"""
Reinforcement Learning Training for Enigma Engine
====================================================

Implements:
- RL-B: RLHF with reward model — small reward model trained from
  DPO preference data, then PPO-style policy gradient on the main model.
- RL-C: Self-play RL — TRAINER model scores STUDENT responses as
  reward signal; policy gradient updates push STUDENT toward higher
  scored outputs.

Both approaches share a RewardModel (small transformer + scalar head)
and a policy gradient training loop with KL penalty against a frozen
reference policy.

Usage:
    from enigma_engine.core.rl_training import (
        RewardModel, RewardTrainer, RLHFTrainer, SelfPlayTrainer,
    )

    # RL-B: Train reward model, then RLHF
    reward_model = RewardModel(base_model)
    reward_trainer = RewardTrainer(reward_model, tokenizer)
    reward_trainer.train(preference_data)

    rlhf = RLHFTrainer(model, tokenizer, reward_model)
    rlhf.train(prompts)

    # RL-C: Self-play with TRAINER as reward
    sp = SelfPlayTrainer(student, tokenizer, trainer_engine)
    sp.train(prompts)
"""

from __future__ import annotations

import copy
import logging
import math
import random
import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _resolve_amp_dtype(amp_dtype: str = "auto") -> torch.dtype:
    """Resolve AMP dtype string to torch.dtype.

    ``"auto"`` picks BF16 on Ampere+ / Blackwell GPUs, FP16 otherwise.
    """
    if amp_dtype == "bfloat16":
        return torch.bfloat16
    if amp_dtype == "float16":
        return torch.float16
    # "auto"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# =============================================================================
# VALUE HEAD — critic for PPO advantage estimation
# =============================================================================


class ValueHead(nn.Module):
    """MLP critic: hidden_state → scalar value.

    Two-layer MLP that projects transformer hidden states to per-token
    state values for GAE advantage estimation.

    Args:
        dim: Hidden dimension of the transformer model.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, 1)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict per-token values.

        Args:
            hidden_states: (B, T, dim) from transformer.

        Returns:
            (B, T) scalar values.
        """
        x = F.relu(self.fc1(hidden_states))
        return self.fc2(x).squeeze(-1)  # (B, T)


# =============================================================================
# ROLLOUT BUFFER — stores experiences for PPO minibatch updates
# =============================================================================


class RolloutBuffer:
    """Stores per-token rollout data for PPO training.

    Each entry is one generated response with per-token log-probs,
    values, rewards, and a mask indicating response tokens.
    Optionally stores full_ids and prompt_len for recomputing
    log-probs during PPO epoch updates.
    """

    def __init__(self):
        self._log_probs: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []
        self._rewards: list[torch.Tensor] = []
        self._masks: list[torch.Tensor] = []
        self._full_ids: list[torch.Tensor | None] = []
        self._prompt_lens: list[int | None] = []
        self._ref_logps: list[torch.Tensor | None] = []

    def store(
        self,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        response_mask: torch.Tensor,
        full_ids: torch.Tensor | None = None,
        prompt_len: int | None = None,
        ref_logps: torch.Tensor | None = None,
    ) -> None:
        """Store one rollout experience.

        All tensors should be 1D with the same length (response tokens).
        full_ids and prompt_len are optional — when provided, PPO epochs
        can recompute fresh log-probs for proper importance sampling.
        ref_logps are reference-model log-probs for per-token KL penalty.
        """
        self._log_probs.append(log_probs.detach())
        self._values.append(values.detach())
        self._rewards.append(rewards.detach())
        self._masks.append(response_mask.detach())
        self._full_ids.append(full_ids.detach() if full_ids is not None else None)
        self._prompt_lens.append(prompt_len)
        self._ref_logps.append(ref_logps.detach() if ref_logps is not None else None)

    def __len__(self) -> int:
        return len(self._log_probs)

    def clear(self) -> None:
        self._log_probs.clear()
        self._values.clear()
        self._rewards.clear()
        self._masks.clear()
        self._full_ids.clear()
        self._prompt_lens.clear()
        self._ref_logps.clear()

    def compute_advantages(
        self,
        gamma: float = 1.0,
        lam: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns across all stored experiences.

        Returns:
            (advantages, returns) — both 1D tensors (concatenated
            across all stored experiences).
        """
        all_advantages = []
        all_returns = []

        for logp, vals, rews, mask in zip(
            self._log_probs,
            self._values,
            self._rewards,
            self._masks,
        ):
            T = vals.shape[0]
            advantages = torch.zeros(T, device=vals.device)
            last_gae = 0.0

            for t in reversed(range(T)):
                if t == T - 1:
                    next_value = 0.0
                else:
                    next_value = vals[t + 1].item()

                delta = rews[t].item() + gamma * next_value - vals[t].item()
                last_gae = delta + gamma * lam * last_gae
                advantages[t] = last_gae

            returns = advantages + vals
            all_advantages.append(advantages)
            all_returns.append(returns)

        return torch.cat(all_advantages), torch.cat(all_returns)

    def get_batched_data(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Concatenate all stored data for minibatch iteration.

        Returns:
            (log_probs, values, rewards, masks) — all 1D concatenated.
        """
        return (
            torch.cat(self._log_probs),
            torch.cat(self._values),
            torch.cat(self._rewards),
            torch.cat(self._masks),
        )


# =============================================================================
# PRIORITIZED REPLAY BUFFER — retains high-reward experiences across epochs
# =============================================================================


class ReplayBuffer:
    """Prioritized experience replay for RLHF training.

    Retains the best rollout experiences across epochs so high-reward
    responses are trained on more frequently.  Experiences are stored
    as detached CPU tensors to avoid VRAM accumulation.

    Priority is based on absolute reward — high-magnitude experiences
    (both positive and negative) are more informative for the policy.

    Args:
        capacity: Maximum number of experiences to retain.
        priority_alpha: Exponent for priority sampling (0 = uniform,
            1 = fully prioritized).
    """

    def __init__(self, capacity: int = 256, priority_alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = priority_alpha
        self._experiences: list[dict[str, torch.Tensor | float]] = []

    def add(
        self,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        response_mask: torch.Tensor,
        reward_scalar: float,
        full_ids: torch.Tensor | None = None,
        prompt_len: int | None = None,
        ref_logps: torch.Tensor | None = None,
    ) -> None:
        """Store one rollout experience (moved to CPU)."""
        exp = {
            "log_probs": log_probs.detach().cpu(),
            "values": values.detach().cpu(),
            "rewards": rewards.detach().cpu(),
            "mask": response_mask.detach().cpu(),
            "priority": abs(reward_scalar) + 1e-6,
            "full_ids": full_ids.detach().cpu() if full_ids is not None else None,
            "prompt_len": prompt_len,
            "ref_logps": ref_logps.detach().cpu() if ref_logps is not None else None,
        }
        self._experiences.append(exp)
        # Evict lowest-priority when over capacity
        if len(self._experiences) > self.capacity:
            self._experiences.sort(key=lambda e: e["priority"])
            self._experiences.pop(0)

    def sample(
        self,
        n: int,
        device: torch.device | str = "cpu",
    ) -> list[dict[str, torch.Tensor]]:
        """Sample *n* experiences weighted by priority.

        Returns list of dicts with tensors moved to *device*.
        """
        if not self._experiences or n <= 0:
            return []

        n = min(n, len(self._experiences))
        priorities = torch.tensor([e["priority"] for e in self._experiences])
        probs = priorities**self.alpha
        probs = probs / probs.sum()

        indices = torch.multinomial(probs, n, replacement=False).tolist()
        result = []
        for idx in indices:
            exp = self._experiences[idx]
            entry: dict[str, Any] = {
                "log_probs": exp["log_probs"].to(device),
                "values": exp["values"].to(device),
                "rewards": exp["rewards"].to(device),
                "mask": exp["mask"].to(device),
            }
            if exp.get("full_ids") is not None:
                entry["full_ids"] = exp["full_ids"].to(device)
            if exp.get("prompt_len") is not None:
                entry["prompt_len"] = exp["prompt_len"]
            if exp.get("ref_logps") is not None:
                entry["ref_logps"] = exp["ref_logps"].to(device)
            result.append(entry)
        return result

    def state_dict(self) -> dict:
        """Serialize buffer contents for checkpoint persistence."""
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "experiences": [
                {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in exp.items()}
                for exp in self._experiences
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore buffer from a serialized state dict."""
        self.capacity = state.get("capacity", self.capacity)
        self.alpha = state.get("alpha", self.alpha)
        self._experiences = state.get("experiences", [])

    def __len__(self) -> int:
        return len(self._experiences)


# =============================================================================
# SHARED HELPERS
# =============================================================================


def _load_state_or_fail(model: nn.Module, state: dict, label: str) -> None:
    """Load a checkpoint state dict, tolerating the LoRA-wrap key differences
    but hard-failing when the checkpoint matches nothing.

    strict=False is needed because a LoRA-wrapped model's keys carry PEFT
    prefixes; but a fully-mismatched checkpoint (wrong architecture, or a
    non-LoRA save into a wrapped model) would otherwise load ZERO weights
    silently. Per the repo's checkpoint contract, refuse that outright.
    """
    result = model.load_state_dict(state, strict=False)
    model_keys = set(model.state_dict())
    loaded = model_keys - set(result.missing_keys)
    if not loaded:
        raise ValueError(
            f"{label} checkpoint matched none of the model's parameters "
            f"({len(state)} checkpoint keys, {len(model_keys)} model keys); "
            "architecture or LoRA-wrap state differs from the saved model."
        )


def _get_response_logps(
    model: nn.Module,
    full_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Compute per-token log-probs for the response portion only.

    Args:
        model: Language model — forward(input_ids) → logits or (logits, loss).
        full_ids: (1, prompt_len + response_len) token ids.
        prompt_len: Length of the prompt prefix.

    Returns:
        (response_len,) tensor of log-probs.
    """
    logits = model(full_ids[:, :-1])
    if isinstance(logits, tuple):
        logits = logits[0]

    log_probs = F.log_softmax(logits, dim=-1)  # (1, L-1, V)
    targets = full_ids[:, 1:]  # (1, L-1)
    per_token = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (1, L-1)

    response_start = max(prompt_len - 1, 0)
    seq_len = per_token.shape[1]
    if response_start >= seq_len:
        logger.warning("prompt_len %d >= sequence length %d -- returning zeros", prompt_len, seq_len + 1)
        return per_token.new_zeros(1)
    return per_token[0, response_start:]


def _get_response_entropy(
    model: nn.Module,
    full_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Compute per-token entropy for the response portion.

    Returns:
        (response_len,) tensor of entropies.
    """
    logits = model(full_ids[:, :-1])
    if isinstance(logits, tuple):
        logits = logits[0]

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # (1, L-1)

    response_start = max(prompt_len - 1, 0)
    seq_len = entropy.shape[1]
    if response_start >= seq_len:
        logger.warning("prompt_len %d >= sequence length %d -- returning zeros", prompt_len, seq_len + 1)
        return entropy.new_zeros(1)
    return entropy[0, response_start:]


def _get_logps_hidden_entropy(
    model: nn.Module,
    full_ids: torch.Tensor,
    prompt_len: int,
    apply_neftune: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single forward pass returning logps, hidden states, and entropy.

    Replaces the three separate calls:
        _get_response_logps + _get_hidden_states + _get_response_entropy

    Runs through model internals manually (embeddings → layers → norm),
    then applies the output projection to get logits.  All three outputs
    are derived from those logits and the pre-head hidden states.

    Mirrors Enigma.forward() behavior for NEFTune when model is in train
    mode so PPO training-mode semantics stay consistent.

    Args:
        model: Enigma model (must have tok_embeddings, layers, norm, output).
        full_ids: (1, prompt_len + response_len) token ids.
        prompt_len: Length of the prompt prefix.

    Returns:
        (response_logps, response_hidden, response_entropy)
        - response_logps:   (resp_len,) per-token log-probs
        - response_hidden:  (1, resp_len, dim) normalized hidden states
        - response_entropy: (resp_len,) per-token entropy
    """
    # --- forward through model internals (single pass) ---
    h = model.tok_embeddings(full_ids[:, :-1])  # (1, T, dim)

    config = getattr(model, "config", None)

    # Keep parity with Enigma.forward() training behavior, but callers
    # collecting rollout/reference log-probs pass apply_neftune=False:
    # embedding noise there randomizes the PPO ratio denominator and the
    # KL estimate.
    if apply_neftune and model.training and config is not None and config.neftune_alpha > 0:
        dims = h.shape[1] * h.shape[2]
        mag = config.neftune_alpha / math.sqrt(dims)
        h = h + torch.empty_like(h).uniform_(-mag, mag)

    use_rope = getattr(config, "use_rope", True)
    if not use_rope and hasattr(model, "pos"):
        T = full_ids.shape[1] - 1
        h = h + model.pos[:, :T]

    freqs = getattr(model, "freqs_cis", None)
    T = h.shape[1]
    mask = None
    if T > 1 and hasattr(model, "_get_causal_mask"):
        mask = model._get_causal_mask(T).to(device=h.device).unsqueeze(0).unsqueeze(0)

    for layer in getattr(model, "layers", []):
        h = layer(h, freqs, mask, False, 0)

    h = model.norm(h)  # (1, T, dim) — normalised hidden states

    # --- logits from the output projection ---
    output_layer = getattr(model, "output", None)
    if output_layer is None:
        raise RuntimeError("Model is missing output projection layer")
    logits = output_layer(h)  # (1, T, V)

    # --- derive all three outputs from logits + h ---
    log_probs = F.log_softmax(logits, dim=-1)  # (1, T, V)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)  # (1, T)

    targets = full_ids[:, 1:]  # (1, T)
    per_token_logps = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (1, T)

    response_start = max(prompt_len - 1, 0)
    seq_len = per_token_logps.shape[1]
    if response_start >= seq_len:
        logger.warning("prompt_len %d >= sequence length %d -- returning zeros", prompt_len, seq_len + 1)
        zeros = per_token_logps.new_zeros(1)
        dummy_hidden = h[:, :1, :]
        return zeros, dummy_hidden, zeros

    return (
        per_token_logps[0, response_start:],  # (resp_len,)
        h[:, response_start:, :],  # (1, resp_len, dim)
        entropy[0, response_start:],  # (resp_len,)
    )


# =============================================================================
# REWARD MODEL
# =============================================================================


class RewardModel(nn.Module):
    """Small transformer + scalar head that outputs a reward score.

    Built on top of an existing Enigma model — reuses its embeddings
    and transformer layers but replaces the language model head with
    a single-value linear projection.  The base weights can be frozen
    or fine-tuned.

    Args:
        base_model: An Enigma model whose architecture is reused.
        freeze_base: If True, freeze all base model weights and only
            train the reward head.  Default True.
    """

    def __init__(self, base_model: nn.Module, freeze_base: bool = True):
        super().__init__()

        # Store config for serialization
        self.base_config = getattr(base_model, "config", None)

        # COPY (not share) the base transformer: these submodules are the
        # policy's own Parameters. Sharing them by reference means freezing
        # the reward head's base below (requires_grad = False) freezes the
        # LIVE policy too, and training the reward model mutates the policy.
        self.tok_embeddings = copy.deepcopy(base_model.tok_embeddings)
        self.layers = copy.deepcopy(base_model.layers)
        self.norm = copy.deepcopy(base_model.norm)

        # Copy RoPE frequencies if present
        if hasattr(base_model, "freqs_cis") and base_model.freqs_cis is not None:
            self.register_buffer("freqs_cis", base_model.freqs_cis.clone())
        else:
            self.freqs_cis = None

        # On-demand causal mask (same pattern as Enigma model)
        self._causal_mask: torch.Tensor | None = None
        self._causal_mask_size: int = 0

        # Copy position embeddings if not using RoPE
        config = getattr(base_model, "config", None)
        self._use_rope = getattr(config, "use_rope", True)
        if not self._use_rope and hasattr(base_model, "pos"):
            self.pos = copy.deepcopy(base_model.pos)

        # Determine hidden dim
        dim = getattr(config, "dim", 512)

        # Reward head: project last hidden state to scalar reward
        self.reward_head = nn.Linear(dim, 1, bias=False)
        nn.init.normal_(self.reward_head.weight, std=0.02)

        # Optionally freeze the transformer
        if freeze_base:
            for param in self.tok_embeddings.parameters():
                param.requires_grad = False
            for param in self.layers.parameters():
                param.requires_grad = False
            for param in self.norm.parameters():
                param.requires_grad = False

    def _get_causal_mask(self, size: int) -> torch.Tensor:
        """Return a (size, size) upper-triangle causal mask, cached and grown on demand."""
        if self._causal_mask is None or self._causal_mask_size < size:
            new_size = max(size, self._causal_mask_size)
            mask = torch.full((new_size, new_size), float("-inf"))
            self._causal_mask = torch.triu(mask, diagonal=1)
            self._causal_mask_size = new_size
        return self._causal_mask[:size, :size]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute reward scores for input sequences.

        Args:
            input_ids: (B, L) token ids.
            attention_mask: optional (B, L) mask.  1 = real token,
                0 = padding.  Used to pick the last non-pad position.

        Returns:
            (B,) tensor of scalar rewards.
        """
        B, T = input_ids.shape

        h = self.tok_embeddings(input_ids)

        if not self._use_rope and hasattr(self, "pos"):
            h = h + self.pos[:, :T]

        # Causal mask (on-demand, same pattern as Enigma model)
        mask = None
        if T > 1:
            mask = self._get_causal_mask(T).to(device=h.device).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            h = layer(h, self.freqs_cis, mask, False, 0)

        h = self.norm(h)  # (B, T, dim)

        # Pick the last non-padding token per sequence
        if attention_mask is not None:
            # Last non-zero position per batch
            lengths = attention_mask.sum(dim=-1).long() - 1
            if (lengths < 0).any():
                logger.warning("RewardModel received all-padding batch -- reward scores will be unreliable.")
            lengths = lengths.clamp(min=0)
            last_hidden = h[torch.arange(B, device=h.device), lengths]
        else:
            last_hidden = h[:, -1, :]  # (B, dim)

        rewards = self.reward_head(last_hidden).squeeze(-1)  # (B,)
        return rewards


# =============================================================================
# REWARD TRAINER — train reward model from preference data
# =============================================================================


@dataclass
class RewardTrainerConfig:
    """Config for reward model training."""

    epochs: int = 3
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_clip: float = 1.0
    margin: float = 0.0  # optional margin in loss
    max_length: int = 512
    use_amp: bool = True
    amp_dtype: str = "auto"  # "auto", "float16", "bfloat16"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8


class RewardTrainer:
    """Train a RewardModel from preference pairs.

    Preference data format: list of dicts with ``prompt``,
    ``chosen`` (preferred response), and ``rejected``
    (dis-preferred response).

    The loss pushes ``reward(chosen) > reward(rejected)``.
    """

    def __init__(
        self,
        reward_model: RewardModel,
        tokenizer: Any,
        config: RewardTrainerConfig | None = None,
    ):
        self.model = reward_model
        self.tokenizer = tokenizer
        self.config = config or RewardTrainerConfig()
        self.device = next(reward_model.parameters()).device

        self.on_progress: Callable[[int, str], None] | None = None
        self._stop_requested = False
        self._lock = threading.Lock()

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def _should_stop(self) -> bool:
        with self._lock:
            return self._stop_requested

    def train(
        self,
        preference_data: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Train reward model on preference pairs.

        Args:
            preference_data: list of {prompt, chosen, rejected}.

        Returns:
            Dict with final_loss, epochs_completed.
        """
        self._stop_requested = False
        self.model.train()

        # Encode pairs
        pairs = self._encode_pairs(preference_data)
        if not pairs:
            raise ValueError("No valid preference pairs after encoding")

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
        )

        amp_dt = _resolve_amp_dtype(self.config.amp_dtype)
        scaler = (
            torch.amp.GradScaler("cuda")
            if (self.config.use_amp and torch.cuda.is_available() and amp_dt != torch.bfloat16)
            else None
        )

        final_loss = 0.0
        epochs_done = 0

        for epoch in range(self.config.epochs):
            if self._should_stop():
                break

            epoch_loss = 0.0
            random.shuffle(pairs)

            for chosen_ids, rejected_ids in pairs:
                if self._should_stop():
                    break

                optimizer.zero_grad()

                with torch.amp.autocast(
                    "cuda",
                    dtype=amp_dt,
                    enabled=self.config.use_amp and torch.cuda.is_available(),
                ):
                    chosen_reward = self.model(chosen_ids)
                    rejected_reward = self.model(rejected_ids)

                    # Bradley-Terry loss
                    loss = -F.logsigmoid(chosen_reward - rejected_reward - self.config.margin).mean()

                if scaler is not None:
                    scaler.scale(loss).backward()
                    if self.config.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if self.config.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    optimizer.step()

                batch_loss = loss.item()
                if math.isnan(batch_loss) or math.isinf(batch_loss):
                    logger.error("Reward training aborted: NaN/Inf loss")
                    self.model.eval()
                    return {"final_loss": float("inf"), "epochs_completed": epochs_done}

                epoch_loss += batch_loss

            avg = epoch_loss / max(len(pairs), 1)
            final_loss = avg
            epochs_done = epoch + 1
            logger.info("Reward epoch %d: loss=%.4f", epoch + 1, avg)

            if self.on_progress:
                pct = int((epoch + 1) / self.config.epochs * 100)
                self.on_progress(pct, f"Reward epoch {epoch + 1}: loss={avg:.4f}")

        self.model.eval()
        return {"final_loss": final_loss, "epochs_completed": epochs_done}

    def _encode_pairs(
        self,
        data: list[dict[str, str]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Encode preference pairs to tensors."""
        pairs = []
        max_len = self.config.max_length

        for item in data:
            prompt = item.get("prompt", "")
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
            if not (prompt and chosen and rejected):
                continue

            c_ids = self.tokenizer.encode(f"User: {prompt}\nAssistant: {chosen}")[:max_len]
            r_ids = self.tokenizer.encode(f"User: {prompt}\nAssistant: {rejected}")[:max_len]

            c_tensor = torch.tensor([c_ids], dtype=torch.long, device=self.device)
            r_tensor = torch.tensor([r_ids], dtype=torch.long, device=self.device)
            pairs.append((c_tensor, r_tensor))

        return pairs


# =============================================================================
# T5-8: PROCESS REWARD MODEL (PRM) -- per-step reward scoring
# =============================================================================


class ProcessRewardModel(nn.Module):
    """Reward model that scores each reasoning step, not just the final answer.

    For chain-of-thought / ``<think>`` blocks, assigns a scalar reward
    to each step boundary (e.g. sentence-ending tokens or explicit step
    markers). Provides denser training signal than outcome-based rewards.

    Args:
        base_model: Base Enigma model whose architecture is reused.
        freeze_base: Freeze transformer weights, train only the step head.
        step_markers: Token IDs that mark reasoning step boundaries.
    """

    def __init__(
        self,
        base_model: nn.Module,
        freeze_base: bool = True,
        step_markers: list[int] | None = None,
    ):
        super().__init__()

        self.base_config = getattr(base_model, "config", None)
        # COPY (not share): these are the policy's own Parameters; sharing
        # them by reference would freeze/mutate the live policy when this
        # reward model is frozen or trained.
        self.tok_embeddings = copy.deepcopy(base_model.tok_embeddings)
        self.layers = copy.deepcopy(base_model.layers)
        self.norm = copy.deepcopy(base_model.norm)

        if hasattr(base_model, "freqs_cis") and base_model.freqs_cis is not None:
            self.register_buffer("freqs_cis", base_model.freqs_cis.clone())
        else:
            self.freqs_cis = None

        self._causal_mask: torch.Tensor | None = None
        self._causal_mask_size: int = 0

        config = getattr(base_model, "config", None)
        self._use_rope = getattr(config, "use_rope", True)
        if not self._use_rope and hasattr(base_model, "pos"):
            self.pos = copy.deepcopy(base_model.pos)

        dim = getattr(config, "dim", 512)

        # Per-token step reward head
        self.step_head = nn.Linear(dim, 1, bias=False)
        nn.init.normal_(self.step_head.weight, std=0.02)

        # Step boundary token IDs
        self.step_markers: set[int] = set(step_markers or [])

        if freeze_base:
            for param in self.tok_embeddings.parameters():
                param.requires_grad = False
            for param in self.layers.parameters():
                param.requires_grad = False
            for param in self.norm.parameters():
                param.requires_grad = False

    def _get_causal_mask(self, size: int) -> torch.Tensor:
        if self._causal_mask is None or self._causal_mask_size < size:
            mask = torch.full((size, size), float("-inf"))
            self._causal_mask = torch.triu(mask, diagonal=1)
            self._causal_mask_size = size
        return self._causal_mask[:size, :size]

    def forward(
        self,
        input_ids: torch.Tensor,
        step_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-step rewards.

        Args:
            input_ids: (B, L) token IDs.
            step_mask: Optional (B, L) boolean mask -- True at step boundary
                positions. If None, auto-detect from step_markers.

        Returns:
            (B, N) rewards where N is the number of step boundary tokens per
            sequence, or (B, L) if step_mask is None and no markers are set.
        """
        B, T = input_ids.shape
        h = self.tok_embeddings(input_ids)

        if not self._use_rope and hasattr(self, "pos"):
            h = h + self.pos[:, :T]

        mask = None
        if T > 1:
            mask = self._get_causal_mask(T).to(device=h.device).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            h = layer(h, self.freqs_cis, mask, False, 0)
        h = self.norm(h)

        # Per-token rewards
        all_rewards = self.step_head(h).squeeze(-1)  # (B, L)

        # Auto-detect step boundaries if no mask provided
        if step_mask is None and self.step_markers:
            step_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for marker in self.step_markers:
                step_mask |= input_ids == marker

        if step_mask is not None and step_mask.any():
            # Return only rewards at step boundary positions
            max_steps = step_mask.sum(dim=1).max().item()
            result = all_rewards.new_zeros(B, max(max_steps, 1))
            for b in range(B):
                positions = step_mask[b].nonzero(as_tuple=True)[0]
                for i, pos in enumerate(positions):
                    if i < result.shape[1]:
                        result[b, i] = all_rewards[b, pos]
            return result

        return all_rewards


@dataclass
class PRMTrainerConfig:
    """Config for Process Reward Model training."""

    epochs: int = 3
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_clip: float = 1.0
    max_length: int = 512
    use_amp: bool = True
    amp_dtype: str = "auto"


class PRMTrainer:
    """Train a ProcessRewardModel from step-level labeled data.

    Training data format: list of dicts with ``input_ids``,
    ``step_positions`` (indices of step boundaries), and
    ``step_labels`` (+1 for correct steps, -1 for wrong steps).
    """

    def __init__(
        self,
        prm: ProcessRewardModel,
        tokenizer: Any,
        config: PRMTrainerConfig | None = None,
    ):
        self.model = prm
        self.tokenizer = tokenizer
        self.config = config or PRMTrainerConfig()
        self.device = next(prm.parameters()).device
        self.on_progress: Callable[[int, str], None] | None = None
        self._stop_requested = False
        self._lock = threading.Lock()

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def _should_stop(self) -> bool:
        with self._lock:
            return self._stop_requested

    def train(
        self,
        step_data: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Train PRM on step-level labeled data.

        Args:
            step_data: List of dicts with keys:
                - "input_ids": list[int] -- tokenized sequence
                - "step_positions": list[int] -- indices of step boundaries
                - "step_labels": list[float] -- +1.0 correct, -1.0 wrong

        Returns:
            Dict with final_loss, epochs_completed.
        """
        self._stop_requested = False
        self.model.train()

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
        )

        amp_dt = _resolve_amp_dtype(self.config.amp_dtype)
        scaler = (
            torch.amp.GradScaler("cuda")
            if (self.config.use_amp and torch.cuda.is_available() and amp_dt != torch.bfloat16)
            else None
        )
        use_amp = self.config.use_amp and torch.cuda.is_available()

        final_loss = float("inf")
        epochs_done = 0

        for epoch in range(self.config.epochs):
            if self._should_stop():
                break

            epoch_loss = 0.0
            n_batches = 0

            for item in step_data:
                if self._should_stop():
                    break

                ids = item.get("input_ids", [])
                positions = item.get("step_positions", [])
                labels = item.get("step_labels", [])

                if not ids or not positions or len(positions) != len(labels):
                    continue

                # Truncate to max length
                ids = ids[: self.config.max_length]
                positions = [p for p in positions if p < len(ids)]
                labels = labels[: len(positions)]
                if not positions:
                    continue

                input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
                step_mask = torch.zeros(1, len(ids), dtype=torch.bool, device=self.device)
                for p in positions:
                    step_mask[0, p] = True
                target = torch.tensor([labels], dtype=torch.float, device=self.device)

                optimizer.zero_grad()

                with torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                    rewards = self.model(input_ids, step_mask=step_mask)
                    # Trim to match target length
                    n_steps = min(rewards.shape[1], target.shape[1])
                    loss = F.mse_loss(rewards[:, :n_steps], target[:, :n_steps])

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / max(n_batches, 1)
            final_loss = avg
            epochs_done = epoch + 1
            logger.info("PRM epoch %d: loss=%.4f", epoch + 1, avg)

            if self.on_progress:
                pct = int((epoch + 1) / self.config.epochs * 100)
                self.on_progress(pct, f"PRM epoch {epoch + 1}: loss={avg:.4f}")

        self.model.eval()
        return {"final_loss": final_loss, "epochs_completed": epochs_done}


# =============================================================================
# RLHF TRAINER (RL-B) — PPO-style policy gradient with reward model
# =============================================================================


@dataclass
class RLHFConfig:
    """Config for RLHF training with PPO."""

    epochs: int = 3
    learning_rate: float = 1e-6
    kl_coeff: float = 0.1  # KL penalty coefficient (initial value)
    kl_target: float = 0.0  # Target KL divergence for adaptive KL (0 = disabled)
    kl_horizon: float = 10000.0  # Adaptive KL update speed (higher = slower adaptation)
    clip_range: float = 0.2  # PPO clipped surrogate ratio
    n_responses: int = 4  # responses per prompt for best-of-N
    max_new_tokens: int = 128
    temperature: float = 0.8
    gradient_clip: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "auto"  # "auto", "float16", "bfloat16"
    max_prompt_length: int = 256
    gamma: float = 1.0  # discount factor
    normalize_rewards: bool = True
    # PPO-specific fields
    value_coeff: float = 0.5  # value loss weight
    entropy_coeff: float = 0.01  # entropy bonus weight
    gae_lambda: float = 0.95  # GAE lambda
    ppo_epochs: int = 4  # minibatch update epochs per rollout
    minibatch_size: int = 4  # rollouts per minibatch
    # Replay buffer settings
    replay_capacity: int = 256  # max stored experiences (0 = disabled)
    replay_ratio: float = 0.25  # fraction of minibatch from replay
    # Optimizer betas (LM-friendly defaults)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    # Checkpoint persistence (empty = disabled)
    checkpoint_dir: str = ""


class RLHFTrainer:
    """RLHF trainer: uses a trained RewardModel to score responses,
    then applies policy gradient updates with KL penalty against
    a frozen reference policy.

    This implements a simplified PPO loop suitable for local training
    with limited compute.

    Args:
        model: The policy model (Enigma) to train.
        tokenizer: Tokenizer for encoding/decoding.
        reward_model: Trained RewardModel.
        config: RLHF training config.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        reward_model: RewardModel,
        config: RLHFConfig | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.config = config or RLHFConfig()
        self.device = next(model.parameters()).device

        self.on_progress: Callable[[int, str], None] | None = None
        self._stop_requested = False
        self._lock = threading.Lock()

        # Adaptive KL coefficient (R9). Updated each epoch if kl_target > 0.
        self._kl_coeff = self.config.kl_coeff

    def _update_adaptive_kl(self, observed_kl: float) -> None:
        """Adjust ``_kl_coeff`` toward ``kl_target`` (PPO-Max style).

        If observed KL is above target, increase coefficient to
        penalise more.  If below, decrease to allow more exploration.
        """
        target = self.config.kl_target
        if target <= 0:
            return  # adaptive KL disabled
        proportional_error = (observed_kl - target) / target
        mult = 1.0 + proportional_error / self.config.kl_horizon
        self._kl_coeff = max(1e-6, self._kl_coeff * mult)

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def _should_stop(self) -> bool:
        with self._lock:
            return self._stop_requested

    def _save_checkpoint(
        self,
        path: Path,
        optimizer: Any,
        value_head: nn.Module,
        replay: ReplayBuffer | None,
        epoch: int,
        metrics: dict,
    ) -> None:
        """Save RLHF training checkpoint with replay buffer state."""
        try:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "value_head_state_dict": value_head.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
            }
            if replay is not None:
                checkpoint["replay_buffer"] = replay.state_dict()
            if hasattr(self.model, "config"):
                checkpoint["model_config"] = self.model.config.__dict__
            from enigma_engine.core.safe_save import atomic_torch_save

            atomic_torch_save(checkpoint, path)
            logger.info("RLHF checkpoint saved: %s", path)
        except Exception as exc:
            logger.error("Failed to save RLHF checkpoint: %s", exc)

    def load_checkpoint(self, path: Path) -> None:
        """Load RLHF checkpoint. Pending state is applied in train().

        Model weights, optimizer, value head, and replay buffer are all
        stashed as ``_pending_*`` and applied inside ``train()``. The
        model state is deferred deliberately: the checkpoint was saved
        AFTER ``_setup_reference()`` wrapped the model with LoRA, so its
        keys are PEFT-prefixed and only load once the wrap is back in
        place (loading into the raw model would key-mismatch).
        """
        from enigma_engine.core.model_registry import safe_load_weights

        checkpoint = safe_load_weights(path, map_location=self.device)

        self._pending_model_state = checkpoint.get("model_state_dict")
        self._pending_optimizer_state = checkpoint.get("optimizer_state_dict")
        self._pending_value_head_state = checkpoint.get("value_head_state_dict")
        self._pending_replay_state = checkpoint.get("replay_buffer")
        self._start_epoch = checkpoint.get("epoch", 0)
        logger.info("RLHF checkpoint loaded: %s", path)

    def _setup_reference(self) -> None:
        """Set up the reference policy for KL penalty.

        Preferred: wrap model with LoRA so the frozen base *is* the
        reference — zero extra VRAM.  Fallback: keep reference on GPU
        if there's enough VRAM, otherwise CPU-offloaded copy.
        """
        self._use_lora_ref = False
        self._ref_model_cpu = None
        self._ref_on_gpu = False

        try:
            from enigma_engine.core.lora_utils import (
                create_lora_model,
                PEFT_AVAILABLE,
                LoraConfig,
            )

            if PEFT_AVAILABLE:
                lora_cfg = LoraConfig(rank=8, alpha=16, dropout=0.0)
                self.model = create_lora_model(self.model, lora_cfg)
                self._use_lora_ref = True
                logger.info("RLHF: using LoRA -- frozen base weights serve as reference policy (no extra VRAM)")
                return
        except Exception as exc:
            logger.debug("LoRA setup failed, using copy: %s", exc)

        ref = copy.deepcopy(self.model)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False

        # Try keeping ref on GPU if enough VRAM (S546 — avoids 1000+ transfers)
        if torch.cuda.is_available() and self.device.type == "cuda":
            try:
                model_bytes = sum(p.nelement() * p.element_size() for p in ref.parameters())
                free_bytes = torch.cuda.mem_get_info(self.device)[0]
                # Keep on GPU if model fits in <50% of remaining VRAM
                if model_bytes < free_bytes * 0.5:
                    self._ref_model_cpu = ref.to(self.device)
                    self._ref_on_gpu = True
                    logger.info(
                        "RLHF: reference model kept on GPU (%.0f MB model, %.0f MB free)",
                        model_bytes / 1e6,
                        free_bytes / 1e6,
                    )
                    return
            except Exception as exc:
                logger.debug("GPU ref check failed: %s", exc)

        # Fallback: CPU-offloaded reference
        ref = ref.cpu()
        self._ref_model_cpu = ref
        logger.info("RLHF: using CPU-offloaded reference model (no extra VRAM, slight I/O overhead)")

    def _get_ref_logps(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Get reference policy log-probs, using LoRA disable or CPU offload."""
        if self._use_lora_ref:
            self.model.eval()
            with torch.no_grad():
                self.model.disable_adapter_layers()
                try:
                    ref_logps = _get_response_logps(self.model, full_ids, prompt_len)
                finally:
                    self.model.enable_adapter_layers()
            self.model.train()
            return ref_logps

        # Reference model already on GPU — no transfer needed (S546)
        if self._ref_on_gpu:
            with torch.no_grad():
                return _get_response_logps(self._ref_model_cpu, full_ids, prompt_len)

        try:
            ref = self._ref_model_cpu.to(self.device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("OOM moving reference model to %s -- freeing cache", self.device)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise
        try:
            with torch.no_grad():
                ref_logps = _get_response_logps(ref, full_ids, prompt_len)
            return ref_logps
        finally:
            ref.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def train(
        self,
        prompts: list[str],
    ) -> dict[str, Any]:
        """Run PPO training on a list of prompts.

        Two-phase loop per epoch:
          Phase 1 — Rollout collection: for each prompt, generate
          n_responses responses, score with reward model, keep the
          best, collect per-token (logprobs, values, rewards).
          Phase 2 — PPO update: multiple epochs of minibatch updates
          with clipped surrogate loss + value loss + entropy bonus.

        Args:
            prompts: Training prompts.

        Returns:
            Dict with metrics.
        """
        if not prompts:
            raise ValueError("No prompts provided for RLHF training")

        self._stop_requested = False
        cfg = self.config

        # Freeze reward model
        self.reward_model.eval()
        for p in self.reward_model.parameters():
            p.requires_grad = False

        # Set up reference policy (LoRA or CPU offload)
        self._setup_reference()

        # Apply the deferred model state now that the LoRA wrap (if any) is
        # back in place, so PEFT-prefixed checkpoint keys line up.
        if getattr(self, "_pending_model_state", None):
            _load_state_or_fail(self.model, self._pending_model_state, "RLHF")
            self._pending_model_state = None

        # Value head for advantage estimation
        model_config = getattr(self.model, "config", None)
        dim = getattr(model_config, "dim", 512)
        value_head = ValueHead(dim).to(self.device)

        self.model.train()

        # Only train parameters that require grad (LoRA adapters or all)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        all_params = trainable + list(value_head.parameters())
        optimizer = torch.optim.AdamW(
            all_params,
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
        )

        amp_dt = _resolve_amp_dtype(cfg.amp_dtype)
        scaler = (
            torch.amp.GradScaler("cuda")
            if (cfg.use_amp and torch.cuda.is_available() and amp_dt != torch.bfloat16)
            else None
        )
        use_amp = cfg.use_amp and torch.cuda.is_available()

        all_rewards: list[float] = []
        all_kl: list[float] = []
        epochs_done = 0
        # Bounded reward history for normalization
        reward_history: deque[float] = deque(maxlen=500)

        # Prioritized replay buffer for cross-epoch experience reuse
        replay = ReplayBuffer(capacity=cfg.replay_capacity) if cfg.replay_capacity > 0 else None

        # Restore pending state from load_checkpoint()
        start_epoch = getattr(self, "_start_epoch", 0)
        if hasattr(self, "_pending_optimizer_state") and self._pending_optimizer_state:
            optimizer.load_state_dict(self._pending_optimizer_state)
            self._pending_optimizer_state = None
        if hasattr(self, "_pending_value_head_state") and self._pending_value_head_state:
            value_head.load_state_dict(self._pending_value_head_state)
            self._pending_value_head_state = None
        if hasattr(self, "_pending_replay_state") and self._pending_replay_state:
            if replay is not None:
                replay.load_state_dict(self._pending_replay_state)
            self._pending_replay_state = None

        # Checkpoint directory for periodic saves
        ckpt_dir = None
        if cfg.checkpoint_dir:
            from pathlib import Path

            ckpt_dir = Path(cfg.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(start_epoch, cfg.epochs):
            if self._should_stop():
                break

            # ---- Phase 1: Collect rollouts ----
            rollout = RolloutBuffer()
            epoch_reward = 0.0
            epoch_kl = 0.0
            n_collected = 0
            random.shuffle(prompts)

            for prompt in prompts:
                if self._should_stop():
                    break

                prompt_ids = self.tokenizer.encode(prompt)
                prompt_ids = prompt_ids[: cfg.max_prompt_length]
                prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
                prompt_len = len(prompt_ids)

                # Generate n_responses, keep the best
                best_gen = None
                best_reward = float("-inf")

                self.model.eval()
                for _ in range(cfg.n_responses):
                    with torch.no_grad():
                        try:
                            gen_ids = self.model.generate(
                                prompt_tensor,
                                max_new_tokens=cfg.max_new_tokens,
                                temperature=cfg.temperature,
                            )
                        except Exception as exc:
                            logger.warning("Generation failed: %s", exc)
                            continue

                    resp_ids = gen_ids[:, prompt_len:]
                    if resp_ids.shape[1] < 1:
                        continue

                    with torch.no_grad():
                        r = self.reward_model(gen_ids).item()

                    if r > best_reward:
                        best_reward = r
                        best_gen = gen_ids

                self.model.train()

                if best_gen is None:
                    continue

                full_ids = best_gen
                resp_len = full_ids.shape[1] - prompt_len
                reward_history.append(best_reward)

                # Normalized reward → per-token reward (terminal)
                reward_scalar = best_reward
                if cfg.normalize_rewards and len(reward_history) > 1:
                    r_mean = sum(reward_history) / len(reward_history)
                    r_var = sum((r - r_mean) ** 2 for r in reward_history) / len(reward_history)
                    r_std = max(r_var**0.5, 1e-8)
                    reward_scalar = (best_reward - r_mean) / r_std

                # Per-token reward: 0 everywhere except last response token
                per_token_rewards = torch.zeros(resp_len, device=self.device)
                per_token_rewards[-1] = reward_scalar

                # Collect old log-probs and values (no grad) — single combined
                # pass. apply_neftune=False: this is the rollout behavior
                # policy's actual log-probs, not a training-noise forward.
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                    old_logps, hidden, _ = _get_logps_hidden_entropy(
                        self.model, full_ids, prompt_len, apply_neftune=False
                    )
                    old_values = value_head(hidden).squeeze(0)  # (resp_len,)

                    ref_logps = self._get_ref_logps(full_ids, prompt_len)

                kl = (old_logps - ref_logps).mean().item()
                response_mask = torch.ones(resp_len, device=self.device)

                rollout.store(
                    log_probs=old_logps,
                    values=old_values,
                    rewards=per_token_rewards,
                    response_mask=response_mask,
                    full_ids=full_ids,
                    prompt_len=prompt_len,
                    ref_logps=ref_logps,
                )

                # Add to replay buffer for cross-epoch reuse
                if replay is not None:
                    replay.add(
                        log_probs=old_logps,
                        values=old_values,
                        rewards=per_token_rewards,
                        response_mask=response_mask,
                        reward_scalar=reward_scalar,
                        full_ids=full_ids,
                        prompt_len=prompt_len,
                        ref_logps=ref_logps,
                    )

                epoch_reward += best_reward
                epoch_kl += kl
                n_collected += 1

            if n_collected == 0:
                epochs_done = epoch + 1
                all_rewards.append(0.0)
                all_kl.append(0.0)
                continue

            # ---- Phase 2: PPO minibatch updates ----

            # Mix in replay experiences from prior epochs
            if replay is not None and len(replay) > 0:
                n_replay = max(1, int(n_collected * cfg.replay_ratio))
                replay_samples = replay.sample(n_replay, device=self.device)
                for s in replay_samples:
                    rollout.store(
                        log_probs=s["log_probs"],
                        values=s["values"],
                        rewards=s["rewards"],
                        response_mask=s["mask"],
                        full_ids=s.get("full_ids"),
                        prompt_len=s.get("prompt_len"),
                        ref_logps=s.get("ref_logps"),
                    )

            advantages, returns = rollout.compute_advantages(gamma=cfg.gamma, lam=cfg.gae_lambda)
            old_logps_all, old_values_all, _, masks_all = rollout.get_batched_data()

            # Normalize advantages
            if advantages.numel() > 1:
                adv_std = advantages.std(unbiased=False)
                if adv_std > 1e-6:
                    advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
                else:
                    advantages = advantages - advantages.mean()

            # Store full_ids and prompt_lens for recomputation
            # We need to re-forward through the model on each PPO epoch.
            # For simplicity with variable-length rollouts, we iterate
            # over the stored rollout entries.
            n_rollouts = len(rollout)

            for ppo_epoch in range(cfg.ppo_epochs):
                if self._should_stop():
                    break

                # Shuffle rollout indices for minibatch
                indices = list(range(n_rollouts))
                random.shuffle(indices)

                # Pre-compute cumulative offsets for O(1) lookup
                _offsets = [0] * (n_rollouts + 1)
                for _i in range(n_rollouts):
                    _offsets[_i + 1] = _offsets[_i] + rollout._log_probs[_i].shape[0]

                for mb_start in range(0, n_rollouts, cfg.minibatch_size):
                    mb_indices = indices[mb_start : mb_start + cfg.minibatch_size]

                    total_policy_loss = torch.tensor(0.0, device=self.device)
                    total_value_loss = torch.tensor(0.0, device=self.device)
                    total_entropy = torch.tensor(0.0, device=self.device)
                    total_kl = torch.tensor(0.0, device=self.device)
                    n_tokens = 0

                    for idx in mb_indices:
                        rollout_len = rollout._log_probs[idx].shape[0]
                        start = _offsets[idx]
                        end = start + rollout_len

                        mb_advantages_i = advantages[start:end]
                        mb_returns_i = returns[start:end]
                        mb_old_logps_i = old_logps_all[start:end]

                        # Recompute fresh log-probs and values if we
                        # have the original full_ids (not replay items)
                        stored_ids = rollout._full_ids[idx]
                        stored_plen = rollout._prompt_lens[idx]
                        if stored_ids is not None and stored_plen is not None:
                            with torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                                # Single combined pass: logps + hidden + entropy
                                new_logps, hidden, entropy = _get_logps_hidden_entropy(
                                    self.model, stored_ids, stored_plen
                                )
                                new_values = value_head(hidden).squeeze(0)
                            log_ratio = (new_logps - mb_old_logps_i).clamp(-20, 20)
                            ratio = torch.exp(log_ratio)
                            entropy_bonus = entropy.mean()

                            # Per-token KL: current policy vs reference
                            stored_ref = rollout._ref_logps[idx]
                            if stored_ref is not None:
                                total_kl = total_kl + (new_logps - stored_ref).mean()
                        else:
                            # Replay items: no full_ids, fall back to
                            # ratio=1 (first PPO epoch is exact)
                            ratio = torch.ones_like(mb_old_logps_i)
                            new_values = old_values_all[start:end]
                            entropy_bonus = -mb_old_logps_i.mean()

                        # Clipped surrogate loss
                        surr1 = ratio * mb_advantages_i
                        surr2 = (
                            torch.clamp(
                                ratio,
                                1.0 - cfg.clip_range,
                                1.0 + cfg.clip_range,
                            )
                            * mb_advantages_i
                        )
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # Value loss (clipped)
                        value_loss = F.mse_loss(new_values, mb_returns_i)

                        total_policy_loss = total_policy_loss + policy_loss
                        total_value_loss = total_value_loss + value_loss
                        total_entropy = total_entropy + entropy_bonus
                        n_tokens += rollout_len

                    mb_count = max(len(mb_indices), 1)
                    loss = (
                        total_policy_loss / mb_count
                        + cfg.value_coeff * total_value_loss / mb_count
                        - cfg.entropy_coeff * total_entropy / mb_count
                        + self._kl_coeff * total_kl / mb_count
                    )

                    # Adaptive KL update (R9) — use scalar KL
                    batch_avg_kl = abs((total_kl / mb_count).item())
                    self._update_adaptive_kl(batch_avg_kl)

                    optimizer.zero_grad()
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        if cfg.gradient_clip > 0:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(all_params, cfg.gradient_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if cfg.gradient_clip > 0:
                            nn.utils.clip_grad_norm_(all_params, cfg.gradient_clip)
                        optimizer.step()

            avg_reward = epoch_reward / max(n_collected, 1)
            avg_kl = epoch_kl / max(n_collected, 1)
            all_rewards.append(avg_reward)
            all_kl.append(avg_kl)
            epochs_done = epoch + 1

            logger.info("RLHF epoch %d: avg_reward=%.4f, avg_kl=%.4f", epoch + 1, avg_reward, avg_kl)

            if self.on_progress:
                pct = int((epoch + 1) / cfg.epochs * 100)
                self.on_progress(pct, f"RLHF epoch {epoch + 1}: reward={avg_reward:.3f}, kl={avg_kl:.4f}")

            if ckpt_dir:
                self._save_checkpoint(
                    ckpt_dir / f"rlhf_epoch{epoch + 1}.pt",
                    optimizer,
                    value_head,
                    replay,
                    epoch + 1,
                    {"avg_rewards": all_rewards, "avg_kl": all_kl},
                )

        self.model.eval()
        if self._ref_model_cpu is not None:
            del self._ref_model_cpu
            self._ref_model_cpu = None

        return {
            "epochs_completed": epochs_done,
            "avg_rewards": all_rewards,
            "avg_kl": all_kl,
            "final_reward": all_rewards[-1] if all_rewards else 0.0,
        }


# =============================================================================
# SELF-PLAY RL (RL-C) — TRAINER as reward signal
# =============================================================================


@dataclass
class SelfPlayConfig:
    """Config for self-play RL training."""

    epochs: int = 3
    learning_rate: float = 1e-6
    kl_coeff: float = 0.05
    kl_target: float = 0.0  # Target KL for adaptive KL (0 = disabled)
    kl_horizon: float = 10000.0  # Adaptive KL update speed
    clip_range: float = 0.2  # PPO clipped surrogate ratio
    n_responses: int = 4
    max_new_tokens: int = 128
    temperature: float = 0.8
    gradient_clip: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "auto"  # "auto", "float16", "bfloat16"
    max_prompt_length: int = 256
    score_prompt: str = (
        "Rate the following response on a scale of 0 to 10 for quality, "
        "helpfulness, and coherence. Respond with ONLY the number.\n\n"
        "Question: {prompt}\n\nResponse: {response}\n\nScore:"
    )
    # PPO-specific fields
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    minibatch_size: int = 4
    # Replay buffer settings
    replay_capacity: int = 256  # max stored experiences (0 = disabled)
    replay_ratio: float = 0.25  # fraction of minibatch from replay
    # Optimizer betas (LM-friendly defaults)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    # Checkpoint persistence (empty = disabled)
    checkpoint_dir: str = ""


class SelfPlayTrainer:
    """Self-play RL: TRAINER model scores STUDENT responses.

    The TRAINER model is used as a reward function — it reads the
    STUDENT's response and outputs a numerical score.  Policy gradient
    updates push the STUDENT toward higher-scoring responses, with
    a KL penalty against the initial STUDENT weights.

    Args:
        student: The model being trained (Enigma).
        tokenizer: Tokenizer for the student model.
        trainer_engine: An EnigmaEngine loaded with the TRAINER model,
            used to generate reward scores via chat.
        config: Self-play config.
    """

    def __init__(
        self,
        student: nn.Module,
        tokenizer: Any,
        trainer_engine: Any,
        config: SelfPlayConfig | None = None,
    ):
        self.student = student
        self.tokenizer = tokenizer
        self.trainer_engine = trainer_engine
        self.config = config or SelfPlayConfig()
        self.device = next(student.parameters()).device

        self.on_progress: Callable[[int, str], None] | None = None
        self._stop_requested = False
        self._lock = threading.Lock()

        # Adaptive KL coefficient (R9)
        self._kl_coeff = self.config.kl_coeff

    def _update_adaptive_kl(self, observed_kl: float) -> None:
        """Adjust ``_kl_coeff`` toward ``kl_target``."""
        target = self.config.kl_target
        if target <= 0:
            return
        proportional_error = (observed_kl - target) / target
        mult = 1.0 + proportional_error / self.config.kl_horizon
        self._kl_coeff = max(1e-6, self._kl_coeff * mult)

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def _should_stop(self) -> bool:
        with self._lock:
            return self._stop_requested

    def _save_checkpoint(
        self,
        path: Path,
        optimizer: Any,
        value_head: nn.Module,
        replay: ReplayBuffer | None,
        epoch: int,
        metrics: dict,
    ) -> None:
        """Save self-play training checkpoint with replay buffer state."""
        try:
            checkpoint = {
                "model_state_dict": self.student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "value_head_state_dict": value_head.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
            }
            if replay is not None:
                checkpoint["replay_buffer"] = replay.state_dict()
            if hasattr(self.student, "config"):
                checkpoint["model_config"] = self.student.config.__dict__
            from enigma_engine.core.safe_save import atomic_torch_save

            atomic_torch_save(checkpoint, path)
            logger.info("Self-play checkpoint saved: %s", path)
        except Exception as exc:
            logger.error("Failed to save self-play checkpoint: %s", exc)

    def load_checkpoint(self, path: Path) -> None:
        """Load self-play checkpoint. Pending state is applied in train().

        Student weights, optimizer, value head, and replay buffer are all
        stashed as ``_pending_*`` and applied inside ``train()``. The
        student state is deferred deliberately: the checkpoint was saved
        AFTER ``_setup_reference()`` wrapped the student with LoRA, so its
        keys are PEFT-prefixed and only load once the wrap is back in
        place (loading into the raw student would key-mismatch).
        """
        from enigma_engine.core.model_registry import safe_load_weights

        checkpoint = safe_load_weights(path, map_location=self.device)

        self._pending_model_state = checkpoint.get("model_state_dict")
        self._pending_optimizer_state = checkpoint.get("optimizer_state_dict")
        self._pending_value_head_state = checkpoint.get("value_head_state_dict")
        self._pending_replay_state = checkpoint.get("replay_buffer")
        self._start_epoch = checkpoint.get("epoch", 0)
        logger.info("Self-play checkpoint loaded: %s", path)

    def _get_trainer_score(self, prompt: str, response: str) -> float:
        """Ask the TRAINER model to score a response.

        Args:
            prompt: The original question.
            response: The student's response.

        Returns:
            Numeric score 0-10, or 5.0 on failure.
        """
        score_prompt = self.config.score_prompt.format(prompt=prompt, response=response)

        base_score = 5.0  # neutral fallback
        try:
            result = self.trainer_engine.chat(score_prompt, max_tokens=16)
            if isinstance(result, dict):
                result = result.get("response", result.get("text", "5"))
            result = str(result).strip()

            # Extract first number from response
            import re

            match = re.search(r"(\d+(?:\.\d+)?)", result)
            if match:
                base_score = float(match.group(1))
                base_score = min(max(base_score, 0.0), 10.0)
        except Exception as exc:
            logger.debug("Trainer scoring failed: %s", exc)

        return base_score

    def _setup_reference(self) -> None:
        """Set up reference policy for KL penalty (same approach as RLHFTrainer)."""
        self._use_lora_ref = False
        self._ref_model_cpu = None

        try:
            from enigma_engine.core.lora_utils import (
                create_lora_model,
                PEFT_AVAILABLE,
                LoraConfig,
            )

            if PEFT_AVAILABLE:
                lora_cfg = LoraConfig(rank=8, alpha=16, dropout=0.0)
                self.student = create_lora_model(self.student, lora_cfg)
                self._use_lora_ref = True
                logger.info("Self-play: using LoRA -- frozen base weights serve as reference policy (no extra VRAM)")
                return
        except Exception as exc:
            logger.debug("LoRA setup failed, using CPU offload: %s", exc)

        # Fallback: CPU-offloaded reference
        ref = copy.deepcopy(self.student)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False
        ref = ref.cpu()
        self._ref_model_cpu = ref
        logger.info("Self-play: using CPU-offloaded reference model (no extra VRAM, slight I/O overhead)")

    def _get_ref_logps(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Get reference policy log-probs, using LoRA disable or CPU offload."""
        if self._use_lora_ref:
            self.student.eval()
            with torch.no_grad():
                self.student.disable_adapter_layers()
                try:
                    ref_logps = _get_response_logps(self.student, full_ids, prompt_len)
                finally:
                    self.student.enable_adapter_layers()
            self.student.train()
            return ref_logps

        try:
            ref = self._ref_model_cpu.to(self.device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("OOM moving reference model to %s -- freeing cache", self.device)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise
        try:
            with torch.no_grad():
                ref_logps = _get_response_logps(ref, full_ids, prompt_len)
            return ref_logps
        finally:
            ref.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def train(
        self,
        prompts: list[str],
    ) -> dict[str, Any]:
        """Run self-play PPO training.

        Two-phase loop per epoch:
          Phase 1 — Rollout: for each prompt, generate n_responses,
          have TRAINER score each, keep the best, collect per-token data.
          Phase 2 — PPO update with clipped surrogate + value + entropy.

        Args:
            prompts: Training prompts.

        Returns:
            Dict with metrics.
        """
        if not prompts:
            raise ValueError("No prompts provided for self-play training")

        self._stop_requested = False
        cfg = self.config

        # Set up reference policy (LoRA or CPU offload)
        self._setup_reference()

        # Apply the deferred student state now that the LoRA wrap (if any)
        # is back in place, so PEFT-prefixed checkpoint keys line up.
        if getattr(self, "_pending_model_state", None):
            _load_state_or_fail(self.student, self._pending_model_state, "self-play")
            self._pending_model_state = None

        # Value head for advantage estimation
        model_config = getattr(self.student, "config", None)
        dim = getattr(model_config, "dim", 512)
        value_head = ValueHead(dim).to(self.device)

        self.student.train()
        trainable = [p for p in self.student.parameters() if p.requires_grad]
        all_params = trainable + list(value_head.parameters())
        optimizer = torch.optim.AdamW(
            all_params,
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
        )

        amp_dt = _resolve_amp_dtype(cfg.amp_dtype)
        scaler = (
            torch.amp.GradScaler("cuda")
            if (cfg.use_amp and torch.cuda.is_available() and amp_dt != torch.bfloat16)
            else None
        )
        use_amp = cfg.use_amp and torch.cuda.is_available()

        all_scores: list[float] = []
        epochs_done = 0

        # Prioritized replay buffer for cross-epoch experience reuse
        replay = ReplayBuffer(capacity=cfg.replay_capacity) if cfg.replay_capacity > 0 else None

        # Restore pending checkpoint state (from load_checkpoint)
        start_epoch = getattr(self, "_start_epoch", 0)
        if hasattr(self, "_pending_optimizer_state") and self._pending_optimizer_state:
            optimizer.load_state_dict(self._pending_optimizer_state)
            self._pending_optimizer_state = None
        if hasattr(self, "_pending_value_head_state") and self._pending_value_head_state:
            value_head.load_state_dict(self._pending_value_head_state)
            self._pending_value_head_state = None
        if hasattr(self, "_pending_replay_state") and self._pending_replay_state:
            if replay is not None:
                replay.load_state_dict(self._pending_replay_state)
            self._pending_replay_state = None

        ckpt_dir = None
        if cfg.checkpoint_dir:
            from pathlib import Path

            ckpt_dir = Path(cfg.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(start_epoch, cfg.epochs):
            if self._should_stop():
                break

            # ---- Phase 1: Collect rollouts ----
            rollout = RolloutBuffer()
            epoch_score = 0.0
            epoch_kl = 0.0
            n_collected = 0
            random.shuffle(prompts)

            for prompt in prompts:
                if self._should_stop():
                    break

                prompt_ids = self.tokenizer.encode(prompt)[: cfg.max_prompt_length]
                prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
                prompt_len = len(prompt_ids)

                # Generate n_responses, score each, keep the best
                best_gen = None
                best_score = float("-inf")

                self.student.eval()
                for _ in range(cfg.n_responses):
                    with torch.no_grad():
                        try:
                            gen_ids = self.student.generate(
                                prompt_tensor,
                                max_new_tokens=cfg.max_new_tokens,
                                temperature=cfg.temperature,
                            )
                        except Exception as exc:
                            logger.debug("Student generation failed: %s", exc)
                            continue

                    resp_ids = gen_ids[0, prompt_len:].tolist()
                    response = self.tokenizer.decode(resp_ids)
                    if not response.strip():
                        continue

                    score = self._get_trainer_score(prompt, response)
                    if score > best_score:
                        best_score = score
                        best_gen = gen_ids

                self.student.train()

                if best_gen is None:
                    continue

                full_ids = best_gen
                resp_len = full_ids.shape[1] - prompt_len

                # Normalize score to [-1, 1] range
                reward_scalar = (best_score - 5.0) / 5.0

                # Per-token reward: terminal only
                per_token_rewards = torch.zeros(resp_len, device=self.device)
                per_token_rewards[-1] = reward_scalar

                # Collect old log-probs and values — single combined pass.
                # apply_neftune=False: the rollout behavior policy's actual
                # log-probs, not a training-noise forward.
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                    old_logps, hidden, _ = _get_logps_hidden_entropy(
                        self.student, full_ids, prompt_len, apply_neftune=False
                    )
                    old_values = value_head(hidden).squeeze(0)
                    ref_logps = self._get_ref_logps(full_ids, prompt_len)

                kl = (old_logps - ref_logps).mean().item()
                response_mask = torch.ones(resp_len, device=self.device)

                rollout.store(
                    log_probs=old_logps,
                    values=old_values,
                    rewards=per_token_rewards,
                    response_mask=response_mask,
                    full_ids=full_ids,
                    prompt_len=prompt_len,
                    ref_logps=ref_logps,
                )

                # Add to replay buffer for cross-epoch reuse
                if replay is not None:
                    replay.add(
                        log_probs=old_logps,
                        values=old_values,
                        rewards=per_token_rewards,
                        response_mask=response_mask,
                        reward_scalar=reward_scalar,
                        full_ids=full_ids,
                        prompt_len=prompt_len,
                        ref_logps=ref_logps,
                    )

                epoch_score += best_score
                epoch_kl += kl
                n_collected += 1

            if n_collected == 0:
                epochs_done = epoch + 1
                all_scores.append(0.0)
                continue

            # ---- Phase 2: PPO minibatch updates ----

            # Mix in replay experiences from prior epochs
            if replay is not None and len(replay) > 0:
                n_replay = max(1, int(n_collected * cfg.replay_ratio))
                replay_samples = replay.sample(n_replay, device=self.device)
                for s in replay_samples:
                    rollout.store(
                        log_probs=s["log_probs"],
                        values=s["values"],
                        rewards=s["rewards"],
                        response_mask=s["mask"],
                        full_ids=s.get("full_ids"),
                        prompt_len=s.get("prompt_len"),
                        ref_logps=s.get("ref_logps"),
                    )

            advantages, returns = rollout.compute_advantages(gamma=1.0, lam=cfg.gae_lambda)
            old_logps_all, old_values_all, _, masks_all = rollout.get_batched_data()

            # Normalize advantages
            if advantages.numel() > 1:
                adv_std = advantages.std(unbiased=False)
                if adv_std > 1e-6:
                    advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
                else:
                    advantages = advantages - advantages.mean()

            n_rollouts = len(rollout)

            for ppo_epoch in range(cfg.ppo_epochs):
                if self._should_stop():
                    break

                indices = list(range(n_rollouts))
                random.shuffle(indices)

                # Pre-compute cumulative offsets for O(1) lookup
                _offsets = [0] * (n_rollouts + 1)
                for _i in range(n_rollouts):
                    _offsets[_i + 1] = _offsets[_i] + rollout._log_probs[_i].shape[0]

                for mb_start in range(0, n_rollouts, cfg.minibatch_size):
                    mb_indices = indices[mb_start : mb_start + cfg.minibatch_size]

                    total_policy_loss = torch.tensor(0.0, device=self.device)
                    total_value_loss = torch.tensor(0.0, device=self.device)
                    total_entropy = torch.tensor(0.0, device=self.device)
                    total_kl = torch.tensor(0.0, device=self.device)

                    for idx in mb_indices:
                        rollout_len = rollout._log_probs[idx].shape[0]
                        start = _offsets[idx]
                        end = start + rollout_len

                        mb_advantages_i = advantages[start:end]
                        mb_returns_i = returns[start:end]
                        mb_old_logps_i = old_logps_all[start:end]

                        # Recompute fresh log-probs and values if we
                        # have the original full_ids (not replay items)
                        stored_ids = rollout._full_ids[idx]
                        stored_plen = rollout._prompt_lens[idx]
                        if stored_ids is not None and stored_plen is not None:
                            with torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp):
                                # Single combined pass: logps + hidden + entropy
                                new_logps, hidden, entropy = _get_logps_hidden_entropy(
                                    self.student, stored_ids, stored_plen
                                )
                                new_values = value_head(hidden).squeeze(0)
                            log_ratio = (new_logps - mb_old_logps_i).clamp(-20, 20)
                            ratio = torch.exp(log_ratio)
                            entropy_bonus = entropy.mean()

                            # Per-token KL: current policy vs reference
                            stored_ref = rollout._ref_logps[idx]
                            if stored_ref is not None:
                                total_kl = total_kl + (new_logps - stored_ref).mean()
                        else:
                            ratio = torch.ones_like(mb_old_logps_i)
                            new_values = old_values_all[start:end]
                            entropy_bonus = -mb_old_logps_i.mean()

                        surr1 = ratio * mb_advantages_i
                        surr2 = (
                            torch.clamp(
                                ratio,
                                1.0 - cfg.clip_range,
                                1.0 + cfg.clip_range,
                            )
                            * mb_advantages_i
                        )
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_loss = F.mse_loss(new_values, mb_returns_i)

                        total_policy_loss = total_policy_loss + policy_loss
                        total_value_loss = total_value_loss + value_loss
                        total_entropy = total_entropy + entropy_bonus

                    mb_count = max(len(mb_indices), 1)
                    loss = (
                        total_policy_loss / mb_count
                        + cfg.value_coeff * total_value_loss / mb_count
                        - cfg.entropy_coeff * total_entropy / mb_count
                        + self._kl_coeff * total_kl / mb_count
                    )

                    # Adaptive KL update (R9) — use scalar KL
                    batch_avg_kl = abs((total_kl / mb_count).item())
                    self._update_adaptive_kl(batch_avg_kl)

                    optimizer.zero_grad()
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        if cfg.gradient_clip > 0:
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(all_params, cfg.gradient_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if cfg.gradient_clip > 0:
                            nn.utils.clip_grad_norm_(all_params, cfg.gradient_clip)
                        optimizer.step()

            avg_score = epoch_score / max(n_collected, 1)
            all_scores.append(avg_score)
            epochs_done = epoch + 1

            logger.info("Self-play epoch %d: avg_score=%.2f", epoch + 1, avg_score)

            if self.on_progress:
                pct = int((epoch + 1) / cfg.epochs * 100)
                self.on_progress(pct, f"Self-play epoch {epoch + 1}: score={avg_score:.2f}")

            if ckpt_dir:
                self._save_checkpoint(
                    ckpt_dir / f"selfplay_epoch{epoch + 1}.pt",
                    optimizer,
                    value_head,
                    replay,
                    epoch + 1,
                    {"avg_scores": all_scores},
                )

        self.student.eval()
        if self._ref_model_cpu is not None:
            del self._ref_model_cpu
            self._ref_model_cpu = None

        return {
            "epochs_completed": epochs_done,
            "avg_scores": all_scores,
            "final_score": all_scores[-1] if all_scores else 0.0,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# R16 — GRPO: Group Relative Policy Optimization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class GRPOConfig:
    """Config for Group Relative Policy Optimization (R16).

    GRPO eliminates the value head entirely — it generates a *group*
    of N responses per prompt, scores them with a reward model (or
    external scorer), and uses the within-group ranking as the advantage
    signal.  No GAE, no value network, no critic head to train.
    """

    epochs: int = 3
    learning_rate: float = 1e-6
    kl_coeff: float = 0.05
    kl_target: float = 0.0
    kl_horizon: float = 10000.0
    clip_range: float = 0.2
    group_size: int = 4  # N responses generated per prompt
    max_new_tokens: int = 128
    temperature: float = 0.8
    gradient_clip: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "auto"
    max_prompt_length: int = 256
    normalize_advantages: bool = True
    entropy_coeff: float = 0.01
    ppo_epochs: int = 1  # update passes per rollout batch
    # Optimizer
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    checkpoint_dir: str = ""


class GRPOTrainer:
    """Group Relative Policy Optimization (R16).

    For each prompt, generates ``group_size`` responses, scores them
    with ``reward_fn``, and computes per-response advantage as the
    normalized deviation from the group mean reward.  No value head
    or GAE required.

    The policy gradient uses a clipped surrogate objective identical
    to PPO, but advantages come from group ranking instead of a
    learned value function.

    Args:
        model: Policy model (Enigma).
        tokenizer: Tokenizer for encoding/decoding.
        reward_fn: Callable that takes ``(prompt_str, response_str)``
            and returns a float reward score.  Can be a RewardModel
            wrapper, an external API call, or any scoring function.
        config: GRPO training config.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        reward_fn: Callable[[str, str], float],
        config: GRPOConfig | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config or GRPOConfig()
        self.device = next(model.parameters()).device

        self.on_progress: Callable[[int, str], None] | None = None
        self._stop_requested = False
        self._lock = threading.Lock()
        self._kl_coeff = self.config.kl_coeff

    def _update_adaptive_kl(self, observed_kl: float) -> None:
        """Adjust ``_kl_coeff`` toward ``kl_target``."""
        target = self.config.kl_target
        if target <= 0:
            return
        proportional_error = (observed_kl - target) / target
        mult = 1.0 + proportional_error / self.config.kl_horizon
        self._kl_coeff = max(1e-6, self._kl_coeff * mult)

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def _should_stop(self) -> bool:
        with self._lock:
            return self._stop_requested

    def train(self, prompts: list[str]) -> dict[str, Any]:
        """Run GRPO training loop.

        Args:
            prompts: List of prompt strings.

        Returns:
            Dict with ``epochs_completed``, ``avg_rewards``, ``final_reward``.
        """
        cfg = self.config
        amp_dtype = _resolve_amp_dtype(cfg.amp_dtype)
        use_amp = cfg.use_amp and self.device.type == "cuda"

        # Frozen reference for KL
        ref_model = copy.deepcopy(self.model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
        )

        scaler = None
        if use_amp and amp_dtype == torch.float16:
            scaler = torch.amp.GradScaler("cuda")

        avg_rewards: list[float] = []
        epochs_done = 0

        for epoch in range(cfg.epochs):
            if self._should_stop():
                break

            epoch_reward = 0.0
            n_scored = 0
            n_updates = 0
            random.shuffle(prompts)

            for prompt in prompts:
                if self._should_stop():
                    break

                prompt_ids = self.tokenizer.encode(prompt)
                prompt_ids = prompt_ids[: cfg.max_prompt_length]
                prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
                prompt_len = len(prompt_ids)

                # Generate a group of N responses
                group_gens: list[torch.Tensor] = []
                group_rewards: list[float] = []
                group_old_logps: list[torch.Tensor] = []

                self.model.eval()
                for _ in range(cfg.group_size):
                    with torch.no_grad():
                        try:
                            gen_ids = self.model.generate(
                                prompt_tensor,
                                max_new_tokens=cfg.max_new_tokens,
                                temperature=cfg.temperature,
                            )
                        except Exception:
                            continue

                    resp_ids = gen_ids[:, prompt_len:]
                    if resp_ids.shape[1] < 1:
                        continue

                    # Rollout-policy log-probs, captured now for the PPO
                    # ratio denominator (the behavior policy that generated
                    # this response), NOT the frozen reference.
                    with torch.no_grad():
                        old_logps = _get_response_logps(self.model, gen_ids, prompt_len)

                    resp_text = self.tokenizer.decode(resp_ids[0].tolist())
                    r = self.reward_fn(prompt, resp_text)
                    group_gens.append(gen_ids)
                    group_rewards.append(r)
                    group_old_logps.append(old_logps)

                self.model.train()

                if len(group_gens) < 2:
                    continue  # Need at least 2 for relative ranking

                # Compute group-relative advantages
                r_tensor = torch.tensor(group_rewards, dtype=torch.float32)
                if cfg.normalize_advantages:
                    r_mean = r_tensor.mean()
                    r_std = r_tensor.std(unbiased=False)
                    if r_std > 1e-6:
                        advantages = ((r_tensor - r_mean) / (r_std + 1e-8)).tolist()
                    else:
                        advantages = (r_tensor - r_mean).tolist()
                else:
                    advantages = r_tensor.tolist()

                epoch_reward += sum(group_rewards) / len(group_rewards)
                n_scored += 1

                # PPO-style update for each response in the group
                for ppo_epoch in range(cfg.ppo_epochs):
                    for gen_ids, adv, old_logps in zip(group_gens, advantages, group_old_logps):
                        full_ids = gen_ids
                        resp_len = full_ids.shape[1] - prompt_len
                        if resp_len < 1:
                            continue

                        optimizer.zero_grad()

                        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                            # Policy log-probs
                            logits = self.model(full_ids)
                            if hasattr(logits, "logits"):
                                logits = logits.logits
                            resp_logits = logits[:, prompt_len - 1 : -1, :]
                            resp_targets = full_ids[:, prompt_len:]
                            log_probs = F.log_softmax(resp_logits, dim=-1)
                            token_logps = log_probs.gather(2, resp_targets.unsqueeze(-1)).squeeze(-1)

                            # Reference log-probs (for the KL penalty only)
                            with torch.no_grad():
                                ref_logits = ref_model(full_ids)
                                if hasattr(ref_logits, "logits"):
                                    ref_logits = ref_logits.logits
                                ref_resp = ref_logits[:, prompt_len - 1 : -1, :]
                                ref_log_probs = F.log_softmax(ref_resp, dim=-1)
                                ref_token_logps = ref_log_probs.gather(2, resp_targets.unsqueeze(-1)).squeeze(-1)

                            # PPO ratio against the ROLLOUT policy: new / old,
                            # so drift from the frozen reference cannot clip
                            # the gradient to zero and stall learning.
                            log_ratio = (token_logps - old_logps).clamp(-20, 20)
                            ratio = log_ratio.exp()

                            adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)

                            surr1 = ratio * adv_t
                            surr2 = (
                                torch.clamp(
                                    ratio,
                                    1.0 - cfg.clip_range,
                                    1.0 + cfg.clip_range,
                                )
                                * adv_t
                            )
                            policy_loss = -torch.min(surr1, surr2).mean()

                            # KL penalty (single-sample estimate, consistent
                            # with RLHF/SelfPlay: KL(policy||ref) ≈ mean(log π - log π_ref))
                            kl = (token_logps - ref_token_logps).mean()
                            kl_penalty = self._kl_coeff * kl

                            # Entropy bonus
                            entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
                            ent_bonus = cfg.entropy_coeff * entropy

                            loss = policy_loss + kl_penalty - ent_bonus

                        if scaler is not None:
                            scaler.scale(loss).backward()
                            if cfg.gradient_clip > 0:
                                scaler.unscale_(optimizer)
                                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            if cfg.gradient_clip > 0:
                                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                            optimizer.step()

                        n_updates += 1

                # Adaptive KL update
                if group_rewards:
                    # Approximate KL from mean log-ratio of last group
                    with torch.no_grad():
                        last_gen = group_gens[-1]
                        logits_p = self.model(last_gen)
                        if hasattr(logits_p, "logits"):
                            logits_p = logits_p.logits
                        logits_r = ref_model(last_gen)
                        if hasattr(logits_r, "logits"):
                            logits_r = logits_r.logits
                        lp = F.log_softmax(logits_p[:, prompt_len - 1 : -1, :], dim=-1)
                        lr = F.log_softmax(logits_r[:, prompt_len - 1 : -1, :], dim=-1)
                        tgts = last_gen[:, prompt_len:]
                        obs_kl = (
                            (
                                lr.gather(2, tgts.unsqueeze(-1)).squeeze(-1)
                                - lp.gather(2, tgts.unsqueeze(-1)).squeeze(-1)
                            )
                            .mean()
                            .item()
                        )
                    self._update_adaptive_kl(abs(obs_kl))

            # Average over prompts that actually scored (produced a group
            # of >= 2 responses), not every prompt including skipped ones.
            avg_r = epoch_reward / max(n_scored, 1)
            avg_rewards.append(avg_r)
            epochs_done = epoch + 1

            logger.info("GRPO epoch %d: avg_reward=%.3f, updates=%d", epoch + 1, avg_r, n_updates)

            if self.on_progress:
                pct = int((epoch + 1) / cfg.epochs * 100)
                self.on_progress(pct, f"GRPO epoch {epoch + 1}: reward={avg_r:.3f}")

        self.model.eval()
        del ref_model

        return {
            "epochs_completed": epochs_done,
            "avg_rewards": avg_rewards,
            "final_reward": avg_rewards[-1] if avg_rewards else 0.0,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# R33 — ReMax: Simplified REINFORCE with Mean-Reward Baseline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class ReMaxConfig:
    """Config for ReMax RL training (Li et al., R33).

    ReMax simplifies PPO by removing the value head entirely.
    Instead of GAE-computed advantages, it uses a simple
    mean-reward baseline:

        advantage_i = reward_i - mean(rewards in batch)

    This eliminates ~50% of trainable parameters (no value head)
    and the complexity of advantage estimation, while achieving
    comparable performance on language model alignment tasks.
    """

    epochs: int = 3
    learning_rate: float = 1e-6
    kl_coeff: float = 0.05
    kl_target: float = 0.0
    kl_horizon: float = 10000.0
    clip_range: float = 0.2
    n_responses: int = 4  # Responses per prompt for baseline estimation
    max_new_tokens: int = 128
    temperature: float = 0.8
    gradient_clip: float = 1.0
    use_amp: bool = True
    amp_dtype: str = "auto"
    max_prompt_length: int = 256
    entropy_coeff: float = 0.01
    # Optimizer
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    checkpoint_dir: str = ""


class ReMaxTrainer:
    """ReMax: REINFORCE with mean-reward baseline (R33).

    For each prompt, generates ``n_responses`` responses, scores them
    with ``reward_fn``, and uses the mean reward as the baseline.
    No value head, no GAE, no critic — just policy gradient with
    a simple variance-reduction baseline.

    The update rule is:
        loss = -sum_i[ (reward_i - mean_reward) * log_prob_i ]

    With optional clipping (same as PPO) and entropy bonus.

    Args:
        model: Policy model.
        tokenizer: Tokenizer for encoding/decoding.
        reward_fn: Callable (prompt, response) → float reward.
        config: ReMax config.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        reward_fn: Callable[[str, str], float],
        config: ReMaxConfig | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config or ReMaxConfig()
        self.device = next(model.parameters()).device

        self.on_progress: Callable[[int, str], None] | None = None
        self._stop_requested = False
        self._lock = threading.Lock()
        self._kl_coeff = self.config.kl_coeff

    def _should_stop(self) -> bool:
        with self._lock:
            return self._stop_requested

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def _update_adaptive_kl(self, mean_kl: float) -> None:
        """Same adaptive KL as R9."""
        if self.config.kl_target <= 0:
            return
        if mean_kl > self.config.kl_target * 1.5:
            self._kl_coeff *= 1.0 + 0.5 / self.config.kl_horizon
        elif mean_kl < self.config.kl_target / 1.5:
            self._kl_coeff *= 1.0 - 0.5 / self.config.kl_horizon
        self._kl_coeff = max(1e-6, min(10.0, self._kl_coeff))

    def train(self, prompts: list[str]) -> dict[str, Any]:
        """Run ReMax training loop.

        Args:
            prompts: List of prompt strings.

        Returns:
            Dict with ``epochs_completed``, ``avg_rewards``, ``final_reward``.
        """
        if not prompts:
            raise ValueError("No prompts provided for ReMax training")

        cfg = self.config
        amp_dtype = _resolve_amp_dtype(cfg.amp_dtype)
        use_amp = cfg.use_amp and self.device.type == "cuda"

        # Frozen reference for KL
        ref_model = copy.deepcopy(self.model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            eps=cfg.adam_eps,
        )

        scaler = None
        if use_amp and amp_dtype == torch.float16:
            scaler = torch.amp.GradScaler("cuda")

        avg_rewards: list[float] = []
        epochs_done = 0

        for epoch in range(cfg.epochs):
            if self._should_stop():
                break

            epoch_reward = 0.0
            n_scored = 0
            n_updates = 0
            random.shuffle(prompts)

            for prompt in prompts:
                if self._should_stop():
                    break

                prompt_ids = self.tokenizer.encode(prompt)
                prompt_ids = prompt_ids[: cfg.max_prompt_length]
                prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
                prompt_len = len(prompt_ids)

                # Generate N responses, capturing each one's rollout-policy
                # log-probs for the PPO ratio denominator.
                responses: list[tuple[torch.Tensor, float, torch.Tensor]] = []

                self.model.eval()
                for _ in range(cfg.n_responses):
                    with torch.no_grad():
                        try:
                            gen_ids = self.model.generate(
                                prompt_tensor,
                                max_new_tokens=cfg.max_new_tokens,
                                temperature=cfg.temperature,
                            )
                        except Exception:
                            continue

                    resp_ids = gen_ids[:, prompt_len:]
                    if resp_ids.shape[1] < 1:
                        continue

                    with torch.no_grad():
                        old_logps = _get_response_logps(self.model, gen_ids, prompt_len)

                    resp_text = self.tokenizer.decode(resp_ids[0].tolist())
                    r = self.reward_fn(prompt, resp_text)
                    responses.append((gen_ids, r, old_logps))

                self.model.train()

                if len(responses) < 2:
                    continue

                # Mean reward baseline
                rewards = [r for _, r, _ in responses]
                mean_reward = sum(rewards) / len(rewards)
                epoch_reward += mean_reward
                n_scored += 1

                # REINFORCE with mean-reward baseline and clipped surrogate
                for gen_ids, reward, old_logps in responses:
                    advantage = reward - mean_reward
                    full_ids = gen_ids
                    resp_len = full_ids.shape[1] - prompt_len
                    if resp_len < 1:
                        continue

                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        # Current policy log-probs + entropy — single combined pass
                        new_logps, _, entropy = _get_logps_hidden_entropy(self.model, full_ids, prompt_len)
                        # Reference log-probs for the KL penalty only
                        with torch.no_grad():
                            ref_logps = _get_response_logps(ref_model, full_ids, prompt_len)

                        # PPO ratio against the ROLLOUT policy (new / old), so
                        # sequential updates within the epoch cannot clip the
                        # gradient to zero once the policy drifts from ref.
                        log_ratio = (new_logps - old_logps).clamp(-20, 20)
                        ratio = torch.exp(log_ratio)

                        # Clipped surrogate (same as PPO)
                        adv_tensor = torch.full_like(ratio, advantage)
                        surr1 = ratio * adv_tensor
                        surr2 = (
                            torch.clamp(
                                ratio,
                                1.0 - cfg.clip_range,
                                1.0 + cfg.clip_range,
                            )
                            * adv_tensor
                        )
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # Entropy bonus (already computed in combined pass above)
                        entropy_bonus = entropy.mean()

                        # KL penalty: KL(policy||ref) ≈ mean(log π - log π_ref)
                        kl = (new_logps - ref_logps).mean()
                        kl_loss = self._kl_coeff * kl

                        loss = policy_loss - cfg.entropy_coeff * entropy_bonus + kl_loss

                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                        optimizer.step()

                    optimizer.zero_grad()
                    n_updates += 1

                    self._update_adaptive_kl(kl.item())

            # Average over prompts that actually scored (>= 2 responses)
            avg_r = epoch_reward / max(n_scored, 1)
            avg_rewards.append(avg_r)
            epochs_done = epoch + 1

            logger.info("ReMax epoch %d: avg_reward=%.3f, updates=%d", epoch + 1, avg_r, n_updates)

            if self.on_progress:
                pct = int((epoch + 1) / cfg.epochs * 100)
                self.on_progress(pct, f"ReMax epoch {epoch + 1}: reward={avg_r:.3f}")

        self.model.eval()
        del ref_model

        return {
            "epochs_completed": epochs_done,
            "avg_rewards": avg_rewards,
            "final_reward": avg_rewards[-1] if avg_rewards else 0.0,
        }
