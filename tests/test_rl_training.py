"""RL stack regressions: reward models must not touch the policy, and the
GRPO/ReMax PPO ratio must be against the rollout policy (so training moves
the weights instead of stalling once the policy drifts from the reference)."""

import torch

from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.core.rl_training import (
    GRPOConfig,
    GRPOTrainer,
    ProcessRewardModel,
    ReMaxConfig,
    ReMaxTrainer,
    RewardModel,
)


class _CharTokenizer:
    """Minimal tokenizer over a tiny fixed vocab for RL loop tests."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(c) % (self.vocab_size - 3)) for c in text][:16] or [2]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + (int(i) % 26)) for i in ids)


def _tiny_policy(vocab_size=32) -> Enigma:
    torch.manual_seed(0)
    return Enigma(
        ForgeConfig(
            vocab_size=vocab_size, dim=32, n_layers=2, n_heads=2,
            max_seq_len=32, dropout=0.0, use_gradient_checkpointing=False,
        )
    )


def _snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _params_changed(model, snapshot):
    return any(not torch.equal(p.detach(), snapshot[n]) for n, p in model.named_parameters())


def test_reward_model_does_not_freeze_the_policy():
    """Building a frozen RewardModel from a policy must leave the policy's
    own parameters trainable (they used to be shared by reference)."""
    policy = _tiny_policy()
    assert all(p.requires_grad for p in policy.parameters())
    RewardModel(policy, freeze_base=True)
    assert all(p.requires_grad for p in policy.parameters()), "policy was frozen by the reward model"


def test_process_reward_model_does_not_freeze_the_policy():
    policy = _tiny_policy()
    ProcessRewardModel(policy, freeze_base=True)
    assert all(p.requires_grad for p in policy.parameters())


def test_training_reward_head_does_not_mutate_policy():
    """Optimizing the reward head must not change the policy's weights."""
    policy = _tiny_policy()
    before = _snapshot(policy)
    rm = RewardModel(policy, freeze_base=True)
    opt = torch.optim.SGD([p for p in rm.parameters() if p.requires_grad], lr=0.1)
    ids = torch.randint(2, 32, (2, 8))
    loss = rm(ids).pow(2).mean()
    loss.backward()
    opt.step()
    assert not _params_changed(policy, before), "reward-head training mutated the policy"


def test_grpo_moves_the_policy_and_reports_finite_reward():
    """A full GRPO loop updates the policy weights (gradient really flows
    through the rollout-ratio surrogate) and returns finite rewards."""
    policy = _tiny_policy()
    tok = _CharTokenizer(policy.config.vocab_size)
    before = _snapshot(policy)
    cfg = GRPOConfig(epochs=1, group_size=3, max_new_tokens=4, learning_rate=1e-2,
                     use_amp=False, ppo_epochs=1, max_prompt_length=8)
    trainer = GRPOTrainer(policy, tok, lambda p, r: float(len(r)), cfg)
    result = trainer.train(["ab", "cd"])
    assert result["avg_rewards"] and all(r == r for r in result["avg_rewards"])  # finite
    assert _params_changed(policy, before), "GRPO did not update the policy"


def test_remax_moves_the_policy_and_reports_finite_reward():
    policy = _tiny_policy()
    tok = _CharTokenizer(policy.config.vocab_size)
    before = _snapshot(policy)
    cfg = ReMaxConfig(epochs=1, n_responses=3, max_new_tokens=4, learning_rate=1e-2,
                      use_amp=False, max_prompt_length=8)
    trainer = ReMaxTrainer(policy, tok, lambda p, r: float(len(r)), cfg)
    result = trainer.train(["ab", "cd"])
    assert result["avg_rewards"] and all(r == r for r in result["avg_rewards"])
    assert _params_changed(policy, before), "ReMax did not update the policy"


def test_rl_checkpoint_load_rejects_total_mismatch():
    """_load_state_or_fail hard-fails when a checkpoint matches nothing,
    rather than silently loading zero weights (checkpoint contract)."""
    import pytest

    from enigma_engine.core.rl_training import _load_state_or_fail

    policy = _tiny_policy()
    # A state dict whose keys match no parameter of the model
    bogus = {"totally.unrelated.key": torch.zeros(2, 2)}
    with pytest.raises(ValueError, match="matched none"):
        _load_state_or_fail(policy, bogus, "test")
    # A matching state dict loads without error
    _load_state_or_fail(policy, dict(policy.state_dict()), "test")
