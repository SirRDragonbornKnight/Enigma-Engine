"""
Mod Router - Central hub for mod connections with background training.

The router:
1. Accepts mod connections on port 9900
2. Routes messages between mods and the engine
3. Runs background training while mods operate
"""

from __future__ import annotations

import json
import logging
import queue
import socket
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Continuous-3 (Pass 156i7): repo-default anchor JSONL path. ModRouter
# wires this into BackgroundTrainer when the file exists, so anchor-set
# rehearsal is on by default for production users without a hand-edit.
# Resolved relative to this file (enigma_engine/router.py → repo root)
# so it works regardless of CWD.
_DEFAULT_ANCHOR_PATH = Path(__file__).resolve().parent.parent / "data" / "anchor_examples.jsonl"

# TEACH-1c (Pass 156z9bj): repo-default corrections JSONL path written
# by the GUI FIX button (`_record_correction_for_last_exchange` in
# gui_logic_chat.py). BackgroundTrainer ingests new rows from this file
# on every replay tick and feeds the user-authored `right_response` as
# a positive SFT example through the existing replay path. Same boot
# pattern as anchors: ModRouter auto-resolves the path when the file
# exists, otherwise None keeps the feature off without warn noise.
_DEFAULT_CORRECTIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "corrections.jsonl"

# Deferred torch import — loaded on first use to avoid 540 MB idle RAM
_torch = None
_torch_lock = threading.Lock()


def _ensure_torch() -> Any:
    """Import torch on first use."""
    global _torch
    if _torch is None:
        with _torch_lock:
            if _torch is None:
                import torch

                _torch = torch
    return _torch


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ModConnection:
    """Represents a connected mod."""

    mod_id: str
    name: str
    socket: socket.socket
    address: tuple
    capabilities: list[str] = field(default_factory=list)
    connected_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


@dataclass
class TrainingExample:
    """A single training example collected from conversations."""

    prompt: str
    response: str
    score: float = 1.0
    source: str = "chat"
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# TRAINING THREAD
# =============================================================================


class BackgroundTrainer(threading.Thread):
    """
    Background training thread that learns from collected examples.

    Runs continuously while the router is active, processing training
    examples in a queue without blocking the main application.

    Smart features (BT-B + BT-D):
    - Replay buffer: rolling FIFO collection of recent examples
      (``deque(maxlen=N)``). Periodic retraining sorts by score and
      rehearses the top half.
    - Anchor set (Continuous-2, Pass 156i5): optional fixed set of
      curated examples loaded from JSONL on disk and rehearsed
      alongside the recent slice on every replay pass. Mitigates
      catastrophic forgetting of skills not present in recent chat.
      Forgetting is bounded by anchor coverage — a 50-example anchor
      set is a floor, not a guarantee. ``anchor_data_path=None``
      preserves legacy recent-only behaviour.
    - DPO pairs: can be populated externally for preference training.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        learning_rate: float = 1e-5,
        batch_size: int = 2,
        save_interval: int = 100,
        checkpoint_dir: str = "models/checkpoints/router_training",
        replay_buffer_size: int = 1000,
        retrain_interval: int = 200,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        max_token_length: int = 4096,
        anchor_data_path: str | Path | None = None,
        anchor_idle_interval_seconds: float | None = None,
        corrections_path: str | Path | None = None,
        dpo_min_pairs: int = 50,
        dpo_beta: float = 0.1,
        dpo_loss_type: str = "dpo",
    ):
        super().__init__(daemon=True)
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.adam_betas = adam_betas
        self.adam_eps = adam_eps
        # Continuous-1 (Pass 156i4): hard cap on per-example token length.
        # Without this, a misbehaving mod or pathological chat input can
        # push a 1M-token sequence through the model and OOM the GPU.
        # 4096 matches the typical max_seq_len budget; oversized samples
        # are skipped (logged at DEBUG) rather than truncated to avoid
        # silent loss-of-context.
        self.max_token_length = int(max_token_length)

        # System prompt for context (set from PromptTab)
        self.system_prompt: str = ""

        # Training state
        self.example_queue: queue.Queue[TrainingExample] = queue.Queue()
        self.examples_processed = 0
        self.total_loss = 0.0
        self.running = False
        self.paused = False

        # Lock to prevent train/eval mode conflicts with inference
        self._train_lock = threading.Lock()

        # Optimizer (created when model is set)
        self.optimizer: Any | None = None

        # Callbacks
        self.on_progress: Callable[[int, float], None] | None = None
        self.on_checkpoint: Callable[[str], None] | None = None
        self.inference_idle_check: Callable[[], bool] | None = None
        self.idle_poll_interval = 0.2

        # --- Smart features (BT-B + BT-D) ---

        # Replay buffer: capped at max size, keeps recent examples
        # for periodic retraining. Mitigates — does NOT prevent —
        # catastrophic forgetting: only skills present in the recent
        # window get rehearsed, so anything outside that window can
        # still drift. See `anchor_data_path` below for the curated
        # floor that closes the gap.
        self.replay_buffer_size: int = replay_buffer_size
        self.replay_buffer: deque[TrainingExample] = deque(maxlen=replay_buffer_size)
        self._replay_lock = threading.Lock()

        # DPO preference pairs: (rejected_example, chosen_response)
        # Can be populated externally for preference training.
        self.dpo_pairs: list[dict[str, str]] = []
        self.dpo_min_pairs: int = max(1, int(dpo_min_pairs))
        self.dpo_beta: float = float(dpo_beta)
        self.dpo_loss_type: str = str(dpo_loss_type)

        # Retrain on replay buffer every N examples processed
        self.retrain_interval: int = retrain_interval

        # Continuous-2 (Pass 156i5): optional anchor-set rehearsal.
        # JSONL file with {"prompt": ..., "response": ..., "score"?: ...}
        # rows. Rehearsed in addition to the score-sorted recent slice
        # every replay pass to mitigate catastrophic forgetting of
        # skills absent from recent chat. Loaded lazily on first
        # replay; missing/unreadable file logs WARNING and falls back
        # to recent-only behaviour.
        self.anchor_data_path: Path | None = Path(anchor_data_path) if anchor_data_path else None
        self._anchor_examples: list[TrainingExample] | None = None
        self._anchor_load_attempted: bool = False

        # Continuous-3c (Pass 156w): anchor-only idle rehearsal.
        # When set to a positive number, the run() loop fires
        # `_retrain_on_replay()` from its idle branch every N seconds
        # of wall time since the last replay, so anchor coverage keeps
        # firing during true quiet periods (no recent chat activity)
        # which is exactly when the per-batch trigger inside
        # `_train_batch` cannot fire. Default None = disabled
        # (preserves existing behaviour for users who don't opt in).
        # 0 / negative inputs collapse to None defensively.
        self.anchor_idle_interval_seconds: float | None = (
            float(anchor_idle_interval_seconds)
            if anchor_idle_interval_seconds and float(anchor_idle_interval_seconds) > 0
            else None
        )
        self._last_anchor_replay_at: float = time.monotonic()

        # TEACH-1c (Pass 156z9bj): corrections-as-SFT ingestion. The
        # GUI FIX button writes one row per correction to the file at
        # `corrections_path`; `ingest_corrections_file()` reads new
        # rows on each replay tick and feeds the `right_response` as
        # a positive `TrainingExample(source="correction")` through
        # the existing `add_example()` path. Idempotent within a
        # process via a byte-offset cursor — restart re-ingests, which
        # is bounded by `replay_buffer_size`.
        #
        # TEACH-1d (Pass 156z9bk): when wrong_response is present, the
        # same ingestion pass also appends one DPO pair
        # {prompt, chosen=right_response, rejected=wrong_response} to
        # `self.dpo_pairs`. `_maybe_train_dpo_pairs()` consumes this
        # list in thresholded batches and calls Trainer.train_dpo().
        # None disables file ingestion.
        self.corrections_path: Path | None = Path(corrections_path) if corrections_path else None
        self._corrections_offset: int = 0

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def set_model(self, model, tokenizer) -> None:
        """Set or update the model to train."""
        with self._train_lock:
            self.model = model
            self.tokenizer = tokenizer
            if model is not None:
                torch = _ensure_torch()
                self.optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.learning_rate,
                    weight_decay=0.01,
                    betas=self.adam_betas,
                    eps=self.adam_eps,
                )
                # Keep model in eval mode — _train_batch switches to
                # train mode only inside the lock and restores eval after.
                model.eval()
                logger.info("BackgroundTrainer: Model set for training")

    def add_example(self, prompt: str, response: str, score: float = 1.0, source: str = "chat") -> None:
        """Add a training example to the queue and replay buffer.

        All examples are queued for training and stored in the
        replay buffer (capped by recency).
        """
        example = TrainingExample(prompt=prompt, response=response, score=score, source=source)

        # Add to training queue
        self.example_queue.put(example)

        # Add to replay buffer (deque auto-trims to maxlen)
        with self._replay_lock:
            self.replay_buffer.append(example)

        logger.debug(
            "Added training example (queue=%d, replay=%d)", self.example_queue.qsize(), len(self.replay_buffer)
        )

    def _inference_busy(self) -> bool:
        """Return True when inference is currently active.

        Fail-open on checker errors so background training does not wedge.
        """
        checker = self.inference_idle_check
        if checker is None:
            return False
        try:
            return not bool(checker())
        except Exception as exc:
            logger.debug("inference_idle_check failed: %s", exc)
            return False

    def _should_run_anchor_idle_replay(self) -> bool:
        """Continuous-3c: gate for the idle anchor scheduler.

        Returns True iff every condition for firing an anchor-only
        rehearsal pass is met right now:

        - feature is opted in (``anchor_idle_interval_seconds`` set)
        - anchors are configured (``anchor_data_path`` not None) —
          the WHOLE POINT of this scheduler is anchor coverage; if
          there are no anchors there is nothing to rehearse and
          firing would be a wasted wakeup
        - trainer is running and not paused
        - model + optimizer are wired (otherwise replay is a no-op)
        - inference is not busy (chat takes priority over background
          rehearsal — same contract as the per-batch path)
        - the configured interval has elapsed since the last replay
          (whether triggered by ``_train_batch`` or by the idle path)

        Throttle by elapsed time, not by absolute schedule, so that
        a burst of regular replays naturally pushes the next idle
        replay further out instead of double-firing.
        """
        if self.anchor_idle_interval_seconds is None:
            return False
        if self.anchor_data_path is None:
            return False
        if self.paused or not self.running:
            return False
        if self.model is None or self.optimizer is None:
            return False
        if self._inference_busy():
            return False
        elapsed = time.monotonic() - self._last_anchor_replay_at
        return elapsed >= self.anchor_idle_interval_seconds

    def run(self) -> None:
        self.running = True
        batch: list[TrainingExample] = []

        logger.info("BackgroundTrainer started")

        while self.running:
            # Check if paused
            if self.paused:
                time.sleep(0.5)
                continue

            # Check if model is available
            if self.model is None or self.tokenizer is None:
                time.sleep(1.0)
                continue

            # Defer training work while inference is active.
            if self._inference_busy():
                time.sleep(self.idle_poll_interval)
                continue

            # Collect batch
            try:
                example = self.example_queue.get(timeout=1.0)
                batch.append(example)
            except queue.Empty:
                # No examples available
                if batch:
                    # Process partial batch after timeout
                    self._train_batch(batch)
                    batch = []
                # Continuous-3c (Pass 156w): true-idle anchor rehearsal.
                # During quiet periods the per-batch trigger inside
                # `_train_batch` cannot fire (no recent chat to call it
                # with), so anchors never rehearse — the exact case
                # they exist for. Fire `_retrain_on_replay()` here on
                # an elapsed-time gate. Off by default; opt-in via
                # `anchor_idle_interval_seconds` constructor kwarg.
                if self._should_run_anchor_idle_replay():
                    elapsed = time.monotonic() - self._last_anchor_replay_at
                    logger.info(
                        "BackgroundTrainer: idle anchor rehearsal (%.0fs since last replay)",
                        elapsed,
                    )
                    self._retrain_on_replay()
                continue

            # Train when batch is full
            if len(batch) >= self.batch_size:
                self._train_batch(batch)
                batch = []

        logger.info("BackgroundTrainer stopped")

    def _train_batch(self, batch: list[TrainingExample]) -> None:
        """Train on a batch of examples.

        Trains on all examples equally.
        Triggers replay retrain periodically (BT-D).
        """
        if not batch or self.model is None:
            return

        try:
            with self._train_lock:
                self.model.train()
                try:
                    total_batch_loss = 0.0

                    # Forward/backward all examples, single optimizer step
                    if self.optimizer is None:
                        logger.warning("No optimizer configured — skipping training batch")
                        return
                    self.optimizer.zero_grad()
                    valid_count = 0
                    batch_len = len(batch)

                    for example in batch:
                        # Prepare input with optional system prompt context
                        if self.system_prompt:
                            text = f"System: {self.system_prompt}\n\nUser: {example.prompt}\n\nAssistant: {example.response}"
                        else:
                            text = f"User: {example.prompt}\n\nAssistant: {example.response}"

                        # Tokenize
                        if hasattr(self.tokenizer, "encode"):
                            tokens = self.tokenizer.encode(text)
                        else:
                            tokens = self.tokenizer(text)

                        if not tokens or len(tokens) < 2:
                            continue

                        # Continuous-1 (Pass 156i4): hard cap on token length
                        # — protects GPU from oversized inputs.
                        if len(tokens) > self.max_token_length:
                            logger.debug(
                                "BackgroundTrainer: skipping oversize example (%d tokens > cap %d)",
                                len(tokens),
                                self.max_token_length,
                            )
                            continue

                        # Convert to tensor
                        torch = _ensure_torch()
                        device = next(self.model.parameters()).device
                        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
                        target_ids = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

                        # Forward pass
                        output = self.model(input_ids)
                        # Unpack tuple if model returns (logits, loss)
                        logits = output[0] if isinstance(output, tuple) else output

                        # Calculate loss
                        loss = _ensure_torch().nn.functional.cross_entropy(
                            logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
                        )

                        # Continuous-1 (Pass 156i4): silent-drift guard.
                        # NaN/Inf loss → skip backward and do NOT increment
                        # valid_count. The optimizer call below is gated
                        # on valid_count > 0, so a fully-NaN batch becomes
                        # a no-op rather than corrupting weights with NaN
                        # gradients (which would propagate to every future
                        # update — the silent-over-months failure mode).
                        if not _ensure_torch().isfinite(loss):
                            logger.warning(
                                "BackgroundTrainer: non-finite loss (%s) on "
                                "example from source=%s — skipping backward; "
                                "check input quality",
                                loss.item(),
                                example.source,
                            )
                            continue

                        # Backward — scale for gradient accumulation
                        (loss / batch_len).backward()

                        total_batch_loss += loss.item()
                        valid_count += 1

                    # Single optimizer step after all examples
                    if valid_count > 0:
                        _ensure_torch().nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                finally:
                    # Always restore eval mode — even on exception — so
                    # dropout / batchnorm don't corrupt inference output.
                    self.model.eval()

            # Update stats
            self.examples_processed += len(batch)
            avg_loss = total_batch_loss / max(1, len(batch))
            self.total_loss = 0.9 * self.total_loss + 0.1 * avg_loss  # EMA

            # Callback
            if self.on_progress:
                self.on_progress(self.examples_processed, self.total_loss)

            # Periodic checkpoint
            if self.examples_processed % self.save_interval == 0:
                self._save_checkpoint()

            # Periodic retrain on replay buffer (BT-D)
            if (
                self.retrain_interval > 0
                and self.examples_processed % self.retrain_interval == 0
                and len(self.replay_buffer) >= self.batch_size
            ):
                self._retrain_on_replay()

            logger.debug(
                "Trained batch: %d examples, loss=%.4f, total=%d", len(batch), avg_loss, self.examples_processed
            )

        except Exception as e:
            logger.error("Training batch error: %s\n%s", e, traceback.format_exc())

    def _load_anchor_examples(self) -> list[TrainingExample]:
        """Load and cache anchor examples from JSONL on first call.

        Returns an empty list when no anchor path is configured, the
        file is missing, or the file cannot be read. WARNING is logged
        once on configuration errors so a misconfigured path does not
        silently degrade to recent-only replay (loud-on-real-issue).
        """
        if self._anchor_load_attempted:
            return self._anchor_examples or []
        self._anchor_load_attempted = True

        if self.anchor_data_path is None:
            self._anchor_examples = []
            return []

        if not self.anchor_data_path.exists():
            logger.warning(
                "BackgroundTrainer anchor set: file not found at %s — "
                "falling back to recent-only replay (forgetting bounded "
                "by recent-buffer coverage only)",
                self.anchor_data_path,
            )
            self._anchor_examples = []
            return []

        examples: list[TrainingExample] = []
        try:
            with self.anchor_data_path.open("r", encoding="utf-8") as fh:
                for line_no, raw in enumerate(fh, 1):
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        prompt = str(row.get("prompt", "")).strip()
                        response = str(row.get("response", "")).strip()
                        score = float(row.get("score", 1.0))
                    except (json.JSONDecodeError, ValueError, TypeError) as exc:
                        logger.debug(
                            "anchor JSONL line %d skipped: %s",
                            line_no,
                            exc,
                        )
                        continue
                    if prompt and response:
                        examples.append(
                            TrainingExample(
                                prompt=prompt,
                                response=response,
                                score=score,
                                source="anchor",
                            )
                        )
            if examples:
                logger.info(
                    "BackgroundTrainer: loaded %d anchor example(s) from %s",
                    len(examples),
                    self.anchor_data_path,
                )
            else:
                # Continuous-2a (Pass 156i6): file-present-zero-yield is
                # a real misconfiguration — loud, not silent INFO 0.
                logger.warning(
                    "BackgroundTrainer anchor set: file %s exists but "
                    "contains no usable rows (all empty/malformed) — "
                    "falling back to recent-only replay",
                    self.anchor_data_path,
                )
        except OSError as exc:
            logger.warning(
                "BackgroundTrainer anchor set: unreadable (%s) — falling back to recent-only replay",
                exc,
            )

        self._anchor_examples = examples
        return examples

    def ingest_corrections_file(
        self,
        path: Path | None = None,
    ) -> int:
        """TEACH-1c: read new correction rows as positive SFT examples.

        Reads the JSONL file at *path* (or ``self.corrections_path``)
        starting from the byte offset where the last call left off,
        parses each row, and queues a ``TrainingExample`` with
        ``response = right_response`` and ``source="correction"`` via
        the existing :meth:`add_example` path. The wrong_response is
        intentionally NOT used here — surfacing it as a DPO "rejected"
        example needs a reference model + beta + KL term and lives
        in :func:`Trainer.train_dpo`; that variant is parked as TEACH-1d.

        Idempotent within a process via the in-memory byte offset.
        Restart re-ingests every row, which is bounded by
        ``replay_buffer_size`` and acceptable for v1.

        Returns the number of new examples ingested. Missing file,
        empty file, or all-rows-malformed return 0 silently (no
        warn noise — this is a steady-state read, not a config error).
        """
        target = Path(path) if path is not None else self.corrections_path
        if target is None or not target.exists():
            return 0

        added = 0
        dpo_added = 0
        bad_rows = 0
        try:
            file_size = target.stat().st_size
            if file_size <= self._corrections_offset:
                # File was truncated/replaced — reset and re-read from start
                # so a hand-cleared corrections.jsonl does not stay invisible.
                if file_size < self._corrections_offset:
                    self._corrections_offset = 0
                else:
                    return 0

            with target.open("rb") as fh:
                fh.seek(self._corrections_offset)
                payload = fh.read()
                new_offset = self._corrections_offset + len(payload)

            text = payload.decode("utf-8", errors="replace")
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    bad_rows += 1
                    continue
                if not isinstance(row, dict):
                    bad_rows += 1
                    continue
                prompt = str(row.get("prompt", "")).strip()
                right = str(row.get("right_response", "")).strip()
                if not prompt or not right:
                    bad_rows += 1
                    continue
                self.add_example(
                    prompt=prompt,
                    response=right,
                    score=1.0,
                    source="correction",
                )
                wrong = str(row.get("wrong_response", "")).strip()
                if wrong:
                    with self._replay_lock:
                        self.dpo_pairs.append(
                            {
                                "prompt": prompt,
                                "chosen": right,
                                "rejected": wrong,
                            }
                        )
                    dpo_added += 1
                added += 1

            self._corrections_offset = new_offset
        except OSError as exc:
            logger.warning(
                "BackgroundTrainer: corrections file %s unreadable (%s)",
                target,
                exc,
            )
            return 0

        if added:
            logger.info(
                "BackgroundTrainer: ingested %d correction(s) + %d DPO pair(s) from %s (skipped %d malformed)",
                added,
                dpo_added,
                target,
                bad_rows,
            )
        elif bad_rows:
            logger.warning(
                "BackgroundTrainer: %d malformed row(s) in %s, no valid corrections ingested this tick",
                bad_rows,
                target,
            )
        return added

    def _create_dpo_trainer(self):
        """Build a one-epoch Trainer instance for DPO replay batches."""
        from enigma_engine.training.training import Trainer, TrainingConfig

        # Keep replay DPO small and stable by design: one epoch over the
        # queued correction pairs, no periodic checkpoints.
        cfg = TrainingConfig(
            epochs=1,
            batch_size=max(1, min(self.batch_size, len(self.dpo_pairs))),
            learning_rate=self.learning_rate,
            save_every=0,
            save_every_steps=0,
        )
        return Trainer(self.model, self.tokenizer, cfg)

    def _maybe_train_dpo_pairs(self) -> int:
        """Run correction-derived DPO training when threshold is met.

        Returns number of pairs consumed (0 when skipped or failed).
        """
        if self.model is None or self.tokenizer is None:
            return 0

        with self._replay_lock:
            if len(self.dpo_pairs) < self.dpo_min_pairs:
                return 0
            pairs = list(self.dpo_pairs)
            self.dpo_pairs.clear()

        try:
            with self._train_lock:
                trainer = self._create_dpo_trainer()
                trainer.train_dpo(
                    pairs,
                    beta=self.dpo_beta,
                    loss_type=self.dpo_loss_type,
                )
            logger.info(
                "BackgroundTrainer: DPO replay consumed %d pair(s)",
                len(pairs),
            )
            return len(pairs)
        except Exception as exc:
            with self._replay_lock:
                # Put pairs back at the front so we do not silently lose
                # user corrections on transient training failures.
                self.dpo_pairs = pairs + self.dpo_pairs
            logger.error(
                "BackgroundTrainer: DPO replay failed (%s)\n%s",
                exc,
                traceback.format_exc(),
            )
            return 0

    def _retrain_on_replay(self) -> None:
        """Retrain on the best examples in the replay buffer (BT-D).

        Takes a snapshot of the top examples from the replay buffer
        and (when configured) the curated anchor set, then runs a mini
        training pass. Mitigates catastrophic forgetting bounded by
        anchor coverage — recent-only replay reinforces what's already
        active, anchors reinforce skills absent from recent chat.
        """
        if self.model is None or self.optimizer is None:
            return

        # TEACH-1c (Pass 156z9bj): pull any new GUI-FIX corrections
        # into the replay buffer BEFORE the timestamp reset and the
        # batch snapshot below, so corrections written since the last
        # tick are eligible to train on this tick (not the next one).
        # No-op when corrections_path is None or the file does not
        # exist; offset-based so re-call is idempotent within process.
        try:
            self.ingest_corrections_file()
        except Exception as exc:
            logger.warning(
                "BackgroundTrainer: ingest_corrections_file failed: %s",
                exc,
            )

        # TEACH-1d (Pass 156z9bk): correction-derived DPO replay.
        # Thresholded by `dpo_min_pairs` so tiny correction batches do
        # not produce unstable preference updates every tick.
        self._maybe_train_dpo_pairs()

        # Continuous-3c audit (Pass 156x2): reset the idle scheduler's
        # cooldown clock at function ENTRY, not on the success path.
        # The empty-batch early-out (anchor file present but yields
        # zero usable rows + empty replay buffer) and the exception
        # path both need to honor the configured interval, otherwise
        # the idle scheduler spins at ~1 Hz logging "idle anchor
        # rehearsal" forever. Comment in the original Pass 156w
        # placement said "failed pass naturally retries" — wrong
        # design: retrying immediately won't fix a broken anchor
        # file, it just spams logs. Wait the full interval.
        self._last_anchor_replay_at = time.monotonic()

        with self._replay_lock:
            # Sort by score (descending) so we retrain on the best examples
            sorted_buf = sorted(self.replay_buffer, key=lambda x: x.score, reverse=True)
            top_k = min(len(sorted_buf), self.replay_buffer_size // 2)
            replay_batch = list(sorted_buf[:top_k])

        # Continuous-2 (Pass 156i5) + Continuous-2a (Pass 156i6):
        # Load anchors BEFORE the empty-batch early-out so anchor-only
        # rehearsal still fires during quiet periods (no recent chat).
        # Anchors are NOT score-sorted — curated order is intentional.
        # Extending replay_batch BEFORE replay_len is computed keeps
        # loss scaling consistent across the combined batch.
        anchor_examples = self._load_anchor_examples()
        if anchor_examples:
            replay_batch = list(replay_batch) + list(anchor_examples)
            logger.debug(
                "Replay batch: %d recent + %d anchor = %d total",
                len(replay_batch) - len(anchor_examples),
                len(anchor_examples),
                len(replay_batch),
            )

        if not replay_batch:
            return

        logger.info("BackgroundTrainer: retraining on %d replay examples", len(replay_batch))

        try:
            with self._train_lock:
                self.model.train()
                # Halve LR for replay to avoid over-fitting on
                # the same examples, then restore after the loop.
                orig_lrs = []
                for pg in self.optimizer.param_groups:
                    orig_lrs.append(pg["lr"])
                    pg["lr"] = pg["lr"] * 0.5
                try:
                    self.optimizer.zero_grad()
                    valid_count = 0
                    replay_len = len(replay_batch)

                    for example in replay_batch:
                        if self.system_prompt:
                            text = (
                                f"System: {self.system_prompt}\n\n"
                                f"User: {example.prompt}\n\n"
                                f"Assistant: {example.response}"
                            )
                        else:
                            text = f"User: {example.prompt}\n\nAssistant: {example.response}"

                        if hasattr(self.tokenizer, "encode"):
                            tokens = self.tokenizer.encode(text)
                        else:
                            tokens = self.tokenizer(text)

                        if not tokens or len(tokens) < 2:
                            continue

                        # Continuous-1 (Pass 156i4): same length cap as
                        # _train_batch — keep the two paths in sync.
                        if len(tokens) > self.max_token_length:
                            logger.debug(
                                "BackgroundTrainer replay: skipping oversize example (%d tokens > cap %d)",
                                len(tokens),
                                self.max_token_length,
                            )
                            continue

                        torch = _ensure_torch()
                        device = next(self.model.parameters()).device
                        input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
                        target_ids = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

                        output = self.model(input_ids)
                        logits = output[0] if isinstance(output, tuple) else output
                        loss = torch.nn.functional.cross_entropy(
                            logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
                        )

                        # Continuous-1 (Pass 156i4): silent-drift guard
                        # mirrors _train_batch — NaN/Inf loss skips the
                        # backward; if every replay sample is bad, the
                        # `if valid_count > 0` gate below blocks the
                        # optimizer update entirely.
                        if not torch.isfinite(loss):
                            logger.warning(
                                "BackgroundTrainer replay: non-finite loss "
                                "(%s) on example from source=%s — skipping "
                                "backward",
                                loss.item(),
                                example.source,
                            )
                            continue

                        (loss / replay_len).backward()
                        valid_count += 1

                    if valid_count > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                finally:
                    # Restore original LR and eval mode
                    if len(orig_lrs) != len(self.optimizer.param_groups):
                        logger.warning(
                            "Param group count changed during replay (%d -> %d), skipping LR restore",
                            len(orig_lrs),
                            len(self.optimizer.param_groups),
                        )
                    else:
                        for pg, lr in zip(self.optimizer.param_groups, orig_lrs):
                            pg["lr"] = lr
                    self.model.eval()

            logger.info("BackgroundTrainer: replay retrain complete")

        except Exception as e:
            logger.error("Replay retrain error: %s\n%s", e, traceback.format_exc())

    def _save_checkpoint(self) -> None:
        """Save a training checkpoint."""
        if self.model is None:
            return

        checkpoint_path = self.checkpoint_dir / f"router_ckpt_{self.examples_processed}.pth"

        try:
            from enigma_engine.core.safe_save import atomic_torch_save

            save_data = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                "examples_processed": self.examples_processed,
                "total_loss": self.total_loss,
                "replay_buffer_size": len(self.replay_buffer),
            }
            if hasattr(self.model, "config"):
                c = self.model.config
                cfg_dict = {
                    "vocab_size": c.vocab_size,
                    "dim": c.dim,
                    "n_layers": c.n_layers,
                    "n_heads": c.n_heads,
                    "n_kv_heads": c.n_kv_heads,
                    "hidden_dim": c.hidden_dim,
                    "max_seq_len": c.max_seq_len,
                    "dropout": c.dropout,
                    "use_rope": c.use_rope,
                    "use_moe": c.use_moe,
                }
                save_data["model_config"] = cfg_dict
                save_data["config"] = cfg_dict
            atomic_torch_save(save_data, checkpoint_path)

            logger.info(f"Saved checkpoint: {checkpoint_path}")

            if self.on_checkpoint:
                self.on_checkpoint(str(checkpoint_path))

        except Exception as e:
            logger.error("Failed to save checkpoint: %s\n%s", e, traceback.format_exc())

    def pause(self) -> None:
        """Pause training."""
        self.paused = True
        logger.info("BackgroundTrainer paused")

    def resume(self) -> None:
        """Resume training."""
        self.paused = False
        logger.info("BackgroundTrainer resumed")

    def stop(self) -> None:
        """Stop the training thread."""
        self.running = False

    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            "running": self.running,
            "paused": self.paused,
            "examples_processed": self.examples_processed,
            "queue_size": self.example_queue.qsize(),
            "average_loss": self.total_loss,
            "has_model": self.model is not None,
            "replay_buffer_size": len(self.replay_buffer),
            "replay_buffer_max": self.replay_buffer_size,
            "dpo_pairs": len(self.dpo_pairs),
        }

    @property
    def train_lock(self) -> threading.Lock:
        """Expose the training lock so inference code can coordinate.

        Inference callers should ``with trainer.train_lock:`` around
        their forward pass to prevent train/eval mode interleaving
        and gradient corruption when the same model is shared.
        """
        return self._train_lock


# =============================================================================
# MOD ROUTER
# =============================================================================


class ModRouter:
    """
    Central router for mod connections.

    Handles:
    - TCP server on port 9900
    - Mod connection management
    - Message routing between mods and engine
    - Background training from conversations
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9900,
        enable_training: bool = True,
        heartbeat_interval: float = 30.0,
        max_connections: int = 50,
        anchor_data_path: str | Path | None = None,
    ):
        self.host = host
        self.port = port
        self.enable_training = enable_training
        self.heartbeat_interval = heartbeat_interval
        self.max_connections = max_connections

        # Continuous-3 (Pass 156i7): resolve anchor path. Explicit
        # caller value wins; otherwise auto-pick the repo default if it
        # exists; otherwise None (recent-only replay, no warn noise).
        if anchor_data_path is None:
            anchor_data_path = _DEFAULT_ANCHOR_PATH if _DEFAULT_ANCHOR_PATH.exists() else None
        self.anchor_data_path: str | Path | None = anchor_data_path

        # Server state
        self.server_socket: socket.socket | None = None
        self.running = False
        self.accept_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None

        # Connected mods
        self.mods: dict[str, ModConnection] = {}
        self.mod_lock = threading.Lock()

        # Multi-purpose prompts for different contexts
        self.prompts: dict[str, str] = {
            "chat": "You are a helpful AI assistant.",
            "gui_usage": "You can control the application using [CMD]command[/CMD] blocks.",
            "training_scorer": "Score this response from 1-100 based on helpfulness, accuracy, and clarity.",
            "mod_router": "Route tasks to the appropriate mod based on the request type.",
        }

        # Training
        self._train_lock = threading.Lock()
        self._inference_idle_check: Callable[[], bool] | None = None
        self.trainer = self._create_trainer() if enable_training else None

        # Callbacks
        self.on_mod_connected: Callable[[ModConnection], None] | None = None
        self.on_mod_disconnected: Callable[[str], None] | None = None
        self.on_message: Callable[[str, dict], None] | None = None

        # Message handlers
        self.message_handlers: dict[str, Callable] = {}

    def _create_trainer(self) -> BackgroundTrainer:
        """Create a trainer configured with current router callbacks."""
        # TEACH-1c (Pass 156z9bj): forward repo-default corrections path
        # when the file exists. Same boot pattern as anchor_data_path —
        # library default stays None (test isolation, explicit opt-in)
        # while the production boot site auto-resolves.
        corrections_path = _DEFAULT_CORRECTIONS_PATH if _DEFAULT_CORRECTIONS_PATH.exists() else None
        trainer = BackgroundTrainer(
            anchor_data_path=self.anchor_data_path,
            corrections_path=corrections_path,
        )
        trainer.inference_idle_check = self._inference_idle_check
        return trainer

    def start(self) -> bool:
        """Start the router server."""
        if self.running:
            logger.warning("Router already running")
            return False

        try:
            # Create server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(10)
            self.server_socket.settimeout(1.0)  # For clean shutdown

            self.running = True

            # Start accept thread
            self.accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
            self.accept_thread.start()

            # Start training thread
            if self.trainer:
                self.trainer.start()

            # Start heartbeat thread
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()

            logger.info(f"Router started on {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start router: {e}")
            self.running = False
            return False

    def stop(self) -> None:
        """Stop the router server."""
        self.running = False

        # Stop trainer
        if self.trainer:
            self.trainer.stop()

        # Close all mod connections
        with self.mod_lock:
            for mod in list(self.mods.values()):
                try:
                    mod.socket.close()
                except Exception as e:
                    logger.debug("Error closing mod socket: %s", e)
            self.mods.clear()

        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception as e:
                logger.debug("Error closing server socket: %s", e)
            self.server_socket = None

        logger.info("Router stopped")

    def _heartbeat_loop(self) -> None:
        """Periodically ping mods and remove dead connections."""
        while self.running:
            time.sleep(self.heartbeat_interval)
            if not self.running:
                break
            # Snapshot mods to ping outside the lock — socket I/O can block
            to_ping: list[tuple[str, "ModConnection"]] = []
            dead: list[str] = []
            with self.mod_lock:
                now = time.time()
                for mod_id, mod in self.mods.items():
                    if now - mod.last_seen > self.heartbeat_interval * 3:
                        dead.append(mod_id)
                    else:
                        to_ping.append((mod_id, mod))
            # Ping outside the lock to avoid blocking other mod ops
            for mod_id, mod in to_ping:
                try:
                    self._send_message(mod.socket, {"type": "ping"})
                except Exception:
                    dead.append(mod_id)
            # Remove dead mods under lock
            with self.mod_lock:
                for mod_id in dead:
                    mod = self.mods.pop(mod_id, None)
                    if mod:
                        try:
                            mod.socket.close()
                        except Exception:
                            pass
            for mod_id in dead:
                logger.info(f"Heartbeat: removed dead mod {mod_id}")
                if self.on_mod_disconnected:
                    self.on_mod_disconnected(mod_id)

    def _accept_loop(self) -> None:
        """Accept incoming mod connections."""
        while self.running:
            try:
                # Reject when at capacity
                with self.mod_lock:
                    at_capacity = len(self.mods) >= self.max_connections
                if at_capacity:
                    time.sleep(0.5)
                    continue

                client_socket, address = self.server_socket.accept()
                logger.info(f"New connection from {address}")

                # Start handler thread
                handler = threading.Thread(target=self._handle_mod, args=(client_socket, address), daemon=True)
                handler.start()

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Accept error: {e}")

    def _handle_mod(self, client_socket: socket.socket, address: tuple) -> None:
        """Handle a connected mod."""
        mod_id = None

        try:
            client_socket.settimeout(30.0)  # 30s timeout for registration

            # Wait for registration message
            data = self._receive_message(client_socket)
            if not data:
                logger.warning(f"No registration from {address}")
                client_socket.close()
                return

            if data.get("type") != "register":
                logger.warning(f"Expected register, got {data.get('type')}")
                client_socket.close()
                return

            # Create mod connection
            mod_id = data.get("mod_id", f"mod_{time.time()}")

            mod = ModConnection(
                mod_id=mod_id,
                name=data.get("name", "Unknown Mod"),
                socket=client_socket,
                address=address,
                capabilities=data.get("capabilities", []),
            )

            # Store mod
            with self.mod_lock:
                self.mods[mod_id] = mod

            # Send acknowledgment
            self._send_message(client_socket, {"type": "registered", "mod_id": mod_id, "status": "ok"})

            logger.info(f"Mod registered: {mod.name} ({mod_id})")

            if self.on_mod_connected:
                self.on_mod_connected(mod)

            # Set normal timeout
            client_socket.settimeout(60.0)

            # Message loop
            while self.running:
                data = self._receive_message(client_socket)
                if data is None:
                    break

                self._handle_message(mod_id, data)

        except socket.timeout:
            logger.debug(f"Mod timeout: {mod_id or address}")
        except ConnectionResetError:
            logger.debug(f"Mod disconnected: {mod_id or address}")
        except Exception as e:
            logger.error(f"Mod handler error: {e}")
        finally:
            # Cleanup
            if mod_id:
                with self.mod_lock:
                    self.mods.pop(mod_id, None)

                if self.on_mod_disconnected:
                    self.on_mod_disconnected(mod_id)

                logger.info(f"Mod disconnected: {mod_id}")

            try:
                client_socket.close()
            except Exception:
                pass

    def _handle_message(self, mod_id: str, data: dict) -> None:
        """Handle a message from a mod."""
        msg_type = data.get("type", "unknown")

        # Check for registered handler
        if msg_type in self.message_handlers:
            try:
                self.message_handlers[msg_type](mod_id, data)
            except Exception as e:
                logger.error(f"Handler error for {msg_type}: {e}")
            return

        # Default handling
        if msg_type == "response":
            # Mod completed a task
            prompt = data.get("prompt", "")
            response = data.get("response", "")
            score = data.get("score", 1.0)

            # Add to training queue
            if self.trainer and prompt and response:
                self.trainer.add_example(prompt, response, score, source=f"mod:{mod_id}")

        elif msg_type == "ping":
            # Respond to ping
            with self.mod_lock:
                mod = self.mods.get(mod_id)
            if mod:
                mod.last_seen = time.time()
                self._send_message(mod.socket, {"type": "pong"})

        elif msg_type == "pong":
            # Heartbeat reply — update last_seen
            with self.mod_lock:
                mod = self.mods.get(mod_id)
            if mod:
                mod.last_seen = time.time()

        # Callback
        if self.on_message:
            self.on_message(mod_id, data)

    def _receive_message(self, sock: socket.socket) -> dict | None:
        """Receive a JSON message."""
        try:
            deadline = time.monotonic() + 30.0  # aggregate timeout

            # Read length prefix (4 bytes)
            length_data = b""
            while len(length_data) < 4:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning("Message receive timed out (header)")
                    return None
                sock.settimeout(min(remaining, 60.0))
                chunk = sock.recv(4 - len(length_data))
                if not chunk:
                    return None
                length_data += chunk

            length = int.from_bytes(length_data, "big")

            if length > 1_000_000:  # 1MB max
                try:
                    client_ip = sock.getpeername()[0]
                except Exception:
                    client_ip = "unknown"
                logger.warning("Message too large from %s: %d bytes", client_ip, length)
                return None

            # Read message
            data = b""
            while len(data) < length:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning("Message receive timed out (body)")
                    return None
                sock.settimeout(min(remaining, 60.0))
                chunk = sock.recv(min(4096, length - len(data)))
                if not chunk:
                    return None
                data += chunk

            return json.loads(data.decode("utf-8"))

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            return None
        except Exception as e:
            logger.debug(f"Receive error: {e}")
            return None

    def _send_message(self, sock: socket.socket, data: dict) -> bool:
        """Send a JSON message."""
        try:
            msg = json.dumps(data).encode("utf-8")
            length = len(msg).to_bytes(4, "big")
            sock.sendall(length + msg)
            return True
        except Exception as e:
            logger.error(f"Send error: {e}")
            return False

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def send_to_mod(self, mod_id: str, message: dict) -> bool:
        """Send a message to a specific mod."""
        with self.mod_lock:
            mod = self.mods.get(mod_id)
            if mod:
                return self._send_message(mod.socket, message)
        return False

    def broadcast(self, message: dict, exclude: list[str] | None = None) -> None:
        """Broadcast a message to all connected mods."""
        exclude_set = set(exclude or [])
        with self.mod_lock:
            targets = [(mod_id, mod) for mod_id, mod in self.mods.items() if mod_id not in exclude_set]
        for _mod_id, mod in targets:
            self._send_message(mod.socket, message)

    def get_connected_mods(self) -> list[dict]:
        """Get list of connected mods."""
        with self.mod_lock:
            return [
                {
                    "mod_id": mod.mod_id,
                    "name": mod.name,
                    "capabilities": mod.capabilities,
                    "connected_at": mod.connected_at,
                    "uptime": time.time() - mod.connected_at,
                }
                for mod in self.mods.values()
            ]

    def add_training_example(self, prompt: str, response: str, score: float = 1.0) -> None:
        """Add a training example from chat."""
        if self.trainer:
            self.trainer.add_example(prompt, response, score, source="chat")

    def set_training_enabled(self, enabled: bool) -> None:
        """Enable or disable the background trainer at runtime."""
        with self._train_lock:
            if enabled:
                if self.trainer is not None:
                    return
                self.trainer = self._create_trainer()
                if self.running:
                    self.trainer.start()
                return

            trainer = self.trainer
            if trainer is None:
                return
            self.trainer = None
        trainer.stop()

    def set_inference_idle_check(self, checker: Callable[[], bool] | None) -> None:
        """Set a callback that reports whether inference is idle.

        Callback should return True when inference is idle/safe for training,
        False when inference is active and training should defer.
        """
        with self._train_lock:
            self._inference_idle_check = checker
            trainer = self.trainer
            if trainer is not None:
                trainer.inference_idle_check = checker

    def set_training_model(self, model, tokenizer, system_prompt: str = "") -> None:
        """Set the model for background training."""
        if self.trainer:
            self.trainer.set_model(model, tokenizer)
            if system_prompt:
                self.trainer.system_prompt = system_prompt

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for training context."""
        if self.trainer:
            self.trainer.system_prompt = prompt

    def get_training_stats(self) -> dict:
        """Get training statistics."""
        if self.trainer:
            return self.trainer.get_stats()
        return {"enabled": False}

    def get_train_lock(self) -> threading.Lock | None:
        """Return the training lock for inference coordination.

        When background training and inference share the same model,
        wrap the inference forward pass with::

            lock = router.get_train_lock()
            if lock:
                with lock:
                    output = model(input_ids)
            else:
                output = model(input_ids)

        Returns None if no trainer is active.
        """
        if self.trainer:
            return self.trainer.train_lock
        return None

    def pause_training(self) -> None:
        """Pause background training."""
        if self.trainer:
            self.trainer.pause()

    def resume_training(self) -> None:
        """Resume background training."""
        if self.trainer:
            self.trainer.resume()

    def get_status(self) -> dict:
        """Get full router status."""
        return {
            "running": self.running,
            "host": self.host,
            "port": self.port,
            "connected_mods": len(self.mods),
            "mods": self.get_connected_mods(),
            "training": self.get_training_stats(),
        }

    def get_prompt(self, purpose: str) -> str:
        """
        Get a prompt by purpose.

        Args:
            purpose: One of 'chat', 'gui_usage', 'training_scorer',
                     'mod_router', 'safety', or a mod name

        Returns:
            The prompt text, or empty string if not found
        """
        return self.prompts.get(purpose, "")

    def set_prompt(self, purpose: str, prompt: str) -> None:
        """
        Set or update a prompt for a specific purpose.

        Args:
            purpose: The prompt identifier (e.g., 'chat', 'safety', mod name)
            prompt: The prompt text
        """
        self.prompts[purpose] = prompt

        # If setting the main system prompt, also update trainer
        if purpose == "chat" and self.trainer:
            self.trainer.system_prompt = prompt

    def add_mod_prompt(self, mod_name: str, prompt: str) -> None:
        """Add a prompt snippet from a mod (called during registration)."""
        if prompt:
            self.prompts[f"mod_{mod_name}"] = prompt

    def get_combined_prompt(self, *purposes: str) -> str:
        """
        Combine multiple prompts into one.

        Args:
            *purposes: Prompt purposes to combine (e.g., 'chat', 'safety')

        Returns:
            Combined prompt text with each section on new lines
        """
        parts = []
        for purpose in purposes:
            prompt = self.prompts.get(purpose, "")
            if prompt:
                parts.append(prompt)
        return "\n\n".join(parts)

    def register_handler(self, msg_type: str, handler: Callable) -> None:
        """Register a custom message handler."""
        self.message_handlers[msg_type] = handler


# =============================================================================
# SINGLETON ROUTER INSTANCE
# =============================================================================

_router_instance: ModRouter | None = None
_router_lock = threading.Lock()


def get_router() -> ModRouter:
    """Get or create the global router instance."""
    global _router_instance
    if _router_instance is None:
        with _router_lock:
            if _router_instance is None:
                _router_instance = ModRouter()
    return _router_instance


def start_router(host: str = "127.0.0.1", port: int = 9900) -> bool:
    """Start the global router."""
    router = get_router()
    router.host = host
    router.port = port
    return router.start()


def stop_router() -> None:
    """Stop the global router."""
    global _router_instance
    if _router_instance:
        _router_instance.stop()
        _router_instance = None
