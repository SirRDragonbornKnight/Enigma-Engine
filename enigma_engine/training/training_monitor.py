"""
Training Monitor for Enigma Engine
====================================

Implements:
- TM-B: Live loss tracking — record loss per step, compute moving
  averages, expose data for GUI charting.
- TM-D: Training history — log all training runs with metadata
  (date, mode, data source, epochs, final loss, duration, model).

All three features share a single ``TrainingMonitor`` class that can
be attached to any Trainer via its ``on_loss`` / ``on_progress``
callbacks.

Usage:
    from enigma_engine.training.training_monitor import (
        TrainingMonitor, TrainingRun,
    )

    monitor = TrainingMonitor()

    # Attach to any trainer
    trainer.on_loss = monitor.record_loss

    # After training
    monitor.finish_run("sft", "chat model", data_source="data/training.txt")
    history = monitor.get_history()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# TRAINING RUN LOG (TM-D)
# =============================================================================


@dataclass
class TrainingRun:
    """Metadata for one completed training run.

    Stored in the history log on disk.
    """

    run_id: str = ""
    timestamp: str = ""
    mode: str = ""  # sft, dpo, rlhf, self-play, progressive, etc.
    model_name: str = ""
    data_source: str = ""
    epochs_completed: int = 0
    total_steps: int = 0
    final_loss: float = 0.0
    best_loss: float = 0.0
    duration_seconds: float = 0.0
    alerts_raised: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "model_name": self.model_name,
            "data_source": self.data_source,
            "epochs_completed": self.epochs_completed,
            "total_steps": self.total_steps,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "duration_seconds": round(self.duration_seconds, 2),
            "alerts_raised": self.alerts_raised,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingRun:
        return cls(
            run_id=d.get("run_id", ""),
            timestamp=d.get("timestamp", ""),
            mode=d.get("mode", ""),
            model_name=d.get("model_name", ""),
            data_source=d.get("data_source", ""),
            epochs_completed=d.get("epochs_completed", 0),
            total_steps=d.get("total_steps", 0),
            final_loss=d.get("final_loss", 0.0),
            best_loss=d.get("best_loss", 0.0),
            duration_seconds=d.get("duration_seconds", 0.0),
            alerts_raised=d.get("alerts_raised", 0),
            extra=d.get("extra", {}),
        )


# =============================================================================
# TRAINING MONITOR
# =============================================================================


class TrainingMonitor:
    """Central training monitor: loss tracking and history.

    Attach this to any trainer by hooking into loss callbacks.

    Args:
        history_path: Path to the JSON history file.
            Defaults to ``data/training_history.json``.
        moving_avg_window: Steps for moving average computation.
    """

    def __init__(
        self,
        history_path: str | Path | None = None,
        moving_avg_window: int = 20,
    ):
        self.history_path = Path(history_path or "data/training_history.json")
        self.moving_avg_window = moving_avg_window

        # Thread safety lock (Suggestion #8A)
        self._lock = threading.Lock()
        # Separate lock for the history file's read-modify-write, so two
        # concurrent run recordings cannot drop each other's entry
        self._history_lock = threading.Lock()

        # Live state (TM-B)
        # Cap at 100k entries to prevent unbounded memory growth on long runs
        self._max_losses = 100_000
        self._losses: list[float] = []
        self._steps: list[int] = []
        self._epoch_losses: list[float] = []
        self._epoch_perplexities: list[float] = []
        self._global_step = 0
        self._start_time: float = 0.0
        self._best_loss = float("inf")

        # Throughput telemetry
        self._step_times: list[float] = []  # seconds per step
        self._tokens_per_sec: list[float] = []  # tokens/sec per step
        self._total_tokens: int = 0
        self._last_step_time: float = 0.0

    # -----------------------------------------------------------------
    # LIVE LOSS TRACKING (TM-B)
    # -----------------------------------------------------------------

    def start_run(self) -> None:
        """Reset live state for a new training run."""
        with self._lock:
            self._losses.clear()
            self._steps.clear()
            self._epoch_losses.clear()
            self._epoch_perplexities.clear()
            self._global_step = 0
            self._best_loss = float("inf")
            self._start_time = time.time()
            self._step_times.clear()
            self._tokens_per_sec.clear()
            self._total_tokens = 0
            self._last_step_time = 0.0

    def record_loss(self, loss: float, step: int | None = None) -> None:
        """Record a single loss value.

        Call this from the trainer's loss callback (once per step or
        once per batch — your choice).

        Args:
            loss: The loss value.
            step: Optional explicit step number. If None, auto-increments.
        """
        with self._lock:
            if step is not None:
                self._global_step = step
            else:
                self._global_step += 1

            self._losses.append(loss)
            self._steps.append(self._global_step)

            # Cap to prevent unbounded memory growth
            if len(self._losses) > self._max_losses:
                # Keep the most recent half
                half = self._max_losses // 2
                self._losses = self._losses[-half:]
                self._steps = self._steps[-half:]

            if loss < self._best_loss:
                self._best_loss = loss

    def record_epoch_loss(self, epoch_loss: float) -> None:
        """Record an epoch-average loss and compute perplexity.

        Perplexity = exp(loss).  Capped at 1e6 to avoid overflow
        when loss is very high (e.g. early training).

        Args:
            epoch_loss: Average cross-entropy loss for this epoch.
        """
        import math

        # Perplexity = exp(loss), capped to avoid overflow
        try:
            ppl = math.exp(min(epoch_loss, 20.0))
            ppl = min(ppl, 1e6)
        except OverflowError:
            ppl = 1e6

        with self._lock:
            self._epoch_losses.append(epoch_loss)
            self._epoch_perplexities.append(ppl)

    def record_throughput(
        self,
        tokens_in_batch: int,
        step_time: float,
    ) -> None:
        """Record throughput metrics for one training step.

        Args:
            tokens_in_batch: Number of tokens processed in this step.
            step_time: Wall-clock time for this step (seconds).
        """
        with self._lock:
            self._total_tokens += tokens_in_batch
            if step_time > 0:
                self._step_times.append(step_time)
                self._tokens_per_sec.append(tokens_in_batch / step_time)

                # Cap to prevent unbounded growth (same as losses)
                if len(self._step_times) > self._max_losses:
                    half = self._max_losses // 2
                    self._step_times = self._step_times[-half:]
                    self._tokens_per_sec = self._tokens_per_sec[-half:]

    @property
    def avg_tokens_per_sec(self) -> float:
        """Average tokens/sec over the run."""
        with self._lock:
            if not self._tokens_per_sec:
                return 0.0
            return sum(self._tokens_per_sec) / len(self._tokens_per_sec)

    @property
    def avg_step_time(self) -> float:
        """Average step time in seconds."""
        with self._lock:
            if not self._step_times:
                return 0.0
            return sum(self._step_times) / len(self._step_times)

    @property
    def total_tokens(self) -> int:
        """Total tokens processed in the run."""
        with self._lock:
            return self._total_tokens

    @property
    def losses(self) -> list[float]:
        """All recorded loss values (for charting) — returns a copy."""
        with self._lock:
            return list(self._losses)

    @property
    def steps(self) -> list[int]:
        """Step numbers corresponding to losses — returns a copy."""
        with self._lock:
            return list(self._steps)

    @property
    def epoch_losses(self) -> list[float]:
        """Per-epoch average losses — returns a copy."""
        with self._lock:
            return list(self._epoch_losses)

    @property
    def epoch_perplexities(self) -> list[float]:
        """Per-epoch perplexities (exp of epoch loss) — returns a copy."""
        with self._lock:
            return list(self._epoch_perplexities)

    @property
    def best_loss(self) -> float:
        with self._lock:
            return self._best_loss

    @property
    def current_loss(self) -> float:
        """Most recent loss value."""
        with self._lock:
            return self._losses[-1] if self._losses else 0.0

    def moving_average(self, window: int | None = None) -> list[float]:
        """Compute a moving average of recorded losses.

        Args:
            window: Window size (defaults to ``moving_avg_window``).

        Returns:
            List of moving-average values (same length as losses,
            with early entries using smaller windows).
        """
        import math

        w = window or self.moving_avg_window
        with self._lock:
            losses_copy = list(self._losses)
        result: list[float] = []
        running_sum = 0.0
        valid_count = 0
        for i in range(len(losses_copy)):
            val = losses_copy[i]
            # Subtract old value leaving the window BEFORE NaN check —
            # otherwise `continue` skips the subtraction and stale
            # values accumulate in running_sum.
            if i >= w:
                old = losses_copy[i - w]
                if not (math.isnan(old) or math.isinf(old)):
                    running_sum -= old
                    valid_count -= 1
            if math.isnan(val) or math.isinf(val):
                # Carry forward previous average (or 0.0)
                result.append(result[-1] if result else 0.0)
                continue
            running_sum += val
            valid_count += 1
            count = min(valid_count, w)
            result.append(running_sum / count if count > 0 else 0.0)
        return result

    def get_chart_data(self) -> dict[str, Any]:
        """Return data suitable for a live loss chart (TM-B).

        Returns:
            Dict with steps, losses, moving_avg, epoch_losses, best_loss.
        """
        with self._lock:
            losses_snap = list(self._losses)
            steps_snap = list(self._steps)
            epoch_losses_snap = list(self._epoch_losses)
            epoch_ppl_snap = list(self._epoch_perplexities)
            best = self._best_loss
            total = self._global_step
            tps_snap = list(self._tokens_per_sec)
            st_snap = list(self._step_times)
            total_tok = self._total_tokens

        # Compute moving average from snapshot (no lock needed)
        import math

        w = self.moving_avg_window
        ma: list[float] = []
        running_sum = 0.0
        valid_count = 0
        for i in range(len(losses_snap)):
            val = losses_snap[i]
            # Subtract old value leaving the window BEFORE NaN check
            if i >= w:
                old = losses_snap[i - w]
                if not (math.isnan(old) or math.isinf(old)):
                    running_sum -= old
                    valid_count -= 1
            if math.isnan(val) or math.isinf(val):
                # Carry forward previous average (or 0.0)
                ma.append(ma[-1] if ma else 0.0)
                continue
            running_sum += val
            valid_count += 1
            count = min(valid_count, w)
            ma.append(running_sum / count if count > 0 else 0.0)

        return {
            "steps": steps_snap,
            "losses": losses_snap,
            "moving_avg": ma,
            "epoch_losses": epoch_losses_snap,
            "epoch_perplexities": epoch_ppl_snap,
            "best_loss": best,
            "total_steps": total,
            "tokens_per_sec": tps_snap,
            "step_times": st_snap,
            "total_tokens": total_tok,
            "avg_tokens_per_sec": (sum(tps_snap) / len(tps_snap) if tps_snap else 0.0),
            "avg_step_time": (sum(st_snap) / len(st_snap) if st_snap else 0.0),
        }

    # -----------------------------------------------------------------
    # TRAINING HISTORY (TM-D)
    # -----------------------------------------------------------------

    def finish_run(
        self,
        mode: str = "",
        model_name: str = "",
        data_source: str = "",
        extra: dict | None = None,
    ) -> TrainingRun:
        """Finalize the current run and save it to history.

        Call this after training completes. Computes duration and
        persists the run metadata.

        Args:
            mode: Training mode (sft, dpo, rlhf, etc.).
            model_name: Name of the model being trained.
            data_source: Path or description of training data.
            extra: Any additional metadata to store.

        Returns:
            The completed TrainingRun.
        """
        import datetime

        # Snapshot live state under lock
        with self._lock:
            duration = time.time() - self._start_time if self._start_time else 0.0
            final_loss = self._losses[-1] if self._losses else 0.0
            best = self._best_loss
            n_epochs = len(self._epoch_losses)
            total = self._global_step
            ppl_snap = list(self._epoch_perplexities)
            total_tok = self._total_tokens
            tps_snap = list(self._tokens_per_sec)

        # Include perplexity in extra metadata (EV-B)
        run_extra = dict(extra or {})
        if ppl_snap:
            run_extra["final_perplexity"] = round(ppl_snap[-1], 4)
            run_extra["best_perplexity"] = round(min(ppl_snap), 4)

        # Include throughput telemetry
        if total_tok > 0:
            run_extra["total_tokens"] = total_tok
        if tps_snap:
            run_extra["avg_tokens_per_sec"] = round(sum(tps_snap) / len(tps_snap), 1)
        if duration > 0 and total_tok > 0:
            run_extra["overall_tokens_per_sec"] = round(total_tok / duration, 1)

        run = TrainingRun(
            run_id=f"{int(time.time())}_{mode}",
            timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
            mode=mode,
            model_name=model_name,
            data_source=data_source,
            epochs_completed=n_epochs,
            total_steps=total,
            final_loss=final_loss,
            best_loss=best,
            duration_seconds=duration,
            alerts_raised=0,
            extra=run_extra,
        )

        self._append_to_history(run)
        logger.info(
            "Training run recorded: %s (%s, %d steps, loss=%.4f, %.0fs)",
            run.run_id,
            mode,
            run.total_steps,
            run.final_loss,
            duration,
        )

        return run

    def get_history(
        self,
        limit: int = 0,
        mode: str = "",
    ) -> list[TrainingRun]:
        """Load training history from disk.

        Args:
            limit: Max runs to return (0 = all, newest first).
            mode: Filter by training mode (empty = all).

        Returns:
            List of TrainingRun, newest first.
        """
        runs = self._load_history()

        if mode:
            runs = [r for r in runs if r.mode == mode]

        # Sort newest first
        runs.sort(key=lambda r: r.timestamp, reverse=True)

        if limit > 0:
            runs = runs[:limit]

        return runs

    def clear_history(self) -> None:
        """Delete all training history."""
        if self.history_path.exists():
            self.history_path.unlink()
            logger.info("Training history cleared")

    def _append_to_history(self, run: TrainingRun) -> None:
        """Append a run to the history file."""
        from enigma_engine.core.safe_save import atomic_write_json

        with self._history_lock:
            runs = self._load_history()
            runs.append(run)
            atomic_write_json(self.history_path, [r.to_dict() for r in runs])

    def _load_history(self) -> list[TrainingRun]:
        """Load history from disk."""
        if not self.history_path.exists():
            return []

        try:
            text = self.history_path.read_text(encoding="utf-8")
            data = json.loads(text)
            return [TrainingRun.from_dict(d) for d in data]
        except Exception as exc:
            logger.debug("Failed to load training history: %s", exc)
            return []
