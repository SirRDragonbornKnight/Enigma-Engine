"""
Enigma Engine - Training Queue & Schedule
============================================

Implements:
- TS-B: Training Queue — queue multiple training jobs to run
  sequentially with progress tracking.
- TS-C: Overnight Plan — schedule a full training plan with
  stages, data, epochs. Save progress, resume on crash.
  Show summary when done.

The TrainingQueue manages a FIFO list of TrainingJob items.
Each job tracks its own mode, data, config, and results.
The queue runs in a background thread and emits progress
via callbacks for GUI integration.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ================================================================
# Training Job — one unit of work in the queue
# ================================================================


@dataclass
class TrainingJob:
    """A single training job to be executed by the queue.

    Attributes:
        job_id: Unique auto-incrementing ID.
        mode: Training mode key (Solo, DPO, LoRA, etc.).
        model_path: Path to the model to train.
        data_path: Optional path to training data file.
        stage: Training stage for guided/adaptive modes.
        epochs: Number of training epochs.
        learning_rate: Learning rate for this job.
        batch_size: Batch size for this job.
        extra_config: Any additional config overrides.
        status: pending, running, completed, failed, cancelled.
        progress: 0-100 completion percentage.
        message: Current status message.
        best_loss: Best loss achieved (set after completion).
        error: Error message if failed.
        created_at: ISO timestamp when job was created.
        started_at: ISO timestamp when job started running.
        completed_at: ISO timestamp when job finished.
    """

    job_id: int = 0
    mode: str = "Solo"
    model_path: str = ""
    data_path: str = ""
    stage: str = "basics"
    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 4
    extra_config: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    progress: int = 0
    message: str = ""
    best_loss: float = float("inf")
    error: str = ""
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingJob:
        """Deserialize from dict."""
        known = cls.__dataclass_fields__
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ================================================================
# Training Queue — manages sequential job execution (TS-B)
# ================================================================


class TrainingQueue:
    """Sequential training job queue.

    Jobs are added and run in FIFO order. Only one job runs at
    a time. The queue can be paused, resumed, and persisted to
    disk for crash recovery.

    Usage:
        queue = TrainingQueue()
        queue.on_progress = lambda job, pct, msg: print(f"{pct}% {msg}")
        queue.on_job_complete = lambda job: print(f"Done: {job.mode}")
        queue.add_job(TrainingJob(mode="Solo", ...))
        queue.add_job(TrainingJob(mode="DPO", ...))
        queue.start()  # Runs all jobs sequentially in background
    """

    def __init__(self, save_path: str | Path | None = None):
        """Initialize the training queue.

        Args:
            save_path: Optional path to save/load queue state.
                       Defaults to data/training_queue.json.
        """
        self._jobs: list[TrainingJob] = []
        self._next_id: int = 1
        self._lock = threading.Lock()
        self._running = False
        self._paused = False
        self._thread: threading.Thread | None = None
        self._current_job: TrainingJob | None = None
        self._stop_requested = False
        self.save_path = Path(save_path) if save_path else None

        # Callbacks — GUI hooks
        self.on_progress: Callable[[TrainingJob, int, str], None] | None = None
        self.on_job_complete: Callable[[TrainingJob], None] | None = None
        self.on_job_failed: Callable[[TrainingJob, str], None] | None = None
        self.on_queue_complete: Callable[[], None] | None = None

        # The actual training executor — set by the GUI/caller
        # Signature: executor(job: TrainingJob) -> float (best_loss)
        # Raises on failure.
        self.executor: Callable[[TrainingJob], float] | None = None

    # ---- Job management ----

    def add_job(self, job: TrainingJob) -> TrainingJob:
        """Add a job to the queue. Returns the job with assigned ID."""
        with self._lock:
            job.job_id = self._next_id
            self._next_id += 1
            job.status = "pending"
            self._jobs.append(job)
        logger.info("Queue: added job #%d (%s)", job.job_id, job.mode)
        self._save_state()
        return job

    def remove_job(self, job_id: int) -> bool:
        """Remove a pending job by ID. Cannot remove running jobs.

        Returns True if removed, False if not found or running.
        """
        removed = False
        with self._lock:
            for i, job in enumerate(self._jobs):
                if job.job_id == job_id:
                    if job.status == "running":
                        return False
                    self._jobs.pop(i)
                    logger.info("Queue: removed job #%d", job_id)
                    removed = True
                    break
        if removed:
            self._save_state()
        return removed

    def cancel_job(self, job_id: int) -> bool:
        """Cancel a pending job (marks as cancelled)."""
        cancelled = False
        with self._lock:
            for job in self._jobs:
                if job.job_id == job_id and job.status == "pending":
                    job.status = "cancelled"
                    logger.info("Queue: cancelled job #%d", job_id)
                    cancelled = True
                    break
        if cancelled:
            self._save_state()
        return cancelled

    def reorder_job(self, job_id: int, new_position: int) -> bool:
        """Move a pending job to a new position in the queue.

        Args:
            job_id: ID of the job to move.
            new_position: 0-based index among pending jobs.

        Returns True if moved, False otherwise.
        """
        reordered = False
        with self._lock:
            # Find the job
            job_idx = None
            for i, job in enumerate(self._jobs):
                if job.job_id == job_id and job.status == "pending":
                    job_idx = i
                    break
            if job_idx is not None:
                job = self._jobs.pop(job_idx)

                # Find position among remaining pending jobs
                # (computed AFTER pop so indices are correct)
                pending_positions = [i for i, j in enumerate(self._jobs) if j.status == "pending"]
                if new_position >= len(pending_positions):
                    self._jobs.append(job)
                elif pending_positions:
                    insert_at = pending_positions[min(new_position, len(pending_positions) - 1)]
                    self._jobs.insert(insert_at, job)
                else:
                    self._jobs.append(job)
                reordered = True
        if reordered:
            self._save_state()
        return reordered

    def clear_completed(self) -> int:
        """Remove all completed/failed/cancelled jobs. Returns count removed."""
        with self._lock:
            before = len(self._jobs)
            self._jobs = [j for j in self._jobs if j.status in ("pending", "running")]
            removed = before - len(self._jobs)
        if removed:
            self._save_state()
        return removed

    # ---- Properties ----

    @property
    def jobs(self) -> list[TrainingJob]:
        """All jobs (snapshot)."""
        with self._lock:
            return list(self._jobs)

    @property
    def pending_count(self) -> int:
        """Number of pending jobs."""
        with self._lock:
            return sum(1 for j in self._jobs if j.status == "pending")

    @property
    def is_running(self) -> bool:
        """Whether the queue is actively processing."""
        return self._running and not self._paused

    @property
    def is_paused(self) -> bool:
        """Whether the queue is paused."""
        return self._paused

    @property
    def current_job(self) -> TrainingJob | None:
        """The currently executing job, or None."""
        with self._lock:
            return self._current_job

    # ---- Queue control ----

    def start(self) -> None:
        """Start processing the queue in a background thread."""
        with self._lock:
            if self._running:
                logger.warning("Queue already running")
                return
            self._running = True
            self._paused = False
            self._stop_requested = False
            # Assigned under the lock: a loop thread from an earlier
            # start() compares itself against this and exits when
            # superseded, so two loops never process jobs concurrently.
            self._thread = threading.Thread(target=self._run_loop, name="TrainingQueue", daemon=True)
            thread = self._thread
        thread.start()
        logger.info("Training queue started")

    def pause(self) -> None:
        """Pause the queue after the current job finishes."""
        with self._lock:
            self._paused = True
        logger.info("Training queue paused")

    def resume(self) -> None:
        """Resume a paused queue."""
        with self._lock:
            was_paused = self._paused
            if was_paused:
                self._paused = False
        if was_paused:
            logger.info("Training queue resumed")
            # If thread died, restart
            if self._thread is None or not self._thread.is_alive():
                self.start()

    def stop(self) -> None:
        """Stop the queue after the current job finishes."""
        with self._lock:
            self._stop_requested = True
            self._running = False
        logger.info("Training queue stop requested")

    def _run_loop(self) -> None:
        """Main queue processing loop (runs in background thread)."""
        me = threading.current_thread()
        try:
            while True:
                with self._lock:
                    # `self._thread is not me` = a later start() superseded
                    # this loop; it must exit even though _running is True.
                    if not self._running or self._stop_requested or self._thread is not me:
                        break
                    paused = self._paused
                if paused:
                    time.sleep(0.5)
                    continue

                # Claim next pending job
                job = self._claim_next_pending()
                if job is None:
                    # No more pending jobs — queue complete
                    break

                self._execute_job(job)

            owns_queue = False
            with self._lock:
                if self._thread is me:
                    self._running = False
                    self._current_job = None
                    owns_queue = True

            if owns_queue and self.on_queue_complete:
                try:
                    self.on_queue_complete()
                except Exception as exc:
                    logger.debug("Queue complete callback error: %s", exc)

            if owns_queue:
                logger.info("Training queue finished")
        except Exception as exc:
            import traceback

            logger.error("Training queue loop error: %s\n%s", exc, traceback.format_exc())
            with self._lock:
                if self._thread is me:
                    self._running = False

    def _claim_next_pending(self) -> TrainingJob | None:
        """Find AND claim the next pending job under one lock acquisition,
        so two loop threads can never both pick up the same job."""
        with self._lock:
            for job in self._jobs:
                if job.status == "pending":
                    job.status = "running"
                    job.started_at = datetime.now().isoformat()
                    self._current_job = job
                    return job
        return None

    def _execute_job(self, job: TrainingJob) -> None:
        """Execute a job already claimed by _claim_next_pending."""
        logger.info("Queue: starting job #%d (%s)", job.job_id, job.mode)
        self._save_state()

        try:
            if self.executor is None:
                raise RuntimeError("No executor set on TrainingQueue")

            best_loss = self.executor(job)

            with self._lock:
                job.status = "completed"
                job.best_loss = best_loss
                job.progress = 100
                job.completed_at = datetime.now().isoformat()

            logger.info("Queue: job #%d completed (loss=%.4f)", job.job_id, best_loss)

            if self.on_job_complete:
                try:
                    self.on_job_complete(job)
                except Exception as exc:
                    logger.debug("Job complete callback error: %s", exc)

        except Exception as exc:
            err_msg = str(exc)
            import traceback

            tb = traceback.format_exc()
            with self._lock:
                job.status = "failed"
                job.error = err_msg
                job.completed_at = datetime.now().isoformat()

            logger.error("Queue: job #%d failed: %s\n%s", job.job_id, err_msg, tb)

            if self.on_job_failed:
                try:
                    self.on_job_failed(job, err_msg)
                except Exception as cb_exc:
                    logger.debug("Job failed callback error: %s", cb_exc)

        with self._lock:
            # Only clear our own claim: after a stop()/start() supersession
            # a new loop thread may already hold a different current job.
            if self._current_job is job:
                self._current_job = None
        self._save_state()

    # ---- Persistence ----

    def _save_state(self) -> None:
        """Save queue state to disk for crash recovery."""
        if self.save_path is None:
            return
        try:
            with self._lock:
                data = {
                    "next_id": self._next_id,
                    "jobs": [j.to_dict() for j in self._jobs],
                    "saved_at": datetime.now().isoformat(),
                }
            from enigma_engine.core.safe_save import atomic_write_json

            atomic_write_json(self.save_path, data, default=str)
        except Exception as exc:
            logger.warning("Queue save error: %s", exc)

    def load_state(self) -> bool:
        """Load queue state from disk. Returns True if loaded.

        Running jobs from a previous session are reset to pending
        so they can be retried.
        """
        if self.save_path is None or not self.save_path.exists():
            return False
        try:
            data = json.loads(self.save_path.read_text(encoding="utf-8"))
            jobs: list[TrainingJob] = []
            for jd in data.get("jobs", []):
                job = TrainingJob.from_dict(jd)
                # Reset interrupted running jobs to pending
                if job.status == "running":
                    job.status = "pending"
                    job.progress = 0
                    job.message = "Resuming after interruption"
                jobs.append(job)
            # Swap under the lock: a running loop thread reads _jobs
            with self._lock:
                self._next_id = int(data.get("next_id", 1))
                self._jobs = jobs
            logger.info("Queue loaded: %d jobs (%d pending)", len(jobs), self.pending_count)
            return True
        except Exception as exc:
            logger.error("Queue load error: %s", exc)
            return False

    # ---- Summary ----

    def summary(self) -> str:
        """Human-readable summary of queue state."""
        with self._lock:
            jobs_snapshot = list(self._jobs)
            current = self._current_job

        lines = ["Training Queue:"]
        counts = {}
        for job in jobs_snapshot:
            counts[job.status] = counts.get(job.status, 0) + 1

        for status, count in sorted(counts.items()):
            lines.append(f"  {status}: {count}")

        if current:
            j = current
            lines.append(f"  Current: #{j.job_id} {j.mode} ({j.progress}% - {j.message})")

        for job in jobs_snapshot:
            # ASCII only: summary() prints to the Windows cp1252 console
            icon = {
                "pending": "o",
                "running": "*",
                "completed": "+",
                "failed": "x",
                "cancelled": "-",
            }.get(job.status, "?")
            loss_str = f" loss={job.best_loss:.4f}" if job.best_loss < float("inf") else ""
            lines.append(f"  {icon} #{job.job_id} {job.mode} [{job.status}]{loss_str}")

        return "\n".join(lines)


# ================================================================
# Overnight Plan — full training schedule (TS-C)
# ================================================================


@dataclass
class OvernightPlan:
    """A full training schedule with multiple stages/modes.

    Combines TS-C: set up a complete plan that runs unattended.
    Saves progress after each job, resumes on crash.

    Attributes:
        name: Human-readable plan name.
        jobs: Ordered list of training job configs.
        auto_checkpoint: Save checkpoint after each job.
        auto_evaluate: Run evaluation after each job.
        notify_on_complete: Show summary when plan finishes.
        status: pending, running, paused, completed, failed.
        current_job_idx: Index of currently executing job.
        results: Summary of each completed job.
        created_at: When the plan was created.
        started_at: When execution began.
        completed_at: When execution finished.
    """

    name: str = "Overnight Training"
    jobs: list[dict[str, Any]] = field(default_factory=list)
    auto_checkpoint: bool = True
    auto_evaluate: bool = False
    notify_on_complete: bool = True
    status: str = "pending"
    current_job_idx: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @property
    def is_complete(self) -> bool:
        """True if all jobs have been processed."""
        if not self.jobs:
            return False
        return self.status == "completed" or self.current_job_idx >= len(self.jobs)

    @property
    def total_jobs(self) -> int:
        return len(self.jobs)

    @property
    def completed_jobs(self) -> int:
        return sum(1 for r in self.results if r.get("status") == "completed")

    @property
    def failed_jobs(self) -> int:
        return sum(1 for r in self.results if r.get("status") == "failed")

    def add_job_config(
        self,
        mode: str,
        model_path: str,
        data_path: str = "",
        stage: str = "basics",
        epochs: int = 10,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        **extra: Any,
    ) -> None:
        """Add a job configuration to the plan."""
        self.jobs.append(
            {
                "mode": mode,
                "model_path": model_path,
                "data_path": data_path,
                "stage": stage,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                **extra,
            }
        )

    def record_result(
        self,
        job_config: dict[str, Any],
        status: str,
        best_loss: float = float("inf"),
        error: str = "",
    ) -> None:
        """Record the result of a completed job."""
        self.results.append(
            {
                "mode": job_config.get("mode", "?"),
                "status": status,
                "best_loss": best_loss,
                "error": error,
                "completed_at": datetime.now().isoformat(),
            }
        )
        self.current_job_idx += 1
        if self.current_job_idx >= len(self.jobs):
            self.status = "completed"
            self.completed_at = datetime.now().isoformat()

    def save(self, path: str | Path) -> None:
        """Save plan to JSON."""
        path = Path(path)
        data = asdict(self)
        from enigma_engine.core.safe_save import atomic_write_json

        atomic_write_json(path, data, default=str)
        logger.info("Overnight plan saved: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> OvernightPlan:
        """Load plan from JSON."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        known = cls.__dataclass_fields__
        filtered = {k: v for k, v in data.items() if k in known}
        plan = cls(**filtered)
        logger.info("Overnight plan loaded: %s (%d/%d jobs done)", path, plan.current_job_idx, plan.total_jobs)
        return plan

    def summary(self) -> str:
        """Human-readable plan summary."""
        lines = [
            f"Overnight Plan: {self.name}",
            f"Status: {self.status.upper()}",
            f"Progress: {self.current_job_idx}/{self.total_jobs} jobs",
        ]
        if self.results:
            total_loss = [r["best_loss"] for r in self.results if r.get("best_loss", float("inf")) < float("inf")]
            if total_loss:
                lines.append(f"Avg loss: {sum(total_loss) / len(total_loss):.4f}")
            lines.append(f"Completed: {self.completed_jobs}, Failed: {self.failed_jobs}")

        for i, job in enumerate(self.jobs):
            if i < len(self.results):
                r = self.results[i]
                status = r.get("status", "?")
                loss = r.get("best_loss", float("inf"))
                loss_str = f" loss={loss:.4f}" if loss < float("inf") else ""
                # ASCII only: the plan summary prints to the cp1252 console
                icon = "+" if status == "completed" else "x"
                lines.append(f"  {icon} {job['mode']} ({job.get('epochs', '?')} epochs){loss_str}")
            elif i == self.current_job_idx and self.status == "running":
                lines.append(f"  * {job['mode']} ({job.get('epochs', '?')} epochs) [RUNNING]")
            else:
                lines.append(f"  o {job['mode']} ({job.get('epochs', '?')} epochs)")

        return "\n".join(lines)

    def to_queue_jobs(self) -> list[TrainingJob]:
        """Convert plan jobs to TrainingJob instances for the queue."""
        result = []
        for i, jc in enumerate(self.jobs):
            if i < self.current_job_idx:
                continue  # Skip already-completed jobs
            job = TrainingJob(
                mode=jc.get("mode", "Solo"),
                model_path=jc.get("model_path", ""),
                data_path=jc.get("data_path", ""),
                stage=jc.get("stage", "basics"),
                epochs=jc.get("epochs", 10),
                learning_rate=jc.get("learning_rate", 1e-4),
                batch_size=jc.get("batch_size", 4),
                extra_config={k: v for k, v in jc.items() if k not in TrainingJob.__dataclass_fields__},
            )
            result.append(job)
        return result
