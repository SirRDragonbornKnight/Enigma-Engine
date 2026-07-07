"""
Enigma Engine - Curated Dataset Manager
==========================================

Implements DA-C: curated dataset with review step.

All guided training curricula + chat exchanges accumulate into
a master dataset. Each entry has an approval status so users
can review/approve/reject on the DOCS page before training.

Dataset is stored as a JSONL file where each line contains:
  - text: the training example text
  - source: where it came from (guided, chat, web_learn, etc.)
  - timestamp: when it was added
  - status: pending, approved, rejected
  - stage: training stage it belongs to
  - metadata: any additional fields
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DatasetEntry:
    """A single entry in the curated dataset.

    Attributes:
        text: The training example text.
        source: Origin identifier (guided, chat, web_learn, etc.).
        timestamp: ISO timestamp when added.
        status: pending, approved, rejected.
        stage: Training stage (basics, conversation, commands, web).
        metadata: Extra info dict (trainer_name, prompt, etc.).
    """

    text: str = ""
    source: str = ""
    timestamp: str = ""
    status: str = "pending"
    stage: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetEntry:
        """Deserialize from dict."""
        known = cls.__dataclass_fields__
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class CuratedDataset:
    """Manages a curated training dataset with approval workflow.

    Entries go through: pending → approved/rejected.
    Only approved entries are used for training.

    Usage:
        ds = CuratedDataset(Path("data/curated_dataset.jsonl"))
        ds.add("Q: What is AI?\\nA: ...", source="guided", stage="basics")
        ds.approve(0)  # approve first entry
        approved = ds.get_approved()  # get only approved entries
    """

    def __init__(self, path: str | Path):
        """Initialize the dataset.

        Args:
            path: Path to the JSONL dataset file.
        """
        self.path = Path(path)
        self._entries: list[DatasetEntry] = []
        self._lock = threading.Lock()
        if self.path.exists():
            self.load()

    # ---- Entry management ----

    def _add_unlocked(
        self,
        text: str,
        source: str,
        stage: str,
        status: str,
        metadata: dict[str, Any] | None,
    ) -> int:
        """Add entry without acquiring lock. Caller must hold self._lock."""
        entry = DatasetEntry(
            text=text,
            source=source,
            stage=stage,
            status=status,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        idx = len(self._entries) - 1
        logger.debug("Dataset: added entry #%d from %s", idx, source)
        return idx

    def add(
        self,
        text: str,
        source: str = "",
        stage: str = "",
        status: str = "pending",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a new entry. Returns its index."""
        if not text or not text.strip():
            logger.debug("Dataset: skipping empty text from %s", source)
            return -1
        with self._lock:
            return self._add_unlocked(text, source, stage, status, metadata)

    def add_batch(
        self,
        texts: list[str],
        source: str = "",
        stage: str = "",
        status: str = "pending",
    ) -> int:
        """Add multiple entries at once. Returns count added.

        Deduplicates within the batch and against existing entries
        — if the same text appears multiple times or is already
        in the dataset, only the first new occurrence is added.
        """
        seen: set[str] = set()
        count = 0
        with self._lock:
            existing = {e.text for e in self._entries}
            for text in texts:
                text = text.strip()
                if text and text not in seen and text not in existing:
                    seen.add(text)
                    self._add_unlocked(text, source, stage, status, None)
                    count += 1
        return count

    def approve(self, index: int) -> bool:
        """Approve an entry by index. Returns True on success."""
        with self._lock:
            if 0 <= index < len(self._entries):
                self._entries[index].status = "approved"
                return True
            return False

    def reject(self, index: int) -> bool:
        """Reject an entry by index. Returns True on success."""
        with self._lock:
            if 0 <= index < len(self._entries):
                self._entries[index].status = "rejected"
                return True
            return False

    def approve_all_pending(self) -> int:
        """Approve all pending entries. Returns count approved."""
        with self._lock:
            count = 0
            for entry in self._entries:
                if entry.status == "pending":
                    entry.status = "approved"
                    count += 1
            return count

    def reject_all_pending(self) -> int:
        """Reject all pending entries. Returns count rejected."""
        with self._lock:
            count = 0
            for entry in self._entries:
                if entry.status == "pending":
                    entry.status = "rejected"
                    count += 1
            return count

    def remove(self, index: int) -> bool:
        """Remove an entry by index. Returns True on success."""
        with self._lock:
            if 0 <= index < len(self._entries):
                self._entries.pop(index)
                return True
            return False

    # ---- Queries (return copies for thread safety) ----

    @property
    def entries(self) -> list[DatasetEntry]:
        """All entries (snapshot copy)."""
        with self._lock:
            return list(self._entries)

    @property
    def count(self) -> int:
        """Total number of entries."""
        with self._lock:
            return len(self._entries)

    @property
    def pending_count(self) -> int:
        """Number of pending entries."""
        with self._lock:
            return sum(1 for e in self._entries if e.status == "pending")

    @property
    def approved_count(self) -> int:
        """Number of approved entries."""
        with self._lock:
            return sum(1 for e in self._entries if e.status == "approved")

    @property
    def rejected_count(self) -> int:
        """Number of rejected entries."""
        with self._lock:
            return sum(1 for e in self._entries if e.status == "rejected")

    def get_approved(self) -> list[DatasetEntry]:
        """Get all approved entries for training."""
        with self._lock:
            return [e for e in self._entries if e.status == "approved"]

    def get_pending(self) -> list[DatasetEntry]:
        """Get all pending entries for review."""
        with self._lock:
            return [e for e in self._entries if e.status == "pending"]

    def get_by_source(self, source: str) -> list[DatasetEntry]:
        """Get entries from a specific source."""
        with self._lock:
            return [e for e in self._entries if e.source == source]

    def get_by_stage(self, stage: str) -> list[DatasetEntry]:
        """Get entries for a specific training stage."""
        with self._lock:
            return [e for e in self._entries if e.stage == stage]

    def get_approved_text(self) -> list[str]:
        """Get just the text of all approved entries — ready for training."""
        with self._lock:
            return [e.text for e in self._entries if e.status == "approved"]

    def get_training_data(self, stage: str = "") -> str:
        """Get approved entries as training text, optionally filtered by stage.

        Returns entries joined by double newlines, suitable for
        passing to Trainer._parse_training_data().
        """
        with self._lock:
            entries = [e for e in self._entries if e.status == "approved"]
            if stage:
                entries = [e for e in entries if e.stage == stage]
            return "\n\n".join(e.text for e in entries)

    # ---- Persistence ----

    def save(self) -> None:
        """Save dataset to JSONL file."""
        from enigma_engine.core.safe_save import atomic_write_text

        # The write stays under the lock: releasing it between snapshot and
        # write lets two concurrent save() calls land in either order, so
        # the older snapshot can end up as the file's final content.
        with self._lock:
            content = "".join(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n" for entry in self._entries)
            approved = sum(1 for e in self._entries if e.status == "approved")
            pending = sum(1 for e in self._entries if e.status == "pending")
            total = len(self._entries)
            atomic_write_text(self.path, content)
        logger.info("Dataset saved: %d entries (%d approved, %d pending)", total, approved, pending)

    def load(self) -> None:
        """Load dataset from JSONL file."""
        if not self.path.exists():
            return
        with self._lock:
            loaded: list[DatasetEntry] = []
            bad = 0
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        # Skip-and-count per line: one corrupt line must not
                        # discard the whole dataset (a later save() would then
                        # clobber every previously curated entry).
                        try:
                            loaded.append(DatasetEntry.from_dict(json.loads(line)))
                        except Exception as exc:
                            bad += 1
                            logger.error("Dataset line %d unreadable, skipped: %s", lineno, exc)
                self._entries = loaded
                if bad:
                    logger.warning("Dataset loaded with %d unreadable line(s) skipped", bad)
                logger.info("Dataset loaded: %d entries from %s", len(self._entries), self.path)
            except Exception as exc:
                logger.error("Dataset load error: %s", exc)

    # ---- Summary ----

    def summary(self) -> str:
        """Human-readable summary."""
        with self._lock:
            total = len(self._entries)
            approved = sum(1 for e in self._entries if e.status == "approved")
            pending = sum(1 for e in self._entries if e.status == "pending")
            rejected = sum(1 for e in self._entries if e.status == "rejected")
            sources: dict[str, int] = {}
            for e in self._entries:
                sources[e.source] = sources.get(e.source, 0) + 1
        lines = [
            f"Curated Dataset: {self.path.name}",
            f"Total: {total} entries",
            f"  Approved: {approved}",
            f"  Pending: {pending}",
            f"  Rejected: {rejected}",
        ]
        if sources:
            lines.append("Sources:")
            for src, cnt in sorted(sources.items()):
                lines.append(f"  {src or '(unknown)'}: {cnt}")
        return "\n".join(lines)
