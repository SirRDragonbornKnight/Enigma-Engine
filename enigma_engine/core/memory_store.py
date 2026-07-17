"""Local memory for Enigma — the runtime-learning layer.

Her weights are frozen between training passes; THIS is where day-to-day
learning lives (the frozen-weights + external-memory consensus). Design
constraints, in order: black-box
(stdlib-only, no embedding service, no deps), inspectable (plain JSONL a human
can read and edit), small (she serves with a 1024-token context — retrieval
must be sharp, not big).

Retrieval is BM25 over whitespace/word tokens. At her scale — hundreds to a
few thousand memories — lexical scoring is the boring, proven choice.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from pathlib import Path
from typing import Any

from enigma_engine.core.safe_save import atomic_write_text

logger = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9']+")

# Words that carry no memory identity ("my dog is Rex" vs "my cat is Whiskers"
# must NOT look similar just because both say "my ... is").
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "my", "your", "our", "their", "his", "her", "its",
    "i", "you", "we", "they", "it", "this", "that",
    "and", "or", "of", "to", "in", "on", "for", "with", "as", "at",
})


def _terms(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _content_terms(text: str) -> set[str]:
    return {t for t in _terms(text) if t not in _STOPWORDS}


def _valid_id(value: Any) -> bool:
    """A usable record id: an int, excluding bool (an int subclass a hand
    edit like ``"id": true`` would otherwise smuggle past the arithmetic)."""
    return isinstance(value, int) and not isinstance(value, bool)


class MemoryStore:
    """Append-mostly JSONL store with BM25 search and budgeted rendering."""

    def __init__(self, path: str | Path):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "memories.jsonl"
        # serve_enigma's endpoints run in FastAPI's threadpool: without a lock,
        # two concurrent add() calls read the same len() and mint duplicate ids.
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        if self.file.exists():
            # utf-8-sig: hand-edited files are inside the contract, and
            # Windows editors save UTF-8 with a BOM that would otherwise
            # corrupt the first line's JSON. Writes stay plain UTF-8.
            with open(self.file, encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict) and rec.get("text"):
                        self._records.append(rec)
        # Hand-edited files are inside the contract (module docstring): a
        # record whose id is missing or not a valid int gets renumbered here
        # so the max+1 id arithmetic in add()/remember() always sees ints.
        next_id = max((r["id"] for r in self._records if _valid_id(r.get("id"))), default=0) + 1
        renumbered = False
        seen_ids: set[int] = set()
        for rec in self._records:
            rid = rec.get("id")
            # Duplicate ids (a copy-paste hand edit) would make id-based
            # delete/supersede ambiguous, so later duplicates get fresh ids.
            if not _valid_id(rid) or rid in seen_ids:
                rec["id"] = next_id
                next_id += 1
                renumbered = True
            seen_ids.add(rec["id"])
        if renumbered:
            # The renumbered ids are live in memory either way; a file that
            # cannot be replaced right now (read-only attribute, another
            # process holding it on Windows) must not stop the store from
            # loading. The next successful rewrite persists them.
            try:
                self._rewrite()
            except OSError as exc:
                logger.warning(
                    "memory store: could not persist renumbered ids to %s (%s); "
                    "continuing with in-memory ids",
                    self.file,
                    exc,
                )

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def add(self, text: str, kind: str = "fact", source: str | None = None) -> dict:
        text = " ".join(str(text).split())
        if not text:
            raise ValueError("empty memory")
        with self._lock:
            # max+1, not len+1: delete()/supersede shrink the list, and a
            # len-based id would collide with a surviving record.
            rec = {"id": max((r["id"] for r in self._records), default=0) + 1, "text": text, "kind": kind}
            if source:
                rec["source"] = source
            self._records.append(rec)
            with open(self.file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rec

    def remember(self, text: str, kind: str = "user_fact", source: str | None = None) -> dict:
        """add() with update semantics -- the ``remember`` tool's entry point.

        Exact duplicate -> returns the existing record (idempotent). A record
        whose CONTENT words overlap the new text heavily (Jaccard >= 0.5,
        stopwords excluded) is treated as the same fact restated ("my dog's
        name is Rex" -> "my dog's name is Bruno") and SUPERSEDED: replaced,
        not left beside the new one to confuse retrieval. Every record gets a
        date stamp so memories can be audited.

        HONEST LIMIT: this is lexical. A correction that rewords most of the
        fact ("red hatchback" -> "silver van", overlap 0.33) coexists with the
        old record instead of replacing it -- resolving that needs semantics,
        which at 182M means a smarter store, not a smarter model. Single-value
        corrections (renames, moves, dates) are the common case and do match."""
        text = " ".join(str(text).split())
        if not text:
            raise ValueError("empty memory")
        new_terms = _content_terms(text)
        with self._lock:
            superseded = None
            for rec in self._records:
                if rec["text"].lower() == text.lower():
                    return dict(rec)  # exact duplicate: keep the original
                old_terms = _content_terms(rec["text"])
                union = new_terms | old_terms
                if union and len(new_terms & old_terms) / len(union) >= 0.5:
                    superseded = rec
                    break
            rec = {
                "id": (max((r["id"] for r in self._records), default=0) + 1),
                "text": text,
                "kind": kind,
                "date": time.strftime("%Y-%m-%d"),
            }
            if source:
                rec["source"] = source
            if superseded is not None:
                rec["superseded"] = superseded["text"]
                self._records.remove(superseded)
                self._records.append(rec)
                self._rewrite()
            else:
                self._records.append(rec)
                with open(self.file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return dict(rec)

    def delete(self, mem_id: int) -> bool:
        """Remove one memory by id. Returns False when the id doesn't exist."""
        with self._lock:
            for rec in self._records:
                if rec["id"] == mem_id:
                    self._records.remove(rec)
                    self._rewrite()
                    return True
            return False

    def clear(self) -> int:
        """Remove ALL memories; returns how many were dropped."""
        with self._lock:
            n = len(self._records)
            self._records = []
            self._rewrite()
            return n

    def _rewrite(self) -> None:
        """Rewrite the JSONL after a mutation (call with the lock held). At
        hundreds of records this is instant and keeps the file inspectable.
        atomic_write_text adds fsync-before-rename (power loss can't leave a
        truncated file). backup=False ON PURPOSE: every _rewrite caller is a
        delete/supersede/clear, and a .bak would keep a full pre-delete copy
        on disk -- "clear my memories" must actually clear (privacy
        regression caught by the 2026-07-17 re-audit). Any .bak an earlier
        build left behind is scrubbed for the same reason."""
        content = "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in self._records)
        atomic_write_text(self.file, content, backup=False)
        self.file.with_suffix(self.file.suffix + ".bak").unlink(missing_ok=True)

    def all(self) -> list[dict]:
        with self._lock:
            return list(self._records)

    def search(self, query: str, k: int = 3) -> list[dict]:
        """BM25 (k1=1.5, b=0.75). Returns up to k records, best first; records
        sharing no term with the query never match. Reads take the same lock
        as writers: scoring zips _records with per-record term vectors, and a
        concurrent supersede between the two passes would misalign the pairs."""
        q_terms = _terms(query)
        with self._lock:
            if not q_terms or not self._records:
                return []
            docs = [_terms(r["text"]) for r in self._records]
            n = len(docs)
            avg_len = sum(len(d) for d in docs) / n
            df: dict[str, int] = {}
            for d in docs:
                for t in set(d):
                    df[t] = df.get(t, 0) + 1
            k1, b = 1.5, 0.75
            scored = []
            for rec, d in zip(self._records, docs):
                score = 0.0
                for t in q_terms:
                    tf = d.count(t)
                    if not tf:
                        continue
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
                if score > 0:
                    scored.append((score, rec))
        scored.sort(key=lambda s: -s[0])
        return [rec for _, rec in scored[:k]]

    def render_context(self, query: str, tokenizer, max_ids: int = 128, k: int = 3) -> str:
        """Top-k matches as a system-prompt block, trimmed to a token budget.
        Empty string when nothing relevant — never pad her context with noise."""
        hits = self.search(query, k=k)
        if not hits:
            return ""
        lines = ["Things you remember:"]
        used = len(tokenizer.encode(lines[0], add_special_tokens=False))
        for rec in hits:
            line = f"- {rec['text']}"
            cost = len(tokenizer.encode(line, add_special_tokens=False)) + 1
            if used + cost > max_ids:
                break
            lines.append(line)
            used += cost
        return "\n".join(lines) if len(lines) > 1 else ""
