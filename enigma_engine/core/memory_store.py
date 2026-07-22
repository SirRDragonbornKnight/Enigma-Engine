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

# The apostrophe SEPARATES words: "dog's" tokenizes to ("dog", "s"), so a
# query about "my dog's name" reaches a stored "User's dog is named Rex."
_WORD = re.compile(r"[a-z0-9]+")

# Words that carry no memory identity ("my dog is Rex" vs "my cat is Whiskers"
# must NOT look similar just because both say "my ... is"). "user"/"s" are
# boilerplate in the stored fact form ("User's X is Y."), and question words
# carry no identity either -- without them a bare "is that right?" scores
# against every record containing "is".
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "my", "your", "our", "their", "his", "her", "its",
    "i", "you", "we", "they", "it", "this", "that",
    "and", "or", "of", "to", "in", "on", "for", "with", "as", "at",
    "s", "user", "users",
    "what", "who", "whom", "whose", "where", "when", "why", "how", "which",
    "do", "does", "did", "can", "could", "would", "should", "will",
    "tell", "me", "about", "again", "remind", "know", "have", "has", "had",
})

_VERB_SUFFIXES = ("ing", "ed")
_PLURAL_SUFFIXES = ("es", "s")


def _stem(token: str) -> str:
    """Fold inflections so name/named/naming/names share one key."""
    if len(token) <= 3:
        return token
    for suffix in _VERB_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    else:
        for suffix in _PLURAL_SUFFIXES:
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                token = token[: -len(suffix)]
                break
    if len(token) > 3 and token.endswith("e"):
        token = token[:-1]
    return token


def _terms(text: str) -> list[str]:
    return [_stem(t) for t in _WORD.findall(text.lower())]


# Stopwords are matched AFTER stemming, so the set carries stemmed forms too
# ("does" stems to "doe", "have" to "hav").
_STOP_STEMS = frozenset(_stem(w) for w in _STOPWORDS) | _STOPWORDS


def _content_term_list(text: str) -> list[str]:
    return [t for t in _terms(text) if t not in _STOP_STEMS]


def _content_terms(text: str) -> set[str]:
    return set(_content_term_list(text))


# The stored fact form ("User's dog is named Rex.", "My bicycle is teal.")
# has an ATTRIBUTE slot and a VALUE slot. Two texts describe the SAME fact
# only when the attribute matches: "User's brother is named Leo." and
# "User's sister is named Leo." share most of their words but are two people.
_FACT = re.compile(
    r"^\s*(?:the\s+)?(?:user's|users|user|my)\s+(?P<attr>[^.]{1,60}?)"
    r"\s+(?:is|are|was|were)\s+(?P<val>[^.]+?)\s*\.?\s*$",
    re.IGNORECASE,
)

# Lexical fallback for texts that are not fact-shaped. Deliberately high:
# a missed supersede leaves two records to rank, a wrong one DESTROYS a fact.
_SUPERSEDE_MIN = 0.75


def _fact_key(text: str) -> frozenset[str] | None:
    """The attribute a fact is ABOUT, or None when the text isn't fact-shaped."""
    match = _FACT.match(" ".join(str(text).split()))
    if not match:
        return None
    key = _content_terms(match.group("attr"))
    return frozenset(key) if key else None


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

        Fact-shaped texts supersede on their ATTRIBUTE, so a changed value
        ("User's car is a red hatchback." -> "... a silver van.") replaces the
        old record however much the wording moved, while a shared value across
        different attributes ("brother"/"sister" both named Leo) does not.

        HONEST LIMIT: texts that are not fact-shaped fall back to lexical
        overlap at a deliberately high bar, so an unusual restatement coexists
        with the old record rather than risking the wrong deletion."""
        text = " ".join(str(text).split())
        if not text:
            raise ValueError("empty memory")
        new_terms = _content_terms(text)
        new_key = _fact_key(text)
        with self._lock:
            superseded = None
            best_score = 0.0
            for rec in self._records:
                if rec["text"].lower() == text.lower():
                    return dict(rec)  # exact duplicate: keep the original
                old_key = _fact_key(rec["text"])
                if new_key is not None and old_key is not None:
                    # Both fact-shaped: the attribute alone decides.
                    if old_key == new_key:
                        superseded = rec
                        break
                    continue
                old_terms = _content_terms(rec["text"])
                union = new_terms | old_terms
                if not union:
                    continue
                score = len(new_terms & old_terms) / len(union)
                # Rank every candidate: the FIRST record over the bar is not
                # necessarily the closest one.
                if score >= _SUPERSEDE_MIN and score > best_score:
                    best_score, superseded = score, rec
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
        concurrent supersede between the two passes would misalign the pairs.

        Scoring runs on CONTENT terms only. A query whose every word is a
        stopword ("is that right?") shares nothing identifying with any record
        and correctly retrieves nothing, rather than ranking on "is"."""
        q_terms = _content_term_list(query)
        with self._lock:
            if not q_terms or not self._records:
                return []
            docs = [_content_term_list(r["text"]) for r in self._records]
            n = len(docs)
            # A record with no content terms contributes length 0; the floor
            # keeps the length-normalization divisor defined.
            avg_len = max(1.0, sum(len(d) for d in docs) / n)
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
