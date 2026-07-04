"""Local memory for Enigma — the runtime-learning layer.

Her weights are frozen between training passes; THIS is where day-to-day
learning lives (the frozen-weights + external-memory consensus, see
SUGGESTIONS.md roadmap step 4). Design constraints, in order: black-box
(stdlib-only, no embedding service, no deps), inspectable (plain JSONL a human
can read and edit), small (she serves with a 1024-token context — retrieval
must be sharp, not big).

Retrieval is BM25 over whitespace/word tokens. At her scale — hundreds to a
few thousand memories — lexical scoring is the boring, proven choice.
"""

from __future__ import annotations

import json
import math
import re
import threading
from pathlib import Path
from typing import Any

_WORD = re.compile(r"[a-z0-9']+")


def _terms(text: str) -> list[str]:
    return _WORD.findall(text.lower())


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
            with open(self.file, encoding="utf-8") as f:
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

    def __len__(self) -> int:
        return len(self._records)

    def add(self, text: str, kind: str = "fact", source: str | None = None) -> dict:
        text = " ".join(str(text).split())
        if not text:
            raise ValueError("empty memory")
        with self._lock:
            rec = {"id": len(self._records) + 1, "text": text, "kind": kind}
            if source:
                rec["source"] = source
            self._records.append(rec)
            with open(self.file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rec

    def all(self) -> list[dict]:
        return list(self._records)

    def search(self, query: str, k: int = 3) -> list[dict]:
        """BM25 (k1=1.5, b=0.75). Returns up to k records, best first; records
        sharing no term with the query never match."""
        q_terms = _terms(query)
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
