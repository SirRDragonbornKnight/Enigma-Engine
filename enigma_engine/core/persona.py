#!/usr/bin/env python
"""Who the AI IS, separated from the machinery that trains and serves her.

The trainer can already build a different model end to end -- pretrain, SFT,
DPO, eval, serve are all parameterized by paths and flags. What it could not do
is build a different AI: the identity was spelled into ~30 literals across a
dozen files, so a run produced a second Enigma rather than someone new.

A persona pack is that identity as DATA. `Persona.load()` with no argument is
Enigma, byte-for-byte what the literals said, so nothing about her changes by
introducing this.

Two kinds of field, and the distinction matters:

* Mechanical -- `name`, `data_dirname`, `transcript_label`. Substituting these
  is safe because the surrounding sentence does not depend on their meaning.
* Name-SEMANTIC -- `name_meaning`. Her identity answers explain what the word
  "Enigma" means ("a closed box, in the good sense"), and no template can
  derive that from a different name. A pack supplies its own or says nothing;
  it is never generated.

The single-machine guards (the tray mutex, the fixed serve port, one shared
data home) are what actually stop two AIs from colliding on this box, so the
data home is derived from the persona rather than hardcoded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# The identity of record. Every value here is what the literals it replaced
# already said, so `Persona.load()` reproduces today's behavior exactly.
ENIGMA = {
    "name": "Enigma",
    "data_dirname": ".enigma_engine",
    "name_meaning": (
        "a closed box, in the good sense: your data goes in, useful answers "
        "come out, and nothing leaks to the outside"
    ),
}

_SAFE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9 _-]{0,31}$")


@dataclass(frozen=True)
class Persona:
    """One AI's identity. Frozen: a persona is chosen at boot, not edited live."""

    name: str = ENIGMA["name"]
    data_dirname: str = ENIGMA["data_dirname"]
    name_meaning: str = ENIGMA["name_meaning"]
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The name reaches a directory name, a stop-sequence and a system
        # prompt. A name carrying a path separator or a newline would not be
        # rejected by any of those on its own.
        if not _SAFE_NAME.fullmatch(self.name or ""):
            raise ValueError(
                f"persona name {self.name!r} must start with a letter and use only "
                "letters, digits, spaces, hyphens and underscores (max 32 chars)"
            )
        if not self.data_dirname or any(c in self.data_dirname for c in '/\\:*?"<>|' + "\n\r\t"):
            raise ValueError(f"persona data_dirname {self.data_dirname!r} is not a bare directory name")

    @property
    def home(self) -> Path:
        """Her data home: voice recipe, generated images, runtime state.

        Per-persona so a second AI on this machine does not overwrite the
        first one's voice and pictures -- one shared home was among the real
        one-AI-per-machine guards."""
        return Path.home() / self.data_dirname

    @property
    def transcript_label(self) -> str:
        """The turn marker a base checkpoint must be cut off at."""
        return f"\n{self.name}:"

    @property
    def tools_preamble(self) -> str:
        """The system line prepended to a tools block when the client sent no
        system message of its own."""
        return (f"You are {self.name}. You can use tools when they are needed; "
                "answer directly when they are not.")

    @classmethod
    def load(cls, path: Path | None = None) -> "Persona":
        """Load a persona pack, or Enigma when there is none.

        A missing file is not an error: this repo IS Enigma, and she is the
        default rather than a special case."""
        if path is None:
            return cls()
        path = Path(path)
        if not path.exists():
            raise SystemExit(f"persona pack not found: {path}")
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"persona pack {path} is not valid JSON ({exc.msg})") from None
        if not isinstance(blob, dict):
            raise SystemExit(f"persona pack {path} must be a JSON object")
        known = {f for f in ("name", "data_dirname", "name_meaning")}
        try:
            return cls(
                name=blob.get("name", ENIGMA["name"]),
                data_dirname=blob.get("data_dirname") or f".{_slug(blob.get('name', ENIGMA['name']))}",
                name_meaning=blob.get("name_meaning", ""),
                extra={k: v for k, v in blob.items() if k not in known},
            )
        except ValueError as exc:  # an invalid pack is a refusal, not a traceback
            raise SystemExit(f"persona pack {path}: {exc}") from None


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "ai"


# The process-wide persona. Set once by a boot path; everything else reads it.
ACTIVE = Persona()


def set_active(persona: Persona) -> Persona:
    global ACTIVE
    ACTIVE = persona
    return ACTIVE
