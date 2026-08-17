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

# Printable ASCII by construction (explicit ranges, so no unicode letter or
# control character matches): the name reaches a directory name, a stop
# sequence AND console prints, and a cp1252 console crashes on a unicode
# print -- the rule the repo's ASCII gate exists to hold.
_SAFE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9 _-]{0,31}$")

# A pack is a DIRECTORY: the mechanical fields live in this file inside it,
# beside the content files `persona_content.load_content` reads. The name is
# spelled once, here, so renaming the format is one edit rather than a grep --
# and `serve --persona <dir>` needs nothing of serve's own.
PACK_MANIFEST = "pack.json"


@dataclass(frozen=True)
class Persona:
    """One AI's identity. Frozen: a persona is chosen at boot, not edited live."""

    name: str = ENIGMA["name"]
    data_dirname: str = ENIGMA["data_dirname"]
    name_meaning: str = ENIGMA["name_meaning"]
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # The name reaches a directory name, a stop-sequence, a system prompt
        # and the console. A name carrying a path separator, a newline or a
        # non-ASCII character would not be rejected by any of those on its own.
        if not _SAFE_NAME.fullmatch(self.name or ""):
            raise ValueError(
                f"persona name {self.name!r} must start with a letter and use only "
                "ASCII letters, digits, spaces, hyphens and underscores (max 32 chars)"
            )
        if not self.data_dirname or any(c in self.data_dirname for c in '/\\:*?"<>|' + "\n\r\t"):
            raise ValueError(f"persona data_dirname {self.data_dirname!r} is not a bare directory name")
        # A dot entry carries no separator and still escapes: "." puts `home`
        # on the profile root and ".." on its parent, and serve's boot WRITES
        # runtime state into whatever home resolves to. The name must be a bare
        # component -- what Path() keeps of it has to be all of it.
        if self.data_dirname in (".", "..") or Path(self.data_dirname).name != self.data_dirname:
            raise ValueError(
                f"persona data_dirname {self.data_dirname!r} must be a single directory "
                "component, not a dot entry or a path"
            )

    @property
    def is_default(self) -> bool:
        """Whether this IS Enigma -- what `Persona.load()` with no pack returns.

        Identity is a VALUE (the literals this dataclass replaced), not a flag
        someone passed, so a pack that spells out those same three values IS
        her -- including one that also carries `extra` keys of its own, which
        say nothing about who she is. Whole-dataclass equality read an extra
        key as a different AI and skipped her legacy state migration.
        Callers use it for the things that are Enigma's alone rather than
        every persona's -- the legacy repo-anchored runtime state."""
        return (self.name, self.data_dirname, self.name_meaning) == (
            ENIGMA["name"], ENIGMA["data_dirname"], ENIGMA["name_meaning"])

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
        default rather than a special case.

        `path` is either the pack DIRECTORY -- whose mechanical fields are read
        from the PACK_MANIFEST inside it -- or that manifest file directly. The
        mechanical half is all serve ever needed, so a bare file still loads;
        the content beside it in a directory is the data-makers' to read."""
        if path is None:
            return cls()
        path = Path(path)
        if not path.exists():
            raise SystemExit(f"persona pack not found: {path}")
        if path.is_dir():
            manifest = path / PACK_MANIFEST
            if not manifest.exists():
                raise SystemExit(
                    f"persona pack directory {path} has no {PACK_MANIFEST} -- a "
                    "directory pack carries it beside its content files"
                )
            path = manifest
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"persona pack {path} is not valid JSON ({exc.msg})") from None
        if not isinstance(blob, dict):
            raise SystemExit(f"persona pack {path} must be a JSON object")
        known = {f for f in ("name", "data_dirname", "name_meaning")}
        # A pack is untrusted data, and the character checks below only run on
        # strings: a numeric name dies in the regex with a TypeError traceback,
        # a non-str name reaches _slug as an AttributeError, and a FALSY
        # non-string data_dirname (0, JSON null) slips past the `or` into the
        # derived default instead of refusing. Type them here, by name.
        for _field in ("name", "data_dirname", "name_meaning"):
            if _field in blob and not isinstance(blob[_field], str):
                raise SystemExit(
                    f"persona pack {path}: {_field} must be a string, not "
                    f"{type(blob[_field]).__name__}"
                )
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
