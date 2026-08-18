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
data home) are what actually stop two AIs from colliding on this box, so each
of them is derived from the persona rather than hardcoded: the home from
`data_dirname`, the mutex and the log names from the name, and the port from
the pack's own `port` field.
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

# Win32 reserves these device names in EVERY directory, case-insensitively and
# with or without an extension ("nul.txt" is the device too). A home named for
# one of them swallows every write and reads back empty.
_RESERVED_DIRNAMES = frozenset(
    ("CON", "PRN", "AUX", "NUL")
    + tuple(f"COM{i}" for i in range(1, 10))
    + tuple(f"LPT{i}" for i in range(1, 10))
)

# The two ports this machine already spends, and what a pack colliding with
# one of them costs. 8000 is the daily serve every launcher points at, so a
# pack claiming it can never come up beside her; 8123 is the eval scratch
# server, and an eval run CLEARS its target's memory store before probing.
RESERVED_PORTS = {
    8000: "the daily serve port every launcher points at",
    8123: "the eval scratch port, and an eval run CLEARS its target's memory store",
}

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
    # The port her launchers bind. None = Enigma's default (8000). It is a
    # MECHANICAL field: which port an AI answers on says nothing about who she
    # is, so a non-None port on the default identity is a flag rather than a
    # different AI -- packs are simply the intended carrier.
    port: int | None = None
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
        # Win32 strips trailing dots and spaces off a path component, so "..."
        # and "atlas." reach the filesystem as ".." and "atlas": the value is a
        # bare component to Path() and a DIFFERENT directory -- the profile
        # root, its parent, or an aliased sibling -- to the writes serve makes.
        if self.data_dirname != self.data_dirname.rstrip(". "):
            raise ValueError(
                f"persona data_dirname {self.data_dirname!r} must not end in a dot or "
                "a space: Win32 strips those, and the home lands somewhere else"
            )
        # The portion before the first dot is what Win32 matches the device
        # names on; ".atlas" has none, and matches nothing.
        if self.data_dirname.split(".", 1)[0].upper() in _RESERVED_DIRNAMES:
            raise ValueError(
                f"persona data_dirname {self.data_dirname!r} is a reserved Win32 "
                "device name, and a home named for one eats every write"
            )
        # Directory names are case-insensitive on Win32, so ".Enigma_Engine" is
        # a different value to the three-field compare and Enigma's OWN home on
        # disk. Only her exact identity may resolve there.
        if (not self.is_default
                and self.data_dirname.casefold() == ENIGMA["data_dirname"].casefold()):
            raise ValueError(
                f"persona data_dirname {self.data_dirname!r} is Enigma's data home "
                "under a different identity (directory names are case-insensitive "
                "on Win32)"
            )
        # name_meaning renders raw into ONE line of the pretrain identity
        # document (make_pretrain_curated.identity_lines), and that line is a
        # whole record in a shard. U+001E is the record separator itself, so it
        # SPLITS the document; a blank line splits it into paragraphs that
        # pretokenize dedups independently; every other control character bakes
        # into the corpus. Unicode stays allowed -- the ASCII rule is
        # console-bound, and training text carries unicode.
        bad = next((c for c in self.name_meaning if ord(c) < 32 or ord(c) == 127), None)
        if bad is not None:
            raise ValueError(
                f"persona name_meaning carries control character U+{ord(bad):04X}; it "
                "renders into a single corpus line"
            )
        if self.port is not None:
            # bool IS an int to Python, so `"port": true` passes an isinstance
            # check and binds port 1 -- a privileged port, silently.
            if isinstance(self.port, bool) or not isinstance(self.port, int):
                raise ValueError(
                    f"persona port {self.port!r} must be an integer, not "
                    f"{type(self.port).__name__}"
                )
            if not 1024 <= self.port <= 65535:
                raise ValueError(
                    f"persona port {self.port} is outside 1024-65535: below 1024 needs "
                    "elevation to bind and 65535 is the last port there is"
                )
            if self.port in RESERVED_PORTS:
                raise ValueError(
                    f"persona port {self.port} collides with {RESERVED_PORTS[self.port]}"
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
    def slug(self) -> str:
        """Her name as a bare identifier: lowercased, runs of anything else
        folded to one underscore.

        The id `/v1/models` publishes, so an OpenAI client asking WHICH model
        answered gets the AI rather than a constant every persona shares.
        Enigma's is "enigma" -- the literal that endpoint carried."""
        return _slug(self.name)

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
            blob = read_pack_json(path)
        except DuplicateJSONKey as exc:
            raise SystemExit(
                f"persona pack {path} names {exc.key!r} twice; the second value "
                "would win and the first would never be read"
            ) from None
        except json.JSONDecodeError as exc:
            raise SystemExit(f"persona pack {path} is not valid JSON ({exc.msg})") from None
        if not isinstance(blob, dict):
            raise SystemExit(f"persona pack {path} must be a JSON object")
        known = {f for f in ("name", "data_dirname", "name_meaning", "port")}
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
                # Straight through, whatever its type: the dataclass owns the
                # port rules, so a pack and a caller building a Persona by hand
                # are refused by the same lines.
                port=blob.get("port"),
                extra={k: v for k, v in blob.items() if k not in known},
            )
        except ValueError as exc:  # an invalid pack is a refusal, not a traceback
            raise SystemExit(f"persona pack {path}: {exc}") from None


class DuplicateJSONKey(ValueError):
    """A JSON object naming one key twice. `key` is the repeated name."""

    def __init__(self, key: str) -> None:
        super().__init__(f"duplicate key {key!r}")
        self.key = key


def _object_pairs(pairs: list[tuple[str, object]]) -> dict:
    """The pairs as a dict, refusing a key that appears twice.

    json's own last-wins is silent, and at ANY object level: two entries for
    one aside key drop the author's first record out of the corpus with
    nothing downstream able to see the hole."""
    out: dict = {}
    for key, value in pairs:
        if key in out:
            raise DuplicateJSONKey(key)
        out[key] = value
    return out


def read_pack_json(path: Path) -> object:
    """A pack JSON file's contents, for the pack readers that share the rules.

    The ONE reader for both halves of a pack -- this module's manifest and
    `persona_content`'s content files -- so the two cannot disagree about what
    a pack file is. Callers keep their own refusal shapes and catch
    `DuplicateJSONKey` and `json.JSONDecodeError`.

    A BOM is tolerated: Windows editors write one, and a BOM decoded as plain
    utf-8 dies as "Expecting value", pointing a pack author at their syntax
    when the defect is their encoding."""
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_object_pairs)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "ai"
