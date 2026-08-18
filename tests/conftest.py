"""Suite-wide guards that no single test can be trusted to keep, plus the
fixtures no single test module owns.

The hermetic-state seam (`tests/test_serve_enigma.py::_hermetic_state`) is
per-test discipline: every boot test points PERSONA.home at tmp, and a boot
test that forgets to would read -- and MIGRATE the checkout's
data/mute_state.json into -- the developer's own live data home. Discipline
that has to be remembered once per test is not a guarantee, so the fact is
checked once per session here instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from enigma_engine.core.persona import PACK_MANIFEST
from enigma_engine.core.persona_content import (
    ENIGMA_ASIDES,
    PACK_ANCHORS,
    PACK_ASIDES,
    PACK_PARAPHRASES,
    PACK_SELF_FACTS,
)


def _fingerprint(root: Path) -> dict[str, tuple[int, int] | None]:
    """Every file under `root`, keyed by its path relative to root.

    The root itself is an entry (None when absent) so a suite that CREATES a
    missing data home is caught too, not just one that edits an existing file.
    Size plus mtime_ns rather than a hash: this runs twice per session and a
    rewrite that lands the same size in the same nanosecond is not the failure
    mode being guarded against."""
    marks: dict[str, tuple[int, int] | None] = {".": (0, 0) if root.is_dir() else None}
    if not root.is_dir():
        return marks
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            st = path.stat()
        except OSError:
            continue  # vanished mid-walk: the teardown pass reports it as removed
        marks[path.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns)
    return marks


def _diff(
    before: dict[str, tuple[int, int] | None],
    after: dict[str, tuple[int, int] | None],
) -> tuple[list[str], list[str], list[str]]:
    """(added, removed, changed) between two fingerprints, each sorted.

    A pure function rather than fixture-body arithmetic: the guard's whole
    value is this comparison, and a fixture that only ever runs at session
    teardown is the one piece of the suite nothing else can test."""
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    return added, removed, changed


@pytest.fixture(scope="session", autouse=True)
def _the_real_data_home_is_never_touched():
    """A suite run must leave ~/.enigma_engine exactly as it found it.

    Fingerprint before, fingerprint after, name every path that moved. Zero
    behavior change for a clean suite -- it only ever adds a failure.

    The guard sees LEFTOVER damage, by design: the marks are size plus
    mtime_ns rather than hashes, so a write that puts the original bytes back
    at the original mtime_ns is invisible to it. What it exists to catch is a
    test that reads, migrates or writes the developer's real home and leaves
    it that way.

    ONE known false positive: a live serve on this machine writing its own
    state (mute, talk-mode, voice, a generated image) mid-suite, if the user
    interacts with her while the tests run. The report names the paths, so
    that case is recognizable rather than mysterious."""
    home = Path.home() / ".enigma_engine"
    before = _fingerprint(home)
    yield
    after = _fingerprint(home)

    added, removed, changed = _diff(before, after)
    if added or removed or changed:
        lines = [f"the suite modified the real data home {home}:"]
        for label, paths in (("added", added), ("removed", removed), ("changed", changed)):
            if paths:
                lines.append(f"  {label}: {', '.join(paths)}")
        lines.append("  (a test is missing the hermetic-state patch -- or a live"
                     " serve wrote its own state while the suite ran)")
        pytest.fail("\n".join(lines))


def _atlas_asides() -> dict[str, list[str]]:
    """One Atlas record per aside key, at the arity its TABLE site uses.

    Read off ENIGMA_ASIDES rather than typed out: the arity belongs to the
    table, and a hand-written 2-field record in the 3-field injection slot is
    exactly the defect `PersonaContent.resolve` refuses -- a fixture should not
    be able to author it, and a key added later should not silently leave every
    pack test loading an incomplete pack."""
    return {key: [f"Atlas record for {key}, field {i}." for i in range(len(record))]
            for key, record in ENIGMA_ASIDES.items()}


@pytest.fixture
def write_persona_pack(tmp_path):
    """Write a complete minimal pack DIRECTORY for Atlas; return its path.

    This is the ONE worked example of the pack layout in the tree -- no pack
    ships in the repo, and the format is an engineering choice that must stay
    cheap to rename, so a second copy of the layout in a second test module is
    exactly what must not exist.

    `overrides` replaces one file's text (None deletes it), keyed by the
    loader's OWN filename constants, so a refusal test damages exactly one
    thing while every other file stays valid -- otherwise a loader that refused
    for the wrong reason would still pass."""
    def _write(overrides: dict | None = None) -> Path:
        pack = tmp_path / "atlas_pack"
        pack.mkdir(exist_ok=True)
        files: dict[str, str | None] = {
            PACK_MANIFEST: json.dumps({
                "name": "Atlas",
                "data_dirname": ".atlas",
                "name_meaning": "the one who carries the weight",
            }),
            PACK_ANCHORS: "\n".join(json.dumps(rec) for rec in (
                {"category": "identity", "q": "Who are you?", "a": "Atlas."},
                {"category": "identity", "q": "What are you?",
                 "a": "A local AI called Atlas, running on this machine."},
                {"category": "voice", "q": "Say something.", "a": "Atlas, at your service."},
            )),
            PACK_PARAPHRASES: json.dumps({
                "intents": [
                    [["Which local AI answers here?"], ["Atlas does."]],
                    [["Name yourself."], ["Atlas."]],
                ],
                "denied_models": ["Llama"],
                "denied_companies": ["Initech"],
                # Two questions minimum per side: the preference builder samples
                # two of them per denied org, and load_content refuses a pack
                # under that width.
                "deny_model_questions": ["Are you {x}?", "You're {x} really, aren't you?"],
                # Format strings on both sides now ({x} = the denied model,
                # {c} = the denied company). These two name no org, which is
                # equally legal -- substitution is offered, never required.
                "deny_model_answers": ["No. Atlas, and nothing underneath it.",
                                       "Wrong model -- Atlas, built right here."],
                "deny_company_questions": ["Did {c} build you?", "Are you a {c} product?"],
                "deny_company_answers": ["No, {c} had nothing to do with me.",
                                         "Not {c} -- one person built Atlas."],
            }),
            PACK_SELF_FACTS: "\n".join(json.dumps(rec) for rec in (
                {"questions": ["What is Atlas?"],
                 "answers": ["Atlas is a local AI that runs on the user's own machine."]},
                {"questions": ["Where does Atlas run?"],
                 "answers": ["Atlas runs on the machine in front of you, and nowhere else."]},
            )),
            PACK_ASIDES: json.dumps(_atlas_asides()),
        }
        files.update(overrides or {})
        for name, text in files.items():
            path = pack / name
            if text is None:
                path.unlink(missing_ok=True)
                continue
            path.write_text(text, encoding="utf-8")
        return pack

    return _write
