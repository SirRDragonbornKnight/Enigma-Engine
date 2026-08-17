"""The curated pretrain shard builder, pinned at the identity seam.

`make_pretrain_curated.py` was parameterized MECHANICALLY only: the persona
supplied the NAME while the CONTENT still arrived by direct import
(`from identity_anchors import EXAMPLES`) and the knowledge renderer was
called with no content at all. A persona pack therefore pretrained
"<name> says: I'm Enigma..." -- content-by-import, which a grep census for
the name cannot see (audit 2026-08-15, finding 2).

What that costs is unrecoverable in a way the later stages are not: a
pretrain pass is not something SFT can talk the weights out of. So the three
facts pinned here are the whole seam -- the anchors follow the content, the
knowledge self-section follows the content, and a pack with no content of its
own (a bare pack FILE) REFUSES instead of rendering her identity under its
name, while a pack DIRECTORY builds that pack's own corpus -- plus the fourth
that keeps the change honest: Enigma's own corpus is byte-identical to what
the pre-change code produced.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

import knowledge_corpus
import make_pretrain_curated as mod
from enigma_engine.core.persona import Persona
from enigma_engine.core.persona_content import default_content

ATLAS = Persona(name="Atlas", data_dirname=".atlas", name_meaning="")


@pytest.fixture
def no_general_diet(monkeypatch):
    """Point the conversational source at nothing.

    `conversational_lines` is mechanical and reads a 105k-record file; these
    tests are about the identity groups, and a missing source returns []."""
    monkeypatch.setattr(mod, "GENERAL", mod.ROOT / "no_such_general_diet.jsonl")


def test_identity_lines_render_the_anchors_the_content_supplies(no_general_diet):
    """The defect in one assertion: swap the anchors and NOTHING of Enigma's
    authored voice may survive into the prose, in either person."""
    content = replace(default_content(), anchors={"identity": [("Who are you?", "Atlas.")]})
    lines = mod.identity_lines(ATLAS, content)

    assert "Atlas." in lines                    # first person, as trained
    assert "Atlas says: Atlas." in lines        # attributed, with the pack's name
    assert not any("Enigma" in line for line in lines), \
        [line for line in lines if "Enigma" in line]
    authored = {" ".join(said.split())
                for pairs in default_content().anchors.values()
                for _user, said in pairs}
    assert not (authored & set(lines))


def test_build_hands_the_content_to_the_knowledge_renderer(no_general_diet):
    """`build()` called gen_knowledge_pretrain_text() with no content, so the
    self-section rendered Enigma's self-facts whatever the persona was -- the
    same defect one call site over from the anchors."""
    content = replace(default_content(), self_facts=[(
        ["What is Atlas?"],
        ["Atlas is a local AI that runs on the user's own machine and answers there."],
    )])
    facts = mod.build(Persona(), content, 0)["facts"]

    assert "Atlas is a local AI that runs on the user's own machine and answers there." in facts
    assert not any("Enigma" in line for line in facts), \
        [line for line in facts if "Enigma" in line][:3]
    for questions, _answers in knowledge_corpus.SELF_FACTS:
        for q in questions:
            assert not any(q in line for line in facts), q
    # World facts are nobody's identity and must NOT follow the pack.
    world_q = knowledge_corpus.WORLD_FACTS[0][0][0]
    assert any(world_q in line for line in facts), world_q


def _pre_wave_2b_identity_lines(persona: Persona, anchors: dict) -> list[str]:
    """`identity_lines` exactly as it stood before the content seam landed.

    Frozen deliberately, and fed the anchors table the old code imported
    directly, so the comparison below measures the ONE thing this wave
    changed: where the anchors come from. Enigma's words are never spelled
    out here -- they are read from the authored module, which stays their
    home."""
    out: list[str] = []
    name = persona.name
    for _cat, pairs in anchors.items():
        for _user, said in pairs:
            said = " ".join(said.split())
            if not said:
                continue
            out.append(said)
            out.append(f"{name} says: {said}")
    if persona.name_meaning:
        out.append(f"{name} is called {name} because it means {persona.name_meaning}.")
    out += [
        f"{name} is a local AI that runs on the user's own machine.",
        f"{name} runs locally, so nothing said to {name} leaves the machine.",
        f"The AI called {name} is not a cloud service and has no telemetry.",
        f"{name} is a language model that runs on the user's own hardware.",
        f"When you talk to {name}, the conversation stays on your computer.",
    ]
    return out


def test_enigmas_identity_group_is_byte_identical_to_the_pre_change_render(no_general_diet):
    """Her corpus does not move because the seam was introduced.

    `default_content().anchors` IS `identity_anchors.EXAMPLES` -- the same
    mapping object, so the same items in the same insertion order -- which is
    why the loop over the content reproduces the loop over the import
    line-for-line."""
    import identity_anchors

    content = default_content()
    assert content.anchors is identity_anchors.EXAMPLES
    got = mod.build(Persona(), content, 0)["identity"]
    assert got == _pre_wave_2b_identity_lines(Persona(), identity_anchors.EXAMPLES)


def test_a_bare_pack_file_refuses_and_writes_nothing(tmp_path, monkeypatch):
    """A pack FILE carries the mechanical fields only -- a name with no content
    behind it -- so the only identity this script could render under it is
    hers. The refusal points at the DIRECTORY format and fires before the
    guard, the build and the writer; an out dir that was never created is the
    receipt."""
    pack = tmp_path / "atlas.json"
    pack.write_text('{"name": "Atlas"}', encoding="utf-8")
    out = tmp_path / "curated"

    def _writer_reached(*_args, **_kwargs):
        raise AssertionError("write_shards ran -- the refusal came too late")

    monkeypatch.setattr(mod, "write_shards", _writer_reached)
    monkeypatch.setattr("sys.argv", ["make_pretrain_curated.py",
                                     "--persona", str(pack), "--out", str(out)])
    with pytest.raises(SystemExit, match="bare pack file"):
        mod.main()
    assert not out.exists()


def test_a_pack_directory_builds_that_packs_own_corpus(
        tmp_path, monkeypatch, no_general_diet, write_persona_pack):
    """The other side of the refusal, and the proof the loader is WIRED: a
    directory pack has content of its own, so main() builds it instead of
    refusing, and the documents it would write carry Atlas rather than her.

    write_shards is intercepted rather than run: the claim under test is what
    the build produced, and the writer's shard/separator contract has its own
    test."""
    pack = write_persona_pack()
    written: dict[str, list[str]] = {}

    def _capture(kept, out_dir):
        written.update(kept)
        return 1

    monkeypatch.setattr(mod, "write_shards", _capture)
    monkeypatch.setattr("sys.argv", ["make_pretrain_curated.py", "--persona", str(pack),
                                     "--out", str(tmp_path / "curated")])
    mod.main()

    assert "Atlas says: Atlas." in written["identity"]
    assert not any("Enigma" in line for line in written["identity"]), \
        [line for line in written["identity"] if "Enigma" in line]
    assert any("Atlas is a local AI" in line for line in written["facts"])
    for questions, _answers in knowledge_corpus.SELF_FACTS:
        for q in questions:
            assert not any(q in line for line in written["facts"]), q


def test_the_default_persona_still_builds(tmp_path, monkeypatch, no_general_diet):
    """The refusal is scoped to packs the content loader cannot serve. Enigma
    is not one, and a guard that refused HER would be caught only by a real
    corpus rebuild."""
    out = tmp_path / "curated"
    monkeypatch.setattr("sys.argv", ["make_pretrain_curated.py", "--out", str(out)])
    mod.main()
    shards = sorted(out.glob("curated_*.txt"))
    assert shards
    assert "Enigma" in shards[0].read_text(encoding="utf-8")
