"""`make_sft_data.py` at the persona seam -- the CLI half.

The generators already took a `content=` argument; `main()` called every one
of them with nothing, so the seam existed in the functions and no command line
could reach it (PERSONA_PACK_AUTHORING's "SFT / DPO gap"). A pack therefore had
no instruct corpus at all, and the nearest thing to one -- pointing the builder
at a pack by hand -- would have written another AI's records into `data/sft`,
over the SERVED model's training receipt.

What is pinned here is the wiring and the four ways it can be quietly wrong:

  - the artifacts land in HER directory anyway, or the pack's build rotates it;
  - a generator keeps rendering Enigma under the pack's name -- the decorative
    failure, which passes any check that only looks for Atlas;
  - the build screens against HER sealed gate, dropping pack records for
    resembling questions that AI is never asked;
  - `teachings.jsonl` -- the user's own channel into ENIGMA's weights -- rides
    into a pack's corpus.

Her own build is the control in every one of them: the default path must come
out exactly as it did, which the shadow byte-identity run proves at corpus
scale and these tests hold at the seam.
"""
from __future__ import annotations

import json

import pytest

import make_sft_data as mod
from enigma_engine.core.persona import Persona


def _run(monkeypatch, tmp_path, argv):
    """Run `main(argv)` with the general diet emptied, and hand back the
    directory that stands in for `data/sft`.

    Only the 105k-record general corpus is stubbed: every generator runs for
    real, because WHICH content they render is the whole claim. `OUT_DIR` moves
    to tmp_path -- a test must never rotate data/sft, those artifacts are the
    served model's training receipt -- and PREAMBLE is registered with
    monkeypatch before the run so main()'s rebind is undone afterwards."""
    hers = tmp_path / "hers"
    empty = tmp_path / "empty_general.jsonl"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setattr(mod, "GENERAL", empty)
    monkeypatch.setattr(mod, "OUT_DIR", hers)
    monkeypatch.setattr(mod, "PREAMBLE", mod.PREAMBLE)
    mod.main(argv)
    return hers


def _records(path):
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln]


def _system_blocks(recs):
    return [m["content"] for r in recs for m in r["messages"] if m["role"] == "system"]


ATLAS_PREAMBLE = Persona(name="Atlas", data_dirname=".atlas", name_meaning="").tools_preamble


# ---- the destination ----


def test_a_pack_with_no_out_refuses_rather_than_rotating_her_mix(
        tmp_path, monkeypatch, write_persona_pack):
    """The default destination is HER data/sft, and `_write_artifact` rotates
    whatever it finds there: a pack built without --out replaces the served
    model's training receipt with another AI's corpus, and the second such run
    takes the .prev generation too.

    The refusal names the directory it is protecting and lands before the
    writer -- a directory that was never created is the receipt."""
    pack = write_persona_pack()

    def _writer_reached(*_args, **_kwargs):
        raise AssertionError("_write_artifact ran -- the refusal came too late")

    monkeypatch.setattr(mod, "_write_artifact", _writer_reached)
    with pytest.raises(SystemExit, match=str(tmp_path / "hers").replace("\\", "\\\\")):
        _run(monkeypatch, tmp_path, ["--persona", str(pack)])
    assert not (tmp_path / "hers").exists()


def test_a_bare_pack_file_refuses_and_writes_nothing(tmp_path, monkeypatch):
    """A pack FILE carries the mechanical fields only -- a name with nothing
    behind it -- so the only identity this build could render under it is hers.
    Same refusal the curated pretrain builder makes, for the same reason, and
    it fires before the guard, the generators and the writer."""
    pack = tmp_path / "atlas.json"
    pack.write_text('{"name": "Atlas"}', encoding="utf-8")

    def _writer_reached(*_args, **_kwargs):
        raise AssertionError("_write_artifact ran -- the refusal came too late")

    monkeypatch.setattr(mod, "_write_artifact", _writer_reached)
    with pytest.raises(SystemExit, match="bare pack file"):
        _run(monkeypatch, tmp_path, ["--persona", str(pack), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_a_packs_artifacts_all_land_in_its_own_out_dir(
        tmp_path, monkeypatch, write_persona_pack):
    """Every file the build writes follows --out, including the two that are
    easy to leave behind: the manifest, and the near-miss review file, which is
    the only one written by an unlink-or-write path rather than the rotating
    writer. A stale review file is planted in BOTH directories -- the pack's
    copy must be cleared as this run's own stale output, hers must be untouched
    by a build that is not hers."""
    pack = write_persona_pack()
    out = tmp_path / "atlas_sft"
    out.mkdir()
    (out / "locked_near_misses.jsonl").write_text("stale pack review\n", encoding="utf-8")
    hers = tmp_path / "hers"
    hers.mkdir()
    (hers / "locked_near_misses.jsonl").write_text("HER review\n", encoding="utf-8")

    _run(monkeypatch, tmp_path, ["--persona", str(pack), "--out", str(out)])

    for name in ("tool_calls.jsonl", "identity.jsonl", "mix.jsonl", "mix.manifest.json"):
        assert (out / name).is_file(), f"{name} did not land in --out"
        assert not (hers / name).exists(), f"{name} landed in her directory"
    assert not (out / "locked_near_misses.jsonl").exists(), \
        "the pack's stale review file was left behind"
    assert (hers / "locked_near_misses.jsonl").read_text(encoding="utf-8") == "HER review\n"


def test_an_out_on_her_own_build_redirects_without_refusing(tmp_path, monkeypatch):
    """--out is a plain redirect for the default persona -- a shadow render of
    her corpus somewhere harmless is exactly what checks a change to this
    builder. Only a PACK has to name a destination."""
    out = tmp_path / "shadow"
    hers = _run(monkeypatch, tmp_path, ["--out", str(out)])

    assert (out / "mix.jsonl").is_file()
    assert not hers.exists(), "the default directory was written as well"
    assert mod.PREAMBLE in _system_blocks(_records(out / "tool_calls.jsonl"))[0]


# ---- the content ----


def test_a_pack_build_renders_the_packs_own_identity_everywhere(
        tmp_path, monkeypatch, write_persona_pack):
    """The four seamed generators in one build: anchors, paraphrases, the
    knowledge self-section and the built-in restraint asides all follow the
    pack, and Enigma's ABSENCE is the half that catches the decorative
    failure -- a generator quietly falling back to `default_content()` renders
    HER under Atlas's name and passes any check that only looks for Atlas."""
    pack = write_persona_pack()
    out = tmp_path / "atlas_sft"
    _run(monkeypatch, tmp_path, ["--persona", str(pack), "--out", str(out)])

    ident = _records(out / "identity.jsonl")
    said = {r["messages"][1]["content"] for r in ident}
    asked = {r["messages"][0]["content"] for r in ident}
    assert "Atlas." in said                                   # anchors
    assert "Atlas does." in said                              # paraphrase intents
    assert "Are you Llama?" in asked                          # templated denials
    assert "No. Atlas, and nothing underneath it." in said

    mix = _records(out / "mix.jsonl")
    knowledge = {r["messages"][0]["content"] for r in mix if r["category"] == "knowledge"}
    assert "What is Atlas?" in knowledge                      # the self-section
    assert "Which planet is the biggest one?" in knowledge, \
        "world facts are nobody's identity and must not follow the pack"

    restraint = {r["messages"][-1]["content"] for r in mix
                 if r["category"] in ("tool_restraint", "builtin_restraint")}
    assert "Atlas record for who_are_you, field 1." in restraint   # the asides

    everything = " ".join(json.dumps(r, ensure_ascii=False) for r in mix)
    assert "Enigma" not in everything, "her identity reached another AI's corpus"
    assert "SirRulean" not in everything


def test_every_system_block_in_a_pack_build_carries_the_packs_preamble(
        tmp_path, monkeypatch, write_persona_pack):
    """PREAMBLE is read by `_system` and `_builtin_system`, which between them
    build nearly every system message in the mix. Left at its import-time
    value, a pack's entire tool corpus tells the model "You are Enigma" while
    every identity record says Atlas -- the two halves of the same build
    disagreeing about who is answering."""
    pack = write_persona_pack()
    out = tmp_path / "atlas_sft"
    _run(monkeypatch, tmp_path, ["--persona", str(pack), "--out", str(out)])

    blocks = _system_blocks(_records(out / "tool_calls.jsonl")) \
        + _system_blocks(_records(out / "mix.jsonl"))
    offered = [b for b in blocks if "Available tools:" in b]
    assert offered, "no tools block in the corpus at all"
    assert all(ATLAS_PREAMBLE in b for b in offered)
    # Memory-only blocks carry no preamble at all, which is the shape serve
    # renders too. The claim is about every block that HAS one.
    assert not any("You are " in b and ATLAS_PREAMBLE not in b for b in blocks), \
        [b for b in blocks if "You are " in b and ATLAS_PREAMBLE not in b][:1]


def test_her_own_build_is_unmoved_by_the_seam(tmp_path, monkeypatch):
    """The control. No --persona is Enigma, and everything the seam touches has
    to come out as it did: her preamble in the system blocks, her anchors in
    identity.jsonl, her self-facts in the knowledge records.

    (Corpus-scale byte identity is proved by the shadow render, not here -- a
    full default build with the real general diet is minutes of tokenizing.)"""
    hers = _run(monkeypatch, tmp_path, [])

    assert mod.PREAMBLE == (
        "You are Enigma. You can use tools when they are needed; "
        "answer directly when they are not.")
    said = {r["messages"][1]["content"] for r in _records(hers / "identity.jsonl")}
    assert "Enigma. That's the name and the nature." in said
    knowledge = {r["messages"][0]["content"] for r in _records(hers / "mix.jsonl")
                 if r["category"] == "knowledge"}
    assert "What is Enigma?" in knowledge
    assert not any("You are Atlas" in b
                   for b in _system_blocks(_records(hers / "tool_calls.jsonl")))


# ---- the gate ----


def test_a_pack_is_screened_against_its_own_sealed_gate_not_hers(
        tmp_path, monkeypatch, write_persona_pack):
    """The build screened with `LockedProbeGuard.load()` -- the default
    manifest, HERS -- whatever it was building. That drops pack records for
    resembling questions that AI will never be asked, while guarding a gate its
    weights are not trained for.

    Both halves are measured. The manifest the build LOADS is recorded, and a
    real bite: the pack seals one probe, a NEAR-PARAPHRASE of it is authored
    into the pack's intents, and only the pack's own guard has any opinion
    about that phrasing. The paraphrase is not an exact match to the sealed
    question, so its absence from the corpus is the fuzzy guard's doing and
    nothing else's."""
    import eval_leak_guard

    sealed_q = "Which local AI answers here?"
    near = "Which local AI is answering here?"
    pack = write_persona_pack({"paraphrases.json": json.dumps({
        "intents": [[[sealed_q, near], ["Atlas does."]],
                    [["Name yourself."], ["Atlas."]]],
        "denied_models": ["Llama"],
        "denied_companies": ["Initech"],
        "deny_model_questions": ["Are you {x}?", "You're {x}, no?", "Admit it, you're {x}."],
        "deny_model_answers": ["No. Atlas, and nothing underneath it."],
        "deny_company_questions": ["Did {c} build you?", "A {c} product?", "So {c} built you?"],
        "deny_company_answers": ["No, {c} had nothing to do with me."],
    })})
    probes = pack / eval_leak_guard.LOCKED_PROBES_NAME
    probes.write_text(json.dumps(
        {"category": "identity", "q": sealed_q, "want_any": ["atlas"], "deny_any": []}) + "\n",
        encoding="utf-8", newline="\n")
    assert eval_leak_guard.main(["seal", str(probes)]) == 0

    real_load = eval_leak_guard.LockedProbeGuard.load
    loaded = {}

    def _record(path=eval_leak_guard.LOCKED_MANIFEST):
        loaded["path"] = path
        return real_load(path)

    hers_guard = real_load(eval_leak_guard.LOCKED_MANIFEST)
    pack_guard = real_load(eval_leak_guard.persona_manifest(pack))
    assert pack_guard.leaks(near), "the pack's own gate does not catch the paraphrase"
    assert not hers_guard.leaks(near), "her gate already catches it -- no bite to measure"

    monkeypatch.setattr(mod.LockedProbeGuard, "load", _record)
    out = tmp_path / "atlas_sft"
    _run(monkeypatch, tmp_path, ["--persona", str(pack), "--out", str(out)])

    # The corpus first: under her gate the paraphrase survives, so this is the
    # behavioral kill and not only a bookkeeping one.
    asked = {r["messages"][0]["content"] for r in _records(out / "identity.jsonl")}
    assert near not in asked, "the pack's own sealed paraphrase trained anyway"
    assert sealed_q not in asked, "the sealed question itself trained"
    assert "Name yourself." in asked, "the screen ate more than the sealed intent"
    assert loaded["path"] == eval_leak_guard.persona_manifest(pack)


def test_a_pack_with_no_sealed_set_says_the_dev_screen_is_inactive(
        tmp_path, monkeypatch, write_persona_pack, capsys):
    """Her `data/eval/behavior_probes.jsonl` is HER dev set: holding its
    questions out of a pack's corpus thins that pack for resembling questions
    its AI is never graded on. A pack reads its own sealed file instead, and
    with none it says so out loud rather than screening against nothing in
    silence."""
    pack = write_persona_pack()
    out = tmp_path / "atlas_sft"
    _run(monkeypatch, tmp_path, ["--persona", str(pack), "--out", str(out)])

    printed = capsys.readouterr().out
    assert "WARN: Atlas has no sealed set" in printed
    assert str(pack / "locked_probes.jsonl") in printed
    assert "persona: Atlas" in printed


def test_a_packs_own_sealed_questions_are_held_out_by_the_exact_screen(
        tmp_path, monkeypatch, write_persona_pack):
    """The other side of that swap: a pack WITH a sealed file gets the same
    exact-match backstop her dev probes give her, read off its own file. The
    manifest is deliberately not sealed here, so the fuzzy guard is inactive
    and the only thing that can hold the question out is the exact screen."""
    pack = write_persona_pack()
    (pack / "locked_probes.jsonl").write_text(json.dumps(
        {"category": "identity", "q": "Which local AI answers here?",
         "want_any": ["atlas"]}) + "\n", encoding="utf-8", newline="\n")
    out = tmp_path / "atlas_sft"
    _run(monkeypatch, tmp_path, ["--persona", str(pack), "--out", str(out)])

    asked = {r["messages"][0]["content"] for r in _records(out / "identity.jsonl")}
    assert "Which local AI answers here?" not in asked
    assert "Name yourself." in asked, "the screen ate more than its probe"


# ---- teachings ----


def test_teachings_are_hers_alone_and_never_a_packs(
        tmp_path, monkeypatch, write_persona_pack, capsys):
    """`teachings.jsonl` is the user's channel into ENIGMA's weights -- facts
    he taught HER, in her own sessions. There is no sense in which another AI
    was told them, so a pack build skips the generator entirely and says so;
    her build reads it exactly as before."""
    taught = {"messages": [{"role": "user", "content": "What is the zzyzx ledger?"},
                           {"role": "assistant", "content": "A test fixture."}],
              "category": "teaching"}
    monkeypatch.setattr(mod, "gen_teaching_examples", lambda *a, **k: [dict(taught)])

    pack = write_persona_pack()
    out = tmp_path / "atlas_sft"
    _run(monkeypatch, tmp_path, ["--persona", str(pack), "--out", str(out)])
    printed = capsys.readouterr().out

    assert not [r for r in _records(out / "mix.jsonl") if r["category"] == "teaching"], \
        "a pack trained on facts the user taught Enigma"
    assert json.loads((out / "mix.manifest.json").read_text(
        encoding="utf-8"))["teaching_records"] == 0
    assert "teachings: SKIPPED" in printed
    assert "Enigma's own channel" in printed


def test_her_build_still_reads_the_teachings_file(tmp_path, monkeypatch, capsys):
    """The other half of the exclusion, and the reason it has to be keyed on
    the persona rather than deleted: her channel still trains."""
    taught = {"messages": [{"role": "user", "content": "What is the zzyzx ledger?"},
                           {"role": "assistant", "content": "A test fixture."}],
              "category": "teaching"}
    monkeypatch.setattr(mod, "gen_teaching_examples", lambda *a, **k: [dict(taught)])

    hers = _run(monkeypatch, tmp_path, [])
    printed = capsys.readouterr().out

    assert "teachings: 1 records" in printed
    assert [r for r in _records(hers / "mix.jsonl") if r["category"] == "teaching"]
    assert json.loads((hers / "mix.manifest.json").read_text(
        encoding="utf-8"))["teaching_records"] == 1
