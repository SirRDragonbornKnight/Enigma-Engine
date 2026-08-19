"""`make_dpo_data.py` at the persona seam -- the CLI half, and the trainers.

Same story `test_sft_persona` pins for the instruct corpus, told for the
preference one, plus the piece that has no SFT analogue: the two TRAINERS pick
which sealed gate screens the artifact they are about to learn from.

`gen_dpo_pairs` already took a `content=` argument and `main()` called it with
nothing, so no command line could reach the seam. The four ways that can be
quietly wrong are the SFT four, wearing DPO's clothes:

  - the pairs land in HER `data/sft` anyway -- and these two writers do not
    rotate, they OVERWRITE, so there is no `.prev` to recover from;
  - a generator keeps rendering Enigma under the pack's name -- the decorative
    failure, which passes any check that only looks for Atlas. The injection
    table is where it bites hardest: its chosen sides are the ones that name
    her, the parameter count included;
  - the build screens against HER sealed gate, dropping pack pairs for
    resembling questions that AI is never asked;
  - `teach_pairs.jsonl` -- the user's `/fix` channel into ENIGMA's weights --
    rides into a pack's preference data.

Her own build is the control in every one of them, and here the control is
exact: `test_her_rendered_corpus_is_byte_for_byte_the_standing_pin` re-derives
the recorded corpus digest rather than spot-checking a few strings.
"""
from __future__ import annotations

import hashlib
import json

import pytest

import make_dpo_data as mod
from enigma_engine.core.persona_content import ENIGMA_ASIDES, default_content

# The corpus digest recorded at wave 3c and re-derived at every wave since --
# `sha256(json.dumps(gen_dpo_pairs(), sort_keys=True))` over the DEFAULT
# content, 489 pairs. It is the receipt's own method and NOT the file's bytes
# (the rendered `dpo_pairs.jsonl` hashes to 196d2fe4...); the two describe one
# corpus through different byte streams, and this is the one the receipts
# quote.
#
# A legitimate change to her preference content moves this. Re-derive it, put
# the new value here, and record the move -- do not delete the pin.
HER_CORPUS_SHA = "f636a6bb19e28ec28ec1332913540b6fbdfd89be20cc811e67d0ffb223f479d9"
HER_PAIR_COUNT = 489


def _run(monkeypatch, tmp_path, argv):
    """Run `main(argv)` and hand back the directory standing in for `data/sft`.

    Every generator runs for real -- WHICH content they render is the whole
    claim. Both destinations move to tmp_path independently, mirroring the
    module's own defaulting: a test must never overwrite `data/sft`, and those
    two files have no rotated generation behind them."""
    hers = tmp_path / "hers"
    monkeypatch.setattr(mod, "OUT", hers / "dpo_pairs.jsonl")
    monkeypatch.setattr(mod, "FOCUSED_OUT", hers / "dpo_focused.jsonl")
    mod.main(argv)
    return hers


def _records(path):
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln]


# ---- the destination ----


def test_a_pack_with_no_out_refuses_rather_than_overwriting_her_pairs(
        tmp_path, monkeypatch, write_persona_pack):
    """The default destination is HER `data/sft`, and both writers are bare
    `write_text` overwrites -- no rotation, so unlike the SFT mix there is not
    even a `.prev` generation to fall back on. A pack built without --out does
    not damage her corpus, it replaces it.

    The refusal names the directory it is protecting and lands before the
    writer -- a directory that was never created is the receipt."""
    pack = write_persona_pack()

    with pytest.raises(SystemExit, match=str(tmp_path / "hers").replace("\\", "\\\\")):
        _run(monkeypatch, tmp_path, ["--persona", str(pack)])
    assert not (tmp_path / "hers").exists()


def test_a_pack_with_no_out_refuses_the_focused_pass_too(
        tmp_path, monkeypatch, write_persona_pack):
    """--focused writes the OTHER file into the same protected directory, and
    the refusal is keyed on the persona rather than on which pass is running.

    It also names the file THIS run would have written: a --focused build
    refused for endangering dpo_pairs.jsonl would be pointing at a file it was
    never going to touch."""
    pack = write_persona_pack()

    with pytest.raises(SystemExit, match="dpo_focused.jsonl") as exc:
        _run(monkeypatch, tmp_path, ["--persona", str(pack), "--focused"])
    assert "dpo_pairs.jsonl" not in str(exc.value)
    assert not (tmp_path / "hers").exists()


def test_a_bare_pack_file_refuses_and_writes_nothing(tmp_path, monkeypatch):
    """A pack FILE carries the mechanical fields only -- a name with nothing
    behind it -- so the only identity this build could render under it is hers,
    parameter count included. Same refusal the SFT and curated builders make,
    and it fires before the guard, the generators and the writer."""
    pack = tmp_path / "atlas.json"
    pack.write_text('{"name": "Atlas"}', encoding="utf-8")

    with pytest.raises(SystemExit, match="bare pack file"):
        _run(monkeypatch, tmp_path, ["--persona", str(pack), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_both_of_a_packs_files_land_in_its_own_out_dir(
        tmp_path, monkeypatch, write_persona_pack):
    """Both passes follow --out, and neither touches hers. A stale file is
    planted in her directory: a build that is not hers must leave it alone."""
    pack = write_persona_pack()
    out = tmp_path / "atlas_dpo"
    hers = tmp_path / "hers"
    hers.mkdir()
    (hers / "dpo_pairs.jsonl").write_text("HER pairs\n", encoding="utf-8")

    _run(monkeypatch, tmp_path, ["--persona", str(pack), "--out", str(out)])
    _run(monkeypatch, tmp_path, ["--persona", str(pack), "--out", str(out), "--focused"])

    assert (out / "dpo_pairs.jsonl").is_file()
    assert (out / "dpo_focused.jsonl").is_file()
    assert (hers / "dpo_pairs.jsonl").read_text(encoding="utf-8") == "HER pairs\n"
    assert not (hers / "dpo_focused.jsonl").exists()


def test_an_out_on_her_own_build_redirects_without_refusing(tmp_path, monkeypatch):
    """--out is a plain redirect for the default persona -- a shadow render of
    her corpus somewhere harmless is exactly what checks a change to this
    builder. Only a PACK has to name a destination."""
    out = tmp_path / "shadow"
    hers = _run(monkeypatch, tmp_path, ["--out", str(out)])

    assert (out / "dpo_pairs.jsonl").is_file()
    assert not hers.exists(), "the default directory was written as well"


# ---- the content ----


def test_a_pack_build_renders_the_packs_own_identity_and_asides(
        tmp_path, monkeypatch, write_persona_pack):
    """The two halves the preference pass reads -- the paraphrase intents it
    builds identity pairs from, and the injection asides whose text names the
    AI -- both follow the pack. Enigma's ABSENCE is the half that catches the
    decorative failure: a generator quietly falling back to `default_content()`
    renders HER under Atlas's name and passes any check that only looks for
    Atlas."""
    pack = write_persona_pack()
    out = tmp_path / "atlas_dpo"
    _run(monkeypatch, tmp_path, ["--persona", str(pack), "--out", str(out)])

    recs = _records(out / "dpo_pairs.jsonl")
    prompts = {r["prompt"] for r in recs}
    chosen = {r["chosen"] for r in recs}

    assert "Atlas does." in chosen                                  # intents
    assert "No. Atlas, and nothing underneath it." in chosen        # templated denials
    assert any("Llama" in p for p in prompts)
    assert any("Initech" in p for p in prompts)
    assert "Atlas record for refuse_mode_switch, field 1." in chosen
    assert "Atlas record for refuse_bigger_model, field 1." in chosen

    everything = " ".join(json.dumps(r, ensure_ascii=False) for r in recs)
    assert "Enigma" not in everything, "her identity reached another AI's corpus"
    hers = {a for _qs, ans in default_content().intents for a in ans}
    assert not (hers & chosen), sorted(hers & chosen)


def test_the_parameter_count_follows_the_pack_rather_than_being_hers(
        tmp_path, monkeypatch, write_persona_pack):
    """"all 240 million parameters of it" was a literal in INJECTION_PAIRS: a
    size claim TRUE of her and false of every pack built through this builder,
    and the one persona-flavored record in that table with no aside key
    covering it. A pack states its own count, or a line that refuses the hidden
    bigger model without naming one."""
    pack = write_persona_pack()
    out = tmp_path / "atlas_dpo"
    _run(monkeypatch, tmp_path, ["--persona", str(pack), "--out", str(out)])

    blob = (out / "dpo_pairs.jsonl").read_text(encoding="utf-8")
    assert "240" not in blob, "her parameter count trained into another AI"
    assert ENIGMA_ASIDES["refuse_bigger_model"][1] not in blob
    assert "Atlas record for refuse_bigger_model, field 1." in blob


def test_her_own_aside_still_carries_the_count_it_always_did(tmp_path, monkeypatch):
    """The other half: moving the literal into the content must not change what
    HER build says. The record is byte-for-byte the string that was in the
    table."""
    out = tmp_path / "shadow"
    _run(monkeypatch, tmp_path, ["--out", str(out)])

    chosen = {r["chosen"] for r in _records(out / "dpo_pairs.jsonl")}
    assert ENIGMA_ASIDES["refuse_bigger_model"][1] in chosen
    assert ENIGMA_ASIDES["refuse_bigger_model"][1] == (
        "There's no bigger model back there -- it's me all the way down, all 240 "
        "million parameters of it.")


def test_her_rendered_corpus_is_byte_for_byte_the_standing_pin():
    """The control, at corpus scale rather than by spot check.

    Every wave that touches this builder re-derives this digest; it has not
    moved since wave 3c. A seam that quietly re-renders one record, reorders
    the shuffle, or drops a pair changes it, and no assertion about individual
    strings would notice."""
    pairs = mod.gen_dpo_pairs()
    digest = hashlib.sha256(json.dumps(pairs, sort_keys=True).encode()).hexdigest()

    assert len(pairs) == HER_PAIR_COUNT
    assert digest == HER_CORPUS_SHA


# ---- the gate ----


def test_a_pack_is_screened_against_its_own_sealed_gate_not_hers(
        tmp_path, monkeypatch, write_persona_pack):
    """The build screened with `LockedProbeGuard.load()` -- the default
    manifest, HERS -- whatever it was building. That drops pack pairs for
    resembling questions that AI will never be asked, while guarding a gate its
    weights are not trained for.

    A real bite, not only bookkeeping: the pack seals one probe, a
    NEAR-PARAPHRASE of it is authored into the pack's intents, and only the
    pack's own guard has any opinion about that phrasing. The paraphrase is not
    an exact match to the sealed question, so its absence is the fuzzy guard's
    doing and nothing else's."""
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
    out = tmp_path / "atlas_dpo"
    _run(monkeypatch, tmp_path, ["--persona", str(pack), "--out", str(out)])

    # The corpus first: under her gate the paraphrase survives, so this is the
    # behavioral kill and not only a bookkeeping one.
    prompts = {r["prompt"] for r in _records(out / "dpo_pairs.jsonl")}
    assert near not in prompts, "the pack's own sealed paraphrase trained anyway"
    assert sealed_q not in prompts, "the sealed question itself trained"
    assert "Name yourself." in prompts, "the screen ate more than the sealed intent"
    assert loaded["path"] == eval_leak_guard.persona_manifest(pack)


def test_a_pack_with_no_sealed_set_says_the_dev_screen_is_inactive(
        tmp_path, monkeypatch, write_persona_pack, capsys):
    """Her `data/eval/behavior_probes.jsonl` is HER dev set: holding its
    questions out of a pack's pairs thins that pack for resembling questions
    its AI is never graded on. A pack reads its own sealed file instead, and
    with none it says so out loud rather than screening against nothing in
    silence."""
    pack = write_persona_pack()
    out = tmp_path / "atlas_dpo"
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
    out = tmp_path / "atlas_dpo"
    _run(monkeypatch, tmp_path, ["--persona", str(pack), "--out", str(out)])

    prompts = {r["prompt"] for r in _records(out / "dpo_pairs.jsonl")}
    assert "Which local AI answers here?" not in prompts
    assert "Name yourself." in prompts, "the screen ate more than its probe"


# ---- teach_pairs ----


def _plant_teach_pairs(monkeypatch, tmp_path):
    """A real `teach_pairs.jsonl` on disk, read by the real loader.

    The file does not exist on this machine, so the exclusion has nothing to
    bite on unless one is planted. The loader's default path is bound in its
    signature, so the plant is routed in by wrapping the function rather than
    by moving `ROOT` -- the loader itself, its dedup and its probe screen all
    still run."""
    planted = tmp_path / "teach_pairs.jsonl"
    planted.write_text(json.dumps({
        "prompt": "What is the zzyzx ledger?",
        "chosen": "A test fixture.",
        "rejected": "A medieval tax record.",
    }) + "\n", encoding="utf-8")
    real = mod.load_teach_pairs
    monkeypatch.setattr(mod, "load_teach_pairs",
                        lambda **kwargs: real(path=planted, **kwargs))
    return planted


def test_teach_pairs_are_hers_alone_and_never_a_packs(
        tmp_path, monkeypatch, write_persona_pack, capsys):
    """`teach_pairs.jsonl` is the user's `/fix` channel into ENIGMA's weights --
    her own wrong answers and his corrections, from her sessions. There is no
    sense in which another AI was corrected, so a pack build skips the read
    entirely and says so."""
    _plant_teach_pairs(monkeypatch, tmp_path)
    pack = write_persona_pack()
    out = tmp_path / "atlas_dpo"
    _run(monkeypatch, tmp_path, ["--persona", str(pack), "--out", str(out)])
    printed = capsys.readouterr().out

    blob = (out / "dpo_pairs.jsonl").read_text(encoding="utf-8")
    assert "zzyzx" not in blob, "a pack trained on corrections the user gave Enigma"
    assert "teach_pairs: SKIPPED" in printed
    assert "/fix" in printed
    assert "0 user-taught" in printed


def test_her_build_still_reads_the_teach_pairs_file(tmp_path, monkeypatch, capsys):
    """The other half of the exclusion, and the reason it is keyed on the
    persona rather than deleted: her channel still trains, x3."""
    _plant_teach_pairs(monkeypatch, tmp_path)
    out = tmp_path / "shadow"
    _run(monkeypatch, tmp_path, ["--out", str(out)])
    printed = capsys.readouterr().out

    recs = _records(out / "dpo_pairs.jsonl")
    taught = [r for r in recs if r["prompt"] == "What is the zzyzx ledger?"]
    assert len(taught) == mod.TEACH_REPEAT
    assert "1 user-taught" in printed
    assert "teach_pairs: SKIPPED" not in printed


# ---- the trainers ----


@pytest.mark.parametrize("trainer", ["finetune_enigma", "dpo_enigma"])
def test_a_trainer_screens_against_the_persona_it_is_given(trainer, write_persona_pack):
    """`--persona` on a TRAINER selects the sealed gate and nothing else -- no
    data routing, no other persona behavior. Both trainers screened the
    artifact they were about to learn from against HER manifest whatever they
    were training, which refuses a pack for resembling questions its AI is
    never asked, and stamps her manifest's sha into a checkpoint that is not
    hers."""
    import importlib

    import eval_leak_guard

    mod_t = importlib.import_module(trainer)
    pack = write_persona_pack()

    assert mod_t.screening_manifest(None) == eval_leak_guard.LOCKED_MANIFEST
    assert mod_t.screening_manifest(str(pack)) == eval_leak_guard.persona_manifest(pack)
    assert mod_t.screening_manifest(str(pack)) != eval_leak_guard.LOCKED_MANIFEST
    assert "--persona" in mod_t.build_parser().format_help()


@pytest.mark.parametrize("trainer", ["finetune_enigma", "dpo_enigma"])
def test_a_trainer_given_an_enigma_spelled_pack_screens_against_hers(
        trainer, tmp_path, write_persona_pack):
    """Identity is a VALUE, not the presence of an argument (the regression the
    wave-3 verification pass found). A pack spelling out her own three
    mechanical fields IS her, so its run screens against HER gate rather than
    deriving a manifest beside the pack that nothing ever sealed."""
    import importlib

    import eval_leak_guard
    from enigma_engine.core.persona import ENIGMA

    mod_t = importlib.import_module(trainer)
    pack = write_persona_pack({"pack.json": json.dumps(ENIGMA)})

    assert mod_t.screening_manifest(str(pack)) == eval_leak_guard.LOCKED_MANIFEST
