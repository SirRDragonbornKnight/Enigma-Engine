"""Facts continued-pretrain corpus (make_facts_pretrain_data): the pure-replay
tail is the window pretrain uses as [val], so it must contain ZERO fact tokens
or [val] stops measuring general-domain retention (final audit 2026-07-16 M3).
A fact doc used to be inserted just under mixed_end and spill past the fence.

The locked-probe screen is the other honesty fence: this stream INSTALLS
knowledge, so an unscreened fact line can teach a locked probe's answer
directly into the weights and turn the gate into a memorization test."""

from __future__ import annotations

import json

import numpy as np
import pytest

import make_facts_pretrain_data
from eval_leak_guard import LockedProbeGuard, seal
from make_facts_pretrain_data import interleave, screen_locked_probes

REPLAY_TOK = 500  # distinct replay token; fact tokens are small ( < 10 )


def _replay(n: int = 200_000) -> np.ndarray:
    return np.full(n, REPLAY_TOK, dtype=np.uint32)


def test_val_tail_is_pure_replay():
    fact_docs = [[1, 2, 3, 4, 5]] * 200
    target, val_reserve = 100_000, 20_000
    out = interleave(fact_docs, _replay(), target, fact_frac=0.05,
                     val_reserve=val_reserve, chunk=2048, seed=1)
    tail = out[target - val_reserve:]
    assert (tail == REPLAY_TOK).all(), "fact tokens leaked into the [val] tail"
    # sanity: facts DID land in the mixed region (the fence isn't just dropping them all)
    assert (out[: target - val_reserve] != REPLAY_TOK).any()


def test_oversized_fact_doc_never_crosses_the_fence():
    # A doc longer than the cadence gap must not straddle mixed_end.
    fact_docs = [list(range(1, 9))] * 50 + [list(range(1, 300))]  # one long doc
    target, val_reserve = 60_000, 10_000
    out = interleave(fact_docs, _replay(), target, fact_frac=0.1,
                     val_reserve=val_reserve, chunk=1024, seed=3)
    tail = out[target - val_reserve:]
    assert (tail == REPLAY_TOK).all()


# --- locked-probe screen -------------------------------------------------

# A locked KNOWLEDGE probe and the fact lines knowledge_corpus emits for the
# same fact (declarative / QA / cloze). These are the shapes that actually
# install the answer, so they are what the screen has to catch.
LOCKED = ["Which planet in our solar system is the largest?"]
LEAKY_QA = "Q: Which planet in our solar system is the largest? A: Jupiter."
LEAKY_DECLARATIVE = "The largest planet in our solar system is Jupiter."
UNRELATED = "The Nile is the longest river in Africa."


def test_eos_from_sidecar_refuses_laundered_types():
    """int() as the guard laundered hand-edit damage: a JSON float 2.7
    truncated to 2 and `true` became 1, both silently -- the exact shapes
    the refusal message claimed to catch (convergence audit, 2026-07-26)."""
    import pytest

    from make_facts_pretrain_data import eos_from_sidecar

    assert eos_from_sidecar({"eos_token_id": 2}, 4718) == 2
    assert eos_from_sidecar({}, 4718) == 2  # missing key keeps the legacy default
    for bad in (2.7, True, False, "2", None, [2]):
        with pytest.raises(SystemExit, match="plain integer"):
            eos_from_sidecar({"eos_token_id": bad}, 4718)
    with pytest.raises(SystemExit, match="outside vocab"):
        eos_from_sidecar({"eos_token_id": 4718}, 4718)


def test_sidecar_vocab_and_dtype_refuse_hand_edits(monkeypatch, tmp_path):
    """eos got the plain-in-range-or-refuse treatment while vocab_size and
    dtype -- read from the SAME hand-editable sidecar -- kept bare trust: a
    "dtype": "float64" edit re-interpreted the whole replay corpus silently,
    and a wrong vocab_size died late as a KeyError or a mis-picked tokenizer.
    All three sidecar keys refuse the same way now."""
    import json as _json
    import sys as _sys

    import pytest

    import make_facts_pretrain_data as mf

    bin_path = tmp_path / "replay.bin"
    bin_path.write_bytes(b"\x00" * 64)

    def run_with(meta):
        (tmp_path / "replay.json").write_text(_json.dumps(meta), encoding="utf-8")
        monkeypatch.setattr(_sys, "argv",
                            ["make_facts_pretrain_data.py", "--source-bin", str(bin_path)])
        mf.main()

    for bad_vocab in (None, "16366", 16366.0, True, 0, -5):
        with pytest.raises(SystemExit, match="vocab_size is not a positive integer"):
            run_with({"vocab_size": bad_vocab, "dtype": "uint16", "eos_token_id": 2})

    for bad_dtype in ("float64", "int64", "uint8", 16, None):
        with pytest.raises(SystemExit, match="not a token dtype"):
            run_with({"vocab_size": 16366, "dtype": bad_dtype, "eos_token_id": 2})


def test_screen_is_a_noop_before_any_seal():
    lines = [LEAKY_QA, LEAKY_DECLARATIVE, UNRELATED]
    assert screen_locked_probes(lines, LockedProbeGuard(None)) == lines


def test_dropped_and_near_miss_lines_go_to_files_not_the_console(tmp_path, capsys):
    """Records to a file, counts to the console: a dropped line scores >=
    threshold against a sealed probe, so printing it puts the sealed set's
    content words into any pasted build log."""
    # Measured 0.5 against the relativity probe -- inside the [0.5, 0.6) review
    # band, so it is KEPT and flagged rather than dropped. Picked by measuring,
    # not by eyeballing the wording: a hand-guessed paraphrase scored 0.33 and
    # silently tested nothing.
    near = "Einstein developed relativity."
    guard = LockedProbeGuard(seal(LOCKED + ["Who developed the theory of relativity?"]))

    kept = screen_locked_probes([LEAKY_QA, near, UNRELATED], guard, review_dir=tmp_path)

    out = capsys.readouterr().out
    assert LEAKY_QA not in out, "a dropped line's text reached the console"
    dropped_rows = [json.loads(x) for x in (tmp_path / "locked_dropped_facts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["text"] for r in dropped_rows] == [LEAKY_QA]
    near_rows = [json.loads(x) for x in (tmp_path / "locked_near_miss_facts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["text"] for r in near_rows] == [near]
    assert near in kept, "review-band lines are flagged, not dropped"


def test_stale_review_files_are_removed_when_a_run_is_clean(tmp_path):
    stale = tmp_path / "locked_dropped_facts.jsonl"
    stale.write_text('{"text": "from an earlier run"}\n', encoding="utf-8")

    screen_locked_probes([UNRELATED], LockedProbeGuard(seal(LOCKED)), review_dir=tmp_path)

    assert not stale.exists(), "a clean run left a stale drop list to mislead the next reader"


def test_sealed_probe_drops_the_leaky_fact_line_and_keeps_the_rest():
    # The adversarial case: a fact line that would teach the locked probe's
    # answer must be GONE, while an unrelated fact survives. Asserting only
    # "unrelated survives" would pass with the screen deleted entirely.
    kept = screen_locked_probes(
        [LEAKY_QA, LEAKY_DECLARATIVE, UNRELATED], LockedProbeGuard(seal(LOCKED))
    )
    assert LEAKY_QA not in kept
    # The declarative form carries no question shape at all, and it is the one
    # that most directly installs the answer -- it has to be caught too.
    assert LEAKY_DECLARATIVE not in kept
    assert UNRELATED in kept


def test_screen_does_not_empty_a_corpus_of_unrelated_facts():
    # Guard against the opposite failure: a screen that drops everything would
    # silently gut the knowledge stream.
    unrelated = [
        UNRELATED,
        "Water boils at 100 degrees Celsius at sea level.",
        "The human heart has four chambers.",
    ]
    assert screen_locked_probes(unrelated, LockedProbeGuard(seal(LOCKED))) == unrelated


def test_the_build_screens_before_it_tokenizes(monkeypatch):
    """The screen being correct is worthless if main() does not call it, or
    calls it after the fact lines are already tokenized into the corpus."""
    order = []

    def spy_screen(lines, guard, review_dir=None):
        order.append("screen")
        return list(lines)

    def spy_tokenize(lines, tokenizer, vocab_size, eos_id=2):
        order.append("tokenize")
        raise SystemExit("stop here -- ordering is what this test pins")

    monkeypatch.setattr(make_facts_pretrain_data, "screen_locked_probes", spy_screen)
    monkeypatch.setattr(make_facts_pretrain_data, "tokenize_fact_docs", spy_tokenize)
    monkeypatch.setattr("sys.argv", ["make_facts_pretrain_data.py"])

    with pytest.raises(SystemExit):
        make_facts_pretrain_data.main()

    assert order == ["screen", "tokenize"]


def test_an_empty_knowledge_corpus_is_not_blamed_on_the_screen(monkeypatch):
    """The 'everything was screened out' error used to fire when the corpus was
    simply empty, pointing the operator at a guard that never ran."""
    monkeypatch.setattr(make_facts_pretrain_data, "gen_knowledge_pretrain_text", lambda: [])
    monkeypatch.setattr(make_facts_pretrain_data, "gen_teaching_pretrain_text", lambda: [])
    monkeypatch.setattr("sys.argv", ["make_facts_pretrain_data.py"])

    with pytest.raises(SystemExit) as exc:
        make_facts_pretrain_data.main()

    assert "no fact lines" in str(exc.value)
    assert "screened out" not in str(exc.value)


# ---------------------------------------------------------------------------
# teachings.jsonl routes here as well as into the SFT mix. SFT cannot install
# a fact -- it only surfaces one the weights already hold -- so a teaching
# that reaches the model ONLY through the mix is trained in the one channel
# that provably cannot learn it.
# ---------------------------------------------------------------------------


def _teachings(tmp_path, *records):
    path = tmp_path / "teachings.jsonl"
    path.write_text(
        "# a comment line\n" + "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_teachings_become_fact_lines(tmp_path):
    path = _teachings(
        tmp_path,
        {
            "questions": ["What is my favourite tea?", "Which tea do I drink?", "Do you know my tea?"],
            "answers": ["Oolong -- you drink it every morning.", "That would be oolong."],
        },
    )
    lines = make_facts_pretrain_data.gen_teaching_pretrain_text(path)

    assert lines, "a written teaching produced no fact lines"
    # Both installable forms: the answer as standalone prose, and a Q/A line
    # per phrasing (the knowledge stream's own shapes).
    assert "Oolong -- you drink it every morning." in lines
    qa = [ln for ln in lines if ln.startswith("Q: ")]
    assert len({ln.split(" A: ")[0] for ln in qa}) == 3, "not every phrasing became a Q/A line"


def test_a_one_word_answer_only_rides_the_qa_form(tmp_path):
    """A bare "Rome." teaches nothing standing alone in a pretrain stream --
    the same bar the knowledge corpus applies."""
    path = _teachings(tmp_path, {"q": "Who is my vet?", "a": "Dr Halloway."})
    lines = make_facts_pretrain_data.gen_teaching_pretrain_text(path)

    assert "Dr Halloway." not in lines
    assert "Q: Who is my vet? A: Dr Halloway." in lines


def test_the_untouched_template_yields_nothing(tmp_path):
    """teachings.jsonl ships as comments only and is the user's authoring
    channel -- the route must be a no-op until they write in it, not a crash."""
    path = tmp_path / "teachings.jsonl"
    path.write_text("# only comments here\n", encoding="utf-8")
    assert make_facts_pretrain_data.gen_teaching_pretrain_text(path) == []
    assert make_facts_pretrain_data.gen_teaching_pretrain_text(tmp_path / "absent.jsonl") == []


def test_teaching_lines_reach_the_build(monkeypatch, tmp_path):
    """The lines must actually be APPENDED to the stream the builder screens
    and tokenizes -- generating them and dropping them on the floor would
    leave the install channel exactly as empty as before."""
    monkeypatch.setattr(make_facts_pretrain_data, "gen_knowledge_pretrain_text", lambda: ["A base fact line here."])
    monkeypatch.setattr(
        make_facts_pretrain_data, "gen_teaching_pretrain_text",
        lambda: ["Q: What is my favourite tea? A: Oolong."],
    )
    seen = {}

    def spy_screen(lines, guard, review_dir=None):
        seen["lines"] = list(lines)
        raise SystemExit("stop here")

    monkeypatch.setattr(make_facts_pretrain_data, "screen_locked_probes", spy_screen)
    monkeypatch.setattr("sys.argv", ["make_facts_pretrain_data.py"])

    with pytest.raises(SystemExit):
        make_facts_pretrain_data.main()

    assert "Q: What is my favourite tea? A: Oolong." in seen["lines"]
    assert "A base fact line here." in seen["lines"], "the knowledge stream must survive too"
