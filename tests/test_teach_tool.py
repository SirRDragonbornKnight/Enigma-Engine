"""Teach-by-chat tool (teach_enigma.py): saved teachings must be readable by
gen_teaching_examples verbatim, preference pairs must match the dpo_pairs
schema, /fix must rewrite history so the chat continues from the corrected
answer, and retract() must remove exactly the retracted records from disk
(audit 2026-07-15: /undo previously left them baked in at x8 weight)."""

from __future__ import annotations

import json

from make_sft_data import gen_teaching_examples
from teach_enigma import (
    augment_teaching,
    fix_last,
    last_exchange,
    paraphrase_question,
    retract,
    save_pair,
    save_teaching,
    statement_twin,
)


def test_saved_teaching_round_trips_through_the_generator(tmp_path):
    path = tmp_path / "teachings.jsonl"
    save_teaching("What is my dog's name?", "Rex -- you told me that.", path=path)
    recs = gen_teaching_examples(path=path)
    assert len(recs) == 1
    msgs = recs[0]["messages"]
    assert msgs[0] == {"role": "user", "content": "What is my dog's name?"}
    assert msgs[1] == {"role": "assistant", "content": "Rex -- you told me that."}


def test_saved_pair_matches_dpo_schema(tmp_path):
    path = tmp_path / "pairs.jsonl"
    save_pair("Q?", "right answer", "wrong answer", path=path)
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec == {"prompt": "Q?", "chosen": "right answer", "rejected": "wrong answer"}


def test_fix_last_rewrites_history():
    history = [
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "garbled nonsense"},
    ]
    fixed = fix_last(history, "A clear answer.")
    assert fixed == ("Q?", "garbled nonsense")
    assert history[-1] == {"role": "assistant", "content": "A clear answer."}


def test_fix_last_refuses_without_an_exchange():
    assert fix_last([], "x") is None
    assert fix_last([{"role": "user", "content": "Q?"}], "x") is None


def test_last_exchange_reads_the_tail():
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "A."},
    ]
    assert last_exchange(history) == ("Q?", "A.")
    assert last_exchange([]) is None


def test_saves_return_pre_append_sizes_and_retract_restores_bytes(tmp_path):
    path = tmp_path / "teachings.jsonl"
    size1 = save_teaching("Q1?", "A1.", path=path)
    snapshot = path.read_bytes()
    size2 = save_teaching("Q2?", "typo'd answer", path=path)
    assert (size1, size2) == (0, len(snapshot))
    assert retract([(path, size2)]) == 1
    assert path.read_bytes() == snapshot  # only the retracted record is gone


def test_retract_truncates_to_earliest_size_per_file(tmp_path):
    # A re-/fix retracts BOTH the first fix's teaching and its preference
    # pair, even with several writes to the same file in the ledger.
    teach = tmp_path / "teachings.jsonl"
    pairs = tmp_path / "pairs.jsonl"
    writes = [
        (teach, save_teaching("Q?", "first correction", path=teach)),
        (pairs, save_pair("Q?", "first correction", "her original", path=pairs)),
        (teach, save_teaching("Q?", "second correction", path=teach)),
    ]
    assert retract(writes) == 3
    assert teach.read_bytes() == b""
    assert pairs.read_bytes() == b""


def test_retract_is_safe_on_empty_ledger_and_missing_files(tmp_path):
    assert retract([]) == 0
    retract([(tmp_path / "never_written.jsonl", 0)])  # must not raise


def test_augment_expands_a_fix_into_diverse_phrasings():
    # A single /fix must produce >=3 DEDUPED question phrasings so the bake
    # teaches the fact, not one exact string (teach-loop auto-augment).
    qs, ans = augment_teaching("What is my dog's name?", "Rex.")
    assert qs[0] == "What is my dog's name?"
    assert len(qs) >= 3
    assert len({q.lower() for q in qs}) == len(qs)
    # 'what is X' gets a declarative statement twin on the answer side,
    # spoken in the ASSISTANT's voice (the user's 'my' flips to 'your').
    assert "Your dog's name is Rex." in ans


def test_statement_twin_only_for_simple_wh_is_questions():
    assert statement_twin("What is the capital of France?", "Paris") == "The capital of France is Paris."
    # behavioral / how / why corrections have no safe generic statement form
    assert statement_twin("Stop repeating yourself.", "Okay.") is None
    assert statement_twin("How do I reset it?", "Hold the button.") is None


def test_statement_twin_speaks_in_the_assistant_voice():
    # audit 2026-07-16: the unflipped twin taught her to call the USER Enigma.
    assert statement_twin("Who are you?", "Enigma.") == "I am Enigma."
    assert statement_twin("What is your name?", "Enigma.") == "My name is Enigma."
    assert statement_twin("What is my dog's name?", "Rex.") == "Your dog's name is Rex."


def test_statement_twin_uninverts_a_pronoun_subject():
    # audit 2026-07-16: 'Where were you born?' twinned to 'You born were ...'.
    # 2026-07-17: sentence-initial function words in the answer lowercase when
    # embedded mid-statement (the old assert BAKED 'born In the forge' into
    # training data); proper nouns keep their capital (next test).
    assert statement_twin("Where were you born?", "In the forge.") == "I was born in the forge."
    assert statement_twin("What is it called?", "The forge.") == "It is called the forge."


def test_statement_twin_keeps_proper_noun_answer_casing():
    assert statement_twin("Where were you born?", "Paris.") == "I was born Paris."
    assert statement_twin("What is your name?", "Sir Knight's Enigma.") == (
        "My name is Sir Knight's Enigma."
    )


def test_paraphrases_stay_grammatical_and_deduped():
    ps = paraphrase_question("What is X?")
    assert "Remind me, what is X?" in ps  # leading capital lowered after the prefix
    assert len(ps) == len(set(ps))


def test_save_teaching_accepts_phrasing_lists_and_round_trips(tmp_path):
    path = tmp_path / "teachings.jsonl"
    save_teaching(["Q1?", "Q2?"], ["A1.", "A2."], path=path)
    recs = gen_teaching_examples(path=path)
    # crossing 2 questions x rotating answers yields several records, and NO
    # per-character explosion of the strings.
    assert len(recs) >= 3
    assert all(len(r["messages"][0]["content"]) > 1 for r in recs)


def test_review_ctrl_c_cancels_instead_of_saving(monkeypatch):
    # audit 2026-07-16: Ctrl+C at the review prompt fell through to
    # accept-and-write -- the abort gesture committed the records.
    import sys as _sys

    from teach_enigma import review_augmentation

    monkeypatch.setattr(_sys.stdin, "isatty", lambda: True)

    def _interrupt(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _interrupt)
    assert review_augmentation(["Q?"], ["A."]) is None


def test_review_eof_still_auto_accepts(monkeypatch):
    # Piped/scripted teaching (stdin runs dry) must keep auto-accepting.
    import sys as _sys

    from teach_enigma import review_augmentation

    monkeypatch.setattr(_sys.stdin, "isatty", lambda: True)

    def _eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    assert review_augmentation(["Q?"], ["A."]) == (["Q?"], ["A."])


def test_retract_does_not_pad_a_hand_shrunk_file(tmp_path):
    path = tmp_path / "teachings.jsonl"
    save_teaching("Q1?", "A1.", path=path)
    stale_offset = save_teaching("Q2?", "A2.", path=path)  # > any tiny hand-edit
    path.write_bytes(b"x\n")  # external edit shrinks the file below the offset
    retract([(path, stale_offset)])
    assert path.read_bytes() == b"x\n"  # left intact, NOT NUL-padded to stale_offset
