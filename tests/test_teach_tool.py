"""Teach-by-chat tool (teach_enigma.py): saved teachings must be readable by
gen_teaching_examples verbatim, preference pairs must match the dpo_pairs
schema, /fix must rewrite history so the chat continues from the corrected
answer, and retract() must remove exactly the retracted records from disk
(audit 2026-07-15: /undo previously left them baked in at x8 weight)."""

from __future__ import annotations

import json

from make_sft_data import gen_teaching_examples
from teach_enigma import fix_last, last_exchange, retract, save_pair, save_teaching


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
