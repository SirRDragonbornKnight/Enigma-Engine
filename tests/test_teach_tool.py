"""Teach-by-chat tool (teach_enigma.py): saved teachings must be readable by
gen_teaching_examples verbatim, preference pairs must match the dpo_pairs
schema, and /fix must rewrite history so the chat continues from the
corrected answer."""

from __future__ import annotations

import json

from make_sft_data import gen_teaching_examples
from teach_enigma import fix_last, last_exchange, save_pair, save_teaching


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
