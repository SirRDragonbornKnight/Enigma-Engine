"""The data shapes added for the v2 SFT regen -- TRAINING side.

Each generator here fills a hole that was MEASURED empty in the shipped mix
(114,244 records): no record carried a second user turn, none contained a
<think> tag, none corrected a remembered fact, and no system block offered a
built-in the request did not trip a regex for. A generator that silently
stopped emitting its shape would leave the same hole and every one of these
records would still pass a schema check, so the tests assert the PROPERTY
that was missing, not just well-formedness.

Scope honesty: this file pins the TRAINING side. That the SERVER renders the
same always-offered block byte-for-byte is locked in
tests/test_serve_enigma.py against serve's own _with_context; the drift lock
here compares against the shared spec table both sides read.
"""

import re

import pytest

from enigma_engine.core.chat_format import (
    BUILTIN_NAMES,
    BUILTIN_TOOLS,
    THINK,
    THINK_END,
    render_tools_system,
)
from enigma_engine.core.tokenizer import get_tokenizer, vocab_file_for_size
from make_sft_data import (
    PREAMBLE,
    _builtin_system,
    gen_builtin_block_examples,
    gen_chat_multiturn_examples,
    gen_memory_correction_examples,
    gen_reasoning_examples,
    vocab_is_digit_uniform,
)


def _tool_calls(rec):
    return [c for m in rec["messages"] for c in (m.get("tool_calls") or [])]


def _user_turns(rec):
    return [m for m in rec["messages"] if m["role"] == "user"]


# --------------------------------------------------------------------------
# The always-offered built-in block
# --------------------------------------------------------------------------


def test_builtin_block_offers_every_builtin_and_nothing_else():
    """The block is the five tools the SERVER can actually execute.

    A name trained into the offer list without a runtime is a promise she
    cannot keep; a runtime missing from the list gets no gradient at all --
    which is how `forget` shipped with an executor and zero training records.
    """
    sys_msg = _builtin_system()
    offered = {t["function"]["name"] for t in BUILTIN_TOOLS}
    assert offered == set(BUILTIN_NAMES)
    for name in offered:
        assert f'"name": "{name}"' in sys_msg, f"{name} missing from the offered block"


def test_the_trained_preamble_is_the_one_serve_prepends():
    """The preamble must come from the SERVE side, not from a local copy.

    Asserting the block equals PREAMBLE + tools would pass with any string in
    PREAMBLE, since both halves would move together -- a tautology. The honest
    comparison is against persona.tools_preamble, which is what serve actually
    puts in front of a tools block.
    """
    from enigma_engine.core.persona import Persona

    assert PREAMBLE == Persona.load().tools_preamble


def test_builtin_block_is_the_shape_serve_renders():
    """Byte-identical to preamble + the server's own tools renderer.

    Both sides read one spec table, so this fails if the training block stops
    being built through render_tools_system -- the drift that would put the
    model in front of a system message it never saw. Paired with the preamble
    test above, which anchors the other half to serve.
    """
    assert _builtin_system() == PREAMBLE + "\n" + render_tools_system(BUILTIN_TOOLS)
    assert "Available tools:" in _builtin_system()


def test_builtin_block_carries_memory_in_serves_join_order():
    block = "Things you remember:\n- User's dog is named Rex."
    out = _builtin_system(block)
    assert out.startswith(block + "\n\n"), "memory must lead, then a blank line"
    assert out.endswith(_builtin_system())


def test_builtin_restraint_records_call_nothing():
    """The half that makes an always-offered block safe.

    Restraint used to come from the gate never OFFERING the tool, so the
    weights never learned to decline one. Every restraint record must show
    all five on offer and no call made -- including the negated ask, which is
    the exact input that used to arm the painter.
    """
    recs = gen_builtin_block_examples()
    restraint = [r for r in recs if r["category"] == "builtin_restraint"]
    assert restraint, "no restraint records"
    for r in restraint:
        assert not _tool_calls(r), f"restraint record called a tool: {r['messages'][1]['content']}"
        assert r["messages"][0]["content"] == _builtin_system()
        assert r["messages"][-1]["role"] == "assistant"
        assert r["messages"][-1]["content"].strip()
    asks = " ".join(r["messages"][1]["content"].lower() for r in restraint)
    # The measured false-fire shapes, each named so a trim cannot quietly
    # drop the one input that motivated the record.
    assert "do not draw anything" in asks, "the negated-draw false-fire is untrained"
    assert "you can say that again" in asks, "the speak-idiom false-fire is untrained"


def test_builtin_call_records_name_a_real_builtin_and_finish_the_trace():
    recs = gen_builtin_block_examples()
    calls = [r for r in recs if r["category"] == "builtin_call"]
    assert calls, "no built-in call records"
    used = set()
    for r in calls:
        made = _tool_calls(r)
        assert made, "a call record with no call"
        for c in made:
            assert c["name"] in BUILTIN_NAMES, f"{c['name']} has no runtime"
            used.add(c["name"])
        roles = [m["role"] for m in r["messages"]]
        assert "tool" in roles, "no tool result -- the trace stops before the answer"
        assert roles[-1] == "assistant" and r["messages"][-1]["content"].strip()
    assert used == set(BUILTIN_NAMES), f"built-ins never called in training: {set(BUILTIN_NAMES) - used}"


def test_builtin_block_trains_the_surfaces_the_regex_gates_missed():
    """Word-number math and an imperative draw were measured MISSES.

    A missed ask meant no offer, so no gradient ever taught it. The surfaces
    here are near-neighbors of the probe phrasings, never the probes
    themselves -- "Draw me a dragon." IS an eval probe, so authoring it into
    the corpus just hands it to the screen and the coverage silently
    vanishes (caught 2026-08-07: 4 of 23 records were being held out).
    """
    asks = " ".join(
        r["messages"][1]["content"].lower()
        for r in gen_builtin_block_examples()
        if r["category"] == "builtin_call"
    )
    assert "seven times eight" in asks, "word-number math (a measured gate miss) is untrained"
    assert "draw me a" in asks, "the imperative draw (a measured gate miss) is untrained"


def test_every_authored_record_survives_the_probe_screen():
    """The four regen-shape corpora are hand-authored, so they must be
    authored to CLEAR the screen -- a record that collides with a probe is
    not trained at all, and the hole it was written to fill stays open while
    every generator-level test stays green. Uses main()'s own screen, so a
    future reseal that newly collides fails HERE instead of silently
    thinning the corpus."""
    from eval_leak_guard import LockedProbeGuard
    from make_sft_data import _eval_probe_questions, probe_screen

    _, held_out = probe_screen(_eval_probe_questions(), LockedProbeGuard.load())
    for gen in (gen_builtin_block_examples, gen_chat_multiturn_examples,
                gen_reasoning_examples, gen_memory_correction_examples):
        gone = [r["messages"][1]["content"] for r in gen() if held_out(r)]
        assert not gone, f"{gen.__name__} authored probe collisions: {gone}"


# --------------------------------------------------------------------------
# Conversational multi-turn
# --------------------------------------------------------------------------


def test_conversations_have_more_than_one_user_turn():
    """The hole this fills: every structured record in the shipped mix had
    exactly ONE user turn, so nothing ever taught her to read a second
    against the first."""
    recs = gen_chat_multiturn_examples()
    assert recs, "generator returned nothing"
    deep = [r for r in recs if len(_user_turns(r)) >= 2]
    assert deep, "no record carries a second user turn -- the gap is still open"
    assert any(len(_user_turns(r)) >= 3 for r in recs), "no three-turn conversation"


def test_conversations_alternate_and_end_on_her():
    for r in gen_chat_multiturn_examples():
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["user", "assistant"] * (len(roles) // 2), f"turns do not alternate: {roles}"
        assert all(m["content"].strip() for m in r["messages"]), "an empty turn"


def test_conversations_carry_no_tools_or_memory():
    """Pure conversation. The tool chains already cover multi-CALL inside one
    request; mixing the two here would confound which shape is being taught."""
    for r in gen_chat_multiturn_examples():
        assert not _tool_calls(r)
        assert r["messages"][0]["role"] == "user", "no system block belongs on these"


# --------------------------------------------------------------------------
# Reasoning traces
# --------------------------------------------------------------------------


def test_reasoning_records_carry_a_well_formed_think_span():
    recs = gen_reasoning_examples()
    assert recs, "generator returned nothing"
    for r in recs:
        text = r["messages"][-1]["content"]
        assert text.count("<think>") == 1 and text.count("</think>") == 1, text[:80]
        assert text.index("<think>") < text.index("</think>")
        answer = text.split("</think>", 1)[1].strip()
        assert answer, "the trace has no answer after it -- she would think and never reply"


@pytest.mark.parametrize("vocab_size", [4718, 16366], ids=["v1", "v2"])
def test_think_tags_are_the_tokenizers_own_ids(vocab_size):
    """The tags cost zero new embedding rows only if they are the vocab's own
    <think>/</think>. Both live tables assign 10/11; a table that assigned
    anything else would train the trace on rows the model does not read as
    thinking, which is why attach_chat_tokens refuses the mismatch."""
    tok = get_tokenizer("bpe", vocab_path=vocab_file_for_size(vocab_size))
    assert tok.token_to_id.get("<think>") == THINK
    assert tok.token_to_id.get("</think>") == THINK_END


def test_reasoning_traces_round_trip_through_the_trainers_renderer():
    """render_training must preserve the tags verbatim -- if the renderer
    dropped or re-encoded them, the corpus would carry reasoning the model is
    never trained to emit."""
    from enigma_engine.core.chat_format import attach_chat_tokens, render_training

    tok = attach_chat_tokens(get_tokenizer("bpe", vocab_path=vocab_file_for_size(16366)))
    rec = gen_reasoning_examples()[0]
    ids, _ = render_training(tok, rec["messages"])
    assert THINK in ids and THINK_END in ids, "the think span did not survive rendering"
    assert ids.index(THINK) < ids.index(THINK_END)


def test_reasoning_does_the_work_rather_than_restating_the_question():
    """A trace that echoes the prompt teaches padding, not reasoning."""
    for r in gen_reasoning_examples():
        question = r["messages"][0]["content"]
        trace = r["messages"][-1]["content"].split("</think>")[0].removeprefix("<think>")
        assert trace.strip() != question.strip()
        assert len(trace.split()) >= 8, f"trace too thin to be work: {trace!r}"


# --------------------------------------------------------------------------
# Memory corrections
# --------------------------------------------------------------------------


def test_corrections_cover_both_halves_of_the_supersede_ruling():
    """Namings and single-valued verbs REPLACE; plain copula values COEXIST.

    Training only the replace half would teach her to erase the first allergy
    when a second arrives -- the data-loss direction the ruling exists to
    prevent.
    """
    recs = gen_memory_correction_examples()
    asks = " ".join(m["content"].lower() for r in recs for m in r["messages"] if m["role"] == "user")
    assert "i moved to austin" in asks, "the replace shape (a move) is untrained"
    assert "also allergic to shellfish" in asks, "the coexist shape (a second allergy) is untrained"


def test_corrections_write_the_new_fact_through_remember():
    recs = gen_memory_correction_examples()
    writing = [r for r in recs if _tool_calls(r)]
    assert writing, "no correction ever writes the corrected fact"
    for r in writing:
        for c in _tool_calls(r):
            assert c["name"] == "remember", "the store decides replace-vs-coexist on write"
            assert c["arguments"]["text"].strip()
        assert r["messages"][0]["content"].startswith("Things you remember:"), \
            "a correction needs the stored fact in front of her"


def test_answers_after_a_correction_do_not_resurface_the_old_value():
    """The failure shape: answering from the fact she was told minutes ago.

    Each follow-up carries only the NEW value in its block, so the old string
    must appear nowhere in the record.
    """
    # Word boundaries, not substrings: the corrected value "Samir" CONTAINS
    # the superseded "Sam", and a substring check flags the right answer.
    superseded = {"denver", "sam", "rex", "teacher", "pickup"}
    for r in gen_memory_correction_examples():
        if _tool_calls(r):
            continue
        blob = " ".join(m["content"].lower() for m in r["messages"])
        leaked = {w for w in superseded if re.search(rf"\b{w}\b", blob)}
        assert not leaked, f"a superseded value resurfaced: {leaked}"


# --------------------------------------------------------------------------
# Math rides the vocab, not a flag
# --------------------------------------------------------------------------


def test_math_is_gated_on_the_vocab_carving_digits_uniformly():
    """v1 merges some digit pairs and not others ('15' is one token, '56' is
    two), and arithmetic cannot be learned through that -- the v4 model
    emitted confidently-wrong numbers. The check MEASURES the table so a bake
    against the wrong vocab drops math instead of repeating that result."""
    assert vocab_is_digit_uniform(None) is False, "the v1 table is not digit-uniform"
    assert vocab_is_digit_uniform(vocab_file_for_size(16366)) is True, "the v2 table is"


# --------------------------------------------------------------------------
# Determinism -- a rebuild must reproduce the artifact
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gen",
    [
        gen_builtin_block_examples,
        gen_chat_multiturn_examples,
        gen_reasoning_examples,
        gen_memory_correction_examples,
    ],
    ids=lambda f: f.__name__,
)
def test_generators_are_deterministic(gen):
    assert gen() == gen()
