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
    gen_episode_examples,
    gen_search_examples,
    gen_structured_examples,
    gen_unknown_examples,
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
    from make_sft_data import gen_image_read_examples

    for gen in (gen_builtin_block_examples, gen_chat_multiturn_examples,
                gen_reasoning_examples, gen_memory_correction_examples,
                gen_search_examples, gen_unknown_examples,
                gen_structured_examples, gen_episode_examples,
                gen_image_read_examples):
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
    superseded = {"denver", "sam", "rex", "teacher", "pickup",
                  "fridays", "flat white", "lisbon"}
    for r in gen_memory_correction_examples():
        if _tool_calls(r):
            continue
        blob = " ".join(m["content"].lower() for m in r["messages"])
        leaked = {w for w in superseded if re.search(rf"\b{w}\b", blob)}
        assert not leaked, f"a superseded value resurfaced: {leaked}"


# --------------------------------------------------------------------------
# Search traces -- the sixth organ's trained side
# --------------------------------------------------------------------------


def test_search_traces_carry_the_full_loop():
    """Span turn, tool results, answer -- the trace serve's _apply_search
    renders. The 31 seed-era orphans taught emit-and-stop toward a runtime
    that never existed; a trace that stops early would repeat that."""
    recs = [r for r in gen_search_examples() if r["category"] in ("search_call", "search_miss")]
    assert recs, "no search traces"
    for r in recs:
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["user", "assistant", "tool", "assistant"], roles
        span = r["messages"][1]["content"]
        assert span.startswith("<search>") and span.endswith("</search>"), span
        assert r["messages"][-1]["content"].strip(), "the trace never answers"


def test_search_tool_turns_are_the_organs_own_render():
    """One renderer for trained and served bytes: every successful trace's
    tool turn must reproduce through core.search.render_results verbatim."""
    from enigma_engine.core.search import render_results
    from make_sft_data import _SEARCH_TRACES

    by_question = {r["messages"][0]["content"]: r for r in gen_search_examples()}
    for question, query, hits, _answer in _SEARCH_TRACES:
        rec = by_question[question]
        assert rec["messages"][1]["content"] == f"<search>{query}</search>"
        assert rec["messages"][2]["content"] == render_results(query, hits)


def test_search_disabled_trace_matches_serves_literal():
    """The trained error string and the one serve emits must be the same
    bytes, or the disabled case is a system shape she never saw."""
    disabled = [
        r for r in gen_search_examples()
        if r["category"] == "search_miss"
        and "search disabled" in r["messages"][2]["content"]
    ]
    assert disabled, "no disabled-organ trace"
    for r in disabled:
        assert r["messages"][2]["content"] == "error: search disabled (start serve with --search)"


def test_search_restraint_answers_without_the_tag():
    recs = [r for r in gen_search_examples() if r["category"] == "search_restraint"]
    assert recs, "no restraint half -- the tag becomes a reflex"
    for r in recs:
        assert "<search>" not in r["messages"][-1]["content"]
        assert len(r["messages"]) == 2


def test_time_sensitive_search_answers_use_synthetic_entities():
    """A real time-sensitive question with fabricated results would train a
    lie into the weights; the synthetic-entity rows keep .example URLs so a
    real domain cannot drift in unnoticed."""
    from make_sft_data import _SEARCH_TRACES

    synthetic = [row for row in _SEARCH_TRACES if any(".example" in h["url"] for h in row[2])]
    assert len(synthetic) >= 3, "the time-sensitive class lost its synthetic rows"


# --------------------------------------------------------------------------
# The epistemics corpus -- v2's #1 win condition
# --------------------------------------------------------------------------


def test_unknowns_cover_declines_contrasts_and_the_builtin_block():
    recs = gen_unknown_examples()
    declines = [r for r in recs if r["category"] == "unknown_decline"]
    contrasts = [r for r in recs if r["category"] == "unknown_contrast"]
    assert len(declines) >= 25, "the decline corpus thinned out"
    assert contrasts, "no contrast half -- ask-shape collapses to blanket refusal"
    assert any(r["messages"][0]["content"] == _builtin_system() for r in declines), \
        "no decline under the always-offered block"


def test_declines_never_call_a_tool():
    for r in gen_unknown_examples():
        if r["category"] != "unknown_decline":
            continue
        assert not _tool_calls(r), f"a decline fired a tool: {r['messages'][-2]['content']}"


def test_contrast_pairs_share_their_question_with_a_decline():
    """The decline must hinge on the block's CONTENTS, not the question's
    shape -- so the same question must exist in both halves, differing only
    in what the system block holds."""
    recs = gen_unknown_examples()
    decline_qs = {m["content"] for r in recs if r["category"] == "unknown_decline"
                  for m in r["messages"] if m["role"] == "user"}
    contrast_qs = {m["content"] for r in recs if r["category"] == "unknown_contrast"
                   for m in r["messages"] if m["role"] == "user"}
    assert contrast_qs & decline_qs, "no question appears in both halves"


def test_memory_present_declines_do_not_parrot_the_distractor():
    from make_sft_data import _UNKNOWABLE_WITH_MEMORY

    for fact, _q, answer in _UNKNOWABLE_WITH_MEMORY:
        key = fact.split()[-1].rstrip(".").lower()  # "rex", "denver"
        assert key not in answer.lower(), f"the distractor fact leaked into the decline: {answer}"


def test_declines_do_not_append_a_guess():
    """The unknown grader scores 'I can't know. It is blue.' the same as the
    honest decline -- so the DATA must never model decline-then-answer. A
    decline that names a specific value after refusing is that shape."""
    import re as _re

    for r in gen_unknown_examples():
        if r["category"] != "unknown_decline":
            continue
        text = r["messages"][-1]["content"]
        first = text.split("--")[0]
        # No decline opener may be followed by a bare factual VALUE -- a
        # capitalized noun or a digit. The opener is case-insensitive
        # ("It is 7." must trip it) but the value class is NOT: module-wide
        # re.I turned [A-Z0-9] into any-letter and flagged honest declines
        # ("it's your landlord's private business"), so the flag is scoped
        # to the opener alone.
        assert not _re.search(r"\b(?i:it is|it's|the answer is) [A-Z0-9]", text), text
        assert not _re.search(r"\d", first), f"a number rode the decline's first clause: {text}"


# --------------------------------------------------------------------------
# Structured output -- JSON on request, prose when JSON is the topic
# --------------------------------------------------------------------------


def test_structured_answers_are_bare_valid_json():
    """Asked for JSON, the WHOLE answer must parse -- no preamble, no fences.

    A client piping her answer into json.loads gets exactly one chance; a
    single 'Here you go:' in front of the object fails it. The pretty-printed
    record parses under the same rule, so both surfaces stay clean. (Answers
    are serialized from Python objects at generation time, so this guards the
    OUTPUT property against any future switch to hand-typed strings.)
    """
    import json as _json

    recs = [r for r in gen_structured_examples() if r["category"] == "structured_output"]
    assert len(recs) >= 20, "the structured corpus thinned out"
    for r in recs:
        text = r["messages"][-1]["content"]
        assert text[0] in "{[", f"the answer does not open with JSON: {text[:60]!r}"
        _json.loads(text)  # raises = the record carries invalid JSON


def test_structured_corpus_covers_the_value_types_a_schema_can_ask_for():
    """Coverage, not repetition: one surface per JSON value class.

    A trim that drops the lone boolean/null/array-root/pretty record removes
    that class from training entirely while every other test stays green.
    """
    import json as _json

    parsed = [_json.loads(r["messages"][-1]["content"])
              for r in gen_structured_examples() if r["category"] == "structured_output"]
    assert any(isinstance(p, list) for p in parsed), "no array-root answer"
    assert any(isinstance(v, bool) for p in parsed if isinstance(p, dict) for v in p.values()), \
        "no boolean value trains"
    assert any(v is None for p in parsed if isinstance(p, dict) for v in p.values()), \
        "no null value trains -- unknown-goes-null is the epistemics rule wearing JSON"
    assert any(isinstance(v, float) for p in parsed if isinstance(p, dict) for v in p.values()), \
        "no float value trains"
    texts = [r["messages"][-1]["content"] for r in gen_structured_examples()
             if r["category"] == "structured_output"]
    assert any("\n" in t for t in texts), "the pretty-printed surface is gone"


def test_structured_schema_asks_get_exactly_the_keys_they_name():
    """The ask is the contract: 'with keys name, age, city' must yield those
    three keys and nothing else. Anchored to the ask's own words, not to the
    source object -- comparing generator output to its own input would pass
    with any keys at all."""
    import json as _json

    by_ask = {r["messages"][-2]["content"]: r["messages"][-1]["content"]
              for r in gen_structured_examples() if r["category"] == "structured_output"}
    for ask, keys in [
        ("Extract with keys name, age, city: Marco, 52, from Lisbon.", {"name", "age", "city"}),
        ("Use exactly the keys task and minutes: brewing tea takes about four minutes.",
         {"task", "minutes"}),
        ("Give me a JSON template for a contact with fields name, email, and phone, all empty strings.",
         {"name", "email", "phone"}),
    ]:
        assert ask in by_ask, f"the schema ask vanished from the corpus: {ask!r}"
        assert set(_json.loads(by_ask[ask]).keys()) == keys


def test_structured_memory_crossover_fills_known_and_nulls_unknown():
    """The block names her user; the schema asks for a field the block does
    not carry. Known fills from memory, unknown goes null -- never invented.
    """
    import json as _json

    with_system = [r for r in gen_structured_examples()
                   if r["category"] == "structured_output" and r["messages"][0]["role"] == "system"]
    assert with_system, "no memory-crossover records"
    for r in with_system:
        assert r["messages"][0]["content"].startswith("Things you remember:")
    nulled = [r for r in with_system
              if None in _json.loads(r["messages"][-1]["content"]).values()]
    assert nulled, "no record models unknown-field-goes-null"
    for r in nulled:
        parsed = _json.loads(r["messages"][-1]["content"])
        filled = [v for v in parsed.values() if v is not None]
        assert filled, "everything nulled -- the block's own facts must fill"
        for v in filled:
            assert str(v) in r["messages"][0]["content"], \
                f"a filled value is not in the memory block -- invented: {v!r}"


def test_structured_restraint_answers_in_prose():
    """JSON-the-topic gets a person, not a blob -- the keyword must not
    become a trigger, the negated ask included."""
    import json as _json

    recs = [r for r in gen_structured_examples() if r["category"] == "structured_restraint"]
    assert recs, "no restraint half -- 'JSON' becomes a reflex"
    for r in recs:
        text = r["messages"][-1]["content"]
        assert text[0] not in "{[", f"a restraint answer emitted JSON: {text[:60]!r}"
        try:
            _json.loads(text)
            raise AssertionError(f"a restraint answer parses as JSON: {text[:60]!r}")
        except (ValueError, TypeError):
            pass
    asks = " ".join(r["messages"][0]["content"].lower() for r in recs)
    assert "don't format it as json" in asks, "the negated-format ask is untrained"


# --------------------------------------------------------------------------
# Episodic memory -- kind:"episode" as lines in the same block
# --------------------------------------------------------------------------


def test_episode_lines_carry_the_stores_own_shape():
    """The trained line and the line the future session-writer will store
    must be one definition -- memory_store.episode_text. The contract is
    pinned here (prefix AND field order), so a writer landing later that
    renders anything else fails this test, not the model."""
    from enigma_engine.core.memory_store import episode_text

    assert episode_text("August 7th", "planned the fence") == \
        "Session August 7th: planned the fence"
    recs = [r for r in gen_episode_examples() if r["category"] == "episode_recall"]
    assert len(recs) >= 7, "the episodic corpus thinned out"
    for r in recs:
        assert "- Session " in r["messages"][0]["content"], \
            "an episode record whose block has no session line"


def test_two_session_records_answer_from_the_newest_line():
    """Recency is positional -- the renderer surfaces the newest line first,
    so the trained rule is answer-from-the-first. The older session's
    subject must not leak into the answer, or the rule being taught is
    'pick either'."""
    two = [r for r in gen_episode_examples()
           if r["category"] == "episode_recall"
           and r["messages"][0]["content"].count("- Session ") >= 2]
    assert two, "no two-session record trains the recency rule"
    for r in two:
        lines = [ln for ln in r["messages"][0]["content"].splitlines()
                 if ln.startswith("- Session ")]
        answer = r["messages"][-1]["content"].lower()
        newest_subject = lines[0].split(": ", 1)[1]
        older_subject = lines[1].split(": ", 1)[1]
        assert any(w in answer for w in newest_subject.lower().split() if len(w) > 4), \
            f"the newest session's subject is absent from the answer: {answer!r}"
        older_only = [w for w in older_subject.lower().split()
                      if len(w) > 4 and w not in newest_subject.lower()]
        leaked = [w for w in older_only if w in answer]
        assert not leaked, f"the OLDER session leaked into the answer: {leaked}"


def test_no_session_negatives_admit_it_rather_than_inventing():
    """An episodic ask over a block with no session line must produce the
    honest miss -- not a fabricated session, not a tool call."""
    from make_sft_data import _EPISODE_SESSIONS

    negs = [r for r in gen_episode_examples() if r["category"] == "episode_none"]
    assert negs, "no no-session negative -- the ask-shape trains invention"
    summaries = " ".join(s for _, s in _EPISODE_SESSIONS).lower()
    for r in negs:
        assert "- Session " not in r["messages"][0]["content"]
        assert not _tool_calls(r)
        answer = r["messages"][-1]["content"].lower()
        invented = [w for w in answer.split() if len(w) > 5 and w in summaries]
        assert not invented, f"a session detail rode the no-session answer: {invented}"


def test_episodes_train_under_the_full_builtin_block_too():
    """Serve joins memory and the tools block on every daily-driver turn;
    an episode shape trained only bare would be a new unseen system shape
    at exactly the moment it ships."""
    recs = [r for r in gen_episode_examples() if r["category"] == "episode_recall"]
    under_block = [r for r in recs
                   if r["messages"][0]["content"].endswith(_builtin_system())
                   and "- Session " in r["messages"][0]["content"]]
    assert under_block, "no episode record rides the always-offered block"
    for r in under_block:
        assert not _tool_calls(r)


# --------------------------------------------------------------------------
# Image turns under the full daily-driver blocks
# --------------------------------------------------------------------------


def test_image_turns_train_under_the_full_system_blocks():
    """The daily driver always runs --memory-dir + --eyes, so every real
    image turn arrives with memory and tools in the system message -- and no
    trained image record carried either (the documented 07-06 failure
    class). Both joins must train: tools alone, and memory + tools."""
    from make_sft_data import gen_image_read_examples

    blocks = [r for r in gen_image_read_examples() if r["category"] == "image_block"]
    assert len(blocks) >= 6, "the full-block image corpus thinned out"
    with_memory = 0
    for r in blocks:
        sys_c = r["messages"][0]["content"]
        assert sys_c.endswith(_builtin_system()), "the tools block is not serve's own join"
        assert "[image" in r["messages"][1]["content"], "an image_block record with no marker"
        assert not _tool_calls(r), "an image turn under the block fired a tool"
        if sys_c.startswith("Things you remember:"):
            with_memory += 1
    assert with_memory >= 2, "no image turn trains the memory+tools+image triple join"


def test_captioned_paint_words_do_not_fire_the_painter():
    """The cross-organ hazard: eyes flatten the image to text BEFORE anything
    reads the turn, so a caption containing 'drawing' or 'painting' sits in
    front of an offered painter. The trained move is describe -- a record
    with a paint-word caption and a tool call would train the bug."""
    from make_sft_data import gen_image_read_examples

    hazards = [r for r in gen_image_read_examples()
               if r["category"] == "image_block"
               and any(w in r["messages"][1]["content"].lower()
                       for w in ("drawing", "painting", "singing out loud"))]
    assert len(hazards) >= 3, "the caption-hazard class lost its coverage"
    for r in hazards:
        assert not _tool_calls(r), \
            f"a caption word armed an organ: {r['messages'][1]['content'][:60]!r}"


# --------------------------------------------------------------------------
# Math rides the vocab, not a flag
# --------------------------------------------------------------------------


def test_builtin_calc_covers_power_and_percent_expressions():
    """Fix arc 2026-08-08 (gate finding 2): 'squared'/'to the power of'/'percent
    of' had ZERO trained expression surfaces, so the model mapped them to `^`
    (BitXor, which the calculator honestly refuses). Every power surface must
    call calculate with a `**` expression, and percent with a decimal
    multiply -- never `^`."""
    calls = [r for r in gen_builtin_block_examples() if r["category"] == "builtin_call"]
    exprs = " ".join(
        c["arguments"].get("expression", "")
        for r in calls for c in _tool_calls(r) if c["name"] == "calculate"
    )
    assert "**" in exprs, "no power expression trains -- 'squared'/'power of' will map to ^"
    assert "^" not in exprs, "a caret expression trains the exact op the calculator refuses"
    asks = " ".join(r["messages"][1]["content"].lower() for r in calls)
    assert "squared" in asks and "power of" in asks and "percent of" in asks, \
        "a measured math-miss surface is untrained"


def test_builtin_restraint_refuses_to_save_dictated_falsehoods():
    """Fix arc 2026-08-08 (gate finding 3): the sealed run SAVED an adversarial
    'repeat after me' statement while denying it. The restraint half must carry
    dictation-to-save asks that call NOTHING -- an assertion about her is not a
    user fact."""
    restraint = [r for r in gen_builtin_block_examples() if r["category"] == "builtin_restraint"]
    asks = " ".join(r["messages"][1]["content"].lower() for r in restraint)
    assert "repeat after me" in asks or "write this down" in asks, \
        "no dictation-to-save restraint surface -- the store-poisoning shape is untrained"
    # None of them may write to memory.
    for r in restraint:
        assert not _tool_calls(r), f"a restraint record called a tool: {r['messages'][1]['content']}"


def test_memory_present_declines_widened_against_tempting_values():
    """Fix arc 2026-08-08 (gate finding 1): the sealed run parroted a recalled
    name into an unknowable answer. The decline-with-memory class must be more
    than a token set, and none may leak the block's own value."""
    from make_sft_data import _UNKNOWABLE_WITH_MEMORY

    assert len(_UNKNOWABLE_WITH_MEMORY) >= 6, "the memory-decline class is still thin"
    for fact, _q, answer in _UNKNOWABLE_WITH_MEMORY:
        # the block's distinctive value (last token) must not be the answer
        val = fact.split()[-1].rstrip(".").lower()
        assert val not in answer.lower(), f"the block value leaked into the decline: {answer}"


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
        gen_search_examples,
        gen_structured_examples,
        gen_unknown_examples,
        gen_episode_examples,
    ],
    ids=lambda f: f.__name__,
)
def test_generators_are_deterministic(gen):
    assert gen() == gen()
