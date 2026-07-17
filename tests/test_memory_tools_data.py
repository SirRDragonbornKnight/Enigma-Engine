"""The combined memory+tools system shape (ultrareview #6).

serve's _with_context joins retrieved memories and the tool preamble with a
blank line whenever a memory hit coincides with offered tools; training used
to carry each block only ALONE, and the model demonstrably fails on unseen
system shapes. gen_memory_tools_examples bakes exactly the joined shape --
these tests lock the byte-level join and both trained behaviors.
"""

from make_sft_data import gen_memory_tools_examples

_PREAMBLE_JOIN = (
    "\n\nYou are Enigma. You can use tools when they are needed; answer "
    "directly when they are not.\nAvailable tools:\n"
)


def test_combined_shape_matches_serves_join():
    recs = gen_memory_tools_examples()
    assert recs, "generator returned nothing"
    for r in recs:
        sys = r["messages"][0]
        assert sys["role"] == "system"
        content = sys["content"]
        # serve's join order: memories first, blank line, preamble + tools.
        assert content.startswith("Things you remember:\n- "), content[:60]
        assert _PREAMBLE_JOIN in content, content[:120]
        assert r["category"] == "memory_tools"


def test_both_behaviors_present():
    recs = gen_memory_tools_examples()
    with_calls = [r for r in recs if any(m.get("tool_calls") for m in r["messages"])]
    without = [r for r in recs if not any(m.get("tool_calls") for m in r["messages"])]
    assert with_calls, "no tool-call records -- the memory block must not suppress tool use"
    assert without, "no answer-from-memory records under offered tools"
    # every tool-call record carries a full trace: call -> result -> final answer
    for r in with_calls:
        roles = [m["role"] for m in r["messages"]]
        assert "tool" in roles
        assert roles[-1] == "assistant"
        assert r["messages"][-1]["content"], "final assistant turn must speak"


def test_generator_is_deterministic():
    assert gen_memory_tools_examples() == gen_memory_tools_examples()
