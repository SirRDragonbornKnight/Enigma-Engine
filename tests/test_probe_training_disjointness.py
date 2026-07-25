"""Dev probes must not collide with make_sft_data's OWN generated surfaces.

The exact-match dev guard (_held_out) silently DROPS a training record whose
question equals a probe question -- by design for probe hygiene, but the
2026-07-20 audit found new memory probes byte-colliding with trained memory
surfaces, which quietly deleted 3 training records at bake (including the v8
anniversary coverage fix). This pins the whole class: every generated
question surface stays disjoint from the live probe set, so widening the dev
eval can never again cannibalize training coverage unnoticed.

(knowledge_corpus disjointness is pinned separately in test_knowledge_data.py;
this file covers the tool/math/memory/image generator streams.)
"""

from __future__ import annotations

import pytest

from make_sft_data import (
    _eval_probe_questions,
    _norm_q,
    gen_image_read_examples,
    gen_math_examples,
    gen_memory_read_examples,
    gen_memory_tools_examples,
    gen_tool_examples,
)

GENERATORS = [
    gen_tool_examples,
    gen_math_examples,
    gen_memory_read_examples,
    gen_memory_tools_examples,
    gen_image_read_examples,
]


@pytest.mark.parametrize("gen", GENERATORS, ids=lambda g: g.__name__)
def test_generated_surfaces_disjoint_from_probes(gen):
    probes = _eval_probe_questions()
    assert probes, "behavior_probes.jsonl missing -- this test proves nothing"
    collisions = [
        _norm_q(rec) for rec in gen() if _norm_q(rec) in probes
    ]
    assert not collisions, (
        f"{gen.__name__} produces question surfaces that equal live probes -- "
        f"the bake would silently DROP these training records: {collisions[:5]}"
    )


def test_a_trimmed_prompt_is_rescreened_before_it_is_emitted():
    """The trim runs AFTER the leak screen and keeps the prompt's TAIL, so a
    record that passed the screen could re-enter the mix carrying exactly the
    text the screen refuses (fix-arc audit, 2026-07-25: score 0.005 pre-trim
    -> 1.0 post-trim) -- and a rebuild re-creates the same trimmed record
    deterministically, making this the one consume-time refusal "rebuild the
    artifact" could not clear. fit_mix_to_block now screens every cut with
    the builder's own predicate."""
    import json

    from make_sft_data import fit_mix_to_block

    filler = " ".join(f"pad{i}" for i in range(400))
    leaky = json.dumps({"prompt": filler + " FORBIDDEN fact at the end",
                        "completion": "Ok."})
    clean = json.dumps({"prompt": filler + " harmless fact at the end",
                        "completion": "Ok."})

    out, trimmed, dropped, leaked = fit_mix_to_block(
        [leaky, clean], block=128, screen=lambda cut: "FORBIDDEN" in cut)
    assert leaked == 1, "the leaky trimmed prompt was emitted unscreened"
    assert trimmed == 1 and dropped == 0
    assert len(out) == 1 and "FORBIDDEN" not in out[0]
    # tail-keeping is the mechanism under test: the survivor still ends in
    # its tail, not its head
    assert "at the end" in json.loads(out[0])["prompt"]

    # without a screen both records trim and pass -- the fit behavior itself
    # is unchanged
    out2, trimmed2, dropped2, leaked2 = fit_mix_to_block([leaky, clean], block=128)
    assert (trimmed2, dropped2, leaked2) == (2, 0, 0) and len(out2) == 2

    # A MID-prompt hit must not doom the record: "a shorter cut keeps the
    # same tail and still leaks" was measured false (round-B audit,
    # 2026-07-25) -- the loop keeps shrinking until a clean cut fits, and
    # only a record with NO clean fitting cut counts as leaked. FORBIDDEN
    # sits mid-prompt here with clean tails of every length from "inside the
    # first fitting cut" down to "far outside it", so under the old
    # break-on-first-leaky-cut at least the short-tail records were dropped.
    mids = [
        json.dumps({"prompt": filler + " FORBIDDEN fact in the middle " +
                    " ".join(f"tail{i}word{j}" for j in range(n)),
                    "completion": "Ok."})
        for i, n in enumerate(range(10, 130, 8))
    ]
    out3, trimmed3, dropped3, leaked3 = fit_mix_to_block(
        mids, block=128, screen=lambda cut: "FORBIDDEN" in cut)
    assert dropped3 == 0
    assert leaked3 == 0, "a record with a clean fitting cut was given up on"
    assert len(out3) == len(mids) and trimmed3 == len(mids)
    assert all("FORBIDDEN" not in json.loads(ln)["prompt"] for ln in out3)
