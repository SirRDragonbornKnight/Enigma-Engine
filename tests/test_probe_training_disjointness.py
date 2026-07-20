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
