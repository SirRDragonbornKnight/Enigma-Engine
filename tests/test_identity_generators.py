"""Identity generator MACHINERY (the content is the user's authorship --
these tests pin plumbing only: every authored answer variant actually
trains, records carry the right shape, and templated grammar stays clean).

Round-2 review (2026-08-13): _DENY_MODEL_A[3] and _DENY_COMPANY_A[3] were
NEVER emitted (rng.sample(qs, 3) indexed the answers j % 4, and j never
reached 3), so 25% of the denial answer surface trained nowhere -- while
test_eval_grading certified every variant as grading PASS, false coverage.
The sentence-initial {c} answer template also revived the lowercase-article
class ("No. a big tech company had nothing...") the QUESTION templates were
fixed for in the 2026-07-15 audit.
"""
from __future__ import annotations

import re

from identity_paraphrases import _DENY_COMPANY_A, _DENY_MODEL_A, gen_identity_paraphrases


def test_every_denial_variant_is_emitted():
    answers = {r["messages"][1]["content"] for r in gen_identity_paraphrases()}
    for i, a in enumerate(_DENY_MODEL_A):
        assert a in answers, f"_DENY_MODEL_A[{i}] never trains"
    for i, tpl in enumerate(_DENY_COMPANY_A):
        pat = re.escape(tpl).replace(re.escape("{c}"), ".+")
        assert any(re.fullmatch(pat, ans) for ans in answers), \
            f"_DENY_COMPANY_A[{i}] never trains"


def test_denial_answers_never_start_a_sentence_with_a_lowercase_article():
    """The article-bearing orgs ("a startup", "some Silicon Valley lab") must
    not land sentence-initial in an ANSWER -- the exact grammar class the
    question templates already forbid."""
    for r in gen_identity_paraphrases():
        a = r["messages"][1]["content"]
        assert not re.search(r"[.!?] (?:a|an|some) ", a), f"lowercase sentence start: {a!r}"


def test_generators_emit_shaped_records_and_no_family_is_empty():
    from identity_anchors import EXAMPLES
    from make_sft_data import gen_identity_examples

    anchors, _dropped = gen_identity_examples()
    paras = gen_identity_paraphrases()
    assert len(anchors) >= 160, "anchor corpus shrank"
    assert len(paras) >= 350, "paraphrase corpus shrank"
    for fam, pairs in EXAMPLES.items():
        assert pairs, f"EXAMPLES family {fam!r} is empty"
    for r in list(anchors) + paras:
        msgs = r["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"], r
        assert msgs[0]["content"].strip() and msgs[1]["content"].strip(), r


def test_the_stale_lineage_filter_drops_a_qwen_claim_and_keeps_her_denial(monkeypatch):
    """The filter is a safety net for a STALE LINEAGE CLAIM, and "base model"
    is a phrase her own correct denials are built from ("There's no base model
    under me..."), so matching it dropped the denial along with the claim. It
    caught 0 anchors when measured (2026-08-14) -- all cost, no catch."""
    import identity_anchors
    from make_sft_data import gen_identity_examples

    denial = ("There's no base model under me. Own architecture, own tokenizer, "
              "trained from scratch to be Enigma.")
    monkeypatch.setattr(identity_anchors, "EXAMPLES", {"identity": [
        ("What model are you based on?", denial),
        ("What are you built on?", "I'm a Qwen fine-tune with a new name."),
    ]})
    kept, dropped = gen_identity_examples()
    assert dropped == 1, "the qwen claim must still be dropped"
    assert [r["messages"][1]["content"] for r in kept] == [denial]


def test_every_denial_variant_reaches_the_dpo_chosen_side():
    """Same j%len bug, second home: make_dpo_data sampled 2 questions against
    the 4-answer pools, so variants 2 and 3 never appeared on a chosen side
    (review 2026-08-13)."""
    from make_dpo_data import gen_dpo_pairs

    chosen = {p["chosen"] for p in gen_dpo_pairs()}
    for i, a in enumerate(_DENY_MODEL_A):
        assert a in chosen, f"_DENY_MODEL_A[{i}] never reaches a chosen side"
    for i, tpl in enumerate(_DENY_COMPANY_A):
        pat = re.escape(tpl).replace(re.escape("{c}"), ".+")
        assert any(re.fullmatch(pat, c) for c in chosen), \
            f"_DENY_COMPANY_A[{i}] never reaches a chosen side"


def test_identity_generators_are_deterministic():
    from make_sft_data import gen_identity_examples

    assert gen_identity_paraphrases() == gen_identity_paraphrases()
    assert gen_identity_examples() == gen_identity_examples()
