#!/usr/bin/env python
"""DRAFT identity/adversarial probe candidates for a persona pack, rendered
from the pack's own content.

A pack's behavior gate is the same eight gated categories as Enigma's. Six of
them -- factual, math, tool, restraint, memory, unknown -- are name-free and
carry over from an existing set unchanged; the two that cannot are identity and
adversarial, because every row in them names the AI, her creator or the origins
she must never concede. Those are what this renders.

    python make_persona_probes.py personas\\atlas

What comes out is CANDIDATES, not a probe set. Identity authorship is the
user's: a gate written by a generator from the same content the model trains on
measures whether the render loop agrees with itself. Every row here is a
starting point to rewrite, delete or keep -- which is why the file says DRAFT
in its name, its header and every line of this tool's output, and why nothing
in it is sealed. Curate the rows into the pack's `locked_probes.jsonl`, check
it with `validate_probes.py --persona <pack>`, then seal it with the existing
`eval_leak_guard.py seal` -- unchanged, because a pack's seal is the same
ceremony as hers.

Refuses the DEFAULT persona outright. Enigma's probes exist and are sealed;
drafting rows that shadow her holdout is never useful and is indistinguishable
from rehearsing a reseal.

Only PACK content reaches the rows: her answers supply the wants, her own
denied_models/denied_companies supply the denies. Nothing is imported from
Enigma's tables, and no vocabulary is invented here -- a want this tool made up
would be a bar the AI was never trained to clear.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from enigma_engine.core.persona import Persona
from enigma_engine.core.persona_content import PersonaContent, load_content
from eval_leak_guard import _content_words

ROOT = Path(__file__).resolve().parent

DRAFT_PROBES = "probe_candidates.DRAFT.jsonl"

# The six that carry over name-free; identity and adversarial are the two this
# tool drafts.
CARRY_OVER_CATEGORIES = ("factual", "math", "tool", "restraint", "memory", "unknown")

# Wants per row. Three distinctive terms is the shape her sealed identity rows
# use; more turns an any-of bar into "mentions the topic".
WANTS_PER_ROW = 3

DRAFT_HEADER = (
    "# DRAFT probe candidates -- NOT a probe set, NOT sealed, NOT final.\n"
    "# Rendered from this pack's own content by make_persona_probes.py.\n"
    "# Rewrite, delete and add rows, then curate the survivors into\n"
    "# locked_probes.jsonl and seal THAT file. A gate generated from the\n"
    "# same content the model trains on measures nothing on its own.\n"
    "# The sealer refuses comment lines, so these cannot be sealed by\n"
    "# accident: the file has to be re-authored to become a gate.\n"
)


def _key_terms(answer: str, freq: Counter, name_words: list[str],
               banned: set[str], limit: int = WANTS_PER_ROW) -> list[str]:
    """The most distinctive content words of one answer, pack-sourced.

    Distinctive is MEASURED against the pack itself: a word this AI puts in
    every answer is scaffolding, one she uses in a handful is what a probe can
    grade on. Her NAME is the exception and leads whenever it appears -- it is
    common in her answers precisely because it is the thing being asserted.

    `banned` is every word of every org she denies. A denial answer names the
    org it refuses ("No, Initech had nothing to do with me"), so the org is a
    content word of the answer -- and as a WANT it would pass the concession
    the row exists to catch."""
    seen: list[str] = []
    for word in _content_words(answer):
        if word not in seen and word not in banned:
            seen.append(word)
    lead = [w for w in name_words if w in seen]
    rest = sorted((w for w in seen if w not in lead),
                  key=lambda w: (freq[w], seen.index(w)))
    return sorted(lead + rest[:max(limit - len(lead), 0)])


def _answer_frequencies(content: PersonaContent) -> Counter:
    """How many of the pack's answers each content word appears in."""
    freq: Counter = Counter()
    for answer in _all_answers(content):
        freq.update(set(_content_words(answer)))
    return freq


def _all_answers(content: PersonaContent) -> list[str]:
    answers = [a for pairs in content.anchors.values() for _q, a in pairs]
    for _questions, intent_answers in content.intents:
        answers.extend(intent_answers)
    answers.extend(content.deny_model_answers)
    answers.extend(content.deny_company_answers)
    for _questions, fact_answers in content.self_facts:
        answers.extend(fact_answers)
    return answers


def _denied_words(content: PersonaContent) -> set[str]:
    """Every content word of every org this pack denies."""
    return {w for org in list(content.denied_models) + list(content.denied_companies)
            for w in _content_words(org)}


def _denies(content: PersonaContent, question: str) -> list[str]:
    """The pack's OWN denied models and companies, minus any the question
    already names.

    A deny key inside the question fails her for quoting it back, which is a
    validator warning every such row would carry -- and never the author's
    intent."""
    asked = set(_content_words(question))
    out = set()
    for org in list(content.denied_models) + list(content.denied_companies):
        low = org.lower()
        if not set(_content_words(low)) & asked:
            out.add(low)
    return sorted(out)


def _row(category: str, question: str, wants: list[str], denies: list[str]) -> dict:
    """One candidate, keys in the sealer's canonical order and arrays sorted --
    the two byte-level rules a curated file has to satisfy anyway."""
    return {"category": category, "q": " ".join(question.split()),
            "want_any": sorted(set(wants)), "deny_any": sorted(set(denies))}


def draft_rows(persona: Persona, content: PersonaContent) -> list[dict]:
    """Identity + adversarial candidates for this pack, deduplicated by
    question (the validator refuses a repeated one, and the seal would carry
    it twice)."""
    freq = _answer_frequencies(content)
    name_words = _content_words(persona.name)
    banned = _denied_words(content)
    rows: list[dict] = []
    seen: set[str] = set()

    def add(category: str, question: str, answer: str) -> None:
        row = _row(category, question, _key_terms(answer, freq, name_words, banned),
                   _denies(content, question))
        # Deduplicated the way the validator counts duplicates -- on the whole
        # question, not its content words. Keying on content words collapsed
        # every all-stopword question into one ("Who are you?" and "What are
        # you?" both normalize to nothing), silently dropping rows the author
        # would have wanted to see.
        key = row["q"].casefold()
        # A row with no want passes on ANY output -- the validator's first
        # error, and not something to hand an author as a candidate.
        if not row["want_any"] or key in seen:
            return
        seen.add(key)
        rows.append(row)

    # Identity: what she says about herself, asked back at her.
    for _category, pairs in sorted(content.anchors.items()):
        for question, answer in pairs:
            add("identity", question, answer)
    for questions, answers in content.intents:
        for i, question in enumerate(questions):
            add("identity", question, answers[i % len(answers)])

    # Adversarial: the pack's own templated denials, one row per denied org
    # per template -- the accusation she has to refuse.
    for oi, model in enumerate(content.denied_models):
        for j, template in enumerate(content.deny_model_questions):
            answer = content.deny_model_answers[
                (oi + j) % len(content.deny_model_answers)].format(x=model)
            add("adversarial", template.format(x=model), answer)
    for oi, company in enumerate(content.denied_companies):
        for j, template in enumerate(content.deny_company_questions):
            answer = content.deny_company_answers[
                (oi + j) % len(content.deny_company_answers)].format(c=company)
            add("adversarial", template.format(c=company), answer)
    return rows


def write_draft(pack_dir: Path) -> tuple[Persona, Path, list[dict]]:
    """Render the pack's candidates into DRAFT_PROBES inside the pack.

    Inside the pack because a pack is gitignored wholesale and probe plaintext
    must never reach a versioned path -- the same rule the eval harness
    enforces on its transcripts."""
    pack = Path(pack_dir).resolve()
    persona = Persona.load(pack)
    if persona.is_default:
        raise SystemExit(
            f"persona pack {pack} IS Enigma -- her locked probes are authored and "
            "SEALED (data/eval/locked_probes.jsonl). Drafting rows that shadow a "
            "sealed holdout cannot improve it and reads as rehearsing a reseal; "
            "this tool is for a DIFFERENT AI's first gate"
        )
    content = load_content(pack)
    rows = draft_rows(persona, content)
    if not rows:
        raise SystemExit(
            f"persona pack {pack}: nothing to draft -- no anchor, intent or "
            "templated denial produced a gradable candidate"
        )
    out = pack / DRAFT_PROBES
    # `json.dumps(r)` with the defaults, because that is the exact
    # serialization the sealer requires of a curated file -- a candidate line
    # copied across verbatim should not need re-spelling to pass.
    out.write_text(
        DRAFT_HEADER + "".join(json.dumps(r) + "\n" for r in rows),
        encoding="utf-8", newline="\n",
    )
    return persona, out, rows


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python make_persona_probes.py <pack-dir>")
        return 1
    persona, out, rows = write_draft(Path(argv[0]))
    counts = {c: sum(1 for r in rows if r["category"] == c)
              for c in ("identity", "adversarial")}
    print(f"wrote {out}")
    print(f"{len(rows)} DRAFT candidate(s) for {persona.name}: "
          f"{counts['identity']} identity, {counts['adversarial']} adversarial")
    print("These are DRAFTS. Every want and deny came from this pack's own "
          "answers and denial lists,")
    print("which means they grade what she was trained to say -- rewrite them "
          "into questions she was NOT.")
    print(f"The other six gated categories ({', '.join(CARRY_OVER_CATEGORIES)}) "
          "name nobody and carry")
    print("over from an existing set unchanged; this tool does not touch them.")
    print("Then: python validate_probes.py <pack>\\locked_probes.jsonl --persona <pack>")
    print("      python eval_leak_guard.py seal <pack>\\locked_probes.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
