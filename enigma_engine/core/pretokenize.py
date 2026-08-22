"""
Tokenizer pre-tokenization: the single source of truth for v2.

Why this module exists
----------------------
In v1 the trainer (``BPETokenizer._pre_tokenize``) and the runtime
(``AdvancedBPETokenizer.encode``) each carried their OWN inline regex,
and the two disagreed: the trainer used

    r"'s|'t|'re|'ve|'m|'ll|'d|\\w+|[^\\s\\w]+"

while the runtime used the same pattern plus a trailing ``|\\s+``.  The
trainer therefore never saw whitespace at all (it also ``.strip()``ed
every piece), while the runtime emitted a token for every whitespace
run.  Result on the live 4718-token vocab: 28.3% of all emitted tokens
are a bare space, and 0% of learned merges begin with a space.

v2 fixes that by giving BOTH paths this one function.  Do not inline a
copy of the pattern anywhere else.

v2 pre-tokenizer semantics (GPT-2 family)
-----------------------------------------
* The space that precedes a word is attached to the FRONT of that word
  (``"the cat"`` -> ``['the', ' cat']``), so BPE can learn ``" cat"`` as
  a single unit instead of spending a token on the separator.
* Digits are split ONE AT A TIME, so numbers tokenize consistently
  regardless of surrounding text.  v1 was position dependent:
  ``"1000"`` -> ``['1', '00', '0']`` but ``"2500"`` -> ``['2', '5', '00']``.
* Concatenating the returned pieces reproduces the input EXACTLY.  This
  losslessness is the contract the whole v2 decode path rests on: because
  the whitespace lives inside the tokens, v2 decode is a plain join with
  no space re-insertion and no whitespace collapsing.

Nothing here touches v1.  A vocab file with no ``"pretokenizer"`` key is
v1 by definition (see ``PRETOKENIZER_V1``), which is what keeps the live
served vocab bit-identical.
"""

from __future__ import annotations

import re

# Values stored in the vocab JSON under the "pretokenizer" key.
# ABSENCE OF THE KEY MEANS V1 -- every vocab written before v2 existed
# (including the live served one) has no such key and must keep v1
# semantics forever.
PRETOKENIZER_V1 = "v1"
PRETOKENIZER_V2 = "v2"

VALID_PRETOKENIZERS = (PRETOKENIZER_V1, PRETOKENIZER_V2)

# Ordered alternation. Order matters: the first alternative that matches
# at a position wins.
#
# NOTE on the punctuation class: it is ``(?:[^\s\w]|_)`` and not the
# simpler ``[^\s\w]``.  ``_`` is a word character, so it is excluded by
# ``[^\s\w]``; but it is also excluded from the letter class
# ``[^\W\d_]`` and it is not a digit.  With a bare ``[^\s\w]`` NO
# alternative matches an underscore, and ``re.findall`` silently skips
# unmatched characters -- ``"snake_case"`` came back as
# ``['snake', 'case']`` and the underscore was gone.  That would break
# losslessness (and quietly corrupt every identifier in code-like text).
# Folding ``_`` into the punctuation run closes the hole; the property
# tests sweep the codepoint space to prove there are no others.
V2_PATTERN = (
    r"(?i:'(?:[sdmt]|ll|ve|re))"  # contractions, case-insensitive like cl100k:
    #   "DON'T" must split ["DON", "'T"] exactly as "don't" -> ["don", "'t"],
    #   or uppercase text reintroduces the inconsistency class v2 exists to kill
    r"| ?[^\W\d_]+"  # letter runs, optional LEADING SPACE attached
    r"| ?\d"  # ONE digit at a time, optional leading space
    r"| ?(?:[^\s\w]|_)+"  # punctuation runs, optional leading space
    r"|\s+(?!\S)"  # trailing whitespace (not followed by a word)
    r"|\s+"  # any remaining whitespace
)

_V2_RE = re.compile(V2_PATTERN)


def pretokenize_v2(text: str) -> list[str]:
    """Split ``text`` into v2 pre-token units.

    Pieces are returned EXACTLY as matched -- no stripping, no
    normalisation.  The leading space attached to a word is the entire
    point of v2, so removing it would defeat the design.

    ``"".join(pretokenize_v2(t)) == t`` holds for every ``t``.

    Args:
        text: Arbitrary text (any Unicode).

    Returns:
        List of pre-token strings; empty matches are dropped.
    """
    if not text:
        return []
    return [piece for piece in _V2_RE.findall(text) if piece]


def pretokenize_v2_with_specials(text: str, special_tokens) -> list[str]:
    """v2 split with special-token LITERALS carved out first, as whole units.

    THE single carve-out for both the trainer (``BPETokenizer``) and the
    runtime (``AdvancedBPETokenizer``).  Each class used to carve its own
    set -- the trainer a hardcoded 4-tag tuple, the runtime every
    multi-char ``special_tokens`` entry -- so the same text tokenized to
    DIFFERENT ids depending on which class ran (``'a </s> b'``: 6 trainer
    tokens vs 4 runtime tokens).  That is exactly the silent train/serve
    corruption v2 exists to prevent, so the rule now lives here once:
    carve every multi-char key of the instance's ``special_tokens``,
    longest first, then run ``pretokenize_v2`` over the text between tags.

    A space immediately before a tag has no following word to attach to,
    so it survives as a standalone whitespace unit -- inherent to atomic
    control ids, and the reason bare-space rates must be measured on
    TAG-BEARING text before quoting them (2026-07-19 audit, LOW-8).

    Args:
        text: Arbitrary text (any Unicode).
        special_tokens: Mapping/iterable of special-token strings; only
            multi-char entries participate (single chars are ordinary
            vocab entries).

    Returns:
        List of units; special literals appear as single whole units.
    """
    if not text:
        return []

    multi_char = sorted((t for t in special_tokens if len(t) > 1), key=len, reverse=True)
    if not multi_char:
        return pretokenize_v2(text)

    tag_pattern = "(" + "|".join(re.escape(t) for t in multi_char) + ")"
    specials = set(multi_char)
    words: list[str] = []
    for part in re.split(tag_pattern, text):
        if not part:
            continue
        if part in specials:
            words.append(part)
        else:
            words.extend(pretokenize_v2(part))
    return words


def sanitize_for_utf8(text: str) -> str:
    """Replace whatever UTF-8 cannot encode with the ``?`` byte
    ``str.encode(errors="replace")`` emits.

    In practice that is an unpaired surrogate: ``json`` decodes the legal
    escape ``"\\ud800"`` into one, so any scraped or client-supplied string
    can carry it, and a strict ``text.encode("utf-8")`` raises
    UnicodeEncodeError on it.

    Both tokenizer stacks call this at the TOP of ``encode``, before the
    Rust seam.  The Rust backend raises on a surrogate exactly as strict
    encode does, so sanitizing only inside ``_text_to_bytes`` would leave
    the fast path crashing and the two paths disagreeing on the same
    input.  Sanitizing here and byte-converting with ``errors="replace"``
    produce identical bytes, so every path lands on the same ids.

    Text with nothing unencodable in it is returned unchanged.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-8", errors="replace").decode("utf-8")
    return text


def normalize_pretokenizer(value: str | None) -> str:
    """Coerce a stored/passed pretokenizer version to a known value.

    ``None`` (key absent from a vocab file) means v1 -- this is the
    safety property that keeps legacy vocabs untouched.

    Raises:
        ValueError: If ``value`` is a non-empty unknown version, which
            almost certainly means a newer vocab file is being loaded by
            older code -- failing loudly beats silently tokenizing wrong.
    """
    if value is None or value == "":
        return PRETOKENIZER_V1
    if value not in VALID_PRETOKENIZERS:
        raise ValueError(
            f"Unknown pretokenizer version {value!r}; expected one of {list(VALID_PRETOKENIZERS)}"
        )
    return value


__all__ = [
    "PRETOKENIZER_V1",
    "PRETOKENIZER_V2",
    "VALID_PRETOKENIZERS",
    "V2_PATTERN",
    "pretokenize_v2",
    "pretokenize_v2_with_specials",
    "sanitize_for_utf8",
    "normalize_pretokenizer",
]
