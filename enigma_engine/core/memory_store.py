"""Local memory for Enigma — the runtime-learning layer.

Her weights are frozen between training passes; THIS is where day-to-day
learning lives (the frozen-weights + external-memory consensus). Design
constraints, in order: black-box
(stdlib-only, no embedding service, no deps), inspectable (plain JSONL a human
can read and edit), small (she serves with a 1024-token context — retrieval
must be sharp, not big).

Retrieval is BM25 over whitespace/word tokens. At her scale — hundreds to a
few thousand memories — lexical scoring is the boring, proven choice.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from pathlib import Path
from typing import Any

from enigma_engine.core.safe_save import atomic_write_text

logger = logging.getLogger(__name__)

# The longest text remember() will file. A memory is a FACT, not a document:
# every record is re-tokenized on every retrieval (search scores the whole
# store) and a retrieved one is echoed into a 1024-token context, so a pasted
# megabyte would cost a full-corpus encode per turn and could not be recalled
# anyway. The cap belongs to the store because the store is the one owner both
# doors (the remember tool and POST /v1/memory) go through.
MAX_MEMORY_CHARS = 4000

# The apostrophe SEPARATES words: "dog's" tokenizes to ("dog", "s"), so a
# query about "my dog's name" reaches a stored "User's dog is named Rex."
_WORD = re.compile(r"[a-z0-9]+")

# Words that carry no memory identity ("my dog is Rex" vs "my cat is Whiskers"
# must NOT look similar just because both say "my ... is"). "user"/"s" are
# boilerplate in the stored fact form ("User's X is Y."), and question words
# carry no identity either -- without them a bare "is that right?" scores
# against every record containing "is".
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "my", "your", "our", "their", "his", "her", "its",
    "i", "you", "we", "they", "it", "this", "that",
    "and", "or", "of", "to", "in", "on", "for", "with", "as", "at",
    "s", "user", "users",
    "what", "who", "whom", "whose", "where", "when", "why", "how", "which",
    # "will" stays OUT of this list: it is a common given name, and a stopped
    # name is unfindable forever (audit 2026-07-22).
    "do", "does", "did", "can", "could", "would", "should",
    "tell", "me", "about", "again", "remind", "know", "have", "has", "had",
})

# The trained storage verbs, folded to their query forms directly. A generic
# suffix-stripper was tried here and collided real words (cared -> car,
# note -> not, audit 2026-07-22): a map over the finite verbs the store
# actually contains cannot.
_NORMALIZE = {
    "named": "name", "naming": "name",
    "called": "call", "calling": "call",
    "goes": "go", "went": "go",
}


def _stem(token: str) -> str:
    """Fold plurals and the trained storage verbs so names/named/name meet.

    Plural-only stripping: -es after a sibilant (bosses -> boss), else -s when
    the word does not end in -ss (names -> name, kids -> kid, boss stays
    boss). No verb-suffix or trailing-e stripping -- both were measured to
    merge unrelated words."""
    mapped = _NORMALIZE.get(token)
    if mapped:
        return mapped
    if len(token) <= 3:
        return token
    if token.endswith("es") and len(token) >= 5 and (token[-3] in "sxz" or token[-4:-2] in ("ch", "sh")):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _terms(text: str) -> list[str]:
    return [_stem(t) for t in _WORD.findall(text.lower())]


# Stopwords are matched AFTER stemming, so the set carries stemmed forms too
# ("does" stems to "doe", "have" to "hav").
_STOP_STEMS = frozenset(_stem(w) for w in _STOPWORDS) | _STOPWORDS


def _content_term_list(text: str) -> list[str]:
    return [t for t in _terms(text) if t not in _STOP_STEMS]


def _content_terms(text: str) -> set[str]:
    return set(_content_term_list(text))


# ---------------------------------------------------------------------------
# QUERY-TIME synonyms. RETRIEVAL ONLY.
# ---------------------------------------------------------------------------
# BM25 matches strings, so a fact she holds can be invisible to the question
# that asks for it: measured 2026-08-10, "What's my job?" retrieved NOTHING
# against a stored "I work as a nurse".
#
# These clusters widen the QUESTION. They must never touch what is STORED,
# SUPERSEDED or DELETED -- that key is subject + value-kind (ruled territory,
# audit 2026-07-22), and folding `job` into `work` there would let one fact
# silently delete another. Which is why this table lives here and is read in
# exactly one place: search(). It is deliberately small and curated; a big
# generated thesaurus is how retrieval starts inventing matches.
_SYNONYM_CLUSTERS: tuple[tuple[str, ...], ...] = (
    ("job", "work", "occupation", "profession", "career", "employer"),
    ("name", "called"),
    ("home", "live", "address", "reside"),
    ("weight", "weigh", "kilo", "pound", "lb", "kg"),
    # "paid" is listed beside "pay": _stem folds plurals only, deliberately --
    # no verb-suffix stripping (it merged unrelated words), so irregular forms
    # have to be spelled out here rather than reached by the stemmer.
    ("pay", "paid", "salary", "wage", "income", "earn"),
    ("age", "old", "born", "birthday"),
    ("phone", "number", "mobile", "cell"),
    ("email", "mail"),
    ("partner", "wife", "husband", "spouse", "married"),
    ("kid", "child", "son", "daughter"),
    ("pet", "dog", "cat"),
    ("car", "vehicle", "drive"),
    ("allergy", "allergic"),
    ("hobby", "pastime", "enjoy"),
    ("school", "study", "student", "degree"),
    ("favourite", "favorite", "prefer"),
    ("doctor", "gp", "physician"),
    ("medication", "medicine", "prescription", "pill"),
)

# A synonym hit counts for a FRACTION of a literal one, so expansion widens the
# net without reordering it: a record using the asked-for word still wins.
_SYNONYM_WEIGHT = 0.35


def _build_synonym_map() -> dict[str, tuple[str, ...]]:
    """Stem both sides once: the query terms scoring runs on are stemmed, so an
    unstemmed table would silently never match ("kilos" arrives as "kilo")."""
    table: dict[str, set[str]] = {}
    for cluster in _SYNONYM_CLUSTERS:
        stems = {_stem(w) for w in cluster}
        for stem in stems:
            table.setdefault(stem, set()).update(stems - {stem})
    return {k: tuple(sorted(v)) for k, v in table.items()}


_QUERY_SYNONYMS = _build_synonym_map()


def _expand_query_terms(q_terms: list[str]) -> list[tuple[str, float]]:
    """The query as (term, weight) pairs: the asked-for words at full weight,
    their synonyms dampened. A synonym already present literally keeps its full
    weight rather than being added twice."""
    weighted: list[tuple[str, float]] = [(t, 1.0) for t in q_terms]
    literal = set(q_terms)
    seen: set[str] = set()
    for term in q_terms:
        for syn in _QUERY_SYNONYMS.get(term, ()):
            if syn in literal or syn in seen:
                continue
            seen.add(syn)
            weighted.append((syn, _SYNONYM_WEIGHT))
    return weighted


# The stored fact form ("User's dog is named Rex.", "My bicycle is teal.")
# has a SUBJECT slot and a VALUE slot. The subject alone is NOT the fact's
# identity -- "dog is named Rex" and "dog is 3 years old" share the subject
# and are two different facts (audit 2026-07-22: keying on the subject made
# them delete each other). The key is subject + the KIND of value: a naming,
# a measurement, or other -- a rename replaces a name, while plain values
# about one subject COEXIST ("car is red" and "car is electric" are two
# facts; the shared coarse kind is not proof of a correction, and a wrong
# supersede destroys a fact). For a MEASUREMENT the value's non-digit terms
# also join the key (review 2026-08-13: the dog's weight ate the dog's age),
# so an age update replaces an age only while the unit wording matches --
# "4 years" replaces "3 years", but "175 pounds" and "180 lbs" COEXIST and
# the stale one is outranked at retrieval, never destroyed. That narrows the
# 2026-07-24 measures-replace ruling in the direction its own tiebreak
# prefers; the width stands as-implemented (blessed 2026-08-26;
# PHASE7_GATE item 6).
_FACT = re.compile(
    r"^\s*(?:the\s+)?(?:user's|users|user|my)\s+(?P<attr>[^.]{1,60}?)"
    r"\s+(?:is|are|was|were)\s+(?P<val>[^.]+?)\s*\.?\s*$",
    re.IGNORECASE,
)

# The second trained storage shape: "User VERBs ..." (lives in Denver, works
# as a teacher, drives a blue pickup, goes by Sam, is allergic to peanuts).
# Not copula-shaped, so _FACT misses it; without this parse a correction
# ("lives in Austin") had to clear the lexical bar and coexisted instead of
# replacing (audit 2026-07-22).
_VERB_FACT = re.compile(
    r"^\s*(?:the\s+user|user|i)\s+(?:(?:is|am|are)\s+)?"
    r"(?P<verb>[a-z]+)(?:\s+(?P<func>in|as|at|by|to|for|on|with|a|an|the)\b)?",
    re.IGNORECASE,
)

# Single-valued RELATIONS -- a person has exactly one of each, so a new value
# replaces the old however the wording moved. Keyed by (verb stem, function
# word), NOT by verb alone: "go by" is a nickname (single-valued) while "go"
# and "go to" are open-ended activities that accumulate (audit 2026-07-22 r3:
# a verb-only set collapsed "goes running"/"goes swimming" into one). Every
# relation NOT listed here is many-valued and COEXISTS -- the safe direction,
# since a wrong supersede destroys a fact and a missed one only leaves two to
# rank. No lexical "is this a correction?" guess is attempted: it cannot tell
# "loves reading books" from "loves writing books" (audit r3), and guessing
# wrong deletes.
# "go by" is NOT here: it means both "known as" and "travel by" ("I go by bus
# to work"), so replacing on it deletes a commute when a nickname arrives
# (audit 2026-07-22 r4). Names correct through the unambiguous "User's name is
# X" copula path instead. A person can hold two jobs or two cars; keeping
# work/drive single-valued bets on correction being commoner than accumulation
# (the trained "updated:" path), and accepts the rare two-of-a-kind as the
# documented limit -- the alternative loses the moved-city / changed-job
# correction the training data promises.
_SINGLE_VALUED_RELATIONS = frozenset({
    ("live", "in"), ("live", "at"),
    ("work", "as"), ("work", "at"),
    ("drive", ""), ("drive", "a"), ("drive", "an"), ("drive", "the"),
    ("sleep", ""), ("wake", ""),
})

# Words a forget ask carries about the ASKING, not about the fact. They are
# stripped from a forget query so "forget that I like tea" keys on {like, tea}
# and cannot subsume a record by sharing "forget". "everything"/"all" land here
# too: a query that survives the filter with nothing left deletes nothing.
_FORGET_NOISE = frozenset(_stem(w) for w in (
    "forget", "forgot", "forgetting", "remember", "remembered", "remembering",
    "delete", "deleted", "remove", "removed", "erase", "erased", "scratch",
    "drop", "everything", "anything", "something", "all", "any", "stuff",
    "please", "longer", "anymore", "said", "told",
    # Contraction fragments: the tokenizer splits "don't" into "don" + "t",
    # and a junk term in the query defeats every subset test ("I don't like
    # tea any more, forget it" matched nothing at all).
    "don", "doesn", "didn", "isn", "aren", "wasn", "cant",
    # ...and the fragments the tokenizer leaves behind from "I'm", "I've",
    # "I'll", "don't": these are not content, and leaving them in made
    # "forget I'm tall" behave differently from "forget that I am tall".
    "m", "ve", "re", "ll", "t",
    # Temporal and conversational filler. These are not subjects, and leaving
    # them in turned a WIPE ask into a needle: "forget everything now" filtered
    # down to {now} and deleted the one record that happened to contain that
    # word -- which supersede itself writes ("...my dog is named Bruno now.").
    # A scoped wipe ("forget everything about my dog") still keeps its subject
    # and still deletes, which is why this belongs here and not in a blanket
    # "any ask containing everything/all is a wipe" rule.
    #
    # Words that can DISCRIMINATE between two records stay OUT of this list,
    # however filler-ish they sound. Stripping "right" made "forget that my
    # right knee hurts" reach the LEFT knee; stripping "memory"/"fact"/"thing"
    # made "forget that my memory is bad" reach "User's mood is bad." Leaving
    # one in costs a match that deletes nothing; taking one out costs a wrong
    # delete, and deletes have no .bak.
    "now", "today", "yesterday", "tomorrow", "again", "still", "just",
    "already", "ok", "okay", "thanks", "thank", "anyway", "yet",
))


def _forget_terms(text: str) -> set[str]:
    """The content terms a forget match keys on, for an ask OR a record.

    Both sides go through THIS function. Filtering only the ask meant a record
    holding a noise word could never equal the ask's terms, so quoting that
    record word for word deleted its shorter sibling instead of itself."""
    return {t for t in _content_terms(text) if t not in _FORGET_NOISE}


# How many matches a refusal names before it summarizes. The refusal is fed
# back into her context as a tool result, and 21 matches measured 728 tokens of
# a 1024-token window. (measured 2026-07-25)
_FORGET_NAMED_MAX = 5

# The phrase that marks a refusal as "I asked which memory you meant". serve
# looks for it in her last turn so the NEXT message is read as the answer --
# a question the user could not answer is not a question. Kept here, beside the
# text that emits it, so the two cannot drift apart.
FORGET_PENDING_MARK = "memories match that"

# THE refusal's exact rendering, anchored to a line start. A bare substring
# test on the MARK could not tell the question from text that merely QUOTES
# it: a stored memory containing the phrase rode out in "forgot: ..." (a
# SUCCESS), was surfaced to the client as if she had asked something, and
# armed answering-mode with no question pending -- so her next ordinary turn
# was read as naming a memory to delete (2026-07-25 fix-arc audit). Only text
# matching this rendering is the handshake; everything else just mentions it.
# Kept beside the raise below for the same no-drift reason as the MARK.
FORGET_PENDING_RE = re.compile(
    r"(?m)^[0-9]+ " + re.escape(FORGET_PENDING_MARK)
    + r" -- say the one you mean word for word, or give its id: "
)


def renders_forget_pending(text: str) -> bool:
    """True only for text carrying THE ambiguity refusal (at a line start),
    never for text that merely quotes its marker phrase."""
    return bool(FORGET_PENDING_RE.search(text))

_NAMING_HEAD = re.compile(r"^(?:named|called|known\s+as)\b", re.IGNORECASE)
_NAME_STEM = _stem("name")  # the attribute spelling of a naming ("name is Sam")

# Lexical fallback for texts neither parse recognizes. Deliberately high:
# a missed supersede leaves two records to rank, a wrong one DESTROYS a fact.
_SUPERSEDE_MIN = 0.75


# CORRECTION lead-ins only. Both fact parses are ^-anchored, so a correction
# that opens with one ("Actually, my dog is named Bruno now.") parsed as
# nothing, keyed as nothing, and COEXISTED with the value it corrected --
# leaving the stale value to outrank it.
#
# The list is deliberately narrow. Stripping general hedges too (sorry, oh,
# well, by the way, wait) turned every hedged ADDITION into a correction and
# fed it to the destructive path: "Sorry, I drive a rental this week" deleted
# "User drives a Toyota", where before the strip it keyed None and coexisted.
# A marker earns its place here only if it announces that what follows
# REPLACES something -- not merely that the speaker is being tentative.
_LEAD_IN = re.compile(
    r"^\s*(?:(?:actually|no|nope|correction|i meant|oops|scratch that)\b[\s,:;-]+)+",
    re.IGNORECASE,
)



def _strip_lead_in(text: str) -> str:
    """Drop leading conversational markers so a correction parses as the fact
    it asserts. Never strips the whole text: a bare "Actually." keeps its
    words rather than becoming an empty, keyless string."""
    stripped = _LEAD_IN.sub("", text).strip()
    return stripped or text


def _value_kind(val: str) -> str:
    """A copula value's kind: name (an explicit naming head) / measure (a
    number) / other. Capitalization is NOT a signal -- keying "Red" as a name
    and "teal" as other made a colour correction coexist as a contradiction
    (audit 2026-07-22 r3)."""
    stripped = val.strip()
    if _NAMING_HEAD.match(stripped):
        return "name"
    if re.search(r"\d", stripped):
        return "measure"
    return "other"


def _fact_key(text: str) -> frozenset[str] | None:
    """The identity of what a fact ASSERTS, or None when unparseable.

    Copula shape: subject terms + the value's kind (so a name and an age about
    one subject are two facts). Verb shape: the verb stem + its function word.
    A name correction rides the copula path as "User's name is X"; there is no
    dedicated 'call me'/'goes by' key -- both double as travel/phone and
    guessing wrong deletes (audit 2026-07-22 r4)."""
    clean = _strip_lead_in(" ".join(str(text).split()))
    match = _FACT.match(clean)
    if match:
        key = _content_terms(match.group("attr"))
        if key:
            kind = _value_kind(match.group("val"))
            if kind == "measure":
                # The value's non-numeric terms name WHICH measure ("3 years
                # old" vs "40 pounds"): without them any two digit-bearing
                # values about one subject shared a key and the second
                # DELETED the first (review 2026-08-13, the dog's weight ate
                # the dog's age). An update of the same measure shares these
                # terms and still replaces; an all-digit value ("rent is
                # 1200") contributes none, and same-measure replacement is
                # exactly right there.
                key |= {t for t in _content_terms(match.group("val")) if not t.isdigit()}
            return frozenset(key | {f"kind:{kind}"})
        return None
    # "User is Sam." -- a self copula with a NAME value is an identity fact;
    # measures ("User is 30 years old") are left to the verb parse, where age
    # and height land on different keys and coexist rather than one coarse
    # "measure" kind collapsing them (audit 2026-07-22 r3).
    self_match = re.match(
        r"^\s*(?:the\s+user|user|i)\s+(?:is|am|are)\s+(?P<val>[^.]+?)\s*\.?\s*$",
        clean,
        re.IGNORECASE,
    )
    if self_match and _value_kind(self_match.group("val")) == "name":
        return frozenset({"self", "kind:name"})
    verb_match = _VERB_FACT.match(clean)
    if verb_match:
        verb = _stem(verb_match.group("verb").lower())
        # A numeric "verb" means the copula value led with a number ("is 30
        # years old") -- not a relation. Leave it to the lexical fallback,
        # where different measures about the user coexist.
        if verb not in _STOP_STEMS and not verb.isdigit():
            func = (verb_match.group("func") or "").lower()
            return frozenset({f"verb:{verb}", f"func:{func}"})
    return None


def _relation_is_single_valued(key: frozenset[str]) -> bool:
    """True when the key names a relation a person has exactly one of."""
    if "self" in key:
        return True
    verb = next((k[5:] for k in key if k.startswith("verb:")), None)
    if verb is None:
        # Copula subject+kind key. A naming or a measurement is one-per-
        # attribute, so a new value replaces -- for measures "one attribute"
        # means the key's unit terms match too (see _fact_key), so replacement
        # reaches same-unit-wording updates only; a naming is proven by the VALUE
        # ("dog is named Rex") or by the ATTRIBUTE ("User's name is Sam" --
        # the value 'Sam' carries no naming head, the attribute does). A
        # plain ("other") value is NOT provably single-valued: "my car is
        # red" and "my car is electric" are two facts sharing one coarse
        # kind, and replacing on it deletes one -- so plain values coexist,
        # accepting that a corrected mood or colour leaves the stale value
        # behind to be outranked, not erased.
        return "kind:name" in key or "kind:measure" in key or _NAME_STEM in key
    func = next((k[5:] for k in key if k.startswith("func:")), "")
    return (verb, func) in _SINGLE_VALUED_RELATIONS


def _valid_id(value: Any) -> bool:
    """A usable record id: an int, excluding bool (an int subclass a hand
    edit like ``"id": true`` would otherwise smuggle past the arithmetic)."""
    return isinstance(value, int) and not isinstance(value, bool)


def episode_text(date: str, summary: str) -> str:
    """The text an episode record carries -- ONE definition, shared.

    Tier-2 reserves kind:"episode" for session memory (ruled into T4 scope
    2026-08-08): an episode is an ordinary record whose text is a dated
    session summary, so it rides add()/recall/render_context unchanged --
    an addition, not a rewrite. The future session-writer stores
    ``add(episode_text(date, summary), kind="episode")``; the SFT data
    renders its trained block lines through this same function, so the
    surface she trains on and the one the store will render cannot drift.
    Recency contract: render_context tie-breaks equal scores newest-id
    first, so when several session lines surface, the FIRST is the newest
    -- the trained rule is answer-from-the-first.
    """
    date = " ".join(str(date).split())
    summary = " ".join(str(summary).split())
    if not date or not summary:
        raise ValueError("an episode needs both a date and a summary")
    return f"Session {date}: {summary}"


class MemoryStore:
    """Append-mostly JSONL store with BM25 search and budgeted rendering."""

    class TooBroad(ValueError):
        """A forget ask that names more memories than it may delete at once."""

    class ConcurrentWriter(RuntimeError):
        """The JSONL moved under this store, so another store owns it now."""

    def __init__(self, path: str | Path):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "memories.jsonl"
        # (mtime_ns, size) as of this store's own last read or write. See
        # _refuse_if_another_writer: the file's shape moving without us is the
        # one cheap sign that a second store is writing the same directory.
        self._seen_stamp: tuple[int, int] | None = None
        # serve_enigma's endpoints run in FastAPI's threadpool: without a lock,
        # two concurrent add() calls read the same len() and mint duplicate ids.
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        if self._load():
            # The renumbered ids are live in memory either way; a file that
            # cannot be replaced right now (read-only attribute, another
            # process holding it on Windows) must not stop the store from
            # loading. The next successful rewrite persists them.
            try:
                self._rewrite()
            except (OSError, self.ConcurrentWriter) as exc:
                logger.warning(
                    "memory store: could not persist renumbered ids to %s (%s); "
                    "continuing with in-memory ids",
                    self.file,
                    exc,
                )

    def _load(self) -> bool:
        """Read the JSONL into `_records` and re-baseline the write stamp.

        Returns whether any record had to be renumbered, so the CALLER decides
        whether to persist that. Keeping this a pure read is what lets
        _refuse_if_another_writer heal with it: a heal that could write would
        turn a refusal into the very clobber the refusal exists to prevent."""
        self._records = []
        if self.file.exists():
            # utf-8-sig: hand-edited files are inside the contract, and
            # Windows editors save UTF-8 with a BOM that would otherwise
            # corrupt the first line's JSON. Writes stay plain UTF-8.
            with open(self.file, encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict) and rec.get("text"):
                        # Whitespace-normalize, the same way add()/remember()
                        # do. Hand-edited files are inside the contract, and a
                        # record text carrying a newline was the last route by
                        # which "forgot: <text>" could FORGE the TooBroad
                        # rendering at a line start in her surfaced reply --
                        # arming forget answering-mode with no question
                        # pending (round-C audit, 2026-07-25).
                        rec["text"] = " ".join(str(rec["text"]).split())
                        if rec["text"]:
                            self._records.append(rec)
        # The file as this store has just read it -- the baseline every later
        # write checks itself against.
        self._seen_stamp = self._file_stamp()
        # Hand-edited files are inside the contract (module docstring): a
        # record whose id is missing or not a valid int gets renumbered here
        # so the max+1 id arithmetic in add()/remember() always sees ints.
        next_id = max((r["id"] for r in self._records if _valid_id(r.get("id"))), default=0) + 1
        renumbered = False
        seen_ids: set[int] = set()
        for rec in self._records:
            rid = rec.get("id")
            # Duplicate ids (a copy-paste hand edit) would make id-based
            # delete/supersede ambiguous, so later duplicates get fresh ids.
            if not _valid_id(rid) or rid in seen_ids:
                rec["id"] = next_id
                next_id += 1
                renumbered = True
            seen_ids.add(rec["id"])
        return renumbered

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    # ------------------------------------------------------- writer safety

    def _file_stamp(self) -> tuple[int, int] | None:
        """(mtime_ns, size) of the JSONL, or None when it does not exist."""
        try:
            st = self.file.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _refuse_if_another_writer(self) -> None:
        """Refuse to write over a file that moved since we last touched it,
        then HEAL (call with the lock held, immediately BEFORE the write).

        ONE writer per memory directory is the convention, and nothing
        enforced it: two MemoryStores on one dir both mint ids from their own
        in-memory maximum (measured: both minted id 1), and the first _rewrite
        either of them runs erases every record the other appended. There is
        no merge to do here -- the records the other store holds are not in
        this process -- so the honest answer is to refuse and say why. Cheap
        by design: a stat, not a re-read.

        REFUSING IS NOT ENOUGH ON ITS OWN, and refusing FOREVER is worse than
        the clobber it prevents: the stamp only re-baselined on a successful
        load/append/rewrite, so ONE foreign write -- a hand edit, which the
        module docstring puts squarely inside the contract -- wedged every
        later mutation of the live serve permanently (audit 2026-08-22). So
        the refusal is followed by a re-read: THE REFUSED OPERATION IS THE
        SACRIFICE, and the store recovers by adopting the on-disk truth, so
        the next operation proceeds against the file as it really is. The heal
        runs BEFORE the raise, and _load() never writes, so the foreign
        records are read, never overwritten.
        """
        if self._file_stamp() != self._seen_stamp:
            self._load()
            raise self.ConcurrentWriter(
                f"{self.file} changed since this store last read it -- another "
                f"MemoryStore is writing the same directory. One writer per "
                f"memory dir: a second store mints colliding ids and its next "
                f"rewrite erases these records. Nothing was written; this store "
                f"has re-read the file, so the next operation will see it."
            )

    def _append(self, rec: dict) -> None:
        """Append one record to the JSONL (call with the lock held)."""
        self._refuse_if_another_writer()
        with open(self.file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._seen_stamp = self._file_stamp()

    def add(self, text: str, kind: str = "fact", source: str | None = None) -> dict:
        text = " ".join(str(text).split())
        if not text:
            raise ValueError("empty memory")
        with self._lock:
            # max+1, not len+1: delete()/supersede shrink the list, and a
            # len-based id would collide with a surviving record.
            rec = {"id": max((r["id"] for r in self._records), default=0) + 1, "text": text, "kind": kind}
            if source:
                rec["source"] = source
            # Persist FIRST, swap after -- the rule every other mutation here
            # follows. Appending in memory first meant a refused write (a
            # second store owning the file) left the record live in this
            # session and on no disk.
            self._append(rec)
            self._records.append(rec)
            return rec

    def remember(self, text: str, kind: str = "user_fact", source: str | None = None) -> dict:
        """add() with update semantics -- the ``remember`` tool's entry point.

        Exact duplicate -> returns the existing record (idempotent). A record
        whose CONTENT words overlap the new text heavily (Jaccard >= 0.5,
        stopwords excluded) is treated as the same fact restated ("my dog's
        name is Rex" -> "my dog's name is Bruno") and SUPERSEDED: replaced,
        not left beside the new one to confuse retrieval. Every record gets a
        date stamp so memories can be audited.

        Fact-shaped texts supersede on their ATTRIBUTE, so a changed value
        ("User's car is a red hatchback." -> "... a silver van.") replaces the
        old record however much the wording moved, while a shared value across
        different attributes ("brother"/"sister" both named Leo) does not.

        HONEST LIMIT: texts that are not fact-shaped fall back to lexical
        overlap at a deliberately high bar, so an unusual restatement coexists
        with the old record rather than risking the wrong deletion.

        Text over MAX_MEMORY_CHARS is REFUSED here rather than filed: this is
        the one owner both doors go through, and a document-sized record is
        re-tokenized on every retrieval and echoed into a context that cannot
        hold it."""
        text = " ".join(str(text).split())
        if not text:
            raise ValueError("empty memory")
        if len(text) > MAX_MEMORY_CHARS:
            raise ValueError(
                f"memory too long: {len(text)} characters, and a memory may be "
                f"at most {MAX_MEMORY_CHARS} -- save the fact, not the document"
            )
        new_terms = _content_terms(text)
        new_key = _fact_key(text)
        with self._lock:
            # Exact-duplicate scan FIRST, over the whole store. Folded into
            # the supersede loop it exited on a key match before reaching a
            # later exact duplicate, deleting one record and minting a twin
            # (audit 2026-07-22 -- reachable on stores seeded via add()).
            for rec in self._records:
                if rec["text"].lower() == text.lower():
                    return dict(rec)
            superseded = None
            best_score = 0.0
            for rec in self._records:
                old_key = _fact_key(rec["text"])
                if new_key is not None and old_key is not None:
                    # Same relation key: replace only when that relation is
                    # single-valued. Many-valued relations (loves, is allergic
                    # to, goes running) coexist -- deleting one to keep another
                    # is the unrecoverable error.
                    if old_key == new_key and _relation_is_single_valued(new_key):
                        superseded = rec
                        break
                    continue
                old_terms = _content_terms(rec["text"])
                union = new_terms | old_terms
                if not union:
                    continue
                score = len(new_terms & old_terms) / len(union)
                # Rank every candidate: the FIRST record over the bar is not
                # necessarily the closest one.
                if score >= _SUPERSEDE_MIN and score > best_score:
                    best_score, superseded = score, rec
            rec = {
                "id": (max((r["id"] for r in self._records), default=0) + 1),
                "text": text,
                "kind": kind,
                "date": time.strftime("%Y-%m-%d"),
            }
            if source:
                rec["source"] = source
            if superseded is not None:
                rec["superseded"] = superseded["text"]
                # Persist FIRST, swap after -- the rule delete()/forget()/
                # clear() follow and _rewrite's docstring states. Mutating
                # first meant a failed write told the user the update failed
                # while the live store already carried it, silently reverting
                # at restart (review 2026-08-13).
                keep = [r for r in self._records if r is not superseded] + [rec]
                self._rewrite(keep)
                self._records = keep
            else:
                self._append(rec)  # persist first, swap after (see add())
                self._records.append(rec)
            return dict(rec)

    def delete(self, mem_id: int) -> bool:
        """Remove one memory by id. Returns False when the id doesn't exist."""
        with self._lock:
            for rec in self._records:
                if rec["id"] == mem_id:
                    keep = [r for r in self._records if r is not rec]
                    self._rewrite(keep)   # persist first: an error means nothing changed
                    self._records = keep
                    return True
            return False

    def forget(self, query: str, only_id: int | None = None) -> list[dict]:
        """Delete the ONE memory the query names; return it as a list.

        One rule: a record is a candidate when its content terms and the ask's
        are a subset one way or the other -- the ask restates the record
        ("forget that I like tea" covers "User likes tea.") or points at it
        ("forget about my dog" covers "User's dog is named Rex."). Exactly one
        candidate is deleted. Two or more raises TooBroad naming them, and
        nothing is touched.

        This is deliberately the least clever version. Four earlier revisions
        tried to decide WHICH of several matches the user meant -- by BM25
        rank, by match direction, by counting a record's terms -- and each one
        shipped a different unrecoverable delete: an ask about a sister
        removing the user's own facts, five one-term facts swept at once, two
        records with the same term set both going. Deletion has no .bak, so
        the matcher does not get to guess. Ambiguity is handed back as a
        question, and `delete(id)` is there for the answer.

        Naming a record EXACTLY resolves an ambiguity the subset rule cannot:
        nested facts ("User likes tea." inside "User likes green tea.") make
        every ask that covers one cover the other, so both spellings raised
        TooBroad and repeating either named candidate raised it again -- advice
        that could not be followed, with no delete-by-name way out. An ask whose
        terms EQUAL one record's terms is that record being named, not a pointer
        at a group, so it wins. Two records with the same terms still tie, which
        is the one case that really is ambiguous.

        A query left contentless by the noise filter ("forget everything",
        "forget everything now") matches nothing -- an unbounded wipe belongs to
        clear(), behind its own explicit ask. A SCOPED wipe keeps its subject
        ("forget everything about my dog" -> {dog}) unless several records share
        that subject, which is an ambiguity like any other.

        `only_id` restricts the delete to one of the candidates the QUERY
        already matched. It answers a TooBroad without becoming a second way in:
        an id that is not among them deletes nothing, so an id she invented --
        and she has no honest source for one outside a refusal -- cannot reach a
        record the ask never named."""
        # None coerced through str() became the literal query "None", whose
        # single term subset-matched any record containing the word "none"
        # and DELETED it (review 2026-08-13). Non-strings of every kind mean
        # "no query".
        query = query if isinstance(query, str) else ""
        if not query.strip():
            return []
        q_terms = _forget_terms(query)
        if not q_terms:
            return []
        with self._lock:
            candidates, keep = [], []
            for rec in self._records:
                # Both sides through the SAME filter. Filtering only the ask let
                # a record carrying a noise word ("User's phone is still
                # broken.") never equal the ask's terms, so quoting that record
                # word for word deleted its shorter sibling instead.
                r_terms = _forget_terms(rec["text"])
                if r_terms and (r_terms <= q_terms or q_terms <= r_terms):
                    candidates.append(rec)
                else:
                    keep.append(rec)
            if only_id is not None:
                chosen = [r for r in candidates if r["id"] == only_id]
                if not chosen:
                    return []
                keep = [r for r in self._records if r is not chosen[0]]
                candidates = chosen
            elif len(candidates) > 1:
                named_exactly = [r for r in candidates
                                 if _forget_terms(r["text"]) == q_terms]
                if len(named_exactly) == 1:
                    # Rebuild from the ORIGINAL order: appending the spared
                    # candidates to the tail reordered the file, and "the recent
                    # ones" is read off the tail.
                    keep = [r for r in self._records if r is not named_exactly[0]]
                    candidates = named_exactly
                else:
                    # Name the IDS, not just the text. Records with identical
                    # term sets cannot be told apart by restating them, so a
                    # text-only refusal was advice with no way to follow it.
                    # Bounded: this string is fed back into her context as a
                    # tool result, and 21 matches measured 728 tokens of a
                    # 1024-token window.
                    shown = candidates[:_FORGET_NAMED_MAX]
                    named = "; ".join(f"#{r['id']} {r['text']}" for r in shown)
                    more = len(candidates) - len(shown)
                    tail = f" (and {more} more)" if more > 0 else ""
                    raise self.TooBroad(
                        f"{len(candidates)} {FORGET_PENDING_MARK} -- say the one you mean "
                        f"word for word, or give its id: {named}{tail}"
                    )
            if candidates:
                # Write BEFORE the in-memory swap. Mutating first meant a failed
                # write reported an error to the user while the record was
                # already gone from the live store -- and the next successful
                # write made that loss permanent.
                self._rewrite(keep)
                self._records = keep
        return candidates

    def clear(self) -> int:
        """Remove ALL memories; returns how many were dropped."""
        with self._lock:
            n = len(self._records)
            self._rewrite([])          # persist first: an error means nothing changed
            self._records = []
            return n

    def _rewrite(self, records: list[dict] | None = None) -> None:
        """Rewrite the JSONL after a mutation (call with the lock held). At
        hundreds of records this is instant and keeps the file inspectable.
        atomic_write_text adds fsync-before-rename (power loss can't leave a
        truncated file). backup=False ON PURPOSE: every _rewrite caller is a
        delete/supersede/clear, and a .bak would keep a full pre-delete copy
        on disk -- "clear my memories" must actually clear (privacy
        regression caught by the 2026-07-17 re-audit). Any .bak an earlier
        build left behind is scrubbed for the same reason.

        Pass the records to WRITE so a caller can persist before swapping them
        in: mutating first meant a failed write told the user the delete had
        failed while the live store had already dropped the record."""
        if records is None:
            records = self._records
        # A rewrite REPLACES the file, so it is the mutation that destroys
        # another store's appends outright -- refuse before writing, never
        # after (see _refuse_if_another_writer).
        self._refuse_if_another_writer()
        content = "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)
        atomic_write_text(self.file, content, backup=False)
        self.file.with_suffix(self.file.suffix + ".bak").unlink(missing_ok=True)
        self._seen_stamp = self._file_stamp()

    def all(self) -> list[dict]:
        with self._lock:
            return list(self._records)

    def search(self, query: str, k: int = 3) -> list[dict]:
        """BM25 (k1=1.5, b=0.75). Returns up to k records, best first; records
        sharing no term with the query never match. Reads take the same lock
        as writers: scoring zips _records with per-record term vectors, and a
        concurrent supersede between the two passes would misalign the pairs.

        Scoring runs on CONTENT terms only. A query whose every word is a
        stopword ("is that right?") shares nothing identifying with any record
        and correctly retrieves nothing, rather than ranking on "is".

        A k of 0 or less asks for nothing and gets nothing."""
        # The slice below counts from the END on a negative k, which would
        # answer a request for nothing with every record but the worst-ranked.
        if k <= 0:
            return []
        q_terms = _content_term_list(query)
        with self._lock:
            if not q_terms or not self._records:
                return []
            docs = [_content_term_list(r["text"]) for r in self._records]
            n = len(docs)
            # A record with no content terms contributes length 0; the floor
            # keeps the length-normalization divisor defined.
            avg_len = max(1.0, sum(len(d) for d in docs) / n)
            df: dict[str, int] = {}
            for d in docs:
                for t in set(d):
                    df[t] = df.get(t, 0) + 1
            k1, b = 1.5, 0.75
            # The query, widened by the synonym clusters: asked-for words at
            # full weight, their synonyms dampened, so a fact stored in other
            # words is REACHABLE without outranking a literal hit.
            weighted_terms = _expand_query_terms(q_terms)
            scored = []
            for rec, d in zip(self._records, docs):
                score = 0.0
                for t, w in weighted_terms:
                    tf = d.count(t)
                    if not tf:
                        continue  # a synonym no record uses simply scores nothing
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    score += w * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
                if score > 0:
                    scored.append((score, rec))
        # Ties break toward the NEWEST record (ids are monotonic). Coexisting
        # plain values ("mood is happy" then "mood is sad") produce identical
        # term vectors, and a stable sort on score alone surfaced the STALE
        # one first -- the coexist default is only honest if the current value
        # outranks the superseded-in-spirit one.
        scored.sort(key=lambda s: (-s[0], -s[1]["id"]))
        return [rec for _, rec in scored[:k]]

    def render_context(
        self, query: str, tokenizer, max_ids: int = 128, k: int = 3,
        focus_query: str | None = None,
    ) -> str:
        """Top-k matches as a system-prompt block, trimmed to a token budget.
        Empty string when nothing relevant — never pad her context with noise.

        `focus_query` is the CURRENT ask when `query` widens to the recent
        thread. Its best match takes the first slot, so prior-turn chatter
        cannot fill all k and evict the answer to the question being asked."""
        hits: list[dict] = []
        if focus_query and k > 0:
            hits = self.search(focus_query, k=1)
        seen = {r["id"] for r in hits}
        for rec in self.search(query, k=k):
            if len(hits) >= k:
                break
            if rec["id"] not in seen:
                hits.append(rec)
                seen.add(rec["id"])
        if not hits:
            return ""
        lines = ["Things you remember:"]
        used = len(tokenizer.encode(lines[0], add_special_tokens=False))
        for rec in hits:
            line = f"- {rec['text']}"
            cost = len(tokenizer.encode(line, add_special_tokens=False)) + 1
            if used + cost > max_ids:
                break
            lines.append(line)
            used += cost
        return "\n".join(lines) if len(lines) > 1 else ""
