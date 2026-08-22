"""
Byte Pair Encoding (BPE) Tokenizer - Built from Scratch

This is a REAL tokenizer that learns patterns from your data.
No external dependencies. No faking it.

How BPE works:
1. Start with characters as initial vocabulary
2. Count all adjacent pairs in the data
3. Merge the most frequent pair into a new token
4. Repeat until desired vocab size

This creates subword tokens that capture common patterns in YOUR specific data.
"""

import json
import logging
import heapq
import random
import re
import threading
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional

from .pretokenize import (
    PRETOKENIZER_V1,
    PRETOKENIZER_V2,
    normalize_pretokenizer,
    pretokenize_v2_with_specials,
    sanitize_for_utf8,
)

logger = logging.getLogger(__name__)


class BPETokenizer:
    """
    A real Byte Pair Encoding tokenizer.

    Learns subword patterns directly from your training data.
    No external dependencies - pure Python implementation.
    Automatically uses Rust backend (enigma_bpe) for encode/decode
    when available (~6x faster).
    """

    # Rust backend availability (class-level, checked once)
    _rust_available: bool | None = None

    def __init__(self, vocab_file: Optional[Path] = None, pretokenizer: Optional[str] = None):
        # Pre-tokenizer version. Defaults to v1 and is overwritten by
        # load() from the vocab file's "pretokenizer" key (absent = v1),
        # so existing vocabs keep exactly the behaviour they were
        # trained with. Pass pretokenizer="v2" to train a new one.
        self.pretokenizer_version: str = normalize_pretokenizer(pretokenizer)

        # Special tokens with reserved IDs
        self.special_tokens = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "<sep>": 4,
            "<mask>": 5,
            "<Q>": 6,
            "<A>": 7,
            "<USER>": 8,
            "<BOT>": 9,
            "<think>": 10,
            "</think>": 11,
            "<search>": 12,  # Stage B-1: inline search-query start
            "</search>": 13,  # Stage B-1: inline search-query end
        }

        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.bos_token = "<s>"
        self.unk_token = "<unk>"

        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.unk_token_id = 3
        self.think_start_id = self.special_tokens["<think>"]
        self.think_end_id = self.special_tokens["</think>"]
        # Stage B-1: None on legacy load. Rust SPECIAL_TOKENS aligned
        # to this dict in B-1b (Pass after 156z9e) so Rust train no
        # longer drops <search>/</search>.
        self.search_start_id: int | None = self.special_tokens.get("<search>")
        self.search_end_id: int | None = self.special_tokens.get("</search>")

        # Vocabulary mappings
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        # BPE merge rules (ordered list of merges)
        self.merges: list[tuple[str, str]] = []
        self.merge_ranks: dict[tuple[str, str], int] = {}

        # Cache for encoding (LRU via OrderedDict)
        self.cache: OrderedDict[str, list[str]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_cap = self._compute_cache_cap()

        # UTF-8 byte-level mode: encode text as UTF-8 bytes
        # before BPE, so any Unicode character can be represented
        # as a sequence of byte tokens (matches GPT-2 / Llama / Qwen3).
        # Default ON (Tok-2) so fresh tokenizers handle Unicode
        # without emitting <unk>. Existing vocab files preserve their
        # stored flag through load() (see line ~631), so legacy
        # tokenizers are not silently re-interpreted.
        self.use_utf8_bytes: bool = True

        # Rust backend (None = not loaded yet)
        self._rust_backend = None

        if vocab_file and vocab_file.exists():
            self.load(vocab_file)  # may overwrite use_utf8_bytes
        else:
            self._init_base_vocab()

        self.vocab_size = len(self.token_to_id)

    def _sync_special_ids(self):
        """Rebuild convenience IDs from special_tokens map."""
        self.pad_token_id = self.special_tokens.get("<pad>", 0)
        self.bos_token_id = self.special_tokens.get("<s>", 1)
        self.eos_token_id = self.special_tokens.get("</s>", 2)
        self.unk_token_id = self.special_tokens.get("<unk>", 3)
        self.think_start_id = self.special_tokens.get("<think>", 10)
        self.think_end_id = self.special_tokens.get("</think>", 11)
        # Stage B-1: None on legacy vocabs (no numeric fallback — would
        # alias a learned merge ID).
        self.search_start_id = self.special_tokens.get("<search>")
        self.search_end_id = self.special_tokens.get("</search>")

    @staticmethod
    def _compute_cache_cap() -> int:
        """Cache cap scaled to RAM via InferenceMemoryBudget (S804)."""
        try:
            from .hardware_detection import InferenceMemoryBudget

            return InferenceMemoryBudget().bpe_cache_cap
        except Exception:
            return 10000  # safe fallback

    def _init_base_vocab(self):
        """Initialize with special tokens and base characters."""
        self.token_to_id = dict(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}

        next_id = len(self.special_tokens)

        # Add all printable ASCII as base tokens
        for i in range(256):
            char = bytes([i]).decode("latin-1")
            if char not in self.token_to_id:
                self.token_to_id[char] = next_id
                self.id_to_token[next_id] = char
                next_id += 1

    @staticmethod
    def _text_to_bytes(text: str) -> str:
        """Convert text to a UTF-8 byte string using latin-1 mapping.

        Each byte of the UTF-8 encoding maps 1:1 to a latin-1 char
        (since the base vocab covers all 256 byte values).  This
        lets the existing BPE operate on byte-level tokens.

        ``errors="replace"`` because an unpaired surrogate (which JSON
        legally carries) is not encodable and used to raise
        UnicodeEncodeError from here -- a 500 on the serve render path
        and a dead worker in pretokenize.  ``sanitize_for_utf8`` makes
        the same substitution at the top of ``encode``, so the Python
        and Rust paths produce identical bytes.
        """
        return text.encode("utf-8", errors="replace").decode("latin-1")

    @staticmethod
    def _bytes_to_text(byte_string: str) -> str:
        """Convert a latin-1 byte string back to UTF-8 text.

        Reverses ``_text_to_bytes``.  Invalid byte sequences
        are replaced rather than raising.
        """
        return byte_string.encode("latin-1").decode("utf-8", errors="replace")

    def train(
        self,
        texts: list[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
        verbose: bool = True,
        on_progress: Callable[[int, str], None] | None = None,
        pretokenizer: Optional[str] = None,
    ):
        """
        Train BPE on a list of texts.

        When ``use_utf8_bytes`` is True, texts are converted to UTF-8
        byte sequences before training so that merges operate on bytes.

        Args:
            texts: List of training texts
            vocab_size: Target vocabulary size
            min_frequency: Minimum pair frequency to merge
            verbose: Print progress
            on_progress: Optional callback(percent, message) for live updates
            pretokenizer: Pre-tokenizer version ("v1" or "v2").  When
                given it overrides the instance setting for this run and
                is persisted by save().  Defaults to the instance value.
        """
        if pretokenizer is not None:
            self.pretokenizer_version = normalize_pretokenizer(pretokenizer)
        is_v2 = self.pretokenizer_version == PRETOKENIZER_V2

        if verbose:
            logger.info("Training BPE tokenizer...")
            logger.info(f"Target vocab size: {vocab_size}")
            logger.info(f"Training texts: {len(texts)}")

        if not texts:
            raise ValueError("Cannot train BPE tokenizer on empty text list")

        if min_frequency < 1:
            raise ValueError(f"min_frequency must be >= 1, got {min_frequency}")

        # Attempt Rust fast path (R-2). The Rust backend hardcodes the v1
        # pre-tokenizer, so a v2 run MUST stay on the Python path or the
        # merges would be learned with v1 word units and silently
        # contradict the "pretokenizer": "v2" written into the file.
        if not is_v2 and self._try_rust_train(texts, vocab_size, min_frequency, on_progress):
            if verbose:
                logger.info(f"Rust BPE train complete: vocab={self.vocab_size}, merges={len(self.merges)}")
            return

        # Reset to base vocabulary
        self._init_base_vocab()
        self.merges = []
        self.merge_ranks = {}
        self.cache: OrderedDict[str, list[str]] = OrderedDict()

        # Pre-tokenize: split into words, keep track of frequencies
        word_freqs: Counter = Counter()

        for text in texts:
            # Split on whitespace and punctuation, keep the pieces
            words = self._pre_tokenize(text)
            for word in words:
                # v2 keeps whitespace-only units (a run of spaces is a
                # legitimate unit that BPE should be able to merge);
                # v1 drops them, as it always has.
                if word if is_v2 else word.strip():
                    # In UTF-8 mode, convert each word to bytes
                    if self.use_utf8_bytes:
                        word = self._text_to_bytes(word)
                    # v2 emits NO </w> end-of-word marker. Under v2 a word
                    # unit already carries its own boundary information in
                    # the attached leading space (" cat" vs "cat"), so </w>
                    # would be redundant -- and worse, it would collide with
                    # lossless decode: v1 decode turns </w> back into a
                    # space, which is exactly the whitespace re-insertion
                    # v2 must not do. Dropping it makes v2 decode a plain
                    # join. The round-trip tests are the arbiter.
                    word_tuple = tuple(word) if is_v2 else tuple(word) + ("</w>",)
                    word_freqs[word_tuple] += 1

        # Add end-of-word marker to vocab (v1 only -- see above)
        if not is_v2 and "</w>" not in self.token_to_id:
            next_id = len(self.token_to_id)
            self.token_to_id["</w>"] = next_id
            self.id_to_token[next_id] = "</w>"

        if verbose:
            logger.info(f"Unique words: {len(word_freqs)}")
            logger.info(f"Base vocab: {len(self.token_to_id)}")

        # BPE training loop
        num_merges = vocab_size - len(self.token_to_id)

        # Build initial pair counts and a reverse index from
        # pairs → set of words containing that pair.  This lets
        # us do incremental updates instead of rescanning all words
        # on every merge (O(affected) instead of O(all_words)).
        pair_freqs: Counter = Counter()
        pair_to_words: dict[tuple[str, str], set[tuple[str, ...]]] = {}

        for word_tuple, freq in word_freqs.items():
            for j in range(len(word_tuple) - 1):
                pair = (word_tuple[j], word_tuple[j + 1])
                pair_freqs[pair] += freq
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(word_tuple)

        if on_progress:
            on_progress(0, f"Starting {num_merges:,} merges ({len(word_freqs):,} unique words)")

        # Max-heap for O(log N) best-pair lookup instead of O(N)
        # linear scan.  Python heapq is a min-heap, so we negate
        # frequencies.  Stale entries are skipped via lazy deletion
        # (check current pair_freqs before accepting a pop).
        heap: list[tuple[int, tuple[str, str]]] = []
        for pair, freq in pair_freqs.items():
            heapq.heappush(heap, (-freq, pair))

        for i in range(num_merges):
            # Pop stale entries until we find one matching current freq
            best_pair = None
            best_freq = 0
            while heap:
                neg_freq, candidate = heapq.heappop(heap)
                current = pair_freqs.get(candidate, 0)
                if current > 0 and current == -neg_freq:
                    best_pair = candidate
                    best_freq = current
                    break
                # Stale entry — skip

            if best_pair is None:
                if verbose:
                    logger.info(f"No more pairs to merge at iteration {i}")
                break

            if best_freq < min_frequency:
                if verbose:
                    logger.info(f"Stopping: best pair frequency {best_freq} < {min_frequency}")
                break

            # Merge the pair
            new_token = best_pair[0] + best_pair[1]

            # Add to vocabulary
            if new_token not in self.token_to_id:
                next_id = len(self.token_to_id)
                self.token_to_id[new_token] = next_id
                self.id_to_token[next_id] = new_token

            # Record the merge
            self.merges.append(best_pair)
            self.merge_ranks[best_pair] = len(self.merges) - 1

            # Incremental update: only process words that contain
            # the merged pair, update pair_freqs and pair_to_words
            affected = pair_to_words.pop(best_pair, set())
            del pair_freqs[best_pair]

            for old_word in affected:
                freq = word_freqs.get(old_word, 0)
                if freq == 0:
                    continue

                # Build new word with merged pair
                new_word_list: list[str] = []
                j = 0
                while j < len(old_word):
                    if j < len(old_word) - 1 and old_word[j] == best_pair[0] and old_word[j + 1] == best_pair[1]:
                        new_word_list.append(new_token)
                        j += 2
                    else:
                        new_word_list.append(old_word[j])
                        j += 1
                new_word = tuple(new_word_list)

                if new_word == old_word:
                    continue

                # Remove old word's pair contributions
                for j in range(len(old_word) - 1):
                    p = (old_word[j], old_word[j + 1])
                    if p != best_pair and p in pair_freqs:
                        pair_freqs[p] -= freq
                        if pair_freqs[p] <= 0:
                            del pair_freqs[p]
                        else:
                            # S712: Repush decremented pair so it
                            # stays visible in the heap.  Without
                            # this, the old heap entry (higher freq)
                            # is stale and gets skipped, making the
                            # pair invisible despite positive freq.
                            heapq.heappush(heap, (-pair_freqs[p], p))
                        if p in pair_to_words:
                            pair_to_words[p].discard(old_word)
                            if not pair_to_words[p]:
                                del pair_to_words[p]

                # Add new word's pair contributions
                for j in range(len(new_word) - 1):
                    p = (new_word[j], new_word[j + 1])
                    pair_freqs[p] = pair_freqs.get(p, 0) + freq
                    if p not in pair_to_words:
                        pair_to_words[p] = set()
                    pair_to_words[p].add(new_word)
                    # Push updated freq onto heap (stale entries
                    # are handled by lazy deletion on pop)
                    heapq.heappush(heap, (-pair_freqs[p], p))

                # Update word_freqs
                del word_freqs[old_word]
                word_freqs[new_word] = freq

            if verbose and (i + 1) % 500 == 0:
                logger.info(
                    f"Merge {i + 1}/{num_merges}: '{best_pair[0]}' + '{best_pair[1]}' -> '{new_token}' (freq: {best_freq})"
                )
            # Yield GIL every merge so the GUI stays responsive.
            # Early merges can take 10+ seconds each with millions
            # of unique words — yielding every 100 was too rare.
            time.sleep(0)
            if (i + 1) % 100 == 0:
                if on_progress:
                    pct = int((i + 1) / num_merges * 100)
                    on_progress(pct, f"Merge {i + 1:,}/{num_merges:,} (freq: {best_freq:,})")

        self.vocab_size = len(self.token_to_id)

        if verbose:
            logger.info(f"Final vocab size: {self.vocab_size}")
            logger.info(f"Total merges: {len(self.merges)}")
            logger.info("Training complete!")

    def _pre_tokenize(self, text: str) -> list[str]:
        """Split text into words for BPE processing."""
        if self.pretokenizer_version == PRETOKENIZER_V2:
            return self._pre_tokenize_v2(text)

        result = []

        # Handle special markers first - extract them before regex processing
        # Pattern to match Q:, A:, User:, Bot:, Human:, Assistant:,
        # and reasoning tags <think>, </think>
        special_pattern = r"(<think>|</think>|<search>|</search>|(?<![A-Za-z])Q:|(?<![A-Za-z])A:|(?<![A-Za-z])User:|(?<![A-Za-z])Bot:|(?<![A-Za-z])Human:|(?<![A-Za-z])Assistant:)"

        parts = re.split(special_pattern, text)

        for part in parts:
            if not part:
                continue

            # Map markers to special tokens
            if part in ("<think>", "</think>", "<search>", "</search>"):
                result.append(part)
            elif part == "Q:":
                result.append("<Q>")
            elif part == "A:":
                result.append("<A>")
            elif part in ("User:", "Human:"):
                result.append("<USER>")
            elif part in ("Bot:", "Assistant:"):
                result.append("<BOT>")
            else:
                # Regular text - split into words
                # This regex splits on spaces and separates punctuation
                pattern = r"'s|'t|'re|'ve|'m|'ll|'d|\w+|[^\s\w]+"
                words = re.findall(pattern, part)

                for word in words:
                    word = word.strip()
                    if word and word not in ["", " "]:
                        result.append(word)

        return result

    def _pre_tokenize_v2(self, text: str) -> list[str]:
        """v2 split, delegated WHOLE to the shared module.

        Both the carve-out set (every multi-char ``special_tokens`` key)
        and the split logic live in ``pretokenize_v2_with_specials`` --
        this class used to carve a hardcoded 4-tag tuple while the
        runtime carved its full ``special_tokens``, which made the same
        text tokenize to different ids per class (2026-07-19 audit,
        HIGH-1). Sharing the function AND the input source closes it.

        v1's "Q:" -> <Q> rewrites remain v1-only: they are lossy, and
        v2's contract is lossless round-tripping -- but the literal
        "<Q>" TAG (like every registered special literal) now maps to
        its control id, exactly as the runtime class always did.
        """
        return pretokenize_v2_with_specials(text, self.special_tokens)

    def _tokenize_word(self, word: str, dropout: float = 0.0) -> list[str]:
        """Tokenize a single word using learned BPE merges.

        Uses a heap-based approach: O(n log n) instead of the naive O(n²)
        linear scan.  All adjacent pairs are inserted with their merge rank;
        each iteration pops the lowest-rank (most common) pair, merges it,
        and pushes any new pairs formed by the merge.

        Args:
            dropout: Probability of skipping each merge (BPE-Dropout).
                0.0 = canonical tokenization.  0.1 = skip 10% of merges
                (training-time subword regularization).
        """
        dropout = max(0.0, min(1.0, dropout))
        # Skip cache when dropout is active (results are stochastic)
        if dropout <= 0.0:
            with self._cache_lock:
                if word in self.cache:
                    self.cache.move_to_end(word)
                    return list(self.cache[word])

        # Check if it's a special token
        if word in self.special_tokens:
            return [word]

        # v2 carries no </w> marker (see train()): the word unit is looked
        # up and split as-is, which is what makes decode a plain join.
        if self.pretokenizer_version == PRETOKENIZER_V2:
            if word in self.token_to_id:
                return [word]
            tokens = list(word)
        else:
            # Check if the whole word is in vocabulary (common word)
            word_with_end = word + "</w>"
            if word_with_end in self.token_to_id:
                return [word_with_end]

            # Start with characters
            tokens = list(word) + ["</w>"]

        if len(tokens) > 1:
            # Build a heap of (rank, insertion_order, position, left, right)
            # insertion_order breaks ties deterministically and lets us
            # lazily invalidate stale entries (position may have been merged).
            heap: list[tuple[int, int, int, str, str]] = []
            seq_counter = 0
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_ranks:
                    heapq.heappush(heap, (self.merge_ranks[pair], seq_counter, i, tokens[i], tokens[i + 1]))
                    seq_counter += 1

            while heap and len(tokens) > 1:
                _rank, _seq, pos, left, right = heapq.heappop(heap)
                # Validate: the pair at 'pos' must still be (left, right)
                if pos >= len(tokens) - 1 or tokens[pos] != left or tokens[pos + 1] != right:
                    continue  # stale entry — skip

                # BPE-Dropout: randomly skip this merge
                if dropout > 0.0 and random.random() < dropout:
                    continue

                # Merge the pair in-place
                merged = left + right
                tokens[pos] = merged
                del tokens[pos + 1]

                # S713: Rebuild heap from current tokens.
                # Position shift from del tokens[pos+1] makes
                # all entries after pos stale.  Token lists are
                # short (per-word), so full rebuild is cheap and
                # guarantees no pairs are orphaned.
                heap = []
                for k in range(len(tokens) - 1):
                    new_pair = (tokens[k], tokens[k + 1])
                    if new_pair in self.merge_ranks:
                        seq_counter += 1
                        heapq.heappush(heap, (self.merge_ranks[new_pair], seq_counter, k, tokens[k], tokens[k + 1]))

        # Cache result (with gradual eviction to avoid GC spikes)
        # Only cache deterministic (non-dropout) tokenizations
        if dropout <= 0.0:
            with self._cache_lock:
                cap = self._cache_cap
                while len(self.cache) >= cap:
                    self.cache.popitem(last=False)
                self.cache[word] = tokens
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True, dropout: float = 0.0) -> list[int]:
        """Encode text to token IDs.

        In UTF-8 byte mode, text is first converted to UTF-8 bytes
        mapped as latin-1 characters, so any Unicode can be encoded.

        Args:
            dropout: BPE-Dropout probability (0.0 = off, 0.1 typical
                for training).  Randomly skips merges to produce
                diverse tokenizations as data augmentation.
        """
        # Before the Rust seam: the backend raises on an unpaired surrogate
        # exactly as strict encode does, so both paths must see the same
        # sanitized text or they disagree byte-for-byte.
        text = sanitize_for_utf8(text)

        # Rust fast path (no dropout support in Rust backend)
        if self._rust_backend is not None and dropout == 0.0:
            return self._rust_backend.encode(text, add_special_tokens)

        ids = []

        if add_special_tokens:
            ids.append(self.bos_token_id)

        # Pre-tokenize on original text, then convert per word
        words = self._pre_tokenize(text)

        for word in words:
            if not word:
                continue

            # Convert each word to byte-level representation
            if self.use_utf8_bytes:
                word = self._text_to_bytes(word)

            # Handle special tokens directly
            if word in self.special_tokens:
                ids.append(self.special_tokens[word])
                continue

            # Tokenize word with BPE
            tokens = self._tokenize_word(word, dropout=dropout)

            for token in tokens:
                if token in self.token_to_id:
                    ids.append(self.token_to_id[token])
                else:
                    # Unknown - encode character by character
                    for char in token:
                        if char in self.token_to_id:
                            ids.append(self.token_to_id[char])
                        else:
                            ids.append(self.unk_token_id)

        if add_special_tokens:
            ids.append(self.eos_token_id)

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.

        In UTF-8 byte mode, the latin-1 byte string is converted
        back to proper UTF-8 text after de-tokenization.
        """
        # Rust fast path
        if self._rust_backend is not None:
            return self._rust_backend.decode(ids, skip_special_tokens)

        if self.pretokenizer_version == PRETOKENIZER_V2:
            return self._decode_v2(ids, skip_special_tokens)

        tokens = []

        for idx in ids:
            if idx in self.id_to_token:
                token = self.id_to_token[idx]

                if token in self.special_tokens:
                    if skip_special_tokens:
                        # Drop all special tokens entirely
                        continue
                    # When keeping special tokens, show readable forms
                    readable = {
                        "<Q>": "Q: ",
                        "<A>": "A: ",
                        "<USER>": "User: ",
                        "<BOT>": "Bot: ",
                    }
                    tokens.append(readable.get(token, token))
                    continue

                tokens.append(token)

        # Join and clean up
        text = "".join(tokens)

        # Remove end-of-word markers and add spaces
        text = text.replace("</w>", " ")

        # Clean up multiple spaces
        text = re.sub(r" +", " ", text)

        # Convert byte-level representation back to UTF-8 text
        if self.use_utf8_bytes:
            text = self._bytes_to_text(text)

        return text.strip()

    def _decode_v2(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """v2 decode: plain lossless concatenation.

        Deliberately does NOT do any of v1's whitespace surgery -- no
        </w>-to-space substitution, no ``re.sub(r" +", " ")`` collapsing,
        no trailing ``.strip()``. Under v2 the whitespace is already
        inside the token pieces, so every one of those steps would
        corrupt the text instead of repairing it.
        """
        pieces: list[str] = []
        for idx in ids:
            token = self.id_to_token.get(idx)
            if token is None:
                continue
            if token in self.special_tokens:
                if skip_special_tokens:
                    continue
                # Emit the literal tag; no "readable form" rewriting,
                # which would break the round-trip.
                pieces.append(token)
                continue
            pieces.append(token)

        text = "".join(pieces)

        if self.use_utf8_bytes:
            text = self._bytes_to_text(text)

        return text

    def save(self, path: Path | str):
        """Save tokenizer to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "token_to_id": self.token_to_id,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
            "use_utf8_bytes": self.use_utf8_bytes,
        }

        # Only stamp the key for non-v1. Absence means v1, so v1 files
        # keep exactly the shape every existing consumer (Rust backend,
        # exporters, the live vocab) already expects.
        if self.pretokenizer_version != PRETOKENIZER_V1:
            data["pretokenizer"] = self.pretokenizer_version

        from enigma_engine.core.safe_save import atomic_write_json

        atomic_write_json(path, data)

        logger.info(f"Saved tokenizer to {path}")

    def load(self, path: Path):
        """Load tokenizer from file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self.token_to_id = data["token_to_id"]
        # Build reverse mapping — values in token_to_id are ints (token IDs)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        self.merges = [tuple(m) for m in data["merges"]]
        self.merge_ranks = {tuple(m): i for i, m in enumerate(self.merges)}

        if "special_tokens" in data:
            self.special_tokens = data["special_tokens"]

        self._sync_special_ids()
        self.use_utf8_bytes = data.get("use_utf8_bytes", False)
        # Absent key = v1: legacy vocabs (including the live served one)
        # must never be re-interpreted as v2.
        self.pretokenizer_version = normalize_pretokenizer(data.get("pretokenizer"))

        self.vocab_size = len(self.token_to_id)
        self.cache: OrderedDict[str, list[str]] = OrderedDict()

        # Try loading Rust backend for fast encode/decode. v2 vocabs
        # additionally require a backend that declares V2_SUPPORTED --
        # a stale built extension would silently split them with v1
        # rules (the guard lives inside _try_load_rust_backend).
        self._try_load_rust_backend(str(path))

        logger.info(
            f"Loaded tokenizer from {path} (vocab: {self.vocab_size}, merges: {len(self.merges)}"
            f"{', rust backend' if self._rust_backend else ''})"
        )

    def _try_load_rust_backend(self, path: str):
        """Attempt to load the Rust BPE backend for faster encode/decode."""
        if BPETokenizer._rust_available is False:
            return
        try:
            import enigma_bpe
            from enigma_bpe import RustBPETokenizer

            BPETokenizer._rust_available = True
            if self.pretokenizer_version != PRETOKENIZER_V1 and not getattr(
                enigma_bpe, "V2_SUPPORTED", False
            ):
                # Older built extension: v1-only splitting. Fail SOFT to
                # Python rather than silently mis-splitting a v2 vocab.
                logger.debug("Rust BPE backend lacks v2 support; using Python for this vocab")
                return
            backend = RustBPETokenizer()
            backend.load(path)
            if getattr(backend, "pretokenizer", PRETOKENIZER_V1) != self.pretokenizer_version:
                # Defense in depth: the backend read the same file, so any
                # disagreement means a version-parsing bug -- do not trust it.
                logger.warning("Rust backend pretokenizer disagrees with vocab file; using Python")
                self._rust_backend = None
                return
            self._rust_backend = backend
        except ImportError:
            BPETokenizer._rust_available = False
            logger.debug("Rust BPE backend not available, using Python")
        except Exception as exc:
            logger.warning(f"Rust BPE backend failed to load: {exc}")
            self._rust_backend = None

    def _try_rust_train(
        self,
        texts: list[str],
        vocab_size: int,
        min_frequency: int,
        on_progress: Callable[[int, str], None] | None,
    ) -> bool:
        """Attempt BPE training via Rust backend.

        Returns True if Rust training succeeded and Python state was
        synced.  Returns False if Rust is unavailable or fails, so the
        caller should fall through to the Python implementation.
        """
        if BPETokenizer._rust_available is False:
            return False
        try:
            from enigma_bpe import RustBPETokenizer

            BPETokenizer._rust_available = True
        except ImportError:
            BPETokenizer._rust_available = False
            return False

        try:
            backend = RustBPETokenizer()
            result = backend.train(
                texts,
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                use_utf8_bytes=self.use_utf8_bytes,
                callback=on_progress,
            )

            # Sync Python state from Rust result
            self.token_to_id = dict(result["token_to_id"])
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
            self.merges = [tuple(m) for m in result["merges"]]
            self.merge_ranks = {m: i for i, m in enumerate(self.merges)}
            self.special_tokens = dict(result["special_tokens"])
            self.vocab_size = len(self.token_to_id)
            self.cache: OrderedDict[str, list[str]] = OrderedDict()
            self._rust_backend = backend
            return True
        except Exception as exc:
            logger.warning(f"Rust BPE train failed, falling back to Python: {exc}")
            return False

    def __call__(
        self,
        text: str,
        return_tensors: str = None,
        padding: bool = None,
        truncation: bool = None,
        max_length: int = None,
        add_special_tokens: bool = True,
    ) -> dict[str, Any]:
        """Tokenize text (HuggingFace-compatible interface)."""
        ids = self.encode(text, add_special_tokens=add_special_tokens)

        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]

        if padding and max_length and len(ids) < max_length:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))

        if return_tensors == "pt":
            import torch

            return {"input_ids": torch.tensor([ids])}

        return {"input_ids": ids}

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> dict[str, int]:
        return self.token_to_id.copy()


def train_bpe_tokenizer(
    data_paths: list[str], vocab_size: int = 32000, output_path: Optional[str] = None
) -> BPETokenizer:
    """
    Train a BPE tokenizer on data files.

    Args:
        data_paths: List of text file paths
        vocab_size: Target vocabulary size
        output_path: Where to save the tokenizer

    Returns:
        Trained BPETokenizer
    """
    # Load all texts
    texts = []
    for path in data_paths:
        p = Path(path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                texts.append(f.read())
            logger.info(f"Loaded {p} ({len(texts[-1])} chars)")

    if not texts:
        raise ValueError("No training data found!")

    # Train tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(texts, vocab_size=vocab_size, verbose=True)

    # Save if path provided
    if output_path:
        tokenizer.save(Path(output_path))

    return tokenizer
