"""
Advanced BPE Tokenizer

Wrapper around BPETokenizer with additional features.
This provides the AdvancedBPETokenizer class expected by inference.py
"""

import json
import logging
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Union

from .pretokenize import (
    PRETOKENIZER_V1,
    PRETOKENIZER_V2,
    normalize_pretokenizer,
    pretokenize_v2_with_specials,
)

logger = logging.getLogger(__name__)

# Reserved-token names decode() strips when skip_special_tokens=True. Kept
# separate from special_tokens, which also drives encode(): adding a name
# there changes how its literal text tokenizes, while this list only affects
# decode output. Covers every reserved name a BPETokenizer-written vocab can
# carry, including the pretraining-era ids this class never encodes itself.
_RESERVED_TOKEN_NAMES = (
    "<pad>",
    "<s>",
    "</s>",
    "<unk>",
    "<sep>",
    "<mask>",
    "<Q>",
    "<A>",
    "<USER>",
    "<BOT>",
    "<think>",
    "</think>",
    "<search>",
    "</search>",
)


class AdvancedBPETokenizer:
    """
    Advanced BPE Tokenizer with additional features.

    Handles multiple tokenizer formats:
    - Standard BPE format (token_to_id, id_to_token, merges)
    - Enigma format (encoder, special_tokens)
    """

    def __init__(self, vocab_file: Optional[Path] = None, **kwargs):
        """
        Initialize the Advanced BPE Tokenizer.

        Args:
            vocab_file: Path to vocabulary JSON file
            **kwargs: Additional arguments (``pretokenizer`` selects the
                pre-tokenization version for a tokenizer built without a
                vocab file; a loaded file always wins)
        """
        # Pre-tokenizer version -- overwritten by load() from the file's
        # "pretokenizer" key. Absent key = v1, which is what keeps the
        # live served vocab on exactly its original behaviour.
        self.pretokenizer_version: str = normalize_pretokenizer(kwargs.get("pretokenizer"))

        # Special tokens with reserved IDs
        self.special_tokens = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "<think>": 4,
            "</think>": 5,
            "<search>": 6,  # Stage B-1: inline search-query start
            "</search>": 7,  # Stage B-1: inline search-query end
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
        # Stage B-1: see bpe_tokenizer for None-on-legacy rationale.
        self.search_start_id: int | None = self.special_tokens.get("<search>")
        self.search_end_id: int | None = self.special_tokens.get("</search>")

        # Vocabulary mappings
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        # BPE merge rules
        self.merges: list[tuple[str, str]] = []
        self.merge_ranks: dict[tuple[str, str], int] = {}

        # Cache for encoding (LRU via OrderedDict)
        self._cache_lock = threading.Lock()
        self.cache: OrderedDict[str, list[str]] = OrderedDict()
        self._cache_cap = self._compute_cache_cap()

        # UTF-8 byte-level mode (Tok-2): when True, words are
        # converted to UTF-8 bytes before BPE so any Unicode
        # codepoint can be encoded via the 256 base byte tokens
        # (matches GPT-2 / Llama / Qwen3). Default ON for fresh
        # tokenizers; legacy vocab files without the flag load
        # with use_utf8_bytes=False (see load()).
        self.use_utf8_bytes: bool = True

        if vocab_file and Path(vocab_file).exists():
            self.load(vocab_file)
        else:
            self._init_base_vocab()

        self.vocab_size = len(self.token_to_id)
        self._sync_decode_skip_ids()

    def _sync_decode_skip_ids(self) -> None:
        """Collect the vocab's reserved-token ids for decode() to strip.
        Includes reserved names present in the file but absent from
        special_tokens (which must stay encode-exact for the trained vocab)."""
        ids = set(self.special_tokens.values())
        for name in _RESERVED_TOKEN_NAMES:
            tid = self.token_to_id.get(name)
            if tid is not None:
                ids.add(tid)
        self._decode_skip_ids = ids

    @staticmethod
    def _compute_cache_cap() -> int:
        """Cache cap scaled to RAM via InferenceMemoryBudget (S805)."""
        try:
            from .hardware_detection import InferenceMemoryBudget

            return InferenceMemoryBudget().advanced_tok_cache_cap
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
        (the base vocab covers all 256 byte values via
        ``_init_base_vocab``), letting BPE operate on byte-level
        tokens. Mirrors ``BPETokenizer._text_to_bytes``.
        """
        return text.encode("utf-8").decode("latin-1")

    @staticmethod
    def _bytes_to_text(byte_string: str) -> str:
        """Convert a latin-1 byte string back to UTF-8 text.

        Reverses ``_text_to_bytes``. Invalid byte sequences are
        replaced rather than raising.
        """
        return byte_string.encode("latin-1").decode("utf-8", errors="replace")

    def load(self, vocab_file: Union[str, Path]) -> None:
        """Load vocabulary from file."""
        vocab_file = Path(vocab_file)

        with open(vocab_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupt vocabulary file {vocab_file}: {exc}") from exc

        # use_utf8_bytes flag: legacy files without the flag default
        # to False so an existing tokenizer trained at character-level
        # is not silently re-interpreted as byte-level (Tok-2).
        self.use_utf8_bytes = bool(data.get("use_utf8_bytes", False))

        # Pre-tokenizer version. Absent key = v1 (the live vocab has no
        # such key and must keep v1 encode/decode bit-for-bit).
        self.pretokenizer_version = normalize_pretokenizer(data.get("pretokenizer"))

        # Handle different formats
        if "encoder" in data:
            # Enigma format
            self.token_to_id = data["encoder"]
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}

            # Update special tokens from file
            if "special_tokens" in data:
                for token, idx in data["special_tokens"].items():
                    self.special_tokens[token] = idx

                # Map standard names to Enigma format
                if "[E:pad]" in data["special_tokens"]:
                    self.pad_token_id = data["special_tokens"]["[E:pad]"]
                    self.pad_token = "[E:pad]"
                if "[E:start]" in data["special_tokens"]:
                    self.bos_token_id = data["special_tokens"]["[E:start]"]
                    self.bos_token = "[E:start]"
                if "[E:end]" in data["special_tokens"]:
                    self.eos_token_id = data["special_tokens"]["[E:end]"]
                    self.eos_token = "[E:end]"
                if "[E:unk]" in data["special_tokens"]:
                    self.unk_token_id = data["special_tokens"]["[E:unk]"]
                    self.unk_token = "[E:unk]"

                # Stage B-1: re-derive search-token IDs from the loaded
                # vocab.  Legacy files written before <search> existed
                # must yield None so the generation hook (Stage B-2)
                # detects "feature unavailable on this model" instead
                # of aliasing a learned ID.
                self.search_start_id = data["special_tokens"].get("<search>")
                self.search_end_id = data["special_tokens"].get("</search>")
                if self.search_start_id is None:
                    self.special_tokens.pop("<search>", None)
                if self.search_end_id is None:
                    self.special_tokens.pop("</search>", None)

            self.vocab_size = data.get("vocab_size", len(self.token_to_id))

        elif "token_to_id" in data:
            # Standard BPE format
            self.token_to_id = data["token_to_id"]
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}

            if "merges" in data:
                self.merges = [tuple(m) for m in data["merges"]]
                self.merge_ranks = {m: i for i, m in enumerate(self.merges)}

            # v2 ONLY: adopt the FILE's full special-token set before the
            # re-sync below. The __init__ defaults carry 8 names but a
            # trainer-written vocab registers 14 (<Q>, <USER>, ...); with
            # re-sync alone the runtime carved a SUBSET of the trainer's
            # specials and the two classes tokenized tag literals to
            # different ids (HIGH-1, caught by the carve-out parity test).
            # v1 deliberately keeps the legacy subset: its output is frozen
            # by the immutability pins, quirks included.
            if self.pretokenizer_version == PRETOKENIZER_V2 and isinstance(data.get("special_tokens"), dict):
                self.special_tokens = {str(k): int(v) for k, v in data["special_tokens"].items()}

            # Re-sync EVERY special token to the LOADED vocab. The __init__
            # defaults use a compact 0..7 range (e.g. <think>=4), but a real
            # BPETokenizer-written vocab can place them elsewhere (<think>=10,
            # </think>=11; 4/5 are <sep>/<mask> there). Without this, encode()
            # would emit the stale default id while decode()/id_to_token use the
            # file's id -> control tokens stop round-tripping. Any special the
            # file lacks is dropped so encode can't emit a phantom id.
            for stok in list(self.special_tokens):
                file_id = self.token_to_id.get(stok)
                if file_id is None:
                    self.special_tokens.pop(stok, None)
                else:
                    self.special_tokens[stok] = file_id
            self.pad_token_id = self.token_to_id.get(self.pad_token, self.pad_token_id)
            self.bos_token_id = self.token_to_id.get(self.bos_token, self.bos_token_id)
            self.eos_token_id = self.token_to_id.get(self.eos_token, self.eos_token_id)
            self.unk_token_id = self.token_to_id.get(self.unk_token, self.unk_token_id)
            self.think_start_id = self.token_to_id.get("<think>")
            self.think_end_id = self.token_to_id.get("</think>")
            self.search_start_id = self.token_to_id.get("<search>")
            self.search_end_id = self.token_to_id.get("</search>")

            self.vocab_size = len(self.token_to_id)
        else:
            raise ValueError(f"Unknown tokenizer format in {vocab_file}")

        self._sync_decode_skip_ids()
        logger.info(f"Loaded tokenizer from {vocab_file} with {self.vocab_size} tokens")

    def save(self, vocab_file: Union[str, Path]) -> None:
        """Save vocabulary to file."""
        vocab_file = Path(vocab_file)

        data = {
            "version": "2.1",
            "vocab_size": self.vocab_size,
            "encoder": self.token_to_id,
            "special_tokens": self.special_tokens,
            "use_utf8_bytes": self.use_utf8_bytes,
        }

        # Absence means v1, so v1 files keep their existing shape.
        if self.pretokenizer_version != PRETOKENIZER_V1:
            data["pretokenizer"] = self.pretokenizer_version

        from enigma_engine.core.safe_save import atomic_write_json

        atomic_write_json(vocab_file, data)

    def _tokenize_word(self, word: str) -> list[str]:
        """Tokenize a single word using learned BPE merges."""
        # Check cache (LRU: move to end on hit)
        with self._cache_lock:
            if word in self.cache:
                self.cache.move_to_end(word)
                return list(self.cache[word])

        # Check if whole word is in vocabulary
        if word in self.token_to_id:
            return [word]

        # Start with characters
        tokens = list(word)

        # Apply merges in priority order (lowest rank = most common)
        while len(tokens) > 1:
            best_pair = None
            best_rank = float("inf")

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            if best_pair is None:
                break

            # Apply the merge
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Cache result (LRU eviction, cap scaled to RAM — S805)
        with self._cache_lock:
            cap = self._cache_cap
            if len(self.cache) > cap:
                evict = cap // 2
                for _ in range(evict):
                    self.cache.popitem(last=False)
            self.cache[word] = tokens
        return tokens

    def _pre_tokenize_v2(self, text: str) -> list[str]:
        """v2 split, delegated WHOLE to the shared module.

        Carve-out set and split logic both live in
        ``pretokenize_v2_with_specials`` so this runtime path and the
        trainer path (``BPETokenizer._pre_tokenize_v2``) cannot diverge
        (2026-07-19 audit, HIGH-1).
        """
        return pretokenize_v2_with_specials(text, self.special_tokens)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs using BPE merges."""
        if not text:
            return []

        ids = []

        if add_special_tokens:
            if self.bos_token_id in self.id_to_token or self.bos_token in self.token_to_id:
                ids.append(self.bos_token_id)

        # If we have merges, use BPE encoding
        if self.merges:
            if self.pretokenizer_version == PRETOKENIZER_V2:
                words = self._pre_tokenize_v2(text)
            else:
                # Build regex with special tokens as alternatives (longest first)
                # so multi-char tokens like <think> match as whole tokens
                special_alts = "|".join(
                    re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True) if len(t) > 1
                )
                base_pattern = r"'s|'t|'re|'ve|'m|'ll|'d|\w+|[^\s\w]+|\s+"
                pattern = special_alts + "|" + base_pattern if special_alts else base_pattern
                words = re.findall(pattern, text)

            for word in words:
                if not word:
                    continue
                # Check special tokens (always on the original text
                # before any byte conversion)
                if word in self.special_tokens:
                    ids.append(self.special_tokens[word])
                    continue
                # Convert to UTF-8 bytes for sub-word lookup so any
                # Unicode codepoint is representable via the 256
                # latin-1 base tokens (Tok-2). Whole-word vocab hits
                # are still possible if a multi-byte sequence was
                # learned as a merged token during training.
                if self.use_utf8_bytes:
                    word = self._text_to_bytes(word)
                # Check if whole word is in vocab
                if word in self.token_to_id:
                    ids.append(self.token_to_id[word])
                    continue
                # Apply BPE merges
                subtokens = self._tokenize_word(word)
                for st in subtokens:
                    if st in self.token_to_id:
                        ids.append(self.token_to_id[st])
                    else:
                        # Fall back to character-level for unknown subtokens
                        for char in st:
                            if char in self.token_to_id:
                                ids.append(self.token_to_id[char])
                            else:
                                ids.append(self.unk_token_id)
        else:
            # No merges loaded — byte-level fallback when enabled,
            # otherwise legacy character-level (Tok-2).
            if self.use_utf8_bytes:
                byte_str = self._text_to_bytes(text)
                for char in byte_str:
                    if char in self.token_to_id:
                        ids.append(self.token_to_id[char])
                    else:
                        ids.append(self.unk_token_id)
            else:
                for char in text:
                    if char in self.token_to_id:
                        ids.append(self.token_to_id[char])
                    else:
                        ids.append(self.unk_token_id)

        if add_special_tokens:
            if self.eos_token_id in self.id_to_token or self.eos_token in self.token_to_id:
                ids.append(self.eos_token_id)

        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.

        In UTF-8 byte mode, the latin-1 byte string assembled from
        token pieces is converted back to UTF-8 text after de-tokenization
        (Tok-2). Unknown IDs render as the unk token.

        This is already a plain lossless concatenation -- unlike
        BPETokenizer.decode it never substituted </w> for a space,
        collapsed runs of spaces, or stripped the result. That is exactly
        what v2 requires (the whitespace lives inside the token pieces),
        so no version gate is needed here; v1 and v2 share this path.
        """
        chars = []

        # Union at call time: attach_chat_tokens and callers can extend
        # special_tokens after load, and the reserved-name set covers vocab
        # ids that special_tokens deliberately omits (encode-exactness).
        special_ids = (set(self.special_tokens.values()) | self._decode_skip_ids) if skip_special_tokens else set()

        for tid in token_ids:
            if tid in special_ids:
                continue
            if tid in self.id_to_token:
                chars.append(self.id_to_token[tid])
            else:
                chars.append(self.unk_token)

        text = "".join(chars)

        if self.use_utf8_bytes:
            text = self._bytes_to_text(text)

        return text

    def encode_chat(self, messages: list[dict[str, str]], add_generation_prompt: bool = True) -> list[int]:
        """Encode a list of chat messages."""
        tokens = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if "[E:system]" in self.token_to_id:
                    tokens.append(self.token_to_id["[E:system]"])
                tokens.extend(self.encode(content))
            elif role == "user":
                if "[E:user]" in self.token_to_id:
                    tokens.append(self.token_to_id["[E:user]"])
                tokens.extend(self.encode(content))
            elif role == "assistant":
                if "[E:assistant]" in self.token_to_id:
                    tokens.append(self.token_to_id["[E:assistant]"])
                tokens.extend(self.encode(content))

        if add_generation_prompt:
            if "[E:assistant]" in self.token_to_id:
                tokens.append(self.token_to_id["[E:assistant]"])

        return tokens

    def get_vocab_stats(self) -> dict[str, Any]:
        """Get vocabulary statistics."""
        return {
            "vocab_size": self.vocab_size,
            "num_merges": len(self.merges),
            "special_tokens": list(self.special_tokens.keys()),
            "cache_size": len(self.cache),
        }

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "AdvancedBPETokenizer":
        """Load tokenizer from file."""
        return cls(vocab_file=Path(path))


__all__ = ["AdvancedBPETokenizer"]
