"""
================================================================================
🔤 FORGE TOKENIZER - TEXT ↔ NUMBERS CONVERTER
================================================================================

The TRANSLATOR between human text and numbers the AI understands!
Converts sentences into sequences of integers for the neural network.

📍 FILE: enigma_engine/core/tokenizer.py
🏷️ TYPE: Text Tokenization
🎯 MAIN FUNCTION: get_tokenizer()
🎯 MAIN CLASSES: SimpleTokenizer, TiktokenWrapper, AdvancedBPETokenizer

┌─────────────────────────────────────────────────────────────────────────────┐
│  TOKENIZATION FLOW:                                                         │
│                                                                             │
│  "Hello world!" → [Tokenizer] → [15496, 995, 0]                            │
│                                                                             │
│  [15496, 995, 0] → [Tokenizer] → "Hello world!"                            │
└─────────────────────────────────────────────────────────────────────────────┘

📊 TOKENIZER HIERARCHY (best to worst):
    1. AdvancedBPETokenizer - Byte-level BPE, handles any input, learns from data
    2. SimpleTokenizer     - Basic character + common words (NO dependencies!)

🏷️ SPECIAL TOKENS:
    • <pad>    - Padding (ID: 0)
    • <s>      - Start of sequence (ID: 1)
    • </s>     - End of sequence (ID: 2)
    • <unk>    - Unknown token (ID: 3)
    • <think>  - Start of reasoning block (ID: 10)
    • </think> - End of reasoning block (ID: 11)

🔗 CONNECTED FILES:
    → USES:      enigma_engine/vocab_model/ (vocabulary files)
    ← USED BY:   enigma_engine/core/model.py (needs vocab_size)
    ← USED BY:   enigma_engine/core/inference.py (encode/decode text)
    ← USED BY:   enigma_engine/core/training.py (prepare training data)

📖 USAGE:
    from enigma_engine.core.tokenizer import get_tokenizer

    # Auto-select best available
    tokenizer = get_tokenizer()

    # Or specify type
    tokenizer = get_tokenizer("bpe")      # Advanced BPE (the live tokenizer)
    tokenizer = get_tokenizer("simple")   # Simple fallback

    # Encode/Decode
    ids = tokenizer.encode("Hello world")
    text = tokenizer.decode(ids)

📁 VOCAB LOCATION: enigma_engine/vocab_model/

📖 SEE ALSO:
    • enigma_engine/core/bpe_tokenizer.py      - BPE implementation
    • enigma_engine/core/advanced_tokenizer.py - Advanced tokenizer
"""

import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Vocabulary directory (shared across all tokenizers)
VOCAB_DIR = Path(__file__).resolve().parent.parent / "vocab_model"


# =============================================================================
# 📋 TOKENIZER PROTOCOL - Type-Safe Interface
# =============================================================================
# This protocol defines the contract that ALL tokenizers must follow.
# Using Protocol enables static type checking without requiring inheritance.


@runtime_checkable
class TokenizerProtocol(Protocol):
    """
    Protocol defining the interface all tokenizers must implement.

    📖 WHY THIS EXISTS:
    - Enables type checking across the codebase
    - Allows any tokenizer (custom, HuggingFace, tiktoken) to be used
    - runtime_checkable allows isinstance() checks at runtime

    📐 USAGE:
        def process_text(tokenizer: TokenizerProtocol, text: str) -> List[int]:
            return tokenizer.encode(text)
    """

    vocab_size: int
    eos_token_id: int
    bos_token_id: int
    pad_token_id: int

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Convert text to token IDs."""
        ...

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text."""
        ...


# =============================================================================
# 🛠️ TOKENIZER UTILITIES - Helper Functions
# =============================================================================
# These functions provide a unified interface for tokenizer operations,
# handling the differences between tokenizer implementations.


def encode_text(
    tokenizer: Any, text: str, add_special_tokens: bool = True, max_length: Optional[int] = None, truncate: bool = True
) -> list[int]:
    """
    Encode text using any tokenizer with a unified interface.

    📖 HANDLES:
    - Enigma AI Engine tokenizers (SimpleTokenizer, AdvancedBPETokenizer)
    - HuggingFace tokenizers (transformers AutoTokenizer)
    - tiktoken tokenizers

    Args:
        tokenizer: Any tokenizer instance
        text: Text to encode
        add_special_tokens: Whether to add BOS/EOS tokens
        max_length: Optional maximum length (truncates if exceeded)
        truncate: Whether to truncate to max_length

    Returns:
        List of token IDs
    """
    # Try direct encode method (Enigma AI Engine tokenizers, HuggingFace)
    if hasattr(tokenizer, "encode"):
        try:
            ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        except TypeError:
            # Some tokenizers don't accept add_special_tokens
            ids = tokenizer.encode(text)
    # Try callable interface (some HuggingFace tokenizers)
    elif callable(tokenizer):
        result = tokenizer(text, add_special_tokens=add_special_tokens)
        ids = result.get("input_ids", result) if isinstance(result, dict) else result
    else:
        raise TypeError(f"Tokenizer {type(tokenizer)} has no encode method")

    # Handle nested lists (batch encoding)
    if ids and isinstance(ids, list) and isinstance(ids[0], list):
        ids = ids[0]

    # Handle tensor returns
    if hasattr(ids, "tolist"):
        ids = ids.tolist()

    # Truncate if needed
    if max_length and truncate and len(ids) > max_length:
        ids = ids[:max_length]

    return ids


def decode_tokens(tokenizer: Any, ids: list[int], skip_special_tokens: bool = True) -> str:
    """
    Decode token IDs using any tokenizer with a unified interface.

    Args:
        tokenizer: Any tokenizer instance
        ids: List of token IDs to decode
        skip_special_tokens: Whether to skip special tokens in output

    Returns:
        Decoded text string
    """
    if hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        except TypeError:
            # Some tokenizers don't accept skip_special_tokens
            return tokenizer.decode(ids)
    else:
        raise TypeError(f"Tokenizer {type(tokenizer)} has no decode method")


def get_vocab_size(tokenizer: Any) -> int:
    """
    Get vocabulary size from any tokenizer.

    Args:
        tokenizer: Any tokenizer instance

    Returns:
        Vocabulary size as integer
    """
    if hasattr(tokenizer, "vocab_size"):
        size = tokenizer.vocab_size
        return size() if callable(size) else size
    if hasattr(tokenizer, "get_vocab_size"):
        return tokenizer.get_vocab_size()
    if hasattr(tokenizer, "__len__"):
        return len(tokenizer)
    if hasattr(tokenizer, "vocab"):
        return len(tokenizer.vocab)
    raise AttributeError(f"Cannot determine vocab_size for {type(tokenizer)}")


def get_special_token_ids(tokenizer: Any) -> dict[str, int]:
    """
    Get special token IDs from any tokenizer.

    Returns dict with keys: 'pad', 'bos', 'eos', 'unk',
    'think_start', 'think_end'
    """
    result = {}

    # Standard attribute names
    for name, attrs in [
        ("pad", ["pad_token_id", "pad_id"]),
        ("bos", ["bos_token_id", "bos_id", "start_token_id"]),
        ("eos", ["eos_token_id", "eos_id", "end_token_id"]),
        ("unk", ["unk_token_id", "unk_id"]),
        ("think_start", ["think_start_id"]),
        ("think_end", ["think_end_id"]),
    ]:
        for attr in attrs:
            if hasattr(tokenizer, attr):
                val = getattr(tokenizer, attr)
                if val is not None:
                    result[name] = val
                    break
        # Default fallbacks
        if name not in result:
            defaults = {
                "pad": 0,
                "bos": 1,
                "eos": 2,
                "unk": 3,
                "think_start": 4,
                "think_end": 5,
            }
            fallback = defaults.get(name, -1)
            logger.warning("Tokenizer missing '%s' attribute, using default %d", name, fallback)
            result[name] = fallback

    return result


# =============================================================================
# 🔤 SIMPLE TOKENIZER - Always Works, No Dependencies!
# =============================================================================
# This is the FALLBACK tokenizer that works without any external libraries.
# It's simple but reliable - perfect for bootstrapping or when tiktoken fails.


class SimpleTokenizer:
    """
    Lightweight character-level tokenizer.

    📖 WHAT THIS DOES:
    Converts text to numbers and back. Simple and reliable!

    📐 HOW IT WORKS:
    1. Has a vocabulary: {"a": 0, "b": 1, "the": 50, ...}
    2. encode(): Split text into tokens, look up their IDs
    3. decode(): Look up IDs to get tokens, join them together

    💡 TOKENIZATION STRATEGY:
    - First tries to match whole WORDS (like "the", "hello")
    - Falls back to individual CHARACTERS if word not in vocab
    - This is a hybrid word+char approach

    📐 EXAMPLE:
        >>> tok = SimpleTokenizer()
        >>> tok.encode("hello world")
        [1, 145, 32, 119, 111, 114, 108, 100, 2]
        #  ↑ <s>  "hello"  space + characters   ↑ </s>

    🔗 CONNECTS TO:
      ← Used by get_tokenizer() as fallback
      ← Used when no trained tokenizer is available
    """

    def __init__(self, vocab_file: Optional[Path] = None):
        """
        Initialize tokenizer.

        Args:
            vocab_file: Optional path to saved vocabulary JSON
        """
        # ─────────────────────────────────────────────────────────────────────
        # SPECIAL TOKENS: Reserved tokens with fixed IDs
        # ─────────────────────────────────────────────────────────────────────
        self.pad_token = "<pad>"  # Padding for batching
        self.eos_token = "</s>"  # End of sequence
        self.unk_token = "<unk>"  # Unknown token (fallback)
        self.bos_token = "<s>"  # Beginning of sequence

        # Special token IDs (MUST be fixed for model compatibility)
        self.special_tokens = {
            "<pad>": 0,  # Padding - used to make sequences same length
            "<s>": 1,  # Start - marks beginning of text
            "</s>": 2,  # End - marks end of text
            "<unk>": 3,  # Unknown - used for characters not in vocab
            "<think>": 4,  # Start of reasoning block
            "</think>": 5,  # End of reasoning block
            "<search>": 6,  # AutoResearch-2 Stage B-1: start of inline search query
            "</search>": 7,  # AutoResearch-2 Stage B-1: end of inline search query
        }

        # Convenient ID lookups
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.think_start_id = 4
        self.think_end_id = 5
        # Stage B-1: search-token accessors. None for legacy vocabs that
        # were saved before <search>/</search> existed; callers must
        # check for None and treat as "feature unavailable on this model".
        self.search_start_id: int | None = 6
        self.search_end_id: int | None = 7

        # ─────────────────────────────────────────────────────────────────────
        # LOAD OR CREATE VOCABULARY
        # ─────────────────────────────────────────────────────────────────────
        if vocab_file and Path(vocab_file).exists():
            self._load_vocab(vocab_file)
        else:
            self._create_default_vocab()

        self.vocab_size = len(self.token_to_id)

    def _create_default_vocab(self):
        """
        Create a basic character + common word vocabulary.

        📖 VOCABULARY STRUCTURE:
        IDs 0-3: Special tokens (<pad>, <s>, </s>, <unk>)
        IDs 4-98: Printable ASCII characters (space, a-z, A-Z, 0-9, etc.)
        IDs 99+: Common English words for efficiency
        """
        # Start with special tokens
        self.token_to_id = dict(self.special_tokens)
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}

        # ─────────────────────────────────────────────────────────────────────
        # ADD ASCII CHARACTERS (codes 32-126)
        # ─────────────────────────────────────────────────────────────────────
        # This gives us: space, !"#$%&'()*+,-./ 0-9 :;<=>?@ A-Z [\]^_` a-z {|}~
        for c in range(32, 127):
            token = chr(c)
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

        # ─────────────────────────────────────────────────────────────────────
        # ADD COMMON WORDS (for efficiency)
        # ─────────────────────────────────────────────────────────────────────
        # Without these, "the" would be 3 tokens: "t", "h", "e"
        # With these, "the" is 1 token - 3x more efficient!
        common = [
            "the",
            "is",
            "a",
            "to",
            "of",
            "and",
            "in",
            "that",
            "it",
            "for",
            "you",
            "was",
            "with",
            "on",
            "are",
            "be",
            "have",
            "this",
            "will",
            "your",
            "from",
            "or",
            "by",
            "not",
            "but",
            "what",
            "all",
            "were",
            "we",
            "when",
            "can",
            "there",
            "an",
            "which",
            "their",
            "if",
            "has",
            "more",
            "also",
            "do",
            "no",
            "my",
            "one",
            "so",
            "our",
            "they",
            "been",
            "would",
            "how",
            "her",
            "him",
            "his",
            "its",
            "may",
            "new",
            "now",
            "old",
            "see",
            "way",
            "who",
            "did",
            "get",
            "just",
            "know",
            "take",
            "come",
            "could",
            "good",
            "some",
            "them",
            "very",
            "after",
            "most",
            "make",
            "should",
            "still",
            "over",
            "such",
            "much",
            "then",
            "first",
            "any",
            "only",
            "other",
            "into",
            "year",
            "hello",
            "hi",
            "yes",
            "please",
            "thank",
            "thanks",
            "sorry",
            "help",
            "AI",
            "I",
            "You",
            "What",
            "How",
            "Why",
            "When",
            "Where",
            "Q:",
            "A:",
            "User:",
            "Bot:",
        ]

        for word in common:
            if word not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[word] = idx
                self.id_to_token[idx] = word

    def _sync_special_ids(self):
        """Rebuild convenience IDs from special_tokens map."""
        self.pad_token_id = self.special_tokens.get("<pad>", 0)
        self.bos_token_id = self.special_tokens.get("<s>", 1)
        self.eos_token_id = self.special_tokens.get("</s>", 2)
        self.unk_token_id = self.special_tokens.get("<unk>", 3)
        self.think_start_id = self.special_tokens.get("<think>", 4)
        self.think_end_id = self.special_tokens.get("</think>", 5)
        # Stage B-1: None on legacy vocab (no fallback ID — would alias
        # an unrelated token). Stage B-2 generation hook checks this.
        self.search_start_id = self.special_tokens.get("<search>")
        self.search_end_id = self.special_tokens.get("</search>")

    def _load_vocab(self, vocab_file: Path):
        """Load vocabulary from JSON file."""
        with open(vocab_file, encoding="utf-8") as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        # Rebuild self.special_tokens from the loaded vocab: keep only
        # tokens that actually exist on disk, with disk's IDs.  This
        # prevents in-memory defaults (e.g. Stage B-1's <search>=6) from
        # leaking into legacy vocabs that lack the token, which would
        # alias an unrelated learned ID.
        loaded_specials: dict[str, int] = {}
        for tok in self.special_tokens:
            if tok in self.token_to_id:
                loaded_specials[tok] = self.token_to_id[tok]
        self.special_tokens = loaded_specials
        self._sync_special_ids()

    def save_vocab(self, vocab_file: Path) -> None:
        """Save vocabulary to JSON file."""
        Path(vocab_file).parent.mkdir(parents=True, exist_ok=True)
        from enigma_engine.core.safe_save import atomic_write_json

        atomic_write_json(vocab_file, self.token_to_id)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Encode text to token IDs.

        📖 ENCODING PROCESS:
        1. Add <s> (start token) if requested
        2. Split text into words
        3. For each word:
           - If word in vocab → use word token
           - Else → use character tokens
        4. Add spaces between words
        5. Add </s> (end token) if requested

        Args:
            text: Input string to encode
            add_special_tokens: Whether to add <s> and </s>

        Returns:
            List of token IDs
        """
        ids = []

        # Start with beginning-of-sequence token
        if add_special_tokens:
            ids.append(self.bos_token_id)

        # ─────────────────────────────────────────────────────────────────────
        # PRE-TOKENIZE: Separate multi-char special tokens from adjacent text
        # so they are encoded as single tokens, not character-by-character.
        # ─────────────────────────────────────────────────────────────────────
        for token in self.special_tokens:
            if len(token) > 1 and token in text:
                text = text.replace(token, f" {token} ")

        # ─────────────────────────────────────────────────────────────────────
        # TOKENIZE: Try words first, fall back to characters
        # ─────────────────────────────────────────────────────────────────────
        words = text.split()
        for i, word in enumerate(words):
            if word in self.token_to_id:
                # Word is in vocabulary - use single token
                ids.append(self.token_to_id[word])
            else:
                # Word not in vocab - tokenize character by character
                for char in word:
                    if char in self.token_to_id:
                        ids.append(self.token_to_id[char])
                    else:
                        # Character not in vocab - use unknown token
                        ids.append(self.unk_token_id)

            # Add space token between words (not after last word)
            if i < len(words) - 1 and " " in self.token_to_id:
                ids.append(self.token_to_id[" "])

        # End with end-of-sequence token
        if add_special_tokens:
            ids.append(self.eos_token_id)

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        📖 DECODING PROCESS:
        1. Look up each ID in id_to_token dictionary
        2. Skip special tokens if requested
        3. Join tokens intelligently (handle spaces)

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip <s>, </s>, <pad>, <unk>

        Returns:
            Decoded text string
        """
        tokens = []

        for idx in ids:
            if idx in self.id_to_token:
                token = self.id_to_token[idx]
                # Skip special tokens if requested
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)

        # ─────────────────────────────────────────────────────────────────────
        # JOIN TOKENS INTELLIGENTLY
        # ─────────────────────────────────────────────────────────────────────
        # We need to add spaces between WORDS but not between CHARACTERS
        result = []
        for i, token in enumerate(tokens):
            if token == " ":
                result.append(" ")
            elif len(token) == 1:
                # Single character - don't add extra space
                result.append(token)
            else:
                # Word token - add space before if needed
                if result and result[-1] != " ":
                    result.append(" ")
                result.append(token)

        return "".join(result).strip()

    def __call__(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True,
    ) -> dict[str, Any]:
        """
        Tokenize text (HuggingFace-compatible interface).

        📖 WHAT THIS DOES:
        Same as encode(), but returns a dictionary and optionally
        handles padding, truncation, and tensor conversion.
        This makes the tokenizer work like HuggingFace tokenizers!

        Args:
            text: Input text to tokenize
            return_tensors: "pt" for PyTorch tensors, None for lists
            padding: Pad to max_length
            truncation: Truncate to max_length
            max_length: Maximum sequence length
            add_special_tokens: Add <s> and </s>

        Returns:
            Dictionary with 'input_ids' key
        """
        ids = self.encode(text, add_special_tokens=add_special_tokens)

        # Truncate if too long
        if truncation and max_length and len(ids) > max_length:
            ids = ids[:max_length]

        # Pad if too short
        if padding and max_length and len(ids) < max_length:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))

        # Convert to PyTorch tensor if requested
        if return_tensors == "pt":
            import torch

            return {"input_ids": torch.tensor([ids])}

        return {"input_ids": ids}

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def get_vocab(self) -> dict[str, int]:
        """Return copy of vocabulary dictionary."""
        return self.token_to_id.copy()


# =============================================================================
# Tiktoken Tokenizer (Fast GPT-style tokenizer)
# =============================================================================


class TiktokenWrapper:
    """Wrapper for tiktoken (GPT-style fast tokenizer)."""

    def __init__(self, encoding: str = "cl100k_base"):
        import tiktoken

        self.enc = tiktoken.get_encoding(encoding)
        # Reserve IDs above the real vocab range so they never collide
        # with actual tiktoken token IDs (0 .. n_vocab-1).
        base = self.enc.n_vocab
        self.pad_token_id = base
        self.bos_token_id = base + 1
        self.eos_token_id = base + 2
        self.unk_token_id = base + 3
        self.think_start_id = base + 4
        self.think_end_id = base + 5
        self.vocab_size = base + 6  # real tokens + 6 special

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs."""
        ids = self.enc.encode(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        if skip_special_tokens:
            # Filter out special tokens (all reserved IDs above real vocab)
            special = {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.unk_token_id,
                self.think_start_id,
                self.think_end_id,
            }
            ids = [i for i in ids if i not in special]
        return self.enc.decode(ids)

    def __call__(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
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


# =============================================================================
# Tokenizer Loading Functions
# =============================================================================

# Tokenizer cache for reuse - maps (tokenizer_type, vocab_path) to instance
_tokenizer_cache: dict[tuple, Any] = {}
_tokenizer_cache_lock = threading.Lock()


def get_tokenizer(tokenizer_type: str = "auto", vocab_path: Optional[str | Path] = None, use_cache: bool = True) -> Any:
    """
    Get the best available tokenizer.

    Args:
        tokenizer_type: Type of tokenizer to load
            - "auto": Best available (tiktoken > bpe)
            - "tiktoken": Fast GPT-style tokenizer (requires tiktoken)
            - "bpe": Advanced BPE tokenizer (the live model's tokenizer)
            - "advanced": Alias for "bpe"
            - "simple": Simple character tokenizer (explicit only)
        vocab_path: Optional path to vocabulary file
        use_cache: Whether to cache and reuse tokenizer instances (default: True)

    Returns:
        Tokenizer instance
    """
    vocab_path = Path(vocab_path) if vocab_path else VOCAB_DIR

    # Check cache first
    cache_key = (tokenizer_type, str(vocab_path))
    if use_cache:
        with _tokenizer_cache_lock:
            if cache_key in _tokenizer_cache:
                logger.debug(f"Using cached tokenizer: {tokenizer_type}")
                return _tokenizer_cache[cache_key]

    def _cache_and_return(tok):
        """Helper to cache tokenizer before returning."""
        if use_cache:
            with _tokenizer_cache_lock:
                _tokenizer_cache[cache_key] = tok
        return tok

    # Try tiktoken first (fastest)
    if tokenizer_type in ("auto", "tiktoken"):
        try:
            tok = TiktokenWrapper()
            logger.info("Loaded tiktoken tokenizer (cl100k_base)")
            return _cache_and_return(tok)
        except ImportError:
            if tokenizer_type == "tiktoken":
                logger.error("tiktoken not available. Install with: pip install tiktoken")
                raise
            # If auto, continue to next best option
            logger.debug("tiktoken not available, trying BPE...")

    # Try Advanced BPE (best custom tokenizer)
    if tokenizer_type in ("auto", "bpe", "advanced"):
        try:
            from .advanced_tokenizer import AdvancedBPETokenizer

            # Check for saved vocabulary
            bpe_vocab = vocab_path / "bpe_vocab.json" if vocab_path.is_dir() else vocab_path

            if bpe_vocab.exists():
                tok = AdvancedBPETokenizer(vocab_file=bpe_vocab)
                logger.info(f"Loaded BPE tokenizer from {bpe_vocab}")
                return _cache_and_return(tok)
            elif tokenizer_type in ("bpe", "advanced"):
                # Return untrained BPE tokenizer
                logger.warning("BPE tokenizer not trained. Use train_tokenizer() first.")
                return _cache_and_return(AdvancedBPETokenizer())
        except ImportError as e:
            logger.warning(f"Could not load BPE tokenizer: {e}")
            if tokenizer_type in ("bpe", "advanced"):
                raise

    # ("char"/"character" was a CharacterTokenizer backend whose module no
    # longer exists — the feature is honestly absent. An explicit request falls
    # through to the RuntimeError below rather than crashing on a missing import.)

    # Explicit simple tokenizer (never auto-selected)
    if tokenizer_type == "simple":
        simple_vocab = vocab_path / "simple_vocab.json" if vocab_path.is_dir() else vocab_path
        if simple_vocab.exists():
            tok = SimpleTokenizer(simple_vocab)
            logger.info(f"Loaded simple tokenizer from {simple_vocab}")
            return _cache_and_return(tok)
        tok = SimpleTokenizer()
        VOCAB_DIR.mkdir(parents=True, exist_ok=True)
        tok.save_vocab(VOCAB_DIR / "simple_vocab.json")
        logger.info("Created new simple tokenizer")
        return _cache_and_return(tok)

    # Auto mode exhausted all viable backends
    raise RuntimeError(
        "No usable tokenizer found. Install tiktoken (pip install tiktoken) "
        "or train a BPE tokenizer (python run.py --train-tokenizer)."
    )


def clear_tokenizer_cache():
    """Clear the tokenizer cache to free memory or force reload."""
    with _tokenizer_cache_lock:
        _tokenizer_cache.clear()
    logger.debug("Tokenizer cache cleared")


def train_tokenizer(
    data_paths: list[str], vocab_size: int = 32000, output_path: Optional[str] = None, tokenizer_type: str = "bpe"
) -> Any:
    """
    Train a tokenizer on text data.

    Args:
        data_paths: List of paths to training text files
        vocab_size: Target vocabulary size
        output_path: Where to save the trained tokenizer
        tokenizer_type: Type of tokenizer to train ("bpe" or "char")

    Returns:
        Trained tokenizer
    """
    # Load training data
    texts = []
    for path in data_paths:
        p = Path(path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                texts.append(f.read())
            logger.info(f"Loaded training data from {path}")

    if not texts:
        raise ValueError("No training data found!")

    logger.info(f"Training {tokenizer_type} tokenizer on {len(texts)} files...")

    if tokenizer_type in ("bpe", "advanced"):
        from .bpe_tokenizer import BPETokenizer

        tokenizer = BPETokenizer()
        tokenizer.train(texts, vocab_size=vocab_size, verbose=True)

        # Save
        save_path = Path(output_path) if output_path else VOCAB_DIR / "bpe_vocab.json"
        tokenizer.save(save_path)

        return tokenizer

    else:
        # "char"/"character" (CharacterTokenizer) was removed — the module no
        # longer exists. Fail honestly instead of crashing on a missing import.
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type!r} (only 'bpe'/'advanced' are trainable)")


# =============================================================================
# Backwards Compatibility
# =============================================================================

# Expose SimpleTokenizer as default (always available)
Tokenizer = SimpleTokenizer


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Main functions
    "get_tokenizer",
    "train_tokenizer",
    "clear_tokenizer_cache",
    # Classes
    "SimpleTokenizer",
    "TiktokenWrapper",
    "Tokenizer",
    # Constants
    "VOCAB_DIR",
]
