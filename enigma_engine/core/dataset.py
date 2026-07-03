"""
Pre-training dataset utilities — download, process, and clean text data.
===========================================================================

Provides tools for preparing text data for language model pre-training:
- process_text_corpus: Load and clean text from files or directories
- clean_text: Normalize whitespace, remove control chars
- estimate_token_count: Quick token count estimate
- download_dataset: Download known datasets via huggingface_hub
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)

# Maximum file size (bytes) for individual files during corpus processing.
# Files exceeding this limit are skipped to prevent OOM crashes.
# 100 GB accommodates large pre-training corpora (e.g. Wikipedia dumps,
# multi-language Common Crawl shards).
MAX_FILE_SIZE: int = 100_000_000_000  # 100 GB


# Files above this threshold are read in chunks instead of all at once.
# This limits peak memory (avoids 2x file-size during clean_text) and
# yields the GIL between chunks so the GUI event loop stays responsive.
# Values scaled to RAM via InferenceMemoryBudget (S806).
def _get_stream_constants() -> tuple[int, int]:
    """Return (stream_threshold, chunk_chars) scaled to available RAM."""
    try:
        from .hardware_detection import InferenceMemoryBudget

        budget = InferenceMemoryBudget()
        return budget.dataset_stream_threshold, budget.dataset_chunk_chars
    except Exception:
        return 500_000_000, 200_000_000  # safe fallback


_STREAM_THRESHOLD: int
_CHUNK_READ_CHARS: int
_STREAM_THRESHOLD, _CHUNK_READ_CHARS = _get_stream_constants()

# =========================================================================
# Known datasets — registry of downloadable pre-training corpora
# =========================================================================

KNOWN_DATASETS: dict[str, dict] = {
    "tinystories": {
        "name": "TinyStories",
        "description": ("Short children's stories (~500M tokens). Good for initial pre-training validation."),
        "repo_id": "roneneldan/TinyStories",
        "repo_type": "dataset",
        "size_estimate": "~2 GB uncompressed text",
    },
    "tinystories-instruct": {
        "name": "TinyStories Instruct",
        "description": ("Instruction-following version of TinyStories. Stories with prompts and completions."),
        "repo_id": "roneneldan/TinyStories",
        "repo_type": "dataset",
        "size_estimate": "~2 GB uncompressed text",
    },
}


# =========================================================================
# Text cleaning
# =========================================================================


def clean_text(text: str) -> str:
    """Clean raw text for pre-training.

    - Remove null bytes and control characters (except newline/tab)
    - Normalize excessive whitespace runs
    - Strip trailing whitespace per line
    - Collapse 3+ consecutive blank lines to 2

    Args:
        text: Raw text to clean.

    Returns:
        Cleaned text.
    """
    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove control chars except \n \r \t
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize \r\n to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse 3+ consecutive newlines to 2 (one blank line max)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# =========================================================================
# Token count estimation
# =========================================================================


def estimate_token_count(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate token count from text length.

    Uses a rough heuristic: ~4 characters per token for English text.
    This matches typical BPE tokenizer behavior.

    Args:
        text: Text to estimate.
        chars_per_token: Average characters per token (default 4.0).

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


# =========================================================================
# Text corpus processing
# =========================================================================


def process_text_corpus(
    source: Path | str,
    *,
    text_key: str = "text",
) -> str:
    """Load and clean text data from a file or directory.

    Handles multiple formats:
    - .txt files: read as-is and clean
    - .jsonl files: extract text from each JSON object using text_key
    - Directories: process all .txt and .jsonl files found

    Args:
        source: Path to a file or directory of text data.
        text_key: JSON key to extract text from in JSONL files.

    Returns:
        Cleaned text ready for pre-training.
    """
    source = Path(source)

    if source.is_dir():
        return _process_directory(source, text_key=text_key)
    elif source.is_file():
        return _process_file(source, text_key=text_key)
    else:
        logger.warning("Source path does not exist: %s", source)
        return ""


def _process_file(path: Path, *, text_key: str = "text") -> str:
    """Process a single file."""
    # Guard against oversized files that would OOM
    try:
        file_size = path.stat().st_size
    except OSError:
        file_size = 0
    if file_size > MAX_FILE_SIZE:
        logger.warning(
            "Skipping %s (%d MB) — exceeds MAX_FILE_SIZE (%d MB)",
            path,
            file_size // 1_000_000,
            MAX_FILE_SIZE // 1_000_000,
        )
        return ""

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return _process_jsonl(path, text_key=text_key)
    elif suffix == ".json":
        return _process_json(path, text_key=text_key)
    else:
        # For large text files, stream-read in chunks to keep
        # the GUI responsive and limit peak memory.
        if file_size > _STREAM_THRESHOLD:
            return "\n\n".join(_chunked_read_text(path))
        # Default: treat as plain text
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Cannot read %s: %s", path, exc)
            return ""
        return clean_text(text)


def _chunked_read_text(
    path: Path,
    *,
    on_progress: "Callable[[int, str], None] | None" = None,
) -> list[str]:
    """Read and clean a large text file in chunks.

    Reads the file in ~200 MB pieces to avoid holding the GIL
    for minutes on a multi-GB string operation.  Each chunk is
    cleaned individually, and ``time.sleep(0)`` yields the GIL
    between chunks so the GUI event loop remains responsive.

    Args:
        on_progress: Optional callback(pct, message) for loading progress.

    Returns:
        List of cleaned text chunks (~200 MB each).
    """
    import time

    parts: list[str] = []
    file_size = path.stat().st_size

    with open(path, encoding="utf-8", errors="replace") as f:
        remainder = ""
        bytes_read = 0
        while True:
            raw = f.read(_CHUNK_READ_CHARS)
            if not raw:
                # Process any leftover text
                if remainder:
                    cleaned = clean_text(remainder)
                    if cleaned:
                        parts.append(cleaned)
                break

            raw = remainder + raw
            bytes_read = f.buffer.tell()

            # Split at last newline to avoid cutting mid-line
            last_nl = raw.rfind("\n")
            if last_nl == -1:
                remainder = raw
                continue

            chunk = raw[: last_nl + 1]
            remainder = raw[last_nl + 1 :]

            cleaned = clean_text(chunk)
            if cleaned:
                parts.append(cleaned)

            pct = min(100, int(bytes_read / max(1, file_size) * 100))
            logger.info(
                "Reading %s: %d%% (%d MB / %d MB)", path.name, pct, bytes_read // 1_000_000, file_size // 1_000_000
            )
            if on_progress is not None:
                on_progress(
                    pct, f"Reading {path.name}: {pct}% ({bytes_read // 1_000_000} / {file_size // 1_000_000} MB)"
                )

            # Yield GIL so GUI thread can process events
            time.sleep(0)

    return parts


def load_text_chunks(
    source: Path | str,
    *,
    text_key: str = "text",
    on_progress: "Callable[[int, str], None] | None" = None,
) -> list[str]:
    """Load text from a file as a list of cleaned chunks.

    For large text files (>500 MB), returns multiple ~200 MB chunks
    instead of one massive string.  This prevents MemoryError on
    multi-GB corpora (e.g. 16 GB combined.txt).

    For small files, directories, and JSONL, falls back to
    :func:`process_text_corpus` and wraps the result in a list.

    Args:
        source: Path to a text file, JSONL file, or directory.
        text_key: JSON key for JSONL files.

    Returns:
        List of cleaned text chunks.  Empty list on failure.
    """
    source = Path(source)
    if source.is_file():
        try:
            file_size = source.stat().st_size
        except OSError:
            file_size = 0
        if file_size > MAX_FILE_SIZE:
            logger.warning("Skipping %s (%d MB) — exceeds MAX_FILE_SIZE", source, file_size // 1_000_000)
            return []
        suffix = source.suffix.lower()
        if suffix not in (".jsonl", ".json") and file_size > _STREAM_THRESHOLD:
            return _chunked_read_text(source, on_progress=on_progress)
    # Small files, directories, JSONL — use existing function
    result = process_text_corpus(source, text_key=text_key)
    return [result] if result else []


def iter_text_chunks(
    source: Path | str,
    *,
    text_key: str = "text",
    on_progress: "Callable[[int, str], None] | None" = None,
) -> "Iterator[str]":
    """Yield cleaned text chunks from a file or directory one at a time.

    Unlike :func:`load_text_chunks`, this never holds more than one chunk
    (~200 MB) in RAM.  For multi-GB corpora this prevents MemoryError.

    For large text files (>500 MB), yields ~200 MB chunks via
    :func:`_chunked_read_text`.  For directories, recursively iterates
    files and yields cleaned text from each.  For small files and JSONL,
    yields the single cleaned result.

    Args:
        source: Path to a text file, JSONL file, or directory.
        text_key: JSON key for JSONL files.
        on_progress: Optional callback(pct, message) for loading progress.

    Yields:
        Cleaned text chunks.  Empty generator on failure.
    """
    source = Path(source)

    if source.is_dir():
        yield from _iter_directory(source, text_key=text_key, on_progress=on_progress)
        return

    if not source.is_file():
        logger.warning("Source path does not exist: %s", source)
        return

    try:
        file_size = source.stat().st_size
    except OSError:
        file_size = 0

    if file_size > MAX_FILE_SIZE:
        logger.warning("Skipping %s (%d MB) — exceeds MAX_FILE_SIZE", source, file_size // 1_000_000)
        return

    suffix = source.suffix.lower()
    if suffix not in (".jsonl", ".json") and file_size > _STREAM_THRESHOLD:
        yield from _iter_chunked_read_text(source, on_progress=on_progress)
        return

    # Small files, JSONL — process and yield single result
    result = _process_file(source, text_key=text_key)
    if result:
        yield result


def _iter_directory(
    dir_path: Path,
    *,
    text_key: str = "text",
    on_progress: "Callable[[int, str], None] | None" = None,
) -> "Iterator[str]":
    """Recursively yield cleaned text from all text files in a directory.

    Walks subdirectories so ``data/pretrain/`` with sub-folders like
    ``fineweb_edu/``, ``gutenberg/``, etc. are handled without needing
    to merge into a single ``combined.txt`` first.
    """
    _TEXT_SUFFIXES = {".txt", ".jsonl", ".json"}
    files: list[Path] = sorted(f for f in dir_path.rglob("*") if f.is_file() and f.suffix.lower() in _TEXT_SUFFIXES)
    if not files:
        return

    total = len(files)
    for i, f in enumerate(files):
        try:
            file_size = f.stat().st_size
        except OSError:
            continue
        if file_size > MAX_FILE_SIZE:
            logger.warning("Skipping %s (%d MB) — exceeds MAX_FILE_SIZE", f, file_size // 1_000_000)
            continue

        suffix = f.suffix.lower()
        if suffix not in (".jsonl", ".json") and file_size > _STREAM_THRESHOLD:
            yield from _iter_chunked_read_text(f)
        else:
            chunk = _process_file(f, text_key=text_key)
            if chunk:
                yield chunk

        if on_progress is not None:
            pct = int((i + 1) / total * 100)
            on_progress(pct, f"Loading files: {i + 1}/{total} ({pct}%)")


def _iter_chunked_read_text(
    path: Path,
    *,
    on_progress: "Callable[[int, str], None] | None" = None,
) -> "Iterator[str]":
    """Yield cleaned ~200 MB text chunks from a large file.

    Generator version of :func:`_chunked_read_text` — yields instead
    of accumulating, so peak RAM stays at one chunk.
    """
    import time

    file_size = path.stat().st_size

    with open(path, encoding="utf-8", errors="replace") as f:
        remainder = ""
        bytes_read = 0
        while True:
            raw = f.read(_CHUNK_READ_CHARS)
            if not raw:
                if remainder:
                    cleaned = clean_text(remainder)
                    if cleaned:
                        yield cleaned
                break

            raw = remainder + raw
            bytes_read = f.buffer.tell()

            last_nl = raw.rfind("\n")
            if last_nl == -1:
                remainder = raw
                continue

            chunk = raw[: last_nl + 1]
            remainder = raw[last_nl + 1 :]

            cleaned = clean_text(chunk)
            if cleaned:
                yield cleaned

            pct = min(100, int(bytes_read / max(1, file_size) * 100))
            logger.info(
                "Reading %s: %d%% (%d MB / %d MB)", path.name, pct, bytes_read // 1_000_000, file_size // 1_000_000
            )
            if on_progress is not None:
                on_progress(
                    pct, f"Reading {path.name}: {pct}% ({bytes_read // 1_000_000} / {file_size // 1_000_000} MB)"
                )

            time.sleep(0)


# Maximum number of JSONL entries to accumulate in memory
_MAX_JSONL_ENTRIES = 1_000_000


def _process_jsonl(path: Path, *, text_key: str = "text") -> str:
    """Extract text from a JSONL file."""
    texts: list[str] = []
    skipped = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if isinstance(obj, dict):
                    val = obj.get(text_key, "")
                    if isinstance(val, str) and val.strip():
                        texts.append(val.strip())
                        if len(texts) >= _MAX_JSONL_ENTRIES:
                            logger.warning(
                                "%s: hit %d entry cap at line %d — remaining lines skipped",
                                path.name,
                                _MAX_JSONL_ENTRIES,
                                line_num,
                            )
                            break
    except OSError as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return ""

    if skipped:
        logger.warning(
            "%s: skipped %d malformed JSONL line(s) out of %d total",
            path.name,
            skipped,
            skipped + len(texts),
        )

    return clean_text("\n\n".join(texts))


def _process_json(path: Path, *, text_key: str = "text") -> str:
    """Extract text from a JSON file (array of objects)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return ""

    if isinstance(data, list):
        texts = []
        for obj in data:
            if isinstance(obj, dict):
                val = obj.get(text_key, "")
                if isinstance(val, str) and val.strip():
                    texts.append(val.strip())
        return clean_text("\n\n".join(texts))

    return ""


def _process_directory(
    dir_path: Path,
    *,
    text_key: str = "text",
) -> str:
    """Process all text/JSONL files in a directory."""
    parts: list[str] = []

    # Collect and sort for deterministic order (recursive like _iter_directory)
    _TEXT_SUFFIXES = {".txt", ".jsonl", ".json"}
    files = sorted(f for f in dir_path.rglob("*") if f.is_file() and f.suffix.lower() in _TEXT_SUFFIXES)
    for f in files:
        suffix = f.suffix.lower()
        if suffix in _TEXT_SUFFIXES:
            chunk = _process_file(f, text_key=text_key)
            if chunk:
                parts.append(chunk)

    return "\n\n".join(parts)


# =========================================================================
# Dataset download
# =========================================================================


def download_dataset(
    name: str,
    dest_dir: Path | str,
    *,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> Path:
    """Download a known pre-training dataset.

    Uses huggingface_hub (already a project dependency) to download
    dataset files.  Returns the path to the downloaded directory.

    Args:
        name: Key from KNOWN_DATASETS (e.g. "tinystories").
        dest_dir: Directory to store downloaded files.
        progress_fn: Optional callback for progress messages.

    Returns:
        Path to the directory containing downloaded files.

    Raises:
        ValueError: If name is not in KNOWN_DATASETS.
        ImportError: If huggingface_hub is not installed.
        RuntimeError: If download fails.
    """
    if name not in KNOWN_DATASETS:
        available = ", ".join(KNOWN_DATASETS.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")

    info = KNOWN_DATASETS[name]
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if progress_fn:
        progress_fn(f"Downloading {info['name']}...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for dataset download. Install with: pip install huggingface-hub"
        ) from None

    try:
        local_dir = dest_dir / name
        snapshot_download(
            repo_id=info["repo_id"],
            repo_type=info.get("repo_type", "dataset"),
            local_dir=str(local_dir),
            ignore_patterns=["*.md", "*.py", ".gitattributes"],
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to download {info['name']}: {exc}") from exc

    if progress_fn:
        progress_fn(f"Downloaded {info['name']} to {local_dir}")

    return local_dir
