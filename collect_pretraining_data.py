"""
Pre-Training Data Collector for Enigma Engine
==============================================
Downloads and cleans text data from free, legal sources for pre-training.

Sources:
  - Wikipedia dump (--wiki-dump) - ALL English Wikipedia articles (~15-20 GB clean text)
  - Wikipedia API (--wiki N) - random articles via API (slower, for small runs)
  - Simple Wikipedia (--simple N) - clear explanations, basic language
  - Project Gutenberg (--books N) - public domain books, rich language (~400+ curated titles)
  - FineWeb-Edu (--fineweb N) - educational web pages from HuggingFace (requires `datasets`)
  - Stack Exchange (--stackexchange) - Q&A dumps from archive.org (requires `py7zr`)
  - Wayback Machine (--wayback N) - educational domains via Internet Archive CDX API
  - Fandom wikis (--fandom N) - fan wikis from fandom.com (games, movies, TV, books)
  - OpenWebText (--openwebtext N) - web text from Reddit links (requires `datasets`)
  - C4 (--c4 N) - cleaned Common Crawl text (requires `datasets`)
  - DCLM-Baseline (--dclm N) - model-filtered web text, beats FineWeb-Edu on MMLU (requires `datasets`)
  - FineMath (--finemath N) - step-by-step math from CommonCrawl, finemath-4+ + infiwebmath-3+ (requires `datasets`)
  - The Stack v1 (--code N) - source code, 16 languages, permissive licenses (requires `datasets` + HF auth)

Usage:
  python collect_pretraining_data.py --wiki-dump            # Download full Wikipedia dump (RECOMMENDED)
  python collect_pretraining_data.py --wiki 5000            # 5K random Wikipedia articles via API
  python collect_pretraining_data.py --books 500            # 500 Gutenberg books
  python collect_pretraining_data.py --simple 500           # 500 Simple Wikipedia articles
  python collect_pretraining_data.py --fineweb 25           # 25 GB of FineWeb-Edu educational text
  python collect_pretraining_data.py --stackexchange        # Stack Exchange Q&A (22 sites, score>=5)
  python collect_pretraining_data.py --wayback 1000         # 1000 pages from educational domains
  python collect_pretraining_data.py --fandom 2000          # 2000 articles from Fandom wikis
  python collect_pretraining_data.py --openwebtext 10       # 10 GB of OpenWebText web text
  python collect_pretraining_data.py --c4 20                # 20 GB of C4 cleaned Common Crawl
  python collect_pretraining_data.py --dclm 15              # 15 GB of DCLM model-filtered web text
  python collect_pretraining_data.py --finemath 10          # 10 GB of FineMath step-by-step math
  python collect_pretraining_data.py --code 10              # 10 GB of The Stack v1 code (needs HF auth)
  python collect_pretraining_data.py --all-sources          # Everything current: wiki, books, fineweb, SE, wayback, fandom, dclm, finemath, code (c4/owt retired -- explicit flags only)
  python collect_pretraining_data.py --books-only           # Download only Gutenberg books
  python collect_pretraining_data.py --resume               # Resume interrupted download
  python collect_pretraining_data.py --stats                # Show current data stats

Output:
  data/pretrain/wikipedia_dump/  - Wikipedia dump article .txt files (from --wiki-dump)
  data/pretrain/wikipedia/       - Wikipedia article .txt files (from --wiki API)
  data/pretrain/simple_wiki/     - Simple Wikipedia .txt files
  data/pretrain/gutenberg/       - Project Gutenberg book .txt files
  data/pretrain/fineweb_edu/     - FineWeb-Edu educational web page .txt files
  data/pretrain/stackexchange/   - Stack Exchange Q&A posts (subdirs per site)
  data/pretrain/wayback/         - Wayback Machine educational pages
  data/pretrain/fandom/          - Fandom wiki articles
  data/pretrain/dclm/            - DCLM-Baseline model-filtered web pages
  data/pretrain/finemath/        - FineMath step-by-step math pages
  data/pretrain/the_stack/       - The Stack (v1) source code files, weighted per-language
"""

import argparse
import bz2
import hashlib
import html as html_mod
import json
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package not found. Install with: pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent / "data" / "pretrain"
WIKI_DIR = BASE_DIR / "wikipedia"
WIKI_DUMP_DIR = BASE_DIR / "wikipedia_dump"
SIMPLE_DIR = BASE_DIR / "simple_wiki"
GUTENBERG_DIR = BASE_DIR / "gutenberg"
FINEWEB_DIR = BASE_DIR / "fineweb_edu"
STACKEX_DIR = BASE_DIR / "stackexchange"
WAYBACK_DIR = BASE_DIR / "wayback"
FANDOM_DIR = BASE_DIR / "fandom"
OPENWEBTEXT_DIR = BASE_DIR / "openwebtext"
C4_DIR = BASE_DIR / "c4"
DCLM_DIR = BASE_DIR / "dclm"
FINEMATH_DIR = BASE_DIR / "finemath"
STACK_DIR = BASE_DIR / "the_stack"
PROGRESS_FILE = BASE_DIR / "progress.json"
COMBINED_FILE = BASE_DIR / "combined.txt"

# Wikipedia dump download
WIKI_DUMP_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
WIKI_DUMP_FILE = BASE_DIR / "enwiki-latest-pages-articles.xml.bz2"

# Wikipedia API
WIKI_API = "https://en.wikipedia.org/w/api.php"
SIMPLE_WIKI_API = "https://simple.wikipedia.org/w/api.php"

# Rate limiting - Wikipedia requires respectful access
WIKI_DELAY = 3.0  # seconds between Wikipedia API calls
GUTENBERG_DELAY = 1.0  # seconds between Gutenberg downloads
WAYBACK_DELAY = 2.0  # seconds between Wayback Machine requests
FANDOM_DELAY = 0.5  # seconds between Fandom API requests
MAX_RETRIES = 8  # max retries before giving up on a source

# Minimum quality thresholds
MIN_ARTICLE_LENGTH = 500  # characters - skip stubs
MIN_BOOK_LENGTH = 10000  # characters - skip fragments
MIN_PARAGRAPH_LENGTH = 50  # characters - skip tiny paragraphs

# One RECORD SEPARATOR (U+001E) between records inside a packed .txt file.
# "\n\n" alone is indistinguishable from a paragraph break, so the reader
# (pretokenize_data.iter_cleaned_docs, which imports this constant) had to
# treat one 5 MB packed FILE as one document and <s>/</s> boundaries were
# lost for every batched source. _sanitize_special_literals strips stray
# U+001E from record text so the separator can never be forged from inside.
DOC_SEP_CHAR = "\x1e"
DOC_SEP_JOIN = f"\n{DOC_SEP_CHAR}\n"
# Every packed-file write passes newline="\n": text-mode write_text otherwise
# translates \n to \r\n on Windows, putting \r\n\x1e\r\n on disk. The
# universal-newlines reader round-trips that invisibly, but a binary reader,
# hash manifest, or grep for the documented pattern finds nothing. (The
# 2026-07-30 v2c pull predates this and is CRLF at rest; its TOKENS are
# unaffected because the reader normalizes on the way in.)

# User agent (required by Wikipedia and Gutenberg policies)
USER_AGENT = (
    "EnigmaEngine-DataCollector/1.0 "
    "(AI pre-training data collection; educational/research use; "
    "https://github.com/enigma-engine; rate-limited to 1 req/3s)"
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------


def clean_wiki_text(text: str) -> str:
    """Clean Wikipedia article text for training."""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[edit\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if len(line) < 20 and line.endswith(":"):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_wikitext(text: str) -> str:
    """Convert raw MediaWiki wikitext to plain text.

    Strips templates, markup, references, tables, images, categories,
    and other wiki syntax.  Not perfect but good enough for training data.
    """
    # HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # <ref>...</ref> and self-closing <ref ... />
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", "", text)

    # Block-level tags whose content should be dropped
    for tag in (
        "gallery",
        "source",
        "syntaxhighlight",
        "pre",
        "math",
        "score",
        "timeline",
        "nowiki",
        "imagemap",
        "mapframe",
    ):
        text = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remaining HTML tags (keep content)
    text = re.sub(r"<[^>]+/?>", "", text)

    # Tables  {| ... |}  (may be nested — iterate)
    for _ in range(5):
        prev = text
        text = re.sub(r"\{\|[^{]*?\|\}", "", text, flags=re.DOTALL)
        if prev == text:
            break
    # Leftover table openers (broken markup)
    text = re.sub(r"\{\|.*", "", text)

    # Templates {{ ... }} (often deeply nested — iterate)
    for _ in range(15):
        prev = text
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
        if prev == text:
            break

    # File / Image / Category links
    text = re.sub(
        r"\[\[(File|Image|Archivo|Fichier|Datei):[^\]]*\]\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\[\[Category:[^\]]*\]\]", "", text, flags=re.IGNORECASE)

    # Piped links [[target|display]] → display
    text = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", text)
    # Plain links [[target]] → target
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)

    # External links [url text] → text,  [url] → ''
    text = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", text)
    text = re.sub(r"\[https?://[^\]]*\]", "", text)

    # Bold / italic
    text = re.sub(r"'{2,5}", "", text)

    # Section headings == title == → title
    text = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.MULTILINE)

    # List / indent markers
    text = re.sub(r"^[*#:;]+\s*", "", text, flags=re.MULTILINE)

    # Common HTML entities
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&ndash;", "\u2013", text)
    text = re.sub(r"&mdash;", "\u2014", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&#\d+;", "", text)

    # Magic words
    text = re.sub(r"__[A-Z]+__", "", text)

    # Whitespace cleanup
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def clean_gutenberg_text(text: str) -> str:
    """Clean Project Gutenberg book text - remove headers/footers."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            newline_idx = text.find("\n", idx)
            if newline_idx != -1:
                start_idx = newline_idx + 1
            break

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    text = text[start_idx:end_idx]
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def quality_filter(text: str, min_length: int) -> bool:
    """Check if text meets minimum quality standards."""
    if len(text) < min_length:
        return False
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.5:
        return False
    sentence_endings = text.count(".") + text.count("?") + text.count("!")
    return sentence_endings >= 3


# Strong signals that text is AI-generated
_AI_STRONG_MARKERS = [
    "as an ai language model",
    "as a large language model",
    "i'm an ai assistant",
    "i am an ai assistant",
    "as an artificial intelligence",
    "my training data",
    "my knowledge cutoff",
    "i was trained by openai",
    "i was trained by anthropic",
    "regenerate response",
    "i cannot provide real-time",
    "i don't have personal experiences",
    "i don't have the ability to browse",
]

# Weak signals — only flag if 4+ different markers appear
_AI_WEAK_MARKERS = [
    "it's important to note that",
    "it is worth noting that",
    "in today's digital age",
    "in today's fast-paced world",
    "dive into the world of",
    "comprehensive guide to",
    "game-changer",
    "mastering the art of",
    "unlock the power of",
    "let's delve into",
    "without further ado",
    "in this article, we will explore",
    "in this comprehensive",
    "revolutionize your",
    "here are some key",
    "here are some tips",
    "overall, it is clear that",
    "in conclusion, it is evident",
    "there are several key factors",
    "it is essential to understand",
]


def detect_ai_content(text: str) -> bool:
    """Return True if text appears to contain AI-generated content.

    Uses a conservative heuristic: flags on any strong marker OR on
    4+ different weak markers. Designed to minimize false positives
    on legitimate academic/educational writing.
    """
    text_lower = text.lower()

    # Any strong marker = flag immediately
    for marker in _AI_STRONG_MARKERS:
        if marker in text_lower:
            return True

    # Count distinct weak markers
    weak_hits = sum(1 for marker in _AI_WEAK_MARKERS if marker in text_lower)
    return weak_hits >= 4


# ---------------------------------------------------------------------------
# Wikipedia dump download + extraction
# ---------------------------------------------------------------------------


def _download_wiki_dump(resume: bool = True) -> bool:
    """Download the English Wikipedia dump bzip2 file with resume support.

    Returns True if file is ready (already existed or download completed).
    The dump is ~22 GB compressed. Downloads with 10 MB chunks and prints
    progress every 100 MB.
    """
    WIKI_DUMP_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded — verify against server Content-Length
    if WIKI_DUMP_FILE.exists():
        existing_size = WIKI_DUMP_FILE.stat().st_size
        # Verify completeness via HEAD request instead of guessing (S723)
        try:
            head = SESSION.head(WIKI_DUMP_URL, timeout=30)
            head.raise_for_status()
            expected = int(head.headers.get("Content-Length", 0))
            if expected > 0 and existing_size >= expected:
                print(f"  [Wiki Dump] Already downloaded: {existing_size / 1e9:.1f} GB (verified)")
                return True
        except Exception:
            pass  # HEAD failed — fall through to resume attempt
        if resume:
            print(f"  [Wiki Dump] Resuming download ({existing_size / 1e6:.0f} MB so far)...")
        else:
            WIKI_DUMP_FILE.unlink()

    # Stream download with progress
    headers = {}
    mode = "wb"
    downloaded = 0

    if resume and WIKI_DUMP_FILE.exists():
        downloaded = WIKI_DUMP_FILE.stat().st_size
        headers["Range"] = f"bytes={downloaded}-"
        mode = "ab"

    print("  [Wiki Dump] Downloading from dumps.wikimedia.org...")
    print("  [Wiki Dump] This is ~22 GB -- expect 30-90 min depending on connection speed.")
    print("  [Wiki Dump] Safe to Ctrl+C and --resume later.")

    try:
        resp = SESSION.get(WIKI_DUMP_URL, headers=headers, stream=True, timeout=60)
        if resp.status_code == 416:
            # Range not satisfiable — file is already complete
            print("  [Wiki Dump] Download already complete.")
            return True
        resp.raise_for_status()

        # A server that ignores Range answers 200 with the FULL body;
        # appending that onto the partial file corrupts the archive (and
        # the size check would then bless it as complete). Start over.
        if headers.get("Range") and resp.status_code != 206:
            print("  [Wiki Dump] Server ignored the resume request -- restarting from zero.")
            mode = "wb"
            downloaded = 0

        # Get total size from Content-Length or Content-Range
        total = None
        if "Content-Range" in resp.headers:
            # Format: "bytes 12345-67890/total"
            total_str = resp.headers["Content-Range"].rsplit("/", 1)[-1]
            if total_str.isdigit():
                total = int(total_str)
        elif "Content-Length" in resp.headers:
            total = downloaded + int(resp.headers["Content-Length"])

        chunk_size = 10 * 1024 * 1024  # 10 MB chunks
        last_print = downloaded
        start_time = time.monotonic()
        last_print_time = start_time

        with open(WIKI_DUMP_FILE, mode) as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Print progress every ~100 MB
                    if downloaded - last_print >= 100_000_000:
                        now = time.monotonic()
                        since_last = now - last_print_time
                        speed = (downloaded - last_print) / max(since_last, 0.1) / 1e6
                        if total:
                            pct = downloaded / total * 100
                            remaining_bytes = total - downloaded
                            eta_min = remaining_bytes / max(speed * 1e6, 1) / 60
                            print(
                                f"  [Wiki Dump] {downloaded / 1e9:.1f}/{total / 1e9:.1f} GB "
                                f"({pct:.0f}%) @ {speed:.0f} MB/s -- ~{eta_min:.0f} min left"
                            )
                        else:
                            print(f"  [Wiki Dump] {downloaded / 1e9:.1f} GB downloaded @ {speed:.0f} MB/s...")
                        last_print = downloaded
                        last_print_time = now

        print(f"  [Wiki Dump] Download complete: {downloaded / 1e9:.1f} GB")
        return True

    except KeyboardInterrupt:
        print(f"\n  [Wiki Dump] Download interrupted at {downloaded / 1e9:.1f} GB. Run with --resume to continue.")
        return False
    except requests.RequestException as e:
        print(f"  [Wiki Dump] Download error: {e}")
        print("  [Wiki Dump] Run with --resume to retry.")
        return False


def _clean_wikitext(wikitext: str) -> str:
    """Convert wikitext markup to clean plaintext using fast regex patterns.

    Handles the major wikitext constructs: templates, links, formatting,
    HTML tags, references, tables, categories, and file embeds. Some minor
    artifacts may remain but are harmless for pre-training data.
    """
    text = wikitext

    # Strips the wiki markup families -- templates (iterated because they
    # nest, bounded at 8 passes), tables, refs and HTML, category/file
    # embeds, links (keeping display text), formatting, headers, magic
    # words, leftover braces -- with the specific families handled before
    # the generic link and brace sweeps so their text cannot leak into the
    # plaintext output.
    for _ in range(8):
        result = re.sub(r"\{\{[^{}]*\}\}", "", text)
        if result == text:
            break
        text = result

    text = re.sub(r"\{\{[^}]{0,500}$", "", text, flags=re.MULTILINE)

    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)

    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<ref[^>]*/>", "", text, flags=re.IGNORECASE)

    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    text = re.sub(r"<[^>]+>", "", text)

    text = re.sub(r"\[\[Category:[^\]]*\]\]", "", text, flags=re.IGNORECASE)

    text = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", "", text, flags=re.IGNORECASE)

    text = re.sub(r"\[\[[^|\]]*\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    text = re.sub(r"https?://\S+", "", text)

    text = re.sub(r"'{2,5}", "", text)

    text = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", text, flags=re.MULTILINE)

    text = re.sub(r"__[A-Z]+__", "", text)

    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)

    text = re.sub(r"\{[^}]*\}", "", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        # Skip lines that look like leftover markup artifacts
        if len(line) < 30 and not line.endswith((".", "!", "?", ":")):
            continue
        # Skip lines that are just bullets/numbers without content
        if re.match(r"^[\*#;:]+\s*$", line):
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def _is_article_namespace(title: str) -> bool:
    """Check if a Wikipedia title is in the main article namespace.

    Rejects talk pages, user pages, Wikipedia meta pages, etc.
    """
    # MediaWiki namespace prefixes to skip
    skip_prefixes = (
        "Talk:",
        "User:",
        "User talk:",
        "Wikipedia:",
        "Wikipedia talk:",
        "File:",
        "File talk:",
        "MediaWiki:",
        "MediaWiki talk:",
        "Template:",
        "Template talk:",
        "Help:",
        "Help talk:",
        "Category:",
        "Category talk:",
        "Portal:",
        "Portal talk:",
        "Draft:",
        "Draft talk:",
        "Module:",
        "Module talk:",
        "TimedText:",
        "TimedText talk:",
        "Gadget:",
        "Gadget talk:",
        "Gadget definition:",
        "Gadget definition talk:",
    )
    return not title.startswith(skip_prefixes)


def process_wiki_dump(progress: dict) -> int:
    """Download and process the full English Wikipedia dump.

    Stream-parses the bzip2 XML ~22 GB file with constant memory usage.
    Extracts article text, cleans wikitext markup, quality-filters,
    and saves individual .txt files to data/pretrain/wikipedia_dump/.

    The XML structure is:
      <mediawiki>
        <page>
          <title>Article Title</title>
          <ns>0</ns>   <!-- namespace: 0 = main article -->
          <redirect title="..." />   <!-- if it's a redirect -->
          <revision>
            <text>...wikitext markup...</text>
          </revision>
        </page>
        ...
      </mediawiki>
    """
    if not _download_wiki_dump(resume=True):
        return 0

    WIKI_DUMP_DIR.mkdir(parents=True, exist_ok=True)

    # Track what we've already processed for resume support
    existing = set(f.stem for f in WIKI_DUMP_DIR.glob("*.txt"))
    saved = len(existing)
    skipped_redirect = 0
    skipped_namespace = 0
    skipped_quality = 0
    skipped_short = 0
    processed = 0
    total_bytes_written = sum(f.stat().st_size for f in WIKI_DUMP_DIR.glob("*.txt"))

    if saved > 0:
        print(f"  [Wiki Dump] Resuming: {saved} articles already extracted ({total_bytes_written / 1e9:.2f} GB)")

    print(f"  [Wiki Dump] Processing dump file ({WIKI_DUMP_FILE.stat().st_size / 1e9:.1f} GB)...")
    print("  [Wiki Dump] This takes 2-6 hours depending on CPU speed.")
    print("  [Wiki Dump] Safe to Ctrl+C and --resume later.")

    # Auto-detect MediaWiki XML namespace from dump header
    ns = None

    start_time = time.monotonic()
    last_print = start_time

    try:
        # Stream decompress + parse — constant memory
        # Use start+end events so we can track root for memory cleanup
        with bz2.open(str(WIKI_DUMP_FILE), "rb") as f:
            context = ET.iterparse(f, events=("start", "end"))  # noqa: S314  -- parses the official Wikipedia dump (trusted source); use defusedxml if a source ever becomes untrusted
            root = None

            for event, elem in context:
                if event == "start":
                    if root is None:
                        root = elem
                    continue

                # Auto-detect namespace from the first element's tag
                if ns is None and "}" in elem.tag:
                    ns = elem.tag.split("}")[0].lstrip("{")
                    print(f"  [Wiki Dump] Detected XML namespace: {ns}")

                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

                if tag != "page":
                    continue

                processed += 1

                # Extract page data
                title_elem = elem.find(f"{{{ns}}}title")
                ns_elem = elem.find(f"{{{ns}}}ns")
                redirect_elem = elem.find(f"{{{ns}}}redirect")
                revision = elem.find(f"{{{ns}}}revision")

                title = title_elem.text if title_elem is not None and title_elem.text else ""

                # --- Process page (all paths lead to cleanup at the end) ---
                skip = False

                # Skip non-article namespaces
                page_ns = ns_elem.text if ns_elem is not None and ns_elem.text else "0"
                if page_ns != "0" or not _is_article_namespace(title):
                    skipped_namespace += 1
                    skip = True

                # Skip redirects
                if not skip and redirect_elem is not None:
                    skipped_redirect += 1
                    skip = True

                # Get wikitext
                wikitext = ""
                if not skip:
                    text_elem = revision.find(f"{{{ns}}}text") if revision is not None else None
                    wikitext = text_elem.text if text_elem is not None and text_elem.text else ""

                    # Skip redirect wikitext (some redirects don't have <redirect>)
                    if wikitext.strip().lower().startswith("#redirect"):
                        skipped_redirect += 1
                        skip = True

                # Skip disambiguation pages
                title_lower = title.lower()
                if not skip:
                    if (
                        "disambiguation" in title_lower
                        or "(disambiguation)" in wikitext[:500].lower()
                        or "{{disambiguation" in wikitext.lower()[:1000]
                    ):
                        skipped_quality += 1
                        skip = True

                # Skip list articles
                if not skip:
                    if title_lower.startswith("list of ") or title_lower.startswith("lists of "):
                        skipped_quality += 1
                        skip = True

                # Make safe filename
                safe_name = ""
                if not skip:
                    safe_name = re.sub(r"[^\w\s-]", "", title)[:80].strip()
                    safe_name = re.sub(r"\s+", "_", safe_name)
                    if not safe_name:
                        skip = True

                # Skip if already exists (resume)
                if not skip and safe_name in existing:
                    skip = True

                # Clean wikitext and save
                if not skip:
                    cleaned = _clean_wikitext(wikitext)

                    if len(cleaned) < MIN_ARTICLE_LENGTH:
                        skipped_short += 1
                        skip = True
                    elif not quality_filter(cleaned, MIN_ARTICLE_LENGTH):
                        skipped_quality += 1
                        skip = True
                    else:
                        filepath = WIKI_DUMP_DIR / f"{safe_name}.txt"
                        filepath.write_text(cleaned, encoding="utf-8")
                        existing.add(safe_name)
                        saved += 1
                        total_bytes_written += len(cleaned.encode("utf-8"))

                # Free memory — clear element and remove from root
                elem.clear()
                if root is not None:
                    try:
                        root.remove(elem)
                    except ValueError:
                        pass

                # Progress report every 60 seconds
                now = time.monotonic()
                if now - last_print >= 60:
                    elapsed_min = (now - start_time) / 60
                    rate = (processed / elapsed_min) if elapsed_min > 0 else 0
                    # Wikipedia has ~7M pages total, ~6.8M articles
                    est_remaining = (7_000_000 - processed) / max(rate, 1)
                    print(
                        f"  [Wiki Dump] {saved:,} articles saved "
                        f"({total_bytes_written / 1e9:.2f} GB) | "
                        f"{processed:,} pages processed | "
                        f"{rate:,.0f} pages/min | "
                        f"~{est_remaining:.0f} min left | "
                        f"skipped: {skipped_redirect:,} redirects, "
                        f"{skipped_short:,} short, "
                        f"{skipped_quality:,} low-quality"
                    )
                    last_print = now

                    # Save progress periodically
                    progress["wiki_dump"] = {
                        "articles_saved": saved,
                        "pages_processed": processed,
                        "bytes_written": total_bytes_written,
                    }
                    save_progress(progress)

    except KeyboardInterrupt:
        elapsed_min = (time.monotonic() - start_time) / 60
        print(
            f"\n  [Wiki Dump] Interrupted after {elapsed_min:.0f} min. "
            f"{saved:,} articles saved ({total_bytes_written / 1e9:.2f} GB). "
            f"Run with --resume to continue."
        )
    except ET.ParseError as e:
        print(f"  [Wiki Dump] XML parse error: {e}")
        print("  [Wiki Dump] The dump file may be corrupted. Delete it and re-download: --wiki-dump")

    # Final stats
    elapsed_min = (time.monotonic() - start_time) / 60
    print(
        f"\n  [Wiki Dump] Done: {saved:,} articles saved ({total_bytes_written / 1e9:.2f} GB) in {elapsed_min:.0f} min."
    )
    print(
        f"  [Wiki Dump] Skipped: {skipped_redirect:,} redirects, "
        f"{skipped_namespace:,} non-article, "
        f"{skipped_short:,} short, {skipped_quality:,} low-quality"
    )

    progress["wiki_dump"] = {
        "articles_saved": saved,
        "pages_processed": processed,
        "bytes_written": total_bytes_written,
    }
    save_progress(progress)
    return saved


# ---------------------------------------------------------------------------
# Progress tracking (for --resume)
# ---------------------------------------------------------------------------


def load_progress() -> dict:
    """Load download progress from disk (tolerates corrupt files)."""
    if PROGRESS_FILE.exists():
        try:
            text = PROGRESS_FILE.read_text(encoding="utf-8")
            if text.strip():
                return json.loads(text)
        except (json.JSONDecodeError, OSError) as e:
            # Corrupt progress file — check for backup
            backup = PROGRESS_FILE.with_suffix(".json.bak")
            if backup.exists():
                try:
                    return json.loads(backup.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass
            print(f"  WARNING: progress.json corrupt ({e}), starting fresh.")
    return {"gutenberg_ids": [], "stats": {}}


def save_progress(progress: dict) -> None:
    """Save download progress atomically (write temp, then rename)."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROGRESS_FILE.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        # Keep one backup of the previous good state
        if PROGRESS_FILE.exists():
            backup = PROGRESS_FILE.with_suffix(".json.bak")
            try:
                PROGRESS_FILE.replace(backup)
            except OSError:
                pass
        tmp.replace(PROGRESS_FILE)
    except OSError as e:
        print(f"  WARNING: Failed to save progress: {e}")


# ---------------------------------------------------------------------------
# Wikipedia downloader (API with proper rate limiting)
# ---------------------------------------------------------------------------


def fetch_wikipedia_articles(api_url: str, output_dir: Path, max_articles: int, label: str, progress: dict) -> int:
    """Download articles from Wikipedia or Simple Wikipedia.

    Uses the MediaWiki Action API with:
    - generator=random + prop=extracts (random high-quality articles)
    - maxlag=5 (backs off when servers are loaded)
    - Batch size of 10 (smaller = less server load per request)
    - 3-second delay between requests
    - Exponential backoff on 429 (rate limit) responses

    Random articles are much higher quality than alphabetical iteration,
    which starts with stubs and disambiguation pages.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = set(f.stem for f in output_dir.glob("*.txt"))
    downloaded = len(existing)

    if downloaded >= max_articles:
        print(f"  [{label}] Already have {downloaded}/{max_articles} articles, skipping.")
        return downloaded

    remaining = max_articles - downloaded
    # Random articles have ~50% quality pass rate, so estimate ~2 batches per 10 articles
    est_minutes = (remaining / 5) * WIKI_DELAY / 60
    print(f"  [{label}] Downloading articles ({downloaded}/{max_articles} exist)...")
    print(f"  [{label}] Estimated time: ~{est_minutes:.0f} minutes for {remaining} articles")

    consecutive_errors = 0
    backoff = 30
    batches_tried = 0
    last_print = downloaded

    while downloaded < max_articles:
        params = {
            "action": "query",
            "format": "json",
            "generator": "random",
            "grnlimit": 10,
            "grnnamespace": 0,  # Main article namespace only
            "prop": "extracts",
            "explaintext": True,
            "exsectionformat": "plain",
            "maxlag": 5,
        }

        try:
            resp = SESSION.get(api_url, params=params, timeout=30)

            if resp.status_code == 429:
                consecutive_errors += 1
                if consecutive_errors >= MAX_RETRIES:
                    print(f"\n  [{label}] Rate-limited {MAX_RETRIES} times. Stopping. Run with --resume later.")
                    break
                retry_after = int(resp.headers.get("Retry-After", backoff))
                wait = max(retry_after, backoff)
                print(f"  [{label}] Rate limited. Waiting {wait}s... ({consecutive_errors}/{MAX_RETRIES})")
                time.sleep(wait)
                backoff = min(backoff * 2, 600)
                continue

            if resp.status_code == 200:
                data = resp.json()
                if "error" in data and data["error"].get("code") == "maxlag":
                    lag = data["error"].get("info", "")
                    print(f"  [{label}] Server busy ({lag}). Waiting 10s...")
                    time.sleep(10)
                    continue

            resp.raise_for_status()
            data = resp.json()
            consecutive_errors = 0
            backoff = 30

        except requests.exceptions.HTTPError as e:
            consecutive_errors += 1
            if consecutive_errors >= MAX_RETRIES:
                print(f"\n  [{label}] Too many HTTP errors. Stopping. Run with --resume later.")
                break
            print(f"  [{label}] HTTP {e}. Waiting {backoff}s... ({consecutive_errors}/{MAX_RETRIES})")
            time.sleep(backoff)
            backoff = min(backoff * 2, 600)
            continue
        except (requests.RequestException, json.JSONDecodeError) as e:
            consecutive_errors += 1
            if consecutive_errors >= MAX_RETRIES:
                print(f"\n  [{label}] Too many errors. Stopping. Run with --resume later.")
                break
            print(f"  [{label}] Error: {e}. Retrying in {backoff}s...")
            time.sleep(backoff)
            continue

        result_pages = data.get("query", {}).get("pages", {})
        batches_tried += 1

        if not result_pages:
            print(f"  [{label}] Empty batch (try {batches_tried}), retrying...")
            time.sleep(WIKI_DELAY)
            continue

        for page_id, page_info in result_pages.items():
            if downloaded >= max_articles:
                break

            title = page_info.get("title", f"page_{page_id}")
            extract = page_info.get("extract", "")

            if not extract:
                continue

            # Skip disambiguation and list pages — poor training data
            title_lower = title.lower()
            if (
                "disambiguation" in title_lower
                or title_lower.startswith("list of")
                or title_lower.startswith("lists of")
                or "(disambiguation)" in extract.lower()[:500]
            ):
                continue

            cleaned = clean_wiki_text(extract)

            if not quality_filter(cleaned, MIN_ARTICLE_LENGTH):
                continue

            safe_name = re.sub(r"[^\w\s-]", "", title)[:80].strip()
            safe_name = re.sub(r"\s+", "_", safe_name)

            if not safe_name or safe_name in existing:
                continue

            filepath = output_dir / f"{safe_name}.txt"
            filepath.write_text(cleaned, encoding="utf-8")
            existing.add(safe_name)
            downloaded += 1

        # Print progress every 10 new articles or every 20 batches
        if downloaded - last_print >= 10 or batches_tried % 20 == 0:
            remaining = max_articles - downloaded
            est = (remaining / 5) * WIKI_DELAY / 60
            print(
                f"  [{label}] {downloaded}/{max_articles} downloaded "
                f"(batch {batches_tried}, ~{est:.0f} min remaining)..."
            )
            last_print = downloaded
            save_progress(progress)

        time.sleep(WIKI_DELAY)

    print(f"  [{label}] Done: {downloaded} articles saved ({batches_tried} batches).")
    save_progress(progress)
    return downloaded


# ---------------------------------------------------------------------------
# Gutenberg downloader
# ---------------------------------------------------------------------------

# ~250 curated high-quality public domain books (by Gutenberg ID).
# Diverse genres: fiction, science, philosophy, history, adventure, drama, education.
# All English. All substantial length.
GUTENBERG_CURATED = [
    # === Classic Fiction & Literature ===
    1342,  # Pride and Prejudice - Austen
    11,  # Alice in Wonderland - Carroll
    1661,  # Sherlock Holmes - Doyle
    84,  # Frankenstein - Shelley
    98,  # A Tale of Two Cities - Dickens
    74,  # Tom Sawyer - Twain
    43,  # Jekyll and Hyde - Stevenson
    76,  # Huckleberry Finn - Twain
    2701,  # Moby Dick - Melville
    1952,  # The Yellow Wallpaper - Gilman
    345,  # Dracula - Stoker
    1400,  # Great Expectations - Dickens
    174,  # Dorian Gray - Wilde
    16328,  # Beowulf
    1232,  # The Prince - Machiavelli
    244,  # A Study in Scarlet - Doyle
    2554,  # Crime and Punishment - Dostoevsky
    2600,  # War and Peace - Tolstoy
    1260,  # Jane Eyre - Bronte
    135,  # Les Miserables - Hugo
    514,  # Little Women - Alcott
    768,  # Wuthering Heights - Bronte
    35,  # The Time Machine - Wells
    5200,  # Metamorphosis - Kafka
    996,  # Don Quixote - Cervantes
    55,  # Wizard of Oz - Baum
    4300,  # Ulysses - Joyce
    161,  # Sense and Sensibility - Austen
    36,  # War of the Worlds - Wells
    219,  # Heart of Darkness - Conrad
    730,  # Oliver Twist - Dickens
    28054,  # Brothers Karamazov - Dostoevsky
    2591,  # Grimm's Fairy Tales
    1184,  # Count of Monte Cristo - Dumas
    829,  # Gulliver's Travels - Swift
    45,  # Anne of Green Gables - Montgomery
    209,  # Turn of the Screw - James
    2542,  # Hound of the Baskervilles - Doyle
    1998,  # Thus Spoke Zarathustra - Nietzsche
    408,  # Souls of Black Folk - Du Bois
    158,  # Emma - Austen
    1399,  # Anna Karenina - Tolstoy
    521,  # Robinson Crusoe - Defoe
    541,  # Age of Innocence - Wharton
    940,  # The Scarlet Letter - Hawthorne
    11030,  # The Secret Garden - Burnett
    6593,  # Tom Jones - Fielding
    37106,  # Little Lord Fauntleroy - Burnett
    19942,  # Candide - Voltaire
    25344,  # The Scarlet Pimpernel - Orczy
    113,  # The Secret Agent - Conrad
    1250,  # Tess of the d'Urbervilles - Hardy
    120,  # Treasure Island - Stevenson
    1727,  # The Odyssey - Homer
    6130,  # The Iliad - Homer
    2814,  # Dubliners - Joyce
    4363,  # Beyond Good and Evil - Nietzsche
    62,  # A Princess of Mars - Burroughs
    26,  # Paradise Lost - Milton
    5197,  # My Antonia - Cather
    7849,  # The Awakening - Chopin
    # === Science, Philosophy & Non-Fiction ===
    4993,  # A Christmas Carol - Dickens
    22381,  # Calculus Made Easy - Thompson
    37134,  # The Elements - Euclid
    4280,  # Critique of Pure Reason - Kant
    5827,  # Problems of Philosophy - Russell
    1228,  # On Liberty - Mill
    20203,  # Autobiography of Benjamin Franklin
    4217,  # Portrait of the Artist - Joyce
    910,  # White Fang - London
    2131,  # Wealth of Nations - Smith
    7370,  # Second Treatise of Government - Locke
    16457,  # The Federalist Papers
    29474,  # Elements of Style - Strunk
    4705,  # Treatise of Human Nature - Hume
    3600,  # Essays First Series - Emerson
    7178,  # Meditations - Marcus Aurelius
    25717,  # Narrative of Frederick Douglass
    1497,  # The Republic - Plato
    3296,  # Confessions - Augustine
    28233,  # Origin of Species - Darwin
    2680,  # Meditations - Descartes
    49010,  # Interpretation of Dreams - Freud
    37090,  # Principles of Psychology - James
    38427,  # Descent of Man - Darwin
    4078,  # Opticks - Newton
    26659,  # Origin of Species 6th ed - Darwin
    15776,  # Practice and Science of Drawing - Speed
    # === Drama & Poetry ===
    1524,  # Hamlet - Shakespeare
    1513,  # Romeo and Juliet - Shakespeare
    1514,  # King Lear - Shakespeare
    1515,  # The Tempest - Shakespeare
    1533,  # Macbeth - Shakespeare
    1041,  # Shakespeare Sonnets
    1112,  # Taming of the Shrew - Shakespeare
    2265,  # Midsummer Night's Dream - Shakespeare
    23042,  # Twelfth Night - Shakespeare
    100,  # Complete Works of Shakespeare
    27761,  # Othello - Shakespeare
    2267,  # Much Ado About Nothing - Shakespeare
    1519,  # Merchant of Venice - Shakespeare
    2235,  # Importance of Being Earnest - Wilde
    1790,  # Julius Caesar - Shakespeare
    23,  # Narrative of Frederick Douglass
    8492,  # Rime of the Ancient Mariner - Coleridge
    22120,  # As You Like It - Shakespeare
    # === Adventure & Travel ===
    103,  # Around the World in 80 Days - Verne
    164,  # Twenty Thousand Leagues - Verne
    236,  # The Jungle Book - Kipling
    215,  # Call of the Wild - London
    2500,  # Siddhartha - Hesse
    47629,  # Mysterious Affair at Styles - Christie
    1259,  # Twenty Years After - Dumas
    3825,  # Meditations on First Philosophy - Descartes
    1257,  # The Three Musketeers - Dumas
    16,  # Peter Pan - Barrie
    308,  # Three Men in a Boat - Jerome
    600,  # Notes from Underground - Dostoevsky
    41,  # Legend of Sleepy Hollow - Irving
    # === Philosophy & Political Theory ===
    3300,  # Essay Concerning Human Understanding - Locke
    10615,  # The Social Contract - Rousseau
    3176,  # Emile - Rousseau
    5740,  # Tractatus Logico-Philosophicus - Wittgenstein
    2412,  # The Apology - Plato
    1656,  # Phaedrus - Plato
    55201,  # Communist Manifesto - Marx/Engels
    7416,  # Democracy in America vol 2 - Tocqueville
    815,  # Democracy in America vol 1 - Tocqueville
    2130,  # Utopia - More
    36034,  # Art of Rhetoric - Aristotle
    8438,  # Politics - Aristotle
    3207,  # Leviathan - Hobbes
    1080,  # A Modest Proposal - Swift
    # === More Fiction ===
    5230,  # The Invisible Man - Wells
    766,  # David Copperfield - Dickens
    2097,  # Autobiography of an Ex-Colored Man - Johnson
    32,  # Herland - Gilman
    14,  # What Is Man - Twain
    356,  # Vindication of the Rights of Woman - Wollstonecraft
    7142,  # Northanger Abbey - Austen
    141,  # Mansfield Park - Austen
    805,  # This Side of Paradise - Fitzgerald
    64317,  # The Great Gatsby - Fitzgerald
    21279,  # Flatland - Abbott
    1064,  # Armadale - Collins
    967,  # The Woman in White - Collins
    779,  # The Moonstone - Collins
    4085,  # Pinocchio - Collodi
    159,  # Same Old Story - Goncharov
    37,  # The Golden Age
    145,  # Middlemarch - Eliot
    11800,  # The Jungle - Sinclair
    932,  # Fall of the House of Usher - Poe
    2148,  # Works of Edgar Allan Poe vol 2
    # === History, Education & Reference ===
    1,  # Declaration of Independence
    2,  # United States Bill of Rights
    5,  # US Constitution
    1322,  # Leaves of Grass - Whitman
    3608,  # On War - Clausewitz
    2610,  # The Analects - Confucius
    678,  # Twelve Years a Slave - Northup
    73,  # A Tramp Abroad - Twain
    1023,  # Bleak House - Dickens
    580,  # Pickwick Papers - Dickens
    132,  # The Art of War - Sun Tzu
    2346,  # Phantom of the Opera - Leroux
    24518,  # First Book in American History - Eggleston
    13846,  # Brief History of the United States
    946,  # The Monster - Crane
    10,  # King James Bible
    16643,  # Walden - Thoreau
    15491,  # The Common Law - Holmes
    14838,  # Art of Money Getting - Barnum
    # === Expanded Fiction & Literature (batch 2) ===
    1080,  # A Modest Proposal - Swift (if not above, dedup handles it)
    25305,  # Nicomachean Ethics - Aristotle
    1497,  # The Republic - Plato (dedup backup)
    6688,  # Metamorphoses - Ovid
    10676,  # The Aeneid - Virgil
    8800,  # Divine Comedy - Dante (Longfellow)
    15474,  # The Canterbury Tales - Chaucer
    3207,  # Leviathan - Hobbes
    4280,  # Critique of Pure Reason - Kant
    7610,  # Enquiry Concerning Human Understanding - Hume
    5740,  # Tractatus - Wittgenstein
    34901,  # Phenomenology of Spirit - Hegel
    39955,  # The Age of Reason - Paine
    38427,  # Descent of Man - Darwin
    4078,  # Opticks - Newton
    30155,  # History of the Peloponnesian War - Thucydides
    2199,  # Book of the Thousand Nights and a Night v1 - Burton
    2200,  # Book of the Thousand Nights and a Night v2 - Burton
    1946,  # Symposium - Plato
    2726,  # Phaedo - Plato
    9296,  # Poetics - Aristotle
    5684,  # The Decameron - Boccaccio
    3608,  # On War - Clausewitz (dedup)
    2981,  # Don Quixote Part 2 - Cervantes
    2000,  # Confessions - Rousseau
    3090,  # Gargantua and Pantagruel - Rabelais
    3527,  # Kim - Kipling
    58585,  # Typee - Melville
    2489,  # Moby Dick (alt) - Melville
    44881,  # A Room with a View - Forster
    2641,  # A Room of One's Own - Woolf
    144,  # Mrs Dalloway - Woolf
    5670,  # To the Lighthouse - Woolf
    4517,  # Ethan Frome - Wharton
    541,  # Age of Innocence - Wharton (dedup)
    1251,  # Le Morte d'Arthur Vol 1 - Malory
    1252,  # Le Morte d'Arthur Vol 2 - Malory
    7386,  # The Story of My Life - Keller
    16119,  # Up from Slavery - Washington
    11,  # Alice in Wonderland (dedup)
    12,  # Through the Looking Glass - Carroll
    19033,  # A Connecticut Yankee - Twain
    86,  # The Prince and the Pauper - Twain
    3186,  # Life on the Mississippi - Twain
    76,  # Huckleberry Finn (dedup)
    30254,  # Autobiography of Mark Twain
    2852,  # The Hound of the Baskervilles (alt) - Doyle
    108,  # The Return of Sherlock Holmes - Doyle
    1661,  # Adventures of Sherlock Holmes (dedup)
    834,  # The Memoirs of Sherlock Holmes - Doyle
    2097,  # Autobiography of an Ex-Colored Man (dedup)
    4583,  # The Waste Land - Eliot
    402,  # Idylls of the King - Tennyson
    1321,  # The Song of Hiawatha - Longfellow
    30368,  # The Rubaiyat of Omar Khayyam - FitzGerald
    271,  # Black Beauty - Sewell
    514,  # Little Women (dedup)
    767,  # Agnes Grey - Bronte
    9182,  # The Tenant of Wildfell Hall - Bronte
    969,  # The Mayor of Casterbridge - Hardy
    153,  # Jude the Obscure - Hardy
    110,  # Tess of the d'Urbervilles - Hardy (alt)
    30,  # The Bible, Douay-Rheims
    8164,  # Treasure Island (alt) - Stevenson
    171,  # Kidnapped - Stevenson
    307,  # The Black Arrow - Stevenson
    1695,  # The Master of Ballantrae - Stevenson
    27827,  # The Kama Sutra
    7370,  # Second Treatise (dedup)
    1497,  # Republic (dedup)
    3600,  # Essays First Series (dedup)
    601,  # Autobiography of John Stuart Mill
    46,  # A Christmas Carol (alt) - Dickens
    700,  # The Old Curiosity Shop - Dickens
    917,  # The Cricket on the Hearth - Dickens
    821,  # Dombey and Son - Dickens
    98,  # Tale of Two Cities (dedup)
    564,  # The Jungle Books - Kipling
    2226,  # Just So Stories - Kipling
    1036,  # Captains Courageous - Kipling
    236,  # Jungle Book (dedup)
    33,  # The Scarlet Letter (alt) - Hawthorne
    77,  # The House of the Seven Gables - Hawthorne
    2081,  # Tanglewood Tales - Hawthorne
    18,  # The Rime of the Ancient Mariner - Coleridge
    8492,  # Rime (alt)
    636,  # Lyrical Ballads - Wordsworth/Coleridge
    2600,  # War and Peace (dedup)
    7178,  # Meditations (dedup)
    2680,  # Meditations - Descartes (dedup)
    1399,  # Anna Karenina (dedup)
    28054,  # Brothers K (dedup)
    600,  # Notes Underground (dedup)
    2554,  # Crime and Punishment (dedup)
    2197,  # The Gambler - Dostoevsky
    1672,  # The Idiot - Dostoevsky
    15823,  # Swann's Way - Proust
    5881,  # Three Men in a Boat (alt) - Jerome
    11231,  # Don Juan - Byron
    5131,  # Childe Harold's Pilgrimage - Byron
    202,  # The Innocents Abroad - Twain
    71,  # Adventures of Tom Sawyer (alt)
    7771,  # The Cossacks - Tolstoy
    985,  # The Kreutzer Sonata - Tolstoy
    2142,  # Resurrection - Tolstoy
    38582,  # Hadji Murad - Tolstoy
    1184,  # Monte Cristo (dedup)
    965,  # The Count of Monte Cristo (alt) - Dumas
    1258,  # The Vicomte de Bragelonne - Dumas
    2759,  # The Man in the Iron Mask - Dumas
    2707,  # Camille - Dumas fils
    37332,  # Lost Illusions - Balzac
    1237,  # Father Goriot - Balzac
    1553,  # Eugenie Grandet - Balzac
    135,  # Les Miserables (dedup)
    17135,  # The Hunchback of Notre-Dame - Hugo
    36098,  # Toilers of the Sea - Hugo
    2610,  # Analects (dedup)
    132,  # Art of War (dedup)
    2814,  # Dubliners (dedup)
    4217,  # Portrait of the Artist (dedup)
    4300,  # Ulysses (dedup)
    4078,  # Opticks (dedup)
    22381,  # Calculus Made Easy (dedup)
    37134,  # Elements (dedup)
    26,  # Paradise Lost (dedup)
    1268,  # The Voyage of the Beagle - Darwin
]

# Deduplicate while preserving order
_seen_ids: set[int] = set()
_deduped: list[int] = []
for _id in GUTENBERG_CURATED:
    if _id not in _seen_ids:
        _seen_ids.add(_id)
        _deduped.append(_id)
GUTENBERG_CURATED = _deduped
del _seen_ids, _deduped


def get_gutenberg_catalog(max_books: int) -> list[dict]:
    """Build catalog from curated book IDs + gutendex.com API discovery.

    Uses the curated list first (guaranteed quality), then paginates
    through gutendex sorted by popularity to find additional English books.
    """
    print(f"  [Gutenberg] Building book catalog (curated: {len(GUTENBERG_CURATED)} books)...")

    books = []
    seen_ids: set[int] = set()

    for book_id in GUTENBERG_CURATED[:max_books]:
        if book_id in seen_ids:
            continue
        seen_ids.add(book_id)
        text_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        books.append({"id": book_id, "title": f"Book {book_id}", "url": text_url})

    # If we need more than the curated list, paginate through gutendex
    if max_books > len(books):
        # Query by topic — broad set of subjects
        subjects = [
            "fiction",
            "science",
            "philosophy",
            "history",
            "adventure",
            "education",
            "psychology",
            "mathematics",
            "biography",
            "poetry",
            "drama",
            "politics",
            "religion",
            "art",
            "music",
            "law",
            "medicine",
            "economics",
            "nature",
            "travel",
            "humor",
            "essays",
            "mythology",
            "folklore",
            "sociology",
            "astronomy",
            "chemistry",
            "physics",
            "geography",
            "engineering",
        ]
        # Also query by popular bookshelves
        bookshelves = [
            "Best Books Ever Listings",
            "Harvard Classics",
            "Banned Books List",
            "Movie Books",
        ]

        gutendex_errors = 0

        # First pass: popular books by download count (sort=-download_count)
        print("  [Gutenberg] Discovering additional books via gutendex API...")
        page = 1
        while len(books) < max_books and page <= 50 and gutendex_errors < 5:
            try:
                resp = SESSION.get(
                    "https://gutendex.com/books/",
                    params={
                        "languages": "en",
                        "page": page,
                        "sort": "popular",
                    },
                    timeout=60,
                )
                if resp.status_code == 429:
                    time.sleep(30)
                    gutendex_errors += 1
                    continue
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    break

                for item in results:
                    bid = item["id"]
                    if bid in seen_ids:
                        continue
                    # Filter: must have text/plain format available
                    formats = item.get("formats", {})
                    has_text = any("text/plain" in k for k in formats)
                    if not has_text:
                        continue
                    seen_ids.add(bid)
                    text_url = f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt"
                    books.append(
                        {
                            "id": bid,
                            "title": item.get("title", f"Book {bid}"),
                            "url": text_url,
                        }
                    )
                    if len(books) >= max_books:
                        break

                if not data.get("next"):
                    break
                page += 1
                time.sleep(1)  # respect rate limits

            except (requests.RequestException, json.JSONDecodeError) as e:
                gutendex_errors += 1
                if gutendex_errors >= 5:
                    print(f"  [Gutenberg] gutendex too unreliable ({e}), using curated list only.")
                    break
                print(f"  [Gutenberg] gutendex error page {page}: {e}")
                time.sleep(5)
                continue

        # Second pass: topic-based search for diversity
        for subject in subjects:
            if len(books) >= max_books or gutendex_errors >= 5:
                break
            try:
                resp = SESSION.get(
                    "https://gutendex.com/books/",
                    params={"topic": subject, "languages": "en", "page": 1},
                    timeout=60,
                )
                if resp.status_code == 429:
                    time.sleep(30)
                    gutendex_errors += 1
                    continue
                resp.raise_for_status()
                for item in resp.json().get("results", []):
                    bid = item["id"]
                    if bid not in seen_ids:
                        seen_ids.add(bid)
                        text_url = f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt"
                        books.append(
                            {
                                "id": bid,
                                "title": item.get("title", f"Book {bid}"),
                                "url": text_url,
                            }
                        )
                        if len(books) >= max_books:
                            break
                time.sleep(1)
            except (requests.RequestException, json.JSONDecodeError):
                gutendex_errors += 1
                continue

        # Third pass: bookshelf-based
        for shelf in bookshelves:
            if len(books) >= max_books or gutendex_errors >= 5:
                break
            try:
                resp = SESSION.get(
                    "https://gutendex.com/books/",
                    params={"bookshelves": shelf, "languages": "en"},
                    timeout=60,
                )
                if resp.status_code == 429:
                    time.sleep(30)
                    gutendex_errors += 1
                    continue
                resp.raise_for_status()
                for item in resp.json().get("results", []):
                    bid = item["id"]
                    if bid not in seen_ids:
                        seen_ids.add(bid)
                        text_url = f"https://www.gutenberg.org/cache/epub/{bid}/pg{bid}.txt"
                        books.append(
                            {
                                "id": bid,
                                "title": item.get("title", f"Book {bid}"),
                                "url": text_url,
                            }
                        )
                        if len(books) >= max_books:
                            break
                time.sleep(1)
            except (requests.RequestException, json.JSONDecodeError):
                gutendex_errors += 1
                continue

    print(f"  [Gutenberg] {len(books)} books in catalog.")
    return books[:max_books]


def fetch_gutenberg_books(max_books: int, progress: dict) -> int:
    """Download books from Project Gutenberg."""
    GUTENBERG_DIR.mkdir(parents=True, exist_ok=True)

    # Build set of book IDs already on disk (extract numeric prefix from filenames)
    existing_ids: set[str] = set()
    for f in GUTENBERG_DIR.glob("*.txt"):
        m = re.match(r"^(\d+)", f.stem)
        if m:
            existing_ids.add(m.group(1))
    downloaded = len(existing_ids)

    if downloaded >= max_books:
        print(f"  [Gutenberg] Already have {downloaded}/{max_books} books, skipping.")
        return downloaded

    catalog = get_gutenberg_catalog(max_books)
    # Merge progress IDs with what's actually on disk
    downloaded_ids = existing_ids | set(progress.get("gutenberg_ids", []))

    remaining = max_books - downloaded
    est_minutes = remaining * GUTENBERG_DELAY / 60
    print(f"  [Gutenberg] Downloading books ({downloaded} exist, ~{est_minutes:.0f} min for {remaining})...")

    for book in catalog:
        if downloaded >= max_books:
            break

        book_id = book["id"]
        if str(book_id) in downloaded_ids:
            continue

        try:
            time.sleep(GUTENBERG_DELAY)
            resp = SESSION.get(book["url"], timeout=60)
            resp.raise_for_status()

            text = resp.content.decode("utf-8", errors="replace")
            cleaned = clean_gutenberg_text(text)

            if not quality_filter(cleaned, MIN_BOOK_LENGTH):
                continue

            safe_title = re.sub(r"[^\w\s-]", "", book["title"])[:80].strip()
            safe_title = re.sub(r"\s+", "_", safe_title)
            filename = f"{book_id}_{safe_title}" if safe_title else str(book_id)

            filepath = GUTENBERG_DIR / f"{filename}.txt"
            filepath.write_text(cleaned, encoding="utf-8")

            downloaded_ids.add(str(book_id))
            downloaded += 1

            if downloaded % 10 == 0:
                remaining = max_books - downloaded
                print(f"  [Gutenberg] {downloaded}/{max_books} books downloaded (~{remaining} left)...")
                progress["gutenberg_ids"] = list(downloaded_ids)
                save_progress(progress)

        except requests.RequestException as e:
            print(f"  [Gutenberg] Error downloading book {book_id}: {e}")
            continue

    print(f"  [Gutenberg] Done: {downloaded} books saved.")
    progress["gutenberg_ids"] = list(downloaded_ids)
    save_progress(progress)
    return downloaded


# ---------------------------------------------------------------------------
# FineWeb-Edu downloader (HuggingFace datasets, streaming)
# ---------------------------------------------------------------------------


def fetch_fineweb_edu(target_gb: float, progress: dict) -> int:
    """Download educational web pages from FineWeb-Edu via HuggingFace datasets.

    Streams the `sample-10BT` subset (smaller than the full 1.3T token dataset).
    Saves individual .txt files to data/pretrain/fineweb_edu/.

    Requires: pip install datasets
    Dataset: HuggingFaceFW/fineweb-edu (ODC-By license)

    Args:
        target_gb: How many GB of text to download (e.g. 10.0).
        progress: Progress dict for resume support.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [FineWeb-Edu] ERROR: 'datasets' package not found.")
        print("  [FineWeb-Edu] Install with: pip install datasets")
        return 0

    FINEWEB_DIR.mkdir(parents=True, exist_ok=True)

    # Resume support — count existing files
    existing_files = list(FINEWEB_DIR.glob("*.txt"))
    existing_bytes = sum(f.stat().st_size for f in existing_files)
    saved = len(existing_files)
    target_bytes = int(target_gb * 1024 * 1024 * 1024)

    if existing_bytes >= target_bytes:
        print(
            f"  [FineWeb-Edu] Already have {existing_bytes / 1e9:.2f} GB "
            f"({saved} files), target is {target_gb:.1f} GB. Skipping."
        )
        return saved

    remaining_gb = (target_bytes - existing_bytes) / 1e9
    print(
        f"  [FineWeb-Edu] Target: {target_gb:.1f} GB | Have: {existing_bytes / 1e9:.2f} GB "
        f"({saved} files) | Need: {remaining_gb:.2f} GB"
    )
    print("  [FineWeb-Edu] Streaming from HuggingFace (sample-10BT subset)...")
    print("  [FineWeb-Edu] Safe to Ctrl+C and --resume later.")

    # Stream the dataset — never loads full thing into memory
    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"  [FineWeb-Edu] Failed to connect to HuggingFace: {e}")
        return saved

    # Resume: skip records that were already processed in previous runs.
    # We track the number of records consumed so the stream can skip ahead.
    records_consumed = progress.get("fineweb_edu", {}).get("records_consumed", 0)
    if records_consumed > 0:
        print(f"  [FineWeb-Edu] Skipping {records_consumed:,} previously consumed records...")
        ds = ds.skip(records_consumed)

    total_bytes = existing_bytes
    batch_text = []
    batch_size = 0
    file_target = 5 * 1024 * 1024  # 5 MB per file — fewer files, faster combine
    leak_rejected = 0
    literals_sanitized = 0
    locked_guard = _locked_probe_guard("FineWeb-Edu")
    start_time = time.monotonic()
    last_print = start_time

    try:
        for record in ds:
            records_consumed += 1
            text = record.get("text", "")
            if not text or len(text) < MIN_ARTICLE_LENGTH:
                continue

            # Basic quality check — must have sentences
            if not quality_filter(text, MIN_ARTICLE_LENGTH):
                continue

            # Same screen the other HF fetchers run: the original FineWeb-Edu
            # pull predates both the literal scrub and the probe seal, and a
            # re-pull is 36% of the diet -- unscreened, it could hand the
            # pretrain a sealed probe's text verbatim.
            text, _hits = _sanitize_special_literals(text)
            if locked_guard is not None and locked_guard.leaks(text):
                leak_rejected += 1
                continue
            # count AFTER the screen: only hits that reached disk
            literals_sanitized += _hits

            text_bytes = len(text.encode("utf-8"))
            batch_text.append(text)
            batch_size += text_bytes

            # Write batch when it hits ~5 MB
            if batch_size >= file_target:
                file_idx = saved
                filepath = FINEWEB_DIR / f"fineweb_{file_idx:08d}.txt"
                filepath.write_text(DOC_SEP_JOIN.join(batch_text), encoding="utf-8", newline="\n")
                saved += 1
                total_bytes += batch_size
                batch_text = []
                batch_size = 0

                # Check if we've hit the target
                if total_bytes >= target_bytes:
                    break

                # Progress every 60 seconds
                now = time.monotonic()
                if now - last_print >= 60:
                    elapsed_min = (now - start_time) / 60
                    speed_mb = (total_bytes - existing_bytes) / max(now - start_time, 1) / 1e6
                    remaining_bytes = target_bytes - total_bytes
                    eta_min = remaining_bytes / max(speed_mb * 1e6, 1) / 60
                    print(
                        f"  [FineWeb-Edu] {total_bytes / 1e9:.2f}/{target_gb:.1f} GB "
                        f"({saved} files) | {speed_mb:.1f} MB/s | "
                        f"~{eta_min:.0f} min left"
                    )
                    last_print = now

                    # Save progress periodically
                    progress["fineweb_edu"] = {
                        "files_saved": saved,
                        "bytes_written": total_bytes,
                        "records_consumed": records_consumed,
                    }
                    save_progress(progress)

        # Write any remaining batch
        if batch_text:
            filepath = FINEWEB_DIR / f"fineweb_{saved:08d}.txt"
            filepath.write_text(DOC_SEP_JOIN.join(batch_text), encoding="utf-8", newline="\n")
            saved += 1
            total_bytes += batch_size

    except KeyboardInterrupt:
        # Write partial batch on interrupt
        if batch_text:
            filepath = FINEWEB_DIR / f"fineweb_{saved:08d}.txt"
            filepath.write_text(DOC_SEP_JOIN.join(batch_text), encoding="utf-8", newline="\n")
            saved += 1
            total_bytes += batch_size
        elapsed_min = (time.monotonic() - start_time) / 60
        print(
            f"\n  [FineWeb-Edu] Interrupted after {elapsed_min:.0f} min. "
            f"{saved} files saved ({total_bytes / 1e9:.2f} GB). "
            f"Run with --resume to continue."
        )
    except Exception as e:
        # Flush the pending batch: records_consumed is saved below, and
        # consumed-but-unwritten records would be skipped forever on resume.
        if batch_text:
            filepath = FINEWEB_DIR / f"fineweb_{saved:08d}.txt"
            filepath.write_text(DOC_SEP_JOIN.join(batch_text), encoding="utf-8", newline="\n")
            saved += 1
            total_bytes += batch_size
        print(f"  [FineWeb-Edu] Error: {e}")
        print("  [FineWeb-Edu] Run with --resume to retry.")

    elapsed_min = (time.monotonic() - start_time) / 60
    print(f"\n  [FineWeb-Edu] Done: {saved} files saved ({total_bytes / 1e9:.2f} GB) in {elapsed_min:.0f} min.")
    if leak_rejected:
        print(f"  [FineWeb-Edu] {leak_rejected:,} records dropped by the sealed-probe screen")
    if literals_sanitized:
        print(f"  [FineWeb-Edu] {literals_sanitized:,} special-token literal(s) space-broken")

    progress["fineweb_edu"] = {
        "files_saved": saved,
        "bytes_written": total_bytes,
        "records_consumed": records_consumed,
    }
    save_progress(progress)
    return saved


# ---------------------------------------------------------------------------
# Stack Exchange downloader (archive.org 7z dumps)
# ---------------------------------------------------------------------------

# High-quality SE sites worth downloading (not StackOverflow — it's 80+ GB)
STACKEX_SITES = [
    "english",  # English language & usage (~350 MB 7z)
    "math",  # Mathematics (~800 MB 7z)
    "physics",  # Physics (~400 MB 7z)
    "philosophy",  # Philosophy (~50 MB 7z)
    "history",  # History (~100 MB 7z)
    "writing",  # Writers & writing (~50 MB 7z)
    "worldbuilding",  # Worldbuilding (~200 MB 7z)
    "scifi",  # Science Fiction & Fantasy (~200 MB 7z)
    "literature",  # Literature (~30 MB 7z)
    "cooking",  # Cooking (~100 MB 7z)
    "travel",  # Travel (~100 MB 7z)
    "music",  # Music (~80 MB 7z)
    "biology",  # Biology (~150 MB 7z)
    "chemistry",  # Chemistry (~150 MB 7z)
    "astronomy",  # Astronomy (~60 MB 7z)
    "psychology",  # Psychology (~30 MB 7z)
    "law",  # Law (~100 MB 7z)
    "economics",  # Economics (~50 MB 7z)
    "politics",  # Politics (~80 MB 7z)
    "linguistics",  # Linguistics (~50 MB 7z)
    "cs",  # Computer Science (~200 MB 7z)
    "stats",  # Cross Validated / Statistics (~400 MB 7z)
]


def _clean_html_body(html_body: str) -> str:
    """Clean HTML from Stack Exchange post body to plain text."""
    # Unescape HTML entities
    text = html_mod.unescape(html_body)
    # Remove code blocks entirely (noisy for language training)
    text = re.sub(r"<pre><code>.*?</code></pre>", "", text, flags=re.DOTALL)
    text = re.sub(r"<code>.*?</code>", "", text, flags=re.DOTALL)
    # Remove all HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_stackexchange(sites: list[str], min_score: int, progress: dict) -> int:
    """Download Stack Exchange data dumps from archive.org.

    Downloads 7z archives containing Posts.xml for each site, extracts
    questions and answers with score >= min_score, cleans HTML, and saves
    as plain text files.

    Requires: pip install py7zr

    Args:
        sites: List of SE site names (e.g. ["english", "math"]).
        min_score: Minimum post score to include.
        progress: Progress dict for resume support.
    """
    try:
        import py7zr
    except ImportError:
        print("  [StackExchange] ERROR: 'py7zr' package not found.")
        print("  [StackExchange] Install with: pip install py7zr")
        return 0

    STACKEX_DIR.mkdir(parents=True, exist_ok=True)
    total_saved = 0
    completed_sites = set(progress.get("stackex_completed", []))

    for site_name in sites:
        site_dir = STACKEX_DIR / site_name
        if site_name in completed_sites:
            if site_dir.exists():
                count = len(list(site_dir.glob("*.txt")))
                total_saved += count
                print(f"  [StackExchange] {site_name}: already done ({count} posts)")
            continue

        # Stack Exchange dumps are at archive.org
        dump_url = f"https://archive.org/download/stackexchange/{site_name}.stackexchange.com.7z"
        archive_path = STACKEX_DIR / f"{site_name}.7z"

        print(f"\n  [StackExchange] Downloading {site_name}.stackexchange.com.7z...")

        # Download with resume support
        try:
            headers = {}
            mode = "wb"
            existing_size = 0
            if archive_path.exists():
                existing_size = archive_path.stat().st_size
                if existing_size > 0:
                    headers["Range"] = f"bytes={existing_size}-"
                    mode = "ab"

            resp = SESSION.get(dump_url, headers=headers, stream=True, timeout=60)

            if resp.status_code == 416:
                print(f"  [StackExchange] {site_name}: download already complete")
            elif resp.status_code == 404:
                print(f"  [StackExchange] {site_name}: not found on archive.org, skipping")
                continue
            else:
                resp.raise_for_status()
                # A server that ignores Range answers 200 with the FULL
                # body; appending it would corrupt the archive. Start over.
                if headers.get("Range") and resp.status_code != 206:
                    print(f"  [StackExchange] {site_name}: server ignored resume -- restarting from zero.")
                    mode = "wb"
                    existing_size = 0
                total_dl = existing_size
                with open(archive_path, mode) as f:
                    for chunk in resp.iter_content(chunk_size=10 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            total_dl += len(chunk)
                print(f"  [StackExchange] {site_name}: downloaded ({total_dl / 1e6:.0f} MB)")

        except requests.RequestException as e:
            print(f"  [StackExchange] {site_name}: download error: {e}")
            print("  [StackExchange] Run with --resume to retry.")
            continue

        # Extract Posts.xml from 7z archive
        print(f"  [StackExchange] {site_name}: extracting Posts.xml...")
        site_dir.mkdir(parents=True, exist_ok=True)
        posts_xml_path = STACKEX_DIR / f"{site_name}_Posts.xml"

        try:
            with py7zr.SevenZipFile(str(archive_path), mode="r") as z:
                # Extract only Posts.xml
                names = z.getnames()
                posts_name = None
                for n in names:
                    if n.lower() == "posts.xml":
                        posts_name = n
                        break
                if posts_name is None:
                    print(f"  [StackExchange] {site_name}: no Posts.xml found, skipping")
                    continue
                z.extract(path=str(STACKEX_DIR), targets=[posts_name])
                # py7zr extracts to STACKEX_DIR/Posts.xml
                extracted = STACKEX_DIR / posts_name
                if extracted.exists() and extracted != posts_xml_path:
                    extracted.rename(posts_xml_path)
        except Exception as e:
            print(f"  [StackExchange] {site_name}: extraction error: {e}")
            continue

        # Parse Posts.xml — stream parse for memory efficiency
        print(f"  [StackExchange] {site_name}: parsing posts (score >= {min_score})...")
        saved = 0
        questions: dict[str, str] = {}  # id -> title, for context

        try:
            for event, elem in ET.iterparse(str(posts_xml_path), events=("end",)):  # noqa: S314  -- parses a downloaded StackExchange data dump (trusted source); use defusedxml if untrusted
                if elem.tag != "row":
                    continue

                post_type = elem.get("PostTypeId", "")
                score = int(elem.get("Score", "0"))
                body = elem.get("Body", "")
                title = elem.get("Title", "")
                post_id = elem.get("Id", "")

                # Track question titles for answer context
                if post_type == "1" and title:
                    questions[post_id] = title

                # Filter: only questions and answers with high scores
                if post_type not in ("1", "2") or score < min_score:
                    elem.clear()
                    continue

                if not body:
                    elem.clear()
                    continue

                # Clean HTML
                cleaned = _clean_html_body(body)
                if not quality_filter(cleaned, MIN_ARTICLE_LENGTH):
                    elem.clear()
                    continue

                # Build text with context
                if post_type == "1" and title:
                    text = f"Q: {title}\n\n{cleaned}"
                elif post_type == "2":
                    parent_id = elem.get("ParentId", "")
                    parent_title = questions.get(parent_id, "")
                    if parent_title:
                        text = f"Q: {parent_title}\n\nA: {cleaned}"
                    else:
                        text = f"A: {cleaned}"
                else:
                    text = cleaned

                filepath = site_dir / f"{site_name}_{post_id}.txt"
                filepath.write_text(text, encoding="utf-8")
                saved += 1

                if saved % 1000 == 0:
                    print(f"  [StackExchange] {site_name}: {saved} posts saved...")

                elem.clear()

        except ET.ParseError as e:
            print(f"  [StackExchange] {site_name}: XML parse error: {e}")
            # A mid-file parse error means only part of the dump was read:
            # the site is NOT complete. Keep the archive so a rerun can
            # re-extract instead of freezing the partial data forever.
            total_saved += saved
            try:
                if posts_xml_path.exists():
                    posts_xml_path.unlink()
            except OSError:
                pass
            print(f"  [StackExchange] {site_name}: {saved} posts saved (INCOMPLETE -- rerun to retry)")
            continue

        total_saved += saved
        completed_sites.add(site_name)
        progress["stackex_completed"] = list(completed_sites)
        save_progress(progress)

        print(f"  [StackExchange] {site_name}: {saved} posts saved")

        # Clean up archive and XML to save disk space
        try:
            if posts_xml_path.exists():
                posts_xml_path.unlink()
            if archive_path.exists():
                archive_path.unlink()
        except OSError:
            pass

    print(f"\n  [StackExchange] Total: {total_saved} posts saved across {len(completed_sites)} sites")
    return total_saved


# ---------------------------------------------------------------------------
# Wayback Machine downloader (Internet Archive CDX API)
# ---------------------------------------------------------------------------

# High-quality educational domains to fetch from Wayback Machine.
# These are freely accessible sites with substantial text content.
WAYBACK_DOMAINS = [
    # Philosophy encyclopedias (excellent long-form writing)
    ("plato.stanford.edu/entries/*", "Stanford Encyclopedia of Philosophy"),
    ("iep.utm.edu/*", "Internet Encyclopedia of Philosophy"),
    # Educational reference
    ("www.britannica.com/topic/*", "Britannica Topics"),
    ("www.britannica.com/science/*", "Britannica Science"),
    ("www.britannica.com/biography/*", "Britannica Biographies"),
    # History & humanities
    ("www.history.com/topics/*", "History.com Topics"),
    ("www.ancient.eu/article/*", "World History Encyclopedia"),
    # Science communication
    ("www.scientificamerican.com/article/*", "Scientific American"),
]


def _extract_text_from_html(html_content: str) -> str:
    """Extract readable text from HTML page content."""
    # Remove scripts and styles
    text = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove nav, header, footer, sidebar elements
    for tag in ["nav", "header", "footer", "aside", "menu"]:
        text = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Unescape HTML entities
    text = html_mod.unescape(text)
    # Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Clean up lines
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if len(line) < 30:
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def fetch_wayback(domains: list[tuple[str, str]], max_pages: int, progress: dict) -> int:
    """Download pages from the Wayback Machine via CDX API.

    Uses the Internet Archive's CDX API to find archived versions of
    pages from specific high-quality domains, then fetches and extracts
    the text content.

    Args:
        domains: List of (url_pattern, label) tuples.
        max_pages: Maximum total pages to download across all domains.
        progress: Progress dict for resume support.
    """
    WAYBACK_DIR.mkdir(parents=True, exist_ok=True)

    existing = set(f.stem for f in WAYBACK_DIR.glob("*.txt"))
    total_saved = len(existing)
    wayback_done = set(progress.get("wayback_done_domains", []))

    if total_saved >= max_pages:
        print(f"  [Wayback] Already have {total_saved}/{max_pages} pages, skipping.")
        return total_saved

    cdx_base = "https://web.archive.org/cdx/search/cdx"

    for url_pattern, label in domains:
        if total_saved >= max_pages:
            break
        if url_pattern in wayback_done:
            print(f"  [Wayback] {label}: already done, skipping")
            continue

        print(f"\n  [Wayback] Searching: {label} ({url_pattern})...")

        # Query CDX API for this domain
        try:
            cdx_params = {
                "url": url_pattern,
                "output": "json",
                "fl": "timestamp,original",
                "collapse": "urlkey",  # Deduplicate by URL
                "limit": min(max_pages - total_saved, 2000),
                "filter": "statuscode:200",
                "mimetype": "text/html",
            }
            resp = SESSION.get(cdx_base, params=cdx_params, timeout=60)
            resp.raise_for_status()
            results = resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"  [Wayback] {label}: CDX query failed: {e}")
            continue

        if len(results) <= 1:  # First row is headers
            print(f"  [Wayback] {label}: no results found")
            continue

        # Skip header row
        urls = results[1:]
        print(f"  [Wayback] {label}: found {len(urls)} archived pages")

        domain_saved = 0
        domain_tried = 0
        skip_dup = 0
        skip_http = 0
        skip_quality = 0
        skip_ai = 0
        consecutive_fails = 0
        for row in urls:
            if total_saved >= max_pages:
                break

            if len(row) < 2:
                continue
            timestamp, original_url = row[0], row[1]
            # Create safe filename from URL
            safe_name = re.sub(r"[^\w]", "_", original_url)[:120]
            safe_name = re.sub(r"_+", "_", safe_name).strip("_")

            if safe_name in existing:
                skip_dup += 1
                continue

            domain_tried += 1

            # Fetch the archived page (id_ = raw content, no Wayback toolbar)
            wayback_url = f"https://web.archive.org/web/{timestamp}id_/{original_url}"
            try:
                delay = WAYBACK_DELAY
                # Back off on consecutive HTTP failures
                if consecutive_fails >= 5:
                    delay = min(WAYBACK_DELAY * 3, 10.0)
                time.sleep(delay)
                page_resp = SESSION.get(wayback_url, timeout=30)
                if page_resp.status_code == 429:
                    # Rate limited — back off and retry later
                    consecutive_fails += 1
                    skip_http += 1
                    if consecutive_fails >= MAX_RETRIES:
                        print(f"  [Wayback] {label}: rate-limited {MAX_RETRIES} times, moving on")
                        break
                    time.sleep(10)
                    continue
                if page_resp.status_code != 200:
                    skip_http += 1
                    consecutive_fails += 1
                    continue
                page_html = page_resp.text
                consecutive_fails = 0  # Reset on success
            except requests.RequestException:
                skip_http += 1
                consecutive_fails += 1
                continue

            # Extract text
            text = _extract_text_from_html(page_html)
            if not quality_filter(text, MIN_ARTICLE_LENGTH):
                skip_quality += 1
                continue

            # Skip AI-generated content
            if detect_ai_content(text):
                skip_ai += 1
                continue

            filepath = WAYBACK_DIR / f"{safe_name}.txt"
            filepath.write_text(text, encoding="utf-8")
            existing.add(safe_name)
            total_saved += 1
            domain_saved += 1

            if domain_saved % 50 == 0:
                print(f"  [Wayback] {label}: {domain_saved} pages saved...")

            # Progress log every 100 attempts
            if domain_tried % 100 == 0:
                print(
                    f"  [Wayback] {label}: tried {domain_tried}/"
                    f"{len(urls)} -- saved {domain_saved}, "
                    f"skip: dup={skip_dup} http={skip_http} "
                    f"quality={skip_quality} ai={skip_ai}"
                )

        wayback_done.add(url_pattern)
        progress["wayback_done_domains"] = list(wayback_done)
        save_progress(progress)
        print(
            f"  [Wayback] {label}: {domain_saved} pages saved "
            f"(tried {domain_tried}, skip: dup={skip_dup} "
            f"http={skip_http} quality={skip_quality} "
            f"ai={skip_ai})"
        )

    print(f"\n  [Wayback] Total: {total_saved} pages saved")
    return total_saved


# ---------------------------------------------------------------------------
# Fandom wiki downloader (MediaWiki API)
# ---------------------------------------------------------------------------

# Curated seed list of well-known Fandom wiki subdomains.  The discovery
# function merges these with the interwiki map from community.fandom.com
# and probes each for article count so we only crawl wikis that have real
# content.
_FANDOM_SEED_WIKIS = [
    # Games
    "minecraft",
    "pokemon",
    "genshin-impact",
    "valorant",
    "fortnite",
    "leagueoflegends",
    "overwatch",
    "terraria",
    "stardewvalley",
    "animalcrossing",
    "roblox",
    "warframe",
    "borderlands",
    "cyberpunk",
    "hollow-knight",
    "elden-ring",
    "baldursgate3",
    "palworld",
    "subnautica",
    "satisfactory",
    "ark",
    "halo",
    "callofduty",
    "battlefield",
    "residentevil",
    "silenthill",
    "metalgear",
    "finalfantasy",
    "kingdomhearts",
    "persona",
    "fire-emblem",
    "xenoblade",
    "splatoon",
    "kirby",
    "megaman",
    "castlevania",
    "doom",
    "monsterhunter",
    "civilization",
    "starcraft",
    "diablo",
    "worldofwarcraft",
    "runescape",
    "guildwars2",
    "dota2",
    "apexlegends",
    "factorio",
    "rimworld",
    "stellaris",
    "citiesskylines",
    "sims",
    "assassinscreed",
    "farcry",
    "grandtheftauto",
    "reddeadredemption",
    "godofwar",
    "horizon",
    "uncharted",
    "lastofus",
    "ratchetandclank",
    "crash-bandicoot",
    "spyro",
    "sonic",
    "donkeykong",
    "metroid",
    "smashbros",
    "bayonetta",
    "devilmaycry",
    "nierautomata",
    "yakuza",
    "dragonquest",
    "bioshock",
    "dishonored",
    "halflife",
    "portal",
    "left4dead",
    "teamfortress",
    "counterstrike",
    "deadbydaylight",
    "fnaf",
    "undertale",
    "deltarune",
    "cuphead",
    "celeste",
    "no-mans-sky",
    "starfield",
    "outerworlds",
    "fallout",
    "elderscrolls",
    "darksouls",
    "bloodborne",
    "sekiro",
    "streetfighter",
    "tekken",
    "mortal-kombat",
    "guilty-gear",
    "destiny",
    "lethal-company",
    "wowwiki",
    "dnd5e",
    "forgottenrealms",
    "pathfinder",
    "warhammer40k",
    "warhammer",
    # Anime/Manga
    "naruto",
    "onepiece",
    "dragonball",
    "bleach",
    "attackontitan",
    "myheroacademia",
    "demonslayer",
    "jujutsukaisen",
    "hunterxhunter",
    "fairytail",
    "souleater",
    "swordartonline",
    "tokyoghoul",
    "deathnote",
    "fullmetalalchemist",
    "evangelion",
    "gundam",
    "sailor-moon",
    "inuyasha",
    "cowboy-bebop",
    "steins-gate",
    "konosuba",
    "re-zero",
    "one-punch-man",
    "vinland-saga",
    "berserk",
    "spy-x-family",
    "chainsaw-man",
    "gintama",
    "fate",
    "typemoon",
    "touhou",
    "digimon",
    "yugioh",
    "haikyu",
    "detective-conan",
    "studio-ghibli",
    "arknights",
    "honkaiimpact3",
    "honkai-star-rail",
    "azurlane",
    # Movies/TV
    "pixar",
    "disney",
    "mcu",
    "jurassicpark",
    "aliens",
    "terminator",
    "matrix",
    "pirates",
    "starwars",
    "startrek",
    "memory-alpha",
    "tardis",
    "walkingdead",
    "strangerthings",
    "breakingbad",
    "simpsons",
    "familyguy",
    "southpark",
    "rickandmorty",
    "adventuretime",
    "steven-universe",
    "gravity-falls",
    "avatar",
    "korra",
    "owl-house",
    "amphibia",
    "transformers",
    "powerrangers",
    "buffy",
    "xfiles",
    "supernatural",
    "gameofthrones",
    "lego",
    "spongebob",
    # Comics
    "marvel",
    "dc",
    "tmnt",
    "invincible",
    "the-boys",
    "sonic-the-hedgehog",
    # Books
    "harrypotter",
    "lotr",
    "asoiaf",
    "cosmere",
    "wot",
    "percy-jackson",
    "warriors",
    "discworld",
    "dune",
    "hunger-games",
    "shadowhunters",
    "narnia",
    "witcher",
    "dragonage",
    "masseffect",
    # Other
    "kpop",
    "vocaloid",
    "mythology",
    "recipes",
    "creepypasta",
    "scp",
    "backrooms",
]

# Wikis to skip during discovery (meta wikis, duplicates, non-content)
_FANDOM_SKIP = {
    "community",
    "search",
    "scratchpad",
    "armchairgm",
    "conworld",
    "future",  # user-generated fiction, not factual content
}

# Minimum articles for a wiki to be worth crawling
_FANDOM_MIN_ARTICLES = 500

# Minimum article length for Fandom (fan wiki articles can be short stubs)
MIN_FANDOM_LENGTH = 1500  # characters — skip stubs and one-liner entries


def discover_fandom_wikis() -> list[tuple[str, str]]:
    """Discover English Fandom wikis via interwiki map + curated seeds.

    Probes each candidate to get article count and site name.  Returns
    wikis with >= _FANDOM_MIN_ARTICLES articles, sorted by article count
    descending.  Deduplicates wikis that resolve to the same site.
    """
    print("  [Fandom] Discovering wikis...")

    # 1. Get interwiki map from community wiki
    iw_wikis: set[str] = set()
    try:
        r = SESSION.get(
            "https://community.fandom.com/api.php",
            params={
                "action": "query",
                "format": "json",
                "meta": "siteinfo",
                "siprop": "interwikimap",
            },
            timeout=15,
        )
        if r.status_code == 200:
            for entry in r.json().get("query", {}).get("interwikimap", []):
                m = re.match(r"https://([a-zA-Z0-9_-]+)\.fandom\.com", entry.get("url", ""))
                if m:
                    iw_wikis.add(m.group(1).lower())
    except (requests.RequestException, json.JSONDecodeError):
        pass

    # 2. Merge with curated seeds
    candidates = iw_wikis | set(w.lower() for w in _FANDOM_SEED_WIKIS)
    candidates -= _FANDOM_SKIP
    print(f"  [Fandom] {len(candidates)} candidate wikis to probe")

    # 3. Probe each for article count (concurrent, 8 threads)
    results: list[tuple[str, str, int]] = []

    def _probe(wiki_id: str):
        try:
            r = SESSION.get(
                f"https://{wiki_id}.fandom.com/api.php",
                params={
                    "action": "query",
                    "format": "json",
                    "meta": "siteinfo",
                    "siprop": "statistics|general",
                },
                timeout=8,
            )
            if r.status_code == 200:
                d = r.json()
                stats = d.get("query", {}).get("statistics", {})
                general = d.get("query", {}).get("general", {})
                articles = stats.get("articles", 0)
                sitename = general.get("sitename", wiki_id)
                lang = general.get("lang", "en")
                if lang == "en" and articles >= _FANDOM_MIN_ARTICLES:
                    return (wiki_id, sitename, articles)
        except (requests.RequestException, json.JSONDecodeError):
            pass
        return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_probe, wid): wid for wid in candidates}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # 4. Deduplicate wikis that share the same sitename (keep highest article count)
    seen_sites: dict[str, tuple[str, str, int]] = {}
    for wiki_id, sitename, articles in results:
        key = sitename.lower().strip()
        if key not in seen_sites or articles > seen_sites[key][2]:
            seen_sites[key] = (wiki_id, sitename, articles)

    deduped = sorted(seen_sites.values(), key=lambda x: -x[2])
    total_articles = sum(a for _, _, a in deduped)
    print(f"  [Fandom] Found {len(deduped)} wikis with {total_articles:,} total articles")

    return [(wid, name) for wid, name, _ in deduped]


def fetch_fandom(wikis: list[tuple[str, str]], max_articles: int, progress: dict) -> int:
    """Download articles from Fandom wikis via MediaWiki revisions API.

    Uses ``prop=revisions`` (not ``prop=extracts`` which Fandom disabled)
    to get raw wikitext, then strips markup to plain text.  Fetches from
    multiple wikis concurrently (different domains = no rate-limit conflict).

    Args:
        wikis: List of (wiki_subdomain, label) tuples.
        max_articles: Maximum total articles across all wikis.  0 = unlimited.
        progress: Progress dict for resume support.
    """
    FANDOM_DIR.mkdir(parents=True, exist_ok=True)

    existing = set(f.stem for f in FANDOM_DIR.glob("*.txt"))
    total_saved_start = len(existing)

    if max_articles > 0 and total_saved_start >= max_articles:
        print(f"  [Fandom] Already have {total_saved_start}/{max_articles} articles, skipping.")
        return total_saved_start

    fandom_done = set(progress.get("fandom_done_wikis", []))
    remaining = [(wid, lab) for wid, lab in wikis if wid not in fandom_done]

    if not remaining:
        print(f"  [Fandom] All wikis already done ({total_saved_start} articles on disk).")
        return total_saved_start

    # Shared mutable state protected by lock
    lock = threading.Lock()
    shared = {
        "total_saved": total_saved_start,
        "ai_rejected": 0,
        "existing": existing,
    }

    def _fetch_wiki(wiki_id: str, label: str) -> int:
        """Fetch all articles from one Fandom wiki.  Returns count saved."""
        api_url = f"https://{wiki_id}.fandom.com/api.php"
        print(f"\n  [Fandom] Fetching from {label} ({wiki_id}.fandom.com)...")

        wiki_saved = 0
        ai_rejected = 0
        ap_continue = None
        consecutive_errors = 0

        while True:
            # Check global cap
            if max_articles > 0:
                with lock:
                    if shared["total_saved"] >= max_articles:
                        break

            # --- Step 1: list page titles via allpages ---
            params: dict = {
                "action": "query",
                "format": "json",
                "list": "allpages",
                "aplimit": 50,
                "apnamespace": 0,
                "apfilterredir": "nonredirects",
            }
            if ap_continue:
                params["apcontinue"] = ap_continue

            try:
                resp = SESSION.get(api_url, params=params, timeout=15)
                if resp.status_code == 429:
                    consecutive_errors += 1
                    if consecutive_errors >= 5:
                        print(f"  [Fandom] {label}: rate-limited too many times, moving on")
                        break
                    time.sleep(10)
                    continue
                resp.raise_for_status()
                data = resp.json()
                consecutive_errors = 0
            except (requests.RequestException, json.JSONDecodeError) as e:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    print(f"  [Fandom] {label}: too many errors ({e}), moving on")
                    break
                time.sleep(3)
                continue

            pages = data.get("query", {}).get("allpages", [])
            if not pages:
                break

            next_continue = data.get("continue", {}).get("apcontinue")
            titles = [p["title"] for p in pages]

            # --- Step 2: fetch raw wikitext via revisions ---
            time.sleep(FANDOM_DELAY)
            try:
                rev_resp = SESSION.get(
                    api_url,
                    params={
                        "action": "query",
                        "format": "json",
                        "titles": "|".join(titles[:50]),
                        "prop": "revisions",
                        "rvprop": "content",
                        "rvslots": "main",
                    },
                    timeout=30,
                )
                if rev_resp.status_code == 429:
                    consecutive_errors += 1
                    if consecutive_errors >= 5:
                        ap_continue = next_continue
                    time.sleep(10)
                    continue
                rev_resp.raise_for_status()
                rev_data = rev_resp.json()
                consecutive_errors = 0
            except (requests.RequestException, json.JSONDecodeError):
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    ap_continue = next_continue
                time.sleep(3)
                continue

            # Advance pagination only after content fetch succeeds
            ap_continue = next_continue

            result_pages = rev_data.get("query", {}).get("pages", {})
            for page_id, page_info in result_pages.items():
                if max_articles > 0:
                    with lock:
                        if shared["total_saved"] >= max_articles:
                            break

                title = page_info.get("title", "")
                revisions = page_info.get("revisions", [])
                if not revisions:
                    continue
                wikitext = revisions[0].get("slots", {}).get("main", {}).get("*", "")
                if not wikitext or len(wikitext) < 200:
                    continue

                # Skip redirects
                if wikitext.strip().upper().startswith("#REDIRECT"):
                    continue

                # Skip list/category/disambiguation pages
                title_lower = title.lower()
                if any(
                    skip in title_lower
                    for skip in [
                        "list of",
                        "lists of",
                        "category:",
                        "template:",
                        "disambiguation",
                        "user:",
                        "talk:",
                    ]
                ):
                    continue

                # Wikitext → plain text
                cleaned = strip_wikitext(wikitext)
                cleaned = clean_wiki_text(cleaned)

                if not quality_filter(cleaned, MIN_FANDOM_LENGTH):
                    continue

                if detect_ai_content(cleaned):
                    ai_rejected += 1
                    continue

                title_clean = re.sub(r"[^\w\s-]", "", title)[:80].strip()
                if not title_clean:
                    continue
                safe_name = re.sub(r"\s+", "_", f"{wiki_id}_{title_clean}")

                with lock:
                    if not safe_name or safe_name in shared["existing"]:
                        continue
                    shared["existing"].add(safe_name)
                    shared["total_saved"] += 1

                filepath = FANDOM_DIR / f"{safe_name}.txt"
                filepath.write_text(cleaned, encoding="utf-8")
                wiki_saved += 1

            if wiki_saved and wiki_saved % 200 == 0:
                print(f"  [Fandom] {label}: {wiki_saved} articles saved...")

            if not ap_continue:
                break  # No more pages in this wiki

            time.sleep(FANDOM_DELAY)

        # Mark wiki done
        with lock:
            fandom_done.add(wiki_id)
            progress["fandom_done_wikis"] = list(fandom_done)
            save_progress(progress)
            shared["ai_rejected"] += ai_rejected

        print(f"  [Fandom] {label}: {wiki_saved} articles saved")
        return wiki_saved

    # Fetch wikis concurrently — each is a different domain
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_fetch_wiki, wid, lab): (wid, lab) for wid, lab in remaining}
        for future in as_completed(futures):
            wid, lab = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  [Fandom] {lab}: error -- {e}")

    total = shared["total_saved"]
    if shared["ai_rejected"]:
        print(f"  [Fandom] {shared['ai_rejected']} articles rejected by AI content filter")
    print(f"\n  [Fandom] Total: {total} articles saved")
    return total


# ---------------------------------------------------------------------------
# Generic HuggingFace streaming fetcher
# ---------------------------------------------------------------------------

# The tokenizer carves multi-char special-token literals out of PLAIN TEXT: a
# source file containing "List<A>" or "text <s>struck</s>" would write real
# control ids (turn markers, BOS, EOS mid-document) into the corpus. Web text
# is nearly clean; source code is full of "<A>" generics and literal tags, so
# every saved text is scrubbed. Must list every multi-char special the live
# vocabs reserve (ids 0-13 in v2); test-pinned against the vocab file.
_SPECIAL_TOKEN_LITERALS = (
    "<pad>", "<s>", "</s>", "<unk>", "<sep>", "<mask>", "<Q>", "<A>",
    "<USER>", "<BOT>", "<think>", "</think>", "<search>", "</search>",
)


def _sanitize_special_literals(text: str) -> tuple[str, int]:
    """Space-break exact special-token literals ("<A>" -> "< A>") so the
    tokenizer cannot carve them into control ids. Returns (text, hits)."""
    hits = 0
    for lit in _SPECIAL_TOKEN_LITERALS:
        if lit in text:
            hits += text.count(lit)
            text = text.replace(lit, lit[0] + " " + lit[1:])
    # The packed-file record separator must never occur INSIDE a record --
    # a stray U+001E would forge a document boundary at pretokenize time.
    # Not counted as a literal hit: it is format armor, not token armor.
    if DOC_SEP_CHAR in text:
        text = text.replace(DOC_SEP_CHAR, " ")
    return text, hits


def _locked_probe_guard(label: str):
    """The sealed-probe screen for collected text. The pretrain path has no
    consume-time guard (pretokenize reads whatever is on disk), so the screen
    lives here, at collection time. None when no seal exists to screen against
    -- said out loud, never silently."""
    try:
        from eval_leak_guard import LockedProbeGuard
    except ImportError:
        print(f"  [{label}] WARN: eval_leak_guard not importable -- collected "
              "text is NOT screened against the sealed probes")
        return None
    guard = LockedProbeGuard.load()
    if not guard:
        print(f"  [{label}] NOTE: no sealed probe manifest on this checkout -- "
              "collected text is not screened")
        return None
    return guard


def _fetch_hf_streaming(
    dataset_name: str,
    config_name: str | None,
    text_field: str,
    output_dir: Path,
    target_gb: float,
    label: str,
    progress: dict,
    progress_key: str,
    min_length: int = MIN_ARTICLE_LENGTH,
    filter_ai: bool = True,
) -> int:
    """Stream a HuggingFace dataset to disk with resume support.

    Generic version of the FineWeb-Edu pattern. Streams records, applies
    quality + AI content filters, batches into ~5 MB files.

    Args:
        dataset_name: HuggingFace dataset ID (e.g. "Skylion007/openwebtext").
        config_name: Dataset config/subset name, or None for default.
        text_field: Name of the text field in each record.
        output_dir: Directory to save .txt files.
        target_gb: How many GB of text to download.
        label: Human-readable label for progress messages.
        progress: Progress dict for resume support.
        progress_key: Key in progress dict for this source.
        min_length: Minimum text length in characters.
        filter_ai: Whether to apply AI content detection filter.

    Returns:
        Number of files saved.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print(f"  [{label}] ERROR: 'datasets' package not found.")
        print(f"  [{label}] Install with: pip install datasets")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume support -- count THIS config's files only, by filename prefix.
    # Two configs sharing one directory (the finemath 50/50 pair) otherwise
    # count each other's bytes: the second config sees the first's files,
    # reads its own target as already met, and delivers nothing.
    prefix = label.lower().replace(" ", "_").replace("-", "_")
    existing_files = list(output_dir.glob(f"{prefix}_*.txt"))
    existing_bytes = sum(f.stat().st_size for f in existing_files)
    saved = len(existing_files)
    target_bytes = int(target_gb * 1024 * 1024 * 1024)

    if existing_bytes >= target_bytes:
        print(
            f"  [{label}] Already have {existing_bytes / 1e9:.2f} GB "
            f"({saved} files), target is {target_gb:.1f} GB. Skipping."
        )
        return saved

    remaining_gb = (target_bytes - existing_bytes) / 1e9
    print(
        f"  [{label}] Target: {target_gb:.1f} GB | Have: {existing_bytes / 1e9:.2f} GB "
        f"({saved} files) | Need: {remaining_gb:.2f} GB"
    )
    print(f"  [{label}] Streaming from HuggingFace ({dataset_name})...")
    print(f"  [{label}] Safe to Ctrl+C and --resume later.")

    # Stream the dataset — never loads full thing into memory
    try:
        load_kwargs: dict = {
            "path": dataset_name,
            "split": "train",
            "streaming": True,
        }
        if config_name is not None:
            load_kwargs["name"] = config_name
        ds = load_dataset(**load_kwargs)
    except Exception as e:
        print(f"  [{label}] Failed to connect to HuggingFace: {e}")
        return saved

    # Resume: skip previously consumed records
    records_consumed = progress.get(progress_key, {}).get("records_consumed", 0)
    if records_consumed > 0:
        print(f"  [{label}] Skipping {records_consumed:,} previously consumed records...")
        ds = ds.skip(records_consumed)

    total_bytes = existing_bytes
    batch_text: list[str] = []
    batch_size = 0
    file_target = 5 * 1024 * 1024  # 5 MB per file
    ai_rejected = 0
    leak_rejected = 0
    literals_sanitized = 0
    locked_guard = _locked_probe_guard(label)
    start_time = time.monotonic()
    last_print = start_time

    try:
        for record in ds:
            records_consumed += 1
            text = record.get(text_field, "")
            if not text or len(text) < min_length:
                continue

            if not quality_filter(text, min_length):
                continue

            if filter_ai and detect_ai_content(text):
                ai_rejected += 1
                continue

            text, _hits = _sanitize_special_literals(text)
            if locked_guard is not None and locked_guard.leaks(text):
                leak_rejected += 1
                continue
            # count AFTER the screen: only hits that reached disk
            literals_sanitized += _hits

            text_bytes = len(text.encode("utf-8"))
            batch_text.append(text)
            batch_size += text_bytes

            # Write batch when it hits ~5 MB
            if batch_size >= file_target:
                filepath = output_dir / f"{prefix}_{saved:08d}.txt"
                filepath.write_text(DOC_SEP_JOIN.join(batch_text), encoding="utf-8", newline="\n")
                saved += 1
                total_bytes += batch_size
                batch_text = []
                batch_size = 0

                if total_bytes >= target_bytes:
                    break

                # Progress every 60 seconds
                now = time.monotonic()
                if now - last_print >= 60:
                    elapsed_min = (now - start_time) / 60
                    speed_mb = (total_bytes - existing_bytes) / max(now - start_time, 1) / 1e6
                    remaining_bytes = target_bytes - total_bytes
                    eta_min = (remaining_bytes / max(speed_mb * 1e6, 1) / 60) if speed_mb > 0 else 0
                    print(
                        f"  [{label}] {total_bytes / 1e9:.2f}/{target_gb:.1f} GB "
                        f"({saved} files) | {speed_mb:.1f} MB/s | "
                        f"~{eta_min:.0f} min left"
                    )
                    last_print = now

                    # Save progress periodically
                    progress[progress_key] = {
                        "files_saved": saved,
                        "bytes_written": total_bytes,
                        "records_consumed": records_consumed,
                    }
                    save_progress(progress)

        # Write remaining batch
        if batch_text:
            filepath = output_dir / f"{prefix}_{saved:08d}.txt"
            filepath.write_text(DOC_SEP_JOIN.join(batch_text), encoding="utf-8", newline="\n")
            saved += 1
            total_bytes += batch_size

    except KeyboardInterrupt:
        if batch_text:
            filepath = output_dir / f"{prefix}_{saved:08d}.txt"
            filepath.write_text(DOC_SEP_JOIN.join(batch_text), encoding="utf-8", newline="\n")
            saved += 1
            total_bytes += batch_size
        elapsed_min = (time.monotonic() - start_time) / 60
        print(
            f"\n  [{label}] Interrupted after {elapsed_min:.0f} min. "
            f"{saved} files ({total_bytes / 1e9:.2f} GB). "
            f"Run with --resume to continue."
        )
    except Exception as e:
        # Flush the pending batch: records_consumed is saved below, and
        # consumed-but-unwritten records would be skipped forever on resume.
        if batch_text:
            filepath = output_dir / f"{prefix}_{saved:08d}.txt"
            filepath.write_text(DOC_SEP_JOIN.join(batch_text), encoding="utf-8", newline="\n")
            saved += 1
            total_bytes += batch_size
        print(f"  [{label}] Error: {e}")
        print(f"  [{label}] Run with --resume to retry.")

    elapsed_min = (time.monotonic() - start_time) / 60
    print(f"\n  [{label}] Done: {saved} files saved ({total_bytes / 1e9:.2f} GB) in {elapsed_min:.0f} min.")
    if ai_rejected:
        print(f"  [{label}] {ai_rejected} records rejected by AI content filter")
    if literals_sanitized:
        print(f"  [{label}] {literals_sanitized} special-token literal(s) space-broken in saved text")
    if leak_rejected:
        print(f"  [{label}] {leak_rejected} records dropped by the sealed-probe screen")

    progress[progress_key] = {
        "files_saved": saved,
        "bytes_written": total_bytes,
        "records_consumed": records_consumed,
    }
    save_progress(progress)
    return saved


def fetch_openwebtext(target_gb: float, progress: dict) -> int:
    """Download general web text from OpenWebText (Skylion007/openwebtext).

    ~8M documents of high-quality web text extracted from URLs shared on
    Reddit with 3+ karma. Public dataset, no authentication required.

    Args:
        target_gb: How many GB to download.
        progress: Progress dict for resume support.
    """
    return _fetch_hf_streaming(
        dataset_name="Skylion007/openwebtext",
        config_name=None,
        text_field="text",
        output_dir=OPENWEBTEXT_DIR,
        target_gb=target_gb,
        label="OpenWebText",
        progress=progress,
        progress_key="openwebtext",
    )


def fetch_c4(target_gb: float, progress: dict) -> int:
    """Download cleaned Common Crawl text from C4 (allenai/c4, English).

    Colossal Clean Crawled Corpus — cleaned web text from Common Crawl.
    ~300+ GB total, stream as much as needed. Public, no auth required.

    Args:
        target_gb: How many GB to download.
        progress: Progress dict for resume support.
    """
    return _fetch_hf_streaming(
        dataset_name="allenai/c4",
        config_name="en",
        text_field="text",
        output_dir=C4_DIR,
        target_gb=target_gb,
        label="C4",
        progress=progress,
        progress_key="c4",
    )


def fetch_dclm(target_gb: float, progress: dict) -> int:
    """Download model-filtered web text from DCLM-Baseline.

    DCLM-Baseline: 4T tokens from Common Crawl, filtered with a fastText
    classifier trained on OpenHermes 2.5 (instruction-tuned quality signal).
    Beats FineWeb-Edu at all compute levels on MMLU. Used by SmolLM3 Stage 1.
    License: CC-BY-4.0. No authentication required.

    Args:
        target_gb: How many GB to download (recommended 10-20 GB).
        progress: Progress dict for resume support.
    """
    return _fetch_hf_streaming(
        dataset_name="mlfoundations/dclm-baseline-1.0",
        config_name=None,
        text_field="text",
        output_dir=DCLM_DIR,
        target_gb=target_gb,
        label="DCLM",
        progress=progress,
        progress_key="dclm",
    )


def fetch_finemath(target_gb: float, progress: dict) -> int:
    """Download high-quality math text from FineMath (two configs, 50/50 split).

    FineMath-4+: 9.6B tokens scored by LLaMA-3.1-70B, filtered to 4+/5 quality.
    InfiMM-WebMath-3+: 20.5B tokens, complementary source. SmolLM3 uses 50/50.
    Combined gives ~50B tokens of step-by-step math from CommonCrawl.
    License: ODC-By. No authentication required.

    Args:
        target_gb: Total GB to download (split evenly between both configs).
        progress: Progress dict for resume support.
    """
    half = max(target_gb / 2, 0.1)
    saved_a = _fetch_hf_streaming(
        dataset_name="HuggingFaceTB/finemath",
        config_name="finemath-4plus",
        text_field="text",
        output_dir=FINEMATH_DIR,
        target_gb=half,
        label="FineMath-4plus",
        progress=progress,
        progress_key="finemath_4plus",
    )
    saved_b = _fetch_hf_streaming(
        dataset_name="HuggingFaceTB/finemath",
        config_name="infiwebmath-3plus",
        text_field="text",
        output_dir=FINEMATH_DIR,
        target_gb=half,
        label="InfiMM-WebMath",
        progress=progress,
        progress_key="infiwebmath_3plus",
    )
    return saved_a + saved_b


# Languages used by SmolLM3, with per-language weights (ruled 2026-07-30).
# The split is weight-proportional, not equal: python carries 3x because
# co-located step-by-step numeric reasoning is the one measurable lever code
# has on the math eval category; the compiled workhorses carry 1.5x; the
# markup/data formats carry 0.3x because FineWeb-Edu and StackExchange
# already supply that shape -- except json at 1.0, which tool-call emission
# is shaped like. A dry language's unused share spreads over the languages
# after it in this order, weight-proportionally; the weights are the floor
# guarantee, the order only routes windfalls.
_STACK_LANGUAGES = [
    ("python", 3.0),
    ("javascript", 1.5),
    ("typescript", 1.5),
    ("java", 1.5),
    ("c", 1.5),
    ("cpp", 1.5),
    ("go", 1.0),
    ("rust", 1.0),
    ("ruby", 1.0),
    ("php", 1.0),
    ("shell", 1.0),
    ("sql", 1.0),
    ("html", 0.3),
    ("css", 0.3),
    ("markdown", 0.3),
    ("json", 1.0),
]
_STACK_PERMISSIVE = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "unlicense",
    "cc0-1.0",
    "0bsd",
}


def fetch_the_stack(target_gb: float, progress: dict) -> int:
    """Download source code from The Stack (bigcode/the-stack).

    v1 is the config that ships file contents inline (`content` +
    `max_stars_repo_licenses`, the fields this loop reads); v2 publishes
    only blob ids that need a separate Software Heritage S3 fetch.

    Iterates over 16 languages (SmolLM3 selection), each capped at a
    WEIGHT-proportional share of the REMAINING target (python 3x, workhorse
    compiled languages 1.5x, markup 0.3x -- see _STACK_LANGUAGES) so no
    single language can fill the pull and the mix is a chosen diet.
    Filters to permissive licenses (MIT/Apache/BSD/ISC/etc.).
    Code data teaches structured reasoning, pattern completion, and precise
    instruction following.

    IMPORTANT: This dataset requires accepting the license on HuggingFace
    (huggingface.co/datasets/bigcode/the-stack -- v1, NOT -v2) and logging in
    with `hf auth login` before running.

    Args:
        target_gb: Total GB to download across all languages.
        progress: Progress dict for resume support.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [The Stack] ERROR: 'datasets' package not found.")
        print("  [The Stack] Install with: pip install datasets")
        return 0

    STACK_DIR.mkdir(parents=True, exist_ok=True)

    existing_files = list(STACK_DIR.glob("*.txt"))
    existing_bytes = sum(f.stat().st_size for f in existing_files)
    target_bytes = int(target_gb * 1024 * 1024 * 1024)
    total_saved = len(existing_files)
    total_bytes = existing_bytes

    if existing_bytes >= target_bytes:
        print(
            f"  [The Stack] Already have {existing_bytes / 1e9:.2f} GB "
            f"({total_saved} files), target is {target_gb:.1f} GB. Skipping."
        )
        return total_saved

    file_target = 5 * 1024 * 1024  # 5 MB per file
    literals_sanitized = 0
    leak_rejected = 0
    locked_guard = _locked_probe_guard("The Stack")
    start_time = time.monotonic()

    remaining_weight = sum(w for _, w in _STACK_LANGUAGES)
    for lang, lang_weight in _STACK_LANGUAGES:
        if total_bytes >= target_bytes:
            break

        # Each language takes its WEIGHT-proportional share of what is LEFT
        # to collect, so no single language can fill the target alone and the
        # mix is a chosen diet rather than a uniform default. Dividing the
        # REMAINDER rather than the target hands an unavailable or exhausted
        # language's share to the ones after it, weight-proportionally, so
        # the target is still met when the data exists.
        lang_budget = (target_bytes - total_bytes) * lang_weight / max(1e-9, remaining_weight)
        remaining_weight -= lang_weight
        lang_start_bytes = total_bytes

        lang_progress_key = f"stack_{lang}"
        lang_dir_prefix = f"stack_{lang}"
        records_consumed = progress.get(lang_progress_key, {}).get("records_consumed", 0)
        remaining_gb = (target_bytes - total_bytes) / 1e9
        print(f"  [The Stack] Language: {lang} | {remaining_gb:.2f} GB remaining")

        try:
            # v1 selects languages by data_dir; its C++ folder is "c++"
            stack_dir_name = {"cpp": "c++"}.get(lang, lang)
            ds = load_dataset(
                "bigcode/the-stack",
                data_dir=f"data/{stack_dir_name}",
                split="train",
                streaming=True,
            )
        except Exception as e:
            err = str(e)
            if "gated" in err.lower() or "401" in err or "403" in err or "access" in err.lower():
                print(f"  [The Stack] Access denied for '{lang}'.")
                print("  [The Stack] This dataset is gated. Run: hf auth login")
                print("  [The Stack] Then accept the license at:")
                print("  [The Stack]   https://huggingface.co/datasets/bigcode/the-stack")
                break  # auth issue — stop all languages, not just this one
            print(f"  [The Stack] Could not load '{lang}': {e} -- skipping")
            continue

        if records_consumed > 0:
            print(f"  [The Stack] Skipping {records_consumed:,} previously consumed records...")
            ds = ds.skip(records_consumed)

        batch_text: list[str] = []
        batch_size = 0

        try:
            for record in ds:
                records_consumed += 1

                # License filter — permissive only
                licenses = record.get("max_stars_repo_licenses", []) or []
                if licenses and not any(lic.lower() in _STACK_PERMISSIVE for lic in licenses):
                    continue

                text = record.get("content", "")
                if not text or len(text) < MIN_ARTICLE_LENGTH:
                    continue

                # Code is where the literals actually live: "List<A>" carves
                # to the real <A> id, a literal "</s>" writes EOS mid-file.
                text, _hits = _sanitize_special_literals(text)
                if locked_guard is not None and locked_guard.leaks(text):
                    leak_rejected += 1
                    continue
                # count AFTER the screen: only hits that reached disk
                literals_sanitized += _hits

                text_bytes = len(text.encode("utf-8"))
                batch_text.append(text)
                batch_size += text_bytes

                if batch_size >= file_target:
                    filepath = STACK_DIR / f"{lang_dir_prefix}_{total_saved:08d}.txt"
                    filepath.write_text(DOC_SEP_JOIN.join(batch_text), encoding="utf-8", newline="\n")
                    total_saved += 1
                    total_bytes += batch_size
                    batch_text = []
                    batch_size = 0

                    if (total_bytes >= target_bytes
                            or total_bytes - lang_start_bytes >= lang_budget):
                        break

                    now = time.monotonic()
                    elapsed_min = (now - start_time) / 60
                    speed_mb = (total_bytes - existing_bytes) / max(now - start_time, 1) / 1e6
                    if elapsed_min > 0 and total_saved % 20 == 0:
                        print(
                            f"  [The Stack/{lang}] {total_bytes / 1e9:.2f}/{target_gb:.1f} GB "
                            f"({total_saved} files) | {speed_mb:.1f} MB/s"
                        )

            if batch_text:
                filepath = STACK_DIR / f"{lang_dir_prefix}_{total_saved:08d}.txt"
                filepath.write_text(DOC_SEP_JOIN.join(batch_text), encoding="utf-8", newline="\n")
                total_saved += 1
                total_bytes += batch_size

        except KeyboardInterrupt:
            if batch_text:
                filepath = STACK_DIR / f"{lang_dir_prefix}_{total_saved:08d}.txt"
                filepath.write_text(DOC_SEP_JOIN.join(batch_text), encoding="utf-8", newline="\n")
                total_saved += 1
                total_bytes += batch_size
            progress[lang_progress_key] = {"records_consumed": records_consumed}
            save_progress(progress)
            elapsed_min = (time.monotonic() - start_time) / 60
            print(
                f"\n  [The Stack] Interrupted after {elapsed_min:.0f} min. "
                f"{total_saved} files ({total_bytes / 1e9:.2f} GB). "
                f"Run with --resume to continue."
            )
            return total_saved
        except Exception as e:
            # Flush the pending batch: records_consumed is saved below, and
            # consumed-but-unwritten records would be skipped forever.
            if batch_text:
                filepath = STACK_DIR / f"{lang_dir_prefix}_{total_saved:08d}.txt"
                filepath.write_text(DOC_SEP_JOIN.join(batch_text), encoding="utf-8", newline="\n")
                total_saved += 1
                total_bytes += batch_size
            print(f"  [The Stack/{lang}] Error: {e} -- moving to next language")

        progress[lang_progress_key] = {"records_consumed": records_consumed}
        save_progress(progress)

    elapsed_min = (time.monotonic() - start_time) / 60
    print(f"\n  [The Stack] Done: {total_saved} files saved ({total_bytes / 1e9:.2f} GB) in {elapsed_min:.0f} min.")
    # The per-language budget is handed forward, not retried, so a run where
    # several languages are short or unavailable finishes UNDER target. Report
    # the shortfall and the per-language split: "Done" alone cannot be told
    # apart from a full pull.
    if total_bytes < 0.9 * target_bytes:
        by_lang = {}
        for f in STACK_DIR.glob("*.txt"):
            by_lang[f.name.rsplit("_", 1)[0]] = by_lang.get(f.name.rsplit("_", 1)[0], 0) + f.stat().st_size
        print(f"  [The Stack] WARNING: {total_bytes / 1e9:.2f} GB is "
              f"{total_bytes / target_bytes:.0%} of the {target_gb:.1f} GB target -- "
              f"languages that ran dry hand their share forward, they are not retried. "
              f"Re-run with --resume to top up.")
        for lang_key, size in sorted(by_lang.items(), key=lambda kv: -kv[1]):
            print(f"  [The Stack]   {lang_key}: {size / 1e9:.2f} GB")
    if literals_sanitized:
        print(f"  [The Stack] {literals_sanitized} special-token literal(s) space-broken in saved code")
    if leak_rejected:
        print(f"  [The Stack] {leak_rejected} records dropped by the sealed-probe screen")
    return total_saved


# ---------------------------------------------------------------------------
# Combine all sources
# ---------------------------------------------------------------------------


def combine_all_sources() -> None:
    """Merge all downloaded text files into one combined.txt for FORGE.

    Includes paragraph-level dedup: identical paragraphs (by SHA-256 hash)
    are written only once across all sources. This catches verbatim content
    shared between Wikipedia and FineWeb-Edu, duplicate Gutenberg entries, etc.

    Uses os.scandir (not sorted glob) for large directories (>10K files) to
    avoid O(N log N) sort + massive memory allocation on dirs like wiki_dump (4.7M files).
    """
    import os as _os

    print("\n--- Combining all sources into combined.txt ---")
    print("  (with paragraph-level deduplication)")

    total_chars = 0
    total_files = 0
    dupes_skipped = 0
    seen_hashes: set[bytes] = set()
    # S789: Cap dedup set to prevent OOM at scale.  50M entries × ~41 bytes
    # (8-byte digest + set overhead) ≈ 2 GB RAM — safe headroom on 61 GB.
    MAX_DEDUP_ENTRIES = 50_000_000
    dedup_warned = False

    source_dirs = [
        ("Wiki Dump", WIKI_DUMP_DIR),
        ("Wikipedia", WIKI_DIR),
        ("Simple Wiki", SIMPLE_DIR),
        ("Gutenberg", GUTENBERG_DIR),
        ("FineWeb-Edu", FINEWEB_DIR),
        ("OpenWebText", OPENWEBTEXT_DIR),
        ("C4", C4_DIR),
        ("DCLM", DCLM_DIR),
        ("FineMath", FINEMATH_DIR),
        ("The Stack", STACK_DIR),
        ("Wayback", WAYBACK_DIR),
        ("Fandom", FANDOM_DIR),
    ]
    # Stack Exchange has subdirectories per site
    if STACKEX_DIR.exists():
        for sub in sorted(STACKEX_DIR.iterdir()):
            if sub.is_dir():
                source_dirs.append((f"SE/{sub.name}", sub))

    # Write to temp file then rename — crash during multi-hour merge
    # must not truncate the previous combined.txt
    tmp_file = COMBINED_FILE.with_suffix(".txt.tmp")

    try:
        with open(tmp_file, "w", encoding="utf-8") as out:
            for label, source_dir in source_dirs:
                if not source_dir.exists():
                    continue

                # Use os.scandir for speed — no sort needed for training data
                dir_files = 0
                try:
                    with _os.scandir(source_dir) as scanner:
                        for entry in scanner:
                            if not entry.is_file(follow_symlinks=False):
                                continue
                            if not entry.name.endswith(".txt"):
                                continue

                            try:
                                text = Path(entry.path).read_text(encoding="utf-8", errors="replace")
                            except OSError:
                                continue
                            if len(text.strip()) < MIN_PARAGRAPH_LENGTH:
                                continue

                            paragraphs = text.split("\n\n")
                            unique_paras = []
                            for para in paragraphs:
                                para = para.strip()
                                if len(para) < MIN_PARAGRAPH_LENGTH:
                                    unique_paras.append(para)
                                    continue
                                # Raw digest (8 bytes, ~41B/entry) instead of
                                # hexdigest (16 chars, ~65B/entry)
                                h = hashlib.sha256(para.encode("utf-8")).digest()[:8]
                                if h in seen_hashes:
                                    dupes_skipped += 1
                                    continue
                                if len(seen_hashes) < MAX_DEDUP_ENTRIES:
                                    seen_hashes.add(h)
                                elif not dedup_warned:
                                    dedup_warned = True
                                    print(
                                        f"  WARNING: Dedup table at capacity "
                                        f"({MAX_DEDUP_ENTRIES:,} entries). "
                                        f"Later paragraphs won't be "
                                        f"deduplicated."
                                    )
                                unique_paras.append(para)

                            cleaned = "\n\n".join(unique_paras).strip()
                            if len(cleaned) < MIN_PARAGRAPH_LENGTH:
                                continue

                            out.write(cleaned)
                            out.write("\n\n")
                            total_chars += len(cleaned)
                            total_files += 1
                            dir_files += 1

                            # Progress for large dirs
                            if dir_files % 100_000 == 0:
                                print(
                                    f"  [{label}] {dir_files:,} files processed "
                                    f"({total_chars / 1024 / 1024 / 1024:.1f} GB)..."
                                )
                except OSError:
                    continue

                if dir_files > 0:
                    print(f"  [{label}] {dir_files:,} files -> {total_chars / 1024 / 1024 / 1024:.1f} GB cumulative")

        # Atomic replace — only overwrites combined.txt after full write succeeds
        tmp_file.replace(COMBINED_FILE)
    except BaseException:
        # Clean up temp file on any failure (including KeyboardInterrupt)
        try:
            if tmp_file.exists():
                tmp_file.unlink()
        except OSError:
            pass
        raise

    size_gb = total_chars / (1024 * 1024 * 1024)
    print(f"\n  Combined {total_files:,} files into combined.txt ({size_gb:.1f} GB)")
    if dupes_skipped:
        print(f"  Deduplication: {dupes_skipped:,} duplicate paragraphs removed")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _count_dir(directory: Path, recursive: bool = False) -> tuple[int, int]:
    """Count .txt files and total size in a directory using os.scandir (fast).

    Returns (file_count, total_bytes).
    """
    import os

    count = 0
    total = 0
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False) and entry.name.endswith(".txt"):
                    count += 1
                    total += entry.stat(follow_symlinks=False).st_size
                elif recursive and entry.is_dir(follow_symlinks=False):
                    sub_count, sub_size = _count_dir(Path(entry.path), recursive=True)
                    count += sub_count
                    total += sub_size
    except OSError:
        pass
    return count, total


def print_stats() -> None:
    """Print summary of collected data."""
    print("\n" + "=" * 60)
    print("  PRE-TRAINING DATA SUMMARY")
    print("=" * 60)

    total_files = 0
    total_size = 0

    sources = [
        ("Wiki Dump", WIKI_DUMP_DIR, False),
        ("Wikipedia", WIKI_DIR, False),
        ("Simple Wiki", SIMPLE_DIR, False),
        ("Gutenberg", GUTENBERG_DIR, False),
        ("FineWeb-Edu", FINEWEB_DIR, False),
        ("OpenWebText", OPENWEBTEXT_DIR, False),
        ("C4", C4_DIR, False),
        ("DCLM", DCLM_DIR, False),
        ("FineMath", FINEMATH_DIR, False),
        ("The Stack", STACK_DIR, False),
        ("StackExchange", STACKEX_DIR, True),  # subdirs per site
        ("Wayback", WAYBACK_DIR, False),
        ("Fandom", FANDOM_DIR, False),
    ]

    for label, directory, recursive in sources:
        if directory.exists():
            count, size = _count_dir(directory, recursive=recursive)
            total_files += count
            total_size += size
            # Human-friendly size
            if size >= 1024 * 1024 * 1024:
                size_str = f"{size / 1024 / 1024 / 1024:6.1f} GB"
            else:
                size_str = f"{size / 1024 / 1024:6.1f} MB"
            # Compact file count for huge dirs
            if count >= 100_000:
                count_str = f"{count / 1_000_000:.1f}M"
            elif count >= 10_000:
                count_str = f"{count / 1_000:.0f}K"
            else:
                count_str = f"{count}"
            print(f"  {label + ':':16s} {count_str:>6s} files  ({size_str})")
        else:
            print(f"  {label + ':':16s}      0 files  (   0.0 MB)")

    if COMBINED_FILE.exists():
        combined_size = COMBINED_FILE.stat().st_size
        if combined_size >= 1024 * 1024 * 1024:
            csize_str = f"{combined_size / 1024 / 1024 / 1024:6.1f} GB"
        else:
            csize_str = f"{combined_size / 1024 / 1024:6.1f} MB"
        print(f"  {'Combined:':16s}      1 file   ({csize_str})")

    print(f"  {'---':50s}")
    if total_size >= 1024 * 1024 * 1024:
        total_str = f"{total_size / 1024 / 1024 / 1024:6.1f} GB"
    else:
        total_str = f"{total_size / 1024 / 1024:6.1f} MB"
    print(f"  {'TOTAL:':16s} {total_files:>6d} files  ({total_str})")
    print()
    # The next step reads SOURCE_DIRS directly; combined.txt is a retired
    # artifact and pointing the summary at it invites rebuilding it.
    print("  Next step: pretokenize_data.py (reads these dirs directly)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Collect pre-training data for Enigma Engine models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_pretraining_data.py                     # Default (2000 wiki + 200 simple + 100 books)
  python collect_pretraining_data.py --wiki 5000         # 5K Wikipedia articles
  python collect_pretraining_data.py --books 300         # 300 Gutenberg books
  python collect_pretraining_data.py --all-sources       # Max: all sources enabled
  python collect_pretraining_data.py --books-only        # Only Gutenberg books
  python collect_pretraining_data.py --stackexchange     # Stack Exchange Q&A dumps
  python collect_pretraining_data.py --wayback 500       # 500 pages from Wayback Machine
  python collect_pretraining_data.py --fineweb 25        # 25 GB of FineWeb-Edu educational text
  python collect_pretraining_data.py --openwebtext 10    # 10 GB of OpenWebText web text
  python collect_pretraining_data.py --c4 20             # 20 GB of C4 cleaned Common Crawl
  python collect_pretraining_data.py --fandom-all        # ALL articles from every Fandom wiki
  python collect_pretraining_data.py --resume            # Continue interrupted download
  python collect_pretraining_data.py --stats             # Show current data stats
        """,
    )

    parser.add_argument(
        "--wiki", type=int, default=2000, help="Number of Wikipedia articles to download (default: 2000)"
    )
    parser.add_argument("--simple", type=int, default=200, help="Number of Simple Wikipedia articles (default: 200)")
    parser.add_argument("--books", type=int, default=100, help="Number of Gutenberg books (default: 100)")
    parser.add_argument(
        "--wiki-dump",
        action="store_true",
        help="Download full English Wikipedia dump (~22 GB compressed, 15-20 GB clean text)",
    )
    parser.add_argument(
        "--all-sources", action="store_true", help="Max defaults: all sources enabled with reasonable limits"
    )
    parser.add_argument("--books-only", action="store_true", help="Download only Gutenberg books (skip Wikipedia)")
    parser.add_argument(
        "--resume", action="store_true", help="Resume interrupted download (keeps existing files + progress)"
    )
    parser.add_argument("--stats", action="store_true", help="Show data stats and exit")
    parser.add_argument("--combine-only", action="store_true", help="Only recombine existing files (skip downloading)")
    parser.add_argument(
        "--no-combine", action="store_true", help="Skip the combine step after downloading (run --combine-only later)"
    )
    parser.add_argument(
        "--fineweb",
        type=float,
        default=0,
        help="Download N GB of FineWeb-Edu educational text (requires `pip install datasets`)",
    )
    parser.add_argument(
        "--stackexchange",
        action="store_true",
        help="Download Stack Exchange Q&A dumps from archive.org (requires `pip install py7zr`)",
    )
    parser.add_argument(
        "--stackexchange-score", type=int, default=5, help="Minimum post score for Stack Exchange (default: 5)"
    )
    parser.add_argument(
        "--wayback", type=int, default=0, help="Max pages to download from Wayback Machine educational domains"
    )
    parser.add_argument("--fandom", type=int, default=0, help="Max articles from Fandom wikis (0 = unlimited/all)")
    parser.add_argument(
        "--fandom-all", action="store_true", help="Download ALL articles from every Fandom wiki (no cap)"
    )
    parser.add_argument(
        "--openwebtext",
        type=float,
        default=0,
        help="Download N GB of OpenWebText web text (requires `pip install datasets`)",
    )
    parser.add_argument(
        "--c4", type=float, default=0, help="Download N GB of C4 cleaned Common Crawl (requires `pip install datasets`)"
    )
    parser.add_argument(
        "--dclm",
        type=float,
        default=0,
        help="Download N GB of DCLM-Baseline model-filtered web text (requires `pip install datasets`)",
    )
    parser.add_argument(
        "--finemath",
        type=float,
        default=0,
        help="Download N GB of FineMath (4+ and InfiMM-WebMath, 50/50 split, requires `pip install datasets`)",
    )
    parser.add_argument(
        "--code",
        type=float,
        default=0,
        help="Download N GB of code from The Stack v1 (requires `pip install datasets` + HuggingFace auth)",
    )

    args = parser.parse_args()

    # Stats only
    if args.stats:
        print_stats()
        return

    # --all-sources overrides
    if args.all_sources:
        args.wiki = max(args.wiki, 5000)
        args.simple = max(args.simple, 500)
        args.books = max(args.books, 2000)
        args.stackexchange = True
        if args.fineweb <= 0:
            args.fineweb = 25.0
        if args.wayback <= 0:
            args.wayback = 1000
        if args.fandom <= 0 and not args.fandom_all:
            args.fandom = 2000
        # C4 and OpenWebText stay OFF here: DCLM supersedes them and their
        # dirs are retired -- a max-defaults run must not refetch 30 GB of
        # retired sources. Both remain reachable by their explicit flags.
        if args.dclm <= 0:
            args.dclm = 15.0
        if args.finemath <= 0:
            args.finemath = 10.0
        if args.code <= 0:
            args.code = 10.0

    # --books-only
    if args.books_only:
        args.wiki = 0
        args.simple = 0

    # Load or reset progress
    if args.resume:
        progress = load_progress()
        # Rebuild gutenberg_ids from files on disk (in case progress.json lost them)
        if GUTENBERG_DIR.exists():
            disk_ids = set()
            for f in GUTENBERG_DIR.glob("*.txt"):
                m = re.match(r"^(\d+)", f.stem)
                if m:
                    disk_ids.add(m.group(1))
            saved_ids = set(progress.get("gutenberg_ids", []))
            progress["gutenberg_ids"] = list(saved_ids | disk_ids)
        print("Resuming previous download...")
    else:
        progress = {"gutenberg_ids": [], "stats": {}}

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    if not args.combine_only:
        print("\n=== Collecting Pre-Training Data ===\n")
        sources = []
        if args.wiki_dump:
            sources.append("Wikipedia dump")
        if args.wiki > 0:
            sources.append(f"{args.wiki} Wikipedia API")
        if args.simple > 0:
            sources.append(f"{args.simple} Simple Wiki")
        if args.books > 0:
            sources.append(f"{args.books} Gutenberg books")
        if args.fineweb > 0:
            sources.append(f"{args.fineweb:.0f} GB FineWeb-Edu")
        if args.stackexchange:
            sources.append(f"Stack Exchange ({len(STACKEX_SITES)} sites)")
        if args.wayback > 0:
            sources.append(f"{args.wayback} Wayback pages")
        if args.fandom > 0 or args.fandom_all:
            cap = "all" if args.fandom_all else str(args.fandom)
            sources.append(f"{cap} Fandom wiki articles")
        if args.openwebtext > 0:
            sources.append(f"{args.openwebtext:.0f} GB OpenWebText")
        if args.c4 > 0:
            sources.append(f"{args.c4:.0f} GB C4")
        if args.dclm > 0:
            sources.append(f"{args.dclm:.0f} GB DCLM")
        if args.finemath > 0:
            sources.append(f"{args.finemath:.0f} GB FineMath")
        if args.code > 0:
            sources.append(f"{args.code:.0f} GB The Stack")
        print(f"  Sources: {', '.join(sources)}")
        print(f"  Output: {BASE_DIR}")
        print()

        # 0. Wikipedia full dump (if requested — this is the big one)
        if args.wiki_dump:
            print("--- Wikipedia Full Dump ---")
            process_wiki_dump(progress)

        # 1. Project Gutenberg (most reliable, run first)
        if args.books > 0:
            print("--- Project Gutenberg ---")
            book_count = fetch_gutenberg_books(args.books, progress)
        else:
            book_count = len(list(GUTENBERG_DIR.glob("*.txt"))) if GUTENBERG_DIR.exists() else 0

        # 2. Simple Wikipedia (smaller wiki, rarely rate-limits)
        if args.simple > 0:
            print("\n--- Simple Wikipedia ---")
            simple_count = fetch_wikipedia_articles(SIMPLE_WIKI_API, SIMPLE_DIR, args.simple, "Simple Wiki", progress)
        else:
            simple_count = len(list(SIMPLE_DIR.glob("*.txt"))) if SIMPLE_DIR.exists() else 0

        # 3. English Wikipedia (largest but rate-limits - run last)
        if args.wiki > 0:
            print("\n--- Wikipedia (English) ---")
            wiki_count = fetch_wikipedia_articles(WIKI_API, WIKI_DIR, args.wiki, "Wikipedia", progress)
        else:
            wiki_count = len(list(WIKI_DIR.glob("*.txt"))) if WIKI_DIR.exists() else 0

        # 4. FineWeb-Edu (educational web pages from HuggingFace)
        if args.fineweb > 0:
            print("\n--- FineWeb-Edu ---")
            fineweb_count = fetch_fineweb_edu(args.fineweb, progress)
        else:
            fineweb_count = len(list(FINEWEB_DIR.glob("*.txt"))) if FINEWEB_DIR.exists() else 0

        # 5. Stack Exchange Q&A dumps (archive.org)
        stackex_count = 0
        if args.stackexchange:
            print("\n--- Stack Exchange ---")
            stackex_count = fetch_stackexchange(STACKEX_SITES, args.stackexchange_score, progress)

        # 6. Wayback Machine (educational domains)
        wayback_count = 0
        if args.wayback > 0:
            print("\n--- Wayback Machine ---")
            wayback_count = fetch_wayback(WAYBACK_DOMAINS, args.wayback, progress)

        # 7. Fandom wikis (games, movies, TV, books)
        fandom_count = 0
        if args.fandom > 0 or args.fandom_all:
            fandom_cap = 0 if args.fandom_all else args.fandom
            print("\n--- Fandom Wikis ---")
            fandom_wikis = discover_fandom_wikis()
            fandom_count = fetch_fandom(fandom_wikis, fandom_cap, progress)

        # 8. OpenWebText (general web text from HuggingFace)
        openwebtext_count = 0
        if args.openwebtext > 0:
            print("\n--- OpenWebText ---")
            openwebtext_count = fetch_openwebtext(args.openwebtext, progress)

        # 9. C4 (cleaned Common Crawl from HuggingFace)
        c4_count = 0
        if args.c4 > 0:
            print("\n--- C4 (Common Crawl) ---")
            c4_count = fetch_c4(args.c4, progress)

        # 10. DCLM-Baseline (model-filtered web text from HuggingFace)
        dclm_count = 0
        if args.dclm > 0:
            print("\n--- DCLM-Baseline ---")
            dclm_count = fetch_dclm(args.dclm, progress)

        # 11. FineMath (high-quality math text from HuggingFace)
        finemath_count = 0
        if args.finemath > 0:
            print("\n--- FineMath ---")
            finemath_count = fetch_finemath(args.finemath, progress)

        # 12. The Stack v1 (source code from HuggingFace — requires auth;
        # v1 is the config with inline content, NOT -v2 -- see fetch_the_stack)
        stack_count = 0
        if args.code > 0:
            print("\n--- The Stack ---")
            stack_count = fetch_the_stack(args.code, progress)

        progress["stats"] = {
            "wiki": wiki_count,
            "simple": simple_count,
            "books": book_count,
            "fineweb": fineweb_count,
            "stackexchange": stackex_count,
            "wayback": wayback_count,
            "fandom": fandom_count,
            "openwebtext": openwebtext_count,
            "c4": c4_count,
            "dclm": dclm_count,
            "finemath": finemath_count,
            "stack": stack_count,
        }
        save_progress(progress)

    # Combine everything (unless --no-combine)
    if not getattr(args, "no_combine", False):
        combine_all_sources()

    # Print summary
    print_stats()


if __name__ == "__main__":
    main()
