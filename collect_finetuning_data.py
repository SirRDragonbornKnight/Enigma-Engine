"""
Fine-Tuning Data Collector for Enigma Engine
=============================================
Downloads and formats instruction-following datasets for fine-tuning.

Sources (all require `pip install datasets`):
  - OASST1 (--oasst)        - Open Assistant conversations (~80K turns)
  - Dolly 15k (--dolly)     - Databricks instruction pairs (~15K)
  - SlimOrca (--slimorca N) - Instruction-following from Open-Orca (N = max samples)
  - OpenThoughts3 (--openthoughts3 N) - Reasoning traces with <think> tags (D-4)
  - SmolTalk2 (--smoltalk2 N --smoltalk2-config NAME [--smoltalk2-split NAME]) - SmolLM3 SFT data (D-11)
  - All sources (--all)     - Download everything

Output format:
  JSONL with {"prompt": "...", "completion": "..."} per line.
  Ready for FORGE → Basic training mode.

Usage:
  python collect_finetuning_data.py --oasst
  python collect_finetuning_data.py --dolly
  python collect_finetuning_data.py --slimorca 50000
  python collect_finetuning_data.py --all
  python collect_finetuning_data.py --stats
"""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/finetune")


def _ensure_datasets():
    """Check that the `datasets` library is installed."""
    try:
        import datasets  # noqa: F401

        return True
    except ImportError:
        logger.error("The `datasets` library is required.\nInstall with: pip install datasets")
        return False


def _dedup_pairs(pairs: list[dict]) -> list[dict]:
    """Remove exact-duplicate prompt+completion pairs."""
    seen = set()
    unique = []
    for item in pairs:
        key = hashlib.sha256((item["prompt"] + item["completion"]).encode()).digest()[:16]
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _write_jsonl(pairs: list[dict], path: Path) -> int:
    """Write pairs to JSONL file, return count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(pairs)


def _write_combined_text(pairs: list[dict], path: Path) -> int:
    """Write pairs as canonical 'User: ...\\n\\nAssistant: ...' blocks.

    Uses the canonical plain-transcript format so the SFT trainer (which reads
    plain text via `Path.read_text`) can consume the collected fine-tune data
    without a JSONL-aware loader. Closes the D-11 consumer-side gap.

    Empty prompts or completions are skipped — a malformed block like
    'User: \\n\\nAssistant: foo' would teach the model that empty input
    is a valid prefix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "w", encoding="utf-8") as f:
        first = True
        for item in pairs:
            prompt = (item.get("prompt") or "").strip()
            completion = (item.get("completion") or "").strip()
            if not prompt or not completion:
                continue
            if not first:
                # Blank line between blocks → "C1\n\n" + "\n" + "User: P2"
                f.write("\n")
            f.write(f"User: {prompt}\n\nAssistant: {completion}\n")
            first = False
            written += 1
    return written


def _clean_text(text: str) -> str:
    """Basic text cleaning for instruction data."""
    if not text:
        return ""
    # Normalize whitespace
    text = " ".join(text.split())
    return text.strip()


# ── OASST1 ─────────────────────────────────────────────────────────


def collect_oasst() -> list[dict]:
    """Download and format Open Assistant Conversations (OASST1).

    Extracts instruction/response pairs from the conversation tree.
    Uses only English, top-ranked messages.
    """
    if not _ensure_datasets():
        return []

    from datasets import load_dataset

    logger.info("Downloading OASST1 dataset...")
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    logger.info(f"OASST1: {len(ds)} messages loaded")

    # Build conversation trees
    messages_by_id = {}
    children = {}
    for msg in ds:
        msg_id = msg["message_id"]
        messages_by_id[msg_id] = msg
        parent = msg.get("parent_id")
        if parent:
            children.setdefault(parent, []).append(msg_id)

    # Extract prompt-response pairs: root message (prompter) →
    # best child (assistant, highest rank)
    pairs = []
    for msg in ds:
        if msg["role"] != "prompter" or msg["parent_id"] is not None:
            continue
        if msg.get("lang", "en") != "en":
            continue

        prompt = _clean_text(msg["text"])
        if not prompt or len(prompt) < 10:
            continue

        # Find best assistant reply
        kid_ids = children.get(msg["message_id"], [])
        assistant_replies = []
        for kid_id in kid_ids:
            kid = messages_by_id.get(kid_id)
            if kid and kid["role"] == "assistant":
                rank = kid.get("rank", 999)
                if rank is None:
                    rank = 999
                assistant_replies.append((rank, kid))

        if not assistant_replies:
            continue

        # Pick highest-ranked (lowest rank number)
        assistant_replies.sort(key=lambda x: x[0])
        best = assistant_replies[0][1]
        # Enforce the English-only contract on the REPLY too, not just the
        # prompt: OASST replies can be a different language than their root.
        if best.get("lang", "en") != "en":
            continue
        completion = _clean_text(best["text"])
        if not completion or len(completion) < 10:
            continue

        pairs.append(
            {
                "prompt": prompt,
                "completion": completion,
            }
        )

    pairs = _dedup_pairs(pairs)
    logger.info(f"OASST1: {len(pairs)} instruction pairs extracted")
    return pairs


# ── Dolly 15k ──────────────────────────────────────────────────────


def collect_dolly() -> list[dict]:
    """Download and format Databricks Dolly 15k.

    Simple instruction/response pairs. Includes optional context
    which gets prepended to the prompt when present.
    """
    if not _ensure_datasets():
        return []

    from datasets import load_dataset

    logger.info("Downloading Dolly 15k dataset...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    logger.info(f"Dolly: {len(ds)} examples loaded")

    pairs = []
    for item in ds:
        instruction = _clean_text(item.get("instruction", ""))
        context = _clean_text(item.get("context", ""))
        response = _clean_text(item.get("response", ""))

        if not instruction or not response:
            continue
        if len(instruction) < 5 or len(response) < 10:
            continue

        # Prepend context to prompt when available
        if context:
            prompt = f"{instruction}\n\nContext: {context}"
        else:
            prompt = instruction

        pairs.append(
            {
                "prompt": prompt,
                "completion": response,
            }
        )

    pairs = _dedup_pairs(pairs)
    logger.info(f"Dolly: {len(pairs)} instruction pairs extracted")
    return pairs


# ── SlimOrca ───────────────────────────────────────────────────────


def collect_slimorca(max_samples: int = 100000) -> list[dict]:
    """Download and format Open-Orca SlimOrca.

    Large instruction-following dataset. Streams to avoid loading
    everything into memory at once.
    """
    if not _ensure_datasets():
        return []

    from datasets import load_dataset

    logger.info(f"Downloading SlimOrca (max {max_samples:,} samples)...")
    ds = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)

    pairs = []
    count = 0
    for item in ds:
        if count >= max_samples:
            break

        conversations = item.get("conversations", [])
        if len(conversations) < 2:
            count += 1
            continue

        # Find system + user + assistant turns
        system = ""
        prompt = ""
        completion = ""
        for turn in conversations:
            role = turn.get("from", "")
            value = _clean_text(turn.get("value", ""))
            if role == "system" and value:
                system = value
            elif role == "human" and value:
                prompt = value
            elif role == "gpt" and value:
                completion = value

        if not prompt or not completion:
            count += 1
            continue
        if len(prompt) < 5 or len(completion) < 10:
            count += 1
            continue

        # Include system prompt if non-generic
        if system and system.lower() not in (
            "you are an ai assistant.",
            "you are a helpful assistant.",
            "",
        ):
            prompt = f"{system}\n\n{prompt}"

        pairs.append(
            {
                "prompt": prompt,
                "completion": completion,
            }
        )
        count += 1

        if count % 10000 == 0:
            logger.info(f"  SlimOrca: {count:,} processed, {len(pairs):,} kept...")

    pairs = _dedup_pairs(pairs)
    logger.info(f"SlimOrca: {len(pairs)} instruction pairs extracted")
    return pairs


# ── OpenThoughts3 (D-4) ────────────────────────────────────────────


def collect_openthoughts3(max_samples: int = 100000) -> list[dict]:
    """Download and format OpenThoughts3-1.2M reasoning data.

    Source: `open-thoughts/OpenThoughts3-1.2M` (Apache-2.0).
    Composition: 850K math + 250K code + 100K science = 1.2M rows.
    Reasoning traces annotated by QwQ-32B and wrapped in
    `<think>...</think>` tags inside the gpt turn.

    Schema per Pass 139 verification:
        {"difficulty": int, "source": str, "domain": str,
         "conversations": [{"from": "human"/"gpt", "value": str}]}

    The `<think>` / `</think>` tags MUST be preserved verbatim — they
    align with our special token IDs (`<think>=10`, `</think>=11`) and
    must not be collapsed by whitespace normalization.
    """
    if not _ensure_datasets():
        return []

    from datasets import load_dataset

    logger.info(f"Downloading OpenThoughts3-1.2M (max {max_samples:,} samples)...")
    try:
        ds = load_dataset(
            "open-thoughts/OpenThoughts3-1.2M",
            split="train",
            streaming=True,
        )
    except Exception as exc:
        logger.error("Failed to load OpenThoughts3: %s", exc)
        return []

    pairs: list[dict] = []
    seen = 0
    for item in ds:
        if seen >= max_samples:
            break
        seen += 1

        conversations = item.get("conversations") or []
        human_turn = next((t for t in conversations if t.get("from") == "human"), None)
        gpt_turn = next((t for t in conversations if t.get("from") == "gpt"), None)
        if not human_turn or not gpt_turn:
            continue

        prompt = (human_turn.get("value") or "").strip()
        # Verbatim — DO NOT pass through _clean_text(): tags need newlines.
        completion = (gpt_turn.get("value") or "").strip()
        if not prompt or not completion:
            continue
        if len(prompt) < 5 or len(completion) < 10:
            continue

        pairs.append({"prompt": prompt, "completion": completion})

        if seen % 10000 == 0:
            logger.info(f"  OpenThoughts3: {seen:,} processed, {len(pairs):,} kept...")

    pairs = _dedup_pairs(pairs)
    logger.info(f"OpenThoughts3: {len(pairs)} reasoning pairs extracted")
    return pairs


# ── SmolTalk2 (D-11) ───────────────────────────────────────────────


def collect_smoltalk2(
    max_samples: int = 100000,
    config: str = "default",
    split: str | None = None,
) -> list[dict]:
    """Download and format SmolTalk2 SFT data.

    Source: `HuggingFaceTB/smoltalk2`. SmolTalk2 ships **many** configs
    (`SFT`, `Mid`, `Preference`) and each config has many named splits
    (e.g. `smoltalk_smollm3_smol_magpie_ultra_no_think`,
    `OpenThoughts3_1.2M_think`) — there is no canonical "train" split.

    On a missing config the HuggingFace loader raises ValueError
    naming the available configs; on a missing split it raises
    ValueError naming the available splits. Either way we log and
    return [] (per learned principle: detect on first attempt, do
    not loop).

    When `split` is None, all splits in the config are concatenated
    until `max_samples` is reached — useful for getting "all SFT data"
    without picking one of 25 split names.

    Schema: standard ChatML — `messages: [{role, content}]` where role
    is "user" / "assistant" (and optionally "system").
    """
    if not _ensure_datasets():
        return []

    from datasets import get_dataset_split_names, load_dataset

    logger.info(f"Downloading SmolTalk2 (config={config!r}, split={split!r}, max {max_samples:,})...")

    # Resolve splits to iterate.
    if split is None:
        try:
            splits = list(get_dataset_split_names("HuggingFaceTB/smoltalk2", config))
        except Exception as exc:
            logger.error(
                "Failed to enumerate SmolTalk2 splits "
                "(config=%r): %s. Pick a valid config and "
                "re-run with --smoltalk2-config NAME.",
                config,
                exc,
            )
            return []
        if not splits:
            logger.error("SmolTalk2 config=%r has no splits.", config)
            return []
        logger.info(f"  SmolTalk2 config={config!r} has {len(splits)} splits; iterating all.")
    else:
        splits = [split]

    pairs: list[dict] = []
    seen = 0
    for split_name in splits:
        if seen >= max_samples:
            break
        try:
            ds = load_dataset(
                "HuggingFaceTB/smoltalk2",
                config,
                split=split_name,
                streaming=True,
            )
        except Exception as exc:
            logger.error(
                "Failed to load SmolTalk2 (config=%r, split=%r): %s.",
                config,
                split_name,
                exc,
            )
            continue

        for item in ds:
            if seen >= max_samples:
                break
            seen += 1

            messages = item.get("messages") or []
            # Optional system prompt, prepended to first user turn.
            system = ""
            prompt = ""
            completion = ""
            for turn in messages:
                role = turn.get("role", "")
                content = _clean_text(turn.get("content", ""))
                if role == "system" and content and not system:
                    system = content
                elif role == "user" and content and not prompt:
                    prompt = content
                elif role == "assistant" and content and not completion:
                    completion = content
                    # first assistant reply is enough for SFT pair
                    break

            if not prompt or not completion:
                continue
            if len(prompt) < 5 or len(completion) < 10:
                continue

            if system and system.lower() not in (
                "you are an ai assistant.",
                "you are a helpful assistant.",
            ):
                prompt = f"{system}\n\n{prompt}"

            pairs.append({"prompt": prompt, "completion": completion})

            if seen % 10000 == 0:
                logger.info(f"  SmolTalk2: {seen:,} processed, {len(pairs):,} kept...")

    pairs = _dedup_pairs(pairs)
    logger.info(f"SmolTalk2: {len(pairs)} instruction pairs extracted")
    return pairs


# ── Combine & Stats ───────────────────────────────────────────────


def combine_all(output_dir: Path) -> Path:
    """Merge all source JSONL files into one combined file."""
    combined_path = output_dir / "combined_finetune.jsonl"
    all_pairs = []

    for jsonl_file in sorted(output_dir.glob("*.jsonl")):
        if jsonl_file.name == "combined_finetune.jsonl":
            continue
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    all_pairs = _dedup_pairs(all_pairs)
    _write_jsonl(all_pairs, combined_path)

    # D-11 wiring (Pass 156i8): also emit canonical-chat-format text so
    # the existing SFT training path (plain-text reader) can consume the
    # collected fine-tune data with zero training-side change.
    text_path = combined_path.with_suffix(".txt")
    text_count = _write_combined_text(all_pairs, text_path)

    size_mb = combined_path.stat().st_size / (1024 * 1024)
    text_size_mb = text_path.stat().st_size / (1024 * 1024)
    logger.info(f"Combined: {len(all_pairs):,} pairs, {size_mb:.1f} MB → {combined_path}")
    # D-11d (Pass 156l): file-present-zero-yield must be loud, not silent.
    # If we have collected pairs but every one had empty prompt/completion,
    # the .txt file is 0 bytes and the SFT path will silently train on
    # nothing. Mirror the file-present-zero-yield WARNING pattern from
    # Pass 156i6 anchor loader.
    if text_count == 0 and len(all_pairs) > 0:
        logger.warning(
            "All %d combined pairs had empty prompt or completion — text file is 0 bytes (%s). Check fetcher output.",
            len(all_pairs),
            text_path,
        )
    else:
        logger.info(f"Combined text: {text_count:,} blocks, {text_size_mb:.1f} MB → {text_path}")
    return combined_path


def show_stats(output_dir: Path):
    """Show statistics for collected fine-tuning data."""
    if not output_dir.exists():
        print("No fine-tuning data collected yet.")
        print("  Run: python collect_finetuning_data.py --all")
        return

    print(f"\n{'Source':<25} {'Pairs':>10} {'Size':>10}")
    print("-" * 47)

    total_pairs = 0
    total_bytes = 0

    for jsonl_file in sorted(output_dir.glob("*.jsonl")):
        n_lines = 0
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n_lines += 1
        size = jsonl_file.stat().st_size
        total_pairs += n_lines
        total_bytes += size

        size_str = f"{size / (1024 * 1024):.1f} MB" if size > 1024 * 1024 else f"{size / 1024:.0f} KB"
        print(f"  {jsonl_file.name:<23} {n_lines:>10,} {size_str:>10}")

    print("-" * 47)
    total_str = f"{total_bytes / (1024 * 1024):.1f} MB" if total_bytes > 1024 * 1024 else f"{total_bytes / 1024:.0f} KB"
    print(f"  {'TOTAL':<23} {total_pairs:>10,} {total_str:>10}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Collect fine-tuning data for Enigma Engine")
    parser.add_argument("--oasst", action="store_true", help="Download OASST1 conversation dataset")
    parser.add_argument("--dolly", action="store_true", help="Download Databricks Dolly 15k")
    parser.add_argument(
        "--slimorca", type=int, nargs="?", const=100000, help="Download SlimOrca (default: 100K samples)"
    )
    parser.add_argument(
        "--openthoughts3",
        type=int,
        nargs="?",
        const=100000,
        help="Download OpenThoughts3-1.2M reasoning data with <think> tags (default: 100K samples)",
    )
    parser.add_argument(
        "--smoltalk2",
        type=int,
        nargs="?",
        const=100000,
        help="Download SmolTalk2 SFT data (default: 100K samples). Requires --smoltalk2-config.",
    )
    parser.add_argument(
        "--smoltalk2-config",
        type=str,
        default="default",
        help="SmolTalk2 config name (e.g. smol_magpie_ultra). On a missing config the loader logs available choices.",
    )
    parser.add_argument(
        "--smoltalk2-split",
        type=str,
        default=None,
        help="SmolTalk2 split name within the chosen config. "
        "If omitted, all splits in the config are concatenated "
        "until --smoltalk2 max is reached.",
    )
    parser.add_argument("--all", action="store_true", help="Download all sources")
    parser.add_argument("--combine-only", action="store_true", help="Re-combine existing source files")
    parser.add_argument("--stats", action="store_true", help="Show collected data statistics")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stats:
        show_stats(output_dir)
        return

    if args.combine_only:
        combine_all(output_dir)
        return

    any_source = args.oasst or args.dolly or args.slimorca or args.openthoughts3 or args.smoltalk2 or args.all
    if not any_source:
        parser.print_help()
        return

    start = time.monotonic()
    collected = []

    if args.oasst or args.all:
        pairs = collect_oasst()
        if pairs:
            path = output_dir / "oasst1.jsonl"
            _write_jsonl(pairs, path)
            logger.info(f"Saved {len(pairs):,} pairs → {path}")
            collected.append(("OASST1", len(pairs)))

    if args.dolly or args.all:
        pairs = collect_dolly()
        if pairs:
            path = output_dir / "dolly_15k.jsonl"
            _write_jsonl(pairs, path)
            logger.info(f"Saved {len(pairs):,} pairs → {path}")
            collected.append(("Dolly 15k", len(pairs)))

    if args.slimorca is not None or args.all:
        max_n = args.slimorca if args.slimorca is not None else 100000
        pairs = collect_slimorca(max_samples=max_n)
        if pairs:
            path = output_dir / "slimorca.jsonl"
            _write_jsonl(pairs, path)
            logger.info(f"Saved {len(pairs):,} pairs → {path}")
            collected.append(("SlimOrca", len(pairs)))

    if args.openthoughts3 is not None or args.all:
        max_n = args.openthoughts3 if args.openthoughts3 is not None else 100000
        pairs = collect_openthoughts3(max_samples=max_n)
        if pairs:
            path = output_dir / "openthoughts3.jsonl"
            _write_jsonl(pairs, path)
            logger.info(f"Saved {len(pairs):,} pairs → {path}")
            collected.append(("OpenThoughts3", len(pairs)))

    if args.smoltalk2 is not None or args.all:
        max_n = args.smoltalk2 if args.smoltalk2 is not None else 100000
        pairs = collect_smoltalk2(
            max_samples=max_n,
            config=args.smoltalk2_config,
            split=args.smoltalk2_split,
        )
        if pairs:
            path = output_dir / "smoltalk2.jsonl"
            _write_jsonl(pairs, path)
            logger.info(f"Saved {len(pairs):,} pairs → {path}")
            collected.append(("SmolTalk2", len(pairs)))

    if collected:
        combine_all(output_dir)

    elapsed = time.monotonic() - start
    m, s = int(elapsed // 60), int(elapsed % 60)
    print(f"\nDone in {m}m {s:02d}s")
    for name, count in collected:
        print(f"  {name}: {count:,} pairs")
    print()
    show_stats(output_dir)


if __name__ == "__main__":
    main()
