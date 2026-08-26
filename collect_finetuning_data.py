"""
Fine-Tuning Data Collector for Enigma Engine
=============================================
Downloads and formats instruction-following datasets for fine-tuning.

Sources (all require `pip install datasets`):
  - OASST1 (--oasst)        - Open Assistant conversations (~80K turns)
  - Dolly 15k (--dolly)     - Databricks instruction pairs (~15K)
  - SlimOrca (--slimorca N) - Instruction-following from Open-Orca (N = max samples)
  - OpenThoughts3 (--openthoughts3 N) - Reasoning traces with <think> tags (D-4).
      EXCLUDED from --all since 2026-07-15: median completion ~14.5k tokens,
      far past the training block (1024 then, 2048 now) -- every record was
      silently dropped at bake (audit).
  - SmolTalk2 (--smoltalk2 N --smoltalk2-config NAME [--smoltalk2-split NAME]) - SmolLM3 SFT data (D-11)
  - No Robots (--no-robots N)   - ~10K HUMAN-written instruction pairs (small-model-native)
  - Everyday Conversations (--everyday N) - simple smalltalk aimed at 1-3B models
  - TriviaQA (--triviaqa N)     - factoid questions with SHORT answers (recall training)
  - NQ-Open (--nq-open N)       - real search queries with short answers (recall training)
  - All sources (--all)     - Download everything above except OpenThoughts3

The 2026-07-15 diet: small-model-native SHORT records (completion caps tuned
when the block was 1024) + short-answer recall sets as the counterweight to
Dolly's extract-from-context bias. Those caps are STYLE filters, not block
filters, and they STAND at block 2048: measured 2026-08-19, 56 of the
105,203 combined records render past 2048 against the serving vocab, and all
but two of them are already longer than the 600-char completion / 800-char
prompt caps allow. Raising the caps to match the block would buy nothing and
would cost the short-record diet; no re-tune is queued.

Output format:
  JSONL with {"prompt": "...", "completion": "..."} per line.

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
import os
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
        # NUL separator keeps ("ab","c") distinct from ("a","bc")
        key = hashlib.sha256((item["prompt"] + "\x00" + item["completion"]).encode()).digest()[:16]
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


# The recorded diet (BACKLOG, 2026-07-15 build): SmolTalk2 completion cap.
# One knob drives three diet behaviors in collect_smoltalk2 (completion cap,
# prompt cap, *_think split skip), so the DEFAULT is the diet, not None.
_DIET_SMOLTALK2_CAP = 600


def _write_jsonl(pairs: list[dict], path: Path) -> int:
    """Write pairs to JSONL, atomically: combine_all rebuilds the LIVE
    combined corpus in place at the end of ANY partial run, and a Ctrl-C
    mid-write left a truncated corpus with no backup (review 2026-08-13)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return len(pairs)


def _write_combined_text(pairs: list[dict], path: Path) -> int:
    """Write pairs as canonical 'User: ...\\n\\nAssistant: ...' blocks.

    The plain-transcript dual-emit format for `collect_search_data.py`, its
    one remaining caller. No training path reads the text: `make_sft_data.py`
    and `make_pretrain_curated.py` load `combined_finetune.jsonl`, and
    `finetune_enigma.py` has no plain-text branch at all.

    Empty prompts or completions are skipped — a malformed block like
    'User: \\n\\nAssistant: foo' would teach the model that empty input
    is a valid prefix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    # newline="": without it Windows wrote CRLF into every emitted block --
    # a stray \r in training bytes if this path is ever wired to a trainer
    # (review 2026-08-13; the tests' read_text() universal-newline mode was
    # structurally blind to it).
    with open(path, "w", encoding="utf-8", newline="") as f:
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
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
    except Exception as exc:
        logger.error("Failed to load OASST1: %s", exc)
        return []
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
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    except Exception as exc:
        logger.error("Failed to load Dolly 15k: %s", exc)
        return []
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
    max_completion_chars: int | None = None,
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
    # With a completion cap, max_samples counts KEPT pairs (long records
    # yield nothing); a 10x processing budget bounds runtime on splits that
    # are mostly long. Without a cap, behavior is unchanged: max_samples
    # counts processed rows.
    budget = max_samples if max_completion_chars is None else 10 * max_samples
    for split_name in splits:
        if len(pairs) >= max_samples or seen >= budget:
            break
        if max_completion_chars is not None and "think" in split_name and "no_think" not in split_name:
            # Think traces cannot fit a chat completion cap; streaming a
            # 1.2M-row think split would burn the whole budget for ~zero
            # yield (the OpenThoughts3 lesson, audit 2026-07-15).
            logger.info(f"  SmolTalk2: skipping think split {split_name!r} (completion cap active)")
            continue
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
            if len(pairs) >= max_samples or seen >= budget:
                break
            seen += 1

            messages = item.get("messages") or []
            # Optional system prompt, prepended to first user turn.
            system = ""
            prompt = ""
            completion = ""
            malformed = False
            for turn in messages:
                role = turn.get("role", "")
                raw = turn.get("content", "") or ""
                # Structured content (a list of content parts) is outside the
                # plain-string schema this collector handles. The whole record
                # is dropped: skipping just the turn could pair the first user
                # prompt with a later assistant reply to a different question.
                if not isinstance(raw, str):
                    malformed = True
                    break
                # `<think>` content needs its newlines verbatim (the tags align
                # with special token IDs) -- same handling as collect_openthoughts3.
                # Non-think content keeps whitespace-collapsing cleanup.
                if "<think>" in raw:
                    content = raw.strip()
                else:
                    content = _clean_text(raw)
                if role == "system" and content and not system:
                    system = content
                elif role == "user" and content and not prompt:
                    prompt = content
                elif role == "assistant" and content and not completion:
                    completion = content
                    # first assistant reply is enough for SFT pair
                    break

            if malformed or not prompt or not completion:
                continue
            if len(prompt) < 5 or len(completion) < 10:
                continue
            if max_completion_chars is not None and (
                len(completion) > max_completion_chars or len(prompt) > 800
            ):
                # Prompt cap rides the same diet switch: LongAlign-style
                # splits pair 64k-char contexts with short answers -- the
                # completion cap alone let 13k of 40k records through with
                # prompts no training block this side of 64k chars can keep
                # as anything but truncated garbage (measured 2026-07-15
                # against block 1024; 2048 is no closer).
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


# ── No Robots ──────────────────────────────────────────────────────


def _first_exchange(
    messages: list[dict],
    max_completion_chars: int,
    max_prompt_chars: int = 800,
    min_prompt_chars: int = 5,
) -> dict | None:
    """First QUALIFYING user->assistant exchange from a ChatML messages
    list, or None. Walks past leading throwaway exchanges ("Hi" -> canned
    greeting; measured: Everyday Conversations yielded 4 of 2,260 records
    when only the literal first pair was considered). Out-of-range records
    are DROPPED, not truncated -- the diet wants data that is short by
    nature, not amputated prose."""
    prompt = ""
    for turn in messages or []:
        role = turn.get("role", "")
        raw = turn.get("content", "")
        if not isinstance(raw, str):
            return None
        content = _clean_text(raw)
        if role == "user":
            prompt = content
        elif role == "assistant" and prompt:
            if min_prompt_chars <= len(prompt) <= max_prompt_chars and 10 <= len(content) <= max_completion_chars:
                return {"prompt": prompt, "completion": content}
            prompt = ""  # this exchange failed the bounds; try the next one
    return None


def collect_no_robots(max_samples: int = 12000, max_completion_chars: int = 600) -> list[dict]:
    """Download and format No Robots (`HuggingFaceH4/no_robots`).

    ~10K instruction pairs written by HUMANS, not distilled from a big
    model -- short, direct, none of the synthetic-teacher verbosity that
    overflows the training block. Schema: ChatML `messages`."""
    if not _ensure_datasets():
        return []
    from datasets import load_dataset

    logger.info(f"Downloading No Robots (max {max_samples:,}, completions <= {max_completion_chars} chars)...")
    try:
        ds = load_dataset("HuggingFaceH4/no_robots", split="train", streaming=True)
    except Exception as exc:
        logger.error("Failed to load No Robots: %s", exc)
        return []

    pairs: list[dict] = []
    for seen, item in enumerate(ds, 1):
        if seen > max_samples:
            break
        pair = _first_exchange(item.get("messages"), max_completion_chars)
        if pair:
            pairs.append(pair)
    pairs = _dedup_pairs(pairs)
    logger.info(f"No Robots: {len(pairs)} instruction pairs extracted")
    return pairs


# ── Everyday Conversations ─────────────────────────────────────────


def collect_everyday_conversations(max_samples: int = 5000, max_completion_chars: int = 600) -> list[dict]:
    """Download and format Everyday Conversations
    (`HuggingFaceTB/everyday-conversations-llama3.1-2k`).

    ~2.2K simple multi-turn chats authored FOR 1-3B models -- smalltalk,
    everyday topics, short replies. First exchange per conversation only
    (later turns depend on context this pair format cannot carry)."""
    if not _ensure_datasets():
        return []
    from datasets import load_dataset

    logger.info(f"Downloading Everyday Conversations (max {max_samples:,})...")
    try:
        ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train_sft", streaming=True)
    except Exception as exc:
        logger.error("Failed to load Everyday Conversations: %s", exc)
        return []

    pairs: list[dict] = []
    for seen, item in enumerate(ds, 1):
        if seen > max_samples:
            break
        # min_prompt_chars=15 skips the scripted "Hi"/"Hey!" opener pair;
        # the substantive exchange is the second one in this dataset.
        pair = _first_exchange(item.get("messages"), max_completion_chars, min_prompt_chars=15)
        if pair:
            pairs.append(pair)
    pairs = _dedup_pairs(pairs)
    logger.info(f"Everyday Conversations: {len(pairs)} instruction pairs extracted")
    return pairs


# ── TriviaQA (short-answer recall) ─────────────────────────────────


def collect_triviaqa(max_samples: int = 25000, max_answer_chars: int = 80) -> list[dict]:
    """Download and format TriviaQA no-context
    (`mandarjoshi/trivia_qa`, config `rc.nocontext`).

    Factoid question -> SHORT canonical answer: pure recall training, the
    counterweight to Dolly's extract-from-context bias (audit 2026-07-15).
    Long answers and long questions are dropped, not truncated."""
    if not _ensure_datasets():
        return []
    from datasets import load_dataset

    logger.info(f"Downloading TriviaQA rc.nocontext (max {max_samples:,})...")
    try:
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train", streaming=True)
    except Exception as exc:
        logger.error("Failed to load TriviaQA: %s", exc)
        return []

    pairs: list[dict] = []
    seen = 0
    for item in ds:
        if seen >= max_samples:
            break
        seen += 1
        question = _clean_text(item.get("question") or "")
        answer = _clean_text((item.get("answer") or {}).get("value") or "")
        if not question or not answer:
            continue
        if len(question) < 10 or len(question) > 300 or len(answer) > max_answer_chars:
            continue
        if not answer.endswith("."):
            answer += "."
        pairs.append({"prompt": question, "completion": answer})
        if seen % 10000 == 0:
            logger.info(f"  TriviaQA: {seen:,} processed, {len(pairs):,} kept...")
    pairs = _dedup_pairs(pairs)
    logger.info(f"TriviaQA: {len(pairs)} QA pairs extracted")
    return pairs


# ── NQ-Open (short-answer recall) ──────────────────────────────────


def collect_nq_open(max_samples: int = 15000, max_answer_chars: int = 80) -> list[dict]:
    """Download and format NQ-Open (`google-research-datasets/nq_open`).

    Real search queries with short answers. Questions arrive lowercase
    with no terminal '?' -- normalized here so the prompt shape matches
    how a user actually types in chat."""
    if not _ensure_datasets():
        return []
    from datasets import load_dataset

    logger.info(f"Downloading NQ-Open (max {max_samples:,})...")
    try:
        ds = load_dataset("google-research-datasets/nq_open", split="train", streaming=True)
    except Exception as exc:
        logger.error("Failed to load NQ-Open: %s", exc)
        return []

    pairs: list[dict] = []
    seen = 0
    for item in ds:
        if seen >= max_samples:
            break
        seen += 1
        question = _clean_text(item.get("question") or "")
        answers = item.get("answer") or []
        answer = _clean_text(answers[0]) if answers and isinstance(answers[0], str) else ""
        if not question or not answer:
            continue
        if len(question) < 10 or len(question) > 300 or len(answer) > max_answer_chars:
            continue
        question = question[0].upper() + question[1:]
        if not question.endswith("?"):
            question += "?"
        if not answer.endswith("."):
            answer += "."
        pairs.append({"prompt": question, "completion": answer})
        if seen % 10000 == 0:
            logger.info(f"  NQ-Open: {seen:,} processed, {len(pairs):,} kept...")
    pairs = _dedup_pairs(pairs)
    logger.info(f"NQ-Open: {len(pairs)} QA pairs extracted")
    return pairs


# ── Combine & Stats ───────────────────────────────────────────────


def combine_all(output_dir: Path) -> Path:
    """Merge all source JSONL files into one combined file."""
    combined_path = output_dir / "combined_finetune.jsonl"
    all_pairs = []

    skipped_foreign = 0
    per_source: dict[str, int] = {}
    for jsonl_file in sorted(output_dir.glob("*.jsonl")):
        if jsonl_file.name == "combined_finetune.jsonl":
            continue
        kept_here = 0
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Only string prompt/completion rows combine; other shapes
                # that share the directory (DPO triples, null/typed values)
                # are skipped, not a crash at the end of a long run.
                if (
                    not isinstance(row, dict)
                    or not isinstance(row.get("prompt"), str)
                    or not isinstance(row.get("completion"), str)
                ):
                    skipped_foreign += 1
                    continue
                all_pairs.append(row)
                kept_here += 1
        if kept_here:
            per_source[jsonl_file.name] = kept_here

    if skipped_foreign:
        logger.warning("Combine: skipped %d row(s) without prompt/completion (foreign shapes)", skipped_foreign)

    # SAY which files fold in: the glob sweeps whatever sits in the dir
    # (leftover excluded sources included -- synthetic_search_seed.jsonl rode
    # into the live mix this way once), and the manifest is the receipt the
    # combined file never had (review 2026-08-13).
    logger.info("Combine folds %d source file(s): %s", len(per_source),
                ", ".join(f"{n}={c}" for n, c in sorted(per_source.items())))
    all_pairs = _dedup_pairs(all_pairs)
    _write_jsonl(all_pairs, combined_path)
    manifest = {
        "sources": per_source,
        "combined_records": len(all_pairs),
        "skipped_foreign": skipped_foreign,
    }
    (output_dir / "combined_finetune.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    size_mb = combined_path.stat().st_size / (1024 * 1024)
    logger.info(f"Combined: {len(all_pairs):,} pairs, {size_mb:.1f} MB -> {combined_path}")
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
    parser.add_argument(
        "--smoltalk2-cap",
        type=int,
        default=_DIET_SMOLTALK2_CAP,
        help="Drop SmolTalk2 records whose completion exceeds N chars "
        "(also skips *_think splits). Default = the 2026-07-15 diet's 600 -- "
        "the None default made --all stream exactly the long-trace class "
        "OpenThoughts3 is excluded for (review 2026-08-13). Pass 0 to uncap "
        "deliberately.",
    )
    parser.add_argument(
        "--no-robots",
        type=int,
        nargs="?",
        const=12000,
        help="Download No Robots human-written pairs (default: 12K samples)",
    )
    parser.add_argument(
        "--everyday",
        type=int,
        nargs="?",
        const=5000,
        help="Download Everyday Conversations smalltalk (default: 5K samples)",
    )
    parser.add_argument(
        "--triviaqa",
        type=int,
        nargs="?",
        const=25000,
        help="Download TriviaQA short-answer recall pairs (default: 25K samples)",
    )
    parser.add_argument(
        "--nq-open",
        type=int,
        nargs="?",
        const=15000,
        help="Download NQ-Open short-answer recall pairs (default: 15K samples)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all sources EXCEPT OpenThoughts3 (median ~14.5k tokens, block-unfit; audit 2026-07-15)",
    )
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

    any_source = (
        args.oasst or args.dolly or args.slimorca or args.openthoughts3 or args.smoltalk2
        or args.no_robots or args.everyday or args.triviaqa or args.nq_open or args.all
    )
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
            logger.info(f"Saved {len(pairs):,} pairs -> {path}")
            collected.append(("OASST1", len(pairs)))

    if args.dolly or args.all:
        pairs = collect_dolly()
        if pairs:
            path = output_dir / "dolly_15k.jsonl"
            _write_jsonl(pairs, path)
            logger.info(f"Saved {len(pairs):,} pairs -> {path}")
            collected.append(("Dolly 15k", len(pairs)))

    # SlimOrca is NOT in --all: it was not in the 105,203-row live build and
    # carries no length filter at all -- --all must reproduce the recorded
    # diet, not a superset of it (review 2026-08-13). Explicit --slimorca
    # still works.
    if args.slimorca is not None:
        max_n = args.slimorca
        pairs = collect_slimorca(max_samples=max_n)
        if pairs:
            path = output_dir / "slimorca.jsonl"
            _write_jsonl(pairs, path)
            logger.info(f"Saved {len(pairs):,} pairs -> {path}")
            collected.append(("SlimOrca", len(pairs)))

    # OpenThoughts3 is NOT in --all: a median completion of ~14.5k tokens meant
    # 100% of records were silently dropped at bake (audit 2026-07-15, block
    # 1024) and it still overflows today's 2048 by ~7x. Explicit
    # --openthoughts3 still works for future longer blocks.
    if args.openthoughts3 is not None:
        pairs = collect_openthoughts3(max_samples=args.openthoughts3)
        if pairs:
            path = output_dir / "openthoughts3.jsonl"
            _write_jsonl(pairs, path)
            logger.info(f"Saved {len(pairs):,} pairs -> {path}")
            collected.append(("OpenThoughts3", len(pairs)))

    if args.smoltalk2 is not None or args.all:
        max_n = args.smoltalk2 if args.smoltalk2 is not None else 100000
        pairs = collect_smoltalk2(
            max_samples=max_n,
            config=args.smoltalk2_config,
            split=args.smoltalk2_split,
            max_completion_chars=(args.smoltalk2_cap or None),  # 0 = deliberate uncap
        )
        if pairs:
            path = output_dir / "smoltalk2.jsonl"
            _write_jsonl(pairs, path)
            logger.info(f"Saved {len(pairs):,} pairs -> {path}")
            collected.append(("SmolTalk2", len(pairs)))

    if args.no_robots is not None or args.all:
        max_n = args.no_robots if args.no_robots is not None else 12000
        pairs = collect_no_robots(max_samples=max_n)
        if pairs:
            path = output_dir / "no_robots.jsonl"
            _write_jsonl(pairs, path)
            logger.info(f"Saved {len(pairs):,} pairs -> {path}")
            collected.append(("No Robots", len(pairs)))

    if args.everyday is not None or args.all:
        max_n = args.everyday if args.everyday is not None else 5000
        pairs = collect_everyday_conversations(max_samples=max_n)
        if pairs:
            path = output_dir / "everyday_conversations.jsonl"
            _write_jsonl(pairs, path)
            logger.info(f"Saved {len(pairs):,} pairs -> {path}")
            collected.append(("Everyday Conversations", len(pairs)))

    if args.triviaqa is not None or args.all:
        max_n = args.triviaqa if args.triviaqa is not None else 25000
        pairs = collect_triviaqa(max_samples=max_n)
        if pairs:
            path = output_dir / "triviaqa.jsonl"
            _write_jsonl(pairs, path)
            logger.info(f"Saved {len(pairs):,} pairs -> {path}")
            collected.append(("TriviaQA", len(pairs)))

    if args.nq_open is not None or args.all:
        max_n = args.nq_open if args.nq_open is not None else 15000
        pairs = collect_nq_open(max_samples=max_n)
        if pairs:
            path = output_dir / "nq_open.jsonl"
            _write_jsonl(pairs, path)
            logger.info(f"Saved {len(pairs):,} pairs -> {path}")
            collected.append(("NQ-Open", len(pairs)))

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
