#!/usr/bin/env python
"""Serve the REAL Enigma — the from-scratch transformer — as an OpenAI-compatible
/v1 endpoint, so Odysseus (or any OpenAI client) can talk to her.

  python serve_enigma.py                       # models/enigma_pretrain_large/latest.pth
  python serve_enigma.py --model models/enigma_pretrain_base_v2/latest.pth
  # then, in Odysseus chat:  /setup local http://127.0.0.1:8000/v1

She is a BASE model (mid-pretraining): no chat template and no tool tokens yet —
those arrive with the instruct pass (special-token IDs 4718-4735 are reserved in
the padded embedding). /v1/chat/completions therefore bridges by rendering the
conversation as a plain-text transcript she continues; /v1/completions is her
native shape.

Replaces the rejected Qwen-wrapper server (the "Muppet"; its <tool_call>
parsing lives in git history and returns with the instruct pass).
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import threading
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from enigma_engine.core.chat_format import (
    CHAT_FORMAT_NAME,
    IM_END,
    ROLES,
    attach_chat_tokens,
    parse_assistant_ids,
    render_chat,
    render_tools_system,
)
from enigma_engine.core.calculator import CalcError, evaluate, format_result
from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.core.tokenizer import get_tokenizer

try:  # Windows consoles default to cp1252 and crash printing unicode.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
MODEL_ID = "enigma"

_p = argparse.ArgumentParser()
_p.add_argument(
    "--model",
    default=str(ROOT / "models" / "enigma_pretrain_large" / "latest.pth"),
    help="Enigma checkpoint (.pth with model_state_dict + config)",
)
_p.add_argument("--host", default="127.0.0.1")
_p.add_argument("--port", type=int, default=8000)
_p.add_argument(
    "--max-context",
    type=int,
    default=1024,
    help="prompt+generation token budget; she trains at block 1024 — longer is mechanically possible but untested",
)
_p.add_argument(
    "--memory-dir",
    default=None,
    help="enable the local memory store (JSONL + BM25); relevant memories are injected into her system context",
)
ARGS, _ = _p.parse_known_args()

print(f"Loading Enigma from {ARGS.model} ...", flush=True)
if not Path(ARGS.model).exists():
    raise SystemExit(
        f"checkpoint not found: {ARGS.model}\n"
        "Pass --model <path to an Enigma .pth checkpoint> (the default only "
        "exists inside a repo checkout with trained models)"
    )
_ck = torch.load(ARGS.model, map_location="cpu", weights_only=False)  # our own checkpoint
if not (isinstance(_ck, dict) and "model_state_dict" in _ck and "config" in _ck):
    raise SystemExit(f"{ARGS.model} is not an Enigma checkpoint (need model_state_dict + config)")
CONFIG = ForgeConfig.from_dict(_ck["config"])
model = Enigma(CONFIG)
model.load_state_dict(_ck["model_state_dict"], strict=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE).eval()
STEP = _ck.get("step")
META = _ck.get("meta") or {}  # finetune_enigma stamps chat_format here
del _ck

# Always keep this many ids of the context free for the reply. Prompt and
# generation share the fixed max_context window; without a reserve a large
# client max_tokens would shrink the prompt budget toward zero and the model
# would answer from a near-empty context (confident garbage). We keep the
# prompt intact (up to max_context - MIN_GEN_TOKENS) and let generation take
# whatever room is left — never the other way around.
MIN_GEN_TOKENS = 64

# Prompt truncation budgets against ARGS.max_context, but the model's KV
# cache holds min(max_seq_len, MAX_CACHE_SEQ_LEN) positions and refuses a
# larger prefill outright — clamp so an oversize --max-context cannot let
# prompts through that the cache will reject.
from enigma_engine.core.model_components import Attention as _Attn

_CACHE_CAP = min(CONFIG.max_seq_len, _Attn.MAX_CACHE_SEQ_LEN)
if ARGS.max_context > _CACHE_CAP:
    print(
        f"  WARN: --max-context {ARGS.max_context} exceeds the model's KV cache "
        f"capacity {_CACHE_CAP}; clamping to {_CACHE_CAP}",
        flush=True,
    )
    ARGS.max_context = _CACHE_CAP
if ARGS.max_context <= MIN_GEN_TOKENS:
    raise SystemExit(
        f"max_context {ARGS.max_context} leaves no prompt budget after the "
        f"{MIN_GEN_TOKENS}-token generation reserve; this model context is too small to serve"
    )

tokenizer = get_tokenizer("bpe")  # the exact tokenizer that built tokens.bin
if getattr(tokenizer, "vocab_size", None) != CONFIG.vocab_size:
    print(
        f"  WARN: tokenizer vocab {getattr(tokenizer, 'vocab_size', '?')} != model vocab {CONFIG.vocab_size}",
        flush=True,
    )
EOS_ID = getattr(tokenizer, "eos_token_id", 2)
BOS_ID = getattr(tokenizer, "bos_token_id", 1)

# Instruct mode: SFT checkpoints (finetune_enigma.py) carry meta.chat_format.
# Base checkpoints get the plain-transcript bridge below. Attaching the chat
# tokens is safe either way — plain text encodes byte-identically.
INSTRUCT = META.get("chat_format") == CHAT_FORMAT_NAME
attach_chat_tokens(tokenizer)

MEMORY = None
if ARGS.memory_dir:
    from enigma_engine.core.memory_store import MemoryStore

    MEMORY = MemoryStore(ARGS.memory_dir)

_n_params = sum(p.numel() for p in model.parameters())
print(
    f"Enigma loaded: {_n_params / 1e6:.1f}M params on {DEVICE}"
    + (f", checkpoint step {STEP:,}" if STEP is not None else "")
    + (f" | INSTRUCT ({META.get('chat_format')})" if INSTRUCT else " | base (transcript bridge)")
    + (f" | memory: {len(MEMORY)} entries" if MEMORY is not None else ""),
    flush=True,
)

app = FastAPI(title="Enigma (from-scratch)")

# One model, one KV-cache — generation must be serialized across requests.
_GEN_LOCK = threading.Lock()

# Transcript turn markers: a base model will happily continue the whole
# conversation, so cut her off when she starts writing the next turn.
_STOP_TEXTS = ("\nUser:", "\nEnigma:")


class Msg(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None  # assistant history (instruct mode)


class ChatReq(BaseModel):
    model: str = MODEL_ID
    messages: list[Msg]
    temperature: float = 0.8
    top_p: float = 0.9
    min_p: float = 0.0  # 0 = off; prunes tokens below min_p * max_prob
    max_tokens: int = 256
    stream: bool = False
    tools: list[dict] | None = None  # OpenAI tool specs (instruct mode)


class CompletionReq(BaseModel):
    model: str = MODEL_ID
    prompt: str
    temperature: float = 0.8
    top_p: float = 0.9
    min_p: float = 0.0  # 0 = off; prunes tokens below min_p * max_prob
    max_tokens: int = 256
    stream: bool = False


def _render_transcript(messages: list[Msg]) -> str:
    """Plain-text bridge for a base model: render the conversation as a
    transcript she can continue. Replaced by a real chat template once the
    instruct pass defines special tokens."""
    lines = []
    for m in messages:
        text = (m.content or "").strip()
        if not text:
            continue
        if m.role == "system":
            lines.append(text)
        elif m.role == "assistant":
            lines.append(f"Enigma: {text}")
        else:
            lines.append(f"User: {text}")
    lines.append("Enigma:")
    return "\n".join(lines)


def _find_stop(text: str, stop_texts: tuple[str, ...]) -> int:
    hits = [i for i in (text.find(s) for s in stop_texts) if i != -1]
    return min(hits) if hits else -1


_GEN_DONE = object()


def _stream_ids_locked(
    x: torch.Tensor, max_tokens: int, temperature: float, top_p: float, min_p: float, stop_tokens: list[int]
):
    """Yield token ids from the model without holding _GEN_LOCK across
    consumer waits. A worker thread owns the lock for the whole generation
    and hands ids over an unbounded queue, so a slow or stalled SSE client
    can never keep other requests blocked on the lock; the ids themselves
    are bounded by max_tokens. A consumer that goes away (client disconnect)
    sets the cancel flag and the worker stops at the next token."""
    q: queue.Queue = queue.Queue()
    cancel = threading.Event()

    def worker():
        try:
            with _GEN_LOCK, torch.no_grad():
                for t in model.generate_stream(
                    x,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop_tokens=stop_tokens,
                    min_p=min_p,
                ):
                    if cancel.is_set():
                        break
                    q.put(int(t.item()))
        except BaseException as exc:
            # Also print: after a client disconnect nobody drains the queue,
            # and a generation error (e.g. CUDA state) must not vanish.
            print(f"generation worker error: {exc!r}", flush=True)
            q.put(exc)
        finally:
            q.put(_GEN_DONE)

    threading.Thread(target=worker, daemon=True).start()
    try:
        while True:
            item = q.get()
            if item is _GEN_DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        cancel.set()


def _generate_text(
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop_texts: tuple[str, ...] = (),
    min_p: float = 0.0,
    stats: dict | None = None,
):
    """Yield text deltas from her KV-cached streaming path.

    Decoding re-decodes the full output each token (O(n²) chars, trivial at
    n<=max_tokens) so BPE merges never split mid-character across deltas. The
    last len(stop)-1 chars are held back until we know a stop marker isn't
    forming, then flushed.

    ``stats`` (if given) is filled once the generator finishes:
    prompt_tokens = ids actually fed, completion_tokens = ids actually
    sampled, finish = "stop" (eos / stop marker) or "length" (budget spent).
    Re-encoding the decoded text is NOT a substitute — strip() plus BPE
    re-merge under-count what the model really produced.
    """
    # encode() brackets text as [BOS]...[EOS]; drop the trailing EOS so she
    # CONTINUES the prompt instead of seeing a finished document, and ensure
    # BOS survives any context trim (mirrors sample_enigma.py).
    ids = tokenizer.encode(prompt)
    if ids and ids[-1] == EOS_ID:
        ids = ids[:-1]
    # Clamp the GENERATION side too: she trains at block 1024, and the RoPE
    # table ends at 2x max_seq_len — an unclamped client max_tokens could walk
    # past both. (2026-06-11 audit finding.)
    # Reserve the prompt first, generation second: keep the most recent
    # prompt context (up to max_context - MIN_GEN_TOKENS ids, leaving 1 for
    # BOS), then give generation whatever room is left. Prompt + generation
    # always fit in max_context; the prompt is never squeezed to a stub.
    prompt_cap = ARGS.max_context - MIN_GEN_TOKENS
    if len(ids) > prompt_cap - 1:
        ids = ids[-(prompt_cap - 1) :]  # keep the most recent context
    if not ids or ids[0] != BOS_ID:
        ids = [BOS_ID] + ids
    max_tokens = max(1, min(int(max_tokens), ARGS.max_context - len(ids)))
    if stats is None:
        stats = {}
    stats["prompt_tokens"] = len(ids)
    stats["completion_tokens"] = 0
    stats["finish"] = "stop"
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    temperature = max(float(temperature), 1e-3)  # sampling requires > 0
    hold = max((len(s) for s in stop_texts), default=1) - 1
    emitted = 0
    saw_eos = False
    out_ids: list[int] = []
    for tid in _stream_ids_locked(x, max_tokens, temperature, top_p, min_p, [EOS_ID]):
        if tid == EOS_ID:
            saw_eos = True
            break
        out_ids.append(tid)
        text = tokenizer.decode(out_ids)
        cut = _find_stop(text, stop_texts)
        if cut != -1:
            stats["completion_tokens"] = len(out_ids)
            if cut > emitted:
                yield text[emitted:cut]
            return
        safe_end = max(emitted, len(text) - hold)
        if safe_end > emitted:
            yield text[emitted:safe_end]
            emitted = safe_end
    stats["completion_tokens"] = len(out_ids)
    if not saw_eos and len(out_ids) >= max_tokens:
        stats["finish"] = "length"  # budget spent, not a natural end
    # Natural end (eos or token budget): flush the held tail.
    text = tokenizer.decode(out_ids)
    cut = _find_stop(text, stop_texts)
    if cut != -1:
        # A stop marker completed inside the held tail: this is a marker
        # stop, not a budget cut.
        text = text[:cut]
        stats["finish"] = "stop"
    if len(text) > emitted:
        yield text[emitted:]


def _gen_ids(
    ids: list[int], max_tokens: int, temperature: float, top_p: float, min_p: float, stop_ids: tuple[int, ...]
):
    """ID-level generation for instruct mode: render_chat already built the
    exact prompt (BOS included, no trailing EOS — the whole encode() EOS
    gotcha is bypassed). Yields raw token ids; stops on EOS/<|im_end|>."""
    # Defensive: prompt + generation must fit in max_context (caller already
    # sized max_tokens against len(ids), but never let a bad caller overflow).
    max_tokens = max(1, min(int(max_tokens), ARGS.max_context - len(ids)))
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    temperature = max(float(temperature), 1e-3)
    for tid in _stream_ids_locked(x, max_tokens, temperature, top_p, min_p, list(stop_ids)):
        if tid in stop_ids:
            break
        yield tid


def _last_user_text(messages: list[Msg]) -> str:
    for m in reversed(messages):
        if m.role == "user" and m.content:
            return m.content
    return ""


# Built-in tools serve executes ITSELF (no client round-trip), in the same
# spec shape make_sft_data trains on (flat params). calculate: a from-scratch
# 182M model can't compute arithmetic in-weights (tokenizer splits numbers
# inconsistently). remember: the ChatGPT-bio-tool pattern -- she calls it when
# the user states a fact worth keeping, serve writes it to the MemoryStore,
# and render_context injects it back on every future relevant ask.
_CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression and return the exact result.",
        "parameters": {"expression": "string"},
    },
}
_REMEMBER_TOOL = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": "Save a fact about the user to long-term memory.",
        "parameters": {"text": "string"},
    },
}
_BUILTIN_NAMES = {"calculate", "remember"}
_MAX_TOOL_HOPS = 3  # bound the execute->regenerate loop so it can't spin

# The calculate tool is offered ONLY when the ask looks arithmetic. Injecting
# it on EVERY request poisoned normal chat: identity/factual training never saw
# a tool system-prompt, so a permanent one dragged the model into "calculator
# mode" (garbage like "your syntax violates the validity criterion" for
# "sum yourself up"). Gate on intent instead -- non-math chat gets no tool
# prompt and behaves; arithmetic gets the calculator.
_ARITH_KEYWORDS = re.compile(
    r"\b(plus|minus|times|divided|divide|multiply|multiplied|product|sum|subtract|"
    r"square root|squared|cubed|percent|modulo|remainder|to the power)\b",
    re.IGNORECASE,
)
# A digit next to an arithmetic operator, e.g. "7 * 8", "100 / 4", "2 ** 10".
_ARITH_SYMBOLS = re.compile(r"\d\s*[-+*/%^]\s*\d|\d\s*(?:x|X)\s*\d")


def _looks_arithmetic(text: str) -> bool:
    if not text:
        return False
    has_digit = any(c.isdigit() for c in text)
    return has_digit and bool(_ARITH_KEYWORDS.search(text) or _ARITH_SYMBOLS.search(text))


# remember is offered only when the message states something save-worthy:
# an explicit remember ask, or a first-person fact/preference. Same rationale
# as the calculate gate -- an ever-present tool prompt degrades normal chat.
_MEMORABLE = re.compile(
    r"\b(remember|don'?t forget|note (that|this)|keep in mind|save (this|that)|"
    r"call me|my name('s| is)|"
    # "my <up to 3 words> is/are": covers "my dog's name is", "my favorite
    # season is" (two attribute words -- a single-\w+ pattern missed it,
    # measured 2026-07-06). Offering is cheap; she decides whether to call.
    r"my (\w+('s)? ){1,3}(is|are)\b|"
    r"i (like|love|hate|prefer|live|work|drive|play|always|never|usually)|"
    r"i'?m (allergic|from|married|working))",
    re.IGNORECASE,
)


def _looks_memorable(text: str) -> bool:
    return bool(text) and MEMORY is not None and bool(_MEMORABLE.search(text))


def _builtin_tools(user_text: str, client_mode: bool) -> list[dict]:
    """The built-ins to offer for this request. calculate rides along in
    client tool-mode (a tool prompt exists anyway; math grammar is distinctive
    enough that it never steals calls). remember is intent-gated ALWAYS:
    offered merely because other tools were, it stole tool calls -- measured
    2026-07-06, 'Check the weather in Toronto' -> remember("User's weather
    for Toronto is correct") instead of get_weather."""
    tools = []
    if client_mode or _looks_arithmetic(user_text):
        tools.append(_CALC_TOOL)
    if _looks_memorable(user_text):  # checks MEMORY is enabled too
        tools.append(_REMEMBER_TOOL)
    return tools


def _execute_builtin(name: str, arguments: dict) -> str:
    """Run a server-side built-in tool and return its result string. Errors
    come back as text (fed to the model) rather than raising -- an engine that
    fails honestly beats one that 500s mid-conversation."""
    # The model can emit any valid JSON as "arguments" (a bare string parses
    # fine); only an object supports .get(), so anything else is a tool error
    # fed back to her, not an exception.
    if not isinstance(arguments, dict):
        return f"error: tool arguments must be a JSON object, got {type(arguments).__name__}"
    if name == "calculate":
        expr = str(arguments.get("expression", "")).strip()
        try:
            return format_result(evaluate(expr))
        except CalcError as exc:
            return f"error: {exc}"
    if name == "remember":
        if MEMORY is None:
            return "error: memory disabled (start serve with --memory-dir)"
        text = str(arguments.get("text", "")).strip()
        if not text:
            return "error: nothing to remember"
        rec = MEMORY.remember(text, source="chat")
        return f"updated: {rec['text']}" if rec.get("superseded") else f"saved: {rec['text']}"
    return f"error: unknown tool {name!r}"


def _with_context(msgs: list[dict], req: ChatReq) -> list[dict]:
    """Fold tool specs and retrieved memories into the system message. The
    built-in calculate tool is ALWAYS offered alongside any client tools."""
    extra = []
    if MEMORY is not None:
        mem = MEMORY.render_context(_last_user_text(req.messages), tokenizer, max_ids=128)
        if mem:
            extra.append(mem)
    # Built-ins are gated on intent (see _builtin_tools); client tools are
    # always honored.
    client_tools = list(req.tools or [])
    all_tools = _builtin_tools(_last_user_text(req.messages), bool(client_tools)) + client_tools
    if all_tools:
        tools_block = render_tools_system(all_tools)
        if not (msgs and msgs[0].get("role") == "system"):
            # Training's tool examples ALWAYS lead with this exact preamble
            # (make_sft_data._system, single \n before "Available tools:");
            # a system message that OPENS with "Available tools:" is a shape
            # the model never saw.
            tools_block = (
                "You are Enigma. You can use tools when they are needed; "
                "answer directly when they are not.\n" + tools_block
            )
        extra.append(tools_block)
    if not extra:
        return msgs
    if msgs and msgs[0].get("role") == "system":
        head = dict(msgs[0])
        head["content"] = "\n\n".join([head.get("content") or ""] + extra).strip()
        return [head] + msgs[1:]
    return [{"role": "system", "content": "\n\n".join(extra)}] + msgs


def _openai_tool_calls(calls: list[dict]) -> list[dict]:
    return [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": c.get("name"), "arguments": json.dumps(c.get("arguments", {}), ensure_ascii=False)},
        }
        for i, c in enumerate(calls)
        if c.get("name")
    ]


def _chat_instruct(req: ChatReq):
    msgs = _with_context([m.model_dump(exclude_none=True) for m in req.messages], req)
    created = int(time.time())
    cid = f"chatcmpl-{created}"

    def _hop(cur_msgs: list[dict]):
        # Reserve the prompt first (render into max_context - MIN_GEN_TOKENS ids),
        # then let generation take the room that's left; a large client max_tokens
        # can no longer starve the prompt.
        prompt_ids = render_chat(
            tokenizer, cur_msgs, add_generation_prompt=True, max_ids=ARGS.max_context - MIN_GEN_TOKENS
        )
        hop_max = max(1, min(int(req.max_tokens), ARGS.max_context - len(prompt_ids)))
        return prompt_ids, hop_max

    def _apply_builtins(cur_msgs: list[dict], out: dict, parsed: list[dict]) -> list[dict]:
        # Append the assistant turn that made the calls, then each built-in's
        # result, so the next hop sees a coherent tool trace.
        named = [{"name": c["name"], "arguments": c.get("arguments") or {}} for c in parsed if c.get("name")]
        nxt = cur_msgs + [{"role": "assistant", "content": out.get("content") or "", "tool_calls": named}]
        for c in parsed:
            if c.get("name") in _BUILTIN_NAMES:
                nxt = nxt + [{"role": "tool", "content": _execute_builtin(c["name"], c.get("arguments") or {})}]
        return nxt

    def _loop_on_builtins(parsed: list[dict], hop: int) -> bool:
        # Execute-and-loop ONLY when every named call is a built-in and hops
        # remain. A mixed batch (built-in + client tool) is surfaced whole
        # instead: executing half and looping would leave the client's call
        # dangling in the trace and never delivered. Likewise a built-in call
        # on the final hop is surfaced, not dropped -- the client seeing a
        # "calculate" call it can answer beats the action silently vanishing.
        named = [c for c in parsed if c.get("name")]
        return bool(named) and all(c["name"] in _BUILTIN_NAMES for c in named) and hop < _MAX_TOOL_HOPS

    if req.stream:

        def events():
            from enigma_engine.core.chat_format import THINK, THINK_END, TOOL_CALL, TOOL_CALL_END

            cur_msgs = msgs
            raw_all: list[str] = []
            for hop in range(_MAX_TOOL_HOPS + 1):
                prompt_ids, hop_max = _hop(cur_msgs)
                gen = _gen_ids(prompt_ids, hop_max, req.temperature, req.top_p, req.min_p, (EOS_ID, IM_END))
                all_ids: list[int] = []
                content_ids: list[int] = []
                emitted = 0
                depth = 0
                for tid in gen:
                    all_ids.append(tid)
                    if tid in (THINK, TOOL_CALL):
                        depth += 1
                        continue
                    if tid in (THINK_END, TOOL_CALL_END):
                        depth = max(0, depth - 1)
                        continue
                    if depth:
                        continue  # span ids surface at the end, parsed — not as text
                    content_ids.append(tid)
                    text = tokenizer.decode(content_ids, skip_special_tokens=True)
                    if len(text) > emitted:
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "id": cid,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": MODEL_ID,
                                    "choices": [
                                        {"index": 0, "delta": {"content": text[emitted:]}, "finish_reason": None}
                                    ],
                                }
                            )
                            + "\n\n"
                        )
                        emitted = len(text)
                out = parse_assistant_ids(tokenizer, all_ids)
                parsed = out["tool_calls"]
                # Unparsable call text is collected across hops -- a malformed
                # action generated alongside an executed built-in surfaces at
                # the end instead of vanishing with the loop.
                raw_all += [c["raw"] for c in parsed if not c.get("name") and c.get("raw")]
                # A built-in-only batch (calculate) is executed here, then we
                # loop to let the model answer from the result -- the client
                # never sees it. Anything else is surfaced whole.
                if _loop_on_builtins(parsed, hop):
                    cur_msgs = _apply_builtins(cur_msgs, out, parsed)
                    continue
                calls = _openai_tool_calls(parsed)
                if raw_all:
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": MODEL_ID,
                                "choices": [
                                    {"index": 0, "delta": {"content": "\n".join(raw_all)}, "finish_reason": None}
                                ],
                            }
                        )
                        + "\n\n"
                    )
                if calls:
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "id": cid,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": MODEL_ID,
                                "choices": [{"index": 0, "delta": {"tool_calls": calls}, "finish_reason": None}],
                            }
                        )
                        + "\n\n"
                    )
                finish = "tool_calls" if calls else ("length" if len(all_ids) >= hop_max else "stop")
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_ID,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                        }
                    )
                    + "\n\n"
                )
                break
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    # Non-stream: run the built-in tool loop to completion, accumulating usage.
    cur_msgs = msgs
    n_prompt = n_out = 0
    out: dict = {"content": "", "tool_calls": []}
    out_ids: list[int] = []
    last_max = 1
    raw_all: list[str] = []
    for hop in range(_MAX_TOOL_HOPS + 1):
        prompt_ids, last_max = _hop(cur_msgs)
        out_ids = list(_gen_ids(prompt_ids, last_max, req.temperature, req.top_p, req.min_p, (EOS_ID, IM_END)))
        out = parse_assistant_ids(tokenizer, out_ids)
        n_prompt += len(prompt_ids)
        n_out += len(out_ids)
        parsed = out["tool_calls"]
        # A tool call whose JSON didn't parse has no name — its raw text is
        # collected across hops and surfaced as content instead of silently
        # dropping the model's action.
        raw_all += [c["raw"] for c in parsed if not c.get("name") and c.get("raw")]
        if _loop_on_builtins(parsed, hop):
            cur_msgs = _apply_builtins(cur_msgs, out, parsed)
            continue
        break

    calls = _openai_tool_calls(out["tool_calls"])
    content = "\n".join(t for t in [out.get("content"), *raw_all] if t)
    message = {"role": "assistant", "content": content or (None if calls else "")}
    if calls:
        message["tool_calls"] = calls
    # honest finish_reason: a generation that spent the whole budget was cut
    # off ("length"), not naturally finished ("stop")
    finish = "tool_calls" if calls else ("length" if len(out_ids) >= last_max else "stop")
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out},
    }


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "enigma"}]}


@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    # Validate roles up front: an unknown role is CLIENT error (400), not a
    # server crash (500). Without this the ValueError from chat_format's
    # renderer leaks a stack trace and returns a generic 500.
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")
    bad = sorted({m.role for m in req.messages if m.role not in ROLES})
    if bad:
        raise HTTPException(status_code=400, detail=f"unknown chat role(s) {bad}; need one of {list(ROLES)}")
    if INSTRUCT:
        return _chat_instruct(req)
    messages = list(req.messages)
    if MEMORY is not None:
        mem = MEMORY.render_context(_last_user_text(messages), tokenizer, max_ids=128)
        if mem:
            messages = [Msg(role="system", content=mem)] + messages
    prompt = _render_transcript(messages)
    created = int(time.time())
    cid = f"chatcmpl-{created}"
    stats: dict = {}
    gen = _generate_text(prompt, req.max_tokens, req.temperature, req.top_p, _STOP_TEXTS, min_p=req.min_p, stats=stats)

    if req.stream:

        def events():
            for delta in gen:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_ID,
                            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                        }
                    )
                    + "\n\n"
                )
            yield (
                "data: "
                + json.dumps(
                    {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": stats.get("finish", "stop")}],
                    }
                )
                + "\n\n"
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    text = "".join(gen).strip()
    # Usage is ground truth from _generate_text: ids actually fed and ids
    # actually sampled. (Re-encoding the stripped text under-counted — BPE
    # re-merge + strip() lost tokens the model really produced.)
    n_prompt = stats["prompt_tokens"]
    n_out = stats["completion_tokens"]
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": stats["finish"]}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out},
    }


@app.post("/v1/completions")
def completions(req: CompletionReq):
    created = int(time.time())
    cid = f"cmpl-{created}"
    stats: dict = {}
    gen = _generate_text(req.prompt, req.max_tokens, req.temperature, req.top_p, min_p=req.min_p, stats=stats)

    if req.stream:

        def events():
            for delta in gen:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": cid,
                            "object": "text_completion",
                            "created": created,
                            "model": MODEL_ID,
                            "choices": [{"index": 0, "text": delta, "finish_reason": None}],
                        }
                    )
                    + "\n\n"
                )
            yield (
                "data: "
                + json.dumps(
                    {
                        "id": cid,
                        "object": "text_completion",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [{"index": 0, "text": "", "finish_reason": stats.get("finish", "stop")}],
                    }
                )
                + "\n\n"
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    text = "".join(gen)
    # Usage is ground truth from _generate_text: ids actually fed and sampled.
    n_prompt = stats["prompt_tokens"]
    n_out = stats["completion_tokens"]
    return {
        "id": cid,
        "object": "text_completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "text": text, "finish_reason": stats["finish"]}],
        "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out},
    }


class MemReq(BaseModel):
    text: str
    kind: str = "fact"


@app.post("/v1/memory")
def memory_add(req: MemReq):
    if MEMORY is None:
        return {"error": "memory disabled — start with --memory-dir"}
    return {"ok": True, "memory": MEMORY.add(req.text, kind=req.kind)}


@app.get("/v1/memory")
def memory_list(q: str | None = None, k: int = 5):
    if MEMORY is None:
        return {"error": "memory disabled — start with --memory-dir"}
    recs = MEMORY.search(q, k=k) if q else MEMORY.all()[-k:]
    return {"count": len(MEMORY), "results": recs}


@app.delete("/v1/memory/{mem_id}")
def memory_delete(mem_id: int):
    """User control over her memory (the ChatGPT-memory-management parallel):
    a saved fact can always be inspected (GET) and removed."""
    if MEMORY is None:
        return {"error": "memory disabled — start with --memory-dir"}
    return {"ok": MEMORY.delete(mem_id), "count": len(MEMORY)}


@app.delete("/v1/memory")
def memory_clear():
    if MEMORY is None:
        return {"error": "memory disabled — start with --memory-dir"}
    return {"ok": True, "cleared": MEMORY.clear()}


def main() -> None:
    """Run the server. Console-script entry point (pyproject [project.scripts])
    and the __main__ path share this."""
    print(f"Enigma OpenAI-compatible API -> http://{ARGS.host}:{ARGS.port}/v1", flush=True)
    print(f"In Odysseus:  /setup local http://{ARGS.host}:{ARGS.port}/v1", flush=True)
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning")


if __name__ == "__main__":
    main()
