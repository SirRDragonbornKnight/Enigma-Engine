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

Replaces the rejected Qwen-wrapper server (the "Muppet", removed 2026-06-11; its
<tool_call> parsing lives in git history and returns with the instruct pass).
"""

from __future__ import annotations

import argparse
import json
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

tokenizer = get_tokenizer("bpe")  # the exact tokenizer that built tokens.bin
if getattr(tokenizer, "vocab_size", None) != CONFIG.vocab_size:
    print(
        f"  WARN: tokenizer vocab {getattr(tokenizer, 'vocab_size', '?')} != model vocab {CONFIG.vocab_size}",
        flush=True,
    )
EOS_ID = getattr(tokenizer, "eos_token_id", 2)
BOS_ID = getattr(tokenizer, "bos_token_id", 1)

# Always keep this many ids of the context free for the reply. Prompt and
# generation share the fixed max_context window; without a reserve a large
# client max_tokens would shrink the prompt budget toward zero and the model
# would answer from a near-empty context (confident garbage). We keep the
# prompt intact (up to max_context - MIN_GEN_TOKENS) and let generation take
# whatever room is left — never the other way around.
MIN_GEN_TOKENS = 64

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

app = FastAPI(title="Modkit · Enigma (from-scratch)")

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
    with _GEN_LOCK, torch.no_grad():
        for t in model.generate_stream(
            x, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p, stop_tokens=[EOS_ID], min_p=min_p
        ):
            tid = int(t.item())
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
        text = text[:cut]
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
    with _GEN_LOCK, torch.no_grad():
        for t in model.generate_stream(
            x, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p, stop_tokens=list(stop_ids), min_p=min_p
        ):
            tid = int(t.item())
            if tid in stop_ids:
                break
            yield tid


def _last_user_text(messages: list[Msg]) -> str:
    for m in reversed(messages):
        if m.role == "user" and m.content:
            return m.content
    return ""


def _with_context(msgs: list[dict], req: ChatReq) -> list[dict]:
    """Fold tool specs and retrieved memories into the system message."""
    extra = []
    if MEMORY is not None:
        mem = MEMORY.render_context(_last_user_text(req.messages), tokenizer, max_ids=128)
        if mem:
            extra.append(mem)
    if req.tools:
        extra.append(render_tools_system(req.tools))
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
    # Reserve the prompt first: render it into up to max_context - MIN_GEN_TOKENS
    # ids, then let generation take whatever room is left (capped by the client's
    # request). A large client max_tokens can no longer starve the prompt.
    prompt_ids = render_chat(
        tokenizer, msgs, add_generation_prompt=True, max_ids=ARGS.max_context - MIN_GEN_TOKENS
    )
    max_tokens = max(1, min(int(req.max_tokens), ARGS.max_context - len(prompt_ids)))
    created = int(time.time())
    cid = f"chatcmpl-{created}"
    gen = _gen_ids(prompt_ids, max_tokens, req.temperature, req.top_p, req.min_p, (EOS_ID, IM_END))

    if req.stream:

        def events():
            from enigma_engine.core.chat_format import (
                THINK,
                THINK_END,
                TOOL_CALL,
                TOOL_CALL_END,
            )

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
                                "choices": [{"index": 0, "delta": {"content": text[emitted:]}, "finish_reason": None}],
                            }
                        )
                        + "\n\n"
                    )
                    emitted = len(text)
            out = parse_assistant_ids(tokenizer, all_ids)
            calls = _openai_tool_calls(out["tool_calls"])
            # surface unparsable tool calls' raw text rather than dropping it
            raw_bits = [c["raw"] for c in out["tool_calls"] if not c.get("name") and c.get("raw")]
            if raw_bits:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "id": cid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_ID,
                            "choices": [
                                {"index": 0, "delta": {"content": "\n".join(raw_bits)}, "finish_reason": None}
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
            finish = "tool_calls" if calls else ("length" if len(all_ids) >= max_tokens else "stop")
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
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    out_ids = list(gen)
    out = parse_assistant_ids(tokenizer, out_ids)
    calls = _openai_tool_calls(out["tool_calls"])
    # A tool call whose JSON didn't parse has no name — surface its raw text
    # as content instead of silently dropping the model's action.
    raw_bits = [c["raw"] for c in out["tool_calls"] if not c.get("name") and c.get("raw")]
    content = "\n".join(t for t in [out["content"], *raw_bits] if t)
    message = {"role": "assistant", "content": content or (None if calls else "")}
    if calls:
        message["tool_calls"] = calls
    # honest finish_reason: a generation that spent the whole budget was cut
    # off ("length"), not naturally finished ("stop")
    finish = "tool_calls" if calls else ("length" if len(out_ids) >= max_tokens else "stop")
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(out_ids),
            "total_tokens": len(prompt_ids) + len(out_ids),
        },
    }


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "modkit"}]}


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


if __name__ == "__main__":
    print(f"Enigma OpenAI-compatible API -> http://{ARGS.host}:{ARGS.port}/v1", flush=True)
    print(f"In Odysseus:  /setup local http://{ARGS.host}:{ARGS.port}/v1", flush=True)
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning")
