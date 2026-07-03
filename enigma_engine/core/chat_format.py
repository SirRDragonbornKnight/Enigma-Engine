"""Chat & tool token format for the from-scratch Enigma — the instruct pass's
foundation. ONE canonical renderer shared by ``finetune_enigma.py`` (training)
and ``serve_enigma.py`` (serving), so the train/serve template can never
diverge — that divergence is the classic instruct-model bug class.

Token plan (KNOWN_ISSUES #8 / the padded embedding): the base vocab is 4718
(IDs 0..4717); the embedding is padded to 4736, leaving 18 free rows that
pretraining never targets. Chat tokens live there:

    4718 <|im_start|>     4719 <|im_end|>
    4720 <|tool_call|>    4721 <|/tool_call|>
    4722 <|tool_result|>  4723 <|/tool_result|>
    4724..4735 reserved for future passes.

Thinking spans reuse the tokenizer's NATIVE ``<think>``=10 / ``</think>``=11
(the IDs the trained ``bpe_vocab.json`` actually assigns) — ``encode()`` already
parses them and the SFT corpus preserves the tags verbatim, so reasoning traces
cost zero new rows. ``attach_chat_tokens`` asserts these constants match the
tokenizer so they can never silently drift (they were 4/5 here once, which
collided with the vocab's ``<sep>``/``<mask>``).

Template (ChatML-shaped; role names are plain text, so no per-role tokens):

    <s><|im_start|>system\\n{...}<|im_end|>\\n
    <|im_start|>user\\n{...}<|im_end|>\\n
    <|im_start|>assistant\\n{...}<|im_end|>\\n ... </s>

Assistant turns may embed ``<|tool_call|>{json}<|/tool_call|>``; tool results
come back as role ``tool``, wrapped by the renderer in
``<|tool_result|>...<|/tool_result|>``.

Nothing here touches the vocab file or the BPE tables: ``attach_chat_tokens``
registers the strings on the tokenizer INSTANCE only (encode's special-token
splitter re.escapes, so the ``|`` characters are safe).
"""

from __future__ import annotations

import json
from typing import Any

BASE_VOCAB = 4718  # real BPE vocab (IDs 0..4717)
PADDED_VOCAB = 4736  # embedding rows in the live checkpoints

IM_START = 4718
IM_END = 4719
TOOL_CALL = 4720
TOOL_CALL_END = 4721
TOOL_RESULT = 4722
TOOL_RESULT_END = 4723
THINK = 10  # native tokenizer IDs (bpe_vocab.json: <think>=10, </think>=11)
THINK_END = 11

CHAT_TOKENS = {
    "<|im_start|>": IM_START,
    "<|im_end|>": IM_END,
    "<|tool_call|>": TOOL_CALL,
    "<|/tool_call|>": TOOL_CALL_END,
    "<|tool_result|>": TOOL_RESULT,
    "<|/tool_result|>": TOOL_RESULT_END,
}

CHAT_FORMAT_NAME = "enigma-chat-v1"  # stamped into SFT checkpoints' meta
ROLES = ("system", "user", "assistant", "tool")

TOOL_SYNTAX = (
    'Call a tool by writing <|tool_call|>{"name": "...", "arguments": {...}}<|/tool_call|> and wait for the result.'
)


def render_tools_system(tools) -> str:
    """OpenAI-style tool specs -> the system-prompt suffix that teaches/reminds
    the call syntax. The SAME text the SFT data uses (make_sft_data.py), so
    serving never drifts from training."""
    specs = []
    for t in tools or []:
        fn = t.get("function", t)  # accept OpenAI nesting or flat specs
        specs.append(
            json.dumps(
                {
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                },
                ensure_ascii=False,
            )
        )
    if not specs:
        return ""
    return "Available tools:\n" + "\n".join(specs) + "\n" + TOOL_SYNTAX


def attach_chat_tokens(tokenizer):
    """Register the chat tokens on a tokenizer INSTANCE (idempotent).

    encode() then emits them as single IDs and decode() can render them.
    The BPE tables (token_to_id/merges) are deliberately untouched, so plain
    text keeps encoding byte-for-byte the same as during pretraining.
    """
    for s, i in CHAT_TOKENS.items():
        have = tokenizer.special_tokens.get(s)
        if have is not None and have != i:
            raise ValueError(f"special token {s!r} already maps to {have}, wanted {i}")
    tokenizer.special_tokens.update(CHAT_TOKENS)
    for s, i in CHAT_TOKENS.items():
        tokenizer.id_to_token[i] = s
    # Fail honestly if the think constants disagree with the tokenizer's actual
    # <think>/</think> ids: render/parse inject THINK/THINK_END as raw ids, so a
    # mismatch would silently train/serve reasoning spans on the wrong token.
    for name, want in (("<think>", THINK), ("</think>", THINK_END)):
        have = tokenizer.token_to_id.get(name)
        if have is not None and have != want:
            raise ValueError(
                f"chat_format {name} id {want} != tokenizer's {have}; "
                f"update THINK/THINK_END to match this vocab before training/serving"
            )
    return tokenizer


def _enc(tokenizer, text: str) -> list[int]:
    """Encode a segment WITHOUT the BOS/EOS bracketing."""
    if not text:
        return []
    return tokenizer.encode(text, add_special_tokens=False)


def _enc_content(tokenizer, text: str, allow_think: bool) -> list[int]:
    """Encode message CONTENT so control markers can never be forged.

    tokenizer.encode maps any literal special-token substring to its control
    ID — so user text containing "<|im_end|>" would close the turn and text
    containing "<|tool_result|>" would forge a tool result. Here any forbidden
    marker is split at its second character and encoded as two plain-text
    pieces (neither half matches the special-token regex); decode() still
    round-trips the exact original characters. Assistant content keeps the
    native <think>/</think> mapping (``allow_think=True``) — the SFT corpus
    carries real reasoning spans; every other role gets those neutralized too.
    """
    if not text:
        return []
    forbidden = list(CHAT_TOKENS)
    if not allow_think:
        forbidden += ["<think>", "</think>"]
    out: list[int] = []
    rest = text
    while rest:
        hits = [(i, s) for s in forbidden for i in (rest.find(s),) if i != -1]
        if not hits:
            out += tokenizer.encode(rest, add_special_tokens=False)
            break
        i, s = min(hits)
        if i:
            out += tokenizer.encode(rest[:i], add_special_tokens=False)
        out += tokenizer.encode(s[:2], add_special_tokens=False)
        out += tokenizer.encode(s[2:], add_special_tokens=False)
        rest = rest[i + len(s) :]
    return out


def _message_chunks(tokenizer, messages: list[dict[str, Any]]):
    """Render each message to (role, ids, trainable_mask) — the single source
    of template truth. trainable_mask marks the positions an SFT pass should
    learn (assistant content + its <|im_end|>); everything else is context."""
    chunks = []
    for m in messages:
        role = m.get("role", "")
        if role not in ROLES:
            raise ValueError(f"unknown chat role {role!r} (need one of {ROLES})")
        content = (m.get("content") or "").strip()
        header = [IM_START] + _enc(tokenizer, role + "\n")
        body: list[int] = _enc_content(tokenizer, content, allow_think=(role == "assistant")) if content else []
        if role == "tool":
            body = [TOOL_RESULT] + body + [TOOL_RESULT_END]
        if role == "assistant":
            for call in m.get("tool_calls") or []:
                fn = call.get("function", call)  # accept OpenAI nesting or flat
                fn_args = fn.get("arguments", {})
                if isinstance(fn_args, str):
                    # OpenAI-spec clients echo arguments back as a JSON STRING;
                    # re-parse so history renders as the OBJECT form the model
                    # was trained on (make_sft_data emits dict arguments).
                    try:
                        fn_args = json.loads(fn_args)
                    except json.JSONDecodeError:
                        pass  # unparsable client input: render as-is
                payload = json.dumps({"name": fn.get("name"), "arguments": fn_args}, ensure_ascii=False)
                body += [TOOL_CALL] + _enc_content(tokenizer, payload, allow_think=False) + [TOOL_CALL_END]
        tail = _enc(tokenizer, "\n")
        ids = header + body + [IM_END] + tail
        mask = [False] * len(header) + [True] * (len(body) + 1) + [False] * len(tail)
        if role != "assistant":
            mask = [False] * len(ids)
        chunks.append((role, ids, mask))
    return chunks


def render_chat(
    tokenizer, messages: list[dict[str, Any]], add_generation_prompt: bool = True, max_ids: int | None = None
) -> list[int]:
    """Messages -> token IDs, ready for the model. BOS-prefixed, NO trailing
    EOS (she continues from here). With ``max_ids``, the oldest non-system
    turns are dropped first; the system message and the newest turn survive,
    and as a last resort the newest turn's own ids are left-truncated."""
    ids, _ = render_training(
        tokenizer, messages, add_generation_prompt=add_generation_prompt, add_eos=False, max_ids=max_ids
    )
    return ids


def render_training(
    tokenizer,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool = False,
    add_eos: bool = True,
    max_ids: int | None = None,
) -> tuple[list[int], list[bool]]:
    """The full renderer: returns (ids, trainable_mask), both BOS-prefixed.

    Training: ``add_eos=True`` appends the document EOS after a final
    assistant turn (trainable — she must learn to stop). Serving:
    ``add_generation_prompt=True`` appends ``<|im_start|>assistant\\n``."""
    bos = getattr(tokenizer, "bos_token_id", 1)
    eos = getattr(tokenizer, "eos_token_id", 2)
    chunks = _message_chunks(tokenizer, messages)

    gen_ids = [IM_START] + _enc(tokenizer, "assistant\n") if add_generation_prompt else []
    fixed = 1 + len(gen_ids) + (1 if add_eos else 0)

    if max_ids is not None and chunks:
        sys_chunk = chunks[0] if chunks[0][0] == "system" else None
        rest = chunks[1:] if sys_chunk else list(chunks)

        def total(parts):
            return fixed + sum(len(c[1]) for c in parts)

        parts = ([sys_chunk] if sys_chunk else []) + rest
        while len(rest) > 1 and total(parts) > max_ids:
            rest.pop(0)  # drop the oldest non-system turn
            parts = ([sys_chunk] if sys_chunk else []) + rest
        if total(parts) > max_ids and rest:
            role, ids_c, mask_c = rest[-1]
            keep = max(8, len(ids_c) - (total(parts) - max_ids))
            rest[-1] = (role, ids_c[-keep:], mask_c[-keep:])
            parts = ([sys_chunk] if sys_chunk else []) + rest
        chunks = parts

    ids: list[int] = [bos]
    mask: list[bool] = [False]
    for _, c_ids, c_mask in chunks:
        ids += c_ids
        mask += c_mask
    if add_eos and messages and messages[-1].get("role") == "assistant":
        ids.append(eos)
        mask.append(True)
    if gen_ids:
        ids += gen_ids
        mask += [False] * len(gen_ids)
    if max_ids is not None and len(ids) > max_ids:
        # Last resort: an oversized SYSTEM chunk alone can exceed the budget
        # (e.g. unbounded client tool specs) — the turn-dropping above never
        # shrinks it. The budget is a HARD promise: past it the ids walk off
        # the trained block / RoPE table. Keep BOS + the tail (newest turn
        # and any generation prompt live there).
        keep = max(1, max_ids - 1)
        ids = ids[:1] + ids[-keep:]
        mask = mask[:1] + mask[-keep:]
    return ids, mask


def parse_assistant_ids(tokenizer, ids: list[int]) -> dict[str, Any]:
    """Decode one assistant turn's generated IDs into
    {content, tool_calls, thinking}. Parsing is ID-level (immune to text
    collisions); generation should stop at <|im_end|>/EOS, but a trailing
    one is tolerated and stripped."""
    eos = getattr(tokenizer, "eos_token_id", 2)
    content_ids: list[int] = []
    think_ids: list[int] = []
    tool_calls: list[dict[str, Any]] = []
    span: list[int] | None = None
    span_kind = ""

    def flush_span():
        if span is None:
            return
        if span_kind == "think":
            think_ids.extend(span)
        else:
            raw = tokenizer.decode(span, skip_special_tokens=True).strip()
            try:
                call = json.loads(raw)
                tool_calls.append({"name": call.get("name"), "arguments": call.get("arguments", {})})
            except (json.JSONDecodeError, AttributeError):
                tool_calls.append({"name": None, "raw": raw})

    for t in ids:
        if t in (IM_END, eos):
            break
        if t == THINK:
            span, span_kind = [], "think"
        elif t == TOOL_CALL:
            span, span_kind = [], "tool"
        elif t in (THINK_END, TOOL_CALL_END):
            flush_span()
            span = None
        elif span is not None:
            span.append(t)
        else:
            content_ids.append(t)
    # Generation can end mid-span (max_tokens budget): flush the dangling span
    # instead of silently discarding the model's output — a truncated think
    # becomes thinking, a truncated tool call surfaces as a raw call.
    flush_span()
    return {
        "content": tokenizer.decode(content_ids, skip_special_tokens=True).strip(),
        "tool_calls": tool_calls,
        "thinking": (tokenizer.decode(think_ids, skip_special_tokens=True).strip() or None),
    }
