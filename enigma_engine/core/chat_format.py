"""Chat & tool token format for the from-scratch Enigma — the instruct pass's
foundation. ONE canonical renderer shared by ``finetune_enigma.py`` (training)
and ``serve_enigma.py`` (serving), so the train/serve template can never
diverge — that divergence is the classic instruct-model bug class.

Token plan (the padded embedding): the base vocab is 4718
(IDs 0..4717); the embedding is padded to 4736, leaving 18 free rows that
pretraining never targets. Chat tokens live there:

    4718 <|im_start|>     4719 <|im_end|>
    4720 <|tool_call|>    4721 <|/tool_call|>
    4722 <|tool_result|>  4723 <|/tool_result|>
    4724 <|image|>        4725 <|/image|>      (vision spans; attach_image_tokens)
    4726..4735 reserved for future passes.

Those numbers are the LIVE v1 layout, kept as constants for the shipped
checkpoints. ``attach_chat_tokens`` DERIVES the base per tokenizer (first
row past the real vocab) and render/parse read ids off the instance via
``chat_token_ids`` -- a hardcoded 4718 would alias real learned tokens on
any bigger (v2) vocab (audit 2026-07-19, HIGH-2).

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

# Image-span delimiters: the NEXT two reserve rows after the chat block
# (v1 layout 4724/4725; derived as base+6/base+7 on any vocab, so the v2
# 16,366-row vocab gets 16,372/16,373). Deliberately NOT in CHAT_TOKENS:
# the chat template never renders them, and keeping them out of
# attach_chat_tokens means the v8 serving path's registered id set is
# byte-identical with or without vision. Like the chat tokens they are
# INSTANCE-attached only -- the vocab file never maps the literals, so no
# corpus text can carve into these rows (the <image>-in-HTML hazard the
# collectors sanitize for table specials cannot exist for these).
IMAGE_START = 4724
IMAGE_END = 4725
IMAGE_TOKENS = {
    "<|image|>": IMAGE_START,
    "<|/image|>": IMAGE_END,
}

CHAT_FORMAT_NAME = "enigma-chat-v1"  # stamped into SFT checkpoints' meta
ROLES = ("system", "user", "assistant", "tool")

TOOL_SYNTAX = (
    'Call a tool by writing <|tool_call|>{"name": "...", "arguments": {...}}<|/tool_call|> and wait for the result.'
)

# The tools the SERVER itself executes. They live here because the SFT data and
# the server must offer byte-identical specs: a description differing by one
# word between training and serving is a system block the model never saw.
# serve binds its own names to these; make_sft_data bakes them through
# render_tools_system, so one edit moves both.
BUILTIN_TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression and return the exact result.",
            "parameters": {"expression": "string"},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save a fact about the user to long-term memory.",
            "parameters": {"text": "string"},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": "Remove a fact about the user from long-term memory when they ask "
            "you to forget it or say it is no longer true. Pass the fact to remove as "
            "'text'. If a previous call reported several matching memories, pass the "
            "'id' of the one they meant instead.",
            "parameters": {"text": "string", "id": "integer"},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "speak",
            "description": "Speak text out loud through the computer speakers.",
            "parameters": {"text": "string"},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "imagine",
            "description": "Generate an image from a text description and save it as a file.",
            "parameters": {"prompt": "string"},
        },
    },
)

BUILTIN_NAMES = frozenset(t["function"]["name"] for t in BUILTIN_TOOLS)


def builtin_tool(name: str) -> dict:
    """The shared spec for one built-in, by name."""
    for t in BUILTIN_TOOLS:
        if t["function"]["name"] == name:
            return t
    raise KeyError(name)


def _flat_params(params) -> dict:
    """JSON-schema tool parameters -> the flat ``{name: type}`` shape ALL the
    SFT data uses (make_sft_data TOOLS). OpenAI clients send a full schema
    ({"type": "object", "properties": {...}}); the model has never seen one,
    and given one it mimics the schema shape in its arguments (measured
    2026-07-05 on the v4 checkpoint). Normalize here so the served spec is
    byte-shaped like training."""
    if isinstance(params, dict) and isinstance(params.get("properties"), dict):
        return {
            k: (v.get("type", "string") if isinstance(v, dict) else "string")
            for k, v in params["properties"].items()
        }
    return params if isinstance(params, dict) else {}


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
                    "parameters": _flat_params(fn.get("parameters", {})),
                },
                ensure_ascii=False,
            )
        )
    if not specs:
        return ""
    return "Available tools:\n" + "\n".join(specs) + "\n" + TOOL_SYNTAX


RESERVED_ROWS = PADDED_VOCAB - BASE_VOCAB  # 18: 6 chat tokens + 12 spare


def real_vocab_rows(tokenizer) -> int:
    """Number of REAL learned rows = the first free embedding row.

    Refuses to guess on a non-contiguous vocab: if ids have holes, "the
    row past the top" and "the count of rows" disagree and neither is a
    safe base for control tokens.
    """
    n = len(tokenizer.token_to_id)
    top = max(tokenizer.token_to_id.values(), default=-1)
    if top != n - 1:
        raise ValueError(f"vocab ids are not contiguous (rows={n}, max id={top}); cannot derive a chat-token base")
    return n


def chat_vocab_rows(tokenizer) -> tuple[int, int]:
    """(base, padded) embedding geometry for THIS tokenizer's vocab.

    base = first row past the real vocab (chat tokens start here);
    padded = base + the 18-row reserve. For the live v1 vocab this is
    exactly (BASE_VOCAB, PADDED_VOCAB) = (4718, 4736).
    """
    base = real_vocab_rows(tokenizer)
    return base, base + RESERVED_ROWS


def attach_chat_tokens(tokenizer):
    """Register the chat tokens on a tokenizer INSTANCE (idempotent).

    encode() then emits them as single IDs and decode() can render them.
    The BPE tables (token_to_id/merges) are deliberately untouched, so plain
    text keeps encoding byte-for-byte the same as during pretraining.

    The ids are DERIVED from the tokenizer, not hardcoded: base = the
    first row past the real vocab, which is 4718 (the constants above)
    for the live v1 vocab and moves with any bigger vocab. A hardcoded
    4718 ALIASES real learned tokens on a larger vocab -- the 2026-07-19
    audit measured id 4718 = ' crashes' on a 5,996-row test vocab, and
    the old name-keyed guard silently overwrote it (HIGH-2). If the
    vocab file already bakes the chat tokens in as real rows, those ids
    are adopted verbatim instead.
    """
    baked = [s for s in CHAT_TOKENS if s in tokenizer.token_to_id]
    if baked and len(baked) != len(CHAT_TOKENS):
        raise ValueError(
            f"vocab bakes {len(baked)}/{len(CHAT_TOKENS)} chat tokens ({baked}); need all or none"
        )
    if baked:
        ids = {s: tokenizer.token_to_id[s] for s in CHAT_TOKENS}
    else:
        base = real_vocab_rows(tokenizer)
        ids = {s: base + offset for offset, s in enumerate(CHAT_TOKENS)}
    for s, i in ids.items():
        have = tokenizer.special_tokens.get(s)
        if have is not None and have != i:
            raise ValueError(f"special token {s!r} already maps to {have}, wanted {i}")
        existing = tokenizer.id_to_token.get(i)
        if existing is not None and existing != s:
            raise ValueError(
                f"chat token {s!r} at id {i} would overwrite {existing!r} -- refusing to alias a real token"
            )
    tokenizer.special_tokens.update(ids)
    for s, i in ids.items():
        tokenizer.id_to_token[i] = s
    # Fail honestly if the tokenizer's actual <think>/</think> ids disagree
    # with what render/parse will inject: a mismatch would silently
    # train/serve reasoning spans on the wrong token. (On every vocab this
    # trainer writes, the special dict pins <think>=10/</think>=11 -- same
    # as v1 -- so the constants double as documentation.)
    for name, want in (("<think>", THINK), ("</think>", THINK_END)):
        have = tokenizer.token_to_id.get(name)
        if have is not None and have != want:
            raise ValueError(
                f"chat_format {name} id {want} != tokenizer's {have}; "
                f"update THINK/THINK_END to match this vocab before training/serving"
            )
    # The Rust fast path snapshots special_tokens at load; the tags just
    # registered change the v2 carve set, so a stale backend would split
    # tag-bearing text as plain text. Detach -- the Python path knows them.
    if getattr(tokenizer, "_rust_backend", None) is not None:
        tokenizer._rust_backend = None
    return tokenizer


def attach_image_tokens(tokenizer):
    """Register the image-span delimiters on a tokenizer INSTANCE (idempotent).

    Same contract as attach_chat_tokens: ids are DERIVED (base + 6/7, the
    rows after the chat block), a vocab that bakes the literals in as real
    rows is adopted verbatim, aliasing a real learned token refuses, and the
    BPE tables stay untouched so plain text encodes byte-for-byte as during
    pretraining. Only the vision path calls this -- a text-only boot never
    registers the ids, so its carve set is unchanged."""
    baked = [s for s in IMAGE_TOKENS if s in tokenizer.token_to_id]
    if baked and len(baked) != len(IMAGE_TOKENS):
        raise ValueError(
            f"vocab bakes {len(baked)}/{len(IMAGE_TOKENS)} image tokens ({baked}); need all or none"
        )
    if baked:
        ids = {s: tokenizer.token_to_id[s] for s in IMAGE_TOKENS}
    else:
        base = real_vocab_rows(tokenizer)
        ids = {s: base + len(CHAT_TOKENS) + offset for offset, s in enumerate(IMAGE_TOKENS)}
    for s, i in ids.items():
        have = tokenizer.special_tokens.get(s)
        if have is not None and have != i:
            raise ValueError(f"special token {s!r} already maps to {have}, wanted {i}")
        existing = tokenizer.id_to_token.get(i)
        if existing is not None and existing != s:
            raise ValueError(
                f"image token {s!r} at id {i} would overwrite {existing!r} -- refusing to alias a real token"
            )
    tokenizer.special_tokens.update(ids)
    for s, i in ids.items():
        tokenizer.id_to_token[i] = s
    # Same stale-snapshot hazard as attach_chat_tokens: the Rust fast path
    # snapshots the carve set at load and must not miss the new tags.
    if getattr(tokenizer, "_rust_backend", None) is not None:
        tokenizer._rust_backend = None
    return tokenizer


def image_token_ids(tokenizer) -> dict[str, int]:
    """The image-delimiter id map registered on THIS instance by
    attach_image_tokens; raises until it has been called, mirroring
    chat_token_ids."""
    try:
        return {s: tokenizer.special_tokens[s] for s in IMAGE_TOKENS}
    except (AttributeError, KeyError) as exc:
        raise ValueError("tokenizer has no image tokens attached; call attach_image_tokens(tokenizer) first") from exc


def chat_token_ids(tokenizer) -> dict[str, int]:
    """The chat-token id map registered on THIS instance by attach_chat_tokens.

    Render/parse read ids from here rather than the module constants, so a
    bigger (v2) vocab gets its derived rows automatically and the constants
    stay what they are: the LIVE v1 checkpoint layout.
    """
    try:
        return {s: tokenizer.special_tokens[s] for s in CHAT_TOKENS}
    except (AttributeError, KeyError) as exc:
        raise ValueError("tokenizer has no chat tokens attached; call attach_chat_tokens(tokenizer) first") from exc


def think_token_ids(tokenizer) -> tuple[int, int]:
    """(<think>, </think>) ids for THIS instance; the v1 constants are the
    fallback for tokenizers that lack the tags (attach_chat_tokens verifies
    equality whenever the vocab does carry them)."""
    t2i = getattr(tokenizer, "token_to_id", {})
    return t2i.get("<think>", THINK), t2i.get("</think>", THINK_END)


def search_token_ids(tokenizer) -> "tuple[int | None, int | None]":
    """(<search>, </search>) ids for THIS instance -- None on vocabs that do
    not carve them. DELIBERATELY no constant fallback, unlike think: the v1
    table predates the tags, and None is how the generation hook detects
    "feature absent on this model" instead of aliasing a learned id (the
    tokenizer's own None-on-legacy contract, Stage B-1)."""
    t2i = getattr(tokenizer, "token_to_id", {})
    return (
        getattr(tokenizer, "search_start_id", None) or t2i.get("<search>"),
        getattr(tokenizer, "search_end_id", None) or t2i.get("</search>"),
    )


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
    native <think>/</think> and <search>/</search> mappings
    (``allow_think=True`` means "assistant-authored") — the SFT corpus
    carries real reasoning and search spans; every other role gets both
    families neutralized, since a user-forged <search> span would otherwise
    land in context as live control ids on a v2 vocab.
    """
    if not text:
        return []
    # IMAGE_TOKENS are forbidden unconditionally: on a vision-attached
    # instance a literal "<|image|>" in user text would forge an image-span
    # boundary; on a text-only instance the split halves encode as the same
    # plain text either way.
    forbidden = list(CHAT_TOKENS) + list(IMAGE_TOKENS)
    if not allow_think:
        forbidden += ["<think>", "</think>", "<search>", "</search>"]
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
    learn (assistant content + its <|im_end|>); everything else is context.

    Chat ids come from the INSTANCE (chat_token_ids), not the module
    constants: on the live v1 vocab they are identical, on a bigger vocab
    the constants would alias real tokens (HIGH-2)."""
    ct = chat_token_ids(tokenizer)
    im_start, im_end = ct["<|im_start|>"], ct["<|im_end|>"]
    tool_call, tool_call_end = ct["<|tool_call|>"], ct["<|/tool_call|>"]
    tool_result, tool_result_end = ct["<|tool_result|>"], ct["<|/tool_result|>"]
    chunks = []
    for m in messages:
        role = m.get("role", "")
        if role not in ROLES:
            raise ValueError(f"unknown chat role {role!r} (need one of {ROLES})")
        content = (m.get("content") or "").strip()
        header = [im_start] + _enc(tokenizer, role + "\n")
        body: list[int] = _enc_content(tokenizer, content, allow_think=(role == "assistant")) if content else []
        if role == "tool":
            body = [tool_result] + body + [tool_result_end]
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
                body += [tool_call] + _enc_content(tokenizer, payload, allow_think=False) + [tool_call_end]
        tail = _enc(tokenizer, "\n")
        ids = header + body + [im_end] + tail
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

    im_start = chat_token_ids(tokenizer)["<|im_start|>"]
    gen_ids = [im_start] + _enc(tokenizer, "assistant\n") if add_generation_prompt else []
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
    {content, tool_calls, thinking, search}. Parsing is ID-level (immune to
    text collisions); generation should stop at <|im_end|>/EOS, but a
    trailing one is tolerated and stripped.

    ``search`` is the first CLOSED <search>...</search> span's query, or
    None. Closed is the contract: a span still open when generation ends is
    a truncated intent, and executing a half-written query would act on
    something the model never finished saying -- the dangling text joins
    content instead, mirroring how a truncated tool call surfaces as raw
    rather than executing. Later closed spans also join content (one lookup
    per hop; the loop regenerates, so a second question gets its own turn).
    On vocabs without the tags (v1) the ids are None and no span can open.
    """
    eos = getattr(tokenizer, "eos_token_id", 2)
    ct = chat_token_ids(tokenizer)
    im_end = ct["<|im_end|>"]
    tool_call, tool_call_end = ct["<|tool_call|>"], ct["<|/tool_call|>"]
    think, think_end = think_token_ids(tokenizer)
    search_start, search_end = search_token_ids(tokenizer)
    content_ids: list[int] = []
    think_ids: list[int] = []
    tool_calls: list[dict[str, Any]] = []
    search_query: str | None = None
    span: list[int] | None = None
    span_kind = ""

    def flush_span(closed: bool = True):
        nonlocal search_query
        if span is None:
            return
        if span_kind == "think":
            think_ids.extend(span)
        elif span_kind == "search":
            raw = tokenizer.decode(span, skip_special_tokens=True).strip()
            if closed and search_query is None and raw:
                search_query = raw
            else:
                content_ids.extend(span)
        else:
            raw = tokenizer.decode(span, skip_special_tokens=True).strip()
            try:
                call = json.loads(raw)
                name = call.get("name")
                # isinstance: a truthy NON-string name ({"name": {"a": 1}})
                # must go to the raw branch too -- downstream does
                # `name in _BUILTIN_NAMES`, which throws on unhashables.
                if isinstance(name, str) and name:
                    tool_calls.append({"name": name, "arguments": call.get("arguments", {})})
                else:
                    # Valid JSON but no usable "name" (e.g. {"tool": ...}):
                    # keep the raw text so callers can surface the action --
                    # a name-less parse without "raw" is invisible to every
                    # downstream filter and the call would vanish silently.
                    tool_calls.append({"name": None, "raw": raw})
            except (json.JSONDecodeError, AttributeError):
                tool_calls.append({"name": None, "raw": raw})

    for t in ids:
        if t in (im_end, eos):
            break
        if t == think:
            span, span_kind = [], "think"
        elif t == tool_call:
            span, span_kind = [], "tool"
        elif search_start is not None and t == search_start:
            span, span_kind = [], "search"
        elif t in (think_end, tool_call_end) or (search_end is not None and t == search_end):
            flush_span()
            span = None
        elif span is not None:
            span.append(t)
        else:
            content_ids.append(t)
    # Generation can end mid-span (max_tokens budget): flush the dangling span
    # instead of silently discarding the model's output — a truncated think
    # becomes thinking, a truncated tool call surfaces as a raw call, and a
    # truncated search query becomes content (never an executed lookup).
    flush_span(closed=False)
    return {
        "content": tokenizer.decode(content_ids, skip_special_tokens=True).strip(),
        "tool_calls": tool_calls,
        "thinking": (tokenizer.decode(think_ids, skip_special_tokens=True).strip() or None),
        "search": search_query,
    }
