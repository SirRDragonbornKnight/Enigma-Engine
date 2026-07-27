# How the AI Works

Enigma is a **fully local, from-scratch** LLM: a 182M-parameter
decoder-only transformer with its own architecture, its own BPE
tokenizer, and its own weights. No cloud, no wrapper around someone
else's model. Pipeline: pretrain -> SFT -> DPO -> serve.

---

## Tokenizer

Enigma uses her own BPE tokenizer (`enigma_engine/core/bpe_tokenizer.py`
family, vocab in `enigma_engine/vocab_model/`):

- **Base vocab 4718** (ids 0..4717) -- trained on the pretraining corpus.
- `<think>` = id 10 and `</think>` = id 11 are **native** tokenizer ids.
- Six chat special tokens live in the padded embedding above the base
  vocab (`enigma_engine/core/chat_format.py`):

| Id | Token |
|----|-------|
| 4718 | `<\|im_start\|>` |
| 4719 | `<\|im_end\|>` |
| 4720 | `<\|tool_call\|>` |
| 4721 | `<\|/tool_call\|>` |
| 4722 | `<\|tool_result\|>` |
| 4723 | `<\|/tool_result\|>` |

---

## The Model

A decoder-only transformer (`enigma_engine/core/model.py`):

1. **Token embedding** -- ids become vectors
2. **Rotary position encoding (RoPE)** -- encodes token positions
3. **Transformer blocks** -- self-attention + feed-forward
4. **RMSNorm** -- normalization before each sub-layer
5. **SwiGLU** -- gated feed-forward activation
6. **Grouped Query Attention (GQA)** -- fewer KV heads than query heads
7. **KV-cache** -- caches key/value pairs so generation is incremental

### Generation

For each request the server tokenizes the rendered conversation, runs
forward passes, and samples one token at a time until a stop token or
the token limit. Sampling parameters (`temperature`, `top_p`, `top_k`,
`max_tokens`, ...) come from the API request, per the OpenAI schema.

---

## Chat Format (shared by train and serve)

`enigma_engine/core/chat_format.py` is the single renderer used by SFT
(`render_training`), DPO, and the server. Conversations render as:

```
<|im_start|>system ... <|im_end|>
<|im_start|>user ... <|im_end|>
<|im_start|>assistant ...
```

Training and serving share the exact same token stream, so what she
learns is byte-for-byte what she serves.

### Tool-call round trip

1. The server puts available tool specs in her system context.
2. She emits `<|tool_call|>{json}<|/tool_call|>` mid-turn.
3. The server parses at the **token-id level** (immune to text quirks).
   Built-in tools run in-process; client-supplied tools are returned to
   the caller as OpenAI `tool_calls`.
4. Results come back wrapped in `<|tool_result|>...<|/tool_result|>`
   and she writes her final answer.

Built-in tools (intent-gated -- each is only offered when the message
looks like it needs it, so tool prompts do not degrade normal chat):

| Tool | Needs | What it does |
|------|-------|-------------|
| `calculate` | (always available for arithmetic-looking asks) | Evaluate arithmetic |
| `remember` | `--memory-dir` | Save a fact to the memory store |
| `forget` | `--memory-dir` | Delete a stored fact on request (exactly one match, or she asks which) |
| `speak` | `--voice` | Say text through the speakers |
| `imagine` | `--image-gen` | Generate a PNG (saved to `~/.enigma_engine/images/`) |

---

## Reasoning (Chain-of-Thought)

`<think>...</think>` spans (native ids 10/11) mark her reasoning.
The SFT corpus may include them in assistant turns; at serve time the
parser extracts thinking separately from the visible answer, and a
generation cut off mid-think is surfaced honestly rather than dropped.

---

## Memory

With `--memory-dir`, the server keeps a JSONL memory store with **BM25**
retrieval (`enigma_engine/core/memory_store.py`). On each request,
memories relevant to the user's message are injected into her system
context. Facts get in two ways:

1. **She saves them** -- the `remember` tool (the ChatGPT bio-tool pattern).
2. **The API** -- `POST` / `GET` / `DELETE` on `/v1/memory`.

Facts leave two ways as well: the `forget` tool from chat ("forget that I
like tea" -- she deletes only when exactly one memory matches, and asks
which one otherwise), or the API.

---

## Organs

Capabilities are local services switched on by serve flags -- not
separate model formats, not plugins:

| Organ | Flag | Backend | Endpoints / tools |
|-------|------|---------|-------------------|
| Voice | `--voice` | Kokoro-82M | `speak` tool, `/v1/audio/speech`, `/v1/audio/voices`, `/v1/audio/stop`, `/v1/audio/talk-mode`, `/v1/audio/status`, `/v1/audio/voice` |
| Ears | `--ears` | faster-whisper | `/v1/audio/transcriptions` |
| Eyes | `--eyes` | her own distilled ViT | OpenAI-style image messages captioned into her context, `/v1/images/describe` |
| Imagination | `--image-gen` | Stable Diffusion (sd-turbo) | `imagine` tool, `/v1/images/generations` |

---

## Model Formats

The engine loads **only its own `.pth` checkpoints** (weights + config,
saved atomically). There are no import loaders: `Enigma.from_huggingface`,
`from_gguf`, and `from_onnx` raise `NotImplementedError` -- honestly.

One-way **GGUF export** exists (`enigma_engine/core/gguf.py`,
`export_to_gguf`), but serving through llama.cpp was REJECTED by ruling
(2026-07-24, reconfirmed 2026-07-27): she serves on her own engine, and
the vendored llama-server binary was deleted. The export remains a data
escape hatch only -- see `external_models.md` for the ruling's receipts.
