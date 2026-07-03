# How the AI Works

Enigma Engine is a **fully local** AI system. Everything runs on your
computer — no cloud, no internet required for inference.

---

## Model Formats

The engine supports multiple model formats and loads them through
specialized loaders:

| Format | Extension | Loader | Description |
|--------|-----------|--------|-------------|
| GGUF | .gguf | gguf_loader.py | Quantized models via llama-cpp-python. Most common, smallest files |
| PyTorch | .pth/.pt | (native) | Native Enigma checkpoints saved with `atomic_torch_save()` |
| SafeTensors | .safetensors | huggingface_loader.py | HuggingFace safe-serialized tensors |
| HuggingFace | folder | huggingface_loader.py | Full HuggingFace model directories (auto_model) |
| GPTQ/AWQ | folder | gptq_awq_loader.py | Quantized HuggingFace models via auto-gptq or autoawq |
| ONNX | .onnx | onnx_loader.py | ONNX Runtime models |
| Ollama | manifest | ollama_loader.py | Reads Ollama's local cache (~/.ollama/models/) |

**Ollama integration:** The Ollama loader discovers locally-installed Ollama
models by parsing manifest files under `manifests/registry.ollama.ai/library/`.
Each manifest points to blob files containing GGUF weights, chat templates, and
system prompts. These models appear in the MODELS page alongside native ones.

---

## Transformer Architecture

### Core Stack

Native Enigma models use this transformer architecture:

1. **Token Embedding** — converts text tokens into vectors
2. **Rotary Position Encoding (RoPE)** — encodes token positions
3. **Transformer Blocks** — self-attention + feed-forward layers
4. **RMSNorm** — normalization before each sub-layer
5. **SwiGLU Activation** — gated feed-forward network
6. **Grouped Query Attention (GQA)** — efficient multi-head attention
7. **KV-Cache** — speeds up generation by caching key/value pairs

### Mixture of Experts (MoE)

Models can optionally use a Mixture of Experts feed-forward layer
(`MoEFeedForward` in model_components.py) instead of a standard FFN.

How it works:
- A **router/gate** network assigns each token to the top-K experts
- Only the selected experts process that token (sparse activation)
- Outputs are combined as a weighted sum using gating scores
- A **load balancing loss** during training prevents all tokens from
  routing to the same expert (`moe_load_balancing` config key)

This lets models have many more parameters without proportional compute
increase. For example, a 30B parameter MoE model might only activate
3B parameters per token.

### Generation

When you send a message, the engine:

1. Tokenizes your text into integer tokens
2. Runs a forward pass through the transformer
3. Samples the next token from the output distribution
4. Repeats until a stop token or max length is reached
5. Decodes the tokens back into text

For GGUF models, steps 2-4 happen inside llama-cpp-python (or the
bundled llama-server binary for unsupported GPU architectures like
Blackwell).

### Sampling Parameters

| Parameter | What It Does |
|-----------|-------------|
| Temperature | Higher = more creative, lower = more focused. 0 = greedy/argmax |
| Top-P | Nucleus sampling — only consider tokens above this cumulative probability |
| Top-K | Only consider the top K most likely tokens |
| Repetition Penalty | Penalizes repeating tokens. Applied as `score / penalty` for positive logits, `score * penalty` for negative |
| Max Tokens | Maximum number of tokens to generate |

These are adjustable on the CONFIG page in the GUI.

---

## Reasoning (Chain-of-Thought)

The engine supports chain-of-thought reasoning using `<think>...</think>`
tags (reasoning.py).

When a model reasons, its output looks like:

```
<think>
15 * 23 = 15 * 20 + 15 * 3 = 300 + 45 = 345
</think>
The answer is 345.
```

The GUI displays the thinking section in a collapsible block — the user
sees the final answer prominently, with the option to expand and read
the reasoning. Think tags are processed regardless of whether the
reasoning toggle is enabled, because many models generate them natively.

If generation is interrupted mid-thought, `strip_incomplete_think()`
cleans up any unclosed `<think>` block so the output stays readable.

---

## Vision Encoder

The engine includes a Vision Transformer (ViT) built entirely from
scratch in vision_encoder.py — no pretrained weights, no downloads.

| Preset | Layers | Dimension | Params |
|--------|--------|-----------|--------|
| tiny | 2 | 128 | ~500K |
| small | 4 | 256 | ~4M |
| medium | 6 | 512 | ~25M |

The vision encoder converts images into patch embeddings that get
projected into the text model's hidden dimension. It trains from
scratch alongside the text model during Image Training mode in FORGE.

Architecture: Image → Patch Embedding → Position Embeddings → N Transformer Blocks → RMSNorm → features

---

## Persistent Memory

The AI has long-term memory across conversations via PersistentMemory
(memory.py). Facts are stored in a human-readable file at
`data/notes/memory.md`.

Facts are added two ways:

1. **Automatic extraction** — pattern matching on each user message
   catches phrases like "my name is Alex", "I work at NASA",
   "I prefer dark themes", etc. Works with any model.
2. **AI command** — `[CMD]memory.remember <fact>[/CMD]` lets capable
   models voluntarily save things they judge important.

The memory file is injected into the system prompt so the AI always
has context about the user. Capped at 200 facts (MAX_FACTS) — oldest
are trimmed first. The user can hand-edit `data/notes/memory.md` at
any time for full transparency.

Related commands: `memory.remember`, `memory.forget`, `memory.notes`,
`memory.clear_notes` (see commands_reference.md).

---

## RAG (Retrieval-Augmented Generation)

The RAG pipeline (rag.py) enables document-grounded Q&A using
TF-IDF vectors with cosine similarity — no external vector store
or sentence-transformer required (runs on numpy alone).

Flow:
1. **Index** — Chunk documents into ~512 character overlapping segments,
   compute TF-IDF vectors, store in memory (optionally persist to
   `data/rag_index.json`)
2. **Query** — Vectorize the user's question, find top-K chunks by
   cosine similarity (default K=5)
3. **Inject** — Prepend the retrieved chunks to the AI's system prompt
   so the model has relevant context

The index is rebuilt on demand and cached. Vocabulary is capped at
16000 terms to keep memory low.

---

## Web Access

The engine can search the web and fetch pages via web_utils.py.

- **`ddg_search(query)`** — searches DuckDuckGo's HTML endpoint and
  parses results (title, URL, snippet)
- **`fetch_page_text(url)`** — fetches a URL and extracts readable text,
  stripping scripts/styles/nav/footer (max 3000 characters)
- **`extract_html_text(html)`** — HTML-to-text parser that filters out
  non-content elements

These functions are used by:
- The `search.web` and `web.fetch` engine commands
- FORGE Web Learn (fetches pages, generates training data from content)
- The web toggle in the CORE chat page

No API keys required — all searches go through DuckDuckGo's public
HTML endpoint.

---

## Hardware Detection

The engine auto-detects your hardware (hardware_detection.py):

- **GPU** — CUDA GPUs get automatic layer offloading
- **VRAM** — determines how many layers fit on the GPU
- **RAM** — fallback for CPU-only inference
- **Context size** — 1024 for models with max_seq_len ≥ 1024, 512 otherwise

For GGUF models, GPU layers are set automatically with
`n_gpu_layers=-1` (all layers on GPU). The engine also adds PyTorch
CUDA DLLs to `os.environ['PATH']` before importing llama-cpp to
ensure correct library loading on Windows.

---

## Chat Templates

Models use chat templates to format conversations:

```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
Hello!
<|im_end|>
<|im_start|>assistant
```

The engine auto-detects the correct template for each model format.
GGUF models use their built-in Jinja templates. HuggingFace models
use `tokenizer.apply_chat_template()`. Native models use ChatML.
