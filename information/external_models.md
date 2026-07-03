# External Models — What Works and What Doesn't

This guide covers using external models (HuggingFace, GGUF, GPTQ/AWQ, ONNX, Ollama)
with Enigma Engine and what features may not work compared to native Enigma models.

---

## Supported External Formats

| Format | Loader | Chat Templates | Training |
|--------|--------|----------------|----------|
| GGUF (llama.cpp) | `gguf_loader.py` | Native via llama.cpp | No |
| HuggingFace (safetensors/bin) | `huggingface_loader.py` | Via tokenizer config | Limited |
| GPTQ quantized | `gptq_awq_loader.py` | Via tokenizer config | No |
| AWQ quantized | `gptq_awq_loader.py` | Via tokenizer config | No |
| ONNX | `onnx_loader.py` | Manual | No |
| Ollama | `ollama_loader.py` | Native via Ollama API | No |

---

## What Works on All Models

- **Chat** — Basic conversation works on all formats
- **System Prompt** — Applied via the CORE page prompt editor
- **Generation Config** — Temperature, top_k, top_p, max tokens
- **History** — Conversation history maintained in memory
- **Auto-save** — Sessions auto-saved regardless of model type
- **Per-model Context** — History and prompt saved per model path
- **Router Assignment** — Any model can be assigned to any route
- **Display Names** — model_info.json works with any format

---

## What Does NOT Work on External Models

### Training (FORGE page)
- **SFT Training** — Only works on native Enigma (PyTorch) models
- **LoRA/QLoRA** — Requires PyTorch model with trainable parameters
- **Evolutionary Training** — Requires multiple PyTorch model instances
- **Tokenizer Training** — BPE tokenizer training is format-independent

> GGUF, GPTQ, AWQ, and ONNX models cannot be fine-tuned through the Forge.
> To train, create a native model in MODELS or use a HuggingFace model.

### KV-Cache Management
- **Clear KV-Cache** — Works on native and GGUF, not guaranteed on ONNX
- **Context Window** — GGUF respects n_ctx; HuggingFace uses max_position_embeddings

### Tool Routing
- **Specialized routing** — Works but each route needs its own engine instance
  unless the same model is assigned (then the chat engine is shared)
- **Mod integration** — Mods communicate via TCP, model format doesn't matter

### Parameter Counting
- **Param count on load** — Accurate for PyTorch models; estimated for GGUF
- **Model size display** — File size shown in MB for all formats

---

## Recommendations

1. **For chat only** — GGUF models are fastest and most memory-efficient
2. **For training** — Use native Enigma models or HuggingFace safetensors
3. **For routing** — Assign the same model to multiple routes to share memory
4. **For mods** — Any format works since mods use TCP messaging

---

## GUI Indicators

The header status indicator (top right) shows model state:
- **Gray dot + "NO MODEL"** — No model loaded
- **Orange dot + "LOADING..."** — Model is loading (may take time for large models)
- **Green dot + "MODEL // DEVICE"** — Model ready
- **Red dot + "LOAD FAILED"** — Loading failed (check chat for error details)

If the indicator stays orange for a long time, the model is either very large
or the load thread encountered an issue. Check the CORE chat for error messages.
