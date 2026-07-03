# Plan — Shrink Enigma's footprint (4-bit, load-on-demand)

**Goal:** keep the full, *generous* 8B Enigma brain, but stop it from squatting on
~16–18 GB of VRAM the whole time it runs. Target: **~5 GB while actively chatting,
~0 GB when idle.** That frees the 5090 for Odysseus, the avatar later, and everything else.

**Status (2026-06-01): DONE via Fallback A (bitsandbytes).** Enigma serves 4-bit by
default now — `python serve_enigma.py` → **~9 GB VRAM** (was ~16 GB), voice intact, 21 GB
free. Two gotchas baked into `serve_enigma.py`: (1) bnb 4-bit needs `device_map="auto"` —
the plain string `"cuda"` silently loads full bf16; (2) call `torch.cuda.empty_cache()`
after load, or the fp16 quantization transient stays *reserved* (~16 GB on `nvidia-smi`
despite a 6 GB model). `--quant bf16` still gives the full-precision max-quality load.

**The GGUF → Ollama route was attempted and ABANDONED (for now).** Full chain worked
*mechanically*: clone llama.cpp → `convert_hf_to_gguf.py` (Qwen3 supported) → f16 GGUF →
`ollama create enigma --quantize q4_K_M` → **5.0 GB model, idle-unload, 6.4 GB at 8192 ctx**.
But the served model is **incoherent via the chat APIs**: `/v1` and `/api/chat` produce
looping word-salad that never emits a stop token (1000+ tokens), at every repeat_penalty
0.0–1.3. Meanwhile raw `/api/generate` (no template) *and* the bnb/transformers version are
coherent — so the **weights are fine; the GGUF chat-serving path is broken for this model**
(special-token / eos-stop mishandling, almost certainly aggravated by the fragile 50-example
fine-tune). Not worth more tuning now. Notes for a future retry:
- Ollama 0.24.0's *safetensors* converter rejects `Qwen3ForCausalLM`; you MUST go via a GGUF
  (llama.cpp clone — master's `convert_hf_to_gguf.py` is now a `conversion/` package, not a
  single file, so clone the repo, don't download one script).
- Before re-quantizing, fix the stop/eos: verify `tokenizer.ggml.eos_token_id` in the GGUF is
  `<|im_end|>` (151645), and that special tokens survive `ollama create`. Test with the
  **native `/api/chat`**, not just `/api/generate --raw`.
- Best revisited **after the model is trained on more data** (the current fine-tune is
  fragile enough that quantization tips it into degeneracy). Until then, bnb @ 9 GB is the
  serving path.

---

## The tradeoff (pick a quant level before you start)

| Build | VRAM (active) | Quality | When |
|---|---|---|---|
| 8B bf16 (`serve_enigma.py` today) | ~16–18 GB | 100% | max quality, always loaded |
| **8B Q4_K_M** ← recommended | **~5–6 GB** | ~95%+, barely noticeable in chat | the "generous but light" pick |
| 8B Q5_K_M | ~6.5 GB | ~97% | a touch more quality |
| 8B Q6_K | ~7.5 GB | ~99% | near-lossless, still half the bf16 cost |

A *smaller base model* (4B) would be ~8–9 GB and noticeably dumber — **quantizing the
8B beats it on every axis** (more brain per GB). So we quantize, not downsize.

**Decision to make:** `q4_K_M` (default) vs `q5_K_M`. Everything below uses `q4_K_M`;
swap the string if you want Q5.

---

## Why Ollama (not just quantizing in serve_enigma.py)

Ollama is the key to the "~0 when idle" win: it **loads the model only when a request
comes in and unloads it after an idle timeout** (default 5 min), freeing the VRAM.
A Python server (bitsandbytes) keeps the model resident the whole time it's up.
You already run Ollama, and **Odysseus speaks to Ollama natively**. So:

- **Primary plan → Ollama 4-bit** (load-on-demand, idle ≈ 0).
- **Fallback A → bitsandbytes 4-bit in `serve_enigma.py`** (one Python server, always-loaded ~5 GB).
- **Fallback B → GGUF by hand via bundled llama.cpp** (only if Ollama's converter chokes).

---

## PRIMARY PLAN — Enigma as a 4-bit Ollama model

Run everything from the repo root: `C:\Users\SirKn\Enigma Engine`

### Step 0 — Free the VRAM
Stop the bf16 server if it's running (Ctrl-C in its window, or kill the python proc).
Confirm the card is mostly free:
```powershell
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

### Step 1 — Write a Modelfile
Create `Modelfile.enigma` in the repo root:
```
FROM ./models/enigma-8b
```
That's enough — Ollama pulls the chat template + tokenizer from the model dir.
*(Optional, to harden the identity against "are you ChatGPT?": add a SYSTEM line, e.g.*
`SYSTEM You are Enigma, a private local AI running on the user's own machine.`*)*

### Step 2 — Import + quantize in one shot
```powershell
ollama create enigma --quantize q4_K_M -f Modelfile.enigma
```
Ollama converts the bf16 safetensors → GGUF and quantizes to Q4_K_M. Takes a few minutes.
- **If it errors that the source must be f16:** do it in two steps —
  ```powershell
  ollama create enigma-f16 -f Modelfile.enigma
  ollama create enigma --quantize q4_K_M -f Modelfile.enigma   # now converts via the f16 it built
  ```
- **If it errors that the architecture is unsupported:** Ollama can't convert Qwen3 on
  this version → go to **Fallback A** (bitsandbytes).

### Step 3 — Smoke test (voice + size)
```powershell
ollama run enigma "Who are you? Answer in one line."
ollama ps        # shows it loaded, its VRAM, and the idle "Until" countdown
```
You want: a reply in Enigma's voice (not a generic assistant), and `ollama ps` showing
~5 GB. Confirm the chat format is clean (no leaked `<|im_start|>` tokens).

### Step 4 — Point Odysseus at it
In Odysseus chat:
```
/setup local http://127.0.0.1:11434/v1
```
Open the model picker → choose **`enigma`** → talk to it. (Ollama's OpenAI-compatible
endpoint lives at `:11434/v1`.)

### Step 5 — Verify the idle win
After ~5 min idle, re-check:
```powershell
ollama ps                                                   # should show enigma gone
nvidia-smi --query-gpu=memory.used --format=csv             # VRAM reclaimed
```
To tune how long it stays warm (e.g. keep it 30 min, or drop to 1 min):
```powershell
$env:OLLAMA_KEEP_ALIVE = "30m"     # set before launching the Ollama service
```

### Done = success criteria
- [ ] `ollama run enigma` replies in Enigma's voice
- [ ] `ollama ps` shows ~5 GB (Q4) while chatting
- [ ] Odysseus chats with `enigma` over `:11434/v1`
- [ ] VRAM returns to ~0 after idle timeout
- [ ] Update `HANDOFF.md` §1 to make Ollama the default chat path (bf16 = "max quality" note)

---

## FALLBACK A — bitsandbytes 4-bit in serve_enigma.py
Use if Ollama can't convert Qwen3. Keeps the single Python server; model stays resident
(~5 GB) the whole time the server is up (no idle-unload — that's the tradeoff vs Ollama).

Edit `serve_enigma.py`:

1. Add a flag near the other args:
```python
_p.add_argument("--quant", choices=["bf16", "4bit"], default="bf16")
```
2. Replace the model-load line with a branch:
```python
if ARGS.quant == "4bit":
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        ARGS.model, quantization_config=bnb, device_map="cuda")
else:
    model = AutoModelForCausalLM.from_pretrained(
        ARGS.model, dtype=torch.bfloat16).to("cuda")
```
   (4-bit uses `device_map`, not `.to("cuda")`.)
3. Run it:
```powershell
python serve_enigma.py --quant 4bit
```
**Caveat — Blackwell:** bitsandbytes needs a build with `sm_120` (5090) support. It's
installed, but if the load throws a CUDA/kernel error, that build predates Blackwell →
use Ollama (Primary) instead, which sidesteps bitsandbytes entirely.

---

## FALLBACK B — GGUF by hand (bundled llama.cpp)
Only if both above fail. Note: the bundled `enigma_engine/bin/llama-server/` ships the
**server**, not the converter/quantizer — you'd need `convert_hf_to_gguf.py` and
`llama-quantize.exe` from a llama.cpp release. Since Ollama *is* llama.cpp underneath,
if Ollama's converter rejects the arch this path likely will too. Steps if you pursue it:
1. `python convert_hf_to_gguf.py models/enigma-8b --outfile enigma-f16.gguf --outtype f16`
2. `llama-quantize.exe enigma-f16.gguf enigma-q4.gguf Q4_K_M`
3. Serve: `llama-server.exe -m enigma-q4.gguf --port 8000` → Odysseus `/setup local http://127.0.0.1:8000/v1`

---

## Rollback / coexistence
Nothing destructive here. To undo the Ollama model: `ollama rm enigma`.
The bf16 server (`python serve_enigma.py`) is untouched and remains the max-quality
option. You can keep both — bf16 on `:8000` for quality, Ollama on `:11434` for the
light everyday driver — and point Odysseus at whichever you want.

## Open questions to resolve when you run this
- Q4 vs Q5 — start Q4; bump to Q5 only if you feel a quality drop in real use.
- Keep-alive — default 5 min is fine; raise it if you chat in bursts and dislike the reload pause.
- Identity hardening — decide whether to bake the SYSTEM line into the Modelfile or leave
  identity to the (re-trainable) LoRA. Bigger corpus is the real fix (see HANDOFF §4).
