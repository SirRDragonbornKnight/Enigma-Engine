# Strategy — current as of 2026-06-11

> Supersedes the 2026-05-26 "Strategy Reset" (Qwen3 fine-tune + Gradio UI). That
> plan was rejected 2026-06-02 — *"actually train Enigma... do not just put
> another ai in it like it is a Muppet"* — and replaced by the Modkit refocus.
> The old document lives in git history.

## The shape

- **Odysseus** (separate repo, `C:\Users\SirKn\odysseus`) = the app/UI: chat,
  MCP client, image gen + editor, TTS/STT, OCR, browser, 40+ agent tools,
  model serving, deep research. We do not build UI in this repo.
- **This repo** = the from-scratch LLM: model forging (pretrain/SFT/DPO) + serving.
  (SUPERSEDED 2026-07-13: the old "Modkit" capability-backend framing — `mods/`, MCP
  `modkit_mcp.py`, in-repo codegen/see_screen/voice — was REMOVED. Capabilities now
  return as organ/hub services exposed to Enigma as tools, not in-repo mods. The pip
  distribution is `enigma-engine`.)
- **Enigma** (the heart) = the from-scratch transformer
  (`enigma_engine/core/model.py`, 182M `large` preset), pretrained on the
  own-built 56.6B-token corpus via `pretrain_enigma.py`.
  **DONE 2026-07-03: full 287,882 steps / 56.6B tokens, val ppl 3.5**
  (`models/enigma_pretrain_large/model.pth`; SHA256-receipted backup at
  `C:\Users\SirKn\Enigma Backups\enigma_pretrain_large_final\`). The finished
  lineage is IMMUTABLE — any continuation gets a new run directory. The full
  phase-by-phase forward plan now lives in `ROADMAP.md`.

## Roadmap (mouth & hands)

1. **Pretraining — DONE 2026-07-03 at the full 287,882 steps (56.6B tokens,
   val ppl 3.5).** `models/enigma_pretrain_large/model.pth`, backed up with
   SHA256 receipts. The bottleneck is no longer code or compute — it is SFT
   data. See `ROADMAP.md` Phase 1 (fatten the tool corpus, chat-shaped general
   data, values/identity authorship) for what comes next.
2. **Mouth — DONE 2026-06-11.** `serve_enigma.py` serves the from-scratch
   model: OpenAI-compatible `/v1`, KV-cache streaming, plain-transcript chat
   bridge until the instruct pass. First live words from the 51k checkpoint
   verified.
3. **Hands — infrastructure BUILT 2026-06-11; the pass itself runs after
   pretraining completes.** `enigma_engine/core/chat_format.py` owns the
   format: chat tokens 4718–4723 in the padded free rows (+ the tokenizer's
   native `<think>`=10/11), ONE ChatML-shaped template shared by training and
   serving, ID-level tool-call parsing. `finetune_enigma.py` is the bespoke
   SFT trainer (assistant-only loss masking, chat-row re-init, shares the
   pretrain arsenal — Muon/WSD work here too). `make_sft_data.py` builds the
   data. `serve_enigma.py` auto-detects instruct checkpoints
   (`meta.chat_format`): real template, OpenAI `tool_calls`, streaming.
   Proven end-to-end on a throwaway nano — it learned to emit
   `<|tool_call|>{…}<|/tool_call|>` unprompted. Before the REAL pass: fatten
   the tool corpus (29 seed examples) and curate the values data.
4. **Memory/skills — v1 BUILT 2026-06-11.** `memory_store.py`: stdlib BM25
   over inspectable JSONL. `serve_enigma.py --memory-dir` injects top hits
   into her system context (128-id budget; silence when nothing matches),
   plus `/v1/memory` GET/POST. She trains at block 1024 — injected context
   stays compact until a length-extension anneal.
5. **Values/identity corpus.** Constitutive alignment: hand-curated examples,
   scaling the proven identity-lock approach. The seed exists
   (`data/sft/identity.jsonl`: 360 from-scratch-true anchors; the old
   Qwen-era claims are false for this model). The curation pass is the user's
   authorship. The WHY lives in the vision memory (Jarvis-class companion
   that provably won't turn evil).
6. **Avatar embodiment.** The avatar is its own repo now (`C:\Users\SirKn\Enigma Avatar\`); its `TODO.md` is AUTHORITATIVE for that
   backlog — read it before any avatar task.

## 2026 landscape check (researched 2026-06-11)

Verdict: nothing in the live run is wrong. The stack matches current
small-model practice (GQA + qk-norm + SwiGLU at ~8/3·dim + RMSNorm + tied
embeddings + RoPE; bf16 autocast + torch.compile; AdamW 0.9/0.95 with decay
only on ≥2-D tensors), and the frozen-weights + external-memory/tools learning
model is the 2026 consensus, not a compromise. The advances below are queued
for FUTURE decisions — none justify touching the paused run mid-schedule:

- **Muon optimizer** — production-proven in 2025-26 (Kimi K2 1T, GLM,
  Megatron support; ~1.3-2× data/compute efficiency vs AdamW). Candidate for
  the instruct pass and any next pretrain. Never mid-run.
  **BUILT 06-11:** `pretrain_enigma.py --optimizer muon`, flag-gated, default
  `adamw` = the live path; resume optimizer-mismatch fails loudly.
- **WSD / decay-to-zero schedules** beat fixed-budget cosine when training
  might continue past the planned budget. Adopt for the NEXT run; the current
  cosine run keeps its recorded schedule.
  **BUILT 06-11:** `--schedule wsd --wsd-decay-frac 0.1`, flag-gated, default
  `cosine` bit-identical to the live formula (regression-tested).
- **Multi-epoch data** (data-constrained scaling laws): up to ~4 epochs of
  the same corpus ≈ fresh tokens; meaningful gains decay around 16. Our run
  is single-epoch (56.6B) — after step 287,882 a continuation over the same
  corpus is legitimate, modern, and the cheapest capability lever we have.
- **Depth vs width:** 2026 small models run deeper-thinner (SmolLM2-135M is
  30 layers; ours is 16×1024). A next-architecture consideration, not an
  error — wider buys throughput on a single consumer GPU.
- **Intra-document attention masking** (Llama 3): negligible effect at block
  1024 by Meta's own measurement; becomes important IF we do the
  length-extension anneal. The 2025 extension recipe is settled: raise RoPE θ
  + continued pretraining on long documents (<10B tokens) — fold both into
  that decision.
- **min-p sampling** is now in every major serving stack (llama.cpp defaults
  it at 0.1); cheap optional add to serve/sample for high-temperature
  stability. Measured benefit is contested — nicety, not a need.
  **BUILT 06-11:** plumbed through generate/generate_stream + the server's
  request models (`min_p`, default 0 = off; the filter already lived in
  `sample_next_token`).
- **Tokenizer:** small vocab is *defensible* at our compute scale (vocab
  scaling laws: embedding FLOPs saved fund longer training; big vocabs
  underfit rare tokens). The standalone-space token (26.6% of the stream)
  remains our real inefficiency — a next-generation tokenizer should merge
  leading spaces GPT-2-style before any re-pretrain. Not fixable for this
  lineage: retokenizing means a new run.

## Principles

- **Black box:** local-only; no cloud egress from the stack.
- **Ships to other machines:** no hardcoded user paths; degrade gracefully
  with NO model present.
- **Keep ideas, not code** — git is the archive; verify before delete.
- **The training arm is the moat:** Odysseus serves models but cannot train
  them. Modkit can.
