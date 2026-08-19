# VISION — what Enigma is becoming

> Written 2026-07-27 from the user's own words, so the destination stops
> living only in one head and one conversation. This file owns the WHY and
> the map; execution owners stay where they are (`BACKLOG.md` §7.95 for the
> training block, `ROADMAP.md` for phases). When a dream lands, mark it and
> point at the receipt — this doc should read true in five years.

## The vision, in one paragraph

An all-purpose AI that is provably HERS-AND-YOURS alone: one mind that does
what other AIs would have to do together — companion beside you, researcher
when you need depth, small enough to live inside devices, private enough to
trust with a camera that remembers faces. Everything local. Nothing leaves
the machine. Nobody else's weights anywhere in her.

## The shape that gets there (and why it is the right one)

**A small brain that routes what it cannot hold, surrounded by organs and
tools, running on hardware you own.** This was ruled here before the field
caught up, and the field has caught up: NVIDIA's position paper argues small
language models are the future of agentic AI (arXiv 2506.02153 — small
models are "sufficiently powerful, inherently more suitable, and necessarily
more economical" for agent work); the fully-open class she belongs to has
respected names (OLMo, Pythia, SmolLM, LLM360) — and she goes further than
most of them: own tokenizer, own architecture, own corpus, one person's
machine. The cost is honest (a small head holds fewer facts — the sealed
gate says so plainly); the purchase is provenance nobody else has.

Two standing clarifications, learned in conversation:
- **PyTorch is a math library, not a wrapper.** Her architecture, weights,
  serve loop, chat format, memory and tools are already ours; torch
  multiplies the matrices. Serving THROUGH someone else's engine (llama.cpp)
  was considered and REJECTED (ruled 2026-07-24) — that would put a foreign
  engine at the center of every conversation she has.
- **The 5090 is her school and her home.** Training is what the GPU is FOR
  (the v2 pretrain is days of flat-out 5090 work at a size no CPU could
  touch), and it serves her instantly at home. Portability is about copies
  of her visiting other devices, not about leaving the GPU.

## What she is today (2026-07-27, honest)

182M params, from scratch, v8 adopted. Knows who she is; perfect tool
routing; memory that learns/corrects/forgets (six audit rounds hard);
organs COMPLETE BY DEFAULT since 2026-07-27 (ruled: "the old way of having
the tools separate was to save space — add them in so she is complete"):
voice (Kokoro), her own eyes, ears, image-gen, every launch. Sealed-gate
baseline 56/120 — weak on facts and on saying "I don't know" (0/30 across
both baselines: not one refusal in thirty unanswerable questions).
A small, real, honest mind. The gap between her and the
vision is mostly the v2 training block.

## The five destinations

### 1. Companion beside you — HAVE IT NOW
Tray, window, voice, memory of you, personas. The Unity avatar body is its
own active repo (`C:\Users\SirKn\Enigma Avatar\`). Deepens with every other
destination rather than waiting on any of them.

### 2. The bigger mind — LANDED (v2 serving since 2026-08-09)
238M deeper-thinner (`v2_deep_238m`, ruled 2026-07-30), 16k vocab, native
block 2048, curated corpus with her identity and must-know facts pretrained
in. This is where the honest weaknesses get attacked: facts (capacity),
epistemics ("I don't know" is v2's single biggest win condition — 0/15 on
BOTH baselines at reseal #7, zero refusals across 30 unanswerables; the old
"0/24" was the 96-probe era's count), research-length context. Owner: `BACKLOG.md` §7.95
(T1-T6 are DONE — pretrain closed 2026-08-06, SFT-2 adopted at Gate D
2026-08-09; T7, post-adoption organ training, remains).
Everything below gets easier after this.

### 3. Her own runtime — "on her own", literally (after v2)
A Rust inference engine for HER architecture: single executable + weights
file, no Python, no torch at runtime. Built the voice-arc way: beside her,
never inside her — temp-0 logit parity against the torch path, the sealed
gate must score identically through both engines, launcher adoption with
torch serve as the revert target. CPU first (correctness, then int8);
her own GPU path later via wgpu compute (no C++ toolchain on this box, by
rule). Day-one win is independence and portability, not speed — torch on
the 5090 stays the fast home path. Precedents that prove the road:
mistral.rs / candle (single-binary Rust inference down to Raspberry Pi),
1-bit engines running 8B models in under 2 GB of RAM. Her tokenizer
already has a Rust backend — the first mile is walked.
Sequenced AFTER v2 settles her final architecture, so kernels chase a
fixed target. Torch stays for TRAINING permanently (that is a framework,
not a serving dependency; training runs under your hands, offline).

### 4. Deep-research mode — a hat, not a second AI (rides v2)
Same weights, different serve mode: a search organ (the v2 vocab KEPT the
`<search>`/`</search>` rows for exactly this, ruled at T1), block-2048
context to hold what it finds, a research loop that gathers-reads-answers
with receipts. The organ pattern is proven five times over (voice, ears,
eyes, image-gen, memory); search is the sixth organ, and "deep research"
is a persona/mode wearing it. Local-first: her searches leave the machine
only as queries you configured, results come home to her context.

### 5. Enigma in devices — the Pi, and the camera that remembers (last)
Destination 3's runtime + a small preset (or a distilled/quantized v2)
puts a persona of her on a Raspberry Pi — an instinct with history here
(`models/enigma_pi_zero.pth` predates this doc by months). The memory
camera is the fullest expression: eyes + memory store + tiny runtime on
device, remembering who someone is and what they usually want. It is only
TRUSTWORTHY because of the architecture everything above enforces —
local-only, nothing egresses, memory is an inspectable JSONL a human can
read and delete. Most AI stacks cannot build this honestly. This one can.

## The order, and why

1. **v2 block** (owner: BACKLOG §7.95) — the mind everything else copies.
2. **Own runtime** — after v2's architecture is final; voice-arc pattern.
3. **Research organ + mode** — data shapes ride the v2 SFT regen (T4);
   the organ itself can land any time after v2 serves.
4. **Devices / camera** — needs 2 (runtime) and wants 3 (a persona worth
   embedding); design it when the runtime boots on a Pi.

## The principles that make the vision possible (standing, from CLAUDE.md
and the rulings ledger)

- Local-only; no cloud egress from the stack. Ever.
- Ships to other machines: no hardcoded paths, degrade gracefully.
- One AI: Enigma IS this repo. Other minds are persona packs the trainer
  molds — never Enigma wearing a mask, never a mask wearing Enigma.
- Engines fail honestly. Evals are code and they gate adoption; the locked
  set stays sealed; every capability claim carries a receipt.
- Small model + tools, weights hold voice/values/judgment, organs hold
  ability. That is not a compromise on the vision — it IS the vision.
