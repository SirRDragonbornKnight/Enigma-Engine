# Forge Quick Test Guide

Everything you need to run a fast smoke test on each Forge mode.
All settings below are chosen for speed — not quality.

---

## Before Any Test

1. Go to **MODELS tab** → create a model called `smoke` with Memory GB = `0.5`
2. That gives you the `pi_zero` preset (~1M params) — tiny, fast, fits any test
3. All forge modes use this same model unless noted

---

## FOUNDATION MODES

### Pre-Train
**What it does:** Trains the model from scratch on raw text. Teaches it language.

| Setting | Value |
|---|---|
| Mode | Pre-Train |
| Student | smoke |
| Data file | `data/smoke_test_basic.txt` (59 KB) |
| Epochs | 1 |
| LR | 0.00005 |
| Batch | 1 |
| Grad Accum | 1 |
| Memory GB | 0.5 |

**Expected:** Runs ~82 steps, loss drops from ~16 to ~10, done in 2-3 min.

---

### Basic (Fine-Tune)
**What it does:** Trains on your own text or JSONL. The everyday fine-tune mode.

| Setting | Value |
|---|---|
| Mode | Basic |
| Student | smoke |
| Data file | `data/enigma_gui_training.txt` (36 KB) OR `data/smoke_test_basic.txt` |
| Epochs | 1 |
| LR | 0.00005 |
| Batch | 1 |

**Expected:** Short run, loss drops, model saved. Done in 1-2 min.

---

### Distill
**What it does:** A teacher model (Ollama/external) generates training data, then trains the student.

**Requires:** An Ollama model running locally (devstral, qwopus, etc.)

| Setting | Value |
|---|---|
| Mode | Distill |
| Trainer (teacher) | Any loaded Ollama model |
| Student | smoke |
| Prompts | Use default / leave as-is |
| Samples | 10 (minimum for a quick test) |

**Expected:** Teacher generates 10 Q&A pairs, then trains student on them. Takes 2-5 min depending on teacher speed.

---

### Image
**What it does:** Trains on image+caption pairs. Teaches visual understanding.

**Requires:** Image files + caption text files in `data/avatar/` or similar folder.
Format: `image.png` + matching `image.txt` with caption.

**Status:** Skip for now unless you have image training data ready.

---

## ADVANCED MODES

### AI-Guided
**What it does:** AI teacher builds a curriculum automatically and trains the student through stages (basics → reasoning → advanced).

**Requires:** Ollama model as teacher.

| Setting | Value |
|---|---|
| Mode | AI-Guided |
| Trainer (teacher) | Any loaded Ollama model |
| Student | smoke |
| Stages | Reduce to 1-2 stages for quick test |

**Expected:** Teacher generates curriculum, student trains on each stage. 5-15 min.

---

### Dialogue
**What it does:** Teacher and student have a conversation. Teacher scores student responses and corrects them. Good for personality training.

**Requires:** Ollama model as teacher.

| Setting | Value |
|---|---|
| Mode | Dialogue |
| Trainer | Any loaded Ollama model |
| Student | smoke |
| Rounds | 5 (minimum) |

**Expected:** 5 conversation rounds, teacher scores each, student trains on corrections. 3-10 min.

---

### RLHF
**What it does:** Trains a reward model on preference data, then uses it to score student responses and improve via policy gradient.

**Requires:** JSONL file with `prompt`, `chosen`, `rejected` fields.

| Setting | Value |
|---|---|
| Mode | RLHF |
| Student | smoke |
| Data file | `data/smoke_test_dpo.jsonl` (10 KB, already has correct format) |
| Reward epochs | 1 |

**Expected:** Phase 1 trains reward model, Phase 2 does RL. 5-10 min.

---

### Self-Play
**What it does:** Teacher generates responses, judges them, student learns from the best ones via RL.

**Requires:** Ollama model as teacher.

| Setting | Value |
|---|---|
| Mode | Self-Play |
| Trainer | Any loaded Ollama model |
| Student | smoke |
| Rounds | 3 |

**Expected:** Teacher generates + judges, student trains on winners. 5-15 min.

---

## ALIGNMENT MODES

> All alignment modes need a JSONL file with `prompt`, `chosen`, `rejected` fields.
> Use `data/smoke_test_dpo.jsonl` for all of them.

### GRPO
**What it does:** RL without a separate critic network. Simpler than PPO/RLHF.

| Setting | Value |
|---|---|
| Mode | GRPO |
| Student | smoke |
| Data file | `data/smoke_test_dpo.jsonl` |

---

### ReMax
**What it does:** REINFORCE with a mean-reward baseline. Even simpler than GRPO.

| Setting | Value |
|---|---|
| Mode | ReMax |
| Student | smoke |
| Data file | `data/smoke_test_dpo.jsonl` |

---

### SimPO
**What it does:** Preference optimization without a reference model. Fastest alignment option.

| Setting | Value |
|---|---|
| Mode | SimPO |
| Student | smoke |
| Data file | `data/smoke_test_dpo.jsonl` |

---

### ORPO
**What it does:** SFT + alignment in one pass. No separate reference model needed.

| Setting | Value |
|---|---|
| Mode | ORPO |
| Student | smoke |
| Data file | `data/smoke_test_dpo.jsonl` |

---

## DATA FILE REFERENCE

| File | Size | Format | Used By |
|---|---|---|---|
| `data/smoke_test_basic.txt` | 59 KB | Plain text Q&A | Pre-Train, Basic |
| `data/enigma_gui_training.txt` | 36 KB | Plain text | Basic |
| `data/smoke_test_dpo.jsonl` | 10 KB | `{prompt, chosen, rejected}` | RLHF, GRPO, ReMax, SimPO, ORPO |
| `data/curated_dataset.jsonl` | 2 KB | `{text, source}` | Basic (JSONL mode) |

---

## KNOWN ISSUE FIXED

**NaN at step ~60 with tiny data + Seq Packing:**
Fixed in `enigma_engine/core/training.py` — packing masks now clamp `-inf` to `-1e9`
so all-padding rows don't explode softmax under torch.compile.

---

## MODES THAT NEED EXTRA SETUP

| Mode | What You Need |
|---|---|
| Distill | Ollama running with at least one model |
| AI-Guided | Ollama running with at least one model |
| Dialogue | Ollama running with at least one model |
| Self-Play | Ollama running with at least one model |
| Image | Image files + caption text files |

Ollama is already installed. Run `ollama list` in terminal to see which models are available.

---

## RECOMMENDED TEST ORDER

1. **Pre-Train** — confirms the core training pipeline works end-to-end
2. **Basic** — confirms fine-tune path works
3. **SimPO** — fastest alignment mode, just needs the DPO file
4. **ORPO** — similar to SimPO, validates alignment in one pass
5. **RLHF** — validates reward model + policy gradient
6. **Distill** — validates Ollama connection + teacher pipeline
7. **Dialogue** — validates conversation loop
8. **AI-Guided** — validates full curriculum system
9. **GRPO / ReMax** — additional RL modes
10. **Self-Play** — most complex, test last
11. **Image** — needs extra data prep, do separately
