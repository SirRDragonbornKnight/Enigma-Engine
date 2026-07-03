# Training Guide

Train your own AI models using the FORGE page in the GUI or the CLI.

---

## Training Modes

The FORGE page offers six training modes, accessible from the mode
dropdown. Each mode has different requirements and use cases.

### Self Study (Solo)

Standard supervised fine-tuning. The model learns to predict the
next token from your training data.

**Requires:** STUDENT model assigned + training data file.

**Data formats:**
- **Plain text** (.txt) — the model learns the raw text
- **Q&A format** — lines starting with `Q:` and `A:`
- **Dialogue format** — lines starting with `User:` and `AI:` (or `Human:` / `Assistant:`)
- **JSONL** — one JSON object per line with `prompt` and `completion` keys
- **Reasoning JSONL** — JSONL with `<think>` tags for chain-of-thought
- **Mixed** — any combination of the above in one file

**CLI:**
```
python run.py --train data/training.txt --epochs 10 --model-size small
```

### Conversation (Dialogue)

Live conversation training between TRAINER and STUDENT models.

1. TRAINER asks questions based on the current training stage
2. STUDENT answers
3. TRAINER scores and provides corrections
4. Corrections become training data
5. STUDENT is fine-tuned on the corrections

**Requires:** Both TRAINER and STUDENT models assigned.
**Params:** Rounds (default 10, max 200).
**Default learning rate:** 0.00005.

### Preference Tuning (DPO)

Direct Preference Optimization — teach the model to prefer
certain responses over others.

**Requires:** STUDENT model + `.jsonl` data file.

**Data format:** Each line must have three fields:
```json
{"prompt": "What is 2+2?", "chosen": "4", "rejected": "22"}
```

The model learns to generate responses closer to `chosen`
and further from `rejected`. Beta value: 0.1.

### Image Training (Vision)

Train the model to understand images using a Vision Transformer.

**Requires:** STUDENT model + folder of image-caption pairs.

**Data format:** Each image needs a matching caption:
- `cat.png` + `cat.txt` (same name, different extension)
- Or a `captions.jsonl` file mapping images to text

**Vision encoder presets:**

| Preset | Layers | Dim | Heads | Params |
|--------|--------|-----|-------|--------|
| Tiny | 2 | 128 | 4 | ~500K |
| Small | 4 | 256 | 4 | ~4M |
| Medium | 6 | 512 | 8 | ~25M |

### Quick Tune (LoRA)

Low-Rank Adaptation — train a small adapter instead of the full model.
Much faster and uses less memory.

**Requires:** STUDENT model + training data file.

**Params:**
- **LoRA Rank** — size of adapter matrices (default 8, range 1–128)
- **LoRA Alpha** — scaling factor (default 16, range 1–256)

Uses the PEFT library when available. Falls back to partial freeze
training (last quarter of layers + output layer).

### Trial & Error (Evolutionary)

Self-play evolution. Generates multiple responses per task,
selects the best, and trains on winners.

**Requires:** STUDENT model + training data file.

**Params:**
- **Generations** — evolution rounds (default 5, range 1–100)
- **N per task** — candidates per generation (default 3, range 2–20)

---

## Train with AI Toggle

When enabled, training becomes an AI-assisted 3-phase process:

1. **Phase 1:** TRAINER generates curriculum for the current stage
   and focus field (format varies by stage — see table below)
2. **Phase 2:** STUDENT is trained on the generated material
3. **Phase 3:** TRAINER tests STUDENT with 10 questions and scores
   each answer 1–10

The data file becomes optional — the TRAINER generates training
data automatically. The current training stage and focus field
guide what the TRAINER teaches. Generated curriculum is saved
to `data/adaptive_{student}_{stage}_{timestamp}.txt` so you can
review it on the DOCS page.

---

## Training Stages

Four progressive stages control what the TRAINER teaches:

| Stage | Focus | Data Format | Restrictions |
|-------|-------|-------------|-------------|
| BASICS | Coherent sentences, greetings, short answers, basic facts | Mixed: statements, greetings, definitions, opinions | 1–2 sentences max. No commands, code, or web |
| CONVERSATION | Natural dialogue, multi-turn responses, follow-ups, personality | `User:` / `AI:` dialogue (2–4 turns) | No commands or web |
| COMMANDS | Tool usage via `[CMD]command[/CMD]` syntax | `Q:` / `A:` with `[CMD]` blocks | No web commands |
| WEB | Web tools: `search.web`, `web.fetch` | `Q:` / `A:` with search/fetch commands | Full capabilities |

Select the active stage using the stage buttons on the FORGE page.
The TRAINER adapts all generated data, supplements, and corrections
to match the selected stage's format automatically.

---

## Focus Field

A text entry on the FORGE page that narrows training to a specific
domain. For example:

- "Python programming"
- "Medical terminology"
- "Customer support for SaaS products"

The focus field is injected into the TRAINER's system prompt and
affects all training modes, data generation, evaluation, and
web learning.

---

## FORGE Tools

### GENERATE DATA

The TRAINER autonomously generates training examples appropriate
for the current stage and focus field. The format adapts to the
stage — basics produces varied text, conversation produces
dialogue, commands and web produce Q/A with tool usage.

Output is saved to `data/generated_{stage}_{trainer_name}.txt`.
If a data file with extra prompts is provided, those supplements
are also formatted to match the current stage.

If **Auto-train** is checked, the generated data file is
automatically selected and training starts immediately.

### WEB LEARN

1. Enter a topic in the web learn field
2. The engine searches DuckDuckGo for related pages
3. Fetches and extracts text from the top N results (1–10 pages)
4. TRAINER reads the page content and generates training examples
5. Saves to `data/generated_{stage}_{trainer_name}.txt`

If **Auto-train** is checked, training starts automatically
on the generated file.

### EVALUATE

TRAINER generates stage-appropriate questions, STUDENT answers
each one, and TRAINER scores responses 1–10. Determines whether
the STUDENT is ready to advance to the next training stage.

No data file required.

### Auto-Train

When checked, both GENERATE DATA and WEB LEARN will automatically
start training on the newly generated data after generation completes.

---

## Hyperparameter Presets

The preset dropdown fills in epoch, learning rate, and batch size:

| Preset | Epochs | Learning Rate | Batch Size |
|--------|--------|---------------|------------|
| Quick | 3 | 0.0001 | 4 |
| Balanced | 10 | 0.00005 | 4 |
| Thorough | 30 | 0.00002 | 2 |
| Custom | (no change) | | |

---

## Training Parameters

| Parameter | GUI Default | Description |
|-----------|------------|-------------|
| Epochs | 10 | Full passes through the data |
| Batch Size | 4 | Samples processed together |
| Learning Rate | 0.00005 | How fast the model learns |
| Vocabulary Size | 8000 | BPE tokenizer vocabulary |

Additional parameters in TrainingConfig:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Weight Decay | 0.01 | L2 regularization |
| Warmup Steps | 100 | Learning rate warmup |
| Gradient Clip | 1.0 | Max gradient norm |
| Early Stopping Patience | 5 | Stop if no improvement for N epochs |
| Max Loss | 100.0 | Abort if loss exceeds this |
| AMP | True | Automatic mixed precision (fp16) |

---

## Model Sizes

**GUI (MODELS tab):** Type a number in the **Memory (GB)** field
and click CREATE. The engine auto-picks the largest architecture
that fits your memory budget.

| Memory (GB) | Auto-picks | ~Params | Train VRAM |
|-------------|-----------|---------|-----------|
| 1 | small | ~27M | ~1 GB |
| 4 | base | ~120M | ~3 GB |
| 8 | large | ~200M | ~6 GB |
| 12+ | xl | ~600M | ~12 GB |

**CLI:** Use `--model-size` with a preset name:

```
python run.py --train data/training.txt --model-size large
```

Available presets: `pi_zero`, `nano`, `tiny`, `small`, `medium`, `large`.

RoPE head dimensions are always even (required for
rotary embeddings).

---

## Training a Tokenizer

Train a BPE tokenizer on your specific data before training a model:

```
python run.py --train-tokenizer data/training.txt --vocab-size 8000
```

This creates a tokenizer optimized for your vocabulary.

---

## Checkpoints

- Training automatically saves checkpoints to `models/checkpoints/`
- Each checkpoint includes both the model weights and architecture config
- Checkpoints use atomic saves (write to `.tmp`, then rename) to
  survive crashes and power loss
- The FORGE shows a progress bar during training

---

## Training History

Every completed training run is recorded in two places:

1. **Global history** — `data/training_history.json` (last 200 runs)
2. **Model identity** — stored in the model's context file alongside
   display name, stats, and other identity data

View training history from the FORGE HISTORY button.

---

## Supported Data Formats

Training data files can use any of these formats (mixed in one file
is fine — the parser detects each automatically):

| Format | Example | When to Use |
|--------|---------|------------|
| **Q&A** | `Q: What is Python?`<br>`A: A programming language.` | Commands, web, factual |
| **Dialogue** | `User: Hello!`<br>`AI: Hi there!` | Conversation, personality |
| **JSONL** | `{"prompt": "...", "completion": "..."}` | Structured data |
| **Raw text** | Any plain paragraphs | General knowledge |
| **Reasoning** | JSONL with `<think>` tags | Chain-of-thought |

You can also use `Human:` / `Assistant:` instead of `User:` / `AI:`.
The parser handles all of these regardless of which stage you select.

The **TRAINER** generates data in the format that matches the
current stage, but your **own data files** can use any format.

---

## Tips

1. **Start small** — train a tiny model first to test your data
2. **Clean data matters** — garbage in, garbage out
3. **Watch the loss** — it should decrease over epochs
4. **NaN detection** — training auto-stops if loss becomes NaN
5. **Early stopping** — stops if loss stops improving
6. **Use stages** — progress BASICS → CONVERSATION → COMMANDS → WEB
7. **Focus field** — narrow training to your domain for better results
8. **Web Learn** — quickly generate domain-specific training data
9. **Any format works** — mix Q&A, dialogue, and raw text freely
10. **Review generated data** — check DOCS page for saved curriculum
