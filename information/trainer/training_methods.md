# Training Methods

Overview of all training methods available in Enigma Engine.

---

## Supervised Fine-Tuning (SFT)

The standard method. Feed the model text and it learns to predict the next token.

**How to use:**
- GUI: Open the FORGE page, select data, set epochs, click START TRAINING
- CLI: `python run.py --train data/training.txt --epochs 10`

**Parameters:**
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Epochs | 10 | 1-1000 | Full passes through the data |
| Batch Size | 4 | 1-256 | Samples processed together |
| Learning Rate | 0.0001 | 0-1 | How fast the model learns |

**What to expect:**
- Loss should decrease over epochs
- Training auto-stops if loss becomes NaN
- Checkpoints save every N epochs (configurable)

---

## LoRA / QLoRA

Low-Rank Adaptation trains small adapter layers instead of the full model.

- **LoRA** — Adds small trainable matrices to attention layers
- **QLoRA** — Same as LoRA but quantizes the base model to 4-bit first

**Benefits:**
- Uses far less VRAM than full fine-tuning
- Faster training
- Adapter files are small (typically 10-100 MB)
- Can be merged back into the base model

---

## Evolutionary Training

An experimental method that uses natural selection to improve outputs.

**How it works:**
1. Run N instances on the same task
2. Score all outputs (rule-based, self-eval, perplexity)
3. Keep the best output
4. Fine-tune the model on the winner
5. Repeat for multiple generations

**Scoring methods:**
- Rule-based checks (do tests pass?)
- Self-evaluation by the model
- Consistency across multiple runs
- Length and format validation
- Perplexity scoring

---

## Tokenizer Training

Before training a model, you can train a BPE tokenizer on your data.

**How to use:**
- GUI: Set vocab size on FORGE page, click TRAIN TOKENIZER
- CLI: `python run.py --train-tokenizer data/training.txt --vocab-size 8000`

This creates a tokenizer optimized for your specific vocabulary and domain.

---

## Tips

1. **Start small** — Train a tiny model first to validate your data
2. **Watch the loss** — Decreasing loss means learning is working
3. **Clean your data** — Quality matters more than quantity
4. **Save often** — Checkpoints let you resume if something fails
5. **Use the DOCS page** — Edit training data files directly in the built-in editor
