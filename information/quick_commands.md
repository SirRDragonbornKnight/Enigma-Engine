# Quick Commands Reference

These are the commands you can run from a terminal or cmd window.
All commands should be run from the Enigma Engine folder.

---

## Launch Commands

| Command | What It Does |
|---------|-------------|
| `python run.py` | Show system info and test imports |
| `python run.py --gui` | Launch the desktop GUI |
| `python run.py --gui --model models/my.pth` | Launch GUI with a model pre-loaded |
| `python run.py --serve` | Start the API server (default port from config) |
| `python run.py --serve --port 8080` | Start API server on a specific port |
| `python run.py --serve --host 0.0.0.0` | Start API server on all network interfaces |
| `python run.py --serve --api-key YOUR_KEY` | Start API server with authentication |
| `python run.py --chat` | Simple CLI chat (requires a loaded model) |
| `python run.py --chat --model models/my.pth` | Chat with a specific model |
| `python run.py --chat --profile creative_writer` | Chat using a specific AI profile |
| `python run.py --chat --temperature 0.7` | Chat with custom temperature |
| `python run.py --help` | Show all available options |

---

## Training Commands

| Command | What It Does |
|---------|-------------|
| `python run.py --train data/training.txt` | Train model on a text file |
| `python run.py --train data/qa.jsonl` | Train model on JSONL data |
| `python run.py --train data/training.txt --epochs 20` | Train with custom epoch count |
| `python run.py --train data/training.txt --batch-size 8` | Train with custom batch size |
| `python run.py --train data/training.txt --lr 0.0003` | Train with custom learning rate |
| `python run.py --train data/training.txt --model-size large` | Train a large model |
| `python run.py --train data/training.txt --seed 42` | Train with a fixed random seed |
| `python run.py --train --resume models/checkpoints/best_model.pt` | Resume training from checkpoint |

### Model Size Presets (CLI `--model-size`)

In the **GUI**, type a GB number in the Memory field instead — the engine auto-picks the best preset.

| Size | Parameters | VRAM Needed |
|------|-----------|-------------|
| pi_zero | ~500K | <1 GB |
| nano | ~1M | <1 GB |
| tiny | ~5M | ~1 GB |
| small | ~27M | ~2 GB |
| medium | ~85M | ~4 GB |
| large | ~200M | ~6 GB |

---

## Tokenizer Commands

| Command | What It Does |
|---------|-------------|
| `python run.py --train-tokenizer data/training.txt` | Train a BPE tokenizer on text |
| `python run.py --train-tokenizer data/ --utf8-bytes` | Train byte-level BPE tokenizer |
| `python run.py --train-tokenizer data/training.txt --vocab-size 16000` | Custom vocabulary size |
| `python run.py --analyze-tokenizer` | Analyze the trained tokenizer |

---

## Benchmark & Evaluation

| Command | What It Does |
|---------|-------------|
| `python run.py --benchmark` | Run coherence benchmark on default model |
| `python run.py --benchmark --model models/my.pth` | Benchmark a specific model |
| `python run.py --golden-eval prompts.json` | Run golden prompt regression eval |

---

## Data Collection

| Command | What It Does |
|---------|-------------|
| `python collect_pretraining_data.py --stats` | Show collected data summary |
| `python collect_pretraining_data.py --all-sources` | Download from all sources |
| `python collect_pretraining_data.py --books-only` | Download only Gutenberg books |
| `python collect_pretraining_data.py --resume` | Resume an interrupted download |

---

## Development & Testing

| Command | What It Does |
|---------|-------------|
| `python -m pytest tests/ -v` | Run all tests (verbose) |
| `python -m pytest tests/ --tb=short -q` | Run all tests (compact output) |
| `ruff check enigma_engine/` | Lint the codebase |
| `ruff check --fix enigma_engine/` | Auto-fix safe lint issues |

---

## Batch File

Double-click `Launch Enigma.bat` to start the GUI without opening
a terminal first. It activates the venv automatically.

---

## Tips

- If this is a fresh install, just run `python run.py` first.
  It will create the virtual environment and install dependencies
  automatically.
- Use `--model-size` to pick how big a model to train. Bigger
  models need more GPU memory but learn better.
- Training can be stopped at any time. The best checkpoint is
  saved automatically.
- The API server is compatible with OpenAI-format requests.
