# Quick Commands Reference

Run everything from the Enigma Engine folder (venv activated).

---

## Serve

| Command | What It Does |
|---------|-------------|
| `python serve_enigma.py --model models/enigma_dpo/model.pth` | OpenAI-compatible /v1 server on port 8000, serving the checkpoint of record |
| `enigma ...` | Same thing (console script installed by `pip install -e .`) |
| `python serve_enigma.py` | Without `--model`, serves the adopted `models/enigma_dpo/model.pth` (default since 2026-07-17) |
| `python serve_enigma.py --port 8123` | Serve on a specific port |
| `python serve_enigma.py --host 0.0.0.0` | Listen on all interfaces |
| `python serve_enigma.py --max-context 1024` | Set the context window (tokens) |
| `python serve_enigma.py --memory-dir data/memory` | Enable the memory store (JSONL + BM25) + /v1/memory API |
| `python serve_enigma.py --voice` | Voice organ: `speak` tool + /v1/audio/speech (Kokoro-82M; run under the repo `venv/`) |
| `python serve_enigma.py --ears` | Ears organ: /v1/audio/transcriptions (faster-whisper) |
| `python serve_enigma.py --eyes` | Eyes organ: image messages captioned into context + /v1/images/describe (her own distilled ViT) |
| `python serve_enigma.py --image-gen` | Imagination organ: `imagine` tool + /v1/images/generations (Stable Diffusion) |
| `python serve_enigma.py --eyes --allow-downloads` | First-ever use of an organ on a machine: permit the one-time weight download. Without the flag the server is fully offline (cache only) |

Organ flags combine freely, e.g. `python serve_enigma.py --voice --ears --eyes --memory-dir data/memory`.

---

## Training Pipeline (pretrain -> facts (optional) -> SFT -> DPO)

| Command | What It Does |
|---------|-------------|
| `python pretrain_enigma.py` | Pretrain from scratch on `data/pretrain/tokens.bin` |
| `python pretrain_enigma.py --sanity` | One forward/backward step, then exit (smoke test) |
| `python make_facts_pretrain_data.py` | Build the facts continued-pretrain stream -> `data/pretrain/facts_tokens.bin` (knowledge install; see training_guide.md Stage 1.5) |
| `python pretrain_enigma.py --tokens-bin data/pretrain/facts_tokens.bin --init-from models/enigma_pretrain_large/latest.pth --out models/enigma_pretrain_facts --tokens 60e6 --lr 1e-4 --warmup 50 --val-general-end 0` | Low-LR continued pretrain that installs the knowledge corpus in weights |
| `python make_sft_data.py` | Build SFT data -> `data/sft/{tool_calls,identity,mix}.jsonl` |
| `python finetune_enigma.py --data data/sft/mix.jsonl --out models/enigma_sft` | SFT the pretrained model into an instruct/tool model |
| `python make_dpo_data.py` | Build DPO preference pairs -> `data/sft/dpo_pairs.jsonl` |
| `python dpo_enigma.py --init models/enigma_sft/model.pth --out models/enigma_dpo` | DPO alignment pass (default lr 5e-7 is the adopted setting) |
| `python sample_enigma.py --ckpt models/enigma_pretrain_large/model.pth` | Sample raw text from a checkpoint |

---

## Evaluation

Serve the candidate on its own port with an isolated memory dir, then run the harness:

| Command | What It Does |
|---------|-------------|
| `python serve_enigma.py --port 8123 --model models/enigma_sft/model.pth --memory-dir data/memory_eval` | Serve the model under test |
| `python eval_behavior.py --base-url http://127.0.0.1:8123` | Behavior eval against the running server (in another shell) |

---

## Data Collection

| Command | What It Does |
|---------|-------------|
| `python collect_pretraining_data.py --stats` | Show collected pretraining data summary |
| `python collect_pretraining_data.py --all-sources` | Download pretraining text (Wikipedia, Gutenberg, FineWeb-Edu, ...) |
| `python collect_finetuning_data.py --all` | Download all instruction datasets except OpenThoughts3 (OASST1, Dolly, SlimOrca, SmolTalk2, No Robots, Everyday Conversations, TriviaQA, NQ-Open) |
| `python collect_finetuning_data.py --no-robots N --everyday N --triviaqa N --nq-open N --smoltalk2 N` | Cherry-pick the short-completion "diet" sources with per-source caps (see `--help`) |
| `python collect_distill_data.py --model <teacher>` | Collect responses from an external OpenAI-compatible teacher as a fine-tune corpus (`--model` is required) |
| `python collect_search_data.py` | Emit the synthetic `<search>` tag training corpus |
| `python collect_vision_data.py --llava-pretrain 100000 --images-dir <extracted images.zip>` | Download image-caption pairs for vision SFT (bare invocation just prints help) |
| `python pretokenize_data.py` | Tokenize `data/pretrain/` sources into `data/pretrain/tokens.bin` |

Each script documents its sources and flags in its docstring (`--help`).

---

## Development & Testing

| Command | What It Does |
|---------|-------------|
| `python -m pytest tests/ -v` | Run all tests (verbose) |
| `python -m pytest tests/ --tb=short -q` | Run all tests (compact output) |

Linting is NOT part of this project's loop: ruff was dropped by user ruling
2026-07-18. Two leftovers remain in pyproject.toml and are equally inert:
the `[tool.ruff]` config and the `ruff>=0.4.0` dev dependency.
Run the tests instead.

---

## Tips

- The server speaks the OpenAI API -- any OpenAI client library or UI works.
- Training scripts checkpoint atomically; interrupt and `--resume` freely.
- `--sanity` on pretrain/SFT/DPO scripts runs one step then exits -- use it before long runs.
