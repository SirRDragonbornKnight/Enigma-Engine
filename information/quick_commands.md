# Quick Commands Reference

Run everything from the Enigma Engine folder (venv activated).

---

## Serve

| Command | What It Does |
|---------|-------------|
| `python serve_enigma.py --model models/enigma_v2_sft2/model.pth` | OpenAI-compatible /v1 server on port 8000, serving the checkpoint of record |
| `enigma ...` | Same thing (console script installed by `pip install -e .`) |
| `python serve_enigma.py` | Without `--model`, serves the adopted `models/enigma_v2_sft2/model.pth` (the v2 lineage, adopted 2026-08-09) |
| `python serve_enigma.py --model models/enigma_dpo/model.pth` | Serve the v8 rollback instead |
| `python serve_enigma.py --port 8123` | Serve on a specific port |
| `python serve_enigma.py --host 0.0.0.0 --unsafe-lan` | Listen on all interfaces -- **WARNING: the API has no authentication.** Anyone on the network can read and change her memory and drive her voice. A non-loopback `--host` is REFUSED without `--unsafe-lan`; stay on 127.0.0.1 unless you know exactly why you need this. |
| `python serve_enigma.py --max-context 2048` | Set the context window (tokens; 2048 is the default and the v2 training block) |
| `python serve_enigma.py --memory-dir data/memory` | Enable the memory store (JSONL + BM25) + /v1/memory API |
| `python serve_enigma.py --voice` | Voice organ: `speak` tool + /v1/audio/speech (Kokoro-82M; run under the repo `venv/`) |
| `python serve_enigma.py --ears` | Ears organ: /v1/audio/transcriptions (faster-whisper) |
| `python serve_enigma.py --eyes` | Eyes organ: image messages captioned into context + /v1/images/describe (her own distilled ViT) |
| `python serve_enigma.py --image-gen` | Imagination organ: `imagine` tool + /v1/images/generations (Stable Diffusion) |
| `python serve_enigma.py --search` | Search organ: `<search>query</search>` spans run a lookup through this machine's own SearXNG (WSL2 docker at 127.0.0.1:8888) and the results return to her context (v2 vocab only) |
| `python serve_enigma.py --eyes --allow-downloads` | First-ever use of an organ on a machine: permit the one-time weight download. Without the flag the server is fully offline (cache only) |
| `python serve_enigma.py --device cpu` | Pick the device for the MODEL: `auto` (default, cuda when available), `cuda`, or `cpu`. `--device cuda` on a box with no CUDA REFUSES rather than falling back silently -- that silent fallback is how a CPU number gets recorded as a GPU one. Scope: the model and the eyes organ follow it; ears/painter/voice still choose their own device |

### Conversation levers (all OFF by default; ON in the daily launcher)

| Command | What It Does |
|---------|-------------|
| `python serve_enigma.py --dry-multiplier 0.8` | DRY sampling strength for requests that do not ask for one themselves (0 = off). A client that names `dry_multiplier` keeps what it asked for, including an explicit `0.0` -- so a caller can turn DRY off against a server that defaults it on. Attacks the 256-token verbatim loops |
| `python serve_enigma.py --tool-span-constrain` | Constrain what she may sample INSIDE a `<\|tool_call\|>` span to valid JSON (xgrammar). Missing wheel or a tokenizer-parity failure disables it with one WARN and decoding is byte-identical |
| `python serve_enigma.py --state-reinject` | Prefix the final user turn with the numeric facts the user stated earlier in THIS conversation (`[context: rent: 1350]`). Conversation-local only -- it never reads the memory store |

`Start-Enigma.ps1` (and therefore `Talk to Enigma.bat` / `Enigma Tray.bat` /
`Enigma HUD.bat`) passes `--state-reinject --tool-span-constrain --dry-multiplier 0.8`:
that is the **daily posture**, adopted 2026-09-01 on measured tables. Eval and scratch
serves build their own argv and stay baseline. Each is one token to remove.

### Waking unprompted (OFF by default)

| Command | What It Does |
|---------|-------------|
| `python serve_enigma.py --wake --wake-watch <folder>` | Let her speak unprompted: a loop beside the request lane that reacts to NEW files in the watched folder. Files already there when she boots are never announced. **Without `--wake-watch` nothing can wake her** -- boot says so out loud |
| `--wake-interval 1800` | Seconds between timer ticks (default 1800). A bare tick does NOT call the model: this lineage cannot emit the `NO_REPLY` sentinel the cheap-silence pattern needs, so ticks would only produce chatter (measured 2026-09-01). File drops always do |
| `--wake-cooldown 900` | Seconds she stays quiet after actually speaking (default 900). A heartbeat she answers with silence costs nothing |
| `--wake-quiet 23-8` | Quiet hours as `H-H`, wrapping midnight (default `23-8`). `0-0` disables |
| `GET /v1/wake/recent?n=20` | The last n things she said unprompted. Present even with `--wake` off (an empty feed is honest); the log lives in her data home, never the repo |

Organ flags combine freely, e.g. `python serve_enigma.py --voice --ears --eyes --memory-dir data/memory`.

---

## Portable / CPU (Enigma-to-go)

| Command | What It Does |
|---------|-------------|
| `python strip_serving_ckpt.py --in models/enigma_v2_sft2/model.pth --out "<new>.pth"` | Drop everything serving never reads (the AdamW optimizer state), keeping exactly `model_state_dict`/`config`/`step`/`meta`. Measured on sft2: **2,728.3 MB -> 909.4 MB**. Refuses an existing `--out` |
| `python quantize_serving_ckpt.py --in "<serving-only>.pth" --out "<int8>.pth"` | int8 weight-only (torchao, eager -- no compiler). Measured: **909.4 MB -> 292.7 MB (~3x smaller)**, top-1 agreement 97/100 vs fp32. **Smaller, NOT faster on this CPU**: 23.7 tok/s int8 vs 30.7 fp32 (no AMX). Refuses an existing `--out`, and refuses any `--out` inside `models/` |
| `python bench_generate.py --model <ckpt> --device cpu --threads 10` | Decode latency. `--threads` caps torch CPU threads (default 10, the Chrome-Remote-Desktop courtesy cap); the receipt line prints the resolved device and `torch.cuda.is_available()` so a CPU number proves it was CPU |
| `.\build_portable.ps1 -Target D:\Enigma-Portable -Force` | Assemble the USB folder: embeddable CPython + vendored wheels + her code + the int8 checkpoint (~944 MB). `Enigma-Portable.bat` serves on 127.0.0.1:8077 and opens her window. Never copies `data\`, `teachings.jsonl`, `models\` or `Enigma Backups\` -- and CHECKS the built folder to prove it |

---

## Training Pipeline (pretrain -> facts (optional) -> SFT -> DPO)

| Command | What It Does |
|---------|-------------|
| `python pretrain_enigma.py` | Pretrain from scratch on `data/pretrain/tokens.bin` |
| `python pretrain_enigma.py --sanity` | One forward/backward step, then exit (smoke test) |
| `python make_facts_pretrain_data.py --out data/pretrain/<new>.bin` | Build the facts continued-pretrain stream (knowledge install; see training_guide.md Stage 1.5; refuses an existing output) |
| `python pretrain_enigma.py --tokens-bin data/pretrain/<new>.bin --init-from models/enigma_v2_238m/model.pth --out models/<new_run_dir> --tokens 60e6 --lr 1e-4 --warmup 50 --val-general-end 0` | Low-LR continued pretrain that installs the knowledge corpus in weights (pretrain has NO existing-artifact guard -- name a genuinely new out dir) |
| `python make_sft_data.py` | Build SFT data -> `data/sft/{tool_calls,identity,mix}.jsonl` |
| `python finetune_enigma.py --data data/sft/mix.jsonl --out models/<new_run_dir>` | SFT the pretrained model into an instruct/tool model |
| `python make_dpo_data.py` | Build DPO preference pairs -> `data/sft/dpo_pairs.jsonl` |
| `python dpo_enigma.py --init models/enigma_v2_sft2/model.pth --out models/<new_run_dir>` | DPO alignment pass (default lr 5e-7 is the adopted setting; --out refuses an existing artifact) |
| `python sample_enigma.py --ckpt models/enigma_v2_238m/model.pth` | Sample raw text from a checkpoint (the v2 base; any .pth works) |

---

## Evaluation

Serve the candidate on its own port with an isolated memory dir, then run the harness:

| Command | What It Does |
|---------|-------------|
| `python serve_enigma.py --port 8123 --model models/<candidate>/model.pth --max-context 2048 --memory-dir data/memory_eval` | Serve the model under test |
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
| `python pretokenize_data.py --vocab <full path> --output-bin <full path> --dtype uint16` | Tokenize `data/pretrain/` sources into a corpus bin. The BARE invocation refuses on purpose: the default output is the write-protected v1 lineage `tokens.bin`, and paths must be passed in full (a wrong `--vocab` path used to fall back silently to an untrained tokenizer). See BACKLOG 7.95 T1 for the recorded v2 invocation. |

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
