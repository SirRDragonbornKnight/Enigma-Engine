# Training Guide

Enigma is trained in three passes, all from the repo root:

```
pretrain -> SFT -> DPO -> (eval) -> serve
```

Every pass renders text through the same
`enigma_engine/core/chat_format.py` used at serve time, so the model is
trained on exactly the bytes she will serve.

---

## Stage 1: Pretraining (from scratch)

Learns language from raw text. Runs on the pre-tokenized corpus at
`data/pretrain/tokens.bin` (uint32 token ids, vocab 4718).

```
python collect_pretraining_data.py --all-sources   # gather raw text
python pretokenize_data.py                         # -> data/pretrain/tokens.bin
python pretrain_enigma.py                          # the long run
```

Useful flags: `--size` (architecture preset, see
[trainer/model_sizes.md](trainer/model_sizes.md)), `--tokens` (target
training tokens), `--optimizer adamw|muon`, `--schedule cosine|wsd`,
`--resume` (continue an interrupted run), `--sanity` (one step, then exit).

---

## Stage 2: SFT (instruct/tool fine-tune)

Turns the pretrained base into an instruct model that follows the chat
format and calls tools.

```
python make_sft_data.py                            # -> data/sft/{tool_calls,identity,mix}.jsonl
python finetune_enigma.py --data data/sft/mix.jsonl --out models/enigma_sft
```

`--init` defaults to `models/enigma_pretrain_large/latest.pth`.
Defaults: 2 epochs, lr 2e-5 (~peak/30 of pretraining), block 1024.

**Data formats** (JSONL, one record per line):

| Format | Example | Notes |
|--------|---------|-------|
| Messages | `{"messages": [{"role": "user", "content": "..."}, ...]}` | Full chat turns; assistant turns may include tool calls |
| Prompt/completion | `{"prompt": "...", "completion": "..."}` | `response`/`answer`/`output` keys also accepted |
| Reasoning | assistant content containing `<think>...</think>` | Think spans are native tokenizer ids 10/11 -- still fully supported |

---

## Stage 3: DPO (preference alignment)

Direct Preference Optimization -- teach the model to prefer certain
responses over others, with no reward model: policy + frozen reference,
one loss. At 182M the realistic targets are format, tone, and identity
integrity, not new knowledge.

```
python make_dpo_data.py                            # -> data/sft/dpo_pairs.jsonl
python dpo_enigma.py --init models/enigma_sft/model.pth --out models/enigma_dpo
```

**Data format** -- each line has three fields:

```json
{"prompt": "What is 2+2?", "chosen": "4", "rejected": "22"}
```

The model learns to generate responses closer to `chosen` and further
from `rejected`. Beta: 0.1. The default learning rate **5e-7 is the
adopted setting** (the old 2e-6 over-optimized and damaged the model --
if you raise it, re-run the eval).

---

## Evaluating a Run

Every SFT/DPO candidate is graded against a *running* server -- the real
production path, not an in-process approximation:

```
python serve_enigma.py --port 8123 --model models/enigma_sft/model.pth --memory-dir data/memory_eval
python eval_behavior.py --base-url http://127.0.0.1:8123     # in another shell
```

`--memory-dir data/memory_eval` keeps eval memories out of your real store.

---

## The Tokenizer

The BPE tokenizer (base vocab 4718, in `enigma_engine/vocab_model/`) is
**fixed**: it produced `tokens.bin` and every checkpoint's embedding is
tied to it. Retraining the tokenizer invalidates all existing weights,
so it is not part of the normal cycle.

---

## Checkpoints

- Each run writes to its `--out` directory under `models/`
  (`latest.pth` during the run, `model.pth` at the end).
- Saves are atomic (write to `.tmp`, then rename) so they survive
  crashes and power loss.
- `--resume` continues an interrupted run; `--sanity` smoke-tests the
  full forward/backward path in one step.
- The served checkpoint of record is `models/enigma_dpo/model.pth`.

---

## Tips

1. **Sanity first** -- run `--sanity` before committing to a long run
2. **Clean data matters** -- garbage in, garbage out
3. **Watch the loss** -- it should decrease; val loss is checked periodically
4. **Small lr for DPO** -- preference tuning over-optimizes fast at this scale
5. **Eval as code** -- trust `eval_behavior.py` scores, not vibes
6. **Any format works for SFT** -- mix messages and prompt/completion records freely
