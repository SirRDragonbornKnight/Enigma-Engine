# Training Guide

Enigma is trained in three passes, all from the repo root:

```
pretrain -> facts continued-pretrain (optional) -> SFT -> DPO -> (eval) -> serve
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
python pretokenize_data.py --vocab <full path> --output-bin <full path> --dtype uint16
python pretrain_enigma.py                          # the long run
```

NOTE: the bare `pretokenize_data.py` invocation refuses on this checkout --
its default output is the write-protected v1 lineage `tokens.bin`, and a
wrong `--vocab` path refuses rather than silently falling back to an
untrained tokenizer. Pass both paths in full; the recorded v2 invocation
lives in BACKLOG 7.95 T1.

Useful flags: `--size` (architecture preset, see
[trainer/model_sizes.md](trainer/model_sizes.md)), `--tokens` (target
training tokens), `--optimizer adamw|muon`, `--schedule cosine|wsd`,
`--resume` (continue an interrupted run), `--sanity` (one step, then exit).

---

## Stage 1.5: Facts continued-pretrain (optional knowledge hop)

SFT surfaces knowledge; it cannot install it (measured 2026-07-15:
"largest planet" -> Jupiter but "biggest planet" -> Saturn). Installation
happens in pretraining, where a fact appears in many textual forms. This
short low-LR pass mixes the `knowledge_corpus.py` fact lines (declarative /
QA / cloze / in-context) into replay chunks from the real corpus, so the
model learns the facts without forgetting the language:

```
python make_facts_pretrain_data.py                 # -> data/pretrain/facts_tokens.bin (60M tokens, ~2% facts)
python pretrain_enigma.py --tokens-bin data/pretrain/facts_tokens.bin \
    --init-from models/enigma_pretrain_large/latest.pth \
    --out models/enigma_pretrain_facts --tokens 60e6 --lr 1e-4 --warmup 50 \
    --val-general-end 0
```

Then point SFT at the facts base instead of the raw one:
`python finetune_enigma.py --init models/enigma_pretrain_facts/latest.pth ...`.
Measured effect on the 90-probe gate: factual 13/20 -> 19/20 (v6 lineage);
the adopted v8 sits on this base.

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

The general-conversation side of the mix comes from
`collect_finetuning_data.py`. `--all` downloads every source EXCEPT
OpenThoughts3 (its completions are block-unfit at 1024). The short-completion
"diet" sources added 2026-07-15 are cherry-pickable with per-source caps:
`--smoltalk2 N` (+ `--smoltalk2-config/--smoltalk2-split/--smoltalk2-cap`),
`--no-robots N`, `--everyday N`, `--triviaqa N`, `--nq-open N` -- see
`--help` for defaults. Length caps differ per source (audit 2026-07-17):
No Robots and Everyday Conversations cap completions at 600 chars;
TriviaQA and NQ-Open cap ANSWERS at 80 chars; SmolTalk2 is UNCAPPED unless
you pass `--smoltalk2-cap 600` (the 2026-07-15 diet did -- and note `--all`
downloads SmolTalk2 uncapped too). Overlong records are dropped at bake,
so an uncapped pull wastes download, not training.

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
fixed FOR A LINEAGE: it produced `tokens.bin`, and every v1 checkpoint's
embedding is tied to it. Retraining the tokenizer invalidates existing
weights, so it is not part of the normal cycle -- it is what STARTS a new
lineage.

That is exactly what the v2 work is. A second vocab already exists
(`bpe_vocab_v2_16k.json`, 16,366 rows, 2.41x chars/token) with its own
corpus (`data/pretrain/tokens_v2b.bin`), waiting on the pretrain in
`BACKLOG.md` §7.95. The two coexist safely because serve and finetune pick
the vocab from the CHECKPOINT's `vocab_size`, not from a global default --
so a v1 checkpoint keeps loading the v1 vocab no matter what else is on
disk. Read this section as "don't swap the vocab under an existing
lineage", not as "the tokenizer never changes".

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
