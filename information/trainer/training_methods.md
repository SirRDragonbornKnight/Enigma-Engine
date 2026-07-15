# Training Methods

Overview of the training methods available in Enigma Engine. The
current pipeline is pretrain -> SFT -> DPO (see
[../training_guide.md](../training_guide.md)).

---

## Pretraining (from scratch)

Next-token prediction over the pre-tokenized corpus
`data/pretrain/tokens.bin`.

**How to use:**
```
python pretrain_enigma.py --size large --tokens 2e9
```

**What to expect:**
- Loss should decrease steadily; val loss is checked every `--eval-every` steps
- Checkpoints save every `--save-every` steps, atomically
- `--resume` continues an interrupted run

---

## Supervised Fine-Tuning (SFT)

Feed the model chat-formatted examples; it learns to predict the
assistant tokens (only assistant tokens are in the loss mask).

**How to use:**
```
python finetune_enigma.py --data data/sft/mix.jsonl --out models/enigma_sft
```

**Parameters (defaults):**
| Parameter | Default | Description |
|-----------|---------|-------------|
| Epochs | 2 | Full passes through the data |
| Learning Rate | 2e-5 | ~peak/30 of pretraining |
| Micro-batch | 8 | Sequences per forward pass |
| Grad accum | 4 | Micro-batches per optimizer step |

---

## DPO (Direct Preference Optimization)

Preference alignment on `{"prompt", "chosen", "rejected"}` pairs --
policy vs frozen reference, no reward model.

**How to use:**
```
python dpo_enigma.py --init models/enigma_sft/model.pth --out models/enigma_dpo
```

Default lr 5e-7 is the adopted setting; higher rates over-optimized in
measured runs.

---

## LoRA / Evolutionary / RLHF (code exists, not in the pipeline)

- **LoRA** utilities live in `enigma_engine/core/lora_utils.py`, but the
  current pipeline scripts do full-parameter training.
- **Evolutionary / self-play / RLHF** code lives in
  `enigma_engine/core/rl_training.py` (RewardModel, RLHFTrainer,
  SelfPlayTrainer); it is dormant and partially implemented (see
  SUGGESTIONS.md -- PPO needs a rewrite before use).

None of these have a supported command today.

---

## The Tokenizer

The BPE tokenizer (base vocab 4718, `enigma_engine/vocab_model/`) is
already trained and **fixed** -- every checkpoint's embedding depends on
it, so retraining it is not part of any method above.

---

## Tips

1. **Start with --sanity** -- one step, then exit; validates the whole path
2. **Watch the loss** -- decreasing loss means learning is working
3. **Clean your data** -- quality matters more than quantity
4. **Eval every run** -- `eval_behavior.py` against a served checkpoint is the scorecard
