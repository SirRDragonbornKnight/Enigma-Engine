# Code Review Tracker — reset 2026-06-11

Pre-refocus findings targeted the Qwen-era engine (`inference.py`,
`engine_generation.py`, `api/server.py`, …) — those modules are deleted; the
findings and their fixes live in git history. Suite baseline today:
**364 passed; ruff clean on the live/changed files.** (Repo-wide `ruff check .`
still flags pre-existing style nits in legacy scratch/collector scripts and
`mods/` — cosmetic, untouched by the 06-11 work; sweep opportunistically.)

## Open

- **PERF (gated):** ToMe token-merging helpers in `model_components.py` use
  Python loops — matters only if `tome_ratio` is ever enabled (0.0 everywhere).
  Deferred.
- **HYGIENE (dormant code):** broad `except Exception` patterns remain in the
  FORGE training stack (`training/training.py`, `core/rl_training.py`). Clean
  opportunistically when that stack is next touched (see CLEANUP_TRACKER
  ruling).
- **torch.compile on Windows:** MFU ceiling ~23–26% from graph breaks; the
  live path is eager + SDPA (`is_causal=True` fast kernel). Chasing the
  compile ceiling is high-risk/low-reward on this stack. Deferred.

## Recently closed (2026-06-10 → 06-11)

- **KV-cache decode mask bug** — rectangular SDPA decode used a top-left
  aligned causal mask, corrupting served generation. Fixed (bottom-right
  aligned mask) and LOCKED by `tests/test_model_kv_cache.py` (cached ==
  uncached, logit-for-logit).
- **model.py cleanup** — dead speculative-decoding suite + MoE expert layer
  removed; checkpoint verified bit-identical before/after
  (`_verify_ckpt.py`, KEYHASH `12edc0bc1ded383d`).
- **Footgun defaults flipped** — `use_differential_attn` True→False,
  `neftune_alpha` 5.0→0.0 in `model_presets.py`; `--no-diff-attn` is now
  redundant belt-and-braces.
- **Trainer hardening** — schedule persisted in checkpoints and restored on
  resume (`--override-schedule` to change), `prev.pth` rotation +
  finite-loss save guard + `--archive-every`, `[val-gen]` second eval window
  (fenced from train sampling), hard-fail on missing `--resume` path.
- **serve_enigma.py EOS bug** — `encode()` brackets prompts as
  `[BOS]…[EOS]`; serving without stripping the trailing EOS made the model
  see a *finished document* and reply with EOS/new-document. Fixed (mirrors
  `sample_enigma.py`).
- **serve max_tokens unclamped** (06-11 audit finding) — generation length
  was client-controlled past block 1024 / the RoPE table. Clamped to
  `--max-context - 2`.
- **val-gen fence redraw int32 overflow** (06-11, caught ~30 steps into the
  resume) — the rejection-fence redraw in `get_batch` called
  `np.random.randint(lo, hi-block-1)` without `dtype=np.int64`; NumPy's legacy
  `randint` keeps C-long (= int32 on Windows) semantics even on NumPy 2.x, and
  `hi` ≈ 56.7e9 → `ValueError: high is out of bounds for int32`. Latent since
  the 06-10 hardening: the fence fires only when a draw lands in the val-gen
  window (~3.4%/step across 192 draws — expected first hit ≈ step 29 of the
  resume; it crashed right on schedule), the pre-06-10 trainer had no fence,
  and the smoke run's corpus was int32-small. The main draw one line up
  already passed the dtype. Fixed to match; checkpoint untouched (crash
  preceded the first save); reproduced + verified in the launch env.
- **Base-mode `usage` off by two** (06-11 readiness probe) — chat/completions
  counted tokens by re-encoding text, and `encode()`'s BOS/EOS bracketing
  inflated both sides (completion_tokens could read > max_tokens). Now counts
  what was fed/generated (`add_special_tokens=False`; +1 for the fed BOS).
  Instruct mode always counted real ids and was exact.
- **Future-run arsenal landed flag-gated** (06-11, from the 2026 landscape
  check): `--optimizer muon` (Moonlight NS5 variant; composite with aux
  AdamW; resume mismatch fails loudly), `--schedule wsd` (decay-to-zero),
  min-p plumbed through generate/serve (default 0). Defaults reproduce the
  live run exactly — locked by `tests/test_pretrain_arsenal.py` (cosine LR
  bit-identical + AdamW grouping order-identical regressions). Smoke-proven
  on a throwaway nano run incl. schedule-lock resume to-the-digit.
- **Instruct-pass infrastructure built** (06-11, "complete enigma engine"):
  `chat_format.py` (tokens 4718–4723, one train==serve template, ID-level
  tool parsing — attaching specials proven not to change plain-text
  encoding), `finetune_enigma.py` (masked SFT in the pretrain pattern),
  `make_sft_data.py`, serve instruct auto-detect + `memory_store.py` (BM25)
  + `/v1/memory`. 18 new tests; end-to-end nano smoke: the format IS
  learnable (it emitted `<|tool_call|>` spans unprompted; malformed JSON
  degrades to a raw fallback, never a crash). Two probe-time catches fixed
  before landing: META read after `del _ck` (boot crash), and a memory test
  budget that ignored the space-heavy tokenizer.
- **Muppet-era scripts resolved** (06-11): `train_enigma_lora.py`,
  `make_enigma_local.py`, `forge.py` deleted (zero importers; git is the
  archive). The reusable EXAMPLES were extracted to `identity_anchors.py`
  (06-30, renamed from `make_enigma_corpus.py`; the Qwen-ChatML LoRA corpus
  writer and `eval_enigma.py` were dropped); `run_training_diagnostic.py`
  stays with the dormant FORGE stack.
