# Code Review Tracker — reset 2026-06-11

Pre-refocus findings targeted the Qwen-era engine (`inference.py`,
`engine_generation.py`, `api/server.py`, …) — those findings and their
fixes live in git history. Suite baseline today (2026-07-06, post
audit-fix pass): **418 passed; repo-wide `ruff check` clean.**

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

## Recently closed (2026-07-06 full-audit fix pass)

All findings from the 5-reader audit + live pipeline test, every one
hand-verified against the code before fixing. Live re-verified after:
eval_behavior 26/29 PASS on a fresh memory dir; stalled-stream
concurrency probe PASS; hand-edited memories.jsonl (BOM + missing/string
ids) loads clean.

- **serve: string tool-arguments 500** — a model-emitted tool call whose
  `"arguments"` was valid JSON but not an object crashed `_execute_builtin`
  (`.get` on a str). Non-objects now come back as an `error:` tool result.
- **serve: `_GEN_LOCK` held across SSE writes** — a streaming client that
  stopped reading parked the lock forever and every request behind it. A
  worker thread now owns the lock for the generation and hands ids over a
  queue (`_stream_ids_locked`); client disconnect cancels the worker.
- **serve: `finish_reason` on tail-flush stop** — a stop marker completing
  inside the held-back tail reported `"length"`; now `"stop"`.
- **serve: raw tool-call text across hops** — malformed (unparsable) call
  text generated alongside an executed built-in vanished when the hop loop
  continued; it now accumulates and surfaces in the final response.
- **MemoryStore: readers now share the writers' lock** (search/all/len) —
  a concurrent supersede could misalign the BM25 record/term-vector zip.
- **MemoryStore: hand-edited files** — records with missing/non-int ids are
  renumbered at load (previously: KeyError/TypeError 500 on the next add);
  reads use `utf-8-sig` so a Windows-editor BOM can't corrupt line 1.
- **AdvancedBPETokenizer.decode leak** — `skip_special_tokens=True` passed
  reserved ids 4-9 through as literal `<sep>/<mask>/<Q>/<A>/<USER>/<BOT>`;
  decode now strips every reserved name found in the vocab file, via a
  decode-only skip set (encode behavior byte-identical, verified).
- **Trainer disk-backed mode deleted the caller's dataset** — `_val_file`
  aliases `data_path` there; cleanup now exempts both aliases.
- **Trainer `_get_sequence_logps` length-averaged** — now summed, matching
  the documented DPO/APO formulas and `dpo_enigma.py`; KTO baseline scaled
  by beta and clamped >= 0 to sit on the reward scale.
- **Gradient noise under fp16 GradScaler** — noise was added to scaled
  grads (effectively /65536); grads are now unscaled before noise/clip.
- **Mid-epoch resume scheduler overrun** — the replayed partial epoch
  pushed cosine past T_max (LR climbing) / WSD below its floor; the
  schedule is stretched by the replayed steps at resume and the WSD lambda
  is floored at `min_lr_ratio`.
- **Eager-path BPE-dropout froze one tokenisation** — multi-epoch eager
  runs now rebuild batches per epoch while dropout is active.
- **Online-DPO epoch accounting** — mid-epoch generated pairs are staged
  for the next epoch instead of joining the list being iterated.
- **kv_share_groups + gradient checkpointing** — refused at the layer
  forward (backward recompute would rebuild followers from projections
  that never ran forward: silently wrong gradients).
- **KV cache oversize prefill** — truncate-and-warn (NaN cascade into
  uniform-random sampling) is now a `ValueError`.
- **TurboQuant rebalance / H2O batch eviction / StreamingLLM zero-points**
  — rebalance re-encodes flipped heads' history; H2O evicts per batch row;
  the window shift moves `_zp_k/_zp_v` with the scales. Regression tests
  added for each.
- **`from_pretrained`/`from_safetensors` no-config fallback** — `cls()`
  was a TypeError (config is required); now `cls(ForgeConfig())`.
- **`apply_lora(merge=False)` silent no-op** — zero-match now raises, same
  guard as `merge_lora_weights`; adapters register only after matching.
- **`atomic_safetensors_save`** — fsyncs before the rename (same
  power-loss window `atomic_torch_save` already closed).
- **NF4 quantization vs weight tying** — Linears whose weight storage is
  shared (tied output head) are skipped via `data_ptr` detection.
- **`forward_multimodal` mask dtype** — cast to `h.dtype` like the main
  forward (fp32 mask on a bf16/fp16 model raised in SDPA).
- **Dead loaders fail honestly** — `from_huggingface/from_gguf/from_onnx`
  raise NotImplementedError (their converter modules don't exist; the old
  ImportError advice to pip-install couldn't help).
- **`training_evaluation.evaluate_tool_usage`** — graded the legacy
  `[CMD]` format; now parses live `<|tool_call|>` JSON and the default
  cases use real pipeline tools.
- **collect_finetuning_data** — dedup key gains a field separator
  (`prompt\x00completion`); SmolTalk2 keeps newlines in think-tagged turns
  (same rule as OpenThoughts3).
- **eval_behavior** — probe questions ASCII-guarded like the detail text.
- **sample_enigma default checkpoint** — now the completed 182M
  `enigma_pretrain_large/model.pth` (was the step-10k base).
- **Docs-only truth syncs** — `pretokenize_data.py` states the real
  `<bos> … <eos> <eos>` per-document layout (existing corpus/lineage;
  behavior deliberately unchanged); `progressive_growing.py` states that
  only depth growth is output-preserving (width is a warm start).

## Recently closed (2026-06-10 → 06-11)

- **KV-cache decode mask bug** — rectangular SDPA decode used a top-left
  aligned causal mask, corrupting served generation. Fixed (bottom-right
  aligned mask) and LOCKED by `tests/test_model_kv_cache.py` (cached ==
  uncached, logit-for-logit).
- **model.py cleanup** — slimmed to the live architecture; checkpoint
  verified bit-identical before/after
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
- **Muppet-era scripts resolved** (06-11): superseded by
  `finetune_enigma.py` (git is the archive). The reusable EXAMPLES live in
  `identity_anchors.py` (06-30, renamed from `make_enigma_corpus.py`);
  `run_training_diagnostic.py` stays with the dormant FORGE stack.
