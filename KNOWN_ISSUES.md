# Known Issues — current as of 2026-07-26

_Navigation layer over `SUGGESTIONS.md` (landscape research + principles),
`_archive/CODE_REVIEW.md` (bugs), `CLEANUP_TRACKER.md` (file state)._

0. **`encode()` misses a chat special that directly follows `}`.** Measured
   2026-07-25: `encode('<|/tool_call|>')` is `[4721]` and
   `encode('} <|/tool_call|>')` ends `[..., 4721]`, but
   `encode('{"a": 1}<|/tool_call|>')` leaves the closing marker as ordinary
   text — so a whole tool-call span encoded as ONE string comes back
   unparseable, with the marker inside the payload. All six chat tokens
   round-trip correctly on their own.
   NOT a live path: `render_training` emits the span markers as IDS (verified
   — every one of the six appears in the rendered ids), and the model generates
   them as ids at serve time, so neither training nor serving encodes such a
   string. It bites anything that builds a tool-call span by string
   concatenation and then encodes it — write the ids directly instead.

1. **TRAINING -- DONE 2026-07-03: full 287,882 steps / 56.6B tokens, val ppl 3.5.** Base model shipped to `models/enigma_pretrain_large/model.pth` (final-save NaN guard passed), backed up with SHA256 receipts at `C:\Users\SirKn\Enigma Backups\enigma_pretrain_large_final\`. The finished lineage is immutable; any continuation run gets a NEW directory. Forward plan: `ROADMAP.md` (bottleneck is now SFT data, not compute).

   _History below = the pause playbook from the 2026-06-12 / 2026-06-25 pauses; kept only for the operational pattern (detached resume, daily shortcuts, Windows Update note) in case a future NEW run needs it._

   **(2026-06-12) TRAINING PAUSED at step 58,500/287,882 (20.3%) on user order**
   ("pause the training at a good point"). Stopped cleanly: killed the python
   worker (PID 17952) immediately after the step-58,500 eval+save flushed —
   `latest.pth` = step 58,500 (06:25:38), `prev.pth` = step 58,250 (rotation
   working). The process had run 10 more steps (to 58,510) before the kill, so
   resume re-does ~10 steps (~40 s) — effectively zero loss. Log
   (`train_large.log`) went silent and stayed silent; nothing left running.
   **Resume command:** the schedule is now RECORDED in the checkpoint (since
   the 51,250 save), so a bare `--resume models/enigma_pretrain_large/latest.pth`
   restores tokens/lr/warmup/batch/etc. from the ckpt — but re-using the full
   proven launch (same flags + `--no-grad-ckpt --archive-every 25000`) is
   safest and matches the recorded schedule. (`--no-diff-attn` appeared in
   the original command; the flag was removed 2026-07-19 — argparse now
   rejects it, drop it from any replayed command.) ETA from here ≈ 10½
   days of stepping if run continuously. **At the user's ~8 h/day cadence**
   (2026-06-12) the ~245 remaining GPU-hours stretch to **~31 active days ->
   mid-July** wall-clock. Stop/start is cheap (<=16 min/stop; the schedule is
   step-based, so chopping it into daily sessions does not distort training).
   - **Daily workflow / desktop shortcuts (2026-06-12):** "Resume Enigma
     Training" (-> `resume_training.ps1`: detached launch with already-running +
     checkpoint guards) starts it; "Enigma Training Log" (-> `tail_training_log.ps1`:
     read-only live tail) watches it. **STOP is not scripted by design -- the
     user asks Claude to stop it at a safe checkpoint.** ⚠️ Launcher `.ps1` files
     MUST stay pure ASCII: PowerShell 5.1 decodes BOM-less scripts as ANSI, so
     em-dashes / smart-quotes break parsing on double-click (caught + fixed
     during the build).
   - **History of this run:** resumed 2026-06-11 from step 51,000 on the user's
     word ("once we are ready start it"); attempt 1 crashed ~30 steps in on the
     int32 fence-redraw bug (fixed, see `_archive/CODE_REVIEW.md`); attempt 2 ran healthy
     ~8 h overnight (51k → 58.5k, ~50–52k tok/s, val ppl steady 3.8) until this
     pause. Detached `cmd /c` → `train_large.log`, survives Claude sessions.
   - (a) ~~**torch.compile fell back to eager**~~ — RESOLVED. triton-windows
     was invisible from outside Claude Desktop's MSIX-virtualized AppData.
     Measured 2026-07-28: `import triton` gives 3.7.0 under torch 2.10.0+cu128
     and the trainer prints `torch.compile: enabled`. The old "no cost" note is
     also wrong for the v2 shapes: on `v2_deep_238m` at the launch shape,
     compiled throughput is ~67.4k tok/s against the 31,311 recorded by the
     eager-era probe (BACKLOG 7.9 carries the re-measured grid). The v1-era
     ~52k figure stands for the v1 shape and is not comparable.
   - (b) **Windows Update still NOT paused** (Settings UI needs a user click —
     changing update settings is a prohibited action for me, and the no-UI
     route needs admin). Queue verified clean of reboot-class items; next Patch
     Tuesday (Jul 14) is past the ETA. **User: Settings → Windows Update →
     Pause updates** before the next long resume. The 06-10 stop was
     KB5094126 force-rebooting at 01:40 (clean OS kill, checkpoint intact).
2. **Validation signal is split by domain.** `[val]` (the corpus tail) is 100%
   anime since the 2026-06-07 append. `[val-gen]` (pre-append window, now
   fenced out of train sampling) is the general-domain gauge; it was ~16%
   train-leaked before the fence landed, so it reads slightly optimistic —
   frozen at that level, not growing.
3. **Context:** she trains at block 1024. `max_seq_len` 4096 + RoPE θ=500k
   give mechanical headroom but quality beyond 1024 is untested. Keep served
   context ≤1024 until a length-extension anneal is decided (interacts with
   the schedule lock — decide deliberately).
4. **Chat has two modes (auto-detected per checkpoint).** BASE checkpoints
   (today's 51k) get the plain-transcript bridge — she may continue the
   dialogue with invented speakers (observed: "Petitioner:"); stop markers
   catch `User:`/`Enigma:` turns only. SFT checkpoints (`meta.chat_format`
   from `finetune_enigma.py`) get the real template + tool calls. SFT
   checkpoints are live (`models/enigma_sft*`); serve auto-detects per
   checkpoint.
5. **base_v2 (122M @ step 2,000) is pipeline-validation quality only** —
   barely trained. Don't judge her by it; probe the large 51k checkpoint.
6. **Vendored weight -- CLOSED 2026-07-25.** `enigma_engine/bin/llama-server/`
   was 1.07 GB (1,066,991,160 bytes) of CUDA DLLs for the GGUF route,
   intentional while the GGUF serving pivot was open. That pivot was
   **REJECTED 2026-07-24** (serving stays from-scratch) and the binary was
   **DELETED 2026-07-25** -- the directory was gitignored and never committed,
   so this checkout has nothing left to delete (~1 GB freed). `core/gguf.py`
   stays dormant by ruling.
7. **Environment quirks (this dev box):** MCP servers load ONLY from the
   project `.mcp.json`; Claude Desktop is MSIX-sandboxed so `%LOCALAPPDATA%`
   writes are virtualized; no Windows admin without explicit user grant.
8. **Tokenizer facts:** token 44 (space) ≈ 26.6% of corpus tokens (29.5% on
   the 2026-07-16 English-sample re-measure) — baked-in
   inefficiency, not a bug. `encode()` brackets text as `[BOS]…[EOS]` — strip
   the trailing EOS before generation or the model sees a finished document
   (`sample_enigma.py` and `serve_enigma.py` both do this).
9. **The python suite is engine-only** -- the live suite count lives in
   `CLEANUP_TRACKER.md` (788 passed as of 2026-07-26). The
   avatar lives in its own repo (`C:\Users\SirKn\Enigma Avatar\`) — its gate
   is `powershell -File tools\verify.ps1` + `python -m pytest python/tests`
   (`node --test` belongs to the Electron predecessor repo).
10. **Model capacity ceilings are measured, not guessed.** `PHASE7_GATE.md`
    holds the receipts: long conversation (block 1024), broad-fact recall
    (~50%), raw arithmetic (bypassed via the server-side `calculate` tool).
    Current SFT data state lives in `BACKLOG.md` 7.95 P2 -- the sealed-gate
    baseline is 56/120 (v8) and 55/120 (v5), measured 2026-07-27 under
    reseal #7. The two are statistically INDISTINGUISHABLE (paired exact
    p = 1.00 over 19 disagreements), so 56/120 is a floor to clear, not a
    score to beat by a probe.
11. **Modkit-refactor module deletions RESTORED 2026-07-13 (audit).** The
    refactor (`0bd9167e`) deleted modules while their callers survived; an
    import-integrity sweep found SIX dangling imports. Restored verbatim from
    `0bd9167e^`: `core/vision_encoder.py` (935 lines, ViT + screen/camera
    capture), `core/audio_encoder.py` (673 lines, Conformer + from-scratch
    mel pipeline), `core/gguf.py` (1,487 lines — `Enigma.export_to_gguf()`
    was a hard crash; the vendored llama-server route in #6 depended on it
    at the time -- that route is closed and gguf.py stays dormant by ruling),
    `core/reasoning.py` (397 lines — SFT data carrying a "thinking" field
    crashed training; the surrounding try only caught JSONDecodeError).
    Verified: imports clean, ruff clean, full suite green, smoke forward ran
    both modalities through `forward_multimodal` (finite logits).
    UPDATE 2026-07-18: `core/reasoning.py` deleted again — this time WITH its
    only caller (the Forge trainer's SFT path) in the compression pass, so
    the dangling-import class this entry documents cannot recur for it.
    LEFT ALONE, deliberately: `core/sentiment.py` (+ its dep
    `core/model_context.py`, ~940 lines combined — would revive the emotional-state
    subsystem; user call). UPDATE 2026-07-17: its guarded rl_training caller was
    removed, so `tests/test_import_integrity.py`'s ALLOWED_MISSING is now EMPTY —
    the import gate is fully strict; restoring sentiment means restoring the module
    AND writing a new caller. (`core/inference.py`, deleted in the pivot `0eab02a3`,
    was dropped from the allowlist 2026-07-13: its only consumer was the Modkit-era
    `mods/codegen`, removed with the rest of modkit — nothing imports it now.)
    STILL ABSENT (honest gap): inference-side multimodal wiring at the
    token/tensor level -- no image/audio placeholder token in
    `chat_format.py`, and `generate()`/KV-cache never calls
    `forward_multimodal`. (Image input DOES reach `serve_enigma.py` as TEXT:
    `flatten_image_content` captions OpenAI `image_url` content into the
    `[image: ...]` marker under `--eyes` -- the caption path, not projector
    wiring.) The projections are untrained: `vision_hidden_size`
    / `audio_hidden_size` remain `None` in every shipped checkpoint; stage-1
    projector training (see `collect_vision_data.py`) is the next step.
    Audio pipeline status (collector, distill, align, the whisper-base
    teacher download) lives in `BACKLOG.md` section 4, step 6 -- this entry
    stays about the restored modules only.

11.5. **CLOSED 2026-07-26 -- every `torch.load` in the pipeline now pins
    `weights_only=True`.** The False pins in finetune/dpo/serve/align/bench/
    distill/_verify_ckpt were pre-torch-2.6 legacy (an audit first caught a
    False pin being ADDED to pretrain as "sibling alignment" -- a live
    security downgrade on torch 2.10, where True is the default those paths
    already ran under). Receipt before the flip: every artifact class loaded
    under True on the real files -- enigma_dpo/model.pth (with meta),
    enigma_pretrain_large/latest.pth (with optimizer state; the
    _verify_ckpt target), enigma_sft/model.pth, and all three
    enigma_vision_align checkpoints (with optimizer+scheduler state), and
    the vision distill --resume class too (the convergence audit found
    models/enigma_vision_distill/latest.pth on disk and loaded it clean
    under True -- the first receipt list wrongly claimed the class had no
    file). Only the AUDIO distill class has no file to receipt; it is
    written by the same tensors-only torch.save pattern, and a failure
    there is a loud UnpicklingError, not silence. Suite 789 green after the
    flip -- the fake-checkpoint fixtures ride the same loaders.

12. **Open findings from the 2026-07-19 compression-pass review** (25 verified;
    the checkpoint-safety subset AND the pre-align fix batch were FIXED same
    day in `vision_align.py`/`model_presets.py` — see
    `tests/test_encoder_persistence.py` + `tests/test_config_compat.py`.
    FIXED and test-pinned 2026-07-19 (round 3): validation mean is now
    token-weighted and an interrupted epoch is never ranked; the four lying
    knob defaults (schedule_type 'wsd', label_smoothing 0.05, bpe_dropout
    0.1, gradient_noise_eta 0.01) are inert and refused when set;
    `ForgeConfig.from_dict`'s known-set is derived from
    `dataclasses.fields()` with to_dict completeness pinned; the serial
    decode/verify/H2D costs got the pooled-overlap rework.
    ROUND 4 (the cleanup batch, same day) closed the remainder: the LR
    schedule is sized in optimizer steps and the loss log is
    boundary-gated (the accum>1 latents); load_checkpoint unwraps via
    `model_registry.get_state_dict` (empty-state-dict fallthrough dead);
    TrainingConfig slimmed to consumed + refused fields (~25 inert Forge
    knobs deleted; `TrainingConfig.from_dict` filters old blobs; to_dict
    is field-derived); the batch_size=0 auto-estimator retired (refusal
    points at `hardware_detection.recommend_training_batch_size`); the
    never-stepped fallback optimizer is lazy and old-checkpoint optimizer
    state is no longer materialized; train/val share one `_forward_ce`;
    one-allocation CPU mask; zero-caller `training/__init__` shim and the
    `--no-diff-attn` no-op removed; stale MoE/LoRA/speculative/MTP prose
    grounded; dead `total_tokens`/`dataset_fingerprint` checkpoint keys
    and the `_emit_loss` val_loss param removed. ENTRY CLOSED - the review
    ledger lives in BACKLOG 7.5 (its two deliberate deferrals included).
    - **Trainer aliases + mutates the caller's TrainingConfig** (batch_size=0
      sentinel overwritten in place; persisted into checkpoints as if
      user-chosen).
    - **Prep/loop caption thresholds disagree** (len<1 vs len<2): 1-token
      captions survive prep, are re-skipped every epoch, and inflate
      `dropped_short_captions` by the epoch factor.
    - **Square-mask crash latent -- CLOSED 2026-07-20 (Arc A trap fixes):**
      pairing an explicit `attention_mask` with a warm KV cache now raises a
      clear ValueError refusal in `forward` (and `forward_multimodal` refuses
      cached multi-token continuation, whose mask was start_pos-blind) instead
      of the latent broadcast crash. A real batched/padded cached decode path
      remains future work for the serving arc; the refusal makes the gap loud.
      Regression-pinned in `tests/test_v2_trap_guards.py` (also covers: KV
      cache cap now follows config.max_seq_len instead of the silent 4096
      clamp; vocab-alignment pad rows can no longer be sampled).
    - Efficiency/cleanup batch: see BACKLOG "2026-07-19 review".
