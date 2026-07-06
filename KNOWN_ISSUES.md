# Known Issues — current as of 2026-07-06

_Navigation layer over `SUGGESTIONS.md` (strategy), `CODE_REVIEW.md` (bugs),
`CLEANUP_TRACKER.md` (file state)._

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
   proven launch (same flags + `--no-diff-attn --no-grad-ckpt --archive-every
   25000`) is safest and matches the recorded schedule. ETA from here ≈ 10½
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
     int32 fence-redraw bug (fixed, see CODE_REVIEW.md); attempt 2 ran healthy
     ~8 h overnight (51k → 58.5k, ~50–52k tok/s, val ppl steady 3.8) until this
     pause. Detached `cmd /c` → `train_large.log`, survives Claude sessions.
   - (a) **torch.compile fell back to eager** — triton-windows lives in Claude
     Desktop's MSIX-virtualized AppData, invisible to this out-of-sandbox
     process. No cost: compile was measured negligible (06-06) and the run hits
     the same ~52k tok/s; the is_causal SDPA fast path is in the model code.
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
6. **Vendored weight:** `enigma_engine/bin/llama-server/` is ~1.06 GB of CUDA
   DLLs for the GGUF route — intentional (5090/sm_120 needs newer CUDA than
   wheel-based llama-cpp). Tree-kill `llama-server.exe` on teardown.
7. **Environment quirks (this dev box):** MCP servers load ONLY from the
   project `.mcp.json`; Claude Desktop is MSIX-sandboxed so `%LOCALAPPDATA%`
   writes are virtualized; no Windows admin without explicit user grant.
8. **Tokenizer facts:** token 44 (space) ≈ 26.6% of corpus tokens — baked-in
   inefficiency, not a bug. `encode()` brackets text as `[BOS]…[EOS]` — strip
   the trailing EOS before generation or the model sees a finished document
   (`sample_enigma.py` and `serve_enigma.py` both do this).
9. **The python suite is engine-only** (418 tests as of 2026-07-06). The
   avatar lives in its own repo (`C:\Users\SirKn\Enigma Avatar\`) with its
   own Node suite — run `node --test` there.
10. **Model capacity ceilings are measured, not guessed.** `PHASE7_GATE.md`
    holds the receipts: long conversation (block 1024), broad-fact recall
    (~50%), raw arithmetic (bypassed via the server-side `calculate` tool).
    Current SFT data state lives in `ROADMAP.md` (Phase 1 DONE, Phase 2
    exit criteria met 26/29).
