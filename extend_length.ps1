# Phase 4 length extension: warm-start a NEW run from the finished base and
# continue-pretrain at block 2048 (detached -- keeps running after this window
# closes). Same on/off workflow as resume_training.ps1.
#
# START is self-service (this script). STOP: ask Claude to stop it right after
# a checkpoint save (a "safe spot"); saves are atomic with a prev.pth backstop.
#
# ---------------------------------------------------------------------------
# DECISIONS baked in below -- review before firing (all measured/derived):
#   SOURCE  models/enigma_pretrain_large/model.pth  (the finished 182M base).
#           If you finish the enigma_pretrain_probe anneal first, repoint this
#           at its exported model.pth -- it is a lower-loss base.
#   BLOCK   2048 (doubles context). rope_theta is already 500000 in the base,
#           so no theta raise is needed -- only longer training windows.
#   GEOM    micro-batch 6 x grad-accum 16 = 196,608 tok/step. Probe-measured
#           ~46,900 tok/s and ~4.9 GB VRAM on the RTX 5090 (grad-ckpt off).
#   TOKENS  10e9 (the recipe ceiling; ~2.5 GPU-days at the measured rate).
#           5-8e9 (~1.5 GPU-days) likely suffices -- lower to eval sooner.
#   LR      3e-4 peak (half the base's 6e-4): converged weights on a new
#           sequence length want a gentler push than a fresh run.
#   SCHED   wsd + 300-step fresh warmup + decay-to-zero over the last 10%.
#   OUT     models/enigma_pretrain_2048 (NEW dir -- --init-from refuses to
#           write into the source model's directory).
# ---------------------------------------------------------------------------

$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
Set-Location $repo

# --- guard: don't launch a second copy (they would fight for the GPU + log) ---
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*pretrain_enigma*' }
if ($existing) {
  Write-Host ("Enigma training is ALREADY running (PID {0})." -f $existing.ProcessId) -ForegroundColor Yellow
  Write-Host "Not launching a second copy." -ForegroundColor Yellow
  Start-Sleep 6
  exit 0
}

# --- resolve the interpreter (PATH first, then the known install; no hardcoded user) ---
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe' }
if (-not (Test-Path $py)) {
  Write-Host ("Python interpreter not found ({0})." -f $py) -ForegroundColor Red
  Start-Sleep 10
  exit 1
}

# --- RESUME vs FRESH -----------------------------------------------------------
# If the 2048 run has already written a checkpoint, this is a re-run after a
# pause: RESUME it (step/lr/schedule restored from the checkpoint) rather than
# warm-starting from step 0 again -- a fresh --init-from would rotate the
# partially-trained checkpoints away and destroy days of progress. Only warm-
# start a brand-new run when no 2048 checkpoint exists yet.
$out2048 = Join-Path $repo 'models\enigma_pretrain_2048'
$hasCkpt = (Test-Path (Join-Path $out2048 'latest.pth')) -or (Test-Path (Join-Path $out2048 'prev.pth'))

if ($hasCkpt) {
  # Resume: NO --init-from (mutually exclusive with --resume). --out is explicit
  # so checkpoints keep landing in the 2048 dir. The run math is restored from
  # the checkpoint's recorded schedule; only operational knobs are passed here.
  $trainArgs = '--resume models/enigma_pretrain_2048/latest.pth --out models/enigma_pretrain_2048 --no-grad-ckpt --no-compile --archive-every 25000'
  $mode = 'Resuming the paused block-2048 run'
} else {
  # Fresh warm-start: the finished base must exist to copy weights from.
  $src = Join-Path $repo 'models\enigma_pretrain_large\model.pth'
  if (-not (Test-Path $src)) {
    Write-Host ("Base checkpoint not found: {0}" -f $src) -ForegroundColor Red
    Write-Host "Cannot warm-start (was the model moved?)." -ForegroundColor Red
    Start-Sleep 10
    exit 1
  }
  # --- the length-extension run math (see DECISIONS block above) ---
  $trainArgs = '--init-from models/enigma_pretrain_large/model.pth --out models/enigma_pretrain_2048 --block 2048 --micro-batch 6 --grad-accum 16 --tokens 10e9 --lr 3e-4 --warmup 300 --weight-decay 0.1 --grad-clip 1.0 --optimizer adamw --schedule wsd --wsd-decay-frac 0.1 --dropout 0.0 --no-diff-attn --no-grad-ckpt --no-compile --archive-every 25000'
  $mode = 'Starting Phase 4 length extension (block 2048, fresh warm-start)'
}

$inner = "/c `"$py`" -u pretrain_enigma.py $trainArgs >> train_2048.log 2>&1"

Write-Host ("{0} (detached)..." -f $mode) -ForegroundColor Cyan
Start-Process -FilePath 'cmd.exe' -ArgumentList $inner -WorkingDirectory $repo -WindowStyle Hidden

Start-Sleep 8
$now = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*pretrain_enigma*' }
if ($now) {
  Write-Host ("Started. python PID {0}. Logging to train_2048.log." -f $now.ProcessId) -ForegroundColor Green
} else {
  Write-Host "Process did not appear yet -- check train_2048.log for an error." -ForegroundColor Yellow
}
Write-Host "--- last log lines ---" -ForegroundColor DarkGray
$log = Join-Path $repo 'train_2048.log'
if (Test-Path $log) { Get-Content $log -Tail 8 }
Write-Host ""
Write-Host "To STOP: ask Claude to stop it at a safe checkpoint." -ForegroundColor Cyan
Write-Host "(This window can be closed; training keeps running.)" -ForegroundColor DarkGray
Start-Sleep 12
