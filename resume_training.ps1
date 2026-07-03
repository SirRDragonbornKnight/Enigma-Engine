# Resume Enigma pretraining (detached -- keeps running after this window closes).
# Built for the daily on/off workflow (2026-06-12).
#
# START is self-service (this script / the desktop shortcut).
# STOP is intentionally NOT scripted here: ask Claude to stop it, so the kill
# lands right after a checkpoint save (a "safe spot"). If you ever must stop
# with no session open, just shutting the PC down is survivable -- saves are
# atomic with a prev.pth backstop; worst case is <=16 min of re-done steps.

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

# --- guard: checkpoint must exist to resume from ---
$ckpt = Join-Path $repo 'models\enigma_pretrain_large\latest.pth'
if (-not (Test-Path $ckpt)) {
  Write-Host ("Checkpoint not found: {0}" -f $ckpt) -ForegroundColor Red
  Write-Host "Cannot resume (was the model moved?)." -ForegroundColor Red
  Start-Sleep 10
  exit 1
}

# --- resolve the interpreter (PATH first, then the known install; no hardcoded user) ---
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe' }
if (-not (Test-Path $py)) {
  Write-Host ("Python interpreter not found ({0})." -f $py) -ForegroundColor Red
  Start-Sleep 10
  exit 1
}

# --- the proven run math (matches the schedule now recorded in the checkpoint) ---
$trainArgs = '--size large --tokens 56.6e9 --lr 6e-4 --warmup 200 --micro-batch 12 --grad-accum 16 --block 1024 --dropout 0.0 --weight-decay 0.1 --grad-clip 1.0 --optimizer adamw --schedule cosine --no-diff-attn --no-grad-ckpt --resume models/enigma_pretrain_large/latest.pth --archive-every 25000'
$inner = "/c `"$py`" -u pretrain_enigma.py $trainArgs >> train_large.log 2>&1"

Write-Host "Resuming Enigma pretraining (detached)..." -ForegroundColor Cyan
Start-Process -FilePath 'cmd.exe' -ArgumentList $inner -WorkingDirectory $repo -WindowStyle Hidden

Start-Sleep 8
$now = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*pretrain_enigma*' }
if ($now) {
  Write-Host ("Started. python PID {0}. Logging to train_large.log." -f $now.ProcessId) -ForegroundColor Green
} else {
  Write-Host "Process did not appear yet -- check train_large.log for an error." -ForegroundColor Yellow
}
Write-Host "--- last log lines ---" -ForegroundColor DarkGray
$log = Join-Path $repo 'train_large.log'
if (Test-Path $log) { Get-Content $log -Tail 6 }
Write-Host ""
Write-Host "To STOP: ask Claude to stop it at a safe checkpoint." -ForegroundColor Cyan
Write-Host "(This window can be closed; training keeps running.)" -ForegroundColor DarkGray
Start-Sleep 12
