# Resume Enigma pretraining (detached -- keeps running after this window closes).
#
# A pretrain checkpoint carries its own schedule (SCHEDULE_KEYS in
# pretrain_enigma.py): corpus, LR, block, optimizer, the grad-checkpoint flag
# and the archive cadence. A resume restores all of it, so this script passes
# no run math -- a flag list written here can only contradict the file, and
# restating one lineage's flags is how a resume continues the wrong model.
#
# Checkpoints written before 2026-07-28 predate two of those keys. What the
# file does not record, this script does not guess:
#   -TokensBin <path>   required when the checkpoint has no corpus recorded
#                       (otherwise the run would train --tokens-bin's default)
#   -NoGradCkpt         opt in to disabling activation checkpointing. Omitted,
#                       checkpointing stays ON: slower, but a lineage sized for
#                       it does not OOM on resume.
# The archive cadence is NOT settable here -- it is restored from the
# checkpoint, so a value passed on the command line would be ignored. Change it
# with a hand-run using --override-schedule and the full launch line.
#
# -Run picks the model directory. The default is the newest directory holding a
# RESUMABLE PRETRAIN checkpoint; vision/SFT/distill checkpoints live under
# models/ too and are not resumable here.
#
# START is self-service (this script / the desktop shortcut).
# STOP is intentionally NOT scripted here: ask Claude to stop it, so the kill
# lands right after a checkpoint save (a "safe spot"). If you ever must stop
# with no session open, just shutting the PC down is survivable -- saves are
# atomic with a prev.pth backstop.

param(
  [string]$Run = '',
  [string]$TokensBin = '',
  [switch]$NoGradCkpt
)

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

# --- resolve the interpreter (repo venv first: that is where the deps live) ---
$py = Join-Path $repo 'venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = (Get-Command python -ErrorAction SilentlyContinue).Source }
if (-not $py -or -not (Test-Path $py)) {
  Write-Host "Python interpreter not found (no repo venv, none on PATH)." -ForegroundColor Red
  exit 1
}

# Ask the checkpoint what it is. An SFT checkpoint has the same top-level shape
# as a pretrain one; what separates them is the schedule -- pretrain counts
# TOKENS, finetune counts EPOCHS over a --data file.
$inspect = @'
import sys, torch
try:
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
except Exception as exc:
    print("INVALID|" + type(exc).__name__); raise SystemExit(0)
if not (isinstance(ck, dict) and "model_state_dict" in ck and "config" in ck and "step" in ck):
    print("INVALID|not an Enigma checkpoint"); raise SystemExit(0)
sched = ck.get("schedule") or {}
if "epochs" in sched or "data" in sched:
    print("INVALID|finetune checkpoint, not pretrain"); raise SystemExit(0)
if "tokens" not in sched:
    print("INVALID|no pretrain schedule recorded"); raise SystemExit(0)
missing = [k for k in ("no_grad_ckpt", "tokens_bin") if k not in sched]
print("OK|%d|%s" % (ck.get("step", -1), ",".join(missing)))
'@
$inspectPy = Join-Path $env:TEMP 'enigma_inspect_ckpt.py'
Set-Content -Path $inspectPy -Value $inspect -Encoding utf8

function Test-Checkpoint($path) {
  # A stderr line from the child would terminate the script under
  # ErrorActionPreference=Stop, turning "skip this checkpoint" into "cannot
  # resume at all". A bad checkpoint must never be fatal to the search.
  try {
    $out = & $py $inspectPy "$path" 2>$null
  } catch {
    return $null
  }
  if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
  $parts = ($out | Select-Object -Last 1).Split('|')
  if ($parts[0] -ne 'OK') { return $null }
  return [pscustomobject]@{
    Path    = $path
    Step    = [int]$parts[1]
    Missing = if ($parts[2]) { $parts[2].Split(',') } else { @() }
  }
}

$modelsDir = Join-Path $repo 'models'
$ckinfo = $null
if ($Run) {
  $candidate = Join-Path $modelsDir (Join-Path $Run 'latest.pth')
  if (-not (Test-Path $candidate)) {
    Write-Host ("Checkpoint not found: {0}" -f $candidate) -ForegroundColor Red
    exit 1
  }
  $ckinfo = Test-Checkpoint $candidate
  if (-not $ckinfo) {
    Write-Host ("{0} is not a resumable pretrain checkpoint." -f $candidate) -ForegroundColor Red
    exit 1
  }
} else {
  Write-Host "Looking for the newest resumable pretrain checkpoint..." -ForegroundColor DarkGray
  $all = Get-ChildItem -Path $modelsDir -Filter 'latest.pth' -Recurse -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending
  $usable = @()
  foreach ($f in $all) {
    $info = Test-Checkpoint $f.FullName
    if (-not $info) {
      Write-Host ("   skipping {0} (not a pretrain checkpoint)" -f $f.Directory.Name) -ForegroundColor DarkGray
      continue
    }
    $usable += $info
    # Prefer one that records its corpus: without -TokensBin those are the only
    # ones that can resume unambiguously, so do not stop the search on a
    # checkpoint that would be refused two lines later.
    if (-not ($info.Missing -contains 'tokens_bin') -or $TokensBin) { $ckinfo = $info; break }
    Write-Host ("   skipping {0} (no corpus recorded; -TokensBin would be needed)" -f $f.Directory.Name) -ForegroundColor DarkGray
  }
  if (-not $ckinfo) {
    Write-Host ("No checkpoint under {0} can resume unambiguously." -f $modelsDir) -ForegroundColor Red
    if ($usable) {
      Write-Host "These are pretrain checkpoints but record no corpus:" -ForegroundColor Red
      foreach ($u in $usable) {
        Write-Host ("   {0,-28} step {1}" -f (Split-Path $u.Path -Parent | Split-Path -Leaf), $u.Step) -ForegroundColor Gray
      }
      Write-Host "Re-run with -Run <dir> -TokensBin data/pretrain/<corpus>.bin" -ForegroundColor Red
    }
    exit 1
  }
}

$runName = (Split-Path $ckinfo.Path -Parent | Split-Path -Leaf)
Write-Host ("Resuming lineage: {0}  (step {1})" -f $runName, $ckinfo.Step) -ForegroundColor Cyan
Write-Host ("Checkpoint:       {0}" -f $ckinfo.Path) -ForegroundColor Cyan

$trainArgs = "--resume `"$($ckinfo.Path)`""

if ($ckinfo.Missing -contains 'tokens_bin') {
  if (-not $TokensBin) {
    Write-Host ""
    Write-Host ("REFUSING: {0} records no corpus, so a resume would train whatever" -f $runName) -ForegroundColor Red
    Write-Host "--tokens-bin defaults to. Pass -TokensBin data/pretrain/<corpus>.bin" -ForegroundColor Red
    exit 1
  }
  $trainArgs += " --tokens-bin `"$TokensBin`""
  Write-Host ("  corpus:         {0} (from -TokensBin; not recorded in the checkpoint)" -f $TokensBin) -ForegroundColor Yellow
} else {
  Write-Host "  corpus:         from the checkpoint" -ForegroundColor DarkGray
  if ($TokensBin) {
    Write-Host "  -TokensBin IGNORED: the checkpoint records its corpus and wins." -ForegroundColor Yellow
  }
}

if ($ckinfo.Missing -contains 'no_grad_ckpt') {
  if ($NoGradCkpt) {
    $trainArgs += ' --no-grad-ckpt'
    Write-Host "  grad ckpt:      OFF (-NoGradCkpt; not recorded in the checkpoint)" -ForegroundColor Yellow
  } else {
    # Guessing here is the expensive mistake in both directions: forcing it off
    # can OOM a lineage sized with it on, forcing it on costs 30-40% throughput.
    # Default to the direction that cannot fail the run.
    Write-Host "  grad ckpt:      ON (not recorded in this checkpoint)." -ForegroundColor Yellow
    Write-Host "                  Pass -NoGradCkpt if this lineage trained without it." -ForegroundColor Yellow
  }
} else {
  Write-Host "  grad ckpt:      from the checkpoint" -ForegroundColor DarkGray
}
Write-Host "  archive-every:  from the checkpoint" -ForegroundColor DarkGray

$log = Join-Path $repo ("train_{0}.log" -f $runName)
# The whole command after /c carries ONE extra enclosing pair of quotes. cmd
# strips the outermost pair when the line holds several quoted tokens, and both
# the interpreter path and this repo contain spaces, so without the extra pair
# cmd receives a mangled line, launches nothing, and reports no error.
$inner = "/c `"`"$py`" -u pretrain_enigma.py $trainArgs >> `"$log`" 2>&1`""

Write-Host ""
Write-Host "Starting in 8s -- Ctrl+C now to pick a different lineage with -Run." -ForegroundColor Yellow
Start-Sleep 8

Write-Host "Resuming Enigma pretraining (detached)..." -ForegroundColor Cyan
Start-Process -FilePath 'cmd.exe' -ArgumentList $inner -WorkingDirectory $repo -WindowStyle Hidden

Start-Sleep 8
$now = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*pretrain_enigma*' }
if (-not $now) {
  # A resume that launched nothing must FAIL, not warn. Run from Task Scheduler
  # this script is invisible, and exiting 0 here reports a dead resume as a
  # successful one -- the days-long silence that follows looks like training.
  Write-Host ("FAILED: no pretrain process appeared. Check {0}." -f $log) -ForegroundColor Red
  Write-Host "--- last log lines ---" -ForegroundColor DarkGray
  if (Test-Path $log) { Get-Content $log -Tail 20 } else { Write-Host "(no log was written at all)" }
  # The desktop shortcut is the common caller and its window closes with the
  # script -- without this pause the FAILED banner is visible for ~0 ms.
  Start-Sleep 12
  exit 1
}
Write-Host ("Started. python PID {0}. Logging to {1}." -f $now.ProcessId, $log) -ForegroundColor Green
Write-Host "--- last log lines ---" -ForegroundColor DarkGray
if (Test-Path $log) { Get-Content $log -Tail 8 }
Write-Host ""
Write-Host "Confirm the banner shows the corpus, step and ckpt= you expect." -ForegroundColor Cyan
Write-Host "(the --size in the banner is a preset LABEL; the architecture comes" -ForegroundColor DarkGray
Write-Host " from the checkpoint config and may print a different preset name.)" -ForegroundColor DarkGray
Write-Host "To STOP: ask Claude to stop it at a safe checkpoint." -ForegroundColor Cyan
Start-Sleep 12
