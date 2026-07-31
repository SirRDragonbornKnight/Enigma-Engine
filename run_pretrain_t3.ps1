# Detached launcher for T3 -- the full v2 pretrain (BACKLOG T3, Gate B = 238m).
#
# THIS SCRIPT DOES NOT DETACH ITSELF. Run it FROM a scheduled task, or the run
# dies with the shell that started it -- and a shell started from a Claude tool
# call takes the run down when the Claude app restarts (2026-06-27, step
# 220,190, no crash text).
#
# resume_training.ps1 cannot do this job: it passes no run math by design, so a
# lineage that has no checkpoint yet has nothing for it to restore. This script
# owns the FIRST launch only. Every later restart goes through resume_training,
# which reads the schedule back out of the checkpoint.
#
# The numbers below are v2c's, not v2b's, and three of them changed with the
# corpus:
#   --tokens-bin       tokens_v2c.bin -- 24.33B tokens, all 31 extents
#                      document-split (validated 2026-07-31)
#   --tokens 24.3e9    one epoch of v2c, a hair under so it never wraps.
#                      v2b's line said 28.3e9; v2c is smaller because dedup
#                      removed 18.1M duplicate paragraphs, not because data was
#                      lost.
#   --val-general-end  11,702,372,163 = the END OF FineWeb-Edu in v2c (v2b's
#                      15,055,680,259 was the same boundary in the old layout).
#                      Without it [val] is the corpus TAIL, which is
#                      SE/worldbuilding alone -- every val number would be
#                      speculative-fiction loss.
# --lr 3e-3 is MEASURED (T2 sweep winner, re-confirmed at 6dp on 30/30 source
# windows), not a placeholder. --micro-batch 6 and --grad-accum 16 (the
# default) are the shape the 24.5 GB / 4.9 d-per-epoch receipt was taken at.

$ErrorActionPreference = "Stop"
$repo = "C:\Users\SirKn\Enigma Engine"
$python = Join-Path $repo "venv\Scripts\python.exe"
$runName = "enigma_v2_238m"

# Same log name resume_training.ps1 would pick, so the "Enigma Training Log"
# shortcut finds this run and a later resume appends to the same file.
$log = Join-Path $repo ("train_{0}.log" -f $runName)
$errLog = Join-Path $repo ("train_{0}.err.log" -f $runName)
$done = Join-Path $repo ("train_{0}.done" -f $runName)

# --- guard: one hot job, one owner (ruled 2026-07-29) ---
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*pretrain_enigma*' }
if ($existing) {
    "REFUSED: pretrain already running (PID $($existing.ProcessId)) at $(Get-Date -Format o)" |
        Set-Content -Path $done -Encoding utf8
    exit 1
}
# A fresh launch must not land on top of an existing lineage: --out would adopt
# the directory and the first save would overwrite another run's checkpoint.
$outDir = Join-Path $repo "models\$runName"
if (Test-Path $outDir) {
    "REFUSED: $outDir already exists -- resume it with resume_training.ps1 -Run $runName, do not re-launch fresh" |
        Set-Content -Path $done -Encoding utf8
    exit 1
}

if (Test-Path $done) { Remove-Item $done -Force }
# -RedirectStandardOutput OVERWRITES; a previous attempt's log is the only
# record of why it died.
foreach ($f in @($log, $errLog)) {
    if (Test-Path $f) { Move-Item $f ($f + ".prev") -Force }
}

# ONE quoted string: -ArgumentList joins array elements without quoting, and
# both the interpreter path and this repo's path carry spaces.
$script = Join-Path $repo "pretrain_enigma.py"
$argString = "`"$script`" --size v2_deep_238m --optimizer muon " +
             "--schedule wsd_sqrt --sdpa-backend cudnn --no-grad-ckpt " +
             "--block 2048 --micro-batch 6 --tokens 24.3e9 --lr 3e-3 " +
             "--tokens-bin data/pretrain/tokens_v2c.bin " +
             "--val-general-end 11702372163 --val-per-source 2000000 " +
             "--out models/$runName --seed 1 --archive-every 1440"

# Block-buffered stdout surfaces in 8 KB clumps once redirected, and the log is
# the only per-step signal a detached run gives.
$env:PYTHONUNBUFFERED = "1"

$proc = Start-Process -FilePath $python -ArgumentList $argString `
    -WorkingDirectory $repo -NoNewWindow -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError $errLog
# Touch the handle WHILE THE CHILD IS ALIVE or ExitCode reads empty after exit
# under PowerShell 5.1 and the marker cannot tell finished from killed.
$null = $proc.Handle

# Priority stays NORMAL, unlike the corpus jobs. Those were CPU-bound walks
# competing with the user's desktop; this is GPU-bound with a data loader that
# must keep the card fed, and demoting it starves the loader for no gain.

$proc.WaitForExit()
"exit=$($proc.ExitCode) finished=$(Get-Date -Format o)" | Set-Content -Path $done -Encoding utf8
