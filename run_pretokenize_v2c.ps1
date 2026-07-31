# Detached launcher for the v2c retokenize (BACKLOG item 11 runbook step 4).
#
# THIS SCRIPT DOES NOT DETACH ITSELF. Run it FROM a scheduled task (same
# recipe as run_collect_rebuild.ps1's header) or the walk dies with your
# shell. The parent process is the throughput wall (walk + dedup + sanitize
# on one core), so BelowNormal on the PARENT matters more than on the
# workers, which self-set it.

$ErrorActionPreference = "Stop"
$repo = "C:\Users\SirKn\Enigma Engine"
$python = Join-Path $repo "venv\Scripts\python.exe"
$log = Join-Path $repo "data\pretokenize_v2c_2026-07-30.log"
$errLog = Join-Path $repo "data\pretokenize_v2c_2026-07-30.err.log"
$done = Join-Path $repo "data\pretokenize_v2c_2026-07-30.done"

if (Test-Path $done) { Remove-Item $done -Force }
# -RedirectStandardOutput OVERWRITES; the previous attempt's log is the only
# record of why it died.
foreach ($f in @($log, $errLog)) {
    if (Test-Path $f) { Move-Item $f ($f + ".prev") -Force }
}

# ONE quoted string: -ArgumentList joins array elements without quoting and
# both paths carry spaces. Output name is the NEXT free version -- an
# existing bin refuses at boot (versioned, never rebuilt in place).
$script = Join-Path $repo "pretokenize_data.py"
$vocab = Join-Path $repo "enigma_engine\vocab_model\bpe_vocab_v2_16k.json"
$argString = "`"$script`" --vocab `"$vocab`" --output-bin data/pretrain/tokens_v2c.bin " +
             "--dtype uint16 --workers 10 --repeat-sources curated=5"

# Block-buffered stdout surfaces in 8 KB clumps once redirected; the log is
# the only live signal a detached walk gives.
$env:PYTHONUNBUFFERED = "1"

$proc = Start-Process -FilePath $python -ArgumentList $argString `
    -WorkingDirectory $repo -NoNewWindow -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError $errLog
# Touch the handle WHILE THE CHILD IS ALIVE or ExitCode reads empty after
# exit under PowerShell 5.1 and the marker cannot tell finished from killed.
$null = $proc.Handle

# BelowNormal on the stub, then sweep the real interpreter it spawns (the
# venv python.exe is a redirector; priority is inherited at creation, so
# setting only the stub races the spawn).
try { $proc.PriorityClass = "BelowNormal" } catch { }
Start-Sleep -Seconds 3
try {
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$($proc.Id)" |
        ForEach-Object { (Get-Process -Id $_.ProcessId -ErrorAction Stop).PriorityClass = "BelowNormal" }
} catch { }

$proc.WaitForExit()
"exit=$($proc.ExitCode) finished=$(Get-Date -Format o)" | Set-Content -Path $done -Encoding utf8
