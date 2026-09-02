# Detached launcher for SFT-5: ONE finetune, no DPO half.
#
#   SFT-5: finetune from models/enigma_v2_238m_facts/model.pth -- the same
#   facts checkpoint SFT-2 came from -- on the current data/sft/mix.jsonl.
#   -> models/enigma_v2_sft5
#
# lr, epochs and seed are the finetune DEFAULTS on purpose (adamw 2e-5 cosine,
# 2 epochs, seed 1337). A launcher that re-states a default is a launcher that
# silently disagrees with it the day the default moves; the recipe lives in
# finetune_enigma.py's argparse, once.
#
# THIS SCRIPT DOES NOT DETACH ITSELF. Run it FROM a scheduled task, or the run
# dies with the shell that started it.
#
# -DryRun prints the exact command and exits BEFORE any guard: a dry run must
# start nothing and WRITE nothing, so what it prints cannot depend on whether
# training happens to be running or on whether the run already exists.

param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repo = "C:\Users\SirKn\Enigma Engine"
$python = Join-Path $repo "venv\Scripts\python.exe"

$sftRun = "enigma_v2_sft5"

$finetune = Join-Path $repo "finetune_enigma.py"
$sftArgs = "`"$finetune`" --data data/sft/mix.jsonl " +
           "--init models/enigma_v2_238m_facts/model.pth " +
           "--out models/$sftRun --block 2048 --micro-batch 4 --grad-accum 8"

if ($DryRun) {
    Write-Output "DRYRUN sft: `"$python`" $sftArgs"
    Write-Output ("DRYRUN logs: {0} | {1}" -f (Join-Path $repo ("train_{0}.log" -f $sftRun)),
                                              (Join-Path $repo ("train_{0}.err.log" -f $sftRun)))
    exit 0
}

# --- guard: one hot job, one owner (ruled 2026-07-29) ---
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*pretrain_enigma*' -or $_.CommandLine -like '*finetune_enigma*' -or $_.CommandLine -like '*dpo_enigma*' }
if ($existing) {
    "REFUSED: training already running (PID $($existing.ProcessId)) at $(Get-Date -Format o)" |
        Set-Content -Path (Join-Path $repo "train_sft5.refused") -Encoding utf8
    exit 1
}
if (Test-Path (Join-Path $repo "models\$sftRun")) {
    "REFUSED: models\$sftRun already exists -- this launcher owns FIRST launches only" |
        Set-Content -Path (Join-Path $repo "train_sft5.refused") -Encoding utf8
    exit 1
}

$env:PYTHONUNBUFFERED = "1"

function Invoke-Step {
    param([string]$Name, [string]$ArgString)
    $log = Join-Path $repo ("train_{0}.log" -f $Name)
    $err = Join-Path $repo ("train_{0}.err.log" -f $Name)
    $done = Join-Path $repo ("train_{0}.done" -f $Name)
    foreach ($f in @($log, $err)) { if (Test-Path $f) { Move-Item $f ($f + ".prev") -Force } }
    if (Test-Path $done) { Remove-Item $done -Force }
    $proc = Start-Process -FilePath $python -ArgumentList $ArgString `
        -WorkingDirectory $repo -NoNewWindow -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError $err
    $null = $proc.Handle
    $proc.WaitForExit()
    "exit=$($proc.ExitCode) finished=$(Get-Date -Format o)" | Set-Content -Path $done -Encoding utf8
    return $proc.ExitCode
}

$code = Invoke-Step -Name $sftRun -ArgString $sftArgs
exit $code
