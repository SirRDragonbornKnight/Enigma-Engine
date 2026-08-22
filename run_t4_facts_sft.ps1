# Detached launcher for T4's two GPU steps, in order (BACKLOG T4, scope ruled
# 2026-08-08):
#   1. facts continued-pretrain -- install the widened knowledge corpus in
#      weights (SFT surfaces facts, it cannot install them). 60M tokens over
#      facts_tokens_v2.bin per make_facts_pretrain_data's doctrine line
#      (--lr 1e-4 --warmup 50 --val-general-end 0), muon because that is the
#      optimizer this lineage's weights were shaped under.
#   2. SFT finetune on data/sft/mix.jsonl (125,523 records, baked 2026-08-08)
#      from step 1's model.pth. The known-good SFT recipe (adamw 2e-5 cosine,
#      2 epochs); --block 2048 EXPLICIT -- it overrode the then-default 1024,
#      which silently skipped over-length records. finetune's OWN default has
#      been 2048 since the SUNSET flip (finetune_enigma.py --block; this line
#      corrected 2026-08-22), so the flag now only restates it, and is kept so
#      the block is stated where the run is read. micro-batch 4 x grad-accum 8
#      keeps the effective batch at the default 32 with half the per-forward
#      VRAM at the doubled block.
#
# THIS SCRIPT DOES NOT DETACH ITSELF. Run it FROM a scheduled task, or the
# runs die with the shell that started them (2026-06-27, step 220,190).
# Step 2 runs only if step 1 exits 0; each step writes its own .done marker.

$ErrorActionPreference = "Stop"
$repo = "C:\Users\SirKn\Enigma Engine"
$python = Join-Path $repo "venv\Scripts\python.exe"

$factsRun = "enigma_v2_238m_facts"
$sftRun = "enigma_v2_sft"

# --- guard: one hot job, one owner (ruled 2026-07-29) ---
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*pretrain_enigma*' -or $_.CommandLine -like '*finetune_enigma*' }
$doneGuard = Join-Path $repo "train_t4.refused"
if ($existing) {
    "REFUSED: training already running (PID $($existing.ProcessId)) at $(Get-Date -Format o)" |
        Set-Content -Path $doneGuard -Encoding utf8
    exit 1
}
foreach ($d in @((Join-Path $repo "models\$factsRun"), (Join-Path $repo "models\$sftRun"))) {
    if (Test-Path $d) {
        "REFUSED: $d already exists -- this launcher owns FIRST launches only" |
            Set-Content -Path $doneGuard -Encoding utf8
        exit 1
    }
}

# Block-buffered stdout surfaces in 8 KB clumps once redirected, and the logs
# are the only per-step signal a detached run gives.
$env:PYTHONUNBUFFERED = "1"

function Invoke-Step {
    param([string]$Name, [string]$ArgString)
    $log = Join-Path $repo ("train_{0}.log" -f $Name)
    $err = Join-Path $repo ("train_{0}.err.log" -f $Name)
    $done = Join-Path $repo ("train_{0}.done" -f $Name)
    foreach ($f in @($log, $err)) {
        if (Test-Path $f) { Move-Item $f ($f + ".prev") -Force }
    }
    if (Test-Path $done) { Remove-Item $done -Force }
    $proc = Start-Process -FilePath $python -ArgumentList $ArgString `
        -WorkingDirectory $repo -NoNewWindow -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError $err
    # Touch the handle WHILE THE CHILD IS ALIVE or ExitCode reads empty after
    # exit under PowerShell 5.1.
    $null = $proc.Handle
    $proc.WaitForExit()
    "exit=$($proc.ExitCode) finished=$(Get-Date -Format o)" | Set-Content -Path $done -Encoding utf8
    return $proc.ExitCode
}

# ONE quoted string per step: -ArgumentList joins array elements without
# quoting, and both the interpreter path and this repo's path carry spaces.
$pretrain = Join-Path $repo "pretrain_enigma.py"
$factsArgs = "`"$pretrain`" --size v2_deep_238m --optimizer muon " +
             "--sdpa-backend cudnn --no-grad-ckpt " +
             "--block 2048 --micro-batch 6 --tokens 60e6 --lr 1e-4 --warmup 50 " +
             "--tokens-bin data/pretrain/facts_tokens_v2.bin " +
             "--val-general-end 0 " +
             "--init-from models/enigma_v2_238m/model.pth " +
             "--out models/$factsRun --seed 1"
$code = Invoke-Step -Name $factsRun -ArgString $factsArgs
if ($code -ne 0) {
    "ABORT: facts step exited $code -- SFT not launched, $(Get-Date -Format o)" |
        Set-Content -Path (Join-Path $repo "train_t4.aborted") -Encoding utf8
    exit $code
}

$finetune = Join-Path $repo "finetune_enigma.py"
$sftArgs = "`"$finetune`" --data data/sft/mix.jsonl " +
           "--init models/$factsRun/model.pth " +
           "--out models/$sftRun --block 2048 --micro-batch 4 --grad-accum 8"
$code = Invoke-Step -Name $sftRun -ArgString $sftArgs
exit $code
