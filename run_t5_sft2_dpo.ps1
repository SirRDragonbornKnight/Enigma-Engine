# Detached launcher for the fix-arc re-SFT then T5 DPO, in order.
#
#   1. SFT-2: re-finetune from the SAME facts checkpoint (models/
#      enigma_v2_238m_facts/model.pth -- unchanged, the fix arc did not touch
#      knowledge_corpus.py) on the patched mix (126,025 records as the mix
#      stood for the 2026-08-08 SFT-2 build -- a dated number, not a check:
#      every make_sft_data rebuild moves it. Contents: power/percent
#      calc surfaces, remember-poisoning restraint, client-tools restraint,
#      widened memory-declines). -> models/enigma_v2_sft2
#   2. T5 DPO from SFT-2 on dpo_pairs.jsonl (350 pairs incl. the new
#      decline-vs-fabricate and injection-resistance classes). THE SAFE RECIPE:
#      lr 5e-7 x 1 epoch -- 2e-6 damaged her (memory, do not raise). block 2048
#      explicit to match how she was trained and served. -> models/enigma_v2_dpo
#
# THIS SCRIPT DOES NOT DETACH ITSELF. Run it FROM a scheduled task, or the runs
# die with the shell that started them. Step 2 runs only if step 1 exits 0.

$ErrorActionPreference = "Stop"
$repo = "C:\Users\SirKn\Enigma Engine"
$python = Join-Path $repo "venv\Scripts\python.exe"

$sftRun = "enigma_v2_sft2"
$dpoRun = "enigma_v2_dpo"

# --- guard: one hot job, one owner (ruled 2026-07-29) ---
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*pretrain_enigma*' -or $_.CommandLine -like '*finetune_enigma*' -or $_.CommandLine -like '*dpo_enigma*' }
if ($existing) {
    "REFUSED: training already running (PID $($existing.ProcessId)) at $(Get-Date -Format o)" |
        Set-Content -Path (Join-Path $repo "train_t5.refused") -Encoding utf8
    exit 1
}
foreach ($d in @((Join-Path $repo "models\$sftRun"), (Join-Path $repo "models\$dpoRun"))) {
    if (Test-Path $d) {
        "REFUSED: $d already exists -- this launcher owns FIRST launches only" |
            Set-Content -Path (Join-Path $repo "train_t5.refused") -Encoding utf8
        exit 1
    }
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

$finetune = Join-Path $repo "finetune_enigma.py"
$sftArgs = "`"$finetune`" --data data/sft/mix.jsonl " +
           "--init models/enigma_v2_238m_facts/model.pth " +
           "--out models/$sftRun --block 2048 --micro-batch 4 --grad-accum 8"
$code = Invoke-Step -Name $sftRun -ArgString $sftArgs
if ($code -ne 0) {
    "ABORT: SFT-2 exited $code -- DPO not launched, $(Get-Date -Format o)" |
        Set-Content -Path (Join-Path $repo "train_t5.aborted") -Encoding utf8
    exit $code
}

# Safe DPO recipe: lr 5e-7 (the default, and the MEASURED-safe value; 2e-6
# damaged her) x 1 epoch. block 2048 explicit.
$dpo = Join-Path $repo "dpo_enigma.py"
$dpoArgs = "`"$dpo`" --data data/sft/dpo_pairs.jsonl " +
           "--init models/$sftRun/model.pth " +
           "--out models/$dpoRun --lr 5e-7 --epochs 1 --block 2048"
$code = Invoke-Step -Name $dpoRun -ArgString $dpoArgs
exit $code
