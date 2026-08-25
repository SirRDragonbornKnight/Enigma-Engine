# run_nightly_suite.ps1 -- nightly full-suite run with a hot-TRAINING guard.
#
# Two measured hazards shape this script (plan audit 2026-08-25):
# 1. NEVER combine ErrorActionPreference=Stop with stderr redirection on a
#    native exe -- PS 5.1 wraps the first stderr line in a terminating
#    NativeCommandError and kills the script mid-flight (measured THREW on
#    this box; same trap resume_training.ps1:81-86 documents). EAP stays
#    default here on purpose.
# 2. A GPU-busy check would skip EVERY night: the daily-posture serve keeps
#    the 238M resident on CUDA. Guard on TRAINING processes by command line
#    instead (the run_t5_sft2_dpo.ps1:26 idiom).
Set-Location $PSScriptRoot
$log = Join-Path $PSScriptRoot "logs\nightly_suite.log"
$stamp = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")

$hot = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match "pretrain_enigma|finetune_enigma|dpo_enigma|pretokenize_data|collect_pretraining" }
if ($hot) {
    Add-Content -Path $log -Value "$stamp SKIPPED (training hot: $(($hot | ForEach-Object ProcessId) -join ','))"
    exit 0
}
$serveUp = Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
    Where-Object { $_.CommandLine -match "serve_enigma" }
$note = if ($serveUp) { " [serve was live: conftest fingerprint false-positive possible]" } else { "" }

$out = & "$PSScriptRoot\venv\Scripts\python.exe" -m pytest tests -q 2>&1
$code = $LASTEXITCODE
$tail = (($out | Select-Object -Last 3) | ForEach-Object { "$_" }) -join " | "
$verdict = if ($code -eq 0) { "PASS" } else { "FAIL exit $code" }
Add-Content -Path $log -Value "$stamp $verdict$note -- $tail"
if ($code -ne 0) {
    ($out | Select-Object -Last 60) | ForEach-Object { "$_" } |
        Add-Content -Path (Join-Path $PSScriptRoot "logs\nightly_suite_last_fail.log")
}
exit $code
