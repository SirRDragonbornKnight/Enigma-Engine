# Live view of the Enigma training log. Read-only; closing it does nothing to
# the run. Ctrl+C or close the window to stop watching.
# Watches the NEWEST train_*.log by default (train_large.log for the 182M
# lineage, train_2048.log once Phase 4 runs); pass -Log to pick one by name.
param([string]$Log = "")
$repo = $PSScriptRoot
if ($Log) {
  $log = Join-Path $repo $Log
} else {
  $newest = Get-ChildItem -Path $repo -Filter 'train_*.log' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($newest) { $log = $newest.FullName } else { $log = Join-Path $repo 'train_large.log' }
}
$host.UI.RawUI.WindowTitle = 'Enigma Training Log'
Write-Host "Live tail of $(Split-Path $log -Leaf)  (Ctrl+C to close)" -ForegroundColor Cyan
Write-Host "------------------------------------------------" -ForegroundColor DarkGray
if (-not (Test-Path $log)) {
  Write-Host "No log file yet -- training hasn't been started." -ForegroundColor Yellow
  Start-Sleep 6
  exit 0
}
Get-Content $log -Tail 40 -Wait
