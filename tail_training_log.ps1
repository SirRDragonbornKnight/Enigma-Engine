# Live view of the Enigma training log. Read-only; closing it does nothing to
# the run. Ctrl+C or close the window to stop watching.
$repo = $PSScriptRoot
$log = Join-Path $repo 'train_large.log'
$host.UI.RawUI.WindowTitle = 'Enigma Training Log'
Write-Host "Live tail of train_large.log  (Ctrl+C to close)" -ForegroundColor Cyan
Write-Host "------------------------------------------------" -ForegroundColor DarkGray
if (-not (Test-Path $log)) {
  Write-Host "No log file yet -- training hasn't been started." -ForegroundColor Yellow
  Start-Sleep 6
  exit 0
}
Get-Content $log -Tail 40 -Wait
