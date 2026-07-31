# Detached launcher for the v2c corpus re-collect (BACKLOG item 11 runbook step 3).
#
# THIS SCRIPT DOES NOT DETACH ITSELF. Detachment comes from running it FROM a
# scheduled task -- launched directly, the pull is a child of your shell and
# dies with it. Register + fire (once; a far-future -Once trigger never
# refires):
#   $a = New-ScheduledTaskAction -Execute powershell.exe -Argument
#     "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File <this file>"
#   Register-ScheduledTask EnigmaCollectRebuild -Action $a -Trigger
#     (New-ScheduledTaskTrigger -Once -At (Get-Date).AddYears(10)) -Principal
#     (New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME"
#       -LogonType Interactive -RunLevel Limited)
#   Start-ScheduledTask EnigmaCollectRebuild
# Then verify the python parent chain ends at svchost, not your shell.
#
# The launcher waits on the child and writes a .done marker carrying the exit
# code. A marker means the child EXITED; no marker means running, killed, or
# timed out (the task's ExecutionTimeLimit kills the whole tree with no
# marker) -- absence carries no information.
#
# --resume is REQUIRED: without it the collector resets progress.json to
# {gutenberg_ids: [], stats: {}} and drops fandom_done_wikis, forcing a full
# Fandom re-pull. The four rebuilt sources start fresh regardless, because
# their resume keys were stripped and their directories emptied.
#
# --no-combine is REQUIRED: the combine step writes a ~95 GB combined.txt that
# pretokenize does not read (its own docstring says so) and that section 9
# ruled dead.
#
# Targets are GiB (the collector multiplies by 1024^3) and reproduce the v2b
# diet: fineweb_edu 43.14 GB / dclm 16.28 / finemath 10.93 / the_stack 11.02
# as measured on disk 2026-07-30, before the clearing move.

$ErrorActionPreference = "Stop"
$repo = "C:\Users\SirKn\Enigma Engine"
$python = Join-Path $repo "venv\Scripts\python.exe"
$log = Join-Path $repo "data\collect_rebuild_2026-07-30.log"
$errLog = Join-Path $repo "data\collect_rebuild_2026-07-30.err.log"
$done = Join-Path $repo "data\collect_rebuild_2026-07-30.done"

if (Test-Path $done) { Remove-Item $done -Force }
# -RedirectStandardOutput OVERWRITES. On a relaunch the previous attempt's log
# is the only record of why it died -- preserve it, don't destroy it.
foreach ($f in @($log, $errLog)) {
    if (Test-Path $f) {
        Move-Item $f ($f + ".prev") -Force
    }
}

# ONE string, not an array: -ArgumentList joins array elements with spaces
# WITHOUT quoting them, so the space in "Enigma Engine" splits the script path
# and python reports "can't open file 'C:\Users\SirKn\Enigma'".
$script = Join-Path $repo "collect_pretraining_data.py"
$argString = "`"$script`" --resume --no-combine --fineweb 40 --dclm 15 --finemath 10 --code 10"

# Unbuffered: redirected to a file, python block-buffers stdout at 8 KB, so
# the 60-second progress lines would surface in ~80-minute clumps and a
# network stall would be indistinguishable from a quiet buffer for over an
# hour. On a detached multi-hour pull the log IS the diagnostic.
$env:PYTHONUNBUFFERED = "1"

$proc = Start-Process -FilePath $python -ArgumentList $argString `
    -WorkingDirectory $repo -NoNewWindow -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError $errLog
# Touch the handle WHILE THE CHILD IS ALIVE or ExitCode reads empty after
# exit under PowerShell 5.1 -- the marker then says "exit=" and a finished
# run is indistinguishable from a killed one, which is the marker's one job.
# (Measured: without this, exit=[]; with it, a child's exit 7 reads 7.)
$null = $proc.Handle

# BelowNormal for the Chrome Remote Desktop budget: the session is driven
# remotely with SMT off, so a foreground-priority job makes the desktop crawl.
# The venv python.exe is a REDIRECTOR STUB that spawns the base interpreter as
# a child within milliseconds; priority is inherited at creation, so setting
# only the stub races the spawn. Set the stub, then sweep its children.
try { $proc.PriorityClass = "BelowNormal" } catch { }
Start-Sleep -Seconds 3
try {
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$($proc.Id)" |
        ForEach-Object { (Get-Process -Id $_.ProcessId -ErrorAction Stop).PriorityClass = "BelowNormal" }
} catch { }

# WaitForExit() on the object, not Wait-Process: Wait-Process leaves the
# process object's ExitCode unpopulated, so the marker recorded "exit=".
$proc.WaitForExit()
"exit=$($proc.ExitCode) finished=$(Get-Date -Format o)" |
    Set-Content -Path $done -Encoding utf8
