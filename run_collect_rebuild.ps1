# Detached launcher for the v2c corpus re-collect (BACKLOG item 11 runbook step 3).
#
# Runs collect_pretraining_data.py OUTSIDE the Claude process tree via Task
# Scheduler, so restarting the Claude app cannot kill a multi-hour pull. The
# launcher waits on the child and writes a .done marker carrying the exit code,
# so a finished run is distinguishable from a killed one.
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

# BelowNormal for the Chrome Remote Desktop budget: the session is driven
# remotely with SMT off, so a foreground-priority job makes the desktop crawl.
try { $proc.PriorityClass = "BelowNormal" } catch { }

# WaitForExit() on the object, not Wait-Process: Wait-Process leaves the
# process object's ExitCode unpopulated, so the marker recorded "exit=".
$proc.WaitForExit()
"exit=$($proc.ExitCode) finished=$(Get-Date -Format o)" |
    Set-Content -Path $done -Encoding utf8
