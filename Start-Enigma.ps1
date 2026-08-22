# Start-Enigma.ps1 -- bring up the Enigma OpenAI-compatible server (port 8000)
# if it isn't already running. Called by Desktop\Odysseus.bat (same pattern as
# Start-Chroma.ps1), by "Talk to Enigma.bat" (which passes -Voice), or run by
# hand.
# Serves the instruct checkpoint WITH her real long-term memory enabled
# (data\memory -- the remember tool writes here).
# -Persona serves a DIFFERENT AI from a pack out of this same checkout, on her
# own port, out of her own memory store; with no -Persona every value below is
# the literal this script carried before it took parameters, which is what
# -DryRun exists to prove.
# ASCII-only output (Windows cp1252 console).
param(
    # RETIRED 2026-07-27 (user ruling: "add them in so she is complete") --
    # every launch now boots ALL organs, so -Voice is accepted for the old
    # callers but changes nothing. Voice loaded does not mean voice talking:
    # speak is intent-gated and talk-mode persists separately. (Offering it
    # every request was tried 2026-08-20 and reverted -- measured worse.)
    [switch]$Voice,
    # Optional single-preset override. EMPTY (default) = her saved Kokoro voice
    # recipe -- the Cortana blend in ~/.enigma_engine/voice.json. Only pass a
    # name to force one preset, e.g. -VoiceName af_bella (see /v1/audio/voices).
    [string]$VoiceName = "",
    # A persona pack DIRECTORY (pack.json beside its content files). Empty is
    # Enigma -- this repo is hers, and a pack is the other case.
    [string]$Persona = "",
    # Override the port. 0 = the pack's own `port` field, or 8000 for Enigma.
    [int]$Port = 0,
    # Print the serve command line this launch would run, and start nothing.
    # Prints BEFORE the port is probed: a dry run must describe the launch it
    # would make, not whatever happens to be listening while it runs.
    [switch]$DryRun
)

$engineDir = "C:\Users\SirKn\Enigma Engine"
# The SERVER runs the repo venv: it carries Kokoro (voice) on top of the same
# torch build as system Python, without pulling Kokoro's G2P stack into the
# pinned system ML environment. The window client (enigma_window.py) stays on
# system Python -- it needs no voice deps.
$python = "C:\Users\SirKn\Enigma Engine\venv\Scripts\python.exe"

# WHO this script serves, and everything derived from her. A persona pack
# serves a DIFFERENT AI out of this same checkout, so "is the process
# serve_enigma.py" answers the wrong question -- the name the server reports
# is the one that decides ownership.
. (Join-Path $PSScriptRoot "Enigma-Persona.ps1")
$self = Resolve-EnigmaPersona -EngineDir $engineDir -PackDir $Persona -Port $Port
$aiName = $self.Name
$bindPort = $self.Port

# Hidden window: a visible console gets closed by accident (see CLAUDE.md
# training-ops gotcha). Logs go to serve_<slug>.log in the repo -- one pair per
# AI, so a second server's boot errors do not land in hers.
$log = $self.Log
# COMPLETE by ruling (2026-07-27): every organ, every launch -- the old
# organs-as-options split was a space saver that stopped earning its keep.
# Each organ WARNs and text serving continues if its backend is missing, so
# the flags are always safe; voice loaded stays SILENT until asked (speak is
# intent-gated; talk-mode persists separately in data\talk_mode.json).
# ADOPTED 2026-08-09 (Gate D): the v2 lineage's SFT-2 checkpoint. It is the
# first candidate the sealed gate could distinguish from v8 (67/120 vs 56,
# paired p=0.0433) and wins or ties every category but factual. The v8
# checkpoint stays at models\enigma_dpo\model.pth as the rollback (byte-
# identical to Enigma Backups\enigma_dpo_v8_adopted). --max-context 2048 is
# NOT optional here: this model trained at block 2048 and serving it at the
# 1024 default silently discards the context the whole v2 lineage was for.
#
# ONE PRE-QUOTED STRING, not an array: PS 5.1 -ArgumentList joins an array
# with spaces and quotes NOTHING, so the first pack path or memory dir
# carrying a space reaches python split in half (trap 1 of the four detach
# traps, pinned in tests/test_repo_hygiene.py). The script path stays
# repo-relative because -WorkingDirectory below is the engine dir.
$argString = "`"serve_enigma.py`" --port $bindPort --model models\enigma_v2_sft2\model.pth" `
    + " --max-context 2048 --memory-dir `"$($self.MemoryDir)`" --eyes --ears --voice --image-gen --search"
if ($VoiceName) { $argString += " --voice-name `"$VoiceName`"" }
if ($Persona) { $argString += " --persona `"$Persona`"" }

if ($DryRun) {
    Write-Output "DRYRUN serve: `"$python`" $argString"
    Write-Output "DRYRUN logs: $log | $($self.ErrLog)"
    Write-Output "DRYRUN start-lock: $($self.StartMutex)"
    exit 0
}

# Checking the port and spawning the server is ONE act. Between the two a
# second launcher -- a double-clicked shortcut, the tray's Talk while the .bat
# is still running -- saw a free port and started a SECOND multi-GB load that
# only discovered the collision when it tried to bind, minutes later. The tray
# already serializes itself on a named mutex; this is the same guard around the
# launch, under a name of its own so the tray and the launcher do not block
# each other, and per-AI so a pack starting is not Enigma starting.
$startLock = New-Object System.Threading.Mutex($false, $self.StartMutex)
try {
    $held = $startLock.WaitOne(0)
} catch [System.Threading.AbandonedMutexException] {
    # The previous holder died mid-launch without releasing. WaitOne hands the
    # mutex over anyway -- take it rather than refusing a launch forever.
    $held = $true
}
if (-not $held) {
    Write-Output "$aiName is already starting (another launcher holds the start lock) -- not starting a second server."
    exit 0
}

try {
    $up = Get-NetTCPConnection -LocalPort $bindPort -State Listen -ErrorAction SilentlyContinue
    if ($up) {
        # Verify the listener really is this AI before claiming she's up.
        $ownerId = $up[0].OwningProcess
        $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$ownerId" -ErrorAction SilentlyContinue).CommandLine
        $procName = (Get-Process -Id $ownerId -ErrorAction SilentlyContinue).ProcessName
        $serving = Get-ServingPersona -Port $bindPort
        if ($serving) {
            # -ceq, not -eq: PowerShell compares strings case-insensitively, and a
            # pack named "enigma" is a different AI wearing her spelling.
            $ours = ($serving -ceq $aiName)
        } else {
            $ours = ($cmd -like "*serve_enigma.py*" -or $procName -in @("enigma", "enigma-ai"))
        }
        if ($ours) {
            Write-Output "$aiName already serving on port $bindPort (pid $ownerId)."
            # A live server keeps the ORGANS it booted with -- flags cannot reach
            # it. Every launch is complete since 2026-07-27, so any live server
            # WITHOUT voice predates the ruling; warn unconditionally.
            $voiceState = "unknown"
            try {
                $s = Invoke-RestMethod -Uri "$($self.BaseUrl)/v1/audio/status" -TimeoutSec 3
                # An older server build may omit the key; $s.voice would be $null
                # and the -eq tests below would silently skip the warn.
                if ($null -ne $s.voice) { $voiceState = $s.voice }
            } catch {
                $voiceState = "unknown"
            }
            if ($voiceState -eq "off" -or $voiceState -eq "error") {
                if ($voiceState -eq "off") {
                    $why = "is already running WITHOUT her voice"
                } else {
                    $why = "is already running but her voice organ is in an error state"
                }
                # "off" cannot distinguish an older voiceless launch from a
                # voice organ that FAILED at boot -- a restart fixes the first
                # and repeats the second, so the advice must cover both.
                Write-Output "WARN: $aiName $why -- organs cannot be added to a live server. Run Stop-Enigma.ps1 and start her again; if this warning returns, the voice organ is failing at boot (check the server console for its WARN line)."
                Add-Type -AssemblyName System.Windows.Forms
                [System.Windows.Forms.MessageBox]::Show(
                    "$aiName $why, and organs cannot be added to a server that is already up. Her window will still open and she can chat in text -- she just will not speak. To hear her: run Stop-Enigma.ps1, then start her again from 'Talk to Enigma'. If this warning comes back after a restart, the voice organ itself is failing at boot -- check the server console for its WARN line.",
                    $aiName) | Out-Null
            }
            exit 0
        }
        # Name the holder as precisely as it can be named: another AI answers for
        # itself, and anything that cannot is still described by its process.
        if ($serving) {
            $who = "$serving (pid $ownerId)"
        } else {
            $who = "pid $ownerId ($procName)"
        }
        Write-Output "WARN: port $bindPort is held by $who, which is NOT $aiName -- not starting a second server."
        # Usually launched hidden -- the refusal must be visible (2026-07-17 audit).
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show(
            "Port $bindPort is in use by $who, so $aiName cannot start. Close that program and try again.",
            $aiName) | Out-Null
        exit 1
    }

    if (-not (Test-Path $python)) {
        # Often launched hidden (tray/bat) -- a popup is the only failure anyone sees.
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show(
            "$($aiName)'s serving environment was not found at $python -- recreate it with: python -m venv venv; venv\Scripts\pip install -e `".[server,voice,ears,eyes,imagegen]`" (from the Enigma Engine folder).",
            $aiName) | Out-Null
        exit 1
    }

    Start-Process -FilePath $python `
        -ArgumentList $argString `
        -WorkingDirectory $engineDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $log `
        -RedirectStandardError $self.ErrLog
    Write-Output "$aiName starting on $($self.BaseUrl)/v1 (log: serve_$($self.Slug).log)"
} finally {
    # Held across the whole check-and-spawn, released however this exits --
    # PowerShell runs finally on `exit` too, so the next launcher is not left
    # waiting on a lock nobody holds.
    $startLock.ReleaseMutex()
    $startLock.Dispose()
}
