# Start-Enigma.ps1 -- bring up the Enigma OpenAI-compatible server (port 8000)
# if it isn't already running. Called by Desktop\Odysseus.bat (same pattern as
# Start-Chroma.ps1), by "Talk to Enigma.bat" (which passes -Voice), or run by
# hand.
# Serves the instruct checkpoint WITH her real long-term memory enabled
# (data\memory -- the remember tool writes here).
# ASCII-only output (Windows cp1252 console).
param(
    # RETIRED 2026-07-27 (user ruling: "add them in so she is complete") --
    # every launch now boots ALL organs, so -Voice is accepted for the old
    # callers but changes nothing. Voice loaded does not mean voice talking:
    # speak is intent-gated and talk-mode persists separately.
    [switch]$Voice,
    # Optional single-preset override. EMPTY (default) = her saved Kokoro voice
    # recipe -- the Cortana blend in ~/.enigma_engine/voice.json. Only pass a
    # name to force one preset, e.g. -VoiceName af_bella (see /v1/audio/voices).
    [string]$VoiceName = ""
)

$engineDir = "C:\Users\SirKn\Enigma Engine"
# The SERVER runs the repo venv: it carries Kokoro (voice) on top of the same
# torch build as system Python, without pulling Kokoro's G2P stack into the
# pinned system ML environment. The window client (enigma_window.py) stays on
# system Python -- it needs no voice deps.
$python = "C:\Users\SirKn\Enigma Engine\venv\Scripts\python.exe"
$port = 8000

$up = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($up) {
    # Verify the listener really is Enigma before claiming she's up.
    $ownerId = $up[0].OwningProcess
    $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$ownerId" -ErrorAction SilentlyContinue).CommandLine
    $procName = (Get-Process -Id $ownerId -ErrorAction SilentlyContinue).ProcessName
    if ($cmd -like "*serve_enigma.py*" -or $procName -in @("enigma", "enigma-ai")) {
        Write-Output "Enigma already serving on port $port (pid $ownerId)."
        # A live server keeps the ORGANS it booted with -- flags cannot reach
        # it. Every launch is complete since 2026-07-27, so any live server
        # WITHOUT voice predates the ruling; warn unconditionally.
        if ($true) {
            $voiceState = "unknown"
            try {
                $s = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/audio/status" -TimeoutSec 3
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
                Write-Output "WARN: Enigma $why -- organs cannot be added to a live server. Run Stop-Enigma.ps1 and start her again; if this warning returns, the voice organ is failing at boot (check the server console for its WARN line)."
                Add-Type -AssemblyName System.Windows.Forms
                [System.Windows.Forms.MessageBox]::Show(
                    "Enigma $why, and organs cannot be added to a server that is already up. Her window will still open and she can chat in text -- she just will not speak. To hear her: run Stop-Enigma.ps1, then start her again from 'Talk to Enigma'. If this warning comes back after a restart, the voice organ itself is failing at boot -- check the server console for its WARN line.",
                    "Enigma") | Out-Null
            }
        }
        exit 0
    }
    Write-Output "WARN: port $port is held by pid $ownerId ($procName), which is NOT Enigma -- not starting a second server."
    # Usually launched hidden -- the refusal must be visible (2026-07-17 audit).
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show(
        "Port $port is in use by another program (pid $ownerId, $procName), so Enigma cannot start. Close that program and try again.",
        "Enigma") | Out-Null
    exit 1
}

if (-not (Test-Path $python)) {
    # Often launched hidden (tray/bat) -- a popup is the only failure anyone sees.
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show(
        "Enigma's serving environment was not found at $python -- recreate it with: python -m venv venv; venv\Scripts\pip install -e `".[server,voice,ears,eyes,imagegen]`" (from the Enigma Engine folder).",
        "Enigma") | Out-Null
    exit 1
}

# Hidden window: a visible console gets closed by accident (see CLAUDE.md
# training-ops gotcha). Logs go to serve_enigma.log in the repo.
$log = Join-Path $engineDir "serve_enigma.log"
# COMPLETE by ruling (2026-07-27): every organ, every launch -- the old
# organs-as-options split was a space saver that stopped earning its keep.
# Each organ WARNs and text serving continues if its backend is missing, so
# the flags are always safe; voice loaded stays SILENT until asked (speak is
# intent-gated; talk-mode persists separately in data\talk_mode.json).
$serveArgs = @("serve_enigma.py", "--port", "$port", "--model", "models\enigma_dpo\model.pth",
               "--memory-dir", "data\memory", "--eyes", "--ears", "--voice", "--image-gen")
if ($VoiceName) { $serveArgs += @("--voice-name", $VoiceName) }
Start-Process -FilePath $python `
    -ArgumentList $serveArgs `
    -WorkingDirectory $engineDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput $log `
    -RedirectStandardError (Join-Path $engineDir "serve_enigma.err.log")
Write-Output "Enigma starting on http://127.0.0.1:$port/v1 (log: serve_enigma.log)"
