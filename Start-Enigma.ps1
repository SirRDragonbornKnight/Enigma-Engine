# Start-Enigma.ps1 -- bring up the Enigma OpenAI-compatible server (port 8000)
# if it isn't already running. Called by Desktop\Odysseus.bat (same pattern as
# Start-Chroma.ps1), by "Talk to Enigma.bat" (which passes -Voice), or run by
# hand.
# Serves the instruct checkpoint WITH her real long-term memory enabled
# (data\memory -- the remember tool writes here).
# ASCII-only output (Windows cp1252 console).
param(
    [switch]$Voice,  # enable the voice organ (speak tool + /v1/audio/speech)
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
        # A live server keeps the ORGANS it booted with -- -Voice cannot reach
        # it. Odysseus.bat starts her voiceless, so without this check a later
        # "Talk to Enigma" attaches to a mute server and reports success.
        if ($Voice) {
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
                Write-Output "WARN: Enigma $why -- organs cannot be added to a live server. Run Stop-Enigma.ps1, then start her again to hear her."
                Add-Type -AssemblyName System.Windows.Forms
                [System.Windows.Forms.MessageBox]::Show(
                    "Enigma $why, and organs cannot be added to a server that is already up. Her window will still open and she can chat in text -- she just will not speak. To hear her: run Stop-Enigma.ps1, then start her again from 'Talk to Enigma'.",
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
        "Enigma's serving environment was not found at $python -- recreate it with: python -m venv venv; venv\Scripts\pip install -e `".[voice,server]`" (from the Enigma Engine folder).",
        "Enigma") | Out-Null
    exit 1
}

# Hidden window: a visible console gets closed by accident (see CLAUDE.md
# training-ops gotcha). Logs go to serve_enigma.log in the repo.
$log = Join-Path $engineDir "serve_enigma.log"
# --eyes = HER OWN vision encoder + projection + this same model (~19M extra,
# no external captioner, no downloads); serve degrades to text-only with a
# WARN if the align checkpoint is missing, so passing it is always safe.
$serveArgs = @("serve_enigma.py", "--port", "$port", "--model", "models\enigma_dpo\model.pth", "--memory-dir", "data\memory", "--eyes")
if ($Voice) {
    $serveArgs += "--voice"
    if ($VoiceName) { $serveArgs += @("--voice-name", $VoiceName) }
}
Start-Process -FilePath $python `
    -ArgumentList $serveArgs `
    -WorkingDirectory $engineDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput $log `
    -RedirectStandardError (Join-Path $engineDir "serve_enigma.err.log")
Write-Output "Enigma starting on http://127.0.0.1:$port/v1 (log: serve_enigma.log)"
