# Start-Enigma.ps1 -- bring up the Enigma OpenAI-compatible server (port 8000)
# if it isn't already running. Called by Desktop\Odysseus.bat (same pattern as
# Start-Chroma.ps1), by "Talk to Enigma.bat" (voiceless -- user ruling
# 2026-07-16), or run by hand.
# Serves the instruct checkpoint WITH her real long-term memory enabled
# (data\memory -- the remember tool writes here).
# ASCII-only output (Windows cp1252 console).
param(
    [switch]$Voice,  # enable the voice organ (speak tool + /v1/audio/speech)
    # Which installed TTS voice she uses (name substring; see /v1/audio/voices).
    # "zira" = the female SAPI voice on this box; the user dislikes the David
    # default. Real fix is the Kokoro swap (BACKLOG section 5).
    [string]$VoiceName = "zira"
)

$engineDir = "C:\Users\SirKn\Enigma Engine"
$python = "C:\Users\SirKn\AppData\Local\Programs\Python\Python312\python.exe"
$port = 8000

$up = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($up) {
    # Verify the listener really is Enigma before claiming she's up.
    $ownerId = $up[0].OwningProcess
    $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$ownerId" -ErrorAction SilentlyContinue).CommandLine
    $procName = (Get-Process -Id $ownerId -ErrorAction SilentlyContinue).ProcessName
    if ($cmd -like "*serve_enigma.py*" -or $procName -in @("enigma", "enigma-ai")) {
        Write-Output "Enigma already serving on port $port (pid $ownerId)."
        exit 0
    }
    Write-Output "WARN: port $port is held by pid $ownerId ($procName), which is NOT Enigma -- not starting a second server."
    exit 1
}

if (-not (Test-Path $python)) {
    # Often launched hidden (tray/bat) -- a popup is the only failure anyone sees.
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show(
        "Python was not found at $python -- Enigma cannot start. Reinstall Python 3.12 from python.org.",
        "Enigma") | Out-Null
    exit 1
}

# Hidden window: a visible console gets closed by accident (see CLAUDE.md
# training-ops gotcha). Logs go to serve_enigma.log in the repo.
$log = Join-Path $engineDir "serve_enigma.log"
$serveArgs = @("serve_enigma.py", "--port", "$port", "--model", "models\enigma_dpo\model.pth", "--memory-dir", "data\memory")
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
