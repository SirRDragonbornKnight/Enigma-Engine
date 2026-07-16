# Start-Enigma.ps1 -- bring up the Enigma OpenAI-compatible server (port 8000)
# if it isn't already running. Called by Desktop\Odysseus.bat (same pattern as
# Start-Chroma.ps1), by "Talk to Enigma.bat" (with -Voice), or run by hand.
# Serves the instruct checkpoint WITH her real long-term memory enabled
# (data\memory -- the remember tool writes here).
# ASCII-only output (Windows cp1252 console).
param(
    [switch]$Voice  # enable the voice organ (speak tool + /v1/audio/speech)
)

$engineDir = "C:\Users\SirKn\Enigma Engine"
$python = "C:\Users\SirKn\AppData\Local\Programs\Python\Python312\python.exe"
$port = 8000

$up = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($up) {
    Write-Output "Enigma already serving on port $port (pid $($up[0].OwningProcess))."
    exit 0
}

# Hidden window: a visible console gets closed by accident (see CLAUDE.md
# training-ops gotcha). Logs go to serve_enigma.log in the repo.
$log = Join-Path $engineDir "serve_enigma.log"
$serveArgs = @("serve_enigma.py", "--port", "$port", "--model", "models\enigma_dpo\model.pth", "--memory-dir", "data\memory")
if ($Voice) { $serveArgs += "--voice" }
Start-Process -FilePath $python `
    -ArgumentList $serveArgs `
    -WorkingDirectory $engineDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput $log `
    -RedirectStandardError (Join-Path $engineDir "serve_enigma.err.log")
Write-Output "Enigma starting on http://127.0.0.1:$port/v1 (log: serve_enigma.log)"
