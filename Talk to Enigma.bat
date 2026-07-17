@echo off
REM Talk to Enigma -- starts her server if it isn't running, then opens the
REM chat in HER OWN WINDOW (enigma_window.py, WebView2 -- no browser, no
REM tabs). The window opens on a boot page and switches to her chat the
REM moment the server answers, however long the cold start takes. VOICE OFF
REM by user ruling 2026-07-16 ("later when it matters") -- add -Voice to the
REM Start-Enigma line to turn her voice back on; the page degrades to
REM "voice: off" honestly. Gaming-friendly on purpose: 182M model (~750MB
REM VRAM, ms-scale replies); eyes/image-gen stay OFF here.

REM This bat often runs HIDDEN (from the tray), so a failure must be a popup,
REM not console text nobody sees.
py -3.12 -c "pass" >nul 2>nul
if errorlevel 1 (
    powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show('Python 3.12 was not found, so Enigma cannot start. Reinstall Python 3.12 from python.org and try again.','Enigma') | Out-Null"
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\SirKn\Enigma Engine\Start-Enigma.ps1"
REM pyw = console-less Python launcher; plain py (visible console) as fallback.
where pyw >nul 2>nul
if errorlevel 1 (
    start "" py -3.12 "%~dp0enigma_window.py"
) else (
    start "" pyw -3.12 "%~dp0enigma_window.py"
)
