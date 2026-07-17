@echo off
REM Talk to Enigma -- starts her server if it isn't running, waits until the
REM port answers, then opens the chat in HER OWN WINDOW (enigma_window.py,
REM WebView2 -- no browser, no tabs). VOICE OFF by user ruling 2026-07-16
REM ("later when it matters") -- add -Voice below to turn her voice back on;
REM the page degrades to "voice: off" honestly. Gaming-friendly on purpose:
REM 182M model (~750MB VRAM, ms-scale replies); eyes/image-gen stay OFF here.
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\SirKn\Enigma Engine\Start-Enigma.ps1"
powershell -NoProfile -Command "for($i=0;$i -lt 120;$i++){ if(Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue){ exit 0 }; Start-Sleep -Milliseconds 500 }; Write-Output 'Server is taking a while -- the window may need a moment.'; exit 0"
REM pyw = console-less Python launcher; plain py (visible console) as fallback.
where pyw >nul 2>nul
if errorlevel 1 (
    start "" py -3.12 "%~dp0enigma_window.py"
) else (
    start "" pyw -3.12 "%~dp0enigma_window.py"
)
