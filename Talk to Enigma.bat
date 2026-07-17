@echo off
REM Talk to Enigma -- starts her server (voice on) if it isn't running, waits
REM until it answers, then opens the chat in HER OWN WINDOW (enigma_window.py,
REM WebView2 -- no browser, no tabs). The window speaks her replies and has a
REM Mute button (also silences the server-side speak tool). Gaming-friendly on
REM purpose: 182M model (~750MB VRAM, ms-scale replies), voice is CPU (SAPI);
REM eyes/image-gen stay OFF here.
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\SirKn\Enigma Engine\Start-Enigma.ps1" -Voice
powershell -NoProfile -Command "for($i=0;$i -lt 120;$i++){ if(Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue){ exit 0 }; Start-Sleep -Milliseconds 500 }; Write-Output 'Server is taking a while -- the window may need a moment.'; exit 0"
REM pyw = console-less Python launcher; plain py (visible console) as fallback.
where pyw >nul 2>nul
if errorlevel 1 (
    start "" py -3.12 "%~dp0enigma_window.py"
) else (
    start "" pyw -3.12 "%~dp0enigma_window.py"
)
