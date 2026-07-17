@echo off
REM Talk to Enigma -- starts her server if it isn't running, then opens the
REM chat in HER OWN WINDOW (enigma_window.py, WebView2 -- no browser, no
REM tabs). The window opens on a boot page and switches to her chat the
REM moment the server answers, however long the cold start takes. VOICE OFF
REM by user ruling 2026-07-16 ("later when it matters") -- add -Voice to the
REM Start-Enigma line to turn her voice back on; the page degrades to
REM "voice: off" honestly. Gaming-friendly on purpose: 182M model (~750MB
REM VRAM, ms-scale replies); eyes/image-gen stay OFF here.

set "PYDIR=C:\Users\SirKn\AppData\Local\Programs\Python\Python312"

REM This bat often runs HIDDEN (from the tray), so a failure must be a popup,
REM not console text nobody sees. Gate on the SAME python the server uses --
REM the py launcher can be missing while python itself is fine.
if not exist "%PYDIR%\python.exe" (
    powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show('Python 3.12 was not found, so Enigma cannot start. Reinstall Python 3.12 from python.org and try again.','Enigma') | Out-Null"
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\SirKn\Enigma Engine\Start-Enigma.ps1"
REM Nonzero = the port is held by something that is NOT Enigma (Start-Enigma
REM already showed the popup) -- do not open a window onto a foreign service.
if errorlevel 1 exit /b 1

REM Console-less launch: pyw, then py, then pythonw.exe directly.
where pyw >nul 2>nul
if not errorlevel 1 (
    start "" pyw -3.12 "%~dp0enigma_window.py"
    goto :eof
)
where py >nul 2>nul
if not errorlevel 1 (
    start "" py -3.12 "%~dp0enigma_window.py"
    goto :eof
)
start "" "%PYDIR%\pythonw.exe" "%~dp0enigma_window.py"
