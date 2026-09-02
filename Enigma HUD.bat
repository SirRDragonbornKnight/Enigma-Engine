@echo off
REM Enigma HUD -- starts her server if it isn't running, then opens the HUD
REM page in HER OWN WINDOW (enigma_window.py, WebView2 -- no browser, no
REM tabs). The same window "Talk to Enigma" opens, pointed at /hud: it opens
REM on a boot page and switches to the HUD the moment the server answers,
REM however long the cold start takes.
REM She boots COMPLETE (2026-07-27 ruling): every launch carries voice, eyes,
REM ears, and image-gen -- the -Voice switch below is an accepted no-op kept
REM so old shortcuts still run, and there is no launcher way to boot her
REM partial. First launch is SILENT: talk-mode starts OFF, so she narrates
REM nothing until you flip "Talk" on in her window (she only speaks on an
REM explicit "say it out loud" ask). The Talk setting PERSISTS across
REM launches (data\talk_mode.json). To mute her sound, use "Enigma Quiet"
REM or the tray's Mute -- those are the real switches.

set "PYDIR=C:\Users\SirKn\AppData\Local\Programs\Python\Python312"

REM This bat often runs HIDDEN (from the tray), so a failure must be a popup,
REM not console text nobody sees. PYDIR is the WINDOW's python (system 3.12,
REM has pywebview); the SERVER runs the repo venv -- Start-Enigma.ps1 checks
REM that one itself and pops its own error if the venv is missing.
if not exist "%PYDIR%\python.exe" (
    powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show('Python 3.12 was not found, so Enigma cannot start. Reinstall Python 3.12 from python.org and try again.','Enigma') | Out-Null"
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\SirKn\Enigma Engine\Start-Enigma.ps1" -Voice
REM Nonzero = the port is held by something that is NOT Enigma (Start-Enigma
REM already showed the popup) -- do not open a window onto a foreign service.
if errorlevel 1 exit /b 1

REM Console-less launch: pyw, then py, then pythonw.exe directly.
where pyw >nul 2>nul
if not errorlevel 1 (
    start "" pyw -3.12 "%~dp0enigma_window.py" --url http://127.0.0.1:8000/hud
    goto :eof
)
where py >nul 2>nul
if not errorlevel 1 (
    start "" py -3.12 "%~dp0enigma_window.py" --url http://127.0.0.1:8000/hud
    goto :eof
)
start "" "%PYDIR%\pythonw.exe" "%~dp0enigma_window.py" --url http://127.0.0.1:8000/hud
