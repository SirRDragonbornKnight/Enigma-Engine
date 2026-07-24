@echo off
REM Talk to Enigma -- starts her server if it isn't running, then opens the
REM chat in HER OWN WINDOW (enigma_window.py, WebView2 -- no browser, no
REM tabs). The window opens on a boot page and switches to her chat the
REM moment the server answers, however long the cold start takes. VOICE ON
REM (Kokoro, her Cortana voice) as of 2026-07-23 -- lifts the 2026-07-16
REM voiceless ruling. First-ever launch is SILENT: talk-mode starts OFF, so she
REM narrates nothing until you flip "Talk" on in her window (she only speaks on
REM an explicit "say it out loud" ask). The Talk setting PERSISTS across
REM launches (data\talk_mode.json). Drop -Voice below to go fully mute.
REM Gaming-friendly on purpose: 182M model (~750MB
REM VRAM, ms-scale replies); eyes/image-gen stay OFF here.

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
    start "" pyw -3.12 "%~dp0enigma_window.py"
    goto :eof
)
where py >nul 2>nul
if not errorlevel 1 (
    start "" py -3.12 "%~dp0enigma_window.py"
    goto :eof
)
start "" "%PYDIR%\pythonw.exe" "%~dp0enigma_window.py"
