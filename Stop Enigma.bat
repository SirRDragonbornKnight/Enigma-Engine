@echo off
REM Stop Enigma -- closes her chat window and shuts the hidden server down.
REM (The server has no visible window on purpose, so this is the clean way
REM to stop it.) Only kills the port-8000 process if it really is
REM serve_enigma.py. Counterpart to "Talk to Enigma.bat".
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\SirKn\Enigma Engine\Stop-Enigma.ps1"
pause
