@echo off
REM Starts the Enigma OpenAI-compatible server (same serve as
REM Start-Enigma.bat; kept for the desktop shortcut name).
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe serve_enigma.py --model models\enigma_dpo\model.pth --memory-dir data\memory
) else (
    python serve_enigma.py --model models\enigma_dpo\model.pth --memory-dir data\memory
)
pause
