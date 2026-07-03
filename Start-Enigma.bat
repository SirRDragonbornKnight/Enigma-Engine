@echo off
title Enigma Server
REM ============================================================
REM  Enigma's local model server.
REM  Serves the model as an OpenAI-compatible API on
REM      http://127.0.0.1:8000/v1
REM  Odysseus talks to this. KEEP THIS WINDOW OPEN while you
REM  chat with Enigma. Close it (or press Ctrl+C) to stop.
REM ============================================================
cd /d "C:\Users\SirKn\Enigma Engine"
"C:\Users\SirKn\AppData\Local\Programs\Python\Python312\python.exe" serve_enigma.py
echo.
echo ===========================================================
echo  Enigma server stopped. Press any key to close this window.
echo ===========================================================
pause >nul
