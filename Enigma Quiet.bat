@echo off
REM Enigma Quiet -- desktop toggle for her voice. Click to silence her (and cut
REM off whatever she is saying right now); click again to turn her voice back
REM on. Flips the same server mute switch the tray and her window use. Safe
REM no-op if the server is not running. Real logic lives in Enigma-Quiet.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\SirKn\Enigma Engine\Enigma-Quiet.ps1"
