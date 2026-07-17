@echo off
REM Enigma Tray -- puts Enigma in the system tray. Right-click the purple-E
REM icon: Talk to Enigma / Mute / Stop Enigma / Exit. Double-click = Talk.
REM Only one instance ever runs (a second launch exits quietly).
REM Want it on every boot? Win+R, type shell:startup, copy this file there.
start "" powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "C:\Users\SirKn\Enigma Engine\Enigma-Tray.ps1"
