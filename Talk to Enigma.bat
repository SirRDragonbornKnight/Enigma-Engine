@echo off
REM Talk to Enigma -- starts her server (voice on) if it isn't running, waits
REM until it answers, then opens the built-in chat page. The page speaks her
REM replies through the browser and has a Mute button (also silences the
REM server-side speak tool). Gaming-friendly on purpose: 182M model (~750MB
REM VRAM, ms-scale replies), voice is CPU (SAPI); eyes/image-gen stay OFF here.
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\SirKn\Enigma Engine\Start-Enigma.ps1" -Voice
powershell -NoProfile -Command "for($i=0;$i -lt 120;$i++){ if(Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue){ exit 0 }; Start-Sleep -Milliseconds 500 }; Write-Output 'Server is taking a while -- the page may need a refresh.'; exit 0"
start "" http://127.0.0.1:8000/
