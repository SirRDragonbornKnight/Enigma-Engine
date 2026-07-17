# Stop-Enigma.ps1 -- shut Enigma down: her chat window (if open) and the
# hidden server on port 8000. Counterpart to Start-Enigma.ps1; called by
# "Stop Enigma.bat" or run by hand.
# Safe by design: the port-8000 process is only killed if its command line
# really is serve_enigma.py -- anything else holding the port is left alone.
# ASCII-only output (Windows cp1252 console).

# 1) Her chat window (the enigma_window.py shim), if one is open. Matched by
# COMMAND LINE, not window title -- any other python window that happens to
# be titled "Enigma" (avatar tooling, a second pywebview app) is not ours.
$windows = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*enigma_window.py*" })
if ($windows.Count -eq 0) {
    Write-Output "window: none open."
} else {
    foreach ($w in $windows) {
        Stop-Process -Id $w.ProcessId -Force
        Write-Output "window: closed (pid $($w.ProcessId))."
    }
}

# 2) The server. Verify ownership before killing.
$conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($null -eq $conn) {
    Write-Output "server: nothing on port 8000 -- already stopped."
} else {
    $ownerId = $conn.OwningProcess
    $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$ownerId" -ErrorAction SilentlyContinue).CommandLine
    $procName = (Get-Process -Id $ownerId -ErrorAction SilentlyContinue).ProcessName
    # Ours = serve_enigma.py directly, OR the enigma/enigma-ai console-script
    # wrappers from pyproject (those run as enigma.exe with no .py on the line).
    if ($cmd -like "*serve_enigma.py*" -or $procName -in @("enigma", "enigma-ai")) {
        Stop-Process -Id $ownerId -Force
        Write-Output "server: stopped (pid $ownerId)."
    } else {
        Write-Output "server: port 8000 is held by pid $ownerId ($procName), which is NOT Enigma -- left alone."
        Write-Output "  command line: $cmd"
    }
}
