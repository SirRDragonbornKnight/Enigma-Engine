# Stop-Enigma.ps1 -- shut Enigma down: her chat window (if open) and the
# hidden server on port 8000. Counterpart to Start-Enigma.ps1; called by
# "Stop Enigma.bat" or run by hand.
# Safe by design: the port-8000 process is only killed if the server there says
# it is Enigma -- another AI, or any other program, is left alone. An EMPTY
# port is not "already stopped": a serve mid-cold-boot has not bound yet, so it
# is matched on its command line instead, by the same ownership rule.
# -Persona stops the AI from THAT pack instead: her port, her window, and
# nothing of Enigma's. With no -Persona every value is the literal this script
# carried before it took parameters, which is what -DryRun exists to prove.
# ASCII-only output (Windows cp1252 console).
param(
    # A persona pack DIRECTORY. Empty is Enigma.
    [string]$Persona = "",
    # Override the port. 0 = the pack's own `port` field, or 8000 for Enigma.
    [int]$Port = 0,
    # Print what this run would kill, and kill nothing.
    [switch]$DryRun
)

# WHO this script stops, and everything derived from her. A persona pack
# serves a DIFFERENT AI out of this same checkout, so "is the process
# serve_enigma.py" answers the wrong question -- it is true of every AI served
# from here. Get-ServingPersona lives in the shared file with the rest: the
# script that starts her and the script that kills her must not drift apart on
# what counts as hers.
. (Join-Path $PSScriptRoot "Enigma-Persona.ps1")
$self = Resolve-EnigmaPersona -EngineDir $PSScriptRoot -PackDir $Persona -Port $Port
$aiName = $self.Name
$bindPort = $self.Port

if ($DryRun) {
    Write-Output "DRYRUN stop: persona=$aiName port=$bindPort"
    Write-Output "DRYRUN window: match=$($self.WindowMatch) extra=$($self.WindowExtra) exclude=$($self.WindowExclude)"
    Write-Output "DRYRUN serve-process: match=$($self.ServeMatch) extra=$($self.ServeExtra) exclude=$($self.ServeExclude)"
    Write-Output "DRYRUN serve-port: match=$($self.ServePortMatch)"
    exit 0
}

# 1) Her chat window (the enigma_window.py shim), if one is open. Matched by
# COMMAND LINE, not window title -- any other python window that happens to
# be titled "Enigma" (avatar tooling, a second pywebview app) is not ours.
# A pack's window carries its pack directory on the line and HERS carries no
# --persona at all, so stopping one AI leaves the other's window standing.
# A pytest run of tests\test_enigma_window.py carries that file name too, and
# force-killing the suite that tests this matcher is not stopping her window.
$windows = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
    Where-Object {
        $_.CommandLine -like $self.WindowMatch -and
        $_.CommandLine -notlike $self.TestRunner -and
        ($self.WindowExtra -eq "" -or $_.CommandLine -like $self.WindowExtra) -and
        ($self.WindowExclude -eq "" -or $_.CommandLine -notlike $self.WindowExclude)
    })
if ($windows.Count -eq 0) {
    Write-Output "window: none open."
} else {
    foreach ($w in $windows) {
        try {
            Stop-Process -Id $w.ProcessId -Force -ErrorAction Stop
            Write-Output "window: closed (pid $($w.ProcessId))."
        } catch {
            Write-Output "window: FAILED to close pid $($w.ProcessId) ($($_.Exception.Message))."
        }
    }
}

# 2) The server. Verify ownership before killing.
$conn = Get-NetTCPConnection -LocalPort $bindPort -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($null -eq $conn) {
    # An empty port is not proof she is down. serve BINDS LAST: boot() reads
    # and sha256s a multi-GB checkpoint and brings the organs up before uvicorn
    # listens, so a cold-booting server holds no port at all -- and "already
    # stopped" left it to bind seconds after Stop said she was gone. It cannot
    # answer /v1/capabilities yet either, so the process line is the honest
    # fallback here, matched by the same ownership rule as her window (a pack's
    # serve carries --persona, hers never does; a pytest run is not a serve).
    # AND BY PORT: the ownership rule alone is true of every serve of hers,
    # so with 8000 empty this force-killed the eval scratch serve on 8123 and
    # any second serve on any other port. ServePortMatch is the port half --
    # this AI's own --port, or (only when that port is serve's own 8000
    # default) a command line carrying no --port at all.
    $booting = @(Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -like $self.ServeMatch -and
            $_.CommandLine -notlike $self.TestRunner -and
            $_.CommandLine -match $self.ServePortMatch -and
            ($self.ServeExtra -eq "" -or $_.CommandLine -like $self.ServeExtra) -and
            ($self.ServeExclude -eq "" -or $_.CommandLine -notlike $self.ServeExclude)
        })
    if ($booting.Count -eq 0) {
        Write-Output "server: nothing on port $bindPort -- already stopped."
    } else {
        foreach ($b in $booting) {
            try {
                Stop-Process -Id $b.ProcessId -Force -ErrorAction Stop
                Write-Output "server: serve was still booting -- stopped (pid $($b.ProcessId))."
            } catch {
                Write-Output "server: FAILED to stop booting pid $($b.ProcessId) ($($_.Exception.Message)) -- try from an elevated shell."
            }
        }
    }
} else {
    $ownerId = $conn.OwningProcess
    $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$ownerId" -ErrorAction SilentlyContinue).CommandLine
    $procName = (Get-Process -Id $ownerId -ErrorAction SilentlyContinue).ProcessName
    # Ours = the server on the port SAYS it is this AI. With no answer, the old
    # process test decides instead: serve_enigma.py directly, OR the
    # enigma/enigma-ai console-script wrappers from pyproject (those run as
    # enigma.exe with no .py on the line).
    $serving = Get-ServingPersona -Port $bindPort
    if ($serving) {
        # -ceq, not -eq: PowerShell compares strings case-insensitively, and a
        # pack named "enigma" is a different AI wearing her spelling. Killing
        # the wrong AI is the failure this whole check exists to prevent.
        $ours = ($serving -ceq $aiName)
    } else {
        $ours = ($cmd -like "*serve_enigma.py*" -or $procName -in @("enigma", "enigma-ai"))
    }
    if ($ours) {
        try {
            Stop-Process -Id $ownerId -Force -ErrorAction Stop
            Write-Output "server: stopped (pid $ownerId)."
        } catch {
            # e.g. access denied when the server was started elevated --
            # saying "stopped" here would be a lie the tray balloon repeats.
            Write-Output "server: FAILED to stop pid $ownerId ($($_.Exception.Message)) -- try from an elevated shell."
        }
    } else {
        # Another AI answers for itself; anything that cannot is described by
        # its process, as before.
        if ($serving) {
            $who = "$serving (pid $ownerId)"
        } else {
            $who = "pid $ownerId ($procName)"
        }
        Write-Output "server: port $bindPort is held by $who, which is NOT $aiName -- left alone."
        Write-Output "  command line: $cmd"
    }
}
