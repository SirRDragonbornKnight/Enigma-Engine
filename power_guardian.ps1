# Enigma Power Guardian
# ---------------------
# Stops Enigma training cleanly the moment the grid drops, rides the UPS bridge
# to the whole-house generator, and AUTO-MEASURES how long the transfer took
# (no need to know the generator's timing in advance -- it learns it each time).
#
# The UPS (APC Back-UPS NS 1500M2) is ONLY a bridge -- the generator carries the
# house -- so this does NOT shut the PC down. It just parks training so the 5090
# isn't hammering at ~480W through the power transition; with the load dropped to
# ~80W idle the UPS can bridge even a slow generator with room to spare.
#
# Training is left STOPPED after an outage (manual restart: resume_training.ps1,
# or tell Claude "start"). Built 2026-06-18 for the daily Enigma workflow.
#
# Run detached + Hidden:
#   Start-Process powershell -WindowStyle Hidden -ArgumentList '-ExecutionPolicy','Bypass','-File','<repo>\power_guardian.ps1'
# Stop: Stop-Process the powershell PID. History lands in power_guardian.log

$ErrorActionPreference = 'Continue'
$repo     = $PSScriptRoot
$glog     = Join-Path $repo 'power_guardian.log'
$DEBOUNCE = 10    # seconds on-battery before it counts as a real outage (ignore flickers)
$POLL     = 4     # seconds between UPS checks while on mains

function Log($msg) {
  $line = ('{0}  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg)
  Add-Content -Path $glog -Value $line
  Write-Host $line
}

function OnBattery {
  # Win32_Battery.BatteryStatus: 1 = discharging (on battery); 2 = on AC (grid OR generator)
  $b = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1
  if (-not $b) { return $false }
  return ([int]$b.BatteryStatus -eq 1)
}

function Charge {
  $b = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($b) { return [int]$b.EstimatedChargeRemaining } else { return -1 }
}

function EnigmaRunning {
  [bool](Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
         Where-Object { $_.CommandLine -like '*pretrain_enigma*' })
}

function StopEnigma {
  Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='cmd.exe'" |
    Where-Object { $_.CommandLine -match 'pretrain_enigma' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

Log "guardian online. UPS poll ${POLL}s, outage debounce ${DEBOUNCE}s. (UPS = bridge only; no PC shutdown.)"
while ($true) {
  if (OnBattery) {
    $t0 = Get-Date
    $stable = $true
    for ($i = 0; $i -lt $DEBOUNCE; $i++) {
      Start-Sleep -Seconds 1
      if (-not (OnBattery)) { $stable = $false; break }
    }
    if (-not $stable) {
      Log "blip ignored: UPS touched battery then recovered in <${DEBOUNCE}s. Training untouched."
      continue
    }

    Log ("OUTAGE: on battery >${DEBOUNCE}s (charge {0}%). Grid down, UPS bridging to generator." -f (Charge))
    if (EnigmaRunning) {
      StopEnigma
      Start-Sleep -Seconds 2
      if (EnigmaRunning) { Log "WARNING: Enigma still running after stop attempt." }
      else { Log "Enigma stopped immediately (resumes from last <=250-step checkpoint; draw ~480W -> ~80W)." }
    } else {
      Log "Enigma not running -- nothing to park."
    }

    # Ride it out: wait for AC to return (generator online or grid restored),
    # measuring the real transfer/outage duration as we go.
    while (OnBattery) { Start-Sleep -Seconds 2 }
    $dur = [int]((Get-Date) - $t0).TotalSeconds
    Log ("POWER RESTORED after ~{0}s on battery (charge {1}%). Generator carried it." -f $dur, (Charge))
    Log "Training left stopped. Restart with resume_training.ps1 or tell Claude 'start'."
  }
  Start-Sleep -Seconds $POLL
}
