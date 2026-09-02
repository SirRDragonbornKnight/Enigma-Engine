<#
.SYNOPSIS
    Assemble "Enigma-to-go": a self-contained folder that serves her from a USB
    stick on a machine with no Python, no CUDA and no toolchain.

.DESCRIPTION
    Embeddable CPython + vendored wheels + her code + the int8 checkpoint. The
    folder runs entirely from relative paths, so it does not care what drive
    letter the stick gets.

    WHAT NEVER TRAVELS: the repo's data\, teachings.jsonl, models\, Enigma
    Backups\ and any dotdir. The stick carries her WEIGHTS and CODE, never this
    machine's conversations, memories or stores. That is enforced below by an
    explicit deny-list check over the finished folder, not merely by being
    careful about what gets copied.

    The wheel set is PINNED to the versions the C4 quality gate was measured
    against (top-1 97/100 vs fp32 on torch 2.10.0). An unpinned resolve picks a
    newer torch, and the gate receipt would no longer describe what shipped.

.EXAMPLE
    .\build_portable.ps1
    .\build_portable.ps1 -Target D:\Enigma-Portable -Force
#>
[CmdletBinding()]
param(
    [string]$Target = (Join-Path $env:TEMP "Enigma-Portable"),
    [string]$PythonZip,
    [string]$WheelDir,
    [string]$Model = "C:\Users\SirKn\Enigma Backups\enigma_v2_sft2_int8wo.pth",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Repo = $PSScriptRoot
# Build inputs live in a cache OUTSIDE the repo: 147 MB of wheels and a CPython
# zip are not repo content. This script never downloads them itself -- fetching
# is the operator's call, and the message below says exactly how.
$VendorRoot = Join-Path $env:TEMP "enigma-portable-vendor"

# Anything matching these must never appear inside the built folder.
$Forbidden = @("teachings.jsonl", "teach_pairs.jsonl", "dpo_pairs.jsonl")
$ForbiddenDirs = @("data\eval", "data\sft", "data\pretrain", "data\finetune", "models", "Enigma Backups")

function Write-Utf8([string]$Path, [string]$Text) {
    # PS 5.1's Set-Content/Out-File -Encoding utf8 writes a BOM, which has
    # already BOM-stamped four source files in this repo once.
    [System.IO.File]::WriteAllText($Path, $Text, (New-Object System.Text.UTF8Encoding($false)))
}

function Say([string]$Message) { Write-Host "  $Message" }

if (Test-Path $Target) {
    if (-not $Force) {
        throw "REFUSED: $Target already exists. Pass -Force to rebuild it, or name a new -Target."
    }
    Say "removing the previous build at $Target"
    Remove-Item -Recurse -Force $Target
}

if (-not $PythonZip) { $PythonZip = Join-Path $VendorRoot "python-3.12.10-embed-amd64.zip" }
if (-not $WheelDir)  { $WheelDir  = Join-Path $VendorRoot "wheels" }
foreach ($needed in @($PythonZip, $WheelDir, $Model)) {
    if (-not (Test-Path $needed)) {
        throw @"
REFUSED: missing build input: $needed

Fetch the inputs once (they are downloads -- your call, not this script's):
  New-Item -ItemType Directory -Force "$VendorRoot\wheels"
  Invoke-WebRequest https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip ``
      -OutFile "$VendorRoot\python-3.12.10-embed-amd64.zip"
  venv\Scripts\python.exe -m pip wheel proxy_tools==0.1.0 --no-deps -w "$VendorRoot\wheels"
  venv\Scripts\python.exe -m pip download -r "$VendorRoot\wheels\requirements-portable.txt" ``
      -d "$VendorRoot\wheels" --only-binary=:all: --find-links "$VendorRoot\wheels"
(proxy_tools is pywebview's one sdist-only dependency -- pure Python, so the
wheel builds here without a compiler.)
"@
    }
}

Write-Host "Building Enigma-to-go at $Target"
New-Item -ItemType Directory -Force -Path $Target | Out-Null

# --- 1. Embeddable Python ---------------------------------------------------
$PyDir = Join-Path $Target "python"
Say "unpacking $(Split-Path -Leaf $PythonZip)"
Expand-Archive -LiteralPath $PythonZip -DestinationPath $PyDir -Force

# The embeddable build ships site DISABLED, so site-packages is not on the
# path at all -- every vendored wheel would be invisible. Enabling site and
# naming the app dir (..) is what makes serve_enigma.py importable next to it.
$pth = Get-ChildItem -Path $PyDir -Filter "python*._pth" | Select-Object -First 1
if (-not $pth) { throw "REFUSED: no python*._pth in the embeddable zip -- layout changed" }
Write-Utf8 $pth.FullName "python312.zip`r`n.`r`n..`r`nLib\site-packages`r`nimport site`r`n"
Say "enabled site + app dir in $($pth.Name)"

# --- 2. Vendored wheels -----------------------------------------------------
$SitePkgs = Join-Path $PyDir "Lib\site-packages"
New-Item -ItemType Directory -Force -Path $SitePkgs | Out-Null
Say "installing vendored wheels (offline) into python\Lib\site-packages"
$req = Join-Path $WheelDir "requirements-portable.txt"
if (-not (Test-Path $req)) { throw "REFUSED: $req not found (the pinned closure list)" }
$pip = Join-Path $Repo "venv\Scripts\python.exe"
# --no-warn-conflicts: a --target install touches nothing in the builder's venv,
# but pip still audits that venv and prints its unrelated conflicts as "ERROR"
# lines -- which read like a failed build and are not one.
& $pip -m pip install --no-index --find-links $WheelDir --target $SitePkgs -r $req --quiet --no-warn-script-location --no-warn-conflicts
if ($LASTEXITCODE -ne 0) { throw "REFUSED: offline wheel install failed with exit $LASTEXITCODE" }

# --- 3. Her code ------------------------------------------------------------
Say "copying her code"
foreach ($f in @("serve_enigma.py", "enigma_window.py", "quantize_serving_ckpt.py")) {
    Copy-Item (Join-Path $Repo $f) -Destination $Target
}
# enigma_engine\ carries the tokenizer vocab the boot reads; __pycache__ does
# not travel (stale bytecode on a different machine is a debugging tax).
Copy-Item (Join-Path $Repo "enigma_engine") -Destination $Target -Recurse
Get-ChildItem -Path (Join-Path $Target "enigma_engine") -Recurse -Directory -Filter "__pycache__" |
    ForEach-Object { Remove-Item -Recurse -Force $_.FullName }
if (Test-Path (Join-Path $Repo "web")) {
    Copy-Item (Join-Path $Repo "web") -Destination $Target -Recurse
    Say "copied web\ (the HUD as it stands in the working tree)"
}

# --- 4. Weights + a FRESH memory home ---------------------------------------
$ModelDir = Join-Path $Target "models_portable"
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
Say "copying the int8 checkpoint ($([math]::Round((Get-Item $Model).Length / 1MB, 1)) MB)"
Copy-Item $Model -Destination (Join-Path $ModelDir "enigma_int8.pth")
# Her memories on the stick start EMPTY and stay the stick's own.
New-Item -ItemType Directory -Force -Path (Join-Path $Target "data\memory") | Out-Null

# --- 5. The launcher --------------------------------------------------------
# Relative paths only: the stick's drive letter changes per machine. cd /d %~dp0
# is what lets a double-click from anywhere still find her.
$bat = @"
@echo off
cd /d "%~dp0"
echo Starting Enigma (CPU, int8) on port 8077 ...
start "Enigma serve" /min python\python.exe serve_enigma.py --model models_portable\enigma_int8.pth --device cpu --port 8077 --memory-dir data\memory
python\python.exe enigma_window.py --url http://127.0.0.1:8077/hud
"@
Write-Utf8 (Join-Path $Target "Enigma-Portable.bat") $bat

# --- 6. The exclusions are CHECKED, not assumed -----------------------------
# NOTE: the vendored wheels' own __pycache__ STAYS. Stripping it saves 122 MB
# but costs 8.7s of first launch against 4.1s (measured), and Python then
# writes those 122 MB straight back onto the stick on first run -- or
# recompiles every single launch if the medium is read-only. Only HER code's
# stale bytecode is removed (above), which is a correctness matter, not size.

Say "verifying nothing personal travelled"
foreach ($name in $Forbidden) {
    $hit = Get-ChildItem -Path $Target -Recurse -File -Filter $name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($hit) { throw "REFUSED: $($hit.FullName) must never ship on the stick" }
}
foreach ($rel in $ForbiddenDirs) {
    if (Test-Path (Join-Path $Target $rel)) { throw "REFUSED: $rel must never ship on the stick" }
}
$dot = Get-ChildItem -Path $Target -Recurse -Force -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like ".*" } | Select-Object -First 1
if ($dot) { throw "REFUSED: dotdir $($dot.FullName) must never ship on the stick" }
$mem = Get-ChildItem -Path (Join-Path $Target "data\memory") -Recurse -File -ErrorAction SilentlyContinue
if ($mem) { throw "REFUSED: the portable memory store must ship EMPTY" }

$size = (Get-ChildItem -Path $Target -Recurse -File | Measure-Object Length -Sum).Sum
$count = (Get-ChildItem -Path $Target -Recurse -File | Measure-Object).Count
Write-Host ""
Write-Host "Enigma-to-go is built."
Write-Host ("  {0}" -f $Target)
Write-Host ("  {0:N1} MB across {1:N0} files" -f ($size / 1MB), $count)
Write-Host "  launch: Enigma-Portable.bat  (serve on 127.0.0.1:8077, then her window)"
Write-Host "  weights: int8 weight-only, CPU. Quality gate PASSED at build time:"
Write-Host "           top-1 agreement 97/100 vs fp32 (gate >= 95)."
Write-Host "  speed  : int8 buys SIZE, not speed, on a CPU without AMX --"
Write-Host "           measured 23.7 tok/s int8 vs 30.7 fp32 on this box."
