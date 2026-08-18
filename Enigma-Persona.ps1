# Enigma-Persona.ps1 -- WHO the launcher chain is acting on, in one place.
# Dot-sourced by Start-Enigma.ps1, Stop-Enigma.ps1, Enigma-Tray.ps1 and
# Enigma-Quiet.ps1; it defines no script-level state and starts nothing, so
# sourcing it is free.
# Two questions live here. Get-ServingPersona asks a PORT who is answering on
# it (the ownership check); Resolve-EnigmaPersona derives everything a
# launcher needs from a persona pack -- and the copies of the first one that
# Start and Stop each carried are gone, so the fallback rule cannot drift
# between the script that starts her and the script that kills her.
# ASCII-only output (Windows cp1252 console).

# The persona serving on $Port, or "" when the port cannot say: nothing
# listening, a program that is not a serve, a build too old to report it, or a
# serve wedged mid-generation. "" means UNKNOWN, never "not ours" -- callers
# fall back to the process test so a server that cannot answer HTTP is still
# recognized as hers and stays killable.
function Get-ServingPersona {
    param([int]$Port)
    try {
        $caps = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/capabilities" -TimeoutSec 3
    } catch {
        return ""
    }
    if ($null -eq $caps) { return "" }
    $name = $caps.persona
    if ($name -isnot [string]) { return "" }
    return $name
}

# Her name as a bare identifier -- the mirror of Persona.slug (core/persona.py:
# lowercase, runs of anything else folded to one underscore). Invariant
# lowercase and a case-SENSITIVE replace on purpose: .NET matches
# case-insensitively by default, and a Turkish locale lowercases "I" to a
# character no slug of Enigma's has ever contained.
function Get-PersonaSlug {
    param([string]$Name)
    $slug = ($Name.ToLowerInvariant() -creplace '[^a-z0-9]+', '_').Trim('_')
    if (-not $slug) { $slug = "ai" }
    return $slug
}

# Everything the four launchers derive from WHO they are running: the name
# they check ownership against, the port they bind, the memory store, the log
# files, the tray mutex and identity, and the .bat the tray's Talk item runs.
#
# No -PackDir is ENIGMA, and every value below is the literal those scripts
# carried before they took parameters -- her launch is unchanged, which is
# what -DryRun exists to prove.
#
# A pack's memory lives WITH the pack ($HOME\<data_dirname>\memory), so two
# AIs started from here do not share the one organ that persists. HERS stays
# the repo-relative data\memory it has always been: the wave-1 mute/talk move
# does not extend to her memories.
#
# The port is not validated here. A pack declaring 8000 or 8123 is refused by
# Persona.load when serve reads the same pack, and a port already held is
# refused by the ownership check in Start-Enigma -- a third copy of the rule
# in PowerShell would only be a fourth thing to drift.
function Resolve-EnigmaPersona {
    param(
        [string]$EngineDir,
        [string]$PackDir = "",
        [int]$Port = 0
    )
    if ($PackDir) {
        $manifest = Join-Path $PackDir "pack.json"
        if (-not (Test-Path $manifest)) {
            throw "persona pack $PackDir has no pack.json -- a pack directory carries it beside its content files"
        }
        $pack = Get-Content -Raw -Encoding UTF8 $manifest | ConvertFrom-Json
        $name = $pack.name
        if ($name -isnot [string] -or -not $name) {
            throw "persona pack $manifest does not name its AI (no string 'name')"
        }
        $slug = Get-PersonaSlug -Name $name
        $dirname = $pack.data_dirname
        if ($dirname -isnot [string] -or -not $dirname) { $dirname = ".$slug" }
        $memoryDir = Join-Path (Join-Path $HOME $dirname) "memory"
        # Written by make_persona_launchers.py, into the pack itself.
        $talkBat = Join-Path $PackDir "Talk.bat"
        if ($Port -gt 0) {
            $bind = $Port
        } elseif ($pack.port) {
            $bind = [int]$pack.port
        } else {
            $bind = 8000
        }
        # Her window is enigma_window.py either way, so the pack path on the
        # command line is what tells two windows apart. The shims pass the
        # directory they were generated into, which is the spelling matched
        # here.
        $windowMatch = "*enigma_window.py*"
        $windowExclude = ""
        $windowExtra = "*$PackDir*"
    } else {
        $name = "Enigma"
        $slug = "enigma"
        $memoryDir = "data\memory"
        $talkBat = Join-Path $EngineDir "Talk to Enigma.bat"
        if ($Port -gt 0) { $bind = $Port } else { $bind = 8000 }
        # HER window is the one started without a pack: a launcher that closed
        # every enigma_window.py would close the other AI's window too.
        $windowMatch = "*enigma_window.py*"
        $windowExclude = "*--persona*"
        $windowExtra = ""
    }
    return [PSCustomObject]@{
        Name          = $name
        Slug          = $slug
        Port          = $bind
        PackDir       = $PackDir
        MemoryDir     = $memoryDir
        Log           = Join-Path $EngineDir "serve_$slug.log"
        ErrLog        = Join-Path $EngineDir "serve_$slug.err.log"
        Mutex         = "Local\" + ($name -replace ' ', '') + "Tray"
        Initial       = $name.Substring(0, 1).ToUpperInvariant()
        TalkBat       = $talkBat
        BaseUrl       = "http://127.0.0.1:$bind"
        WindowMatch   = $windowMatch
        WindowExtra   = $windowExtra
        WindowExclude = $windowExclude
    }
}
