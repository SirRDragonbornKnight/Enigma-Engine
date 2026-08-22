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
# The RESOLVED bind port is validated here for a NON-DEFAULT persona: a pack
# that resolves to a reserved port (8000 daily, 8123 eval) is refused, whether
# it came from -Port, the pack's own port field, or the portless fall-through.
# Persona.load refuses a pack DECLARING a reserved port when serve reads it,
# but the -Port override and the portless-falls-to-8000 case reach neither that
# check nor the ownership test -- so the resolve refuses them, in the one place
# Start and Stop both read.
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
        # A non-default persona binding a reserved port squats the eval scratch
        # port an eval run CLEARS (8123) or Enigma's own daily serve (8000).
        # Refuse it at resolve, whichever source produced it, so a pack cannot
        # reach either port through -Port, its own field, or the fall-through.
        if ($bind -eq 8000 -or $bind -eq 8123) {
            if ($bind -eq 8000) {
                $why = "Enigma's daily serve port"
            } else {
                $why = "the eval scratch port, which an eval run CLEARS"
            }
            throw "persona pack $PackDir resolves to port $bind, which is $why -- give the pack its own port (1024-65535, not 8000 or 8123)"
        }
        # Her window is enigma_window.py either way, so the pack path on the
        # command line is what tells two windows apart. The shims pass the
        # directory they were generated into, which is the spelling matched
        # here. Her SERVE is matched the same way and for the same reason: one
        # script serves every AI in this checkout, so the process name answers
        # the wrong question and the pack path is what makes it hers.
        $windowMatch = "*enigma_window.py*"
        $windowExclude = ""
        $windowExtra = "*$PackDir*"
        $serveMatch = "*serve_enigma.py*"
        $serveExclude = ""
        $serveExtra = "*$PackDir*"
    } else {
        $name = "Enigma"
        $slug = "enigma"
        $memoryDir = "data\memory"
        $talkBat = Join-Path $EngineDir "Talk to Enigma.bat"
        if ($Port -gt 0) { $bind = $Port } else { $bind = 8000 }
        # HER window is the one started without a pack: a launcher that closed
        # every enigma_window.py would close the other AI's window too. Same
        # rule for HER serve -- a pack's carries --persona, hers never does.
        $windowMatch = "*enigma_window.py*"
        $windowExclude = "*--persona*"
        $windowExtra = ""
        $serveMatch = "*serve_enigma.py*"
        $serveExclude = "*--persona*"
        $serveExtra = ""
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
        StartMutex    = "Local\" + ($name -replace ' ', '') + "Start"
        Initial       = $name.Substring(0, 1).ToUpperInvariant()
        TalkBat       = $talkBat
        BaseUrl       = "http://127.0.0.1:$bind"
        WindowMatch   = $windowMatch
        WindowExtra   = $windowExtra
        WindowExclude = $windowExclude
        ServeMatch    = $serveMatch
        ServeExtra    = $serveExtra
        ServeExclude  = $serveExclude
        # A pytest run of tests/test_enigma_window.py or tests/test_serve_enigma.py
        # carries the matched FILE NAME on its own command line, so the process
        # matchers above see the suite that is testing them and Stop force-killed
        # it. Nothing this chain launches ever runs through pytest.
        TestRunner    = "*pytest*"
    }
}
