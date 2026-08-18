# Enigma-Quiet.ps1 -- desktop toggle for her voice, no window needed. Reads the
# server mute state and flips it; when muting, it ALSO hushes the utterance
# playing right now (mute alone only gates the NEXT one). Counterpart to the
# tray Mute item and her window's Mute button. ASCII-only output.
# -Persona toggles the AI from THAT pack, on her own port, instead.
param(
    # A persona pack DIRECTORY. Empty is Enigma.
    [string]$Persona = "",
    # Override the port. 0 = the pack's own `port` field, or 8000 for Enigma.
    [int]$Port = 0
)

. (Join-Path $PSScriptRoot "Enigma-Persona.ps1")
$self = Resolve-EnigmaPersona -EngineDir $PSScriptRoot -PackDir $Persona -Port $Port
$aiName = $self.Name
$base = "$($self.BaseUrl)/v1/audio"
try {
    $cur = Invoke-RestMethod -Uri "$base/mute" -TimeoutSec 2
    $target = -not [bool]$cur.muted
    # Set mute FIRST, then stop: muting gates the next utterance (the speak tool
    # and /v1/audio/speech both honor mute), so there is no gap where a reply
    # queued between the two calls could start playing un-muted. Stop then only
    # has to cut off the utterance already sounding.
    $body = @{ muted = $target } | ConvertTo-Json -Compress
    $now = Invoke-RestMethod -Uri "$base/mute" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 2
    if ($now.muted) {
        try { Invoke-RestMethod -Uri "$base/stop" -Method Post -TimeoutSec 2 | Out-Null } catch { }
        Write-Output "$aiName muted."
    } else {
        Write-Output "$aiName voice back on."
    }
} catch {
    Write-Output "$aiName server is not running -- nothing to toggle."
}
