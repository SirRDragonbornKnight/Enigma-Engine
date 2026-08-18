# Enigma-Tray.ps1 -- Enigma lives in the system tray. Right-click the icon:
# Talk to Enigma / Mute / Stop Enigma / Exit tray icon. Double-click = Talk.
# One instance only (named mutex). Zero dependencies: WinForms NotifyIcon,
# icon drawn at runtime (dark disc, purple E) so no asset file to lose.
# Launched hidden by "Enigma Tray.bat". ASCII-only (Windows cp1252 console).
# -Persona puts a DIFFERENT AI in the tray instead: her own mutex, icon
# letter, labels, port and Talk shim, so two AIs can hold two icons at once.
param(
    # A persona pack DIRECTORY. Empty is Enigma.
    [string]$Persona = "",
    # Override the port. 0 = the pack's own `port` field, or 8000 for Enigma.
    [int]$Port = 0
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$script:engineDir = "C:\Users\SirKn\Enigma Engine"
# WHOSE tray icon this is. Every name, URL and path below is derived from her,
# so with no -Persona the tray is byte-for-byte the one it has always been.
. (Join-Path $PSScriptRoot "Enigma-Persona.ps1")
$self = Resolve-EnigmaPersona -EngineDir $script:engineDir -PackDir $Persona -Port $Port
$script:aiName = $self.Name
$script:bindPort = $self.Port
$script:packDir = $Persona
$script:talkBat = $self.TalkBat
$script:muteUrl = "$($self.BaseUrl)/v1/audio/mute"
$script:stopUrl = "$($self.BaseUrl)/v1/audio/stop"
$script:talkUrl = "$($self.BaseUrl)/v1/audio/talk-mode"

# One tray icon per AI, ever. A second launch of the SAME AI exits quietly --
# the mutex carries her name, so a second AI is not a second launch.
$created = $false
$script:mutex = New-Object System.Threading.Mutex($true, $self.Mutex, [ref]$created)
if (-not $created) { exit 0 }

# -- icon: dark disc, her purple initial, drawn in memory --
$bmp = New-Object System.Drawing.Bitmap(32, 32)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$bg = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(30, 30, 46))
$fg = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(203, 166, 247))
$g.FillEllipse($bg, 1, 1, 30, 30)
$font = New-Object System.Drawing.Font("Segoe UI", 15, [System.Drawing.FontStyle]::Bold)
$fmt = New-Object System.Drawing.StringFormat
$fmt.Alignment = [System.Drawing.StringAlignment]::Center
$fmt.LineAlignment = [System.Drawing.StringAlignment]::Center
$g.DrawString($self.Initial, $font, $fg, (New-Object System.Drawing.RectangleF(0, 0, 32, 32)), $fmt)
$g.Dispose()

$script:notify = New-Object System.Windows.Forms.NotifyIcon
$script:notify.Icon = [System.Drawing.Icon]::FromHandle($bmp.GetHicon())
$script:notify.Text = $script:aiName
$script:notify.Visible = $true

$menu = New-Object System.Windows.Forms.ContextMenu
$script:miTalk = New-Object System.Windows.Forms.MenuItem "Talk to $($script:aiName)"
$script:miTalkMode = New-Object System.Windows.Forms.MenuItem "Talk mode"
$script:miHush = New-Object System.Windows.Forms.MenuItem "Hush (stop talking)"
$script:miMute = New-Object System.Windows.Forms.MenuItem "Mute"
$script:miStop = New-Object System.Windows.Forms.MenuItem "Stop $($script:aiName)"
$script:miExit = New-Object System.Windows.Forms.MenuItem "Exit tray icon"
$sep1 = New-Object System.Windows.Forms.MenuItem "-"
$sep2 = New-Object System.Windows.Forms.MenuItem "-"
$menu.MenuItems.AddRange(@($script:miTalk, $sep1, $script:miTalkMode, $script:miHush, $script:miMute, $sep2, $script:miStop, $script:miExit))
$script:notify.ContextMenu = $menu

$talk = {
    Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c", "`"$script:talkBat`"" `
        -WindowStyle Hidden
}
$script:miTalk.add_Click($talk)
$script:notify.add_DoubleClick($talk)

$script:miMute.add_Click({
    # One POST -- the target state comes from the label the Popup handler just
    # painted (the menu can't be shown without Popup firing first). Halves the
    # time this handler blocks the UI thread.
    $target = ($script:miMute.Text -eq "Mute")
    try {
        $body = @{ muted = $target } | ConvertTo-Json -Compress
        $now = Invoke-RestMethod -Uri $script:muteUrl -Method Post -Body $body `
            -ContentType "application/json" -TimeoutSec 2
        if ($now.muted) { $tip = "Muted." } else { $tip = "Voice back on." }
        $script:notify.ShowBalloonTip(1200, $script:aiName, $tip, [System.Windows.Forms.ToolTipIcon]::None)
    } catch {
        $script:notify.ShowBalloonTip(1500, $script:aiName, "Server is not running.", [System.Windows.Forms.ToolTipIcon]::Info)
    }
})

$script:miHush.add_Click({
    # One-shot: stop the utterance playing now. Leaves mute + talk mode as-is.
    try {
        Invoke-RestMethod -Uri $script:stopUrl -Method Post -TimeoutSec 2 | Out-Null
        $script:notify.ShowBalloonTip(1000, $script:aiName, "Hushed.", [System.Windows.Forms.ToolTipIcon]::None)
    } catch {
        $script:notify.ShowBalloonTip(1500, $script:aiName, "Server is not running.", [System.Windows.Forms.ToolTipIcon]::Info)
    }
})

$script:miTalkMode.add_Click({
    # Target comes from the label the Popup handler just painted (Popup always
    # fires before the menu can show), same trick as Mute.
    $target = ($script:miTalkMode.Text -eq "Talk mode: off")
    try {
        $body = @{ enabled = $target } | ConvertTo-Json -Compress
        $now = Invoke-RestMethod -Uri $script:talkUrl -Method Post -Body $body `
            -ContentType "application/json" -TimeoutSec 2
        if ($now.enabled) { $tip = "Talk mode on -- she speaks every reply." } else { $tip = "Talk mode off." }
        $script:notify.ShowBalloonTip(1200, $script:aiName, $tip, [System.Windows.Forms.ToolTipIcon]::None)
    } catch {
        $script:notify.ShowBalloonTip(1500, $script:aiName, "Server is not running.", [System.Windows.Forms.ToolTipIcon]::Info)
    }
})

# Refresh the Mute + Talk-mode labels from live server state when the menu opens.
$menu.add_Popup({
    try {
        $state = Invoke-RestMethod -Uri $script:muteUrl -TimeoutSec 1
        if ($state.muted) { $script:miMute.Text = "Unmute" } else { $script:miMute.Text = "Mute" }
        $script:miMute.Enabled = $true
    } catch {
        $script:miMute.Text = "Mute (server off)"
        $script:miMute.Enabled = $false
    }
    try {
        $ts = Invoke-RestMethod -Uri $script:talkUrl -TimeoutSec 1
        if ($ts.enabled) { $script:miTalkMode.Text = "Talk mode: on" } else { $script:miTalkMode.Text = "Talk mode: off" }
        $script:miTalkMode.Enabled = $true
    } catch {
        $script:miTalkMode.Text = "Talk mode (server off)"
        $script:miTalkMode.Enabled = $false
    }
})

$script:miStop.add_Click({
    # Report what Stop-Enigma actually did -- "Stopped." when nothing was
    # running (or the port was foreign) is a lie the user acts on. It stops
    # THIS AI: the same pack and port this tray was started with.
    $out = (& "$script:engineDir\Stop-Enigma.ps1" -Persona $script:packDir -Port $script:bindPort) -join " "
    if ($out -match "FAILED") { $tip = "Could not stop $($script:aiName) -- run Stop Enigma from the Desktop to see why." }
    elseif ($out -match "left alone") { $tip = "Port $($script:bindPort) is not $($script:aiName) -- left it alone." }
    elseif ($out -match "already stopped" -and $out -match "none open") { $tip = "Nothing was running." }
    else { $tip = "Stopped." }
    $script:notify.ShowBalloonTip(1500, $script:aiName, $tip, [System.Windows.Forms.ToolTipIcon]::None)
})

$script:miExit.add_Click({
    [System.Windows.Forms.Application]::Exit()
})

try {
    [System.Windows.Forms.Application]::Run()
} finally {
    # Normal exit or exception: never leave a ghost icon in the tray.
    $script:notify.Visible = $false
    $script:notify.Dispose()
}
