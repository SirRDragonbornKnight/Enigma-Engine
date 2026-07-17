# Enigma-Tray.ps1 -- Enigma lives in the system tray. Right-click the icon:
# Talk to Enigma / Mute / Stop Enigma / Exit tray icon. Double-click = Talk.
# One instance only (named mutex). Zero dependencies: WinForms NotifyIcon,
# icon drawn at runtime (dark disc, purple E) so no asset file to lose.
# Launched hidden by "Enigma Tray.bat". ASCII-only (Windows cp1252 console).

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$script:engineDir = "C:\Users\SirKn\Enigma Engine"
$script:muteUrl = "http://127.0.0.1:8000/v1/audio/mute"

# One tray icon, ever. A second launch exits quietly.
$created = $false
$script:mutex = New-Object System.Threading.Mutex($true, "Local\EnigmaTray", [ref]$created)
if (-not $created) { exit 0 }

# -- icon: dark disc, her purple E, drawn in memory --
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
$g.DrawString("E", $font, $fg, (New-Object System.Drawing.RectangleF(0, 0, 32, 32)), $fmt)
$g.Dispose()

$script:notify = New-Object System.Windows.Forms.NotifyIcon
$script:notify.Icon = [System.Drawing.Icon]::FromHandle($bmp.GetHicon())
$script:notify.Text = "Enigma"
$script:notify.Visible = $true

$menu = New-Object System.Windows.Forms.ContextMenu
$script:miTalk = New-Object System.Windows.Forms.MenuItem "Talk to Enigma"
$script:miMute = New-Object System.Windows.Forms.MenuItem "Mute"
$script:miStop = New-Object System.Windows.Forms.MenuItem "Stop Enigma"
$script:miExit = New-Object System.Windows.Forms.MenuItem "Exit tray icon"
$sep = New-Object System.Windows.Forms.MenuItem "-"
$menu.MenuItems.AddRange(@($script:miTalk, $script:miMute, $sep, $script:miStop, $script:miExit))
$script:notify.ContextMenu = $menu

$talk = {
    Start-Process -FilePath "cmd.exe" `
        -ArgumentList "/c", "`"$script:engineDir\Talk to Enigma.bat`"" `
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
        $script:notify.ShowBalloonTip(1200, "Enigma", $tip, [System.Windows.Forms.ToolTipIcon]::None)
    } catch {
        $script:notify.ShowBalloonTip(1500, "Enigma", "Server is not running.", [System.Windows.Forms.ToolTipIcon]::Info)
    }
})

# Refresh the Mute label from live server state each time the menu opens.
$menu.add_Popup({
    try {
        $state = Invoke-RestMethod -Uri $script:muteUrl -TimeoutSec 1
        if ($state.muted) { $script:miMute.Text = "Unmute" } else { $script:miMute.Text = "Mute" }
        $script:miMute.Enabled = $true
    } catch {
        $script:miMute.Text = "Mute (server off)"
        $script:miMute.Enabled = $false
    }
})

$script:miStop.add_Click({
    # Report what Stop-Enigma actually did -- "Stopped." when nothing was
    # running (or the port was foreign) is a lie the user acts on.
    $out = (& "$script:engineDir\Stop-Enigma.ps1") -join " "
    if ($out -match "left alone") { $tip = "Port 8000 is not Enigma -- left it alone." }
    elseif ($out -match "already stopped" -and $out -match "none open") { $tip = "Nothing was running." }
    else { $tip = "Stopped." }
    $script:notify.ShowBalloonTip(1500, "Enigma", $tip, [System.Windows.Forms.ToolTipIcon]::None)
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
