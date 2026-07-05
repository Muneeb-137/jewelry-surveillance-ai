# JewelGuard backend — frees port 8000, then starts uvicorn.
# Usage:
#   .\start_backend.ps1              (uses DEFAULT_MODE in source_config.py)
#   .\start_backend.ps1 -Mode webcam
#   .\start_backend.ps1 -Mode video
# Or double-click: start_webcam.bat / start_video.bat

param(
    [ValidateSet("webcam", "video", "")]
    [string]$Mode = ""
)

$ErrorActionPreference = "SilentlyContinue"
$Port = 8000
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if ($Mode) {
    $env:JEWELGUARD_MODE = $Mode
} elseif (-not $env:JEWELGUARD_MODE) {
    $env:JEWELGUARD_MODE = "webcam"
}

$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listeners) {
    $listeners | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
        Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped old server on port $Port (PID $_)"
    }
    Start-Sleep -Seconds 1
}

Write-Host "JewelGuard mode: $env:JEWELGUARD_MODE"
$runtime = if ($env:JEWELGUARD_RUNTIME) { $env:JEWELGUARD_RUNTIME } else { if ($env:JEWELGUARD_MODE -eq 'video') { 'demo' } else { 'live' } }
Write-Host "Runtime profile: $runtime (demo=MP4 review, live=CCTV/webcam)"
if ($env:JEWELGUARD_MODE -eq "video") {
    $profile = if ($env:JEWELGUARD_VIDEO_PROFILE) { $env:JEWELGUARD_VIDEO_PROFILE } else { "near" }
    Write-Host "Video profile:   $profile (near=store/close, far=distant)"
    if ($runtime -eq "demo") {
        $speed = if ($env:JEWELGUARD_VIDEO_PLAYBACK_SPEED) { $env:JEWELGUARD_VIDEO_PLAYBACK_SPEED } else { "0.35" }
        Write-Host "Playback speed:  ${speed}x (demo slow-mo)"
        Write-Host "Scene-cut reset: on (multi-angle MP4s)"
    } else {
        Write-Host "Playback speed:  1.0x (real-time)"
        Write-Host "Scene-cut reset: off (static camera logic)"
    }
}
Write-Host "Settings file:   backend\source_config.py"
Write-Host "Backend:         http://127.0.0.1:$Port"
Write-Host "Press Ctrl+C to stop.`n"
python -m uvicorn backend.app:app --host 127.0.0.1 --port $Port
