@echo off
REM VaultVision LIVE — RTSP / IP camera (real-time, no angle-cut reset)
REM Set your camera URL before running:
REM   set JEWELGUARD_RTSP_URL=rtsp://user:pass@192.168.1.50:554/stream1
if "%JEWELGUARD_RTSP_URL%"=="" (
  echo ERROR: Set JEWELGUARD_RTSP_URL to your camera RTSP URL first.
  echo Example: set JEWELGUARD_RTSP_URL=rtsp://admin:pass@192.168.1.50/stream1
  pause
  exit /b 1
)
set JEWELGUARD_RUNTIME=live
set JEWELGUARD_MODE=rtsp
set JEWELGUARD_VIDEO_PROFILE=near
call "%~dp0start_backend.bat"
