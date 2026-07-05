@echo off

REM VaultVision LIVE — test an MP4 at real-time (no angle-cut reset)

REM Same tracking as CCTV; useful for single-angle test files.

REM   set JEWELGUARD_VIDEO_PATH=C:\path\to\clip.mp4

set JEWELGUARD_RUNTIME=live

set JEWELGUARD_MODE=video

set JEWELGUARD_VIDEO_PROFILE=near

set JEWELGUARD_VIDEO_PLAYBACK_SPEED=1.0

set JEWELGUARD_VIDEO_SCENE_CUT_RESET=0

call "%~dp0start_backend.bat"

