@echo off

REM VaultVision DEMO — sample MP4 review (slow playback + angle-cut P-ID reset)

REM For live CCTV use start_rtsp.bat instead.

REM

REM Optional: set JEWELGUARD_VIDEO_PATH=C:\path\to\clip.mp4

set JEWELGUARD_RUNTIME=demo

set JEWELGUARD_MODE=video

set JEWELGUARD_VIDEO_PROFILE=near

set JEWELGUARD_VIDEO_PLAYBACK_SPEED=0.35

set JEWELGUARD_VIDEO_SCENE_CUT_RESET=1

call "%~dp0start_backend.bat"

