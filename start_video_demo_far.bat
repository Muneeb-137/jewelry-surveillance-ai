@echo off

REM VaultVision DEMO — far profile (distant / small faces in sample MP4s)

set JEWELGUARD_RUNTIME=demo

set JEWELGUARD_MODE=video

set JEWELGUARD_VIDEO_PROFILE=far

set JEWELGUARD_VIDEO_PLAYBACK_SPEED=0.35

set JEWELGUARD_VIDEO_SCENE_CUT_RESET=1

call "%~dp0start_backend.bat"

