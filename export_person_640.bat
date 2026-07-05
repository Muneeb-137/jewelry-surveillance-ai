@echo off
REM Export 640px OpenVINO person model (video/RTSP — better crowd detection)
cd /d "%~dp0"
python ml\models\export_person_640.py
pause
