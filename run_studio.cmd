@echo off
setlocal
cd /d "%~dp0"
python background_correction_studio.py
exit /b %ERRORLEVEL%
