@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\build_portable.ps1" %*
exit /b %ERRORLEVEL%
