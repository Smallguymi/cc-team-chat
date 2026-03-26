@echo off
title CC Team Chat
echo Starting CC Team Chat...
echo Open your browser at http://localhost:8000
echo.
echo Usage:
echo   start.bat                  ^(uses .\userdata as project^)
echo   start.bat C:\my\project    ^(use any folder as project^)
echo.
cd /d "%~dp0"
set PROJECT=%~1
if "%PROJECT%"=="" set PROJECT=userdata
"%LOCALAPPDATA%\Programs\Python\Python314\python.exe" app.py --project "%PROJECT%"
pause
