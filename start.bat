@echo off
title CC Team Chat
cd /d "%~dp0"

echo Checking port 8000...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr " 0.0.0.0:8000 "') do (
    echo.
    echo  Port 8000 is already in use by process PID %%a.
    echo.
    echo  To stop the old instance, run ONE of these:
    echo    taskkill /PID %%a /F
    echo    OR close the terminal window where CC Team Chat is running.
    echo.
    pause
    exit /b 1
)

echo Starting CC Team Chat...
echo Open your browser at http://localhost:8000
echo.
echo Usage:
echo   start.bat                  ^(uses .\userdata as project^)
echo   start.bat C:\my\project    ^(use any folder as project^)
echo.
set PROJECT=%~1
if "%PROJECT%"=="" set PROJECT=userdata
"%LOCALAPPDATA%\Programs\Python\Python314\python.exe" app.py --project "%PROJECT%"
pause
