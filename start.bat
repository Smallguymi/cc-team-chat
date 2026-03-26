@echo off
title CC Team Chat
echo Starting CC Team Chat...
echo Open your browser at http://localhost:8000
echo Press Ctrl+C to stop.
echo.
cd /d "%~dp0"
"%LOCALAPPDATA%\Programs\Python\Python314\python.exe" app.py
pause
