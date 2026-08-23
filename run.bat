@echo off
setlocal enabledelayedexpansion
title Pneumonia Screening Tool

cd /d "%~dp0"

if not exist venv (
    echo.
    echo   No environment found yet. Run setup_and_run.bat first
    echo   ^(just once^) to install everything, then use this
    echo   shortcut from now on.
    echo.
    pause
    exit /b 1
)

netstat -ano | findstr ":8501" | findstr "LISTENING" >nul 2>nul
if not errorlevel 1 goto :open_window

start "Pneumonia Screening Tool - SERVER (keep open, closing this stops the app)" /min cmd /c "venv\Scripts\streamlit.exe run app.py --server.headless true --server.port 8501"

set READY=0
for /l %%i in (1,1,45) do (
    if "!READY!"=="0" (
        powershell -NoProfile -Command "try { (New-Object Net.Sockets.TcpClient('localhost',8501)).Close(); exit 0 } catch { exit 1 }" >nul 2>nul
        if not errorlevel 1 set READY=1
    )
    if "!READY!"=="0" timeout /t 1 /nobreak >nul
)

if "%READY%"=="0" (
    echo   The app is taking longer than expected to start.
    echo   Try opening http://localhost:8501 yourself in a moment.
    pause
    exit /b 1
)

:open_window
start "" msedge --app=http://localhost:8501 --window-size=1200,860
exit /b 0
