@echo off
setlocal enabledelayedexpansion
title Medical Imaging Screening Tool

cd /d "%~dp0"

if not exist venv (
    echo.
    echo   No environment found in this folder:
    echo       %~dp0
    echo.
    echo   Run setup_and_run.bat from this exact folder first ^(just
    echo   once^), then use run.bat from here every time after that.
    echo.
    pause
    exit /b 1
)

netstat -ano | findstr ":8501" | findstr "LISTENING" >nul 2>nul
if not errorlevel 1 goto :open_window

start "Medical Imaging Screening Tool - SERVER (keep open, closing this stops the app)" /min cmd /c "venv\Scripts\streamlit.exe run app.py --server.headless true --server.port 8501"

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
