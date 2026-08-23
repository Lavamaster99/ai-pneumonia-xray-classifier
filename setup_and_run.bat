@echo off
setlocal enabledelayedexpansion
title Chest X-Ray Pneumonia Screening Tool - Setup
color 0B

cd /d "%~dp0"

cls
echo.
echo   ######################################################
echo   #                                                    #
echo   #     CHEST X-RAY PNEUMONIA SCREENING TOOL            #
echo   #     One-click setup for Windows                    #
echo   #                                                    #
echo   ######################################################
echo.
echo   This will check for Python, set up an isolated
echo   environment, install everything needed, and launch
echo   the dashboard in your browser. Nothing is installed
echo   outside this project folder except Python itself
echo   (if it isn't already on your system).
echo.
echo   ------------------------------------------------------
echo.

rem ---------------------------------------------------------------
rem [1/4] Python
rem ---------------------------------------------------------------
echo   [1/4] Checking for Python...

rem Windows ships a fake "python" App Execution Alias on PATH even
rem when Python isn't installed -- it prints a Microsoft Store
rem message instead of a version number, so checking that "python"
rem merely exists on PATH gives a false positive. Checking the
rem actual version output is the real test.
set PYTHON_OK=0
for /f "delims=" %%v in ('python --version 2^>^&1') do (
    echo %%v | findstr /b /c:"Python 3" >nul && set PYTHON_OK=1
)

if "%PYTHON_OK%"=="0" (
    echo         Python not found on this system.
    where winget >nul 2>nul
    if errorlevel 1 (
        echo.
        echo   Could not install Python automatically ^(winget not found^).
        echo   Please install Python 3.10+ yourself:
        echo       https://www.python.org/downloads/
        echo   Tick "Add python.exe to PATH" during install, then run this
        echo   file again.
        echo.
        pause
        exit /b 1
    )
    echo         Installing Python via winget -- this may take a few minutes...
    echo.
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo.
        echo   Automatic install failed. Please install Python manually:
        echo       https://www.python.org/downloads/
        echo   ^(tick "Add python.exe to PATH"^), then run this file again.
        echo.
        pause
        exit /b 1
    )
    echo.
    echo   ------------------------------------------------------
    echo   Python is installed. Please CLOSE this window and
    echo   double-click setup_and_run.bat again -- this refreshes
    echo   your PATH so Windows can find the new install.
    echo   ------------------------------------------------------
    echo.
    pause
    exit /b 0
)
echo         Python found and working.
echo.

rem ---------------------------------------------------------------
rem [2/4] Virtual environment
rem ---------------------------------------------------------------
echo   [2/4] Setting up an isolated environment...
if not exist venv (
    echo         Creating venv\ ...
    python -m venv venv
) else (
    echo         Already exists, reusing it.
)
echo.

rem ---------------------------------------------------------------
rem [3/4] Dependencies
rem ---------------------------------------------------------------
echo   [3/4] Installing dependencies...
echo         ^(first run only -- TensorFlow is a large download,
echo          this can take several minutes^)
echo.
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo   Something went wrong installing dependencies.
    echo   Scroll up to see the actual error message.
    echo.
    pause
    exit /b 1
)
echo.
echo         Done.
echo.

rem Skip Streamlit's first-run "enter your email" prompt, which would
rem otherwise block here waiting for input on a brand-new install.
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    echo [general] > "%USERPROFILE%\.streamlit\credentials.toml"
    echo email = "" >> "%USERPROFILE%\.streamlit\credentials.toml"
)

rem ---------------------------------------------------------------
rem [4/4] Launch
rem ---------------------------------------------------------------
echo   [4/4] Launching the dashboard...
echo.
echo   ------------------------------------------------------
echo    Setup complete. Opening in your browser now.
echo    To stop the app later, just close this window.
echo   ------------------------------------------------------
echo.
streamlit run app.py

pause
