@echo off
setlocal enabledelayedexpansion
title Medical Imaging Screening Tool - Setup
color 0B

cd /d "%~dp0"

cls
echo.
echo   ######################################################
echo   #                                                    #
echo   #     MEDICAL IMAGING SCREENING TOOL                  #
echo   #     One-click installer for Windows                #
echo   #                                                    #
echo   ######################################################
echo.
echo   This sets up an isolated Python environment right here
echo   in this folder ^(nothing system-wide except Python itself
echo   if it isn't already installed^). Keep this folder --
echo   there's nothing to install elsewhere and nothing added
echo   to your Desktop or Start Menu. Next time, just run
echo   run.bat from here to open the app again.
echo.
echo   ------------------------------------------------------
echo.

rem This file is an automation script, not the app itself -- it needs
rem app.py, requirements.txt, the trained models, etc. sitting next to
rem it in the same folder to actually have anything to install.
rem Catching a standalone download of just the .bat file HERE, with a
rem clear explanation, beats failing confusingly partway through.
if not exist "%~dp0app.py" (
    echo   This copy of setup_and_run.bat is on its own, without the
    echo   rest of the project ^(app.py, requirements.txt, the trained
    echo   models, etc.^) that it needs to actually install.
    echo.
    echo   Fix: download the full project zip ^(the one this file came
    echo   in^), extract ALL of it, then run the setup_and_run.bat
    echo   that's inside that extracted folder.
    echo.
    pause
    exit /b 1
)

rem ---------------------------------------------------------------
rem [1/3] Python
rem ---------------------------------------------------------------
echo   [1/3] Checking for Python...

rem Windows ships a fake "python" App Execution Alias on PATH even
rem when Python isn't installed -- it prints a Microsoft Store message
rem instead of a version number, so checking that "python" merely
rem exists on PATH gives a false positive. Checking the actual version
rem output is the real test.
rem
rem The "py" launcher (installed alongside every official/winget Python)
rem is checked FIRST because it is immune to this problem -- it lives in
rem C:\Windows\ and is never shadowed by the App Execution Alias, unlike
rem the bare "python"/"python3" commands. This matters because a
rem previously-installed Python can still be invisible to "python" if
rem the Store alias happens to sit earlier on PATH -- "py" sidesteps
rem that entirely.
set PYTHON_OK=0
set PYCMD=

for /f "delims=" %%v in ('py -3 --version 2^>^&1') do (
    echo %%v | findstr /b /c:"Python 3" >nul && (set PYTHON_OK=1& set "PYCMD=py -3")
)
if "%PYTHON_OK%"=="0" (
    for /f "delims=" %%v in ('python --version 2^>^&1') do (
        echo %%v | findstr /b /c:"Python 3" >nul && (set PYTHON_OK=1& set "PYCMD=python")
    )
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
    echo   double-click setup_and_run.bat again.
    echo.
    echo   If it still says "not found" after that: Windows has a
    echo   built-in "python" shortcut that can hide a real install.
    echo   Go to Settings, search "App execution aliases", and turn
    echo   OFF the switches next to python.exe / python3.exe, then
    echo   run this file once more.
    echo   ------------------------------------------------------
    echo.
    pause
    exit /b 0
)
echo         Python found and working ^(using "%PYCMD%"^).
echo.

rem ---------------------------------------------------------------
rem [2/3] Virtual environment
rem ---------------------------------------------------------------
echo   [2/3] Setting up an isolated environment...
if not exist venv (
    echo         Creating venv\ ...
    %PYCMD% -m venv venv
) else (
    echo         Already exists, reusing it.
)
echo.

rem ---------------------------------------------------------------
rem [3/3] Dependencies
rem ---------------------------------------------------------------
echo   [3/3] Installing dependencies...
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

echo   ------------------------------------------------------
echo    Setup done. Keep this folder -- there's nothing
echo    installed anywhere else. Next time, just run run.bat
echo    from here to open the app again, no need to redo setup.
echo   ------------------------------------------------------
echo.
echo   Opening it now for the first time...
echo.

rem "start" here (not "call") launches run.bat as its own independent
rem process rather than continuing inline in this one -- more robust
rem regardless of how THIS window itself was opened.
start "" /d "%~dp0" run.bat
