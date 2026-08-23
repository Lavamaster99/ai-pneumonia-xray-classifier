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
rem [2/4] Virtual environment
rem ---------------------------------------------------------------
echo   [2/4] Setting up an isolated environment...
if not exist venv (
    echo         Creating venv\ ...
    %PYCMD% -m venv venv
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
rem [4/4] Desktop shortcut, so this full setup only ever runs once
rem ---------------------------------------------------------------
echo   [4/4] Creating a desktop shortcut...

set SHORTCUT_NAME=Pneumonia Screening Tool.lnk

rem Desktop is a "Known Folder" that can be redirected (e.g. into
rem OneDrive) -- %USERPROFILE%\Desktop in plain batch doesn't know
rem about that redirection, but PowerShell's GetFolderPath does, so
rem PowerShell resolves the real path AND reports success itself
rem rather than batch re-guessing where the file landed.
for /f "delims=" %%r in ('powershell -NoProfile -Command ^
    "try {" ^
    "  $s = (New-Object -ComObject WScript.Shell).CreateShortcut([System.IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), '%SHORTCUT_NAME%'));" ^
    "  $s.TargetPath = (Join-Path '%~dp0' 'run.bat');" ^
    "  $s.WorkingDirectory = '%~dp0';" ^
    "  $s.WindowStyle = 7;" ^
    "  $s.IconLocation = 'shell32.dll,167';" ^
    "  $s.Description = 'Chest X-Ray Pneumonia Screening Tool';" ^
    "  $s.Save();" ^
    "  Write-Output 'OK'" ^
    "} catch { Write-Output 'FAIL' }"') do set SHORTCUT_RESULT=%%r

if "%SHORTCUT_RESULT%"=="OK" (
    echo         Done -- look for "%SHORTCUT_NAME%" on your Desktop.
) else (
    echo         Couldn't create it automatically, but that's fine --
    echo         run.bat in this folder does the same thing.
)
echo.

echo   ------------------------------------------------------
echo    Setup is done and only needs to happen once.
echo    From now on, use the Desktop shortcut to open the app
echo    directly -- no need to run this setup file again.
echo   ------------------------------------------------------
echo.
echo   Opening it now for the first time...
echo.

call "%~dp0run.bat"
