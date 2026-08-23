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
echo   This installs the app to your PC ^(under your user
echo   folder, nothing system-wide except Python itself if
echo   it isn't already installed^), sets it up, and adds a
echo   desktop shortcut. Once it's done, you can delete
echo   whatever folder you downloaded/extracted this from --
echo   the real install lives elsewhere from here on.
echo.
echo   ------------------------------------------------------
echo.

rem This file is an automation script, not the app itself -- it needs
rem app.py, requirements.txt, the trained model, etc. sitting next to
rem it in the same folder to actually have anything to install.
rem Catching a standalone download of just the .bat file HERE, with a
rem clear explanation, beats failing confusingly partway through.
if not exist "%~dp0app.py" (
    echo   This copy of setup_and_run.bat is on its own, without the
    echo   rest of the project ^(app.py, requirements.txt, the trained
    echo   model, etc.^) that it needs to actually install.
    echo.
    echo   Fix: download the full project zip ^(the one this file came
    echo   in^), extract ALL of it, then run the setup_and_run.bat
    echo   that's inside that extracted folder.
    echo.
    pause
    exit /b 1
)

rem ---------------------------------------------------------------
rem [1/5] Python
rem ---------------------------------------------------------------
echo   [1/5] Checking for Python...

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
rem [2/5] Copy the app to a permanent location
rem ---------------------------------------------------------------
echo   [2/5] Installing to your PC...

set "INSTALL_DIR=%LOCALAPPDATA%\MedicalScreeningTool"
set "SOURCE_DIR=%~dp0"
if "%SOURCE_DIR:~-1%"=="\" set "SOURCE_DIR=%SOURCE_DIR:~0,-1%"

if /i "%SOURCE_DIR%"=="%INSTALL_DIR%" (
    echo         Already running from the installed location.
) else (
    echo         Copying app files to:
    echo             %INSTALL_DIR%
    robocopy "%SOURCE_DIR%" "%INSTALL_DIR%" /E /XD venv data site .git .agents .claude __pycache__ /XF *.pyc >nul
    if not exist "%INSTALL_DIR%\app.py" (
        echo.
        echo   Copying the app files failed. Nothing was installed.
        echo   Try running this as Administrator, or check that
        echo   %INSTALL_DIR% isn't blocked by another program.
        echo.
        pause
        exit /b 1
    )
    echo         Done. Once setup finishes, it's safe to delete the
    echo         folder you downloaded/extracted this from -- the
    echo         real install now lives at the path above.
)
echo.

cd /d "%INSTALL_DIR%"

rem ---------------------------------------------------------------
rem [3/5] Virtual environment
rem ---------------------------------------------------------------
echo   [3/5] Setting up an isolated environment...
if not exist venv (
    echo         Creating venv\ ...
    %PYCMD% -m venv venv
) else (
    echo         Already exists, reusing it.
)
echo.

rem ---------------------------------------------------------------
rem [4/5] Dependencies
rem ---------------------------------------------------------------
echo   [4/5] Installing dependencies...
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
rem [5/5] Desktop shortcut, so this full setup only ever runs once
rem ---------------------------------------------------------------
echo   [5/5] Creating a desktop shortcut...

set SHORTCUT_NAME=Medical Screening Tool.lnk

rem Desktop is a "Known Folder" that can be redirected (e.g. into
rem OneDrive) -- %USERPROFILE%\Desktop in plain batch doesn't know
rem about that redirection, but PowerShell's GetFolderPath does, so
rem PowerShell resolves the real path AND reports success itself
rem rather than batch re-guessing where the file landed.
for /f "delims=" %%r in ('powershell -NoProfile -Command ^
    "try {" ^
    "  $s = (New-Object -ComObject WScript.Shell).CreateShortcut([System.IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), '%SHORTCUT_NAME%'));" ^
    "  $s.TargetPath = (Join-Path '%INSTALL_DIR%' 'run.bat');" ^
    "  $s.WorkingDirectory = '%INSTALL_DIR%';" ^
    "  $s.WindowStyle = 7;" ^
    "  $s.IconLocation = 'shell32.dll,167';" ^
    "  $s.Description = 'Medical Imaging Screening Tool';" ^
    "  $s.Save();" ^
    "  Write-Output 'OK'" ^
    "} catch { Write-Output 'FAIL' }"') do set SHORTCUT_RESULT=%%r

if "%SHORTCUT_RESULT%"=="OK" (
    echo         Done -- look for "%SHORTCUT_NAME%" on your Desktop.
) else (
    echo         Couldn't create it automatically, but that's fine --
    echo         run.bat at %INSTALL_DIR% does the same thing.
)
echo.

echo   ------------------------------------------------------
echo    Installed to: %INSTALL_DIR%
echo    Setup only needs to happen once. From now on, use the
echo    Desktop shortcut -- and you can delete the folder you
echo    downloaded this from, if you haven't already.
echo   ------------------------------------------------------
echo.
echo   Opening it now for the first time...
echo.

rem "start" here (not "call") launches run.bat as its own independent
rem process rather than continuing inline in this one -- more robust
rem regardless of how THIS window itself was opened, and it means this
rem installer window doesn't have to stay alive for the app to run.
start "" /d "%INSTALL_DIR%" run.bat
