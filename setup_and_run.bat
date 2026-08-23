@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  Chest X-Ray Pneumonia Screening Tool -- one-click setup
echo ============================================================
echo.

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on this computer.
    where winget >nul 2>nul
    if errorlevel 1 (
        echo Could not find winget either, so Python can't be installed automatically.
        echo Please install Python 3.10+ from https://www.python.org/downloads/
        echo ^(tick "Add python.exe to PATH" during install^), then run this file again.
        pause
        exit /b 1
    )
    echo Installing Python automatically via winget -- this may take a few minutes...
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo Automatic install failed. Please install Python manually from
        echo https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^), then run this file again.
        pause
        exit /b 1
    )
    echo Python installed. Please close this window and double-click setup_and_run.bat again
    echo ^(this refreshes your PATH so Windows can find the new Python install^).
    pause
    exit /b 0
)

if not exist venv (
    echo Creating a virtual environment...
    python -m venv venv
)

echo Installing dependencies ^(first run only takes a few minutes -- TensorFlow is a large download^)...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Something went wrong installing dependencies. Scroll up to see the error.
    pause
    exit /b 1
)

if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    echo [general] > "%USERPROFILE%\.streamlit\credentials.toml"
    echo email = "" >> "%USERPROFILE%\.streamlit\credentials.toml"
)

echo.
echo Setup complete. Launching the dashboard -- your browser should open automatically...
echo ^(To stop it later, close this window.^)
echo.
streamlit run app.py

pause
