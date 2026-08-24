#!/bin/bash
# Medical Imaging Screening Tool -- one-click installer for macOS and Linux.
# Run this once (bash setup_and_run.sh, or ./setup_and_run.sh after a
# chmod +x). It sets up an isolated Python environment right here in this
# folder and opens the dashboard. Keep the folder -- there's nothing
# installed elsewhere. Next time, just run run.sh from here again.

set -u
cd "$(dirname "$0")"

echo ""
echo "  ########################################################"
echo "  #                                                      #"
echo "  #     MEDICAL IMAGING SCREENING TOOL                   #"
echo "  #     One-click installer for macOS / Linux            #"
echo "  #                                                      #"
echo "  ########################################################"
echo ""
echo "  This sets up an isolated Python environment right here in"
echo "  this folder. Keep this folder -- there's nothing to install"
echo "  elsewhere and nothing added to your Desktop or applications"
echo "  menu. Next time, just run run.sh from here to open the app"
echo "  again."
echo ""
echo "  --------------------------------------------------------"
echo ""

# This file is an automation script, not the app itself -- it needs
# app.py, requirements.txt, the trained models, etc. sitting next to it
# in the same folder to actually have anything to install. Catching a
# standalone download of just this script here, with a clear
# explanation, beats failing confusingly partway through.
if [ ! -f "$(dirname "$0")/app.py" ]; then
    echo "  This copy of setup_and_run.sh is on its own, without the rest"
    echo "  of the project (app.py, requirements.txt, the trained models,"
    echo "  etc.) that it needs to actually install."
    echo ""
    echo "  Fix: download the full project zip (the one this file came"
    echo "  in), extract ALL of it, then run the setup_and_run.sh that's"
    echo "  inside that extracted folder."
    echo ""
    read -r -p "  Press Enter to close..."
    exit 1
fi

OS_NAME="$(uname -s)"

# -----------------------------------------------------------------------
# [1/3] Python
# -----------------------------------------------------------------------
echo "  [1/3] Checking for Python..."

PYCMD=""
if command -v python3 >/dev/null 2>&1 && python3 --version 2>&1 | grep -q "^Python 3"; then
    PYCMD="python3"
elif command -v python >/dev/null 2>&1 && python --version 2>&1 | grep -q "^Python 3"; then
    PYCMD="python"
fi

if [ -z "$PYCMD" ]; then
    echo "        Python 3 not found on this system."
    echo ""
    if [ "$OS_NAME" = "Darwin" ]; then
        echo "  Install Python 3 first, then run this script again:"
        echo "      - Easiest: install Homebrew (https://brew.sh), then run"
        echo "        \"brew install python3\""
        echo "      - Or download the installer from https://www.python.org/downloads/"
    else
        echo "  Install Python 3 first, then run this script again. On"
        echo "  most Linux distros:"
        echo "      Debian / Ubuntu:  sudo apt install python3 python3-venv python3-pip"
        echo "      Fedora:           sudo dnf install python3"
        echo "      Arch:             sudo pacman -S python"
    fi
    echo ""
    read -r -p "  Press Enter to close..."
    exit 1
fi
echo "        Python found and working (using \"$PYCMD\")."
echo ""

# -----------------------------------------------------------------------
# [2/3] Virtual environment
# -----------------------------------------------------------------------
echo "  [2/3] Setting up an isolated environment..."
if [ ! -d venv ]; then
    echo "        Creating venv/ ..."
    "$PYCMD" -m venv venv
    if [ ! -f venv/bin/python ]; then
        echo ""
        echo "  Creating the virtual environment failed. On Debian/Ubuntu"
        echo "  this usually means the venv module isn't installed:"
        echo "      sudo apt install python3-venv"
        echo "  then run this script again."
        echo ""
        read -r -p "  Press Enter to close..."
        exit 1
    fi
else
    echo "        Already exists, reusing it."
fi
echo ""

# -----------------------------------------------------------------------
# [3/3] Dependencies
# -----------------------------------------------------------------------
echo "  [3/3] Installing dependencies..."
echo "        (first run only -- TensorFlow is a large download, this"
echo "         can take several minutes)"
echo ""
venv/bin/python -m pip install --upgrade pip >/dev/null
if ! venv/bin/pip install -r requirements.txt; then
    echo ""
    echo "  Something went wrong installing dependencies."
    echo "  Scroll up to see the actual error message."
    echo ""
    read -r -p "  Press Enter to close..."
    exit 1
fi
echo ""
echo "        Done."
echo ""

# Skip Streamlit's first-run "enter your email" prompt, which would
# otherwise block here waiting for input on a brand-new install.
mkdir -p "$HOME/.streamlit"
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
    printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

echo "  --------------------------------------------------------"
echo "   Setup done. Keep this folder -- there's nothing installed"
echo "   anywhere else. Next time, just run run.sh from here to"
echo "   open the app again, no need to redo setup."
echo "  --------------------------------------------------------"
echo ""
echo "  Opening it now for the first time..."
echo ""

exec "$(pwd)/run.sh"
