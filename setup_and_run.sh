#!/bin/bash
# Medical Imaging Screening Tool -- one-click installer for macOS and Linux.
# Run this once (bash setup_and_run.sh, or ./setup_and_run.sh after a
# chmod +x). It installs the app under your user profile, sets up an
# isolated Python environment, adds a launcher you can use from then on,
# and opens the dashboard.

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
echo "  This installs the app under your user profile, sets it up,"
echo "  and adds a launcher. Once it's done, you can delete whatever"
echo "  folder you downloaded/extracted this from -- the real"
echo "  install lives elsewhere from here on."
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
# [1/5] Python
# -----------------------------------------------------------------------
echo "  [1/5] Checking for Python..."

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
# [2/5] Copy the app to a permanent location
# -----------------------------------------------------------------------
echo "  [2/5] Installing..."

if [ "$OS_NAME" = "Darwin" ]; then
    INSTALL_DIR="$HOME/Library/Application Support/MedicalScreeningTool"
else
    INSTALL_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/MedicalScreeningTool"
fi
SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ "$SOURCE_DIR" = "$INSTALL_DIR" ]; then
    echo "        Already running from the installed location."
else
    echo "        Copying app files to:"
    echo "            $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
    # rsync mirrors the source in, skipping dev-only directories that
    # don't belong in an end-user install.
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete \
            --exclude venv --exclude data --exclude site \
            --exclude .git --exclude .agents --exclude .claude \
            --exclude __pycache__ --exclude '*.pyc' \
            "$SOURCE_DIR/" "$INSTALL_DIR/"
    else
        cp -R "$SOURCE_DIR/." "$INSTALL_DIR/"
        rm -rf "$INSTALL_DIR/venv" "$INSTALL_DIR/data" "$INSTALL_DIR/site" \
               "$INSTALL_DIR/.git" "$INSTALL_DIR/.agents" "$INSTALL_DIR/.claude"
        find "$INSTALL_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
        find "$INSTALL_DIR" -name "*.pyc" -delete
    fi
    if [ ! -f "$INSTALL_DIR/app.py" ]; then
        echo ""
        echo "  Copying the app files failed. Nothing was installed."
        echo ""
        read -r -p "  Press Enter to close..."
        exit 1
    fi
    echo "        Done. Once setup finishes, it's safe to delete the"
    echo "        folder you downloaded/extracted this from -- the real"
    echo "        install now lives at the path above."
fi
echo ""

cd "$INSTALL_DIR"

# -----------------------------------------------------------------------
# [3/5] Virtual environment
# -----------------------------------------------------------------------
echo "  [3/5] Setting up an isolated environment..."
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
# [4/5] Dependencies
# -----------------------------------------------------------------------
echo "  [4/5] Installing dependencies..."
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

# -----------------------------------------------------------------------
# [5/5] Launcher, so this full setup only ever runs once
# -----------------------------------------------------------------------
echo "  [5/5] Creating a launcher..."

if [ "$OS_NAME" = "Darwin" ]; then
    DESKTOP_DIR="$HOME/Desktop"
    LAUNCHER="$DESKTOP_DIR/Medical Screening Tool.command"
    mkdir -p "$DESKTOP_DIR"
    cat > "$LAUNCHER" <<EOF
#!/bin/bash
exec "$INSTALL_DIR/run.sh"
EOF
    chmod +x "$LAUNCHER"
    echo "        Done -- look for \"Medical Screening Tool.command\" on"
    echo "        your Desktop. (First double-click: macOS will warn it's"
    echo "        from an unidentified developer -- right-click it, choose"
    echo "        Open, then confirm. Only needed once.)"
else
    APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    mkdir -p "$APPS_DIR"
    DESKTOP_FILE="$APPS_DIR/medical-screening-tool.desktop"
    cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=Medical Imaging Screening Tool
Comment=Chest X-ray pneumonia and breast ultrasound screening -- CNN + Grad-CAM
Exec="$INSTALL_DIR/run.sh"
Terminal=false
Categories=Science;Education;
EOF
    chmod +x "$DESKTOP_FILE"
    # Also drop a copy on the Desktop, if there is one -- most desktop
    # environments require right-click -> "Allow Launching" (or similar)
    # the first time a .desktop file is run from there, which is a
    # standard Linux trust prompt this script can't skip.
    if [ -d "$HOME/Desktop" ]; then
        cp "$DESKTOP_FILE" "$HOME/Desktop/"
        chmod +x "$HOME/Desktop/medical-screening-tool.desktop"
    fi
    echo "        Done -- added to your applications menu as \"Medical"
    echo "        Imaging Screening Tool\" (and to your Desktop, if you"
    echo "        have one -- first launch from there may need right-click"
    echo "        -> Allow Launching, a standard Linux security prompt for"
    echo "        new .desktop files)."
fi
echo ""

echo "  --------------------------------------------------------"
echo "   Installed to: $INSTALL_DIR"
echo "   Setup only needs to happen once. From now on, use the"
echo "   launcher created above -- and you can delete the folder"
echo "   you downloaded this from, if you haven't already."
echo "  --------------------------------------------------------"
echo ""
echo "  Opening it now for the first time..."
echo ""

exec "$INSTALL_DIR/run.sh"
