#!/bin/bash
# Lightweight launcher for every run after the first setup_and_run.sh.
# Starts the Streamlit server if it isn't already running, waits for it
# to come up, then opens the dashboard in its own window.

cd "$(dirname "$0")"

if [ ! -d venv ]; then
    echo ""
    echo "  No environment found in this folder:"
    echo "      $(pwd)"
    echo ""
    echo "  Run setup_and_run.sh from this exact folder first (just"
    echo "  once), then use this launcher from now on."
    echo ""
    read -r -p "  Press Enter to close..."
    exit 1
fi

PORT=8501

already_running() {
    if command -v curl >/dev/null 2>&1; then
        curl -s -o /dev/null "http://localhost:$PORT" && return 0
    fi
    if command -v nc >/dev/null 2>&1; then
        nc -z localhost "$PORT" 2>/dev/null && return 0
    fi
    return 1
}

if ! already_running; then
    nohup venv/bin/streamlit run app.py \
        --server.headless true --server.port "$PORT" \
        >/tmp/medical-screening-tool.log 2>&1 &

    READY=0
    for _ in $(seq 1 45); do
        if already_running; then
            READY=1
            break
        fi
        sleep 1
    done

    if [ "$READY" = "0" ]; then
        echo "  The app is taking longer than expected to start."
        echo "  Try opening http://localhost:$PORT yourself in a moment."
        echo "  (Log: /tmp/medical-screening-tool.log)"
        read -r -p "  Press Enter to close..."
        exit 1
    fi
fi

URL="http://localhost:$PORT"

open_app_window() {
    # Prefer a Chrome-family browser in --app mode, which drops the
    # address bar/tabs and makes this feel like an installed app rather
    # than "a website" -- same effect as the Windows launcher's
    # `msedge --app=...`. Falls back to just opening the default browser
    # if none of those are available.
    if [ "$(uname -s)" = "Darwin" ]; then
        for browser in "Google Chrome" "Microsoft Edge" Chromium Brave; do
            if osascript -e "id of application \"$browser\"" >/dev/null 2>&1; then
                open -na "$browser" --args --app="$URL" --window-size=1200,860
                return
            fi
        done
        open "$URL"
    else
        for cmd in google-chrome google-chrome-stable microsoft-edge chromium chromium-browser brave-browser; do
            if command -v "$cmd" >/dev/null 2>&1; then
                "$cmd" --app="$URL" --window-size=1200,860 >/dev/null 2>&1 &
                return
            fi
        done
        xdg-open "$URL" >/dev/null 2>&1 &
    fi
}

open_app_window
