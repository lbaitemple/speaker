#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
CFG="$VENV/pyvenv.cfg"

# Allow the venv to see system-installed packages (e.g. MangDang LCD)
if [ -f "$CFG" ]; then
    sed -i 's/^include-system-site-packages = false/include-system-site-packages = true/' "$CFG"
    echo "[setup] include-system-site-packages = true"
fi

# Install / update Python dependencies
echo "[setup] Installing requirements..."
 sudo   apt install python3.12-venv  portaudio19-dev  python3-pyaudio -y
"$VENV/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"

# Kill any previous instance so it releases the audio device before we start
if pkill -f "discord_app_speaker.py" 2>/dev/null; then
    echo "[setup] Waiting for previous instance to release audio device..."
    sleep 2
fi

# Launch the bot (no exec — shell must continue after Python exits)
echo "[start] Starting discord_app_speaker..."
"$VENV/bin/python3" "$SCRIPT_DIR/discord_app_speaker.py" || true

# Restore robot control regardless of how the bot exited (Ctrl+C, crash, login failure)
echo "[shutdown] Restarting robot.service..."
sudo systemctl restart robot.service
