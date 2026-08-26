#!/usr/bin/env bash
# One-time setup for GeoViz on macOS or Raspberry Pi 5.
# Creates a venv, installs dependencies, optionally installs Demucs and
# pre-downloads the htdemucs model.
set -euo pipefail
cd "$(dirname "$0")"

PY=python3
if ! $PY -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
  for cand in python3.12 python3.11; do
    if command -v $cand >/dev/null; then PY=$cand; break; fi
  done
fi
echo "Using $($PY --version)"

OS="$(uname -s)"
IS_RPI=false
if [ "$OS" = "Linux" ] && grep -qi "raspberry pi" /proc/device-tree/model 2>/dev/null; then
  IS_RPI=true
fi

if $IS_RPI; then
  echo "Raspberry Pi detected — installing system packages (needs sudo)…"
  sudo apt-get update
  sudo apt-get install -y python3-venv python3-dev libportaudio2 libsndfile1 \
    libgl1 libegl1 libxkbcommon-x11-0 libxcb-cursor0 ffmpeg
fi

$PY -m venv venv
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

# Demucs (torch) — always on macOS; opt-in on RPi because it is heavy/slow.
if $IS_RPI; then
  read -r -p "Install Demucs v4 stem separation on this Pi? Large download, slow separation [y/N] " ans
  if [[ "${ans:-n}" =~ ^[Yy]$ ]]; then
    pip install -r requirements-stems.txt
  else
    echo "Skipping Demucs — the app will use fast DSP pseudo-stems."
  fi
else
  pip install -r requirements-stems.txt
fi

# Pre-download the htdemucs model so first run doesn't stall.
if python -c 'import demucs' 2>/dev/null; then
  echo "Downloading Demucs htdemucs model (one-time)…"
  python - <<'EOF'
from demucs.pretrained import get_model
get_model("htdemucs")
print("Model cached.")
EOF
fi

chmod +x run.sh
echo
echo "Setup complete. Start the app with:  ./run.sh"
