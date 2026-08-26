#!/usr/bin/env bash
# Launch NeanderthalV (macOS and Raspberry Pi 5).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d venv ]; then
  echo "No venv found — running ./setup.sh first…"
  ./setup.sh
fi
# shellcheck disable=SC1091
source venv/bin/activate

# Raspberry Pi: prefer the KMS GL driver and cap Qt scaling oddities.
if grep -qi "raspberry pi" /proc/device-tree/model 2>/dev/null; then
  export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
  export MESA_GL_VERSION_OVERRIDE=3.3
fi

exec python main.py "$@"
