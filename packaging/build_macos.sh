#!/usr/bin/env bash
# Build the single-bundle macOS app: dist/NeanderthalV.app
set -euo pipefail
cd "$(dirname "$0")/.."

source venv/bin/activate
pip install pyinstaller
pyinstaller --noconfirm packaging/neanderthalv.spec
echo
echo "Built dist/NeanderthalV.app — copy it anywhere and double-click to run."
