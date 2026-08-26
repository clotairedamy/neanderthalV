#!/usr/bin/env bash
# Build the single-bundle macOS app: dist/GeoViz.app
set -euo pipefail
cd "$(dirname "$0")/.."

source venv/bin/activate
pip install pyinstaller
pyinstaller --noconfirm packaging/geoviz.spec
echo
echo "Built dist/GeoViz.app — copy it anywhere and double-click to run."
