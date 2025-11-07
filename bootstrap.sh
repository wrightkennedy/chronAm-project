#!/bin/bash
set -euo pipefail

# Bootstrap script for ChronAm on macOS/Linux. Creates an isolated venv next to the repo
# and launches the PyQt GUI once dependencies finish installing.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${SCRIPT_DIR}/chronam-env"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d "$ENV_DIR" ]; then
  echo "Creating ChronAm virtual environment at $ENV_DIR"
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi

echo "Activating chronam-env"
# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"

echo "Upgrading pip"
python -m pip install --upgrade pip

echo "Installing ChronAm dependencies"
python -m pip install -r "$SCRIPT_DIR/requirements.txt"

echo "Launching ChronAm GUI"
python "$SCRIPT_DIR/app.py"
