#!/usr/bin/env bash
# Usage: bash scripts/install_env.sh [/path/to/new/venv]
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
[[ $# -le 1 ]] || { echo "Usage: $0 [/path/to/new/venv]" >&2; exit 2; }
DEST=${1:-"$ROOT/.venv"}
BOOTSTRAP_PYTHON=${BOOTSTRAP_PYTHON:-python3.12}
"$BOOTSTRAP_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
[[ ! -e "$DEST" ]] || { echo "Destination already exists: $DEST" >&2; exit 2; }
"$BOOTSTRAP_PYTHON" -m venv "$DEST"
"$DEST/bin/python" -m pip install --upgrade pip
"$DEST/bin/python" -m pip install -r "$ROOT/requirements.txt"
"$DEST/bin/python" -m pip install --no-deps -e "$ROOT"
"$DEST/bin/python" -m pip check
echo "Activate with: source $DEST/bin/activate"
