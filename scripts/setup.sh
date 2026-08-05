#!/usr/bin/env bash
# One-time setup: activate gsplat_env and install non-torch deps.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

echo "Python: $(which python3)"
echo "Torch:  $(python3 -c 'import torch; print(torch.__version__)')"

pip install -r "${SCRIPT_DIR}/../requirements.txt"
echo "Setup complete."
