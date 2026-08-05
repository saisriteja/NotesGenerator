#!/usr/bin/env bash
# One-shot lecture pipeline wrapper (sources conda env + runs run_lecture.py)
#
# Usage:
#   bash scripts/run_lecture.sh \
#     --video "/path/to/lecture.mp4" \
#     --output lecture_1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

exec python3 "${SCRIPT_DIR}/run_lecture.py" "$@"
