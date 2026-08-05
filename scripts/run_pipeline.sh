#!/usr/bin/env bash
# End-to-end pipeline runner.
#
# Usage:
#   source scripts/env.sh
#   bash scripts/run_pipeline.sh "https://www.youtube.com/watch?v=VIDEO_ID"
#
# Optional env vars:
#   COOKIES_BROWSER=chrome   — pass to yt-dlp for bot-protected videos
#   WHISPER_MODEL=medium
#   QWEN_MODEL=Qwen/Qwen3-VL-4B-Instruct
#   VLM_BATCH_SIZE=8
#   NOTES_BATCH_SIZE=4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

URL="${1:?Usage: run_pipeline.sh <youtube_url_or_id>}"

VIDEO_ID="$(python3 -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from common import extract_video_id; print(extract_video_id(sys.argv[1]))" "${URL}")"

RUN_DIR="${NOTE_GENERATOR_ROOT}/runs/run_${VIDEO_ID}"
COOKIES_BROWSER="${COOKIES_BROWSER:-}"
WHISPER_MODEL="${WHISPER_MODEL:-medium}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
VLM_BATCH_SIZE="${VLM_BATCH_SIZE:-8}"
NOTES_BATCH_SIZE="${NOTES_BATCH_SIZE:-4}"

echo "=== Pipeline run: ${RUN_DIR} ==="

COOKIE_ARGS=()
if [[ -n "${COOKIES_BROWSER}" ]]; then
  COOKIE_ARGS=(--cookies-from-browser "${COOKIES_BROWSER}")
elif [[ -n "${COOKIES_FILE:-}" ]]; then
  COOKIE_ARGS=(--cookies "${COOKIES_FILE}")
fi

python3 "${SCRIPT_DIR}/01_download.py" "${URL}" --run-dir "${RUN_DIR}" "${COOKIE_ARGS[@]}"

# Modules 2 and 3 are independent; run sequentially here (parallel optional).
python3 "${SCRIPT_DIR}/02_transcribe.py" --run-dir "${RUN_DIR}" --model "${WHISPER_MODEL}"
python3 "${SCRIPT_DIR}/03_detect_scenes.py" --run-dir "${RUN_DIR}"
python3 "${SCRIPT_DIR}/04_extract_keyframes.py" --run-dir "${RUN_DIR}"
python3 "${SCRIPT_DIR}/05_caption_slides.py" --run-dir "${RUN_DIR}" --model "${QWEN_MODEL}" --batch-size "${VLM_BATCH_SIZE}"
python3 "${SCRIPT_DIR}/06_align.py" --run-dir "${RUN_DIR}"
python3 "${SCRIPT_DIR}/07_merge.py" --run-dir "${RUN_DIR}"
python3 "${SCRIPT_DIR}/08_generate_notes.py" --run-dir "${RUN_DIR}" --model "${QWEN_MODEL}" --batch-size "${NOTES_BATCH_SIZE}"

echo "=== Done ==="
echo "Checkpoint: ${RUN_DIR}/merged/scenes_merged.json"
echo "Notes:      ${RUN_DIR}/notes/lecture_notes.md"
