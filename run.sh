#!/usr/bin/env bash
# Batch-run lectures 1–9; skip any that already have report.md, zip when done.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source scripts/env.sh

# Don't inherit Cursor/IDE proxy env (breaks HuggingFace even with local cache)
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
unset SOCKS_PROXY SOCKS5_PROXY socks_proxy socks5_proxy

LOG="${ROOT}/runs/batch.log"
mkdir -p "${ROOT}/runs"
exec > >(tee -a "$LOG") 2>&1

echo "========== batch started $(date -Is) =========="

run_report_tail() {
  local rd="$1"
  python3 scripts/05_caption_slides.py --run-dir "$rd" --batch-size 8 --force
  python3 scripts/06_align.py --run-dir "$rd"
  python3 scripts/07_merge.py --run-dir "$rd"
  python3 scripts/09_generate_report.py \
    --run-dir "$rd" \
    --force-outline \
    --max-model-len 16384 \
    --window-minutes 8 \
    --section-tokens 6000 \
    --refine-passes 2 \
    --refine-tokens 4096 \
    --gpu-mem 0.82
}

for i in 1 2 3 4 5 6 7 8 9; do
  d="${ROOT}/runs/lecture_${i}"
  video="${ROOT}/lecture/lectures/lecture ${i}.mp4"

  if [ -f "$d/report/report.md" ]; then
    echo "SKIP lecture_${i} — report already exists"
    continue
  fi

  if [ ! -f "$video" ] && [ "$i" -eq 1 ]; then
    video="${ROOT}/lecture/Stanford CS329A Self-Improving AI Agents ｜ Part 1 ｜ Course Overview [6YnLB0XbTnI].mp4"
  fi

  if [ ! -f "$video" ]; then
    echo "ERROR: video not found for lecture_${i}: $video"
    continue
  fi

  echo "========== lecture_${i} $(date -Is) =========="

  kf_count="$(find "$d/scenes/keyframes" -name '*.png' 2>/dev/null | wc -l)"
  if [ "$kf_count" -gt 0 ] && [ -f "$d/scenes/scene_list.json" ]; then
    echo "RESUME lecture_${i} from step 05 ($kf_count keyframes ready)"
    run_report_tail "$d"
  else
    bash scripts/run_lecture.sh \
      --video "$video" \
      --output "lecture_${i}" \
      --force-all
  fi

  if [ -d "$d/report" ]; then
    (cd "$d" && zip -r "lecture_${i}_report.zip" report/)
    echo "Zipped: $d/lecture_${i}_report.zip"
  else
    echo "WARN: lecture_${i} has no report/ — skipping zip"
  fi
done

echo "========== batch finished $(date -Is) =========="

# Final zip pass for any reports created outside this run
for i in 1 2 3 4 5 6 7 8 9; do
  d="${ROOT}/runs/lecture_${i}"
  if [ -d "$d/report" ] && [ ! -f "$d/lecture_${i}_report.zip" ]; then
    (cd "$d" && zip -r "lecture_${i}_report.zip" report/)
    echo "Zipped: $d/lecture_${i}_report.zip"
  fi
done
