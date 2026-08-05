#!/usr/bin/env bash
# Batch-run GPU MODE lectures from split yt-dlp streams (.mp4/.webm pairs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUMODE="${ROOT}/gpumode"
cd "$ROOT"
source scripts/env.sh

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
unset SOCKS_PROXY SOCKS5_PROXY socks_proxy socks5_proxy

LOG="${ROOT}/runs/gpumode_batch.log"
mkdir -p "${ROOT}/runs"
exec > >(tee -a "$LOG") 2>&1

echo "========== gpumode batch started $(date -Is) =========="

mapfile -t VIDEO_FILES < <(
  python3 - <<'PY'
from pathlib import Path
import subprocess

gpumode = Path("gpumode")
stems: dict[str, Path] = {}

def has_video(path: Path) -> bool:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and result.stdout.strip().lower() == "video"

def stem(path: Path) -> str:
    import re
    name = path.name
    match = re.search(r"\.f\d+\.(mp4|webm|webp|mkv)$", name, re.I)
    if match:
        return name[: match.start()]
    return path.stem

def rank(path: Path) -> tuple:
    name = path.name.lower()
    return (
        path.suffix.lower() != ".mp4",
        ".f137." not in name and ".f399." not in name,
        -path.stat().st_size,
    )

for path in sorted(gpumode.iterdir()):
    if not path.is_file():
        continue
    if path.suffix.lower() not in {".mp4", ".webm", ".webp"}:
        continue
    if path.name.endswith(".part") or path.stat().st_size < 1024 * 1024:
        continue
    if not has_video(path):
        continue
    key = stem(path)
    current = stems.get(key)
    if current is None or rank(path) < rank(current):
        stems[key] = path

for path in sorted(stems.values(), key=lambda p: p.name):
    print(path)
PY
)

echo "Found ${#VIDEO_FILES[@]} lectures in ${GPUMODE}"

FAILED=()
for video in "${VIDEO_FILES[@]}"; do
  base="$(basename "$video")"
  slug="$(python3 -c "import re,sys; n=sys.argv[1]; m=re.search(r'\.f\d+\.', n); print(re.sub(r'[^A-Za-z0-9]+','_', n[:m.start()] if m else n.rsplit('.',1)[0]).strip('_').lower())" "$base")"
  out="gpumode_${slug}"
  d="${ROOT}/runs/${out}"

  if [ -f "$d/report/report.md" ]; then
    echo "SKIP ${out} — report already exists"
    continue
  fi

  echo "========== ${out} $(date -Is) =========="
  echo "Video: ${video}"

  if ! bash scripts/run_lecture.sh \
    --video "$video" \
    --output "$out" \
    --force-all; then
    echo "ERROR: ${out} failed — continuing with next lecture"
    FAILED+=("$out")
    continue
  fi

  if [ -d "$d/report" ]; then
    (cd "$d" && zip -r "${out}_report.zip" report/)
    echo "Zipped: $d/${out}_report.zip"
  fi
done

echo "========== gpumode batch finished $(date -Is) =========="
if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "Failed (${#FAILED[@]}): ${FAILED[*]}"
  exit 1
fi
