# NotesGenerator

Turn lecture videos into detailed markdown reports. Designed for **Google Colab** with a GPU runtime.

## Install

```bash
pip install NotesGenerator[gpu]
```

Colab (GPU runtime):

```python
!pip install NotesGenerator[gpu]
!apt-get install -y -qq ffmpeg libsm6 libxext6 libgl1
```

## Quick start

```bash
# Local video
notesgenerator --video /path/to/lecture.mp4 --output my_lecture

# YouTube URL
notesgenerator --url "https://www.youtube.com/watch?v=VIDEO_ID" --output my_lecture
```

Output: `runs/<output>/report/report.md` (relative to your working directory).

## Python API / Colab

```python
from notesgenerator.colab_env import configure_colab_env

configure_colab_env()  # sets cache under ./models/, outputs under ./runs/
```

Open the bundled notebook:

```python
import notesgenerator
from pathlib import Path
print(Path(notesgenerator.__file__).parent / "colab" / "note_taker.ipynb")
```

Or use `colab/note_taker.ipynb` from this repo.

## Pipeline steps

1. Prepare video + extract audio (or download from YouTube)
2. Transcribe with Whisper
3. Detect slide/scene changes
4. Extract keyframes
5. Caption slides with Qwen3-VL (vLLM)
6. Align transcript to scenes
7. Merge checkpoint
8. Generate structured report

## Requirements

- **GPU runtime** (Colab T4/A100/L4). vLLM + Qwen3-VL need CUDA.
- ~8–12 GB GPU RAM for Qwen3-VL-4B with default settings.
- `ffmpeg` on the system PATH.
- First run downloads model weights (~several GB) to `models/hub/`.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `NOTE_TAKER_ROOT` | current directory | Where `runs/` and `models/` are created |
| `QWEN_MODEL` | `Qwen/Qwen3-VL-4B-Instruct` | VLM for captions + report |
| `HF_HOME` | `{NOTE_TAKER_ROOT}/models/hub` | HuggingFace cache |

## PyPI publishing

This package uses [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) via GitHub Actions.

**Trusted publisher settings on PyPI:**

| Field | Value |
|-------|-------|
| PyPI Project Name | `NotesGenerator` |
| Owner | `saisritejakuppaEros` |
| Repository name | `NotesGenerator` |
| Workflow name | `workflow.yml` |
| Environment name | `pypi` |

**GitHub:** create a `pypi` environment under repo Settings → Environments.

**Publish:** create a GitHub Release (tag e.g. `v0.1.0`) — the workflow builds and uploads automatically.

Before each release, bump `__version__` in `src/notesgenerator/__init__.py`.

## Development

```bash
cd note_taker
pip install -e ".[gpu]"
python -m build
```
