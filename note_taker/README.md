# NotesGenerator

Lecture video → markdown report (GPU required).

## Colab

```python
!pip install -q "NotesGenerator[gpu] @ git+https://github.com/saisritejakuppaEros/NotesGenerator.git#subdirectory=note_taker"
!apt-get install -y -qq ffmpeg imagemagick libsm6 libxext6 libgl1

from notesgenerator.colab_env import configure_colab_env
configure_colab_env()

!notesgenerator --video "/content/lecture.mp4" --output my_lecture
```

Outputs:
- `/content/runs/my_lecture/report/report.md`
- `/content/my_lecture_report.zip` (auto)

Progress prints appear live for each pipeline step. Images compressed with ImageMagick (palette PNG) after keyframe extraction and after report generation.

## Options

```bash
notesgenerator --video lecture.mp4 --output my_lecture
notesgenerator --url "https://youtube.com/watch?v=ID" --output my_lecture
notesgenerator --video lecture.mp4 --output my_lecture --data-dir /content
notesgenerator --video lecture.mp4 --output my_lecture --no-zip
notesgenerator --video lecture.mp4 --output my_lecture --no-compress

# Ground reports in research PDFs (requires papers extra)
pip install "NotesGenerator[gpu,papers]"
notesgenerator --video lecture.mp4 --output my_lecture \
  --papers paper1.pdf paper2.pdf

# Last-frame keyframes (default) vs midpoint
notesgenerator --video lecture.mp4 --output my_lecture --keyframe-strategy last
```

Report step vLLM settings are **auto-detected** (8192 context on T4, 16384 on larger GPUs). On shared GPUs, memory utilization is auto-clamped to available VRAM. Override with `--max-model-len` / `--gpu-mem` if needed.

## What's new in v0.1.5

- **Single VLM load** — slide captioning and report generation share one Qwen3-VL session (faster, less VRAM churn)
- **Research paper grounding** — optional `--papers` PDFs parsed with docling and injected into report prompts (`NotesGenerator[papers]`)
- **Last-frame keyframes** — default `--keyframe-strategy last` captures completed slide animations
- **Better transcript alignment** — overlap-based scene matching preserves timestamped segments
- **Teaching-level reports** — prerequisites, learning roadmap, granular topic sections, paper context
- **Shared-GPU VRAM clamp** — auto-lowers `gpu_mem` when other jobs occupy the GPU
- **vLLM batching** — parallel outline/topic generation for faster report writing on large GPUs
