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
```

Report step vLLM settings are **auto-detected** (8192 context on T4, 16384 on larger GPUs). Override with `--max-model-len` / `--gpu-mem` if needed.
