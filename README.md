# NotesGenerator

Turn lecture videos into markdown reports.

**Documentation:** [docs/](docs/index.html) — static site for [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-from-a-branch) (Settings → Pages → branch `main`, folder `/docs`).

| Page | Description |
|------|-------------|
| [Overview](docs/index.html) | Features, models, quick example |
| [Quickstart](docs/quickstart.html) | Colab + local install |
| [How it works](docs/how-it-works.html) | Pipeline flowcharts & folder layout |
| [Configuration](docs/configuration.html) | CLI flags & GPU tuning |
| [Examples](docs/examples.html) | Sample report structure |

## Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qqzgnirnHFXrWATLAeHg4-73V3NpGCzo?usp=sharing)

Or run manually:

```python
!pip install -q "NotesGenerator[gpu] @ git+https://github.com/saisritejakuppaEros/NotesGenerator.git#subdirectory=note_taker"
!apt-get install -y -qq ffmpeg imagemagick libsm6 libxext6 libgl1

from notesgenerator.colab_env import configure_colab_env
configure_colab_env()  # outputs go to /content/runs/, fixes torch/torchaudio on Colab

!notesgenerator --video "/content/cv.mp4" --output my_lecture
# → /content/runs/my_lecture/report/report.md
# → /content/my_lecture_report.zip  (auto-created)
```

Download zip:

```python
from google.colab import files
files.download("/content/my_lecture_report.zip")
```

After PyPI publish: `pip install NotesGenerator[gpu]`
