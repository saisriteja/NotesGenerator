# NotesGenerator

Turn lecture videos into markdown reports.

## Colab

```python
!pip install -q "NotesGenerator[gpu] @ git+https://github.com/saisritejakuppaEros/NotesGenerator.git#subdirectory=note_taker"
!apt-get install -y -qq ffmpeg imagemagick libsm6 libxext6 libgl1

from notesgenerator.colab_env import configure_colab_env
configure_colab_env()  # outputs go to /content/runs/

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
