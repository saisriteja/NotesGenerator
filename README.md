# NotesGenerator

Turn lecture videos into markdown reports.

## Colab

```python
!pip install "NotesGenerator[gpu] @ git+https://github.com/saisritejakuppaEros/NotesGenerator.git#subdirectory=note_taker"
!apt-get install -y -qq ffmpeg libsm6 libxext6 libgl1

from notesgenerator.colab_env import configure_colab_env
configure_colab_env()

!notesgenerator --video "/content/lecture.mp4" --output my_lecture
```

After PyPI publish: `pip install NotesGenerator[gpu]`

See [note_taker/README.md](note_taker/README.md) for details.
