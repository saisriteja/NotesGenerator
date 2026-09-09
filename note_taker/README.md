# NotesGenerator

Lecture video → markdown report pipeline (GPU required).

## Install

```bash
# From GitHub (until PyPI is live)
pip install "git+https://github.com/saisritejakuppaEros/NotesGenerator.git#subdirectory=note_taker[gpu]"

# From PyPI (after publish)
pip install NotesGenerator[gpu]
```

Colab also needs: `apt-get install -y ffmpeg libsm6 libxext6 libgl1`

## Run

```bash
notesgenerator --video lecture.mp4 --output my_lecture
notesgenerator --url "https://www.youtube.com/watch?v=VIDEO_ID" --output my_lecture
```

Output: `runs/<output>/report/report.md`

## Colab notebook

Open `src/notesgenerator/colab/note_taker.ipynb` or run the cells in the root README.

## Dev

```bash
cd note_taker
pip install -e ".[gpu]"
python -m build
```
