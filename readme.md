# Lecture Notes Pipeline

## One-shot run (everything + time analysis)

```bash
source /devwork/teja/note_generator/scripts/env.sh
cd /devwork/teja/note_generator

bash scripts/run_lecture.sh \
  --video "/devwork/teja/note_generator/lecture/Stanford CS329A Self-Improving AI Agents ｜ Part 1 ｜ Course Overview [6YnLB0XbTnI].mp4" \
  --output lecture_1
```

Or directly:

```python
python3 scripts/run_lecture.py \
  --video "/devwork/teja/note_generator/lecture/Stanford CS329A Self-Improving AI Agents ｜ Part 1 ｜ Course Overview [6YnLB0XbTnI].mp4" \
  --output lecture_1
```

### What it runs (in order)

| Step | Script | Output |
|------|--------|--------|
| 00 | Prepare video + audio | `runs/lecture_1/raw/` |
| 02 | Whisper transcribe | `transcript/whisper_segments.json` |
| 03 | Scene detect (threshold 35) | `scenes/scene_list.json` |
| 04 | Keyframes | `scenes/keyframes/*.png` |
| 05 | Qwen VLM captions | `captions/vlm_captions.json` |
| 06 | Align | `merged/aligned_scenes.json` |
| 07 | Merge | `merged/scenes_merged.json` |
| 09 | Detailed report | `report/report.md` |

### Output layout

```
runs/lecture_1/
├── pipeline_timing.json    ← per-step time analysis
├── raw/video.mp4
├── merged/scenes_merged.json
└── report/
    ├── report.md           ← final Obsidian doc
    ├── topic_outline.json
    └── attachments/*.png
```

At the end you get a **time analysis table** in the terminal + `pipeline_timing.json`.

### Options

```bash
bash scripts/run_lecture.sh \
  --video "/path/to/lecture.mp4" \
  --output lecture_1 \
  --force-all \              # re-prepare, re-detect, re-caption
  --whisper-model medium \
  --scene-threshold 35 \
  --refine-passes 2
```

### Zip for Obsidian

```bash
cd runs/lecture_1/report
zip -r lecture_1_report.zip report.md attachments topic_outline.json main_scenes.json
```
