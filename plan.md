# YouTube Lecture → Obsidian Notes Pipeline — Module Reference

Reference doc for what each module does, what file it lives in, and its exact input/output contract. Built for Colab, sized for videos up to ~120 min.

---

## 0. Directory layout (created once, at the start of a run)

```
/content/run_<video_id>/
├── raw/
│   ├── video.mp4
│   └── audio.wav
├── transcript/
│   └── whisper_segments.json
├── scenes/
│   ├── scene_list.json
│   └── keyframes/
│       ├── scene_001.png
│       ├── scene_002.png
│       └── ...
├── captions/
│   └── vlm_captions.json
├── merged/
│   └── scenes_merged.json      ← the checkpoint file
├── notes/
│   └── lecture_notes.md        ← final Obsidian file
└── attachments/                ← copy of keyframes, ships with the .md
    ├── scene_001.png
    └── ...
```

Every module reads from one folder and writes to the next. Nothing gets overwritten in place, so any module can be re-run alone as long as its input folder already exists.

---

## Module 1 — Downloader

**File:** `01_download.py`

| | |
|---|---|
| **Input** | YouTube URL (string) |
| **Output** | `raw/video.mp4`, `raw/audio.wav` (16kHz mono) |
| **Tool** | `yt-dlp` |
| **Notes** | Only network step. For a 120-min video, download once and never re-run — cache in Colab's `/content/` for the session. If Colab disconnects, re-download is your main time cost, so consider saving `raw/` to Drive right after this step. |

---

## Module 2 — Transcription

**File:** `02_transcribe.py`

| | |
|---|---|
| **Input** | `raw/audio.wav` |
| **Output** | `transcript/whisper_segments.json` → list of `{start, end, text}` |
| **Tool** | `faster-whisper` (GPU) |
| **Notes** | For 120 min, use `medium` or `small` model unless accuracy on math terms is poor — `large-v3` is slower but better at technical vocabulary (integrals, "epsilon", etc.). This step and Module 3 don't depend on each other — can run in either order or in parallel. Runtime on a T4: roughly 0.15–0.3× real-time, so a 120-min lecture ≈ 20–35 min transcription. |

---

## Module 3 — Scene Detection

**File:** `03_detect_scenes.py`

| | |
|---|---|
| **Input** | `raw/video.mp4` |
| **Output** | `scenes/scene_list.json` → list of `{scene_id, start, end}` |
| **Tool** | `PySceneDetect` (`ContentDetector`, threshold ~27) |
| **Notes** | For 120-min slide lectures expect roughly 80–200 scenes depending on how often slides change. If detector over-triggers on animations/builds *within* one slide, raise threshold (30–35) or add `min_scene_len` (e.g. 2 sec) to avoid junk micro-scenes. Cheap and fast — runs in a minute or two even for 2 hours of video. |

---

## Module 4 — Keyframe Extraction

**File:** `04_extract_keyframes.py`

| | |
|---|---|
| **Input** | `raw/video.mp4` + `scenes/scene_list.json` |
| **Output** | `scenes/keyframes/scene_XXX.png` (one PNG per scene) |
| **Tool** | `opencv-python` |
| **Notes** | Grabs the frame at each scene's **midpoint** (`(start+end)/2`), not the first frame, to avoid transition blur. One image per scene, so ~80–200 images for a 2-hour lecture, not thousands. |

---

## Module 5 — VLM Captioning

**File:** `05_caption_slides.py`

| | |
|---|---|
| **Input** | `scenes/keyframes/*.png` |
| **Output** | `captions/vlm_captions.json` → list of `{scene_id, vlm_description}` |
| **Tool** | Gemma (VLM) |
| **Notes** | This is the slowest step if run scene-by-scene — for 150 images budget several minutes to ~20 min depending on model size and Colab GPU. Fixed prompt, no back-and-forth, no re-tries unless output is empty. If you're on a small/quantized Gemma, batch images in groups if the model supports multi-image input to cut overhead. |

**Fixed prompt used per image:**
> "This is a slide from an AI/math lecture. Transcribe any equations (LaTeX), list key terms/diagrams, and summarize what the slide shows in 2–3 sentences."

---

## Module 6 — Alignment

**File:** `06_align.py`

| | |
|---|---|
| **Input** | `transcript/whisper_segments.json` + `scenes/scene_list.json` |
| **Output** | intermediate, held in memory / merged directly into Module 7's output |
| **Tool** | plain Python (no ML) |
| **Notes** | For each scene window `(start, end)`, pulls all Whisper segments whose `start` falls inside that window and concatenates their `.text`. Purely arithmetic — instant even for 120-min transcripts. |

---

## Module 7 — Merge (checkpoint)

**File:** `07_merge.py`

| | |
|---|---|
| **Input** | scene list + keyframe paths + VLM captions + aligned transcript text |
| **Output** | `merged/scenes_merged.json` — **the single most important file in the pipeline** |
| **Tool** | plain Python |
| **Notes** | One record per scene: |

```json
{
  "scene_id": 3,
  "start": 145.2,
  "end": 210.8,
  "image_path": "scenes/keyframes/scene_003.png",
  "vlm_description": "...",
  "transcript": "..."
}
```

> **Save this file to Drive immediately after generating it.** Everything upstream (download, transcribe, detect, caption) is the expensive/slow part. Everything downstream (Module 8) is cheap text generation you'll want to re-run and tweak repeatedly. This checkpoint is what lets you redesign note *style* without ever touching Whisper or Gemma again.

---

## Module 8 — Note Generation

**File:** `08_generate_notes.py`

| | |
|---|---|
| **Input** | `merged/scenes_merged.json` |
| **Output** | `notes/lecture_notes.md` (final file) + `attachments/*.png` (copied keyframes) |
| **Tool** | LLM (Gemma or a stronger model if available) |
| **Notes** | One prompt per scene (or batch 5–10 scenes per call to cut latency if the model is strong enough to hold context). Deterministic, single-shot — no reflection/re-planning loop. Output per scene follows a fixed template, then all scene blocks get concatenated into one `.md` file. |

**Per-scene output template:**

```markdown
## Scene 3 — [topic] (02:25–03:30)
![[scene_003.png]]

**Slide content:** ...
**Explanation (from audio):** ...
**Key equations:** ...
```

---

## End-to-end runtime estimate (120-min video, T4 GPU)

| Module | Approx. time |
|---|---|
| 1. Download | 1–5 min (network dependent) |
| 2. Transcribe | 20–35 min |
| 3. Scene detect | 1–3 min |
| 4. Keyframe extract | <1 min |
| 5. VLM caption (~150 scenes) | 10–20 min |
| 6. Align | instant |
| 7. Merge | instant |
| 8. Note generation (~150 scenes) | 10–20 min |
| **Total** | **~45–85 min** |

Modules 2 and 5/8 dominate. If you want to shorten future iterations: run 1–7 once, checkpoint, then only ever re-run Module 8 while you dial in note style.

---

## What can be parallelized
- Module 2 (transcribe) and Module 3 (scene detect) are independent — can run back to back without waiting on each other, or in two Colab cells fired close together.
- Module 5 (VLM captioning) can start as soon as Module 4 finishes; it doesn't need the transcript at all.
- Module 6 needs both 2 and 3 done, so it's the natural join point.