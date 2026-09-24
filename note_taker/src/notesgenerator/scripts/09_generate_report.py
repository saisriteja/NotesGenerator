#!/usr/bin/env python3
"""Module 9 — Detailed lecture report with full-session topic/time breakdown.

Produces ``report/report.md`` with:
- Complete timeline for the entire lecture duration
- Topic index → nested time segments (revisit timestamps)
- Granular detailed sections + figures
- Multi-pass refinement against full context
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from common import add_run_dir_arg, load_json, save_json
from qwen_vllm import DEFAULT_MODEL, QwenVllmEngine, VllmConfig, qwen_vllm_session, text_messages

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SKIP_KEYWORDS = re.compile(
    r"\b(logo|title slide|welcome|contact information|course assistants|"
    r"no equations|no visible|empty stage|branding|institutional identifier)\b",
    re.I,
)
KEEP_KEYWORDS = re.compile(
    r"\b(equation|latex|\$\$|diagram|architecture|graph|scaling law|"
    r"algorithm|framework|theorem|paper|arxiv|benchmark|plot|chart|"
    r"table|workflow|agent|model)\b",
    re.I,
)
JUNK_SEGMENT_TITLE = re.compile(
    r"\b("
    r"no equations?(?:\s+on\s+slide|\s+or\s+diagrams?|\s+present|\s+or\s+latex)?|"
    r"no visible(?:\s+equations?)?|slide with no visuals?|no diagrams?|"
    r"discussion questions on slide|no visuals?"
    r")\b",
    re.I,
)
_SLIDE_ABSENCE_SENTENCE = re.compile(
    r"(?:^|[.!?]\s+)(?:The )?(?:slide )?(?:contains |has |presents? |shows? )?"
    r"no (?:equations?|diagrams?|visual aids?|latex)[^.!?]*[.!?]\s*",
    re.I | re.MULTILINE,
)


def fmt_time(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def fmt_range(start: float, end: float) -> str:
    return f"{fmt_time(start)}–{fmt_time(end)}"


def parse_json_block(text: str) -> Any:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    text = re.sub(r"//[^\n]*", "", text)
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            obj, _end = decoder.raw_decode(text, i)
            return obj
        except json.JSONDecodeError:
            continue
    raise ValueError(f"No parseable JSON: {text[:240]!r}...")


def truncate(text: str, limit: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def prompt_char_budget(max_model_len: int, output_tokens: int, reserve: int = 768) -> int:
    """Rough character budget for prompt text (≈3 chars/token)."""
    input_tokens = max(512, max_model_len - output_tokens - reserve)
    return input_tokens * 3


def fit_prompt(text: str, budget: int) -> str:
    """Trim prompt context without markers the model might copy into output."""
    text = (text or "").strip()
    if len(text) <= budget:
        return text
    cut = text[: max(0, budget - 1)]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


def transcript_for_range(
    whisper_segments: list[dict],
    start: float,
    end: float,
    max_chars: int = 6000,
    window_sec: float = 300.0,
) -> str:
    """Pull transcript excerpts by time range (works when few scenes span long spans)."""
    if not whisper_segments or end <= start:
        return "(no transcript)"

    lines: list[str] = []
    used = 0
    t = start
    while t < end and used < max_chars:
        win_end = min(t + window_sec, end)
        texts = [
            seg.get("text", "").strip()
            for seg in whisper_segments
            if seg.get("text") and seg["start"] >= t and seg["start"] < win_end
        ]
        if texts:
            chunk = truncate(" ".join(texts), min(800, max_chars - used))
            lines.append(f"{fmt_range(t, win_end)}: {chunk}")
            used += len(chunk)
        t = win_end
    return "\n".join(lines) or "(no transcript)"


def compact_segments_json(segs: list[dict], max_chars: int = 8000) -> str:
    compact = []
    for seg in segs:
        item = dict(seg)
        item["summary"] = str(item.get("summary", ""))
        item["key_points"] = [
            truncate(str(p), 120) for p in (item.get("key_points") or [])[:8]
        ]
        compact.append(item)
    text = json.dumps(compact, indent=2, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    # Drop figures from JSON if still too large (they are duplicated in layout rules).
    slim = []
    for seg in compact:
        item = {k: v for k, v in seg.items() if k != "figures"}
        slim.append(item)
    return fit_prompt(json.dumps(slim, indent=2, ensure_ascii=False), max_chars)


def scene_importance_score(scene: dict) -> float:
    vlm = scene.get("vlm_description") or ""
    transcript = scene.get("transcript") or ""
    blob = f"{vlm} {transcript}"
    score = 0.0
    if KEEP_KEYWORDS.search(blob):
        score += 3.0
    if re.search(r"\\[\(\[]|\$\$|equation|LaTeX", blob, re.I):
        score += 2.0
    if "http" in blob or "arxiv" in blob.lower():
        score += 1.5
    if len(transcript) > 80:
        score += 1.0
    if SKIP_KEYWORDS.search(vlm):
        score -= 4.0
    return score


def scene_by_id(scenes: list[dict]) -> dict[int, dict]:
    return {s["scene_id"]: s for s in scenes}


def scenes_in_range(scenes: list[dict], start: float, end: float) -> list[dict]:
    return [s for s in scenes if s["start"] < end and s["end"] > start]


def compact_catalog(scenes: list[dict], vlm: int = 120, audio: int = 100) -> str:
    lines = []
    for s in scenes:
        lines.append(
            f"{s['scene_id']:03d}|{fmt_range(s['start'], s['end'])}|"
            f"{truncate(s.get('vlm_description', ''), vlm)}|"
            f"{truncate(s.get('transcript', ''), audio)}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 1 — topic / time outline (full lecture coverage)
# ---------------------------------------------------------------------------

WINDOW_OUTLINE_PROMPT = """You segment ONE time window of a lecture into sub-topics.

Return ONLY JSON:
{{
  "segments": [
    {{
      "start_sec": {win_start},
      "end_sec": 300.0,
      "title": "short sub-topic name",
      "summary": "2-3 sentences — what is taught",
      "key_points": ["point 1", "point 2", "point 3"],
      "scene_ids": [1, 2]
    }}
  ]
}}

Rules:
- Cover the ENTIRE window {win_start}–{win_end} with contiguous segments (no gaps).
- Use scene_ids from the catalog below.
- Be specific about concepts, formulas, and examples taught in each segment.
- title: what the lecturer is teaching (3–10 words). NEVER describe what is missing from the slide ("No equations on slide", "No diagrams", etc.).
- summary: 2–3 complete sentences about the concepts taught — not about slide visuals. Synthesize — do NOT copy transcript verbatim.
- key_points: complete bullet phrases, not speech fragments.

Scene catalog (id|time|slide|audio):
{catalog}
"""

MERGE_TOPICS_PROMPT = """Merge lecture segment outlines into 6–14 major TOPICS covering the FULL lecture.

Return ONLY JSON:
{{
  "title": "lecture title",
  "subtitle": "one line",
  "topics": [
    {{
      "topic_id": 1,
      "title": "Major topic name",
      "start_sec": 0.0,
      "end_sec": 600.0,
      "summary": "paragraph overview — complete sentences only",
      "segments": [
        {{
          "start_sec": 0.0,
          "end_sec": 120.0,
          "title": "sub-topic",
          "summary": "2–3 complete sentences",
          "key_points": ["..."],
          "scene_ids": [1, 2]
        }}
      ]
    }}
  ],
  "references": [
    {{"label": "...", "url": "...", "timestamp": "MM:SS", "note": "..."}}
  ]
}}

Text quality rules (STRICT):
- Every title is a short descriptive phrase — NO markdown (**bold**), NO LaTeX blocks, NO "..."
- Segment titles describe what is being taught — NEVER "No equations on slide" or similar slide-absence labels
- Every summary is polished prose (2–4 sentences), ends with . ! or ? — focus on concepts, not slide visuals
- Do NOT paste raw transcript ("So here you have...", "That's your input...")
- Preserve all start_sec, end_sec, scene_ids from input segments

Segment batches:
{segments_json}
"""

OUTLINE_POLISH_PROMPT = """Polish lecture outline metadata for a student-facing report.

Return ONLY JSON with the SAME structure and the SAME topic_id / start_sec / end_sec / scene_ids values.
Rewrite ONLY text fields: title, subtitle, summary, segment titles, segment summaries, key_points.

Rules:
- title (topic & segment): 3–10 word descriptive phrase. NO markdown, NO LaTeX, NO ellipsis.
- Segment titles: what is taught in that period — NEVER slide-absence labels ("No equations on slide", etc.).
- summary: 2–4 complete sentences about concepts taught; must end with . ! or ?; never truncate mid-thought.
- key_points: complete concise bullets.
- Synthesize concepts from transcript — do NOT copy speech verbatim.

Current outline:
{outline_json}

Transcript reference (for accuracy only — do not copy phrasing):
{transcript_excerpt}
"""


def outline_window_fallback(
    window_scenes: list[dict], win_start: float, win_end: float
) -> list[dict]:
    if not window_scenes:
        return [
            {
                "start_sec": win_start,
                "end_sec": win_end,
                "title": "Segment",
                "summary": "No transcript in this window.",
                "key_points": [],
                "scene_ids": [],
            }
        ]
    fallback = []
    batch = 3
    for i in range(0, len(window_scenes), batch):
        chunk = window_scenes[i : i + batch]
        fallback.append(
            {
                "start_sec": chunk[0]["start"],
                "end_sec": chunk[-1]["end"],
                "title": truncate(chunk[0].get("vlm_description", ""), 50) or "Section",
                "summary": "Content covered in this segment.",
                "key_points": [],
                "scene_ids": [s["scene_id"] for s in chunk],
            }
        )
    return fallback


def outline_window_prompt(
    scenes: list[dict], win_start: float, win_end: float
) -> list[dict] | None:
    window_scenes = scenes_in_range(scenes, win_start, win_end)
    if not window_scenes:
        return None
    prompt = WINDOW_OUTLINE_PROMPT.format(
        win_start=win_start,
        win_end=win_end,
        catalog=compact_catalog(window_scenes),
    )
    return text_messages(prompt)


def parse_outline_window_response(
    text: str,
    scenes: list[dict],
    win_start: float,
    win_end: float,
) -> list[dict]:
    window_scenes = scenes_in_range(scenes, win_start, win_end)
    try:
        parsed = parse_json_block(text)
        segs = parsed.get("segments", parsed) if isinstance(parsed, dict) else parsed
        if isinstance(segs, list) and segs:
            return segs
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
        print(f"    warning: window outline failed ({exc})")
    return outline_window_fallback(window_scenes, win_start, win_end)


def outline_window(
    engine, scenes: list[dict], win_start: float, win_end: float
) -> list[dict]:
    window_scenes = scenes_in_range(scenes, win_start, win_end)
    if not window_scenes:
        return outline_window_fallback(window_scenes, win_start, win_end)
    messages = outline_window_prompt(scenes, win_start, win_end)
    if messages is None:
        return outline_window_fallback(window_scenes, win_start, win_end)
    text = engine.generate_one(messages)
    return parse_outline_window_response(text, scenes, win_start, win_end)


def build_topic_outline(
    engine,
    scenes: list[dict],
    window_minutes: int,
    whisper_segments: list[dict] | None = None,
    max_model_len: int = 16384,
    outline_tokens: int = 4096,
    outline_batch_size: int = 4,
) -> dict:
    if not scenes:
        return {"title": "Lecture", "subtitle": "", "topics": [], "references": []}

    end_time = max(s["end"] for s in scenes)
    window_sec = window_minutes * 60
    all_segments: list[dict] = []

    windows: list[tuple[float, float]] = []
    t = 0.0
    while t < end_time:
        windows.append((t, min(t + window_sec, end_time)))
        t = windows[-1][1]

    pending: list[tuple[float, float, list[dict] | None]] = []
    for win_start, win_end in windows:
        pending.append((win_start, win_end, outline_window_prompt(scenes, win_start, win_end)))

    print(
        f"  outline: {len(windows)} window(s), vLLM batch_size={outline_batch_size} ..."
    )
    win_num = 0
    for start in range(0, len(pending), outline_batch_size):
        batch = pending[start : start + outline_batch_size]
        batch_msgs = [item[2] for item in batch if item[2] is not None]
        batch_texts: list[str] = []
        if batch_msgs:
            batch_texts = engine.generate_batch_chunked(
                batch_msgs,
                len(batch_msgs),
                max_tokens=outline_tokens,
            )
        text_idx = 0
        for win_start, win_end, messages in batch:
            win_num += 1
            print(f"  outline window {win_num}: {fmt_range(win_start, win_end)} ...")
            if messages is None:
                segs = outline_window_fallback([], win_start, win_end)
            else:
                segs = parse_outline_window_response(
                    batch_texts[text_idx], scenes, win_start, win_end
                )
                text_idx += 1
            all_segments.extend(segs)

    print("  merging into major topics ...")
    merge_prompt = MERGE_TOPICS_PROMPT.format(
        segments_json=json.dumps(all_segments[:80], indent=2, ensure_ascii=False)[:12000]
    )
    try:
        outline = parse_json_block(engine.generate_one(text_messages(merge_prompt)))
        if isinstance(outline, dict) and outline.get("topics"):
            return polish_outline(
                engine,
                outline,
                scenes,
                whisper_segments,
                max_model_len,
                outline_tokens,
            )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"  warning: topic merge failed ({exc}), using heuristic grouping")

    return polish_outline(
        engine,
        heuristic_topic_outline(scenes, window_sec),
        scenes,
        whisper_segments,
        max_model_len,
        outline_tokens,
    )


def heuristic_topic_outline(scenes: list[dict], window_sec: float) -> dict:
    """Fallback topic tree from time windows."""
    end_time = max(s["end"] for s in scenes)
    topics = []
    tid = 1
    t = 0.0
    while t < end_time:
        win_end = min(t + window_sec * 2, end_time)
        win_scenes = scenes_in_range(scenes, t, win_end)
        segments = []
        sub_t = t
        while sub_t < win_end:
            sub_end = min(sub_t + window_sec / 2, win_end)
            sub_scenes = scenes_in_range(scenes, sub_t, sub_end)
            if sub_scenes:
                segments.append(
                    {
                        "start_sec": sub_scenes[0]["start"],
                        "end_sec": sub_scenes[-1]["end"],
                        "title": f"Segment at {fmt_time(sub_t)}",
                        "summary": "Material covered in this segment.",
                        "key_points": [],
                        "scene_ids": [s["scene_id"] for s in sub_scenes[:5]],
                    }
                )
            sub_t = sub_end
        topics.append(
            {
                "topic_id": tid,
                "title": f"Part {tid} ({fmt_range(t, win_end)})",
                "start_sec": t,
                "end_sec": win_end,
                "summary": "Overview of material covered in this part of the lecture.",
                "segments": segments,
            }
        )
        tid += 1
        t = win_end
    return {
        "title": "Lecture Report",
        "subtitle": "",
        "topics": topics,
        "references": [],
    }


SPEECH_START = re.compile(
    r"^(So |And |Here |Now |Okay |Um |That's |This is |You have )",
    re.I,
)


def is_junk_segment_title(title: str) -> bool:
    title = (title or "").strip()
    if not title or len(title) < 4:
        return True
    if SKIP_KEYWORDS.search(title) or JUNK_SEGMENT_TITLE.search(title):
        return True
    return False


def is_weak_segment_title(title: str) -> bool:
    title = (title or "").strip()
    if is_junk_segment_title(title):
        return True
    if title.endswith("..."):
        return True
    if re.match(r"^The (?:lecture|slide)\b", title, re.I):
        return True
    if re.match(r"^(Its|It|These|Their|To|Students)\b", title, re.I):
        return True
    return False


def teaching_phrase(text: str) -> str:
    """Turn meta lecture/slide phrasing into a plain concept phrase."""
    text = (text or "").strip().rstrip(".")
    if not text:
        return ""
    text = re.sub(
        r"^The lecture (?:explains|emphasizes|highlights|encourages|introduces|discusses) "
        r"(?:how |that )?",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"^The slide (?:presents|shows|contains|lacks) ",
        "",
        text,
        flags=re.I,
    )
    text = text.strip(" .")
    if not text or re.match(r"^(Its|It|These|Their|To|Students)\b", text, re.I):
        return ""
    return text[0].upper() + text[1:]


def content_from_summary(summary: str) -> str:
    """Keep teaching content; drop slide-absence meta from summaries."""
    summary = (summary or "").strip()
    if ":" in summary:
        left, right = summary.split(":", 1)
        if is_weak_segment_title(left):
            summary = right.strip()
    sentences = re.split(r"(?<=[.!?])\s+", summary)
    kept: list[str] = []
    for sentence in sentences:
        if not sentence.strip():
            continue
        if JUNK_SEGMENT_TITLE.search(sentence):
            continue
        if re.search(
            r"\b(slide (?:contains no|lacks|presents)|contains no equations?|"
            r"no visual aids?|reinforcing the (?:conceptual|abstract)|"
            r"abstract nature of)\b",
            sentence,
            re.I,
        ):
            continue
        phrase = teaching_phrase(sentence)
        if phrase and not is_weak_segment_title(phrase):
            kept.append(phrase if phrase.endswith((".", "!", "?")) else phrase + ".")
    return " ".join(kept).strip() or teaching_phrase(summary) or summary


def title_from_transcript(lookup: dict[int, dict], scene_ids: list) -> str:
    for sid in scene_ids:
        scene = lookup.get(int(sid))
        if not scene:
            continue
        for field in ("transcript", "vlm_description"):
            text = str(scene.get(field, "")).strip()
            if len(text) < 24:
                continue
            if SPEECH_START.match(text):
                text = re.sub(
                    r"^(?:So |And |Here |Now |Okay |Um |That's |This is |You have )+",
                    "",
                    text,
                    flags=re.I,
                ).strip()
            if len(text) < 24 or is_junk_segment_title(text):
                continue
            phrase = teaching_phrase(text.split(".")[0])
            if phrase and not is_weak_segment_title(phrase):
                return truncate(phrase, 60)
            return truncate(text, 60)
    return ""


def derive_segment_title(
    seg: dict,
    lookup: dict[int, dict] | None = None,
    *,
    topic_title: str = "",
) -> str:
    for point in seg.get("key_points") or []:
        point = teaching_phrase(str(point).strip())
        if point and not is_weak_segment_title(point):
            return truncate(point, 60)

    if lookup:
        from_transcript = title_from_transcript(lookup, seg.get("scene_ids") or [])
        if from_transcript and not is_weak_segment_title(from_transcript):
            return from_transcript

    summary = content_from_summary(str(seg.get("summary", "")))
    if summary:
        first = re.split(r"[.!?]", summary)[0].strip()
        phrase = teaching_phrase(first)
        if phrase and not is_weak_segment_title(phrase):
            return truncate(phrase, 60)

    if topic_title:
        return truncate(topic_title, 60)
    return "Lecture segment"


def sanitize_segment_fields(
    seg: dict,
    lookup: dict[int, dict] | None = None,
    *,
    topic_title: str = "",
) -> dict:
    item = dict(seg)
    if is_weak_segment_title(str(item.get("title", ""))):
        item["title"] = derive_segment_title(
            item, lookup, topic_title=topic_title
        )
    item["summary"] = content_from_summary(str(item.get("summary", "")))
    if not item["summary"] or is_weak_segment_title(item["summary"]):
        item["summary"] = item["title"]
    return item


def sanitize_outline(outline: dict, scenes: list[dict] | None = None) -> dict:
    lookup = scene_by_id(scenes or [])
    cleaned = dict(outline)
    topics: list[dict] = []
    for topic in outline.get("topics", []):
        item = dict(topic)
        topic_title = str(topic.get("title", ""))
        item["segments"] = [
            sanitize_segment_fields(seg, lookup, topic_title=topic_title)
            for seg in topic.get("segments", [])
        ]
        topics.append(item)
    cleaned["topics"] = topics
    return cleaned


def segment_title_by_time(outline: dict) -> dict[str, str]:
    titles: dict[str, str] = {}
    for topic in outline.get("topics", []):
        for seg in topic.get("segments", []):
            key = fmt_range(seg.get("start_sec", 0), seg.get("end_sec", 0))
            titles[key] = str(seg.get("title", ""))
    return titles


def sanitize_segment_headings(md: str, outline: dict | None = None) -> str:
    title_map = segment_title_by_time(outline) if outline else {}

    def _fix(match: re.Match[str]) -> str:
        time_range = match.group(1)
        title = match.group(2).strip()
        if is_weak_segment_title(title):
            title = title_map.get(time_range) or teaching_phrase(title)
            if is_weak_segment_title(title):
                title = "Lecture segment"
        return f"### {time_range} — {title}"

    return re.sub(
        r"^###\s+(\d{2}:\d{2}(?::\d{2})?[–-]\d{2}:\d{2}(?::\d{2})?)\s+—\s+(.+)$",
        _fix,
        md,
        flags=re.MULTILINE,
    )


def strip_slide_absence_sentences(md: str) -> str:
    md = _SLIDE_ABSENCE_SENTENCE.sub("", md)
    md = re.sub(
        r"(?m)^.*\b(reinforcing the conceptual nature|emphasizing the conceptual framework)\b.*\n?",
        "",
        md,
        flags=re.I,
    )
    return md


def looks_incomplete(text: str, *, is_title: bool = False) -> bool:
    text = (text or "").strip()
    min_len = 4 if is_title else 12
    if not text or len(text) < min_len:
        return True
    if is_title and is_weak_segment_title(text):
        return True
    if text.endswith("..."):
        return True
    if "**" in text or "```" in text:
        return True
    if re.search(r"Equations \(LaTeX\)", text, re.I):
        return True
    if not is_title and SPEECH_START.match(text):
        return True
    if not is_title and text[-1] not in ".!?" and len(text.split()) > 8:
        return True
    return False


def outline_quality_issues(outline: dict) -> list[dict]:
    issues: list[dict] = []
    for topic in outline.get("topics", []):
        tid = int(topic.get("topic_id") or 0)
        for field in ("title", "summary"):
            val = topic.get(field, "")
            if looks_incomplete(val, is_title=(field == "title")):
                issues.append(
                    {
                        "topic_id": tid,
                        "type": "incomplete_metadata",
                        "severity": "major",
                        "field": field,
                        "description": f"Topic {field} is incomplete or low quality: {truncate(val, 80)}",
                    }
                )
        for seg in topic.get("segments", []):
            for field in ("title", "summary"):
                val = seg.get(field, "")
                if looks_incomplete(val, is_title=(field == "title")):
                    issues.append(
                        {
                            "topic_id": tid,
                            "type": "incomplete_metadata",
                            "severity": "major",
                            "field": f"segment.{field}",
                            "description": f"Segment {field} is incomplete or raw transcript: {truncate(val, 80)}",
                        }
                    )
    return issues


def merge_outline_structure(original: dict, polished: dict) -> dict:
    """Keep timing/scene_ids from original; take polished text fields."""
    orig_topics = {
        int(t.get("topic_id", 0)): t for t in original.get("topics", [])
    }
    topics: list[dict] = []
    for pt in polished.get("topics", []):
        tid = int(pt.get("topic_id", 0))
        ot = orig_topics.get(tid, pt)
        topic = {
            "topic_id": tid,
            "title": pt.get("title") or ot.get("title", ""),
            "start_sec": ot.get("start_sec", pt.get("start_sec", 0)),
            "end_sec": ot.get("end_sec", pt.get("end_sec", 0)),
            "summary": pt.get("summary") or ot.get("summary", ""),
            "segments": [],
        }
        orig_segs = ot.get("segments", [])
        pol_segs = pt.get("segments", [])
        for j, os in enumerate(orig_segs):
            ps = pol_segs[j] if j < len(pol_segs) else {}
            topic["segments"].append(
                {
                    "start_sec": os.get("start_sec", ps.get("start_sec", 0)),
                    "end_sec": os.get("end_sec", ps.get("end_sec", 0)),
                    "title": ps.get("title") or os.get("title", ""),
                    "summary": ps.get("summary") or os.get("summary", ""),
                    "key_points": ps.get("key_points") or os.get("key_points", []),
                    "scene_ids": os.get("scene_ids") or ps.get("scene_ids", []),
                }
            )
        if not topic["segments"] and orig_segs:
            topic["segments"] = orig_segs
        topics.append(topic)

    if not topics:
        return original

    return {
        "title": polished.get("title") or original.get("title", "Lecture Report"),
        "subtitle": polished.get("subtitle") or original.get("subtitle", ""),
        "topics": topics,
        "references": polished.get("references") or original.get("references", []),
    }


def polish_outline(
    engine,
    outline: dict,
    scenes: list[dict],
    whisper_segments: list[dict] | None,
    max_model_len: int = 16384,
    outline_tokens: int = 4096,
) -> dict:
    issues = outline_quality_issues(outline)
    if not issues:
        return outline

    print(f"  polishing outline ({len(issues)} metadata issue(s)) ...")
    budget = prompt_char_budget(max_model_len, outline_tokens)
    transcript_excerpt = ""
    if whisper_segments:
        transcript_excerpt = compact_whisper_transcript(whisper_segments, budget // 3)
    elif scenes:
        transcript_excerpt = transcript_highlights(scenes, max_windows=12, chars=500)

    prompt = OUTLINE_POLISH_PROMPT.format(
        outline_json=fit_prompt(
            json.dumps(outline, indent=2, ensure_ascii=False), budget // 2
        ),
        transcript_excerpt=transcript_excerpt or "(no transcript)",
    )
    try:
        polished = parse_json_block(engine.generate_one(text_messages(prompt)))
        if isinstance(polished, dict) and polished.get("topics"):
            merged = merge_outline_structure(outline, polished)
            remaining = outline_quality_issues(merged)
            if len(remaining) < len(issues):
                print(f"  outline polished ({len(issues)} → {len(remaining)} issues)")
                return merged
            print(f"  outline polish partial ({len(remaining)} issues remain)")
            return merged
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
        print(f"  warning: outline polish failed ({exc})")
    return outline


def pick_main_scenes_from_outline(outline: dict, scenes: list[dict], max_n: int) -> list[dict]:
    lookup = scene_by_id(scenes)
    seen: set[int] = set()
    picked: list[dict] = []
    for topic in outline.get("topics", []):
        for seg in topic.get("segments", []):
            for sid in seg.get("scene_ids", []):
                sid = int(sid)
                if sid in seen or sid not in lookup:
                    continue
                sc = lookup[sid]
                if scene_importance_score(sc) < 0 and len(picked) > max_n // 2:
                    continue
                seen.add(sid)
                picked.append(
                    {
                        "scene_id": sid,
                        "section": seg.get("title") or topic.get("title", ""),
                        "reason": truncate(sc.get("vlm_description", ""), 100),
                    }
                )
                if len(picked) >= max_n:
                    return picked
    # Fill from top scored scenes
    for sc in sorted(scenes, key=scene_importance_score, reverse=True):
        if sc["scene_id"] not in seen:
            picked.append(
                {
                    "scene_id": sc["scene_id"],
                    "section": "Key slide",
                    "reason": "High content score",
                }
            )
        if len(picked) >= max_n:
            break
    return picked


# ---------------------------------------------------------------------------
# Phase 2 — detailed sections (one per topic) + refinement passes
# ---------------------------------------------------------------------------

TOPIC_DETAIL_PROMPT = """You are writing FIRST-PRINCIPLES lecture notes for a student who knows NOTHING about this topic yet.
Explain like you are teaching a curious beginner — simple words first, then depth. Never hand-wave.

Topic {topic_id}: **{topic_title}** ({time_range})
Papers available for grounding: {papers_available}

## Layout rules (STRICT)
Each segment MUST use this exact block order:
1. `### {{segment_time}} — {{segment_title}}` — use the `segment_time` field from JSON (MM:SS or HH:MM:SS). NEVER use raw `start_sec`/`end_sec` numbers.
2. **Figures first** — copy every string from the segment `figures` list exactly (one per line), immediately after the heading
3. **All teaching sections below** (every section below is MANDATORY for every segment — write substantial paragraphs, not one-liners)

## Math rules (STRICT)
- TeX/LaTeX only: inline `$...$`, display `$$...$$`
- Define EVERY symbol when you introduce an equation
- NEVER use Unicode math or plain-text formulas

## MANDATORY sections (after figures, for EACH segment, in this order)

- **What is this?** — Precise definition in plain language (2–4 sentences). Assume zero prior knowledge.
- **Why do we need it?** — What problem does this solve? Why did anyone invent this idea?
- **What existed before?** — Prior approaches and their limitations.
- **Core intuition** — Analogy or story that makes the idea click. No jargon without explanation.
- **What the lecturer said** — Preserve the lecturer's key explanations as quoted or paraphrased transcript (include specific phrases, examples, numbers they mention).
- **Deep technical explanation** — Full mechanism explained step-by-step in prose.
- **Mathematics** — Every formula in TeX; derive or explain where each term comes from.
- **How it works (step-by-step)** — Numbered list of the mechanism/process.
- **Paper connection** — {paper_instruction}
- **Concrete example** — Walk through one specific example with numbers or a mini scenario.
- **If I were implementing this** — What code/modules/data structures would I build? Be specific.
- **Common misunderstandings** — 2–3 mistakes beginners make and why they are wrong.
- **Builds on / leads to** — Link to concepts from earlier/later in the lecture.
- `> 🔁 Revisit lecture:` {{segment_time}} (same human-readable `segment_time` from JSON)

## Quality rules
- Timestamps must be human-readable (e.g. `01:00:23–01:03:20`), never raw seconds like `3623.053`.
- Never write truncation markers or meta phrases like "truncated for context limit".
- Do NOT write repetitive disclaimer lists ("We will not assume..."). Never repeat the same sentence or bullet.
- Do NOT describe what is missing from slides ("no equations on slide", "slide contains no diagrams"). Teach what is happening.
- Segment headings must name the concept being taught, not slide visuals.
- Do NOT summarize in 1–2 sentences where a paragraph is needed.
- Do NOT skip segments. Do NOT merge segments.
- Do NOT put figures after text.
- Prefer completeness over brevity — redundancy that aids understanding is OK.
- Use the slide descriptions AND transcript AND paper context below — synthesize all three.

Segments JSON (copy `figures` lines exactly after each ### heading):
{segments_json}

Context (slides, transcript, papers):
{context}
"""

PREREQUISITES_PROMPT = """You are preparing a student to understand a technical lecture from scratch.

Lecture: **{title}** — _{subtitle}_

Write a `## Prerequisites` section in Markdown for a beginner. Be granular and concrete.

Structure (STRICT):
1. Opening paragraph (3–5 sentences): what minimal background you assume and what the lecture will teach.
2. For EACH prerequisite area relevant to THIS lecture (e.g. probability, LLM inference, unit testing):
   - `### Area name`
   - Nested bullet tree of specific sub-skills (not vague labels like "know ML")
   - Under each leaf: **Why you need this:** one sentence linking to something the lecture or paper uses
3. `### Prerequisite dependency order` — numbered list: learn A before B because...
4. If papers are provided: `### Paper-specific prerequisites` — concepts needed to read the cited papers.

Rules:
- Be specific to THIS lecture (not a generic CS curriculum).
- Explain jargon when you use it.
- If a prerequisite itself needs another concept, nest it (recursive dependencies).
- Write 800–2000 words. Teach with bullets and short paragraphs — not disclaimer lists.
- Do NOT write repetitive "We will not assume..." or "You don't need to know..." sentences.
- Do NOT repeat the same sentence, bullet, or phrase. Stop when the structure above is complete.
- End with a complete sentence — never trail off mid-thought.

Outline:
{outline_summary}

Transcript excerpt:
{transcript_excerpt}

{paper_block}

Return ONLY the Markdown starting with `## Prerequisites`. No preamble.
"""

LEARNING_ROADMAP_PROMPT = """Write a `## Learning Roadmap` for a technical lecture. This is NOT a summary — it is a map for learning.

Lecture: **{title}** — _{subtitle}_

Answer these in order (use ### headings):
1. **What is this lecture about?** (2–3 sentences, plain language)
2. **What problem are we solving?** — Why does this problem exist? Who cares?
3. **Why does it matter?** — Real-world stakes, research motivation.
4. **What will you understand after?** — Concrete capabilities, not vague "understand X".
5. **What will you be able to implement?** — Specific engineering outcomes.
6. **Concept roadmap (in order)** — Numbered list of concepts; each entry: name → one-line intuition → what it enables next.
7. **How concepts build on each other** — Short dependency paragraph (A requires B because...).
8. **Paper grounding** — {paper_roadmap_instruction}

Rules:
- Write for a beginner who will read the detailed notes next.
- Be specific to this lecture's content.
- 500–1200 words.
- Do NOT write repetitive disclaimer lists ("We will not assume..."). Never repeat the same sentence.
- End with a complete sentence.

Outline:
{outline_summary}

{paper_block}

Return ONLY Markdown starting with `## Learning Roadmap`. No preamble.
"""

REFINE_TOPIC_PROMPT = """Expand and refine ONE topic section to FIRST-PRINCIPLES teaching depth. Return ONLY this topic's markdown.

Topic {topic_id}: **{topic_title}** ({time_range})
Papers available: {papers_available}

The draft may be too shallow. Your job is to make it teach like explaining to a beginner who knows nothing.

Tasks:
1. Cover EVERY segment — do not drop or merge
2. Figures immediately after each `###` heading (before text)
3. EVERY segment must have ALL sections: What is this?, Why do we need it?, What existed before?, Core intuition, What the lecturer said, Deep technical explanation, Mathematics, How it works (step-by-step), Paper connection, Concrete example, If I were implementing this, Common misunderstandings, Builds on / leads to, Revisit lecture timestamp
4. Expand any section that is missing, one sentence, or vague — add paragraphs
5. Pull specific details from transcript and paper context below
6. TeX only for math; define all symbols
7. Do NOT add meta commentary about missing content or slide visuals ("no equations on slide")
8. Use `segment_time` from JSON for every ### heading and Revisit line (MM:SS or HH:MM:SS — never raw seconds)
9. Never output "truncated for context limit" or similar meta text
10. Segment headings must name the concept taught — not what is absent from the slide

Required segments:
{segments_json}

Context:
{context}

Current draft (expand this):
{draft}

Return the complete topic section starting with `## Topic {topic_id}:`. No preamble.
"""

COVERAGE_AUDIT_PROMPT = """Audit a lecture report against the original Whisper transcript.

Return ONLY JSON:
{{
  "passes": false,
  "coverage_score": 85,
  "issues": [
    {{
      "topic_id": 1,
      "time_range": "MM:SS–MM:SS",
      "severity": "major",
      "type": "missing_content",
      "description": "what is wrong or missing",
      "transcript_excerpt": "relevant transcript quote",
      "suggested_fix": "what to add or correct"
    }}
  ],
  "summary": "one paragraph assessment"
}}

Rules:
- coverage_score: 0–100 — how completely the report reflects the transcript
- passes=true ONLY if coverage_score >= 90 AND there are no major incorrect statements
- type is one of: missing_content, incorrect, incomplete, incomplete_metadata
- severity is major or minor
- List every meaningful gap: concepts, formulas, definitions, examples in transcript but absent from report
- Flag report statements that contradict the transcript
- Flag incomplete metadata: summaries ending with "...", raw transcript pasted as titles/summaries, markdown/LaTeX dumps in titles, speech fragments ("So here you have...")
- Check BOTH the timeline/index front matter AND the detailed topic sections

Lecture duration: {duration}

Topic outline:
{outline_summary}

Whisper transcript (timestamped):
{transcript}

Report front matter (timeline + index):
{front_matter}

Report detailed sections:
{report_body}
"""

COVERAGE_FIX_PROMPT = """Fix ONE topic section based on a coverage audit against the lecture transcript.

Topic {topic_id}: **{topic_title}** ({time_range})

Audit issues to address (fix ALL of these):
{issues_json}

Whisper transcript for this topic:
{transcript}

Required segments (preserve structure — figures immediately after each ### heading):
{segments_json}

Current section:
{draft}

Tasks:
1. Add ALL missing transcript content identified in the audit
2. Correct any inaccurate statements
3. Keep layout: ### heading → figures → What is covered / Key points / Equations & definitions → Revisit timestamp
4. TeX math only (`$...$` inline, `$$...$$` display)
5. Do NOT add "what you might miss", skip warnings, or similar meta sections

Return the complete topic section starting with `## Topic {topic_id}:`. No preamble.
"""


_CONTEXT_TRUNC_RE = re.compile(
    r"\.\.\.\s*\[truncated for context limit\]", re.I
)
_ASSUMPTION_SPAM_RE = re.compile(r"We will not assume[^.]+\.", re.I)
_RAW_SEC_HEADING_RE = re.compile(
    r"^###\s+(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)\s+—\s+",
    re.MULTILINE,
)


def remove_assumption_disclaimers(text: str) -> str:
    """Strip 'We will not assume...' disclaimer spam without truncating the rest."""
    if len(_ASSUMPTION_SPAM_RE.findall(text)) < 2:
        return text

    parts = re.split(r"(\n\n+)", text)
    kept: list[str] = []
    for part in parts:
        if part.startswith("\n") or not _ASSUMPTION_SPAM_RE.search(part):
            kept.append(part)
    text = "".join(kept)

    text = re.sub(
        r"(?m)^You may think you know nothing[^\n]*\n+",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"(?m)^We['\u2019]ll start from zero[^\n]*\n+",
        "",
        text,
    )
    text = re.sub(
        r"(?m)^But you must understand[^\n]*\n+",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r'[ \t]*[""][^"\n]*[""]?[ \t]*', " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def collapse_tail_repetition(text: str, *, min_unit: int = 25, min_repeats: int = 3) -> str:
    """Trim degenerate loops where the same phrase repeats at the end."""
    if len(text) < min_unit * min_repeats:
        return text
    for unit_len in range(min(len(text) // min_repeats, 500), min_unit - 1, -1):
        unit = text[-unit_len:]
        if not unit.strip():
            continue
        repeats = 1
        pos = len(text) - unit_len
        while pos >= unit_len and text[pos - unit_len : pos] == unit:
            repeats += 1
            pos -= unit_len
        if repeats >= min_repeats:
            return text[: len(text) - unit_len * (repeats - 1)].rstrip()
    return text


def strip_dangling_tail(md: str) -> str:
    """Remove incomplete trailing fragments (e.g. cut-off 'The paper')."""
    lines = md.splitlines()
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if last.startswith(("#", "-", "*", ">", "|", "!", "`")):
            break
        if len(last) >= 60 or re.search(r'[.!?"\'\)]$', last):
            break
        lines.pop()
    return "\n".join(lines)


def strip_orphan_quote_fragments(md: str) -> str:
    """Remove lines that are mostly quote fragments left after disclaimer removal."""
    lines: list[str] = []
    for line in md.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append(line)
            continue
        if re.match(r"^We['\u2019]ll start from zero\b", stripped, re.I):
            continue
        without_quotes = re.sub(r'[""\s]+', "", stripped)
        if len(without_quotes) < 20 and re.search(r'[""]', stripped):
            continue
        lines.append(line)
    return "\n".join(lines)


def collapse_duplicate_lines(md: str) -> str:
    """Drop lines that repeat 3+ times in a row."""
    out: list[str] = []
    prev_key: str | None = None
    streak = 0
    for line in md.splitlines():
        key = line.strip()
        if key and key == prev_key:
            streak += 1
            if streak >= 2:
                continue
        else:
            streak = 0
        if key:
            prev_key = key
        out.append(line)
    return "\n".join(out)


def sanitize_generated_markdown(md: str) -> str:
    """Remove common LLM degeneration patterns from generated markdown."""
    md = remove_assumption_disclaimers(md)
    md = strip_orphan_quote_fragments(md)
    md = collapse_tail_repetition(md)
    md = collapse_duplicate_lines(md)
    md = strip_dangling_tail(md)
    return md


def audit_report_block(md: str, label: str) -> list[str]:
    """Return human-readable warnings for remaining quality issues."""
    issues: list[str] = []
    if _CONTEXT_TRUNC_RE.search(md):
        issues.append(f"{label}: truncation marker still present")
    if _ASSUMPTION_SPAM_RE.search(md):
        issues.append(f"{label}: 'We will not assume' disclaimer list remains")
    if _RAW_SEC_HEADING_RE.search(md):
        issues.append(f"{label}: raw-second timestamps in headings")
    lines = [line.strip() for line in md.strip().splitlines() if line.strip()]
    if lines:
        last = lines[-1]
        if (
            len(last) < 40
            and not re.search(r"[.!?\"'\)]$", last)
            and not last.startswith(("#", "-", "*", ">", "|", "!", "`"))
        ):
            issues.append(f"{label}: dangling fragment at end ({last[:48]!r})")
    prev: str | None = None
    dup_streak = 0
    for line in lines:
        if line == prev:
            dup_streak += 1
            if dup_streak >= 2:
                issues.append(f"{label}: repeated consecutive lines")
                break
        else:
            dup_streak = 0
        prev = line
    return issues


def sanitize_report_blocks(
    report_dir: Path,
    outline: dict,
    *,
    rewrite: bool = True,
) -> list[str]:
    """Sanitize prerequisites, roadmap, and topic files; return audit warnings."""
    warnings: list[str] = []
    topics_dir = report_dir / "topics"

    for name in ("prerequisites.md", "learning_roadmap.md"):
        path = report_dir / name
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8")
        clean = strip_miss_sections(raw, outline)
        warnings.extend(audit_report_block(clean, name))
        if rewrite and clean != raw:
            path.write_text(clean.strip() + "\n", encoding="utf-8")

    for topic in outline.get("topics", []):
        tid = topic.get("topic_id", 0)
        path = topic_markdown_path(topics_dir, tid)
        if not path.exists():
            warnings.append(f"topic_{tid:02d}.md: missing")
            continue
        raw = path.read_text(encoding="utf-8")
        clean = strip_miss_sections(raw, outline)
        warnings.extend(audit_report_block(clean, path.name))
        if rewrite and clean != raw:
            path.write_text(clean.strip() + "\n", encoding="utf-8")

    return warnings


def normalize_segment_timestamps(md: str) -> str:
    """Convert raw-second timestamps in headings/revisit lines to MM:SS / HH:MM:SS."""

    def _heading(m: re.Match[str]) -> str:
        return f"### {fmt_range(float(m.group(1)), float(m.group(2)))} — {m.group(3)}"

    md = re.sub(
        r"^###\s+(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)\s+—\s+(.+)$",
        _heading,
        md,
        flags=re.MULTILINE,
    )

    def _revisit(m: re.Match[str]) -> str:
        return f"> 🔁 Revisit lecture: {fmt_range(float(m.group(1)), float(m.group(2)))}"

    md = re.sub(
        r"^>\s*🔁\s*Revisit lecture:\s*(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)\s*$",
        _revisit,
        md,
        flags=re.MULTILINE,
    )
    return md


def strip_miss_sections(md: str, outline: dict | None = None) -> str:
    """Remove skip/warning blocks and leaked prompt artifacts from generated markdown."""
    md = _CONTEXT_TRUNC_RE.sub("", md)
    md = re.sub(
        r"(?ms)^## What you might miss if you skip this part:\s*\n.*?(?=^## |^> 🔁|\Z)",
        "",
        md,
    )
    md = sanitize_generated_markdown(md)
    md = strip_slide_absence_sentences(md)
    md = sanitize_segment_headings(md, outline)
    md = normalize_segment_timestamps(md)
    return re.sub(r"\n{3,}", "\n\n", md)


def topic_markdown_path(topics_dir: Path, topic_id: int) -> Path:
    return topics_dir / f"topic_{topic_id:02d}.md"


def figure_md(scene_id: int, *, for_topics_dir: bool = False) -> str:
    prefix = "../attachments" if for_topics_dir else "attachments"
    return f"![image]({prefix}/scene_{scene_id:03d}.png)"


FIGURE_LINE = re.compile(
    r"^!\[image\]\((?:\.\./)?attachments/scene_\d{3}\.png\)\s*$"
    r"|^!\[\[scene_\d{3}\.png\]\]\s*$",
    re.MULTILINE,
)


def scene_ids_in_text(text: str) -> list[int]:
    return [int(s) for s in dict.fromkeys(re.findall(r"scene_(\d{3})\.png", text))]


def rewrite_figure_paths(md: str, *, for_topics_dir: bool) -> str:
    """Normalize wikilinks / wrong prefixes to the canonical markdown image path."""
    md = re.sub(
        r"!\[image\]\((?:\.\./)?attachments/scene_(\d{3})\.png\)",
        lambda m: figure_md(int(m.group(1)), for_topics_dir=for_topics_dir),
        md,
    )
    md = re.sub(
        r"!\[\[scene_(\d{3})\.png\]\]",
        lambda m: figure_md(int(m.group(1)), for_topics_dir=for_topics_dir),
        md,
    )
    return md


def enrich_segments(
    segs: list[dict],
    lookup: dict[int, dict],
    *,
    for_topics_dir: bool = False,
    topic_title: str = "",
) -> list[dict]:
    """Add human-readable timestamps and markdown figure links for each segment."""
    enriched = []
    for seg in segs:
        item = sanitize_segment_fields(seg, lookup, topic_title=topic_title)
        start = float(item.get("start_sec", 0))
        end = float(item.get("end_sec", start))
        item["segment_time"] = fmt_range(start, end)
        figures = []
        for sid in item.get("scene_ids") or []:
            sid = int(sid)
            if lookup.get(sid):
                figures.append(figure_md(sid, for_topics_dir=for_topics_dir))
        item["figures"] = figures
        enriched.append(item)
    return enriched


def normalize_segment_images(md: str, *, for_topics_dir: bool = False) -> str:
    """Move slide images to immediately after each ### heading, before text."""
    md = rewrite_figure_paths(md, for_topics_dir=for_topics_dir)
    if "### " not in md:
        return md

    chunks = re.split(r"(?m)(?=^### )", md)
    out: list[str] = []
    for chunk in chunks:
        if not chunk.startswith("### "):
            out.append(chunk)
            continue
        header_line, _, rest = chunk.partition("\n")
        scene_ids = scene_ids_in_text(rest)
        images = [figure_md(sid, for_topics_dir=for_topics_dir) for sid in scene_ids]
        rest_clean = FIGURE_LINE.sub("", rest)
        rest_clean = re.sub(r"\n{3,}", "\n\n", rest_clean.strip())
        block = header_line + "\n\n"
        if images:
            block += "\n".join(images) + "\n\n"
        if rest_clean:
            block += rest_clean + "\n"
        out.append(block)
    return "".join(out)


def normalize_report_layout(md: str, outline: dict | None = None) -> str:
    """Normalize image placement across the full report."""
    md = normalize_segment_images(md, for_topics_dir=False)
    return strip_miss_sections(md, outline)


def context_for_scenes(
    scenes: list[dict],
    scene_ids: list[int],
    lookup: dict,
    *,
    vlm_limit: int = 1500,
    audio_limit: int = 2500,
) -> str:
    lines = []
    for sid in scene_ids[:12]:
        sc = lookup.get(int(sid))
        if not sc:
            continue
        lines.append(
            f"scene_{sid:03d}|{fmt_range(sc['start'], sc['end'])}|"
            f"{truncate(sc.get('vlm_description', ''), vlm_limit)}|"
            f"{truncate(sc.get('transcript', ''), audio_limit)}"
        )
    return "\n".join(lines) or "(no slides)"


def load_paper_context(run_dir: Path, *, max_chars: int = 32000) -> str:
    manifest_path = run_dir / "papers" / "manifest.json"
    if not manifest_path.exists():
        return ""

    manifest = load_json(manifest_path)
    blocks: list[str] = []
    per_paper = max(4000, max_chars // max(len(manifest), 1))
    for entry in manifest:
        md_path = run_dir / "papers" / entry.get("markdown", "")
        if not md_path.exists():
            continue
        title = entry.get("title") or md_path.stem
        body = md_path.read_text(encoding="utf-8").strip()
        blocks.append(f"### Paper: {title}\n{fit_prompt(body, per_paper)}")

    if not blocks:
        return ""
    return "Research paper grounding (parsed PDFs — use for equations, method, motivation, results):\n" + "\n\n".join(blocks)


def paper_block_for_prompt(paper_context: str) -> str:
    if not paper_context.strip():
        return "No research papers provided."
    return f"Research papers (authoritative technical source):\n{fit_prompt(paper_context, 14000)}"


def paper_instruction(has_papers: bool) -> str:
    if has_papers:
        return (
            "MANDATORY — Map this segment to the paper: cite specific sections, equations, "
            "figures, method steps, or results from the paper context. Explain what the paper "
            "says vs. what the lecturer simplified. Quote key equations in TeX."
        )
    return "No papers provided — connect to standard literature if the lecturer mentions prior work."


def paper_roadmap_instruction(has_papers: bool) -> str:
    if has_papers:
        return (
            "For each provided paper: motivation → contribution → key method → "
            "important equations → how the lecture maps to paper sections."
        )
    return "Note any papers or methods cited in the lecture outline."


def context_for_topic(
    topic: dict,
    scenes: list[dict],
    lookup: dict,
    whisper_segments: list[dict],
    paper_context: str = "",
) -> str:
    start = float(topic.get("start_sec", 0))
    end = float(topic.get("end_sec", 0))
    scene_ids: list[int] = []
    for seg in topic.get("segments", []):
        scene_ids.extend(int(s) for s in (seg.get("scene_ids") or []))

    slide_ctx = context_for_scenes(scenes, scene_ids, lookup)
    transcript_ctx = transcript_for_range(
        whisper_segments, start, end, max_chars=14000, window_sec=180.0
    )
    parts = [
        f"Slide context:\n{slide_ctx}",
        f"Full transcript for this topic (preserve lecturer details):\n{transcript_ctx}",
    ]
    if paper_context:
        parts.append(f"Paper grounding:\n{fit_prompt(paper_context, 10000)}")
    return "\n\n".join(parts)


def write_prerequisites_section(
    engine,
    outline: dict,
    whisper_segments: list[dict],
    paper_context: str,
    max_model_len: int,
) -> str:
    transcript_excerpt = compact_whisper_transcript(
        whisper_segments, max_chars=min(12000, prompt_char_budget(max_model_len, 4096) // 2)
    )
    body = engine.generate_one(
        text_messages(
            PREREQUISITES_PROMPT.format(
                title=outline.get("title", "Lecture"),
                subtitle=outline.get("subtitle", ""),
                outline_summary=outline_summary(outline),
                transcript_excerpt=transcript_excerpt,
                paper_block=paper_block_for_prompt(paper_context),
            )
        )
    )
    text = strip_miss_sections(body.strip())
    if not text.startswith("## Prerequisites"):
        text = "## Prerequisites\n\n" + text
    return text.strip() + "\n"


def write_learning_roadmap_section(
    engine,
    outline: dict,
    paper_context: str,
    max_model_len: int,
) -> str:
    has_papers = bool(paper_context.strip())
    body = engine.generate_one(
        text_messages(
            LEARNING_ROADMAP_PROMPT.format(
                title=outline.get("title", "Lecture"),
                subtitle=outline.get("subtitle", ""),
                outline_summary=outline_summary(outline),
                paper_block=paper_block_for_prompt(paper_context),
                paper_roadmap_instruction=paper_roadmap_instruction(has_papers),
            )
        )
    )
    text = strip_miss_sections(body.strip())
    if not text.startswith("## Learning Roadmap"):
        text = "## Learning Roadmap\n\n" + text
    return text.strip() + "\n"


def _learning_roadmap_messages(
    outline: dict, paper_context: str, max_model_len: int
) -> list[dict]:
    has_papers = bool(paper_context.strip())
    return text_messages(
        LEARNING_ROADMAP_PROMPT.format(
            title=outline.get("title", "Lecture"),
            subtitle=outline.get("subtitle", ""),
            outline_summary=outline_summary(outline),
            paper_block=paper_block_for_prompt(paper_context),
            paper_roadmap_instruction=paper_roadmap_instruction(has_papers),
        )
    )


def _prerequisites_messages(
    outline: dict,
    whisper_segments: list[dict],
    paper_context: str,
    max_model_len: int,
) -> list[dict]:
    transcript_excerpt = compact_whisper_transcript(
        whisper_segments, max_chars=min(12000, prompt_char_budget(max_model_len, 4096) // 2)
    )
    return text_messages(
        PREREQUISITES_PROMPT.format(
            title=outline.get("title", "Lecture"),
            subtitle=outline.get("subtitle", ""),
            outline_summary=outline_summary(outline),
            transcript_excerpt=transcript_excerpt,
            paper_block=paper_block_for_prompt(paper_context),
        )
    )


def write_front_matter_sections(
    engine,
    outline: dict,
    whisper_segments: list[dict],
    paper_context: str,
    max_model_len: int,
    *,
    front_batch_size: int = 2,
) -> tuple[str, str]:
    """Generate learning roadmap + prerequisites (batched when both needed)."""
    msgs = [
        _learning_roadmap_messages(outline, paper_context, max_model_len),
        _prerequisites_messages(outline, whisper_segments, paper_context, max_model_len),
    ]
    if front_batch_size > 1:
        bodies = engine.generate_batch_chunked(
            msgs, len(msgs), max_tokens=4096
        )
    else:
        bodies = [engine.generate_one(m, max_tokens=4096) for m in msgs]

    roadmap = strip_miss_sections(bodies[0].strip())
    if not roadmap.startswith("## Learning Roadmap"):
        roadmap = "## Learning Roadmap\n\n" + roadmap

    prereq = strip_miss_sections(bodies[1].strip())
    if not prereq.startswith("## Prerequisites"):
        prereq = "## Prerequisites\n\n" + prereq

    return roadmap.strip() + "\n", prereq.strip() + "\n"


def topic_detail_messages(
    topic: dict,
    scenes: list[dict],
    whisper_segments: list[dict],
    max_model_len: int,
    section_tokens: int,
    paper_context: str = "",
) -> list[dict]:
    lookup = scene_by_id(scenes)
    segs = enrich_segments(
        topic.get("segments", []),
        lookup,
        for_topics_dir=True,
        topic_title=str(topic.get("title", "")),
    )
    budget = prompt_char_budget(max_model_len, section_tokens)
    has_papers = bool(paper_context.strip())
    segs_json = compact_segments_json(segs, max_chars=min(10000, budget // 4))
    context = fit_prompt(
        context_for_topic(topic, scenes, lookup, whisper_segments, paper_context),
        budget // 2,
    )
    return text_messages(
        TOPIC_DETAIL_PROMPT.format(
            topic_id=topic.get("topic_id", 0),
            topic_title=topic.get("title", "Topic"),
            time_range=fmt_range(topic.get("start_sec", 0), topic.get("end_sec", 0)),
            papers_available="yes" if has_papers else "no",
            paper_instruction=paper_instruction(has_papers),
            segments_json=segs_json,
            context=context,
        )
    )


def topic_section_header(topic: dict) -> str:
    return (
        f"## Topic {topic.get('topic_id', 0)}: {topic.get('title', 'Topic')}\n"
        f"> ⏱ Session: {fmt_range(topic.get('start_sec', 0), topic.get('end_sec', 0))}\n"
    )


def finalize_topic_section(
    topic: dict, body: str, outline: dict | None = None
) -> str:
    header = topic_section_header(topic)
    return strip_miss_sections(
        header + "\n" + normalize_segment_images(body, for_topics_dir=True),
        outline,
    )


def write_topic_section(
    engine,
    topic: dict,
    scenes: list[dict],
    whisper_segments: list[dict],
    max_model_len: int,
    section_tokens: int,
    paper_context: str = "",
    outline: dict | None = None,
) -> str:
    body = engine.generate_one(
        topic_detail_messages(
            topic, scenes, whisper_segments, max_model_len, section_tokens, paper_context
        ),
        max_tokens=section_tokens,
    )
    return finalize_topic_section(topic, body, outline)


def write_topics_batched(
    engine,
    topics: list[dict],
    scenes: list[dict],
    whisper_segments: list[dict],
    max_model_len: int,
    section_tokens: int,
    paper_context: str,
    *,
    topic_batch_size: int,
    outline: dict | None = None,
) -> dict[int, str]:
    """Generate multiple topic sections via vLLM continuous batching."""
    if not topics:
        return {}

    messages_list = [
        topic_detail_messages(
            topic, scenes, whisper_segments, max_model_len, section_tokens, paper_context
        )
        for topic in topics
    ]
    labels = [
        f"topic {topic.get('topic_id', 0)}: {str(topic.get('title', 'Topic'))[:40]}"
        for topic in topics
    ]
    print(
        f"  vLLM batch-write {len(topics)} topics "
        f"(batch_size={topic_batch_size}): {', '.join(labels)} ..."
    )
    bodies = engine.generate_batch_chunked(
        messages_list, topic_batch_size, max_tokens=section_tokens
    )
    return {
        int(topic.get("topic_id", 0)): finalize_topic_section(topic, body, outline)
        for topic, body in zip(topics, bodies)
    }


def refine_topic_messages(
    topic: dict,
    body: str,
    scenes: list[dict],
    whisper_segments: list[dict],
    max_model_len: int,
    refine_tokens: int,
    paper_context: str,
) -> list[dict]:
    lookup = scene_by_id(scenes)
    segs = enrich_segments(
        topic.get("segments", []),
        lookup,
        for_topics_dir=True,
        topic_title=str(topic.get("title", "")),
    )
    budget = prompt_char_budget(max_model_len, refine_tokens)
    segs_json = compact_segments_json(segs, max_chars=min(6000, budget // 4))
    has_papers = bool(paper_context.strip())
    context = fit_prompt(
        context_for_topic(topic, scenes, lookup, whisper_segments, paper_context),
        budget // 2,
    )
    draft_cap = min(12000, budget // 2)
    return text_messages(
        REFINE_TOPIC_PROMPT.format(
            topic_id=topic.get("topic_id", 0),
            topic_title=topic.get("title", "Topic"),
            time_range=fmt_range(topic.get("start_sec", 0), topic.get("end_sec", 0)),
            papers_available="yes" if has_papers else "no",
            segments_json=segs_json,
            context=context,
            draft=fit_prompt(body, draft_cap),
        )
    )


def refine_topics_batched(
    engine,
    topics: list[dict],
    drafts: dict[int, str],
    scenes: list[dict],
    whisper_segments: list[dict],
    max_model_len: int,
    refine_tokens: int,
    paper_context: str,
    *,
    passes: int,
    topic_batch_size: int,
    outline: dict | None = None,
) -> dict[int, str]:
    """Run refinement passes; batch independent topics within each pass."""
    current = dict(drafts)
    for pass_num in range(1, passes + 1):
        print(f"  refine pass {pass_num}/{passes} ({len(topics)} topics) ...")
        messages_list = [
            refine_topic_messages(
                topic,
                current[int(topic.get("topic_id", 0))],
                scenes,
                whisper_segments,
                max_model_len,
                refine_tokens,
                paper_context,
            )
            for topic in topics
        ]
        bodies = engine.generate_batch_chunked(
            messages_list, topic_batch_size, max_tokens=refine_tokens
        )
        for topic, body in zip(topics, bodies):
            tid = int(topic.get("topic_id", 0))
            current[tid] = strip_miss_sections(
                normalize_segment_images(body, for_topics_dir=True),
                outline,
            )
    return current


def render_master_timeline(outline: dict) -> str:
    lines = [
        "## Complete Session Timeline",
        "",
        "Full lecture coverage — use this to check you did not miss any period.",
        "",
        "| Time | Topic | Sub-segments | Summary |",
        "|------|-------|--------------|---------|",
    ]
    for topic in outline.get("topics", []):
        t_range = fmt_range(topic.get("start_sec", 0), topic.get("end_sec", 0))
        subs = "; ".join(
            f"{fmt_range(s.get('start_sec', 0), s.get('end_sec', 0))} {s.get('title', '')}"
            for s in topic.get("segments", [])[:6]
        )
        lines.append(
            f"| {t_range} | **{topic.get('title', '')}** | {subs} | "
            f"{topic.get('summary', '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_topic_index(outline: dict) -> str:
    lines = ["## Topic Index (with time breakdown)", ""]
    for topic in outline.get("topics", []):
        tid = topic.get("topic_id", 0)
        lines.append(
            f"### Topic {tid}: {topic.get('title', '')} "
            f"({fmt_range(topic.get('start_sec', 0), topic.get('end_sec', 0))})"
        )
        lines.append("")
        for seg in topic.get("segments", []):
            title = str(seg.get("title", "")).strip()
            summary = str(seg.get("summary", "")).strip()
            if summary and summary != title and not summary.startswith(title):
                detail = f"{title}: {summary}"
            else:
                detail = title
            lines.append(
                f"- **{fmt_range(seg.get('start_sec', 0), seg.get('end_sec', 0))}** — "
                f"{detail}"
            )
        lines.append("")
    return "\n".join(lines)


def transcript_highlights(scenes: list[dict], max_windows: int = 20, chars: int = 400) -> str:
    if not scenes:
        return "(none)"
    end_time = max(s["end"] for s in scenes)
    window = max(end_time / max_windows, 240)
    lines: list[str] = []
    t = 0.0
    while t < end_time and len(lines) < max_windows:
        window_end = t + window
        texts = [
            s["transcript"]
            for s in scenes
            if s["transcript"] and s["start"] >= t and s["start"] < window_end
        ]
        if texts:
            lines.append(f"{fmt_range(t, window_end)}: {truncate(' '.join(texts), chars)}")
        t = window_end
    return "\n".join(lines)


def render_references(outline: dict) -> str:
    refs = outline.get("references") or []
    if not refs:
        return ""
    lines = ["## References & Additional Materials", ""]
    for ref in refs:
        ts = ref.get("timestamp", "")
        url = ref.get("url", "")
        label = ref.get("label", "Resource")
        note = ref.get("note", "")
        if url:
            lines.append(f"- [{label}]({url}) — {note} _(lecture ~{ts})_")
        else:
            lines.append(f"- **{label}** — {note} _(lecture ~{ts})_")
    lines.append("")
    return "\n".join(lines)


def render_mindmap_prompt(title: str) -> str:
    return f"""Add ONLY a `## Concept Dependency Map` section with a ```mermaid flowchart TD``` diagram for: {title}

Show the CONCEPTUAL DEPENDENCY structure (not a generic topic tree):
- Start with Problem / Motivation at the top
- Prerequisites feed into CoreConcept
- CoreConcept → Mechanism → Architecture/Method → Training/Inference → Evaluation → Limitations → Extensions
- Use labeled arrows showing WHY one concept requires another (e.g. "requires understanding of X")
- Include 8–15 nodes; group related ideas

Return ONLY the section markdown starting with `## Concept Dependency Map`."""


def refine_topic_section(
    engine,
    body: str,
    topic: dict,
    scenes: list[dict],
    whisper_segments: list[dict],
    passes: int,
    max_model_len: int,
    refine_tokens: int,
    paper_context: str = "",
    outline: dict | None = None,
    topic_batch_size: int = 1,
) -> str:
    """Refine one topic (uses batched helper for consistency)."""
    if passes <= 0:
        return body

    tid = int(topic.get("topic_id", 0))
    refined = refine_topics_batched(
        engine,
        [topic],
        {tid: body},
        scenes,
        whisper_segments,
        max_model_len,
        refine_tokens,
        paper_context,
        passes=passes,
        topic_batch_size=max(1, topic_batch_size),
        outline=outline,
    )
    return refined[tid]


def compact_whisper_transcript(
    whisper_segments: list[dict], max_chars: int = 18000
) -> str:
    lines: list[str] = []
    for seg in whisper_segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{fmt_time(seg.get('start', 0))}] {text}")
    return fit_prompt("\n".join(lines), max_chars)


def outline_summary(outline: dict) -> str:
    lines: list[str] = []
    for topic in outline.get("topics", []):
        tid = topic.get("topic_id", 0)
        lines.append(
            f"Topic {tid}: {topic.get('title', '')} "
            f"({fmt_range(topic.get('start_sec', 0), topic.get('end_sec', 0))})"
        )
    return "\n".join(lines) or "(no topics)"


def report_front_matter_for_audit(md: str, max_chars: int = 6000) -> str:
    marker = "## Detailed Breakdown"
    front = md.split(marker, 1)[0].strip() if marker in md else md
    return fit_prompt(front, max_chars)


def report_body_for_audit(md: str, max_chars: int = 10000) -> str:
    marker = "## Detailed Breakdown"
    body = md.split(marker, 1)[-1] if marker in md else md
    return fit_prompt(body.strip(), max_chars)


def run_coverage_audit(
    engine,
    outline: dict,
    whisper_segments: list[dict],
    report_md: str,
    duration_sec: float,
    max_model_len: int,
    verify_tokens: int,
) -> dict:
    budget = prompt_char_budget(max_model_len, verify_tokens)
    prompt = COVERAGE_AUDIT_PROMPT.format(
        duration=fmt_time(duration_sec),
        outline_summary=outline_summary(outline),
        transcript=compact_whisper_transcript(whisper_segments, budget // 3),
        front_matter=report_front_matter_for_audit(report_md, budget // 4),
        report_body=report_body_for_audit(report_md, budget // 2),
    )
    try:
        return parse_json_block(engine.generate_one(text_messages(prompt)))
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
        print(f"    warning: coverage audit parse failed ({exc})")
        return {
            "passes": True,
            "coverage_score": 100,
            "issues": [],
            "summary": f"Audit skipped due to parse error: {exc}",
        }


def fix_topic_from_audit(
    engine,
    body: str,
    topic: dict,
    issues: list[dict],
    scenes: list[dict],
    whisper_segments: list[dict],
    max_model_len: int,
    verify_tokens: int,
    lookup: dict[int, dict],
) -> str:
    segs = enrich_segments(
        topic.get("segments", []),
        lookup,
        for_topics_dir=True,
        topic_title=str(topic.get("title", "")),
    )
    budget = prompt_char_budget(max_model_len, verify_tokens)
    segs_json = compact_segments_json(segs, max_chars=min(6000, budget // 4))
    transcript = fit_prompt(
        transcript_for_range(
            whisper_segments,
            float(topic.get("start_sec", 0)),
            float(topic.get("end_sec", 0)),
            max_chars=budget // 3,
        ),
        budget // 3,
    )
    topic_id = topic.get("topic_id", 0)
    fixed = engine.generate_one(
        text_messages(
            COVERAGE_FIX_PROMPT.format(
                topic_id=topic_id,
                topic_title=topic.get("title", "Topic"),
                time_range=fmt_range(topic.get("start_sec", 0), topic.get("end_sec", 0)),
                issues_json=json.dumps(issues, indent=2, ensure_ascii=False),
                transcript=transcript,
                segments_json=segs_json,
                draft=fit_prompt(body, min(8000, budget // 2)),
            )
        )
    )
    return normalize_segment_images(strip_miss_sections(fixed), for_topics_dir=True)


def topics_with_metadata_issues(issues: list[dict]) -> set[int]:
    tids: set[int] = set()
    for issue in issues:
        if issue.get("type") in ("incomplete_metadata", "incomplete", "low_quality_summary"):
            tid = int(issue.get("topic_id") or 0)
            if tid:
                tids.add(tid)
    return tids


def regenerate_topics(
    engine,
    topic_ids: set[int],
    topics: list[dict],
    bodies: list[str],
    topics_dir: Path,
    scenes: list[dict],
    whisper_segments: list[dict],
    max_model_len: int,
    section_tokens: int,
    refine_passes: int,
    refine_tokens: int,
    paper_context: str = "",
    *,
    topic_batch_size: int = 2,
    outline: dict | None = None,
) -> list[str]:
    updated = list(bodies)
    to_regen = [t for t in topics if int(t.get("topic_id") or 0) in topic_ids]
    if not to_regen:
        return updated

    print(f"    regenerate {len(to_regen)} topic(s) (batch_size={topic_batch_size}) ...")
    drafts = write_topics_batched(
        engine,
        to_regen,
        scenes,
        whisper_segments,
        max_model_len,
        section_tokens,
        paper_context,
        topic_batch_size=topic_batch_size,
        outline=outline,
    )
    if refine_passes > 0:
        drafts = refine_topics_batched(
            engine,
            to_regen,
            drafts,
            scenes,
            whisper_segments,
            max_model_len,
            refine_tokens,
            paper_context,
            passes=refine_passes,
            topic_batch_size=topic_batch_size,
            outline=outline,
        )

    for topic in to_regen:
        tid = int(topic.get("topic_id") or 0)
        idx = next((i for i, t in enumerate(topics) if t.get("topic_id") == tid), None)
        if idx is None:
            continue
        body = drafts[tid].strip() + "\n"
        topic_markdown_path(topics_dir, tid).write_text(body, encoding="utf-8")
        updated[idx] = body
    return updated


def verify_and_improve_report(
    engine,
    run_dir: Path,
    outline: dict,
    scenes: list[dict],
    whisper_segments: list[dict],
    topics_dir: Path,
    topic_bodies: list[str],
    title: str,
    subtitle: str,
    references_md: str,
    mindmap_md: str,
    max_model_len: int,
    verify_tokens: int,
    verify_passes: int,
    section_tokens: int,
    refine_passes: int,
    refine_tokens: int,
    outline_path: Path,
    paper_context: str = "",
    prerequisites_md: str = "",
    roadmap_md: str = "",
    *,
    topic_batch_size: int = 2,
) -> tuple[str, list[str], dict]:
    if verify_passes <= 0 or not whisper_segments:
        draft = merge_report(
            title,
            subtitle,
            outline,
            topic_bodies,
            references_md,
            mindmap_md,
            prerequisites_md,
            roadmap_md,
        )
        return normalize_report_layout(draft, outline), topic_bodies, outline

    outline = sanitize_outline(outline, scenes)
    lookup = scene_by_id(scenes)
    topics = outline.get("topics", [])
    topic_by_id = {t.get("topic_id", 0): t for t in topics}
    bodies = list(topic_bodies)
    duration_sec = max(s["end"] for s in scenes) if scenes else 0.0

    draft = normalize_report_layout(
        merge_report(
            title,
            subtitle,
            outline,
            bodies,
            references_md,
            mindmap_md,
            prerequisites_md,
            roadmap_md,
        ),
        outline,
    )

    for pass_num in range(1, verify_passes + 1):
        print(f"Phase 4: coverage verify pass {pass_num}/{verify_passes} ...")

        meta_issues = outline_quality_issues(outline)
        if meta_issues:
            print(f"  {len(meta_issues)} outline metadata issue(s) — polishing ...")
            outline = polish_outline(
                engine,
                outline,
                scenes,
                whisper_segments,
                max_model_len,
                verify_tokens,
            )
            save_json(outline_path, outline)
            title = outline.get("title", title)
            subtitle = outline.get("subtitle", subtitle)
            topics = outline.get("topics", [])
            topic_by_id = {t.get("topic_id", 0): t for t in topics}
            regen_ids = topics_with_metadata_issues(meta_issues)
            if regen_ids:
                engine.config.max_new_tokens = section_tokens
                bodies = regenerate_topics(
                    engine,
                    regen_ids,
                    topics,
                    bodies,
                    topics_dir,
                    scenes,
                    whisper_segments,
                    max_model_len,
                    section_tokens,
                    refine_passes,
                    refine_tokens,
                    paper_context,
                    topic_batch_size=topic_batch_size,
                    outline=outline,
                )
                outline = sanitize_outline(outline, scenes)
                draft = normalize_report_layout(
                    merge_report(
                        title,
                        subtitle,
                        outline,
                        bodies,
                        references_md,
                        mindmap_md,
                        prerequisites_md,
                        roadmap_md,
                    ),
                    outline,
                )

        engine.config.max_new_tokens = verify_tokens
        audit = run_coverage_audit(
            engine,
            outline,
            whisper_segments,
            draft,
            duration_sec,
            max_model_len,
            verify_tokens,
        )
        save_json(run_dir / "report" / f"coverage_audit_pass_{pass_num}.json", audit)

        score = audit.get("coverage_score", "?")
        meta_ok = not outline_quality_issues(outline)
        if audit.get("passes") and meta_ok:
            print(f"  coverage OK (score {score})")
            break

        issues = audit.get("issues") or []
        if not issues and meta_ok:
            print(f"  audit failed (score {score}) but no issues listed — stopping")
            break

        print(f"  score {score}, fixing {len(issues)} issue(s) ...")

        meta_audit_ids = topics_with_metadata_issues(issues)
        if meta_audit_ids or not meta_ok:
            outline = polish_outline(
                engine,
                outline,
                scenes,
                whisper_segments,
                max_model_len,
                verify_tokens,
            )
            save_json(outline_path, outline)
            title = outline.get("title", title)
            subtitle = outline.get("subtitle", subtitle)
            topics = outline.get("topics", [])
            topic_by_id = {t.get("topic_id", 0): t for t in topics}
            regen_ids = meta_audit_ids | topics_with_metadata_issues(
                outline_quality_issues(outline)
            )
            if regen_ids:
                engine.config.max_new_tokens = section_tokens
                bodies = regenerate_topics(
                    engine,
                    regen_ids,
                    topics,
                    bodies,
                    topics_dir,
                    scenes,
                    whisper_segments,
                    max_model_len,
                    section_tokens,
                    refine_passes,
                    refine_tokens,
                    paper_context,
                    topic_batch_size=topic_batch_size,
                    outline=outline,
                )

        by_topic: dict[int, list[dict]] = {}
        for issue in issues:
            if issue.get("type") == "incomplete_metadata":
                continue
            tid = int(issue.get("topic_id") or 0)
            if tid in topic_by_id:
                by_topic.setdefault(tid, []).append(issue)

        for tid, topic_issues in sorted(by_topic.items()):
            topic = topic_by_id[tid]
            idx = next(
                (i for i, t in enumerate(topics) if t.get("topic_id") == tid),
                None,
            )
            if idx is None:
                continue
            print(f"    fix topic {tid}: {len(topic_issues)} issue(s)")
            fixed = fix_topic_from_audit(
                engine,
                bodies[idx],
                topic,
                topic_issues,
                scenes,
                whisper_segments,
                max_model_len,
                verify_tokens,
                lookup,
            )
            fixed = strip_miss_sections(fixed, outline).strip() + "\n"
            topic_markdown_path(topics_dir, tid).write_text(fixed, encoding="utf-8")
            bodies[idx] = fixed

        draft = normalize_report_layout(
            merge_report(
                title,
                subtitle,
                outline,
                bodies,
                references_md,
                mindmap_md,
                prerequisites_md,
                roadmap_md,
            ),
            outline,
        )

        if pass_num == verify_passes:
            print(f"  reached max verify passes ({verify_passes})")

    return draft, bodies, outline


def merge_report(
    title: str,
    subtitle: str,
    outline: dict,
    topic_bodies: list[str],
    references_md: str,
    mindmap_md: str,
    prerequisites_md: str = "",
    roadmap_md: str = "",
) -> str:
    """Assemble final report from per-topic files + front matter + tail sections."""
    parts: list[str] = [
        f"# {title}",
        f"_{subtitle}_" if subtitle else "",
        "",
    ]
    if roadmap_md.strip():
        parts.extend([roadmap_md.strip(), "", "---", ""])
    if prerequisites_md.strip():
        parts.extend([prerequisites_md.strip(), "", "---", ""])
    parts.extend([
        render_master_timeline(outline),
        render_topic_index(outline),
        "---",
        "",
    ])

    for body in topic_bodies:
        report_body = body.strip().replace("../attachments/", "attachments/")
        parts.append(report_body)
        parts.append("")
        parts.append("---")
        parts.append("")

    if references_md:
        parts.append(references_md)
    if mindmap_md:
        parts.append(mindmap_md)

    text = "\n".join(p for p in parts if p is not None).strip()
    text = re.sub(r"\n---\n\n---\n", "\n---\n", text)
    return text + "\n"


def count_topic_sections(md: str) -> int:
    return len(re.findall(r"(?m)^## Topic \d+:", md))


def load_topic_bodies(
    topics_dir: Path,
    outline: dict,
    *,
    rewrite: bool = False,
) -> list[str]:
    bodies: list[str] = []
    for topic in outline.get("topics", []):
        tid = topic.get("topic_id", 0)
        path = topic_markdown_path(topics_dir, tid)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name} — run step 09 without --merge-only to generate it"
            )
        raw = path.read_text(encoding="utf-8")
        body = strip_miss_sections(raw, outline)
        if rewrite and body != raw:
            path.write_text(body.strip() + "\n", encoding="utf-8")
        bodies.append(body)
    return bodies


def merge_only_report(run_dir: Path) -> Path:
    """Rebuild report.md from existing topic_*.md files (no LLM)."""
    report_dir = run_dir / "report"
    topics_dir = report_dir / "topics"
    outline_path = report_dir / "topic_outline.json"
    out_path = report_dir / "report.md"
    mindmap_path = report_dir / "mindmap.md"
    scenes_path = run_dir / "merged" / "scenes_merged.json"

    if not outline_path.exists():
        raise FileNotFoundError(f"Outline not found: {outline_path}")

    scenes = load_json(scenes_path) if scenes_path.exists() else []
    outline = sanitize_outline(load_json(outline_path), scenes)
    save_json(outline_path, outline)
    block_warnings = sanitize_report_blocks(report_dir, outline, rewrite=True)
    topic_bodies = load_topic_bodies(topics_dir, outline, rewrite=False)
    mindmap_md = (
        mindmap_path.read_text(encoding="utf-8") if mindmap_path.exists() else ""
    )

    prereq_path = report_dir / "prerequisites.md"
    roadmap_path = report_dir / "learning_roadmap.md"
    prerequisites_md = (
        prereq_path.read_text(encoding="utf-8") if prereq_path.exists() else ""
    )
    roadmap_md = (
        roadmap_path.read_text(encoding="utf-8") if roadmap_path.exists() else ""
    )
    draft = merge_report(
        outline.get("title", "Lecture Report"),
        outline.get("subtitle", ""),
        outline,
        topic_bodies,
        render_references(outline),
        mindmap_md,
        prerequisites_md,
        roadmap_md,
    )
    draft = normalize_report_layout(draft, outline)
    draft_warnings = audit_report_block(draft, "report.md")
    out_path.write_text(draft.strip() + "\n", encoding="utf-8")

    for warning in block_warnings + draft_warnings:
        print(f"  audit: {warning}")

    n_topics = count_topic_sections(draft)
    expected = len(outline.get("topics", []))
    print(f"Merged {n_topics}/{expected} topics → {out_path}")
    if n_topics < expected:
        print(
            f"WARNING: {expected - n_topics} topic(s) missing — "
            "run without --merge-only to generate them"
        )
    return out_path


def collect_outline_scene_ids(outline: dict) -> set[int]:
    ids: set[int] = set()
    for topic in outline.get("topics", []):
        for seg in topic.get("segments", []):
            for sid in seg.get("scene_ids") or []:
                ids.add(int(sid))
    return ids


def copy_report_images(
    scene_ids: set[int], keyframes_dir: Path, attachments_dir: Path
) -> int:
    attachments_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for sid in sorted(scene_ids):
        src = keyframes_dir / f"scene_{sid:03d}.png"
        dst = attachments_dir / f"scene_{sid:03d}.png"
        if src.exists():
            shutil.copy2(src, dst)
            n += 1
    return n


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def generate_report(
    run_dir: Path,
    model: str,
    max_model_len: int,
    max_main_scenes: int,
    window_minutes: int,
    section_tokens: int,
    refine_tokens: int,
    refine_passes: int,
    verify_passes: int,
    verify_tokens: int,
    force_outline: bool,
    force_topics: bool,
    gpu_memory_utilization: float = 0.82,
    engine: QwenVllmEngine | None = None,
    outline_batch_size: int = 4,
    topic_batch_size: int = 2,
) -> Path:
    scenes = load_json(run_dir / "merged" / "scenes_merged.json")
    whisper_path = run_dir / "transcript" / "whisper_segments.json"
    whisper_segments = load_json(whisper_path) if whisper_path.exists() else []
    paper_context = load_paper_context(run_dir)
    report_dir = run_dir / "report"
    topics_dir = report_dir / "topics"
    attachments_dir = report_dir / "attachments"
    outline_path = report_dir / "topic_outline.json"
    out_path = report_dir / "report.md"
    report_dir.mkdir(parents=True, exist_ok=True)
    topics_dir.mkdir(parents=True, exist_ok=True)

    config = VllmConfig(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    def run_report(active_engine: QwenVllmEngine) -> tuple[str, dict]:
        if outline_path.exists() and not force_outline:
            print(f"Reusing outline: {outline_path}")
            outline = load_json(outline_path)
        else:
            print(f"Phase 1/3: topic + time outline ({len(scenes)} scenes) ...")
            active_engine.config.max_new_tokens = 4096
            outline = build_topic_outline(
                active_engine,
                scenes,
                window_minutes,
                whisper_segments,
                max_model_len,
                4096,
                outline_batch_size,
            )
            save_json(outline_path, outline)
            print(f"  → {len(outline.get('topics', []))} major topics")

        outline = sanitize_outline(outline, scenes)
        save_json(outline_path, outline)

        main_scenes = pick_main_scenes_from_outline(outline, scenes, max_main_scenes)
        save_json(report_dir / "main_scenes.json", {"main_scenes": main_scenes, **outline})

        topics = outline.get("topics", [])
        title = outline.get("title", "Lecture Report")
        subtitle = outline.get("subtitle", "")

        prereq_path = report_dir / "prerequisites.md"
        roadmap_path = report_dir / "learning_roadmap.md"
        regen_front = force_topics or force_outline or not prereq_path.exists()
        if regen_front:
            print("Phase 1.5: learning roadmap + prerequisites (vLLM batch) ...")
            active_engine.config.max_new_tokens = 4096
            roadmap_md, prerequisites_md = write_front_matter_sections(
                active_engine,
                outline,
                whisper_segments,
                paper_context,
                max_model_len,
            )
            roadmap_path.write_text(roadmap_md, encoding="utf-8")
            prereq_path.write_text(prerequisites_md, encoding="utf-8")
            print("  → learning_roadmap.md, prerequisites.md")
        else:
            print("  reuse learning_roadmap.md + prerequisites.md")
            roadmap_md = roadmap_path.read_text(encoding="utf-8")
            prerequisites_md = prereq_path.read_text(encoding="utf-8")

        print(
            f"Phase 2/3: first-principles topic sections — {len(topics)} topics "
            f"(batch_size={topic_batch_size}) → {topics_dir}/"
        )

        active_engine.config.max_new_tokens = section_tokens
        bodies_by_tid: dict[int, str] = {}
        to_write: list[dict] = []
        for topic in topics:
            tid = int(topic.get("topic_id", 0))
            topic_path = topic_markdown_path(topics_dir, tid)
            label = topic.get("title", "Topic")[:50]

            if topic_path.exists() and not force_topics:
                print(f"  reuse topic {tid}: {label} ...")
                bodies_by_tid[tid] = strip_miss_sections(
                    topic_path.read_text(encoding="utf-8"), outline
                )
            else:
                to_write.append(topic)

        if to_write:
            drafts = write_topics_batched(
                active_engine,
                to_write,
                scenes,
                whisper_segments,
                max_model_len,
                section_tokens,
                paper_context,
                topic_batch_size=topic_batch_size,
                outline=outline,
            )
            if refine_passes > 0:
                active_engine.config.max_new_tokens = refine_tokens
                drafts = refine_topics_batched(
                    active_engine,
                    to_write,
                    drafts,
                    scenes,
                    whisper_segments,
                    max_model_len,
                    refine_tokens,
                    paper_context,
                    passes=refine_passes,
                    topic_batch_size=topic_batch_size,
                    outline=outline,
                )
            for topic in to_write:
                tid = int(topic.get("topic_id", 0))
                body = drafts[tid].strip() + "\n"
                topic_path = topic_markdown_path(topics_dir, tid)
                topic_path.write_text(body, encoding="utf-8")
                bodies_by_tid[tid] = body
                print(f"    → {topic_path.name}")

        topic_bodies = [bodies_by_tid[int(t.get("topic_id", 0))] for t in topics]

        print(f"Phase 3/3: merge {len(topic_bodies)} topics into report.md ...")
        active_engine.config.max_new_tokens = 2048
        mindmap_md = active_engine.generate_one(
            text_messages(render_mindmap_prompt(title))
        )
        (report_dir / "mindmap.md").write_text(mindmap_md.strip() + "\n", encoding="utf-8")
        references_md = render_references(outline)

        draft, topic_bodies, outline = verify_and_improve_report(
            active_engine,
            run_dir,
            outline,
            scenes,
            whisper_segments,
            topics_dir,
            topic_bodies,
            title,
            subtitle,
            references_md,
            mindmap_md,
            max_model_len,
            verify_tokens,
            verify_passes,
            section_tokens,
            refine_passes,
            refine_tokens,
            outline_path,
            paper_context,
            prerequisites_md,
            roadmap_md,
            topic_batch_size=topic_batch_size,
        )
        save_json(outline_path, outline)

        print("Final pass: sanitize + audit all report blocks ...")
        block_warnings = sanitize_report_blocks(report_dir, outline, rewrite=True)
        outline = sanitize_outline(outline, scenes)
        save_json(outline_path, outline)
        draft = normalize_report_layout(
            merge_report(
                title,
                subtitle,
                outline,
                load_topic_bodies(topics_dir, outline, rewrite=False),
                render_references(outline),
                (report_dir / "mindmap.md").read_text(encoding="utf-8")
                if (report_dir / "mindmap.md").exists()
                else "",
                prereq_path.read_text(encoding="utf-8")
                if prereq_path.exists()
                else "",
                roadmap_path.read_text(encoding="utf-8")
                if roadmap_path.exists()
                else "",
            ),
            outline,
        )
        draft_warnings = audit_report_block(draft, "report.md")
        for warning in block_warnings + draft_warnings:
            print(f"  audit: {warning}")
        if not block_warnings and not draft_warnings:
            print("  audit: all blocks clean")

        return draft, outline

    if engine is not None:
        draft, outline = run_report(engine)
    else:
        with qwen_vllm_session(config) as active_engine:
            draft, outline = run_report(active_engine)

    out_path.write_text(draft.strip() + "\n", encoding="utf-8")
    n_topics = count_topic_sections(draft)
    expected = len(outline.get("topics", []))
    if n_topics < expected:
        print(
            f"WARNING: merged report has {n_topics}/{expected} topics — "
            "check report/topics/ for missing topic_XX.md files"
        )
    else:
        print(f"Report complete: {n_topics} topics merged")
    n = copy_report_images(
        collect_outline_scene_ids(outline),
        run_dir / "scenes" / "keyframes",
        attachments_dir,
    )
    print(f"Copied {n} images → {attachments_dir}")
    print(f"Report → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed report with topic/time breakdown")
    add_run_dir_arg(parser)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="vLLM max context (8192 fits T4; use 16384 on 24GB+ GPUs)",
    )
    parser.add_argument("--max-main-scenes", type=int, default=25)
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=8,
        help="Minutes per outline window (smaller = finer time breakdown)",
    )
    parser.add_argument("--section-tokens", type=int, default=6000)
    parser.add_argument("--refine-tokens", type=int, default=4096)
    parser.add_argument(
        "--refine-passes",
        type=int,
        default=2,
        help="Per-topic refinement iterations against slide/transcript context",
    )
    parser.add_argument(
        "--verify-passes",
        type=int,
        default=2,
        help="Post-merge coverage audits vs Whisper transcript (max iterations)",
    )
    parser.add_argument(
        "--verify-tokens",
        type=int,
        default=4096,
        help="Max tokens for coverage audit/fix passes",
    )
    parser.add_argument("--force-outline", action="store_true")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Rebuild report.md from existing report/topics/*.md (no LLM)",
    )
    parser.add_argument(
        "--force-topics",
        action="store_true",
        help="Regenerate all per-topic markdown files (default: reuse existing)",
    )
    parser.add_argument(
        "--gpu-mem",
        type=float,
        default=0.82,
        help="vLLM gpu_memory_utilization (lower if GPU is partially in use)",
    )
    parser.add_argument(
        "--outline-batch-size",
        type=int,
        default=4,
        help="Outline windows per vLLM batch (parallel continuous batching)",
    )
    parser.add_argument(
        "--topic-batch-size",
        type=int,
        default=2,
        help="Topic sections per vLLM batch during write/refine (raise on large GPUs)",
    )
    args = parser.parse_args()

    if args.merge_only:
        merge_only_report(args.run_dir)
        return

    generate_report(
        args.run_dir,
        args.model or DEFAULT_MODEL,
        args.max_model_len,
        args.max_main_scenes,
        args.window_minutes,
        args.section_tokens,
        args.refine_tokens,
        args.refine_passes,
        args.verify_passes,
        args.verify_tokens,
        args.force_outline,
        args.force_topics,
        args.gpu_mem,
        args.outline_batch_size,
        args.topic_batch_size,
    )


if __name__ == "__main__":
    main()
