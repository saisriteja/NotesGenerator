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
from qwen_vllm import DEFAULT_MODEL, VllmConfig, qwen_vllm_session, text_messages

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
    text = (text or "").strip()
    if len(text) <= budget:
        return text
    return text[: max(0, budget - 40)] + "\n\n... [truncated for context limit]"


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
        item["summary"] = truncate(str(item.get("summary", "")), 400)
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
- Be specific so a student knows exactly what they might have missed.

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
      "summary": "paragraph overview",
      "segments": [
        {{
          "start_sec": 0.0,
          "end_sec": 120.0,
          "title": "sub-topic",
          "summary": "...",
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

Segment batches:
{segments_json}
"""


def outline_window(
    engine, scenes: list[dict], win_start: float, win_end: float
) -> list[dict]:
    window_scenes = scenes_in_range(scenes, win_start, win_end)
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
    prompt = WINDOW_OUTLINE_PROMPT.format(
        win_start=win_start,
        win_end=win_end,
        catalog=compact_catalog(window_scenes),
    )
    try:
        parsed = parse_json_block(engine.generate_one(text_messages(prompt)))
        segs = parsed.get("segments", parsed) if isinstance(parsed, dict) else parsed
        if isinstance(segs, list) and segs:
            return segs
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as exc:
        print(f"    warning: window outline failed ({exc})")
    # Fallback: one segment per ~3 scenes
    fallback = []
    batch = 3
    for i in range(0, len(window_scenes), batch):
        chunk = window_scenes[i : i + batch]
        fallback.append(
            {
                "start_sec": chunk[0]["start"],
                "end_sec": chunk[-1]["end"],
                "title": truncate(chunk[0].get("vlm_description", ""), 50) or "Section",
                "summary": truncate(" ".join(s.get("transcript", "") for s in chunk), 200),
                "key_points": [],
                "scene_ids": [s["scene_id"] for s in chunk],
            }
        )
    return fallback


def build_topic_outline(
    engine,
    scenes: list[dict],
    window_minutes: int,
) -> dict:
    if not scenes:
        return {"title": "Lecture", "subtitle": "", "topics": [], "references": []}

    end_time = max(s["end"] for s in scenes)
    window_sec = window_minutes * 60
    all_segments: list[dict] = []

    t = 0.0
    win_num = 0
    while t < end_time:
        win_end = min(t + window_sec, end_time)
        win_num += 1
        print(f"  outline window {win_num}: {fmt_range(t, win_end)} ...")
        segs = outline_window(engine, scenes, t, win_end)
        all_segments.extend(segs)
        t = win_end

    print("  merging into major topics ...")
    merge_prompt = MERGE_TOPICS_PROMPT.format(
        segments_json=json.dumps(all_segments[:80], indent=2, ensure_ascii=False)[:12000]
    )
    try:
        outline = parse_json_block(engine.generate_one(text_messages(merge_prompt)))
        if isinstance(outline, dict) and outline.get("topics"):
            return outline
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"  warning: topic merge failed ({exc}), using heuristic grouping")

    return heuristic_topic_outline(scenes, window_sec)


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
                        "summary": truncate(
                            " ".join(s.get("transcript", "") for s in sub_scenes), 250
                        ),
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
                "summary": truncate(
                    " ".join(s.get("transcript", "") for s in win_scenes[:8]), 300
                ),
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

TOPIC_DETAIL_PROMPT = """Write a GRANULAR, detailed Markdown section for ONE lecture topic.

Topic {topic_id}: **{topic_title}** ({time_range})

## Layout rules (STRICT — follow exactly)
Each time segment MUST use this block order:
1. `### {{segment_time}} — {{segment_title}}`
2. **Figures first** — copy every string from the segment `figures` list exactly (one per line), immediately after the heading
3. **Then text** — explanation sections below the images

## Math rules (STRICT)
- Write ALL mathematics in TeX/LaTeX only — Obsidian-compatible
- Inline math: `$...$` (e.g. `$L \\propto C^{{-0.444}}$`)
- Display equations on their own line: `$$...$$`
- Use `\\frac`, `\\cdot`, `\\times`, `\\log`, subscripts `C_{{train}}`, superscripts `10^{{-4}}`
- NEVER use Unicode math (⁰, ⁴, ×, ∝, etc.) or plain-text formulas like `Compute^0.444`

## Text sections (after figures)
For EACH segment include, in this order:
- **What is covered:** detailed explanation (teach the material)
- **Key points:** bullet list
- **Equations & definitions:** every formula in TeX (`$...$` or `$$...$$`); define symbols
- **What you might miss if you skip this part:** 1-2 bullets
- `> 🔁 Revisit lecture:` {{segment_time}}

Do NOT put figures after the text. Do NOT skip any segment.

Segments JSON (each segment has a `figures` list — copy those lines exactly to the top of that segment):
{segments_json}

Slide/transcript context:
{context}
"""

REFINE_TOPIC_PROMPT = """Refine ONE topic section from a lecture report. Return ONLY this topic's markdown.

Topic {topic_id}: **{topic_title}** ({time_range})

Tasks:
1. Cover EVERY segment in the outline below — do not drop or merge segments
2. **Layout:** copy `figures` from the segment JSON immediately after each `###` heading, BEFORE text
3. **Math:** TeX only — `$...$` inline, `$$...$$` display; no Unicode math
4. Add missing detail, equations, key points; keep `> 🔁 Revisit lecture:` timestamps

Required segments (all must appear):
{segments_json}

Transcript/slide context for this topic:
{context}

Current topic draft:
{draft}

Return the complete topic section starting with `## Topic {topic_id}:`. No preamble.
"""


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
    segs: list[dict], lookup: dict[int, dict], *, for_topics_dir: bool = False
) -> list[dict]:
    """Add markdown figure links for each segment."""
    enriched = []
    for seg in segs:
        item = dict(seg)
        figures = []
        for sid in seg.get("scene_ids") or []:
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


def normalize_report_layout(md: str) -> str:
    """Normalize image placement across the full report."""
    marker = "## Detailed Breakdown"
    if marker not in md:
        return normalize_segment_images(md)

    prefix, body = md.split(marker, 1)
    return prefix + marker + normalize_segment_images(body, for_topics_dir=False)


def context_for_scenes(
    scenes: list[dict],
    scene_ids: list[int],
    lookup: dict,
    *,
    vlm_limit: int = 400,
    audio_limit: int = 600,
) -> str:
    lines = []
    for sid in scene_ids[:8]:
        sc = lookup.get(int(sid))
        if not sc:
            continue
        lines.append(
            f"scene_{sid:03d}|{fmt_range(sc['start'], sc['end'])}|"
            f"{truncate(sc.get('vlm_description', ''), vlm_limit)}|"
            f"{truncate(sc.get('transcript', ''), audio_limit)}"
        )
    return "\n".join(lines) or "(no slides)"


def context_for_topic(
    topic: dict,
    scenes: list[dict],
    lookup: dict,
    whisper_segments: list[dict],
) -> str:
    start = float(topic.get("start_sec", 0))
    end = float(topic.get("end_sec", 0))
    scene_ids: list[int] = []
    for seg in topic.get("segments", []):
        scene_ids.extend(int(s) for s in (seg.get("scene_ids") or []))

    slide_ctx = context_for_scenes(scenes, scene_ids, lookup)
    transcript_ctx = transcript_for_range(whisper_segments, start, end)
    return (
        f"Slide context (truncated per scene):\n{slide_ctx}\n\n"
        f"Transcript by time window:\n{transcript_ctx}"
    )


def write_topic_section(
    engine,
    topic: dict,
    scenes: list[dict],
    whisper_segments: list[dict],
    max_model_len: int,
    section_tokens: int,
) -> str:
    lookup = scene_by_id(scenes)
    segs = enrich_segments(topic.get("segments", []), lookup, for_topics_dir=True)
    budget = prompt_char_budget(max_model_len, section_tokens)
    segs_json = compact_segments_json(segs, max_chars=min(8000, budget // 3))
    context = fit_prompt(
        context_for_topic(topic, scenes, lookup, whisper_segments),
        budget // 3,
    )

    body = engine.generate_one(
        text_messages(
            TOPIC_DETAIL_PROMPT.format(
                topic_id=topic.get("topic_id", 0),
                topic_title=topic.get("title", "Topic"),
                time_range=fmt_range(topic.get("start_sec", 0), topic.get("end_sec", 0)),
                segments_json=segs_json,
                context=context,
            )
        )
    )
    header = (
        f"## Topic {topic.get('topic_id', 0)}: {topic.get('title', 'Topic')}\n"
        f"> ⏱ Session: {fmt_range(topic.get('start_sec', 0), topic.get('end_sec', 0))}\n"
    )
    return header + "\n" + normalize_segment_images(body, for_topics_dir=True)


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
            f"| {t_range} | **{topic.get('title', '')}** | {truncate(subs, 120)} | "
            f"{truncate(topic.get('summary', ''), 100)} |"
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
            lines.append(
                f"- **{fmt_range(seg.get('start_sec', 0), seg.get('end_sec', 0))}** — "
                f"{seg.get('title', '')}: {truncate(seg.get('summary', ''), 150)}"
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
    return f"""Add ONLY a `## Mind Map` section with ```mermaid mindmap``` for: {title}
Root = lecture topic. Include all major topics and 2-3 sub-concepts each.
Return ONLY the Mind Map section markdown."""


def refine_topic_section(
    engine,
    body: str,
    topic: dict,
    scenes: list[dict],
    whisper_segments: list[dict],
    passes: int,
    max_model_len: int,
    refine_tokens: int,
) -> str:
    """Refine one topic at a time — avoids truncating the full report."""
    if passes <= 0:
        return body

    lookup = scene_by_id(scenes)
    segs = enrich_segments(topic.get("segments", []), lookup, for_topics_dir=True)
    budget = prompt_char_budget(max_model_len, refine_tokens)
    segs_json = compact_segments_json(segs, max_chars=min(6000, budget // 4))
    context = fit_prompt(
        context_for_topic(topic, scenes, lookup, whisper_segments),
        budget // 4,
    )
    draft_cap = min(8000, budget // 2)

    topic_id = topic.get("topic_id", 0)
    for i in range(passes):
        print(f"    refine topic {topic_id} pass {i + 1}/{passes} ...")
        body = engine.generate_one(
            text_messages(
                REFINE_TOPIC_PROMPT.format(
                    topic_id=topic_id,
                    topic_title=topic.get("title", "Topic"),
                    time_range=fmt_range(topic.get("start_sec", 0), topic.get("end_sec", 0)),
                    segments_json=segs_json,
                    context=context,
                    draft=fit_prompt(body, draft_cap),
                )
            )
        )
        body = normalize_segment_images(body, for_topics_dir=True)
    return body


def merge_report(
    title: str,
    subtitle: str,
    outline: dict,
    topic_bodies: list[str],
    references_md: str,
    mindmap_md: str,
) -> str:
    """Assemble final report from per-topic files + front matter + tail sections."""
    parts: list[str] = [
        f"# {title}",
        f"_{subtitle}_" if subtitle else "",
        "",
        render_master_timeline(outline),
        render_topic_index(outline),
        "## Detailed Breakdown",
        "",
        "> **Reading order:** each segment shows **slide figures first** "
        "(`![image](attachments/scene_XXX.png)`), then the explanation. All math uses **TeX** "
        "(`$...$` inline, `$$...$$` display).",
        "",
        "### Topic files (Obsidian)",
        "",
    ]
    for topic in outline.get("topics", []):
        tid = topic.get("topic_id", 0)
        ttitle = topic.get("title", "Topic")
        parts.append(f"- [[topics/topic_{tid:02d}|Topic {tid}: {ttitle}]]")
    parts.extend(["", "---", ""])

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


def load_topic_bodies(topics_dir: Path, outline: dict) -> list[str]:
    bodies: list[str] = []
    for topic in outline.get("topics", []):
        tid = topic.get("topic_id", 0)
        path = topic_markdown_path(topics_dir, tid)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name} — run step 09 without --merge-only to generate it"
            )
        bodies.append(path.read_text(encoding="utf-8"))
    return bodies


def merge_only_report(run_dir: Path) -> Path:
    """Rebuild report.md from existing topic_*.md files (no LLM)."""
    report_dir = run_dir / "report"
    topics_dir = report_dir / "topics"
    outline_path = report_dir / "topic_outline.json"
    out_path = report_dir / "report.md"
    mindmap_path = report_dir / "mindmap.md"

    if not outline_path.exists():
        raise FileNotFoundError(f"Outline not found: {outline_path}")

    outline = load_json(outline_path)
    topic_bodies = load_topic_bodies(topics_dir, outline)
    mindmap_md = (
        mindmap_path.read_text(encoding="utf-8") if mindmap_path.exists() else ""
    )

    draft = merge_report(
        outline.get("title", "Lecture Report"),
        outline.get("subtitle", ""),
        outline,
        topic_bodies,
        render_references(outline),
        mindmap_md,
    )
    draft = normalize_report_layout(draft)
    out_path.write_text(draft.strip() + "\n", encoding="utf-8")

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
    force_outline: bool,
    force_topics: bool,
    gpu_memory_utilization: float = 0.82,
) -> Path:
    scenes = load_json(run_dir / "merged" / "scenes_merged.json")
    whisper_path = run_dir / "transcript" / "whisper_segments.json"
    whisper_segments = load_json(whisper_path) if whisper_path.exists() else []
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

    with qwen_vllm_session(config) as engine:
        if outline_path.exists() and not force_outline:
            print(f"Reusing outline: {outline_path}")
            outline = load_json(outline_path)
        else:
            print(f"Phase 1/3: topic + time outline ({len(scenes)} scenes) ...")
            engine.config.max_new_tokens = 4096
            outline = build_topic_outline(engine, scenes, window_minutes)
            save_json(outline_path, outline)
            print(f"  → {len(outline.get('topics', []))} major topics")

        main_scenes = pick_main_scenes_from_outline(outline, scenes, max_main_scenes)
        save_json(report_dir / "main_scenes.json", {"main_scenes": main_scenes, **outline})

        topics = outline.get("topics", [])
        print(f"Phase 2/3: granular sections — {len(topics)} topics → {topics_dir}/")

        engine.config.max_new_tokens = section_tokens
        title = outline.get("title", "Lecture Report")
        subtitle = outline.get("subtitle", "")

        topic_bodies: list[str] = []
        for topic in topics:
            tid = topic.get("topic_id", 0)
            topic_path = topic_markdown_path(topics_dir, tid)
            label = topic.get("title", "Topic")[:50]

            if topic_path.exists() and not force_topics:
                print(f"  reuse topic {tid}: {label} ...")
                body = topic_path.read_text(encoding="utf-8")
            else:
                print(f"  write topic {tid}/{len(topics)}: {label} ...")
                body = write_topic_section(
                    engine, topic, scenes, whisper_segments, max_model_len, section_tokens
                )
                if refine_passes > 0:
                    engine.config.max_new_tokens = refine_tokens
                    body = refine_topic_section(
                        engine,
                        body,
                        topic,
                        scenes,
                        whisper_segments,
                        refine_passes,
                        max_model_len,
                        refine_tokens,
                    )
                body = normalize_segment_images(body, for_topics_dir=True).strip() + "\n"
                topic_path.write_text(body, encoding="utf-8")
                print(f"    → {topic_path.name}")

            topic_bodies.append(body)

        print(f"Phase 3/3: merge {len(topic_bodies)} topics into report.md ...")
        engine.config.max_new_tokens = 2048
        mindmap_md = engine.generate_one(
            text_messages(render_mindmap_prompt(title))
        )
        (report_dir / "mindmap.md").write_text(mindmap_md.strip() + "\n", encoding="utf-8")

        draft = merge_report(
            title,
            subtitle,
            outline,
            topic_bodies,
            render_references(outline),
            mindmap_md,
        )
        draft = normalize_report_layout(draft)

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
    parser.add_argument("--max-model-len", type=int, default=16384)
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
        help="Cross-check/refinement iterations against full context",
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
        args.force_outline,
        args.force_topics,
        args.gpu_mem,
    )


if __name__ == "__main__":
    main()
