"""Map lecture videos to runs/ pipeline names and notes/ delivery folders."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from notesgenerator._paths import data_root, runs_dir

GENERIC_TITLES = {"Lecture Report", "Lecture Title", ""}
LECTURE_NOTES_COURSE = "self_improving_ai"


def lectures_dir() -> Path:
    return data_root() / "lectures"


def notes_dir() -> Path:
    return data_root() / "notes"


def sanitize(name: str, max_len: int = 120) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = re.sub(r"\s+", " ", name).strip().rstrip(".")
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name or "untitled"


def course_display_name(course: str) -> str:
    numbered = re.fullmatch(r"lecture_(\d+)", course)
    if numbered:
        return f"lecture {numbered.group(1)}"
    return course


def notes_course_name(pipeline_course: str) -> str:
    if re.fullmatch(r"lecture_\d+", pipeline_course):
        return LECTURE_NOTES_COURSE
    return course_display_name(pipeline_course)


def output_name_for(video: Path) -> str:
    """Run name under runs/, e.g. cs224w/IMpkHvQ0LA4 or lecture_2."""
    lectures = lectures_dir()
    rel = video.relative_to(lectures)
    dir_name = str(rel.parent)
    base = video.stem
    top = rel.parts[0]

    if dir_name.startswith("lecture ") and base == dir_name:
        return f"lecture_{dir_name.removeprefix('lecture ')}"
    return f"{top}/{base}"


def pipeline_course_for(video: Path) -> str:
    output = output_name_for(video)
    if output.startswith("lecture_"):
        return output
    return output.split("/", 1)[0]


def video_id_for(video: Path) -> str:
    output = output_name_for(video)
    if output.startswith("lecture_"):
        return output
    return video.stem


def build_video_map() -> dict[str, tuple[str, str | None]]:
    mapping: dict[str, tuple[str, str | None]] = {}
    lectures = lectures_dir()
    if not lectures.is_dir():
        return mapping

    for mp4 in lectures.rglob("*.mp4"):
        rel = mp4.relative_to(lectures)
        parts = rel.parts
        course = parts[0]
        series = parts[1] if len(parts) >= 3 else None
        mapping[mp4.stem] = (course, series)
        output = output_name_for(mp4)
        if output.startswith("lecture_"):
            mapping[output] = (course, None)
    return mapping


def lecture_title(course: str, video_id: str) -> str:
    outline = runs_dir() / course / video_id / "report" / "topic_outline.json"
    if outline.exists():
        try:
            data = json.loads(outline.read_text())
            title = (data.get("title") or "").strip()
            if title not in GENERIC_TITLES:
                return title
        except (json.JSONDecodeError, OSError):
            pass
    return video_id


def find_series_name(course: str) -> str | None:
    course_dir = lectures_dir() / course
    if not course_dir.is_dir():
        return None
    subdirs = sorted(d for d in course_dir.iterdir() if d.is_dir())
    if len(subdirs) == 1:
        return subdirs[0].name
    return None


def destination_dir(
    course: str,
    video_id: str,
    video_map: dict[str, tuple[str, str | None]] | None = None,
) -> Path:
    """Final notes folder for a lecture report zip."""
    if video_map is None:
        video_map = build_video_map()

    notes_course = notes_course_name(course)
    base = notes_dir() / notes_course

    if re.fullmatch(r"lecture_\d+", course) or re.fullmatch(r"lecture_\d+", video_id):
        numbered = course if re.fullmatch(r"lecture_\d+", course) else video_id
        return base / course_display_name(numbered)

    _, series_from_map = video_map.get(video_id, (course, None))
    series = series_from_map or find_series_name(course)
    if series:
        base = base / sanitize(series)

    if video_id.startswith("lecture_"):
        return base

    title = lecture_title(course, video_id)
    if title == video_id:
        lecture_folder = video_id
    else:
        lecture_folder = sanitize(f"{title} [{video_id}]")
    return base / lecture_folder


def report_zip_path(output_name: str) -> Path:
    return data_root() / f"{output_name}_report.zip"


def find_report_zip(output_name: str) -> Path | None:
    candidate = report_zip_path(output_name)
    return candidate if candidate.exists() else None


def has_notes_output(video: Path, video_map: dict[str, tuple[str, str | None]] | None = None) -> bool:
    """True when this lecture already has a report zip under notes/."""
    if video_map is None:
        video_map = build_video_map()

    course = pipeline_course_for(video)
    video_id = video_id_for(video)
    dest = destination_dir(course, video_id, video_map)

    if dest.is_dir() and any(dest.glob("*_report.zip")):
        return True

    notes = notes_dir()
    if not notes.is_dir():
        return False

    stem = video.stem
    if any(notes.rglob(f"{stem}_report.zip")):
        return True
    if video_id != stem and any(notes.rglob(f"{video_id}_report.zip")):
        return True

    marker = f"[{stem}]"
    for folder in notes.rglob("*"):
        if folder.is_dir() and marker in folder.name and any(folder.glob("*_report.zip")):
            return True
    return False


def organize_report_zip(
    zip_path: Path,
    course: str,
    video_id: str,
    video_map: dict[str, tuple[str, str | None]] | None = None,
) -> Path:
    """Move a report zip into notes/<course>/... preserving the existing layout."""
    if video_map is None:
        video_map = build_video_map()

    dest_dir = destination_dir(course, video_id, video_map)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_zip = dest_dir / zip_path.name

    if dest_zip.exists() and dest_zip.resolve() == zip_path.resolve():
        return dest_zip
    if dest_zip.exists() and dest_zip.stat().st_size == zip_path.stat().st_size:
        zip_path.unlink(missing_ok=True)
        return dest_zip
    if dest_zip.exists():
        dest_zip = dest_dir / f"{zip_path.stem}_dup{zip_path.suffix}"

    shutil.move(str(zip_path), str(dest_zip))
    return dest_zip


def organize_output(output_name: str, video_map: dict[str, tuple[str, str | None]] | None = None) -> Path | None:
    """Organize the report zip for a pipeline output name, if present."""
    zip_path = find_report_zip(output_name)
    if zip_path is None:
        return None

    if output_name.startswith("lecture_"):
        course = output_name
        video_id = output_name
    else:
        course, video_id = output_name.split("/", 1)

    return organize_report_zip(zip_path, course, video_id, video_map)
