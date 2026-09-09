#!/usr/bin/env python3
"""Compress scene keyframe images with ImageMagick.

Keeps pixel dimensions at full resolution by default (no downscaling).
Optional palette or JPEG/WebP modes for larger savings.

Examples:
  python3 scripts/compress_keyframes.py --run-dir runs/lecture_1
  python3 scripts/compress_keyframes.py --run-dir runs/lecture_1 --strategy palette
  python3 scripts/compress_keyframes.py --run-dir runs/lecture_1 --strategy jpeg --quality 85
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Allow running as script from scripts/ dir
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imagemagick_util import available, convert, ensure_installed, identify_size
from common import add_run_dir_arg, die

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def image_dimensions(path: Path) -> tuple[int, int]:
    return identify_size(path)


def compress_with_imagemagick(
    src: Path,
    dst: Path,
    *,
    strategy: str,
    quality: int,
    min_scale: float,
) -> None:
    orig_w, orig_h = image_dimensions(src)
    min_w = max(1, int(orig_w * min_scale))
    min_h = max(1, int(orig_h * min_scale))

    extra = ["-strip"]
    if strategy == "optimized-png":
        extra.extend(["-define", "png:compression-level=9"])
    elif strategy == "palette":
        extra.extend(["-colors", "256", "-define", "png:compression-level=9"])
    elif strategy in {"jpeg", "webp"}:
        extra.extend(["-quality", str(quality)])
    else:
        die(f"Unknown strategy: {strategy}")

    convert(src, dst, extra)

    out_w, out_h = image_dimensions(dst)
    if out_w < min_w or out_h < min_h:
        die(
            f"Compressed image {dst.name} is below min scale "
            f"({out_w}x{out_h} < {min_w}x{min_h})"
        )


def collect_images(
    run_dir: Path,
    *,
    include_attachments: bool,
    include_report: bool,
    report_only: bool,
) -> list[Path]:
    paths: list[Path] = []

    if report_only:
        report_attachments = run_dir / "report" / "attachments"
        if report_attachments.is_dir():
            paths.extend(
                sorted(p for p in report_attachments.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            )
        return paths

    keyframes = run_dir / "scenes" / "keyframes"
    if keyframes.is_dir():
        paths.extend(sorted(p for p in keyframes.iterdir() if p.suffix.lower() in IMAGE_EXTS))

    if include_report:
        report_attachments = run_dir / "report" / "attachments"
        if report_attachments.is_dir():
            paths.extend(
                sorted(p for p in report_attachments.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            )

    if include_attachments:
        attachments = run_dir / "attachments"
        if attachments.is_dir():
            paths.extend(
                sorted(p for p in attachments.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            )
    return paths


def output_path(src: Path, strategy: str) -> Path:
    if strategy == "jpeg":
        return src.with_suffix(".jpg")
    if strategy == "webp":
        return src.with_suffix(".webp")
    return src


def human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def compress_images(
    run_dir: Path,
    *,
    strategy: str,
    quality: int,
    min_scale: float,
    in_place: bool,
    include_attachments: bool,
    include_report: bool,
    report_only: bool,
    dry_run: bool,
) -> None:
    images = collect_images(
        run_dir,
        include_attachments=include_attachments,
        include_report=include_report,
        report_only=report_only,
    )
    if not images:
        if report_only:
            die(f"No images found under {run_dir}/report/attachments")
        die(f"No images found under {run_dir}/scenes/keyframes")

    before_total = 0
    after_total = 0
    converted = 0

    print(
        f"Compressing {len(images)} image(s) in {run_dir.name} "
        f"(strategy={strategy}, quality={quality}, min_scale={min_scale:.0%})",
        flush=True,
    )

    for src in images:
        before = src.stat().st_size
        before_total += before
        dst = output_path(src, strategy)

        if dry_run:
            with tempfile.NamedTemporaryFile(suffix=dst.suffix) as tmp:
                tmp_path = Path(tmp.name)
                compress_with_imagemagick(
                    src,
                    tmp_path,
                    strategy=strategy,
                    quality=quality,
                    min_scale=min_scale,
                )
                after = tmp_path.stat().st_size
        elif in_place:
            with tempfile.NamedTemporaryFile(suffix=dst.suffix, delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                compress_with_imagemagick(
                    src,
                    tmp_path,
                    strategy=strategy,
                    quality=quality,
                    min_scale=min_scale,
                )
                after = tmp_path.stat().st_size
                if dst == src:
                    shutil.move(str(tmp_path), str(dst))
                else:
                    shutil.move(str(tmp_path), str(dst))
                    src.unlink(missing_ok=True)
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            compress_with_imagemagick(
                src,
                dst,
                strategy=strategy,
                quality=quality,
                min_scale=min_scale,
            )
            after = dst.stat().st_size

        after_total += after
        converted += 1
        pct = 100.0 * (1 - after / before) if before else 0.0
        out_name = dst.name if in_place or not dry_run else f"{src.name} -> {dst.name}"
        orig_w, orig_h = image_dimensions(dst if (in_place and not dry_run) else src)
        print(
            f"  {out_name}: {human_size(before)} -> {human_size(after)} "
            f"({pct:.0f}% smaller, {orig_w}x{orig_h})",
            flush=True,
        )

    saved = before_total - after_total
    saved_pct = 100.0 * saved / before_total if before_total else 0.0
    print(
        f"\nDone: {converted} images, "
        f"{human_size(before_total)} -> {human_size(after_total)} "
        f"({saved_pct:.0f}% saved, {human_size(saved)} freed)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress keyframe images with ImageMagick")
    add_run_dir_arg(parser)
    parser.add_argument(
        "--strategy",
        choices=["optimized-png", "palette", "jpeg", "webp"],
        default="optimized-png",
        help="optimized-png: lossless strip+compress; palette: 256-color PNG; jpeg/webp: lossy",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=85,
        help="JPEG/WebP quality (default 85)",
    )
    parser.add_argument(
        "--min-scale",
        type=float,
        default=0.5,
        help="Never shrink below this fraction of original width/height (default 0.5)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace originals (default writes alongside for non-PNG strategies)",
    )
    parser.add_argument(
        "--attachments",
        action="store_true",
        help="Also compress run attachments/",
    )
    parser.add_argument(
        "--include-report",
        action="store_true",
        help="Also compress report/attachments/ (used by end-of-pipeline step)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Compress only report/attachments/ (for download bundles)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Measure savings without writing output files",
    )
    args = parser.parse_args()

    if args.min_scale <= 0 or args.min_scale > 1:
        die("--min-scale must be in (0, 1]")

    if not ensure_installed():
        print("Skipping compression (ImageMagick not available).", flush=True)
        return

    compress_images(
        args.run_dir,
        strategy=args.strategy,
        quality=args.quality,
        min_scale=args.min_scale,
        in_place=args.in_place or args.strategy in {"optimized-png", "palette"},
        include_attachments=args.attachments,
        include_report=args.include_report,
        report_only=args.report_only,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
