#!/usr/bin/env python3
"""Parse research PDFs to markdown via docling."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import add_run_dir_arg, die, save_json


def parse_papers(pdf_paths: list[Path], papers_dir: Path, force: bool) -> list[dict]:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        die(
            "docling is required for paper parsing. "
            'Install with: pip install "NotesGenerator[papers]"'
        )

    manifest_path = papers_dir / "manifest.json"
    if manifest_path.exists() and not force:
        from common import load_json

        existing = load_json(manifest_path)
        print(f"Papers already parsed: {manifest_path} ({len(existing)} entries)")
        return existing

    papers_dir.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()
    manifest: list[dict] = []

    for pdf_path in pdf_paths:
        pdf_path = pdf_path.resolve()
        if not pdf_path.exists():
            print(f"Warning: missing PDF: {pdf_path}")
            continue

        md_name = pdf_path.stem + ".md"
        md_path = papers_dir / md_name
        if md_path.exists() and not force:
            print(f"Reusing parsed paper: {md_path.name}")
        else:
            print(f"Parsing PDF: {pdf_path.name} ...")
            result = converter.convert(str(pdf_path))
            md_path.write_text(result.document.export_to_markdown(), encoding="utf-8")
            print(f"  → {md_path}")

        title = pdf_path.stem.replace("-", " ").replace("_", " ").title()
        manifest.append(
            {
                "source_pdf": str(pdf_path),
                "markdown": md_name,
                "title": title,
            }
        )

    save_json(manifest_path, manifest)
    print(f"Wrote {len(manifest)} paper(s) -> {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse research PDFs to markdown")
    add_run_dir_arg(parser)
    parser.add_argument(
        "pdfs",
        nargs="+",
        type=Path,
        help="One or more PDF file paths",
    )
    parser.add_argument("--force", action="store_true", help="Re-parse even if outputs exist")
    args = parser.parse_args()

    parse_papers(args.pdfs, args.run_dir / "papers", args.force)


if __name__ == "__main__":
    main()
