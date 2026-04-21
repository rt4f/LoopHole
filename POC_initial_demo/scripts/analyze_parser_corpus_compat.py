"""Analyze parser compatibility on an MLIR corpus and emit JSON/Markdown summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from loophole.parser_corpus_compat import analyze_parser_compatibility


def _to_json(summary) -> Dict[str, Any]:
    return {
        "root": summary.root,
        "total_files": summary.total_files,
        "compatible_files": summary.compatible_files,
        "incompatible_files": summary.incompatible_files,
        "class_frequency": summary.class_frequency,
        "files": [
            {
                "file_path": row.file_path,
                "compatible": row.compatible,
                "classes": row.classes,
                "diagnostics_count": row.diagnostics_count,
                "loop_count": row.loop_count,
                "read_count": row.read_count,
                "write_count": row.write_count,
            }
            for row in summary.files
        ],
    }


def _to_markdown(summary) -> str:
    lines = []
    lines.append("# Parser Corpus Compatibility Summary")
    lines.append("")
    lines.append(f"- Corpus root: `{summary.root}`")
    lines.append(f"- Total files: {summary.total_files}")
    lines.append(f"- Compatible files: {summary.compatible_files}")
    lines.append(f"- Incompatible files: {summary.incompatible_files}")
    lines.append("")

    lines.append("## Incompatibility Class Frequency")
    lines.append("")
    lines.append("| Class | Frequency |")
    lines.append("|---|---:|")
    if summary.class_frequency:
        for name, count in summary.class_frequency.items():
            lines.append(f"| {name} | {count} |")
    else:
        lines.append("| none | 0 |")
    lines.append("")

    lines.append("## File-Level Results")
    lines.append("")
    lines.append("| File | Compatible | Classes | Loops | Reads | Writes | Diagnostics |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in summary.files:
        classes = ", ".join(row.classes) if row.classes else "-"
        lines.append(
            f"| {row.file_path} | {'yes' if row.compatible else 'no'} | {classes} | {row.loop_count} | {row.read_count} | {row.write_count} | {row.diagnostics_count} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", required=True, help="Path to folder containing .mlir files")
    parser.add_argument("--json-out", default=None, help="Optional JSON output file")
    parser.add_argument("--md-out", default=None, help="Optional Markdown output file")
    args = parser.parse_args()

    corpus_root = Path(args.corpus_dir).resolve()
    summary = analyze_parser_compatibility(corpus_root)

    as_json = _to_json(summary)
    print(json.dumps(as_json, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(as_json, indent=2) + "\n", encoding="utf-8")

    if args.md_out:
        Path(args.md_out).write_text(_to_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
