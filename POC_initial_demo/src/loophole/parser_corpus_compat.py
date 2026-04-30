"""Utilities to analyze parser compatibility on a corpus of MLIR files."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from loophole.affine_extractor import AffineExtractor


@dataclass(frozen=True)
class CorpusFileResult:
    file_path: str
    compatible: bool
    classes: List[str]
    diagnostics_count: int
    loop_count: int
    read_count: int
    write_count: int


@dataclass(frozen=True)
class CorpusCompatibilitySummary:
    root: str
    total_files: int
    compatible_files: int
    incompatible_files: int
    class_frequency: Dict[str, int]
    files: List[CorpusFileResult]


def _classify_incompatibilities(mlir_text: str, loop_count: int, write_count: int, diagnostics_error_count: int) -> List[str]:
    classes: List[str] = []

    if loop_count == 0:
        classes.append("no_loop_construct_detected")
    if write_count == 0:
        classes.append("no_store_detected")
    if diagnostics_error_count > 0:
        classes.append("parser_error_diagnostics")

    if "iter_args(" in mlir_text:
        classes.append("scf_iter_args_not_modeled")
    if "affine.if" in mlir_text or "scf.if" in mlir_text:
        classes.append("conditional_region_not_modeled")
    if "scf.while" in mlir_text or "affine.parallel" in mlir_text:
        classes.append("non_for_loop_control_not_modeled")
    return sorted(set(classes))


def analyze_parser_compatibility(root: Path) -> CorpusCompatibilitySummary:
    """Run AffineExtractor over every .mlir file under root and summarize misses."""
    if not root.exists():
        raise FileNotFoundError(f"Corpus root does not exist: {root}")

    extractor = AffineExtractor(validate_with_mlir=False)

    rows: List[CorpusFileResult] = []
    freq: Counter[str] = Counter()

    for path in sorted(root.rglob("*.mlir")):
        mlir_text = path.read_text(encoding="utf-8")
        info = extractor.extract(mlir_text)

        diag_error_count = sum(1 for d in info.diagnostics if d.level == "error")
        classes = _classify_incompatibilities(
            mlir_text=mlir_text,
            loop_count=len(info.loop_order),
            write_count=len(info.writes),
            diagnostics_error_count=diag_error_count,
        )

        compatible = len(classes) == 0
        for cls in classes:
            freq[cls] += 1

        rows.append(
            CorpusFileResult(
                file_path=str(path.relative_to(root).as_posix()),
                compatible=compatible,
                classes=classes,
                diagnostics_count=len(info.diagnostics),
                loop_count=len(info.loop_order),
                read_count=len(info.reads),
                write_count=len(info.writes),
            )
        )

    compatible_count = sum(1 for row in rows if row.compatible)

    return CorpusCompatibilitySummary(
        root=str(root),
        total_files=len(rows),
        compatible_files=compatible_count,
        incompatible_files=len(rows) - compatible_count,
        class_frequency=dict(sorted(freq.items(), key=lambda item: (-item[1], item[0]))),
        files=rows,
    )
