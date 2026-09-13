"""Unit tests for weekly benchmark report script behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from loophole.lifter import LiftResultState


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generate_weekly_benchmark_report.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("weekly_benchmark_report_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discover_fixtures_skips_lifted_outputs(tmp_path) -> None:
    module = _load_script_module()
    (tmp_path / "a.mlir").write_text("module {}", encoding="utf-8")
    (tmp_path / "a_lifted.mlir").write_text("module {}", encoding="utf-8")

    files = module._discover_fixtures(tmp_path)

    assert [p.name for p in files] == ["a.mlir"]


def test_generate_report_emits_trend_and_canonical_summary(monkeypatch, tmp_path) -> None:
    module = _load_script_module()
    (tmp_path / "matmul.mlir").write_text("module {}", encoding="utf-8")
    (tmp_path / "conv2d.mlir").write_text("module {}", encoding="utf-8")

    equivalent = SimpleNamespace(value="EQUIVALENT")
    not_equivalent = SimpleNamespace(value="NOT_EQUIVALENT")

    fake_results = iter(
        [
            SimpleNamespace(
                result_state=LiftResultState.PROVED,
                sketch_name="stablehlo.dot_general",
                verification=SimpleNamespace(
                    result=equivalent,
                    failed_implication=None,
                    sympy_z3_disagreement=None,
                    mismatch_summary=None,
                    counterexample_bindings={},
                ),
                is_accepted=lambda strict_mode=False: True,
                sympy_confidence=1.0,
                total_elapsed_ms=10.0,
                error=None,
            ),
            SimpleNamespace(
                result_state=LiftResultState.REFUTED,
                sketch_name=None,
                verification=SimpleNamespace(
                    result=not_equivalent,
                    failed_implication="source_to_sketch",
                    sympy_z3_disagreement=None,
                    mismatch_summary="mismatch",
                    counterexample_bindings={"i": "1"},
                ),
                is_accepted=lambda strict_mode=False: False,
                sympy_confidence=0.1,
                total_elapsed_ms=30.0,
                error="refuted",
            ),
        ]
    )

    class _FakeLifter:
        def __init__(self, *args, **kwargs):
            pass

        def lift(self, _src: str):
            return next(fake_results)

    monkeypatch.setattr(module, "Lifter", _FakeLifter)

    baseline_payload = {
        "label": "baseline_week",
        "generated_at_utc": "2026-04-01T00:00:00+00:00",
        "summary": {
            "proved": 2,
            "unproved_timeout": 0,
            "refuted": 0,
            "avg_elapsed_ms": 10.0,
        },
    }

    payload = module.generate_report(
        fixtures_dir=tmp_path,
        target="stablehlo",
        z3_timeout_ms=10_000,
        label="week_2",
        baseline_payload=baseline_payload,
        regression_threshold_fraction=0.05,
        docker_image="ghcr.io/schizoid-man/loophole-polygeist:llvm17",
    )

    assert payload["schema_version"] == "1.2"
    assert payload["trend_summary"]["available"] is True
    assert payload["trend_summary"]["delta"]["proved"] == -1
    assert payload["canonical_stablehlo_summary"]["applicable"] is True
    assert payload["canonical_stablehlo_summary"]["proved"] == 1
    assert payload["run_metadata"]["docker_image"] == "ghcr.io/schizoid-man/loophole-polygeist:llvm17"
