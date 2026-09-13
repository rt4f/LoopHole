"""
Policy test: parser/emitter hardening tests must use loophole.tests.fixtures
as the single fixture source (no inline MLIR fixtures in those test files).
"""

from pathlib import Path


def _repo_test_file(relpath: str) -> Path:
    return Path(__file__).resolve().parents[1] / relpath


def test_affine_extractor_uses_shared_fixtures_only() -> None:
    p = _repo_test_file("unit/test_affine_extractor.py")
    text = p.read_text(encoding="utf-8")
    assert "from loophole.tests.fixtures import" in text
    assert "func.func @" not in text


def test_emitter_uses_shared_fixtures_only() -> None:
    p = _repo_test_file("unit/test_emitter.py")
    text = p.read_text(encoding="utf-8")
    assert "from loophole.tests.fixtures import" in text
    assert "func.func @" not in text
