"""
Smoke tests for the standalone demo script.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_SCRIPT = PROJECT_ROOT / "examples" / "run_demo.py"


def _run_demo(mode: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LOOPHOLE_DEMO_FIXTURE_LIMIT"] = "1"
    env["LOOPHOLE_DEMO_Z3_TIMEOUT_MS"] = "500"

    return subprocess.run(
        [sys.executable, str(DEMO_SCRIPT), "--mode", mode],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_demo_script_plain_mode_smoke() -> None:
    proc = _run_demo("plain")
    output = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 0, output
    assert "Traceback" not in output


def test_demo_script_demo_mode_smoke() -> None:
    proc = _run_demo("demo")
    output = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 0, output
    assert "Traceback" not in output
