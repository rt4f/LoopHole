"""Utilities for artifact-level MLIR validation via external verifier tools."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MlirValidationResult:
    ok: bool
    command: str
    returncode: int
    stdout: str
    stderr: str


def find_mlir_verifier(preferred_cmd: Optional[str] = None) -> Optional[str]:
    """Return an MLIR verifier command if available on PATH.

    Discovery order:
      1. explicit preferred command
      2. LOOPHOLE_MLIR_VERIFY_CMD environment variable
      3. known verifier binaries
    """
    if preferred_cmd:
        if shutil.which(shlex.split(preferred_cmd)[0]):
            return preferred_cmd
        return None

    env_cmd = os.getenv("LOOPHOLE_MLIR_VERIFY_CMD", "").strip()
    if env_cmd and shutil.which(shlex.split(env_cmd)[0]):
        return env_cmd

    for candidate in ("mlir-opt", "mlir-opt-18", "mlir-opt-17"):
        if shutil.which(candidate):
            return candidate

    return None


def validate_mlir_artifact(
    mlir_text: str,
    verifier_cmd: str,
    timeout_sec: int = 15,
) -> MlirValidationResult:
    """Validate emitted MLIR by invoking an external verifier command."""
    with tempfile.TemporaryDirectory(prefix="loophole_mlir_validate_") as tmp_dir:
        artifact_path = Path(tmp_dir) / "artifact.mlir"
        artifact_path.write_text(mlir_text, encoding="utf-8")

        command = shlex.split(verifier_cmd) + [str(artifact_path)]
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

    return MlirValidationResult(
        ok=proc.returncode == 0,
        command=" ".join(command),
        returncode=proc.returncode,
        stdout=(proc.stdout or "").strip(),
        stderr=(proc.stderr or "").strip(),
    )
