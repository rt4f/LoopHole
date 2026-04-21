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


INTERNAL_VERIFIER_CMD = "loophole-internal-mlir-verify"
DOCKER_VERIFIER_CMD = "loophole-docker-mlir-verify"
DEFAULT_DOCKER_MLIR_IMAGE = "loophole-polygeist:llvm17"
DEFAULT_DOCKER_MLIR_IMAGE_CANDIDATES = (
    DEFAULT_DOCKER_MLIR_IMAGE,
    "ghcr.io/schizoid-man/loophole-polygeist:llvm17",
)
DOCKER_IMAGE_ENV_VAR = "LOOPHOLE_MLIR_DOCKER_IMAGE"
DOCKER_VERIFY_DISABLE_ENV_VAR = "LOOPHOLE_DISABLE_DOCKER_MLIR_VERIFY"


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
            3. Docker toolchain mlir-opt
            4. known verifier binaries
            5. internal fallback verifier
    """
    if preferred_cmd:
        if shutil.which(shlex.split(preferred_cmd)[0]):
            return preferred_cmd
        return None

    env_cmd = os.getenv("LOOPHOLE_MLIR_VERIFY_CMD", "").strip()
    if env_cmd and shutil.which(shlex.split(env_cmd)[0]):
        return env_cmd

    docker_disabled = os.getenv(DOCKER_VERIFY_DISABLE_ENV_VAR, "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not docker_disabled and _docker_mlir_verifier_available():
        return DOCKER_VERIFIER_CMD

    for candidate in ("mlir-opt", "mlir-opt-18", "mlir-opt-17"):
        if shutil.which(candidate):
            return candidate

    # Fall back to a lightweight built-in syntax verifier so local test runs can
    # still execute artifact-validation tests without requiring external tooling.
    return INTERNAL_VERIFIER_CMD


def _docker_mlir_image() -> str:
    configured = os.getenv(DOCKER_IMAGE_ENV_VAR, "").strip()
    if configured:
        return configured

    for image in DEFAULT_DOCKER_MLIR_IMAGE_CANDIDATES:
        if _docker_image_available(image):
            return image

    return DEFAULT_DOCKER_MLIR_IMAGE


def _docker_image_available(image: str) -> bool:
    try:
        inspect = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (subprocess.SubprocessError, OSError):
        return False

    return inspect.returncode == 0


def _docker_mlir_verifier_available() -> bool:
    if not shutil.which("docker"):
        return False

    try:
        daemon = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (subprocess.SubprocessError, OSError):
        return False

    if daemon.returncode != 0:
        return False

    image = _docker_mlir_image()
    return _docker_image_available(image)


def is_internal_mlir_verifier(verifier_cmd: str) -> bool:
    return verifier_cmd.strip() == INTERNAL_VERIFIER_CMD


def is_docker_mlir_verifier(verifier_cmd: str) -> bool:
    return verifier_cmd.strip() == DOCKER_VERIFIER_CMD


def _validate_mlir_artifact_internal(mlir_text: str) -> MlirValidationResult:
    """Best-effort structural MLIR validation used when external tools are absent."""
    errors = []
    text = mlir_text.strip()

    if not text:
        errors.append("artifact is empty")

    if text.count("{") != text.count("}"):
        errors.append("unbalanced braces")

    if "func.func" not in text:
        errors.append("missing func.func declaration")

    if "return" not in text:
        errors.append("missing return terminator")

    # Require at least one recognizable op namespace in the emitted body.
    if not any(ns in text for ns in ("linalg.", "stablehlo.", "arith.")):
        errors.append("no recognized MLIR op namespace found (linalg/stablehlo/arith)")

    if errors:
        return MlirValidationResult(
            ok=False,
            command=INTERNAL_VERIFIER_CMD,
            returncode=1,
            stdout="",
            stderr="; ".join(errors),
        )

    return MlirValidationResult(
        ok=True,
        command=INTERNAL_VERIFIER_CMD,
        returncode=0,
        stdout="internal verifier passed",
        stderr="",
    )


def _validate_mlir_artifact_docker(mlir_text: str, timeout_sec: int = 30) -> MlirValidationResult:
    """Validate emitted MLIR by invoking mlir-opt in the configured Docker image."""
    image = _docker_mlir_image()
    with tempfile.TemporaryDirectory(prefix="loophole_mlir_validate_") as tmp_dir:
        artifact_path = Path(tmp_dir) / "artifact.mlir"
        artifact_path.write_text(mlir_text, encoding="utf-8")

        mount_arg = f"{tmp_dir}:/work"
        command = [
            "docker",
            "run",
            "--rm",
            "-v",
            mount_arg,
            image,
            "mlir-opt",
            "/work/artifact.mlir",
        ]

        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            return MlirValidationResult(
                ok=False,
                command=" ".join(command),
                returncode=124,
                stdout=(exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "docker mlir verifier timed out").strip() if isinstance(exc.stderr, str) else "docker mlir verifier timed out",
            )
        except OSError as exc:
            return MlirValidationResult(
                ok=False,
                command=" ".join(command),
                returncode=127,
                stdout="",
                stderr=str(exc),
            )

    return MlirValidationResult(
        ok=proc.returncode == 0,
        command=" ".join(command),
        returncode=proc.returncode,
        stdout=(proc.stdout or "").strip(),
        stderr=(proc.stderr or "").strip(),
    )


def validate_mlir_artifact(
    mlir_text: str,
    verifier_cmd: str,
    timeout_sec: int = 15,
) -> MlirValidationResult:
    """Validate emitted MLIR by invoking an external verifier command."""
    if is_docker_mlir_verifier(verifier_cmd):
        return _validate_mlir_artifact_docker(mlir_text, timeout_sec=max(timeout_sec, 30))

    if is_internal_mlir_verifier(verifier_cmd):
        return _validate_mlir_artifact_internal(mlir_text)

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
