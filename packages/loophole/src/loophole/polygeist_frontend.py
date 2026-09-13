"""
polygeist_frontend.py - C/C++ frontend integration via Polygeist cgeist.

This module provides a small wrapper around cgeist so the rest of the
LoopHole pipeline can consume generated MLIR text without shell scripting.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence


DEFAULT_DOCKER_POLYGEIST_IMAGE = "loophole-polygeist:llvm17"
DEFAULT_DOCKER_POLYGEIST_IMAGE_CANDIDATES = (
    DEFAULT_DOCKER_POLYGEIST_IMAGE,
    "ghcr.io/schizoid-man/loophole-polygeist:llvm17",
)
DOCKER_POLYGEIST_IMAGE_ENV_VAR = "LOOPHOLE_CGEIST_DOCKER_IMAGE"
DOCKER_MLIR_IMAGE_ENV_VAR = "LOOPHOLE_MLIR_DOCKER_IMAGE"
DOCKER_CGEIST_DISABLE_ENV_VAR = "LOOPHOLE_DISABLE_DOCKER_CGEIST"


class PolygeistFrontendError(RuntimeError):
    """Raised when cgeist invocation fails."""

    def __init__(
        self,
        message: str,
        *,
        command: Optional[Sequence[str]] = None,
        returncode: Optional[int] = None,
        stderr: str = "",
        stdout: str = "",
    ) -> None:
        super().__init__(message)
        self.command = list(command) if command else None
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


@dataclass
class CgeistInvocation:
    """Input parameters used to construct a cgeist command."""

    source_files: List[str]
    language: str = "c"
    function: Optional[str] = None
    include_dirs: List[str] = field(default_factory=list)
    defines: List[str] = field(default_factory=list)
    std: Optional[str] = None
    extra_clang_args: List[str] = field(default_factory=list)
    output_file: Optional[str] = None
    timeout_sec: int = 60
    cwd: Optional[str] = None


@dataclass
class CgeistResult:
    """Result object returned by successful cgeist invocation."""

    command: List[str]
    returncode: int
    elapsed_ms: float
    stdout: str
    stderr: str
    mlir_text: str


class PolygeistFrontend:
    """Wrapper for converting C/C++ source files to MLIR via cgeist."""

    def __init__(
        self,
        cgeist_bin: str = "cgeist",
        default_timeout_sec: int = 60,
        verbose: bool = False,
        docker_image: Optional[str] = None,
        prefer_docker: bool = False,
        enable_docker_fallback: bool = True,
    ) -> None:
        self.cgeist_bin = cgeist_bin
        self.default_timeout_sec = default_timeout_sec
        self.verbose = verbose
        self.docker_image = docker_image
        self.prefer_docker = prefer_docker
        self.enable_docker_fallback = enable_docker_fallback

    def build_command(self, invocation: CgeistInvocation) -> List[str]:
        """Build a cgeist command line from invocation options."""
        if not invocation.source_files:
            raise ValueError("At least one source file is required.")

        if invocation.language not in {"c", "cpp"}:
            raise ValueError("language must be 'c' or 'cpp'.")

        cmd: List[str] = [self.cgeist_bin, "-S", "-x"]
        cmd.append("c++" if invocation.language == "cpp" else "c")

        if invocation.function:
            cmd.append(f"-function={invocation.function}")
        else:
            cmd.append("--function=*")

        cmd.append("--memref-fullrank")
        cmd.append("-raise-scf-to-affine")

        if invocation.std:
            cmd.append(f"-std={invocation.std}")

        for include_dir in invocation.include_dirs:
            cmd.extend(["-I", include_dir])

        for define in invocation.defines:
            cmd.append(f"-D{define}")

        cmd.extend(invocation.extra_clang_args)

        if invocation.output_file:
            cmd.extend(["-o", invocation.output_file])

        cmd.extend(invocation.source_files)
        return cmd

    def _docker_disabled(self) -> bool:
        return os.getenv(DOCKER_CGEIST_DISABLE_ENV_VAR, "").strip().lower() in {
            "1",
            "true",
            "yes",
        }

    def _docker_available(self) -> bool:
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

        return daemon.returncode == 0

    def _docker_image_available(self, image: str) -> bool:
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

    def _resolve_docker_image(self) -> str:
        if self.docker_image:
            return self.docker_image

        configured = os.getenv(DOCKER_POLYGEIST_IMAGE_ENV_VAR, "").strip()
        if configured:
            return configured

        inherited = os.getenv(DOCKER_MLIR_IMAGE_ENV_VAR, "").strip()
        if inherited:
            return inherited

        for candidate in DEFAULT_DOCKER_POLYGEIST_IMAGE_CANDIDATES:
            if self._docker_image_available(candidate):
                return candidate

        return DEFAULT_DOCKER_POLYGEIST_IMAGE

    def _resolve_host_path(self, path_text: str, base_dir: Path) -> Path:
        path = Path(path_text)
        if not path.is_absolute():
            path = base_dir / path
        return path.resolve()

    def _common_mount_root(self, paths: Sequence[Path]) -> Path:
        try:
            root = Path(os.path.commonpath([str(path) for path in paths]))
        except ValueError as exc:
            raise PolygeistFrontendError(
                "Input/output/include paths must be on the same drive for Docker volume mounting."
            ) from exc
        if root.is_file():
            root = root.parent
        if not root.exists():
            raise PolygeistFrontendError(f"Computed Docker mount root does not exist: {root}")
        return root

    def _to_container_path(self, host_path: Path, mount_root: Path) -> str:
        try:
            rel = host_path.relative_to(mount_root)
        except ValueError as exc:
            raise PolygeistFrontendError(
                f"Path '{host_path}' is not under Docker mount root '{mount_root}'."
            ) from exc
        return f"/workspace/{rel.as_posix()}"

    def _container_frontend_binary(self, invocation: CgeistInvocation) -> str:
        if self.cgeist_bin in {"cgeist", "cgeist++"}:
            return self.cgeist_bin
        return "cgeist++" if invocation.language == "cpp" else "cgeist"

    def _build_docker_command(
        self,
        invocation: CgeistInvocation,
        image: str,
    ) -> tuple[List[str], Optional[Path]]:
        base_dir = Path(invocation.cwd).resolve() if invocation.cwd else Path.cwd().resolve()
        source_paths = [self._resolve_host_path(source, base_dir) for source in invocation.source_files]
        include_paths = [self._resolve_host_path(include, base_dir) for include in invocation.include_dirs]

        output_host_path: Optional[Path] = None
        mount_inputs: List[Path] = [*(path.parent for path in source_paths), *include_paths]
        if invocation.output_file:
            output_host_path = self._resolve_host_path(invocation.output_file, base_dir)
            output_host_path.parent.mkdir(parents=True, exist_ok=True)
            mount_inputs.append(output_host_path.parent)

        mount_root = self._common_mount_root(mount_inputs)

        cgeist_cmd: List[str] = [
            self._container_frontend_binary(invocation),
            "-S",
        ]

        if invocation.function:
            cgeist_cmd.append(f"-function={invocation.function}")
        else:
            cgeist_cmd.append("--function=*")

        cgeist_cmd.append("--memref-fullrank")
        cgeist_cmd.append("-raise-scf-to-affine")

        if invocation.std:
            cgeist_cmd.append(f"-std={invocation.std}")

        for include_path in include_paths:
            cgeist_cmd.extend(["-I", self._to_container_path(include_path, mount_root)])

        for define in invocation.defines:
            cgeist_cmd.append(f"-D{define}")

        cgeist_cmd.extend(invocation.extra_clang_args)

        if output_host_path:
            cgeist_cmd.extend(["-o", self._to_container_path(output_host_path, mount_root)])

        cgeist_cmd.extend(self._to_container_path(path, mount_root) for path in source_paths)

        docker_cmd: List[str] = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{mount_root}:/workspace",
            image,
            *cgeist_cmd,
        ]
        return docker_cmd, output_host_path

    def _run_docker_cgeist(
        self,
        invocation: CgeistInvocation,
        timeout_sec: int,
    ) -> CgeistResult:
        image = self._resolve_docker_image()
        command, output_host_path = self._build_docker_command(invocation, image)

        if self.verbose:
            print("[cgeist/docker]", " ".join(command))

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except FileNotFoundError as exc:
            raise PolygeistFrontendError(
                "docker CLI not found while trying Docker cgeist fallback",
                command=command,
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise PolygeistFrontendError(
                f"docker cgeist timed out after {timeout_sec} seconds",
                command=command,
                stderr=exc.stderr or "",
                stdout=exc.stdout or "",
            ) from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        if proc.returncode != 0:
            raise PolygeistFrontendError(
                f"docker cgeist failed with exit code {proc.returncode}",
                command=command,
                returncode=proc.returncode,
                stderr=stderr,
                stdout=stdout,
            )

        mlir_text = stdout
        if output_host_path:
            if not output_host_path.exists():
                raise PolygeistFrontendError(
                    "docker cgeist succeeded but output file was not created",
                    command=command,
                    returncode=proc.returncode,
                    stderr=stderr,
                    stdout=stdout,
                )
            mlir_text = output_host_path.read_text(encoding="utf-8")

        if not mlir_text.strip():
            raise PolygeistFrontendError(
                "docker cgeist returned empty MLIR output",
                command=command,
                returncode=proc.returncode,
                stderr=stderr,
                stdout=stdout,
            )

        return CgeistResult(
            command=command,
            returncode=proc.returncode,
            elapsed_ms=elapsed_ms,
            stdout=stdout,
            stderr=stderr,
            mlir_text=mlir_text,
        )

    def _try_docker_cgeist(
        self,
        invocation: CgeistInvocation,
        timeout_sec: int,
    ) -> Optional[CgeistResult]:
        if not self.enable_docker_fallback or self._docker_disabled():
            return None
        if not self._docker_available():
            return None
        return self._run_docker_cgeist(invocation, timeout_sec)

    def generate_mlir(self, invocation: CgeistInvocation) -> CgeistResult:
        """Execute cgeist and return generated MLIR text."""
        timeout_sec = invocation.timeout_sec or self.default_timeout_sec

        if self.prefer_docker:
            docker_result = self._try_docker_cgeist(invocation, timeout_sec)
            if docker_result is not None:
                return docker_result

        command = self.build_command(invocation)

        if self.verbose:
            print("[cgeist]", " ".join(command))

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=invocation.cwd,
                check=False,
            )
        except FileNotFoundError as exc:
            docker_result = self._try_docker_cgeist(invocation, timeout_sec)
            if docker_result is not None:
                return docker_result
            raise PolygeistFrontendError(
                f"cgeist binary not found: {self.cgeist_bin}",
                command=command,
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise PolygeistFrontendError(
                f"cgeist timed out after {timeout_sec} seconds",
                command=command,
                stderr=exc.stderr or "",
                stdout=exc.stdout or "",
            ) from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        stdout = proc.stdout or ""
        stderr = proc.stderr or ""

        if proc.returncode != 0:
            raise PolygeistFrontendError(
                f"cgeist failed with exit code {proc.returncode}",
                command=command,
                returncode=proc.returncode,
                stderr=stderr,
                stdout=stdout,
            )

        mlir_text = stdout
        if invocation.output_file:
            output_path = Path(invocation.output_file)
            if not output_path.exists():
                raise PolygeistFrontendError(
                    "cgeist succeeded but output file was not created",
                    command=command,
                    returncode=proc.returncode,
                    stderr=stderr,
                    stdout=stdout,
                )
            mlir_text = output_path.read_text(encoding="utf-8")

        if not mlir_text.strip():
            raise PolygeistFrontendError(
                "cgeist returned empty MLIR output",
                command=command,
                returncode=proc.returncode,
                stderr=stderr,
                stdout=stdout,
            )

        return CgeistResult(
            command=command,
            returncode=proc.returncode,
            elapsed_ms=elapsed_ms,
            stdout=stdout,
            stderr=stderr,
            mlir_text=mlir_text,
        )
