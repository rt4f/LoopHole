"""
polygeist_frontend.py - C/C++ frontend integration via Polygeist cgeist.

This module provides a small wrapper around cgeist so the rest of the
LoopHole pipeline can consume generated MLIR text without shell scripting.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence


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
    ) -> None:
        self.cgeist_bin = cgeist_bin
        self.default_timeout_sec = default_timeout_sec
        self.verbose = verbose

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

    def generate_mlir(self, invocation: CgeistInvocation) -> CgeistResult:
        """Execute cgeist and return generated MLIR text."""
        command = self.build_command(invocation)
        timeout_sec = invocation.timeout_sec or self.default_timeout_sec

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
