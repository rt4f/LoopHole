"""Unit tests for PolygeistFrontend wrapper around cgeist."""

import subprocess
from types import SimpleNamespace

import pytest

from loophole.polygeist_frontend import (
    CgeistInvocation,
    PolygeistFrontend,
    PolygeistFrontendError,
)


def test_build_command_cpp_multi_file() -> None:
    frontend = PolygeistFrontend(cgeist_bin="mock-cgeist")
    invocation = CgeistInvocation(
        source_files=["kernels/a.cpp", "kernels/b.cpp"],
        language="cpp",
        function="entry",
        include_dirs=["include", "third_party/include"],
        defines=["N=32", "USE_FAST_PATH"],
        std="c++17",
        extra_clang_args=["-O2", "-ffast-math"],
    )

    command = frontend.build_command(invocation)

    assert command[:4] == ["mock-cgeist", "-S", "-x", "c++"]
    assert "-function=entry" in command
    assert "-std=c++17" in command
    assert "-I" in command
    assert "-DN=32" in command
    assert "-DUSE_FAST_PATH" in command
    assert command[-2:] == ["kernels/a.cpp", "kernels/b.cpp"]


def test_generate_mlir_success_from_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    frontend = PolygeistFrontend(cgeist_bin="mock-cgeist")

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="module {\n}\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    invocation = CgeistInvocation(source_files=["kernel.c"], language="c")
    result = frontend.generate_mlir(invocation)

    assert result.returncode == 0
    assert result.mlir_text.strip().startswith("module")


def test_generate_mlir_reads_output_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    frontend = PolygeistFrontend(cgeist_bin="mock-cgeist")

    def fake_run(*args, **kwargs):
        command = args[0]
        output_idx = command.index("-o") + 1
        output_path = command[output_idx]
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("module {\n  func.func @k()\n}\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    out_file = tmp_path / "generated.mlir"
    invocation = CgeistInvocation(
        source_files=["kernel.c"],
        language="c",
        output_file=str(out_file),
    )
    result = frontend.generate_mlir(invocation)

    assert out_file.exists()
    assert "func.func" in result.mlir_text


def test_generate_mlir_raises_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    frontend = PolygeistFrontend(cgeist_bin="mock-cgeist")

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="parse error")

    monkeypatch.setattr(subprocess, "run", fake_run)

    invocation = CgeistInvocation(source_files=["broken.c"], language="c")

    with pytest.raises(PolygeistFrontendError) as exc_info:
        frontend.generate_mlir(invocation)

    assert "exit code 1" in str(exc_info.value)
    assert exc_info.value.returncode == 1
    assert "parse error" in exc_info.value.stderr


def test_generate_mlir_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    frontend = PolygeistFrontend(cgeist_bin="mock-cgeist")

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)

    invocation = CgeistInvocation(source_files=["slow.c"], language="c", timeout_sec=3)

    with pytest.raises(PolygeistFrontendError) as exc_info:
        frontend.generate_mlir(invocation)

    assert "timed out" in str(exc_info.value)


def test_generate_mlir_falls_back_to_docker_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frontend = PolygeistFrontend(cgeist_bin="missing-cgeist")
    invocation = CgeistInvocation(source_files=["kernel.c"], language="c")

    monkeypatch.setattr(frontend, "_docker_available", lambda: True)
    monkeypatch.setattr(frontend, "_resolve_docker_image", lambda: "ghcr.io/schizoid-man/loophole-polygeist:llvm17")
    monkeypatch.setattr(
        frontend,
        "_build_docker_command",
        lambda *_args, **_kwargs: (["docker", "run", "image", "cgeist"], None),
    )

    calls = []

    def fake_run(command, *args, **kwargs):
        calls.append(command)
        if command[0] == "missing-cgeist":
            raise FileNotFoundError("missing cgeist")
        return SimpleNamespace(returncode=0, stdout="module {\n}\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = frontend.generate_mlir(invocation)

    assert calls[0][0] == "missing-cgeist"
    assert calls[1][0] == "docker"
    assert result.command[0] == "docker"
    assert result.mlir_text.strip().startswith("module")


def test_generate_mlir_prefers_docker_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    frontend = PolygeistFrontend(cgeist_bin="mock-cgeist", prefer_docker=True)
    invocation = CgeistInvocation(source_files=["kernel.c"], language="c")

    monkeypatch.setattr(frontend, "_docker_available", lambda: True)
    monkeypatch.setattr(frontend, "_resolve_docker_image", lambda: "ghcr.io/schizoid-man/loophole-polygeist:llvm17")
    monkeypatch.setattr(
        frontend,
        "_build_docker_command",
        lambda *_args, **_kwargs: (["docker", "run", "image", "cgeist"], None),
    )

    def fake_run(command, *args, **kwargs):
        if command[0] == "mock-cgeist":
            pytest.fail("Host cgeist command should not run when prefer_docker=True and Docker is available")
        return SimpleNamespace(returncode=0, stdout="module {\n}\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = frontend.generate_mlir(invocation)

    assert result.command[0] == "docker"
    assert result.mlir_text.strip().startswith("module")


def test_generate_mlir_binary_not_found_without_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    frontend = PolygeistFrontend(cgeist_bin="missing-cgeist")
    invocation = CgeistInvocation(source_files=["kernel.c"], language="c")

    monkeypatch.setattr(frontend, "_docker_available", lambda: False)

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("missing cgeist")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(PolygeistFrontendError) as exc_info:
        frontend.generate_mlir(invocation)

    assert "cgeist binary not found" in str(exc_info.value)
