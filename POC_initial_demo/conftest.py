"""
conftest.py — shared pytest configuration and fixtures.
"""
import os
import pathlib
import pytest

from loophole.affine_extractor import AffineExtractor
from loophole.z3_checker import Z3EquivalenceChecker
from loophole.emitter import LinalgEmitter, StableHLOEmitter
from loophole.lifter import Lifter, resolve_policy_profile
from loophole.tests.fixtures import (
    MATMUL_MLIR,
    TRANSPOSE_2D_MLIR,
    CONV_1D_MLIR,
    CONV_2D_SIMPLE_MLIR,
    DOT_PRODUCT_MLIR,
    MATVEC_MLIR,
    ELEMENTWISE_ADD_MLIR,
    REDUCE_SUM_MLIR,
)
from loophole.mlir_validator import find_mlir_verifier

# ---------------------------------------------------------------------------
# Fixture file directory
# ---------------------------------------------------------------------------

FIXTURES_DIR = pathlib.Path(__file__).parent / "tests" / "fixtures"


def load_mlir_fixture(name: str) -> str:
    """Load an MLIR fixture from tests/fixtures/<name>.mlir."""
    path = FIXTURES_DIR / f"{name}.mlir"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Shared component fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def affine_extractor():
    """Session-scoped AffineExtractor (stateless)."""
    return AffineExtractor()


@pytest.fixture(scope="session")
def z3_checker():
    """Session-scoped Z3 checker with a reasonable timeout."""
    return Z3EquivalenceChecker(timeout_ms=15_000)


@pytest.fixture(scope="session")
def linalg_emitter():
    """Session-scoped Linalg emitter."""
    return LinalgEmitter()


@pytest.fixture(scope="session")
def stablehlo_emitter():
    """Session-scoped StableHLO emitter."""
    return StableHLOEmitter()


@pytest.fixture(scope="session")
def lifter_linalg():
    profile = resolve_policy_profile()
    return Lifter.from_policy_profile(target="linalg", profile_name=profile.name)


@pytest.fixture(scope="session")
def lifter_stablehlo():
    profile = resolve_policy_profile()
    return Lifter.from_policy_profile(target="stablehlo", profile_name=profile.name)


@pytest.fixture(scope="session")
def policy_profile_name() -> str:
    """Active policy profile name resolved from CLI/env defaults."""
    return resolve_policy_profile().name


# ---------------------------------------------------------------------------
# MLIR text fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def matmul_mlir():
    return MATMUL_MLIR

@pytest.fixture
def transpose_mlir():
    return TRANSPOSE_2D_MLIR

@pytest.fixture
def conv1d_mlir():
    return CONV_1D_MLIR

@pytest.fixture
def conv2d_mlir():
    return CONV_2D_SIMPLE_MLIR

@pytest.fixture
def dot_product_mlir():
    return DOT_PRODUCT_MLIR

@pytest.fixture
def matvec_mlir():
    return MATVEC_MLIR

@pytest.fixture
def elementwise_add_mlir():
    return ELEMENTWISE_ADD_MLIR

@pytest.fixture
def reduce_sum_mlir():
    return REDUCE_SUM_MLIR


# ---------------------------------------------------------------------------
# Parsed LoopNestInfo fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def matmul_loop_info():
    return AffineExtractor().extract(MATMUL_MLIR)

@pytest.fixture(scope="session")
def transpose_loop_info():
    return AffineExtractor().extract(TRANSPOSE_2D_MLIR)

@pytest.fixture(scope="session")
def conv1d_loop_info():
    return AffineExtractor().extract(CONV_1D_MLIR)

@pytest.fixture(scope="session")
def conv2d_loop_info():
    return AffineExtractor().extract(CONV_2D_SIMPLE_MLIR)

@pytest.fixture(scope="session")
def dot_product_loop_info():
    return AffineExtractor().extract(DOT_PRODUCT_MLIR)


# ---------------------------------------------------------------------------
# MLIR artifact validation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mlir_verifier_cmd() -> str:
    """Resolve verifier command or skip unless strict CI mode requires it."""
    cmd = find_mlir_verifier()
    require = os.getenv("LOOPHOLE_REQUIRE_MLIR_VERIFY", "0").strip() in {"1", "true", "TRUE", "yes"}
    if cmd:
        return cmd
    if require:
        pytest.fail(
            "LOOPHOLE_REQUIRE_MLIR_VERIFY is set, but no MLIR verifier command was found on PATH "
            "(tried LOOPHOLE_MLIR_VERIFY_CMD, mlir-opt, mlir-opt-18, mlir-opt-17)."
        )
    pytest.skip("No MLIR verifier command available on PATH; artifact validation skipped.")
