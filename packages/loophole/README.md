# loophole (Python package)

The LoopHole lifter: parses MLIR Affine IR, matches loop nests against an
operation sketch library, proves equivalence with Z3, and emits Linalg or
StableHLO. The CLI also drives Polygeist (`cgeist`) inside Docker to go from
C/C++ to MLIR.

For first-time setup (Docker image + venv) see [SETUP.md](../../SETUP.md).

## Install (dev)

```bash
cd packages/loophole
python -m venv .venv
.venv/Scripts/activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## CLI

| Command | What it does |
|---|---|
| `loophole compile` | C/C++ → MLIR with Polygeist in Docker |
| `loophole cgeist` | Lower-level `cgeist` wrapper (local binary or Docker) |
| `loophole lift` | MLIR Affine IR → Linalg / StableHLO (with Z3 proof) |
| `loophole lift-c` | One step: C/C++ → lifted tensor IR |
| `loophole verify` | Z3 equivalence check only |
| `loophole batch` | Lift a directory, optional JSON report / trusted profile |
| `loophole replay` | Replay a counterexample from a batch report |
| `loophole demo` / `sketches` | Built-in demo / list sketches |

## Layout

```
packages/loophole/
├── pyproject.toml      # package metadata, deps, pytest config
├── requirements.txt    # pinned-ish deps (used by CI and the Docker image)
├── conftest.py         # shared pytest fixtures
├── src/loophole/       # library + CLI (cli.py is the entry point)
│   └── tests/fixtures.py   # canonical MLIR fixtures (also used by `loophole demo`)
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/       # .mlir inputs, corpus, expected lifted outputs
├── examples/           # C kernels + run_demo.py
└── scripts/            # report generators and Docker report helpers
```

## Tests

```bash
python -m pytest -q
```

Set `PYTHONUTF8=1` on Windows. See [SETUP.md](../../SETUP.md#45-tests) for the
current baseline and known failures.
