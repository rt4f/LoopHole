# LoopHole

**Automated lifting of legacy scalar loop code into MLIR tensor dialects.**

LoopHole takes loop-based C/C++ kernels (via [Polygeist](https://github.com/llvm/Polygeist) → MLIR Affine IR),
recognizes the tensor operation they implement, **proves equivalence with Z3**, and emits
high-level **Linalg** or **StableHLO**. Downstream ML compilers (XLA, IREE) can then run the
code on accelerators without manual rewriting.

```
C/C++ ──(Polygeist, in Docker)──▶ Affine IR ──(extract · sketch match · Z3 proof)──▶ Linalg / StableHLO ──▶ XLA · IREE · LLVM
```

## Quick start

Full instructions: **[SETUP.md](SETUP.md)**.

```powershell
docker build -f docker/polygeist/Dockerfile -t loophole-polygeist:llvm17 -t ghcr.io/schizoid-man/loophole-polygeist:llvm17 packages/loophole
cd packages/loophole
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt; pip install -e ".[dev]"
loophole lift-c examples/smoke_matmul.c --target linalg
```

## Repository layout

```
LoopHole/
├── packages/
│   └── loophole/          Python package: lifter, Z3 verifier, emitters, CLI, tests, examples
├── docker/
│   └── polygeist/         Supported toolchain image (Polygeist cgeist + LLVM/MLIR 17)
├── docs/                  All documentation (index: docs/README.md)
│   ├── guides/            How-to
│   ├── architecture/      How it works
│   ├── background/        Problem statement, state of the art, references
│   ├── planning/          Roadmaps and team plans
│   ├── status/            Status reports, audits, handoffs
│   ├── decisions/         Architecture Decision Records
│   └── worklog/           Per-lane (A/B/C) sprint records
├── research/              Research phase: experiments, benchmarks, notebooks
├── papers/                Thesis report and IEEE paper (LaTeX)
├── tools/legacy/          Superseded tooling (unsupported)
├── .github/workflows/     CI and Docker image publishing
├── SETUP.md               Environment setup
└── CONTRIBUTING.md        Where things go and how to contribute
```

## Where to start reading

| If you want to… | Read |
|---|---|
| Understand everything from scratch (start here) | [Engineering handbook](docs/handbook/README.md): five PDFs covering orientation, architecture, every module, toolchain/tests/CI, and the tech-debt audit |
| Run it | [SETUP.md](SETUP.md) |
| Understand the problem | [Problem statement](docs/background/problem-statement.md) → [Project overview](docs/background/project-overview.md) → [State of the art](docs/background/state-of-the-art.md) |
| Understand the code | [POC documentation](docs/architecture/poc-documentation.md), [package README](packages/loophole/README.md) |
| Know where the project stands | [Project status](docs/status/project-status.md) → [Future plan](docs/planning/future-plan.md) → [Plan v2](docs/planning/three-person-execution-plan-v2-12-weeks.md) |
| Start research work | [research/](research/README.md), [Novel research directions](docs/background/novel-research-directions.md) |
| Add something to the repo | [CONTRIBUTING.md](CONTRIBUTING.md) |
| See everything | [docs/README.md](docs/README.md) |

## State of the art (Feb 2026)

| Framework | Venue | Best speedup | Mechanism |
|---|---|---|---|
| mlirSynth | PACT 2023 | 21.6× (TPU) | Bottom-up enumerative synthesis |
| Tenspiler | ECOOP 2024 | 105× (kernel avg) | SMT-verified lifting via Rosette |
| Tensorize | CGO 2025 | **4,102× (GPU)** | Symbolic tracing + algebraic solving |
| STAGG | PLDI 2025 | 99% accuracy, 3.19 s avg | LLM-guided probabilistic grammar + A\* |

## License

See [LICENSE](LICENSE).
