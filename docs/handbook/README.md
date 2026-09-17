# LoopHole Engineering Handbook

The current, verified description of the LoopHole codebase, written in September 2026 for a team that did not write the original code. Every defect it calls "verified" was reproduced against the real toolchain.

| Part | PDF | Source |
|---|---|---|
| 1. Orientation and project overview | [01-orientation.pdf](01-orientation.pdf) | [source](source/01-orientation.md) |
| 2. Architecture and pipeline | [02-architecture.pdf](02-architecture.pdf) | [source](source/02-architecture.md) |
| 3. Module reference | [03-module-reference.pdf](03-module-reference.pdf) | [source](source/03-module-reference.md) |
| 4. Toolchain, tests, CI and operations | [04-toolchain-tests-ci.pdf](04-toolchain-tests-ci.pdf) | [source](source/04-toolchain-tests-ci.md) |
| 5. Tech-debt audit and remediation plan | [05-tech-debt-audit.pdf](05-tech-debt-audit.pdf) | [source](source/05-tech-debt-audit.md) |

New to the project? Read Part 1, then Part 2 chapters 1 to 4.

## Keeping it current

The Markdown in `source/` is the source of truth; the PDFs are generated. If your PR makes a handbook statement false, update the Markdown in the same PR and rebuild:

```bash
pip install reportlab
python docs/handbook/build_handbook.py        # all parts
python docs/handbook/build_handbook.py 03     # one part
```

Diagrams are defined as data in `diagrams.py`. The supported Markdown subset is described at the top of `build_handbook.py`.
