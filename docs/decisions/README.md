# Architecture Decision Records (ADRs)

Short records of significant decisions: what was decided, why, and what it costs.
They are never deleted. If a decision changes, write a new ADR that supersedes the old one.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-monorepo-layout.md) | Monorepo layout with packages/, research/, papers/ | Accepted |

## Writing one

1. Copy [0000-template.md](0000-template.md) to `NNNN-short-title.md` (next number).
2. Fill it in and keep it under a page.
3. Add a row above.

Good candidates: choosing a solver or search strategy, adding a target dialect,
changing the trust/verification policy, adding a package, changing the toolchain version.
