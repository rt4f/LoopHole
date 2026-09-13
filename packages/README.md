# packages/

One folder per installable component. Today there is one:

| Package | Description |
|---|---|
| [loophole](loophole/) | Affine IR → Linalg/StableHLO lifter, Z3 verifier, and CLI |

## Adding a package

When the research phase produces a component that has its own dependencies or
release cadence (for example a learned search guide, a benchmark harness, or a
new frontend), give it its own folder here rather than growing `loophole`:

```
packages/<name>/
├── pyproject.toml        # name, deps, entry points
├── README.md             # what it is, how to install, how to test
├── src/<import_name>/
└── tests/
```

Conventions:

- `src/` layout, so tests run against the installed package.
- The package depends on `loophole` via a normal dependency (`pip install -e ../loophole`), never via relative imports across folders.
- Add a CI job for it in `.github/workflows/ci.yml` (copy the `loophole` job and change `working-directory`).
- Record why it exists as an ADR in [docs/decisions](../docs/decisions/).
