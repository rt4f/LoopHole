# papers/

| Folder | What | Entry point |
|---|---|---|
| [thesis-report/](thesis-report/) | Project report (LaTeX) | `main.tex` → `main.pdf` |
| [ieee-paper/](ieee-paper/) | IEEE paper draft (LaTeX) | `main.tex` → `main.pdf` |

The factual source for the IEEE paper is
[ieee-paper/docs/implementation_findings.md](ieee-paper/docs/implementation_findings.md).

## Building

```bash
cd papers/thesis-report
latexmk -pdf main.tex
```

Only sources, images, and the final `main.pdf` are committed. Build artifacts
(`*.aux`, `*.synctex.gz`, `*.log`, …) are git-ignored.

## Adding a paper

Create `papers/<venue-or-name>/` with its own `main.tex`. Cite results from
`research/experiments/*` and `research/benchmarks/*`, not from ad-hoc runs.
