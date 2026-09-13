# Phase 3 Batch Workflow UX

## Scope

This update completes Person C Phase 3 task C-11:

- C-11: Improve batch workflow UX for larger fixture directories.
- Done when: users can run and share structured project-wide reports quickly.

## Implemented

### 1) Large-directory file selection controls

`loophole batch` now supports explicit file selection and run-size controls:

- `--include-glob` (repeatable)
- `--exclude-glob` (repeatable)
- `--max-files`
- `--table-limit`

This lets users target fixture subsets quickly and avoid noisy terminal output for large trees.

### 2) Report-focused policy and strictness controls

Batch runs now align with policy-profile semantics and strict acceptance reporting:

- `--profile`
- `--strict`
- `--report-path`

Phase 4 extension:

- profile aliases `trusted` and `exploratory` are accepted and resolve to canonical profiles.
- reports now preserve both requested and resolved profile names under `run_metadata`.

Report summaries now include profile and strictness context, accepted counts, and filter metadata, which improves reproducibility when sharing results.

Additional reproducibility metadata now includes fixture fingerprint hash and per-file source hash in report payloads.

### 3) Artifact output layout and throughput ergonomics

Batch output behavior now scales better for large fixture directories:

- Emitted artifacts preserve relative source layout under output directory.
- Duplicate file-name collisions across subdirectories are avoided.
- `--write-emitted/--no-write-emitted` controls artifact generation for report-only runs.
- Generated `*_lifted.mlir` files are ignored as future batch inputs by default to avoid self-recursive reprocessing.

### 4) Optional verifier integration in batch mode

Batch mode now supports emitted-MLIR validation options already used in single-file flows:

- `--validate-emitted`
- `--mlir-verifier`

## Test Coverage

Added integration coverage in:

- `tests/integration/test_cli_batch_report.py`

New assertions cover:

- include/exclude/max-file selection behavior in report payload
- collision-safe emitted output layout for duplicated source basenames
- report-only runs with `--no-write-emitted`

## Result

Person C Phase 3 tasks C-08, C-09, C-10, and C-11 are now implemented and documented in the `docs/worklog/lane-c` lane.
