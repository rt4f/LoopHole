# CLI States and Batch JSON Report

## C-04: CLI Result-State Reporting

The CLI uses explicit state names derived from `LiftResult.result_state`:

- `PROVED`: formally equivalent by Z3.
- `UNPROVED_TIMEOUT`: candidate matched and emitted, but proof ended in timeout/unknown.
- `REFUTED`: not equivalent or otherwise rejected.

This state vocabulary is now applied consistently in:

- single-file lift status summary,
- batch table output,
- demo output and summary table.

## C-05: Batch JSON Schema

`loophole batch --report` writes JSON with a stable top-level schema:

- `schema_version`
- `compatible_schema_versions`
- `generated_at_utc`
- `input_dir`
- `output_dir`
- `target`
- `z3_timeout_ms`
- `run_metadata`
- `selection`
- `summary`
- `canonical_stablehlo_summary`
- `results`

`summary` includes:

- total files,
- counts for PROVED / UNPROVED_TIMEOUT / REFUTED,
- accepted counts under loose and strict policies.

`run_metadata` includes reproducibility context:

- profile requested and resolved,
- strict-mode requested and effective flags,
- fixture fingerprint hash,
- python/platform/tool version fields,
- docker image tag (when provided).

`canonical_stablehlo_summary` includes:

- required canonical op list,
- observed and missing canonical ops,
- proved/unproved/refuted counts on canonical-matched results,
- unmatched canonical candidate files.

Each `results` item includes:

- file path and file name,
- matched sketch,
- state label,
- acceptance booleans,
- verification label,
- confidence,
- elapsed time,
- error text (if any),
- emitted output file path (if written).
- source SHA256 for reproducible reruns.

Compatibility note:

- New reports use schema `1.1`.
- `compatible_schema_versions` includes `1.0` for migration tooling.

## Usage

```bash
loophole batch tests/fixtures --report
loophole batch tests/fixtures --report --report-path reports/latest_batch.json
```
