# C-15 Batch Profile Presets and Reproducibility Metadata

Last updated: 2026-04-20

This document defines profile preset UX and reproducibility metadata additions for batch and weekly reporting.

## Profile Preset UX

Canonical profiles:
- `local-explore`
- `ci-strict`

User-facing aliases:
- `exploratory` -> `local-explore`
- `trusted` -> `ci-strict`

CLI behavior:
1. `--profile` accepts canonical names and aliases.
2. Reports retain canonical resolved profile in `summary.profile`.
3. Requested profile is preserved in `run_metadata.profile_requested`.

## Report Metadata Additions (Schema 1.1)

Added fields:
- `run_metadata.run_id`
- `run_metadata.profile_requested`
- `run_metadata.profile_resolved`
- `run_metadata.strict_requested`
- `run_metadata.strict_mode`
- `run_metadata.fixtures_fingerprint_sha256`
- `run_metadata.fixture_count`
- `run_metadata.python_version`
- `run_metadata.platform`
- `run_metadata.loophole_version`
- `run_metadata.docker_image`

Per-result reproducibility field:
- `results[].source_sha256`

Selection expansion:
- `selection.processed_file_rels`

## Fingerprint Method

The fixture fingerprint is SHA256 over sorted pairs:
1. relative fixture path
2. source file SHA256

This allows deterministic reruns when fixture corpus and contents match.

## Compatibility Policy

1. New reports use `schema_version: 1.1`.
2. `compatible_schema_versions` includes `1.0` for migration tooling.
3. Existing `1.0` fields remain present and unchanged where possible.

## Example

Trusted batch run with alias:

```powershell
loophole batch POC_initial_demo/tests/fixtures --target stablehlo --profile trusted --report --report-path POC_initial_demo/reports/trusted_batch.json
```

The output records:
- `summary.profile = ci-strict`
- `run_metadata.profile_requested = trusted`
- `run_metadata.profile_resolved = ci-strict`
