# C-12 Quickstart and Runbook (Trusted and Exploratory)

Last updated: 2026-04-20

This runbook consolidates the C-12 workflow for trusted and exploratory usage and assumes Docker-first MLIR workflows.

## Profile Mapping

| User-facing profile | Canonical profile | Strict mode | Intended use |
|---|---|---|---|
| exploratory | local-explore | false | Fast iteration, allow unproved timeout outputs |
| trusted | ci-strict | true | Trust-gated runs for release and CI checks |

Notes:
- Existing profile names `local-explore` and `ci-strict` remain canonical.
- Aliases `exploratory` and `trusted` are now accepted in CLI profile options.

## Prerequisites

1. Docker Desktop is installed and running.
2. Use prebuilt image or build locally:

```powershell
docker pull ghcr.io/schizoid-man/loophole-polygeist:llvm17
```

## Trusted Quickstart (Proof-gated)

1. Compile C/C++ to MLIR with Docker-backed Polygeist:

```powershell
loophole compile POC_initial_demo/examples/smoke_matmul.c -o POC_initial_demo/reports/smoke_matmul.mlir --docker-image ghcr.io/schizoid-man/loophole-polygeist:llvm17
```

2. Lift with trusted profile:

```powershell
loophole lift POC_initial_demo/reports/smoke_matmul.mlir --target linalg --profile trusted --report
```

3. Batch trusted run with JSON report:

```powershell
loophole batch POC_initial_demo/tests/fixtures --target stablehlo --profile trusted --report --report-path POC_initial_demo/reports/stablehlo_trusted_report.json
```

Expected behavior:
- Any `UNPROVED_TIMEOUT` in trusted mode exits non-zero.
- Any `REFUTED` result is rejected.

## Exploratory Quickstart (Coverage-first)

1. Single-file exploratory run:

```powershell
loophole lift POC_initial_demo/tests/fixtures/transpose.mlir --target stablehlo --profile exploratory --report
```

2. Batch exploratory run:

```powershell
loophole batch POC_initial_demo/tests/fixtures --target stablehlo --profile exploratory --report --report-path POC_initial_demo/reports/stablehlo_exploratory_report.json
```

Expected behavior:
- `UNPROVED_TIMEOUT` remains visible but accepted under loose policy.
- `REFUTED` remains rejected and visible in summary/report.

## Docker Runner Helpers (Recommended)

Use helper scripts to ensure report workflows run in the Docker image:

```powershell
powershell -ExecutionPolicy Bypass -File POC_initial_demo/scripts/run_batch_report_in_docker.ps1 -Target stablehlo -Profile trusted
powershell -ExecutionPolicy Bypass -File POC_initial_demo/scripts/run_weekly_report_in_docker.ps1 -Target stablehlo -Label week_01
```

## Reproducibility Checklist

1. Keep fixture set stable between comparisons.
2. Keep profile and timeout constant for trend analysis.
3. Keep Docker image tag pinned (`llvm17`) for baseline continuity.
4. Check report `run_metadata` fields:
- `run_id`
- `profile_requested`
- `profile_resolved`
- `fixtures_fingerprint_sha256`
- `docker_image`

## Common Recovery Steps

1. If Docker mount path fails on Windows, keep input and output paths on the same drive.
2. If strict mode exits on timeout, rerun with exploratory profile for triage and preserve diagnostics.
3. If report schema parsers expect `1.0`, use `compatible_schema_versions` in `1.1` reports during migration.
