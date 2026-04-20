param(
    [string]$Image = "ghcr.io/schizoid-man/loophole-polygeist:llvm17",
    [string]$FixturesDir = "tests/fixtures",
    [ValidateSet("linalg", "stablehlo", "both")]
    [string]$Target = "linalg",
    [int]$Z3TimeoutMs = 10000,
    [string]$Label = "weekly",
    [string]$OutputDir = "reports/benchmarks",
    [string]$BaselineReport = "",
    [double]$RegressionThresholdFraction = 0.05
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")

$baselineArg = if ($BaselineReport) { " --baseline-report '$BaselineReport'" } else { "" }

$containerCommand = @(
    "cd /workspace/src/POC_initial_demo",
    "python3 -m venv /tmp/loophole-venv",
    "source /tmp/loophole-venv/bin/activate",
    "python -m pip install --upgrade pip",
    "python -m pip install -r requirements.txt",
    "python -m pip install -e .",
    "python scripts/generate_weekly_benchmark_report.py --fixtures-dir '$FixturesDir' --target '$Target' --z3-timeout-ms $Z3TimeoutMs --label '$Label' --output-dir '$OutputDir' --regression-threshold-fraction $RegressionThresholdFraction --docker-image '$Image'$baselineArg"
) -join "; "

$dockerArgs = @(
    "run",
    "--rm",
    "-v",
    "${repoRoot}:/workspace/src",
    "-e",
    "LOOPHOLE_DOCKER_IMAGE=$Image",
    $Image,
    "bash",
    "-lc",
    $containerCommand
)

Write-Host "[info] Running weekly report in Docker image: $Image"
docker @dockerArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
