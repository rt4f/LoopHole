param(
    [string]$Image = "ghcr.io/schizoid-man/loophole-polygeist:llvm17",
    [string]$InputDir = "tests/fixtures",
    [string]$OutputDir = "reports",
    [ValidateSet("linalg", "stablehlo", "both")]
    [string]$Target = "linalg",
    [string]$Profile = "default",
    [switch]$Strict,
    [switch]$ValidateEmitted
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..\..")

$strictArg = if ($Strict) { " --strict" } else { "" }
$validateArg = if ($ValidateEmitted) { " --validate-emitted" } else { "" }

$containerCommand = @(
    "cd /workspace/src/packages/loophole",
    "python3 -m venv /tmp/loophole-venv",
    "source /tmp/loophole-venv/bin/activate",
    "python -m pip install --upgrade pip",
    "python -m pip install -r requirements.txt",
    "python -m pip install -e .",
    "python -m loophole.cli batch '$InputDir' --output-dir '$OutputDir' --target '$Target' --profile '$Profile' --report$strictArg$validateArg"
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

Write-Host "[info] Running batch report in Docker image: $Image"
docker @dockerArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
