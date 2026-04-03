param(
    [string]$RepoWindowsPath = "C:\Users\Yasho\LoopHole",
    [ValidateSet("native-clone", "windows-tree")]
    [string]$Mode = "native-clone",
    [string]$ToolchainWslRoot = "~/loophole-toolchain"
)

# Convert C:\foo\bar -> /mnt/c/foo/bar (works even if wslpath interop is flaky)
$normalized = $RepoWindowsPath -replace '\\', '/'
if ($normalized -match '^([A-Za-z]):/(.*)$') {
    $drive = $matches[1].ToLower()
    $tail = $matches[2]
    $RepoWslPath = "/mnt/$drive/$tail"
} else {
    $RepoWslPath = wsl wslpath -a "$RepoWindowsPath"
}

if (-not $RepoWslPath) {
    Write-Error "Could not translate Windows path to WSL path."
    exit 1
}

Write-Host "Running WSL install script at: $RepoWslPath"
if ($Mode -eq "native-clone") {
    wsl bash -lc "cd '$RepoWslPath' && chmod +x scripts/wsl_install_polygeist_native.sh scripts/wsl_smoke_test.sh && ./scripts/wsl_install_polygeist_native.sh '$ToolchainWslRoot'"
} else {
    wsl bash -lc "cd '$RepoWslPath' && chmod +x scripts/wsl_install_polygeist.sh scripts/wsl_smoke_test.sh && ./scripts/wsl_install_polygeist.sh '$RepoWslPath'"
}
if ($LASTEXITCODE -ne 0) {
    Write-Error "Toolchain install failed."
    exit $LASTEXITCODE
}

Write-Host "Running smoke test"
if ($Mode -eq "native-clone") {
    wsl bash -lc "cd '$RepoWslPath' && ./scripts/wsl_smoke_test.sh '$RepoWslPath' '$ToolchainWslRoot/polygeist/build/bin'"
} else {
    wsl bash -lc "cd '$RepoWslPath' && ./scripts/wsl_smoke_test.sh '$RepoWslPath'"
}
exit $LASTEXITCODE
