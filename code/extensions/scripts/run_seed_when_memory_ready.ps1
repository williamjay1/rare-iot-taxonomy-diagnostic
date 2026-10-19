param(
    [ValidateSet(42, 43, 44)][int]$Seed = 44,
    [double]$MinimumFreeGiB = 8
)

$ErrorActionPreference = 'Stop'
$Root = 'workspace\extensions'
$Python = 'legacy_project\.venv\Scripts\python.exe'
$WaitLock = Join-Path $Root "dispatch\seed$Seed.wait.lock"
$Handle = [System.IO.File]::Open($WaitLock, [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)

try {
    $ReadyChecks = 0
    while ($ReadyChecks -lt 2) {
        $Memory = Get-CimInstance Win32_OperatingSystem
        $FreeGiB = [double]$Memory.FreePhysicalMemory / 1048576
        Write-Output ("{0} seed{1} waiting: free={2:F2} GiB, required={3:F2} GiB" -f
            (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Seed, $FreeGiB, $MinimumFreeGiB)
        if ($FreeGiB -ge $MinimumFreeGiB) {
            $ReadyChecks++
        } else {
            $ReadyChecks = 0
        }
        if ($ReadyChecks -lt 2) {
            Start-Sleep -Seconds 60
        }
    }
    Write-Output "Resource gate passed; starting the isolated seed$Seed worker, n_jobs=2"
    & $Python -B -u (Join-Path $Root 'scripts\run_matched_extensions.py') `
        --phases lofo scarcity --seeds $Seed --worker-only `
        --lease-dir (Join-Path $Root 'dispatch\leases') `
        --dataset-cache-dir (Join-Path $Root 'datasets') `
        --out-dir (Join-Path $Root "workers\seed$Seed")
    $Result = $LASTEXITCODE
} finally {
    $Handle.Dispose()
    Remove-Item -LiteralPath $WaitLock
}
exit $Result
