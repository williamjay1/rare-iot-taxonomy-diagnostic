$ErrorActionPreference = 'Stop'
$Root = 'workspace\extensions'
$Python = 'legacy_project\.venv\Scripts\python.exe'
$WaitLock = Join-Path $Root 'dispatch\capture.wait.lock'
$Handle = [System.IO.File]::Open($WaitLock, [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)

try {
    $ReadyChecks = 0
    while ($ReadyChecks -lt 2) {
        $FreeGiB = [double](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1048576
        $Seed44Started = Test-Path -LiteralPath (Join-Path $Root 'workers\seed44\run_status.json')
        Write-Output ("{0} capture waiting: free={1:F2} GiB, required=8.00 GiB; seed44_started={2}" -f
            (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $FreeGiB, $Seed44Started)
        if ($Seed44Started -and $FreeGiB -ge 8) {
            $ReadyChecks++
        } else {
            $ReadyChecks = 0
        }
        if ($ReadyChecks -lt 2) {
            Start-Sleep -Seconds 60
        }
    }
    Write-Output 'Resource gate passed; starting capture controls, n_jobs=2; separate output root'
    & $Python -B -u (Join-Path $Root 'scripts\run_capture_controls.py') --defer-shared-summary
    $Result = $LASTEXITCODE
} finally {
    $Handle.Dispose()
    Remove-Item -LiteralPath $WaitLock
}
exit $Result
