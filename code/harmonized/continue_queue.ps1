param([int]$PairedLauncherPid = 21448)

$ErrorActionPreference = 'Stop'
$root = 'workspace\harmonized'
$python = 'legacy_project\.venv\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = Join-Path $root 'results\grouped37\temp'
$env:TMP = $env:TEMP
$env:JOBLIB_TEMP_FOLDER = $env:TEMP

function Invoke-CheckedPython {
    param([string[]]$PythonArguments)
    & $python -B -u @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python stage failed ($LASTEXITCODE): $PythonArguments"
    }
}

Write-Output "$(Get-Date -Format o) Queue waiting for existing paired launcher $PairedLauncherPid; no restart."
$active = Get-Process -Id $PairedLauncherPid -ErrorAction SilentlyContinue
if ($null -ne $active) {
    Wait-Process -Id $PairedLauncherPid
}
$paired = Join-Path $root 'results\paired46'
foreach ($seed in 42..47) {
    if (-not (Test-Path -LiteralPath (Join-Path $paired "seed_$seed\COMPLETED.json"))) {
        throw "Paired run exited without completion of seed $seed; no confirmation started."
    }
}
$table = Import-Csv -LiteralPath (Join-Path $paired 'harmonized_results.csv')
if ($table.Count -ne 78) {
    throw "Paired run has $($table.Count) rows, expected 78."
}
if (-not (Test-Path -LiteralPath (Join-Path $paired 'invocation_runtime.json'))) {
    throw 'Paired invocation did not save its final runtime.'
}
Write-Output "$(Get-Date -Format o) Paired six seeds complete; starting grouped37 sequentially."
Invoke-CheckedPython -PythonArguments @((Join-Path $root 'run_grouped37_confirmation.py'))
Invoke-CheckedPython -PythonArguments @((Join-Path $root 'derive_ece10.py'))
Invoke-CheckedPython -PythonArguments @((Join-Path $root 'verify_harmonized_outputs.py'))
Invoke-CheckedPython -PythonArguments @((Join-Path $root 'test_protocol_helpers.py'))
Invoke-CheckedPython -PythonArguments @((Join-Path $root 'test_ece10.py'))
Invoke-CheckedPython -PythonArguments @((Join-Path $root 'build_experiment_report.py'))
Write-Output "$(Get-Date -Format o) COMPLETE: both six-seed protocols and independent output verification."
