param([int]$PairedResumeLauncherPid = 30508, [int]$ResourceQueuePid = 7524)

$ErrorActionPreference = 'Stop'
$root = 'workspace\harmonized'
$python = 'legacy_project\.venv\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = Join-Path $root 'results\grouped37\temp'
$env:TMP = $env:TEMP
$env:JOBLIB_TEMP_FOLDER = $env:TEMP
$lease = [System.IO.File]::Open((Join-Path $root 'dispatch\recovery_finalizer.lock'), [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
Write-Output "$(Get-Date -Format o) Waiting for paired recovery and original resource queue; no model processes are changed."
foreach ($processIdToWait in @($PairedResumeLauncherPid, $ResourceQueuePid)) {
    if ($null -ne (Get-Process -Id $processIdToWait -ErrorAction SilentlyContinue)) {
        Wait-Process -Id $processIdToWait
    }
}
foreach ($protocol in @('paired46', 'grouped37')) {
    foreach ($seed in 42..47) {
        if (-not (Test-Path -LiteralPath (Join-Path $root "results\$protocol\seed_$seed\COMPLETED.json"))) {
            throw "Incomplete $protocol seed $seed; completion remains forbidden."
        }
    }
}
if (Test-Path -LiteralPath (Join-Path $root 'MANUSCRIPT_HANDOFF.json')) {
    $handoff = Get-Content -LiteralPath (Join-Path $root 'MANUSCRIPT_HANDOFF.json') -Raw | ConvertFrom-Json
    if ($handoff.completion -eq $true -and $handoff.status -eq 'COMPLETE_VERIFIED') {
        Write-Output "$(Get-Date -Format o) Original queue already completed finalization; no second writer."
        exit 0
    }
    throw 'Unexpected non-complete handoff exists.'
}
foreach ($script in @('run_grouped37_confirmation.py', 'finalize_worker_runtime.py', 'derive_ece10.py', 'verify_harmonized_outputs.py', 'test_protocol_helpers.py', 'test_ece10.py', 'test_prediction_execution.py', 'test_resume_audit.py', 'test_handoff_guards.py', 'build_experiment_report.py')) {
    & $python -B -u (Join-Path $root $script)
    if ($LASTEXITCODE -ne 0) { throw "Finalization failed: $script ($LASTEXITCODE)" }
}
Write-Output "$(Get-Date -Format o) COMPLETE_VERIFIED after metadata-only recovery."
