param([int]$PairedLauncherPid = 21448)

$ErrorActionPreference = 'Stop'
$root = 'workspace\harmonized'
$python = 'legacy_project\.venv\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = Join-Path $root 'results\grouped37\temp'
$env:TMP = $env:TEMP
$env:JOBLIB_TEMP_FOLDER = $env:TEMP
$dispatch = Join-Path $root 'dispatch'
New-Item -ItemType Directory -Path $dispatch -Force | Out-Null
$queueLease = [System.IO.File]::Open((Join-Path $dispatch 'resource_queue.lock'), [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
$pending = [System.Collections.Generic.List[int]]::new()
$workers = @{}
foreach ($seed in 42..47) {
    if (-not (Test-Path -LiteralPath (Join-Path $root "results\grouped37\seed_$seed\COMPLETED.json"))) {
        $pending.Add($seed)
    }
}

function Invoke-CheckedPython {
    param([string[]]$PythonArguments)
    & $python -B -u @PythonArguments
    if ($LASTEXITCODE -ne 0) { throw "Python stage failed ($LASTEXITCODE): $PythonArguments" }
}

Write-Output "$(Get-Date -Format o) Resource queue: paired PID $PairedLauncherPid unchanged; admission limits read from resource_policy.json. Max 2 grouped workers; n_jobs=2."
$lastObservation = ''
while ($pending.Count -gt 0 -or $workers.Count -gt 0) {
    $policy = Get-Content -LiteralPath (Join-Path $root 'resource_policy.json') -Raw | ConvertFrom-Json
    if ($policy.estimator_n_jobs -ne 2 -or $policy.maximum_grouped_workers -gt 2 -or $policy.minimum_free_ram_gib_before_launch -lt 8) {
        throw 'Resource policy violates the preserved two-thread estimator or safe admission limits.'
    }
    foreach ($seed in @($workers.Keys)) {
        $process = $workers[$seed]
        if ($process.HasExited) {
            $process.WaitForExit()
            if ($process.ExitCode -ne 0) { throw "Grouped seed $seed failed with exit $($process.ExitCode); see dispatch logs. No automatic duplicate fit." }
            $completed = Join-Path $root "results\grouped37\seed_$seed\COMPLETED.json"
            $runtime = Join-Path $root "results\grouped37\seed_$seed\runtime.json"
            if (-not (Test-Path -LiteralPath $completed) -or -not (Test-Path -LiteralPath $runtime)) {
                throw "Grouped seed $seed exited without completed checkpoint and runtime."
            }
            Write-Output "$(Get-Date -Format o) Grouped seed $seed completed."
            $workers.Remove($seed)
        }
    }
    $freeGiB = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    $actualRuntimes = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
        $_.ExecutablePath -like '*codex-primary-runtime\dependencies\python\python.exe' -and $_.CommandLine -like '*.py*'
    }).Count
    $observation = "pending=$($pending.Count) grouped_active=$($workers.Count) actual_runtimes=$actualRuntimes free_floor_GiB=$([math]::Floor($freeGiB))"
    if ($observation -ne $lastObservation) {
        Write-Output "$(Get-Date -Format o) $observation"
        $lastObservation = $observation
    }
    if ($pending.Count -gt 0 -and $workers.Count -lt $policy.maximum_grouped_workers -and $freeGiB -ge $policy.minimum_free_ram_gib_before_launch -and $actualRuntimes -le $policy.maximum_actual_python_runtimes_before_launch) {
        $seed = $pending[0]
        $pending.RemoveAt(0)
        $stamp = Get-Date -Format 'yyyyMMddTHHmmss'
        $stdout = Join-Path $dispatch "grouped37_seed_${seed}_$stamp.stdout.log"
        $stderr = Join-Path $dispatch "grouped37_seed_${seed}_$stamp.stderr.log"
        $argsList = @('-B', '-u', (Join-Path $root 'run_grouped37_seed_worker.py'), '--seed', [string]$seed)
        $process = Start-Process -FilePath $python -ArgumentList $argsList -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $workers[$seed] = $process
        Write-Output "$(Get-Date -Format o) Started exclusively leased grouped seed $seed launcher PID $($process.Id), free GiB $([math]::Round($freeGiB,2))."
    }
    if ($pending.Count -gt 0 -or $workers.Count -gt 0) { Start-Sleep -Seconds $policy.poll_seconds }
}
Write-Output "$(Get-Date -Format o) All grouped workers complete. Waiting for existing paired launcher without restarting it."
$active = Get-Process -Id $PairedLauncherPid -ErrorAction SilentlyContinue
if ($null -ne $active) { Wait-Process -Id $PairedLauncherPid }
foreach ($seed in 42..47) {
    if (-not (Test-Path -LiteralPath (Join-Path $root "results\paired46\seed_$seed\COMPLETED.json"))) {
        throw "Paired run exited without seed $seed; finalization forbidden."
    }
}
$paired = Import-Csv -LiteralPath (Join-Path $root 'results\paired46\harmonized_results.csv')
if ($paired.Count -ne 78) { throw "Paired rows $($paired.Count), expected 78." }

# All workers have exited: a single writer performs final aggregation only.
Invoke-CheckedPython -PythonArguments @((Join-Path $root 'run_grouped37_confirmation.py'))
Invoke-CheckedPython -PythonArguments @((Join-Path $root 'finalize_worker_runtime.py'))
Invoke-CheckedPython -PythonArguments @((Join-Path $root 'derive_ece10.py'))
Invoke-CheckedPython -PythonArguments @((Join-Path $root 'verify_harmonized_outputs.py'))
Invoke-CheckedPython -PythonArguments @((Join-Path $root 'test_protocol_helpers.py'))
Invoke-CheckedPython -PythonArguments @((Join-Path $root 'test_ece10.py'))
Invoke-CheckedPython -PythonArguments @((Join-Path $root 'test_handoff_guards.py'))
Invoke-CheckedPython -PythonArguments @((Join-Path $root 'build_experiment_report.py'))
Write-Output "$(Get-Date -Format o) COMPLETE_VERIFIED: completion-only manuscript handoff written."
