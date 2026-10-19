$ErrorActionPreference = 'Stop'
$Root = 'workspace\extensions'
$Audit = Join-Path $Root 'dispatch\recovery_20261019_0412'
$Expected = @{'42'=20192;'43'=34500;'44'=25640}

if ((Get-Item -LiteralPath $Root).Attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw 'Recovery root must be a physical directory'
}
$Processes = @(Get-CimInstance Win32_Process)
$Active = @($Processes | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -like '*extensions*'
})
if ($Active.Count) { throw 'An extension Python process exists; do not duplicate it' }
$AlreadyArchived = Test-Path -LiteralPath $Audit
if (Test-Path -LiteralPath (Join-Path $Audit 'recovery_audit.json')) { throw 'Recovery was already prepared; inspect its manifest' }
foreach ($Seed in $Expected.Keys) {
    if (Get-Process -Id $Expected[$Seed] -ErrorAction SilentlyContinue) {
        throw "Old PID still exists for seed $Seed; inspect its identity first"
    }
    $Lease = if ($AlreadyArchived) { Join-Path $Audit "seed$Seed.stale.lock" } else { Join-Path $Root "dispatch\leases\seed$Seed.lock" }
    $Value = Get-Content -Raw -LiteralPath $Lease | ConvertFrom-Json
    if ($Value.pid -ne $Expected[$Seed] -or $Value.seed -ne [int]$Seed -or
        $Value.config_sha256 -ne '363b512827ad871e7cce09fe76dbf503776ff113c6020762705dcbabc209f04f') {
        throw "Unexpected lease for seed $Seed"
    }
}
$CaptureWaiting = @($Processes | Where-Object {
    $_.Name -match '^(powershell|pwsh)\.exe$' -and
    $_.CommandLine -like '*run_capture_when_memory_ready.ps1*'
})
if ($CaptureWaiting.Count) { throw 'A capture resource waiter already exists' }
$CaptureLock = Join-Path $Root 'dispatch\capture.wait.lock'
if (-not $AlreadyArchived) {
    $Probe = [IO.File]::Open($CaptureLock, [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    $Probe.Dispose()
    New-Item -ItemType Directory -Path $Audit | Out-Null
} elseif (-not (Test-Path -LiteralPath (Join-Path $Audit 'capture.wait.stale.lock'))) {
    throw 'Partially archived recovery requires manual inspection'
}
$Markers = @(Get-ChildItem -LiteralPath (Join-Path $Root 'workers') -Filter completed.json -Recurse -File)
$Evidence = @($Markers | ForEach-Object {
    [ordered]@{path=$_.FullName;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}
})
$Archived = @()
foreach ($Seed in $Expected.Keys) {
    $Lease = Join-Path $Root "dispatch\leases\seed$Seed.lock"
    $Destination = Join-Path $Audit "seed$Seed.stale.lock"
    if (-not $AlreadyArchived) {
        Move-Item -LiteralPath $Lease -Destination $Destination
        Copy-Item -LiteralPath (Join-Path $Root "workers\seed$Seed\run_status.json") -Destination (Join-Path $Audit "seed$Seed.previous_run_status.json")
    }
    $Archived += $Destination
}
if (-not $AlreadyArchived) { Move-Item -LiteralPath $CaptureLock -Destination (Join-Path $Audit 'capture.wait.stale.lock') }
$Report = [ordered]@{
    snapshot_utc=[DateTime]::UtcNow.ToString('o')
    classification='Previous owner PIDs and resource waiter no longer exist; checkpoint-only queue recovery'
    old_pids=$Expected
    completed_refit_cells=$Markers.Count
    completed_markers=$Evidence
    archived_stale_seed_leases=$Archived
    completed_cells_must_be_skipped=$true
    existing_model_checkpoints_must_be_loaded=$true
    scientific_configuration_changed=$false
    n_jobs_per_worker=2
    capture_minimum_free_gib=8
    unfinished_calibrator_optimizer_state_not_saved=$true
}
$Report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $Audit 'recovery_audit.json') -Encoding utf8
$Report | ConvertTo-Json -Depth 8
