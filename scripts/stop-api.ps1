param([ValidateRange(1024,65535)][int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$statePath = Join-Path $projectRoot ".runtime\api-$Port.json"
if (-not (Test-Path -LiteralPath $statePath)) { Write-Output 'No managed API process recorded.'; return }
$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
$targets = @()
if ($state.WorkerProcessId) {
    $targets += @{ Id=$state.WorkerProcessId; Ticks=$state.WorkerStartTimeUtcTicks; Path=$state.WorkerPython }
}
$targets += @{ Id=$state.ProcessId; Ticks=$state.StartTimeUtcTicks; Path=(Join-Path $projectRoot '.venv\Scripts\python.exe') }
$verified = @()
foreach ($target in $targets) {
    $process = Get-Process -Id $target.Id -ErrorAction SilentlyContinue
    if (-not $process) { continue }
    if ($process.StartTime.ToUniversalTime().Ticks -ne [long]$target.Ticks -or $process.Path -ne $target.Path) {
        throw 'Recorded process identity changed; refusing to stop a different process.'
    }
    $verified += $process
}
# Stop only the verified worker and venv launcher; no global process-tree query.
foreach ($process in $verified) {
    $process.Refresh()
    if (-not $process.HasExited) { Stop-Process -InputObject $process -Force -ErrorAction Stop }
}
Remove-Item -LiteralPath $statePath
Write-Output "Stopped the managed API on port $Port."
