param([ValidateRange(1024,65535)][int]$Port = 8766)
$ErrorActionPreference = 'Stop'
$statePath = Join-Path $env:LOCALAPPDATA "local-aigc-detectors\runtime\api-$Port.json"
if (-not (Test-Path -LiteralPath $statePath)) { Write-Output 'No managed API process recorded.'; return }
$state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
$targets = @()
if ($state.WorkerProcessId) {
    $targets += @{ Id=$state.WorkerProcessId; Ticks=$state.WorkerStartTimeUtcTicks; Path=$state.WorkerPython }
}
$targets += @{ Id=$state.ProcessId; Ticks=$state.StartTimeUtcTicks; Path=$state.Python }
$verified = @()
foreach ($target in $targets) {
    $process = Get-Process -Id $target.Id -ErrorAction SilentlyContinue
    if (-not $process) { continue }
    if ($process.StartTime.ToUniversalTime().Ticks -ne [long]$target.Ticks -or $process.Path -ne $target.Path) {
        throw 'Recorded process identity changed; refusing to stop a different process.'
    }
    $verified += $process
}
# Stop only recorded processes whose identities match.
foreach ($process in $verified) {
    $process.Refresh()
    if (-not $process.HasExited) { Stop-Process -InputObject $process -Force -ErrorAction Stop }
}
Remove-Item -LiteralPath $statePath
Write-Output "Stopped the managed API on port $Port."
