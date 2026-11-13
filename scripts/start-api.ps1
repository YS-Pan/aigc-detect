param(
    [ValidateRange(1024,65535)][int]$Port = 8766,
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'auto',
    [string]$Python = 'python',
    [switch]$Foreground
)
$ErrorActionPreference = 'Stop'
$pythonExe = (Get-Command $Python -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
$baseUrl = "http://127.0.0.1:$Port"
try {
    $health = Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 3
    if ($health.service -eq 'local-aigc-detectors') {
        Write-Output "API already running: $baseUrl (device=$($health.device), status=$($health.status))"
        return
    }
    throw "Port $Port is serving a different application."
} catch {
    $probe = [System.Net.Sockets.TcpClient]::new()
    try { $probe.Connect('127.0.0.1', $Port); $occupied = $true } catch { $occupied = $false } finally { $probe.Dispose() }
    if ($occupied) { throw "Port $Port is occupied and did not return this API's health response." }
}
$env:PYTHONUTF8 = '1'
$stateDir = Join-Path $env:LOCALAPPDATA 'local-aigc-detectors'
$logDir = Join-Path $stateDir 'logs'
$runtimeDir = Join-Path $stateDir 'runtime'
New-Item -ItemType Directory -Path $logDir,$runtimeDir -Force | Out-Null
if ($Foreground) {
    Push-Location $runtimeDir
    try {
        & $pythonExe -m aigc_detector serve --port $Port --device $Device
        $exitCode = $LASTEXITCODE
    } finally { Pop-Location }
    exit $exitCode
}
$service = Start-Process -FilePath $pythonExe -ArgumentList @('-m','aigc_detector','serve','--port',"$Port",'--device',$Device) -WorkingDirectory $runtimeDir -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir "api-$Port.out.log") -RedirectStandardError (Join-Path $logDir "api-$Port.err.log") -PassThru
$state = @{ ProcessId=$service.Id; StartTimeUtc=$service.StartTime.ToUniversalTime().ToString('o'); StartTimeUtcTicks=$service.StartTime.ToUniversalTime().Ticks; Port=$Port; Python=$pythonExe }
$state | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimeDir "api-$Port.json") -Encoding utf8
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    Start-Sleep -Milliseconds 1000
    $service.Refresh()
    if ($service.HasExited) { throw "API exited. Read $logDir/api-$Port.err.log." }
    try { $health = Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 2 } catch { continue }
    if ($health.service -eq 'local-aigc-detectors') {
        $worker = Get-Process -Id $health.process_id -ErrorAction Stop
        $state.WorkerProcessId = $worker.Id
        $state.WorkerStartTimeUtcTicks = $worker.StartTime.ToUniversalTime().Ticks
        $state.WorkerPython = $worker.Path
        $state | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimeDir "api-$Port.json") -Encoding utf8
        Write-Output "API ready: $baseUrl (PID $($service.Id), device=$($health.device), status=$($health.status))"
        Write-Output "API reference: $baseUrl/docs"
        return
    }
}
throw "API did not become ready within the startup wait. See $logDir/api-$Port.err.log and $runtimeDir/api-$Port.json."
