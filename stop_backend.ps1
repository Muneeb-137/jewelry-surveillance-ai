# Stop whatever is listening on port 8000.
$ErrorActionPreference = "SilentlyContinue"
$Port = 8000

$listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $listeners) {
    Write-Host "Nothing running on port $Port."
    exit 0
}

$listeners | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
    Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped PID $_"
}

Write-Host "Port $Port is free."
