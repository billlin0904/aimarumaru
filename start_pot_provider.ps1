<#
.SYNOPSIS
Start the local bgutil PO Token Provider with Docker Desktop on Windows.

.DESCRIPTION
The provider binds only to 127.0.0.1:4416 for Aimarumaru and yt-dlp on
the same Windows host. Do not expose this port through Cloudflare Tunnel.
#>

[CmdletBinding()]
param(
    [string]$ContainerName = "aimarumaru-pot-provider",
    [string]$Image = "brainicism/bgutil-ytdlp-pot-provider:1.3.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 4416
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker was not found. Install and start Docker Desktop first."
}

& docker version --format '{{.Server.Version}}' 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop is not running. Start it and run this script again."
}

$running = & docker ps --format '{{.Names}}' | Where-Object { $_ -eq $ContainerName }
if ($running) {
    Write-Host "PO Token Provider is already running: $ContainerName"
    exit 0
}

$existing = & docker ps -a --format '{{.Names}}' |
    Where-Object { $_ -eq $ContainerName }
if ($existing) {
    & docker rm -f $ContainerName | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Could not remove the stopped PO Token Provider container."
    }
}
& docker run -d `
    --name $ContainerName `
    --restart unless-stopped `
    --init `
    -p "127.0.0.1:${Port}:4416" `
    $Image | Out-Null

if ($LASTEXITCODE -ne 0) {
    throw "Could not start PO Token Provider. Check the container logs in Docker Desktop."
}

Write-Host "PO Token Provider started: http://127.0.0.1:$Port"
