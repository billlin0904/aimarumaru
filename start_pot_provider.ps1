<#!
.SYNOPSIS
在 Windows Docker Desktop 啟動本機 bgutil PO Token Provider。

.DESCRIPTION
Provider 只繫結到 127.0.0.1:4416，供同一台 Windows 主機上的
Aimarumaru / yt-dlp 使用；請勿將此連接埠公開到 Cloudflare Tunnel。
#>

[CmdletBinding()]
param(
    [string]$ContainerName = "aimarumaru-pot-provider",
    [string]$Image = "brainicism/bgutil-ytdlp-pot-provider:latest",
    [ValidateRange(1, 65535)]
    [int]$Port = 4416
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "找不到 Docker。請先安裝並啟動 Docker Desktop。"
}

& docker version --format '{{.Server.Version}}' 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop 尚未啟動。請啟動後再執行此腳本。"
}

$running = & docker ps --format '{{.Names}}' | Where-Object { $_ -eq $ContainerName }
if ($running) {
    Write-Host "PO Token Provider 已在執行：$ContainerName"
    exit 0
}

& docker rm -f $ContainerName 2>$null | Out-Null
& docker run -d `
    --name $ContainerName `
    --restart unless-stopped `
    --init `
    -p "127.0.0.1:${Port}:4416" `
    $Image | Out-Null

if ($LASTEXITCODE -ne 0) {
    throw "無法啟動 PO Token Provider。請查看 Docker Desktop 的容器日誌。"
}

Write-Host "PO Token Provider 已啟動：http://127.0.0.1:$Port"
