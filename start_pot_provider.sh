#!/usr/bin/env bash

# 啟動 bgutil 的本機 PO Token Provider。它只監聽 localhost，供同一台
# Aimarumaru/yt-dlp 使用；不應透過 Cloudflare Tunnel 或公開網際網路。
set -Eeuo pipefail

NAME="${POT_PROVIDER_CONTAINER_NAME:-aimarumaru-pot-provider}"
IMAGE="${POT_PROVIDER_IMAGE:-brainicism/bgutil-ytdlp-pot-provider:latest}"
HOST_PORT="${POT_PROVIDER_PORT:-4416}"

command -v docker >/dev/null 2>&1 || {
  echo "找不到 Docker；請安裝 Docker，或依 bgutil provider 文件以 Node.js 啟動 HTTP server。" >&2
  exit 1
}

if docker ps --format '{{.Names}}' | grep -Fxq "$NAME"; then
  echo "PO Token Provider 已在執行：$NAME"
  exit 0
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d \
  --name "$NAME" \
  --restart unless-stopped \
  --init \
  -p "127.0.0.1:${HOST_PORT}:4416" \
  "$IMAGE" >/dev/null

echo "PO Token Provider 已啟動：http://127.0.0.1:${HOST_PORT}"
