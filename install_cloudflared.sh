#!/usr/bin/env bash

set -Eeuo pipefail

KEYRING_DIR="/usr/share/keyrings"
KEYRING_FILE="${KEYRING_DIR}/cloudflare-main.gpg"
REPOSITORY_FILE="/etc/apt/sources.list.d/cloudflared.list"

log() {
    printf '[cloudflared] %s\n' "$*"
}

fail() {
    printf '[cloudflared] ERROR: %s\n' "$*" >&2
    exit 1
}

command -v apt-get >/dev/null 2>&1 \
    || fail "此安裝腳本僅支援使用 apt-get 的 Debian/Ubuntu 系統。"

if [[ "${EUID}" -eq 0 ]]; then
    SUDO=()
elif command -v sudo >/dev/null 2>&1; then
    SUDO=(sudo)
else
    fail "安裝 cloudflared 需要 root 權限，且目前找不到 sudo。"
fi

if ! command -v curl >/dev/null 2>&1; then
    log "安裝 curl 與 CA 憑證"
    "${SUDO[@]}" apt-get update
    "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive \
        apt-get install -y curl ca-certificates
fi

log "加入 Cloudflare GPG key"
"${SUDO[@]}" mkdir -p --mode=0755 "${KEYRING_DIR}"
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
    | "${SUDO[@]}" tee "${KEYRING_FILE}" >/dev/null

log "加入 Cloudflare apt repository"
printf '%s\n' \
    "deb [signed-by=${KEYRING_FILE}] https://pkg.cloudflare.com/cloudflared any main" \
    | "${SUDO[@]}" tee "${REPOSITORY_FILE}" >/dev/null

log "安裝 cloudflared"
"${SUDO[@]}" apt-get update
"${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive \
    apt-get install -y cloudflared

log "安裝完成：$(cloudflared --version)"
