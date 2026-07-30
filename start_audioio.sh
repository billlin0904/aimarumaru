#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/vault/aimarumaru}"
VENV_DIR="${VENV_DIR:-/vault/venvs/aimarumaru}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${PROJECT_DIR}/requirements.txt}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8090}"

export HF_HOME="${HF_HOME:-/vault/cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/vault/cache}"
export AUDIOIO_LOG_TIMEZONE="${AUDIOIO_LOG_TIMEZONE:-Asia/Taipei}"
export AUDIOIO_LOG_LEVEL="${AUDIOIO_LOG_LEVEL:-INFO}"
export PYTHONUNBUFFERED=1

log() {
    printf '[setup] %s\n' "$*"
}

fail() {
    printf '[setup] ERROR: %s\n' "$*" >&2
    exit 1
}

command -v "${PYTHON_BIN}" >/dev/null 2>&1 \
    || fail "找不到 ${PYTHON_BIN}，請先安裝 Python 3.10 以上版本。"

[[ -d "${PROJECT_DIR}" ]] \
    || fail "找不到專案目錄：${PROJECT_DIR}"
[[ -f "${REQUIREMENTS_FILE}" ]] \
    || fail "找不到 requirements.txt：${REQUIREMENTS_FILE}"

mkdir -p "${VENV_DIR%/*}" "${HF_HOME}" "${XDG_CACHE_HOME}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    log "建立虛擬環境：${VENV_DIR}"
    "${PYTHON_BIN}" -m venv "${VENV_DIR}" \
        || fail "建立 venv 失敗；Ubuntu/Debian 請先安裝 python3-venv。"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

REQUIREMENTS_HASH="$(
    "${VENV_DIR}/bin/python" -c \
        'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
        "${REQUIREMENTS_FILE}"
)"
REQUIREMENTS_MARKER="${VENV_DIR}/.requirements.sha256"
INSTALLED_HASH=""
if [[ -f "${REQUIREMENTS_MARKER}" ]]; then
    INSTALLED_HASH="$(<"${REQUIREMENTS_MARKER}")"
fi

if [[ "${REQUIREMENTS_HASH}" != "${INSTALLED_HASH}" ]]; then
    log "安裝或更新 Python 套件"
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r "${REQUIREMENTS_FILE}"
    printf '%s\n' "${REQUIREMENTS_HASH}" > "${REQUIREMENTS_MARKER}"
else
    log "requirements.txt 未變更，略過套件安裝"
fi

for binary in ffmpeg ffprobe; do
    command -v "${binary}" >/dev/null 2>&1 \
        || fail "找不到 ${binary}；請先執行 apt-get update && apt-get install -y ffmpeg。"
done

cd "${PROJECT_DIR}"

FFMPEG_VERSION="$(ffmpeg -version 2>/dev/null)"
FFMPEG_VERSION="${FFMPEG_VERSION%%$'\n'*}"

log "Python：$(python --version 2>&1)"
log "FFmpeg：${FFMPEG_VERSION}"
log "模型快取：${HF_HOME}"
log "啟動 Audio IO：http://${HOST}:${PORT}"

exec python -m uvicorn main:app \
    --host "${HOST}" \
    --port "${PORT}"
