#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
START_SCRIPT="${START_SCRIPT:-${SCRIPT_DIR}/start_audioio.sh}"
AUDIOIO_GPU_UUID="${AUDIOIO_GPU_UUID:-GPU-aad845f7-e9e3-f475-6792-799f510bd2f4}"
OLLAMA_GPU_UUID="${OLLAMA_GPU_UUID:-GPU-7b8b0572-a31f-b3ed-1e39-c0680af38f9d}"
OLLAMA_CHECK_MODE="${OLLAMA_CHECK_MODE:-warn}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_COMMAND="${OLLAMA_COMMAND:-ollama}"
OLLAMA_START_TIMEOUT_SECONDS="${OLLAMA_START_TIMEOUT_SECONDS:-30}"
KOTOBAMARU_PROJECT_DIR="${KOTOBAMARU_PROJECT_DIR:-/vault/kotobamaru}"
KOTOBAMARU_VENV_DIR="${KOTOBAMARU_VENV_DIR:-/vault/venvs/kotobamaru}"
KOTOBAMARU_HOST="${KOTOBAMARU_HOST:-127.0.0.1}"
KOTOBAMARU_PORT="${KOTOBAMARU_PORT:-8100}"
KOTOBAMARU_PID_FILE="${KOTOBAMARU_PID_FILE:-${KOTOBAMARU_PROJECT_DIR}/data/kotobamaru.pid}"
KOTOBAMARU_START_SCRIPT="${KOTOBAMARU_START_SCRIPT:-${KOTOBAMARU_PROJECT_DIR}/start_kotobamaru.sh}"
KOTOBAMARU_LOG_FILE="${KOTOBAMARU_LOG_FILE:-${KOTOBAMARU_PROJECT_DIR}/logs/kotobamaru-server.log}"
KOTOBAMARU_START_TIMEOUT_SECONDS="${KOTOBAMARU_START_TIMEOUT_SECONDS:-60}"
KOTOBAMARU_URL="http://${KOTOBAMARU_HOST}:${KOTOBAMARU_PORT}"
OLLAMA_PID_FILE="${OLLAMA_PID_FILE:-${KOTOBAMARU_PROJECT_DIR}/data/ollama.pid}"
OLLAMA_LOG_FILE="${OLLAMA_LOG_FILE:-${KOTOBAMARU_PROJECT_DIR}/logs/ollama-server.log}"

log() {
    printf '[dual-gpu] %s\n' "$*"
}

warn() {
    printf '[dual-gpu] WARNING: %s\n' "$*" >&2
}

fail() {
    printf '[dual-gpu] ERROR: %s\n' "$*" >&2
    exit 1
}

gpu_exists() {
    local expected_uuid="$1"
    local detected_uuid
    while IFS= read -r detected_uuid; do
        detected_uuid="${detected_uuid//$'\r'/}"
        [[ "${detected_uuid}" == "${expected_uuid}" ]] && return 0
    done <<< "${AVAILABLE_GPU_UUIDS}"
    return 1
}

url_is_ready() {
    local url="$1"
    curl --silent --show-error --fail --max-time 2 "${url}" >/dev/null 2>&1
}

wait_for_url() {
    local service_name="$1"
    local url="$2"
    local timeout_seconds="$3"
    local second
    for ((second = 0; second < timeout_seconds; second += 1)); do
        if url_is_ready "${url}"; then
            log "${service_name} 已就緒：${url}"
            return 0
        fi
        sleep 1
    done
    return 1
}

run_systemctl() {
    if [[ "${EUID}" -eq 0 ]]; then
        systemctl "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo systemctl "$@"
    else
        return 1
    fi
}

ollama_systemd_is_available() {
    command -v systemctl >/dev/null 2>&1 \
        && [[ -d /run/systemd/system ]] \
        && systemctl cat ollama.service >/dev/null 2>&1
}

ollama_pid_has_expected_gpu() {
    local pid="$1"
    [[ -r "/proc/${pid}/environ" ]] || return 1
    tr '\0' '\n' < "/proc/${pid}/environ" \
        | grep -Fxq "CUDA_VISIBLE_DEVICES=${OLLAMA_GPU_UUID}"
}

check_ollama_binding() {
    [[ "${OLLAMA_CHECK_MODE}" != "off" ]] || return 0
    if ! ollama_systemd_is_available; then
        log "未偵測到 systemd，Ollama 將由腳本直接綁定 GPU 1"
        return 0
    fi

    local ollama_environment
    ollama_environment="$(
        systemctl show ollama.service --property=Environment --value 2>/dev/null \
            || true
    )"
    if [[ "${ollama_environment}" == *"CUDA_VISIBLE_DEVICES=${OLLAMA_GPU_UUID}"* ]]; then
        log "Ollama 已綁定保留 GPU：${OLLAMA_GPU_UUID}"
        return 0
    fi

    local message
    message="Ollama systemd 尚未綁定 ${OLLAMA_GPU_UUID}；請先執行 sudo systemctl edit ollama。"
    if [[ "${OLLAMA_CHECK_MODE}" == "required" ]]; then
        fail "${message}"
    fi
    warn "${message}"
}

start_ollama_manually() {
    local ollama_binary
    ollama_binary="$(command -v "${OLLAMA_COMMAND}" 2>/dev/null || true)"
    [[ -n "${ollama_binary}" ]] \
        || fail "找不到 Ollama 執行檔：${OLLAMA_COMMAND}"

    if [[ -f "${OLLAMA_PID_FILE}" ]]; then
        local existing_pid
        existing_pid="$(<"${OLLAMA_PID_FILE}")"
        if [[ "${existing_pid}" =~ ^[1-9][0-9]*$ ]] \
            && kill -0 "${existing_pid}" 2>/dev/null; then
            fail "Ollama PID ${existing_pid} 仍存在，但 API 尚未就緒；請先檢查 ${OLLAMA_LOG_FILE}。"
        fi
        rm -f -- "${OLLAMA_PID_FILE}"
    fi

    mkdir -p \
        "$(dirname "${OLLAMA_PID_FILE}")" \
        "$(dirname "${OLLAMA_LOG_FILE}")"
    log "容器無 systemd，直接以 GPU 1 啟動 Ollama"
    nohup env \
        CUDA_DEVICE_ORDER=PCI_BUS_ID \
        CUDA_VISIBLE_DEVICES="${OLLAMA_GPU_UUID}" \
        OLLAMA_HOST="${OLLAMA_HOST}" \
        "${ollama_binary}" serve \
        > "${OLLAMA_LOG_FILE}" 2>&1 &
    local ollama_pid=$!
    printf '%s\n' "${ollama_pid}" > "${OLLAMA_PID_FILE}"
    log "Ollama PID：${ollama_pid}（${OLLAMA_PID_FILE}）"
}

ensure_ollama_running() {
    local health_url="${OLLAMA_URL%/}/api/tags"
    if url_is_ready "${health_url}"; then
        if [[ "${OLLAMA_CHECK_MODE}" == "required" ]] \
            && ! ollama_systemd_is_available; then
            local managed_pid=""
            [[ -f "${OLLAMA_PID_FILE}" ]] && managed_pid="$(<"${OLLAMA_PID_FILE}")"
            if [[ ! "${managed_pid}" =~ ^[1-9][0-9]*$ ]] \
                || ! kill -0 "${managed_pid}" 2>/dev/null \
                || ! ollama_pid_has_expected_gpu "${managed_pid}"; then
                fail "Ollama 已在執行，但無法確認它只使用 GPU 1；請先停止舊 Ollama，或設定 OLLAMA_CHECK_MODE=warn。"
            fi
        fi
        log "Ollama 已在執行：${OLLAMA_URL}"
        return 0
    fi

    if ollama_systemd_is_available; then
        log "啟動 Ollama systemd 服務"
        run_systemctl start ollama.service \
            || fail "無法啟動 Ollama；請先執行 sudo systemctl start ollama。"
    else
        start_ollama_manually
    fi

    if ! wait_for_url "Ollama" "${health_url}" "${OLLAMA_START_TIMEOUT_SECONDS}"; then
        tail -n 30 "${OLLAMA_LOG_FILE}" >&2 || true
        fail "等待 Ollama 啟動逾時：${OLLAMA_URL}"
    fi
}

ensure_kotobamaru_running() {
    local health_url="${KOTOBAMARU_URL}/health"
    if url_is_ready "${health_url}"; then
        log "Kotobamaru 已在執行：${KOTOBAMARU_URL}"
        return 0
    fi

    [[ -x "${KOTOBAMARU_START_SCRIPT}" ]] \
        || fail "Kotobamaru 啟動腳本不存在或不可執行：${KOTOBAMARU_START_SCRIPT}"
    mkdir -p "$(dirname "${KOTOBAMARU_LOG_FILE}")"
    log "啟動 Kotobamaru：${KOTOBAMARU_URL}"
    nohup env \
        PROJECT_DIR="${KOTOBAMARU_PROJECT_DIR}" \
        VENV_DIR="${KOTOBAMARU_VENV_DIR}" \
        HOST="${KOTOBAMARU_HOST}" \
        PORT="${KOTOBAMARU_PORT}" \
        PID_FILE="${KOTOBAMARU_PID_FILE}" \
        OLLAMA_URL="${OLLAMA_URL}" \
        OLLAMA_URLS="${OLLAMA_URL}" \
        CUDA_VISIBLE_DEVICES="" \
        "${KOTOBAMARU_START_SCRIPT}" \
        > "${KOTOBAMARU_LOG_FILE}" 2>&1 &

    if ! wait_for_url \
        "Kotobamaru" \
        "${health_url}" \
        "${KOTOBAMARU_START_TIMEOUT_SECONDS}"; then
        tail -n 30 "${KOTOBAMARU_LOG_FILE}" >&2 || true
        fail "等待 Kotobamaru 啟動逾時：${KOTOBAMARU_URL}"
    fi
}

command -v nvidia-smi >/dev/null 2>&1 || fail "找不到 nvidia-smi。"
command -v curl >/dev/null 2>&1 || fail "找不到 curl。"
command -v grep >/dev/null 2>&1 || fail "找不到 grep。"
command -v nohup >/dev/null 2>&1 || fail "找不到 nohup。"
[[ -x "${START_SCRIPT}" ]] || fail "啟動腳本不存在或不可執行：${START_SCRIPT}"
[[ "${AUDIOIO_GPU_UUID}" != "${OLLAMA_GPU_UUID}" ]] \
    || fail "Audio IO 與 Ollama 不可指定同一個 GPU UUID。"
[[ "${OLLAMA_CHECK_MODE}" =~ ^(warn|required|off)$ ]] \
    || fail "OLLAMA_CHECK_MODE 只支援 warn、required、off。"
[[ "${OLLAMA_START_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
    || fail "OLLAMA_START_TIMEOUT_SECONDS 必須是正整數。"
[[ "${KOTOBAMARU_START_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
    || fail "KOTOBAMARU_START_TIMEOUT_SECONDS 必須是正整數。"
[[ "${KOTOBAMARU_PORT}" =~ ^[1-9][0-9]*$ ]] \
    || fail "KOTOBAMARU_PORT 必須是正整數。"

AVAILABLE_GPU_UUIDS="$(
    nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null
)"
gpu_exists "${AUDIOIO_GPU_UUID}" \
    || fail "找不到 Audio IO GPU：${AUDIOIO_GPU_UUID}"
gpu_exists "${OLLAMA_GPU_UUID}" \
    || fail "找不到 Ollama GPU：${OLLAMA_GPU_UUID}"

check_ollama_binding
ensure_ollama_running
ensure_kotobamaru_running

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${AUDIOIO_GPU_UUID}"
export TRANSLATE_API_BASE="${TRANSLATE_API_BASE:-${KOTOBAMARU_URL}}"

log "Audio IO GPU：${AUDIOIO_GPU_UUID}"
log "Ollama 保留 GPU：${OLLAMA_GPU_UUID}"
log "Kotobamaru API：${TRANSLATE_API_BASE}"
log "啟動：${START_SCRIPT}"

exec "${START_SCRIPT}" "$@"
