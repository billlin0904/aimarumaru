#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/vault/aimarumaru}"
PID_FILE="${PID_FILE:-${PROJECT_DIR}/audioio.pid}"
GRACE_SECONDS="${GRACE_SECONDS:-30}"
EXPECTED_COMMAND_PATTERN="${EXPECTED_COMMAND_PATTERN:-uvicorn main:app}"

log() {
    printf '[stop] %s\n' "$*"
}

fail() {
    printf '[stop] ERROR: %s\n' "$*" >&2
    exit 1
}

remove_matching_pid_file() {
    local pid="$1"
    if [[ -f "${PID_FILE}" ]] && [[ "$(<"${PID_FILE}")" == "${pid}" ]]; then
        rm -f -- "${PID_FILE}"
    fi
}

PID="${1:-}"
if [[ -z "${PID}" ]]; then
    [[ -f "${PID_FILE}" ]] || fail "找不到 PID 檔：${PID_FILE}"
    PID="$(<"${PID_FILE}")"
fi

[[ "${PID}" =~ ^[1-9][0-9]*$ ]] || fail "PID 必須是正整數：${PID}"
[[ "${GRACE_SECONDS}" =~ ^[0-9]+$ ]] \
    || fail "GRACE_SECONDS 必須是非負整數：${GRACE_SECONDS}"

if ! kill -0 "${PID}" 2>/dev/null; then
    log "PID ${PID} 已不存在，清除舊 PID 檔"
    remove_matching_pid_file "${PID}"
    exit 0
fi

COMMAND_LINE=""
if [[ -r "/proc/${PID}/cmdline" ]]; then
    COMMAND_LINE="$(tr '\0' ' ' < "/proc/${PID}/cmdline")"
fi

if [[ "${FORCE_PID:-0}" != "1" ]]; then
    [[ -n "${COMMAND_LINE}" ]] \
        || fail "無法讀取 PID ${PID} 的命令列；若確定要停止，請設定 FORCE_PID=1。"
    [[ "${COMMAND_LINE}" == *"${EXPECTED_COMMAND_PATTERN}"* ]] \
        || fail "PID ${PID} 不是預期的 Audio IO 服務：${COMMAND_LINE}"
fi

log "傳送 SIGTERM 給 PID ${PID}"
kill -TERM "${PID}"

for ((second = 0; second < GRACE_SECONDS; second += 1)); do
    if ! kill -0 "${PID}" 2>/dev/null; then
        remove_matching_pid_file "${PID}"
        log "Audio IO 已停止"
        exit 0
    fi
    sleep 1
done

if kill -0 "${PID}" 2>/dev/null; then
    log "等待 ${GRACE_SECONDS} 秒後仍未停止，傳送 SIGKILL"
    kill -KILL "${PID}"
fi

remove_matching_pid_file "${PID}"
log "Audio IO 已停止"
