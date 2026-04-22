#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_DIR}/setup_r_wsl.log") 2>&1

echo "[INFO] setup_r_wsl.sh started at $(date -Iseconds)"
echo "[INFO] uname: $(uname -a)"

if ! grep -qiE "(microsoft|wsl)" /proc/version; then
  echo "[WARN] This does not look like WSL. Continuing with Linux setup anyway."
fi

bash "${PROJECT_ROOT}/scripts/setup_r_linux.sh"

echo "[INFO] setup_r_wsl.sh completed at $(date -Iseconds)"
