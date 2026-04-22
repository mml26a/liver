#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_DIR}/setup_r_macos.log") 2>&1

echo "[INFO] setup_r_macos.sh started at $(date -Iseconds)"
echo "[INFO] project root: ${PROJECT_ROOT}"

if ! command -v brew >/dev/null 2>&1; then
  echo "[ERROR] Homebrew is required on macOS. Install from https://brew.sh first."
  exit 1
fi

brew update
brew install r gdal geos proj udunits pkg-config || true

if ! command -v Rscript >/dev/null 2>&1; then
  echo "[ERROR] Rscript is not available after brew install."
  exit 1
fi

Rscript "${PROJECT_ROOT}/scripts/install_r_packages.R"
Rscript "${PROJECT_ROOT}/scripts/check_environment.R"

echo "[INFO] setup_r_macos.sh completed at $(date -Iseconds)"
