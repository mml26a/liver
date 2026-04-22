#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${LOG_DIR}/setup_r_linux.log") 2>&1

echo "[INFO] setup_r_linux.sh started at $(date -Iseconds)"
echo "[INFO] project root: ${PROJECT_ROOT}"

SUDO=""
if [[ "${EUID}" -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "[ERROR] This script requires root or sudo privileges."
    exit 1
  fi
fi

if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  ${SUDO} apt-get update
  ${SUDO} apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff-dev \
    libjpeg-dev \
    libcairo2-dev \
    libudunits2-dev \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libsqlite3-dev \
    make \
    g++ \
    pkg-config \
    cmake \
    git \
    curl \
    ca-certificates
else
  echo "[ERROR] Unsupported Linux package manager. Please add support in scripts/setup_r_linux.sh."
  exit 1
fi

if ! command -v Rscript >/dev/null 2>&1; then
  echo "[ERROR] Rscript is still not available after installation."
  exit 1
fi

Rscript "${PROJECT_ROOT}/scripts/install_r_packages.R"
Rscript "${PROJECT_ROOT}/scripts/check_environment.R"

echo "[INFO] setup_r_linux.sh completed successfully at $(date -Iseconds)"
