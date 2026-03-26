#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env.8gpu_custom_stack_best"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_gpt.py"

cd "${REPO_ROOT}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

mkdir -p logs

echo "Launching 8-GPU run with RUN_ID=${RUN_ID}"
echo "Train script: ${TRAIN_SCRIPT}"
echo "Repo root: ${REPO_ROOT}"

torchrun --standalone --nproc_per_node=8 "${TRAIN_SCRIPT}"
