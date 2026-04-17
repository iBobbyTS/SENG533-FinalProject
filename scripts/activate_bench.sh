#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT/.venv-bench"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Missing virtual environment at $VENV_DIR. Run scripts/setup_all.sh first." >&2
  exit 1
fi

export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$ROOT/cache/huggingface/hub"
export HF_DATASETS_CACHE="$ROOT/cache/huggingface/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
source "$VENV_DIR/bin/activate"
