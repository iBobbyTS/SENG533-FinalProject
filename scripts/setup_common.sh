#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT/.venv-bench"
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$ROOT/cache/huggingface/hub"
export HF_DATASETS_CACHE="$ROOT/cache/huggingface/datasets"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT/requirements-bench.txt"
