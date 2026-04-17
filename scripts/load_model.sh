#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || "$1" == --* ]]; then
  echo "Usage: $0 <model> [--base-url URL] [--context-length N]" >&2
  exit 1
fi

MODEL="$1"
shift

source "$ROOT/scripts/activate_bench.sh"
python -m benchmark_suite.model_management load --model "$MODEL" "$@"
