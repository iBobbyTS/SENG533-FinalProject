#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-initial}"
shift || true
source "$ROOT/scripts/activate_bench.sh"
python -m benchmark_suite.run_benchmark --benchmark vqa --profile "$PROFILE" "$@"
