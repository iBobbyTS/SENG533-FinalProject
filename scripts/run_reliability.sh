#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-smoke}"
shift || true
source "$ROOT/scripts/activate_bench.sh"
python -m benchmark_suite.run_reliability --profile "$PROFILE" "$@"
