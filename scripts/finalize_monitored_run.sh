#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/activate_bench.sh"
python -m benchmark_suite.finalize_monitored_run "$@"
