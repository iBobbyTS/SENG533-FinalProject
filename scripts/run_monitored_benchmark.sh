#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <benchmark> [profile] [benchmark args...]" >&2
  exit 1
fi

BENCHMARK="$1"
shift
PROFILE="probe"
if [[ $# -gt 0 && "$1" != --* ]]; then
  PROFILE="$1"
  shift
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS_ROOT="$ROOT/results"
MONITORED_ROOT="$RESULTS_ROOT/$RUN_ID/monitored"

source "$ROOT/scripts/activate_bench.sh"
mkdir -p "$MONITORED_ROOT"

python -m benchmark_suite.run_monitored_benchmark \
  --benchmark "$BENCHMARK" \
  --profile "$PROFILE" \
  --results-root "$RESULTS_ROOT" \
  --run-id "$RUN_ID" \
  --meta-out "$MONITORED_ROOT/$BENCHMARK.meta.json" \
  --memory-out "$MONITORED_ROOT/$BENCHMARK.memory.json" \
  --power-out "$MONITORED_ROOT/power.jsonl" \
  --row-out "$MONITORED_ROOT/$BENCHMARK.row.json" \
  "$@"
