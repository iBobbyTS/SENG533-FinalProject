#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 || "$1" == --* ]]; then
  echo "Usage: $0 <instance-id> [--base-url URL]" >&2
  exit 1
fi

INSTANCE_ID="$1"
shift

source "$ROOT/scripts/activate_bench.sh"
python -m benchmark_suite.model_management unload --instance-id "$INSTANCE_ID" "$@"
