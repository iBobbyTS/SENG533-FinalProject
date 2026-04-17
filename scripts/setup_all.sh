#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/setup_common.sh"
python -m benchmark_suite.setup_external --bench all --root "$ROOT"
