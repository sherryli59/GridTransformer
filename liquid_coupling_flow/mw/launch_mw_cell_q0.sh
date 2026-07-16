#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
LOG=${LOG:-reports/logs-2026-07-15/mw_cell_q0_train.out}
mkdir -p "$(dirname "$LOG")"
exec python -u -m liquid_coupling_flow.mw.mw_cell_train "$@" >"$LOG" 2>&1
