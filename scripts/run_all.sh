#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_static.sh
bash scripts/run_dynamic.sh

STATIC_DIR=$(cat outputs/last_run_static.txt)
DYNAMIC_DIR=$(cat outputs/last_run_dynamic.txt)

python -m analysis.analyze_metrics \
  --run_dir "$STATIC_DIR" \
  --compare_dir "$DYNAMIC_DIR"
