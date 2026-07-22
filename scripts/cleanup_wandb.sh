#!/usr/bin/env bash
# Delete wandb runs from the local wandb server.
# Usage:
#   bash scripts/cleanup_wandb.sh           # defaults to --state crashed
#   bash scripts/cleanup_wandb.sh --all     # delete all runs
#   bash scripts/cleanup_wandb.sh failed    # delete only failed runs
#   bash scripts/cleanup_wandb.sh crashed   # delete only crashed runs
set -euo pipefail

WANDB_BASE_URL="${WANDB_BASE_URL:-http://127.0.0.1:9090}"
WANDB_API_KEY="${WANDB_API_KEY:-local-cbfc7c82164f68aa14bcadfb7efafcee35f99dbf}"
WANDB_PROJECT="${WANDB_PROJECT:-polar-swegym-grpo}"

STATE="${1:-crashed}"

export WANDB_BASE_URL WANDB_API_KEY

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ "$STATE" = "--all" ]; then
    echo "Deleting ALL runs from project: ${WANDB_PROJECT}"
    "${PYTHON_BIN}" - << PYEOF
import os
os.environ["WANDB_BASE_URL"] = "${WANDB_BASE_URL}"
os.environ["WANDB_API_KEY"] = "${WANDB_API_KEY}"
import wandb
api = wandb.Api()
runs = list(api.runs("${WANDB_PROJECT}"))
print(f"Found {len(runs)} runs")
for r in runs:
    print(f"  Deleting {r.id} ({r.state})")
    r.delete()
print("Done.")
PYEOF
else
    echo "Deleting ${STATE} runs from project: ${WANDB_PROJECT}"
    "${PYTHON_BIN}" - << PYEOF
import os
os.environ["WANDB_BASE_URL"] = "${WANDB_BASE_URL}"
os.environ["WANDB_API_KEY"] = "${WANDB_API_KEY}"
import wandb
api = wandb.Api()
runs = [r for r in api.runs("${WANDB_PROJECT}") if r.state == "${STATE}"]
print(f"Found {len(runs)} ${STATE} runs")
for r in runs:
    print(f"  Deleting {r.id}")
    r.delete()
print("Done.")
PYEOF
fi
