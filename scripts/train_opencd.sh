#!/usr/bin/env bash
# Train an Open-CD model and register the run in results/runs.csv.
#
#   bash scripts/train_opencd.sh <config_name> <tag> <seed> [extra --cfg-options ...]
#
# Examples:
#   # Gate 0 smoke test (300 iters) -> prints estimated GPU-hours for 40k iters
#   bash scripts/train_opencd.sh changeformer_b0_taktpixel smoke 0 train_cfg.max_iters=300 train_cfg.val_interval=300
#   # full baseline
#   bash scripts/train_opencd.sh changeformer_b0_taktpixel baseline 0
#
# Work dirs (checkpoints, logs) go to outputs/work_dirs/<run_id>  (NOT in git).
# Small artifacts (config dump, scalars, best metrics) are copied to results/runs/<run_id>/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG_NAME=$1; TAG=$2; SEED=$3; shift 3
OCD="$ROOT/third_party/open-cd"
[ -d "$OCD" ] || { echo "Run setup.sh first"; exit 1; }

# sync our configs into Open-CD so _base_ paths resolve
mkdir -p "$OCD/configs/taktpixel"
cp "$ROOT"/configs/opencd/*.py "$OCD/configs/taktpixel/"

RUN_ID=$(python "$ROOT/scripts/runs.py" start --config "$CFG_NAME" --tag "$TAG" --seed "$SEED" \
         --notes "$*")
WORK="$ROOT/outputs/work_dirs/$RUN_ID"
mkdir -p "$WORK"
echo "run_id=$RUN_ID  work_dir=$WORK"

cd "$OCD"
set +e
python tools/train.py "configs/taktpixel/${CFG_NAME}.py" --work-dir "$WORK" \
  --cfg-options randomness.seed="$SEED" "$@" 2>&1 | tee "$WORK.log"
STATUS=${PIPESTATUS[0]}
set -e

cd "$ROOT"
python scripts/collect_run.py --run-id "$RUN_ID" --work-dir "$WORK" \
  --status "$([ "$STATUS" -eq 0 ] && echo done || echo failed)"
exit "$STATUS"
