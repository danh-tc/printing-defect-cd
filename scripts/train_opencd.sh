#!/usr/bin/env bash
# Train an Open-CD model, register the run in results/runs.csv and back up
# checkpoints to the Hugging Face Hub while training (if HF_REPO is set).
#
#   bash scripts/train_opencd.sh <config_name> <tag> <seed> [extra cfg-options ...]
#   RESUME=<run_id> bash scripts/train_opencd.sh <config_name> <tag> <seed>   # resume (pulls from HF if needed)
#
# Examples:
#   bash scripts/train_opencd.sh changeformer_b0_taktpixel smoke 0 train_cfg.max_iters=300 train_cfg.val_interval=300
#   bash scripts/train_opencd.sh changeformer_b0_taktpixel baseline 0
#
# Run long jobs inside tmux so an SSH/VS Code disconnect does not kill them.
# Work dirs go to outputs/work_dirs/<run_id> (not in git); small artifacts to results/runs/<run_id>/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG_NAME=$1; TAG=$2; SEED=$3; shift 3
OCD="$ROOT/third_party/open-cd"
[ -d "$OCD" ] || { echo "Run setup.sh first"; exit 1; }
# shellcheck disable=SC1091
[ -f "$ROOT/.env" ] && set -a && source "$ROOT/.env" && set +a
HF_INTERVAL=${HF_INTERVAL:-600}

mkdir -p "$OCD/configs/taktpixel"
cp "$ROOT"/configs/opencd/*.py "$OCD/configs/taktpixel/"

RESUME_ARGS=()
if [ -n "${RESUME:-}" ]; then
  RUN_ID=$RESUME
  WORK="$ROOT/outputs/work_dirs/$RUN_ID"
  if [ ! -f "$WORK/last_checkpoint" ] && [ -n "${HF_REPO:-}" ]; then
    python "$ROOT/scripts/hf_sync.py" pull --run-id "$RUN_ID" --work-dir "$WORK"
  fi
  [ -f "$WORK/last_checkpoint" ] || { echo "No checkpoint to resume in $WORK"; exit 1; }
  python "$ROOT/scripts/runs.py" reopen "$RUN_ID"
  RESUME_ARGS=(--resume)
else
  RUN_ID=$(python "$ROOT/scripts/runs.py" start --config "$CFG_NAME" --tag "$TAG" --seed "$SEED" --notes "$*")
  WORK="$ROOT/outputs/work_dirs/$RUN_ID"
fi
mkdir -p "$WORK"
echo "run_id=$RUN_ID  work_dir=$WORK"

WATCH_PID=""
if [ -n "${HF_REPO:-}" ]; then
  python "$ROOT/scripts/hf_sync.py" watch --run-id "$RUN_ID" --work-dir "$WORK" \
    --interval "$HF_INTERVAL" >> "$WORK.hf.log" 2>&1 &
  WATCH_PID=$!
  trap '[ -n "$WATCH_PID" ] && kill "$WATCH_PID" 2>/dev/null || true' EXIT
  echo "HF backup -> $HF_REPO/runs/$RUN_ID every ${HF_INTERVAL}s (log: $WORK.hf.log)"
else
  echo "WARNING: HF_REPO not set — checkpoints exist only on this machine."
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"  # for custom_imports (pdcd.opencd_ext)
cd "$OCD"
set +e
python tools/train.py "configs/taktpixel/${CFG_NAME}.py" --work-dir "$WORK" "${RESUME_ARGS[@]}" \
  --cfg-options randomness.seed="$SEED" "$@" 2>&1 | tee -a "$WORK.log"
STATUS=${PIPESTATUS[0]}
set -e
cd "$ROOT"

if [ -n "$WATCH_PID" ]; then
  kill "$WATCH_PID" 2>/dev/null || true; wait "$WATCH_PID" 2>/dev/null || true; WATCH_PID=""
  python scripts/hf_sync.py push --final --run-id "$RUN_ID" --work-dir "$WORK" || echo "final HF push failed; retry: python scripts/hf_sync.py push --run-id $RUN_ID"
fi
python scripts/collect_run.py --run-id "$RUN_ID" --work-dir "$WORK" \
  --status "$([ "$STATUS" -eq 0 ] && echo done || echo failed)"
exit "$STATUS"
