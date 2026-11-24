#!/usr/bin/env bash
# Save everything small and important to git BEFORE the rented machine is released.
#
#   bash scripts/sync_results.sh "phase1: dataset analysis"
#
# Copies small analysis/eval outputs from outputs/ into results/, then commits and pushes
# results/ + PROGRESS.md. Checkpoints (*.pth) are not committed to git; they are pushed
# to the Hugging Face Hub when HF_REPO is set (in .env).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MSG=${1:-"sync results"}

mkdir -p results
# analysis + evaluation tables (csv/json/txt only, < 20 MB each)
for d in outputs/analysis outputs/eval outputs/review outputs/review_failures outputs/compare outputs/preds/absdiff; do
  [ -d "$d" ] || continue
  find "$d" -type f \( -name '*.csv' -o -name '*.json' -o -name '*.txt' \) -size -20M | while read -r f; do
    dst="results/${f#outputs/}"
    mkdir -p "$(dirname "$dst")"
    cp "$f" "$dst"
  done
done

# checkpoints -> Hugging Face Hub (private repo); see scripts/hf_sync.py
[ -f .env ] && set -a && source .env && set +a
if [ -n "${HF_REPO:-}" ] && [ -d outputs/work_dirs ]; then
  for w in outputs/work_dirs/*/; do
    [ -d "$w" ] || continue
    python scripts/hf_sync.py push --run-id "$(basename "$w")" --work-dir "$w" || echo "HF push failed for $w"
  done
else
  echo "WARNING: HF_REPO not set — checkpoints are NOT backed up."
fi

git add results PROGRESS.md
if git diff --cached --quiet; then
  echo "nothing to commit"
else
  git commit -q -m "$MSG"
  git push -q
  echo "pushed: $MSG"
fi
