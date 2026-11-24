#!/usr/bin/env bash
# Save everything small and important to git BEFORE the rented machine is released.
#
#   bash scripts/sync_results.sh "phase1: dataset analysis"
#
# Copies small analysis/eval outputs from outputs/ into results/, then commits and pushes
# results/ + PROGRESS.md. Checkpoints (*.pth) are NOT committed; set RCLONE_REMOTE
# (e.g. "gdrive:pdcd") to also upload best checkpoints with rclone.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MSG=${1:-"sync results"}

mkdir -p results
# analysis + evaluation tables (csv/json/txt only, < 20 MB each)
for d in outputs/analysis outputs/eval outputs/review outputs/preds/absdiff; do
  [ -d "$d" ] || continue
  find "$d" -type f \( -name '*.csv' -o -name '*.json' -o -name '*.txt' \) -size -20M | while read -r f; do
    dst="results/${f#outputs/}"
    mkdir -p "$(dirname "$dst")"
    cp "$f" "$dst"
  done
done

if [ -n "${RCLONE_REMOTE:-}" ] && command -v rclone >/dev/null; then
  find outputs/work_dirs -name 'best_*.pth' | while read -r c; do
    rclone copy "$c" "$RCLONE_REMOTE/$(dirname "${c#outputs/}")" --progress
  done
fi

git add results PROGRESS.md
if git diff --cached --quiet; then
  echo "nothing to commit"
else
  git commit -q -m "$MSG"
  git push -q
  echo "pushed: $MSG"
fi
