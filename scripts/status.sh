#!/usr/bin/env bash
# Quick "where are we" summary for a new session (human or Claude).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "== Today: $(date '+%Y-%m-%d %H:%M')   deadline AMI 2027: 2026-12-10 =="
echo; echo "== PROGRESS.md: current status =="
sed -n '/^## Current status/,/^## Checklist/p' PROGRESS.md | sed '$d'
echo "== Next actions =="
sed -n '/^## Next actions/,/^## /p' PROGRESS.md | sed '1d;$d'
echo; echo "== Runs (results/runs.csv) =="
python scripts/runs.py list 2>/dev/null || echo "(python env not active)"
echo; echo "== Checkpoints on HF =="
[ -f .env ] && set -a && . ./.env && set +a
if [ -n "${HF_REPO:-}" ]; then python scripts/hf_sync.py list 2>/dev/null || echo "(cannot reach HF)"; else echo "HF_REPO not set (.env missing)"; fi
echo; echo "== Git =="
git log --oneline -5; git status --short | head -20
echo; echo "== Environment =="
[ -d .venv ] && echo "venv: ok" || echo "venv: MISSING -> bash setup.sh"
[ -d third_party/open-cd ] && echo "open-cd: ok" || echo "open-cd: MISSING -> bash setup.sh"
[ -f .data_root ] && echo "data: $(cat .data_root)" || echo "data: MISSING -> bash setup.sh"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
