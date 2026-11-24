# CLAUDE.md — instructions for Claude sessions on this repo

The GPU machine is rented by the hour and wiped between rentals; the Claude extension
is reinstalled each time. **All memory of past work lives in this repo.**

User is in GMT+7 (Vietnam) and writes in Vietnamese: report times/ETAs in GMT+7.
The rented machine clock is UTC, so run_ids and log timestamps are UTC.

## At the start of every session
1. Read `PROGRESS.md` (current phase, decisions, next actions) and run `bash scripts/status.sh`.
2. If `.env` is missing, ask the user for HF_TOKEN / HF_REPO (see `.env.example`).
3. If `.venv` / `third_party/open-cd` / `.data_root` is missing → `bash setup.sh` (≈10–20 min).
4. `source .venv/bin/activate` before any python.
5. Continue from **Next actions** in `PROGRESS.md`; don't redo finished items.

## While working
- After finishing any task: update `PROGRESS.md` (tick checklist, add a dated line to
  *Session log*, rewrite *Next actions*), then `bash scripts/sync_results.sh "<msg>"`.
- Record every decision (gate outcomes, chosen failure case, chosen method) in
  *Decision log* with the evidence (file path in `results/`).
- Before the machine is released: always run `sync_results.sh` (git + HF push).
- Checkpoints live on the Hugging Face Hub (`HF_REPO` in `.env`, private), backed up
  every 10 min during training. Restore: `python scripts/hf_sync.py pull --run-id R [--best-only]`.
  Resume an interrupted run: `RESUME=<run_id> bash scripts/train_opencd.sh <cfg> <tag> <seed>`.
- `.env` is not in git; recreate it from `.env.example` on each new machine.

## Research rules (do not break)
- **Test split is used once, at the end, on frozen models** (`--final`). All selection,
  tuning, gate decisions use **val**.
- Train runs go through `scripts/train_opencd.sh` so they are logged in `results/runs.csv`.
- Same training budget for the original and improved model; 3 seeds only for the final
  comparison; benchmark and ablation = 1 seed.
- Scope: 1 dataset, 1 failure case, 1 improvement. Don't add methods unless the main
  research question needs them.

## Layout
- `src/pdcd/` library (data, morphology, metrics, analysis, baselines)
- `scripts/` CLI tools; `configs/opencd/` Open-CD configs (copied into Open-CD at train time)
- `results/` small artifacts tracked in git; `outputs/`, `data/`, `third_party/`, `.venv/` are not
- Env: Ubuntu, RTX 3090, Python 3.10, torch 2.1.2+cu121, mmcv 2.1.0, mmseg 1.2.2, Open-CD pinned in `setup.sh`
