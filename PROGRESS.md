# PROGRESS — AMI 2027 (deadline 2026-12-10)

## Current status
- Phase: **1 – Preparation** (24–25/11)
- Code for Phase 1 written and unit-tested on synthetic data; not yet run on the real dataset.
- Next gate: **Gate 0 (25/11)** — GPU-hours per 40k-iter run.

## Next actions
1. On the GPU machine: `cp .env.example .env` (fill HF_TOKEN, HF_REPO) → `bash setup.sh` → `bash scripts/status.sh`.
2. `python scripts/analyze_dataset.py --data $(cat .data_root) --out outputs/analysis`
   → check `leakage.txt`, `summary.csv` (multi-component images, thickness ≤ 4 px share).
3. `python scripts/review_samples.py --data $(cat .data_root) --n 50` → fill `review_sheet.csv`.
4. Gate 0 smoke test:
   `bash scripts/train_opencd.sh changeformer_b0_taktpixel smoke 0 train_cfg.max_iters=300 train_cfg.val_interval=300`
   (repeat for `snunet_c16_taktpixel`) → note est. GPU-h in Decision log.
5. `bash scripts/sync_results.sh "phase1: analysis + gate0"`.

## Checklist
### Phase 1 – Preparation (24–25/11)
- [x] Repo scaffold, metrics, morphology, abs-diff baseline, eval pipeline (35 tests)
- [x] setup.sh, Open-CD configs, run registry, progress tracking
- [x] Checkpoint backup/resume via Hugging Face Hub (`scripts/hf_sync.py`)
- [ ] Setup on GPU machine works end-to-end
- [ ] Dataset analysis (`results/analysis/`)
- [ ] Manual review of 50 pairs (`results/review/review_sheet.csv`)
- [ ] Leakage check
- [ ] Gate 0: GPU-hours measured
### Phase 2 – Benchmark & failure analysis (26–29/11)
- [ ] B0 Abs-diff/ΔE tuned on val + evaluated
- [ ] B1 SNUNet-CD trained (1 seed) + evaluated on val
- [ ] B2 ChangeFormer trained (1 seed) + evaluated on val
- [ ] Candidate fixes implemented (ΔE input channel; Skeleton Recall / Blob loss)
- [ ] 3–5 related papers for the observed failure case
- [ ] Review 30–50 failure cases
- [ ] Pilot of candidate fix
- [ ] Gate 1 decision
### Phase 3 – Improvement (30/11–06/12)
- [ ] Improved model seed 1 → Gate 2 (02/12)
- [ ] Seeds 2–3 for baseline and improved
- [ ] Ablation (1 seed)
### Phase 4 – Submission (07–10/12)
- [ ] Test set once (`--final`)
- [ ] Paper draft, review, CCIS format, submit

## Decision log
| Date | Decision | Evidence |
|---|---|---|
| 2026-11-24 | Baseline-driven direction; baselines = Abs-diff/ΔE, SNUNet-CD, ChangeFormer (ChangerEx optional) | proposal review rounds |
| 2026-11-24 | Thickness = P90 of EDT on skeleton of 2× upsampled component (old 2·median−1 under-estimated even widths by 1 px) | `tests/test_morphology.py` |
| 2026-11-24 | Open-CD (likyoo/open-cd, pinned commit) instead of OpenCD-Lite; eval at full resolution (no resize) | `configs/opencd/_taktpixel_data.py` |

## Session log
- 2026-11-24: Added HF Hub checkpoint backup (every 10 min + final), resume via RESUME=<run_id>.
- 2026-11-24: Phase 1 code + setup/tracking scripts pushed.
