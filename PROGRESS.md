# PROGRESS — AMI 2027 (deadline 2026-12-10)

## Current status
- Phase: **1 – Preparation** (24–25/11)
- GPU machine set up end-to-end (2026-11-24). Dataset analysis + leakage check done; 50 review panels generated (sheet not yet filled).
- **Gate 0 passed (24/11):** ChangeFormer-b0 ~0.5 GPU-h, SNUNet-c16 ~1.1 GPU-h per 40k run → both backbones kept.
- Next gate: **Gate 1 (29/11)**.

## Next actions
1. New machine: `cp .env.example .env` (fill HF_TOKEN) → `bash setup.sh`.
2. Full baselines (1 seed, 40k, run inside tmux):
   `bash scripts/train_opencd.sh snunet_c16_taktpixel baseline 0` and `bash scripts/train_opencd.sh changeformer_b0_taktpixel baseline 0`
   → `opencd_predict.py --split val` → `evaluate.py ... --exclude results/analysis/dup_with_train.csv`.
3. B0 abs-diff: `run_absdiff.py --tune --predict val` → `evaluate.py` (same flags).
4. Fill `results/review/review_sheet.csv` (panels in `outputs/review/panels/`, regenerate with `review_samples.py --n 50`).
5. Before Gate 1: re-bin thickness (stride-relative G1/G2 nearly empty, see Decision log).

## Checklist
### Phase 1 – Preparation (24–25/11)
- [x] Repo scaffold, metrics, morphology, abs-diff baseline, eval pipeline (35 tests)
- [x] setup.sh, Open-CD configs, run registry, progress tracking
- [x] Checkpoint backup/resume via Hugging Face Hub (`scripts/hf_sync.py`)
- [x] Setup on GPU machine works end-to-end
- [x] Dataset analysis (`results/analysis/`)
- [ ] Manual review of 50 pairs (panels generated, sheet to fill) (`results/review/review_sheet.csv`)
- [x] Leakage check (23 exact dups val/test↔train → `results/analysis/dup_with_train.csv`)
- [x] Gate 0: GPU-hours measured (CF 0.5 h, SNUNet 1.1 h per 40k)
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
| 2026-11-24 | Open-CD (likyoo/open-cd, pinned commit) instead of OpenCD-Lite; eval at full resolution (no resize) | `configs/opencd/*_taktpixel.py` |
| 2026-11-24 | Keep official splits (comparability with dataset paper). 23 val/test pairs are byte-identical (A,B,mask) to train pairs (val 10: 6 pinhole + 4 line; test 13: 9 pinhole + 4 line → 4/17 val and 4/18 test `line` images). Report main numbers with `--exclude dup_with_train.csv`, official-split numbers as secondary. Split is tile-level: all 82 sheet ids appear in train/val/test → note as limitation. | `results/analysis/leakage.txt`, `results/analysis/dup_with_train.csv` |
| 2026-11-24 | Data facts: all tiles 256×256; 6 classes (train 4267: color_shift 1380, black_spot 1262, hair 836, pinhole 328, friction 318, line 143). Defects are mostly thick: train comps 11 832, thickness ≤2 px: 1, ≤4 px: 3.9 %, ≤8 px: 16.8 % (median 8 px line … 16 px black_spot). Stride-4 bins G1/G2 are ~empty → a 'very thin defect' failure case is unlikely; re-bin (e.g. train quantiles) before Gate 1. color_shift = 19.5 % defect pixels, ~6 comps/img; line is rare (143 train). | `results/analysis/summary.csv`, `bins.json` |
| 2026-11-24 | **Gate 0 passed**: 300-iter smoke on RTX 3090 — ChangeFormer-b0 0.048 s/iter (~0.5 GPU-h/40k, 1.5 GB), SNUNet-c16 0.098 s/iter (~1.1 GPU-h/40k, 4.9 GB). Keep both backbones; budget is not a constraint. | `results/runs/1124-1334-changeformer_b0-smoke-s0/`, `results/runs/1124-1336-snunet_c16-smoke-s0/` |

## Session log
- 2026-11-24 (GPU #1): Gate 0 smoke tests. Fixed (a) Open-CD config: duplicate keys across multiple `_base_` → overrides inlined, `_taktpixel_data.py` removed; (b) `hf_sync.py push` skipped .pth < 30 s old, so the final push after training never uploaded the last/best ckpt → `--final` flag in train_opencd.sh. Verified HF pull restores identical ckpt; opencd_predict → evaluate works end-to-end.
- 2026-11-24 (GPU #1): setup done, dataset via HF mirror; analyze_dataset + leakage check; 50 review panels; `evaluate.py --exclude` added. Visual check of 12 panels: masks mostly tight; some `line` masks cover only part of the visible streak (possible label noise); color_shift shows misregistration ghosts.
- 2026-11-24 (GPU #1): setup fixes — (a) openmim→openxlab downgrades setuptools to 60.2 (no PEP 660) → setup.sh re-upgrades before `pip install -e open-cd`;
  (b) unpinned opencv pulled numpy 2.x (breaks torch 2.1.2) → requirements.txt caps numpy<2, opencv<4.11;
  (c) Zenodo returns 403 for GPU-host IPs → download_data.sh falls back to private HF dataset mirror `HF_DATA_REPO`.
- 2026-11-24: Added HF Hub checkpoint backup (every 10 min + final), resume via RESUME=<run_id>.
- 2026-11-24: Phase 1 code + setup/tracking scripts pushed.
