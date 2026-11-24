# PROGRESS — AMI 2027 (deadline 2026-12-10)

## Current status
- Phase: **1 – Preparation** (24–25/11)
- GPU machine set up end-to-end (2026-11-24). Dataset analysis + leakage check done; 50 review panels generated (sheet not yet filled).
- **Gate 0 passed (24/11):** ChangeFormer-b0 ~0.5 GPU-h, SNUNet-c16 ~1.1 GPU-h per 40k run → both backbones kept.
- Benchmark B0/B1/B2 done on val (24/11). Preliminary Gate 1 evidence below; waiting on label review.
- Next gate: **Gate 1 (29/11)**.

## Next actions
1. New machine: `cp .env.example .env` (fill HF_TOKEN) → `bash setup.sh`. Restore preds: `hf_sync.py pull --run-id <run> --best-only` → `opencd_predict.py`.
2. **User:** fill `results/review_failures/failure_sheet.csv` (50 joint misses; panels via `scripts/failure_panels.py`, legend GT red / B1 green / B2 blue / yellow box = missed comp) and `results/review/review_sheet.csv`.
3. Pilot H-aug (1 seed, 40k, ChangeFormer = cheaper): paired or no photometric aug → compare low-contrast recall vs `1124-1450-changeformer_b0-baseline-s0`.
4. Literature (3–5 papers): low-contrast / subtle defects in reference-based print inspection; augmentation consistency in change detection.
5. Decide primary component metric (original vs merged-within-3px) before G1.

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
- [x] B0 Abs-diff/ΔE tuned on val + evaluated (`results/eval/absdiff_val/`)
- [x] B1 SNUNet-CD trained (1 seed) + evaluated on val (`1124-1341-snunet_c16-baseline-s0`)
- [x] B2 ChangeFormer trained (1 seed) + evaluated on val (`1124-1450-changeformer_b0-baseline-s0`)
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
| 2026-11-24 | B0 best on val (pixel F1 grid): rgb, blur σ=2, thr=30, min_area=16 → pix F1 0.769 / IoU 0.624; comp recall@0.3 0.698, FP/img 5.75 (val, dups excluded). Weakest: line (comp recall 0.03), friction (0.45); color_shift FP/img 12.5. Best σ is at the grid edge (2.0) → optionally extend grid (3, 4) before the paper table. Note: B0 is tuned and scored on val (test later, once). | `results/eval/absdiff_val/summary.csv`, `results/preds/absdiff/grid.csv` |
| 2026-11-24 | B1 SNUNet-c16 40k (1.13 GPU-h, best = iter 40000, OpenCD mIoU 90.57): pix F1 0.905 / IoU 0.826, comp recall@0.3 0.833, FP/img 1.62 (B0: 0.769 / 0.624 / 0.698 / 5.75). Weak: line (comp recall 0.345, n=29), friction (0.73, FP/img 3.2), thin+low-contrast (G2 low 0.47, G3 low 0.47, n=34/103), small area (0.68). | `results/eval/1124-1341-snunet_c16-baseline-s0_val/` |
| 2026-11-24 | B2 ChangeFormer-b0 40k (0.56 GPU-h, best iter 40000, OpenCD mIoU 89.79): pix F1 0.896 / IoU 0.812, comp recall@0.3 0.845, FP/img 1.63. **B1 ≈ B2** overall → no architecture-specific weakness. | `results/eval/1124-1450-changeformer_b0-baseline-s0_val/` |
| 2026-11-24 | **Prelim. G1 evidence — common weakness = low contrast (ΔE), not thinness.** Low- vs high-contrast comp recall B1 0.71/0.93, B2 0.75/0.92 (n 597/808). Holds within categories: color_shift low 0.62/0.68 vs high 0.91/0.90; line (all low, n 26) 0.38/0.46; friction low 0.72/0.71. Thin T1/T2 low-contrast *excluding* color_shift: 0.84/0.85 → thickness alone is not the issue (re-binned with train quartiles 10/14.4/20.6 px). 162 GT comps missed by both DL models (11.5 % of comps, 4.6 % of GT px): median ΔE 16.9 vs 33.5 for comps both hit. | `results/compare/baselines_val/` |
| 2026-11-24 | Component metric is fragmentation-sensitive: merging comps within ~3 px (7×7 dilation for labelling) halves FP/img (B1 1.62→0.77, B2 1.63→1.01), line recall B1 0.34→0.45, B2 0.41→0.59. Report a merged-component variant as robustness check (decide primary metric at G1). | inline analysis 2026-11-24 (re-run to add as script if adopted) |
| 2026-11-24 | **Hypothesis H-aug:** Open-CD `MultiImgPhotoMetricDistortion` (brightness ±10, contrast/saturation 0.8–1.2, hue ±10) is applied *independently* to A and B. On 60 train pairs × 5 draws it shifts background ΔE from ~0.6 to median 4.9 / P75 9.7 / P90 14.7 — the same range as line (12.0) and friction (12.8) defects, labelled 'no change'. Likely teaches the model to ignore low-ΔE differences. Cheap pilot: same 40k run with paired (identical) or no photometric aug. | `opencd/datasets/transforms/transforms.py` L682-716 |
| 2026-11-24 | Visual check (12 joint-miss panels): line GT = chain of fragments along a continuous faint streak; several friction GT blobs sit on halftone areas with barely visible defect; color_shift defect extent within misregistration is ambiguous → label-noise risk must be checked before G1 (`results/review_failures/failure_sheet.csv`). | `outputs/review_failures/panels/` |

## Session log
- 2026-11-24 (GPU #1): B1/B2 40k baselines in tmux (1.13 + 0.56 GPU-h), val eval, `compare_models.py` + `failure_panels.py`, preliminary G1 evidence (low contrast), H-aug found.
- 2026-11-24 (GPU #1): Gate 0 smoke tests. Fixed (a) Open-CD config: duplicate keys across multiple `_base_` → overrides inlined, `_taktpixel_data.py` removed; (b) `hf_sync.py push` skipped .pth < 30 s old, so the final push after training never uploaded the last/best ckpt → `--final` flag in train_opencd.sh. Verified HF pull restores identical ckpt; opencd_predict → evaluate works end-to-end.
- 2026-11-24 (GPU #1): setup done, dataset via HF mirror; analyze_dataset + leakage check; 50 review panels; `evaluate.py --exclude` added. Visual check of 12 panels: masks mostly tight; some `line` masks cover only part of the visible streak (possible label noise); color_shift shows misregistration ghosts.
- 2026-11-24 (GPU #1): setup fixes — (a) openmim→openxlab downgrades setuptools to 60.2 (no PEP 660) → setup.sh re-upgrades before `pip install -e open-cd`;
  (b) unpinned opencv pulled numpy 2.x (breaks torch 2.1.2) → requirements.txt caps numpy<2, opencv<4.11;
  (c) Zenodo returns 403 for GPU-host IPs → download_data.sh falls back to private HF dataset mirror `HF_DATA_REPO`.
- 2026-11-24: Added HF Hub checkpoint backup (every 10 min + final), resume via RESUME=<run_id>.
- 2026-11-24: Phase 1 code + setup/tracking scripts pushed.
