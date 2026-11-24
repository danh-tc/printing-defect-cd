# printing-defect-cd

Baseline-driven research for industrial printing defect segmentation using
reference-based change detection on **Taktpixel2025PD-CD** (target: AMI 2027,
deadline 10/12/2026).

Strategy: **Benchmark → Failure analysis → Focused literature review →
One targeted improvement → Final evaluation.** The method is not fixed in
advance; it is chosen at Gate 1 from validation-set evidence.

## Plan (24/11 → 10/12/2026)

| Phase | Dates | GPU | Research / code | Gate |
|---|---|---|---|---|
| 1 Prep | 24–25/11 | ChangeFormer smoke test | Env, data, review 50 pairs, leakage check, dataset stats | **G0 (25/11)**: hours/run ≤ ~10–12 h (1 GPU) or ≤ ~20 h (2 GPU), else 1 backbone |
| 2 Benchmark | 26–29/11 | SNUNet-CD, ChangeFormer (1 seed) | Abs-diff/ΔE baseline, eval; write Intro + Dataset; **27–28: implement 2 candidate fixes**; **28–29: read 3–5 papers**; review 30–50 failures | **G1 (29/11)**: clear weakness on both DL models, enough samples, not label noise, fix not already published, pilot shows signal |
| 3 Improve | 30/11–06/12 | Improved seed 1 → +2 seeds baseline & improved, ablation (1 seed) | Method, Related Work | **G2 (02/12)**: target group improves, no big drop overall / FP rise |
| 4 Submit | 07–10/12 | Test set **once** | Results, review, CCIS format | — |

Budget: 11 core runs + 4 reserve. Test split is never used for selection.

## Layout

```
src/pdcd/
  data.py          dataset index (A/B/OUT), category from filename prefix
  morphology.py    components: area, thickness, skeleton length, mean ΔE (CIE76)
  metrics.py       pixel metrics, one-to-one component matching (IoU 0.1/0.3/0.5), FP/image
  analysis.py      train-fitted bins (thickness/stride, area, contrast), recall tables
  baselines/absdiff.py   B0: RGB / ΔE difference + threshold
scripts/
  analyze_dataset.py   Phase 1 stats, leakage report, bins.json
  review_samples.py    panels + sheet for manual annotation review
  run_absdiff.py       tune B0 on val, write predictions
  evaluate.py          evaluate any prediction folder (pixel, component, stratified)
  prepare_opencd.py    LEVIR-CD-style layout (A/B/label 0-255) for Open-CD / OpenCD-Lite
  make_synthetic.py    tiny fake dataset for smoke tests
tests/                 pytest: thickness estimator + metrics
```

## Setup on a rented GPU (Ubuntu + RTX 3090)

```bash
git clone https://github.com/danh-tc/printing-defect-cd && cd printing-defect-cd
cp .env.example .env && nano .env   # HF_TOKEN (write) + HF_REPO
bash setup.sh                    # venv, torch 2.1.2+cu121, mmcv 2.1.0, Open-CD, dataset, tests
source .venv/bin/activate
bash scripts/status.sh           # where we left off
```

`setup.sh` is idempotent. Use `CUDA_VARIANT=cu118` if the driver is older than 530.

## Session workflow (machine is wiped between rentals)

- **Memory lives in git:** `PROGRESS.md` (status, decisions, next actions),
  `results/runs.csv` (every training run), `results/` (small CSV/JSON outputs).
  `CLAUDE.md` tells a fresh Claude session to read these first.
- Train via `bash scripts/train_opencd.sh <config> <tag> <seed> [cfg-options]`. It registers the run,
  copies logs/metrics to `results/runs/<run_id>/` and prints estimated GPU-hours.
- Predictions for evaluation: `python scripts/opencd_predict.py --config <cfg> --ckpt <pth> --split val --out ...`
- **Checkpoints → Hugging Face Hub** (private repo `HF_REPO`): `train_opencd.sh` uploads the newest
  `iter_*.pth` + `best_*.pth` every 10 min (`HF_INTERVAL`) and at the end; older iters are pruned.
  - Interrupted run: `RESUME=<run_id> bash scripts/train_opencd.sh <cfg> <tag> <seed>` (pulls from HF, `--resume`).
  - Predict on a new machine: `python scripts/hf_sync.py pull --run-id <run_id> --best-only`.
  - `python scripts/hf_sync.py list` shows what is backed up.
- **Before releasing the machine:** `bash scripts/sync_results.sh "<msg>"` (git push results + HF push checkpoints).

## Phase 1 commands

```bash
D=data/Taktpixel2025PD-CD
python scripts/analyze_dataset.py --data $D --out outputs/analysis          # stats + bins.json
python scripts/review_samples.py  --data $D --split train --n 50            # fill review_sheet.csv
python scripts/run_absdiff.py     --data $D --out outputs/preds/absdiff --tune --predict val
python scripts/evaluate.py        --data $D --split val --pred outputs/preds/absdiff/val \
                                  --bins outputs/analysis/bins.json --out outputs/eval/absdiff_val
python scripts/prepare_opencd.py  --data $D --out data/opencd_taktpixel     # for DL baselines
```

Evaluating deep models: export their val predictions as PNG masks with the
same file names as GT, then run `evaluate.py` on that folder. Test requires `--final`.

## Definitions

- **Thickness** `T = max(1, P90 over skeleton of EDT)` computed on a 2× nearest-upsampled
  component. Exact for axis-aligned lines 1–8 px; for compact blobs it is the inscribed
  diameter (~20% below nominal diameter of a digital disk). See `tests/test_morphology.py`.
- **Contrast** = mean CIE76 ΔE between master and target inside the GT component.
- **Thickness groups** (stride s=4): G1 T/s≤0.5, G2 ≤1, G3 ≤2, G4 >2.
  Area and contrast bins are fitted on train (`bins.json`) and reused for val/test.
- **Component matching**: Hungarian one-to-one on IoU; match if IoU ≥ thr.
  Category metrics are image-level (category comes from the file name).
