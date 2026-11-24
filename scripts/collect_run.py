#!/usr/bin/env python
"""After a run: copy small artifacts into results/runs/<run_id>/, estimate GPU-hours,
record best val metrics in results/runs.csv.

mmengine writes <work_dir>/<timestamp>/vis_data/scalars.json (one JSON per line).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_scalars(work: Path):
    files = sorted(work.glob("*/vis_data/scalars.json"))
    rows = []
    for f in files:
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return rows, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--status", default="done")
    ap.add_argument("--full-iters", type=int, default=40000, help="for GPU-hour extrapolation")
    a = ap.parse_args()
    work = Path(a.work_dir)
    dst = ROOT / "results" / "runs" / a.run_id
    dst.mkdir(parents=True, exist_ok=True)

    rows, files = load_scalars(work)
    train = [r for r in rows if "time" in r and "loss" in r]
    val = [r for r in rows if "mIoU" in r]
    sec_per_iter = sum(r["time"] for r in train[2:]) / max(len(train[2:]), 1) if train else None
    last_iter = max((r.get("iter", r.get("step", 0)) for r in train), default=0)
    gpu_hours = sec_per_iter * last_iter / 3600 if sec_per_iter else None
    best = max(val, key=lambda r: r["mIoU"]) if val else {}

    summary = {
        "run_id": a.run_id, "status": a.status, "last_iter": last_iter,
        "sec_per_iter": sec_per_iter, "gpu_hours": gpu_hours,
        "est_gpu_hours_full": sec_per_iter * a.full_iters / 3600 if sec_per_iter else None,
        "best_val": best, "max_memory_mb": max((r.get("memory", 0) for r in train), default=None),
    }
    (dst / "summary.json").write_text(json.dumps(summary, indent=2))
    for f in files:
        shutil.copy(f, dst / "scalars.json")
    for cfg in work.glob("*.py"):
        shutil.copy(cfg, dst / cfg.name)
    log = Path(str(work) + ".log")
    if log.exists():  # keep only the tail; full log stays in outputs/
        (dst / "train_tail.log").write_text("\n".join(log.read_text(errors="ignore").splitlines()[-200:]))

    metrics = {k: best[k] for k in ("mIoU", "mFscore", "step") if k in best}
    cmd = [sys.executable, str(ROOT / "scripts/runs.py"), "finish", a.run_id, "--status", a.status,
           "--metrics", json.dumps(metrics)]
    if gpu_hours is not None:
        cmd += ["--gpu-hours", str(gpu_hours)]
    subprocess.check_call(cmd)

    if sec_per_iter:
        print(f"\n[Gate 0] {sec_per_iter:.3f} s/iter -> ~{summary['est_gpu_hours_full']:.1f} GPU-h "
              f"for {a.full_iters} iters (peak mem {summary['max_memory_mb']} MB)")
    print(f"Saved run artifacts to {dst}. Commit them with: bash scripts/sync_results.sh \"<msg>\"")


if __name__ == "__main__":
    main()
