#!/usr/bin/env python
"""Run registry kept in git: results/runs.csv.

    python scripts/runs.py start --config changeformer_b0_taktpixel --tag baseline --seed 0
    python scripts/runs.py finish <run_id> --status done --metrics '{"mIoU": 80.1}' --gpu-hours 3.2
    python scripts/runs.py list
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import socket
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "results" / "runs.csv"
FIELDS = ["run_id", "config", "tag", "seed", "status", "started", "finished",
          "gpu_hours", "host", "git_commit", "metrics", "notes"]


def _read():
    if not REG.exists():
        return []
    with REG.open() as f:
        return list(csv.DictReader(f))


def _write(rows):
    REG.parent.mkdir(parents=True, exist_ok=True)
    with REG.open("w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


def _now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def _commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return ""


def start(a):
    rows = _read()
    run_id = f"{dt.datetime.now():%m%d-%H%M}-{a.config.split('_taktpixel')[0]}-{a.tag}-s{a.seed}"
    rows.append({"run_id": run_id, "config": a.config, "tag": a.tag, "seed": a.seed,
                 "status": "running", "started": _now(), "finished": "", "gpu_hours": "",
                 "host": socket.gethostname(), "git_commit": _commit(), "metrics": "",
                 "notes": a.notes})
    _write(rows)
    print(run_id)


def finish(a):
    rows = _read()
    for r in rows:
        if r["run_id"] == a.run_id:
            r.update(status=a.status, finished=_now())
            if a.metrics:
                r["metrics"] = json.dumps(json.loads(a.metrics), separators=(",", ":"))
            if a.gpu_hours is not None:
                r["gpu_hours"] = f"{a.gpu_hours:.2f}"
            break
    else:
        raise SystemExit(f"unknown run_id {a.run_id}")
    _write(rows)


def show(_):
    rows = _read()
    if not rows:
        print("(no runs yet)"); return
    for r in rows:
        print(f"{r['run_id']:<45} {r['status']:<8} gpu_h={r['gpu_hours'] or '-':<6} {r['metrics'][:80]}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(required=True)
    s = sp.add_parser("start"); s.set_defaults(fn=start)
    s.add_argument("--config", required=True); s.add_argument("--tag", required=True)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--notes", default="")
    f = sp.add_parser("finish"); f.set_defaults(fn=finish)
    f.add_argument("run_id"); f.add_argument("--status", default="done")
    f.add_argument("--metrics", default=""); f.add_argument("--gpu-hours", type=float)
    l = sp.add_parser("list"); l.set_defaults(fn=show)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
