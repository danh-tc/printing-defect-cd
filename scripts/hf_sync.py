#!/usr/bin/env python
"""Back up / restore training checkpoints on the Hugging Face Hub (private model repo).

Remote layout:  <HF_REPO>/runs/<run_id>/{latest iter_*.pth, best_*.pth, *.py config, train.log}

    export HF_REPO=<user>/pdcd-checkpoints      # created private on first push
    export HF_TOKEN=hf_xxx                      # or: huggingface-cli login

    python scripts/hf_sync.py push  --run-id R --work-dir outputs/work_dirs/R
    python scripts/hf_sync.py watch --run-id R --work-dir outputs/work_dirs/R --interval 600
    python scripts/hf_sync.py pull  --run-id R [--best-only]
    python scripts/hf_sync.py list

Only the newest iter_*.pth is kept remotely (older ones are deleted) plus best_*.pth,
so a run costs ~2 checkpoints of storage and can be resumed with `--resume`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ".hf_uploaded.json"


def _api():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("pip install huggingface_hub")
    repo = os.environ.get("HF_REPO")
    if not repo:
        sys.exit("Set HF_REPO=<user>/<repo> (e.g. in .env)")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    return api, repo


def _iter_num(p: Path) -> int:
    m = re.search(r"iter_(\d+)\.pth$", p.name)
    return int(m.group(1)) if m else -1


def _local_files(work: Path):
    iters = sorted(work.glob("iter_*.pth"), key=_iter_num)
    files = ([iters[-1]] if iters else []) + sorted(work.glob("best_*.pth")) + sorted(work.glob("*.py"))
    log = Path(str(work) + ".log")
    if log.exists():
        files.append(log)
    return files


def push(work: Path, run_id: str, quiet=False) -> int:
    api, repo = _api()
    state_f = work / STATE
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    prefix = f"runs/{run_id}"
    n = 0
    for f in _local_files(work):
        if not f.exists():
            continue
        st = f.stat()
        key = f.name if f.suffix != ".log" else "train.log"
        sig = f"{st.st_size}:{int(st.st_mtime)}"
        if state.get(key) == sig:
            continue
        if f.suffix == ".pth" and time.time() - st.st_mtime < 30:
            continue  # may still be being written; next round
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f"{prefix}/{key}", repo_id=repo,
                        commit_message=f"{run_id}: {key}")
        state[key] = sig
        n += 1
        if not quiet:
            print(f"[hf] uploaded {key} ({st.st_size / 1e6:.1f} MB)")
    # keep only newest iter_*.pth and newest best_*.pth remotely
    local_keep = {f.name for f in _local_files(work)}
    remote = [p for p in api.list_repo_files(repo) if p.startswith(prefix + "/")]
    for p in remote:
        name = p.rsplit("/", 1)[1]
        if name.endswith(".pth") and name not in local_keep and (name.startswith("iter_") or name.startswith("best_")):
            api.delete_file(p, repo_id=repo, commit_message=f"{run_id}: prune {name}")
            state.pop(name, None)
    state_f.write_text(json.dumps(state, indent=2))
    return n


def watch(work: Path, run_id: str, interval: int):
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    print(f"[hf] watching {work} every {interval}s")
    while not stop["flag"]:
        try:
            push(work, run_id, quiet=False)
        except Exception as e:  # network hiccups must not kill training
            print(f"[hf] push failed (will retry): {e}")
        for _ in range(interval):
            if stop["flag"]:
                break
            time.sleep(1)


def pull(run_id: str, work: Path, best_only: bool):
    from huggingface_hub import snapshot_download
    _, repo = _api()
    patterns = [f"runs/{run_id}/best_*.pth", f"runs/{run_id}/*.py"] if best_only else [f"runs/{run_id}/*"]
    snap = Path(snapshot_download(repo, repo_type="model", allow_patterns=patterns,
                                  token=os.environ.get("HF_TOKEN")))
    src = snap / "runs" / run_id
    if not src.exists():
        sys.exit(f"run {run_id} not found in {repo}")
    work.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        dst = work / f.name
        if not dst.exists():
            dst.write_bytes(f.read_bytes())
        print(f"[hf] restored {dst}")
    iters = sorted(work.glob("iter_*.pth"), key=_iter_num)
    if iters:  # mmengine --resume reads this file
        (work / "last_checkpoint").write_text(str(iters[-1].resolve()))
        print(f"[hf] last_checkpoint -> {iters[-1].name}")


def list_runs():
    api, repo = _api()
    runs = {}
    for p in api.list_repo_files(repo):
        parts = p.split("/")
        if len(parts) == 3 and parts[0] == "runs":
            runs.setdefault(parts[1], []).append(parts[2])
    for r, fs in sorted(runs.items()):
        print(f"{r:<45} {', '.join(sorted(f for f in fs if f.endswith('.pth')))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "watch", "pull", "list"])
    ap.add_argument("--run-id")
    ap.add_argument("--work-dir")
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--best-only", action="store_true")
    a = ap.parse_args()
    if a.cmd == "list":
        return list_runs()
    if not a.run_id:
        sys.exit("--run-id required")
    work = Path(a.work_dir) if a.work_dir else ROOT / "outputs" / "work_dirs" / a.run_id
    if a.cmd == "push":
        print(f"[hf] {push(work, a.run_id)} file(s) uploaded")
    elif a.cmd == "watch":
        watch(work, a.run_id, a.interval)
    else:
        pull(a.run_id, work, a.best_only)


if __name__ == "__main__":
    main()
