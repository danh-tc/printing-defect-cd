#!/usr/bin/env bash
# One-shot, idempotent setup for a fresh rented Ubuntu + RTX 3090 machine.
#
#   bash setup.sh                 # full: system deps, venv, Open-CD, dataset, checks
#   bash setup.sh --skip-data     # env only
#   CUDA_VARIANT=cu118 bash setup.sh   # if the driver is too old for CUDA 12.1
#
# Re-running is safe: finished steps are skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY=${PY:-python3}
VENV=${VENV:-$ROOT/.venv}
DATA_DIR=${DATA_DIR:-$ROOT/data}
CUDA_VARIANT=${CUDA_VARIANT:-cu121}           # cu121 (driver >= 530) or cu118 (driver >= 520)
OPENCD_COMMIT=${OPENCD_COMMIT:-790a1972b538baaaffd4c3022a4d9afaa3f861e5}  # pinned, 2026-06-15
SKIP_DATA=0
for a in "$@"; do
  case "$a" in
    --skip-data) SKIP_DATA=1 ;;
    *) echo "unknown arg: $a"; exit 1 ;;
  esac
done

[ -f "$ROOT/.env" ] && set -a && source "$ROOT/.env" && set +a

log() { printf '\n\033[1;32m[setup]\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------- 1. system
log "1/6 system packages"
SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO="sudo"
NEED=""
for p in git unzip curl libgl1 libglib2.0-0; do dpkg -s "$p" >/dev/null 2>&1 || NEED="$NEED $p"; done
PYVER=$($PY -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
dpkg -s "python${PYVER}-venv" >/dev/null 2>&1 || NEED="$NEED python${PYVER}-venv"
if [ -n "$NEED" ]; then
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq $NEED
fi
case "$PYVER" in 3.8|3.9|3.10|3.11) ;; *) echo "Python $PYVER unsupported by mmcv 2.1 wheels; set PY=python3.10"; exit 1 ;; esac

log "GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || { echo "nvidia-smi failed"; exit 1; }

# ---------------------------------------------------------------- 2. venv
log "2/6 virtualenv at $VENV"
[ -d "$VENV" ] || $PY -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -U pip wheel "setuptools<70"

# ---------------------------------------------------------------- 3. torch + OpenMMLab
log "3/6 PyTorch 2.1.2 ($CUDA_VARIANT) + OpenMMLab"
if ! python -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.1.2') else 1)" 2>/dev/null; then
  pip install -q torch==2.1.2 torchvision==0.16.2 --index-url "https://download.pytorch.org/whl/${CUDA_VARIANT}"
fi
pip install -q "numpy<2" openmim mmengine==0.10.4
if ! python -c "import mmcv,sys; sys.exit(0 if mmcv.__version__=='2.1.0' else 1)" 2>/dev/null; then
  # --only-binary: fail fast instead of compiling mmcv for 30+ minutes
  pip install -q mmcv==2.1.0 --only-binary mmcv \
    -f "https://download.openmmlab.com/mmcv/dist/${CUDA_VARIANT}/torch2.1/index.html"
fi
pip install -q mmpretrain==1.2.0 mmsegmentation==1.2.2 mmdet==3.3.0 ftfy regex
pip install -q -r requirements.txt

# ---------------------------------------------------------------- 4. Open-CD
log "4/6 Open-CD @ ${OPENCD_COMMIT:0:7}"
OCD="$ROOT/third_party/open-cd"
if [ ! -d "$OCD/.git" ]; then
  git clone -q https://github.com/likyoo/open-cd "$OCD"
fi
git -C "$OCD" fetch -q --depth 1 origin "$OPENCD_COMMIT" 2>/dev/null || true
git -C "$OCD" checkout -q "$OPENCD_COMMIT"
# openmim -> openxlab pins setuptools~=60.2 (no PEP 660 editable support, and pip>=25
# dropped the `setup.py develop` fallback). Re-upgrade before the editable install;
# the openxlab pin conflict is harmless (only used by `mim download`).
if ! python -c "import opencd" 2>/dev/null; then
  pip install -q "setuptools>=64,<70" 2>&1 | grep -v "openxlab" || true
  pip install -q -e "$OCD" --no-build-isolation
fi

# ---------------------------------------------------------------- 5. data
if [ "$SKIP_DATA" -eq 0 ]; then
  log "5/6 dataset -> $DATA_DIR"
  bash scripts/download_data.sh "$DATA_DIR"
  RAW="$DATA_DIR/Taktpixel2025PD-CD"
  [ -d "$RAW/train" ] || RAW=$(dirname "$(find "$DATA_DIR" -maxdepth 3 -type d -name train | head -1)")
  echo "raw dataset: $RAW"
  if [ ! -d "$DATA_DIR/opencd_taktpixel/test/label" ]; then
    python scripts/prepare_opencd.py --data "$RAW" --out "$DATA_DIR/opencd_taktpixel"
  fi
  mkdir -p "$OCD/data"
  ln -sfn "$DATA_DIR/opencd_taktpixel" "$OCD/data/opencd_taktpixel"
  echo "$RAW" > "$ROOT/.data_root"
else
  log "5/6 dataset skipped"
fi

# ---------------------------------------------------------------- 6. checks
log "6/6 checks"
python - <<'EOF'
import torch, mmcv, mmseg, mmengine, opencd
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
print(f"mmcv {mmcv.__version__} mmseg {mmseg.__version__} mmengine {mmengine.__version__} opencd {opencd.__version__}")
from mmcv.ops import nms  # verifies compiled CUDA ops
assert torch.cuda.is_available(), "CUDA not available"
EOF
python -m pytest -q tests

if [ -n "${HF_REPO:-}" ]; then
  python - <<'PYEOF' || echo "WARNING: HF login failed. Put a write HF_TOKEN in .env."
import os
from huggingface_hub import HfApi
user = HfApi(token=os.environ.get("HF_TOKEN")).whoami()["name"]
print(f"HF login ok: {user} -> {os.environ['HF_REPO']}")
PYEOF
else
  echo "WARNING: no .env with HF_REPO/HF_TOKEN - checkpoints will not be backed up. See .env.example"
fi

log "done. Next: source .venv/bin/activate && bash scripts/status.sh"
