#!/usr/bin/env bash
# Download and verify Taktpixel2025PD-CD (Zenodo DOI 10.5281/zenodo.15318946).
#
# Source order:
#   1. Zenodo (often 403 "unusual traffic" from rented GPU/datacenter IPs)
#   2. Private HF dataset mirror: $HF_DATA_REPO (default <HF user of HF_REPO>/taktpixel2025pd-cd),
#      file Taktpixel2025PD-CD.zip. Needs HF_TOKEN (read) in .env.
# Either way the MD5 of the original Zenodo zip is verified.
set -euo pipefail
DEST=${1:-data}
URL="https://zenodo.org/records/15318946/files/Taktpixel2025PD-CD.zip?download=1"
MD5="157ee2371e1fad27b77442a78d7ee63c"
ZIP_NAME="Taktpixel2025PD-CD.zip"
HF_REPO=${HF_REPO:-}
HF_DATA_REPO=${HF_DATA_REPO:-${HF_REPO%%/*}/taktpixel2025pd-cd}

mkdir -p "$DEST"
ZIP="$DEST/$ZIP_NAME"

ok_md5() { [ -f "$ZIP" ] && echo "$MD5  $ZIP" | md5sum -c --status -; }

if ! ok_md5; then
  rm -f "$ZIP"
  echo "[data] trying Zenodo ..."
  if ! curl -L --fail -sS -o "$ZIP" "$URL" || ! ok_md5; then
    rm -f "$ZIP"
    if [[ "$HF_DATA_REPO" == /* ]]; then   # neither HF_DATA_REPO nor HF_REPO set
      echo "[data] Zenodo failed and no HF_DATA_REPO/HF_REPO set in .env"; exit 1
    fi
    echo "[data] Zenodo failed -> HF dataset mirror $HF_DATA_REPO"
    python - "$HF_DATA_REPO" "$ZIP_NAME" "$DEST" <<'PYEOF'
import os, shutil, sys
from huggingface_hub import hf_hub_download
repo, name, dest = sys.argv[1:]
p = hf_hub_download(repo, name, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
shutil.copy(p, os.path.join(dest, name))
PYEOF
  fi
fi
echo "$MD5  $ZIP" | md5sum -c -
unzip -q -n "$ZIP" -d "$DEST"
echo "Extracted to $DEST. Check that $DEST/Taktpixel2025PD-CD/{train,val,test}/{A,B,OUT} exist;"
echo "if the archive uses a different top-level folder, pass that path as --data."
