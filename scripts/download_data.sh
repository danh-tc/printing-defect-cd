#!/usr/bin/env bash
# Download and verify Taktpixel2025PD-CD from Zenodo (DOI 10.5281/zenodo.15318946).
set -euo pipefail
DEST=${1:-data}
URL="https://zenodo.org/records/15318946/files/Taktpixel2025PD-CD.zip?download=1"
MD5="157ee2371e1fad27b77442a78d7ee63c"

mkdir -p "$DEST"
ZIP="$DEST/Taktpixel2025PD-CD.zip"
[ -f "$ZIP" ] || curl -L --fail -o "$ZIP" "$URL"
echo "$MD5  $ZIP" | md5sum -c -
unzip -q -n "$ZIP" -d "$DEST"
echo "Extracted to $DEST. Check that $DEST/Taktpixel2025PD-CD/{train,val,test}/{A,B,OUT} exist;"
echo "if the archive uses a different top-level folder, pass that path as --data."
