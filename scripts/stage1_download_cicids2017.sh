#!/usr/bin/env bash
# Stage 1 — download CICIDS2017 from the York BCCC lab mirror.
#
# NOTE: this is BCCC's re-extraction of the original CIC-IDS2017 pcaps using
# their own flow extractor, not the original UNB CICFlowMeter CSVs. Column
# names/feature set may differ from the classic CIC-IDS2017 papers -- Stage 2
# will inspect the actual schema before writing the parser. Cite the BCCC
# re-extraction (not just the original Sharafaldin et al. 2018 CIC-IDS2017
# paper) in the thesis when using this source.
#
# Source: http://bccc.laps.yorku.ca/BCCC-CIC-IDS-2017/CSVs.zip (no
# registration needed at this mirror -- found directly reachable; the
# yorku.ca dataset-request form is the vendor's documented path but this
# direct link works).
#
# Usage: bash scripts/stage1_download_cicids2017.sh [output_dir]

set -euo pipefail

OUT_DIR="${1:-data/raw/cicids2017}"
mkdir -p "$OUT_DIR"

URL="http://bccc.laps.yorku.ca/BCCC-CIC-IDS-2017/CSVs.zip"
DEST="$OUT_DIR/CSVs.zip"

if [ -f "$DEST" ]; then
  echo "[skip] $DEST already exists"
else
  echo "[fetch] $URL"
  curl -L --fail --retry 3 -o "$DEST.part" "$URL"
  mv "$DEST.part" "$DEST"
fi

EXTRACT_DIR="$OUT_DIR/CSVs"
if [ -d "$EXTRACT_DIR" ]; then
  echo "[skip] $EXTRACT_DIR already extracted"
else
  echo "[extract] $DEST -> $EXTRACT_DIR"
  mkdir -p "$EXTRACT_DIR"
  unzip -q "$DEST" -d "$EXTRACT_DIR"
fi

echo "Done. Contents:"
find "$OUT_DIR" -maxdepth 3 | sort
