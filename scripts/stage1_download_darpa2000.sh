#!/usr/bin/env bash
# Stage 1 — download the DARPA2000 LLDOS multi-stage attack datasets.
#
# Source: MIT Lincoln Laboratory archive (archive.ll.mit.edu). No registration
# required. These are the headline correlation-evaluation datasets: real,
# labeled, multi-phase attacker campaigns (probe -> exploit -> install -> DDoS).
#
# Usage: bash scripts/stage1_download_darpa2000.sh [output_dir]

set -euo pipefail

OUT_DIR="${1:-data/raw/darpa2000}"
mkdir -p "$OUT_DIR"

BASE="https://archive.ll.mit.edu/ideval/data/2000"

# scenario -> tar.gz containing tcpdump (inside+dmz, per-phase), BSM audit logs,
# and the mid-level XML "truth" labels that mark which packets belong to which
# attack phase.
declare -A SCENARIOS=(
  ["LLS_DDOS_1.0"]="$BASE/LLS_DDOS_1.0/data_and_labeling/LLS_DDOS_1.0.tar.gz"
  ["LLS_DDOS_2.0.2"]="$BASE/LLS_DDOS_2.0.2/data_and_labeling/LLS_DDOS_2.0.2.tar.gz"
)

for name in "${!SCENARIOS[@]}"; do
  url="${SCENARIOS[$name]}"
  dest="$OUT_DIR/${name}.tar.gz"
  if [ -f "$dest" ]; then
    echo "[skip] $dest already exists"
  else
    echo "[fetch] $name  <-  $url"
    curl -L --fail --retry 3 -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
  fi

  extract_dir="$OUT_DIR/$name"
  if [ -d "$extract_dir" ]; then
    echo "[skip] $extract_dir already extracted"
  else
    echo "[extract] $dest -> $extract_dir"
    mkdir -p "$extract_dir"
    tar -xzf "$dest" -C "$extract_dir"
  fi
done

echo "Done. Contents:"
find "$OUT_DIR" -maxdepth 3 -type d | sort
