#!/usr/bin/env bash
# PPI / hotspot fine-tune recipe (SI §4.3). Provide a directory of complex PDBs
# to actually condition on interfaces; synthetic monomers fall back to motif tasks.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/train.py \
  experiment=ppi \
  model=tiny \
  device=cpu \
  max_steps=50 \
  output_dir=./outputs/ppi_toy \
  data.pdb_dir="${1:-}" \
  data.crop_size=96 \
  diffuser.so3_type=gaussian
