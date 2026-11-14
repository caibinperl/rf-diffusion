#!/usr/bin/env bash
# Active-site fine-tune recipe (SI §4.2): 70% triple-contact motifs, motif loss x10.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/train.py \
  experiment=active_site \
  model=tiny \
  device=cpu \
  max_steps=50 \
  output_dir=./outputs/active_site_toy \
  data.synthetic_n=64 \
  data.crop_size=64 \
  diffuser.so3_type=gaussian
