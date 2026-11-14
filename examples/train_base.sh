#!/usr/bin/env bash
# Educational base-model training on synthetic compact backbones.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/train.py \
  experiment=base \
  model=tiny \
  device=cpu \
  max_steps=50 \
  log_every=5 \
  ckpt_every=50 \
  output_dir=./outputs/base_toy \
  data.synthetic_n=64 \
  data.synthetic_len=48 \
  data.crop_size=48 \
  diffuser.so3_type=gaussian \
  optim.micro_batch_size=1 \
  optim.pseudo_batch_size=4
