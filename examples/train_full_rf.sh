#!/usr/bin/env bash
# Full-scale launch matching Supplementary Methods Table 6.
# Requires: official rfdiffusion install, RF weights, PDB training set, 8x A100.
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${RF_CKPT:-models/RF_structure_prediction_weights.pt}"
PDB="${PDB_DIR:?Set PDB_DIR to the RoseTTAFold monomer PDB directory}"
python scripts/train.py \
  experiment=base \
  model=rf \
  rf_ckpt="$CKPT" \
  device=cuda \
  max_steps=null \
  epochs=5 \
  examples_per_epoch=25600 \
  output_dir=./outputs/base_rf \
  data.pdb_dir="$PDB" \
  data.crop_size=384 \
  diffuser.T=200 \
  diffuser.so3_type=igso3 \
  diffuser.truncation_level=2000 \
  diffuser.num_omega=1000 \
  diffuser.num_sigma=500 \
  optim.lr=0.0005 \
  optim.pseudo_batch_size=64 \
  optim.micro_batch_size=1 \
  optim.decay_every=10000 \
  optim.decay_factor=0.95
