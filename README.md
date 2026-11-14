# RFdiffusion training tutorial

A **training pipeline** reproduced from [RosettaCommons/RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) and the supplementary material of Watson *et al.*, *Nature* 2023.

The official repository **releases inference code only** (discussions [#239](https://github.com/RosettaCommons/RFdiffusion/discussions/239), [#270](https://github.com/RosettaCommons/RFdiffusion/discussions/270)). This repository rewrites the training scripts from the paper's Algorithm 3, Table 6, and the public `rfdiffusion/diffusion.py` / `igso3.py`, for learning and small-scale experiments. A full reproduction still needs the RoseTTAFold backbone, a PDB training set, and multi-GPU A100s.

## Contents

| Path | Description |
| --- | --- |
| `scripts/train.py` | Hydra training entry point (Algorithm 3) |
| `rf_diffusion/` | Diffusion, losses, data, model, and the training loop |
| `examples/` | Launch scripts for base / active-site / PPI / full RF |
| `configs/train.yaml` | Hyperparameters aligned with Table 6 of the paper |

## Installation

```bash
conda env create -f environment.yml
conda activate rf-diffusion
```

To plug in the official RoseTTAFold backbone (optional, `model=rf`):

```bash
git clone https://github.com/RosettaCommons/RFdiffusion.git
# Install SE3nv and rfdiffusion following the official README
# Download the RF structure-prediction weights RF_structure_prediction_weights.pt
```

## Quick start (synthetic data, CPU, about one minute)

```bash
chmod +x examples/*.sh
./examples/train_base.sh
```

Or:

```bash
python scripts/train.py experiment=base model=tiny device=cpu max_steps=50 \
  diffuser.so3_type=gaussian data.crop_size=48
```

## Paper recipe (8×A100, about 3 days)

```bash
export PDB_DIR=/path/to/rf_monomer_pdb
export RF_CKPT=models/RF_structure_prediction_weights.pt
./examples/train_full_rf.sh
```

Switch tasks:

```bash
python scripts/train.py experiment=active_site   # SI §4.2, three-residue active site
python scripts/train.py experiment=ppi           # SI §4.3, complex + hotspot
```

## Tests

```bash
python tests/test_train_step.py
```

## Citation

```
Watson et al. De novo design of protein structure and function with RFdiffusion.
Nature 620, 1089–1100 (2023). https://doi.org/10.1038/s41586-023-06415-8
```

Official inference code: https://github.com/RosettaCommons/RFdiffusion
