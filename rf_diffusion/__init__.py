"""Educational RFdiffusion training package.

The official RosettaCommons/RFdiffusion repository ships inference only.
This package reconstructs the training procedure from:

* Watson et al., Nature 2023, Supplementary Methods (Algorithms 1--3, Tables 4--6)
* The public inference implementation (`rfdiffusion/diffusion.py`, `igso3.py`)

It is intended for study and small-scale experiments.  Full-scale training
still requires the RoseTTAFold backbone, PDB clusters, and multi-GPU compute
described in the paper (8 x A100, 5 epochs, ~3 days).
"""

from .config import TrainConfig, default_config
from .diffusion import SE3Diffuser
from .losses import diffusion_loss
from .trainer import Trainer

__all__ = [
    "TrainConfig",
    "default_config",
    "SE3Diffuser",
    "diffusion_loss",
    "Trainer",
]
