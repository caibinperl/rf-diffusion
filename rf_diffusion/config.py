"""Dataclasses that mirror Supplementary Methods Table 6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class DiffuserConfig:
    T: int = 200
    b_0: float = 0.01
    b_T: float = 0.07
    schedule_type: str = "linear"
    so3_type: str = "igso3"
    so3_schedule_type: str = "linear"
    min_sigma: float = 0.02
    max_sigma: float = 1.5
    # Paper Table 6. Inference yaml uses 1.5 / 2.5; training uses 1.06 / 1.77.
    min_b: float = 1.06
    max_b: float = 1.77
    crd_scale: float = 0.25
    truncation_level: int = 200
    num_omega: int = 512
    num_sigma: int = 128
    cache_dir: str = "./schedule_cache"


@dataclass
class LossConfig:
    w_trans: float = 0.5
    w_rot: float = 1.0
    w_2d: float = 1.0
    d_clamp: float = 10.0
    p_clamp: float = 0.9
    gamma: float = 0.99
    motif_loss_weight: float = 1.0
    # Active-site fine-tune upweights motif displacement 10x (SI 4.2).
    active_site_motif_weight: float = 10.0


@dataclass
class DataConfig:
    pdb_dir: Optional[str] = None
    pdb_index: Optional[str] = None
    crop_size: int = 384
    min_length: int = 40
    max_length: int = 512
    unconditional_prob: float = 0.20
    motif_contiguous_prob: float = 0.50
    motif_mask_frac_min: float = 0.20
    motif_mask_frac_max: float = 1.00
    hotspot_frac_max: float = 0.20
    hotspot_distance: float = 10.0
    synthetic_n: int = 256
    synthetic_len: int = 64
    num_workers: int = 0


@dataclass
class OptimConfig:
    lr: float = 5e-4
    weight_decay: float = 0.0
    warmup_steps: int = 0
    decay_every: int = 10_000
    decay_factor: float = 0.95
    grad_clip: float = 1.0
    pseudo_batch_size: int = 64
    micro_batch_size: int = 1


@dataclass
class TrainConfig:
    experiment: Literal["base", "active_site", "ppi", "fold"] = "base"
    model: Literal["tiny", "rf"] = "tiny"
    rf_ckpt: Optional[str] = None
    output_dir: str = "./outputs"
    seed: int = 0
    device: str = "cuda"
    max_steps: Optional[int] = None
    epochs: int = 5
    examples_per_epoch: int = 25600
    log_every: int = 10
    ckpt_every: int = 500
    self_cond_prob: float = 0.50
    use_wandb: bool = False
    wandb_project: str = "rfdiffusion-train"
    diffuser: DiffuserConfig = field(default_factory=DiffuserConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


def default_config() -> TrainConfig:
    return TrainConfig()


def config_from_dict(d: dict) -> TrainConfig:
    cfg = TrainConfig()
    for section in ("diffuser", "loss", "data", "optim"):
        if section in d and d[section] is not None:
            section_obj = getattr(cfg, section)
            for k, v in d[section].items():
                setattr(section_obj, k, v)
    for k, v in d.items():
        if k in ("diffuser", "loss", "data", "optim"):
            continue
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    return cfg
