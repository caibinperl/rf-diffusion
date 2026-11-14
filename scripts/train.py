#!/usr/bin/env python
"""RFdiffusion training entry point.

Official training code was never released (RosettaCommons/RFdiffusion#239,#270).
This script follows Supplementary Methods Algorithm 3 and Table 6.

Examples
--------
python scripts/train.py experiment=base model=tiny max_steps=50 data.synthetic_n=32
python scripts/train.py experiment=active_site model=tiny
python scripts/train.py experiment=base model=rf rf_ckpt=models/RF_structure_prediction_weights.pt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rf_diffusion.config import config_from_dict
from rf_diffusion.trainer import train_from_cfg


def _deep_set(d: dict, dotted: str, value):
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    raw = value
    if isinstance(raw, str):
        if raw.lower() in ("null", "none"):
            raw = None
        elif raw.lower() in ("true", "false"):
            raw = raw.lower() == "true"
        else:
            try:
                raw = int(raw)
            except ValueError:
                try:
                    raw = float(raw)
                except ValueError:
                    pass
    cur[keys[-1]] = raw


def load_cfg(argv=None) -> dict:
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg_path = ROOT / "configs" / "train.yaml"
    overrides = []
    i = 0
    while i < len(argv):
        if argv[i] in ("--config-name", "-cn") and i + 1 < len(argv):
            cfg_path = ROOT / "configs" / f"{argv[i + 1]}.yaml"
            i += 2
            continue
        if argv[i].startswith("--config-path"):
            i += 2 if "=" not in argv[i] else 1
            continue
        if "=" in argv[i] and not argv[i].startswith("-"):
            overrides.append(argv[i])
        i += 1
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}
    for item in overrides:
        k, v = item.split("=", 1)
        _deep_set(cfg, k, v)
    return cfg


def main() -> None:
    os.chdir(ROOT)
    try:
        import hydra
        from omegaconf import DictConfig, OmegaConf

        @hydra.main(version_base=None, config_path="../configs", config_name="train")
        def _hydra_main(cfg: DictConfig) -> None:
            os.chdir(ROOT)
            plain = OmegaConf.to_container(cfg, resolve=True)
            train_cfg = config_from_dict(plain)
            Path(train_cfg.output_dir).mkdir(parents=True, exist_ok=True)
            OmegaConf.save(cfg, Path(train_cfg.output_dir) / "hydra.yaml")
            print(OmegaConf.to_yaml(cfg))
            train_from_cfg(train_cfg)

        _hydra_main()
        return
    except ImportError:
        pass

    plain = load_cfg()
    train_cfg = config_from_dict(plain)
    Path(train_cfg.output_dir).mkdir(parents=True, exist_ok=True)
    print(yaml.safe_dump(plain, sort_keys=False))
    train_from_cfg(train_cfg)


if __name__ == "__main__":
    main()
