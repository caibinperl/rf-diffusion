#!/usr/bin/env python
"""Scan a PDB directory and write a JSON index for RFdiffusion training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rf_diffusion.geometry import parse_pdb_backbone


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pdb_dir", required=True)
    p.add_argument("--out", default="data/pdb_index.json")
    p.add_argument("--min_length", type=int, default=40)
    p.add_argument("--max_length", type=int, default=1024)
    args = p.parse_args()
    paths = sorted(Path(args.pdb_dir).rglob("*.pdb"))
    kept = []
    for path in paths:
        try:
            rec = parse_pdb_backbone(path)
        except Exception as e:
            print(f"skip {path}: {e}")
            continue
        L = rec["xyz"].shape[0]
        if args.min_length <= L <= args.max_length:
            kept.append(str(path.resolve()))
            print(f"ok  {path.name}  L={L}")
        else:
            print(f"len {path.name}  L={L}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(kept, indent=2))
    print(f"Wrote {len(kept)} entries to {out}")


if __name__ == "__main__":
    main()
