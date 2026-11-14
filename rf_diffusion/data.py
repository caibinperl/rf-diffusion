"""Datasets: synthetic compact backbones and PDB crops.

Training-task mix (SI §4.1):
  20% unconditional, 80% motif-conditional.
Motif residues: randomly 20--100% of the chain, contiguous with p=0.5.
Examples are centered on the motif COM (or the full-chain COM if unconditional).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig, TrainConfig
from .geometry import (
    AA1_TO_IDX,
    IDEAL_C,
    IDEAL_N,
    center_xyz,
    frames_to_xyz,
    generate_cbeta,
    parse_pdb_backbone,
    rot_matmul,
    xyz_to_frames,
)


def _rotation_about(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z = axis.unbind(-1)
    c, s = torch.cos(angle), torch.sin(angle)
    C = 1 - c
    R = torch.stack(
        [
            c + x * x * C, x * y * C - z * s, x * z * C + y * s,
            y * x * C + z * s, c + y * y * C, y * z * C - x * s,
            z * x * C - y * s, z * y * C + x * s, c + z * z * C,
        ],
        dim=-1,
    ).reshape(*angle.shape, 3, 3)
    return R


def make_helical_backbone(
    length: int,
    noise: float = 0.4,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Toy α-helix-like N-CA-C coordinates, slightly perturbed."""
    rise, twist = 1.5, 100.0 * torch.pi / 180.0
    radius = 2.3
    i = torch.arange(length).float()
    x = radius * torch.cos(i * twist)
    y = radius * torch.sin(i * twist)
    z = i * rise
    ca = torch.stack([x, y, z], dim=-1)
    if noise:
        ca = ca + noise * torch.randn(ca.shape, generator=generator) * 0.15
    R, _ = xyz_to_frames(
        torch.stack(
            [
                ca + torch.tensor([-0.8, 0.8, 0.0]),
                ca,
                ca + torch.tensor([1.2, 0.0, 0.2]),
            ],
            dim=1,
        ).unsqueeze(0)
    )
    # Re-idealize to valid N-CA-C
    xyz = frames_to_xyz(R[0], ca)
    return xyz


def make_compact_protein(length: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """A few helical segments packed into a compact blob."""
    nseg = max(2, length // 20)
    segs = []
    remaining = length
    for s in range(nseg):
        slen = remaining if s == nseg - 1 else max(8, remaining // (nseg - s))
        remaining -= slen
        helix = make_helical_backbone(slen, noise=0.5, generator=generator)
        axis = torch.randn(3, generator=generator)
        ang = torch.rand((), generator=generator) * 2 * torch.pi
        R = _rotation_about(axis, ang)
        offset = torch.randn(3, generator=generator) * 6.0
        helix = torch.einsum("ij,laj->lai", R, helix) + offset
        segs.append(helix)
    xyz = torch.cat(segs, dim=0)[:length]
    return center_xyz(xyz.unsqueeze(0))[0]


def _free_segment_starts(occupied: torch.Tensor, n: int) -> torch.Tensor:
    """Starts `s` where `occupied[s:s+n]` is entirely False."""
    length = int(occupied.shape[0])
    if n <= 0 or n > length:
        return torch.empty(0, dtype=torch.long)
    window = (~occupied).unfold(0, n, 1).all(dim=-1)
    return torch.where(window)[0]


def optional_neighbor(r: int, length: int) -> Optional[int]:
    """With probability 1/2, pick one sequence neighbor of `r` that exists."""
    if np.random.rand() >= 0.5:
        return None
    choices = []
    if r > 0:
        choices.append(r - 1)
    if r + 1 < length:
        choices.append(r + 1)
    if not choices:
        return None
    return int(choices[np.random.randint(0, len(choices))])


def chain_residue_index(chain: np.ndarray) -> torch.Tensor:
    """Residue index with a +200 jump at each chain break, matching RoseTTAFold."""
    n = len(chain)
    idx = torch.arange(n, dtype=torch.long)
    offset = 0
    for i in range(1, n):
        if chain[i] != chain[i - 1]:
            offset += 200
        idx[i] = i + offset
    return idx


def sample_motif_mask(length: int, cfg: DataConfig) -> torch.Tensor:
    """True = motif (fixed).  Masked fraction is the *diffused* fraction."""
    mask_frac = np.random.uniform(cfg.motif_mask_frac_min, cfg.motif_mask_frac_max)
    n_diffused = int(round(mask_frac * length))
    n_diffused = min(max(n_diffused, 0), length)
    n_motif = length - n_diffused
    motif = torch.zeros(length, dtype=torch.bool)
    if n_motif == 0:
        return motif
    if n_motif == length or np.random.rand() < cfg.motif_contiguous_prob:
        start = np.random.randint(0, length - n_motif + 1)
        motif[start : start + n_motif] = True
        return motif
    if n_motif == 1:
        motif[np.random.randint(0, length)] = True
        return motif
    for _ in range(50):
        n1 = int(np.random.randint(1, n_motif))
        n2 = n_motif - n1
        motif.zero_()
        starts1 = _free_segment_starts(motif, n1)
        if len(starts1) == 0:
            continue
        i1 = int(starts1[np.random.randint(0, len(starts1))])
        motif[i1 : i1 + n1] = True
        starts2 = _free_segment_starts(motif, n2)
        if len(starts2) == 0:
            continue
        i2 = int(starts2[np.random.randint(0, len(starts2))])
        motif[i2 : i2 + n2] = True
        return motif
    motif.zero_()
    start = np.random.randint(0, length - n_motif + 1)
    motif[start : start + n_motif] = True
    return motif


def sample_active_site_mask(xyz: torch.Tensor) -> Optional[torch.Tensor]:
    """Triple-contact motif (SI §4.2): 3 residues >10 apart in sequence, Cb<6 Å."""
    L = xyz.shape[0]
    if L < 25:
        return None
    N, Ca, C = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    Cb = generate_cbeta(N, Ca, C)
    d = torch.cdist(Cb, Cb)
    idx = torch.arange(L)
    seqd = (idx[:, None] - idx[None, :]).abs()
    close = (d < 6.0) & (seqd > 10)
    # brute-force a random triad
    candidates = torch.nonzero(close, as_tuple=False)
    if len(candidates) < 3:
        return None
    for _ in range(200):
        i = int(np.random.randint(0, L))
        js = torch.where(close[i])[0]
        if len(js) < 2:
            continue
        perm = js[torch.randperm(len(js))[:2]]
        j, k = int(perm[0]), int(perm[1])
        if close[j, k]:
            mask = torch.zeros(L, dtype=torch.bool)
            for r in (i, j, k):
                mask[r] = True
                nb = optional_neighbor(r, L)
                if nb is not None:
                    mask[nb] = True
            return mask
    return None


def hotspot_mask(
    xyz: torch.Tensor,
    chain: np.ndarray,
    binder_id,
    max_frac: float = 0.2,
    cutoff: float = 10.0,
) -> torch.Tensor:
    """Interface residues on the fixed chains, subsampled to 0--20% (SI §4.3.1).

    `binder_id` is the chain that is diffused. Hotspots lie on every other chain,
    within `cutoff` Å (Cβ) of the binder. The kept fraction can be zero.
    """
    L = xyz.shape[0]
    hot = torch.zeros(L, dtype=torch.bool)
    labels = list(chain)
    is_binder = torch.tensor([c == binder_id for c in labels])
    is_target = ~is_binder
    if not bool(is_binder.any()) or not bool(is_target.any()):
        return hot
    N, Ca, C = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    Cb = generate_cbeta(N, Ca, C)
    d = torch.cdist(Cb, Cb)
    iface = (d[:, is_binder].min(dim=1).values < cutoff) & is_target
    idx = torch.where(iface)[0]
    if len(idx) == 0:
        return hot
    keep = int(np.random.uniform(0.0, max_frac) * len(idx))
    keep = min(max(keep, 0), len(idx))
    if keep == 0:
        return hot
    sel = idx[torch.randperm(len(idx))[:keep]]
    hot[sel] = True
    return hot


class ProteinDataset(Dataset):
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.data_cfg = cfg.data
        self.records: List[Dict] = []
        if self.data_cfg.pdb_dir:
            self._index_pdbs()
        if not self.records:
            self._build_synthetic()

    def _index_pdbs(self) -> None:
        paths = sorted(glob.glob(str(Path(self.data_cfg.pdb_dir) / "**/*.pdb"), recursive=True))
        if self.data_cfg.pdb_index and Path(self.data_cfg.pdb_index).exists():
            with open(self.data_cfg.pdb_index) as f:
                listed = json.load(f)
            paths = listed
        for p in paths:
            try:
                rec = parse_pdb_backbone(p)
            except Exception:
                continue
            L = rec["xyz"].shape[0]
            if L < self.data_cfg.min_length or L > self.data_cfg.max_length:
                continue
            self.records.append({"path": p, "length": L, "kind": "pdb"})

    def _build_synthetic(self) -> None:
        rng = np.random.RandomState(0)
        for i in range(self.data_cfg.synthetic_n):
            L = int(rng.randint(40, max(41, self.data_cfg.synthetic_len + 1)))
            self.records.append({"kind": "synthetic", "length": L, "seed": int(rng.randint(1e9))})

    def __len__(self) -> int:
        return max(len(self.records), 1)

    def _load(self, rec: Dict) -> Dict[str, torch.Tensor]:
        if rec["kind"] == "pdb":
            raw = parse_pdb_backbone(rec["path"])
            xyz = torch.tensor(raw["xyz"])
            seq = torch.tensor(raw["seq"])
            chain = raw["chain"]
        else:
            rng = torch.Generator().manual_seed(int(rec["seed"]))
            L = rec["length"]
            xyz = make_compact_protein(L, generator=rng)
            seq = torch.randint(0, 20, (L,), generator=rng)
            chain = np.array(["A"] * L, dtype=object)
        return {"xyz": xyz, "seq": seq, "chain": chain}

    def _crop(self, xyz: torch.Tensor, seq: torch.Tensor, chain: np.ndarray):
        L = xyz.shape[0]
        crop = self.data_cfg.crop_size
        if L <= crop:
            return xyz, seq, chain
        start = np.random.randint(0, L - crop + 1)
        return xyz[start : start + crop], seq[start : start + crop], chain[start : start + crop]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx % len(self.records)]
        item = self._load(rec)
        xyz, seq, chain = self._crop(item["xyz"], item["seq"], item["chain"])
        L = xyz.shape[0]
        experiment = self.cfg.experiment

        task = "unconditional" if np.random.rand() < self.data_cfg.unconditional_prob else "motif"
        motif_mask = torch.zeros(L, dtype=torch.bool)
        hotspot = torch.zeros(L, dtype=torch.bool)
        is_complex = len(np.unique(chain)) > 1

        if experiment == "active_site" and np.random.rand() < 0.70:
            m = sample_active_site_mask(xyz)
            if m is not None:
                motif_mask = m
                task = "active_site"
            else:
                task = "motif"

        if experiment == "ppi" and is_complex and np.random.rand() < 0.50:
            # Noise one randomly chosen chain; keep the other chains fixed.
            chain_ids = list(np.unique(chain))
            binder_id = chain_ids[int(np.random.randint(0, len(chain_ids)))]
            binder = torch.tensor([c == binder_id for c in chain])
            motif_mask = ~binder
            hotspot = hotspot_mask(
                xyz,
                chain,
                binder_id,
                self.data_cfg.hotspot_frac_max,
                self.data_cfg.hotspot_distance,
            )
            task = "ppi"
        elif task == "motif" and not motif_mask.any():
            motif_mask = sample_motif_mask(L, self.data_cfg)

        if motif_mask.any():
            xyz = center_xyz(xyz.unsqueeze(0), mask=motif_mask.unsqueeze(0))[0]
        else:
            xyz = center_xyz(xyz.unsqueeze(0))[0]

        return {
            "xyz": xyz.float(),
            "seq": seq.long(),
            "motif_mask": motif_mask,
            "hotspot": hotspot,
            "idx": chain_residue_index(chain),
            "task": task,
            "length": torch.tensor(L),
        }


def collate_pad(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    Lmax = max(int(b["length"]) for b in batch)
    B = len(batch)
    xyz = torch.zeros(B, Lmax, 3, 3)
    seq = torch.full((B, Lmax), 21, dtype=torch.long)
    motif = torch.zeros(B, Lmax, dtype=torch.bool)
    hotspot = torch.zeros(B, Lmax, dtype=torch.bool)
    pad_mask = torch.zeros(B, Lmax, dtype=torch.bool)
    idx = torch.zeros(B, Lmax, dtype=torch.long)
    for i, b in enumerate(batch):
        L = int(b["length"])
        xyz[i, :L] = b["xyz"]
        seq[i, :L] = b["seq"]
        motif[i, :L] = b["motif_mask"]
        hotspot[i, :L] = b["hotspot"]
        pad_mask[i, :L] = True
        idx[i, :L] = b["idx"]
        if L < Lmax and L > 0:
            base = int(b["idx"][-1]) + 200
            idx[i, L:] = base + torch.arange(1, Lmax - L + 1)
    return {
        "xyz": xyz,
        "seq": seq,
        "motif_mask": motif,
        "hotspot": hotspot,
        "pad_mask": pad_mask,
        "idx": idx,
        "length": torch.stack([b["length"] for b in batch]),
    }
