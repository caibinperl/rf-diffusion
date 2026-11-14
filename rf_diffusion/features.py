"""Pack a noised backbone and a sequence task into network inputs. Corresponds to Supplementary Methods Table 4.

The model sees three things at once: the masked sequence, the backbone at step t,
and the 1D / 2D template features computed from that backbone. Motif residues keep
their amino-acid identity and clean coordinates; every other residue is set to
MASK in sequence, and its coordinates are the output of `SE3Diffuser.forward_noise`.

There are two independent time channels. Do not treat them as the same quantity:

- Last channel of `t1d`: motif positions are written as 1, and the rest as 1 − t/T.
  This is the scalar used by RoseTTAFold templates.
- `sinusoidal_time`: encodes the discrete step t as sin/cos, which the small denoiser
  adds onto residue embeddings.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .geometry import MASK_IDX, N_AA, generate_cbeta
from .losses import xyz_to_c6d


def sinusoidal_time(t: torch.Tensor, dim: int = 32, T: int = 200) -> torch.Tensor:
    """Encode a diffusion step as a Transformer-style sin/cos embedding.

    Frequencies match Vaswani positional encoding: ω_k = 10000^(−k / (dim/2)),
    k = 0…dim/2−1, and the output is (sin(t ω), cos(t ω)). t is the raw step
    index (1…T) and is not divided by T. The `T` argument exists only to match
    the caller's signature and does not enter the computation. If dim is odd,
    the last entry is padded with 0.

    Args:
        t: (B,) or (B, L), a 1-based discrete step
    Returns:
        (B, dim) when t is (B,); (B, L, dim) when t is (B, L)
    """
    half = dim // 2
    freq = torch.exp(
        -torch.arange(half, device=t.device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / half)
    )
    arg = t.float().unsqueeze(-1) * freq
    emb = torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


def build_t1d(
    seq: torch.Tensor,
    t: torch.Tensor,
    motif_mask: torch.Tensor,
    T: int,
    hotspot: Optional[torch.Tensor] = None,
    ss: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One-dimensional template features. Base width 22, matching Table 4's "20 amino acids + time".

    `seq` is first turned into a 22-class one-hot (0–19 amino acids, 20 is X, 21 is MASK),
    then the last channel is overwritten with time, so there is no separate mask bit:

    - 0–19: amino acids. Masked residues are all zeros here.
    - 20: 1 only when the residue itself is X.
    - 21: time. Motif residues are 1; the rest are 1 − t/T (near 1 at t=1, and 0 at t=T).

    The network recognizes residues to be designed by "channels 0–20 all zero".
    `seq` should already have been processed by `mask_sequence` before the call:
    motifs keep their identity, and the rest are MASK.

    Optional channels are appended only for the task that uses them:
    - hotspot: (B, L), PPI only. 1 on residues the binder should touch. Width becomes 23.
    - ss: (B, L, 4) helix / strand / coil / mask, fold conditioning only. Width becomes 26.

    Args:
        seq: (B, L) integers
        t: (B,) in 1…T
        motif_mask: (B, L), True = motif
    Returns:
        (B, L, 22 + optional channels)
    """
    B, L = seq.shape
    aa = F.one_hot(seq.clamp(0, N_AA - 1), num_classes=N_AA).float()  # 22
    # Table 4 uses 20 AA + mask + timestep.  We keep 22-class one-hot and
    # overwrite the last channel with the continuous timestep used by RF.
    time_chan = torch.ones(B, L, device=seq.device, dtype=torch.float32)
    t_scaled = 1.0 - t.float().view(B, 1) / T
    time_chan = torch.where(motif_mask, torch.ones_like(time_chan), t_scaled.expand(B, L))
    t1d = aa.clone()
    t1d[..., -1] = time_chan
    extras = []
    if hotspot is not None:
        extras.append(hotspot.float().unsqueeze(-1))
    if ss is not None:
        extras.append(ss.float())
    if extras:
        t1d = torch.cat([t1d, *extras], dim=-1)
    return t1d


def build_t2d(xyz_t: torch.Tensor) -> torch.Tensor:
    """Build residue-pair features from the step-t backbone, shape (B, L, L, 44), aligned with Table 4.

    Geometry comes from `xyz_to_c6d`. Distances use a virtual Cβ, and the three angles are

    - ω: dihedral CA_i–Cβ_i–Cβ_j–CA_j
    - θ: dihedral N_i–CA_i–Cβ_i–Cβ_j
    - φ: planar angle CA_i–Cβ_i–Cβ_j (unsigned, in [0, π])

    The 44 channels are:

    - 37 distance one-hot channels. Bin edges run from 2.5 Å to 20 Å in steps of 0.5 Å,
      36 edges in total. Distances below 2.5 Å fall in bin 0, and distances at or above
      20 Å fall in the last bin.
    - 6 angle channels: sin and cos of ω, θ, and φ, avoiding the discontinuity at ±π.
    - 1 channel: whether that pair distance is finite (1 for normal coordinates, 0 for NaN).
      NaNs in the angles are set to 0.

    Args:
        xyz_t: (B, L, 3, 3), N/CA/C, in Å
    """
    c6d = xyz_to_c6d(xyz_t)
    dist = c6d["dist"]
    dmin, dmax, nbins = 2.0, 20.0, 36
    edges = torch.linspace(dmin + (dmax - dmin) / nbins, dmax, nbins, device=xyz_t.device)
    db = torch.bucketize(dist.clamp(max=dmax + 1), edges)
    dist_oh = F.one_hot(db.clamp(0, nbins), num_classes=nbins + 1).float()
    ang = torch.stack(
        [
            torch.sin(c6d["omega"]), torch.cos(c6d["omega"]),
            torch.sin(c6d["theta"]), torch.cos(c6d["theta"]),
            torch.sin(c6d["phi"]), torch.cos(c6d["phi"]),
        ],
        dim=-1,
    )
    missing = (~torch.isnan(dist)).float().unsqueeze(-1)
    ang = torch.nan_to_num(ang, nan=0.0)
    return torch.cat([dist_oh, ang, missing], dim=-1)


def t1d_channels(experiment: str) -> int:
    """Width of `build_t1d` for one training task.

    Base and active-site tasks are 22 (one-hot with the time scalar in the last
    channel). PPI appends one hotspot channel (23). Fold conditioning appends a
    4-class secondary-structure one-hot: helix, strand, coil, mask (26).
    """
    if experiment == "ppi":
        return 23
    if experiment == "fold":
        return 26
    return 22


def ss_onehot(xyz: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Coarse secondary structure from Cα geometry, shape (B, L, 4).

    Classes are helix, strand, coil, mask. Helix is a Cα(i)–Cα(i+4) distance in
    [5.0, 6.5] Å. Strand is an extended stretch (Cα(i)–Cα(i+2) ≥ 6 Å and
    Cα(i)–Cα(i+4) ≥ 9 Å) that was not already called helix. Everything else on a
    real residue is coil. Padding is the mask class.
    """
    ca = xyz[..., 1, :]
    B, L, _ = ca.shape
    labels = torch.full((B, L), 2, device=ca.device, dtype=torch.long)
    if L >= 5:
        d4 = torch.linalg.norm(ca[:, 4:] - ca[:, :-4], dim=-1)
        helix_pair = (d4 >= 5.0) & (d4 <= 6.5)
        helix = torch.zeros(B, L, dtype=torch.bool, device=ca.device)
        span = L - 4
        for k in range(5):
            helix[:, k : k + span] |= helix_pair
        labels = torch.where(helix, torch.zeros_like(labels), labels)
        d2 = torch.linalg.norm(ca[:, 2:] - ca[:, :-2], dim=-1)
        extended = torch.zeros(B, L, dtype=torch.bool, device=ca.device)
        for k in range(5):
            extended[:, k : k + span] |= d4 >= 9.0
        strand = torch.zeros(B, L, dtype=torch.bool, device=ca.device)
        span2 = L - 2
        for k in range(3):
            strand[:, k : k + span2] |= d2 >= 6.0
        strand = strand & extended & ~helix
        labels = torch.where(strand, torch.ones_like(labels), labels)
    if pad_mask is not None:
        labels = torch.where(pad_mask, labels, torch.full_like(labels, 3))
    return F.one_hot(labels, num_classes=4).to(dtype=xyz.dtype)


def mask_sequence(seq: torch.Tensor, motif_mask: torch.Tensor) -> torch.Tensor:
    """Keep the amino acid only on motif residues; set the rest to `MASK_IDX` (21).

    This is the sequence input for conditional generation: the network can see the
    fragment that must stay fixed, and cannot see the sequence to be designed.
    """
    return torch.where(motif_mask, seq, torch.full_like(seq, MASK_IDX))


def assemble_model_inputs(
    xyz0: torch.Tensor,
    xyz_t: torch.Tensor,
    seq: torch.Tensor,
    t: torch.Tensor,
    motif_mask: torch.Tensor,
    T: int,
    xyz_selfcond: Optional[torch.Tensor] = None,
    hotspot: Optional[torch.Tensor] = None,
    ss: Optional[torch.Tensor] = None,
    idx: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Assemble one forward input for the denoiser. `xyz0` is not part of the return value; sequence masking uses only `seq`.

    Returned keys:

    - seq: (B, L) masked sequence.
    - xyz_t: (B, L, 3, 3) coordinates at step t, in Å.
    - xyz_selfcond: the previous x_0 prediction. If omitted, all zeros, meaning this
      step has no self-conditioning. When it is enabled, the training loop first runs
      the network at t+1 and fills in the detached prediction.
    - t: (B,) discrete step.
    - t1d, t2d: the two template features above. t2d looks only at xyz_t, not at self-conditioning.
    - motif_mask: passed through unchanged. The loss upweights these residues; the diffuser leaves them un-noised.
    - idx: (B, L) residue indices. Chain breaks jump by 200 when the batch supplies them.
    - msa: (B, 1, L, 22). RoseTTAFold expects an MSA. Unconditional design has no homologs,
      so this is a single row of one-hot of the masked sequence, with depth 1.

    Args:
        xyz_t: the noised backbone. Motif positions should still be the clean coordinates.
        seq: the unmasked true sequence.
    """
    seq_masked = mask_sequence(seq, motif_mask)
    t1d = build_t1d(seq_masked, t, motif_mask, T, hotspot=hotspot, ss=ss)
    t2d = build_t2d(xyz_t)
    if xyz_selfcond is None:
        xyz_selfcond = torch.zeros_like(xyz_t)
    B, L = seq.shape
    if idx is None:
        idx = torch.arange(L, device=seq.device)[None].expand(B, L)
    return {
        "seq": seq_masked,
        "xyz_t": xyz_t,
        "xyz_selfcond": xyz_selfcond,
        "t": t,
        "t1d": t1d,
        "t2d": t2d,
        "motif_mask": motif_mask,
        "idx": idx,
        "msa": F.one_hot(seq_masked, num_classes=N_AA).float().unsqueeze(1),
    }
