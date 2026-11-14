"""Training losses: L_Frame + w2D L_2D (Supplementary Methods §4.1)."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .config import LossConfig
from .geometry import generate_cbeta, rotation_frobenius, xyz_to_frames


def _dihedral(a, b, c, d, eps: float = 1e-8):
    b0 = a - b
    b1 = c - b
    b2 = d - c
    b1 = b1 / (torch.linalg.norm(b1, dim=-1, keepdim=True) + eps)
    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1
    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)


def _planar(a, b, c, eps: float = 1e-8):
    v = a - b
    w = c - b
    v = v / (torch.linalg.norm(v, dim=-1, keepdim=True) + eps)
    w = w / (torch.linalg.norm(w, dim=-1, keepdim=True) + eps)
    return torch.acos((v * w).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6))


def xyz_to_c6d(xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Inter-residue dist / omega / theta / phi from N-CA-C.

    xyz: (B, L, 3, 3)
    """
    N, Ca, C = xyz[..., 0, :], xyz[..., 1, :], xyz[..., 2, :]
    Cb = generate_cbeta(N, Ca, C)
    dist = torch.cdist(Cb, Cb)
    B, L = dist.shape[:2]
    # Pairwise geometries: (B, L, L)
    Ca_i = Ca[:, :, None, :].expand(B, L, L, 3)
    Ca_j = Ca[:, None, :, :].expand(B, L, L, 3)
    Cb_i = Cb[:, :, None, :].expand(B, L, L, 3)
    Cb_j = Cb[:, None, :, :].expand(B, L, L, 3)
    N_i = N[:, :, None, :].expand(B, L, L, 3)
    omega = _dihedral(Ca_i, Cb_i, Cb_j, Ca_j)
    theta = _dihedral(N_i, Ca_i, Cb_i, Cb_j)
    phi = _planar(Ca_i, Cb_i, Cb_j)
    return {"dist": dist, "omega": omega, "theta": theta, "phi": phi}


def c6d_to_bins(c6d: Dict[str, torch.Tensor], dmax: float = 20.0) -> Dict[str, torch.Tensor]:
    dist = c6d["dist"].clamp(max=dmax)
    # 36 bins of 0.5 Å from 2--20 plus one overflow bin -> 37
    db = torch.bucketize(dist, torch.linspace(2.5, 20.0, 36, device=dist.device))
    astep = 2.0 * torch.pi / 36
    ob = torch.bucketize(c6d["omega"], torch.linspace(-torch.pi + astep, torch.pi, 36, device=dist.device))
    tb = torch.bucketize(c6d["theta"], torch.linspace(-torch.pi + astep, torch.pi, 36, device=dist.device))
    pb = torch.bucketize(c6d["phi"], torch.linspace(astep, torch.pi, 18, device=dist.device))
    no_contact = dist >= dmax
    ob = torch.where(no_contact, torch.full_like(ob, 36), ob)
    tb = torch.where(no_contact, torch.full_like(tb, 36), tb)
    pb = torch.where(no_contact, torch.full_like(pb, 18), pb)
    return {"dist": db.long(), "omega": ob.long(), "theta": tb.long(), "phi": pb.long()}


def frame_distance(
    xyz_true: torch.Tensor,
    xyz_pred: torch.Tensor,
    cfg: LossConfig,
    clamp: bool = True,
    residue_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """d_Frame averaged over residues. xyz: (B, L, 3, 3)."""
    R_true, t_true = xyz_to_frames(xyz_true)
    R_pred, t_pred = xyz_to_frames(xyz_pred)
    trans = torch.linalg.norm(t_true - t_pred, dim=-1)
    if clamp:
        trans = torch.clamp(trans, max=cfg.d_clamp)
    rot = rotation_frobenius(R_pred, R_true)
    per = cfg.w_trans * trans.square() + cfg.w_rot * rot
    if residue_weight is not None:
        per = per * residue_weight
        return per.sum(dim=-1) / residue_weight.sum(dim=-1).clamp_min(1.0)
    return per.mean(dim=-1)


def loss_2d(
    logits: Dict[str, torch.Tensor],
    xyz_true: torch.Tensor,
    pair_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-entropy on distogram / anglegrams. logits[k]: (B, C, L, L)."""
    bins = c6d_to_bins(xyz_to_c6d(xyz_true))
    total = 0.0
    n = 0
    for key, n_class in (("dist", 37), ("omega", 37), ("theta", 37), ("phi", 19)):
        if key not in logits:
            continue
        lg = logits[key]
        tgt = bins[key]
        B, _, L, _ = lg.shape
        loss = F.cross_entropy(lg.reshape(B, n_class, L * L), tgt.reshape(B, L * L), reduction="none")
        loss = loss.reshape(B, L, L)
        if pair_mask is not None:
            loss = loss * pair_mask
            denom = pair_mask.sum(dim=(-2, -1)).clamp_min(1.0)
            total = total + loss.sum(dim=(-2, -1)) / denom
        else:
            total = total + loss.mean(dim=(-2, -1))
        n += 1
    return total / max(n, 1)


def diffusion_loss(
    xyz_true: torch.Tensor,
    xyz_pred: torch.Tensor,
    cfg: LossConfig,
    logits_c6d: Optional[Dict[str, torch.Tensor]] = None,
    motif_mask: Optional[torch.Tensor] = None,
    motif_weight: float = 1.0,
    intermediate: Optional[list] = None,
    pad_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """L_Diffusion = L_Frame + w2D L_2D.

    If `intermediate` is a list of xyz predictions from structure blocks,
    L_Frame uses exponential weighting with gamma (SI §4.1).
    Motif residues are upweighted inside L_Frame, which is the displacement
    term. Padding has weight 0. L_2D is cross-entropy of the predicted bins
    against the true structure, so its gradient does not flow through
    predicted coordinates.
    """
    B, L = xyz_true.shape[:2]
    clamp = bool(torch.rand(1).item() < cfg.p_clamp)
    res_w = torch.ones(B, L, device=xyz_true.device, dtype=xyz_true.dtype)
    if pad_mask is not None:
        res_w = res_w * pad_mask.to(dtype=res_w.dtype)
    if motif_mask is not None:
        res_w = torch.where(motif_mask, res_w * motif_weight, res_w)

    preds = intermediate if intermediate else [xyz_pred]
    I = len(preds)
    weights = torch.tensor(
        [cfg.gamma ** (I - 1 - i) for i in range(I)],
        device=xyz_true.device,
        dtype=xyz_true.dtype,
    )
    weights = weights / weights.sum()
    l_frame = 0.0
    for w, pred in zip(weights, preds):
        l_frame = l_frame + w * frame_distance(xyz_true, pred, cfg, clamp=clamp, residue_weight=res_w)
    l_frame = l_frame.mean()

    pair_mask = None
    if pad_mask is not None:
        pair_mask = pad_mask[:, :, None] & pad_mask[:, None, :]

    out = {"loss_frame": l_frame, "loss_2d": xyz_true.new_zeros(())}
    if logits_c6d is not None:
        out["loss_2d"] = loss_2d(logits_c6d, xyz_true, pair_mask=pair_mask).mean()
    out["loss"] = out["loss_frame"] + cfg.w_2d * out["loss_2d"]
    with torch.no_grad():
        ca_err = torch.linalg.norm(xyz_true[..., 1, :] - xyz_pred[..., 1, :], dim=-1)
        if pad_mask is not None:
            w = pad_mask.to(dtype=ca_err.dtype)
            out["ca_rmsd"] = torch.sqrt((ca_err.square() * w).sum() / w.sum().clamp_min(1.0))
        else:
            out["ca_rmsd"] = torch.sqrt(ca_err.square().mean())
        if motif_mask is not None and motif_mask.any():
            out["motif_ca_rmsd"] = torch.sqrt(ca_err[motif_mask].square().mean())
        else:
            out["motif_ca_rmsd"] = xyz_true.new_zeros(())
    return out
