"""Denoisers used by the training loop.

`TinyRFDenoiser` is a self-contained x0-predictor so this repository can train
without compiling NVIDIA SE(3)-Transformer.  Its I/O matches the quantities in
Supplementary Methods Tables 4--5 (noisy frames in, denoised frames + optional
distogram out).

`RoseTTAFoldDenoiser` wraps the official `rfdiffusion.RoseTTAFoldModule` when
that package is installed, initializing from RF structure-prediction weights
as in the paper.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import build_t2d, sinusoidal_time
from .geometry import N_AA, frames_to_xyz, xyz_to_frames


class PairBiasBlock(nn.Module):
    """Pre-norm self-attention whose logits are shifted by a pair feature."""

    def __init__(self, d_res: int, n_heads: int, d_pair: int):
        super().__init__()
        if d_res % n_heads != 0:
            raise ValueError(f"d_res={d_res} must be divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_res // n_heads
        self.norm1 = nn.LayerNorm(d_res)
        self.qkv = nn.Linear(d_res, 3 * d_res)
        self.out = nn.Linear(d_res, d_res)
        self.pair_bias = nn.Linear(d_pair, n_heads)
        self.norm2 = nn.LayerNorm(d_res)
        self.ff = nn.Sequential(nn.Linear(d_res, 4 * d_res), nn.GELU(), nn.Linear(4 * d_res, d_res))

    def forward(self, h: torch.Tensor, pair: torch.Tensor, key_pad: Optional[torch.Tensor]) -> torch.Tensor:
        B, L, D = h.shape
        x = self.norm1(h)
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn + self.pair_bias(pair).permute(0, 3, 1, 2)
        if key_pad is not None:
            attn = attn.masked_fill(key_pad[:, None, None, :], torch.finfo(attn.dtype).min)
        w = torch.softmax(attn, dim=-1)
        w = torch.nan_to_num(w, nan=0.0)
        msg = torch.matmul(w, v).permute(0, 2, 1, 3).reshape(B, L, D)
        h = h + self.out(msg)
        h = h + self.ff(self.norm2(h))
        return h


class TinyRFDenoiser(nn.Module):
    """Pair-biased residue network that predicts denoised N-CA-C frames.

    The trunk sees sequence position, the 44-d pair geometry (`t2d`), and an
    optional self-conditioning draft. Motif frames are not overwritten: keeping
    them is a loss, so an upweighted motif term actually trains the network.
    """

    def __init__(
        self,
        d_res: int = 128,
        d_pair: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        T: int = 200,
        t1d_dim: int = 22,
    ):
        super().__init__()
        self.T = T
        self.d_res = d_res
        self.t1d_dim = t1d_dim
        self.aa_emb = nn.Embedding(N_AA, d_res)
        self.time_mlp = nn.Sequential(nn.Linear(32, d_res), nn.SiLU(), nn.Linear(d_res, d_res))
        self.idx_mlp = nn.Sequential(nn.Linear(32, d_res), nn.SiLU(), nn.Linear(d_res, d_res))
        self.coord_mlp = nn.Sequential(nn.Linear(9, d_res), nn.SiLU(), nn.Linear(d_res, d_res))
        self.selfcond_mlp = nn.Sequential(nn.Linear(12, d_res), nn.SiLU(), nn.Linear(d_res, d_res))
        self.t1d_mlp = nn.Linear(t1d_dim, d_res)
        self.pair_proj = nn.Linear(44, d_pair)
        self.sc_pair = nn.Linear(1, d_pair)
        self.layers = nn.ModuleList([PairBiasBlock(d_res, n_heads, d_pair) for _ in range(n_layers)])
        self.ca_out = nn.Linear(d_res, 3)
        self.rot6_out = nn.Linear(d_res, 6)
        self.dist_head = nn.Linear(d_pair + 2 * d_res, 37)
        self.omega_head = nn.Linear(d_pair + 2 * d_res, 37)
        self.theta_head = nn.Linear(d_pair + 2 * d_res, 37)
        self.phi_head = nn.Linear(d_pair + 2 * d_res, 19)

    @staticmethod
    def _local_coords(xyz: torch.Tensor) -> torch.Tensor:
        ca = xyz[..., 1, :]
        return (xyz - ca.unsqueeze(-2)).reshape(*xyz.shape[:-2], 9)

    @staticmethod
    def _rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
        a1 = x[..., :3]
        a2 = x[..., 3:]
        e1 = a1 / (torch.linalg.norm(a1, dim=-1, keepdim=True) + 1e-8)
        e2 = a2 - (e1 * a2).sum(dim=-1, keepdim=True) * e1
        e2 = e2 / (torch.linalg.norm(e2, dim=-1, keepdim=True) + 1e-8)
        e3 = torch.cross(e1, e2, dim=-1)
        return torch.stack([e1, e2, e3], dim=-1)

    def forward(
        self,
        xyz_t: torch.Tensor,
        t: torch.Tensor,
        seq: torch.Tensor,
        motif_mask: torch.Tensor,
        t1d: torch.Tensor,
        t2d: Optional[torch.Tensor] = None,
        xyz_selfcond: Optional[torch.Tensor] = None,
        selfcond_mask: Optional[torch.Tensor] = None,
        idx: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        del motif_mask  # supervised by the loss, not clamped in the output
        B, L = seq.shape
        if t1d.shape[-1] != self.t1d_dim:
            raise ValueError(f"t1d width {t1d.shape[-1]} != model t1d_dim {self.t1d_dim}")
        if idx is None:
            idx = torch.arange(L, device=seq.device)[None].expand(B, L)
        if t2d is None:
            t2d = build_t2d(xyz_t)
        if t2d.dim() == 5:
            t2d = t2d[:, 0]
        h = self.aa_emb(seq.clamp(0, N_AA - 1))
        t_emb = sinusoidal_time(t.view(B, 1).expand(B, L), dim=32, T=self.T)
        h = h + self.time_mlp(t_emb)
        h = h + self.idx_mlp(sinusoidal_time(idx, dim=32, T=self.T))
        h = h + self.coord_mlp(self._local_coords(xyz_t))
        h = h + self.t1d_mlp(t1d)

        if xyz_selfcond is None:
            xyz_selfcond = torch.zeros_like(xyz_t)
        sc_ca = xyz_selfcond[..., 1, :]
        sc_in = torch.cat([self._local_coords(xyz_selfcond), sc_ca - xyz_t[..., 1, :]], dim=-1)
        sc = self.selfcond_mlp(sc_in)
        if selfcond_mask is not None:
            sc = sc * selfcond_mask.to(dtype=sc.dtype).view(B, 1, 1)
        h = h + sc

        pair = self.pair_proj(t2d)
        sc_dist = torch.cdist(sc_ca, sc_ca).unsqueeze(-1)
        sc_pair = self.sc_pair(sc_dist)
        if selfcond_mask is not None:
            sc_pair = sc_pair * selfcond_mask.to(dtype=sc_pair.dtype).view(B, 1, 1, 1)
        pair = pair + sc_pair

        key_pad = None if pad_mask is None else ~pad_mask
        for layer in self.layers:
            h = layer(h, pair, key_pad)

        R_t, t_ca = xyz_to_frames(xyz_t)
        R_upd = self._rot6d_to_matrix(self.rot6_out(h))
        R_pred = torch.einsum("blij,bljk->blik", R_upd, R_t)
        t_pred = t_ca + self.ca_out(h)
        xyz_pred = frames_to_xyz(R_pred, t_pred)

        pair_h = torch.cat(
            [pair, h[:, :, None, :].expand(B, L, L, h.size(-1)), h[:, None, :, :].expand(B, L, L, h.size(-1))],
            dim=-1,
        )
        logits = {
            "dist": self.dist_head(pair_h).permute(0, 3, 1, 2),
            "omega": self.omega_head(pair_h).permute(0, 3, 1, 2),
            "theta": self.theta_head(pair_h).permute(0, 3, 1, 2),
            "phi": self.phi_head(pair_h).permute(0, 3, 1, 2),
        }
        return {"xyz": xyz_pred, "logits_c6d": logits, "intermediate": [xyz_pred]}


class RoseTTAFoldDenoiser(nn.Module):
    """Thin wrapper around the official RFdiffusion RoseTTAFold module."""

    def __init__(self, T: int = 200, ckpt: Optional[str] = None, d_t1d: int = 22, d_t2d: int = 44):
        super().__init__()
        try:
            from rfdiffusion.RoseTTAFoldModel import RoseTTAFoldModule
        except ImportError as e:
            raise ImportError(
                "RoseTTAFoldDenoiser requires the official rfdiffusion package. "
                "Install https://github.com/RosettaCommons/RFdiffusion and pass model=tiny "
                "to use the educational denoiser."
            ) from e
        self.T = T
        self.net = RoseTTAFoldModule(
            n_extra_block=4,
            n_main_block=32,
            n_ref_block=4,
            d_msa=256,
            d_msa_full=64,
            d_pair=128,
            d_templ=64,
            n_head_msa=8,
            n_head_pair=4,
            n_head_templ=4,
            d_hidden=32,
            d_hidden_templ=32,
            p_drop=0.15,
            d_t1d=d_t1d,
            d_t2d=d_t2d,
            T=T,
            use_motif_timestep=True,
            freeze_track_motif=False,
            SE3_param_full={
                "num_layers": 1, "num_channels": 32, "num_degrees": 2, "n_heads": 4,
                "div": 4, "l0_in_features": 8, "l0_out_features": 8, "l1_in_features": 3,
                "l1_out_features": 2, "num_edge_features": 32,
            },
            SE3_param_topk={
                "num_layers": 1, "num_channels": 32, "num_degrees": 2, "n_heads": 4,
                "div": 4, "l0_in_features": 64, "l0_out_features": 64, "l1_in_features": 3,
                "l1_out_features": 2, "num_edge_features": 64,
            },
            input_seq_onehot=True,
        )
        if ckpt:
            state = torch.load(ckpt, map_location="cpu")
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            missing, unexpected = self.net.load_state_dict(state, strict=False)
            print(f"Loaded RF weights from {ckpt}. missing={len(missing)} unexpected={len(unexpected)}")

    def forward(
        self,
        xyz_t,
        t,
        seq,
        motif_mask,
        t1d,
        t2d=None,
        xyz_selfcond=None,
        selfcond_mask=None,
        msa=None,
        idx=None,
        **kwargs,
    ):
        B, L = seq.shape
        if seq.dtype in (torch.int32, torch.int64, torch.long):
            seq = F.one_hot(seq.clamp(0, N_AA - 1), num_classes=N_AA).float()
        else:
            seq = seq.float()
        if msa is None:
            msa = seq.unsqueeze(1)
        # MSA_emb.Linear expects 22 + 22 + 2 + 2. A single design sequence is
        # stored twice, with the four extra channels left at zero.
        if msa.shape[-1] == N_AA:
            msa = torch.cat([msa, msa, msa.new_zeros(*msa.shape[:-1], 4)], dim=-1)
        elif msa.shape[-1] < 48:
            msa = F.pad(msa, (0, 48 - msa.shape[-1]))
        msa_full = seq.unsqueeze(1)
        if msa_full.shape[-1] < 25:
            msa_full = F.pad(msa_full, (0, 25 - msa_full.shape[-1]))
        if idx is None:
            idx = torch.arange(L, device=seq.device)[None].expand(B, L)
        xyz = xyz_t
        if xyz.shape[-2] < 27:
            xyz = F.pad(xyz, (0, 0, 0, 27 - xyz.shape[-2]))
        xyz_sc = xyz_selfcond if xyz_selfcond is not None else torch.zeros_like(xyz)
        if selfcond_mask is not None:
            xyz_sc = xyz_sc * selfcond_mask.to(dtype=xyz_sc.dtype).view(B, 1, 1, 1)
        if xyz_sc.shape[-2] < 27:
            xyz_sc = F.pad(xyz_sc, (0, 0, 0, 27 - xyz_sc.shape[-2]))
        alpha_t = torch.zeros(B, 1, L, 30, device=seq.device)
        t2d = t2d if t2d is not None else torch.zeros(B, 1, L, L, 44, device=seq.device)
        if t2d.dim() == 4:
            t2d = t2d.unsqueeze(1)
        if t1d.dim() == 3:
            t1d = t1d.unsqueeze(1)
        net_kwargs = dict(
            t1d=t1d,
            t2d=t2d,
            xyz_t=xyz_sc,
            alpha_t=alpha_t,
            motif_mask=motif_mask,
            cyclic_reses=None,
        )
        try:
            logits, logits_aa, logits_exp, xyz_out, alpha_s, lddt = self.net(
                msa, msa_full, seq, xyz, idx, t, **net_kwargs
            )
        except TypeError:
            net_kwargs.pop("cyclic_reses")
            logits, logits_aa, logits_exp, xyz_out, alpha_s, lddt = self.net(
                msa, msa_full, seq, xyz, idx, t, **net_kwargs
            )
        # xyz_out: (n_block, B, L, 3, 3) intermediate structures
        if xyz_out.dim() == 5:
            xyz_final = xyz_out[-1]
            intermediate = [xyz_out[i] for i in range(xyz_out.shape[0])]
        else:
            xyz_final = xyz_out
            intermediate = [xyz_out]
        dist, omega, theta, phi = logits
        return {
            "xyz": xyz_final[..., :3, :],
            "logits_c6d": {"dist": dist, "omega": omega, "theta": theta, "phi": phi},
            "intermediate": [x[..., :3, :] for x in intermediate],
        }


def build_model(name: str, T: int, ckpt: Optional[str] = None, t1d_dim: int = 22) -> nn.Module:
    if name == "tiny":
        return TinyRFDenoiser(T=T, t1d_dim=t1d_dim)
    if name == "rf":
        return RoseTTAFoldDenoiser(T=T, ckpt=ckpt, d_t1d=t1d_dim)
    raise ValueError(name)
