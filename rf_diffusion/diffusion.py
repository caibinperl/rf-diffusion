"""SE(3) diffusion: Euclidean DDPM on the CA translation, IGSO(3) on the residue orientation.

Corresponds to Supplementary Methods §§2.2–2.3 and `rfdiffusion/diffusion.py`.
Each residue is one rigid frame. Translation moves only the CA; rotation turns
about that residue's own CA. N/C are placed back from the frame with ideal bond
geometry, so a noised backbone stays valid.

Do not mix the two clocks:
- For the translation schedule, t is a discrete step, 1…T (1 is the least noise,
  T is almost pure noise).
- Rotation looks up `IGSO3.sigma` with continuous time t/T ∈ (0, 1].

Residues with `motif_mask` True are a fixed motif: the forward process does not
noise them, and the reverse process does not update them. Coordinates exposed
to callers are in Å; inside the schedule they are multiplied by `crd_scale`
(default 0.25) before translational diffusion.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from .config import DiffuserConfig
from .geometry import frames_to_xyz, rot_matmul, xyz_to_frames
from .so3 import IGSO3, matrix_to_rotvec, rotvec_to_matrix


def center_noise(eps: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Subtract the mean of `eps` over mobile residues so translational noise does not move the COM.

    `mask` True marks a fixed residue. Those entries are set to zero, and the mean
    is taken only over the complement. With no mask, every residue is mobile.
    An empty mobile set leaves the tensor at zero.
    """
    if mask is None:
        return eps - eps.mean(dim=-2, keepdim=True)
    mobile = (~mask).to(dtype=eps.dtype).unsqueeze(-1)
    denom = mobile.sum(dim=-2, keepdim=True).clamp_min(1.0)
    mean = (eps * mobile).sum(dim=-2, keepdim=True) / denom
    return (eps - mean) * mobile


def get_beta_schedule(T: int, b0: float, bT: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear β schedule for the translation part, scaled from the paper default T=200.

    β_t increases linearly from b0 to bT over T steps. α_t = 1 − β_t,
    ᾱ_t = ∏_{s=1}^{t} α_s. Closed-form noising uses ᾱ_t:
    x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε.

    b0 and bT are first multiplied by 200/T. At T=200 this matches the paper.
    Fewer steps make each β larger, and more steps make each β smaller, so the
    noise accumulated along the whole chain stays on the same scale. T is at least 15.

    Returns:
        beta, alpha, alphabar: each (T,), index 0 corresponds to t=1.
    """
    assert T >= 15
    b0 = b0 * 200 / T
    bT = bT * 200 / T
    beta = torch.linspace(b0, bT, T)
    alpha = 1.0 - beta
    alphabar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alphabar


class SE3Diffuser:
    """The forward process takes a structure to step t; the reverse walks back to t−1 given x_0.

    Translation uses the DDPM Gaussian closed form and the posterior
    q(x_{t−1} | x_t, x_0). Rotation samples directly from IGSO(3) in the forward
    direction (not a step-by-step random walk); the reverse uses the score of the
    relative rotation R_t R_0ᵀ for one reverse-SDE step (SI Algorithm 1).
    When `so3_type == "gaussian"`, rotation falls back to an isotropic Gaussian
    in the tangent space.
    """

    def __init__(self, cfg: DiffuserConfig):
        self.cfg = cfg
        self.T = cfg.T
        self.crd_scale = cfg.crd_scale
        self.beta, self.alpha, self.alphabar = get_beta_schedule(cfg.T, cfg.b_0, cfg.b_T)
        self.so3 = IGSO3(
            T=cfg.T,
            min_sigma=cfg.min_sigma,
            max_sigma=cfg.max_sigma,
            min_b=cfg.min_b,
            max_b=cfg.max_b,
            cache_dir=cfg.cache_dir,
            schedule=cfg.so3_schedule_type,
            L=cfg.truncation_level,
            num_omega=cfg.num_omega,
            num_sigma=cfg.num_sigma,
        )

    def to(self, device: torch.device) -> "SE3Diffuser":
        """Move the translation schedule tables onto the given device. IGSO(3) lookups stay in numpy."""
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alphabar = self.alphabar.to(device)
        return self

    def _gather(self, schedule: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Index a schedule table by a 1-based diffusion step. t=1 corresponds to index 0."""
        return schedule[t.long() - 1].to(dtype=torch.float32)

    def forward_translations(
        self,
        xyz0: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Closed-form noising of the CA translation. Moves position only; does not change residue orientation.

        In the scaled coordinate system (the caller has already multiplied by `crd_scale`):

            CA_t = √ᾱ_t · CA_0 + √(1−ᾱ_t) · ε,   ε ~ N(0, I)

        The same displacement is added to N, CA, and C, a rigid translation of the residue.
        Residues with `mask` True have their displacement set to zero.

        Args:
            xyz0: (B, L, 3, 3), atom order N, CA, C
            t: (B,) discrete steps, 1…T
            mask: (B, L), True means fixed and not noised
        Returns:
            (B, L, 3, 3), in the same coordinate system as xyz0
        """
        ca0 = xyz0[..., 1, :]
        ab = self._gather(self.alphabar, t).view(-1, 1, 1)
        eps = center_noise(torch.randn_like(ca0), mask)
        cat = torch.sqrt(ab) * ca0 + torch.sqrt((1.0 - ab).clamp_min(0.0)) * eps
        delta = cat - ca0
        if mask is not None:
            delta = delta * (~mask).unsqueeze(-1).to(delta.dtype)
        return xyz0 + delta.unsqueeze(-2)

    def _sample_rotvec(self, t_b: int, n: int, device, dtype) -> torch.Tensor:
        """Draw n rotation vectors, the rotational noise at this discrete step.

        By default, sample from IGSO(3) (uniform axis, angle from the inverse CDF).
        When `so3_type == "gaussian"`, use a tangent-space Gaussian instead: N(0, σ(t/T)² I).
        """
        if self.cfg.so3_type == "gaussian":
            sig = float(self.so3.sigma(torch.tensor(t_b / self.T)))
            return torch.randn(n, 3, device=device, dtype=dtype) * sig
        return torch.tensor(
            self.so3.sample_rotvec([t_b], n_samples=n)[0],
            dtype=dtype,
            device=device,
        )

    def forward_rotations(
        self,
        xyz0: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply rotational noise about each residue's own CA, then place ideal N/C back.

        R_t = R_noise @ R_0 (left multiply; the noise acts on the lab-frame side).
        CA stays the CA from the input; N/C are replaced by ideal geometry, which
        happens even when the angle is 0. When `mask` is True the rotation vector
        is zeroed, but this step still idealizes, so motif coordinates have to be
        written back by the `where` at the end of `forward_noise`.

        The batch is looped because each sample has its own t, and IGSO(3) sampling
        looks up the numpy table per step.
        """
        B, L = xyz0.shape[:2]
        R0, ca = xyz_to_frames(xyz0)
        xyz_out = xyz0.clone()
        for b in range(B):
            tb = int(t[b].item())
            rotvec = self._sample_rotvec(tb, L, xyz0.device, xyz0.dtype)
            if mask is not None:
                rotvec = rotvec * (~mask[b]).unsqueeze(-1).to(rotvec.dtype)
            R_noise = rotvec_to_matrix(rotvec)
            R_t = rot_matmul(R_noise, R0[b])
            xyz_out[b] = frames_to_xyz(R_t, ca[b])
        return xyz_out

    def forward_noise(
        self,
        xyz0_angstrom: torch.Tensor,
        t: torch.Tensor,
        motif_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Algorithm 3: draw a full backbone at step t in one shot. Input and output are both in Å.

        Translation and rotation are independent and both act on the original
        structure, rather than being stacked serially:

        1. Multiply coordinates by `crd_scale` to enter scaled space.
        2. The translation branch produces a noised CA.
        3. The rotation branch turns the original frame. CA is still at the old
           position, and N/C are already ideal coordinates.
        4. Translate the rotated residue so its CA lines up with the translation branch's CA.
        5. Replace motif residues, the whole row, with the scaled original coordinates
           (true bond lengths, not only the CA).
        6. Divide by `crd_scale` to return to Å.

        Args:
            xyz0_angstrom: (B, L, 3, 3)
            t: (B,) in 1…T
            motif_mask: (B, L), True = fixed motif
        """
        xyz = xyz0_angstrom * self.crd_scale
        xyz_t = self.forward_translations(xyz, t, mask=motif_mask)
        # Rotation is applied to the *original* scaled frames, then translation
        # is re-applied so CA matches the Euclidean-noised location.
        xyz_rot = self.forward_rotations(xyz, t, mask=motif_mask)
        ca_t = xyz_t[..., 1, :]
        ca_rot = xyz_rot[..., 1, :]
        xyz_t = xyz_rot + (ca_t - ca_rot).unsqueeze(-2)
        if motif_mask is not None:
            m = motif_mask.unsqueeze(-1).unsqueeze(-1)
            xyz_t = torch.where(m, xyz, xyz_t)
        return xyz_t / self.crd_scale

    def graft_forward_rotation(
        self,
        xyz0_angstrom: torch.Tensor,
        ca_angstrom: torch.Tensor,
        t: torch.Tensor,
        motif_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Keep caller-supplied CA positions and redraw orientations from the forward process at `t`.

        A Gaussian translation chain is Markov, so one reverse step from x_{t+1} with the true
        x_0 is an unbiased sample of q(x_t | x_0). Rotation is not: the forward process draws
        directly from IGSO(3) at σ(t/T), and one reverse-SDE step does not land on that marginal.
        Self-conditioning therefore keeps the posterior CA and replaces the rotation here.
        Motif rows are copied from `xyz0_angstrom`. Both inputs are in Å.
        """
        xyz = xyz0_angstrom * self.crd_scale
        ca = ca_angstrom * self.crd_scale
        xyz_rot = self.forward_rotations(xyz, t, mask=motif_mask)
        xyz_t = xyz_rot + (ca - xyz_rot[..., 1, :]).unsqueeze(-2)
        if motif_mask is not None:
            m = motif_mask.unsqueeze(-1).unsqueeze(-1)
            xyz_t = torch.where(m, xyz, xyz_t)
        return xyz_t / self.crd_scale

    def posterior_mean_variance(
        self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean and variance of the translation posterior q(x_{t−1} | x_t, x_0).

        DDPM Gaussian posterior. x is the CA coordinate (scaled space), and t is 1-based:

            μ = c1 · x_0 + c2 · x_t
            c1 = √ᾱ_{t−1} · β_t / (1 − ᾱ_t)
            c2 = √α_t · (1 − ᾱ_{t−1}) / (1 − ᾱ_t)
            σ² = β_t · (1 − ᾱ_{t−1}) / (1 − ᾱ_t)

        At t=1 there is no earlier step, so ᾱ_0 is taken to be 1. Then c1=1, c2=0,
        σ²=0, and the mean is x_0.

        Args:
            x0, xt: (B, L, 3)
            t: (B,)
        Returns:
            mean, var: both (B, L, 3); var is the same on all three coordinates.
        """
        beta_t = self._gather(self.beta, t).view(-1, 1, 1)
        alpha_t = self._gather(self.alpha, t).view(-1, 1, 1)
        ab_t = self._gather(self.alphabar, t).view(-1, 1, 1)
        ab_tm1 = torch.ones_like(ab_t)
        t_gt1 = t > 1
        if t_gt1.any():
            ab_tm1[t_gt1] = self._gather(self.alphabar, (t[t_gt1] - 1)).view(-1, 1, 1)
        mean = (
            torch.sqrt(ab_tm1) * beta_t / (1.0 - ab_t).clamp_min(1e-8)
        ) * x0 + (
            torch.sqrt(alpha_t) * (1.0 - ab_tm1) / (1.0 - ab_t).clamp_min(1e-8)
        ) * xt
        var = beta_t * (1.0 - ab_tm1) / (1.0 - ab_t).clamp_min(1e-8)
        return mean, var

    def reverse_translations(
        self,
        xyz_t: torch.Tensor,
        xyz_0: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """Walk the CA from t to t−1 with the translation posterior. N/C take the same rigid displacement.

        CA_{t−1} = μ + √σ² · ε. `noise_scale` scales ε. When t≤1 the noise is forced
        to zero, because the posterior variance of that step is already 0. Residues
        with `mask` True have their displacement set to zero.

        xyz_* is in scaled space, shape (B, L, 3, 3). The return has the same shape;
        orientation has not been updated yet.
        """
        ca_t = xyz_t[..., 1, :]
        ca_0 = xyz_0[..., 1, :]
        mean, var = self.posterior_mean_variance(ca_0, ca_t, t)
        noise = center_noise(torch.randn_like(ca_t) * noise_scale, mask)
        noise = torch.where((t <= 1).view(-1, 1, 1), torch.zeros_like(noise), noise)
        ca_prev = mean + torch.sqrt(var.clamp_min(0.0)) * noise
        delta = ca_prev - ca_t
        if mask is not None:
            delta = delta * (~mask).unsqueeze(-1).to(delta.dtype)
        return xyz_t + delta.unsqueeze(-2)

    def reverse_rotations(
        self,
        xyz_t: torch.Tensor,
        xyz_0: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """One reverse-SDE step for rotation. The score comes from p(R_t | R_0) (SI Algorithm 1).

        The relative rotation R_t R_0ᵀ is exactly the turn the forward process stacked
        on. Its axis-angle length is ω, and the score points along the axis with
        length ∂ log f / ∂ω. That derivative is negative on (0, π], so the drift
        rotates the frame back toward R_0 instead of opening it further.

        Under the Gaussian approximation, score = −v / σ², the score of N(0, σ² I)
        on the rotation vector.

        Discrete update (continuous time s = t/T, step size Δs = 1/T):

            δ = g(s)² Δs · score + g(s) √Δs · z

        δ is left-multiplied onto R_t, then N/C are placed back with ideal geometry.
        This step does not move the CA. When t≤1, z is set to zero. Residues with
        `mask` True have δ set to zero.

        During training R_0 is the true structure; at sampling time R_0 should be
        replaced by the network's predicted x_0.
        """
        B, L = xyz_t.shape[:2]
        Rt, ca_t = xyz_to_frames(xyz_t)
        R0, _ = xyz_to_frames(xyz_0)
        xyz_out = xyz_t.clone()
        for b in range(B):
            tb = int(t[b].item())
            R0t = torch.einsum("nij,nkj->nik", Rt[b], R0[b])
            rotvec = matrix_to_rotvec(R0t)
            omega = torch.linalg.norm(rotvec, dim=-1).clamp_min(1e-8)
            if self.cfg.so3_type == "gaussian":
                sig = float(self.so3.sigma(torch.tensor(tb / self.T)))
                sn = -omega / (sig ** 2 + 1e-8)
            else:
                sn = torch.tensor(
                    self.so3.score_norm(tb, omega.detach().cpu().numpy()),
                    dtype=rotvec.dtype,
                    device=rotvec.device,
                )
            score = rotvec * (sn / omega).unsqueeze(-1)
            continuous_t = tb / self.T
            rot_g = self.so3.g(continuous_t).to(device=rotvec.device, dtype=rotvec.dtype)
            z = torch.randn_like(rotvec) * noise_scale
            if tb <= 1:
                z = torch.zeros_like(z)
            delta = (rot_g ** 2) * self.so3.step_size * score + rot_g * np.sqrt(self.so3.step_size) * z
            if mask is not None:
                delta = delta * (~mask[b]).unsqueeze(-1).to(delta.dtype)
            perturb = rotvec_to_matrix(delta)
            R_prev = rot_matmul(perturb, Rt[b])
            xyz_out[b] = frames_to_xyz(R_prev, ca_t[b])
        return xyz_out

    def reverse_step(
        self,
        xyz_t_angstrom: torch.Tensor,
        xyz_0_angstrom: torch.Tensor,
        t: torch.Tensor,
        motif_mask: Optional[torch.Tensor] = None,
        noise_scale: float = 1.0,
    ) -> torch.Tensor:
        """One reverse step: t → t−1. Input and output are both in Å. x_0 is the clean structure or the network prediction.

        As in the forward pass, translation and rotation are computed separately
        and then stitched: the rotation result supplies the new orientation and
        ideal N/C, the translation result supplies the new CA, and the residue is
        finally translated onto that CA. Motif residues are replaced, the whole
        row, by the coordinates in x_0.

        `noise_scale` scales both the translation posterior noise and the rotational Brownian increment.
        """
        xyz_t = xyz_t_angstrom * self.crd_scale
        xyz_0 = xyz_0_angstrom * self.crd_scale
        xyz_trans = self.reverse_translations(xyz_t, xyz_0, t, mask=motif_mask, noise_scale=noise_scale)
        xyz_rot = self.reverse_rotations(xyz_t, xyz_0, t, mask=motif_mask, noise_scale=noise_scale)
        ca = xyz_trans[..., 1, :]
        ca_rot = xyz_rot[..., 1, :]
        xyz_prev = xyz_rot + (ca - ca_rot).unsqueeze(-2)
        if motif_mask is not None:
            m = motif_mask.unsqueeze(-1).unsqueeze(-1)
            xyz_prev = torch.where(m, xyz_0, xyz_prev)
        return xyz_prev / self.crd_scale
