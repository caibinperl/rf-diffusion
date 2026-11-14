"""IGSO(3) utilities reconstructed from rfdiffusion/igso3.py.

The isotropic Gaussian on SO(3) is Brownian motion on the rotation group, i.e. the
heat kernel. The density depends only on the rotation angle ω ∈ (0, π]; the axis is
uniform on the sphere.

This module does two things:
1. Convert between axis-angle (rotation vectors) and 3×3 rotation matrices.
2. Precompute the angle CDF and the score, for forward noising samples and the reverse SDE.

Time is measured on two scales. Do not mix them when reading the code:
- The `t` argument of `f_igso3` / `igso3_density_angle` is the variance σ², not a diffusion step.
- In `IGSO3` methods, the discrete step is t ∈ {1, …, T}, continuous time is t/T ∈ [0, 1],
  the noise scale σ(t/T) comes from the linear schedule, and the variance is then σ².
"""

from __future__ import annotations

import os
import pickle
from typing import Dict

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciR


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Rotation vector → rotation matrix.

    For a rotation vector v ∈ R³, the direction is the axis and the norm ||v|| is the
    angle (radians; the scipy convention lands in [0, π]). The output maps the local
    frame into the lab frame and satisfies R = exp([v]_×).

    scipy accepts only CPU numpy, so the tensor is detached and then moved back to the
    original device / dtype. This conversion does not need gradients: in the reverse
    step the perturbation is added in rotation-vector space and only then turned into a matrix.

    Args:
        rotvec: (..., 3)
    Returns:
        (..., 3, 3)
    """
    shape = rotvec.shape[:-1]
    mat = SciR.from_rotvec(rotvec.detach().cpu().reshape(-1, 3).numpy()).as_matrix()
    return torch.tensor(mat, dtype=rotvec.dtype, device=rotvec.device).reshape(*shape, 3, 3)


def matrix_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix → rotation vector, the inverse of `rotvec_to_matrix`.

    The returned vector points along the axis, and its norm is the angle ω relative
    to the identity. Reverse diffusion uses it to read off the angle between the
    current frame and the predicted x0 frame, then looks up the score.

    Args:
        R: (..., 3, 3), a rotation matrix with determinant +1.
    Returns:
        (..., 3)
    """
    shape = R.shape[:-2]
    vec = SciR.from_matrix(R.detach().cpu().reshape(-1, 3, 3).numpy()).as_rotvec()
    return torch.tensor(vec, dtype=R.dtype, device=R.device).reshape(*shape, 3)


def f_igso3(omega: torch.Tensor, t: torch.Tensor, L: int = 200) -> torch.Tensor:
    """Value of the IGSO(3) heat kernel at angle ω, truncated to the first L irreps.

    The heat kernel on SO(3) with variance t = σ² depends only on the angle. Its
    expansion in spherical harmonics (characters) is

        f(ω, t) = Σ_{l=0}^{L-1} (2l+1) exp(-l(l+1) t / 2) · sin((l+1/2) ω) / sin(ω/2)

    l indexes the irreducible representations of SO(3), and
    χ_l(ω) = sin((l+1/2)ω) / sin(ω/2) is the character. The Laplacian eigenvalue is
    l(l+1); the Brownian-motion convention multiplies by another 1/2, so the decay
    rate is l(l+1)t/2. Larger l decays faster. At large t only the low-order terms
    survive, and L=200 is enough.

    f is a density with respect to Haar measure and has not yet been multiplied by
    the angle volume element. The marginal used for sampling multiplies by
    (1 - cos ω) / π in `igso3_density_angle`.

    The factor 1/2 in the exponent is the public `rfdiffusion/igso3.py` convention
    (`f_igso3`): σ here is √2 times the scale in Leach et al., so that variance
    t = σ² matches Brownian motion under the canonical SO(3) inner product.

    Args:
        omega: (N,) angles in radians, in (0, π]. As ω→0, sin(ω/2)→0, so clamp first.
        t: scalar variance σ², not a discrete diffusion step.
        L: truncation order.
    Returns:
        (N,) the unnormalized heat kernel f(ω, t).
    """
    ls = torch.arange(L, device=omega.device, dtype=omega.dtype)[None]
    omega = omega[:, None]
    # Small-angle safe sin(omega/2)
    half = omega.clamp(min=1e-8) / 2
    return (
        (2 * ls + 1)
        * torch.exp(-ls * (ls + 1) * t / 2)
        * torch.sin(omega * (ls + 0.5))
        / torch.sin(half)
    ).sum(dim=-1)


def igso3_density_angle(omega: np.ndarray, t: float, L: int = 200) -> np.ndarray:
    """Marginal density of the angle ω, used for the CDF and inverse-transform sampling.

    In axis-angle coordinates, Haar measure separates the axis (uniform on S²) from
    the angle, whose volume element is (1 - cos ω) dω / π. The heat kernel f is
    already isotropic, so the angle density is

        p(ω | t) = f(ω, t) · (1 - cos ω) / π ,    ω ∈ (0, π]

    Integrating over ω should give about 1. `calculate_igso3` then turns the discrete
    cumulative sum into a CDF.

    Args:
        omega: (N,) angles.
        t: variance σ².
    Returns:
        (N,) angle density.
    """
    om = torch.tensor(omega, dtype=torch.float64, device="cpu")
    f = f_igso3(om, torch.tensor(t, dtype=torch.float64, device="cpu"), L=L).numpy()
    return f * (1.0 - np.cos(omega)) / np.pi


def calculate_igso3(
    *,
    num_sigma: int,
    num_omega: int,
    min_sigma: float,
    max_sigma: float,
    L: int = 200,
) -> Dict[str, np.ndarray]:
    """Precompute the IGSO(3) lookup tables: angle CDF, score, and the expected score norm.

    The expansion sums L terms at every (σ, ω), which is too expensive to do on the
    fly during training, so it is evaluated once on a grid of σ and ω and stored as
    the five arrays below (lengths are given with each field):

    - discrete_omega: (num_omega,) angle grid. linspace(0, π) drops 0, so the nodes
      are π/N, 2π/N, …, π. ω=0 would make sin(ω/2) zero.
    - discrete_sigma: (num_sigma,) noise-scale grid, log-uniform on [min_sigma, max_sigma],
      likewise dropping the left endpoint so σ=0 is not stored.
    - cdf: (num_sigma, num_omega) cumulative distribution of the angle density at each σ.
      Rectangular integration cumsum(p) · Δω, then divided by the last node so that
      CDF(π) = 1. Sampling draws u uniform and does `np.interp(u, cdf, omega)`, the inverse CDF.
    - score_norm: (num_sigma, num_omega) ∂/∂ω log f(ω, σ²).
      This is the score of the heat kernel with respect to the angle, not the score
      of the marginal density p(ω): the volume element (1-cos ω) belongs to the
      measure and does not enter the score. Isotropy puts the score vector in the
      Lie algebra along the axis, with length equal to this scalar. The reverse step
      uses score(v) = score_norm(ω) · v / ||v||.
      Positions where f < 1e-20 are clamped to a constant, so the gradient there is 0.
    - exp_score_norms: (num_sigma,) root-mean-square score-vector norm,
      sqrt(E[score_norm²]), with the expectation taken under that σ's angle density.
      Present to match the upstream cache format; `IGSO3` methods in this file do not read it.

    Variance is always passed to `f_igso3` as t = σ².
    """
    discrete_omega = np.linspace(0, np.pi, num_omega + 1)[1:]
    discrete_sigma = 10 ** np.linspace(np.log10(min_sigma), np.log10(max_sigma), num_sigma + 1)[1:]
    pdf_vals = np.stack(
        [igso3_density_angle(discrete_omega, float(sig ** 2), L=L) for sig in discrete_sigma]
    )
    d_omega = np.pi / num_omega
    cdf_vals = np.cumsum(pdf_vals, axis=1) * d_omega
    cdf_vals = cdf_vals / np.clip(cdf_vals[:, -1:], 1e-12, None)

    score_norm = []
    for sig in discrete_sigma:
        omega_t = torch.tensor(discrete_omega, dtype=torch.float64, requires_grad=True)
        log_f = torch.log(
            f_igso3(omega_t, torch.tensor(float(sig ** 2), dtype=torch.float64), L=L).clamp_min(1e-20)
        )
        g = torch.autograd.grad(log_f.sum(), omega_t)[0]
        score_norm.append(g.detach().numpy())
    score_norm = np.stack(score_norm)
    exp_score_norms = np.sqrt(
        np.sum(score_norm ** 2 * pdf_vals, axis=1) / np.clip(np.sum(pdf_vals, axis=1), 1e-12, None)
    )
    return {
        "cdf": cdf_vals.astype(np.float64),
        "score_norm": score_norm.astype(np.float64),
        "exp_score_norms": exp_score_norms.astype(np.float64),
        "discrete_omega": discrete_omega.astype(np.float64),
        "discrete_sigma": discrete_sigma.astype(np.float64),
    }


def load_or_calc_igso3(cache_dir: str, **kwargs) -> Dict[str, np.ndarray]:
    """Load the IGSO(3) lookup table; if there is no cache, call `calculate_igso3` and pickle it.

    The cache filename concatenates kwargs sorted by key, so a change in grid
    resolution or in the σ range forces a recompute. Under the linear schedule,
    `IGSO3` replaces max_sigma with σ(1), and the cache key uses that replaced value.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = "_".join(f"{k}{kwargs[k]}" for k in sorted(kwargs))
    path = os.path.join(cache_dir, f"igso3_{key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    vals = calculate_igso3(**kwargs)
    with open(path, "wb") as f:
        pickle.dump(vals, f)
    return vals


class IGSO3:
    """Forward sampling and reverse score for the rotation part.

    Noise scale on continuous time s ∈ [0, 1] (linear schedule, paper Table 6):

        σ(s) = σ_min + s · b_min + ½ s² (b_max - b_min)

    A discrete diffusion step t ∈ {1, …, T} corresponds to s = t/T. The forward
    process draws a rotation vector from IGSO(3) at that scale; the reverse process
    uses the score at the same scale, then multiplies by the diffusion coefficient g(s).

    The lookup table is loaded on the first access to `vals`. When `schedule == "linear"`,
    the table's σ upper bound is replaced by σ(1), not by the max_sigma passed to the
    constructor (the default 1.5 becomes about 1.435).
    """

    def __init__(
        self,
        T: int,
        min_sigma: float,
        max_sigma: float,
        min_b: float,
        max_b: float,
        cache_dir: str = "./schedule_cache",
        schedule: str = "linear",
        L: int = 200,
        num_omega: int = 512,
        num_sigma: int = 128,
    ):
        self.T = T
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.min_b = min_b
        self.max_b = max_b
        self.schedule = schedule
        self.step_size = 1.0 / T
        if schedule == "linear":
            self.max_sigma = float(self.sigma(torch.tensor(1.0)))
        self._vals = None
        self._cache_kwargs = dict(
            num_sigma=num_sigma,
            num_omega=num_omega,
            min_sigma=min_sigma,
            max_sigma=self.max_sigma,
            L=L,
        )
        self._cache_dir = cache_dir

    @property
    def vals(self):
        """Lazily load the lookup table. Contents are the return value of `calculate_igso3`."""
        if self._vals is None:
            self._vals = load_or_calc_igso3(self._cache_dir, **self._cache_kwargs)
        return self._vals

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise scale σ(s) at continuous time s ∈ [0, 1].

        Linear schedule: σ(s) = σ_min + s·b_min + ½ s² (b_max − b_min).
        The argument named `t` is continuous time, not a discrete step and not the
        variance expected by `f_igso3`.
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        t = t.to(dtype=torch.float64)
        if self.schedule == "linear":
            return self.min_sigma + t * self.min_b + 0.5 * (t ** 2) * (self.max_b - self.min_b)
        raise ValueError(self.schedule)

    def sigma_idx(self, sigma: np.ndarray) -> np.ndarray:
        """Map σ onto an index of `discrete_sigma`.

        `np.digitize` returns the index of the first grid line to the right; subtracting
        1 gives the last grid line that does not exceed this σ. A σ smaller than the
        whole grid yields -1, and the caller then clips into the valid range.
        """
        return np.digitize(sigma, self.vals["discrete_sigma"]) - 1

    def t_to_idx(self, t: np.ndarray) -> np.ndarray:
        """Discrete diffusion step t (about 1…T, same as the translation schedule) → lookup-table row.

        Convert to continuous time t/T, compute σ(t/T), then snap to the nearest
        σ grid point that does not exceed it.
        """
        continuous_t = np.asarray(t, dtype=np.float64) / self.T
        sig = self.sigma(torch.tensor(continuous_t)).numpy()
        return np.clip(self.sigma_idx(sig), 0, len(self.vals["discrete_sigma"]) - 1)

    def sample_angles(self, ts, n_samples: int) -> np.ndarray:
        """Inverse-CDF sample of rotation angles at the given discrete steps.

        Each t uses its own CDF row. u ~ Uniform(0, 1),
        ω = interp(u; CDF_t → discrete_omega).

        Args:
            ts: discrete steps, a scalar or (B,).
            n_samples: how many angles to draw per t.
        Returns:
            (len(ts), n_samples) in radians.
        """
        ts = np.atleast_1d(np.asarray(ts))
        out = []
        u = np.random.rand(len(ts), n_samples)
        for i, t in enumerate(ts):
            idx = int(self.t_to_idx(t))
            ang = np.interp(u[i], self.vals["cdf"][idx], self.vals["discrete_omega"])
            out.append(ang)
        return np.stack(out, axis=0)

    def sample_rotvec(self, ts, n_samples: int) -> np.ndarray:
        """Sample IGSO(3) rotation vectors, used to add rotational noise to each residue frame in the forward process.

        Isotropy: axis and angle are independent. The axis is a standard normal
        renormalized to unit length (uniform on the sphere), and the angle comes
        from `sample_angles`. The rotation vector is axis times angle.

        Returns:
            (len(ts), n_samples, 3)
        """
        axis = np.random.randn(len(np.atleast_1d(ts)), n_samples, 3)
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True).clip(min=1e-8)
        ang = self.sample_angles(ts, n_samples)
        return axis * ang[..., None]

    def score_norm(self, t, omega: np.ndarray) -> np.ndarray:
        """Look up ∂/∂ω log f at discrete step t and angle omega.

        Linear interpolation inside the table. The return value is the signed length
        of the score vector: the reverse step multiplies it by the unit axis to get
        the score in the Lie algebra.
        """
        idx = int(self.t_to_idx(np.array(t)))
        return np.interp(omega, self.vals["discrete_omega"], self.vals["score_norm"][idx])

    def g(self, t: float) -> torch.Tensor:
        """Continuous-time diffusion coefficient g(s) = sqrt(d/ds σ(s)²).

        The time derivative of σ² is 2 σ dσ/ds, and under the linear schedule
        dσ/ds = b_min + s (b_max − b_min), so

            g(s) = sqrt(max(2 σ(s) dσ/ds, 0))

        The drift of the reverse SDE carries g² · Δs · score, and the noise term
        carries g · sqrt(Δs) · z. The argument `t` is continuous time s ∈ [0, 1],
        not a discrete step.
        """
        t = float(t)
        sig = float(self.sigma(torch.tensor(t)))
        dsigma_dt = self.min_b + t * (self.max_b - self.min_b)
        return torch.sqrt(torch.tensor(max(2.0 * sig * dsigma_dt, 0.0)))
