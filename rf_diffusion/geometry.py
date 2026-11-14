"""Rigid backbone frames, ideal backbone geometry, and PDB I/O.

Diffusion does not noise the three points N/CA/C directly. Each residue is one
rigid frame (R, t): t is the CA coordinate, and the three columns of R are the
local axes in the RoseTTAFold convention, expressed in the lab frame. A point x
in the local frame becomes R @ x + t in the lab frame.

N and C are not stored on their own. When coordinates are needed they are placed
back from the frame with fixed ideal bond lengths and angles (`IDEAL_N` / `IDEAL_C`),
so a round trip "idealizes" the bond geometry and keeps only the orientation and
the CA position.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Three-letter → one-letter. MSE (selenomethionine) is treated as MET.
# Residues missing from the table are later recorded as X.
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "UNK": "X",
}
# RoseTTAFold's order of the 20 amino acids, indices 0–19.
# X=20 is unknown, and MASK_IDX=21 is a masked position.
AA1_TO_IDX = {aa: i for i, aa in enumerate("ARNDCQEGHILKMFPSTWYV")}
AA1_TO_IDX["X"] = 20
MASK_IDX = 21
N_AA = 22

# Ideal N, CA, C in the residue local frame (RoseTTAFold convention, in Å).
# CA is at the origin; C lies on +e1, with CA–C = 1.5233 Å;
# N lies in the e1–e2 plane, corresponding to CA–N ≈ 1.46 Å and bond angle N–CA–C ≈ 111°.
IDEAL_N = torch.tensor([-0.5272, 1.3593, 0.0])
IDEAL_C = torch.tensor([1.5233, 0.0, 0.0])
IDEAL_CA = torch.tensor([0.0, 0.0, 0.0])


def rigid_from_3_points(
    N: torch.Tensor,
    Ca: torch.Tensor,
    C: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a rotation matrix from backbone N, CA, C (RoseTTAFold convention).

    Gram–Schmidt about CA gives a right-handed frame. The three columns are R:

    - e1 = normalize(C − CA): the local x axis, along CA→C.
    - e2 = normalize( (N − CA) with its component along e1 removed ): the axis in
      the N–CA–C plane that points toward N.
    - e3 = e1 × e2: the plane normal.

    R maps local coordinates into the lab frame (columns are the local axes).
    t is taken directly as CA; there is no translation along the normal.

    Args:
        N, Ca, C: (..., 3)
    Returns:
        R: (..., 3, 3), Ca: (..., 3)
    """
    v1 = C - Ca
    v2 = N - Ca
    e1 = v1 / (torch.linalg.norm(v1, dim=-1, keepdim=True) + eps)
    u2 = v2 - (e1 * v2).sum(dim=-1, keepdim=True) * e1
    e2 = u2 / (torch.linalg.norm(u2, dim=-1, keepdim=True) + eps)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-1)
    return R, Ca


def frames_to_xyz(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Place ideal N, CA, C back from a rigid frame.

    xyz = R @ local + t. The local coordinate of CA is 0, so CA lands exactly on t.
    N and C land at the ideal bond lengths and angles; bond-length deviations in a
    real structure are wiped out. The network predicts an update to the frame, and
    the last step uses this function to turn it back into coordinates.

    Args:
        R: (..., 3, 3)
        t: (..., 3), the CA coordinate
    Returns:
        xyz: (..., 3, 3), with N, CA, C along the second-to-last dimension
    """
    device, dtype = R.device, R.dtype
    n = IDEAL_N.to(device=device, dtype=dtype)
    c = IDEAL_C.to(device=device, dtype=dtype)
    ca = IDEAL_CA.to(device=device, dtype=dtype)
    local = torch.stack([n, ca, c], dim=0)  # (3, 3)
    xyz = torch.einsum("...ij,aj->...ai", R, local) + t.unsqueeze(-2)
    return xyz


def xyz_to_frames(xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Turn coordinates back into a frame. Atom order of xyz must be N, CA, C.

    A thin wrapper around `rigid_from_3_points`. If the input is already the output
    of `frames_to_xyz`, a round trip recovers the original R and t. If the input is
    a crystal coordinate, placing the resulting frame back idealizes N/C.

    Args:
        xyz: (..., 3, 3)
    Returns:
        R: (..., 3, 3), t: (..., 3) CA coordinate
    """
    return rigid_from_3_points(xyz[..., 0, :], xyz[..., 1, :], xyz[..., 2, :])


def generate_cbeta(N: torch.Tensor, Ca: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Place a virtual Cβ from N, CA, C (the fixed linear combination used by RoseTTAFold / RFdiffusion).

    b = CA − N, c = C − CA, a = b × c, then

        Cβ = −0.58273431 a + 0.56802827 b − 0.54067466 c + CA

    a supplies the out-of-plane direction the tetrahedron needs. The coefficients
    match ideal bond lengths and angles; they are not fit from a structure.
    Glycine has no real Cβ, and every residue is treated the same here.
    Diffusion coordinates do not include Cβ; it is used only for pair geometry (c6d),
    active-site distances, and interface hotspots.

    Args:
        N, Ca, C: (..., 3)
    Returns:
        (..., 3) virtual Cβ
    """
    b = Ca - N
    c = C - Ca
    a = torch.cross(b, c, dim=-1)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca


def rot_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rotation composition a @ b: apply b first, then a (column-vector convention).

    Forward noising writes R_noise @ R_0, and the reverse step writes perturb @ R_t.
    Both are left multiplies, so the incremental rotation acts on the lab-frame side
    of the current frame.
    """
    return torch.einsum("...ij,...jk->...ik", a, b)


def apply_rot(R: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by R: R @ x. The batch dimensions of R and x must broadcast."""
    return torch.einsum("...ij,...j->...i", R, x)


def rotation_frobenius(R_pred: torch.Tensor, R_true: torch.Tensor) -> torch.Tensor:
    """Rotation loss between two frames, ||I − R_predᵀ R_true||_F².

    R_predᵀ R_true is the relative rotation from the predicted frame to the true
    frame. At angle θ this quantity equals 8 sin²(θ/2). It is 0 when the frames
    agree, and about 2θ² at small angles. `frame_distance` uses it as the rotation
    term of L_frame and averages it together with the CA displacement.

    Args:
        R_pred, R_true: (..., 3, 3)
    Returns:
        (...,) one scalar per residue
    """
    I = torch.eye(3, device=R_pred.device, dtype=R_pred.dtype)
    rel = I - torch.einsum("...ji,...jk->...ik", R_pred, R_true)
    return (rel * rel).sum(dim=(-2, -1))


def center_xyz(xyz: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Translate the structure onto the CA center of mass. N, CA, and C share one subtracted vector, so frame orientation is unchanged.

    Only residues with mask True enter the center of mass. With no mask, every CA
    is used. The weight sum is clamped to at least 1, so an empty selection does
    not divide by zero. Training examples do this before noising, so translational
    diffusion is relative to the center of mass rather than the lab origin.

    Args:
        xyz: (..., L, 3, 3), atom order N, CA, C
        mask: (..., L), True means included in the center of mass
    Returns:
        same shape as xyz
    """
    ca = xyz[..., 1, :]
    if mask is None:
        com = ca.mean(dim=-2, keepdim=True)
    else:
        w = mask.to(ca.dtype).unsqueeze(-1)
        com = (ca * w).sum(dim=-2, keepdim=True) / w.sum(dim=-2, keepdim=True).clamp_min(1.0)
    return xyz - com.unsqueeze(-2)


def parse_pdb_backbone(path: str | Path, chain: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Read backbone N/CA/C from a PDB and return coordinates and sequence indices.

    ATOM records are read, plus HETATM records whose residue name is MSE
    (selenomethionine is often stored that way). Only atoms named N, CA, and C
    are kept. A blank chain ID is treated as "A". Passing `chain` drops every
    other chain. A residue key is (chain, residue number, insertion code), so
    10 and 10A stay distinct. Output is sorted by (chain, number, insertion code).

    A residue missing any of the three backbone atoms is dropped entirely.
    Three-letter codes are mapped through `AA3_TO_1`; unrecognized ones are
    recorded as X (index 20).

    Returns:
        xyz: (L, 3, 3) float32, order N, CA, C, in Å
        seq: (L,) int64, 0–20
        chain: (L,) chain IDs
        resi: (L,) int64, PDB residue numbers
    """
    atoms: Dict[Tuple[str, int, str, str], np.ndarray] = {}
    seq_map: Dict[Tuple[str, int, str], str] = {}
    with open(path) as f:
        for line in f:
            is_atom = line.startswith("ATOM")
            is_het = line.startswith("HETATM")
            if not is_atom and not is_het:
                continue
            if len(line) < 54:
                continue
            resn = line[17:20].strip()
            if is_het and resn != "MSE":
                continue
            name = line[12:16].strip()
            if name not in ("N", "CA", "C"):
                continue
            ch = line[21].strip() or "A"
            if chain is not None and ch != chain:
                continue
            resi = int(line[22:26])
            icode = line[26].strip()
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float32)
            atoms[(ch, resi, icode, name)] = xyz
            seq_map[(ch, resi, icode)] = AA3_TO_1.get(resn, "X")

    residues = sorted({(ch, resi, icode) for (ch, resi, icode, _) in atoms})
    L = len(residues)
    xyz = np.full((L, 3, 3), np.nan, dtype=np.float32)
    seq = np.full((L,), 20, dtype=np.int64)
    chains: List[str] = []
    resis: List[int] = []
    atom_names = ("N", "CA", "C")
    for i, (ch, resi, icode) in enumerate(residues):
        chains.append(ch)
        resis.append(resi)
        seq[i] = AA1_TO_IDX.get(seq_map[(ch, resi, icode)], 20)
        for j, an in enumerate(atom_names):
            if (ch, resi, icode, an) in atoms:
                xyz[i, j] = atoms[(ch, resi, icode, an)]
    ok = ~np.isnan(xyz).any(axis=(1, 2))
    return {
        "xyz": xyz[ok],
        "seq": seq[ok],
        "chain": np.array(chains, dtype=object)[ok],
        "resi": np.array(resis, dtype=np.int64)[ok],
    }


def write_pdb(path: str | Path, xyz: np.ndarray, seq: Optional[np.ndarray] = None) -> None:
    """Write N/CA/C coordinates as a single-chain PDB.

    The chain ID is fixed to A, and residue numbers run consecutively from 1;
    original numbering is not kept. Each residue writes only N, CA, and C, with
    occupancy 1.00 and B-factor 0.00.

    `seq` uses the same indices as `AA1_TO_IDX`: 0–19 are the standard amino acids,
    20 is written as UNK, and 21 (mask) is written as GLY. With no sequence, every
    residue is written as GLY (index 7).

    Args:
        xyz: (L, 3, 3), order N, CA, C
        seq: (L,) amino-acid indices, optional
    """
    aa = "ARNDCQEGHILKMFPSTWYVXG"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        serial = 1
        for i in range(xyz.shape[0]):
            s = int(seq[i]) if seq is not None else 7
            s = min(max(s, 0), 21)
            resn = {
                "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
                "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
                "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
                "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
                "X": "UNK", "G2": "GLY",
            }.get(aa[s] if s < 20 else ("X" if s == 20 else "G"), "GLY")
            for j, an in enumerate((" N  ", " CA ", " C  ")):
                x, y, z = xyz[i, j]
                f.write(
                    f"ATOM  {serial:5d} {an} {resn} A{i+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {an.strip()[0]}\n"
                )
                serial += 1
        f.write("END\n")
