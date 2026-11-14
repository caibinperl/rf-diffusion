"""Smoke tests for one RFdiffusion training step (synthetic data, Gaussian SO(3))."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from rf_diffusion.config import DataConfig, LossConfig, TrainConfig
from rf_diffusion.data import (
    ProteinDataset,
    chain_residue_index,
    optional_neighbor,
    sample_motif_mask,
)
from rf_diffusion.diffusion import SE3Diffuser
from rf_diffusion.features import assemble_model_inputs, t1d_channels
from rf_diffusion.geometry import frames_to_xyz, parse_pdb_backbone
from rf_diffusion.losses import diffusion_loss
from rf_diffusion.model import TinyRFDenoiser
from rf_diffusion.trainer import Trainer, resolve_device


def test_forward_noise_changes_scaffold_not_motif():
    cfg = TrainConfig()
    cfg.diffuser.so3_type = "gaussian"
    cfg.diffuser.truncation_level = 32
    cfg.diffuser.num_omega = 32
    cfg.diffuser.num_sigma = 8
    diff = SE3Diffuser(cfg.diffuser)
    ds = ProteinDataset(cfg)
    item = ds[0]
    xyz = item["xyz"].unsqueeze(0)
    motif = item["motif_mask"].unsqueeze(0)
    t = torch.tensor([80])
    xyz_t = diff.forward_noise(xyz, t, motif_mask=motif)
    if motif.any():
        err = (xyz_t[motif] - xyz[motif]).abs().max()
        assert err < 1e-4, err
    delta = (xyz_t[~motif] - xyz[~motif]).abs().mean()
    assert delta > 1e-3


def test_one_train_step():
    cfg = TrainConfig()
    cfg.device = "cpu"
    cfg.model = "tiny"
    cfg.max_steps = 2
    cfg.log_every = 1
    cfg.ckpt_every = 999
    cfg.data.synthetic_n = 8
    cfg.data.synthetic_len = 32
    cfg.data.crop_size = 32
    cfg.diffuser.so3_type = "gaussian"
    cfg.diffuser.T = 50
    cfg.optim.micro_batch_size = 1
    cfg.optim.pseudo_batch_size = 1
    cfg.self_cond_prob = 0.0
    cfg.output_dir = str(ROOT / "outputs" / "pytest")
    trainer = Trainer(cfg)
    trainer.train()
    assert trainer.step == 2
    assert (Path(cfg.output_dir) / "ckpt_last.pt").exists()


def _line(L: int = 8, B: int = 1) -> torch.Tensor:
    ca = torch.zeros(B, L, 3)
    ca[..., 0] = torch.arange(L).float()
    R = torch.eye(3).expand(B, L, 3, 3).contiguous()
    return frames_to_xyz(R, ca)


def test_translation_noise_keeps_com():
    cfg = TrainConfig()
    cfg.diffuser.so3_type = "gaussian"
    cfg.diffuser.T = 50
    diff = SE3Diffuser(cfg.diffuser)
    torch.manual_seed(0)
    xyz = _line(20, B=2)
    t = torch.full((2,), 50)
    ab = diff.alphabar[49]
    xyz_t = diff.forward_noise(xyz, t)
    noise = xyz_t[..., 1, :] - torch.sqrt(ab) * xyz[..., 1, :]
    assert noise.mean(dim=1).abs().max() < 1e-3


def test_pair_geometry_changes_frame_update():
    torch.manual_seed(0)
    model = TinyRFDenoiser(d_res=32, d_pair=16, n_layers=1, n_heads=4, T=20, t1d_dim=22)
    model.eval()
    xyz1 = _line(8)
    ca = torch.zeros(1, 8, 3)
    ca[0, :, 0] = torch.tensor([0.0, 0.3, 0.6, 4.0, 4.2, 9.0, 9.2, 14.0])
    R = torch.eye(3).expand(1, 8, 3, 3).contiguous()
    xyz2 = frames_to_xyz(R, ca)
    seq = torch.zeros(1, 8, dtype=torch.long)
    t = torch.tensor([10])
    motif = torch.zeros(1, 8, dtype=torch.bool)
    t1d = torch.zeros(1, 8, 22)
    t1d[..., -1] = 0.5
    with torch.no_grad():
        y1 = model(xyz1, t, seq, motif, t1d)
        y2 = model(xyz2, t, seq, motif, t1d)
    d1 = y1["xyz"][..., 1, :] - xyz1[..., 1, :]
    d2 = y2["xyz"][..., 1, :] - xyz2[..., 1, :]
    assert not torch.allclose(d1, d2, atol=1e-5)


def test_padding_is_not_supervised():
    xyz = _line(6)
    pred = xyz.clone()
    pred[0, -1] = pred[0, -1] + 4.0
    pad = torch.ones(1, 6, dtype=torch.bool)
    pad[0, -1] = False
    out = diffusion_loss(xyz, pred, LossConfig(), pad_mask=pad)
    assert float(out["loss"]) < 1e-4
    assert float(out["ca_rmsd"]) < 1e-4


def test_motif_weight_increases_loss():
    xyz = _line(6)
    pred = xyz.clone()
    pred[0, 0] = pred[0, 0] + 2.0
    motif = torch.zeros(1, 6, dtype=torch.bool)
    motif[0, 0] = True
    lo = diffusion_loss(xyz, pred, LossConfig(p_clamp=0.0), motif_mask=motif, motif_weight=1.0)
    hi = diffusion_loss(xyz, pred, LossConfig(p_clamp=0.0), motif_mask=motif, motif_weight=10.0)
    assert float(hi["loss"]) > float(lo["loss"])


def test_motif_segments_do_not_overlap():
    cfg = DataConfig()
    cfg.motif_contiguous_prob = 0.0
    cfg.motif_mask_frac_min = 0.5
    cfg.motif_mask_frac_max = 0.5
    np.random.seed(0)
    for _ in range(30):
        motif = sample_motif_mask(40, cfg)
        assert int(motif.sum()) == 20
        runs = int((motif[1:] & ~motif[:-1]).sum()) + int(motif[0])
        assert runs <= 2


def test_neighbor_can_be_either_side():
    np.random.seed(1)
    seen = set()
    for _ in range(200):
        nb = optional_neighbor(5, 20)
        if nb is not None:
            seen.add(nb)
    assert seen == {4, 6}


def test_chain_index_jumps_at_breaks():
    chain = np.array(list("AAABBB"))
    idx = chain_residue_index(chain)
    assert int(idx[2]) == 2
    assert int(idx[3]) - int(idx[2]) == 201


def test_synthetic_backbone_is_seeded():
    cfg = TrainConfig()
    cfg.data.synthetic_n = 4
    cfg.data.synthetic_len = 32
    ds = ProteinDataset(cfg)
    rec = next(r for r in ds.records if r["kind"] == "synthetic")
    a = ds._load(rec)
    b = ds._load(rec)
    assert torch.equal(a["xyz"], b["xyz"])
    assert torch.equal(a["seq"], b["seq"])


def test_parse_mse_and_insertion(tmp_path=None):
    import tempfile
    from pathlib import Path

    text = """\
HETATM    1  N   MSE A   1      11.000  12.000  13.000  1.00  0.00           N
HETATM    2  CA  MSE A   1      12.000  12.000  13.000  1.00  0.00           C
HETATM    3  C   MSE A   1      13.000  12.000  13.000  1.00  0.00           C
ATOM      4  N   ALA A   1A     21.000  22.000  23.000  1.00  0.00           N
ATOM      5  CA  ALA A   1A     22.000  22.000  23.000  1.00  0.00           C
ATOM      6  C   ALA A   1A     23.000  22.000  23.000  1.00  0.00           C
ATOM      7  N   ALA A   2      31.000  32.000  33.000  1.00  0.00           N
ATOM      8  CA  ALA A   2      32.000  32.000  33.000  1.00  0.00           C
ATOM      9  C   ALA A   2      33.000  32.000  33.000  1.00  0.00           C
END
"""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "toy.pdb"
        path.write_text(text)
        rec = parse_pdb_backbone(path)
        assert rec["xyz"].shape[0] == 3
        assert int(rec["seq"][0]) == 12  # MSE -> MET
        cfg = TrainConfig()
        cfg.data.pdb_dir = folder
        cfg.data.min_length = 1
        cfg.data.max_length = 2
        cfg.data.synthetic_n = 1
        skipped = ProteinDataset(cfg)
        assert all(r["kind"] == "synthetic" for r in skipped.records)
        cfg.data.max_length = 10
        kept = ProteinDataset(cfg)
        assert any(r["kind"] == "pdb" for r in kept.records)


def test_t1d_width_matches_task():
    assert t1d_channels("base") == 22
    assert t1d_channels("active_site") == 22
    assert t1d_channels("ppi") == 23
    assert t1d_channels("fold") == 26
    xyz = _line(4)
    seq = torch.zeros(1, 4, dtype=torch.long)
    t = torch.tensor([3])
    motif = torch.zeros(1, 4, dtype=torch.bool)
    base = assemble_model_inputs(xyz, xyz, seq, t, motif, 20)
    assert base["t1d"].shape[-1] == 22
    hot = torch.zeros(1, 4)
    ppi = assemble_model_inputs(xyz, xyz, seq, t, motif, 20, hotspot=hot)
    assert ppi["t1d"].shape[-1] == 23
    ss = torch.zeros(1, 4, 4)
    fold = assemble_model_inputs(xyz, xyz, seq, t, motif, 20, ss=ss)
    assert fold["t1d"].shape[-1] == 26


def test_resolve_device_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_self_cond_respects_terminal_time():
    cfg = TrainConfig()
    cfg.device = "cpu"
    cfg.model = "tiny"
    cfg.self_cond_prob = 1.0
    cfg.data.synthetic_n = 2
    cfg.data.synthetic_len = 16
    cfg.data.crop_size = 16
    cfg.diffuser.so3_type = "gaussian"
    cfg.diffuser.T = 20
    cfg.optim.micro_batch_size = 2
    cfg.output_dir = str(ROOT / "outputs" / "pytest_selfcond")
    trainer = Trainer(cfg)
    batch = next(iter(trainer.loader))
    trainer._sample_t = lambda B: torch.ones(B, dtype=torch.long, device=trainer.device)
    on = trainer._forward_batch(batch)
    assert float(on["self_cond"]) == 1.0
    assert torch.isfinite(on["loss"])
    trainer._sample_t = lambda B: torch.full((B,), cfg.diffuser.T, device=trainer.device)
    off = trainer._forward_batch(batch)
    assert float(off["self_cond"]) == 0.0


if __name__ == "__main__":
    test_forward_noise_changes_scaffold_not_motif()
    print("ok noise")
    test_translation_noise_keeps_com()
    print("ok com")
    test_pair_geometry_changes_frame_update()
    print("ok pair")
    test_padding_is_not_supervised()
    print("ok pad")
    test_motif_weight_increases_loss()
    print("ok motif weight")
    test_motif_segments_do_not_overlap()
    print("ok motif segments")
    test_neighbor_can_be_either_side()
    print("ok neighbor")
    test_chain_index_jumps_at_breaks()
    print("ok chain index")
    test_synthetic_backbone_is_seeded()
    print("ok seed")
    test_parse_mse_and_insertion()
    print("ok pdb")
    test_t1d_width_matches_task()
    print("ok t1d")
    test_resolve_device_cpu()
    print("ok device")
    test_self_cond_respects_terminal_time()
    print("ok self-cond")
    test_one_train_step()
    print("ok train step")
