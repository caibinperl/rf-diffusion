"""Training loop implementing Supplementary Methods Algorithm 3."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import TrainConfig
from .data import ProteinDataset, collate_pad
from .diffusion import SE3Diffuser
from .features import assemble_model_inputs, ss_onehot, t1d_channels
from .losses import diffusion_loss
from .model import build_model


def resolve_device(name: str) -> torch.device:
    """Use the requested device when the backend exists, otherwise CPU."""
    requested = str(name).lower()
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        return torch.device("cpu")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        torch.manual_seed(cfg.seed)
        self.diffuser = SE3Diffuser(cfg.diffuser).to(self.device)
        t1d_dim = t1d_channels(cfg.experiment)
        self.model = build_model(cfg.model, T=cfg.diffuser.T, ckpt=cfg.rf_ckpt, t1d_dim=t1d_dim).to(self.device)
        self._wandb = None
        if cfg.use_wandb:
            try:
                import wandb

                wandb.init(
                    project=cfg.wandb_project,
                    config={
                        "experiment": cfg.experiment,
                        "model": cfg.model,
                        "T": cfg.diffuser.T,
                        "lr": cfg.optim.lr,
                    },
                )
                self._wandb = wandb
            except ImportError:
                print("wandb is not installed; continuing without it")
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        self.dataset = ProteinDataset(cfg)
        self.loader = DataLoader(
            self.dataset,
            batch_size=cfg.optim.micro_batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            collate_fn=collate_pad,
            drop_last=False,
        )
        self.step = 0
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "train_log.jsonl"

    def _lr_at(self, step: int) -> float:
        lr = self.cfg.optim.lr
        if self.cfg.optim.warmup_steps and step < self.cfg.optim.warmup_steps:
            lr = lr * step / max(self.cfg.optim.warmup_steps, 1)
        decays = step // max(self.cfg.optim.decay_every, 1)
        return lr * (self.cfg.optim.decay_factor ** decays)

    def _sample_t(self, B: int) -> torch.Tensor:
        return torch.randint(1, self.cfg.diffuser.T + 1, (B,), device=self.device)

    def _task_features(self, xyz0: torch.Tensor, hotspot: torch.Tensor, pad: torch.Tensor):
        """Hotspot only for PPI, secondary structure only for fold. Width matches `t1d_channels`."""
        exp = self.cfg.experiment
        hotspot_in = hotspot if exp == "ppi" else None
        ss = ss_onehot(xyz0, pad) if exp == "fold" else None
        return hotspot_in, ss

    def _run_model(self, inputs: Dict[str, torch.Tensor], pad: torch.Tensor, selfcond_mask: torch.Tensor):
        return self.model(
            xyz_t=inputs["xyz_t"],
            t=inputs["t"],
            seq=inputs["seq"],
            motif_mask=inputs["motif_mask"],
            t1d=inputs["t1d"],
            t2d=inputs["t2d"],
            xyz_selfcond=inputs["xyz_selfcond"],
            selfcond_mask=selfcond_mask,
            msa=inputs["msa"],
            idx=inputs["idx"],
            pad_mask=pad,
        )

    def _forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        xyz0 = batch["xyz"].to(self.device)
        seq = batch["seq"].to(self.device)
        motif = batch["motif_mask"].to(self.device)
        hotspot = batch["hotspot"].to(self.device)
        pad = batch["pad_mask"].to(self.device)
        idx = batch["idx"].to(self.device)
        B = xyz0.shape[0]
        t = self._sample_t(B)
        T = self.cfg.diffuser.T
        hotspot_in, ss = self._task_features(xyz0, hotspot, pad)

        # Direct forward sample: correct translational and rotational marginals at t.
        xyz_t = self.diffuser.forward_noise(xyz0, t, motif_mask=motif)
        xyz_self = torch.zeros_like(xyz_t)
        selfcond_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
        # t = T has no noisier step. Decide per example, not for the whole batch.
        use_self = (torch.rand(B, device=self.device) < self.cfg.self_cond_prob) & (t < T)
        if bool(use_self.any()):
            t_next = torch.where(use_self, t + 1, torch.ones_like(t))
            with torch.no_grad():
                xyz_tp1 = self.diffuser.forward_noise(xyz0, t_next, motif_mask=motif)
                xyz_back = self.diffuser.reverse_step(xyz_tp1, xyz0, t_next, motif_mask=motif)
                # Posterior CA is unbiased. Rotation is redrawn at the labeled time t.
                xyz_tf = self.diffuser.graft_forward_rotation(
                    xyz0, xyz_back[..., 1, :], t, motif_mask=motif
                )
                inputs_prev = assemble_model_inputs(
                    xyz0, xyz_tp1, seq, t_next, motif, T,
                    hotspot=hotspot_in, ss=ss, idx=idx,
                )
                pred_prev = self._run_model(
                    inputs_prev, pad, torch.zeros(B, dtype=torch.bool, device=self.device)
                )
                draft = pred_prev["xyz"].detach()
            take = use_self.view(B, 1, 1, 1)
            xyz_t = torch.where(take, xyz_tf, xyz_t)
            xyz_self = torch.where(take, draft, xyz_self)
            selfcond_mask = use_self

        inputs = assemble_model_inputs(
            xyz0, xyz_t, seq, t, motif, T,
            xyz_selfcond=xyz_self, hotspot=hotspot_in, ss=ss, idx=idx,
        )
        pred = self._run_model(inputs, pad, selfcond_mask)
        motif_w = self.cfg.loss.motif_loss_weight
        if self.cfg.experiment == "active_site":
            motif_w = self.cfg.loss.active_site_motif_weight
        losses = diffusion_loss(
            xyz_true=xyz0,
            xyz_pred=pred["xyz"],
            cfg=self.cfg.loss,
            logits_c6d=pred.get("logits_c6d"),
            motif_mask=motif & pad,
            motif_weight=motif_w,
            intermediate=pred.get("intermediate"),
            pad_mask=pad,
        )
        losses["t_mean"] = t.float().mean()
        losses["self_cond"] = selfcond_mask.float().mean()
        return losses

    def save(self, tag: str = "last") -> None:
        path = self.output_dir / f"ckpt_{tag}.pt"
        torch.save(
            {
                "step": self.step,
                "model": self.model.state_dict(),
                "opt": self.opt.state_dict(),
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def _log(self, metrics: Dict[str, float]) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        if self._wandb is not None:
            self._wandb.log(metrics)

    def _accum(self) -> int:
        return max(self.cfg.optim.pseudo_batch_size // max(self.cfg.optim.micro_batch_size, 1), 1)

    def _optimizer_step(self, n_micro: int) -> None:
        """Apply accumulated gradients. A short final group is rescaled to a full step."""
        accum = self._accum()
        if n_micro <= 0:
            return
        if n_micro != accum:
            scale = accum / float(n_micro)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

    def train(self) -> None:
        cfg = self.cfg
        accum = self._accum()
        steps_per_epoch = math.ceil(cfg.examples_per_epoch / max(cfg.optim.micro_batch_size, 1))
        max_steps = cfg.max_steps if cfg.max_steps is not None else cfg.epochs * steps_per_epoch
        print(
            f"RFdiffusion training  experiment={cfg.experiment}  model={cfg.model}\n"
            f"device={self.device}  T={cfg.diffuser.T}  crop={cfg.data.crop_size}\n"
            f"micro_batch={cfg.optim.micro_batch_size}  accum={accum}  max_steps={max_steps}"
        )
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        data_iter = iter(self.loader)
        pbar = tqdm(total=max_steps, desc="train")
        running = {}
        pending = 0
        t0 = time.time()
        while self.step < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.loader)
                batch = next(data_iter)
            lr = self._lr_at(self.step)
            for g in self.opt.param_groups:
                g["lr"] = lr
            losses = self._forward_batch(batch)
            (losses["loss"] / accum).backward()
            pending += 1
            if pending == accum:
                self._optimizer_step(pending)
                pending = 0
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach().cpu())
            self.step += 1
            pbar.update(1)
            if self.step % cfg.log_every == 0:
                avg = {k: v / cfg.log_every for k, v in running.items()}
                avg.update({"step": self.step, "lr": lr, "sec": time.time() - t0})
                self._log(avg)
                pbar.set_postfix(
                    {
                        "loss": f"{avg['loss']:.3f}",
                        "frame": f"{avg['loss_frame']:.3f}",
                        "ca": f"{avg['ca_rmsd']:.2f}",
                    }
                )
                running = {}
            if self.step % cfg.ckpt_every == 0:
                self.save(tag=str(self.step))
        if pending:
            self._optimizer_step(pending)
        self.save(tag="last")
        pbar.close()
        if self._wandb is not None:
            self._wandb.finish()
        print(f"Finished {self.step} steps. Checkpoints in {self.output_dir}")


def train_from_cfg(cfg: TrainConfig) -> Trainer:
    trainer = Trainer(cfg)
    trainer.train()
    return trainer
