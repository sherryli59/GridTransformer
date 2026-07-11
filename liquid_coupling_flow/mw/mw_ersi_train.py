"""Training entry point for the 3-D, monatomic mW eRSI velocity field."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from liquid_coupling_flow.mw.mw_ersi_common import _add_learndiffeq_path
from liquid_coupling_flow.mw.mw_ersi_data import L_for_N, write_ersi_dataset


def _make_flow(*, N: int, K: int, hidden_nf: int, n_layers: int, L: float, lr: float, ot: bool):
    """Construct the vendored RFM shell with its traceable, exact-divergence velocity.

    The vendored velocity factory only exposes the older ``particles_equivariant``
    backend.  It has no registration hook for ``egnn_traceable``.  We therefore
    construct the supported RFM shell then replace its velocity child before the
    optimiser is built; training, OT coupling and likelihood integration remain
    the vendored implementations while every velocity/divergence call uses the
    traceable backend.
    """
    _add_learndiffeq_path()
    from learndiffeq.flow_matching.riemannian_fm import RiemannianFlowMatching
    from learndiffeq.particles.distributions.uniform import UniformParticles
    from learndiffeq.particles.velocities.egnn_traceable import EGNN_dynamics

    species = torch.zeros(N, dtype=torch.long)
    rho0 = UniformParticles(species=species, n_particles=N, dim_phys=3, L=L, return_sorted=True)
    flow = RiemannianFlowMatching(
        n_particles=N, dim_phys=3, L=L, rho0=rho0, lr=lr,
        velocity_type="particles_equivariant",
        velocity_kwargs={"hidden_nf": hidden_nf, "n_layers": n_layers, "L": L, "n_species": 1},
        ot_particles=ot, use_linear_assignment_particles=ot,
        force_automatic_div=False,
    )
    flow.b = EGNN_dynamics(n_particles=N, n_dimension=3, hidden_nf=hidden_nf, n_layers=n_layers,
                           max_neighbors=K, L=L, n_species=1)
    return flow


class _ERSIDataModule(pl.LightningDataModule):
    """Small Lightning-compatible module preserving the chronological validation split."""

    def __init__(self, train_x, train_a, val_x, val_a, batch, seed):
        super().__init__()
        self.train_x, self.train_a = train_x, train_a
        self.val_x, self.val_a = val_x, val_a
        self.batch, self.seed = int(batch), int(seed)
        self._epoch = 0

    def setup(self, stage=None):
        self.train_ds = TensorDataset(self.train_x, self.train_a)
        self.val_ds = TensorDataset(self.val_x, self.val_a)

    def train_dataloader(self):
        g = torch.Generator().manual_seed(self.seed + self._epoch)
        return DataLoader(self.train_ds, batch_size=self.batch, shuffle=True, generator=g, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch, shuffle=False, num_workers=0)

    def augment_train(self):
        """Apply one deterministic random torus/point-group augmentation per epoch."""
        g = torch.Generator().manual_seed(self.seed + 10_000 + self._epoch)
        x = self.train_x
        B, N, D = x.shape
        # Monatomic particles may be permuted independently in each configuration.
        perms = torch.argsort(torch.rand(B, N, generator=g), dim=1)
        x = torch.gather(x, 1, perms[..., None].expand(-1, -1, D))
        trans = torch.rand(B, 1, D, generator=g, dtype=x.dtype)
        axes = torch.argsort(torch.rand(B, D, generator=g), dim=1)
        x = torch.gather(x, 2, axes[:, None, :].expand(-1, N, -1))
        signs = (2 * torch.randint(0, 2, (B, 1, D), generator=g, dtype=torch.long) - 1).to(x.dtype)
        self.train_x = torch.remainder((x + trans * self.L) * signs, self.L)
        self._epoch += 1
        self.setup()

    @property
    def L(self):
        # All coordinates use the same cubic volume; infer it once from the caller-set value.
        return self._L

    @L.setter
    def L(self, value):
        self._L = float(value)


def train(*, N: int = 64, K: int = 12, hidden_nf: int = 128, n_layers: int = 4,
          steps: int = 10_000, batch: int = 32, lr: float = 3e-4, ot: bool = True,
          out: str = "mw_ersi_N64.pt", bank_paths=None, seed: int = 0,
          thin_events: int = 2, val_frac: float = 0.1, device: str = "cpu") -> dict:
    """Fit an mW eRSI velocity and save a portable reconstruction checkpoint.

    CPU is the safe default because the shared GPU belongs to active campaign
    runs.  Pass ``device='cuda'`` explicitly for a controller-authorized run.
    """
    if N < 2 or not 1 <= K < N:
        raise ValueError(f"require 2 <= N and 1 <= K < N, got N={N}, K={K}")
    if steps < 1 or batch < 1:
        raise ValueError("steps and batch must be positive")
    if not bank_paths:
        raise ValueError("bank_paths is required; training data is never guessed")
    if device != "cpu" and not (device == "cuda" and torch.cuda.is_available()):
        raise ValueError(f"requested unavailable device {device!r}")

    _add_learndiffeq_path()
    from pytorch_lightning.callbacks import Callback

    pl.seed_everything(seed, workers=True)
    L = L_for_N(N)
    with tempfile.TemporaryDirectory(prefix="mw_ersi_data_") as tmp:
        pos, species = os.path.join(tmp, "positions.pt"), os.path.join(tmp, "species.pt")
        write_ersi_dataset(pos, species, bank_paths, thin_events=thin_events, val_frac=val_frac, seed=seed)
        val_pos = str(Path(pos).with_name("positions_val.pt"))
        val_species = str(Path(species).with_name("species_val.pt"))
        train_x = torch.load(pos, weights_only=False)
        train_a = torch.load(species, weights_only=False)
        val_x = torch.load(val_pos, weights_only=False)
        val_a = torch.load(val_species, weights_only=False)

        dm = _ERSIDataModule(train_x, train_a, val_x, val_a, batch, seed)
        dm.L = L
        model = _make_flow(N=N, K=K, hidden_nf=hidden_nf, n_layers=n_layers, L=L, lr=lr, ot=ot)

        losses = []

        class _RecordAndAugment(Callback):
            def on_train_epoch_start(self, trainer, pl_module):
                trainer.datamodule.augment_train()

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                value = outputs["loss"] if isinstance(outputs, dict) else outputs
                losses.append(float(value.detach().cpu()))

        trainer = pl.Trainer(
            max_steps=steps, accelerator="gpu" if device == "cuda" else "cpu", devices=1,
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            num_sanity_val_steps=0, deterministic=True, callbacks=[_RecordAndAugment()],
        )
        trainer.fit(model, datamodule=dm)

    if not losses:
        raise RuntimeError("eRSI trainer completed without recording a training loss")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": "mw_ersi_traceable_rfm_v1",
        "state_dict": model.b.state_dict(), "N": N, "K": K, "hidden_nf": hidden_nf,
        "n_layers": n_layers, "L": L, "dim_phys": 3, "n_species": 1, "ot": bool(ot),
        "step": int(steps), "solver": {"method": "rk4", "n_steps": 40},
    }, out_path)
    return {"path": str(out_path), "loss_first": losses[0], "loss_last": losses[-1], "n_steps": len(losses)}


__all__ = ["train", "_make_flow"]
