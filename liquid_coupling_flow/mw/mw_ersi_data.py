"""Adapt mW reference banks to the vendored eRSI tensor format."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from liquid_coupling_flow.mw.mw_ersi_common import L_for_N


def _val_path(path: str) -> str:
    p = Path(path)
    return str(p.with_name(f"{p.stem}_val{p.suffix}"))


def write_ersi_dataset(out_pos: str, out_species: str, bank_paths: Iterable[str], *,
                       thin_events: int = 2, val_frac: float = 0.1, seed: int = 0) -> dict:
    """Write time-ordered train/validation eRSI tensors from mW ``mc_run`` banks.

    ``seed`` is deliberately accepted for a stable public interface, although the
    leakage-safe split is chronological and therefore has no randomized branch.
    Validation tensors are written next to their corresponding training files as
    ``*_val.pt``.
    """
    del seed
    paths = [str(p) for p in bank_paths]
    if not paths:
        raise ValueError("bank_paths must contain at least one reference bank")
    if thin_events < 1:
        raise ValueError("thin_events must be >= 1")
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be strictly between 0 and 1")

    cfgs = []
    expected_shape = None
    for path in paths:
        bank = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(bank, dict) or "cfgs" not in bank:
            raise ValueError(f"{path} is not an mW mc_run bank with a 'cfgs' tensor")
        x = torch.as_tensor(bank["cfgs"])
        if x.ndim != 3 or x.shape[-1] != 3 or x.shape[0] == 0:
            raise ValueError(f"{path}: expected non-empty cfgs [n,N,3], got {tuple(x.shape)}")
        if expected_shape is None:
            expected_shape = tuple(x.shape[1:])
        elif tuple(x.shape[1:]) != expected_shape:
            raise ValueError(f"{path}: cfg shape {tuple(x.shape[1:])} != {expected_shape}")
        cfgs.append(x[::thin_events])

    x = torch.cat(cfgs, dim=0).to(dtype=torch.float32)
    N = x.shape[1]
    L = L_for_N(N)
    x = torch.remainder(x, L)
    n_val = max(1, int(round(x.shape[0] * val_frac)))
    if n_val >= x.shape[0]:
        raise ValueError("dataset has no training examples after chronological validation split")
    train, val = x[:-n_val].contiguous(), x[-n_val:].contiguous()
    species_train = torch.zeros(train.shape[:2], dtype=torch.long)
    species_val = torch.zeros(val.shape[:2], dtype=torch.long)

    for path in (out_pos, out_species, _val_path(out_pos), _val_path(out_species)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(train, out_pos)
    torch.save(species_train, out_species)
    torch.save(val, _val_path(out_pos))
    torch.save(species_val, _val_path(out_species))
    return {"n_train": int(train.shape[0]), "n_val": int(val.shape[0]), "N": N, "L": L}


__all__ = ["L_for_N", "write_ersi_dataset"]
