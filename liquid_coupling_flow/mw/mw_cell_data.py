"""Raw-frame cache and fresh augmented-state draws for global cell q0 training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from liquid_coupling_flow.mw.mw_cell_geom import sample_color_order, sample_orientation
from liquid_coupling_flow.mw.mw_cell_q0 import AugmentedCellState


@dataclass
class CellFrameBank:
    N: int
    L: float
    train: torch.Tensor
    validation: torch.Tensor
    source: str


def _chain_split(payload: dict, held_fraction: float):
    x = payload["cfgs"].detach().cpu().float()
    chain = payload.get("chain_id")
    if chain is None:
        cut = max(1, int((1.0 - float(held_fraction)) * len(x)))
        return x[:cut], x[cut:]
    chain = torch.as_tensor(chain).cpu()
    values = chain.unique(sorted=True)
    n_hold = max(1, int(torch.ceil(torch.tensor(len(values) * float(held_fraction))).item()))
    held = values[-n_hold:]
    mask = (chain[:, None] == held[None]).any(1)
    return x[~mask], x[mask]


def _subsample(x: torch.Tensor, count: int | None, gen: torch.Generator) -> torch.Tensor:
    if count is None or int(count) <= 0 or len(x) <= int(count):
        return x.contiguous()
    return x[torch.randperm(len(x), generator=gen)[:int(count)]].contiguous()


def build_raw_cache(sources, out: str | Path, *, train_per_size=4096,
                    validation_per_size=256, held_fraction=0.125, seed=20260715):
    """Persist immutable wrapped frames only; all q0 auxiliaries stay fresh."""
    paths = [str(p) for p in sources]
    gen = torch.Generator().manual_seed(int(seed))
    banks = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "cfgs" not in payload:
            raise ValueError(f"{path} has no cfgs tensor")
        train, validation = _chain_split(payload, held_fraction)
        N = int(train.shape[1])
        L = float(payload.get("L", (N / 0.4564) ** (1.0 / 3.0)))
        banks.append({
            "N": N, "L": L, "train": _subsample(train, train_per_size, gen),
            "validation": _subsample(validation, validation_per_size, gen),
            "source": path,
        })
    payload = {
        "version": "mw_cell_raw_frames_v1", "seed": int(seed),
        "held_fraction": float(held_fraction), "banks": banks,
        "augmentation": "fresh priority/grid_shift/O_h/color_order per draw",
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return payload


def load_or_build_raw_cache(sources, out: str | Path, **kwargs):
    out = Path(out)
    source_strings = [str(p) for p in sources]
    if out.exists():
        payload = torch.load(out, map_location="cpu", weights_only=False)
        if payload.get("version") != "mw_cell_raw_frames_v1":
            raise ValueError(f"unsupported cache at {out}")
        if [b["source"] for b in payload["banks"]] != source_strings:
            raise ValueError(f"cache sources do not match requested sources at {out}")
        return payload
    return build_raw_cache(source_strings, out, **kwargs)


def frame_banks(cache_payload: dict) -> list[CellFrameBank]:
    return [CellFrameBank(**b) for b in cache_payload["banks"]]


def fresh_augmented_state(x: torch.Tensor, L: float, G: int, gen=None) -> AugmentedCellState:
    """Attach independent target auxiliaries to data frames without caching them."""
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("x must be [B,N,3]")
    B, N = x.shape[:2]
    h = float(L) / int(G)
    priorities = torch.rand(B, N, device=x.device, dtype=x.dtype, generator=gen)
    shift = torch.rand(B, 3, device=x.device, dtype=x.dtype, generator=gen) * h
    orientation = []
    colors = []
    for _ in range(B):
        _, idx = sample_orientation(device=x.device, dtype=x.dtype, gen=gen)
        orientation.append(idx)
        colors.append(sample_color_order(device=x.device, gen=gen))
    return AugmentedCellState(
        x=torch.remainder(x, float(L)), priorities=priorities, shift=shift,
        orientation_index=torch.tensor(orientation, device=x.device, dtype=torch.long),
        color_order=torch.stack(colors),
    )
