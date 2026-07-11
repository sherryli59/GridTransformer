"""Overfit-first training run for the positions-only G2 cavity generator."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from liquid_coupling_flow.ka3d_cavity_carve import carve_batch
from liquid_coupling_flow.ka3d_cavity_generator import CavityGenerator, cavity_fm_loss, collate_cavities


def _take(batch, idx):
    return {k: (v[idx] if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == batch["x_in"].shape[0] else v)
            for k, v in batch.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt"))
    p.add_argument("--out", type=Path, default=Path("liquid_coupling_flow/artifacts/ka3d_cavity_generator_g2.pt"))
    p.add_argument("--n-boundaries", type=int, default=48)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--n-ctx-max", type=int, default=192)
    p.add_argument("--n-max", type=int, default=112)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); torch.manual_seed(0)
    d = torch.load(a.dataset, map_location=a.device, weights_only=False)
    x, s, L = d["x"][:a.n_boundaries].float(), d["s"][:a.n_boundaries].long(), float(d["L"])
    pairs = carve_batch(x, s, L, [1.6, 2.4], 1,
                        torch.Generator(device=a.device).manual_seed(12))
    for pair in pairs: pair["T"] = float(d["T"])
    if max(pair["n_in"] for pair in pairs) > a.n_max:
        raise ValueError(f"--n-max={a.n_max} is smaller than a carved training interior")
    n_max = a.n_max
    batch = collate_cavities(pairs, n_max=n_max, n_ctx_max=a.n_ctx_max)
    model = CavityGenerator(n_max, a.hidden, a.layers, a.n_ctx_max).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    t0 = time.time(); first = None
    for step in range(a.steps):
        idx = torch.randint(len(pairs), (min(a.batch_size, len(pairs)),), device=a.device)
        loss = cavity_fm_loss(model, _take(batch, idx))
        if first is None: first = float(loss)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10); opt.step()
        if step % 100 == 0 or step == a.steps - 1:
            print(f"[cavity-gen] step={step} loss={float(loss):.5f} elapsed={(time.time()-t0)/60:.1f}m", flush=True)
    with torch.no_grad():
        sample = model.sample(batch["s_in"], batch["mask"], batch["x_ctx"], batch["s_ctx"],
                              batch["ctx_mask"], batch["R"], batch["T"])
        rms = (((sample - batch["x_in"]) ** 2).sum(-1)[batch["mask"]].mean().sqrt()).item()
    cpu_pairs = [{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in pair.items()} for pair in pairs]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": {"n_max": n_max, "hidden": a.hidden,
                "layers": a.layers, "n_ctx_max": a.n_ctx_max}, "pairs": cpu_pairs,
                "dataset": str(a.dataset), "first_loss": first, "final_loss": float(loss),
                "sample_target_rms": rms}, a.out)
    print(f"[cavity-gen] DONE first={first:.5f} final={float(loss):.5f} sample-target-rms={rms:.4f} -> {a.out}", flush=True)


if __name__ == "__main__": main()
