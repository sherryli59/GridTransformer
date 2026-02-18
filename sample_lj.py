from __future__ import annotations

import argparse
import math
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from grid_transformer.training.ar import GraphormerAR


def _is_power_of_two(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


def _hilbert_d2xy(order_n: int, d: int) -> tuple[int, int]:
    """Distance-to-2D Hilbert coordinate for square side length `order_n`."""
    x = 0
    y = 0
    t = int(d)
    s = 1
    while s < order_n:
        rx = (t // 2) & 1
        ry = (t ^ rx) & 1
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s <<= 1
    return x, y


def hilbert_to_ijk(h: torch.LongTensor, G: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Map 1D Hilbert IDs back to 2D grid indices (ix, iy).

    h: Long tensor of arbitrary shape with values in [0, G*G-1].
    G: grid resolution.
    """
    if not _is_power_of_two(int(G)):
        raise ValueError(f"G must be a power of two for Hilbert mapping, got {G}")
    h_flat = h.reshape(-1).to(torch.long)
    if (h_flat < 0).any() or (h_flat >= int(G * G)).any():
        raise ValueError("Hilbert token out of valid range [0, G^2-1].")
    device = h.device
    lut = torch.empty((G * G, 2), dtype=torch.long)
    for d in range(G * G):
        x, y = _hilbert_d2xy(G, d)
        lut[d, 0] = x
        lut[d, 1] = y
    lut = lut.to(device)
    xy = lut[h_flat]
    ix = xy[:, 0].reshape_as(h)
    iy = xy[:, 1].reshape_as(h)
    return ix, iy


def _infer_grid_size_from_vocab(K: int) -> int:
    G = int(round(math.sqrt(K)))
    if G * G != K:
        raise ValueError(f"Vocabulary size K={K} is not a perfect square, cannot infer grid size.")
    return G


def _load_graphormer_checkpoint(path: str, device: torch.device) -> GraphormerAR:
    model = GraphormerAR.load_from_checkpoint(path, map_location=device)
    model.to(device).eval()
    return model


@torch.no_grad()
def autoregressive_unique_hilbert_sample(
    model: GraphormerAR,
    *,
    n_particles: int,
    G: int,
    Lx: float,
    Ly: float,
    nsamples: int,
    sample_mode: str = "multinomial",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> dict[str, torch.Tensor]:
    """
    Sample N unique Hilbert cell IDs autoregressively with exact log-likelihood tracking.
    """
    if n_particles <= 0:
        raise ValueError(f"n_particles must be positive, got {n_particles}")
    if G <= 0:
        raise ValueError(f"G must be positive, got {G}")
    if nsamples <= 0:
        raise ValueError(f"nsamples must be positive, got {nsamples}")

    device = next(model.parameters()).device
    K = int(model.K)
    if K != G * G:
        raise ValueError(f"Model vocabulary K={K} does not match G^2={G * G}")
    if n_particles > K:
        raise ValueError(f"n_particles={n_particles} exceeds number of cells K={K}")
    if model.sos_id is None:
        raise ValueError("Checkpoint has no sos_id; cannot run autoregressive sampling.")
    sos_id = int(model.sos_id)

    gen = generator
    if gen is None and seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

    seq_out = torch.empty((nsamples, n_particles), dtype=torch.long, device=device)
    seq_in = torch.empty((nsamples, n_particles), dtype=torch.long, device=device)
    seq_in[:, 0] = sos_id
    occupied = torch.zeros((nsamples, K), dtype=torch.bool, device=device)
    logp_discrete = torch.zeros((nsamples,), dtype=torch.float32, device=device)

    box_size = torch.tensor([float(Lx), float(Ly)], dtype=torch.float32, device=device).unsqueeze(0).expand(nsamples, -1)

    for t in range(n_particles):
        logits = model(seq_in[:, : t + 1], box_size=box_size)[:, -1, :]  # [B, K]
        if temperature != 1.0:
            logits = logits / float(max(1e-8, temperature))

        # Hard occupancy constraint: already-picked cells are forbidden.
        logits = logits.masked_fill(occupied, float("-inf"))

        if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
            v, _ = torch.topk(logits, k=top_k, dim=-1)
            kth = v[:, -1:].expand_as(logits)
            logits = torch.where(logits < kth, torch.full_like(logits, -1e9), logits)

        log_probs = F.log_softmax(logits, dim=-1)
        if sample_mode == "argmax":
            nxt = log_probs.argmax(dim=-1)
        elif sample_mode == "multinomial":
            probs = torch.exp(log_probs)
            nxt = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(1)
        else:
            raise ValueError(f"Unknown sample_mode '{sample_mode}'")

        seq_out[:, t] = nxt
        logp_discrete += log_probs.gather(1, nxt.unsqueeze(1)).squeeze(1)
        occupied.scatter_(1, nxt.unsqueeze(1), True)
        if t + 1 < n_particles:
            seq_in[:, t + 1] = nxt

    # Decode Hilbert IDs -> grid cell indices.
    ix, iy = hilbert_to_ijk(seq_out, G)
    ix_f = ix.to(torch.float32)
    iy_f = iy.to(torch.float32)

    dx = float(Lx) / float(G)
    dy = float(Ly) / float(G)

    x_center = (ix_f + 0.5) * dx
    y_center = (iy_f + 0.5) * dy
    centers = torch.stack([x_center, y_center], dim=-1)  # [B, N, 2]

    # Uniform de-quantization in each selected cell.
    if gen is None:
        jitter = torch.rand_like(centers) - 0.5
    else:
        jitter = torch.rand(centers.shape, device=device, generator=gen, dtype=centers.dtype) - 0.5
    scales = torch.tensor([dx, dy], dtype=centers.dtype, device=device).view(1, 1, 2)
    x_base = centers + jitter * scales
    x_base[..., 0] = torch.remainder(x_base[..., 0], float(Lx))
    x_base[..., 1] = torch.remainder(x_base[..., 1], float(Ly))

    cell_volume = dx * dy
    logp_base = logp_discrete - float(n_particles) * math.log(cell_volume)

    return {
        "hilbert_ids": seq_out,
        "ix": ix,
        "iy": iy,
        "x_center": centers,
        "x_base": x_base,
        "logp_discrete": logp_discrete,
        "logp_base": logp_base,
        "cell_volume": torch.tensor(cell_volume, dtype=torch.float32, device=device),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sample LJ base states from a GraphormerAR lj_cell checkpoint.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to GraphormerAR checkpoint.")
    ap.add_argument("--save", type=str, default="lj_samples/out.npz", help="Output NPZ path.")
    ap.add_argument("--nsamples", type=int, default=64, help="Number of samples to generate.")
    ap.add_argument(
        "--sample_batch_size",
        type=int,
        default=0,
        help="Sampling chunk size. If <=0, generate all nsamples in one batch.",
    )
    ap.add_argument("--num_particles", type=int, default=16, help="Number of particles/tokens to sample.")
    ap.add_argument("--grid_size", type=int, default=None, help="Grid side G. If omitted, inferred from K via sqrt(K).")
    ap.add_argument("--Lx", type=float, required=True, help="Box length in x.")
    ap.add_argument("--Ly", type=float, required=True, help="Box length in y.")
    ap.add_argument("--sample_mode", type=str, default="multinomial", choices=("multinomial", "argmax"))
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", help="Sampling device: auto/cpu/cuda.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = _load_graphormer_checkpoint(args.ckpt, device)
    K = int(model.K)
    G = int(args.grid_size) if args.grid_size is not None else _infer_grid_size_from_vocab(K)
    total = int(args.nsamples)
    if total <= 0:
        raise ValueError(f"--nsamples must be > 0, got {total}")
    chunk = int(args.sample_batch_size) if int(args.sample_batch_size) > 0 else total

    shared_gen = torch.Generator(device=device)
    shared_gen.manual_seed(int(args.seed))

    chunks = []
    done = 0
    while done < total:
        bsz = min(chunk, total - done)
        out_chunk = autoregressive_unique_hilbert_sample(
            model,
            n_particles=int(args.num_particles),
            G=G,
            Lx=float(args.Lx),
            Ly=float(args.Ly),
            nsamples=bsz,
            sample_mode=args.sample_mode,
            temperature=float(args.temperature),
            top_k=args.top_k,
            generator=shared_gen,
        )
        chunks.append(out_chunk)
        done += bsz

    out = {
        "hilbert_ids": torch.cat([c["hilbert_ids"] for c in chunks], dim=0),
        "ix": torch.cat([c["ix"] for c in chunks], dim=0),
        "iy": torch.cat([c["iy"] for c in chunks], dim=0),
        "x_center": torch.cat([c["x_center"] for c in chunks], dim=0),
        "x_base": torch.cat([c["x_base"] for c in chunks], dim=0),
        "logp_discrete": torch.cat([c["logp_discrete"] for c in chunks], dim=0),
        "logp_base": torch.cat([c["logp_base"] for c in chunks], dim=0),
        "cell_volume": chunks[0]["cell_volume"],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    np.savez(
        args.save,
        hilbert_ids=out["hilbert_ids"].cpu().numpy().astype(np.int32),
        ix=out["ix"].cpu().numpy().astype(np.int32),
        iy=out["iy"].cpu().numpy().astype(np.int32),
        x_center=out["x_center"].cpu().numpy().astype(np.float32),
        x_base=out["x_base"].cpu().numpy().astype(np.float32),
        logp_discrete=out["logp_discrete"].cpu().numpy().astype(np.float32),
        logp_base=out["logp_base"].cpu().numpy().astype(np.float32),
        cell_volume=np.array(float(out["cell_volume"].item()), dtype=np.float32),
        G=np.array(G, dtype=np.int32),
        K=np.array(K, dtype=np.int32),
        N=np.array(args.num_particles, dtype=np.int32),
        L=np.array([args.Lx, args.Ly], dtype=np.float32),
        temperature=np.array(args.temperature, dtype=np.float32),
        sample_mode=np.array(args.sample_mode),
        top_k=np.array(-1 if args.top_k is None else int(args.top_k), dtype=np.int32),
    )

    print(f"Saved {args.save} (nsamples={args.nsamples}, N={args.num_particles}, G={G}, K={K})")
    print(f"Mean logp_discrete: {out['logp_discrete'].mean().item():.6f}")
    print(f"Mean logp_base:     {out['logp_base'].mean().item():.6f}")


if __name__ == "__main__":
    main()
