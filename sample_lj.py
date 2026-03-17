from __future__ import annotations

import argparse
import math
import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from grid_transformer.data.lj_abs_dataset import AbsoluteCoordinateTokenizer
from grid_transformer.data.lj_transferable import RelativeDeltaTokenizer
from grid_transformer.models.ar_registry import AR_ARCH_CHOICES, load_ar_checkpoint


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


def _compute_target_n(
    *,
    num_particles: Optional[int],
    density: Optional[float],
    box_lengths: Sequence[float],
) -> int:
    if density is not None:
        volume = float(np.prod(np.asarray(box_lengths, dtype=np.float64)))
        target_n = int(round(float(density) * volume))
        if target_n <= 0:
            raise ValueError(f"density implies non-positive particle count: {target_n}")
        return target_n
    if num_particles is None:
        raise ValueError("Provide --num_particles or --density.")
    if int(num_particles) <= 0:
        raise ValueError(f"num_particles must be > 0, got {num_particles}")
    return int(num_particles)


def _sampling_log_probs(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: Optional[int],
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled_logits = logits / float(temperature)
    if top_k is not None and top_k > 0 and top_k < scaled_logits.shape[-1]:
        v, _ = torch.topk(scaled_logits, k=top_k, dim=-1)
        kth = v[:, -1:].expand_as(scaled_logits)
        scaled_logits = torch.where(
            scaled_logits < kth,
            torch.full_like(scaled_logits, -1e9),
            scaled_logits,
        )
    return F.log_softmax(scaled_logits, dim=-1)


@torch.no_grad()
def autoregressive_unique_hilbert_sample(
    model: torch.nn.Module,
    *,
    n_particles: int,
    G: int,
    Lx: float,
    Ly: float,
    nsamples: int,
    sample_mode: str = "multinomial",
    temperature: float = 0.7,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    density: Optional[float] = None,
    periodic: bool = True,
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
    rho = float(density) if density is not None else float(n_particles / max(1e-8, (Lx * Ly)))
    density_tensor = torch.full((nsamples,), rho, dtype=torch.float32, device=device)

    for t in range(n_particles):
        logits = model(seq_in[:, : t + 1], box_size=box_size, density=density_tensor)[:, -1, :]  # [B, K]

        # Hard occupancy constraint: already-picked cells are forbidden.
        logits = logits.masked_fill(occupied, float("-inf"))
        log_probs = _sampling_log_probs(logits, temperature=temperature, top_k=top_k)
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
    if periodic:
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
        "density": torch.tensor(rho, dtype=torch.float32, device=device),
    }


@torch.no_grad()
def autoregressive_relative_delta_sample(
    model: torch.nn.Module,
    *,
    n_particles: int,
    box_lengths: Sequence[float],
    nsamples: int,
    tokenizer: RelativeDeltaTokenizer,
    sample_mode: str = "multinomial",
    temperature: float = 0.7,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    density: Optional[float] = None,
    periodic: bool = True,
) -> dict[str, torch.Tensor]:
    if n_particles <= 1:
        raise ValueError(f"n_particles must be > 1, got {n_particles}")
    if nsamples <= 0:
        raise ValueError(f"nsamples must be positive, got {nsamples}")

    device = next(model.parameters()).device
    K = int(model.K)
    if int(tokenizer.vocab_size) != K:
        base_vocab = int(tokenizer.base_vocab_size)
        hint = ""
        if base_vocab == K and tokenizer.use_long_jump_token:
            hint = (
                " Set `--relative_no_long_jump` to match this checkpoint, or sample with a "
                "checkpoint trained with a long-jump token."
            )
        elif base_vocab + 1 == K and (not tokenizer.use_long_jump_token):
            hint = (
                " This checkpoint appears to expect a long-jump token. Remove "
                "`--relative_no_long_jump`."
            )
        raise ValueError(
            f"Tokenizer vocab ({tokenizer.vocab_size}) does not match model K ({K}).{hint} "
            "Also ensure the checkpoint was trained with `--dataset lj_transferable` before "
            "using `--mode relative`."
        )
    if model.sos_id is None:
        raise ValueError("Checkpoint has no sos_id; cannot run autoregressive sampling.")
    sos_id = int(model.sos_id)
    coord_dim = int(tokenizer.dim)
    box_np = np.asarray(box_lengths, dtype=np.float32).reshape(-1)
    if box_np.size != coord_dim:
        raise ValueError(f"box_lengths must have length {coord_dim}, got {box_np.size}")

    gen = generator
    if gen is None and seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

    box_volume = float(np.prod(box_np, dtype=np.float64))
    rho = float(density) if density is not None else float(n_particles / max(1e-8, box_volume))
    density_tensor = torch.full((nsamples,), rho, dtype=torch.float32, device=device)
    box_size = torch.from_numpy(box_np).to(device=device, dtype=torch.float32).unsqueeze(0).expand(nsamples, -1)

    n_predict_particles = n_particles - 1
    factorized = bool(getattr(tokenizer, "factorized", False))
    n_predict_tokens = n_predict_particles * coord_dim if factorized else n_predict_particles

    seq_out = torch.empty((nsamples, n_predict_tokens), dtype=torch.long, device=device)
    seq_in = torch.empty((nsamples, n_predict_tokens), dtype=torch.long, device=device)
    seq_in[:, 0] = sos_id

    # Particle 0 is implicit during autoregressive generation and starts at the origin.
    # The model predicts particle 1 relative to particle 0 first, then particle 2
    # relative to particle 1, etc. COM=0 is enforced only after the full chain exists.
    x_base = torch.zeros((nsamples, n_particles, coord_dim), dtype=torch.float32, device=device)
    deltas = torch.zeros((nsamples, n_predict_particles, coord_dim), dtype=torch.float32, device=device)
    logp_discrete = torch.zeros((nsamples,), dtype=torch.float32, device=device)

    for t in range(n_predict_tokens):
        if factorized:
            x_base_rep = torch.repeat_interleave(x_base, coord_dim, dim=1)
            coords_in = x_base_rep[:, : t + 1, :].clone()
        else:
            coords_in = x_base[:, : t + 1, :].clone()

        logits = model(
            seq_in[:, : t + 1],
            coords=coords_in,
            box_size=box_size,
            density=density_tensor,
        )[:, -1, :]
        
        log_probs = _sampling_log_probs(logits, temperature=temperature, top_k=top_k)
        if sample_mode == "argmax":
            nxt = log_probs.argmax(dim=-1)
        elif sample_mode == "multinomial":
            probs = torch.exp(log_probs)
            nxt = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(1)
        else:
            raise ValueError(f"Unknown sample_mode '{sample_mode}'")

        seq_out[:, t] = nxt
        logp_discrete += log_probs.gather(1, nxt.unsqueeze(1)).squeeze(1)

        if factorized:
            if t % coord_dim == coord_dim - 1:
                particle_tokens = seq_out[:, t - coord_dim + 1 : t + 1]
                delta_t = tokenizer.decode(particle_tokens).squeeze(1).to(device=device, dtype=torch.float32)
                p_idx = (t // coord_dim) + 1
                raw_pos = x_base[:, p_idx - 1, :] + delta_t
                if periodic:
                    raw_pos = raw_pos - box_size * torch.round(raw_pos / box_size.clamp_min(1e-8))
                deltas[:, p_idx - 1, :] = delta_t
                x_base[:, p_idx, :] = raw_pos
        else:
            delta_t = tokenizer.decode(nxt).to(device=device, dtype=torch.float32)
            raw_pos = x_base[:, t, :] + delta_t
            if periodic:
                raw_pos = raw_pos - box_size * torch.round(raw_pos / box_size.clamp_min(1e-8))
            deltas[:, t, :] = delta_t
            x_base[:, t + 1, :] = raw_pos

        if t + 1 < n_predict_tokens:
            seq_in[:, t + 1] = nxt

    # Recover the final absolute frame by shifting the whole chain so COM=0.
    com = x_base.mean(dim=1, keepdim=True)
    final_positions = x_base - com
    
    if periodic:
        # Wrap final coordinates back into the primary simulation box [-L/2, L/2]
        final_positions = torch.remainder(final_positions + box_size/2, box_size) - box_size/2

    return {
        "token_ids": seq_out,
        "x_base": final_positions, # Returns fully formed (B, N, 3) geometry
        "deltas": deltas,          # Returns (B, N-1, 3) relative jumps
        "logp_discrete": logp_discrete,
        "density": torch.tensor(rho, dtype=torch.float32, device=device),
    }


@torch.no_grad()
def autoregressive_absolute_coordinate_sample(
    model: torch.nn.Module,
    *,
    n_particles: int,
    Lx: float,
    Ly: float,
    nsamples: int,
    tokenizer: AbsoluteCoordinateTokenizer,
    sample_mode: str = "multinomial",
    temperature: float = 0.7,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> dict[str, torch.Tensor]:
    if n_particles <= 0:
        raise ValueError(f"n_particles must be positive, got {n_particles}")
    if nsamples <= 0:
        raise ValueError(f"nsamples must be positive, got {nsamples}")

    device = next(model.parameters()).device
    K = int(model.K)
    if int(tokenizer.vocab_size) != K:
        raise ValueError(f"Absolute tokenizer vocab ({tokenizer.vocab_size}) does not match model K ({K}).")
    if model.sos_id is None:
        raise ValueError("Checkpoint has no sos_id; cannot run autoregressive sampling.")
    sos_id = int(model.sos_id)

    gen = generator
    if gen is None and seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))

    seq_len = int(n_particles) * 2
    seq_out = torch.empty((nsamples, seq_len), dtype=torch.long, device=device)
    seq_in = torch.empty((nsamples, seq_len), dtype=torch.long, device=device)
    seq_in[:, 0] = sos_id
    logp_discrete = torch.zeros((nsamples,), dtype=torch.float32, device=device)

    for t in range(seq_len):
        logits = model(seq_in[:, : t + 1])[:, -1, :]
        log_probs = _sampling_log_probs(logits, temperature=temperature, top_k=top_k)
        if sample_mode == "argmax":
            nxt = log_probs.argmax(dim=-1)
        elif sample_mode == "multinomial":
            probs = torch.exp(log_probs)
            nxt = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(1)
        else:
            raise ValueError(f"Unknown sample_mode '{sample_mode}'")

        seq_out[:, t] = nxt
        logp_discrete += log_probs.gather(1, nxt.unsqueeze(1)).squeeze(1)
        if t + 1 < seq_len:
            seq_in[:, t + 1] = nxt

    token_pairs = seq_out.view(nsamples, n_particles, 2)
    box_size = torch.tensor([float(Lx), float(Ly)], dtype=torch.float32, device=device).unsqueeze(0).expand(nsamples, -1)
    x_center = tokenizer.decode(token_pairs, box_size)
    cell = box_size[:, None, :] / float(tokenizer.bins)
    if gen is None:
        jitter = torch.rand_like(x_center) - 0.5
    else:
        jitter = torch.rand(x_center.shape, device=device, generator=gen, dtype=x_center.dtype) - 0.5
    x_base = torch.remainder(x_center + jitter * cell, box_size[:, None, :])

    return {
        "token_ids": seq_out,
        "token_pairs": token_pairs,
        "x_center": x_center,
        "x_base": x_base,
        "logp_discrete": logp_discrete,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sample LJ base states from a GraphormerAR checkpoint.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to GraphormerAR checkpoint.")
    ap.add_argument("--save", type=str, default="lj_samples/out.npz", help="Output NPZ path.")
    ap.add_argument("--nsamples", type=int, default=64, help="Number of samples to generate.")
    ap.add_argument(
        "--sample_batch_size",
        type=int,
        default=0,
        help="Sampling chunk size. If <=0, generate all nsamples in one batch.",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="hilbert",
        choices=("hilbert", "relative", "abs"),
        help="Sampling tokenization mode.",
    )
    ap.add_argument(
        "--num_particles",
        type=int,
        default=None,
        help="Number of tokens/particles to sample. If omitted, --density must be provided.",
    )
    ap.add_argument(
        "--density",
        type=float,
        default=None,
        help="Target number density rho. If set, N is computed from rho times box volume/area.",
    )
    ap.add_argument("--grid_size", type=int, default=None, help="Grid side G for hilbert mode; if omitted, inferred from K.")
    ap.add_argument("--Lx", type=float, required=True, help="Box length in x.")
    ap.add_argument("--Ly", type=float, required=True, help="Box length in y.")
    ap.add_argument("--Lz", type=float, default=None, help="Optional box length in z for 3D relative mode.")
    ap.add_argument("--coord_dim", type=int, default=None, help="Optional coordinate dimension override for relative mode.")
    ap.add_argument(
        "--periodic",
        dest="periodic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override checkpoint geometry. If omitted, sampling follows the checkpoint torus setting.",
    )
    ap.add_argument("--sample_mode", type=str, default="multinomial", choices=("multinomial", "argmax"))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", help="Sampling device: auto/cpu/cuda.")
    ap.add_argument(
        "--relative_window",
        type=float,
        default=3.0,
        help="Relative-mode local displacement window W.",
    )
    ap.add_argument(
        "--relative_bins",
        type=int,
        default=64,
        help="Relative-mode number of bins per axis.",
    )
    ap.add_argument(
        "--relative_no_long_jump",
        action="store_true",
        help="Relative mode: disable dedicated long-jump token and clamp displacements.",
    )
    ap.add_argument(
        "--factorized",
        action="store_true",
        help="Enable 1D sequential x,y,z tokenization.",
    )
    ap.add_argument(
        "--abs_bins",
        type=int,
        default=None,
        help="Absolute-coordinate mode bins per axis. If omitted, inferred from model vocabulary size.",
    )
    ap.add_argument(
        "--ar_arch",
        type=str,
        choices=AR_ARCH_CHOICES,
        default="auto",
        help="AR checkpoint architecture: auto/standard/ida/vanilla. Auto infers from checkpoint metadata.",
    )
    ap.add_argument(
        "--use_ida",
        dest="use_ida",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Legacy alias used only when --ar_arch=auto. True=>ida, False=>standard.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if float(args.temperature) <= 0.0:
        raise ValueError(f"--temperature must be > 0, got {args.temperature}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, resolved_arch = load_ar_checkpoint(
        args.ckpt,
        device,
        ar_arch=str(args.ar_arch),
        use_ida=bool(args.use_ida),
    )
    K = int(model.K)
    periodic = bool(args.periodic) if args.periodic is not None else bool(getattr(model, "torus", True))
    inferred_coord_dim = args.coord_dim
    if inferred_coord_dim is None:
        inferred_coord_dim = getattr(getattr(model, "hparams", None), "spatial_dim", None)
    coord_dim = int(inferred_coord_dim) if inferred_coord_dim is not None else 2
    if coord_dim not in (2, 3):
        raise ValueError(f"Sampling currently supports coord_dim 2 or 3, got {coord_dim}")
    if coord_dim == 3:
        if args.Lz is None:
            raise ValueError("--Lz is required for coord_dim=3 relative sampling.")
        box_lengths = [float(args.Lx), float(args.Ly), float(args.Lz)]
    else:
        box_lengths = [float(args.Lx), float(args.Ly)]

    target_n = _compute_target_n(
        num_particles=args.num_particles,
        density=args.density,
        box_lengths=box_lengths,
    )

    total = int(args.nsamples)
    if total <= 0:
        raise ValueError(f"--nsamples must be > 0, got {total}")
    chunk = int(args.sample_batch_size) if int(args.sample_batch_size) > 0 else total

    shared_gen = torch.Generator(device=device)
    shared_gen.manual_seed(int(args.seed))

    chunks = []
    done = 0
    if args.mode == "hilbert":
        G = int(args.grid_size) if args.grid_size is not None else _infer_grid_size_from_vocab(K)
        while done < total:
            bsz = min(chunk, total - done)
            out_chunk = autoregressive_unique_hilbert_sample(
                model,
                n_particles=target_n,
                G=G,
                Lx=float(args.Lx),
                Ly=float(args.Ly),
                nsamples=bsz,
                sample_mode=args.sample_mode,
                temperature=float(args.temperature),
                top_k=args.top_k,
                generator=shared_gen,
                density=args.density,
                periodic=periodic,
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
            "density": chunks[0]["density"],
        }

        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        np.savez(
            args.save,
            mode=np.array(args.mode),
            hilbert_ids=out["hilbert_ids"].cpu().numpy().astype(np.int32),
            ix=out["ix"].cpu().numpy().astype(np.int32),
            iy=out["iy"].cpu().numpy().astype(np.int32),
            x_center=out["x_center"].cpu().numpy().astype(np.float32),
            x_base=out["x_base"].cpu().numpy().astype(np.float32),
            logp_discrete=out["logp_discrete"].cpu().numpy().astype(np.float32),
            logp_base=out["logp_base"].cpu().numpy().astype(np.float32),
            cell_volume=np.array(float(out["cell_volume"].item()), dtype=np.float32),
            rho=np.array(float(out["density"].item()), dtype=np.float32),
            G=np.array(G, dtype=np.int32),
            K=np.array(K, dtype=np.int32),
            N=np.array(target_n, dtype=np.int32),
            L=np.array([args.Lx, args.Ly], dtype=np.float32),
            periodic=np.array(int(periodic), dtype=np.int32),
            temperature=np.array(args.temperature, dtype=np.float32),
            sample_mode=np.array(args.sample_mode),
            top_k=np.array(-1 if args.top_k is None else int(args.top_k), dtype=np.int32),
            ar_arch=np.array(resolved_arch),
        )

        print(
            f"Saved {args.save} (mode=hilbert, nsamples={args.nsamples}, N={target_n}, "
            f"G={G}, K={K}, periodic={periodic}, ar_arch={resolved_arch})"
        )
        print(f"Mean logp_discrete: {out['logp_discrete'].mean().item():.6f}")
        print(f"Mean logp_base:     {out['logp_base'].mean().item():.6f}")
        return

    tokenizer = RelativeDeltaTokenizer(
        window=float(args.relative_window),
        bins=int(args.relative_bins),
        dim=coord_dim,
        use_long_jump_token=(not args.relative_no_long_jump),
        factorized=bool(args.factorized),
    )

    if args.mode == "abs":
        abs_bins = int(args.abs_bins) if args.abs_bins is not None else int(K)
        tokenizer_abs = AbsoluteCoordinateTokenizer(bins=abs_bins)
        while done < total:
            bsz = min(chunk, total - done)
            out_chunk = autoregressive_absolute_coordinate_sample(
                model,
                n_particles=target_n,
                Lx=float(args.Lx),
                Ly=float(args.Ly),
                nsamples=bsz,
                tokenizer=tokenizer_abs,
                sample_mode=args.sample_mode,
                temperature=float(args.temperature),
                top_k=args.top_k,
                generator=shared_gen,
            )
            chunks.append(out_chunk)
            done += bsz

        out = {
            "token_ids": torch.cat([c["token_ids"] for c in chunks], dim=0),
            "token_pairs": torch.cat([c["token_pairs"] for c in chunks], dim=0),
            "x_center": torch.cat([c["x_center"] for c in chunks], dim=0),
            "x_base": torch.cat([c["x_base"] for c in chunks], dim=0),
            "logp_discrete": torch.cat([c["logp_discrete"] for c in chunks], dim=0),
        }

        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        np.savez(
            args.save,
            mode=np.array(args.mode),
            token_ids=out["token_ids"].cpu().numpy().astype(np.int32),
            token_pairs=out["token_pairs"].cpu().numpy().astype(np.int32),
            x_center=out["x_center"].cpu().numpy().astype(np.float32),
            x_base=out["x_base"].cpu().numpy().astype(np.float32),
            logp_discrete=out["logp_discrete"].cpu().numpy().astype(np.float32),
            K=np.array(K, dtype=np.int32),
            N=np.array(target_n, dtype=np.int32),
            L=np.array([args.Lx, args.Ly], dtype=np.float32),
            bins=np.array(int(tokenizer_abs.bins), dtype=np.int32),
            periodic=np.array(1, dtype=np.int32),
            temperature=np.array(args.temperature, dtype=np.float32),
            sample_mode=np.array(args.sample_mode),
            top_k=np.array(-1 if args.top_k is None else int(args.top_k), dtype=np.int32),
            ar_arch=np.array(resolved_arch),
        )

        print(
            f"Saved {args.save} (mode=abs, nsamples={args.nsamples}, N={target_n}, "
            f"K={K}, bins={tokenizer_abs.bins}, ar_arch={resolved_arch})"
        )
        print(f"Mean logp_discrete: {out['logp_discrete'].mean().item():.6f}")
        return

    while done < total:
        bsz = min(chunk, total - done)
        out_chunk = autoregressive_relative_delta_sample(
            model,
            n_particles=target_n,
            box_lengths=box_lengths,
            nsamples=bsz,
            tokenizer=tokenizer,
            sample_mode=args.sample_mode,
            temperature=float(args.temperature),
            top_k=args.top_k,
            generator=shared_gen,
            density=args.density,
            periodic=periodic,
        )
        chunks.append(out_chunk)
        done += bsz

    out = {
        "token_ids": torch.cat([c["token_ids"] for c in chunks], dim=0),
        "x_base": torch.cat([c["x_base"] for c in chunks], dim=0),
        "deltas": torch.cat([c["deltas"] for c in chunks], dim=0),
        "logp_discrete": torch.cat([c["logp_discrete"] for c in chunks], dim=0),
        "density": chunks[0]["density"],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
    np.savez(
        args.save,
        mode=np.array(args.mode),
        token_ids=out["token_ids"].cpu().numpy().astype(np.int32),
        x_base=out["x_base"].cpu().numpy().astype(np.float32),
        deltas=out["deltas"].cpu().numpy().astype(np.float32),
        logp_discrete=out["logp_discrete"].cpu().numpy().astype(np.float32),
        rho=np.array(float(out["density"].item()), dtype=np.float32),
        K=np.array(K, dtype=np.int32),
        N=np.array(target_n, dtype=np.int32),
        L=np.array(box_lengths, dtype=np.float32),
        window=np.array(float(args.relative_window), dtype=np.float32),
        bins=np.array(int(args.relative_bins), dtype=np.int32),
        use_long_jump=np.array(int(not args.relative_no_long_jump), dtype=np.int32),
        factorized=np.array(int(bool(args.factorized)), dtype=np.int32),
        periodic=np.array(int(periodic), dtype=np.int32),
        temperature=np.array(args.temperature, dtype=np.float32),
        sample_mode=np.array(args.sample_mode),
        top_k=np.array(-1 if args.top_k is None else int(args.top_k), dtype=np.int32),
        ar_arch=np.array(resolved_arch),
    )

    print(
        f"Saved {args.save} (mode=relative, nsamples={args.nsamples}, N={target_n}, "
        f"K={K}, bins={args.relative_bins}, window={args.relative_window}, factorized={bool(args.factorized)}, periodic={periodic}, "
        f"ar_arch={resolved_arch})"
    )
    print(f"Mean logp_discrete: {out['logp_discrete'].mean().item():.6f}")


if __name__ == "__main__":
    main()
