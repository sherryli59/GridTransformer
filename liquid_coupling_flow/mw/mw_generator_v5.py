"""Exact local-origin autoregressive generator for the N=64 mW liquid (v5a).

The scaffold chooses an ordering and a region to inspect.  The density itself is
expressed relative to a causal, prefix-derived origin.  A continuous local frame
is used for the conditioning tokens, but deliberately not for the output chart:
an arbitrary rotation does not map the periodic cubic box onto itself.  Output
coordinates instead use the global minimum-image cube and a circular RQ-spline
head.  This makes every conditional a normalized density on the three-torus and
keeps sample/log_prob exact on every draw (there is no wrapped-tail exception).
"""
from __future__ import annotations

import math
import os
import torch
import torch.nn as nn

from liquid_coupling_flow.transforms_spline import (
    CircularRQSplineElementwise,
    DEFAULT_MIN_DERIVATIVE,
)
from liquid_coupling_flow.mw.mw_energy import RHO_STAR
from liquid_coupling_flow.mw.mw_generator import (
    ART,
    DEV,
    EV_RMIN,
    R_SHELL1,
    _augment_batch,
    build_frames,
    canonical_order,
    load_training_bank,
    mw_scaffold,
    wrap_pm,
)


class CircularSpline3Head(nn.Module):
    """AR-chained circular RQ splines on ``[-bound, bound)^3``.

    A uniform base on each circle supplies the normalized base density.  Previous
    coordinates enter through sine/cosine embeddings so the conditioning is also
    continuous across the periodic seam.
    """

    def __init__(self, d_model: int, num_bins: int = 32, bound: float = 4.0):
        super().__init__()
        self.bound = float(bound)
        self.period = 2.0 * self.bound
        self.spline = CircularRQSplineElementwise(num_bins=num_bins, L=self.period)
        self.num_bins = int(num_bins)
        P = self.spline.params_per_dim
        self.heads = nn.ModuleList([nn.Linear(d_model + 2 * i, P) for i in range(3)])
        const = float(torch.log(torch.expm1(torch.tensor(1.0 - DEFAULT_MIN_DERIVATIVE))))
        for head in self.heads:
            nn.init.zeros_(head.weight)
            with torch.no_grad():
                head.bias.zero_()
                head.bias[2 * self.num_bins:] = const

    def _circle(self, u: torch.Tensor) -> torch.Tensor:
        return torch.remainder(u + self.bound, self.period)

    def _embed(self, q: torch.Tensor) -> torch.Tensor:
        a = 2.0 * math.pi * q / self.period
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

    def logp_a(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q = self._circle(a)
        _, ld = self.spline.inverse(q, self.heads[0](h))
        return -math.log(self.period) + ld

    def log_prob(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        lp = torch.zeros(h.shape[:-1], device=h.device, dtype=h.dtype)
        ctx = h
        for i in range(3):
            q = self._circle(u[..., i:i + 1])
            _, ld = self.spline.inverse(q, self.heads[i](ctx))
            lp = lp - math.log(self.period) + ld.squeeze(-1)
            ctx = torch.cat([ctx, self._embed(q)], -1)
        return lp

    def sample(self, h: torch.Tensor, gen=None):
        lp = torch.zeros(h.shape[:-1], device=h.device, dtype=h.dtype)
        ctx, us = h, []
        for i in range(3):
            z = torch.rand(h.shape[:-1] + (1,), device=h.device, dtype=h.dtype, generator=gen)
            z = z * self.period
            q, ld = self.spline.forward(z, self.heads[i](ctx))
            lp = lp - math.log(self.period) - ld.squeeze(-1)
            us.append(q - self.bound)
            ctx = torch.cat([ctx, self._embed(q)], -1)
        return torch.cat(us, -1), lp


class MWLocalFrameAR(nn.Module):
    """Causal local-origin mW generator with exact torus conditionals."""

    def __init__(self, knn=12, d_model=128, n_layers=2, n_heads=4,
                 num_bins=32, tail_bound=4.0, sigma_origin=1.30,
                 periods=(0.5, 1.0, 2.0, 4.0)):
        super().__init__()
        self.knn = int(knn)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.num_bins = int(num_bins)
        self.tail_bound = float(tail_bound)
        self.sigma_origin = float(sigma_origin)
        self.periods = tuple(float(p) for p in periods)
        # frame xyz, r, radial Fourier, inverse-r2, r/r_shell, occupancy, tetra angle
        n_feat = 3 + 1 + 2 * len(self.periods) + 4
        self.embed = nn.Linear(n_feat, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=0.0,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = CircularSpline3Head(d_model, num_bins=num_bins, bound=tail_bound)

    def _fourier(self, r):
        out = []
        for p in self.periods:
            out += [torch.sin(2 * math.pi * r / p), torch.cos(2 * math.pi * r / p)]
        return torch.cat(out, -1)

    def _origins(self, xo, anchors, L):
        """Vectorized causal Gaussian-centroid origins, shape ``[B,N,3]``."""
        B, N, _ = xo.shape
        d = wrap_pm(xo[:, None] - anchors[None, :, None], L)  # [B, query j, particle i, 3]
        jj = torch.arange(N, device=xo.device)
        causal = jj[None, None, :] < jj[None, :, None]
        w = torch.exp(-d.square().sum(-1) / (2 * self.sigma_origin ** 2)) * causal
        denom = w.sum(-1, keepdim=True)
        shift = (w[..., None] * d).sum(2) / denom.clamp_min(1e-12)
        origins = torch.remainder(anchors[None] + shift, L)
        return torch.where(denom > 0, origins, anchors[None].expand(B, -1, -1))

    def _geometry(self, xo, anchors, L):
        """Return origins, local frames, neighbor vectors and validity for all steps."""
        B, N, _ = xo.shape
        origins = self._origins(xo, anchors, L)
        rel_all = wrap_pm(xo[:, None] - origins[:, :, None], L)
        dist2 = rel_all.square().sum(-1)
        jj = torch.arange(N, device=xo.device)
        causal = jj[None, None, :] < jj[None, :, None]
        K = min(self.knn, N)
        idx = dist2.masked_fill(~causal, float("inf")).topk(K, 2, largest=False).indices
        rel = torch.gather(rel_all, 2, idx[..., None].expand(-1, -1, -1, 3))
        valid = torch.gather(causal.expand(B, N, N), 2, idx)
        rel = rel * valid[..., None]
        n_valid = valid.sum(-1)
        frames = build_frames(rel.reshape(B * N, K, 3), n_valid.reshape(-1)).reshape(B, N, 3, 3)
        return origins, frames, rel, valid

    def _token_features(self, frames, rel, valid):
        r = rel.norm(dim=-1, keepdim=True)
        xyz = torch.einsum("...ij,...mj->...mi", frames, rel)
        inv_r2 = 1.0 / r.clamp_min(EV_RMIN).square()
        occupancy = valid.sum(-1, keepdim=True).to(r.dtype) / self.knn
        occupancy = occupancy[..., None].expand(*r.shape[:-2], r.shape[-2], 1)
        ref = rel[..., :1, :]
        cos = (rel * ref).sum(-1, keepdim=True) / (r * r[..., :1, :]).clamp_min(1e-8)
        tetra = cos + 1.0 / 3.0
        feat = torch.cat([xyz, r, self._fourier(r), inv_r2, r / R_SHELL1,
                          occupancy, tetra], -1)
        return feat * valid[..., None]

    def _hidden(self, xo, anchors, L):
        B, N, _ = xo.shape
        origins, frames, rel, valid = self._geometry(xo, anchors, L)
        K = rel.shape[2]
        tok = self.embed(self._token_features(frames, rel, valid))
        seq = torch.cat([self.cls.expand(B, N, 1, -1), tok], 2)
        pad = torch.cat([torch.zeros(B, N, 1, dtype=torch.bool, device=xo.device), ~valid], 2)
        flat, mask = seq.reshape(B * N, K + 1, -1), pad.reshape(B * N, K + 1)
        outs = []
        for i in range(0, B * N, 8192):
            outs.append(self.encoder(flat[i:i + 8192], src_key_padding_mask=mask[i:i + 8192])[:, 0])
        return torch.cat(outs).reshape(B, N, -1), origins

    def _hidden_step(self, prefix, anchor, L):
        """Current-step mirror of ``_hidden`` without recomputing older queries."""
        B, j, _ = prefix.shape
        if j == 0:
            origin = anchor.expand(B, 3)
        else:
            da = wrap_pm(prefix - anchor, L)
            w = torch.exp(-da.square().sum(-1) / (2 * self.sigma_origin ** 2))
            origin = torch.remainder(anchor + (w[..., None] * da).sum(1)
                                     / w.sum(1, keepdim=True).clamp_min(1e-12), L)
        k = min(self.knn, j)
        if k:
            all_rel = wrap_pm(prefix - origin[:, None], L)
            idx = all_rel.square().sum(-1).topk(k, 1, largest=False).indices
            real = torch.gather(all_rel, 1, idx[..., None].expand(-1, -1, 3))
        else:
            real = prefix.new_zeros(B, 0, 3)
        K = self.knn
        rel = torch.cat([real, prefix.new_zeros(B, K - k, 3)], 1)
        valid = torch.arange(K, device=prefix.device)[None] < k
        valid = valid.expand(B, -1)
        frame = build_frames(rel, torch.full((B,), k, device=prefix.device, dtype=torch.long))
        tok = self.embed(self._token_features(frame, rel, valid))
        seq = torch.cat([self.cls.expand(B, 1, -1), tok], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=prefix.device), ~valid], 1)
        return self.encoder(seq, src_key_padding_mask=pad)[:, 0], origin

    def _ordered(self, x, L, preordered):
        x = torch.remainder(x, L)
        anchors, rank, R = mw_scaffold(x.shape[1], L, x.device)
        if preordered:
            return x, anchors, R
        perm = canonical_order(x, L, R, rank)
        return torch.gather(x, 1, perm[..., None].expand(-1, -1, 3)), anchors, R

    def log_prob(self, x, L, preordered=False):
        xo, anchors, _ = self._ordered(x, L, preordered)
        h, origins = self._hidden(xo, anchors, L)
        # Global minimum-image chart: unlike a continuously rotated cube, this is
        # exactly the cubic torus fundamental domain.
        s = L / (2 * self.tail_bound)
        u = wrap_pm(xo - origins, L) / s
        return self.head.log_prob(h, u).sum(1) - 3 * x.shape[1] * math.log(s)

    @torch.no_grad()
    def sample(self, B, N, L, gen=None, return_logq=True):
        device = next(self.parameters()).device
        anchors, _, _ = mw_scaffold(N, L, device)
        s = L / (2 * self.tail_bound)
        x = torch.zeros(B, N, 3, device=device)
        logq = torch.zeros(B, device=device)
        for j in range(N):
            h, origin = self._hidden_step(x[:, :j], anchors[j], L)
            u, lp = self.head.sample(h, gen=gen)
            x[:, j] = torch.remainder(origin + s * u, L)
            logq += lp - 3 * math.log(s)
        return (x, logq) if return_logq else x

    @torch.no_grad()
    def teacher_forced_structure(self, x, L, nbins=120, gen=None):
        """Pooled-predecessor TF g(r), peak error, and excluded-core mass."""
        xo, anchors, _ = self._ordered(x, L, preordered=False)
        h, origins = self._hidden(xo, anchors, L)
        u, _ = self.head.sample(h, gen=gen)
        s = L / (2 * self.tail_bound)
        placed = torch.remainder(origins + s * u, L)
        d = wrap_pm(placed[:, :, None] - xo[:, None], L).norm(dim=-1)
        N = xo.shape[1]
        jj = torch.arange(N, device=x.device)
        causal = jj[None, :, None] > jj[None, None, :]
        vals = d[causal.expand(len(xo), -1, -1)]
        edges = torch.linspace(0, L / 2, nbins + 1, device=x.device)
        # torch.histogram is CPU-only; counts rejoin the device for the normalization arithmetic
        counts = torch.histogram(vals.float().cpu(), bins=nbins, range=(0.0, L / 2))[0].to(x.device)
        r = (edges[1:] + edges[:-1]) / 2
        dr = edges[1] - edges[0]
        norm = len(xo) * (N * (N - 1) / 2) * (4 * math.pi * r.square() * dr) / (L ** 3)
        gr = counts / norm.clamp_min(1e-12)
        ipeak = (r - R_SHELL1).abs().argmin()
        peak_err = (gr[ipeak] - 2.18).abs() / 2.18
        core_mass = gr[r < 1.0].mean()
        return {"r": r, "gr": gr, "peak_err": peak_err, "core_mass": core_mass,
                "composite": peak_err + core_mass}


def _checkpoint(model, step, val_nll, struct, L):
    return {
        "state_dict": model.state_dict(), "step": step, "val_nll": val_nll,
        "struct": {k: (float(v) if v.numel() == 1 else v.cpu()) for k, v in struct.items()},
        "train_N": round(RHO_STAR * L ** 3), "architecture": "local_origin_circular_spline_v5a_exact",
        **{k: getattr(model, k) for k in ("knn", "d_model", "n_layers", "n_heads",
                                          "num_bins", "tail_bound", "sigma_origin", "periods")},
    }


def train(steps=30_000, batch=32, lr=3e-4, val_every=500, seed=53,
          out="mw_gen_N64_v5.pt", primary_thin=2, val_frac=0.1,
          extension="mw_ref_N64_ext.pt", device=None, art_path=None,
          extra_banks=None, **model_kw):
    """Train v5 and save independent best-NLL and best-structure checkpoints."""
    torch.manual_seed(seed)
    device = device or DEV
    if extra_banks is None:
        ext = extension if os.path.exists(extension) else os.path.join(ART, extension)
        extra_banks = [(ext, 32, 1)]
    xtr, xva, L = load_training_bank(art_path=art_path, thin_events=primary_thin,
                                     val_frac=val_frac, extra_banks=extra_banks)
    xtr, xva = xtr.to(device), xva.to(device)
    model = MWLocalFrameAR(**model_kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    aug_gen = torch.Generator(device=device).manual_seed(seed + 1)
    eval_gen = torch.Generator(device=device).manual_seed(seed + 2)
    out = out if os.path.isabs(out) else os.path.join(ART, out)
    stem = out[:-3] if out.endswith(".pt") else out
    paths = {"nll": stem + "_best_nll.pt", "struct": stem + "_best_struct.pt",
             "last": stem + "_last.pt"}
    best_nll = best_struct = float("inf")
    N = xtr.shape[1]
    for step in range(steps):
        idx = torch.randint(len(xtr), (batch,), device=device)
        xb = _augment_batch(xtr[idx], L, aug_gen)
        loss = -model.log_prob(xb, L).mean() / N
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite v5 loss at step {step}")
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % val_every == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vals = [-model.log_prob(xva[i:i + 64], L).sum()
                        for i in range(0, len(xva), 64)]
                val = float(torch.stack(vals).sum() / (len(xva) * N))
                # Resetting makes checkpoint comparisons use common random numbers.
                eval_gen.manual_seed(seed + 2)
                struct = model.teacher_forced_structure(xva[:64], L, gen=eval_gen)
            model.train()
            ck = _checkpoint(model, step, val, struct, L)
            torch.save(ck, paths["last"])
            if val < best_nll:
                best_nll = val; torch.save(ck, paths["nll"])
            score = float(struct["composite"])
            if score < best_struct:
                best_struct = score; torch.save(ck, paths["struct"])
            print(f"v5 step {step}: train {float(loss):.4f} val {val:.4f} "
                  f"TF peak_err {float(struct['peak_err']):.4f} core {float(struct['core_mass']):.4f} "
                  f"composite {score:.4f}", flush=True)
    return {**paths, "best_nll": best_nll, "best_struct": best_struct}


if __name__ == "__main__":
    train()
