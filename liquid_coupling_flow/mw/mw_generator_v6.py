"""mW generator v6 -- v4 global-AR body + v5 circular-spline torus head.

This is the synthesis of the two prior generators (design:
docs/superpowers/specs/2026-07-10-mw-v6-global-spline-design.md):

  - Body: v4's ``MWGlobalAR`` -- global causal AR over ALL placed particles,
    ``CurveRail`` + per-layer ``GeometricBias`` + ``CausalGeoBlock`` x n_layers.
    This is the component that GENERALIZES (v4 val 0.51 vs v5's localized-body
    3.46), reused by import and NOT re-tuned.
  - Head: v5's ``CircularSpline3Head`` -- AR-chained circular RQ-splines on the
    3-torus, uniform base, periodic seam. Sharper than v4's full-cov tanh MDN
    (free-code overfit -11.3 vs +1.65 NLL/particle) AND toroidally exact (no
    ``tanh`` open-box chart distortion at cell boundaries).
  - Chart: TRANSFER-SAFE. u = wrap_pm(x_j - t_j, L)/s with s = L/(2*bound),
    bound = 4.0 FIXED (v5 convention). This normalizes |u| < bound for ANY N --
    unlike v4's s = (L/R)/2 which gives |u| < R (R = 4 at N=64 but 6 at N=216)
    and would break the fixed-bound spline at transfer. CRITICAL for Phase 3.
  - mW three-body features (``use_geo_feat``, enhancement): v4's body has no
    angular encoding, but mW's Stillinger-Weber three-body term is tetrahedral.
    A per-anchor pooled summary over the k=12 nearest PLACED-prefix neighbors
    (tetra pair term + excluded-volume 1/r^2 + occupancy) is projected through a
    ZERO-INITIALIZED linear and ADDED to the body input, so v6 starts == v4-body.

Exactness (co-first-class, since log q0 enters SMC importance weights): the
origin (anchor t_j) and chart are deterministic torus translations, |Jacobian|
= 1; the spline head is a normalized density on the 3-torus with no wrapped-tail
exception; and the three-body features are computed inside ``_hidden`` causally
from the prefix positions, so ``sample()``'s per-step ``[:, -1]`` value is the
SAME code path as ``log_prob``'s vectorized position j. sample <-> log_prob
(preordered) agree to <= 1e-4 per particle on every draw.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from liquid_coupling_flow.mw.mw_energy import RHO_STAR
from liquid_coupling_flow.mw.mw_generator import (
    ART,
    DEV,
    EV_RMIN,
    R_SHELL1,
    _augment_batch,
    canonical_order,
    load_training_bank,
    mw_scaffold,
    wrap_pm,
)
from liquid_coupling_flow.mw.mw_generator_v4 import (
    CausalGeoBlock,
    CurveRail,
    GeometricBias,
)
from liquid_coupling_flow.mw.mw_generator_v5 import CircularSpline3Head

N_GEO_FEAT = 3  # [tetra pair mean, inv-r^2 mean, occupancy]


class MWGlobalSpline(nn.Module):
    """v4's global-AR body with v5's circular-spline torus head (v6).

    Structurally identical to ``mw_generator_v4.MWGlobalAR`` except: (1) the head
    is a ``CircularSpline3Head`` (toroidally exact) instead of the tanh MDN, (2)
    the chart uses the fixed-bound scale s = L/(2*bound) so |u| < bound for any N,
    and (3) an optional zero-initialized three-body geometric feature is added to
    the body input.
    """

    def __init__(self, d_model=256, n_layers=4, n_heads=8, rail_k=8,
                 num_bins=32, bound=4.0, knn=12, use_geo_feat=True):
        super().__init__()
        self.d_model, self.n_layers, self.n_heads = d_model, n_layers, n_heads
        self.rail_k, self.num_bins = rail_k, num_bins
        self.bound = float(bound)
        self.knn = int(knn)
        self.use_geo_feat = bool(use_geo_feat)
        self.prev_proj = nn.Linear(3, d_model)
        self.phase_proj = nn.Sequential(nn.Linear(8, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.rail = CurveRail(d_model, n_heads, rail_k)
        self.geo = GeometricBias(n_heads)
        self.blocks = nn.ModuleList([CausalGeoBlock(d_model, n_heads) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.head = CircularSpline3Head(d_model, num_bins=num_bins, bound=self.bound)
        # Created LAST so its RNG draw does not shift the shared-body init stream:
        # a fresh MWGlobalSpline(use_geo_feat=False) has bit-identical body weights.
        # Zero-initialized => v6 starts IDENTICAL to the v4 body (feature is a pure
        # additive enhancement; test_geo_feat_zero_init_matches_no_feat guards this).
        self.geo_feat_proj = nn.Linear(N_GEO_FEAT, d_model)
        nn.init.zeros_(self.geo_feat_proj.weight)
        nn.init.zeros_(self.geo_feat_proj.bias)

    @staticmethod
    def _phase(T, device):
        z = (torch.arange(T, device=device) + 0.5) / T
        fs = []
        for f in (1.0, 2.0, 4.0, 8.0):
            fs += [torch.sin(2 * math.pi * f * z), torch.cos(2 * math.pi * f * z)]
        return torch.stack(fs, -1)

    def _geo_feat(self, x, anchors, L):
        """Per-anchor three-body summary over the k nearest PLACED-prefix neighbors.

        Vectorized-causal: for query anchor j (curve rank), candidates are the
        already-placed particles i < j. Returns ``[B, T, N_GEO_FEAT]`` with, per j:
          - mean over neighbor pairs (a,b) of (cos theta_ab + 1/3), theta_ab the
            angle at t_j (tetrahedral, mW Stillinger-Weber);
          - mean over neighbors of 1/clamp_min(r, EV_RMIN)^2 (excluded volume);
          - n_valid / knn occupancy.
        Computing it inside ``_hidden`` from prefix positions makes ``sample``'s
        per-step ``[:, -1]`` the SAME code path as ``log_prob`` position j (exact).
        """
        B, T, _ = x.shape
        K = min(self.knn, T)
        jj = torch.arange(T, device=x.device)
        causal = jj[None, None, :] < jj[None, :, None]                     # [1,T(query j),T(cand i)]: i<j
        d = wrap_pm(x[:, None, :, :] - anchors[:T][None, :, None, :], L)   # [B,T,T,3] cand rel to anchor j
        dist2 = d.square().sum(-1)                                          # [B,T,T]
        idx = dist2.masked_fill(~causal, 1e9).topk(K, dim=2, largest=False).indices   # [B,T,K] nearest-first
        rel = torch.gather(d, 2, idx[..., None].expand(-1, -1, -1, 3))      # [B,T,K,3]
        valid = torch.gather(causal.expand(B, T, T), 2, idx)               # [B,T,K] bool
        rel = rel * valid[..., None].to(rel.dtype)
        vf = valid.to(x.dtype)
        r = rel.norm(dim=-1)                                                # [B,T,K]
        n_valid = valid.sum(-1)                                             # [B,T]
        nv = n_valid.clamp_min(1).to(x.dtype)
        inv_r2 = 1.0 / r.clamp_min(EV_RMIN).square()                        # [B,T,K] == min(1/r^2, 4)
        inv_r2_mean = (inv_r2 * vf).sum(-1) / nv                            # [B,T]
        occ = n_valid.to(x.dtype) / self.knn                               # [B,T]
        unit = rel / r.clamp_min(1e-6)[..., None]                           # [B,T,K,3]
        cos = torch.einsum("btkd,btld->btkl", unit, unit)                  # [B,T,K,K]
        tetra = cos + 1.0 / 3.0
        eye = torch.eye(K, device=x.device, dtype=torch.bool)
        pair_valid = (valid[..., :, None] & valid[..., None, :] & (~eye)).to(x.dtype)  # [B,T,K,K]
        npairs = pair_valid.sum((-1, -2)).clamp_min(1.0)
        tetra_mean = (tetra * pair_valid).sum((-1, -2)) / npairs           # [B,T]
        return torch.stack([tetra_mean, inv_r2_mean, occ], dim=-1)         # [B,T,N_GEO_FEAT]

    def _hidden(self, u, x, anchors, L):
        """u/x contain targets/positions for the currently known teacher-forced prefix."""
        B, T, _ = u.shape
        prev = torch.cat([torch.zeros(B, 1, 3, device=u.device, dtype=u.dtype), u[:, :-1]], 1)
        h = self.prev_proj(prev) + self.phase_proj(self._phase(anchors.shape[0], u.device)[:T])[None]
        if self.use_geo_feat:
            h = h + self.geo_feat_proj(self._geo_feat(x, anchors, L))
        h[:, :1] = h[:, :1] + self.bos
        h = self.rail(h, anchors, L, L / round(anchors.shape[0] ** (1 / 3)))
        key_pos = torch.cat([anchors[:1][None].expand(B, -1, -1), x[:, :-1]], 1)
        gb = self.geo(anchors[:T], key_pos, L)
        for block in self.blocks:
            h = block(h, gb)
        return self.final_ln(h)

    def log_prob(self, x, L, preordered=False):
        B, N, _ = x.shape
        x = torch.remainder(x, L)
        t, rank, R = mw_scaffold(N, L, x.device)
        if not preordered:
            perm = canonical_order(x, L, R, rank)
            x = torch.gather(x, 1, perm[..., None].expand(-1, -1, 3))
        s = L / (2 * self.bound)
        # Global minimum-image chart, fixed bound: |u| < bound for ANY N (transfer-safe).
        u = wrap_pm(x - t[None], L) / s
        h = self._hidden(u, x, t, L)
        return self.head.log_prob(h, u).sum(1) - 3 * N * math.log(s)

    @torch.no_grad()
    def sample(self, B, N, L, gen=None, return_logq=True):
        device = next(self.parameters()).device
        t, _, R = mw_scaffold(N, L, device)
        s = L / (2 * self.bound)
        x = torch.zeros(B, N, 3, device=device)
        u = torch.zeros_like(x)
        logq = torch.zeros(B, device=device)
        for j in range(N):
            h = self._hidden(u[:, :j + 1], x[:, :j + 1], t, L)[:, -1]
            uj, lp = self.head.sample(h, gen=gen)
            u[:, j] = uj
            x[:, j] = torch.remainder(t[j] + s * uj, L)
            logq += lp - 3 * math.log(s)
        return (x, logq) if return_logq else x

    @torch.no_grad()
    def teacher_forced_structure(self, x, L, nbins=120, gen=None):
        """Pooled-predecessor TF g(r), peak error, and excluded-core mass."""
        xo = torch.remainder(x, L)
        t, rank, R = mw_scaffold(xo.shape[1], L, xo.device)
        perm = canonical_order(xo, L, R, rank)
        xo = torch.gather(xo, 1, perm[..., None].expand(-1, -1, 3))
        s = L / (2 * self.bound)
        u = wrap_pm(xo - t[None], L) / s
        h = self._hidden(u, xo, t, L)
        u_s, _ = self.head.sample(h, gen=gen)
        placed = torch.remainder(t[None] + s * u_s, L)
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
        "train_N": round(RHO_STAR * L ** 3), "architecture": "global_spline_v6",
        **{k: getattr(model, k) for k in ("d_model", "n_layers", "n_heads", "rail_k",
                                          "num_bins", "bound", "knn", "use_geo_feat")},
    }


def train(steps=30_000, batch=32, lr=3e-4, val_every=500, seed=63,
          out="mw_gen_N64_v6.pt", primary_thin=2, val_frac=0.1,
          extension="mw_ref_N64_ext.pt", device=None, art_path=None,
          extra_banks=None, **model_kw):
    """Train v6 and save independent best-NLL and best-structure checkpoints.

    Structure ported verbatim from v5's ``train``: multi-bank loader (thin-2
    primary + extension), on-the-fly augmentation, AdamW lr 3e-4 wd 1e-4,
    clip_grad_norm 5.0, best_nll + best_struct + last checkpoints, and a TF
    pooled-predecessor g(r) composite (peak_err + core_mass) every val cycle.
    """
    torch.manual_seed(seed)
    device = device or DEV
    if extra_banks is None:
        ext = extension if os.path.exists(extension) else os.path.join(ART, extension)
        extra_banks = [(ext, 32, 1)]
    xtr, xva, L = load_training_bank(art_path=art_path, thin_events=primary_thin,
                                     val_frac=val_frac, extra_banks=extra_banks)
    xtr, xva = xtr.to(device), xva.to(device)
    model = MWGlobalSpline(**model_kw).to(device)
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
            raise FloatingPointError(f"nonfinite v6 loss at step {step}")
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
            print(f"v6 step {step}: train {float(loss):.4f} val {val:.4f} "
                  f"TF peak_err {float(struct['peak_err']):.4f} core {float(struct['core_mass']):.4f} "
                  f"composite {score:.4f}", flush=True)
    return {**paths, "best_nll": best_nll, "best_struct": best_struct}


if __name__ == "__main__":
    train()
