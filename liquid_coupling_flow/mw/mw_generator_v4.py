"""Global-prefix mW AR transformer (v4).

This is the deliberately stronger control against ``mw_generator.MWGenerator``:
global causal prefix attention, previous-displacement content, per-layer geometric
attention bias, an explicit Gilbert-curve rail, and a 64-component full-covariance
mixture head.  The mixture is tanh-bounded to the minimum-image cube, making the
sample -> wrapped position map one-to-one (up to measure-zero box boundaries).
"""
from __future__ import annotations

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from liquid_coupling_flow.mw.mw_energy import RHO_STAR
from liquid_coupling_flow.mw.mw_generator import (
    ART, DEV, _augment_batch, canonical_order, load_training_bank, mw_scaffold, wrap_pm,
)


class FullCovTanhMDN(nn.Module):
    """Mixture of full-covariance Gaussians followed by u=bound*tanh(z)."""

    def __init__(self, d_model=256, n_mix=64, dim=3, min_scale=1e-3):
        super().__init__()
        self.n_mix, self.dim, self.min_scale = n_mix, dim, min_scale
        self.n_tril = dim * (dim + 1) // 2
        self.out = nn.Linear(d_model, n_mix * (1 + dim + self.n_tril))
        self.register_buffer("tri_i", torch.tril_indices(dim, dim)[0])
        self.register_buffer("tri_j", torch.tril_indices(dim, dim)[1])

    def _params(self, h):
        p = self.out(h).reshape(*h.shape[:-1], self.n_mix, 1 + self.dim + self.n_tril)
        logits = p[..., 0]
        mean = p[..., 1:1 + self.dim]
        raw = p[..., 1 + self.dim:]
        L = raw.new_zeros(*raw.shape[:-1], self.dim, self.dim)
        L[..., self.tri_i, self.tri_j] = raw
        diag = torch.arange(self.dim, device=h.device)
        L[..., diag, diag] = F.softplus(L[..., diag, diag]) + self.min_scale
        return logits, mean, L

    def log_prob(self, h, u, bound):
        logits, mean, L = self._params(h)
        y = (u / bound).clamp(-1 + 1e-6, 1 - 1e-6)
        z = torch.atanh(y)
        dz = z.unsqueeze(-2) - mean
        sol = torch.linalg.solve_triangular(L, dz.unsqueeze(-1), upper=False).squeeze(-1)
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        lp_z = -0.5 * (sol.square().sum(-1) + self.dim * math.log(2 * math.pi)) \
               - torch.log(diag).sum(-1)
        log_jac = (-math.log(bound) - torch.log1p(-y.square())).sum(-1)
        return torch.logsumexp(F.log_softmax(logits, -1) + lp_z, -1) + log_jac

    def sample(self, h, bound, gen=None):
        logits, mean, L = self._params(h)
        shape = logits.shape[:-1]
        flat_logits = logits.reshape(-1, self.n_mix)
        comp = torch.multinomial(F.softmax(flat_logits, -1), 1, generator=gen).squeeze(-1)
        ar = torch.arange(comp.numel(), device=h.device)
        mf = mean.reshape(-1, self.n_mix, self.dim)[ar, comp]
        Lf = L.reshape(-1, self.n_mix, self.dim, self.dim)[ar, comp]
        eps = torch.randn(comp.numel(), self.dim, device=h.device, generator=gen)
        z = mf + torch.einsum("bij,bj->bi", Lf, eps)
        u = bound * torch.tanh(z)
        lp = self.log_prob(h.reshape(-1, h.shape[-1]), u, bound).reshape(shape)
        return u.reshape(*shape, self.dim), lp


class CurveRail(nn.Module):
    """Per-query cross-attention onto the next K fixed Gilbert waypoints."""

    def __init__(self, d_model, n_heads, k=8):
        super().__init__()
        self.k = k
        self.vec = nn.Sequential(nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.order = nn.Embedding(k, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, anchors, L, cell_width):
        B, T, D = x.shape
        j = torch.arange(T, device=x.device)
        off = torch.arange(1, self.k + 1, device=x.device)
        idx = (j[:, None] + off[None]).clamp_max(anchors.shape[0] - 1)
        rel = wrap_pm(anchors[idx] - anchors[j, None], L) / cell_width
        valid = (j[:, None] + off[None]) < anchors.shape[0]
        feat = torch.cat([rel, rel.norm(dim=-1, keepdim=True)], -1)
        rail = self.vec(feat) + self.order(off - 1)[None]
        rail = rail[None].expand(B, -1, -1, -1).reshape(B * T, self.k, D)
        q = x.reshape(B * T, 1, D)
        pad = (~valid)[None].expand(B, -1, -1).reshape(B * T, self.k)
        # MHA returns NaN if every key is padded at the last query. Keep one zero-valued key.
        all_pad = pad.all(-1)
        if all_pad.any():
            pad = pad.clone(); pad[all_pad, 0] = False
            rail = rail.clone(); rail[all_pad, 0] = 0
        y, _ = self.attn(q, rail, rail, key_padding_mask=pad, need_weights=False)
        return x + self.proj(y.reshape(B, T, D))


class GeometricBias(nn.Module):
    def __init__(self, n_heads, n_rbf=32, rmax=4.0):
        super().__init__()
        self.register_buffer("centers", torch.linspace(0, rmax, n_rbf))
        self.gamma = 1.0 / (rmax / (n_rbf - 1)) ** 2
        self.net = nn.Sequential(nn.Linear(n_rbf + 4, 64), nn.GELU(), nn.Linear(64, n_heads))

    def forward(self, anchors_q, key_pos, L):
        # query anchor -> previously placed particle carried by each causal token
        d = wrap_pm(key_pos[:, None] - anchors_q[None, :, None], L)
        r = d.norm(dim=-1, keepdim=True)
        rbf = torch.exp(-self.gamma * (r - self.centers) ** 2)
        feat = torch.cat([rbf, d / r.clamp_min(1e-6), r], -1)
        return self.net(feat).permute(0, 3, 1, 2).contiguous()


class CausalGeoBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.dh = n_heads, d_model // n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x, geo_bias):
        B, T, D = x.shape
        qkv = self.qkv(self.ln1(x)).reshape(B, T, 3, self.n_heads, self.dh)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh) + geo_bias
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        score = score.masked_fill(~causal[None, None], float("-inf"))
        a = F.softmax(score, -1)
        y = torch.matmul(a, v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        return x + self.ff(self.ln2(x))


class MWGlobalAR(nn.Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=8, n_mix=64, rail_k=8):
        super().__init__()
        self.d_model, self.n_layers, self.n_heads = d_model, n_layers, n_heads
        self.n_mix, self.rail_k = n_mix, rail_k
        self.prev_proj = nn.Linear(3, d_model)
        self.phase_proj = nn.Sequential(nn.Linear(8, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.rail = CurveRail(d_model, n_heads, rail_k)
        self.geo = GeometricBias(n_heads)
        self.blocks = nn.ModuleList([CausalGeoBlock(d_model, n_heads) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)
        self.head = FullCovTanhMDN(d_model, n_mix=n_mix)

    @staticmethod
    def _phase(T, device):
        z = (torch.arange(T, device=device) + 0.5) / T
        fs = []
        for f in (1.0, 2.0, 4.0, 8.0):
            fs += [torch.sin(2 * math.pi * f * z), torch.cos(2 * math.pi * f * z)]
        return torch.stack(fs, -1)

    def _hidden(self, u, x, anchors, L):
        """u/x contain targets/positions for the currently known teacher-forced prefix."""
        B, T, _ = u.shape
        prev = torch.cat([torch.zeros(B, 1, 3, device=u.device, dtype=u.dtype), u[:, :-1]], 1)
        h = self.prev_proj(prev) + self.phase_proj(self._phase(anchors.shape[0], u.device)[:T])[None]
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
        s = (L / R) / 2.0
        bound = float(R)
        # unique minimum-image coordinate in [-R,R]^3 after division by half-cell width
        u = wrap_pm(x - t[None], L) / s
        h = self._hidden(u, x, t, L)
        return self.head.log_prob(h, u, bound).sum(-1) - 3 * N * math.log(s)

    @torch.no_grad()
    def sample(self, B, N, L, gen=None, return_logq=True):
        device = next(self.parameters()).device
        t, _, R = mw_scaffold(N, L, device)
        s, bound = (L / R) / 2.0, float(R)
        x = torch.zeros(B, N, 3, device=device)
        u = torch.zeros_like(x)
        logq = torch.zeros(B, device=device)
        for j in range(N):
            h = self._hidden(u[:, :j + 1], x[:, :j + 1], t, L)[:, -1]
            uj, lp = self.head.sample(h, bound, gen)
            u[:, j] = uj
            x[:, j] = torch.remainder(t[j] + s * uj, L)
            logq += lp - 3 * math.log(s)
        return (x, logq) if return_logq else x

    def suffix_log_prob(self, x, m, L):
        """Exact log q0 of slots N-m..N-1 given slots < N-m. x is PREORDERED [B,N,3]
        (slot j == AR step j, anchor t_j) -- the same convention as
        log_prob(..., preordered=True), of which this is the per-slot tail sum."""
        B, N, _ = x.shape
        x = torch.remainder(x, L)
        t, rank, R = mw_scaffold(N, L, x.device)
        s, bound = (L / R) / 2.0, float(R)
        u = wrap_pm(x - t[None], L) / s
        h = self._hidden(u, x, t, L)
        return self.head.log_prob(h, u, bound)[:, N - m:].sum(-1) - 3 * m * math.log(s)

    @torch.no_grad()
    def sample_suffix(self, x, m, L, gen=None):
        """Redraw slots N-m..N-1 from the exact AR conditional given slots < N-m.
        Mirrors sample()'s j-loop with the prefix pre-filled. Returns (x_new, logq_fwd);
        logq_fwd == suffix_log_prob(x_new, m, L) by construction (shared head/_hidden)."""
        device = next(self.parameters()).device
        B, N, _ = x.shape
        t, _, R = mw_scaffold(N, L, device)
        s, bound = (L / R) / 2.0, float(R)
        x = torch.remainder(x.to(device).clone(), L)
        u = wrap_pm(x - t[None], L) / s
        logq = torch.zeros(B, device=device)
        for j in range(N - m, N):
            h = self._hidden(u[:, :j + 1], x[:, :j + 1], t, L)[:, -1]
            uj, lp = self.head.sample(h, bound, gen)
            u[:, j] = uj
            x[:, j] = torch.remainder(t[j] + s * uj, L)
            logq = logq + lp - 3 * math.log(s)
        return x, logq


def _val_nll(model, x, L, chunk=64):
    was_training = model.training; model.eval(); total = 0.0
    with torch.no_grad():
        for i in range(0, len(x), chunk):
            total += float((-model.log_prob(x[i:i + chunk], L)).sum())
    model.train(was_training)
    return total / (len(x) * x.shape[1])


def train(steps=30000, batch=32, lr=2e-4, val_every=500, seed=23,
          out="mw_gen_N64_v4.pt", primary_thin=2, val_frac=0.1,
          extension="mw_ref_N64_ext.pt", device=None, warm=None,
          d_model=256, n_layers=4, n_heads=8, n_mix=64, rail_k=8):
    torch.manual_seed(seed); device = device or DEV
    ext = extension if os.path.exists(extension) else os.path.join(ART, extension)
    xtr, xva, L = load_training_bank(thin_events=primary_thin, val_frac=val_frac,
                                     extra_banks=[(ext, 32, 1)])
    xtr, xva = xtr.to(device), xva.to(device); N = xtr.shape[1]
    out_path = out if os.path.isabs(out) else os.path.join(ART, out)
    last_path = out_path[:-3] + "_last.pt"
    if warm is not None:
        warm_path = warm if os.path.exists(warm) else os.path.join(ART, warm)
        wc = torch.load(warm_path, map_location=device, weights_only=False)
        kw = {k: wc[k] for k in ("d_model", "n_layers", "n_heads", "n_mix", "rail_k")}
        model = MWGlobalAR(**kw).to(device); model.load_state_dict(wc["state_dict"])
        best, step_offset = float(wc["val_nll"]), int(wc["step"]) + 1
        print(f"v4 warm-continue from {warm_path}: step={wc['step']} best_val={best:.4f}; fresh Adam", flush=True)
    else:
        model = MWGlobalAR(d_model, n_layers, n_heads, n_mix, rail_k).to(device)
        best, step_offset = float("inf"), 0
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    aug_gen = torch.Generator(device=device).manual_seed(seed + 1)
    kw = dict(d_model=model.d_model, n_layers=model.n_layers, n_heads=model.n_heads,
              n_mix=model.n_mix, rail_k=model.rail_k)
    print(f"v4 train: N={N} train={len(xtr)} val={len(xva)} model={kw} device={device}", flush=True)
    for local_step in range(steps):
        step = step_offset + local_step
        idx = torch.randint(len(xtr), (batch,), device=device)
        xb = _augment_batch(xtr[idx], L, aug_gen)
        loss = -model.log_prob(xb, L).mean() / N
        if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss at {step}")
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if local_step % val_every == 0 or local_step == steps - 1:
            val = _val_nll(model, xva, L)
            print(f"step {step}: train {float(loss):.4f} val {val:.4f}", flush=True)
            ck = {"state_dict": model.state_dict(), **kw, "val_nll": val, "step": step,
                  "train_N": N, "primary_thin": primary_thin, "val_frac": val_frac,
                  "extension": ext, "architecture": "global_causal_geo_rail_fullcov_tanh_mdn"}
            torch.save(ck, last_path)
            if val < best: best = val; torch.save(ck, out_path)
    return {"out": out_path, "last": last_path, "best_val": best}


if __name__ == "__main__":
    train()
