# liquid_coupling_flow/ka_flowhead.py
"""Exact AR rational-quadratic spline-flow placement head for the local-frame generator (Phase 1 / Feature B).
Replaces the binned (a,b) categorical with a continuous flow sharp enough to represent the hard-core contact
peak the categorical smears. Gaussian base on R + RQSplineElementwise (identity linear tails); AR factorization
p(a|h)*p(b|a,h). Identity-initialized for stable training. Returns the NORMALIZED-offset log-density; the caller
adds the arc_scale Jacobian-to-physical. Quetzal structure (transformer context -> small continuous head),
exact instead of diffusion so the SMC corrector's likelihood stays exact."""
from __future__ import annotations
import math, torch, torch.nn as nn
from liquid_coupling_flow.transforms_spline import RQSplineElementwise, DEFAULT_MIN_DERIVATIVE

_LOG2PI = math.log(2 * math.pi)


def _base_logp(z):                                          # standard-normal log-density, summed over last dim
    return (-0.5 * z ** 2 - 0.5 * _LOG2PI)


class SplineFlowHead(nn.Module):
    def __init__(self, d_model, num_bins=8, tail_bound=4.0):
        super().__init__()
        self.spline = RQSplineElementwise(num_bins=num_bins, tail_bound=tail_bound)
        self.num_bins = num_bins
        P = self.spline.params_per_dim                      # 3K - 1
        self.head_a = nn.Linear(d_model, P)
        self.head_b = nn.Linear(d_model + 1, P)             # condition b on the continuous a
        self._identity_init()

    def _identity_init(self):
        # zero weights; widths/heights bias 0 (-> uniform bins); interior-derivative bias = const giving
        # softplus(const)+min_deriv == 1 -> the RQS is exactly the identity at init (flow == Gaussian base).
        K = self.num_bins
        const = float(torch.log(torch.expm1(torch.tensor(1.0 - DEFAULT_MIN_DERIVATIVE))))
        for head in (self.head_a, self.head_b):
            nn.init.zeros_(head.weight)
            with torch.no_grad():
                head.bias.zero_()
                head.bias[2 * K:] = const                   # interior derivatives (P = 3K-1: [2K:] is the K-1 derivs)

    def log_prob(self, h, ab):
        a = ab[..., 0:1]; b = ab[..., 1:2]
        za, lda = self.spline.inverse(a, self.head_a(h))                 # a -> base
        lpa = _base_logp(za) + lda
        zb, ldb = self.spline.inverse(b, self.head_b(torch.cat([h, a], -1)))
        lpb = _base_logp(zb) + ldb
        return (lpa + lpb).squeeze(-1)

    def sample(self, h, gen=None):
        za = torch.randn(h.shape[:-1] + (1,), device=h.device, generator=gen)
        a, lda = self.spline.forward(za, self.head_a(h))                 # base -> a
        lpa = _base_logp(za) - lda
        zb = torch.randn(h.shape[:-1] + (1,), device=h.device, generator=gen)
        b, ldb = self.spline.forward(zb, self.head_b(torch.cat([h, a], -1)))
        lpb = _base_logp(zb) - ldb
        return torch.cat([a, b], -1), (lpa + lpb).squeeze(-1)


# ---------------------------------------------------------------------------
# KAFlowHeadModel — local-frame AR generator with spline-flow placement head
# ---------------------------------------------------------------------------
import torch.nn.functional as F
from liquid_coupling_flow.ka_localframe import KALocalFrameModel, _wrap_pm


class KAFlowHeadModel(KALocalFrameModel):
    """Local-frame AR generator with the categorical (a,b) head replaced by the exact spline flow.
    Species head + frame + KNN context + curve ordering are inherited unchanged; exact likelihood preserved
    (continuous flow density replaces P(bin)/bin_area; the arc_scale Jacobian is the same `jac` term).

    frame_mode="scaffold" (default): behaviour byte-identical to Task 2 (no rotation).
    frame_mode="inertial": rotate the (a,b) offset into the PCA principal-axis frame of the placed
    neighbours before scoring/sampling. |det R|=1 => exactness preserved (no Jacobian term)."""

    def __init__(self, *args, num_bins=8, tail_bound=4.0, frame_mode="scaffold", **kw):
        super().__init__(*args, **kw)
        self.flow = SplineFlowHead(self.d_model, num_bins=num_bins, tail_bound=tail_bound)
        self.frame_mode = frame_mode

    # ------------------------------------------------------------------
    # Inertial-frame helpers
    # ------------------------------------------------------------------

    def _inertial_R(self, nbr_rel):
        """Per-particle 2x2 rotation from the principal axis of placed-neighbour offsets.
        nbr_rel [..., k, 2].  Returns R [..., 2, 2] with R^T R = I and det R = +1."""
        x, y = nbr_rel[..., 0], nbr_rel[..., 1]          # [..., k]
        cxx = (x * x).mean(-1)
        cyy = (y * y).mean(-1)
        cxy = (x * y).mean(-1)
        theta = 0.5 * torch.atan2(2 * cxy, cxx - cyy)   # principal axis angle
        c, s = torch.cos(theta), torch.sin(theta)
        # R = [[c, -s], [s, c]]  (rotation by -theta, maps principal axis -> x-axis)
        row0 = torch.stack([c, -s], -1)                  # [..., 2]
        row1 = torch.stack([s,  c], -1)                  # [..., 2]
        return torch.stack([row0, row1], -2)              # [..., 2, 2]

    def _inertial_R_logprob(self, xo, origin, L, N):
        """Return the [B,N,2,2] inertial rotation stack as computed by log_prob (with valid-mask + j<2
        identity). Used only by tests — scaffold path is unaffected."""
        B = xo.shape[0]
        sc = self.geo._scaffold(N, xo.device)
        d_ksc = _wrap_pm(xo[:, None, :, :] - sc[None, :, None, :], L)
        dist2 = (d_ksc ** 2).sum(-1)
        jj = torch.arange(N, device=xo.device)
        causal = jj[None, None, :] < jj[None, :, None]
        idx = dist2.masked_fill(~causal, 1e9).topk(self.knn, dim=2, largest=False).indices
        nbr_pos = torch.gather(xo[:, None].expand(B, N, N, 2), 2, idx[..., None].expand(-1, -1, -1, 2))
        nbr_rel = _wrap_pm(nbr_pos - origin[:, :, None, :], L)
        valid = torch.gather(causal.expand(B, N, N), 2, idx)
        nbr_rel = nbr_rel * valid[..., None]
        R = self._inertial_R(nbr_rel)
        eye = torch.eye(2, device=R.device).expand_as(R)
        R = torch.where((jj < 2)[None, :, None, None], eye, R)
        return R

    def _inertial_R_sample_step(self, pos, sc, L, N):
        """Return the [B,N,2,2] inertial rotation stack as computed step-by-step in sample/_step.
        Used only by tests — scaffold path is unaffected."""
        B = pos.shape[0]; device = pos.device
        R_stack = []
        eye2 = torch.eye(2, device=device)
        for j in range(N):
            if j < 2:
                R_stack.append(eye2.expand(B, 2, 2))
            else:
                k = min(self.knn, j)
                d_sc = _wrap_pm(pos[:, :j] - sc[j][None, None], L)
                dist2_j = (d_sc ** 2).sum(-1)
                w = torch.exp(-dist2_j / (2 * 1.3 ** 2))
                wsum = w.sum(1, keepdim=True).clamp_min(1e-12)
                origin_j = torch.remainder(sc[j][None] + (w[..., None] * d_sc).sum(1) / wsum, L)
                idx_j = dist2_j.topk(k, dim=1, largest=False).indices
                nbr_pos_j = torch.gather(pos[:, :j], 1, idx_j[..., None].expand(-1, -1, 2))
                nbr_rel_j = _wrap_pm(nbr_pos_j - origin_j[:, None, :], L)
                R_stack.append(self._inertial_R(nbr_rel_j))
        return torch.stack(R_stack, dim=1)  # [B,N,2,2]

    def _species_logits(self, h, rem):
        lg = self.head_species(h)
        return lg if rem is None else lg.masked_fill(rem <= 0, float("-inf"))

    def log_prob(self, x, s, canonical=None, preordered=False):
        B, N = x.shape[0], x.shape[1]
        s = s.long()
        s = s.expand(B, N).clone() if s.dim() == 1 else s
        L = self._Lof(N)
        if preordered:
            xo, so = x, s
        else:
            order = self.geo._curve_order(x, N)
            xo = torch.gather(x, 1, order[..., None].expand(-1, -1, 2))
            so = torch.gather(s, 1, order)
        sc = self.geo._scaffold(N, x.device)
        context, origin = self._local(xo, so, sc, L, N)
        ab = _wrap_pm(xo - origin, L) / self._arc_scale(N)
        if self.frame_mode == "inertial":
            # Recompute the causal-KNN nbr_rel used inside _local (same neighbour set) to build R.
            # This mirrors the logic in _local: causal mask k<j, topk by dist2 to scaffold, rel to origin.
            d_ksc = _wrap_pm(xo[:, None, :, :] - sc[None, :, None, :], L)     # [B,N,N,2] pos_k - sc_j
            dist2 = (d_ksc ** 2).sum(-1)                                        # [B,N,N]
            jj = torch.arange(N, device=xo.device)
            causal = jj[None, None, :] < jj[None, :, None]                     # k < j  [1,N,N]
            idx = dist2.masked_fill(~causal, 1e9).topk(self.knn, dim=2, largest=False).indices  # [B,N,KNN]
            nbr_pos = torch.gather(xo[:, None].expand(B, N, N, 2), 2, idx[..., None].expand(-1, -1, -1, 2))
            nbr_rel_lp = _wrap_pm(nbr_pos - origin[:, :, None, :], L)          # [B,N,KNN,2]
            # Part 1a: apply valid mask so spurious future-particle slots (filled with 1e9 dist but still
            # returned by topk) contribute zero to the PCA covariance — matching sample's _step which only
            # gathers min(knn,j) real causal neighbours and never returns future particles.
            valid = torch.gather(causal.expand(B, N, N), 2, idx)               # [B,N,KNN] bool
            nbr_rel_lp = nbr_rel_lp * valid[..., None]                         # zero spurious offsets
            R = self._inertial_R(nbr_rel_lp)                                    # [B,N,2,2]
            # Part 1b: force R = identity for j<2 to match sample's explicit identity guard (j<2 uses
            # no neighbours, so _inertial_R returns an arbitrary rotation; sample uses R=I there).
            eye = torch.eye(2, device=R.device).expand_as(R)
            R = torch.where((jj < 2)[None, :, None, None], eye, R)
            ab = torch.einsum("bnij,bnj->bni", R, ab)                          # rotate DATA offset into frame
        lp_ab = self.flow.log_prob(context, ab)                          # [B,N] continuous flow log-density
        s_logits = self.head_species(context)
        if self.canonical if canonical is None else canonical:
            oh = F.one_hot(so, self.n_species).to(s_logits.dtype)
            rem = oh.sum(1, keepdim=True) - (oh.cumsum(1) - oh)
            s_logits = s_logits.masked_fill(rem <= 0, float("-inf"))
        lp_s = F.log_softmax(s_logits, -1).gather(-1, so[..., None]).squeeze(-1)
        jac = self.d * N * math.log(self._arc_scale(N))                  # NO bin_w vol term (flow is continuous)
        return (lp_ab + lp_s).sum(1) - jac

    @torch.no_grad()
    def sample(self, B, N, n_B=None, device=None, return_logq=False):
        L = self._Lof(N); sc = self.geo._scaffold(N, device); arc = self._arc_scale(N)
        pos = torch.zeros(B, N, 2, device=device); sp = torch.zeros(B, N, dtype=torch.long, device=device)
        rem = None
        if n_B is not None:
            rem = torch.zeros(B, self.n_species, device=device)
            rem[:, 0] = N - n_B; rem[:, 1] = n_B
        for j in range(N):
            h, origin = self._step(pos, sp, sc[j], j, L)
            sj = torch.multinomial(F.softmax(self._species_logits(h, rem), -1), 1).squeeze(-1)
            ab_frame, _ = self.flow.sample(h)                            # [B,2] in frame coords
            if self.frame_mode == "inertial":
                # Recompute nbr_rel at step j (placed prefix 0..j-1) to build R, matching _step logic.
                if j < 2:
                    # No / too few neighbours: R is arbitrary; use identity (theta=0 -> R=I).
                    ab = ab_frame
                else:
                    k = min(self.knn, j)
                    d_sc = _wrap_pm(pos[:, :j] - sc[j][None, None], L)  # [B,j,2]
                    dist2_j = (d_sc ** 2).sum(-1)                        # [B,j]
                    w = torch.exp(-dist2_j / (2 * 1.3 ** 2))
                    wsum = w.sum(1, keepdim=True).clamp_min(1e-12)
                    origin_j = torch.remainder(sc[j][None] + (w[..., None] * d_sc).sum(1) / wsum, L)
                    idx_j = dist2_j.topk(k, dim=1, largest=False).indices  # [B,k]
                    nbr_pos_j = torch.gather(pos[:, :j], 1, idx_j[..., None].expand(-1, -1, 2))
                    nbr_rel_j = _wrap_pm(nbr_pos_j - origin_j[:, None, :], L)  # [B,k,2]
                    R_j = self._inertial_R(nbr_rel_j)                    # [B,2,2]
                    # Rotate frame coords back to physical: ab = R^T @ ab_frame
                    ab = torch.einsum("bij,bj->bi", R_j.transpose(-1, -2), ab_frame)
            else:
                ab = ab_frame
            if rem is not None:
                rem[torch.arange(B, device=device), sj] -= 1
            pos[:, j] = torch.remainder(origin + ab * arc, L)
            sp[:, j] = sj
        perm = self.geo._curve_order(pos, N)
        pos = torch.gather(pos, 1, perm[..., None].expand(-1, -1, 2))
        sp = torch.gather(sp, 1, perm)
        if return_logq:
            return pos, sp, self.log_prob(pos, sp)                       # exactness trick (matches base model)
        return pos, sp


# --- append to liquid_coupling_flow/ka_flowhead.py ---
import os, time
ART = os.path.join(os.path.dirname(__file__), "artifacts")


def train(steps=20000, train_N=100, num_bins=8, tail_bound=4.0, warm=None, lr=3e-4,
          out=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    from liquid_coupling_flow.ka_gridformer_train import augment
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    data, sp, L = ref["x"].to(device), ref["s"].to(device).long(), ref["L"]; N = data.shape[1]
    m = KAFlowHeadModel(rho=1.2, n_bins=192, knn=16, num_bins=num_bins, tail_bound=tail_bound).to(device)
    if warm is not None:
        from liquid_coupling_flow.ka_exposure_lf import load_compat
        load_compat(m, torch.load(os.path.join(ART, warm), map_location=device, weights_only=False)["state_dict"])
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4); B, t0 = 128, time.time()
    warmup = 500
    for step in range(steps):
        lr_scale = min(1.0, (step + 1) / warmup)
        for g in opt.param_groups:
            g["lr"] = lr * lr_scale
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        loss = (-m.log_prob(augment(data[idx], L), sp) / N).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"NON-FINITE loss at step {step} (convergence-gate violation)")
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if step % 1000 == 0:
            print(f"  step {step:5d} nll/N {loss.item():.3f} lr {opt.param_groups[0]['lr']:.1e} {time.time()-t0:.0f}s", flush=True)
    out = out or f"ka_flowhead_N{train_N}_k{num_bins}{'_scratch' if warm is None else ''}.pt"
    torch.save({"state_dict": m.state_dict(), "rho": 1.2, "n_bins": 192, "knn": 16,
                "num_bins": num_bins, "tail_bound": tail_bound, "step": steps}, os.path.join(ART, out))
    print(f"saved {out}", flush=True); return out


@torch.no_grad()
def _categorical_pos_logdensity(train_N, device):
    """Mean per-particle NORMALIZED position log-density of the categorical baseline: log P(bin) - d*log(bin_w)."""
    from liquid_coupling_flow.ka_exposure_lf import _load
    m = _load("ka_localframe_N100_20k.pt", device)
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    s, L, data = ref["s"].to(device).long(), ref["L"], ref["x"].to(device)[:512]; N = data.shape[1]
    order = m.geo._curve_order(data, N); xo = torch.gather(data, 1, order[..., None].expand(-1, -1, 2))
    so = torch.gather(s.expand(data.shape[0], N) if s.dim() == 1 else s[:512], 1, order)
    context, origin = m._local(xo, so, m.geo._scaffold(N, device), L, N)
    ab = _wrap_pm(xo - origin, L) / m._arc_scale(N)
    ba, bb = m._bin(ab[..., 0]), m._bin(ab[..., 1])
    la = F.log_softmax(m.head_a(context), -1).gather(-1, ba[..., None]).squeeze(-1)
    lb = F.log_softmax(m.head_b(context + m.bin_a_emb(ba)), -1).gather(-1, bb[..., None]).squeeze(-1)
    return float(((la + lb) - m.d * math.log(m.bin_w)).mean())


def convergence_gate(train_N=100, device="cuda" if torch.cuda.is_available() else "cpu"):
    """(1) overfit a tiny batch; (2) flow normalized position log-density beats categorical; (3) no NaN."""
    from liquid_coupling_flow.ka_gridformer_train import augment
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    data, sp, L = ref["x"].to(device), ref["s"].to(device).long(), ref["L"]; N = data.shape[1]
    m = KAFlowHeadModel(rho=1.2, n_bins=192, knn=16).to(device); m.train()
    tiny = data[:8]
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3); nans = False
    first = None
    for step in range(400):
        loss = (-m.log_prob(tiny, sp) / N).mean()
        if first is None: first = loss.item()
        if not torch.isfinite(loss): nans = True; break
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    overfit_nll = loss.item()
    print(f"OVERFIT-TINY: nll/N {first:.3f} -> {overfit_nll:.3f} (should drop substantially)  NaN={nans}", flush=True)
    cat = _categorical_pos_logdensity(train_N, device)
    print(f"categorical normalized position log-density (baseline to beat): {cat:.3f}", flush=True)
    return {"overfit_first": first, "overfit_last": overfit_nll, "nan": nans, "categorical_pos_logdensity": cat}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "gate":
        convergence_gate()
    else:
        train(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
