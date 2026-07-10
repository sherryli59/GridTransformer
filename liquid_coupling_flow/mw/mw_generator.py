"""3D local-frame AR generator for the liquid-mW glass campaign.

Part 1 (this file, Task 8): the exactness-bearing geometry layer —
    - `mw_scaffold`: a gilbert3d curve on an R x R x R grid, giving per-cell
      curve rank and cell-center anchors t_j in curve-visit order.
    - `canonical_order`: a deterministic particle -> AR-step permutation
      (curve-cell rank, then distance-to-cell-center, both stable), defining
      q0 as a function of the configuration alone (no RNG, storage-order
      invariant).
    - `build_frames`: vectorized 3D Gram-Schmidt local frames from the two
      nearest valid neighbor displacement vectors, with a fallback hierarchy
      that guarantees an orthonormal right-handed triad (R R^T = I, det = +1)
      for every batch row, including degenerate (0/1-valid, collinear) cases.

Part 2 (this file, Task 9): the model —
    - `Spline3Head`: AR-chained 3-coordinate exact spline-flow head
      (p(a|h) p(b|h,a) p(c|h,a,b)), identity-initialized (== N(0,1)^3 base).
    - `MWGenerator`: the causal local-frame transformer generator. Offset
      convention: u_j = Rf_j @ wrap_pm(x_j - t_j, L) / s, s = (L/R)/2 (half
      cell width); log q_x,j = log q_u,j - 3 log s. Per-neighbor features are
      geometry-only (frame coords + radial distance + fixed-period Fourier
      encodings of r + excluded-volume pair features [min(1/r^2,4), r/1.19])
      so the model carries no curve-specific or absolute-position information
      -- the size-transfer invariant this campaign needs (see MEMORY.md
      "curve-conditioning-blocks-transfer").

Part 3 (this file, Task 10): training CLI + Phase-2 entry diagnostics --
    - `load_training_bank`: reuses the certified G2 reference MC bank
      (event-major MC snapshots, arbitrary storage order) as MLE training
      data, thinned for decorrelation and split by event order (val strictly
      later-in-time than train).
    - `train` / `load_generator`: val-loss-checkpointed MLE training CLI
      (always-save-last + best-on-val, per this repo's checkpointing rule).
    - `orderspread`: Phase-2 entry diagnostics on reference configs --
      log-density spread under random storage orderings, canonicalization
      stability under the frozen MC step size, and the trained sampler's
      carry-forward exactness counters (n_wrapped, noncanonical fraction).
      Recorded for the record, not gated.

Ordering modes (which q0 lives where):
    - SMC base / importance weights: `log_prob(x, L, preordered=True)` scores
      x in its GIVEN storage order (slot j = AR step j, anchor t_j, causal
      prefix = slots < j). A sample's true density is the AR density in
      generation order, and `sample()` returns slot j == step j, so this is
      the exact density of `sample()`'s own output; canonical-lifting a
      sampled config CHANGES the value whenever the config is noncanonical
      (a particle strayed from its generation-step cell), which would bias
      importance weights.
    - Training / scoring reference configs: `log_prob(x, L)` (default,
      preordered=False) canonicalizes first -- the deterministic
      `canonical_order` factorization of unordered data.
"""
from __future__ import annotations
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from liquid_coupling_flow.transforms_spline import RQSplineElementwise, DEFAULT_MIN_DERIVATIVE
from liquid_coupling_flow.mw.mw_energy import RHO_STAR

_LOG2PI = math.log(2 * math.pi)
ART = os.path.join(os.path.dirname(__file__), "artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# excluded-volume pair features (ARM-FULL P1 lineage), appended to each neighbor token:
R_SHELL1 = 1.19        # first-shell g(r) peak location (sigma units) -- the physical contact scale
EV_RMIN = 0.5          # 1/r^2 clamp radius: feature = 1/clamp_min(r, 0.5)^2 == min(1/r^2, 4.0)
N_PAIR_FEAT = 2


def wrap_pm(v, L):
    """Minimum-image wrap of a displacement into (-L/2, L/2] componentwise (module-local: mw/ stays
    self-contained rather than importing ka_flowhead's `_wrap_pm`)."""
    return v - L * torch.round(v / L)


def mw_scaffold(N, L, device):
    """Gilbert-curve scaffold on an R x R x R grid (R = N^(1/3), asserts cube).

    Returns:
        t: [N,3] float32 cell-center anchors, in curve-visit order.
        rank: [R,R,R] int64, rank[i,j,k] = curve-visit index of cell (i,j,k).
        R: int, grid side length.
    """
    R = round(N ** (1 / 3))
    assert R ** 3 == N, f"N={N} not a cube"
    from grid_transformer.data.gilbert import gilbert3d_path
    path = torch.as_tensor(gilbert3d_path(R, R, R), device=device)           # [N,3] cells in curve order
    rank = torch.zeros(R, R, R, dtype=torch.long, device=device)
    rank[path[:, 0], path[:, 1], path[:, 2]] = torch.arange(N, device=device)
    t = (path.float() + 0.5) * (L / R)
    return t, rank, R


def canonical_order(x, L, R, rank):
    """Deterministic particle -> AR-step permutation.

    Sorts by (curve rank of the containing cell, then distance to that
    cell's center), both argsorts stable, so the result is a function of the
    configuration only (independent of storage order and of RNG).

    Args:
        x: [B,N,3] positions, REQUIRED in [0, L) (wrap upstream; asserted —
            .long() truncates toward zero, so e.g. x = -1e-7 would silently
            misclassify into cell 0, and this function defines q0).
        L: box length.
        R: grid side length (from mw_scaffold).
        rank: [R,R,R] int64 curve rank per cell (from mw_scaffold).

    Returns:
        perm: [B,N] int64, perm[b] is a permutation of range(N).
    """
    assert bool((x >= 0).all() and (x < L).all()), "canonical_order requires x in [0,L) (wrap upstream)"
    B, N, _ = x.shape
    w = L / R
    cell = (x / w).long().clamp(0, R - 1)
    rk = rank[cell[..., 0], cell[..., 1], cell[..., 2]]                       # [B,N]
    d2 = ((x - (cell.float() + 0.5) * w) ** 2).sum(-1)
    o1 = torch.argsort(d2, dim=1, stable=True)
    o2 = torch.argsort(torch.gather(rk, 1, o1), dim=1, stable=True)
    return torch.gather(o1, 1, o2)


def build_frames(nbr_rel, n_valid, col_tol=0.99):
    """Vectorized 3D Gram-Schmidt local frames.

    Args:
        nbr_rel: [.,k,3] displacement vectors sorted nearest-first (rows
            beyond n_valid are arbitrary/invalid, including all-zero).
            Nominally-valid rows with zero norm (<= 1e-9) are also treated
            as invalid — they fall through to the same fallback hierarchy.
        n_valid: [.] int count of valid rows per batch element.
        col_tol: |cos| threshold above which a candidate second vector is
            treated as collinear with e1 and rejected.

    Fallback hierarchy for the second axis:
        0 valid  -> e1 = [1,0,0] (identity-equivalent), e2 = coordinate axis
                    least aligned with e1.
        1 valid  -> e1 from the single neighbor, e2 = coordinate axis least
                    aligned with e1.
        >=2 valid, nearest two collinear (|cos| > col_tol) -> scan later
                    valid neighbors for the first non-collinear one; if none,
                    fall back to axis completion.
        else     -> e2 from the nearest non-collinear valid neighbor.

    Returns:
        Rf: [.,3,3], rows are the frame axes (e1, e2, e3); R R^T = I and
            det(R) = +1 for every row, including all fallback paths.
    """
    B, k, _ = nbr_rel.shape
    d1 = nbr_rel[:, 0]
    # Degenerate-"valid" guard: an exactly-zero row passes F.normalize silently as
    # [0,0,0] (and its cos vs e1 is 0 <= col_tol, so the ok-mask would even SELECT
    # it as d2) -> treat zero-norm rows as invalid in both the e1 and d2 paths.
    nz = nbr_rel.norm(dim=-1) > 1e-9                                          # [B,k]
    e1 = F.normalize(torch.where(((n_valid >= 1) & nz[:, 0])[:, None], d1,
         torch.tensor([1., 0., 0.], device=d1.device).expand_as(d1)), dim=-1)
    # second vector: first nonzero neighbor with |cos| <= col_tol among indices 1..k-1, else axis fallback
    cos = torch.einsum("bkd,bd->bk", F.normalize(nbr_rel, dim=-1), e1).abs()
    ok = (cos <= col_tol) & nz & (torch.arange(k, device=d1.device)[None] < n_valid[:, None]) \
         & (torch.arange(k, device=d1.device)[None] >= 1)
    idx2 = torch.where(ok.any(1), ok.float().argmax(1), torch.zeros_like(n_valid))
    d2 = torch.gather(nbr_rel, 1, idx2[:, None, None].expand(-1, 1, 3)).squeeze(1)
    # axis fallback (0/1 valid, or all collinear): coordinate axis least aligned with e1
    axes = torch.eye(3, device=d1.device)
    ax = axes[e1.abs().argmin(-1)]
    use_ax = (~ok.any(1)) | (n_valid < 2)
    d2 = torch.where(use_ax[:, None], ax, d2)
    u2 = d2 - (d2 * e1).sum(-1, keepdim=True) * e1
    e2 = F.normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-2)


# ---------------------------------------------------------------------------
# Task 9: model
# ---------------------------------------------------------------------------

class Spline3Head(nn.Module):
    """3-coordinate AR spline head: p(a|h) p(b|h,a) p(c|h,a,b). Identity init (== N(0,1)^3 base).

    Ported from ka_flowhead.SplineFlowHead's 2-coordinate identity-init pattern (zero weights;
    bias[2K:] = const so softplus(const)+min_deriv == 1 -> the RQS is the identity at init and the
    head is, at construction, a pure standard-normal base entirely independent of h)."""

    def __init__(self, d_model, num_bins=8, tail_bound=4.0):
        super().__init__()
        self.spline = RQSplineElementwise(num_bins=num_bins, tail_bound=tail_bound)
        self.num_bins = num_bins
        P = self.spline.params_per_dim
        self.heads = nn.ModuleList([nn.Linear(d_model + i, P) for i in range(3)])
        const = float(torch.log(torch.expm1(torch.tensor(1.0 - DEFAULT_MIN_DERIVATIVE))))
        for hd in self.heads:
            nn.init.zeros_(hd.weight)
            with torch.no_grad():
                hd.bias.zero_(); hd.bias[2 * num_bins:] = const

    def logp_a(self, h, a):                              # exposed for the normalization test
        z, ld = self.spline.inverse(a, self.heads[0](h))
        return -0.5 * z ** 2 - 0.5 * math.log(2 * math.pi) + ld

    def log_prob(self, h, u):
        lp, ctx = 0.0, h
        for i in range(3):
            ui = u[..., i:i + 1]
            z, ld = self.spline.inverse(ui, self.heads[i](ctx))
            lp = lp + (-0.5 * z ** 2 - 0.5 * math.log(2 * math.pi) + ld).squeeze(-1)
            ctx = torch.cat([ctx, ui], -1)
        return lp

    def sample(self, h, gen=None):
        us, lp, ctx = [], 0.0, h
        for i in range(3):
            z = torch.randn(h.shape[:-1] + (1,), device=h.device, generator=gen)
            ui, ld = self.spline.forward(z, self.heads[i](ctx))
            lp = lp + (-0.5 * z ** 2 - 0.5 * math.log(2 * math.pi) - ld).squeeze(-1)
            us.append(ui); ctx = torch.cat([ctx, ui], -1)
        return torch.cat(us, -1), lp


class MWGenerator(nn.Module):
    """Causal local-frame AR generator over an mw_scaffold gilbert3d curve.

    Per-step j (curve rank), the anchor t_j conditions a causal-kNN context of already-placed
    particles (canonical order, i < j only). Per-neighbor features are geometry-only: frame
    coordinates + radial distance + fixed-period Fourier encodings of r + excluded-volume pair
    features [min(1/r^2, 4), r/1.19] (ARM-FULL P1 lineage: direct excluded-volume inductive bias;
    1.19 sigma = first-shell g(r) peak) -- no curve index, no absolute position, so the
    conditioning is a function of local geometry alone (the transfer invariant; see
    test_prefix_storage_invariance / test_translation_by_lattice below).

    log_prob(x, L) does ONE teacher-forced vectorized pass (the `_inertial_R_logprob` pattern from
    ka_flowhead.py, ported to 3D: causal mask via broadcast distances to every anchor, topk-kNN,
    gather). sample(B, N, L) mirrors the same per-step feature construction sequentially.
    """

    def __init__(self, knn=12, d_model=128, n_layers=2, n_heads=4, num_bins=8, tail_bound=4.0,
                 periods=(0.5, 1.0, 2.0, 4.0), n_pair_feat=N_PAIR_FEAT):
        super().__init__()
        assert n_pair_feat in (0, N_PAIR_FEAT), \
            f"n_pair_feat must be 0 (v1 12-dim tokens) or {N_PAIR_FEAT} (v2), got {n_pair_feat}"
        self.knn = knn
        self.d_model = d_model
        self.periods = tuple(periods)
        self.n_pair_feat = n_pair_feat
        n_feat = 3 + 1 + 2 * len(self.periods) + n_pair_feat     # frame(3) + r(1) + sincos(2P) [+ pair(2)]
        self.embed = nn.Linear(n_feat, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model,
                                            dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = Spline3Head(d_model, num_bins=num_bins, tail_bound=tail_bound)

    def _fourier(self, r):
        """r: [...,1] radial distance -> [...,2*len(periods)] fixed-period sin/cos encoding."""
        feats = []
        for p in self.periods:
            feats.append(torch.sin((2 * math.pi / p) * r))
            feats.append(torch.cos((2 * math.pi / p) * r))
        return torch.cat(feats, dim=-1)

    def _tokens(self, Rf, nbr_rel):
        """Rf: [...,3,3], nbr_rel: [...,k,3] (vector to anchor, world frame) -> token feats
        [...,k,12+n_pair_feat].

        Layout: [frame coords(3), r(1), sincos(2P=8)] + (when n_pair_feat=2) [min(1/r^2, 4)(1),
        r/1.19(1)]. The pair tail is the ARM-FULL P1 excluded-volume inductive bias: 1/r^2 clamped
        at r >= EV_RMIN=0.5 (== capped at 4.0; implemented as 1/clamp_min(r,0.5)^2 so the clamped
        branch has exactly zero gradient -- no inf at the zeroed invalid-slot rows r=0), and r
        scaled by R_SHELL1=1.19 sigma (the first-shell g(r) peak location, the physical unit for
        "contact distance"). n_pair_feat=0 reproduces the v1 12-dim layout exactly (checkpoint
        back-compat).

        SINGLE source of truth for per-neighbor features: both log_prob's vectorized pass and
        sample's per-step pass call this, so they cannot desynchronize
        (test_sample_logprob_consistency_perturbed is the mirror-correctness guard)."""
        r = nbr_rel.norm(dim=-1, keepdim=True)                                   # [...,k,1]
        feat_xyz = torch.einsum("...ij,...mj->...mi", Rf, nbr_rel)               # [...,k,3] frame coords
        four = self._fourier(r)                                                   # [...,k,2P]
        feats = [feat_xyz, r, four]
        if self.n_pair_feat > 0:
            inv_r2 = 1.0 / r.clamp_min(EV_RMIN) ** 2                              # [...,k,1] == min(1/r^2, 4)
            feats += [inv_r2, r / R_SHELL1]
        return torch.cat(feats, dim=-1)                                           # [...,k,12+n_pair_feat]

    # ------------------------------------------------------------------
    # log_prob: one teacher-forced vectorized pass
    # ------------------------------------------------------------------

    def log_prob(self, x, L, preordered=False):
        """Exact AR log-density of x under this generator. Returns [B].

        preordered=False (default; training / scoring reference configs): canonicalize first via
        `canonical_order` -- the deterministic factorization of unordered data.
        preordered=True (SMC base / importance weights): score x in its GIVEN storage order (slot j =
        AR step j, anchor t_j, causal prefix = slots < j). This is the exact density of `sample()`'s
        own output (which returns slot j == generation step j); canonical-lifting a sampled config
        changes the value whenever it is noncanonical, biasing IS weights (KA lineage:
        ka_flowhead.log_prob's `preordered` arg)."""
        B, N, _ = x.shape
        device = x.device
        x = torch.remainder(x, L)
        t, rank, R = mw_scaffold(N, L, device)
        s = (L / R) / 2.0
        if preordered:
            xo = x                                                                # slot j == AR step j
        else:
            perm = canonical_order(x, L, R, rank)
            xo = torch.gather(x, 1, perm[..., None].expand(-1, -1, 3))           # [B,N,3] canonical order

        K = min(self.knn, N)
        jj = torch.arange(N, device=device)
        causal = jj[None, None, :] < jj[None, :, None]                            # [1,N(step j),N(cand k)]: k<j
        d = wrap_pm(xo[:, None, :, :] - t[None, :, None, :], L)                   # [B,N(step),N(cand),3]
        dist2 = (d ** 2).sum(-1)                                                   # [B,N,N]
        # Exact-distance topk ties are measure-zero for continuous positions, and both passes rank
        # bit-identical dist2 values, so tie-breaking cannot desynchronize sample vs log_prob.
        idx = dist2.masked_fill(~causal, 1e9).topk(K, dim=2, largest=False).indices  # [B,N,K] nearest-first
        nbr_pos = torch.gather(xo[:, None, :, :].expand(B, N, N, 3), 2,
                                idx[..., None].expand(-1, -1, -1, 3))               # [B,N,K,3]
        nbr_rel = wrap_pm(nbr_pos - t[None, :, None, :], L)                        # [B,N,K,3] rel to anchor t_j
        valid = torch.gather(causal.expand(B, N, N), 2, idx)                       # [B,N,K] bool
        nbr_rel = nbr_rel * valid[..., None].to(nbr_rel.dtype)                     # zero invalid slots
        n_valid = valid.sum(-1)                                                     # [B,N]

        Rf = build_frames(nbr_rel.reshape(B * N, K, 3), n_valid.reshape(B * N)).reshape(B, N, 3, 3)

        off = wrap_pm(xo - t[None], L)                                             # [B,N,3]
        u = torch.einsum("bnij,bnj->bni", Rf, off) / s                             # [B,N,3] world -> frame

        tok_feat = self._tokens(Rf, nbr_rel)                                       # [B,N,K,14]
        tok = self.embed(tok_feat)                                                  # [B,N,K,d]
        cls = self.cls.expand(B, N, 1, self.d_model)
        seq = torch.cat([cls, tok], dim=2)                                          # [B,N,K+1,d]
        pad_mask = torch.cat([torch.zeros(B, N, 1, dtype=torch.bool, device=device), ~valid], dim=2)

        BN = B * N
        seq_flat = seq.reshape(BN, K + 1, self.d_model)
        mask_flat = pad_mask.reshape(BN, K + 1)
        outs = []
        for i0 in range(0, BN, 8192):
            sl = slice(i0, i0 + 8192)
            out = self.encoder(seq_flat[sl], src_key_padding_mask=mask_flat[sl])
            outs.append(out[:, 0])                                                  # CLS position
        h = torch.cat(outs, dim=0).reshape(B, N, self.d_model)

        lq_u = self.head.log_prob(h, u)                                             # [B,N]
        return lq_u.sum(1) - 3 * N * math.log(s)

    # ------------------------------------------------------------------
    # sample: sequential j-loop mirroring log_prob's feature construction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, B, N, L, gen=None):
        device = next(self.parameters()).device
        t, rank, R = mw_scaffold(N, L, device)
        s = (L / R) / 2.0
        x = torch.zeros(B, N, 3, device=device)
        logq = torch.zeros(B, device=device)
        n_wrapped = 0
        for j in range(N):
            tj = t[j]
            k = min(self.knn, j)
            if k > 0:
                placed = x[:, :j, :]                                                # [B,j,3]
                d = wrap_pm(placed - tj[None, None, :], L)                          # [B,j,3]
                dist2 = (d ** 2).sum(-1)                                             # [B,j]
                idx = dist2.topk(k, dim=1, largest=False).indices                    # [B,k] nearest-first (ties: see log_prob)
                real = torch.gather(d, 1, idx[..., None].expand(-1, -1, 3))          # [B,k,3] rel to anchor t_j
            else:
                real = torch.zeros(B, 0, 3, device=device)
            pad_n = self.knn - k
            frame_in = torch.cat([real, torch.zeros(B, pad_n, 3, device=device)], dim=1) if pad_n > 0 else real
            n_valid = torch.full((B,), k, dtype=torch.long, device=device)
            Rf = build_frames(frame_in, n_valid)                                     # [B,3,3]

            tok_feat = self._tokens(Rf, real)                                        # [B,k,14]
            tok = self.embed(tok_feat)                                               # [B,k,d]
            cls = self.cls.expand(B, 1, self.d_model)
            seq = torch.cat([cls, tok], dim=1)                                       # [B,k+1,d]
            h = self.encoder(seq)[:, 0]                                              # [B,d] (no padding: all real)

            u, lq_u = self.head.sample(h, gen=gen)                                   # [B,3], [B]
            off = torch.einsum("bij,bi->bj", Rf, u * s)                              # frame -> world
            n_wrapped += int((off.abs() > L / 2).any(-1).sum().item())               # counts (config,step) events, not configs
            x[:, j, :] = torch.remainder(tj[None, :] + off, L)
            logq = logq + lq_u - 3 * math.log(s)
        return x, logq, n_wrapped


# ---------------------------------------------------------------------------
# Task 10: training CLI + Phase-2 entry diagnostics
# ---------------------------------------------------------------------------

CHAINS_B = 16                  # independent MC chains in the G2 reference bank (mw_gates.REF_B);
                                # event-major layout: cfgs[event*CHAINS_B + chain] -- mc_run's
                                # collection loop appends one [B,N,3] snapshot per collection event.

DEFAULT_MODEL_KW = dict(knn=12, d_model=128, n_layers=2, n_heads=4, num_bins=8, tail_bound=4.0,
                         periods=(0.5, 1.0, 2.0, 4.0), n_pair_feat=N_PAIR_FEAT)


def load_training_bank(art_path=None, thin_events=6, val_frac=0.1):
    """Reuse the certified G2 reference bank (mw_gates.g2's `ref` MC run) as MLE training data.

    cfgs [n_events*CHAINS_B, N, 3] is event-major (index = event*CHAINS_B + chain: mc_run's
    collection loop appends one [CHAINS_B,N,3] snapshot per collection event, `every` sweeps apart --
    4 sweeps for the G2 bank). Thin by taking every `thin_events`-th EVENT (all chains kept, for
    decorrelation), then split by EVENT ORDER (not interleaved) so validation is strictly
    later-in-time than training -- a cheap guard against leakage through residual autocorrelation.

    Args:
        art_path: bank .pt path (default artifacts/mw_g2_N64.pt).
        thin_events: keep every this-th collection event.
        val_frac: fraction of the THINNED events (not raw events) held out, taken from the end.

    Returns:
        x_train, x_val: [.,N,3] float32, wrapped into [0,L).
        L: box length, (N/RHO_STAR)^(1/3).
    """
    if art_path is None:
        art_path = os.path.join(ART, "mw_g2_N64.pt")
    bank = torch.load(art_path, map_location="cpu")
    cfgs = bank["ref"]["cfgs"]                                            # [n_events*CHAINS_B, N, 3]
    n_tot, N, _ = cfgs.shape
    assert n_tot % CHAINS_B == 0, f"bank size {n_tot} not a multiple of CHAINS_B={CHAINS_B}"
    n_events = n_tot // CHAINS_B
    cfgs = cfgs.reshape(n_events, CHAINS_B, N, 3)[::thin_events]          # [n_thin,CHAINS_B,N,3]
    n_thin = cfgs.shape[0]
    n_val_events = max(1, int(round(n_thin * val_frac)))
    n_train_events = n_thin - n_val_events
    assert n_train_events > 0, f"val_frac={val_frac} leaves no training events (n_thin={n_thin})"
    x_train = cfgs[:n_train_events].reshape(-1, N, 3)                     # earlier events
    x_val = cfgs[n_train_events:].reshape(-1, N, 3)                       # later events
    L = (N / RHO_STAR) ** (1.0 / 3.0)
    x_train = torch.remainder(x_train, L)
    x_val = torch.remainder(x_val, L)
    return x_train, x_val, L


def _augment_batch(x, L, gen):
    """Exact on-the-fly data augmentation: per batch element, a continuous torus translation
    followed by a random element of the 48-element cubic group O_h.

    (a) translation: x -> remainder(x + v, L), v ~ U[0,L)^3 drawn per config -- exact symmetry of
        the periodic box.
    (b) O_h element as a SIGNED COORDINATE PERMUTATION about the box center c = L/2:
        y_i = s_i * (x - c)_{perm(i)} + c, perm ~ U(S_3), s ~ U{+-1}^3. Signed permutation
        matrices with arbitrary sign pattern realize the full 48-element group O_h (24 proper
        rotations x optional inversion); the improper half is exact too (SW/mW is achiral and the
        box symmetric). Deliberately NOT continuous rotations -- those are not a symmetry of the
        periodic box.

    All randomness through `gen` (device must match x). Returns [B,N,3] in [0,L).
    """
    B, N, _ = x.shape
    device = x.device
    v = torch.rand(B, 1, 3, device=device, generator=gen) * L
    x = torch.remainder(x + v, L)
    perm = torch.rand(B, 3, device=device, generator=gen).argsort(dim=1)          # [B,3] ~ U(S_3)
    signs = (torch.randint(0, 2, (B, 3), device=device, generator=gen) * 2 - 1).to(x.dtype)
    c = L / 2.0
    xc = x - c                                                                     # [-c, c)
    xp = torch.gather(xc, 2, perm[:, None, :].expand(-1, N, -1))                   # xp[...,i] = xc[...,perm[i]]
    return torch.remainder(c + signs[:, None, :] * xp, L)


def _val_nll(model, x_val, L, N, chunk=64):
    """Full-val NLL/particle, chunked, no_grad. Restores the model's train/eval mode on exit."""
    was_training = model.training
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, x_val.shape[0], chunk):
            xb = x_val[i:i + chunk]
            total += float((-model.log_prob(xb, L)).sum().item())
    if was_training:
        model.train()
    return total / (x_val.shape[0] * N)


def _ckpt_hparams(ckpt):
    """Model hyperparams from a checkpoint dict, with n_pair_feat back-compat.

    n_pair_feat: use the stored key when present; otherwise INFER it from the embed weight's input
    dim (embed.weight.shape[1] - (3+1+2P)) -- unambiguous, and covers BOTH pre-schema v1 ckpts
    (12-dim tokens -> 0) and key-less ckpts written by an in-flight v2 process holding the
    pre-schema module in memory (14-dim tokens -> 2)."""
    kw = {k: ckpt[k] for k in ("knn", "d_model", "n_layers", "n_heads", "num_bins",
                                 "tail_bound", "periods")}
    if "n_pair_feat" in ckpt:
        kw["n_pair_feat"] = ckpt["n_pair_feat"]
    else:
        base = 3 + 1 + 2 * len(kw["periods"])
        kw["n_pair_feat"] = int(ckpt["state_dict"]["embed.weight"].shape[1]) - base
    return kw


def _model_from_ckpt(ckpt):
    """Construct an (untrained) MWGenerator from a checkpoint's recorded hyperparams."""
    return MWGenerator(**_ckpt_hparams(ckpt))


def train(steps=20000, batch=32, lr=3e-4, val_every=500, out="mw_gen_N64.pt", seed=0,
          art_path=None, thin_events=6, val_frac=0.1, device=None, augment=True, warm=None,
          **model_kw):
    """MLE-train MWGenerator on the banked reference configs (load_training_bank).

    Loss = -log_prob(batch, L).mean()/N, CANONICAL mode (preordered=False): the banked training
    data comes off the MC chains in arbitrary storage order (not a curve-visit order), and
    canonical_order is the deterministic AR factorization of unordered data. `model_kw` overrides
    MWGenerator's default hyperparams (DEFAULT_MODEL_KW); `art_path`/`thin_events`/`val_frac` let
    tests point at a small synthetic bank without touching the production data path.

    augment=True applies _augment_batch (exact torus translations + cubic O_h) to each TRAIN batch
    only; validation stays fixed/unaugmented so val NLL is comparable across evals (checkpointing).

    warm: optional checkpoint path to warm-continue from. The model is constructed from THAT
    checkpoint's recorded hyperparams (`model_kw` is IGNORED -- the loaded state_dict fixes the
    architecture) and its state_dict loaded; best_val initializes from the checkpoint's val_nll so
    the continuation only overwrites the best file if it genuinely improves on the loaded model.
    The Adam optimizer is fresh (no optimizer state is carried).

    Val-loss checkpointing is a hard requirement (train-loss checkpointing is the measured trap in
    this repo, MEMORY "exposure-bias-noise-run"): every val_every steps this evaluates full-val
    NLL/particle and saves BOTH the best-on-val checkpoint (`out`) and an always-saved last
    checkpoint (`out` with a `_last` suffix), so a long run's progress survives a val regression.

    Returns {"out_path", "last_path", "history", "best_val"} (history: per-eval step/train_loss/val_nll).
    """
    torch.manual_seed(seed)
    device = device or DEV

    x_train, x_val, L = load_training_bank(art_path=art_path, thin_events=thin_events, val_frac=val_frac)
    x_train, x_val = x_train.to(device), x_val.to(device)
    N = x_train.shape[1]

    if warm is not None:
        warm_path = warm if os.path.exists(warm) else os.path.join(ART, warm)   # cwd-relative or ART name
        wc = torch.load(warm_path, map_location=device)
        kw = _ckpt_hparams(wc)
        model = _model_from_ckpt(wc).to(device)
        model.load_state_dict(wc["state_dict"])
        best_val = float(wc["val_nll"])
        for k, cur in (("thin_events", thin_events), ("val_frac", val_frac)):
            if k in wc and wc[k] != cur:                  # data-provenance guard: best_val was measured
                print(f"WARNING train: warm ckpt {k}={wc[k]} != current {cur} -- carried best_val "
                      f"was measured on a DIFFERENT val split; comparability not guaranteed", flush=True)
        print(f"warm-continue from {warm_path}: loaded step {wc.get('step')} "
              f"val_nll {best_val:.4f} (best_val carried; fresh Adam)", flush=True)
    else:
        kw = {**DEFAULT_MODEL_KW, **model_kw}
        model = MWGenerator(**kw).to(device)
        best_val = float("inf")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    out_path = out if os.path.isabs(out) else os.path.join(ART, out)
    assert out_path.endswith(".pt"), f"out must end with .pt, got {out!r}"
    last_path = out_path[:-3] + "_last.pt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    aug_gen = torch.Generator(device=device).manual_seed(seed + 1) if augment else None

    uniform_nll = 3.0 * math.log(L)                       # calibration reference: -log(L^-3N)/N
    print(f"train: N={N} L={L:.4f} n_train={x_train.shape[0]} n_val={x_val.shape[0]} device={device} "
          f"augment={augment} warm={warm} uniform_baseline_nll_per_particle={uniform_nll:.4f}",
          flush=True)

    def _ckpt(val_nll, step):
        return {"state_dict": model.state_dict(), "knn": kw["knn"], "d_model": kw["d_model"],
                "n_layers": kw["n_layers"], "n_heads": kw["n_heads"], "num_bins": kw["num_bins"],
                "tail_bound": kw["tail_bound"], "periods": kw["periods"],
                "n_pair_feat": kw["n_pair_feat"], "val_nll": val_nll, "step": step, "train_N": N,
                "thin_events": thin_events, "val_frac": val_frac}         # data provenance (warm guard)

    history = []
    for step in range(steps):
        idx = torch.randint(0, x_train.shape[0], (batch,), device=device)
        xb = x_train[idx]
        if augment:
            xb = _augment_batch(xb, L, aug_gen)
        loss = -model.log_prob(xb, L).mean() / N
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss = float(loss.item())

        if step % val_every == 0 or step == steps - 1:
            val_nll = _val_nll(model, x_val, L, N)
            print(f"step {step}: train {train_loss:.4f} val {val_nll:.4f}", flush=True)
            history.append({"step": step, "train_loss": train_loss, "val_nll": val_nll})
            ckpt = _ckpt(val_nll, step)
            torch.save(ckpt, last_path)                    # always-save-last
            if val_nll < best_val:
                best_val = val_nll
                torch.save(ckpt, out_path)                  # best-on-val

    return {"out_path": out_path, "last_path": last_path, "history": history, "best_val": best_val}


def load_generator(path, device=None):
    """Construct a MWGenerator from a checkpoint's recorded hyperparams and load its weights."""
    device = device or DEV
    ckpt = torch.load(path, map_location=device)
    model = _model_from_ckpt(ckpt)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def orderspread(ckpt="mw_gen_N64.pt", M=64, K=32, seed=0, bank_path=None, device=None):
    """Phase-2 entry diagnostics on reference configs for a trained generator (recorded, NOT gated).

    (i)   log q~ spread under K RANDOM storage orderings (preordered=True, forced-order scoring):
          per the SMC-base role log_prob(..., preordered=True) scores x in ITS GIVEN storage order,
          so re-labeling storage moves the value -- this measures that spread against the 2D-WCA
          campaign's 0.027 nats/particle ordering-penalty benchmark (MEMORY
          "ordered-space-mh-exactness-efficiency").
    (ii)  canonicalization stability under the frozen MC step-size perturbation: how often the
          canonical_order permutation itself changes (a particle crosses a cell boundary), and the
          resulting |Delta log_prob| (canonical mode) on those flips -- the scale of the
          mutation-MH discontinuity this ordering choice would induce.
    (iii) the trained sampler's carry-forward exactness counters: n_wrapped and the noncanonical
          fraction of canonical_order(sample) == arange(N).

    Saves all arrays to artifacts/mw_orderspread_N64.pt; returns the same dict.
    """
    device = device or DEV
    ckpt_path = ckpt if os.path.isabs(ckpt) else os.path.join(ART, ckpt)
    N = torch.load(ckpt_path, map_location="cpu")["train_N"]
    model = load_generator(ckpt_path, device)
    L = (N / RHO_STAR) ** (1.0 / 3.0)

    if bank_path is None:
        bank_path = os.path.join(ART, "mw_g2_N64.pt")
    bank = torch.load(bank_path, map_location="cpu")
    cfgs_all = torch.remainder(bank["ref"]["cfgs"], L)
    ref_step = float(bank["ref"]["step"])

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(cfgs_all.shape[0], generator=g)[:M]
    cfgs = cfgs_all[idx].to(device)                                       # [M,N,3]

    t, rank, R = mw_scaffold(N, L, device)

    # (i) K random orderings -> per-config log q~ spread
    gK = torch.Generator(device=device).manual_seed(seed + 1)
    lps = torch.empty(K, M)
    with torch.no_grad():
        for k in range(K):
            perm = torch.stack([torch.randperm(N, device=device, generator=gK) for _ in range(M)])
            xp = torch.gather(cfgs, 1, perm[..., None].expand(-1, -1, 3))
            lps[k] = model.log_prob(xp, L, preordered=True).cpu()
    std_per_config = lps.std(dim=0) / N                                   # [M] nats/particle
    mean_std, max_std = float(std_per_config.mean()), float(std_per_config.max())
    print(f"orderspread (i): K={K} random orderings over M={M} configs -> log q~ std/N "
          f"mean={mean_std:.4f} max={max_std:.4f} (2D-WCA ordering-penalty ref: 0.027 nats/particle)",
          flush=True)

    # (ii) canonicalization stability under the frozen MC step-size perturbation
    gp = torch.Generator(device=device).manual_seed(seed + 2)
    pert = ref_step * torch.randn(M, N, 3, device=device, generator=gp)
    cfgs_pert = torch.remainder(cfgs + pert, L)
    perm0 = canonical_order(cfgs, L, R, rank)
    perm1 = canonical_order(cfgs_pert, L, R, rank)
    changed = (perm0 != perm1).any(dim=1)                                 # [M]
    p_changed = float(changed.float().mean())
    with torch.no_grad():
        lp0 = model.log_prob(cfgs, L)
        lp1 = model.log_prob(cfgs_pert, L)
    dlp = (lp1 - lp0).abs()
    n_flips = int(changed.sum())
    dlp_flip = dlp[changed].cpu()
    print(f"orderspread (ii): P(canonical order changed)={p_changed:.4f} (n_flips={n_flips}/{M}, "
          f"step={ref_step:.4f}); |delta log_prob| on flips: "
          f"mean={float(dlp_flip.mean()) if n_flips else float('nan'):.4f} "
          f"max={float(dlp_flip.max()) if n_flips else float('nan'):.4f}", flush=True)

    # (iii) trained sampler's carry-forward exactness counters
    gS = torch.Generator(device=device).manual_seed(seed + 3)
    with torch.no_grad():
        xs, _, n_wrapped = model.sample(512, N, L, gen=gS)
    perm_s = canonical_order(xs, L, R, rank)
    noncanon = (perm_s != torch.arange(N, device=device)[None]).any(dim=1)
    noncanon_frac = float(noncanon.float().mean())
    print(f"orderspread (iii): sample(512) -> n_wrapped={n_wrapped} (of {512 * N} steps), "
          f"noncanonical_frac={noncanon_frac:.4f}", flush=True)

    out = {"M": M, "K": K, "seed": seed, "N": N, "L": L, "ref_step": ref_step,
           "lps": lps, "std_per_config": std_per_config, "mean_std": mean_std, "max_std": max_std,
           "perm0": perm0.cpu(), "perm1": perm1.cpu(), "changed": changed.cpu(), "p_changed": p_changed,
           "dlp": dlp.cpu(), "dlp_flip": dlp_flip, "n_flips": n_flips,
           "n_wrapped": n_wrapped, "noncanon_frac": noncanon_frac}
    os.makedirs(ART, exist_ok=True)
    out_path = os.path.join(ART, "mw_orderspread_N64.pt")
    torch.save(out, out_path)
    print(f"orderspread: saved -> {out_path}", flush=True)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="mW generator training + Phase-2 entry diagnostics")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--steps", type=int, default=20000)
    p_tr.add_argument("--batch", type=int, default=32)
    p_tr.add_argument("--lr", type=float, default=3e-4)
    p_tr.add_argument("--val_every", type=int, default=500)
    p_tr.add_argument("--out", type=str, default="mw_gen_N64.pt")
    p_tr.add_argument("--seed", type=int, default=0)
    p_tr.add_argument("--thin_events", type=int, default=6)
    p_tr.add_argument("--val_frac", type=float, default=0.1)
    p_tr.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p_tr.add_argument("--warm", type=str, default=None,
                      help="checkpoint path to warm-continue from (hyperparams+weights loaded, "
                           "best_val carried, fresh Adam; model hyperparam flags are ignored)")
    p_tr.add_argument("--knn", type=int, default=DEFAULT_MODEL_KW["knn"])
    p_tr.add_argument("--d_model", type=int, default=DEFAULT_MODEL_KW["d_model"])
    p_tr.add_argument("--n_layers", type=int, default=DEFAULT_MODEL_KW["n_layers"])
    p_tr.add_argument("--n_heads", type=int, default=DEFAULT_MODEL_KW["n_heads"])
    p_tr.add_argument("--num_bins", type=int, default=DEFAULT_MODEL_KW["num_bins"])
    p_tr.add_argument("--tail_bound", type=float, default=DEFAULT_MODEL_KW["tail_bound"])

    p_os = sub.add_parser("orderspread")
    p_os.add_argument("--ckpt", type=str, default="mw_gen_N64.pt")
    p_os.add_argument("--M", type=int, default=64)
    p_os.add_argument("--K", type=int, default=32)
    p_os.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.cmd == "train":
        train(steps=args.steps, batch=args.batch, lr=args.lr, val_every=args.val_every, out=args.out,
              seed=args.seed, thin_events=args.thin_events, val_frac=args.val_frac,
              augment=args.augment, warm=args.warm, knn=args.knn, d_model=args.d_model,
              n_layers=args.n_layers, n_heads=args.n_heads, num_bins=args.num_bins,
              tail_bound=args.tail_bound)
    elif args.cmd == "orderspread":
        orderspread(ckpt=args.ckpt, M=args.M, K=args.K, seed=args.seed)
