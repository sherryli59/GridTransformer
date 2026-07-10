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
      encodings of r) so the model carries no curve-specific or absolute-
      position information -- the size-transfer invariant this campaign
      needs (see MEMORY.md "curve-conditioning-blocks-transfer").

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from liquid_coupling_flow.transforms_spline import RQSplineElementwise, DEFAULT_MIN_DERIVATIVE

_LOG2PI = math.log(2 * math.pi)


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
    coordinates + radial distance + fixed-period Fourier encodings of r -- no curve index, no
    absolute position, so the conditioning is a function of local geometry alone (the transfer
    invariant; see test_prefix_storage_invariance / test_translation_by_lattice below).

    log_prob(x, L) does ONE teacher-forced vectorized pass (the `_inertial_R_logprob` pattern from
    ka_flowhead.py, ported to 3D: causal mask via broadcast distances to every anchor, topk-kNN,
    gather). sample(B, N, L) mirrors the same per-step feature construction sequentially.
    """

    def __init__(self, knn=12, d_model=128, n_layers=2, n_heads=4, num_bins=8, tail_bound=4.0,
                 periods=(0.5, 1.0, 2.0, 4.0)):
        super().__init__()
        self.knn = knn
        self.d_model = d_model
        self.periods = tuple(periods)
        n_feat = 3 + 1 + 2 * len(self.periods)                    # frame coords(3) + r(1) + sincos(2P)
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
        """Rf: [...,3,3], nbr_rel: [...,k,3] (vector to anchor, world frame) -> token feats [...,k,12]."""
        r = nbr_rel.norm(dim=-1, keepdim=True)                                   # [...,k,1]
        feat_xyz = torch.einsum("...ij,...mj->...mi", Rf, nbr_rel)               # [...,k,3] frame coords
        four = self._fourier(r)                                                   # [...,k,2P]
        return torch.cat([feat_xyz, r, four], dim=-1)                             # [...,k,12]

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

        tok_feat = self._tokens(Rf, nbr_rel)                                       # [B,N,K,12]
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
                idx = dist2.topk(k, dim=1, largest=False).indices                    # [B,k] nearest-first
                real = torch.gather(d, 1, idx[..., None].expand(-1, -1, 3))          # [B,k,3] rel to anchor t_j
            else:
                real = torch.zeros(B, 0, 3, device=device)
            pad_n = self.knn - k
            frame_in = torch.cat([real, torch.zeros(B, pad_n, 3, device=device)], dim=1) if pad_n > 0 else real
            n_valid = torch.full((B,), k, dtype=torch.long, device=device)
            Rf = build_frames(frame_in, n_valid)                                     # [B,3,3]

            tok_feat = self._tokens(Rf, real)                                        # [B,k,12]
            tok = self.embed(tok_feat)                                               # [B,k,d]
            cls = self.cls.expand(B, 1, self.d_model)
            seq = torch.cat([cls, tok], dim=1)                                       # [B,k+1,d]
            h = self.encoder(seq)[:, 0]                                              # [B,d] (no padding: all real)

            u, lq_u = self.head.sample(h, gen=gen)                                   # [B,3], [B]
            off = torch.einsum("bij,bi->bj", Rf, u * s)                              # frame -> world
            n_wrapped += int((off.abs() > L / 2).any(-1).sum().item())
            x[:, j, :] = torch.remainder(tj[None, :] + off, L)
            logq = logq + lq_u - 3 * math.log(s)
        return x, logq, n_wrapped
