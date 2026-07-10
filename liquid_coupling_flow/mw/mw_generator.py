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

Task 9 appends the transformer + spline model that consumes this geometry.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


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
