"""Alchemical (x, lambda) species kernel — PATHWISE species exchange via NCMC (spec
2026-07-07-teleport-and-alchemical-design.md).

Plain A<->B swaps are endpoint moves and are measured DEAD (0/102,400) — the sigma_AB mismatch cannot be paid
in one step. This kernel pays it in increments: each particle carries lambda_i in [0,1]; pair parameters are
BILINEAR in (lambda_i, lambda_j) over the KA 2x2 tables (handles the non-additive sigma_AB=0.8 exactly at the
corners); an NCMC protocol drives an unlike pair (1,0)->(0,1) through T interpolation steps, each followed by
n_relax Metropolis displacement steps at the intermediate Hamiltonian (the cage BREATHES DURING the exchange —
the pathwise advantage over swap-and-breathe's one-shot resample). Protocol work W accumulates ONLY the
lambda-increment energy jumps (DB propagators cancel, Crooks); accept the whole trajectory with min(1, e^{-beta W}),
restoring the ORIGINAL state on rejection. Pair selection is unordered+uniform (symmetric); the reverse protocol
is the same move on the same pair => exact MH. Composition exactly preserved (antisymmetric drive); endpoints
exactly binary => no reweighting.

Flow-drift upgrade (behind this gate): replace/augment the relaxation drift with the EGNN flow — density-free
role; NCMC needs only work bookkeeping.
"""
from __future__ import annotations
import os, sys, time, torch
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR

ART = os.path.join(os.path.dirname(__file__), "artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_S = torch.tensor(SIGMA)   # [2,2] indices 0=A,1=B
_E = torch.tensor(EPS)


def _pair_tables(lam):
    """Per-config bilinear pair parameters from per-particle lambda. lam [B,N] -> sigma,eps [B,N,N].
    Bilinear over the 2x2 KA tables: exact at the corners incl the non-additive sigma_AB."""
    li = lam[:, :, None]; lj = lam[:, None, :]
    wAA = (1 - li) * (1 - lj); wAB = li * (1 - lj) + lj * (1 - li); wBB = li * lj
    S = _S.to(lam.device); E = _E.to(lam.device)
    sig = wAA * S[0, 0] + wAB * S[0, 1] + wBB * S[1, 1]
    eps = wAA * E[0, 0] + wAB * E[0, 1] + wBB * E[1, 1]
    return sig, eps


def alch_energy(x, lam, L, per_particle=False):
    """Shifted/cutoff/min-image LJ with lambda-interpolated pair parameters. x [B,N,2], lam [B,N].
    At lam == species it equals ka_energy exactly (same shift convention, rc = 2.5*sigma(lam))."""
    B, N, _ = x.shape
    sig, eps = _pair_tables(lam)
    rc = RCUT_FACTOR * sig
    diff = x[:, :, None, :] - x[:, None, :, :]
    diff = diff - L * torch.round(diff / L)
    r2 = (diff ** 2).sum(-1)
    eye = torch.eye(N, device=x.device, dtype=torch.bool)
    r2 = r2.masked_fill(eye, 1e12)
    inv6 = (sig ** 2 / r2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    src6 = (sig / rc) ** 6
    e = torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e))
    if per_particle:
        return e.sum(-1)
    return 0.5 * e.sum(dim=(1, 2))


def _alch_disp_step(x, lam, L, beta, step=0.04, gen=None):
    """One parallel-Metropolis displacement step under the interpolated Hamiltonian (exact per-particle rows)."""
    B, N, _ = x.shape
    prop = torch.remainder(x + step * torch.randn(x.shape, device=x.device, generator=gen), L)
    sig, eps = _pair_tables(lam)
    rc = RCUT_FACTOR * sig
    def rows(y_i, y_all):
        d = y_i[:, :, None, :] - y_all[:, None, :, :]
        d = d - L * torch.round(d / L)
        r2 = (d ** 2).sum(-1)
        eye = torch.eye(N, device=x.device, dtype=torch.bool)
        r2 = r2.masked_fill(eye, 1e12)
        inv6 = (sig ** 2 / r2) ** 3
        e = 4 * eps * (inv6 ** 2 - inv6)
        src6 = (sig / rc) ** 6
        return torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e)).sum(-1)
    dE = rows(prop, x) - rows(x, x)                                  # [B,N] one-mover row deltas
    acc = torch.log(torch.rand(B, N, device=x.device, generator=gen)) < (-beta * dE)
    return torch.where(acc[:, :, None], prop, x)


@torch.no_grad()
def ncmc_swap(x, lam, i, j, L, beta=2.0, T_steps=8, n_relax=2, step=0.04, gen=None):
    """NCMC alchemical exchange of the unlike pair (i, j): drive (lam_i, lam_j) antisymmetrically to the
    swapped endpoints over T_steps, with n_relax displacement steps at each intermediate lambda. W accumulates
    the lambda-increment jumps only. Accept whole trajectory w.p. min(1, e^{-beta W}); reject => restore.
    Returns (x, lam, info) with info['work'] [B]."""
    B = x.shape[0]
    x0, lam0 = x.clone(), lam.clone()
    li0, lj0 = lam[:, i].clone(), lam[:, j].clone()
    W = torch.zeros(B, device=x.device)
    xc, lc = x.clone(), lam.clone()
    for k in range(1, T_steps + 1):
        t = k / T_steps
        l_new = lc.clone()
        l_new[:, i] = (1 - t) * li0 + t * lj0
        l_new[:, j] = (1 - t) * lj0 + t * li0
        W = W + alch_energy(xc, l_new, L) - alch_energy(xc, lc, L)   # perturbation work
        lc = l_new
        for _ in range(n_relax):
            xc = _alch_disp_step(xc, lc, L, beta, step=step, gen=gen)
    u = torch.rand(B, device=x.device, generator=gen)
    accept = torch.log(u) < -beta * W
    x_out = torch.where(accept[:, None, None], xc, x0)
    lam_out = torch.where(accept[:, None], lc, lam0)
    return x_out, lam_out, {"work": W, "accept": accept.float().mean().item(), "accept_mask": accept}


def gate(T_list=(1, 4, 16, 64), n_moves=200, B=128, beta=2.0):
    """THE GATE: accepted exchanges per GPU-second vs path length T, at beta=2 equilibrium. T=1 ~ plain-swap
    (dead) sanity anchor; the question is whether some T buys exchange cheaper than swap-and-breathe v2."""
    from liquid_coupling_flow.ka_cluster_flow import _scaffold
    lad = torch.load(os.path.join(ART, "pt_ladder_N100.pt"), map_location=DEV, weights_only=False)
    x0 = lad["configs_per_rung"][0][:B].to(DEV)
    s = lad["s"].to(DEV).long()
    gen = torch.Generator(device=DEV).manual_seed(0)
    A = (s == 0).nonzero().squeeze(-1); Bidx = (s == 1).nonzero().squeeze(-1)
    out = {}
    for T in T_list:
        x = x0.clone(); lam = s.float()[None].expand(B, -1).clone()
        accs, t0 = [], time.time()
        for m in range(n_moves):
            i = int(A[torch.randint(0, len(A), (1,), device=DEV, generator=gen)])
            j = int(Bidx[torch.randint(0, len(Bidx), (1,), device=DEV, generator=gen)])
            x, lam, info = ncmc_swap(x, lam, i, j, L=(100 / 1.2) ** 0.5, beta=beta, T_steps=T, n_relax=2, gen=gen)
            accs.append(info["accept"])
        wall = time.time() - t0
        rate = sum(accs) / len(accs)
        out[T] = {"accept": rate, "wall": wall, "acc_per_s": rate * n_moves * B / wall}
        print(f"NCMC T={T:3d}: accept {rate:.5f}  {wall:.0f}s/{n_moves} moves  "
              f"accepted-exchanges/s {out[T]['acc_per_s']:.2f}", flush=True)
    torch.save(out, os.path.join(ART, "alchemical_gate_N100.pt"))
    print("compare: swap-and-breathe v2 = 1.9% per ~70s/exchange-chain [[swap-breathe-kernel]]", flush=True)


if __name__ == "__main__":
    gate()
