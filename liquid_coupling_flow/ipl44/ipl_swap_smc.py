"""Swap-MH kernel + position Metropolis + sweeps-to-equilibrium benchmark for IPL44 (spec
docs/superpowers/specs/2026-07-02-ipl44-lever-push-design.md). The learned swap proposer uses PAIRED
RE-EVALUATION: forward weights from weight_fn at the current state, reverse weights from weight_fn at the
swapped state -> textbook-exact MH with a state-dependent proposal. Random proposer = constant weights
(ratio contribution exactly 0 in log space)."""
from __future__ import annotations
import torch

EPS = 1e-12


def uniform_weight_fn(x, s):
    return torch.full_like(s, 0.5, dtype=x.dtype)


def _pick(weights, mask):
    """Sample one index per row from weights restricted to mask; returns idx [B]."""
    w = weights * mask + EPS * mask
    return torch.multinomial(w, 1).squeeze(1)


def _sel_prob(weights, mask, idx):
    """Probability that a masked multinomial over `weights` would select `idx`."""
    w = weights * mask + EPS * mask
    return w[torch.arange(w.shape[0], device=w.device), idx] / w.sum(1)


def _swap_log_ratio(pB_f, pB_r, s, s_new, i, j):
    """log g'(reverse)/g(forward) for swapping A-particle i with B-particle j.
    Forward at state s: pick i among A's with weight pB_f, j among B's with weight (1-pB_f).
    Reverse at state s_new: pick j among A's with weight pB_r, i among B's with weight (1-pB_r)."""
    dt = pB_f.dtype
    isA, isB = (s == 0).to(dt), (s == 1).to(dt)
    isA_n, isB_n = (s_new == 0).to(dt), (s_new == 1).to(dt)
    g_i = _sel_prob(pB_f, isA, i)
    g_j = _sel_prob(1.0 - pB_f, isB, j)
    gp_j = _sel_prob(pB_r, isA_n, j)
    gp_i = _sel_prob(1.0 - pB_r, isB_n, i)
    return (gp_j + EPS).log() + (gp_i + EPS).log() - (g_i + EPS).log() - (g_j + EPS).log()


def swap_attempt(x, s, U, beta, energy_fn, weight_fn):
    """One batched MH swap attempt: swap one A with one B per config. Returns (s_new, U_new, accepted [B])."""
    B = x.shape[0]; ar = torch.arange(B, device=x.device)
    pB_f = weight_fn(x, s)
    dt = pB_f.dtype
    isA, isB = (s == 0).to(dt), (s == 1).to(dt)
    i = _pick(pB_f, isA)                     # A-particle that "wants to be B"
    j = _pick(1.0 - pB_f, isB)               # B-particle that "wants to be A"
    s_prop = s.clone(); s_prop[ar, i] = 1; s_prop[ar, j] = 0
    U_prop = energy_fn(x, s_prop)
    pB_r = weight_fn(x, s_prop)               # paired re-evaluation at the swapped state -> exact reverse
    log_ratio = -beta * (U_prop - U) + _swap_log_ratio(pB_f, pB_r, s, s_prop, i, j)
    acc = torch.log(torch.rand(B, device=x.device)) < log_ratio
    s_new = torch.where(acc[:, None], s_prop, s)
    U_new = torch.where(acc, U_prop, U)
    return s_new, U_new, acc


def position_sweep(x, s, U, beta, L, step, energy_fn):
    """One sweep = N sequential batched single-particle Metropolis moves. Returns (x, U, acc_rate)."""
    B, N, _ = x.shape; n_acc = 0.0
    for i in torch.randperm(N).tolist():
        xp = x.clone()
        xp[:, i] = torch.remainder(xp[:, i] + step * torch.randn(B, 2, device=x.device), L)
        Up = energy_fn(xp, s)
        acc = torch.log(torch.rand(B, device=x.device)) < (-beta * (Up - U))
        x = torch.where(acc[:, None, None], xp, x); U = torch.where(acc, Up, U)
        n_acc += acc.float().mean().item()
    return x, U, n_acc / N


def run_chain(x0, s0, n_sweeps, beta, L, energy_fn, weight_fn=None, n_swap=8, step=0.08, record_every=10):
    """Alternate position sweeps and swap attempts; record batch-ensemble metrics every record_every sweeps."""
    from liquid_coupling_flow.ipl44.ipl_energy import ipl_gr_partials
    x, s = x0.clone(), s0.clone(); U = energy_fn(x, s)
    rec = {"sweep": [], "U_median": [], "gbb_peak": [], "swap_acc": [], "pos_acc": []}
    for k in range(n_sweeps + 1):
        if k % record_every == 0:
            _, _, _, gbb = ipl_gr_partials(x.cpu(), s.cpu(), L)
            rec["sweep"].append(k); rec["U_median"].append(float(U.median()))
            rec["gbb_peak"].append(float(gbb.max()))
        if k == n_sweeps:
            break
        x, U, pacc = position_sweep(x, s, U, beta, L, step, energy_fn)
        sacc = 0.0
        if weight_fn is not None:
            for _ in range(n_swap):
                s, U, a = swap_attempt(x, s, U, beta, energy_fn, weight_fn)
                sacc += a.float().mean().item()
            sacc /= n_swap
        if (k + 1) % record_every == 0 or k == 0:
            rec["swap_acc"].append(sacc); rec["pos_acc"].append(pacc)
    rec["x"], rec["s"] = x, s
    return rec


def sweeps_to_band(curves, U_ref_med, gbb_ref_peak, u_tol=0.05, g_tol=0.15):
    """First recorded sweep where BOTH |U_med-ref|/|ref| < u_tol and |gbb-ref|/ref < g_tol; None if never."""
    for k, u, g in zip(curves["sweep"], curves["U_median"], curves["gbb_peak"]):
        if abs(u - U_ref_med) / abs(U_ref_med) < u_tol and abs(g - gbb_ref_peak) / gbb_ref_peak < g_tol:
            return k
    return None
