"""Hard-walled 3D 80:20 KABLJ cavity point-to-set convergence gate.

The required bulk-reference archive prevents an unequilibrated random start
from being mistaken for a PTS measurement. It must contain ``x[n,N,3]``,
``s[N]`` or ``s[n,N]``, ``L``, ``rho=1.2``, ``T=0.5``,
``composition_B=0.2``, and ``converged=True``.

Run after producing a separately validated 3D bulk/PT reference::

  python -m liquid_coupling_flow.ka_cavity_3d --reference PATH_TO_REFERENCE
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from liquid_coupling_flow.ka_cavity import assert_mobile_inside, cavity_inside
from liquid_coupling_flow.ka_energy import EPS, RCUT_FACTOR, SIGMA, ka_energy, ka_pair_row_scatter

RHO, T, X_B, D = 1.2, 0.5, 0.2, 3


def _pick(mask):
    valid = mask.any(1)
    weights = mask.float()
    weights[~valid, 0] = 1.0
    return torch.multinomial(weights, 1).squeeze(1), valid


def _pair_energy(xi, xj, si, sj, L):
    diff = xi - xj; diff = diff - L * torch.round(diff / L)
    r2 = diff.square().sum(-1).clamp_min(1e-12)
    sig = torch.as_tensor(SIGMA, device=xi.device, dtype=xi.dtype)[si.long(), sj.long()]
    eps = torch.as_tensor(EPS, device=xi.device, dtype=xi.dtype)[si.long(), sj.long()]
    inv6 = (sig.square() / r2).pow(3)
    e = 4 * eps * (inv6.square() - inv6)
    shift = 4 * eps * ((1 / RCUT_FACTOR) ** 12 - (1 / RCUT_FACTOR) ** 6)
    return torch.where(r2 < (RCUT_FACTOR * sig).square(), e - shift, torch.zeros_like(e))


def local_displacement(x, s, U, mobile, beta, L, step, center=None, R=None, stats=None):
    """One exact O(N) single-site move per batch row, with optional hard wall."""
    B = x.shape[0]; rows = torch.arange(B, device=x.device)
    i, valid = _pick(mobile); old = x[rows, i]
    prop = torch.remainder(old + step * torch.randn_like(old), L)
    inside = torch.ones(B, dtype=torch.bool, device=x.device) if R is None else cavity_inside(prop, center, R, L)
    old_e = ka_pair_row_scatter(x, s, (rows, i), old, L)
    new_e = ka_pair_row_scatter(x, s, (rows, i), prop, L)
    acc = valid & inside & (torch.log(torch.rand(B, device=x.device)) < -beta * (new_e - old_e))
    # The caller owns this Markov-chain state. In-place replacement avoids an
    # O(BN) copy for every O(N) local proposal.
    x[rows[acc], i[acc]] = prop[acc]
    if stats is not None:
        stats["mobile_proposals"] += int(valid.sum())
        stats["wall_reject"] += int((valid & ~inside).sum())
    return x, torch.where(acc, U + new_e - old_e, U), acc


def local_identity_swap(x, s, U, mobile, beta, L):
    """Exact A/B identity exchange, using only the two affected pair rows."""
    B = x.shape[0]; rows = torch.arange(B, device=x.device)
    i, ok_a = _pick(mobile & (s == 0)); j, ok_b = _pick(mobile & (s == 1)); valid = ok_a & ok_b
    sp = s.clone(); sp[rows, i], sp[rows, j] = s[rows, j], s[rows, i]
    ri, rj = x[rows, i], x[rows, j]
    old_rows = ka_pair_row_scatter(x, s, (rows, i), ri, L) + ka_pair_row_scatter(x, s, (rows, j), rj, L)
    new_rows = ka_pair_row_scatter(x, sp, (rows, i), ri, L) + ka_pair_row_scatter(x, sp, (rows, j), rj, L)
    # i-j appears in both pair rows; subtract one copy of its change.
    dE = new_rows - old_rows - (_pair_energy(ri, rj, sp[rows, i], sp[rows, j], L) - _pair_energy(ri, rj, s[rows, i], s[rows, j], L))
    acc = valid & (torch.log(torch.rand(B, device=x.device)) < -beta * dE)
    return torch.where(acc[:, None], sp, s), torch.where(acc, U + dE, U), acc


def occupancy(x, L, cell):
    g = max(1, int(round(L / cell)))
    ijk = torch.clamp((torch.remainder(x, L) / L * g).long(), 0, g - 1)
    flat = (ijk[..., 0] * g + ijk[..., 1]) * g + ijk[..., 2]
    out = torch.zeros(x.shape[0], g ** 3, dtype=torch.bool, device=x.device)
    return out.scatter_(1, flat, torch.ones_like(flat, dtype=torch.bool))


def _deep_excluded(center, R, shell, L, cell):
    g = max(1, int(round(L / cell))); a = (torch.arange(g, device=center.device, dtype=center.dtype) + .5) * L / g
    grid = torch.stack(torch.meshgrid(a, a, a, indexing="ij"), -1).reshape(-1, 3)
    delta = grid[None] - center[:, None]; delta = delta - L * torch.round(delta / L)
    return ~(delta.square().sum(-1) < (R - shell) ** 2)


def _q(x, ref_occ, excluded, L, cell):
    now = occupancy(x, L, cell); keep = ~excluded
    return float(((now & ref_occ & keep).sum(1).float() / (ref_occ & keep).sum(1).clamp_min(1).float()).mean())


def _load_reference(path, device):
    d = torch.load(path, map_location=device, weights_only=False)
    need = {"x", "s", "L", "rho", "T", "composition_B", "converged"}
    if missing := need - set(d): raise ValueError(f"reference lacks fields: {sorted(missing)}")
    if not bool(d["converged"]): raise ValueError("reference bulk-equilibration gate did not pass")
    if not (math.isclose(float(d["rho"]), RHO) and math.isclose(float(d["T"]), T) and math.isclose(float(d["composition_B"]), X_B)):
        raise ValueError("reference must be 3D 80:20 KABLJ at rho=1.2, T=0.5")
    x, s, L = d["x"].to(device), d["s"].to(device).long(), float(d["L"])
    if x.ndim != 3 or x.shape[-1] != D: raise ValueError(f"x must be [n,N,3], got {tuple(x.shape)}")
    if not math.isclose(x.shape[1] / L ** 3, RHO, rel_tol=3e-3): raise ValueError("N/L^3 != 1.2")
    if s.ndim == 1: s = s[None].expand(x.shape[0], -1)
    if s.shape != x.shape[:2]: raise ValueError("s must be [N] or [n,N]")
    return x, s, L


def _melt(xref, s, mobile, center, R, L, sweeps, seed):
    torch.manual_seed(seed); x = xref.clone(); U = ka_energy(x, s, L); stats = {"wall_reject": 0, "mobile_proposals": 0}
    nmove = max(1, int(mobile.sum(1).float().mean().round()))
    for _ in range(sweeps * nmove): x, U, _ = local_displacement(x, s, U, mobile, 1 / 1.5, L, .12, center, R, stats)
    assert_mobile_inside(x, mobile, center, R, L)
    return x, stats


def run_gate(reference, out, radii=(1.6, 2.2, 2.8, 3.2), n_centers=8, n_iter=3000, shell=1.0, cell=.3, seed=0, device="cuda"):
    """Run two arms for every radius, averaging over independent bulk configurations and centres."""
    xall, sall, L = _load_reference(reference, device)
    if n_centers > len(xall): raise ValueError(f"need {n_centers} independent references, archive has {len(xall)}")
    if 2 * max(radii) + RCUT_FACTOR > L: raise ValueError("need L >= 2*max(R)+r_cut to isolate periodic cavities")
    torch.manual_seed(seed); take = torch.randperm(len(xall), device=xall.device)[:n_centers]
    xref, s0 = xall[take].clone(), sall[take].clone(); center = torch.rand(n_centers, 3, device=device) * L; qref = occupancy(xref, L, cell)
    rows = []
    for nr, R in enumerate(radii):
        delta = xref - center[:, None]; delta = delta - L * torch.round(delta / L); mobile = delta.square().sum(-1) < R * R
        excluded = _deep_excluded(center, R, shell, L, cell); nmove = max(1, int(mobile.sum(1).float().mean().round()))
        arms = {}
        melt_q = None
        for arm in ("ref", "scramble"):
            stats = {"wall_reject": 0, "mobile_proposals": 0}
            if arm == "ref":
                x = xref.clone()
            else:
                x, melt_stats = _melt(xref, s0, mobile, center, R, L, 400, seed + 100 * nr)
                stats["wall_reject"] += melt_stats["wall_reject"]
                stats["mobile_proposals"] += melt_stats["mobile_proposals"]
                melt_q = _q(x, qref, excluded, L, cell)
            s, U, qs = s0.clone(), ka_energy(x, s0, L), []
            for it in range(n_iter + 1):
                if it % max(1, n_iter // 10) == 0: qs.append(_q(x, qref, excluded, L, cell))
                if it == n_iter: break
                for _ in range(nmove): x, U, _ = local_displacement(x, s, U, mobile, 1 / T, L, .08, center, R, stats)
                for _ in range(8): s, U, _ = local_identity_swap(x, s, U, mobile, 1 / T, L)
            assert_mobile_inside(x, mobile, center, R, L); arms[arm] = {"Qinf": float(np.mean(qs[-3:])), "Q": qs, **stats}
        gap = abs(arms["ref"]["Qinf"] - arms["scramble"]["Qinf"])
        row = {"R": R, "n_mobile_mean": float(mobile.sum(1).float().mean()), "n_centers": n_centers, "melt_Q": melt_q, "ref": arms["ref"], "scramble": arms["scramble"], "gap": gap, "verdict": "CONVERGED" if gap <= .05 else "STALL"}
        rows.append(row); print(f"[3d cavity R={R:.1f}] mobile~{row['n_mobile_mean']:.0f} centers={n_centers} meltQ={row['melt_Q']:.3f} ref={arms['ref']['Qinf']:.3f} scr={arms['scramble']['Qinf']:.3f} gap={gap:.3f} -> {row['verdict']}", flush=True)
    payload = {"model": "3D 80:20 KABLJ", "rho": RHO, "T": T, "N": xref.shape[1], "L": L, "hard_spherical_wall": True, "reference": str(reference), "rows": rows, "created_unix": time.time()}
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    p = argparse.ArgumentParser(); p.add_argument("--reference", required=True, type=Path); p.add_argument("--out", type=Path, default=Path("reports/logs-2026-07-09/pts_cavity_3d_ka.json")); p.add_argument("--radii", nargs="+", type=float, default=[1.6, 2.2, 2.8, 3.2]); p.add_argument("--n-centers", type=int, default=8); p.add_argument("--n-iter", type=int, default=3000); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); run_gate(a.reference, a.out, tuple(a.radii), a.n_centers, a.n_iter, device=a.device)


if __name__ == "__main__": main()
