"""G2 exactness gate: SMC final population vs independent reference MC at the ambient point
(N=8,64 at beta=1/T*, rho=rho*). Three comparisons, SMC side WEIGHTED by w=softmax(logw) (the
annealer never resamples-to-uniform at the last rung, so the population must be treated as
weighted, not as an equal-weight sample):
  1. <U>/N   : weighted SMC mean vs reference mean.
               SE_smc via weighted bootstrap (200 resamples: draw B indices ~ w, plain mean of
               the resample). SE_ref via independent-chain time-means (the G1 pattern: chains
               never interact -> B chain-means are iid). PASS |diff| < 3*(SE_smc+SE_ref).
  2. P(U/N)  : total variation on bins spanning the UNION of both samples' ranges (shared support
               so a systematic shift can't hide in disjoint binning); SMC histogram weighted by w,
               reference unweighted. PASS TV < 0.05.
  3. g(r)    : reference g(r) vs g(r) of the SMC population RESAMPLED by weight (multinomial B
               draws ~ w) — g_r itself takes no weights, so the weighting has to happen upstream
               of it. PASS max|dg| < 0.1 (g_r's default rmax=L/2 already restricts to r < L/2).

This module is reused by Phase-2/3 re-gates (same g2() at later checkpoints of the base).

Run: python -m liquid_coupling_flow.mw.mw_gates g2 <N> [B]
"""
from __future__ import annotations
import math, os, sys, torch
from liquid_coupling_flow.mw.mw_base import UniformBase
from liquid_coupling_flow.mw.mw_smc import smc_run
from liquid_coupling_flow.mw.mw_reference import mc_run, g_r
from liquid_coupling_flow.mw.mw_energy import T_STAR, RHO_STAR

ART = os.path.join(os.path.dirname(__file__), "artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

REF_B = 16                                  # independent MC chains (fixed, matches the G1 pattern)
REF_BUDGETS = {8: (20000, 20000), 64: (40000, 8000)}   # N -> (n_equil, n_collect)


def _weighted_bootstrap_se(u_smc, w, n_boot=200, seed=0):
    """SE of the weighted SMC mean via weighted bootstrap: each of n_boot resamples draws B
    indices with replacement, probability w=softmax(logw), then takes the PLAIN mean of u over
    that resample — the weighting is already baked into the resampling distribution, not into a
    second weighted average."""
    device = u_smc.device
    gen = torch.Generator(device=device).manual_seed(seed)
    B = u_smc.shape[0]
    w_ = w.to(u_smc.dtype)
    boot = torch.empty(n_boot, dtype=torch.float64)
    for b in range(n_boot):
        idx = torch.multinomial(w_, B, replacement=True, generator=gen)
        boot[b] = u_smc[idx].double().mean()
    return float(boot.std())


def _tv_shared_bins(u_smc, w, u_ref, nbins=40):
    """Total variation between the SMC-weighted and reference-unweighted P(U/N), on bins spanning
    the union of both samples' ranges. torch.histogram is CPU-only -> move there explicitly."""
    u_smc_c, w_c, u_ref_c = u_smc.double().cpu(), w.double().cpu(), u_ref.double().cpu()
    lo = float(min(u_smc_c.min(), u_ref_c.min()))
    hi = float(max(u_smc_c.max(), u_ref_c.max()))
    if hi <= lo:
        hi = lo + 1e-6
    counts_smc, edges = torch.histogram(u_smc_c, bins=nbins, range=(lo, hi), weight=w_c)
    counts_ref, _ = torch.histogram(u_ref_c, bins=nbins, range=(lo, hi))
    p_smc = counts_smc / counts_smc.sum().clamp_min(1e-300)
    p_ref = counts_ref / counts_ref.sum().clamp_min(1e-300)
    tv = 0.5 * float((p_smc - p_ref).abs().sum())
    return tv, edges, p_smc, p_ref


def g2(N, B=512, n_ref_equil=None, n_ref_collect=None, save_tag="_g2", n_sweeps=10,
       final_sweeps=300):
    """G2 gate at size N: independent displacement-MC reference vs annealed-SMC final population,
    both at the ambient point (beta=1/T*, L from rho*). Saves full populations (both arms) + all
    metrics to artifacts/mw_g2_N{N}.pt (repo rule: full data, never summaries-only) and returns
    that same dict.

    Mutation budget (the first real N=8 run FAILED all three metrics on mutation-limited depth):
    the SMC arm's MH step is the REFERENCE's frozen adapted step (spec section 2 — the lam=1-tuned
    step; letting it default to 0.15 vs the adapted 0.0666 at beta*=10.38 was a real mismatch),
    n_sweeps=10 per rung, plus a final_sweeps=300 lam=1 finisher (pi_1-invariant, weight path
    untouched -> exactness preserved).

    Reference cache: if the CANONICAL artifact mw_g2_N{N}.pt exists and its saved ref budgets
    exactly match the requested ones, its ref dict is reused instead of re-running mc_run (the
    N=64 reference costs ~2h; gate retries must not pay it again). Exact budget match only."""
    if n_ref_equil is None or n_ref_collect is None:
        if N not in REF_BUDGETS:
            raise ValueError(f"g2: no default reference budget for N={N}; "
                              f"pass n_ref_equil/n_ref_collect explicitly")
        d_equil, d_collect = REF_BUDGETS[N]
        n_ref_equil = d_equil if n_ref_equil is None else n_ref_equil
        n_ref_collect = d_collect if n_ref_collect is None else n_ref_collect

    beta = 1.0 / T_STAR
    L = (N / RHO_STAR) ** (1.0 / 3.0)

    print(f"G2 N={N}: L={L:.4f} beta={beta:.4f} ref(equil={n_ref_equil},collect={n_ref_collect},"
          f"every=4,B={REF_B}) smc(B={B},n_sweeps={n_sweeps},final_sweeps={final_sweeps})",
          flush=True)

    # reference cache: exact budget match against the CANONICAL artifact only (keep it simple);
    # artifacts written before the cache existed lack "ref_budget" and correctly fall through.
    ref = None
    canon_path = os.path.join(ART, f"mw_g2_N{N}.pt")
    if os.path.exists(canon_path):
        try:
            prev = torch.load(canon_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"G2 N={N}: ref cache unreadable ({e}); running mc_run fresh", flush=True)
            prev = None
        if prev is not None and prev.get("ref_budget") == (n_ref_equil, n_ref_collect):
            ref = prev["ref"]
            print(f"G2 N={N}: ref CACHED from artifact ({canon_path})", flush=True)
    if ref is None:
        ref = mc_run(N, L, beta, n_ref_equil, n_ref_collect, every=4, seed=0, B=REF_B)

    print(f"G2 N={N}: smc step = ref frozen adapted step {ref['step']:.4f}", flush=True)
    out = smc_run(UniformBase(N, L), N, L, beta, B=B, n_sweeps=n_sweeps, step=ref["step"],
                  seed=0, save_tag=save_tag, final_sweeps=final_sweeps)

    w = torch.softmax(out["logw"].double(), 0)      # double precision softmax (ess()'s reasoning)

    # --- 1: <U>/N ---
    u_smc = out["U"] / N
    u_ref = ref["U"] / N
    mean_smc = float((w * u_smc.double()).sum())
    se_smc = _weighted_bootstrap_se(u_smc, w, n_boot=200, seed=0)
    chain_means = u_ref.reshape(-1, REF_B).mean(0)    # G1 pattern: independent-chain time-means
    mean_ref = float(chain_means.mean())
    se_ref = float(chain_means.std() / math.sqrt(REF_B))
    diff1 = abs(mean_smc - mean_ref)
    tol1 = 3.0 * (se_smc + se_ref)
    pass1 = diff1 < tol1
    print(f"G2 N={N} <U>/N: SMC {mean_smc:.4f}+/-{se_smc:.4f} vs ref {mean_ref:.4f}+/-{se_ref:.4f} "
          f"| |diff| {diff1:.4f} < tol {tol1:.4f} -> {'PASS' if pass1 else 'FAIL'}", flush=True)

    # --- 2: P(U/N) TV ---
    tv, edges, p_smc, p_ref = _tv_shared_bins(u_smc, w, u_ref, nbins=40)
    pass2 = tv < 0.05
    print(f"G2 N={N} P(U/N) TV: {tv:.4f} < 0.05 -> {'PASS' if pass2 else 'FAIL'}", flush=True)

    # --- 3: g(r) ---
    gen = torch.Generator(device=out["x"].device).manual_seed(0)
    idx = torch.multinomial(w.to(torch.float64), out["x"].shape[0], replacement=True, generator=gen)
    x_resampled = out["x"][idx]
    r_ref, g_ref_arr = g_r(ref["cfgs"], L)
    r_smc, g_smc_arr = g_r(x_resampled.cpu(), L)
    max_dg = float((g_smc_arr - g_ref_arr).abs().max())
    pass3 = max_dg < 0.1
    print(f"G2 N={N} g(r): max|dg| {max_dg:.4f} < 0.10 -> {'PASS' if pass3 else 'FAIL'}", flush=True)

    overall = pass1 and pass2 and pass3
    print(f"G2 N={N} OVERALL -> {'PASS' if overall else 'FAIL'}", flush=True)

    result = {
        "N": N, "L": L, "beta": beta,
        "ref_budget": (n_ref_equil, n_ref_collect),        # cache key for g2 retries
        "ref": {"cfgs": ref["cfgs"], "U": ref["U"], "acc": ref["acc"], "step": ref["step"],
                "flat_budget": ref["flat_budget"], "coll_drift": ref["coll_drift"], "traj": ref["traj"]},
        "smc": {"x": out["x"], "U": out["U"], "logw": out["logw"], "logZ": out["logZ"],
                "history": out["history"], "evals": out["evals"], "wall": out["wall"],
                "B": B, "n_sweeps": n_sweeps, "final_sweeps": final_sweeps, "step": ref["step"]},
        "mean_U_per_N": {"smc": mean_smc, "se_smc": se_smc, "ref": mean_ref, "se_ref": se_ref,
                          "diff": diff1, "tol": tol1, "pass": pass1},
        "tv_U_per_N": {"tv": tv, "nbins": 40, "edges": edges, "p_smc": p_smc, "p_ref": p_ref,
                        "pass": pass2},
        "g_r": {"r": r_ref, "g_ref": g_ref_arr, "g_smc": g_smc_arr, "max_dg": max_dg, "pass": pass3},
        "overall_pass": overall,
    }
    os.makedirs(ART, exist_ok=True)
    # canonical filename mw_g2_N{N}.pt for the default save_tag; any OTHER tag (e.g. the smoke
    # test's "_g2smoke") gets folded in so a fast/tiny run can never silently clobber a real,
    # hours-long gate result at the same N.
    tag_suffix = "" if save_tag == "_g2" else save_tag
    path = os.path.join(ART, f"mw_g2{tag_suffix}_N{N}.pt")
    torch.save(result, path)
    print(f"G2 N={N}: saved -> {path}", flush=True)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "g2":
        print("usage: python -m liquid_coupling_flow.mw.mw_gates g2 <N> [B]", file=sys.stderr)
        sys.exit(1)
    N_arg = int(sys.argv[2])
    B_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    g2(N_arg, B=B_arg)
