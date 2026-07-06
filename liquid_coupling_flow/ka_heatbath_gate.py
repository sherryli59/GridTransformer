"""Stage A2 gates (GA2) for the beta-conditioned full-cage single-site heat-bath.

  stationarity  — exactness: from the N=100 reference, apply heat-bath sweeps; <U>/N + g_BB must hold flat
                  (an exact pi-invariant kernel), the MH energy guard preventing the pure-Gibbs collapse.
  accept        — acceptance x mean-jump vs plain displacement sweeps at matched cost (does the learned
                  proposal move further per accepted move than a random step?).
  amortization  — acceptance as a function of beta-rung: roughly flat => one model serves all temperatures
                  (the anti-init-dependence property).
  insmc         — the depth test: local-SMC baseline vs +heat-bath mutation. The Stage-A1 negative said the
                  barrier needs MORE of the cage; A2 sees the full cage -> the real depth candidate.

Run:  python -m liquid_coupling_flow.ka_heatbath_gate {stationarity|accept|amortization|insmc}
"""
from __future__ import annotations
import os, sys, time, torch
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold, ART
from liquid_coupling_flow.ka_heatbath import _load, single_site_mh_sweep
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_observables import partial_gr

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "ka_heatbath_N100.pt"


def _env(B=128, N=100, ref="pt_ladder"):
    sc, L, geo = _scaffold(N, DEV)
    if ref == "pt_ladder":                                           # the trustworthy Stage-0 cold rung (-3.264)
        lad = torch.load(os.path.join(ART, f"pt_ladder_N{N}.pt"), map_location=DEV, weights_only=False)
        x = lad["configs_per_rung"][0][:B].to(DEV); s0 = lad["s"].to(DEV).long()
    else:
        r = torch.load(os.path.join(ART, f"ka_reference_N{N}.pt"), map_location=DEV, weights_only=False)
        x = r["x"][:B].to(DEV); s0 = r["s"].to(DEV).long()
    pos, s = slot_order(x, s0, geo, N)
    HB = _load(torch.load(os.path.join(ART, CKPT), map_location=DEV, weights_only=False), DEV)
    return sc, L, geo, pos, s, HB, N


def _gbb_peak(pos, s, L):
    _, g = partial_gr(pos, s, L, rmax=3.0, nbins=60, pair=(1, 1))
    return float(max(g))


def stationarity(T=60, B=128, beta=2.0):
    sc, L, geo, pos, s, HB, N = _env(B=B)
    gen = torch.Generator(device=DEV).manual_seed(0)
    u0 = (ka_energy(pos, s, L) / N).mean().item(); g0 = _gbb_peak(pos, s, L)
    print(f"[heatbath stationarity] start: <U>/N {u0:.4f}  g_BB {g0:.3f}", flush=True)
    hist = [(0, u0, g0)]; t0 = time.time()
    for t in range(1, T + 1):
        pos, info = single_site_mh_sweep(HB, pos, s, sc, L, geo, beta=beta, gen=gen)
        if t % 10 == 0 or t == 1:
            u = (ka_energy(pos, s, L) / N).mean().item(); g = _gbb_peak(pos, s, L)
            hist.append((t, u, g))
            print(f"  sweep {t:3d}: <U>/N {u:.4f}  g_BB {g:.3f}  accept {info['accept']:.3f}  "
                  f"({(time.time()-t0)/t:.1f}s/sweep)", flush=True)
    us = [u for _, u, _ in hist[1:]]; band = max(us) - min(us); drift = us[-1] - u0
    verdict = "STATIONARY" if abs(drift) < 0.03 and band < 0.05 else "DRIFTS"
    print(f"[heatbath stationarity] band {band:.4f} drift {drift:+.4f} -> {verdict}", flush=True)
    torch.save({"hist": hist, "band": band, "drift": drift}, os.path.join(ART, "heatbath_stationarity_N100.pt"))


def accept(T=10, B=128, beta=2.0):
    """Acceptance x mean-jump vs plain displacement (matched sweeps). Jump = mean |x_new - x_old| of the site."""
    from liquid_coupling_flow.ka_local_smc import _disp_sweeps
    sc, L, geo, pos, s, HB, N = _env(B=B)
    gen = torch.Generator(device=DEV).manual_seed(0)
    p0 = pos.clone(); t0 = time.time(); accs = []
    for _ in range(T):
        p0, info = single_site_mh_sweep(HB, p0, s, sc, L, geo, beta=beta, gen=gen); accs.append(info["accept"])
    th = (time.time() - t0) / T; uh = (ka_energy(p0, s, L) / N).mean().item()
    pd = pos.clone(); t0 = time.time()
    for _ in range(T):
        pd = _disp_sweeps(pd, s[0], L, kT=1.0 / beta, n=1)
    td = (time.time() - t0) / T; ud = (ka_energy(pd, s, L) / N).mean().item()
    print(f"GA2 accept (beta={beta}, {T} sweeps): heatbath accept {sum(accs)/len(accs):.3f} <U>/N {uh:.4f} "
          f"{th:.1f}s/sweep | displacement <U>/N {ud:.4f} {td:.2f}s/sweep", flush=True)
    torch.save({"hb_accept": sum(accs)/len(accs), "hb_U": uh, "disp_U": ud, "th": th, "td": td},
               os.path.join(ART, "heatbath_accept_N100.pt"))


def amortization(B=128, T=4):
    """Acceptance per beta-rung (from the ladder configs): flat => one model serves all rungs."""
    sc, L, geo = _scaffold(100, DEV)
    lad = torch.load(os.path.join(ART, "pt_ladder_N100.pt"), map_location=DEV, weights_only=False)
    HB = _load(torch.load(os.path.join(ART, CKPT), map_location=DEV, weights_only=False), DEV)
    s0 = lad["s"].to(DEV).long(); betas = lad["betas"]; gen = torch.Generator(device=DEV).manual_seed(0)
    rows = []
    for l, b in enumerate(betas):
        x = lad["configs_per_rung"][l][:B].to(DEV); pos, s = slot_order(x, s0, geo, 100)
        for _ in range(T):
            pos, info = single_site_mh_sweep(HB, pos, s, sc, L, geo, beta=b, gen=gen)
        rows.append((b, info["accept"])); print(f"  beta {b:.2f}: accept {info['accept']:.3f}", flush=True)
    accs = [a for _, a in rows]
    print(f"GA2 amortization: accept range {min(accs):.3f}-{max(accs):.3f} over rungs -> "
          f"{'FLAT (amortizes)' if max(accs)-min(accs) < 0.15 else 'rung-dependent'}", flush=True)
    torch.save({"rows": rows}, os.path.join(ART, "heatbath_amortization_N100.pt"))


def insmc(n_heatbath=1, B=256, max_rungs=40, n_mut=2, n_disp=40, seed=0):
    from liquid_coupling_flow.ka_local_smc import smc_run
    print(f"=== in-SMC depth: baseline vs +heatbath (n_heatbath={n_heatbath}), B={B} ===", flush=True)
    base = smc_run("arm0", N=100, B=B, max_rungs=max_rungs, n_mut=n_mut, n_disp=n_disp, seed=seed, n_heatbath=0)
    ub = base["history"][-1]["U_mean"]; wb = base["wall"]
    print(f"BASELINE   final U/N {ub:.4f}  {wb:.0f}s", flush=True)
    aug = smc_run("arm0", N=100, B=B, max_rungs=max_rungs, n_mut=n_mut, n_disp=n_disp, seed=seed, n_heatbath=n_heatbath)
    ua = aug["history"][-1]["U_mean"]; wa = aug["wall"]
    print(f"+heatbath  final U/N {ua:.4f}  {wa:.0f}s", flush=True)
    print(f"VERDICT: depth delta {ua-ub:+.4f} (neg=deeper) at {wa/max(wb,1e-9):.2f}x wall", flush=True)
    torch.save({"baseline": base["history"], "heatbath": aug["history"], "U_base": ub, "U_hb": ua,
                "wall_base": wb, "wall_hb": wa}, os.path.join(ART, "heatbath_insmc_depth.pt"))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stationarity"
    {"stationarity": stationarity, "accept": accept, "amortization": amortization, "insmc": insmc}[cmd]()
