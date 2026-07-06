"""Stage A1 gates for the cSMC cluster move (ka_cluster_csmc.csmc_cluster_move).

GATES:
  stationarity  — THE exactness gate: start from the N=100 PT reference (equilibrium) and apply cSMC cluster
                  sweeps. An exactly pi-invariant kernel HOLDS <U>/N and g_BB flat; a non-invariant/diffuse
                  kernel drifts up (the pure-Gibbs-collapse signature). Run for the exact core (resample off)
                  and the efficiency variants (resample/PGAS) to empirically gate the add-ons.
  ga1           — utility vs the validated MTM cluster mover at beta=2: acceptance (1-ref_survival) and
                  cluster rearrangement per wall-clock. Same ARM-FULL proposal, apples-to-apples.

Run:  python -m liquid_coupling_flow.ka_cluster_csmc_gate stationarity [M] [resample] [ancestor]
      python -m liquid_coupling_flow.ka_cluster_csmc_gate ga1 [M]
"""
from __future__ import annotations
import os, sys, time, torch
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold, _load, ART
from liquid_coupling_flow.ka_cluster_csmc import csmc_cluster_move, csmc_sweep
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_observables import partial_gr

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _env(B=128, N=100):
    sc, L, geo = _scaffold(N, DEV)
    ref = torch.load(os.path.join(ART, f"ka_reference_N{N}.pt"), map_location=DEV, weights_only=False)
    x = ref["x"][:B].to(DEV); s0 = ref["s"].to(DEV).long()
    pos, s = slot_order(x, s0, geo, N)
    P = _load(torch.load(os.path.join(ART, "ka_cluster_flow_full_N100.pt"), map_location=DEV,
                         weights_only=False), DEV)
    return sc, L, geo, pos, s, P, N


def _gbb_peak(pos, s, L):
    _, g = partial_gr(pos, s, L, rmax=3.0, nbins=60, pair=(1, 1))
    return float(max(g))


def stationarity(M=16, resample=False, ancestor=False, T=60, B=128, beta=2.0):
    sc, L, geo, pos, s, P, N = _env(B=B)
    gen = torch.Generator(device=DEV).manual_seed(0)
    u0 = (ka_energy(pos, s, L) / N).mean().item(); g0 = _gbb_peak(pos, s, L)
    tag = f"cSMC M={M} resample={resample} ancestor={ancestor}"
    print(f"[{tag}] start (reference): <U>/N {u0:.4f}  g_BB {g0:.3f}", flush=True)
    hist = [(0, u0, g0)]
    t0 = time.time()
    for t in range(1, T + 1):
        pos, s, info = csmc_sweep(P, pos, s, sc, L, geo, M=M, beta=beta,
                                  resample=resample, ancestor=ancestor, gen=gen)
        if t % 10 == 0 or t == 1:
            u = (ka_energy(pos, s, L) / N).mean().item(); g = _gbb_peak(pos, s, L)
            hist.append((t, u, g))
            print(f"  sweep {t:3d}: <U>/N {u:.4f}  g_BB {g:.3f}  accept {info['accept']:.3f}  "
                  f"({(time.time()-t0)/t:.1f}s/sweep)", flush=True)
    us = [u for _, u, _ in hist[1:]]
    band = max(us) - min(us)
    drift = us[-1] - u0
    verdict = "STATIONARY" if abs(drift) < 0.03 and band < 0.05 else "DRIFTS (non-invariant?)"
    print(f"[{tag}] band {band:.4f}  net-drift {drift:+.4f}  -> {verdict}", flush=True)
    torch.save({"hist": hist, "M": M, "resample": resample, "ancestor": ancestor, "u0": u0, "g0": g0,
                "band": band, "drift": drift}, os.path.join(ART, f"csmc_stationarity_M{M}_r{int(resample)}_a{int(ancestor)}.pt"))
    return hist


def ga1(M=16, T=10, B=128, beta=2.0):
    """Utility vs MTM at beta=2 from reference: acceptance + wall-clock. Both around ARM-FULL."""
    from liquid_coupling_flow.ka_cluster_mtm import mtm_sweep
    sc, L, geo, pos, s, P, N = _env(B=B)
    gen = torch.Generator(device=DEV).manual_seed(0)
    # cSMC
    pc = pos.clone(); t0 = time.time()
    for _ in range(T):
        pc, _, ic = csmc_sweep(P, pc, s, sc, L, geo, M=M, beta=beta, gen=gen)
    tc = (time.time() - t0) / T
    uc = (ka_energy(pc, s, L) / N).mean().item()
    # MTM (mtm_sweep uses torch.randperm on CPU -> pass gen=None; utility comparison needs no fixed MTM seed)
    pm = pos.clone(); t0 = time.time()
    for _ in range(T):
        pm, sm, im = mtm_sweep(P, pm, s, sc, L, M=M, beta=beta, k=7, gen=None)
    tm = (time.time() - t0) / T
    um = (ka_energy(pm, s, L) / N).mean().item()
    print(f"GA1 (M={M}, {T} sweeps from reference, beta={beta}):", flush=True)
    print(f"  cSMC : accept {ic['accept']:.3f}  <U>/N {uc:.4f}  {tc:.1f}s/sweep", flush=True)
    print(f"  MTM  : accept {im.get('acceptance', float('nan')):.3f}  <U>/N {um:.4f}  {tm:.1f}s/sweep", flush=True)
    torch.save({"csmc": {"accept": ic["accept"], "U": uc, "t": tc},
                "mtm": {"accept": im.get("acceptance"), "U": um, "t": tm}, "M": M},
               os.path.join(ART, f"csmc_ga1_M{M}.pt"))


def insmc_depth(n_csmc=1, csmc_M=16, B=256, max_rungs=40, n_mut=2, n_disp=40, seed=0):
    """GA1 in-SMC depth gate (primary success criterion): does adding a cSMC mutation to the local-SMC stack
    reach deeper <U>/N than the ARM-0 baseline? Runs baseline (n_csmc=0) then +cSMC at the SAME schedule;
    reports final depth + wall-clock (the matched-cost read). Saves both trajectories."""
    from liquid_coupling_flow.ka_local_smc import smc_run
    print(f"=== in-SMC depth: baseline (n_csmc=0) vs +cSMC (n_csmc={n_csmc}, M={csmc_M}), B={B} ===", flush=True)
    base = smc_run("arm0", N=100, B=B, max_rungs=max_rungs, n_mut=n_mut, n_disp=n_disp, seed=seed, n_csmc=0)
    ub = base["history"][-1]["U_mean"]; wb = base["wall"]
    print(f"BASELINE   final U/N {ub:.4f}  {wb:.0f}s  rungs {len(base['history'])}", flush=True)
    aug = smc_run("arm0", N=100, B=B, max_rungs=max_rungs, n_mut=n_mut, n_disp=n_disp, seed=seed,
                  n_csmc=n_csmc, csmc_M=csmc_M)
    ua = aug["history"][-1]["U_mean"]; wa = aug["wall"]
    print(f"+cSMC      final U/N {ua:.4f}  {wa:.0f}s  rungs {len(aug['history'])}", flush=True)
    print(f"VERDICT: depth delta {ua-ub:+.4f} (neg=deeper=win) at {wa/max(wb,1e-9):.2f}x wall", flush=True)
    torch.save({"baseline": base["history"], "csmc": aug["history"], "n_csmc": n_csmc, "csmc_M": csmc_M,
                "U_base": ub, "U_csmc": ua, "wall_base": wb, "wall_csmc": wa},
               os.path.join(ART, f"csmc_insmc_depth_M{csmc_M}.pt"))


def matched_wall(n_disp=176, B=256, max_rungs=40, n_mut=2, seed=0):
    """Matched-WALL control for the in-SMC depth gate: the +cSMC run cost 4.36x the baseline's wall. Give the
    pure-displacement baseline the SAME wall (scale n_disp ~4.36x) and ask whether it reaches the same depth.
    If yes -> cSMC's extra depth was just bought with time (no real lever). If baseline stays shallower ->
    cSMC's depth is a genuine mixing win."""
    from liquid_coupling_flow.ka_local_smc import smc_run
    print(f"=== matched-wall baseline: n_csmc=0, n_disp={n_disp} (~4.4x mutation budget), B={B} ===", flush=True)
    base = smc_run("arm0", N=100, B=B, max_rungs=max_rungs, n_mut=n_mut, n_disp=n_disp, seed=seed, n_csmc=0)
    ub = base["history"][-1]["U_mean"]; wb = base["wall"]
    prev = os.path.join(ART, "csmc_insmc_depth_M16.pt")
    uc = torch.load(prev, weights_only=False)["U_csmc"] if os.path.exists(prev) else float("nan")
    print(f"MATCHED-WALL baseline final U/N {ub:.4f}  {wb:.0f}s  rungs {len(base['history'])}", flush=True)
    print(f"  vs +cSMC (prior) {uc:.4f}  ->  delta {uc-ub:+.4f} (neg = cSMC deeper at matched wall = real lever)",
          flush=True)
    torch.save({"history": base["history"], "U": ub, "wall": wb, "n_disp": n_disp, "U_csmc": uc},
               os.path.join(ART, f"csmc_matched_wall_nd{n_disp}.pt"))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stationarity"
    if cmd == "stationarity":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 16
        rs = len(sys.argv) > 3 and sys.argv[3] == "1"
        an = len(sys.argv) > 4 and sys.argv[4] == "1"
        stationarity(M=M, resample=rs, ancestor=an)
    elif cmd == "ga1":
        ga1(M=int(sys.argv[2]) if len(sys.argv) > 2 else 16)
    elif cmd == "insmc":
        insmc_depth(n_csmc=int(sys.argv[2]) if len(sys.argv) > 2 else 1,
                    csmc_M=int(sys.argv[3]) if len(sys.argv) > 3 else 16)
    elif cmd == "matched":
        matched_wall(n_disp=int(sys.argv[2]) if len(sys.argv) > 2 else 176)
