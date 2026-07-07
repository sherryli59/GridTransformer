"""GB1 gates for the collective move model (ka_move_model.MoveModel).

  accept       — one-shot MH acceptance + abort rate at beta=2: (a) from the equilibrium reference, and
                 (b) from the SMC-STUCK population (smc_pilot_arm0_N100.pt final configs, ~-3.13) — the
                 deployment target: does the learned collective move fire where the stack is stuck?
  stationarity — exactness in deployment form (move sweep MIXED with displacement): <U>/N + g_BB flat from
                 the reference; catches occupancy/DB bugs beyond the unit tests.
  insmc        — the decisive depth gate: A2 stack (disp+SB+heat-bath) vs A2 stack + move sweep, matched
                 schedule. Bar: past ~-3.14 toward the true -3.276.

Run: python -m liquid_coupling_flow.ka_move_gate {accept|stationarity|insmc}
"""
from __future__ import annotations
import os, sys, time, torch
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold, ART
from liquid_coupling_flow.ka_move_model import _load, move_mh_sweep
from liquid_coupling_flow.ka_local_smc import _disp_sweeps, _canonicalize
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_observables import partial_gr

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "ka_move_model_N100.pt"


def _model():
    return _load(torch.load(os.path.join(ART, CKPT), map_location=DEV, weights_only=False), DEV)


def _ref_env(B=128, N=100):
    sc, L, geo = _scaffold(N, DEV)
    lad = torch.load(os.path.join(ART, f"pt_ladder_N{N}.pt"), map_location=DEV, weights_only=False)
    x = lad["configs_per_rung"][0][:B].to(DEV); s0 = lad["s"].to(DEV).long()
    pos, s = slot_order(x, s0, geo, N)
    return sc, L, geo, pos, s


def accept(T=3, B=128):
    m = _model()
    sc, L, geo, pos, s = _ref_env(B=B)
    gen = torch.Generator(device=DEV).manual_seed(0)
    # (a) from equilibrium
    p = pos.clone()
    for _ in range(T):
        p, _, info = move_mh_sweep(m, p, s, sc, L, geo, beta=2.0, gen=gen)
    print(f"accept@equilibrium: {info['accept']:.4f}  abort {info['abort_frac']:.3f}", flush=True)
    # (b) from the SMC-stuck population
    pilot = os.path.join(ART, "smc_pilot_arm0_N100.pt")
    if os.path.exists(pilot):
        d = torch.load(pilot, map_location=DEV, weights_only=False)
        ps, ss = slot_order(d["pos"][:B].to(DEV), d["s"][:B].to(DEV).long()[0], geo, 100)
        u0 = (ka_energy(ps, ss, L) / 100).mean().item()
        for _ in range(T):
            ps, ss, info2 = move_mh_sweep(m, ps, ss, sc, L, geo, beta=2.0, gen=gen)
        u1 = (ka_energy(ps, ss, L) / 100).mean().item()
        print(f"accept@SMC-stuck: {info2['accept']:.4f}  abort {info2['abort_frac']:.3f}  "
              f"U/N {u0:.4f}->{u1:.4f} ({T} sweeps)", flush=True)
    torch.save({"eq_accept": info["accept"]}, os.path.join(ART, "move_accept_N100.pt"))


def stationarity(T=40, B=128, beta=2.0, n_disp=40):
    """Deployment form: [move sweep + n_disp displacement] blocks from the reference; band must stay tight."""
    m = _model()
    sc, L, geo, pos, s = _ref_env(B=B)
    pos, s = _canonicalize(pos, s)
    gen = torch.Generator(device=DEV).manual_seed(0)
    u0 = (ka_energy(pos, s, L) / 100).mean().item()
    _, g0 = partial_gr(pos, s, L, rmax=3.0, nbins=60, pair=(1, 1))
    print(f"[move stationarity] start <U>/N {u0:.4f}  g_BB {max(g0):.3f}", flush=True)
    hist = [(0, u0, float(max(g0)))]; t0 = time.time()
    for t in range(1, T + 1):
        pos, s, info = move_mh_sweep(m, pos, s, sc, L, geo, beta=beta, gen=gen)
        pos, s2 = _canonicalize(pos, s)
        pos = _disp_sweeps(pos, s2[0], L, kT=1.0 / beta, n=n_disp); s = s2
        if t % 5 == 0 or t == 1:
            u = (ka_energy(pos, s, L) / 100).mean().item()
            _, g = partial_gr(pos, s, L, rmax=3.0, nbins=60, pair=(1, 1))
            hist.append((t, u, float(max(g))))
            print(f"  block {t:3d}: <U>/N {u:.4f}  g_BB {max(g):.3f}  mv-accept {info['accept']:.4f}  "
                  f"({(time.time()-t0)/t:.1f}s/block)", flush=True)
    us = [u for _, u, _ in hist[1:]]
    band = max(us) - min(us); drift = us[-1] - u0
    print(f"[move stationarity] band {band:.4f} drift {drift:+.4f} -> "
          f"{'STATIONARY' if band < 0.05 and abs(drift) < 0.03 else 'DRIFTS'}", flush=True)
    torch.save({"hist": hist, "band": band, "drift": drift}, os.path.join(ART, "move_stationarity_N100.pt"))


def insmc(B=256, max_rungs=40, n_mut=2, n_disp=40, seed=0):
    """A2 stack vs A2 stack + move sweep (the decisive depth gate)."""
    from liquid_coupling_flow.ka_local_smc import smc_run
    print(f"=== in-SMC depth: A2 stack (n_heatbath=1) vs +move, B={B} ===", flush=True)
    base = smc_run("arm0", N=100, B=B, max_rungs=max_rungs, n_mut=n_mut, n_disp=n_disp, seed=seed,
                   n_heatbath=1)
    ub = base["history"][-1]["U_mean"]; wb = base["wall"]
    print(f"A2 STACK   final U/N {ub:.4f}  {wb:.0f}s", flush=True)
    aug = smc_run("arm0", N=100, B=B, max_rungs=max_rungs, n_mut=n_mut, n_disp=n_disp, seed=seed,
                  n_heatbath=1, n_move=1)
    ua = aug["history"][-1]["U_mean"]; wa = aug["wall"]
    print(f"+MOVE      final U/N {ua:.4f}  {wa:.0f}s", flush=True)
    print(f"VERDICT: depth delta {ua-ub:+.4f} (neg=deeper) at {wa/max(wb,1e-9):.2f}x wall", flush=True)
    torch.save({"base": base["history"], "move": aug["history"], "U_base": ub, "U_move": ua,
                "wall_base": wb, "wall_move": wa}, os.path.join(ART, "move_insmc_depth.pt"))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "accept"
    {"accept": accept, "stationarity": stationarity, "insmc": insmc}[cmd]()
