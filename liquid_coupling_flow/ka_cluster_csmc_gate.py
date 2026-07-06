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
from liquid_coupling_flow.ka_cluster_csmc import csmc_cluster_move
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


def csmc_sweep(P, pos, s, sc, L, geo, k=7, beta=2.0, M=16, n_moves=None, resample=False, ancestor=False, gen=None):
    """Positional cSMC cluster sweep (mirrors swap_breathe_sweep): re-slot-order per move (positions change),
    apply the move, scatter cluster positions back. Species untouched. Returns pos, s, mean-info."""
    B, N, _ = pos.shape; dev = pos.device
    n_moves = N if n_moves is None else n_moves
    survs, moved = [], []
    for _ in range(n_moves):
        seed = int(torch.randint(0, N, (1,), device=dev, generator=gen).item())
        order = geo._curve_order(pos, N)
        pos_o = torch.gather(pos, 1, order[..., None].expand(-1, -1, 2))
        s_o = torch.gather(s, 1, order)
        cl = KC.cluster_slots(seed, sc, k, L)
        pos_n, _, info = csmc_cluster_move(P, pos_o, s_o, cl, sc, L, beta=beta, M=M,
                                           resample=resample, ancestor=ancestor, gen=gen)
        idx = order[:, cl]
        pos = pos.clone()
        pos[torch.arange(B, device=dev)[:, None], idx] = pos_n[:, cl]
        survs.append(info["ref_survival"]); moved.append(1.0 - info["ref_survival"])
    return pos, s, {"ref_survival": sum(survs) / len(survs), "accept": sum(moved) / len(moved)}


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
    # MTM
    pm = pos.clone(); gen2 = torch.Generator(device=DEV).manual_seed(0); t0 = time.time()
    for _ in range(T):
        pm, sm, im = mtm_sweep(P, pm, s, sc, L, M=M, beta=beta, k=7, gen=gen2)
    tm = (time.time() - t0) / T
    um = (ka_energy(pm, s, L) / N).mean().item()
    print(f"GA1 (M={M}, {T} sweeps from reference, beta={beta}):", flush=True)
    print(f"  cSMC : accept {ic['accept']:.3f}  <U>/N {uc:.4f}  {tc:.1f}s/sweep", flush=True)
    print(f"  MTM  : accept {im.get('acceptance', float('nan')):.3f}  <U>/N {um:.4f}  {tm:.1f}s/sweep", flush=True)
    torch.save({"csmc": {"accept": ic["accept"], "U": uc, "t": tc},
                "mtm": {"accept": im.get("acceptance"), "U": um, "t": tm}, "M": M},
               os.path.join(ART, f"csmc_ga1_M{M}.pt"))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stationarity"
    if cmd == "stationarity":
        M = int(sys.argv[2]) if len(sys.argv) > 2 else 16
        rs = len(sys.argv) > 3 and sys.argv[3] == "1"
        an = len(sys.argv) > 4 and sys.argv[4] == "1"
        stationarity(M=M, resample=rs, ancestor=an)
    elif cmd == "ga1":
        ga1(M=int(sys.argv[2]) if len(sys.argv) > 2 else 16)
