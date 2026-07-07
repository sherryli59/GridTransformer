"""PT+A2 gates (learned-augmented PT; spec 2026-07-07-pt-augmentation-and-scale-crossover-design.md).

  pt0 — A2 zero-shot N=256 validity (the transfer pre-gate, no reference needed):
        (i) frozen-cage DB invariance at N=256 (move ONLY the site; log_q_site must be unchanged),
        (ii) acceptance on N=256 cold-ladder configs (healthy ~ N=100's 33%),
        (iii) short mixed stationarity band from the N=256 cold rung.
  pt1 — N=100 matched-WALL control: plain PT vs PT+A2 (hb_every=10, hb_moves=25), single seed each,
        cold <U>/N-vs-WALL curves (wall = sweep x measured s/sweep). Bar: PT+A2's curve sits below plain's
        at equal wall (reaches -3.264 sooner and/or pushes toward the true -3.276).

Run: python -m liquid_coupling_flow.ka_pt_aug_gate {pt0|pt1}
"""
from __future__ import annotations
import os, sys, time, torch
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold, ART
from liquid_coupling_flow.ka_heatbath import _load as _hb_load, single_site_mh_sweep
from liquid_coupling_flow.ka_local_smc import _disp_sweeps, _canonicalize
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_mcmc import make_species

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def pt0(B=64, N=256):
    HB = _hb_load(torch.load(os.path.join(ART, "ka_heatbath_N100.pt"), map_location=DEV, weights_only=False), DEV)
    sc, L, geo = _scaffold(N, DEV)
    lad = torch.load(os.path.join(ART, f"pt_ladder_N{N}.pt"), map_location=DEV, weights_only=False)
    pos, s = slot_order(lad["configs_per_rung"][0][:B].to(DEV), lad["s"].to(DEV).long(), geo, N)
    beta = torch.full((B,), 2.0, device=DEV)
    # (i) frozen-cage DB invariance at N=256
    site = 7
    xq = torch.rand(B, 2, device=DEV) * L
    lqA = HB.log_q_site(pos, s, site, xq, sc, L, beta)
    pos2 = pos.clone(); pos2[:, site] = torch.rand(B, 2, device=DEV) * L
    lqB = HB.log_q_site(pos2, s, site, xq, sc, L, beta)
    db = float((lqA - lqB).abs().max())
    print(f"PT0(i) frozen-cage DB @N=256: max|dlogq| {db:.2e} -> {'PASS' if db < 1e-4 else 'FAIL'}", flush=True)
    # (ii) acceptance
    gen = torch.Generator(device=DEV).manual_seed(0)
    p = pos.clone()
    for _ in range(2):
        p, info = single_site_mh_sweep(HB, p, s, sc, L, geo, beta=2.0, n_moves=100, gen=gen)
    print(f"PT0(ii) acceptance @N=256 cold: {info['accept']:.3f} (N=100 was ~0.33)", flush=True)
    # (iii) short mixed stationarity
    p, s2 = _canonicalize(p, s)
    u0 = (ka_energy(p, s2, L) / N).mean().item(); us = [u0]
    for t in range(8):
        p, info = single_site_mh_sweep(HB, p, s2, sc, L, geo, beta=2.0, n_moves=100, gen=gen)
        p, s2 = _canonicalize(p, s2)
        p = _disp_sweeps(p, s2[0], L, kT=0.5, n=20)
        us.append((ka_energy(p, s2, L) / N).mean().item())
    band = max(us) - min(us)
    print(f"PT0(iii) mixed stationarity: start {u0:.4f} end {us[-1]:.4f} band {band:.4f} "
          f"(NOTE: N=256 ladder under-converged, downward drift toward true eq expected/OK; watch for blowup)",
          flush=True)
    torch.save({"db": db, "accept": info["accept"], "us": us}, os.path.join(ART, "pt0_gate_N256.pt"))


def pt1(n_equil=16000, n_collect=2000, hb_every=10, hb_moves=25):
    from liquid_coupling_flow.ka_reference import parallel_tempering
    N = 100; L = (N / 1.2) ** 0.5
    sd = make_species(N, 0.35).to(DEV)
    T_ladder = 0.5 * (1.25 / 0.5) ** (torch.arange(10) / 9)
    out = {}
    for name, hb in (("plain", None), ("hb", "ka_heatbath_N100.pt")):
        t0 = time.time()
        cfg, traj, ex, _ = parallel_tempering(N, L, sd, T_ladder, DEV, n_per=8, n_equil=n_equil,
                                              n_collect=n_collect, every=8, track_every=250, seed=0,
                                              hb_ckpt=hb, hb_every=hb_every, hb_moves=hb_moves)
        wall = time.time() - t0
        U = (ka_energy(cfg, sd, L) / N).mean().item()
        spersweep = wall / (n_equil + n_collect)
        out[name] = {"traj": traj, "wall": wall, "s_per_sweep": spersweep, "U_final": U, "exch": ex}
        print(f"PT1 {name}: final cold <U>/N {U:.4f}  wall {wall:.0f}s ({spersweep*1e3:.0f} ms/sweep)  "
              f"exch {ex:.2f}", flush=True)
    # matched-wall comparison: interpolate plain's cold-U at the wall PT+A2 spent
    pl, hb_ = out["plain"], out["hb"]
    for frac in (0.25, 0.5, 1.0):
        w = frac * min(pl["wall"], hb_["wall"])
        def u_at(o, w):
            tr = [(sw * o["s_per_sweep"], u) for sw, u in o["traj"]]
            best = min(tr, key=lambda p: abs(p[0] - w))
            return best[1]
        print(f"  at wall {w:.0f}s: plain {u_at(pl, w):.4f}  PT+A2 {u_at(hb_, w):.4f}", flush=True)
    torch.save(out, os.path.join(ART, "pt1_matched_wall_N100.pt"))


if __name__ == "__main__":
    {"pt0": pt0, "pt1": pt1}[sys.argv[1] if len(sys.argv) > 1 else "pt0"]()
