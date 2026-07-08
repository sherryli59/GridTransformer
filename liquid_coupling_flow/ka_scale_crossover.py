"""3-ARM SCALE CROSSOVER — the headline transfer experiment (spec
2026-07-07-pt-augmentation-and-scale-crossover-design.md, updated per PT1's verdict to THREE arms).

At each N in {100, 256, 576}: (1) A2-stack SMC (flow seeds + disp + SB + heat-bath; all learned parts
N=100-trained) runs to ladder completion and SETS the wall budget W_N; (2) plain PT and (3) PT+A2 each run
for ~W_N (sweep count from a timing probe). Metric: cold/final U/N at matched wall vs N, arms against each
other (no trusted reference at N>=256 — that is the point). Per-unit saves throughout.

PRE-CHECKS (run first, `precheck`): N=576 scaffold + A2 frozen-cage DB (N-agnostic, must be ~0) + flowhead
seed smoke + SB one-move smoke. Any failure => that component drops out at 576 (recorded, not fatal).

Run: python -m liquid_coupling_flow.ka_scale_crossover {precheck|run [N...]}
"""
from __future__ import annotations
import os, sys, time, torch
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold, ART
from liquid_coupling_flow.ka_reference import parallel_tempering
from liquid_coupling_flow.ka_mcmc import make_species
from liquid_coupling_flow.ka_energy import ka_energy

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _energy_chunked(cfg, sd, L, N, chunk=128):
    """ka_energy over a big config stack builds [n,N,N,2] => 10.4 GiB at N=576 (the crash that killed BOTH
    N=576 PT-arm attempts at their final measurement line). Chunk it."""
    us = [ka_energy(cfg[i:i + chunk], sd, L) for i in range(0, cfg.shape[0], chunk)]
    return torch.cat(us)


def precheck(N=576):
    ok = {}
    # (a) scaffold machinery
    try:
        sc, L, geo = _scaffold(N, DEV)
        ok["scaffold"] = True
        print(f"precheck scaffold@{N}: OK (L={L:.2f})", flush=True)
    except Exception as e:
        ok["scaffold"] = False; print(f"precheck scaffold@{N}: FAIL {e}", flush=True); return ok
    # (b) A2 frozen-cage DB at N (the N-agnostic exactness test)
    try:
        from liquid_coupling_flow.ka_heatbath import _load
        HB = _load(torch.load(os.path.join(ART, "ka_heatbath_N100.pt"), map_location=DEV, weights_only=False), DEV)
        B = 8
        pos = torch.rand(B, N, 2, device=DEV) * L
        s = make_species(N, 0.35).long().to(DEV)[None].expand(B, N)
        beta = torch.full((B,), 2.0, device=DEV)
        site = 11
        xq = torch.rand(B, 2, device=DEV) * L
        lqA = HB.log_q_site(pos, s, site, xq, sc, L, beta)
        pos2 = pos.clone(); pos2[:, site] = torch.rand(B, 2, device=DEV) * L
        lqB = HB.log_q_site(pos2, s, site, xq, sc, L, beta)
        db = float((lqA - lqB).abs().max())
        ok["a2_db"] = db < 1e-4
        print(f"precheck A2-DB@{N}: max|dlogq| {db:.2e} -> {'PASS' if ok['a2_db'] else 'FAIL'}", flush=True)
    except Exception as e:
        ok["a2_db"] = False; print(f"precheck A2-DB@{N}: FAIL {e}", flush=True)
    # (c) flowhead seed smoke
    try:
        from liquid_coupling_flow.ka_flowhead import KAFlowHeadModel
        ck = torch.load(os.path.join(ART, "ka_flowhead_N100_k8_scratch.pt"), map_location=DEV, weights_only=False)
        gm = KAFlowHeadModel(rho=1.2, n_bins=ck["n_bins"], knn=ck["knn"], num_bins=ck["num_bins"],
                             tail_bound=ck["tail_bound"]).to(DEV)
        gm.load_state_dict(ck["state_dict"]); gm.eval()
        x, sp, _ = gm.sample(2, N, n_B=round(0.35 * N), device=DEV, return_logq=True)
        ok["seeds"] = bool(torch.isfinite(x).all())
        print(f"precheck flowhead seeds@{N}: {'OK' if ok['seeds'] else 'FAIL'} shape {tuple(x.shape)}", flush=True)
    except Exception as e:
        ok["seeds"] = False; print(f"precheck flowhead seeds@{N}: FAIL {e}", flush=True)
    # (d) SB one move
    try:
        from liquid_coupling_flow.ka_swap_breathe import swap_breathe_sweep
        from liquid_coupling_flow.ka_cluster_flow import _load as _loadP
        P = _loadP(torch.load(os.path.join(ART, "ka_cluster_flow_full_N100.pt"), map_location=DEV,
                              weights_only=False), DEV)
        pos = torch.rand(4, N, 2, device=DEV) * L
        s = make_species(N, 0.35).long().to(DEV)[None].expand(4, N).clone()
        _, _, info = swap_breathe_sweep(P, pos, s, sc, L, geo, beta=2.0, n_moves=2)
        ok["sb"] = True
        print(f"precheck SB@{N}: OK", flush=True)
    except Exception as e:
        ok["sb"] = False; print(f"precheck SB@{N}: FAIL {e}", flush=True)
    torch.save(ok, os.path.join(ART, f"crossover_precheck_N{N}.pt"))
    return ok


def run_size(N, B_smc=None):
    """One size: SMC sets W; PT arms match it. Per-unit saves."""
    from liquid_coupling_flow.ka_local_smc import smc_run
    L = (N / 1.2) ** 0.5
    sd = make_species(N, 0.35).to(DEV)
    B_smc = B_smc or max(64, 25600 // N)
    # ARM 1: A2-stack SMC (sets the wall budget)
    out = smc_run("arm0", N=N, B=B_smc, n_heatbath=1, max_rungs=64, save_tag=f"_xover")
    W = out["wall"]; u_smc = out["history"][-1]["U_mean"]
    del out
    import gc; gc.collect(); torch.cuda.empty_cache()               # OOM@576: SMC left ~21GB held in-process
    print(f"XOVER N={N} SMC: U/N {u_smc:.4f}  wall {W:.0f}s (budget-setter)", flush=True)
    # timing probe for PT sweeps
    T_ladder = 0.5 * (1.25 / 0.5) ** (torch.arange(10) / 9)
    t0 = time.time()
    parallel_tempering(N, L, sd, T_ladder, DEV, n_per=8, n_equil=80, n_collect=20, every=4, track_every=50, seed=0)
    s_sweep = (time.time() - t0) / 100
    res = {"N": N, "smc": {"U": u_smc, "wall": W}, "s_per_sweep_pt": s_sweep}
    torch.save(res, os.path.join(ART, f"crossover_N{N}_partial.pt"))
    # ARMs 2+3: plain PT and PT+A2 at ~matched wall
    for name, hb in (("pt_plain", None), ("pt_hb", "ka_heatbath_N100.pt")):
        factor = 1.0 if hb is None else 1.3 if N >= 256 else 2.16      # hb overhead estimate; wall re-measured
        n_sw = max(2000, int(W / (s_sweep * factor)))
        n_coll = max(400, n_sw // 10)
        t0 = time.time()
        cfg, traj, ex, _ = parallel_tempering(N, L, sd, T_ladder, DEV, n_per=8, n_equil=n_sw - n_coll,
                                              n_collect=n_coll, every=8, track_every=500, seed=0,
                                              hb_ckpt=hb, hb_every=10, hb_moves=25)
        wall = time.time() - t0
        torch.save({"cfg": cfg.cpu(), "traj": traj, "exch": ex}, os.path.join(ART, f"crossover_N{N}_{name}_cfgs.pt"))
        U = (_energy_chunked(cfg, sd, L, N) / N).mean().item()
        res[name] = {"U": U, "wall": wall, "n_sweeps": n_sw, "exch": ex, "traj": traj}
        del cfg
        import gc; gc.collect(); torch.cuda.empty_cache()
        torch.save(res, os.path.join(ART, f"crossover_N{N}_partial.pt"))
        print(f"XOVER N={N} {name}: U/N {U:.4f}  wall {wall:.0f}s ({n_sw} sweeps, exch {ex:.2f})", flush=True)
    print(f"XOVER N={N} SUMMARY: SMC {u_smc:.4f}@{W:.0f}s | plain {res['pt_plain']['U']:.4f}@"
          f"{res['pt_plain']['wall']:.0f}s | PT+A2 {res['pt_hb']['U']:.4f}@{res['pt_hb']['wall']:.0f}s", flush=True)
    return res


def main(Ns=(100, 256, 576)):
    all_res = {}
    for N in Ns:
        if N >= 500:
            ok = precheck(N)
            if not ok.get("scaffold"):
                print(f"XOVER N={N} SKIPPED (scaffold fail)", flush=True); continue
        all_res[N] = run_size(N)
    torch.save(all_res, os.path.join(ART, "crossover_all.pt"))
    print("CROSSOVER COMPLETE", flush=True)


def pt_arms_only(N):
    """Recovery: run just the PT arms using the saved SMC budget from crossover_N{N}_partial.pt."""
    d = torch.load(os.path.join(ART, f"crossover_N{N}_partial.pt"), map_location="cpu", weights_only=False)
    W, s_sweep = d["smc"]["wall"], d["s_per_sweep_pt"]
    L = (N / 1.2) ** 0.5
    sd = make_species(N, 0.35).to(DEV)
    T_ladder = 0.5 * (1.25 / 0.5) ** (torch.arange(10) / 9)
    res = dict(d)
    for name, hb in (("pt_plain", None), ("pt_hb", "ka_heatbath_N100.pt")):
        if name in res:
            continue
        factor = 1.0 if hb is None else 1.3 if N >= 256 else 2.16
        n_sw = max(2000, int(W / (s_sweep * factor)))
        n_coll = max(400, n_sw // 10)
        t0 = time.time()
        cfg, traj, ex, _ = parallel_tempering(N, L, sd, T_ladder, DEV, n_per=8, n_equil=n_sw - n_coll,
                                              n_collect=n_coll, every=8, track_every=500, seed=0,
                                              hb_ckpt=hb, hb_every=10, hb_moves=25)
        wall = time.time() - t0
        torch.save({"cfg": cfg.cpu(), "traj": traj, "exch": ex}, os.path.join(ART, f"crossover_N{N}_{name}_cfgs.pt"))
        U = (_energy_chunked(cfg, sd, L, N) / N).mean().item()
        res[name] = {"U": U, "wall": wall, "n_sweeps": n_sw, "exch": ex, "traj": traj}
        torch.save(res, os.path.join(ART, f"crossover_N{N}_partial.pt"))
        print(f"XOVER N={N} {name}: U/N {U:.4f}  wall {wall:.0f}s ({n_sw} sweeps, exch {ex:.2f})", flush=True)
        del cfg
        import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"XOVER N={N} SUMMARY: SMC {res['smc']['U']:.4f}@{res['smc']['wall']:.0f}s | "
          f"plain {res['pt_plain']['U']:.4f}@{res['pt_plain']['wall']:.0f}s | "
          f"PT+A2 {res['pt_hb']['U']:.4f}@{res['pt_hb']['wall']:.0f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pt_arms":
        pt_arms_only(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "precheck":
        precheck(int(sys.argv[2]) if len(sys.argv) > 2 else 576)
    else:
        main(tuple(int(a) for a in sys.argv[1:] if a.isdigit()) or (100, 256, 576))
