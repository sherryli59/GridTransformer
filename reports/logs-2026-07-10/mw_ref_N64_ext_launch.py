"""One-off harness: EXTEND the certified N=64 mW reference with 32 warm-started chains.

Warm-start = 32 well-separated configs from the certified G2 bank (ref["cfgs"][::1000][:32],
1000-index spacing = 250 sweeps apart across chains), so n_equil is a short 500-sweep safety
burn-in (mc_run init_cfgs, commit b8de008). Collection: 16000 sweeps, every=8 -> 2000 events
x 32 chains = 64k configs of new trajectory at the ambient point (beta=1/T*, rho=rho*).
Partial saves every 2000 sweeps via ckpt_path (checkpoint-incrementally directive).

Run: nohup python reports/logs-2026-07-10/mw_ref_N64_ext_launch.py > reports/logs-2026-07-10/mw_ref_N64_ext.out 2>&1 &
"""
import os, torch
from liquid_coupling_flow.mw.mw_reference import mc_run, ART
from liquid_coupling_flow.mw.mw_energy import T_STAR, RHO_STAR

N, B = 64, 32
L = (N / RHO_STAR) ** (1.0 / 3.0)
beta = 1.0 / T_STAR

bank_path = os.path.join(ART, "mw_g2_N64.pt")
bank = torch.load(bank_path, map_location="cpu")
init = bank["ref"]["cfgs"][::1000][:32]                          # [32,64,3], 1000-index spacing
assert init.shape == (B, N, 3), init.shape
print(f"mw_ref_ext: N={N} B={B} L={L:.4f} beta={beta:.4f} n_equil=500 n_collect=16000 every=8 "
      f"seed=7 init from {bank_path} [::1000][:32]", flush=True)

partial = os.path.join(ART, "mw_ref_N64_ext.partial.pt")
final = os.path.join(ART, "mw_ref_N64_ext.pt")
out = mc_run(N, L, beta, n_equil=500, n_collect=16000, every=8, seed=7, B=B,
             init_cfgs=init, ckpt_path=partial)
out["init_src"] = {"bank": bank_path, "slice": "ref.cfgs[::1000][:32]"}
out["budget"] = {"n_equil": 500, "n_collect": 16000, "every": 8, "seed": 7, "B": B}
torch.save(out, final)
print(f"mw_ref_ext DONE: cfgs {tuple(out['cfgs'].shape)} step {out['step']:.4f} acc {out['acc']:.3f} "
      f"flat_budget {out['flat_budget']:.4f} coll_drift {out['coll_drift']:.4f} -> {final}", flush=True)
