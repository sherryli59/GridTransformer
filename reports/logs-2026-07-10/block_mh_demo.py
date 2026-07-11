"""Concept test for the exact block-MH kernel on a free cluster (no boundary yet).

Two runs at T*=0.5 (beta=2): (1) from the TRUE config -> should stay near true energy at high
acceptance (equilibrium/stationarity check); (2) from a one-shot warm start (drifted/clashy) ->
should RELAX toward the true energy (the warm-start + block-MH pipeline). Reports acceptance and
energy/clash trajectories. Uses the CURRENT (full-order, non-masked) checkpoint -> a lower bound
on what masked training will give.
"""
import shutil, statistics as st
from pathlib import Path
import torch
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_scaffold_ar import KA3DScaffoldAR, label_to_scaffold
from liquid_coupling_flow.ka3d_block import block_mh_step
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda" if torch.cuda.is_available() else "cpu"
SCR = Path("/tmp/claude-1002/-mnt-ssd-GridTransformer/271167b9-117a-400e-b11f-c2c1bfa0b6df/scratchpad")
ckp = SCR / "blockmh_ckpt.pt"; shutil.copy("liquid_coupling_flow/artifacts/ka3d_free_scaffold_ar_best.pt", ckp)
ck = torch.load(ckp, map_location=dev, weights_only=False)
m = KA3DScaffoldAR().to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
print(f"ckpt step={ck['step']}", flush=True)

d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
R, beta, K, SWEEPS = 2.0, 2.0, 4, 150
gen = torch.Generator(device=dev).manual_seed(7)
empty_x, empty_s = torch.zeros(0, 3, device=dev), torch.zeros(0, dtype=torch.long, device=dev)


def energy_total(xo, so):
    return ka_energy(xo[None], so.long()[None], 100.0)[0]


def clash_frac(xo, cutoff=0.8):
    dd = torch.cdist(xo, xo) + torch.eye(xo.shape[0], device=dev) * 1e9
    return float((dd.min(1).values < cutoff).float().mean())


# one held cluster
center = X[950][123 % X.shape[1]]
p = carve(X[950], S[950], center, R, L)
xr = _mic(p["x_in"], center, L)
x_true, s_true, _ = label_to_scaffold(xr, p["s_in"], R)
n = x_true.shape[0]; n_B = int((s_true == 1).sum())
warm_x, warm_s = m.sample_pair(empty_x, empty_s, n - n_B, n_B, R)      # drifted one-shot warm start
print(f"n={n} true U/N={float(energy_total(x_true, s_true))/n:+.3f} clash={clash_frac(x_true):.2f} | "
      f"warm U/N={float(energy_total(warm_x, warm_s))/n:+.2f} clash={clash_frac(warm_x):.2f}", flush=True)

for label, (x0, s0) in [("from-true", (x_true.clone(), s_true.clone())),
                        ("from-warm", (warm_x.clone(), warm_s.clone()))]:
    x, s = x0, s0
    acc = mv = 0
    print(f"\n[{label}]", flush=True)
    for sweep in range(SWEEPS + 1):
        if sweep % 30 == 0:
            print(f"  sweep {sweep:3d}  U/N={float(energy_total(x, s))/n:+.3f}  clash={clash_frac(x):.2f}  "
                  f"acc={100*acc/max(mv,1):.0f}%", flush=True)
        if sweep == SWEEPS:
            break
        for _ in range(max(1, n // K)):
            block = torch.zeros(n, dtype=torch.bool, device=dev)
            block[torch.randperm(n, device=dev, generator=gen)[:K]] = True
            x, s, a, _ = block_mh_step(m, x, s, block, empty_x, empty_s, R, beta, energy_total, gen)
            acc += int(a); mv += 1
print("\ndone", flush=True)
