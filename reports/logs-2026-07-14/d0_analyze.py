"""D0 analysis: traces, distributions, crystallization check, T*_work verdict."""
import glob, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 64
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for f in sorted(glob.glob("liquid_coupling_flow/mw/artifacts/d0_probe_T*_N64.pt")):
    res = torch.load(f, map_location="cpu", weights_only=False)
    sw = [s for s, _ in res["traj"]]; uv = [u for _, u in res["traj"]]
    ax[0].plot(sw, uv, label=f"T*={res['tstar']:.4f}")
    ax[1].plot(res["gr_r"], res["gr"], label=f"T*={res['tstar']:.4f}")
    ax[2].hist((res["U"] / N).numpy(), bins=40, alpha=0.5, label=f"T*={res['tstar']:.4f}")
    # crystallization flag: step-drop = any 500-sweep window falling > 0.05/particle
    import numpy as np
    u = np.array(uv); w = 5
    drops = u[:-w] - u[w:]
    print(f"T*={res['tstar']:.4f} coll_drift={res['coll_drift']:.4f} "
          f"max_window_drop={drops.max() if len(drops) else 0:+.4f} "
          f"U/N final={float(res['U'].mean()/N):+.4f}")
ax[0].set(xlabel="sweep", ylabel="U/N", title="D0 traces"); ax[0].legend()
ax[1].set(xlabel="r", ylabel="g(r)", title="g(r) per T*"); ax[1].legend()
ax[2].set(xlabel="U/N", title="collected U/N distributions"); ax[2].legend()
fig.tight_layout(); out = "reports/logs-2026-07-14/d0_regime_probe.png"
fig.savefig(out, dpi=140); print(f"PLOT: /mnt/ssd/GridTransformer/{out}")
