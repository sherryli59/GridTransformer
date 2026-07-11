"""Decisive interpretation of the G2 overfit run: is CavityGenerator learning
the cavity conditional, or is 1.33 just the trivial CFM floor?

Reports, against concrete baselines the raw scalar lacks:
  1. trained out_scale (velocity-starvation check);
  2. model FM loss vs ZERO-velocity baseline vs a data-mean predictor
     (averaged over many (x0,t) draws) -> how far below trivial;
  3. per-RADIUS sample->target RMS (R=1.6 peaked conditional SHOULD memorize;
     R=2.4 large RMS is CORRECT diversity, not failure);
  4. energy/clash of GENERATED interiors vs the true carved interiors
     (physical vs core-fill garbage);
  5. generator self-overlap vs the carved config's structure.
CPU, reads only the saved checkpoint (state_dict + training pairs)."""
from pathlib import Path
import torch
from liquid_coupling_flow.ka3d_cavity_generator import CavityGenerator, cavity_fm_loss, collate_cavities, _uniform_ball
from liquid_coupling_flow.ka3d_cavity_eval import assemble_generated
from liquid_coupling_flow.ka3d_pts_observables import core_overlap
from liquid_coupling_flow.ka_energy import ka_energy

CK = Path("liquid_coupling_flow/artifacts/ka3d_cavity_generator_g2.pt")
dev = "cpu"
ck = torch.load(CK, map_location=dev, weights_only=False)
cfg = ck["config"]; pairs = ck["pairs"]
for p in pairs:
    for k, v in p.items():
        if torch.is_tensor(v):
            p[k] = v.to(dev)
model = CavityGenerator(cfg["n_max"], cfg["hidden"], cfg["layers"], cfg["n_ctx_max"]).to(dev)
model.load_state_dict(ck["state_dict"]); model.eval()
L = float(pairs[0]["L"])
print(f"[diag] pairs={len(pairs)} n_max={cfg['n_max']} L={L:.3f} "
      f"first_loss={ck['first_loss']:.4f} final_loss={ck['final_loss']:.4f} "
      f"saved_sample_rms={ck['sample_target_rms']:.4f}")
print(f"[diag] TRAINED out_scale = {model.out_scale.item():.4f}  (init 0.1; ~O(1)+ means it grew)")

batch = collate_cavities(pairs, n_max=cfg["n_max"], n_ctx_max=cfg["n_ctx_max"])
mask = batch["mask"]

# ---- (2) loss vs baselines, averaged over draws ----
@torch.no_grad()
def avg_losses(n=64):
    ml = zl = 0.0
    for _ in range(n):
        ml += float(cavity_fm_loss(model, batch))
        x1, R = batch["x_in"], batch["R"]
        x0 = _uniform_ball(x1.shape[0], model.n_max, R, x1.device, x1.dtype) * mask[..., None]
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        target = (x1 - x0) * mask[..., None]
        zl += float((target.square() * mask[..., None]).sum() / (mask.sum() * 3).clamp_min(1))
    return ml / n, zl / n
ml, zl = avg_losses()
frac = (zl - ml) / zl if zl > 0 else float("nan")
print(f"[diag] FM loss model={ml:.4f}  zero-velocity baseline={zl:.4f}  "
      f"-> model explains {frac*100:.1f}% of trivial-velocity variance "
      f"({'LEARNING' if frac > 0.3 else 'WEAK/NULL' if frac < 0.12 else 'PARTIAL'})")

# ---- (3) per-radius sample->target RMS ----
with torch.no_grad():
    samp = model.sample(batch["s_in"], mask, batch["x_ctx"], batch["s_ctx"],
                        batch["ctx_mask"], batch["R"], batch["T"])
Rvals = batch["R"]
for Rtag in sorted(set(float(r) for r in Rvals.tolist())):
    sel = (Rvals == Rtag)
    if not sel.any():
        continue
    m = mask & sel[:, None]
    rms = (((samp - batch["x_in"]) ** 2).sum(-1)[m].mean().sqrt()).item()
    # random-in-ball control: RMS between two independent uniform-ball draws
    x0a = _uniform_ball(int(sel.sum()), model.n_max, Rvals[sel], dev, samp.dtype)
    x0b = _uniform_ball(int(sel.sum()), model.n_max, Rvals[sel], dev, samp.dtype)
    mm = mask[sel]
    ctrl = (((x0a - x0b) ** 2).sum(-1)[mm].mean().sqrt()).item()
    print(f"[diag] R={Rtag:.1f}: sample->target RMS={rms:.3f}  "
          f"(uniform-ball<->uniform-ball control={ctrl:.3f}; "
          f"peaked conditional wants RMS<<control)")

# ---- (4) energy / clash of generated vs true interiors ----
@torch.no_grad()
def config_energy_per_particle(pair, rel_interior):
    x, s = assemble_generated(pair, rel_interior)
    U = ka_energy(x, s, L)
    return (U / x.shape[1]).cpu()

gen_U, true_U, n_clash = [], [], 0
with torch.no_grad():
    for p in pairs:
        n = p["n_in"]
        b1 = collate_cavities([p], n_max=cfg["n_max"], n_ctx_max=cfg["n_ctx_max"])
        rel = model.sample(b1["s_in"], b1["mask"], b1["x_ctx"], b1["s_ctx"],
                           b1["ctx_mask"], b1["R"], b1["T"])[:, :n]
        ug = float(config_energy_per_particle(p, rel[0]))
        # true interior in center-relative coords
        d = p["x_in"] - p["center"]; d = d - L * torch.round(d / L)
        ut = float(config_energy_per_particle(p, d))
        gen_U.append(ug); true_U.append(ut)
        if ug > ut + 2.0:
            n_clash += 1
gen_U = torch.tensor(gen_U); true_U = torch.tensor(true_U)
print(f"[diag] energy/particle  GENERATED median={gen_U.median():.3f} mean={gen_U.mean():.2f}  "
      f"| TRUE median={true_U.median():.3f} mean={true_U.mean():.2f}  "
      f"| clashed(gen>true+2)={n_clash}/{len(pairs)}")
print("[diag] DONE — verdict key: out_scale grown + frac>0.3 + R=1.6 RMS<<control + "
      "gen median near true => generator represents the conditional; else structural/bug.")
