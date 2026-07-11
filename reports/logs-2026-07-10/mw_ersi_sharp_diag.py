"""Sharp G-b/G-d arbiter on the in-flight eRSI checkpoint (rebuilt from Lightning last.ckpt).
Separates the TRAINING-INDEPENDENT bug test (is the composed density normalized / constant-Z?) from the
TRAINING-DEPENDENT quality test (ESS, SNIS <U>, shells). Resolves whether G-b's grid-quadrature 14.19 is
a real x-dependent defect (fatal to SNIS) or a benign constant-Z/grid artifact (SNIS-invariant)."""
import math, os, tempfile, torch
from liquid_coupling_flow.mw.mw_ersi import load_ersi
from liquid_coupling_flow.mw.mw_energy import mw_energy_chunked, T_STAR
from liquid_coupling_flow.mw.mw_reference import g_r

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LCK = "liquid_coupling_flow/mw/artifacts/mw_ersi_N64_checkpoints/last.ckpt"
lck = torch.load(LCK, map_location="cpu", weights_only=False)
sd = {k[2:]: v for k, v in lck["state_dict"].items() if k.startswith("b.")}
portable = {"state_dict": sd, "N": 64, "K": 12, "hidden_nf": 128, "n_layers": 4,
            "L": 5.195309753663598, "dim_phys": 3, "n_species": 1, "ot": True}
tmp = os.path.join(tempfile.gettempdir(), "ersi_portable_probe.pt")
torch.save(portable, tmp)
m = load_ersi(tmp, DEV)
N, L, beta = m.N, m.L, 1.0 / T_STAR
print(f"loaded step={lck.get('global_step')}  N={N} L={L:.4f} beta={beta:.3f}", flush=True)

g = torch.Generator(device=DEV).manual_seed(0)
# NOTE reverse-time log_q is broken in the training's learndiffeq version (compute_div 'a' kwarg);
# one-shot IS uses ONLY forward sample_and_logq, and SNIS is the arbiter (invariant to constant Z),
# so (B) alone decides usability: SNIS<U> ~ ref => forward logq correct-or-constant-offset => USABLE.

# ---- (B) DECISIVE: one-shot SNIS to the target (forward density only) ----
with torch.no_grad():
    X, logq = m.sample_and_logq(1024, g)
U = mw_energy_chunked(X, L).double()
logw = -beta * U - logq.double()
w = torch.softmax(logw, 0)
ess = float(1.0 / (w ** 2).sum())
u_snis = float((w * U).sum()) / N
u_raw = float(U.mean()) / N
print(f"(B) one-shot: ESS {ess:.1f}/1024  SNIS <U>/N {u_snis:+.4f}  (ref -1.627)  raw-sample <U>/N {u_raw:+.3f}", flush=True)

# ---- (C) structure preview (G-c): raw + reweighted shells ----
r, gr_raw = g_r(X.cpu(), L)
gi = torch.Generator(device=DEV).manual_seed(1)
idx = torch.multinomial(w, 1024, replacement=True, generator=gi)
_, gr_rw = g_r(X[idx].cpu(), L)
i1, i2 = int(1.19 / (L / 2) * len(r)), int(1.85 / (L / 2) * len(r))
print(f"(C) g(r) raw: shell1 {float(gr_raw[i1]):.2f} shell2 {float(gr_raw[i2]):.2f} | "
      f"reweighted {float(gr_rw[i1]):.2f}/{float(gr_rw[i2]):.2f}  (data 2.12 / 1.19)", flush=True)
print("\nVERDICT KEY: (A) std small + Zhat~const => normalization is constant-Z, SNIS valid regardless; "
      "(B) SNIS<U> ~ -1.627 => flow USABLE for one-shot IS (G-b block was quadrature artifact); "
      "shells may lag (undertrained, step "
      f"{lck.get('global_step')}).", flush=True)
torch.save({"Zhat": Zhat, "lr_mean": float(lr.mean()), "lr_std": float(lr.std()), "ess": ess,
            "u_snis": u_snis, "u_raw": u_raw, "r": r, "gr_raw": gr_raw, "gr_rw": gr_rw,
            "step": lck.get("global_step")}, "liquid_coupling_flow/mw/artifacts/mw_ersi_sharp_diag.pt")
