"""Measure p = P(one full q0 draw lands in the reference basin) -- the ONE number that decides whether the
unflawed single-chain kernel (full-regen independence-MTM at high lambda) can cross. Evidence so far: ref
configs are NOT pointwise under-weighted (z~-0.5) and the lambda=1 tempered weight favors ref by +330-670
nats (instant acceptance once drawn) -> viability = draw frequency alone. Draw NDRAW full alien regens per
cavity (AR forward, energy-free), score q~_box(draw vs reference); a HIT = q~ > HIT_THR (well above the bulk
floor ~0.06, toward the pinned value). Report hit-rate p per cavity + the max q~ seen (how close q0 ever
gets). lam05 Rext, R=2.0, l=0.368."""
import sys, statistics as st, functools
import torch
sys.path.insert(0, "reports/logs-2026-07-13")
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5; R = 2.0; ART = "liquid_coupling_flow/artifacts"; L_BOX = 0.368
RHO = 1.149; BULK = RHO * L_BOX ** 3
NDRAW = 8192; BATCH = 512; HIT_THR = 0.30; NCAV = 4
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


@functools.lru_cache(maxsize=16)
def n_boxes(l, R):
    g = torch.arange(-int(R / l) - 1, int(R / l) + 2) * l + l / 2
    gx, gy, gz = torch.meshgrid(g, g, g, indexing="ij")
    return int((gx ** 2 + gy ** 2 + gz ** 2 < R * R).sum())


def box_id_set(x, l, R):
    mm = x.norm(dim=-1) < R; ijk = torch.floor(x[mm] / l).long(); off = ijk - ijk.min(0).values
    span = off.max(0).values + 1
    return set((off[:, 0] * span[1] * span[2] + off[:, 1] * span[2] + off[:, 2]).tolist())


gen = torch.Generator(device=dev).manual_seed(0); ncav = 0
print(f"=== basin mass p under q0 (NDRAW={NDRAW}/cavity, HIT = q~_box > {HIT_THR}; bulk floor ~{BULK:.3f}) ===", flush=True)
print(f"{'cav':>4} {'n':>4} | {'hits':>5} {'p':>9} | {'max q~':>7} {'q~ q99':>7} {'q~ mean':>8}", flush=True)
allp = []
for ci in range(24):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < 14:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xin = _mic(p["x_in"], c, L)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    ref_ids = box_id_set(xin, L_BOX, R); Nb = n_boxes(L_BOX, R)
    qs = []
    g = torch.Generator(device=dev).manual_seed(500 + ci)
    for b0 in range(0, NDRAW, BATCH):
        nb = min(BATCH, NDRAW - b0)
        Xa, Sa, _ = m.sample_block_b(xo[None].expand(nb, n, 3).clone(), so[None].expand(nb, n).clone(),
                                     allm, bnd, sb, R, gen=g)
        Xa = Xa.cpu()
        for k in range(nb):
            q = len(ref_ids & box_id_set(Xa[k], L_BOX, R)) / (L_BOX ** 3 * Nb)
            qs.append(q - BULK)
    qt = torch.tensor(qs)
    hits = int((qt > HIT_THR).sum()); pr = hits / NDRAW
    allp.append(pr)
    print(f"{ci:>4} {n:>4} | {hits:>5} {pr:>9.5f} | {float(qt.max()):>7.3f} {float(qt.quantile(.99)):>7.3f} "
          f"{float(qt.mean()):>8.3f}", flush=True)
    ncav += 1
    if ncav >= NCAV:
        break
print(f"\np mean {st.mean(allp):.5f}. If p ~ 0 across {NDRAW}/cavity: upper bound p < {1/NDRAW:.1e} per draw;", flush=True)
print(f"full-regen MTM needs ~1/(p*M_try) attempts -> judge feasibility (draws are energy-free AR forwards;", flush=True)
print(f"each MTM try costs one KA energy eval, cheap here). max q~ shows how CLOSE q0 ever gets.", flush=True)
