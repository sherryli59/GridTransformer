"""THROUGH-xi_PTS island-SMC sweep with the PAPER'S observable (Hocky-Markland-Reichman PRL 108, 225506
(2012), the canonical cavity-PTS paper on KA 80:20 LJ). Their overlap is a COARSE-GRAINED BOX-OCCUPATION
product, NOT species-min-dist:
    q(R) = (l^3 N_box)^-1  sum_i  n_i^ref n_i^gen ,   n_i in {0,1} = box-i occupied (ANY particle)
report the bulk-subtracted q~ = q - q_bulk (what their Fig 2 plots); ceiling q(self)=rho. q_bulk = reference
interior vs an INDEPENDENT data config's interior at the SAME cavity geometry (Cavagna random-config test).
Headline l=0.3 non-species (Hocky); species-resolved + species-min-dist shown for continuity.

State-point caveat (printed): our model is rho=1.149 T=0.5; paper is rho=1.20 T=0.55. Same observable, nearby
state point -> expect same ballpark/shape, NOT an exact number match. Paper Fig 2c anchors (LJ T=0.55, by-eye
+-0.05): q~ ~ 0.33 @ R=2.2, ~0.29 @ R=2.4.

Island-SMC: tempered pi_lam ~ q0^(1-lam) e^(-lam beta U), exact geometric bridge, UNIFORM island recombination
for the observable (logZ-weighting is winner-take-all noise for pinned R -- diag_recombine_check). Sweeps R
through xi_PTS. 1blob-K8 (primary). Rext RL model. Saves raw populations per cavity for re-scoring (durability).
"""
import argparse, time, sys, statistics as st
import torch
sys.path.insert(0, "reports/logs-2026-07-14")
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import island, overlap as overlap_spmin
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda"; RCTX = 2.5; ART = "liquid_coupling_flow/artifacts"
ANCHOR = {2.2: 0.33, 2.4: 0.29}          # paper Fig 2c, LJ T=0.55, by-eye +-0.05


def box_ids(x, l, R):
    m = x.norm(dim=-1) < R
    ijk = torch.floor(x[m] / l).long()
    off = ijk - ijk.min(0).values
    span = off.max(0).values + 1
    return (off[:, 0] * span[1] * span[2] + off[:, 1] * span[2] + off[:, 2]).tolist()


def n_boxes_in_sphere(l, R):
    g = torch.arange(-int(R / l) - 1, int(R / l) + 2) * l + l / 2
    gx, gy, gz = torch.meshgrid(g, g, g, indexing="ij")
    return int((gx ** 2 + gy ** 2 + gz ** 2 < R * R).sum())


def box_overlap(xa, xb, R, l, sa=None, sb=None):
    """(l^3 N_box)^-1 * #cells occupied in BOTH (same species too if sa,sb given)."""
    Nb = n_boxes_in_sphere(l, R)
    if sa is None:
        shared = len(set(box_ids(xa, l, R)) & set(box_ids(xb, l, R)))
    else:
        ma, mb = xa.norm(dim=-1) < R, xb.norm(dim=-1) < R
        A = set(zip(box_ids(xa, l, R), sa[ma].tolist())); B = set(zip(box_ids(xb, l, R), sb[mb].tolist()))
        shared = len(A & B)
    return shared / (l ** 3 * Nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="+", default=[1.6, 2.0, 2.5, 3.0, 3.5])
    ap.add_argument("--islands", type=int, default=6); ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--T", type=int, default=20); ap.add_argument("--n-mut", type=int, default=3)
    ap.add_argument("--ncav", type=int, default=6); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--l-box", type=float, default=0.3)
    ap.add_argument("--ckpt", default=f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt")
    ap.add_argument("--out", default="reports/logs-2026-07-14/smc_pts_hocky.pt")
    a = ap.parse_args()
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
    m.load_state_dict(torch.load(a.ckpt, map_location=dev, weights_only=False)["state_dict"], strict=False)
    m.eval(); m.use_frame = False
    D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
    X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
    rho = X.shape[1] / L ** 3
    cfg = {"move": "1blob", "K": a.K, "swap": False, "name": f"1blob-K{a.K}"}
    print(f"=== through-xi_PTS SMC sweep, HOCKY box-occupation observable ===", flush=True)
    print(f"model={a.ckpt.split('/')[-1]}  rho={rho:.3f} T=0.5  |  PAPER rho=1.20 T=0.55 (Fig2c ~0.33@2.2,0.29@2.4)", flush=True)
    print(f"CAVEAT: same observable, nearby state point (rho 1.149 vs 1.20, T 0.5 vs 0.55) -> ballpark/shape, not exact", flush=True)
    print(f"config={cfg['name']} J={a.islands} M={a.m} T={a.T} n_mut={a.n_mut} ncav={a.ncav} l_box={a.l_box}", flush=True)
    print(f"{'R':>4} | {'q~_box':>16} | {'q_box':>6} {'qbulk':>6} {'qself':>6} | {'q~_spec':>7} | {'q_spmin':>7} | isl-std", flush=True)
    results = {}
    for R in a.radii:
        t0 = time.time(); gen = torch.Generator(device=dev).manual_seed(0); ncav = 0
        rows = []
        for ci in range(24):
            c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
            if p["n_in"] < a.K + 6:
                continue
            xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
            xin = _mic(p["x_in"], c, L); sin = p["s_in"]
            xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
            # bulk baseline: ref interior vs OTHER data configs' interior at the SAME center c (independent)
            qb, qb_s = [], []
            for cj in range(24):
                if cj == ci:
                    continue
                pj = carve(X[cj], S[cj], c, R, L)
                if pj["n_in"] < 1:
                    continue
                xj = _mic(pj["x_in"], c, L); sj = pj["s_in"]
                qb.append(box_overlap(xin, xj, R, a.l_box)); qb_s.append(box_overlap(xin, xj, R, a.l_box, sin, sj))
                if len(qb) >= 6:
                    break
            qbulk = st.mean(qb); qbulk_s = st.mean(qb_s); qself = box_overlap(xin, xin, R, a.l_box)
            # island-SMC, UNIFORM recombination
            qbox_isl, qboxs_isl, qsp_isl, popX, popS = [], [], [], [], []
            for j in range(a.islands):
                g = torch.Generator(device=dev).manual_seed(1000 + 31 * ci + j)
                Xp, Sp, _, _ = island(m, xo, so, bnd, sb, R, cfg, a.m, a.T, a.n_mut, g)
                qbox_isl.append(st.mean([box_overlap(xin, Xp[k], R, a.l_box) for k in range(Xp.shape[0])]))
                qboxs_isl.append(st.mean([box_overlap(xin, Xp[k], R, a.l_box, sin, Sp[k]) for k in range(Xp.shape[0])]))
                qsp_isl.append(overlap_spmin(Xp, Sp, xin, sin))
                popX.append(Xp.cpu()); popS.append(Sp.cpu())
            q_box = st.mean(qbox_isl); q_boxs = st.mean(qboxs_isl); q_sp = st.mean(qsp_isl)
            rows.append({"R": R, "ci": ci, "n_in": int(p["n_in"]), "qbulk": qbulk, "qbulk_s": qbulk_s,
                         "qself": qself, "q_box": q_box, "q_boxs": q_boxs, "q_sp": q_sp,
                         "qbox_isl": qbox_isl, "qsp_isl": qsp_isl,
                         "popX": popX, "popS": popS, "xin": xin.cpu(), "sin": sin.cpu()})
            results[R] = rows; torch.save(results, a.out)                      # incremental per-cavity durability
            ncav += 1
            if ncav >= a.ncav:
                break
        qb = st.mean([r["q_box"] for r in rows]); qbk = st.mean([r["qbulk"] for r in rows])
        qsf = st.mean([r["qself"] for r in rows]); qtil = qb - qbk
        qtil_s = st.mean([r["q_boxs"] for r in rows]) - st.mean([r["qbulk_s"] for r in rows])
        qsp = st.mean([r["q_sp"] for r in rows])
        islstd = st.mean([st.pstdev(r["qbox_isl"]) for r in rows]); se = islstd / (a.islands ** 0.5)
        anch = ANCHOR.get(R); astr = f"  [paper~{anch:.2f}]" if anch else ""
        print(f"{R:>4} | {qtil:>+7.3f}+/-{se:.3f}{'':>3} | {qb:>6.3f} {qbk:>6.3f} {qsf:>6.3f} | {qtil_s:>+7.3f} | "
              f"{qsp:>7.3f} | {islstd:.3f}{astr}", flush=True)
    torch.save(results, a.out)
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
