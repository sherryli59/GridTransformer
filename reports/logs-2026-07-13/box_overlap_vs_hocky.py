"""Recompute the point-to-set overlap on our SAVED island-SMC populations in HOCKY-MARKLAND-REICHMAN's
convention (PRL 108, 225506 (2012), the canonical cavity-PTS paper on our exact model: standard KA 80:20
LJ, rho=1.2). Their overlap is a COARSE-GRAINED BOX-OCCUPATION product, NOT our species-matched
nearest-neighbour distance:
    q(R) = (l^3 N_box)^-1  sum_i  n_i^ref n_i^gen ,   n_i in {0,1} = box-i occupied
with N_box = # cubic cells (side l) whose center lies inside the cavity |x|<R. Two IDENTICAL configs give
q = rho (=1.2, the ceiling); two INDEPENDENT configs give the bulk q_bulk, which we MEASURE empirically
(reference-vs-other-data-config, the Cavagna random-config test) rather than assume, so the result is
robust to their exact (l^3 N_box) normalisation. Report the bulk-subtracted q~ = q - q_bulk, which is what
their Fig 2 plots. l is an SI detail (not in the 4-page main text) -> SWEEP l in {0.2,0.3,0.4,0.5}.

Runs on island_smc_production.pt (already on disk); rebuilds each deterministic cavity to recover the
reference interior. NON-species-resolved to match Hocky (their n_i counts ANY particle); also reports a
species-resolved variant. Compare island-reweighted q~(R) to the paper's T=0.55 LJ anchor (Fig 2c ~ 0.30
at R=2.2-2.4). Caveats printed inline: their T=0.55 vs our 0.5; their l unknown."""
import sys, statistics as st
import torch
sys.path.insert(0, "reports/logs-2026-07-13")
from run_island_smc import build_cavity   # deterministic cavity == the one island-SMC ran

dev = "cpu"
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].float(), D["s"].long(), float(D["L"])
RES = torch.load("reports/logs-2026-07-13/island_smc_production.pt", map_location=dev, weights_only=False)


def box_ids(x, l, R):
    """Cell index (integer 3-vector packed to one int) for each particle in the |x|<R cavity, spacing l."""
    m = x.norm(dim=-1) < R
    ijk = torch.floor(x[m] / l).long()                         # signed cell coords
    off = ijk - ijk.min(0).values                              # non-negative
    span = off.max(0).values + 1
    return (off[:, 0] * span[1] * span[2] + off[:, 1] * span[2] + off[:, 2]).tolist()


def n_boxes_in_sphere(l, R):
    """# cubic cells whose CENTER lies within radius R (the N_box normaliser)."""
    g = torch.arange(-int(R / l) - 1, int(R / l) + 2) * l + l / 2
    gx, gy, gz = torch.meshgrid(g, g, g, indexing="ij")
    return int((gx**2 + gy**2 + gz**2 < R * R).sum())


def box_overlap(xa, xb, R, l, sa=None, sb=None):
    """(l^3 N_box)^-1 * #cells occupied in BOTH. If species given, require SAME species in the shared cell."""
    Nb = n_boxes_in_sphere(l, R)
    if sa is None:
        A, B = set(box_ids(xa, l, R)), set(box_ids(xb, l, R))
        shared = len(A & B)
    else:
        ma, mb = xa.norm(dim=-1) < R, xb.norm(dim=-1) < R
        A = set(zip(box_ids(xa, l, R), sa[ma].tolist()))
        B = set(zip(box_ids(xb, l, R), sb[mb].tolist()))
        shared = len(A & B)
    return shared / (l**3 * Nb)


ANCHOR = {2.2: 0.33, 2.4: 0.29}   # LJ T=0.55 q~ read off Fig 2c (by eye, +-0.05); T=0.5 slightly higher
LS = [0.2, 0.3, 0.4, 0.5]
print("=== PTS overlap in Hocky box-occupation convention on island-SMC populations ===", flush=True)
print("    (q~ = q - q_bulk; q_bulk = ref-vs-independent-data; ceiling q(self)=rho=1.2)", flush=True)
print(f"    ANCHOR (paper, LJ T=0.55, Fig 2c, by-eye +-0.05): {ANCHOR}", flush=True)
print(f"    CAVEATS: paper T=0.55 vs ours 0.5 (ours colder -> q~ slightly higher); their box size l is an SI", flush=True)
print(f"    detail we sweep. Non-species (Hocky) is the headline; species-resolved shown for reference.\n", flush=True)

for (R, c), v in RES.items():
    isl = v["islands"]
    xo, so, bnd, sb, xin, sin = build_cavity(X, S, L, R, c)
    xin = xin.float(); sin = sin.long()
    # bulk baseline: reference vs OTHER data configs' interiors at the SAME cavity geometry (independent)
    others = [build_cavity(X, S, L, R, cc)[4:6] for cc in range(2, 8)]
    # island reweight by exp(logZ_j)
    lz = torch.tensor([i["logZ"] for i in isl]); wj = torch.softmax(lz, 0)
    print(f"R={R} c={c}  ({len(isl)} islands, ref n_in={xin.shape[0]})", flush=True)
    for l in LS:
        Nb = n_boxes_in_sphere(l, R)
        qbulk = st.mean([box_overlap(xin, xo2.float(), R, l) for (xo2, _) in others])
        qself = box_overlap(xin, xin, R, l)
        # per-island mean box-overlap of generated sample vs reference, then reweight
        qgen_isl = []
        for i in isl:
            Xg, Sg = i["X"].float(), i["S"].long()
            qs = [box_overlap(xin, Xg[k], R, l) for k in range(Xg.shape[0])]
            qgen_isl.append(st.mean(qs))
        qgen = float(sum(wj[j] * qgen_isl[j] for j in range(len(isl))))
        qtil = qgen - qbulk
        anch = ANCHOR.get(R)
        astr = f"  [paper~{anch:.2f}]" if (anch and abs(l - 0.3) < 1e-9) else ""
        print(f"    l={l:.1f}: q(self)={qself:.3f}  q_bulk={qbulk:.3f}  q_gen={qgen:.3f}  "
              f"-> q~_gen={qtil:+.3f}{astr}", flush=True)
    # species-resolved at l=0.3 for reference
    l = 0.3
    qbulk_s = st.mean([box_overlap(xin, xo2.float(), R, l, sin, ss2.long()) for (xo2, ss2) in others])
    qgen_isl_s = []
    for i in isl:
        Xg, Sg = i["X"].float(), i["S"].long()
        qgen_isl_s.append(st.mean([box_overlap(xin, Xg[k], R, l, sin, Sg[k]) for k in range(Xg.shape[0])]))
    qgen_s = float(sum(wj[j] * qgen_isl_s[j] for j in range(len(isl))))
    print(f"    [species-resolved l=0.3]: q_bulk={qbulk_s:.3f}  q_gen={qgen_s:.3f}  q~={qgen_s-qbulk_s:+.3f}", flush=True)
    print(f"    [old species-min-dist<0.3 q_island_reweighted]: {v.get('q_island_reweighted', float('nan')):.3f}\n", flush=True)
