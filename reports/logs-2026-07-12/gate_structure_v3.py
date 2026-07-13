"""TASK 6 structure-fix gate, v3 = crash-proof rerun of gate_structure.py (same RNG stream, same metrics).

v2 lessons wired in:
  - _mic gets the BOX LENGTH L (v1 passed the cavity radius 2.0 -> whole box folded into [-1,1]^3:
    +-1e22 energies identical across arms, whole-exterior bnd, 4082-particle cage -> OOM).
  - energies/floors use the rmax=5.0 local cage (exact for block terms; avoids the full-cavity O(m^2)).
  - per-cavity try/except: a stiff/garbage AR draw can collapse dopri5's step size and raise
    AssertionError at max_steps (documented in CavityBlockFlow); v2 died with NO summary after cav 11.
    Here a failed cavity prints its traceback and is SKIPPED; the summary always prints.
  - frame sanity asserts on the DATA arm (bnd count, finite data energy) so a frame bug can never
    again masquerade as a model result.
"""
import traceback
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; RCTX = 2.5; BIGL = 100.0; K = 8; M = 8; POS_TEMP = 0.4; N_CAGE = 48; DUMMY_R = 50.0
FLOW_CKPT = "liquid_coupling_flow/artifacts/ka3d_cavity_egnn_flow_best.pt"


def fixed_size_cage(cage_x, sp_cage, centroid, n_cage):
    """48-nearest-to-centroid truncation + far-dummy padding (replicates train_cavity_egnn_flow.fixed_size_cage)."""
    Mn, m = cage_x.shape[0], cage_x.shape[1]
    if m >= n_cage:
        idx = torch.topk((cage_x[0] - centroid).norm(dim=-1), n_cage, largest=False).indices
        return cage_x[:, idx], sp_cage[:, idx]
    pad = n_cage - m
    dpos = torch.tensor([DUMMY_R, 0., 0.], device=dev, dtype=cage_x.dtype)[None, None].expand(Mn, pad, 3)
    return torch.cat([cage_x, dpos], 1), torch.cat([sp_cage, torch.zeros(Mn, pad, dtype=sp_cage.dtype, device=dev)], 1)


def local_cage(cage_x, cage_s, centroid, rmax=5.0):
    """Restrict the cage to particles within rmax of the block centroid: exact for the BLOCK energy (all
    block-cage pairs are < r_cut=2.5 << rmax) and O(local) instead of O(full-boundary^2)."""
    keep = (cage_x[0] - centroid).norm(dim=-1) < rmax
    return cage_x[:, keep], cage_s[:, keep]


def block_E_per(xb, sb, cage_x, cage_s):
    """TRUE KA energy of {block U cage} per config [M], double precision (clashy r->0 stays finite)."""
    allx = torch.cat([xb, cage_x], 1).double(); alls = torch.cat([sb, cage_s], 1)
    return ka_energy(allx, alls.long(), BIGL)


def lbfgs_floor(xb0, sb, cage_x, cage_s, iters=120):
    x = xb0.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([x], lr=0.3, max_iter=iters, line_search_fn="strong_wolfe", tolerance_grad=1e-6)

    def closure():
        opt.zero_grad(); e = block_E_per(x, sb, cage_x, cage_s).sum(); e.backward(); return e
    opt.step(closure)
    with torch.no_grad():
        return block_E_per(x, sb, cage_x, cage_s)


def clash_rate(xb, cage_x, cut=0.8):
    Kk = xb.shape[1]; allx = torch.cat([xb, cage_x], 1)
    dm = torch.cdist(xb, allx); dm[:, torch.arange(Kk), torch.arange(Kk)] = 9.0
    return float((dm.min(2).values < cut).float().mean())


def gbb_hist(blocks_x, blocks_s, cage_x, cage_s, edges):
    counts = torch.zeros(len(edges) - 1)
    for m in range(blocks_x.shape[0]):
        bx, bs = blocks_x[m], blocks_s[m]; ax = torch.cat([bx, cage_x[m]]); as_ = torch.cat([bs, cage_s[m]])
        bmask = bs == 1
        if bmask.sum() == 0:
            continue
        d = torch.cdist(bx[bmask], ax[as_ == 1])
        d = d[d > 1e-6]
        counts += torch.histc(d.cpu(), bins=len(edges) - 1, min=float(edges[0]), max=float(edges[-1]))
    return counts


ck = torch.load(FLOW_CKPT, map_location=dev, weights_only=False)
a = ck["args"]
flow = CavityBlockFlow(k=a["k"], n_cage=a["n_cage"], r_c=a.get("r_c", 2.5), hidden_nf=a["hidden_nf"],
                       n_layers=a["n_layers"], n_species=2, max_neighbors=a["max_neighbors"]).to(dev).double()
flow.load_state_dict(ck["state_dict"]); flow.eval()
print(f"flow ckpt step {ck['step']} held_fm {ck['held_loss']:.4f}  (k={a['k']} n_cage={a['n_cage']} hid={a['hidden_nf']} L={a['n_layers']})", flush=True)

ar_ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt", map_location=dev, weights_only=False)
ar = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
ar.load_state_dict(ar_ck["state_dict"], strict=False)
ar.eval(); ar.use_frame = False

d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
gen = torch.Generator(device=dev).manual_seed(0)

res = {k: [] for k in ["dE_data", "dE_ar", "dE_flow", "fl_data", "fl_ar", "fl_flow",
                        "cl_ar", "cl_flow", "div_ar", "div_flow"]}
edges = torch.linspace(0.6, 2.6, 41)
gbb = {"data": torch.zeros(40), "ar": torch.zeros(40), "flow": torch.zeros(40)}
ncav = 0; nfail = 0
for ci in range(16):                                                        # held reference chains (snapshot 1)
    # RNG draws BEFORE the try so a failed cavity consumes the same stream as v2 (cav 1-11 reproduce)
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, 2.0, L)
    if p["n_in"] < K + 6:
        continue
    try:
        # _mic's 3rd arg is the BOX LENGTH (NOT the cavity radius) — see module docstring
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], 2.0)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (2.0 + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        assert bnd.shape[0] < 600, f"frame sanity: n_bnd={bnd.shape[0]} (expected ~400; folded frame?)"
        n = xo.shape[0]; anch = fixed_ball_scaffold(n, 2.0, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
        block_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        block_mask[(anch - anch[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
        xo_b = xo[None].expand(M, n, 3).contiguous(); so_b = so[None].expand(M, n).contiguous()
        # AR@0.4 sample
        xo_ar, so_ar, logq_ar = ar.sample_block_b(xo_b, so_b, block_mask, bnd, sb, 2.0, gen=gen, pos_temp=POS_TEMP)
        ar_blk = xo_ar[:, block_mask]; sp_blk = so_ar[:, block_mask]
        ret_x = xo_ar[:, ~block_mask]; ret_s = so_ar[:, ~block_mask]
        cage_full = torch.cat([bnd[None].expand(M, -1, -1), ret_x], 1)          # FULL cage (for the flow's truncation)
        cage_s_full = torch.cat([sb[None].expand(M, -1), ret_s], 1)
        centroid0 = xo[block_mask].mean(0)
        cage_en, cage_s_en = local_cage(cage_full, cage_s_full, centroid0, rmax=5.0)   # local cage for ENERGY (exact)
        # flow correction (48-nearest cage for the flow's message passing only)
        centroid = xo[block_mask].mean(0)
        cage_fix, cage_s_fix = fixed_size_cage(cage_full, cage_s_full, centroid, N_CAGE)
        x1_blk, _ = flow.composed_logq(ar_blk.double(), logq_ar.double(), cage_fix.double(), sp_blk, cage_s_fix, R=2.0)
        flow_blk = x1_blk.float()
        # data block
        data_blk = xo[None, block_mask].expand(M, K, 3).contiguous(); data_s = so[None, block_mask].expand(M, K)
        Ed = block_E_per(data_blk, data_s, cage_en, cage_s_en).mean().item()
        assert abs(Ed) < 1e6, f"frame sanity: |E_data|={Ed:.3g} not physical (folded frame?)"
        res["dE_data"].append(0.0)
        res["dE_ar"].append((block_E_per(ar_blk, sp_blk, cage_en, cage_s_en).mean().item() - Ed) / K)
        res["dE_flow"].append((block_E_per(flow_blk, sp_blk, cage_en, cage_s_en).mean().item() - Ed) / K)
        res["fl_data"].append((lbfgs_floor(data_blk, data_s, cage_en, cage_s_en).mean().item() - Ed) / K)
        res["fl_ar"].append((lbfgs_floor(ar_blk, sp_blk, cage_en, cage_s_en).mean().item() - Ed) / K)
        res["fl_flow"].append((lbfgs_floor(flow_blk, sp_blk, cage_en, cage_s_en).mean().item() - Ed) / K)
        res["cl_ar"].append(clash_rate(ar_blk, cage_en)); res["cl_flow"].append(clash_rate(flow_blk, cage_en))

        # per-cage diversity: cross-chain overlap (fraction of block particles within 0.3 across chains) -> ->1 = collapsed
        def xdiv(bx):
            return st.mean([float((torch.cdist(bx[i], bx[j]).min(1).values < 0.3).float().mean())
                            for i in range(M) for j in range(i + 1, M)])
        res["div_ar"].append(xdiv(ar_blk)); res["div_flow"].append(xdiv(flow_blk))
        for key, bx in (("data", data_blk), ("ar", ar_blk), ("flow", flow_blk)):
            sblk = data_s if key == "data" else sp_blk
            gbb[key] += gbb_hist(bx, sblk, cage_full, cage_s_full, edges)
        ncav += 1
        print(f"  cav {ncav}: dE/p AR {res['dE_ar'][-1]:+8.1f} FLOW {res['dE_flow'][-1]:+8.1f} | "
              f"floor AR {res['fl_ar'][-1]:+6.2f} FLOW {res['fl_flow'][-1]:+6.2f} DATA {res['fl_data'][-1]:+5.2f} | "
              f"clash AR {100*res['cl_ar'][-1]:.0f}% FLOW {100*res['cl_flow'][-1]:.0f}% | div AR {res['div_ar'][-1]:.2f} FLOW {res['div_flow'][-1]:.2f}", flush=True)
    except Exception as e:
        nfail += 1
        traceback.print_exc()
        # a partially-appended cavity would skew medians: drop any keys longer than the shortest
        nmin = min(len(v) for v in res.values())
        for kk in res:
            del res[kk][nmin:]
        print(f"  cav idx {ci}: FAILED ({type(e).__name__}: {e}) -> skipped", flush=True)
        continue

md = lambda k: st.median(res[k])
print(f"\n=== TASK 6 STRUCTURE-FIX GATE ({ncav} held cavities, {nfail} failed/skipped, K={K}, pos_temp={POS_TEMP}) ===", flush=True)
print(f"  dE/particle above data:   AR {md('dE_ar'):+9.1f}   FLOW {md('dE_flow'):+9.1f}   (data 0)", flush=True)
print(f"  L-BFGS floor/particle:    AR {md('fl_ar'):+7.2f}   FLOW {md('fl_flow'):+7.2f}   DATA {md('fl_data'):+6.2f}", flush=True)
print(f"  clash rate:               AR {100*md('cl_ar'):3.0f}%   FLOW {100*md('cl_flow'):3.0f}%", flush=True)
print(f"  cross-chain overlap (->1 collapsed):  AR {md('div_ar'):.2f}   FLOW {md('div_flow'):.2f}", flush=True)
ctr = 0.5 * (edges[:-1] + edges[1:])
for key in ("data", "ar", "flow"):
    pk = ctr[gbb[key].argmax()].item() if gbb[key].sum() > 0 else float("nan")
    print(f"  g_BB peak location ({key}): {pk:.2f}", flush=True)
torch.save({"res": res, "gbb": gbb, "edges": edges, "ncav": ncav, "nfail": nfail,
            "K": K, "M": M, "pos_temp": POS_TEMP, "flow_step": ck["step"]},
           "reports/logs-2026-07-12/gate_structure_v3.pt")
print("saved -> reports/logs-2026-07-12/gate_structure_v3.pt", flush=True)
beat = md("fl_flow") < md("fl_ar") - 1.0
clean = md("dE_flow") < md("dE_ar")
divok = md("div_flow") < 0.85
print("\nGATE:", "PASS -- flow BEATS the AR structure floor toward data" if (beat and clean and divok)
      else "SEE GATES -- flow does NOT clearly beat the AR floor (structure fix weak/absent)", flush=True)
