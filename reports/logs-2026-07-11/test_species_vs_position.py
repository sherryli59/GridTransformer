"""LOCALIZE the AR block frustration: is it the SPECIES-per-slot choice or the position/topology?

Four inherent-structure floors (L-BFGS on TRUE KA energy, cage frozen), same slots each time:
  A  AR pos   + AR species    (measured HIGH, +40..328/part)
  B  AR pos   + DATA species  -> if LOW: AR positions are fine, AR SPECIES is the culprit
  C  DATA pos + AR species    -> if HIGH: AR species incompatible even with perfect positions => SPECIES culprit
  D  DATA pos + DATA species  (control, ~-0.7/part)
Species are per-slot (scaffold labels), so B/C just swap the species vector on the same slots. Decisive for
where the base fix must go: a full-cage SPECIES proposal vs a position corrector."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; R = 2.0; RCTX = 2.5; K = 12; M = 8; BIGL = 100.0


def load_base():
    ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax.pt", map_location=dev, weights_only=False)
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
    return m


def block_E(xblk, sblk, cage_x, cage_s):
    allx = torch.cat([xblk, cage_x], 1); alls = torch.cat([sblk, cage_s], 1)
    return ka_energy(allx, alls.long(), BIGL)


def floor(xb0, sblk, cage_x, cage_s, iters=120):
    x = xb0.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([x], lr=0.4, max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad(); e = block_E(x, sblk, cage_x, cage_s).sum(); e.backward(); return e
    opt.step(closure)
    with torch.no_grad():
        return block_E(x, sblk, cage_x, cage_s).mean().item()


m = load_base()
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"]); gen = torch.Generator(device=dev).manual_seed(0)
A, B, C, D, mism, ncav = [], [], [], [], [], 0
for ci in range(900, 970):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < K + 6:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev); blk[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    order = torch.argsort(blk.to(torch.uint8), stable=True); n_ret = int((~blk).sum()); blk_re = blk[order]
    xo_re, so_re = xo[order], so[order]
    cage_x = torch.cat([bnd[None].expand(M, bnd.shape[0], 3), xo_re[None, :n_ret].expand(M, n_ret, 3)], 1)
    cage_s = torch.cat([sb[None].expand(M, bnd.shape[0]), so_re[None, :n_ret].expand(M, n_ret)], 1)
    data_blk = xo_re[None, n_ret:].expand(M, K, 3).contiguous(); data_bs = so_re[None, n_ret:].expand(M, K)
    E_data = block_E(data_blk, data_bs, cage_x, cage_s).mean().item()
    x0, s0, _ = m.sample_block_b(xo_re[None].expand(M, n, 3), so_re[None].expand(M, n), blk_re, bnd, sb, R, gen=gen)
    ar_blk = x0[:, n_ret:].contiguous(); ar_bs = s0[:, n_ret:]
    # species mismatch rate: AR species-per-slot vs data species-per-slot
    mism.append(float((ar_bs != data_bs).float().mean()))
    A.append((floor(ar_blk, ar_bs, cage_x, cage_s) - E_data) / K)
    B.append((floor(ar_blk, data_bs, cage_x, cage_s) - E_data) / K)
    C.append((floor(data_blk, ar_bs, cage_x, cage_s) - E_data) / K)
    D.append((floor(data_blk, data_bs, cage_x, cage_s) - E_data) / K)
    ncav += 1
    print(f"  cav {ncav}: A(ARpos+ARsp) {A[-1]:+8.1f}  B(ARpos+DATAsp) {B[-1]:+7.2f}  C(DATApos+ARsp) {C[-1]:+8.1f}  D {D[-1]:+5.2f}  | sp-mismatch {100*mism[-1]:.0f}%", flush=True)
    if ncav >= 10:
        break


def med(v):
    return st.median(v)


print(f"\nK={K}, {ncav} cavities  (median dE/block-particle above equilibrium)", flush=True)
print(f"  A  AR pos   + AR species    {med(A):+8.2f}", flush=True)
print(f"  B  AR pos   + DATA species  {med(B):+8.2f}   (LOW => AR species is the culprit)", flush=True)
print(f"  C  DATA pos + AR species    {med(C):+8.2f}   (HIGH => AR species incompatible even w/ perfect pos)", flush=True)
print(f"  D  DATA pos + DATA species  {med(D):+8.2f}   (control)", flush=True)
print(f"  species mismatch (AR vs data, per slot): {100*st.mean(mism):.0f}%", flush=True)
if med(B) < 5.0 and med(C) > 10.0:
    print("VERDICT: SPECIES is the frustration -> fix = full-cage SPECIES proposal; positions then relax clean", flush=True)
elif med(B) > 10.0:
    print("VERDICT: POSITION/TOPOLOGY frustration persists even with data species -> deeper than species", flush=True)
else:
    print("VERDICT: mixed -> see numbers", flush=True)
