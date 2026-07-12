"""DECISIVE gate for the physics-relaxation corrector, in TRUE KA ENERGY (not the soft-core proxy).

The old large-K failure: AR blocks are clashy -> block energy sits ~+40-48/particle ABOVE the equilibrium
(data) block, so the exact MTM samples high-energy garbage. Here we ask: does the deterministic declash
relaxation bring the block's TRUE KA energy down toward equilibrium, while (a) staying diverse (multi-basin,
not collapsed) and (b) keeping the map INJECTIVE (every per-step I-eta*H positive-definite => exact logdet,
diffeomorphism => carried-latent weight logu = -beta*U(y) - logq_AR(x0) + logdet is exact)?

Reference config for each cavity = the equilibrium DATA interior; cage = {boundary} U {retained data interior}
is IDENTICAL across data/raw/relaxed, so ka_energy(block+cage) differences are block-only. We report
per-block-particle energy ABOVE the data block."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; R = 2.0; RCTX = 2.5; K = 12; M = 16; CUT = 0.8; BIGL = 100.0
# chosen declash flow: gentle, NET-CONTRACTING (negative logdet => injective => exact); clash ~11% < 29% floor
S_SC, A_SC, W_SC, P_SC, T_SC, ETA_SC = 0.95, 0.45, 6.0, 3, 30, 8e-4


def load_base():
    ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax.pt", map_location=dev, weights_only=False)
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
    return m


def softcore_U(xb, cage_x):
    allx = torch.cat([xb, cage_x], 1); Kk, P = xb.shape[1], allx.shape[1]
    r2 = (xb[:, :, None, :] - allx[:, None, :, :]).square().sum(-1)
    r2 = r2.masked_fill(torch.eye(Kk, P, device=dev, dtype=torch.bool)[None], 1e18)
    return 0.5 * W_SC * (S_SC * S_SC / (r2 + A_SC * A_SC)).pow(P_SC).sum((1, 2))


def relax(xb, cage_x, T=T_SC, eta=ETA_SC, check_inj=False):
    """T steps x <- x - eta*grad(U); exact logdet = sum log|det(I-eta*H)|; min per-step eigenvalue if asked."""
    M = xb.shape[0]; Kk = xb.shape[1]; logdet = xb.new_zeros(M); mineig = xb.new_full((M,), 1e9)
    for _ in range(T):
        x = xb.detach().requires_grad_(True)
        g = torch.autograd.grad(softcore_U(x, cage_x).sum(), x, create_graph=True)[0]
        flat_g = g.reshape(M, 3 * Kk)
        H = torch.stack([torch.autograd.grad(flat_g[:, i].sum(), x, retain_graph=True)[0].reshape(M, 3 * Kk)
                         for i in range(3 * Kk)], 1)
        with torch.no_grad():
            Jm = torch.eye(3 * Kk, device=dev)[None] - eta * H
            logdet = logdet + torch.slogdet(Jm)[1]
            if check_inj:
                mineig = torch.minimum(mineig, torch.linalg.eigvalsh(0.5 * (Jm + Jm.transpose(1, 2)))[:, 0])
            xb = xb - eta * g
    return xb.detach(), logdet.detach(), mineig.detach()


def block_energy(xblk, sblk, cage_x, cage_s):
    """TRUE KA energy of {block U cage} per config [M] (cage-cage constant cancels in block comparisons)."""
    allx = torch.cat([xblk, cage_x], 1); alls = torch.cat([sblk, cage_s], 1)
    return ka_energy(allx, alls.long(), BIGL)


def clash_div(xblk, cage_x, blk_all_x=None):
    Mn = xblk.shape[0]; allx = torch.cat([xblk, cage_x], 1); Kk = xblk.shape[1]
    dm = torch.cdist(xblk, allx)                                                        # [M,K,K+cage]
    dm[:, torch.arange(Kk), torch.arange(Kk)] = 9.0
    clash = float((dm.min(2).values < CUT).float().mean())
    div = [float((torch.cdist(xblk[i], xblk[j]).min(1).values < 0.3).float().mean())
           for i in range(min(Mn, 8)) for j in range(i + 1, min(Mn, 8))]
    return clash, (st.mean(div) if div else 0.0)


m = load_base()
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"]); gen = torch.Generator(device=dev).manual_seed(0)
dE_raw, dE_rel, c_raw, c_rel, div_raw, div_rel, minei, ncav = [], [], [], [], [], [], [], 0
for ci in range(900, 970):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < K + 6:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)                  # data interior (labeled)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev); blk[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    order = torch.argsort(blk.to(torch.uint8), stable=True); n_ret = int((~blk).sum())
    xo_re, so_re = xo[order], so[order]
    # cage = boundary U retained DATA interior (identical across data/raw/relaxed)
    cage_x = torch.cat([bnd[None].expand(M, bnd.shape[0], 3), xo_re[None, :n_ret].expand(M, n_ret, 3)], 1)
    cage_s = torch.cat([sb[None].expand(M, bnd.shape[0]), so_re[None, :n_ret].expand(M, n_ret)], 1)
    # equilibrium DATA block energy (reference)
    data_blk = xo_re[None, n_ret:].expand(M, K, 3); data_bs = so_re[None, n_ret:].expand(M, K)
    E_data = block_energy(data_blk, data_bs, cage_x, cage_s).mean().item()
    # AR proposal block (raw) + relaxed
    x0, s0, _ = m.sample_block_b(xo_re[None].expand(M, n, 3), so_re[None].expand(M, n), blk[order], bnd, sb, R, gen=gen)
    xb0 = x0[:, n_ret:]; sblk = s0[:, n_ret:]
    E_raw = block_energy(xb0, sblk, cage_x, cage_s)
    xb1, ld, me = relax(xb0, cage_x, check_inj=True)
    E_rel = block_energy(xb1, sblk, cage_x, cage_s)
    dE_raw.append((E_raw.mean().item() - E_data) / K); dE_rel.append((E_rel.mean().item() - E_data) / K)
    cr, dr = clash_div(xb0, cage_x); cc, dc = clash_div(xb1, cage_x)
    c_raw.append(cr); c_rel.append(cc); div_raw.append(dr); div_rel.append(dc); minei.append(me.min().item())
    ncav += 1
    if ncav >= 12:
        break

print(f"K={K}, {ncav} equilibrium cavities  (flow p={P_SC} a={A_SC} w={W_SC:.0f} eta={ETA_SC:.0e} T={T_SC})", flush=True)
print(f"  dE/block-particle ABOVE equilibrium:  raw {st.mean(dE_raw):+6.2f}  ->  relaxed {st.mean(dE_rel):+6.2f}", flush=True)
print(f"  clash:     raw {100*st.mean(c_raw):3.0f}%  ->  relaxed {100*st.mean(c_rel):3.0f}%   (full-cage floor 29%)", flush=True)
print(f"  diversity: raw {100*st.mean(div_raw):3.0f}%  ->  relaxed {100*st.mean(div_rel):3.0f}%   (->100 = collapsed)", flush=True)
print(f"  INJECTIVITY (min per-step eig of I-eta*H over all cavities): {min(minei):+.4f}  ({'OK diffeo' if min(minei) > 0 else 'FOLD -> not exact'})", flush=True)
ok = st.mean(dE_rel) < 5.0 and 100 * st.mean(c_rel) < 29 and 100 * st.mean(div_rel) < 85 and min(minei) > 0
print("VERDICT:", "PASS -> clean low-energy diverse EXACT proposal; build the MTM" if ok else "see gates", flush=True)
