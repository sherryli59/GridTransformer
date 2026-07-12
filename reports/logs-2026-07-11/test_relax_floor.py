"""CEILING PROBE: what is the true-KA-energy FLOOR that POSITION-ONLY relaxation of an AR block can reach?

Relaxing block positions (cage frozen) toward the inherent structure = L-BFGS minimize ka_energy(block+cage)
over block xyz. If the floor is near equilibrium (~0-5/particle above data), the frustration is POSITIONAL
=> a stable/exact relaxation integrator (backward Euler on true repulsion) is worth building. If the floor
stays high (+tens/particle), the frustration is in the AR base's SPECIES/TOPOLOGY choice (two incompatible
particles routed to one pocket by the half-cage conditional) => position relaxation cannot rescue it and the
fix must change the base, not add a position corrector. Decisive either way."""
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
    return ka_energy(allx, alls.long(), BIGL)                                           # [M]


def lbfgs_floor(xb0, sblk, cage_x, cage_s, iters=120):
    """Minimize true KA block energy over block positions (cage frozen). Returns floor energy per config."""
    x = xb0.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([x], lr=0.4, max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        e = block_E(x, sblk, cage_x, cage_s).sum()
        e.backward()
        return e
    opt.step(closure)
    with torch.no_grad():
        return block_E(x, sblk, cage_x, cage_s)


m = load_base()
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"]); gen = torch.Generator(device=dev).manual_seed(0)
dE_raw, dE_floor, dE_datafloor, ncav = [], [], [], 0
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
    data_blk = xo_re[None, n_ret:].expand(M, K, 3); data_bs = so_re[None, n_ret:].expand(M, K)
    E_data = block_E(data_blk, data_bs, cage_x, cage_s).mean().item()
    # AR proposal block (raw), then LBFGS floor (relax positions, KEEP AR species)
    x0, s0, _ = m.sample_block_b(xo_re[None].expand(M, n, 3), so_re[None].expand(M, n), blk_re, bnd, sb, R, gen=gen)
    xb0 = x0[:, n_ret:].contiguous(); sblk = s0[:, n_ret:]
    E_raw = block_E(xb0, sblk, cage_x, cage_s)
    E_floor = lbfgs_floor(xb0, sblk, cage_x, cage_s)
    # control: relax the DATA block from its own positions (should stay ~E_data => LBFGS is sane)
    E_dfloor = lbfgs_floor(data_blk.contiguous(), data_bs, cage_x, cage_s)
    dE_raw.append((E_raw.mean().item() - E_data) / K)
    dE_floor.append((E_floor.mean().item() - E_data) / K)
    dE_datafloor.append((E_dfloor.mean().item() - E_data) / K)
    ncav += 1
    print(f"  cav {ncav}: raw {dE_raw[-1]:+.1e}  AR-floor {dE_floor[-1]:+7.2f}  data-floor {dE_datafloor[-1]:+6.2f}  /part", flush=True)
    if ncav >= 10:
        break

print(f"\nK={K}, {ncav} cavities  (LBFGS position-only relaxation on TRUE KA energy)", flush=True)
print(f"  dE/block-particle above equilibrium:", flush=True)
print(f"    raw AR         {st.mean(dE_raw):+.2e}", flush=True)
print(f"    AR-floor       {st.mean(dE_floor):+7.2f}   (position-relaxed AR block, AR species kept)", flush=True)
print(f"    data-floor     {st.mean(dE_datafloor):+7.2f}   (control: relaxed data block => LBFGS sanity)", flush=True)
verdict = ("POSITIONAL frustration: relaxation CAN clean it -> build backward-Euler exact integrator"
           if st.mean(dE_floor) < 5.0 else
           "SPECIES/TOPOLOGY frustration: position relaxation floors HIGH -> fix the base, not a position corrector")
print("VERDICT:", verdict, flush=True)
