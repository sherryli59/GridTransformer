"""WHY is even a clash-free K=8 block +1.4/particle above data? Decisive test: frozen-cage MC (freeze all
non-block; single-particle Metropolis on the K=8 block) from THREE starts, track dE/particle vs sweeps:
  DATA-start : if stays ~0 => data IS the constrained equilibrium => +1.4 is a proposal/relaxation failure;
               if drifts UP to +1.4 => +1.4 IS the constrained-ensemble energy (data co-relaxed w/ surroundings).
  AR-start   : reaches ~0 => relaxation works (AR under-relaxed); plateaus at +1.4 while data-start is flat
               => GLASSY TRAPPING (dynamic slowing down) -- MC stuck in a metastable basin, can't reach data.
  RAND-start : control (uniform-in-ball init).
K=8 blob, R=2.5, beta=2 (T*=0.5), single-particle displacement MC, step 0.12 sigma. Several cavities."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; RCTX = 2.5; R = 2.5; BIGL = 100.0; BETA = 2.0; K = 8; NCAV = 8
FT = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_blockcond_knn24_best.pt"
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24).to(dev)
m.load_state_dict(torch.load(FT, map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False


def block_energy(xblk, sblk, xfix, sfix):
    """Total energy of (block ++ frozen) config [1]. Frozen-frozen part constant -> cancels in dE."""
    x = torch.cat([xblk, xfix])[None]; s = torch.cat([sblk, sfix])[None]
    return ka_energy(x.double(), s.long(), BIGL).float()[0]


@torch.no_grad()
def mc_relax(x0, sblk, xfix, sfix, R, sweeps, gen, step=0.12):
    """Single-particle Metropolis on the block within |x|<R, frozen surroundings. Returns dE/particle traj
    + acceptance rate (overall and in the plateau window, last 25% of sweeps)."""
    x = x0.clone(); E = block_energy(x, sblk, xfix, sfix); traj = {}
    checkpoints = {0, 100, 500, 2000, sweeps}
    if 0 in checkpoints:
        traj[0] = float(E)
    acc = tot = acc_late = tot_late = 0
    for sw in range(1, sweeps + 1):
        late = sw > 0.75 * sweeps
        for i in range(K):
            xn = x.clone()
            xn[i] = x[i] + torch.randn(3, generator=gen, device=dev) * step
            tot += 1; tot_late += int(late)
            if xn[i].norm() >= R:                             # keep in ball (counts as rejected)
                continue
            En = block_energy(xn, sblk, xfix, sfix)
            if torch.rand((), generator=gen, device=dev).log() < (-BETA * (En - E)):
                x, E = xn, En; acc += 1; acc_late += int(late)
        if sw in checkpoints:
            traj[sw] = float(E)
    return x, traj, (100 * acc / tot, 100 * acc_late / max(tot_late, 1))


starts = {"DATA": [], "AR": [], "RAND": []}
gen = torch.Generator(device=dev).manual_seed(0)
SW = 2000
ncav = 0
for ci in range(16):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < K + 6:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
    seed = int(torch.randint(n, (), generator=gen, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev)
    blk[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    bidx = blk.nonzero().squeeze(1)
    xfix = torch.cat([xo[~blk], bnd]); sfix = torch.cat([so[~blk], sb])
    x_data = xo[bidx]; s_blk = so[bidx]
    E_data = block_energy(x_data, s_blk, xfix, sfix)
    # AR proposal
    Xp, Sp, _ = m.sample_block_b(xo[None], so[None], blk, bnd, sb, R, gen=gen)
    x_ar = Xp[0][bidx]
    # random-in-ball init (same species multiset)
    xr = torch.randn(K, 3, generator=gen, device=dev); xr = xr / xr.norm(dim=-1, keepdim=True) * (torch.rand(K, 1, generator=gen, device=dev) ** (1 / 3)) * R * 0.9
    for name, x0 in (("DATA", x_data), ("AR", x_ar), ("RAND", xr)):
        _, traj, accr = mc_relax(x0, s_blk, xfix, sfix, R, SW, gen)
        rec = {sw: (traj[sw] - float(E_data)) / K for sw in traj}
        rec["acc"], rec["acc_late"] = accr
        starts[name].append(rec)
    ncav += 1
    if ncav >= NCAV:
        break

print(f"=== frozen-cage MC dE/particle vs data (K={K}, beta={BETA}, {ncav} cav) ===", flush=True)
print(f"{'start':>6} | " + " ".join(f"sweep{sw:>5}" for sw in (0, 100, 500, 2000)) + " | acc% acc_late%", flush=True)
for name in ("DATA", "AR", "RAND"):
    row = f"{name:>6} | "
    for sw in (0, 100, 500, 2000):
        vals = [d[sw] for d in starts[name] if sw in d]
        row += f"  {st.mean(vals):>+8.2f}"
    row += f" | {st.mean([d['acc'] for d in starts[name]]):4.0f} {st.mean([d['acc_late'] for d in starts[name]]):5.0f}"
    print(row, flush=True)
print("\nREAD: DATA-start flat ~0 & AR/RAND plateau >0 => GLASSY TRAPPING (dynamic slowing down): the data"
      "\nbasin is stable but MC can't reach it from elsewhere. DATA-start drifts UP => +1.4 is the true"
      "\nconstrained-ensemble energy (data co-relaxed w/ now-frozen surroundings).", flush=True)
