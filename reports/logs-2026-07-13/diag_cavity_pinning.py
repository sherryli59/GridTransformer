"""CORRECTED basin probe: the uniform-seed version conflated glassy under-equilibration with basins
(gave fake multi-basin at R=2.0, contradicting the oracle 2%-rearrangement pinned finding). Proper PTS
pinning measurement starts from the ALREADY-EQUILIBRATED data config and watches the overlap decay under
frozen-boundary T=0.5 MC:
  PINNED (R<xi):  overlap(config_t, data) plateaus HIGH -> the config wiggles inside the data basin.
  UNPINNED (R>xi): overlap DECAYS toward the independent-config floor q_inf -> it explores other basins.
q_inf baseline = overlap between two INDEPENDENT equilibrium configs of the SAME cage, approximated by the
overlap of the data interior against a uniform-random interior (accidental overlap). Confinement: radial
clamp (starting from data the interior is already bound, so clamp is nearly inactive)."""
import torch, statistics as st
from liquid_coupling_flow.ka_pmc_3d import parallel_mc_disp
from liquid_coupling_flow.ka_cavity_3d import local_identity_swap
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda"; RCTX = 2.5; BETA = 2.0; BIGL = 100.0; STEP = 0.05; N_SWEEP = 300; REC = 30; A_OVL = 0.3
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
gen = torch.Generator(device=dev).manual_seed(0)


def overlap(xi, si, xj, sj):
    d = torch.cdist(xi, xj).masked_fill(si[:, None] != sj[None, :], 9.0)
    return (d.min(1).values < A_OVL).float().mean()


print("=== CAVITY PINNING (overlap decay from DATA start, frozen-boundary T=0.5) ===", flush=True)
for R in (2.0, 3.0, 4.0):
    traj = {t: [] for t in range(0, N_SWEEP + 1, REC)}; qinf = []
    for trial in range(4):
        while True:
            c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[trial], S[trial], c, R, L)
            if p["n_in"] >= 12:
                break
        xin = _mic(p["x_in"], c, L); sin = p["s_in"].long(); ni = xin.shape[0]
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm].long()
        data_x = xin.clone()
        off = BIGL / 2; nb = bnd.shape[0]
        x = torch.cat([xin, bnd], 0)[None].clone() + off; s = torch.cat([sin, sb], 0)[None].clone()
        mobile = torch.zeros(1, ni + nb, dtype=torch.bool, device=dev); mobile[:, :ni] = True
        ctr = torch.full((1, 3), off, device=dev)
        # q_inf baseline: data vs uniform-random interior (accidental same-species overlap)
        uni = (torch.rand(ni, 3, generator=gen, device=dev) * 2 - 1); uni = uni * (R * 0.9 / uni.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
        qinf.append(float(overlap(data_x, sin, uni, sin)))
        for sw in range(N_SWEEP + 1):
            if sw in traj:
                traj[sw].append(float(overlap(x[0, :ni] - off, s[0, :ni], data_x, sin)))
            x = parallel_mc_disp(x, s, BIGL, BETA, STEP, mobile=mobile)
            if sw % 5 == 0:
                s, _, _ = local_identity_swap(x, s, torch.zeros(1, device=dev), mobile, BETA, BIGL)
            rel = x[:, :ni] - ctr[:, None]; rr = rel.norm(dim=-1, keepdim=True)
            x[:, :ni] = ctr[:, None] + rel * (R * 0.999 / rr.clamp_min(1e-9)).clamp(max=1.0)
    ts = sorted(traj)
    plate = st.mean(traj[ts[-1]])
    q0 = st.mean(traj[0]); qi = st.mean(qinf)
    verdict = "PINNED" if (plate - qi) > 0.5 * (q0 - qi) else "UNPINNED (decays to independent floor)"
    curve = "  ".join(f"{t}:{st.mean(traj[t]):.2f}" for t in ts[::5])
    print(f"R={R}: q(t) {curve}  | plateau {plate:.2f} vs q_inf {qi:.2f} (start {q0:.2f}) -> {verdict}", flush=True)
torch.save({"note": "pinning decay"}, "reports/logs-2026-07-13/diag_cavity_pinning.pt")
print("saved", flush=True)
