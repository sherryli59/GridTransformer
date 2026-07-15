"""BCY SHRINKAGE-PT baseline -- faithful replication of Berthier-Charbonneau-Yaida (JCP 144, 024501 (2016),
arXiv:1510.06320) cavity equilibration. THE classical baseline (no learned parts anywhere):
  - n replicas at (T_a, lambda_a): bottom (T,1.0) = target; up the ladder hotter + diameter-SHRUNK.
  - Deformed Hamiltonian V(r; lam~ sigma), shifted-LJ rc=2.5 lam~ sigma (our ka_energy convention with
    sigma -> lam~ sigma): lam~ = lambda_a for mobile-mobile, (1+lambda_a)/2 for mobile-pinned (their eq).
  - Moves: single-particle displacement l*nhat, l ~ U[0,0.3], hard wall |x|<R. N_cav moves per sweep.
    Species FROZEN (their displacement-only dynamics; note: our learned product resamples species).
  - Exchange: adjacent (T,lambda) Metropolis with cross-evaluated deformed energies, every EXCH sweeps.
  - Certificate (theirs): init-A = reference; init-B = randomized at TOP conditions (T=1.0, lambda=0.6,
    RAND_SW sweeps); converged when running q-bar agree within q_tol=0.1.
Deviations from the paper (noted): ladder linear T 0.5->1.0 x lambda 1.0->0.6 (their tables vary per size);
exchange every 10 sweeps (theirs ~1000, harmless-faster mixing); SW=20000 (theirs 1e4-1e6).
Observable: exact Hocky fixed-grid q~ on the bottom replica. Usage: [R] argv."""
import sys, functools, time, statistics as st
import torch
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA, EPS
torch.set_grad_enabled(False)

dev = "cuda"; RCTX = 2.5; RHO = 1.149
R = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
# BCY Table IV (T=0.51, R=2.0) MIDPOINT-DENSIFIED (probe: their spacing gives dlog ~ -9 at OUR state
# point -> marginal; halving dlam -> ~-4.5 -> 5-15% acc; their caveat endorses local retuning)
_BASE = [1.0000, 0.9825, 0.9640, 0.9450, 0.9250, 0.9050, 0.8850, 0.8640, 0.8423, 0.8200, 0.7960]
LAM_LADDER = []
for _i, _v in enumerate(_BASE):
    LAM_LADDER.append(_v)
    if _i + 1 < len(_BASE):
        LAM_LADDER.append(0.5 * (_v + _BASE[_i + 1]))
T_BOT = 0.5; T_DEC = 1.0; LAM_DEC = 0.8
NR = len(LAM_LADDER); NCH = 1; LAM_TOP = LAM_LADDER[-1]
SW = int(__import__('sys').argv[2]) if len(__import__('sys').argv) > 2 else 60000
EXCH_MEAN = 10;  # dense exchange (efficiency-only deviation, same invariant) REC = 1000; RAND_SW = 10000; Q_TOL = 0.1; STEP_MAX = 0.3  # per-PAIR Poisson exchange @1/1000 sw (their 'every 1000 sweeps on average'; per-pair reading)
L_HOCKY = (0.06 / RHO) ** (1.0 / 3.0); BULK_HOCKY = 0.06; RCUT_F = 2.5
OUT = f"reports/logs-2026-07-14/bcy_shrinkage_R{R}.pt"
ART = "liquid_coupling_flow/artifacts"
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
LAMs = torch.tensor(LAM_LADDER, device=dev)
Ts = T_BOT + (T_DEC - T_BOT) * (1.0 - LAMs) / (1.0 - LAM_DEC)          # their exact linear (T,lam) relation
T_TOP = float(Ts[-1])
BETAs = 1.0 / Ts


@functools.lru_cache(maxsize=16)
def n_boxes(l, R):
    g = torch.arange(-int(R / l) - 1, int(R / l) + 2) * l + l / 2
    gx, gy, gz = torch.meshgrid(g, g, g, indexing="ij")
    return int((gx ** 2 + gy ** 2 + gz ** 2 < R * R).sum())


def box_set(x, l, R):
    mm = x.norm(dim=-1) < R; ijk = torch.floor(x[mm] / l).long()
    h = int(R / l) + 2; side = 2 * h + 1
    return set(((ijk[:, 0] + h) * side * side + (ijk[:, 1] + h) * side + (ijk[:, 2] + h)).tolist())


def q_hocky(a, b, R):
    return len(box_set(a, L_HOCKY, R) & box_set(b, L_HOCKY, R)) / (L_HOCKY ** 3 * n_boxes(L_HOCKY, R))


def pair_e(r2, sig, eps):
    """shifted LJ at effective sigma (rc = 2.5 sigma), r2 [B,K]; sig,eps broadcastable."""
    inv6 = (sig ** 2 / r2.clamp_min(1e-12)) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    src6 = (1.0 / RCUT_F) ** 6                                  # (sig/rc)^6 with rc=2.5 sig
    eshift = 4 * eps * (src6 ** 2 - src6)
    return torch.where(r2 < (RCUT_F * sig) ** 2, e - eshift, torch.zeros_like(e))


class Cavity:
    """One cavity: batched BCY-PT state over B = 2 stacks x NR x NCH walkers, species frozen."""
    def __init__(self, xin, sin, bnd, sb, n):
        self.n = n; self.m = bnd.shape[0]; self.Nt = n + self.m
        self.bnd = bnd; self.xin0 = xin
        s_all = torch.cat([sin, sb]).long()
        t_sig = torch.tensor(SIGMA, device=dev); t_eps = torch.tensor(EPS, device=dev)
        self.sig0 = t_sig[s_all[:, None], s_all[None, :]]        # [Nt,Nt] base sigma
        self.eps0 = t_eps[s_all[:, None], s_all[None, :]]
        B = 2 * NR * NCH
        lam = LAMs[None, :, None].expand(2, NR, NCH).reshape(-1)  # [B]
        lam_pair = torch.empty(B, self.Nt, device=dev)            # lam~ for (mobile i, j)
        lam_pair[:, :n] = lam[:, None]                            # mobile-mobile
        lam_pair[:, n:] = (1 + lam[:, None]) / 2                  # mobile-pinned
        self.lam_pair = lam_pair; self.beta = BETAs[None, :, None].expand(2, NR, NCH).reshape(-1).clone()
        self.lam = lam

    def full_U(self, Xm, lam_vec=None, beta_dummy=None):
        """Deformed total energy involving MOBILE particles for each batch entry. Xm [B,n,3]."""
        B = Xm.shape[0]
        xa = torch.cat([Xm, self.bnd[None].expand(B, self.m, 3)], 1)         # [B,Nt,3]
        d2 = torch.cdist(xa, xa) ** 2
        eye = torch.eye(self.Nt, dtype=torch.bool, device=Xm.device)
        d2 = d2.masked_fill(eye[None], 1e12)                       # mask self-pairs BEFORE the LJ (float32
        # overflow at r2~0 made diag inf -> inf-inf = NaN -> every exchange silently rejected)
        lam_pair = self.lam_pair if lam_vec is None else self._pair(lam_vec)
        sig = self.sig0[None] * lam_pair[:, None, :]              # column-factor lam~ of partner j; only
        # mobile-row blocks are summed below, so (i mobile, j mobile)->lam, (i mobile, j pinned)->(1+lam)/2
        # NOTE sig row scaling: pair (i,j) uses lam~ of the MOBILE partner; mobile-mobile symmetric OK;
        # boundary-boundary rows get 1 but are excluded below anyway.
        e = pair_e(d2, sig, self.eps0[None])
        e = e - torch.diag_embed(torch.diagonal(e, dim1=1, dim2=2))
        e_mm = e[:, :self.n, :self.n].sum((1, 2)) * 0.5
        e_mb = e[:, :self.n, self.n:].sum((1, 2))
        return e_mm + e_mb

    def _pair(self, lam_vec):
        B = lam_vec.shape[0]
        lp = torch.empty(B, self.Nt, device=dev)
        lp[:, :self.n] = lam_vec[:, None]; lp[:, self.n:] = (1 + lam_vec[:, None]) / 2
        return lp

    def row_U(self, Xm, i, xi):
        """Pair-row energy of mobile particle i at position xi [B,3] vs all others (deformed)."""
        B = Xm.shape[0]
        xa = torch.cat([Xm, self.bnd[None].expand(B, self.m, 3)], 1)
        d2 = ((xa - xi[:, None]) ** 2).sum(-1)                    # [B,Nt]
        d2[:, i] = 1e12
        sig_row = self.sig0[i][None] * self.lam_pair              # lam~ of mobile i vs j: use j's class
        # for pair (i mobile, j mobile) lam~=lam ; (i mobile, j pinned) lam~=(1+lam)/2 -> lam_pair encodes by j
        eps_row = self.eps0[i][None]
        return pair_e(d2, sig_row, eps_row).sum(1)


results = {}
gen = torch.Generator(device=dev).manual_seed(0); ncav = 0
for ci in range(24):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < 14:
        continue
    t0 = time.time()
    xin = _mic(p["x_in"], c, L); sin = p["s_in"]
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xin.shape[0]
    cav = Cavity(xin, sin, bnd, sb, n)
    B = 2 * NR * NCH
    Xm = xin[None].expand(B, n, 3).clone()                                   # all walkers at reference
    g = torch.Generator(device=dev).manual_seed(1500 + ci)
    # init-B randomization at TOP conditions (T=1.0, lambda=0.6) for RAND_SW sweeps, stack-B walkers only
    top_beta = torch.full((B,), 1.0 / 1.00, device=dev)          # their randomization state:
    top_lam = torch.full((B,), 0.60, device=dev)                   # (T=1.00, lam=0.6), 1e4 sweeps
    lam_pair_save = cav.lam_pair; beta_save = cav.beta
    cav.lam_pair = cav._pair(top_lam); cav.beta = top_beta
    Bmask = torch.zeros(B, dtype=torch.bool, device=dev); Bmask.view(2, NR, NCH)[1] = True
    U = cav.full_U(Xm)
    for sw in range(RAND_SW):
        for i in torch.randperm(n, generator=g, device=dev).tolist():
            l = STEP_MAX * torch.rand(B, 1, device=dev, generator=g)
            nh = torch.randn(B, 3, device=dev, generator=g); nh = nh / nh.norm(dim=-1, keepdim=True)
            xi_new = Xm[:, i] + l * nh
            ok = (xi_new.norm(dim=-1) < R) & Bmask                            # only stack B moves
            dU = cav.row_U(Xm, i, xi_new) - cav.row_U(Xm, i, Xm[:, i])
            acc = ok & (torch.rand(B, device=dev, generator=g).log() < -cav.beta * dU)
            Xm[acc, i] = xi_new[acc]; U = torch.where(acc, U + dU, U)
    cav.lam_pair = lam_pair_save; cav.beta = beta_save
    U = cav.full_U(Xm)
    print(f"=== cav {ci} (n={n}, R={R}) BCY shrinkage-PT: NR={NR} (T 0.5-1.0 x lam 1.0-0.6), SW={SW}, "
          f"B randomized {RAND_SW}sw@top ===", flush=True)
    qA_run, qB_run, traj = [], [], []
    ex_acc = torch.zeros(NR - 1); ex_try = torch.zeros(NR - 1)
    tag = torch.arange(NR, device=dev)[None, :, None].expand(2, NR, NCH).clone()  # replica-flow labels
    seen_top = torch.zeros(2, NR, NCH, dtype=torch.bool, device=dev)
    trips = 0
    for sw in range(1, SW + 1):
        for i in torch.randperm(n, generator=g, device=dev).tolist():
            l = STEP_MAX * torch.rand(B, 1, device=dev, generator=g)
            nh = torch.randn(B, 3, device=dev, generator=g); nh = nh / nh.norm(dim=-1, keepdim=True)
            xi_new = Xm[:, i] + l * nh
            ok = xi_new.norm(dim=-1) < R
            dU = cav.row_U(Xm, i, xi_new) - cav.row_U(Xm, i, Xm[:, i])
            acc = ok & (torch.rand(B, device=dev, generator=g).log() < -cav.beta * dU)
            Xm[acc, i] = xi_new[acc]; U = torch.where(acc, U + dU, U)
        att = (torch.rand(NR - 1, generator=g, device=dev) < 1.0 / EXCH_MEAN).nonzero().squeeze(1).tolist()
        if att:                                                               # sparse Poisson per-pair exchange
            Xr = Xm.view(2, NR, NCH, n, 3)
            for r in att:
                xa = Xr[:, r].reshape(-1, n, 3); xb = Xr[:, r + 1].reshape(-1, n, 3)
                la_v = torch.full((xa.shape[0],), float(LAMs[r]), device=dev)
                lb_v = torch.full((xa.shape[0],), float(LAMs[r + 1]), device=dev)
                Uaa = cav.full_U(xa, la_v); Uab = cav.full_U(xb, la_v)
                Uba = cav.full_U(xa, lb_v); Ubb = cav.full_U(xb, lb_v)
                dlog = -BETAs[r] * (Uab - Uaa) - BETAs[r + 1] * (Uba - Ubb)
                swp = torch.rand(xa.shape[0], device=dev, generator=g).log() < dlog
                swp2 = swp.view(2, NCH)
                ex_acc[r] += float(swp2.float().sum()); ex_try[r] += swp2.numel()
                for stk in range(2):
                    for ch in range(NCH):
                        if swp2[stk, ch]:
                            tmp = Xr[stk, r, ch].clone(); Xr[stk, r, ch] = Xr[stk, r + 1, ch]; Xr[stk, r + 1, ch] = tmp
                            t1 = int(tag[stk, r, ch]); tag[stk, r, ch] = tag[stk, r + 1, ch]; tag[stk, r + 1, ch] = t1
                            s1 = bool(seen_top[stk, r, ch]); seen_top[stk, r, ch] = seen_top[stk, r + 1, ch]; seen_top[stk, r + 1, ch] = s1
                # round-trip bookkeeping: a walker identity at the TOP marks seen_top; reaching BOTTOM with it counts
                seen_top[:, NR - 1, :] = True
                done = seen_top[:, 0, :].clone()
                trips += int(done.sum()); seen_top[:, 0, :] = False
            Xm = Xr.reshape(B, n, 3); U = cav.full_U(Xm)
        if sw % REC == 0:
            Xr = Xm.view(2, NR, NCH, n, 3)
            qA = st.mean([q_hocky(Xr[0, 0, ch].cpu(), xin.cpu(), R) - BULK_HOCKY for ch in range(NCH)])
            qB = st.mean([q_hocky(Xr[1, 0, ch].cpu(), xin.cpu(), R) - BULK_HOCKY for ch in range(NCH)])
            qA_run.append(qA); qB_run.append(qB)
            half = len(qA_run) // 2
            rA, rB = st.mean(qA_run[half:]), st.mean(qB_run[half:])
            traj.append({"sweep": sw, "qA": qA, "qB": qB, "runA": rA, "runB": rB,
                         "X_bot": Xr[:, 0].cpu().clone()})
            if sw % (REC * 10) == 0:
                ea = (ex_acc / ex_try.clamp(min=1))
                print(f"  sw {sw:>6}: inst A {qA:+.2f} B {qB:+.2f} | run-mean A {rA:+.3f} B {rB:+.3f} "
                      f"| gap {abs(rA-rB):.3f} {'CONVERGED' if abs(rA-rB) < Q_TOL else ''} "
                      f"| exch min/med {float(ea.min()):.2f}/{float(ea.median()):.2f} trips {trips}", flush=True)
                results[(ci, "traj")] = traj; torch.save(results, OUT)
    half = len(qA_run) // 2
    rA, rB = st.mean(qA_run[half:]), st.mean(qB_run[half:])
    results[(ci, "final")] = {"qA": rA, "qB": rB, "gap": abs(rA - rB),
                              "converged": abs(rA - rB) < Q_TOL, "wall_s": time.time() - t0}
    torch.save(results, OUT)
    print(f"  FINAL: run-mean A {rA:+.3f} B {rB:+.3f} gap {abs(rA-rB):.3f} "
          f"{'CONVERGED (q_tol 0.1)' if abs(rA-rB) < Q_TOL else 'NOT converged'} ({time.time()-t0:.0f}s)", flush=True)
    ncav += 1
    if ncav >= 2:
        break
print(f"saved -> {OUT}", flush=True)
