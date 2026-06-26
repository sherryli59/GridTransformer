"""Single-site Metropolis-Hastings kernel with a learned (NonCausalLF) local proposal.
See docs/superpowers/specs/2026-06-25-ka-single-site-mh-kernel-design.md."""
from __future__ import annotations
import math, torch, torch.nn.functional as F
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR
from liquid_coupling_flow.ka_gridformer import _wrap_pm

_SIG = torch.tensor(SIGMA); _EPS = torch.tensor(EPS); _RC = RCUT_FACTOR


def site_energy(xj, sj, pos, s, j, L):
    """Sum of shifted-LJ pair energies of a particle at xj (species int sj) vs all pos_k, k!=j.
    xj [B,2], pos [B,N,2], s [N] long, returns [B]."""
    B, N, _ = pos.shape
    sig = _SIG.to(pos.device, pos.dtype)[sj, s.long()]            # [N]  pair sigma j-vs-k
    eps = _EPS.to(pos.device, pos.dtype)[sj, s.long()]            # [N]
    rc = _RC * sig
    d = xj[:, None, :] - pos                                      # [B,N,2]
    d = d - L * torch.round(d / L)
    r2 = (d ** 2).sum(-1)                                         # [B,N]
    r2[:, j] = 1e12                                              # exclude self
    inv6 = (sig ** 2 / r2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    src6 = (sig / rc) ** 6
    e = torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e))
    return e.sum(-1)                                              # [B]


def site_dE(pos, s, j, xj_new, L):
    sj = int(s[j])
    return site_energy(xj_new, sj, pos, s, j, L) - site_energy(pos[:, j], sj, pos, s, j, L)


def _bin_probs(m, ctx_j):
    """Return (la [B,nbins], lb [B,nbins,nbins]) log-probs: lb[:,a] = log P(bin_b | bin_a)."""
    la = F.log_softmax(m.head_a(ctx_j), -1)                       # [B,nbins]
    nb = m.n_bins
    a_emb = m.bin_a_emb(torch.arange(nb, device=ctx_j.device))    # [nbins,d]
    lb = F.log_softmax(m.head_b(ctx_j[:, None, :] + a_emb[None]), -1)  # [B,nbins,nbins]
    return la, lb

def _offsets_to_bins(off, m):
    """off [B] in arc-units along one dim -> long bin idx in [0,nbins); out-of-range -> -1."""
    AR = m.arc_range
    b = ((off + AR) / m.bin_w).floor().long()
    return torch.where((off >= -AR) & (off < AR), b, torch.full_like(b, -1))

@torch.no_grad()
def site_logq(m, ctx_j, origin_j, xj, arc, L):
    """Folded log q(xj | x_{-j}). xj [B,2] -> [B]."""
    la, lb = _bin_probs(m, ctx_j)
    AR = m.arc_range; period = L / arc                            # torus period in arc-units
    off = _wrap_pm(xj - origin_j, L) / arc                        # [B,2] in [-period/2, period/2]
    B = xj.shape[0]; q = torch.zeros(B, device=xj.device)
    for da in (-1, 0, 1):                                         # fold aliased grid offsets
        for db in (-1, 0, 1):
            oa = off[:, 0] + da * period; ob = off[:, 1] + db * period
            ba = _offsets_to_bins(oa, m); bb = _offsets_to_bins(ob, m)
            ok = (ba >= 0) & (bb >= 0)
            if not ok.any():
                continue
            lpa = la.gather(1, ba.clamp_min(0)[:, None]).squeeze(1)
            lpb = lb.gather(1, ba.clamp_min(0)[:, None, None].expand(-1, 1, m.n_bins)).squeeze(1) \
                    .gather(1, bb.clamp_min(0)[:, None]).squeeze(1)
            contrib = (lpa + lpb).exp() / (m.bin_w * arc) ** 2     # density in PHYSICAL units
            q = q + torch.where(ok, contrib, torch.zeros_like(contrib))
    return q.clamp_min(1e-30).log()

@torch.no_grad()
def learned_position_sweep(m, pos, s, sc, L, N, kT, arc, rng):
    """N random-scan single-site MH attempts; per-move _local recompute (correctness-first, spec §3.4)."""
    B = pos.shape[0]; sp = s.expand(B, N); n_acc = 0
    order = torch.randperm(N, generator=rng, device=pos.device)
    for j in order.tolist():
        ctx, origin = m._local(pos, sp, sc, N=N, L=L)             # current config
        ctx_j, origin_j = ctx[:, j], origin[:, j]
        xj_new, logq_new = site_propose(m, ctx_j, origin_j, arc, L)
        logq_old = site_logq(m, ctx_j, origin_j, pos[:, j], arc, L)
        dE = site_dE(pos, s, j, xj_new, L)
        logacc = (-dE / kT) + logq_old - logq_new
        acc = torch.log(torch.rand(B, generator=rng, device=pos.device)) < logacc
        pos[:, j] = torch.where(acc[:, None], xj_new, pos[:, j]); n_acc += int(acc.sum())
    return pos, n_acc

@torch.no_grad()
def uniform_position_sweep(pos, s, L, N, kT, step, rng):
    """Matched baseline: N single-site symmetric Gaussian-displacement attempts."""
    B = pos.shape[0]; n_acc = 0
    order = torch.randperm(N, generator=rng, device=pos.device)
    for j in order.tolist():
        xj_new = torch.remainder(pos[:, j] + step * torch.randn(B, 2, generator=rng, device=pos.device), L)
        dE = site_dE(pos, s, j, xj_new, L)
        acc = torch.log(torch.rand(B, generator=rng, device=pos.device)) < (-dE / kT)
        pos[:, j] = torch.where(acc[:, None], xj_new, pos[:, j]); n_acc += int(acc.sum())
    return pos, n_acc

@torch.no_grad()
def mock_position_sweep(pos, s, L, N, kT, sigma, rng):
    """Single-site MH with an analytic, x_j-INDEPENDENT torus-Gaussian conditional
    q(x_j|x_{-j}) = N_torus(pos_{(j+1)%N}, sigma^2). The proposal centre is another particle (so it depends
    only on x_{-j}, mirroring §3.2's context-independence). The density is IMAGE-SUMMED so it exactly matches
    the wrapped sampler at the boundary -> validates the MH/sweep arithmetic independent of the learned model."""
    B = pos.shape[0]; n_acc = 0; norm = 2 * math.pi * sigma ** 2
    def logq(xj, mu):
        acc = torch.zeros(B, device=pos.device)
        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                d = xj - mu - torch.tensor([ix * L, iy * L], device=pos.device, dtype=pos.dtype)
                acc = acc + torch.exp(-(d ** 2).sum(-1) / (2 * sigma ** 2))
        return (acc / norm).clamp_min(1e-30).log()
    for j in torch.randperm(N, generator=rng, device=pos.device).tolist():
        mu = pos[:, (j + 1) % N]                                  # x_{-j}-only proposal centre
        xj_new = torch.remainder(mu + sigma * torch.randn(B, 2, generator=rng, device=pos.device), L)
        dE = site_dE(pos, s, j, xj_new, L)
        logacc = (-dE / kT) + logq(pos[:, j], mu) - logq(xj_new, mu)
        acc = torch.log(torch.rand(B, generator=rng, device=pos.device)) < logacc
        pos[:, j] = torch.where(acc[:, None], xj_new, pos[:, j]); n_acc += int(acc.sum())
    return pos, n_acc


@torch.no_grad()
def swap_sweep(pos, s, L, kT, n_swap, rng):
    """A<->B position swaps; MH on the two particles' local energy change."""
    B = pos.shape[0]; n_acc = 0
    A = (s == 0).nonzero().squeeze(-1); Bi = (s == 1).nonzero().squeeze(-1)
    if len(A) == 0 or len(Bi) == 0:
        return pos, 0
    for _ in range(n_swap):
        i = int(A[torch.randint(len(A), (1,), generator=rng, device=pos.device)])
        j = int(Bi[torch.randint(len(Bi), (1,), generator=rng, device=pos.device)])
        xi, xj = pos[:, i].clone(), pos[:, j].clone()
        e_old = site_energy(xi, int(s[i]), pos, s, i, L) + site_energy(xj, int(s[j]), pos, s, j, L)
        posp = pos.clone(); posp[:, i] = xj; posp[:, j] = xi
        e_new = site_energy(xj, int(s[i]), posp, s, i, L) + site_energy(xi, int(s[j]), posp, s, j, L)
        acc = torch.log(torch.rand(B, generator=rng, device=pos.device)) < (-(e_new - e_old) / kT)
        pos[:, i] = torch.where(acc[:, None], xj, xi); pos[:, j] = torch.where(acc[:, None], xi, xj)
        n_acc += int(acc.sum())
    return pos, n_acc


@torch.no_grad()
def preflight(m, ref_path, kT=0.5, B=32, n_warm=40):
    ref = torch.load(ref_path, map_location=next(m.parameters()).device, weights_only=False)
    dev = next(m.parameters()).device
    s = ref["s"].to(dev).long(); L = ref["L"]; N = ref["x"].shape[1]
    arc = m._arc_scale(N); sc = m.geo._scaffold(N, dev)
    geom_ok = bool(m.arc_range * arc >= L / 2)
    assert geom_ok, f"GEOMETRIC FAIL: arc_range*arc={m.arc_range*arc:.3f} < L/2={L/2:.3f}; enlarge arc_range"
    # equilibrium out-of-range fraction (curve-ordered reference)
    x = ref["x"][:512].to(dev); order = m.geo._curve_order(x, N)
    xo = torch.gather(x, 1, order[..., None].expand(-1, -1, 2)); so = torch.gather(s.expand(x.shape[0], N), 1, order)
    _, origin = m._local(xo, so, sc, N=N, L=L)
    off = (_wrap_pm(xo - origin, L) / arc).abs().amax(-1)
    oor = float((off > m.arc_range).float().mean())
    assert oor == 0.0, f"COVERAGE FAIL: out-of-range {oor:.4%} > 0; enlarge arc_range / add fallback move"
    # warm acceptance + reverse-zero fraction from a uniform seed
    pos = torch.rand(B, N, 2, device=dev) * L
    g = torch.Generator(device=dev).manual_seed(0); acc_tot = 0; rz = 0; rz_den = 0
    for _ in range(n_warm):
        pos, na = learned_position_sweep(m, pos, s, sc, L, N, kT, arc, g); acc_tot += na
        ctx, ori = m._local(pos, s.expand(B, N), sc, N=N, L=L)     # sample reverse-zero on current config
        for j in (0, N // 2, N - 1):
            rz += int((site_logq(m, ctx[:, j], ori[:, j], pos[:, j], arc, L) < -60).sum()); rz_den += B
    return {"geom_ok": geom_ok, "out_of_range_frac": oor,
            "accept_frac": acc_tot / (n_warm * N * B), "reverse_zero_frac": rz / max(rz_den, 1)}


@torch.no_grad()
def site_propose(m, ctx_j, origin_j, arc, L):
    """Sample xj' ~ q(.|x_{-j}); return (xj' [B,2], logq' [B]) with logq' the folded density at xj'."""
    la, lb = _bin_probs(m, ctx_j); B = ctx_j.shape[0]
    ba = torch.multinomial(la.exp(), 1).squeeze(1)                # [B]
    lb_a = lb.gather(1, ba[:, None, None].expand(-1, 1, m.n_bins)).squeeze(1)  # [B,nbins]
    bb = torch.multinomial(lb_a.exp(), 1).squeeze(1)
    a = m._bin_center(ba) + (torch.rand(B, device=ctx_j.device) - 0.5) * m.bin_w
    b = m._bin_center(bb) + (torch.rand(B, device=ctx_j.device) - 0.5) * m.bin_w
    xj = torch.remainder(origin_j + torch.stack([a, b], -1) * arc, L)
    return xj, site_logq(m, ctx_j, origin_j, xj, arc, L)


from liquid_coupling_flow.ka_observables import partial_gr
from liquid_coupling_flow.ka_energy import ka_energy


def _gbb(pos, s, L):
    """g_BB peak height and spurious contacts (mean g below 0.88)."""
    rc, gg = partial_gr(pos, s, L, min(L / 2, 4.0), 60, (1, 1))
    import numpy as np
    rc = np.asarray(rc); gg = np.asarray(gg)
    return float(gg.max()), float(gg[rc < 0.88].mean())


def _overlap(pos, L, thr=0.7):
    """Fraction of nearest-neighbour pairs closer than thr (clash indicator)."""
    d = pos[:, :, None, :] - pos[:, None, :, :]; d = d - L * torch.round(d / L)
    r = (d ** 2).sum(-1).sqrt(); N = pos.shape[1]
    r = r + torch.eye(N, device=pos.device)[None] * 1e3
    return float((r.min(2).values.reshape(-1) < thr).float().mean())


def __art__():
    import os
    return os.path.join(os.path.dirname(__file__), "artifacts")


@torch.no_grad()
def tune_uniform_step(s, L, N, kT, target=0.5, B=16, iters=18):
    """Bisection on Gaussian step size to hit ~target single-site acceptance on equilibrated configs."""
    dev = s.device; ref_step_lo, ref_step_hi = 0.01, 1.0
    ref = torch.load(f"{__art__()}/ka_reference_N100.pt", map_location=dev, weights_only=False)
    base = ref["x"][:B].to(dev)
    for _ in range(iters):
        step = 0.5 * (ref_step_lo + ref_step_hi); g = torch.Generator(device=dev).manual_seed(0)
        pos = base.clone(); _, na = uniform_position_sweep(pos, s, L, N, kT, step, g)
        frac = na / (N * B)
        if frac > target:
            ref_step_lo = step
        else:
            ref_step_hi = step
    return 0.5 * (ref_step_lo + ref_step_hi)


def sweeps_to_reference(sweeps, vals, band, k_blocks=4):
    """Sweep at which the first run of >= k_blocks consecutive in-band records BEGINS. None if never."""
    lo, hi = band; n = len(vals)
    for start in range(n - k_blocks + 1):
        if all(lo <= vals[start + k] <= hi for k in range(k_blocks)):
            return sweeps[start]
    return None


@torch.no_grad()
def run_chain(m, pos, s, sc, L, N, kT, n_sweeps, record_every, n_swap, kind, step, arc, rng):
    """Run a chain of MH sweeps, recording observables every record_every sweeps.

    kind="learned" -> learned_position_sweep; else -> uniform_position_sweep.
    Returns dict with trajectory lists {sweeps, U, gbb_peak, gbb_spur, overlap, accept} and x_final.
    """
    out = {"sweeps": [], "U": [], "gbb_peak": [], "gbb_spur": [], "overlap": [], "accept": []}
    for sweep in range(n_sweeps + 1):
        if sweep % record_every == 0:
            out["sweeps"].append(sweep)
            out["U"].append(float((ka_energy(pos, s, L) / N).median()))
            p, q = _gbb(pos, s, L)
            out["gbb_peak"].append(p); out["gbb_spur"].append(q)
            out["overlap"].append(_overlap(pos, L))
        if sweep == n_sweeps:
            break
        if kind == "learned":
            pos, na = learned_position_sweep(m, pos, s, sc, L, N, kT, arc, rng)
        else:
            pos, na = uniform_position_sweep(pos, s, L, N, kT, step, rng)
        out["accept"].append(na / (N * pos.shape[0]))
        pos, _ = swap_sweep(pos, s, L, kT, n_swap, rng)
    out["x_final"] = pos
    return out


def _load_model_for_bench(dev, N):
    from liquid_coupling_flow.ka_noncausal import NonCausalLF
    ck = torch.load(f"{__art__()}/ka_noncausal_N{N}.pt", map_location=dev, weights_only=False)
    m = NonCausalLF(rho=1.2, n_bins=192, knn=ck["knn"], canonical=False).to(dev).eval(); m.load_state_dict(ck["state_dict"]); return m


def _ar_seed(dev, N, nB, B):
    from liquid_coupling_flow.ka_localframe import KALocalFrameModel
    ck = torch.load(f"{__art__()}/ka_localframe_N100_20k.pt", map_location=dev, weights_only=False)
    mab = KALocalFrameModel(rho=ck["rho"], n_bins=ck["n_bins"], knn=ck["knn"]).to(dev).eval()
    mab.load_state_dict(ck["state_dict"], strict=False)
    return mab.sample(B, N, n_B=nB, device=dev)[0]


@torch.no_grad()
def benchmark(N=100, kT=0.5, B=64, n_sweeps=600, record_every=25, device=None):
    import os, time, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = _load_model_for_bench(dev, N)
    pf = preflight(m, f"{__art__()}/ka_reference_N{N}.pt", kT, B=32, n_warm=40)
    print("PRE-FLIGHT:", pf, flush=True)
    assert pf["geom_ok"] and pf["out_of_range_frac"] == 0.0 and pf["accept_frac"] >= 0.05, "pre-flight gate failed"
    ref = torch.load(f"{__art__()}/ka_reference_N{N}.pt", map_location=dev, weights_only=False)
    s = ref["s"].to(dev).long(); L = ref["L"]; data = ref["x"].to(dev); sc = m.geo._scaffold(N, dev); arc = m._arc_scale(N)
    # reference band (±2σ) for U and g_BB peak from independent reference draws
    U_ref = (ka_energy(data[:512], s, L) / N); u_lo, u_hi = float(U_ref.mean() - 2 * U_ref.std()), float(U_ref.mean() + 2 * U_ref.std())
    gpk = _gbb(data[:512], s, L)[0]; band_g = (gpk - 0.3, gpk + 0.3)
    step = tune_uniform_step(s, L, N, kT); print(f"tuned uniform step={step:.3f}", flush=True)
    n_swap = N // 8
    seeds = {"uniform": torch.rand(B, N, 2, device=dev) * L,
             "AR": _ar_seed(dev, N, int((s == 1).sum()), B)}
    traj = {}
    for seedname, x0 in seeds.items():
        for kind, stp in (("uniform", step), ("learned", 0.0)):
            g = torch.Generator(device=dev).manual_seed(0); t0 = time.time()
            tj = run_chain(m, x0.clone(), s, sc, L, N, kT, n_sweeps, record_every, n_swap, kind, stp, arc, g)
            su = sweeps_to_reference(tj["sweeps"], tj["U"], (u_lo, u_hi), 4)
            sg = sweeps_to_reference(tj["sweeps"], tj["gbb_peak"], band_g, 4)
            reach = max(su, sg) if (su is not None and sg is not None) else None
            traj[(seedname, kind)] = (tj, reach)
            print(f"[{seedname}/{kind}] reach U&gBB -> {reach}  (U {tj['U'][0]:+.2f}->{tj['U'][-1]:+.2f}, "
                  f"gBB {tj['gbb_peak'][0]:.2f}->{tj['gbb_peak'][-1]:.2f}, {time.time()-t0:.0f}s)", flush=True)
    for seedname in seeds:
        ru = traj[(seedname, "uniform")][1]; rl = traj[(seedname, "learned")][1]
        print(f"SWEEPS-SAVED [{seedname}]: uniform {ru} - learned {rl} = "
              f"{None if (ru is None or rl is None) else ru - rl}", flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for (seedname, kind), (tj, _) in traj.items():
        c = {"uniform": "C0", "learned": "C2"}[kind]; ls = {"uniform": "-", "AR": "--"}[seedname]
        ax[0].plot(tj["sweeps"], tj["U"], c, ls=ls, label=f"{seedname}/{kind}")
        ax[1].plot(tj["sweeps"], tj["gbb_peak"], c, ls=ls, label=f"{seedname}/{kind}")
    ax[0].axhspan(u_lo, u_hi, color="grey", alpha=0.2); ax[0].set_ylabel("<U>/N"); ax[0].set_xlabel("sweeps")
    ax[1].axhspan(*band_g, color="grey", alpha=0.2); ax[1].set_ylabel("g_BB peak"); ax[1].set_xlabel("sweeps"); ax[1].legend(fontsize=8)
    fig.suptitle(f"Learned single-site MH vs matched uniform (N={N}, T={kT}): sweeps to PT reference")
    out = f"{__art__()}/ka_mh_kernel_benchmark_N{N}.png"; fig.tight_layout(); fig.savefig(out, dpi=120)
    print("saved", out, flush=True)


if __name__ == "__main__":
    import sys
    benchmark(n_sweeps=int(sys.argv[1]) if len(sys.argv) > 1 else 600,
              B=int(sys.argv[2]) if len(sys.argv) > 2 else 64)
