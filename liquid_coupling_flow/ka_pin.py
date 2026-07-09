"""Random-pinning point-to-set: masked (frozen-subset) exact kernels + constrained-run driver.
Pinned particles never move but contribute to the full KA energy; every learned component is behind
exact Metropolis. See docs/superpowers/specs/2026-07-08-ka-point-to-set-pinning-design.md."""
import math, torch
from liquid_coupling_flow.ipl44.ipl_swap_smc import _tame, _pick, _sel_prob, _cb_sample, _cb_logprob, uniform_weight_fn, EPS
from liquid_coupling_flow.ka_energy import ka_energy, ka_forces


def _masked_swap_log_ratio(pB_f, pB_r, s, s_new, mobile, i, j):
    """Selection log-ratio for masked_swap: normalizers restricted to MOBILE same-species particles (the
    frozen-aware analogue of ipl_swap_smc._swap_log_ratio). Exact for any weight_fn."""
    dt = pB_f.dtype
    isA = ((s == 0) & mobile).to(dt);      isB = ((s == 1) & mobile).to(dt)
    isAn = ((s_new == 0) & mobile).to(dt); isBn = ((s_new == 1) & mobile).to(dt)
    g_i = _sel_prob(pB_f, isA, i);         g_j = _sel_prob(1.0 - pB_f, isB, j)
    gp_j = _sel_prob(pB_r, isAn, j);       gp_i = _sel_prob(1.0 - pB_r, isBn, i)
    return (gp_j + EPS).log() + (gp_i + EPS).log() - (g_i + EPS).log() - (g_j + EPS).log()


def pin_mask(B, N, c, device, generator=None):
    """Random pinning: freeze ceil(c*N) particles per config. Returns mobile bool[B,N] (True = mobile)."""
    n_pin = math.ceil(c * N)
    mobile = torch.ones(B, N, dtype=torch.bool, device=device)
    for b in range(B):
        perm = torch.randperm(N, generator=generator, device=device if generator is None else generator.device)
        mobile[b, perm[:n_pin].to(device)] = False
    return mobile


def masked_mala(x, s, U, mobile, beta, L, dt, energy_fn, force_fn, fmax=60.0):
    """Tamed-MALA restricted to mobile particles: drift & noise are masked to zero on frozen sites, so their
    proposal residuals vanish and the MH ratio involves only mobile DOFs. Frozen contribute to energy/forces.
    Returns (x, U, acc_rate)."""
    m = mobile[..., None].to(x.dtype)                                  # [B,N,1]
    F = _tame(force_fn(x, s), fmax) * m
    mu = x + 0.5 * dt * dt * beta * F
    xp = torch.remainder(mu + dt * torch.randn_like(x) * m, L)         # frozen: mu=x, noise=0 -> xp=x
    Up = energy_fn(xp, s)
    Fp = _tame(force_fn(xp, s), fmax) * m
    mup = xp + 0.5 * dt * dt * beta * Fp
    d_f = xp - mu; d_f = d_f - L * torch.round(d_f / L)                # frozen residual = 0
    d_r = x - mup; d_r = d_r - L * torch.round(d_r / L)
    logq = (-(d_r ** 2).sum((1, 2)) + (d_f ** 2).sum((1, 2))) / (2 * dt * dt)
    log_ratio = -beta * (Up - U) + logq
    acc = torch.log(torch.rand(x.shape[0], device=x.device)) < log_ratio
    x = torch.where(acc[:, None, None], xp, x); U = torch.where(acc, Up, U)
    return x, U, float(acc.float().mean())


def masked_swap(x, s, U, mobile, beta, energy_fn, weight_fn):
    """Paired-reeval MH swap of one mobile-A with one mobile-B per config (frozen never picked); the selection
    normalizer is restricted to MOBILE same-species particles, so this is exact for any weight_fn."""
    B = x.shape[0]; ar = torch.arange(B, device=x.device)
    pB_f = weight_fn(x, s); dt = pB_f.dtype
    isA = ((s == 0) & mobile).to(dt); isB = ((s == 1) & mobile).to(dt)
    i = _pick(pB_f, isA); j = _pick(1.0 - pB_f, isB)
    s_prop = s.clone(); s_prop[ar, i] = 1; s_prop[ar, j] = 0
    U_prop = energy_fn(x, s_prop)
    pB_r = weight_fn(x, s_prop)
    log_ratio = -beta * (U_prop - U) + _masked_swap_log_ratio(pB_f, pB_r, s, s_prop, mobile, i, j)
    acc = torch.log(torch.rand(B, device=x.device)) < log_ratio
    return torch.where(acc[:, None], s_prop, s), torch.where(acc, U_prop, U), acc


def masked_block_relabel(x, s, U, mobile, beta, energy_fn, table_fn, k):
    """Gumbel-top-k block relabel over MOBILE sites only (score=-inf on frozen), conditional-Bernoulli redraw
    preserving the block count. x-only randomized selection cancels in the MH ratio -> exact, PROVIDED
    k <= every row's mobile count (asserted below; otherwise frozen sites would fill the block)."""
    B, N = s.shape
    assert k <= int(mobile.sum(1).min()), "block size k exceeds a row's mobile count -> would relabel frozen sites"
    W = table_fn(x).clamp(1e-6, 1 - 1e-6)
    score = -((W - 0.5).abs() + 1e-3).log()
    score = score.masked_fill(~mobile, -1e30)                        # frozen never selected
    gumbel = -torch.log(-torch.log(torch.rand_like(score) + 1e-12) + 1e-12)
    blk = torch.argsort(score + gumbel, dim=1, descending=True)[:, :k]
    wblk = W.gather(1, blk); sblk = s.gather(1, blk)
    m = sblk.sum(1).long()
    sblk_new = _cb_sample(wblk, m)
    s_prop = s.scatter(1, blk, sblk_new)
    U_prop = energy_fn(x, s_prop)
    log_ratio = -beta * (U_prop - U) + _cb_logprob(wblk, sblk) - _cb_logprob(wblk, sblk_new)
    acc = torch.log(torch.rand(B, device=x.device)) < log_ratio
    return torch.where(acc[:, None], s_prop, s), torch.where(acc, U_prop, U), acc


def constrained_run(x_ref, s_ref, mobile, T, L, n_iter, table_fn=None, dt=0.01,
                    record_every=5, arm="ref", scramble_x=None, n_swap=4, k_block=8, fmax=60.0):
    """Re-equilibrate mobile particles among frozen pins; record occupancy overlap Q(t) vs x_ref.
    arm='ref': mobile start at reference positions (Q decays from 1); arm='scramble': mobile start at
    scramble_x (Q rises). Species channel: masked_swap (+ masked_block_relabel if table_fn given), both exact.
    Learned part (table_fn) enters block-relabel only, behind MH."""
    from liquid_coupling_flow.ka_pin_overlap import cell_occupancy, pinned_cells, overlap_Q
    beta = 1.0 / T
    x = x_ref.clone(); s = s_ref.clone()
    if arm == "scramble":
        assert scramble_x is not None
        m3 = mobile[..., None]
        x = torch.where(m3, torch.remainder(scramble_x, L), x)          # only mobile scrambled
    efn = lambda a, b: ka_energy(a, b, L); ffn = lambda a, b: ka_forces(a, b, L)
    U = efn(x, s)
    occ_ref = cell_occupancy(x_ref, L); excl = pinned_cells(x_ref, mobile, L)
    rec = {"t": [], "Q": [], "U": [], "arm": arm}
    for it in range(n_iter + 1):
        if it % record_every == 0:
            rec["t"].append(it)
            rec["Q"].append(overlap_Q(cell_occupancy(x, L), occ_ref, excl))
            rec["U"].append(float((U / x.shape[1]).median()))
        if it == n_iter:
            break
        x, U, _ = masked_mala(x, s, U, mobile, beta, L, dt, efn, ffn, fmax)
        for _ in range(n_swap):
            s, U, _ = masked_swap(x, s, U, mobile, beta, efn, uniform_weight_fn)
        if table_fn is not None:
            s, U, _ = masked_block_relabel(x, s, U, mobile, beta, efn, table_fn, k_block)
    rec["x_final"] = x; rec["s_final"] = s; rec["x_ref"] = x_ref; rec["mobile"] = mobile
    return rec


def load_geometry_table(N, L, nB, ckpt_path, device):
    """Load the N=100-trained two-time joint flow (knn=32) and return table_fn(x)->P(species=B) [B,N], which
    is s-INDEPENDENT (queried at t_pos=1, t_spec=0 with the canonical labelling) -> valid block-relabel table."""
    from liquid_coupling_flow.ipl44.joint_flow import JointSpeciesFlow
    ck = torch.load(ckpt_path, map_location=device, weights_only=False); cfg = ck["cfg"]
    jf = JointSpeciesFlow(n_particles=N, L=L, hidden_nf=cfg["hidden_nf"], n_layers=cfg["n_layers"],
                          two_time=True).to(device)
    jf.load_state_dict(ck["state_dict"]); jf.eval(); jf.knn = 32
    s_canon = torch.zeros(1, N, dtype=torch.long, device=device); s_canon[:, :nB] = 1

    def table_fn(x):
        with torch.no_grad():
            Bx = x.shape[0]
            _, lg = jf(torch.ones(Bx, 1, 1, device=device), x, s_canon.expand(Bx, -1),
                       t_spec=torch.zeros(Bx, 1, 1, device=device))
            return torch.softmax(lg, -1)[..., 1]
    return table_fn
