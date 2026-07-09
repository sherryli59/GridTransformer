"""Random-pinning point-to-set: masked (frozen-subset) exact kernels + constrained-run driver.
Pinned particles never move but contribute to the full KA energy; every learned component is behind
exact Metropolis. See docs/superpowers/specs/2026-07-08-ka-point-to-set-pinning-design.md."""
import math, torch
from liquid_coupling_flow.ipl44.ipl_swap_smc import _tame, _pick, _swap_log_ratio, _cb_sample, _cb_logprob, uniform_weight_fn
from liquid_coupling_flow.ka_energy import ka_energy, ka_forces


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
    """Paired-reeval MH swap of one mobile-A with one mobile-B per config (frozen never picked). Exact."""
    B = x.shape[0]; ar = torch.arange(B, device=x.device)
    pB_f = weight_fn(x, s); dt = pB_f.dtype
    isA = ((s == 0) & mobile).to(dt); isB = ((s == 1) & mobile).to(dt)
    i = _pick(pB_f, isA); j = _pick(1.0 - pB_f, isB)
    s_prop = s.clone(); s_prop[ar, i] = 1; s_prop[ar, j] = 0
    U_prop = energy_fn(x, s_prop)
    pB_r = weight_fn(x, s_prop)
    log_ratio = -beta * (U_prop - U) + _swap_log_ratio(pB_f, pB_r, s, s_prop, i, j)
    acc = torch.log(torch.rand(B, device=x.device)) < log_ratio
    return torch.where(acc[:, None], s_prop, s), torch.where(acc, U_prop, U), acc


def masked_block_relabel(x, s, U, mobile, beta, energy_fn, table_fn, k):
    """Gumbel-top-k block relabel over MOBILE sites only (score=-inf on frozen), conditional-Bernoulli redraw
    preserving the block count. x-only randomized selection cancels in the MH ratio -> exact."""
    B, N = s.shape
    W = table_fn(x).clamp(1e-6, 1 - 1e-6)
    score = -((W - 0.5).abs() + 1e-3).log()
    score = score.masked_fill(~mobile, -1e30)                        # frozen never selected
    gumbel = -torch.log(-torch.log(torch.rand_like(score) + 1e-12) + 1e-12)
    blk = torch.argsort(score + gumbel, dim=1, descending=True)[:, :k]
    wblk = W.gather(1, blk); sblk = s.gather(1, blk)
    m = sblk.sum(1).long()
    sblk_new = _cb_sample(wblk, m)
    log_ratio = -beta * (energy_fn(x, s.scatter(1, blk, sblk_new)) - U) \
        + _cb_logprob(wblk, sblk) - _cb_logprob(wblk, sblk_new)
    s_prop = s.scatter(1, blk, sblk_new)
    U_prop = energy_fn(x, s_prop)
    acc = torch.log(torch.rand(B, device=x.device)) < log_ratio
    return torch.where(acc[:, None], s_prop, s), torch.where(acc, U_prop, U), acc


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
