from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineCoupling(nn.Module):
    """Diagonal affine map on [...,3] with tanh-bounded log-scale (identity when params=0).
    forward:  u_out = u * exp(s) + t,  s = smax*tanh(params[...,:3]),  t = params[...,3:]
    logdet = sum(s). inverse: u = (u_out - t) * exp(-s)."""
    def __init__(self, smax: float = 2.0):
        super().__init__()
        self.smax = float(smax)

    def _st(self, params):
        s = self.smax * torch.tanh(params[..., :3])
        t = params[..., 3:]
        return s, t

    def forward(self, u, params):
        s, t = self._st(params)
        return u * torch.exp(s) + t, s.sum(-1)

    def inverse(self, u_out, params):
        s, t = self._st(params)
        return (u_out - t) * torch.exp(-s)


class CageConditioner(nn.Module):
    """Per-active-particle affine params from a distance-RBF + species-pair message over the conditioning
    set (other block particles + cage). Physical positions; batched [M,A,*] x [M,C,*] -> [M,A,6]. Last
    layer zero-init => identity coupling at init (composition starts == base)."""
    def __init__(self, d_model: int = 96, n_rbf: int = 12, rbf_max: float = 3.0, n_species: int = 2):
        super().__init__()
        self.n_species = n_species
        mu = torch.linspace(0.0, rbf_max, n_rbf)
        self.register_buffer("mu", mu); self.w = float(mu[1] - mu[0])
        self.pair = nn.Embedding(n_species * n_species, 8)
        self.msg = nn.Sequential(nn.Linear(n_rbf + 8, d_model), nn.SiLU(),
                                 nn.Linear(d_model, d_model), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 6))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def params(self, active_x, active_s, cond_x, cond_s, R):
        d = (active_x[:, :, None, :] - cond_x[:, None, :, :]).norm(dim=-1)          # [M,A,C]
        r = torch.exp(-((d[..., None] - self.mu) ** 2) / (2 * self.w ** 2))         # [M,A,C,n_rbf]
        pr = (active_s[:, :, None] * self.n_species + cond_s[:, None, :]).clamp(0, self.n_species ** 2 - 1)
        msg = self.msg(torch.cat([r, self.pair(pr)], -1))                          # [M,A,C,d]
        agg = msg.mean(2)                                                          # permutation-invariant over cond
        return self.head(agg)                                                     # [M,A,6]


from liquid_coupling_flow.ka3d_scaffold_ar import ball_squash


class BlockCorrector(nn.Module):
    """Stacked conditional affine-coupling flow over the K block particles' u-coords. Layer L transforms
    the parity-L half of block particles conditioned on the other half + the frozen cage."""
    def __init__(self, n_layers: int = 6, d_model: int = 96, n_rbf: int = 12, smax: float = 2.0):
        super().__init__()
        self.n_layers = n_layers
        self.couple = AffineCoupling(smax=smax)
        self.conds = nn.ModuleList([CageConditioner(d_model, n_rbf) for _ in range(n_layers)])

    def _phys(self, u, anchor_y, R):
        x, _ = ball_squash(u + anchor_y[None], R)                                 # [M,K,3]
        return x

    def _run(self, u, s_blk, anchor_y, cage_x, cage_s, R, invert):
        M, K, _ = u.shape
        dev = u.device
        idx = torch.arange(K, device=dev)
        order = range(self.n_layers - 1, -1, -1) if invert else range(self.n_layers)
        logdet = u.new_zeros(M)
        # Fixed (non-evolving) per-slot anchor position, used as the ACTIVE side's own query location so
        # the conditioner is a function only of quantities invariant to this layer's transform (the
        # inactive half + cage + each active particle's fixed anchor) -- required for exact invertibility:
        # using the evolving xu[:, active] instead would differ between forward (pre-transform) and
        # inverse (post-transform, same layer) since that IS the value being solved for.
        anchor_x = self._phys(u.new_zeros(u.shape), anchor_y, R)                  # [M,K,3], layer-invariant
        for L in order:
            active = (idx % 2) == (L % 2)                                         # alternating half
            passive = ~active
            xu = self._phys(u, anchor_y, R)                                       # current physical block
            cond_x = torch.cat([xu[:, passive], cage_x], 1)                       # other block + cage
            cond_s = torch.cat([s_blk[:, passive], cage_s], 1)
            params_full = u.new_zeros(M, K, 6)
            params_full[:, active] = self.conds[L].params(
                anchor_x[:, active], s_blk[:, active], cond_x, cond_s, R)
            if invert:
                u = u.clone()
                u[:, active] = self.couple.inverse(u[:, active], params_full[:, active])
                logdet = logdet - self.couple.forward(u[:, active] * 0, params_full[:, active])[1].sum(-1)  # -sum s
            else:
                ua, ld = self.couple.forward(u[:, active], params_full[:, active])
                u = u.clone(); u[:, active] = ua
                logdet = logdet + ld.sum(-1)
        return u, logdet

    def forward(self, u_blk, s_blk, anchor_y_blk, cage_x, cage_s, R):
        return self._run(u_blk, s_blk, anchor_y_blk, cage_x, cage_s, R, invert=False)

    def inverse(self, u_out, s_blk, anchor_y_blk, cage_x, cage_s, R):
        return self._run(u_out, s_blk, anchor_y_blk, cage_x, cage_s, R, invert=True)


from liquid_coupling_flow.ka3d_scaffold_ar import ball_unsquash, fixed_ball_scaffold


class CorrectedBlockModel(nn.Module):
    """Frozen base (KA3DScaffoldEBMBatched) + BlockCorrector, composed to an exact-log_q block proposal."""
    def __init__(self, base, corrector):
        super().__init__()
        self.base = base
        self.corrector = corrector

    def _geom(self, xo, block_mask, bnd, s_bnd, R):
        """Reorder [retained; block]; return block anchors' unsquashed coords + the frozen cage (physical)."""
        n = xo.shape[1]; dev = xo.device
        anchors = fixed_ball_scaffold(n, R, dev, xo.dtype)
        order = torch.argsort(block_mask.to(torch.uint8), stable=True)
        n_ret = int((~block_mask).sum())
        anchor_y, _ = ball_unsquash(anchors[order], R)          # [n,3]
        return order, n_ret, anchor_y[n_ret:]                    # block anchors = suffix

    def _cage(self, xo_re, so_re, n_ret, bnd, s_bnd):
        """Physical cage = boundary ++ retained interior (everything not in the block). xo_re [M,n,3]."""
        M = xo_re.shape[0]; m = bnd.shape[0]
        cage_x = torch.cat([bnd[None].expand(M, m, 3), xo_re[:, :n_ret]], 1)
        cage_s = torch.cat([s_bnd[None].expand(M, m), so_re[:, :n_ret]], 1)
        return cage_x, cage_s

    def sample_block_corrected(self, xo, so, block_mask, bnd, s_bnd, R, gen=None):
        base = self.base
        order, n_ret, ay_blk = self._geom(xo, block_mask, bnd, s_bnd, R)
        x0, s0, lq0 = base.sample_block_b(xo, so, block_mask, bnd, s_bnd, R, gen=gen)   # [M,n,3],[M,n],[M]
        x0_re = x0[:, order]; s0_re = s0[:, order]
        xb0 = x0_re[:, n_ret:]                                                          # block, [M,K,3]
        y0, ld_yx = ball_unsquash(xb0, R)                                               # [M,K,3],[M,K]
        u0 = y0 - ay_blk[None]
        cage_x, cage_s = self._cage(x0_re, s0_re, n_ret, bnd, s_bnd)
        u1, ldF = self.corrector.forward(u0, s0_re[:, n_ret:], ay_blk, cage_x, cage_s, R)
        xb1, ld_xy = ball_squash(u1 + ay_blk[None], R)
        x1_re = x0_re.clone(); x1_re[:, n_ret:] = xb1
        x1 = torch.empty_like(x0); x1[:, order] = x1_re
        logq = lq0 - ld_yx.sum(1) - ldF - ld_xy.sum(1)
        return x1, s0, logq

    def block_log_prob_corrected(self, xo, so, block_mask, bnd, s_bnd, R):
        base = self.base
        order, n_ret, ay_blk = self._geom(xo, block_mask, bnd, s_bnd, R)
        xo_re = xo[:, order]; so_re = so[:, order]
        xb1 = xo_re[:, n_ret:]
        y1, ld_xy = ball_unsquash(xb1, R)                                               # unsquash of x1 (given config)
        u1 = y1 - ay_blk[None]
        cage_x, cage_s = self._cage(xo_re, so_re, n_ret, bnd, s_bnd)
        u0, ldFinv = self.corrector.inverse(u1, so_re[:, n_ret:], ay_blk, cage_x, cage_s, R)
        xb0, ld_yx = ball_squash(u0 + ay_blk[None], R)
        x0_re = xo_re.clone(); x0_re[:, n_ret:] = xb0
        x0 = torch.empty_like(xo); x0[:, order] = x0_re
        # The frozen base's forward pass is NOT run-to-run invariant to grad-tracking mode: scoring an
        # x0 that carries an autograd graph (via the corrector's trainable params) selects a different
        # attention-kernel path than scoring a plain (requires_grad=False) tensor, at ~2e-3 magnitude on
        # this cavity -- confirmed by A/B: base.block_log_prob_b(x, ...) on BIT-IDENTICAL x differs by
        # 2.0e-3 solely based on x.requires_grad, with no torch.no_grad() (verified: neither .detach() on
        # x0 alone, nor position precision, mattered -- only the ambient torch.is_grad_enabled() state
        # did). The base is frozen (all params requires_grad_(False)); match the gate's own no-grad call.
        with torch.no_grad():
            lq0 = base.block_log_prob_b(x0, so, block_mask, bnd, s_bnd, R)
        # Inverse-path sign: ball_unsquash(x) and ball_squash(y) evaluated at corresponding points are exact
        # negatives of each other (log|dy/dx| == -log|dx/dy|), and ldFinv == -ldF exactly. Substituting those
        # identities into the sample-path formula (lq0 - ld_S1 - ldF - ld_U0) flips BOTH ball-map terms to
        # "+" here (they are NOT mirrored with the sample path's "-, -"): logq = lq0 + ld_U1 + ldFinv + ld_S0.
        logq = lq0 + ld_xy.sum(1) + ldFinv + ld_yx.sum(1)
        return logq


def load_base_and_cavity(dev, ci, R, K, M):
    """Test/gate helper: load frozen base + build ONE carved cavity replicated to M chains + a K-blob mask."""
    import torch
    from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
    from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
    from liquid_coupling_flow.ka3d_cavity_ar import _mic
    from liquid_coupling_flow.ka3d_cavity_carve import carve
    ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax.pt", map_location=dev, weights_only=False)
    base = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
    base.load_state_dict(ck["state_dict"], strict=False); base.eval(); base.use_frame = False
    for p in base.parameters():
        p.requires_grad_(False)
    d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
    X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
    gen = torch.Generator(device=dev).manual_seed(ci)
    c = torch.rand(3, generator=gen, device=dev) * L
    p = carve(X[ci], S[ci], c, R, L)
    xo1, so1, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + 2.5)
    bnd, sb = xout[bm], p["s_out"][bm]
    n = xo1.shape[0]
    a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev)
    blk[(a - a[seed]).norm(dim=-1).topk(min(K, n), largest=False).indices] = True
    xo = xo1[None].expand(M, n, 3).contiguous(); so = so1[None].expand(M, n).contiguous()
    return base, (xo, so, blk, bnd, sb, R)
