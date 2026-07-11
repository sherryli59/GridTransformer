"""No-retrain test motivated by 2D: MTM tolerates MILD clashes (2D: 10% clash-free -> 30% MTM). 3D's
broad conditional makes CATASTROPHIC (r->0) clashes. Cutting the broad tail via LOW-TEMPERATURE
sampling (z~N(0,T^2), T<1) with the correctly-tempered proposal density q_T should turn catastrophic
proposals into mild ones -> MTM accepts. Tests T in {1.0,0.7,0.5,0.35} on the current 3D model, no retrain.

Tempered flow: sample z*T through spline; logq_T(u)= normal_logp(z/T)-log T +/- logdet (base variance T^2)."""
import math, shutil, statistics as st
from pathlib import Path
import torch
import torch.nn.functional as F
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_scaffold_ar import (KA3DScaffoldAR, label_to_scaffold, fixed_ball_scaffold,
                                                   ball_squash, ball_unsquash)
from liquid_coupling_flow.ka3d_block import _order
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"
SCR = Path("/tmp/claude-1002/-mnt-ssd-GridTransformer/271167b9-117a-400e-b11f-c2c1bfa0b6df/scratchpad")
ckp = SCR / "lowt.pt"; shutil.copy("liquid_coupling_flow/artifacts/ka3d_blob_sharp.pt", ckp)
ck = torch.load(ckp, map_location=dev, weights_only=False)
m = KA3DScaffoldAR(flow_bins=48, flow_tail=2.75).to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
print(f"ckpt step={ck['step']}", flush=True)
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"]); R = 2.0; beta = 2.0
E3, E1 = torch.zeros(0, 3, device=dev), torch.zeros(0, dtype=torch.long, device=dev)
gen = torch.Generator(device=dev).manual_seed(3)
LOG2PI = math.log(2 * math.pi)


@torch.no_grad()
def block_geom(x, s, blk):
    n = x.shape[0]; anchors = fixed_ball_scaffold(n, R, dev); order = _order(blk)
    n_ret = int((~blk).sum()); xo, so, anch = x[order], s[order], anchors[order]
    return n, n_ret, xo, so, anch, ball_unsquash(anch, R)[0], order


@torch.no_grad()
def tempered_sample_block(x, s, blk, T):
    """Regenerate block with base variance T^2; return xo_new, so_new, logq_T."""
    n, n_ret, xo, so, anch, anch_y, order = block_geom(x, s, blk)
    xo = xo.clone(); so = so.clone(); combined = xo.clone(); scomb = so.clone()
    kind = torch.zeros(n, dtype=torch.long, device=dev); idx = torch.arange(n, device=dev)
    logq = xo.new_zeros(())
    for jj in range(n_ret, n):
        h, frame = m._frame_context(combined, scomb, kind, (idx < jj)[None], anch[jj:jj+1], anch[jj:jj+1],
                                    slot_feat=m._slot_features(anch[jj:jj+1], jj, n, R))
        ctx = h + m.sp_out_emb(so[jj:jj+1]); vals = []
        for a in range(3):
            z = torch.randn(1, 1, device=dev, generator=gen) * T
            ui, ld = m.flow.spline.forward(z, m.flow.heads[a](ctx))
            logq = logq + (-0.5 * (z / T) ** 2 - 0.5 * LOG2PI - math.log(T) - ld).squeeze()
            vals.append(ui); ctx = torch.cat([ctx, ui], -1)
        y = anch_y[jj] + torch.einsum("naj,na->nj", frame, torch.cat(vals, -1))[0]
        pos, ldxy = ball_squash(y, R); xo[jj] = pos; combined[jj] = pos
        logq = logq - ldxy.squeeze()
    xn = x.clone(); xn[order] = xo
    return xn, s, logq


@torch.no_grad()
def tempered_block_logp(x, s, blk, T):
    """logq_T of the TRUE block given retained (reverse move)."""
    n, n_ret, xo, so, anch, anch_y, order = block_geom(x, s, blk)
    kind = torch.zeros(n, dtype=torch.long, device=dev); idx = torch.arange(n, device=dev)
    y, ldyx = ball_unsquash(xo, R)
    lp = xo.new_zeros(())
    for jj in range(n_ret, n):
        h, frame = m._frame_context(xo, so, kind, (idx < jj)[None], anch[jj:jj+1], anch[jj:jj+1],
                                    slot_feat=m._slot_features(anch[jj:jj+1], jj, n, R))
        ctx = h + m.sp_out_emb(so[jj:jj+1]); u = torch.einsum("naj,nj->na", frame, (y[jj:jj+1] - anch_y[jj:jj+1]))
        for a in range(3):
            zi, ld = m.flow.spline.inverse(u[:, a:a+1], m.flow.heads[a](ctx))
            lp = lp + (-0.5 * (zi / T) ** 2 - 0.5 * LOG2PI - math.log(T) + ld).squeeze()
            ctx = torch.cat([ctx, u[:, a:a+1]], -1)
        lp = lp + ldyx[jj]
    return lp


def energy(x, s):
    return float(ka_energy(x[None], s.long()[None], 100.0)[0])


K = 4
for T in (1.0, 0.7, 0.5, 0.35):
    acc, dU_best = [], []
    for ci in range(900, 925):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 2:
            continue
        x, s, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R); n = x.shape[0]
        a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
        blk = torch.zeros(n, dtype=torch.bool, device=dev); blk[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
        u_old = energy(x, s); lq_x = float(tempered_block_logp(x, s, blk, T))
        lus = [-beta * u_old - lq_x]; dUs = []
        for _ in range(16):
            xn, sn, lq = tempered_sample_block(x, s, blk, T)
            un = energy(xn, sn); lus.append(-beta * un - float(lq)); dUs.append(un - u_old)
        dU_best.append(min(dUs))
        lu = torch.tensor(lus[1:], device=dev); sfwd = torch.logsumexp(lu, 0)
        js = int(torch.multinomial(torch.softmax(lu, 0), 1, generator=gen))
        lu_rev = lu.clone(); lu_rev[js] = lus[0]
        acc.append(float(torch.rand((), device=dev, generator=gen).log() < (sfwd - torch.logsumexp(lu_rev, 0))))
    n = len(acc)
    print(f"T={T:.2f}: MTM accept={100*sum(acc)/n:.0f}%  median best-trial dU={st.median(dU_best):+.2f}  "
          f"(2D ref: 30% accept, mild dU)", flush=True)
