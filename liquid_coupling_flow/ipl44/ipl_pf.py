"""Professor forcing (bounded arm, G0-PF passed: drift share 0.44): fine-tune the IPL44 curve-flow transformer
with an adversarial term that makes FREE-RUNNING hidden-state segments indistinguishable from TEACHER-FORCED
ones. Density parameterization untouched -> exact likelihood preserved; PF only reshapes training gradients.

H_TF: one causal call m._local(x_true, s_true, ...) + curve_feat -> [B,N,d_model].
H_FR: rollout from a true prefix of random start j0, `freelen` free steps; sampled outputs are written detached
(gradient reaches theta through each step's encoder, the standard PF approximation for mixed discrete/continuous
outputs). D = GRU over the segment. L_G = NLL + beta_pf*BCE(D(FR),1); L_D = BCE(D(TF),1)+BCE(D(FR),0).

Gate G-PF: FR discard (U > 2*U_ref_max, 512 samples) 0.987 -> <0.9 or core mass -30%, val NLL degradation <5%.
Usage: python -m liquid_coupling_flow.ipl44.ipl_pf TAG [beta_pf] [steps] [freelen] [d_gru]"""
import os, sys, time, torch, torch.nn as nn, torch.nn.functional as F
from liquid_coupling_flow.ipl44.ipl_model import make_ipl_model
from liquid_coupling_flow.ipl44.ipl_energy import ipl_box, ipl_energy

dev = "cuda" if torch.cuda.is_available() else "cpu"
N, L = ipl_box(); Lf = float(L)
tag = sys.argv[1]
beta_pf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
freelen = int(sys.argv[4]) if len(sys.argv) > 4 else 8
d_gru = int(sys.argv[5]) if len(sys.argv) > 5 else 64
B, lr, warmup, ema_decay = 32, 1e-4, 200, 0.999
ART = os.path.join(os.path.dirname(__file__), "data")

ck = torch.load(f"{ART}/ipl44_curveflow.pt", map_location=dev, weights_only=False)
m = make_ipl_model(num_bins=ck["num_bins"], tail_bound=ck["tail_bound"], knn=ck["knn"],
                   arc_range=ck["arc_range"], device=dev)
m.load_state_dict(ck["state_dict"]); nB = ck["n_B"]

D = "/mnt/ssd/GridTransformer/datasets"
x = torch.remainder(torch.load(f"{D}/ipl44_T0.1_positions.pt", weights_only=False).float(), Lf)
sp = torch.load(f"{D}/ipl44_T0.1_species.pt", weights_only=False).long()
xtr, stra, xva, sva = x[:9000].to(dev), sp[:9000].to(dev), x[9000:].to(dev), sp[9000:].to(dev)
U_ref_max = float(ipl_energy(xva, sva).max())


class Disc(nn.Module):
    def __init__(self, d_in, d_h):
        super().__init__()
        self.gru = nn.GRU(d_in, d_h, batch_first=True)
        self.out = nn.Linear(d_h, 1)

    def forward(self, hseq):
        _, hn = self.gru(hseq)
        return self.out(hn[-1]).squeeze(-1)


disc = Disc(m.d_model, d_gru).to(dev)
opt_g = torch.optim.Adam(m.parameters(), lr=lr)
opt_d = torch.optim.Adam(disc.parameters(), lr=1e-4)
import copy
ema = copy.deepcopy(m).eval()


def tf_hidden(x1, s1):
    """Causal TF hidden states for curve-ordered configs: [B,N,d_model]."""
    order = m.geo._curve_order(x1, N)
    xo = torch.gather(x1, 1, order[..., None].expand(-1, -1, 2)); so = torch.gather(s1, 1, order)
    sc = m.geo._scaffold(N, dev)
    ctx, _ = m._local(xo, so, sc, Lf, N)
    return ctx + m._curve_feat(N, dev)[None], xo, so


def fr_segment(xo, so, j0):
    """Roll `freelen` free steps from the true prefix 0..j0-1; collect hidden states WITH theta-grad."""
    sc = m.geo._scaffold(N, dev); arc = m._arc_scale(N); cf = m._curve_feat(N, dev)
    pos = xo.clone(); spc = so.clone()
    rem = torch.zeros(xo.shape[0], m.n_species, device=dev)
    rem[:, 0] = (so[:, j0:] == 0).sum(1).float() + 0  # remaining counts consistent with suffix budget
    rem[:, 1] = (so[:, j0:] == 1).sum(1).float()
    hs = []
    for j in range(j0, min(j0 + freelen, N)):
        h, origin = m._step(pos, spc, sc[j], j, Lf)
        h = h + cf[j]
        hs.append(h)
        with torch.no_grad():
            sj = torch.multinomial(F.softmax(m._species_logits(h, rem), -1), 1).squeeze(-1)
            ab, _ = m.flow.sample(h)
            rem[torch.arange(xo.shape[0], device=dev), sj] -= 1
        pos = pos.clone(); spc = spc.clone()
        pos[:, j] = torch.remainder(origin + ab * arc, Lf).detach(); spc[:, j] = sj
    return torch.stack(hs, dim=1)                                        # [B, freelen, d_model]


@torch.no_grad()
def gen_metrics(model, n=512):
    xs, ss = model.sample(n, N, n_B=nB, device=dev)
    U = ipl_energy(torch.remainder(xs, Lf), ss)
    discard = float((U > 2 * U_ref_max).float().mean())
    # core mass proxy: fraction of NN distances below 0.9 (sigma_AA)
    d = xs[:, :, None, :] - xs[:, None, :, :]; d = d - Lf * torch.round(d / Lf)
    r = d.norm(dim=-1) + torch.eye(N, device=dev) * 99
    core = float((r.min(-1).values < 0.9).float().mean())
    nll = float((-model.log_prob(xva[:512], sva[:512]) / N).mean())
    return discard, core, nll


m.eval()
d0, c0, n0 = gen_metrics(m)                                              # baseline in EVAL mode (matches report)
m.train()
print(f"[{tag}] beta_pf {beta_pf} freelen {freelen} gru {d_gru} | BASELINE discard {d0:.3f} core {c0:.4f} val nll/N {n0:.4f}", flush=True)

best = {"discard": d0}; t0 = time.time()
for step in range(steps):
    for g in opt_g.param_groups:
        g["lr"] = lr * min(1.0, (step + 1) / warmup)
    idx = torch.randint(0, xtr.shape[0], (B,), device=dev)
    H_tf, xo, so = tf_hidden(xtr[idx], stra[idx])
    j0 = int(torch.randint(4, N - freelen, (1,)))
    H_fr = fr_segment(xo, so, j0)
    seg_tf = H_tf[:, j0:j0 + freelen]
    # --- D step (detached) ---
    ld = F.binary_cross_entropy_with_logits(disc(seg_tf.detach()), torch.ones(B, device=dev)) + \
         F.binary_cross_entropy_with_logits(disc(H_fr.detach()), torch.zeros(B, device=dev))
    opt_d.zero_grad(); ld.backward(); opt_d.step()
    # --- G step: NLL + fool D on FR ---
    nll = (-m.log_prob(xtr[idx], stra[idx]) / N).mean()
    adv = F.binary_cross_entropy_with_logits(disc(H_fr), torch.ones(B, device=dev))
    lg = nll + beta_pf * adv
    if not torch.isfinite(lg):
        print(f"NON-FINITE at {step}"); opt_g.zero_grad(); continue
    opt_g.zero_grad(); lg.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt_g.step()
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), m.parameters()):
            pe.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
    if step % 500 == 0 or step == steps - 1:
        dd, cc, nn_ = gen_metrics(ema)
        ckpt = {"state_dict": ema.state_dict(), "raw_state_dict": m.state_dict(), "step": step,
                "beta_pf": beta_pf, "freelen": freelen, "d_gru": d_gru,
                "discard": dd, "core": cc, "nll": nn_,
                "num_bins": ck["num_bins"], "tail_bound": ck["tail_bound"], "knn": ck["knn"],
                "arc_range": ck["arc_range"], "n_B": nB}
        torch.save(ckpt, f"{ART}/pf_{tag}_last.pt")
        star = ""
        if dd < best["discard"] and nn_ < n0 * 1.05:
            best = {"discard": dd}; torch.save(ckpt, f"{ART}/pf_{tag}_best.pt"); star = " *BEST*"
        print(f"[{tag}] step {step:5d} nll {float(nll):.4f} adv {float(adv):.3f} ld {float(ld):.3f} "
              f"| gen discard {dd:.3f} core {cc:.4f} val nll/N {nn_:.4f}{star} | {time.time()-t0:.0f}s", flush=True)
print(f"[{tag}] DONE best discard {best['discard']:.3f} (baseline {d0:.3f})", flush=True)
