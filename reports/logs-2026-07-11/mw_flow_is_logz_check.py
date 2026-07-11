"""Closer: one-shot IS logZ bound from the COMMITTED 2026-07-10 diagnostic (192 flow samples + logq,
step-28k ckpt, none of the new SMC code). Exact-Jacobian proof => q~ exactly normalized => E[Zhat]=Z
=> Markov: P(log Zhat > logZ + t) <= e^-t. Result: log Zhat = 1222.07 (max logw 1227.33; exactly ~1
sample expected above logZ+5 at logZ~1222 — self-consistent). => TRUE logZ ~ 1222; the certified
uniform-path 1198.1 was ~24 nats LOW (mutation-limited systematic bias, REPRODUCIBLE across its 3
seeds — multi-seed agreement is agreement OF THE BIAS). Flow-path SMC logZ (1215-1228) was RIGHT."""
import torch, math
d = torch.load("liquid_coupling_flow/mw/artifacts/mw_ersi_sharp_diag2.pt", map_location="cpu", weights_only=False)
from liquid_coupling_flow.mw.mw_energy import mw_energy_chunked, T_STAR, RHO_STAR
N = 64; L = (N/RHO_STAR)**(1/3); beta = 1.0/T_STAR
X, lq = d["X"], d["logq"].double()
U = mw_energy_chunked(X.cuda(), L).double().cpu()
logw = -beta*U - lq
lzhat = float(torch.logsumexp(logw, 0) - math.log(logw.shape[0]))
print(f"log Zhat(IS, B=192) = {lzhat:.2f}   max logw = {float(logw.max()):.2f}")
