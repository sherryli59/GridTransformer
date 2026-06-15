"""Train the AR particle flow and SAVE a checkpoint + a batch of samples.

The demos never persisted weights, so every plot meant a full retrain. This trains
once (forward-KL on cached 2D LJ MCMC data) and writes everything a diagnostic plot
needs to `artifacts/ar_flow_ckpt.pt`:
    {config, state_dict, samples [S,N,2], logq [S], (MCMC data lives in /tmp cache)}.

Run:  python -m liquid_coupling_flow.train_ar_ckpt
"""

from __future__ import annotations

import os
import time

import torch

from liquid_coupling_flow.particles_ar import ARParticleFlow
from liquid_coupling_flow.demo_particles_mle import get_data
from liquid_coupling_flow.energy import lj_energy
from liquid_coupling_flow.particles import min_image

ART = os.path.join(os.path.dirname(__file__), "artifacts")
CKPT = os.path.join(ART, "ar_flow_ckpt.pt")


def main(device="cuda", steps=7000, lr=1e-3, num_bins=24, hidden=128, n_samples=8000,
         ctx0_reduce="sum", ctxc_reduce="sum", out_name="ar_flow_ckpt.pt"):
    os.makedirs(ART, exist_ok=True)
    torch.manual_seed(0)
    cfg = dict(N=16, L=5.0, kT=1.0, cutoff=2.4, num_bins=num_bins, hidden=hidden,
               ctx0_reduce=ctx0_reduce, ctxc_reduce=ctxc_reduce)
    N, L, kT, cutoff = cfg["N"], cfg["L"], cfg["kT"], cfg["cutoff"]

    data = get_data(N, L, kT, cutoff, device).to(device)
    U_data = (lj_energy(data, L, cutoff=cutoff, shift=True) / N).mean().item()
    print(f"data {data.shape[0]} configs  <U>/N {U_data:.3f}  "
          f"ctx0={ctx0_reduce} ctxc={ctxc_reduce}", flush=True)

    flow = ARParticleFlow(N=N, L=L, num_bins=num_bins, hidden=hidden, cutoff=cutoff,
                          ctx0_reduce=ctx0_reduce, ctxc_reduce=ctxc_reduce).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    B = 512
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        loss = -flow.log_prob(data[idx]).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"  step {step:4d}  -logq {loss.item():.3f}  (base {-flow.base_logp:.1f})  "
                  f"{time.time()-t0:.0f}s", flush=True)

    flow.eval()
    with torch.no_grad():
        xs, logq = [], []
        for _ in range(0, n_samples, 2000):
            x_b, lq_b = flow.sample(2000, device=device)
            xs.append(x_b); logq.append(lq_b)
        x = torch.cat(xs)[:n_samples]
        logq = torch.cat(logq)[:n_samples]
        # quick health check
        U = lj_energy(x, L, cutoff=cutoff, shift=True)
        logw = -U / kT - logq
        logw = logw - torch.logsumexp(logw, 0)
        w = logw.exp()
        ess = float(1.0 / (w ** 2).sum() / w.shape[0])
        _, r = min_image(x, L)
        r = r + torch.eye(N, device=x.device)[None] * 1e3
        overlap = float((r.min(dim=2).values.reshape(-1) < 0.8).float().mean())
    print(f"samples: overlaps {overlap:.3f}  x-ESS {100*ess:.2f}%", flush=True)

    out_path = os.path.join(ART, out_name)
    torch.save({"config": cfg, "state_dict": flow.state_dict(),
                "samples": x.cpu(), "logq": logq.cpu(),
                "overlap": overlap, "ess": ess, "U_data": U_data}, out_path)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main(device="cuda" if torch.cuda.is_available() else "cpu")
