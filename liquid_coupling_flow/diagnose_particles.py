"""Diagnostic: why does the particle flow stay at the identity (NLL = base const)?

Checks, in order:
  1. conditioner output variance + gradient flow at init (is the GNN producing/learning
     position-dependent params, or collapsed?);
  2. can the flow fit a STRUCTURED target (jittered lattice: strong x-x correlations, no
     overlaps)? If -log q drops well below the base constant, the conditioner works and the
     LJ failure is about the energy/data regime; if it stays pinned at the base const, the
     coupling/conditioner itself can't inject correlations.
"""

from __future__ import annotations

import torch

from liquid_coupling_flow.particles import build_particle_flow


def jittered_lattice(M, N, L, jitter, device):
    k = int(round(N ** 0.5))
    assert k * k == N, "use a square N"
    g = torch.arange(k, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)  # [N,2]
    base = grid * (L / k)
    x = base[None] + jitter * torch.randn(M, N, 2)
    return torch.remainder(x, L).to(device)


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    N, L = 16, 5.0
    flow = build_particle_flow(N=N, d=2, L=L, n_layers=12, num_bins=8, cutoff=2.4, hidden=64).to(dev)

    x = jittered_lattice(256, N, L, 0.15, dev)
    a = torch.rand(256, N, 2, device=dev) * L

    # --- 1. init: param variance + gradient flow ---
    p_xcond = flow.layers[1].conditioner(x)     # cond on real structure x
    print(f"init: params(cond on structured x) std={p_xcond.std():.3e} (zero-init -> expect ~0)")
    loss = -flow.log_prob(x, a).mean()
    print(f"init loss {loss.item():.3f}   base const {flow.base_logp:.3f}")
    loss.backward()
    nm = flow.layers[1].conditioner.node_mlp[-1].weight.grad
    em = [p.grad for p in flow.layers[1].conditioner.edge_mlp.parameters() if p.grad is not None]
    print(f"grad node_mlp.last {nm.norm():.3e}   edge_mlp {sum(g.norm()**2 for g in em)**0.5:.3e}")

    # --- 2. can it fit a structured target? ---
    flow.zero_grad()
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    for step in range(800):
        xb = jittered_lattice(512, N, L, 0.15, dev)
        ab = torch.rand(512, N, 2, device=dev) * L
        loss = -flow.log_prob(xb, ab).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
        opt.step()
        if step % 200 == 0 or step == 799:
            with torch.no_grad():
                ps = flow.layers[1].conditioner(xb).std().item()
            print(f"  step {step:3d}  -logq {loss.item():.3f}  (base {flow.base_logp:.1f})  param-std {ps:.3e}")


if __name__ == "__main__":
    main()
