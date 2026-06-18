"""Phase 3: train BOTH benchmark arms (AR + coupling) on a small-N KA reference by
forward-KL (MLE), then check in-distribution that each reproduces the reference. Saves
the trained flows for the size-transfer benchmark. Trains at small N (AR's per-particle
loop is slow at large N -> train small, transfer large, which is the design anyway)."""
from __future__ import annotations
import os, time, numpy as np, torch
from liquid_coupling_flow.ka_flow_ar import KAARFlow
from liquid_coupling_flow.ka_flow_coupling import KACouplingFlow
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.particles import min_image

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def overlap_frac(x, L, N, thresh=0.7):
    _, r = min_image(x, L)
    r = r + torch.eye(N, device=x.device)[None] * 1e3
    return float((r.min(dim=2).values.reshape(-1) < thresh).float().mean())


def train_one(flow, data, s, L, N, device, steps, lr=1e-3, B=128, tag=""):
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        loss = -flow.log_prob(data[idx], s).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"  [{tag}] step {step:4d} -logq {loss.item():.2f} (base {-flow.base_logp:.1f}) "
                  f"{time.time()-t0:.0f}s", flush=True)
    flow.eval()
    with torch.no_grad():
        x, _ = flow.sample(min(512, data.shape[0]), s, device=device)
        ov = overlap_frac(x, L, N)
        U = (ka_energy(x, s, L) / N)
        good = torch.isfinite(U)
        Uf = U[good].median().item()
    return ov, Uf


def main(N=100, steps=3000, device="cuda" if torch.cuda.is_available() else "cpu"):
    ref = torch.load(os.path.join(ART, f"ka_reference_N{N}.pt"), map_location=device)
    data, s, L = ref["x"].to(device), ref["s"].to(device), ref["L"]
    U_ref = (ka_energy(data, s, L) / N)
    ov_ref = overlap_frac(data[:512], L, N)
    print(f"reference N={N}: {data.shape[0]} configs, <U>/N {U_ref.mean():.3f}, overlaps {ov_ref:.3f}", flush=True)

    arms = {
        "AR": KAARFlow(N=N, L=L, num_bins=24, hidden=128, cutoff=2.4).to(device),
        "coupling": KACouplingFlow(N=N, L=L, n_cycles=4, num_bins=24, hidden=128, cutoff=2.4).to(device),
    }
    out = {"N": N, "L": L, "s": s.cpu(), "ref_U": U_ref.mean().item(), "ref_ov": ov_ref}
    for name, flow in arms.items():
        print(f"=== training {name} ({sum(p.numel() for p in flow.parameters())/1e3:.0f}k params) ===", flush=True)
        ov, Uf = train_one(flow, data, s, L, N, device, steps, tag=name)
        print(f"  {name}: in-dist sample overlaps {ov:.3f} (ref {ov_ref:.3f}), "
              f"<U>/N median {Uf:.3f} (ref {U_ref.mean():.3f})", flush=True)
        out[name] = {"state_dict": flow.state_dict(), "overlaps": ov, "U_median": Uf}
    torch.save(out, os.path.join(ART, f"ka_arms_N{N}.pt"))
    print(f"saved both trained arms -> artifacts/ka_arms_N{N}.pt", flush=True)


if __name__ == "__main__":
    main()
