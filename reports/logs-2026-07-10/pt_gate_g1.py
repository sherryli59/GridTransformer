"""Tiny, resumable G1 sanity for the 3D KABLJ cavity PT machinery.

This intentionally is not a paper reproduction: four boundaries, three radii,
one target temperature, plus one high-temperature mixing smoke.  Each completed
unit is checkpointed so an interrupted GPU run can resume.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from liquid_coupling_flow.ka3d_pt_cavity import equilibrate_cavity, two_arm_converged


def _tail_mean(q):
    n = max(3, (len(q) + 2) // 3)
    return float(torch.tensor(q[-n:]).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt"))
    p.add_argument("--artifact", type=Path, default=Path("liquid_coupling_flow/artifacts/ka3d_pt_g1_tiny.pt"))
    p.add_argument("--n-centers", type=int, default=4)
    p.add_argument("--radii", type=float, nargs="+", default=[1.6, 2.2, 2.4])
    p.add_argument("--skip-high-t", action="store_true")
    p.add_argument("--n-sweep", type=int, default=1500)
    p.add_argument("--n-rep", type=int, default=12)
    p.add_argument("--exch-every", type=int, default=50)
    p.add_argument("--t-rec", type=int, default=50)
    p.add_argument("--randomize-sweeps", type=int, default=200)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    d = torch.load(a.dataset, map_location=a.device, weights_only=False)
    xall, sall, L = d["x"].float(), d["s"].long(), float(d["L"])
    radii = tuple(a.radii)
    state = torch.load(a.artifact, map_location="cpu", weights_only=False) if a.artifact.exists() else {
        "dataset": str(a.dataset), "N": int(xall.shape[1]), "L": L, "radii": radii,
        "n_centers": a.n_centers, "n_sweep": a.n_sweep, "rows": []}
    done = {(float(r["T"]), float(r["R"]), int(r["center_id"])) for r in state["rows"]}
    t0 = time.time()
    units = [(0.5, R, c) for R in radii for c in range(a.n_centers)]
    if not a.skip_high_t:
        units.append((1.0, 1.6, 0))
    for unit_id, (temp, R, c) in enumerate(units):
        if (temp, R, c) in done:
            continue
        # Different bulk frames and particle-centred cavities give distinct
        # frozen boundaries while guaranteeing a non-empty overlap core.
        frame = -(c + 1)
        x = xall[frame].unsqueeze(0).to(a.device); s = sall[frame].unsqueeze(0).to(a.device)
        # Preserve the same quenched boundary centre as R changes, matching the
        # paper's paired-radius design.
        particle_id = (37 * c + 11) % x.shape[1]
        center = x[0, particle_id].clone()
        delta = x - center; delta = delta - L * torch.round(delta / L)
        mobile = delta.square().sum(-1) < R * R
        kw = dict(R=R, L=L, T=temp, n_rep=a.n_rep, n_sweep=a.n_sweep,
                  exch_every=a.exch_every, t_rec=a.t_rec,
                  randomize_sweeps=a.randomize_sweeps)
        ref = equilibrate_cavity(x, s, mobile, center, init="ref", seed=1000 + unit_id, **kw)
        rnd = equilibrate_cavity(x, s, mobile, center, init="random", seed=2000 + unit_id, **kw)
        conv = two_arm_converged(ref["q_c_traj"], rnd["q_c_traj"], q_tol=0.1)
        row = {"T": temp, "R": R, "center_id": c, "n_mobile": int(mobile.sum()),
               "q_ref": _tail_mean(ref["q_c_traj"]), "q_random": _tail_mean(rnd["q_c_traj"]),
               "gap": abs(_tail_mean(ref["q_c_traj"]) - _tail_mean(rnd["q_c_traj"])),
               "converged": conv, "exchange_prob_ref": ref["exchange_prob"],
               "exchange_prob_random": rnd["exchange_prob"], "ref_traj": ref["q_c_traj"],
               "random_traj": rnd["q_c_traj"], "ref_samples": ref["x_samples"],
               "random_samples": rnd["x_samples"], "s_samples": ref["s_samples"]}
        state["rows"].append(row); a.artifact.parent.mkdir(parents=True, exist_ok=True); torch.save(state, a.artifact)
        print(f"[G1 T={temp:.1f} R={R:.1f} c={c}] mobile={row['n_mobile']} "
              f"ref={row['q_ref']:.3f} rnd={row['q_random']:.3f} gap={row['gap']:.3f} "
              f"conv={conv} elapsed={(time.time()-t0)/60:.1f}m", flush=True)
    cold = [r for r in state["rows"] if r["T"] == 0.5]
    all_conv = all(r["converged"] for r in cold)
    means = {R: sum(0.5 * (r["q_ref"] + r["q_random"]) for r in cold if r["R"] == R) /
                 max(1, sum(r["R"] == R for r in cold)) for R in radii}
    trend = len(radii) < 2 or means[min(radii)] > means[max(radii)]
    state["verdict"] = {"all_two_arm_converged": all_conv, "high_to_lower_overlap": trend,
                        "mean_q_by_R": means, "passed": all_conv and trend}
    torch.save(state, a.artifact)
    print(f"[G1 VERDICT] means={means} all_converged={all_conv} decreasing={trend} "
          f"-> {'PASS' if all_conv and trend else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
