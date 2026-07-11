"""G2 generator P(q_c) comparison on the same boundaries as fresh PT arms."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from liquid_coupling_flow.ka3d_cavity_carve import carve, reassemble
from liquid_coupling_flow.ka3d_cavity_eval import compare_overlap_distributions, generator_overlap_samples, pt_overlap_samples
from liquid_coupling_flow.ka3d_cavity_generator import CavityGenerator
from liquid_coupling_flow.ka3d_pt_cavity import equilibrate_cavity, two_arm_converged


def main():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", type=Path, default=Path("liquid_coupling_flow/artifacts/ka3d_cavity_generator_g2.pt")); p.add_argument("--out", type=Path, default=Path("liquid_coupling_flow/artifacts/ka3d_generator_gate_g2.pt")); p.add_argument("--n-sweep", type=int, default=2000); p.add_argument("--n-gen", type=int, default=40); p.add_argument("--n-boundaries", type=int, default=2); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args(); ck = torch.load(a.checkpoint, map_location=a.device, weights_only=False); cfg = ck["config"]
    model = CavityGenerator(cfg["n_max"], cfg["hidden"], cfg["layers"], cfg["n_ctx_max"]).to(a.device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    data = torch.load(ck["dataset"], map_location=a.device, weights_only=False)
    bx, bs, L = data["x"].float(), data["s"].long(), float(data["L"])
    results = []
    for R in (1.6, 2.4):
        for c in range(a.n_boundaries):
            # Last frames are disjoint from train_cavity_g2's first frames.
            xb, sb = bx[-(c + 1)], bs[-(c + 1)]
            center = xb[(53 * c + 17) % xb.shape[0]].clone()
            pair = carve(xb, sb, center, R, L); pair["T"] = .5
            if pair["n_in"] > model.n_max:
                raise ValueError(f"held-out cavity has {pair['n_in']} particles > model.n_max={model.n_max}")
            x, s = reassemble(pair); x=x[None]; s=s[None]; mobile = pair["mobile"][None]
            kw=dict(R=R,L=L,T=.5,n_rep=12,n_sweep=a.n_sweep,exch_every=25,t_rec=50,randomize_sweeps=500)
            seed0=int(R*1000)+10*c
            ref=equilibrate_cavity(x,s,mobile,center,init="ref",seed=seed0+1,**kw)
            rnd=equilibrate_cavity(x,s,mobile,center,init="random",seed=seed0+2,**kw)
            conv=two_arm_converged(ref["q_c_traj"],rnd["q_c_traj"],.1)
            qpt=pt_overlap_samples(ref,rnd,center,L); qgen=generator_overlap_samples(model,pair,a.n_gen)
            metrics=compare_overlap_distributions(qgen,qpt); metrics.update({"R":R,"center_id":c,"converged":conv,"q_pt":qpt.cpu(),"q_generator":qgen.cpu()}); results.append(metrics)
            print(f"[G2 R={R} c={c}] conv={conv} gen={metrics['generator_mean']:.3f}+-{metrics['generator_sem']:.3f} pt={metrics['pt_mean']:.3f}+-{metrics['pt_sem']:.3f} diff={metrics['mean_abs_diff']:.3f} TV={metrics['histogram_tv']:.3f}",flush=True)
    passed=all(r["converged"] and r["mean_abs_diff"] <= max(.1,2*(r["generator_sem"]+r["pt_sem"])) for r in results)
    a.out.parent.mkdir(parents=True,exist_ok=True); torch.save({"results":results,"passed":passed,"checkpoint":str(a.checkpoint)},a.out)
    print(f"[G2 VERDICT] {'PASS' if passed else 'FAIL'} -> {a.out}",flush=True)


if __name__ == "__main__": main()
