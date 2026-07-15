"""Continue an exact thermal-SMC endpoint with target-local MCMC sweeps."""
from __future__ import annotations

import argparse
import torch

from liquid_coupling_flow.mw.mw_thermal_smc import continue_target_smc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweeps", type=int, required=True)
    ap.add_argument("--report-every", type=int, default=20)
    ap.add_argument("--step", type=float)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    result = torch.load(a.input, map_location="cpu", weights_only=False)
    out = continue_target_smc(
        result, a.sweeps, report_every=a.report_every, step=a.step,
        seed=a.seed, save_path=a.out, verbose=True)
    print(f"saved {a.out}; counter={out['energy_counter']}", flush=True)


if __name__ == "__main__":
    main()
