import numpy as np
import torch
from liquid_coupling_flow.ptu.tables import build_tables_u
from liquid_coupling_flow.ptu.ladder import Ladder
from liquid_coupling_flow.ptu.qc import bcy_qc
from liquid_coupling_flow.tests.test_ptu_kernels import cavity  # fixture


def test_qc_calibration(cavity):
    *_, xin, sin, bnd, sb = cavity
    g = torch.Generator().manual_seed(7)
    x = xin.numpy()
    assert bcy_qc(x, x, g) > 0.98                       # self ~ 1
    rng = np.random.default_rng(0)
    u = rng.standard_normal((x.shape[0], 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    xr = u * 2.0 * rng.random((x.shape[0], 1)) ** (1 / 3)
    assert bcy_qc(x, xr, g) < 0.05                      # random ~ 0


def test_ladder_smoke_and_trips_counter(cavity):
    allx, alls, n, n_tot, xin, *_ = cavity
    # IDENTICAL rungs (coords/temps all equal): this toy verifies TRIPS/exchange
    # BOOKKEEPING, not ladder physics (physics is Task 5's job). With identical
    # tables and betas, every exchange dlog=0 -> acceptance is exactly 1 ->
    # walkers deterministically shuttle end-to-end and trips accumulate
    # regardless of seed (a real coarse ladder makes trips a seed lottery --
    # measured ~8% per-seed at 300 sweeps on a genuine [1.0,0.6,0.2] toy).
    coords = [1.0, 1.0, 1.0]                            # 3-rung toy, u-mode, identical
    temps = [0.5, 0.5, 0.5]
    lad = Ladder(allx, alls, n, n_tot, "u", coords, temps, nch=2, exch_mean=1, seed=0)
    lad.randomize_stack_B(50)
    g = torch.Generator().manual_seed(7)
    res = lad.run(300, rec_every=100, qc_fn=lambda a, b: bcy_qc(a, b, g))
    # identical-rung tables -> dlog=0 always -> deterministic shuttling, trips >> 1
    assert res["trips"] >= 1
    assert len(res["qcA"]) == 3 and len(res["qcB"]) == 3
    assert res["exch_att"].min() > 0
    # stack A starts at reference: its first recorded qc must be high
    assert res["qcA"][0] > 0.4
