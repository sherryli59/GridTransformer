import os, torch
from liquid_coupling_flow.mw.mw_generator_v4 import MWGlobalAR
from liquid_coupling_flow.mw.mw_block_ar import MWPeriodicBlockAR
from liquid_coupling_flow.mw.mw_smc_portfolio import smc_run_portfolio

N, L = 27, 3.9


def test_portfolio_smoke_completes_and_ledgers():
    torch.manual_seed(0)
    q0 = MWGlobalAR(d_model=32, n_layers=1, n_heads=2, n_mix=4, rail_k=4).eval()
    blk = MWPeriodicBlockAR(d_model=32, n_rbf=6).eval()
    out = smc_run_portfolio(q0, blk, N, L, beta=2.0, B=8, ess_target=0.5,
                            n_site_sweeps=1, suffix_cfg=(1, 0.3, 8, 12),
                            twoblob_cfg=(1, 0.7, 3, 1.2), seed=0, save_tag="smoketest")
    assert out["history"][-1]["lam"] == 1.0
    assert out["evals"]["cost_units"] > 0
    lams = [h["lam"] for h in out["history"]]
    assert all(b >= a for a, b in zip(lams, lams[1:]))
    assert os.path.exists("liquid_coupling_flow/mw/artifacts/mw_kportfolio_smoketest_N27.pt")
    assert torch.isfinite(out["logZ"])


def test_real_block_model_through_two_blob():
    """Review-mandated: wire the REAL untrained MWPeriodicBlockAR through two_blob_move
    inside a short portfolio run. It will mostly reject (expected); the harness must still
    complete, history must carry twoblob_acc entries, and no interface exception may escape."""
    torch.manual_seed(1)
    q0 = MWGlobalAR(d_model=32, n_layers=1, n_heads=2, n_mix=4, rail_k=4).eval()
    blk = MWPeriodicBlockAR(d_model=32, n_rbf=6).eval()
    out = smc_run_portfolio(q0, blk, N, L, beta=2.0, B=8, ess_target=0.5,
                            n_site_sweeps=1, suffix_cfg=(1, 0.3, 8, 12),
                            twoblob_cfg=(2, 0.7, 3, 1.2), seed=3, save_tag="tbtest")
    # harness completed the full bridge
    assert out["history"][-1]["lam"] == 1.0
    assert torch.isfinite(out["logZ"])
    # two_blob ran on the real block model and was ledgered every rung with lam < lam_tb
    tb_rows = [h for h in out["history"] if "twoblob_acc" in h]
    assert len(tb_rows) > 0
    # acceptance is a valid probability even when every proposal rejects (no div-by-zero)
    assert all(0.0 <= h["twoblob_acc"] <= 1.0 for h in tb_rows)
