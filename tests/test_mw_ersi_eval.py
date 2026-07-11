import torch

from liquid_coupling_flow.mw.mw_ersi import load_ersi
from liquid_coupling_flow.mw.mw_ersi_data import L_for_N
from liquid_coupling_flow.mw.mw_ersi_eval import oneshot_is
from liquid_coupling_flow.mw.mw_ersi_train import train


def test_oneshot_is_runs(tmp_path):
    N = 8
    L = L_for_N(N)
    bank, ckpt = tmp_path / "bank.pt", tmp_path / "model.pt"
    torch.save({"cfgs": torch.rand(256, N, 3) * L}, bank)
    train(N=N, K=4, hidden_nf=16, n_layers=1, steps=10, batch=16,
          out=str(ckpt), bank_paths=[str(bank)], seed=0)
    out = oneshot_is(load_ersi(str(ckpt), "cpu"), B=32, seed=0)
    assert torch.isfinite(out["logw"]).all() and 1.0 <= out["ess"] <= 32.0
    assert "gr_raw" in out and out["gr_raw"][1].shape[0] > 0
