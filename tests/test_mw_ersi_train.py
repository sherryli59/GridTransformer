import torch

from liquid_coupling_flow.mw.mw_ersi_data import L_for_N
from liquid_coupling_flow.mw.mw_ersi_train import train


def test_train_smoke(tmp_path):
    N = 8
    L = L_for_N(N)
    bank = tmp_path / "bank.pt"
    torch.save({"cfgs": torch.rand(256, N, 3) * L}, bank)
    out = tmp_path / "m.pt"
    info = train(N=N, K=4, hidden_nf=16, n_layers=1, steps=40, batch=16, lr=1e-3,
                 out=str(out), bank_paths=[str(bank)], seed=0, checkpoint_every=10)
    assert out.exists()
    ck = torch.load(out, weights_only=False)
    assert ck["N"] == N and ck["K"] == 4 and "state_dict" in ck
    assert torch.isfinite(torch.tensor([info["loss_first"], info["loss_last"]])).all()
    assert (tmp_path / "m_checkpoints" / "last.ckpt").exists()
    assert (tmp_path / "m_checkpoints" / "metrics" / "version_0" / "metrics.csv").exists()
