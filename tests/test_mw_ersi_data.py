import torch

from liquid_coupling_flow.mw.mw_ersi_data import L_for_N, write_ersi_dataset


def test_dataset_format(tmp_path):
    N = 8
    L = L_for_N(N)
    cfgs = torch.rand(320, N, 3) * L * 1.5 - 0.2 * L
    bank = tmp_path / "bank.pt"
    torch.save({"cfgs": cfgs}, bank)
    pos_p, sp_p = tmp_path / "pos.pt", tmp_path / "sp.pt"
    info = write_ersi_dataset(str(pos_p), str(sp_p), [str(bank)], thin_events=2, val_frac=0.1, seed=0)
    pos = torch.load(pos_p, weights_only=False)
    sp = torch.load(sp_p, weights_only=False)
    assert pos.shape[1:] == (N, 3) and (pos >= 0).all() and (pos < L).all()
    assert sp.shape == pos.shape[:2] and (sp == 0).all()
    assert info["N"] == N and abs(info["L"] - L) < 1e-9 and info["n_train"] > 0
    assert (tmp_path / "pos_val.pt").exists() and (tmp_path / "sp_val.pt").exists()
