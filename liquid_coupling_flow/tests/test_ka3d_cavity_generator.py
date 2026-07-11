import torch

from liquid_coupling_flow.ka3d_cavity_carve import carve_batch
from liquid_coupling_flow.ka3d_cavity_generator import CavityGenerator, cavity_fm_loss, collate_cavities


def _batch():
    torch.manual_seed(1); B, N = 3, 256; L = (N / 1.2) ** (1 / 3)
    x = torch.rand(B, N, 3) * L; s = torch.zeros(B, N, dtype=torch.long); s[:, -20:] = 1
    # R=1.6 is valid in this deliberately roomy test box.
    pairs = carve_batch(x, s, L, [1.6], 1, torch.Generator().manual_seed(2))
    for p in pairs: p["T"] = .5
    return collate_cavities(pairs, n_max=max(p["n_in"] for p in pairs), n_ctx_max=24)


def test_forward_mask_and_sample_spherical_support():
    b = _batch(); model = CavityGenerator(b["x_in"].shape[1], hidden_nf=16, n_layers=2, n_ctx_max=24)
    v = model(torch.zeros(3), b["x_in"], b["mask"], b["s_in"], b["x_ctx"],
              b["s_ctx"], b["ctx_mask"], b["R"], b["T"])
    assert v.shape == b["x_in"].shape and torch.isfinite(v).all()
    assert torch.equal(v[~b["mask"]], torch.zeros_like(v[~b["mask"]]))
    x = model.sample(b["s_in"], b["mask"], b["x_ctx"], b["s_ctx"], b["ctx_mask"], b["R"], b["T"], 4)
    assert (x.norm(dim=-1)[b["mask"]] < b["R"][:, None].expand_as(b["mask"])[b["mask"]]).all()


def test_tiny_training_reduces_flow_matching_loss():
    torch.manual_seed(5); b = _batch(); model = CavityGenerator(b["x_in"].shape[1], 24, 2, 24)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    torch.manual_seed(10); initial = float(cavity_fm_loss(model, b))
    for _ in range(35):
        loss = cavity_fm_loss(model, b); opt.zero_grad(); loss.backward(); opt.step()
    torch.manual_seed(10); final = float(cavity_fm_loss(model, b))
    assert final < initial
