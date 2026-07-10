"""Tiny-scale smoke of the 3D dataset generator: it runs end-to-end and emits a valid, finite-energy dataset of
the expected shape. (Production scale is a GPU launch; this only guards the driver.)"""
import torch

from liquid_coupling_flow.ka_dataset_3d import generate_dataset


def test_generate_dataset_smoke_tiny(tmp_path):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = tmp_path / "tiny.pt"
    generate_dataset(out, N=64, n_chains=3, equil_sweeps=10, decorr_sweeps=5, polish_sweeps=2,
                     per_chain=2, device=dev, seed=0)
    d = torch.load(out, map_location="cpu", weights_only=False)
    assert d["x"].shape == (3 * 2, 64, 3) and d["s"].shape == (3 * 2, 64)
    assert torch.isfinite(d["energy_per_N"]).all()
    assert abs(d["composition_B"] - 0.2) < 1e-9 and d["x"].shape[1] / d["L"] ** 3 == d["N"] / d["L"] ** 3
    # species count preserved per config (80:20 -> round(0.2*64)=13 B particles)
    assert torch.equal(d["s"].sum(1), torch.full((6,), round(0.2 * 64), dtype=d["s"].dtype))
