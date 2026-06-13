"""Clean-input validation NLL: val split mechanics + noise gating in eval mode.

The checkpoint monitor uses val/loss because the train loss is computed on
noise-corrupted inputs under input-corruption training (exposure-bias plan),
so min-train-loss would pin best.ckpt to the lowest-noise epoch.
"""

import numpy as np
import torch

from grid_transformer.models.transformer import GraphormerAR
from grid_transformer.training.lightning_module import LJTransferableDataModule


def _model(**kwargs):
    return GraphormerAR(
        K=16,
        d_model=32,
        n_layer=1,
        n_head=4,
        ida_spatial_dim=3,
        use_continuous_head=True,
        continuous_input=True,
        num_mixtures=3,
        arc_repr=True,
        **kwargs,
    )


class _DummyDataset:
    def __init__(self, n: int):
        half = n // 2
        self.sample_lengths = np.array([27] * half + [125] * (n - half), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.sample_lengths)

    def __getitem__(self, idx: int):
        return {"idx": torch.tensor(idx)}


def _datamodule(val_frac: float, n: int = 100, seed: int = 0) -> LJTransferableDataModule:
    dm = LJTransferableDataModule(batch_size=8, seed=seed, val_frac=val_frac)
    dm._train_ds = _DummyDataset(n)
    dm._split_val()
    return dm


def test_split_is_disjoint_exhaustive_and_deterministic():
    dm_a = _datamodule(val_frac=0.1)
    dm_b = _datamodule(val_frac=0.1)
    assert dm_a._val_indices is not None and dm_a._train_indices is not None
    assert len(dm_a._val_indices) == 10
    combined = np.sort(np.concatenate([dm_a._train_indices, dm_a._val_indices]))
    np.testing.assert_array_equal(combined, np.arange(100))
    np.testing.assert_array_equal(dm_a._val_indices, dm_b._val_indices)


def test_val_frac_zero_disables_validation():
    dm = _datamodule(val_frac=0.0)
    assert dm._val_indices is None
    assert dm.val_dataloader() is None


def test_dataloaders_cover_split_with_homogeneous_length_batches():
    dm = _datamodule(val_frac=0.1)
    full_lengths = dm._train_ds.sample_lengths

    seen_val = []
    for batch in dm.val_dataloader():
        idx = batch["idx"].numpy()
        assert np.unique(full_lengths[idx]).size == 1  # bucketed: one size per batch
        seen_val.extend(idx.tolist())
    np.testing.assert_array_equal(np.sort(seen_val), dm._val_indices)

    seen_train = []
    for batch in dm.train_dataloader():
        idx = batch["idx"].numpy()
        assert np.unique(full_lengths[idx]).size == 1
        seen_train.extend(idx.tolist())
    np.testing.assert_array_equal(np.sort(seen_train), dm._train_indices)


def test_noise_is_inactive_in_eval_mode():
    torch.manual_seed(0)
    model = _model(
        continuous_input_noise_delta_s=0.5,
        continuous_input_noise_fine=0.5,
        continuous_input_noise_prob=1.0,
    )
    model.eval()
    x = torch.zeros(2, 6, 4)
    y, rms = model._apply_continuous_input_noise(x)
    torch.testing.assert_close(y, x)
    assert float(rms) == 0.0


def test_training_and_validation_steps_route_prefixes(monkeypatch):
    model = _model()
    calls = []
    monkeypatch.setattr(
        model,
        "_shared_step",
        lambda batch, idx, prefix: (calls.append(prefix), torch.tensor(0.0))[1],
    )
    model.training_step({}, 0)
    model.validation_step({}, 0)
    assert calls == ["train", "val"]
