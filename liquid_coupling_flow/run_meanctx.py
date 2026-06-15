"""Workstream A: retrain the N=16 flow with INTENSIVE coord-0 context (mean, not sum)
then re-run the size-transfer test. The extensive sum was diagnosed as the reason the
raw flow collapses to uniform at larger N; mean-pooling is size-invariant.

Run:  python -m liquid_coupling_flow.run_meanctx
"""

from __future__ import annotations

import torch

from liquid_coupling_flow.train_ar_ckpt import main as train
from liquid_coupling_flow.size_transfer import main as transfer

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train(device=dev, steps=7000, ctx0_reduce="mean", out_name="ar_flow_meanctx_ckpt.pt")
    transfer(device=dev, ckpt_name="ar_flow_meanctx_ckpt.pt")
