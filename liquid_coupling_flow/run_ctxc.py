"""Isolate the _ctx_c fix: keep coord-0 context as the original SUM (it fit train-size
better than mean), but make the EXCLUDED-VOLUME context _ctx_c intensive (weighted mean,
not sum) -- the component diagnosed as still L-extensive (it pools over a coord-0 strip
spanning the full box height). Train N=16, re-run the size-transfer test.

Win condition: raw overlaps at N=64 drop toward the train-size level (was 0.43 sum/sum,
0.37 mean/sum; uniform baseline ~0.72).

Run:  python -m liquid_coupling_flow.run_ctxc
"""

from __future__ import annotations

import torch

from liquid_coupling_flow.train_ar_ckpt import main as train
from liquid_coupling_flow.size_transfer import main as transfer

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train(device=dev, steps=7000, ctx0_reduce="sum", ctxc_reduce="mean",
          out_name="ar_flow_ctxcmean_ckpt.pt")
    transfer(device=dev, ckpt_name="ar_flow_ctxcmean_ckpt.pt")
