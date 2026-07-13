"""Split the gated_big training step cost: data pipeline (gated AR + OT) vs FM loss fwd+bwd.
3 steps, big config (hid192/L6, 8 cav x 16 draws, chunk 32), CUDA-synced timers."""
import sys, time
import torch
sys.path.insert(0, "reports/logs-2026-07-12")
from train_cavity_egnn_flow import (load_ar_model, load_split, sample_training_batch, augment_so3,
                                    fm_loss, TRAIN_DATA, GatedARBase)
from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow

dev = "cuda"
gen = torch.Generator(device=dev).manual_seed(0)
ar = GatedARBase(load_ar_model(dev), cut=0.55, max_rounds=32)
X, S, L = load_split(TRAIN_DATA, dev, expect_chains=112)
flow = CavityBlockFlow(n_cage=48, k=8, r_c=2.5, hidden_nf=192, n_layers=6, n_species=2,
                       max_neighbors=16).to(dev)
opt = torch.optim.AdamW(flow.parameters(), lr=3e-4)


def sync():
    torch.cuda.synchronize()


for it in range(3):
    sync(); t0 = time.time()
    batch = sample_training_batch(ar, X, S, L, gen, 8, 48, [1.6, 2.0, 2.5, 3.0],
                                  n_cav=8, M=16, pos_temp=0.4)
    sync(); t_data = time.time() - t0
    batch = augment_so3(batch, gen)
    n_rows = batch["x_t"].shape[0]
    opt.zero_grad(set_to_none=True)
    sync(); t0 = time.time()
    t_fwd = 0.0
    for lo in range(0, n_rows, 32):
        sl = slice(lo, min(lo + 32, n_rows))
        sub = {kk: (vv[sl] if torch.is_tensor(vv) else vv) for kk, vv in batch.items()}
        sync(); tf = time.time()
        l = fm_loss(flow, sub, 8) * (sub["x_t"].shape[0] / n_rows)
        sync(); t_fwd += time.time() - tf
        l.backward()
    sync(); t_loss = time.time() - t0
    opt.step()
    print(f"step {it}: data {t_data:6.2f}s | loss fwd+bwd {t_loss:6.2f}s (fwd alone {t_fwd:5.2f}s) "
          f"| rows {n_rows} | total {t_data + t_loss:6.2f}s", flush=True)
