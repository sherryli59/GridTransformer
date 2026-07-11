"""Profile the block-training hot path to find the CPU-bound bottleneck."""
import time
import torch
from liquid_coupling_flow.ka3d_cavity_ar import _mic, morton_code_3d
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_scaffold_ar import KA3DScaffoldAR, label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_block import block_log_prob

dev = "cuda"
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
m = KA3DScaffoldAR().to(dev)
E3, E1 = torch.zeros(0, 3, device=dev), torch.zeros(0, dtype=torch.long, device=dev)

pool = []
for i in range(6):
    c = carve(X[i], S[i], X[i][7 * i % 512], 2.0, L)
    x, s, _ = label_to_scaffold(_mic(c["x_in"], X[i][7 * i % 512], L), c["s_in"], 2.0)
    n = x.shape[0]
    blk = torch.zeros(n, dtype=torch.bool, device=dev); blk[torch.randperm(n, device=dev)[:4]] = True
    pool.append((x, s, blk, n))
x0, s0, blk0, n0 = pool[0]
print(f"n0={n0}", flush=True)


def timed(name, fn, iters=50, sync=True):
    for _ in range(3):
        fn()
    if sync:
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        fn()
    if sync:
        torch.cuda.synchronize()
    print(f"  {name:34s} {1000*(time.time()-t)/iters:7.2f} ms", flush=True)


print("component timings:", flush=True)
timed("morton_code_3d(n0)", lambda: morton_code_3d(x0, 2.0))
timed("fixed_ball_scaffold(n0)", lambda: fixed_ball_scaffold(n0, 2.0, dev))
timed("block_log_prob fwd", lambda: block_log_prob(m, x0, s0, blk0, E3, E1, 2.0))


def fwd_bwd():
    m.zero_grad(set_to_none=True)
    lp = block_log_prob(m, x0, s0, blk0, E3, E1, 2.0)
    (-lp).backward()


timed("block_log_prob fwd+bwd", fwd_bwd, iters=30)


def full_step():
    m.zero_grad(set_to_none=True)
    loss = X.new_zeros(())
    for (x, s, blk, n) in pool:
        loss = loss - block_log_prob(m, x, s, blk, E3, E1, 2.0) / int(blk.sum())
    (loss / len(pool)).backward()


timed("full training step (batch=6)", full_step, iters=20)
