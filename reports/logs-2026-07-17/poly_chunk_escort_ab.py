"""PoC A/B: does a chunk-stressed-trained flow reduce NCMC work? COST-BLIND (user directive).

Chunked protocol: C=4 instant Delta-lambda=0.25 switches (work accumulated exactly), each
followed by R local sweeps. Arm ESCORT additionally applies one flow-MH block move IMMEDIATELY
after each switch (fresh stress = where the 41% learnable signal lives), before the sweeps.
Flow move = SwapBlockFlow proposal on the pair-centered k=8 block, IDENTITY sigma path at the
current interpolated sigmas, MH-accepted against the fixed-lambda Hamiltonian via block_dU +
exact logq ratio -> a detailed-balanced kernel at pi_lambda; the W ledger is untouched by
flow moves (they are DB at fixed lambda, contributing no switching work).
Verdict: W(escort) vs W(plain) distributions per |dsig| bin; PoC = statistically lower W.
"""
import sys, time
import numpy as np
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
sys.path.insert(0, "/mnt/ssd/GridTransformer/reports/logs-2026-07-17")
from poly_ncmc_v2 import local_sweep, build_local_set
from poly_gate_acceptance import block_dU
from liquid_coupling_flow.poly.model import seed_numba
from liquid_coupling_flow.poly.swap_flow import SwapBlockFlow, sigma_to_bin
from poly_ncmc_probe import _pair_env_e
MAX_PAIR_DIST = 2.0    # near pairs only: a single joint block is the right escort geometry

T = 0.085; BETA = 1.0 / T
BINS = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.45), (0.45, 0.9)]
CHUNKS = [0.25, 0.5, 0.75, 1.0]
R_SW = 50
PER_BIN = 70
CKPT = "liquid_coupling_flow/artifacts/poly_chunkescort_T0.085_k8_best.pt"

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(CKPT, map_location=dev, weights_only=False)
ar = ck["args"]
flow = SwapBlockFlow(k_max=ar["k_max"], m_env=ar["m_env"], hidden_nf=ar["hidden_nf"],
                     n_layers=ar["n_layers"], n_sig_bins=ar.get("n_sig_bins", 8),
                     base_w=ar.get("base_w", 0.05)).to(dev)
flow.load_state_dict(ck["state_dict"]); flow.eval()
gen = torch.Generator(); gen.manual_seed(3)

D = torch.load("reports/logs-2026-07-17/poly_bank_fixed_run3.pt", weights_only=False)
rec = D[(3, T)]
L = float(rec["L"]); N = rec["x"].shape[1]
buf = np.zeros(N, dtype=np.int64)


def flow_move(x, sg, i, j):
    """One DB flow-MH move on the pair-centered k=8 block at the current (fixed) sigmas.
    Returns (accepted, n/a). Mutates x in place on accept."""
    mid = 0.5 * (x[i] + x[j]); dd = x - mid; dd -= L * np.round(dd / L)
    order = np.argsort((dd ** 2).sum(1))
    others = [m for m in order if m != i and m != j][:6]
    idx = np.array([i, j] + others, dtype=np.int64)          # pair always in block, no dupes
    mask = np.zeros(N, bool); mask[idx] = True
    cen = x[idx].mean(0)
    def cent(a):
        b = a - cen
        return b - L * np.round(b / L)
    env_all = np.where(~mask)[0]
    de = x[env_all] - cen; de -= L * np.round(de / L)
    env_idx = env_all[np.argsort((de ** 2).sum(1))[:64]]
    xb = cent(x[idx]).astype(np.float32)
    ex = cent(x[env_idx]).astype(np.float32)
    sb = sg[idx].copy(); es = sg[env_idx].copy()
    t = lambda a, d=torch.float32: torch.as_tensor(a, dtype=d, device=dev).unsqueeze(0)
    xn, lqf = flow.propose_batch(t(xb), t(sb, torch.float64), t(sb, torch.float64),
                                 t(ex), t(es, torch.float64), gen)
    xn0 = np.asarray(xn[0], dtype=np.float64)
    lqr = flow.logq_of_batch(t(xb), torch.as_tensor(xn0, dtype=torch.float32, device=dev).unsqueeze(0),
                             t(sb, torch.float64), t(sb, torch.float64), t(ex), t(es, torch.float64))
    dU = block_dU(xn0, sb, cent(x[env_idx]).astype(np.float64), es, L) - \
        block_dU(cent(x[idx]).astype(np.float64), sb, cent(x[env_idx]).astype(np.float64), es, L)
    a_log = -BETA * dU + float(lqr[0]) - float(lqf[0])
    if np.log(np.random.random() + 1e-300) < a_log:
        x[idx] = xn0 + cen
        return True
    return False


def run_attempt(fr, s0, i, j, escort, seed):
    x = fr.copy(); sg = s0.copy(); si, sj = sg[i], sg[j]
    seed_numba(seed)
    np.random.seed(seed % (2**31))
    W = 0.0
    fl_acc = 0; fl_att = 0
    for lam in CHUNKS:
        eA = _pair_env_e(x, sg, i, j, L)
        sg[i] = (1 - lam) * si + lam * sj
        sg[j] = (1 - lam) * sj + lam * si
        eB = _pair_env_e(x, sg, i, j, L)
        W += eB - eA
        if escort:
            fl_att += 1
            fl_acc += bool(flow_move(x, sg, i, j))
        nS = build_local_set(x, i, j, 3.0, L, buf)
        for _ in range(R_SW):
            local_sweep(x, sg, L, BETA, i, j, 3.0, 0.12, buf, nS)
    return W, fl_acc, fl_att


if __name__ == "__main__":
    rng = np.random.default_rng(9)
    out = {}
    for arm in ("plain", "escort"):
        t0 = time.time()
        res = {b: [] for b in BINS}; need = {b: PER_BIN for b in BINS}
        fa = ft = 0
        guard = 0
        rng = np.random.default_rng(9)          # same pair sequence both arms
        while any(v > 0 for v in need.values()) and guard < 25000:
            guard += 1
            fr_i = guard % 16
            i = int(rng.integers(N)); j = int(rng.integers(N - 1)); j += (j >= i)
            s0 = rec["sigs"][fr_i].astype(np.float64)
            ds = abs(float(s0[i] - s0[j]))
            bb = next((b for b in BINS if b[0] <= ds < b[1]), None)
            if bb is None or need[bb] <= 0:
                continue
            xf = rec["x"][fr_i]
            dv = xf[i] - xf[j]; dv = dv - L * np.round(dv / L)
            if float((dv ** 2).sum()) ** 0.5 > MAX_PAIR_DIST:
                need[bb] += 1
                continue
            need[bb] -= 1
            W, a, t_ = run_attempt(rec["x"][fr_i].astype(np.float64), s0, i, j,
                                   arm == "escort", 60000 + guard)
            res[bb].append(W); fa += a; ft += t_
        out[arm] = {str(b): np.array(v) for b, v in res.items()}
        line = [f"{arm:>7}"]
        for b in BINS:
            Wv = np.array(res[b])
            acc = np.minimum(1, np.exp(np.clip(-BETA * Wv, -700, 0)))
            line.append(f"{b}: Wmed {np.median(Wv):+.2f} acc {acc.mean():.4f}")
        extra = f" | flow acc {fa}/{ft}" if arm == "escort" else ""
        print(f"[{time.time()-t0:5.0f}s] " + " | ".join(line) + extra, flush=True)
        torch.save(out, "reports/logs-2026-07-17/poly_chunk_escort_ab_T0.085.pt")
    print("AB DONE", flush=True)
