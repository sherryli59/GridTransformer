"""LEARNED PAIR SELECTOR: screened NCMC diameter-swap kernel (model + deployment).

Exactness contract (screened NCMC kernel)
------------------------------------------
Pick a pair uniformly in the |dsigma| in [0.1, 0.9] window (the window is symmetric:
|dsigma| is invariant under swapping the pair's labels). Evaluate a screening
probability s_theta(x, pair) in [s_min, 1] (s_min=0.02 is a floor that keeps the chain
ergodic -- no pair is EVER strictly forbidden). With probability s_theta run the NCMC
protocol; otherwise the attempt ends right there (no move, no energy change). If the
protocol runs, accept the resulting configuration with

    A = min(1, exp(-beta*W) * s_theta(x_end, pair) / s_theta(x_start, pair))

The screening ratio s_theta(x_end)/s_theta(x_start) is exactly what makes this
detailed-balanced despite s_theta being state-dependent: the FORWARD proposal density
of ever reaching x_end via this pair carries a factor s_theta(x_start) (probability we
chose to run the protocol from x_start), and the REVERSE proposal (x_end -> x_start,
same pair, same |dsigma| so same window membership) carries s_theta(x_end). Those are
exactly the two factors that appear in the acceptance ratio's proposal-density term,
so they cancel against the same factors in the *unconditional* forward/reverse
transition probabilities and what's left is the plain Metropolis-Hastings ratio in W
familiar from the un-screened kernel (Nilmeier et al., PNAS 2011 argument, now with an
extra state-dependent but exactly-cancelling accept/reject gate). A neural-network
approximation error in s_theta can only ever waste attempts (screen out a pair that
would have been cheap) or withhold potential extra throughput (screen in a pair that
was doomed, wasting CPU but not violating detailed balance) -- it can NEVER bias the
stationary distribution, because the ratio s_theta(x_end)/s_theta(x_start) is evaluated
with the SAME function on both sides of the SAME transition, whatever that function is.

This module provides:
  - SelectorNet: MLP(40 -> 128 -> 128 -> 1, GELU) regressing standardized W-hat.
  - train_selector(): trains on poly_selector_dataset_T<T>.pt (90/10 split BY FRAME,
    not by sample -- samples drawn from the same frame share strongly correlated local
    environments, so a per-sample split would leak).
  - screening_prob(what, beta, s_min, c): the deployment gate, s_theta = clip(exp(-beta
    * max(what,0) * c), s_min, 1).
  - evaluate_screening(): THE GATE -- on held-out run-3 frames (never seen in training),
    compares CPU-per-accepted-exchange for screened vs unscreened NCMC using continuous
    (expected-value) acceptance/cost bookkeeping rather than a single finite-sample
    binary draw (lower-variance estimate of the same quantity).
"""
import sys, time, json
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/mnt/ssd/GridTransformer")
sys.path.insert(0, "/mnt/ssd/GridTransformer/reports/logs-2026-07-17")
from liquid_coupling_flow.poly.model import seed_numba  # noqa: E402
from poly_ncmc_v2 import ncmc_work_local  # noqa: E402  -- imported, never modified
from poly_selector_data import (  # noqa: E402
    pair_features, sample_pair_in_window, FEATURE_NAMES, N_FEATURES, R_LOC, N_STEPS,
    STEP, DS_LO, DS_HI,
)

MODEL_PATH_FMT = "reports/logs-2026-07-17/poly_selector_model_T{T}.pt"
S_MIN_DEFAULT = 0.02
C_DEFAULT = 0.5
NN_EVAL_COST = 1.0   # single-move-attempt units; ~1e-4 x a full protocol (negligible)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SelectorNet(nn.Module):
    def __init__(self, in_dim=N_FEATURES, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def standardize(a, mean, std):
    return (a - mean) / std


def predict_what(model, scaler, feats_raw):
    """feats_raw: (n, N_FEATURES) numpy or torch. Returns raw-space W-hat, numpy."""
    if isinstance(feats_raw, np.ndarray):
        feats_raw = torch.as_tensor(feats_raw, dtype=torch.float32)
    fmean = torch.as_tensor(scaler["feat_mean"], dtype=torch.float32)
    fstd = torch.as_tensor(scaler["feat_std"], dtype=torch.float32)
    with torch.no_grad():
        z = (feats_raw - fmean) / fstd
        pred_std = model(z)
        what = pred_std * scaler["target_std"] + scaler["target_mean"]
    return what.numpy()


def screening_prob(what, beta, s_min=S_MIN_DEFAULT, c=C_DEFAULT):
    """Deployment screening probability. what: predicted W-hat (raw units, scalar or
    array). c is a sharpness knob (default 0.5, softened so the s-ratio doesn't kill
    real accepts -- see module contract). Always in [s_min, 1]."""
    what = np.asarray(what, dtype=np.float64)
    raw = np.exp(-beta * np.maximum(what, 0.0) * c)
    return np.clip(raw, s_min, 1.0)


# ---------------------------------------------------------------------------
# Dataset / split / training
# ---------------------------------------------------------------------------
def load_dataset(path):
    D = torch.load(path, weights_only=False)
    return D


def frame_split(meta, val_frac=0.1, seed=31):
    """meta: (n,5) array [run, frame, i, j, ds]. Returns boolean is_val mask, split BY
    (run,frame) not by sample."""
    keys = meta[:, :2]
    uniq = sorted(set(map(tuple, keys.tolist())))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_val = max(1, round(val_frac * len(uniq)))
    val_keys = set(uniq[p] for p in perm[:n_val])
    is_val = np.array([tuple(r) in val_keys for r in keys.tolist()])
    return is_val, sorted(val_keys)


def train_selector(dataset_path, T, epochs=200, lr=1e-3, batch_size=256, seed=31,
                    huber_delta=1.0, hidden=128, val_frac=0.1):
    D = load_dataset(dataset_path)
    feats = D["features"].astype(np.float64)
    W = D["W"].astype(np.float64)
    meta = D["meta"]
    assert feats.shape[1] == N_FEATURES

    is_val, val_keys = frame_split(meta, val_frac=val_frac, seed=seed)
    Xtr, Xva = feats[~is_val], feats[is_val]
    Wtr, Wva = W[~is_val], W[is_val]
    print(f"[selector-train] n={len(W)} train={len(Wtr)} val={len(Wva)} "
          f"val_frames={val_keys}", flush=True)

    feat_mean = Xtr.mean(axis=0); feat_std = Xtr.std(axis=0) + 1e-8
    target_mean = float(Wtr.mean()); target_std = float(Wtr.std() + 1e-8)
    scaler = {"feat_mean": feat_mean, "feat_std": feat_std,
              "target_mean": target_mean, "target_std": target_std}

    Xtr_t = torch.as_tensor(standardize(Xtr, feat_mean, feat_std), dtype=torch.float32)
    Xva_t = torch.as_tensor(standardize(Xva, feat_mean, feat_std), dtype=torch.float32)
    Ytr_t = torch.as_tensor(standardize(Wtr, target_mean, target_std), dtype=torch.float32)
    Yva_t = torch.as_tensor(standardize(Wva, target_mean, target_std), dtype=torch.float32)

    torch.manual_seed(seed)
    model = SelectorNet(in_dim=N_FEATURES, hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.HuberLoss(delta=huber_delta)

    n = Xtr_t.shape[0]
    rng = np.random.default_rng(seed + 1)
    best_val = float("inf"); best_state = None
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        ep_loss = 0.0
        for b0 in range(0, n, batch_size):
            idx = perm[b0:b0 + batch_size]
            xb = Xtr_t[idx]; yb = Ytr_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= n

        model.eval()
        with torch.no_grad():
            vpred = model(Xva_t)
            vloss = loss_fn(vpred, Yva_t).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  ep {ep:4d} train_huber={ep_loss:.4f} val_huber={vloss:.4f}", flush=True)

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        vpred_std = model(Xva_t).numpy()
    What_va = vpred_std * target_std + target_mean
    mae = float(np.mean(np.abs(What_va - Wva)))

    from scipy.stats import spearmanr
    rho, _ = spearmanr(What_va, Wva)

    decile = max(1, int(np.ceil(0.1 * len(Wva))))
    order = np.argsort(What_va)  # ascending predicted W-hat = predicted cheapest first
    cheapest_idx = order[:decile]
    true_cheap = Wva < 1.0
    n_true_cheap = int(true_cheap.sum())
    tail_recall = float(true_cheap[cheapest_idx].sum() / max(n_true_cheap, 1))

    metrics = {
        "n_train": len(Wtr), "n_val": len(Wva), "val_frames": val_keys,
        "val_mae": mae, "val_spearman": float(rho), "val_huber": best_val,
        "tail_recall_decile": tail_recall, "n_true_cheap_val": n_true_cheap,
        "decile_size": decile,
    }
    print(f"[selector-train] val MAE={mae:.3f} Spearman={rho:.3f} "
          f"tail-recall(decile)={tail_recall:.3f} (n_true_cheap={n_true_cheap}/{len(Wva)})",
          flush=True)
    return model, scaler, metrics


# ---------------------------------------------------------------------------
# THE GATE: screened vs unscreened CPU-per-accepted-exchange on held-out run 3
# ---------------------------------------------------------------------------
def _load_run3_frames(T):
    D = torch.load("reports/logs-2026-07-17/poly_bank_fixed_run3.pt", weights_only=False)
    rec = D[(3, T)]
    L = float(rec["L"])
    return [{"run": 3, "frame": fr, "L": L,
              "x": rec["x"][fr].astype(np.float64), "sig": rec["sigs"][fr].astype(np.float64)}
             for fr in range(rec["x"].shape[0])]


def _run_attempts(frames, T, n_attempts, seed):
    """Fresh NCMC attempts (r_loc=R_LOC, n_steps=N_STEPS) on held-out frames. Returns
    per-attempt: ds, W, cost (nS_step.sum()), x_start,sig_start,i,j (for feature eval
    at start), x_end,sig_end (mutated arrays post-protocol, for feature eval at end)."""
    rng = np.random.default_rng(seed)
    seed_numba(seed + 5000)
    beta = 1.0 / T
    schedule = np.linspace(1.0 / N_STEPS, 1.0, N_STEPS)
    n_frames = len(frames)
    out = []
    for k in range(n_attempts):
        fr = frames[k % n_frames]
        i, j = sample_pair_in_window(rng, fr["sig"])
        x0 = fr["x"]; s0 = fr["sig"]; L = fr["L"]
        x = x0.copy(); sig = s0.copy()
        dW_step = np.zeros(N_STEPS); nS_step = np.zeros(N_STEPS)
        W = ncmc_work_local(x, sig, L, beta, i, j, schedule, 1, R_LOC, STEP, dW_step, nS_step)
        out.append({
            "i": i, "j": j, "L": L, "ds": abs(float(s0[i] - s0[j])),
            "W": float(W), "cost": float(nS_step.sum()),
            "x_start": x0, "sig_start": s0, "x_end": x, "sig_end": sig,
        })
    return out


def evaluate_screening(model, scaler, T, n_attempts=600, s_min=S_MIN_DEFAULT, c=C_DEFAULT,
                        seed_screened=9001, seed_unscreened=9002):
    beta = 1.0 / T
    frames = _load_run3_frames(T)

    screened = _run_attempts(frames, T, n_attempts, seed_screened)
    unscreened = _run_attempts(frames, T, n_attempts, seed_unscreened)

    # --- screened arm: continuous (expected-value) bookkeeping ---
    cpu_screened = 0.0; acc_screened = 0.0; transport_screened = 0.0
    for a in screened:
        feat_start = pair_features(a["x_start"], a["sig_start"], a["L"], a["i"], a["j"])
        feat_end = pair_features(a["x_end"], a["sig_end"], a["L"], a["i"], a["j"])
        what_start = float(predict_what(model, scaler, feat_start[None, :])[0])
        what_end = float(predict_what(model, scaler, feat_end[None, :])[0])
        s_start = float(screening_prob(what_start, beta, s_min, c))
        s_end = float(screening_prob(what_end, beta, s_min, c))
        p_run = s_start
        cost_k = NN_EVAL_COST * (1.0 + p_run) + p_run * a["cost"]
        # A = min(1, exp(-beta W) * s_end/s_start); unconditional accept = p_run * A
        #   = min(p_run, exp(-beta W) * s_end)
        accept_k = min(p_run, np.exp(np.clip(-beta * a["W"], -700, 700)) * s_end)
        cpu_screened += cost_k
        acc_screened += accept_k
        transport_screened += accept_k * a["ds"]

    # --- unscreened arm: baseline, s_theta==1 everywhere (no gating) ---
    cpu_unscreened = 0.0; acc_unscreened = 0.0; transport_unscreened = 0.0
    for a in unscreened:
        cost_k = a["cost"]
        accept_k = min(1.0, np.exp(np.clip(-beta * a["W"], -700, 700)))
        cpu_unscreened += cost_k
        acc_unscreened += accept_k
        transport_unscreened += accept_k * a["ds"]

    cpu_per_acc_screened = cpu_screened / max(acc_screened, 1e-12)
    cpu_per_acc_unscreened = cpu_unscreened / max(acc_unscreened, 1e-12)
    gate_ratio = cpu_per_acc_unscreened / cpu_per_acc_screened

    result = {
        "n_attempts": n_attempts, "T": T, "s_min": s_min, "c": c,
        "cpu_per_accept_screened": cpu_per_acc_screened,
        "cpu_per_accept_unscreened": cpu_per_acc_unscreened,
        "gate_ratio": gate_ratio, "gate_pass": bool(gate_ratio >= 3.0),
        "expected_accepts_screened": acc_screened, "expected_accepts_unscreened": acc_unscreened,
        "transport_per_cpu_screened": transport_screened / max(cpu_screened, 1e-12),
        "transport_per_cpu_unscreened": transport_unscreened / max(cpu_unscreened, 1e-12),
    }
    print(f"[THE GATE] CPU/accept screened={cpu_per_acc_screened:.1f} "
          f"unscreened={cpu_per_acc_unscreened:.1f} ratio={gate_ratio:.2f}x "
          f"(need >=3x) -> {'PASS' if result['gate_pass'] else 'FAIL'}", flush=True)
    return result


def save_selector(path, model, scaler, metrics, gate_result, T):
    torch.save({
        "state_dict": model.state_dict(), "scaler": scaler, "metrics": metrics,
        "gate_result": gate_result, "T": T, "feature_names": FEATURE_NAMES,
        "hidden": 128, "s_min": S_MIN_DEFAULT, "c": C_DEFAULT,
        "r_loc": R_LOC, "n_steps": N_STEPS, "ds_window": (DS_LO, DS_HI),
    }, path)


def load_selector(path):
    D = torch.load(path, weights_only=False)
    model = SelectorNet(in_dim=N_FEATURES, hidden=D["hidden"])
    model.load_state_dict(D["state_dict"])
    model.eval()
    return model, D["scaler"], D


if __name__ == "__main__":
    T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.085
    dataset_path = sys.argv[2] if len(sys.argv) > 2 else f"reports/logs-2026-07-17/poly_selector_dataset_T{T}.pt"
    model, scaler, metrics = train_selector(dataset_path, T)
    gate_result = evaluate_screening(model, scaler, T)
    out_path = MODEL_PATH_FMT.format(T=T)
    save_selector(out_path, model, scaler, metrics, gate_result, T)
    print(f"[selector] saved model -> {out_path}", flush=True)
    print(json.dumps({"metrics": metrics, "gate": gate_result}, indent=2, default=str))
