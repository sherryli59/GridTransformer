"""mu(T) ladder driver: finish T=0.1 (Newton + revalidate), then descend 0.085/0.075/0.065.
Recipe per rung (proven at T=0.2): warm-start mu from previous T -> short adaptive loop
(20 iters, tracks the structure as it re-equilibrates) -> long fixed-mu equilibrated
measurement -> ONE full-strength Newton step -> re-validate. Declares converged-for-purpose
at L1<=0.04 or after 2 Newton rounds (tail whack-a-mole saturates ~5-7% frac_sup in ~1%-mass
bins; that is tail-mixing hysteresis, not mu bias -- documented at T=0.2).
Everything saved incrementally; ladder state in poly_mu_ladder_status.txt after each stage.
"""
import subprocess, sys, json, time
import numpy as np
import torch

REPO = "/mnt/ssd/GridTransformer"
LOGD = f"{REPO}/reports/logs-2026-07-17"
ART = f"{REPO}/liquid_coupling_flow/artifacts"
STATUS = f"{LOGD}/poly_mu_ladder_status.txt"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def run(cmd):
    log(f"RUN {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"FAILED rc={r.returncode}: {r.stderr[-800:]}")
        sys.exit(1)
    return r.stdout


def newton(T, val_pt):
    V = torch.load(val_pt, weights_only=False)
    mp = f"{ART}/poly_mu_T{T}.pt"
    M = torch.load(mp, weights_only=False)
    mu = np.asarray(M["mu"], dtype=np.float64).copy()
    pm, pt_ = np.asarray(V["p_meas"]), np.asarray(V["p_target"])
    upd = T * np.log(pt_ / np.maximum(pm, 1e-6))
    mu += upd
    mu -= np.sum(mu * pt_)
    M["mu"] = mu
    M.setdefault("newton_steps", []).append({"from": val_pt, "max_upd": float(np.abs(upd).max())})
    torch.save(M, mp)
    log(f"T={T}: Newton step, max|dmu|={np.abs(upd).max():.4f}")


def validate(T, tag, sweeps=150000):
    out_pt = f"{LOGD}/poly_mu_validate_T{T}_{tag}.pt"
    out = run(["nice", "-19", "python", f"{LOGD}/poly_mu_validate.py", "--T", str(T),
               "--sweeps", str(sweeps), "--out", out_pt])
    log(out.strip().replace("\n", " | "))
    V = torch.load(out_pt, weights_only=False)
    rel = np.asarray(V["rel"])
    l1 = float(np.abs(np.asarray(V["p_meas"]) - np.asarray(V["p_target"])).sum())
    return out_pt, float(rel.max()), l1


def finish_rung(T, mu_init=None, loop_iters=20):
    if mu_init:
        run(["nice", "-19", "python", f"{LOGD}/poly_mu_calibrate.py", "--T", str(T),
             "--iters", str(loop_iters), "--sweeps_per_iter", "6000",
             "--mu_init", mu_init, "--out", f"{ART}/poly_mu_T{T}.pt"])
        log(f"T={T}: warm-start loop ({loop_iters} iters) done")
    for round_ in (1, 2):
        vp, fs, l1 = validate(T, f"r{round_}")
        log(f"T={T} round {round_}: frac_sup={fs:.4f} L1={l1:.4f}")
        if l1 <= 0.04 and fs <= 0.08:
            log(f"T={T}: CONVERGED-FOR-PURPOSE (L1 {l1:.3f}, tail {fs:.3f})")
            return
        newton(T, vp)
    vp, fs, l1 = validate(T, "final")
    log(f"T={T} FINAL: frac_sup={fs:.4f} L1={l1:.4f} "
        f"({'ok' if l1 <= 0.05 else 'REVIEW NEEDED'})")


if __name__ == "__main__":
    log("=== mu ladder driver start ===")
    # T=0.1: loop already ran; measurement poly_mu_validate_T0.1_v1 may exist -- Newton off it
    import os
    v1 = f"{LOGD}/poly_mu_validate_T0.1_v1.pt"
    if os.path.exists(v1):
        newton(0.1, v1)
        finish_rung(0.1, mu_init=None)          # rounds of validate+newton only
    else:
        finish_rung(0.1, mu_init=None)
    prev = f"{ART}/poly_mu_T0.1.pt"
    for T in (0.085, 0.075, 0.065):
        finish_rung(T, mu_init=prev)
        prev = f"{ART}/poly_mu_T{T}.pt"
    log("=== mu ladder driver DONE ===")
