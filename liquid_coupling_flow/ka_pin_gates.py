"""Blocking gates for point-to-set (in order): G-corr (correctness vs brute-force at an easy point),
G-conv (two-arm agreement AND run-length >= 3 tau), G-stick (third-init + 2x extension at the binding point),
drift_check (long-wavelength contamination). Failure of G-conv => report the point as a BOUND, never fit."""
import numpy as np
from liquid_coupling_flow.ka_pin_extract import stretched_exp_fit


def _qinf_tau(run):
    t = np.asarray(run["t"], float)
    Q = np.asarray(run["Q"], float)
    # For increasing data (Q[-1] > Q[0]), fit the residual (Q[-1] - Q),
    # which decays to 0. The actual plateau is Q[-1].
    if Q[-1] > Q[0]:
        Q_residual = Q[-1] - Q
        f = stretched_exp_fit(t, Q_residual)
        return Q[-1], f["tau"]
    else:
        # For decreasing data, fit normally
        f = stretched_exp_fit(t, Q)
        return f["Qinf"], f["tau"]


def g_conv(ref_run, scr_run, tol=0.02):
    q_ref, tau_r = _qinf_tau(ref_run); q_scr, tau_s = _qinf_tau(scr_run)
    tmax = min(max(ref_run["t"]), max(scr_run["t"]))
    run_ok = (tmax >= 3.0 * tau_r) and (tmax >= 3.0 * tau_s)         # Qinf<->tau degeneracy guard
    gap = abs(q_ref - q_scr)
    return {"passed": bool(run_ok and gap <= tol), "q_ref": q_ref, "q_scr": q_scr,
            "gap": float(gap), "tau": float(max(tau_r, tau_s)), "run_ok": bool(run_ok)}


def g_corr(learned_run, brute_run, tol=0.02):
    ql, _ = _qinf_tau(learned_run); qb, _ = _qinf_tau(brute_run)
    return {"passed": bool(abs(ql - qb) <= tol), "gap": float(abs(ql - qb))}


def g_stick(ref_run, third_run, ext_run, tol=0.02):
    """Binding-point co-sticking: a THIRD init (different equilibrium reference) and a 2x-extended run must
    reach the same plateau as the reference arm; two arms can co-stick at a wrong plateau in a glass."""
    q_ref, _ = _qinf_tau(ref_run); q_third, _ = _qinf_tau(third_run); q_ext, _ = _qinf_tau(ext_run)
    g3, ge = abs(q_ref - q_third), abs(q_ref - q_ext)
    return {"passed": bool(g3 <= tol and ge <= tol), "gap_third": float(g3), "gap_ext": float(ge)}


def drift_check(occ_registered_Q, occ_std_Q, tol=0.02):
    """Long-wavelength spot-check at largest l_c: disparity between locally-registered and standard Q_inf."""
    d = abs(occ_registered_Q - occ_std_Q)
    return {"flagged": bool(d > tol), "disparity": float(d)}
