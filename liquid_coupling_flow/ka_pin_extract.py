"""Extractors for point-to-set: stretched-exponential plateau Q_inf and threshold length xi with
vertical->horizontal error propagation (dxi = dQ / |dQ_inf/dl|)."""
import numpy as np
from scipy.optimize import curve_fit


def _sexp(t, Qinf, A, tau, beta):
    return Qinf + A * np.exp(-((t / np.maximum(tau, 1e-9)) ** beta))


def stretched_exp_fit(t, Q):
    t = np.asarray(t, float); Q = np.asarray(Q, float)
    if t.size < 4:
        qinf = float(Q[-1]) if Q.size else float("nan")
        return {"Qinf": qinf, "A": 0.0, "tau": 1.0, "beta": 1.0, "ok": False}
    p0 = [Q[-1], max(Q[0] - Q[-1], 1e-3), max(t[-1] / 3, 1.0), 0.7]
    bounds = ([0.0, 0.0, 1e-3, 0.2], [1.0, 1.5, 1e6, 1.5])
    try:
        p, _ = curve_fit(_sexp, t, Q, p0=p0, bounds=bounds, maxfev=20000)
        return {"Qinf": float(p[0]), "A": float(p[1]), "tau": float(p[2]), "beta": float(p[3]), "ok": True}
    except Exception:
        return {"Qinf": float(Q[-1]), "A": 0.0, "tau": float(t[-1]), "beta": 1.0, "ok": False}


def xi_threshold(lvals, Qinf, Qinf_err, thr, Qrand=0.108, dQ_tol=0.02):
    """xi = l where (Qinf - Qrand) crosses thr, by linear interpolation of the excess vs l.
    dxi = (dQ_tol (+) Qinf_err_local) / |local slope|. kind='range' if multiple crossings."""
    l = np.asarray(lvals, float); e = np.asarray(Qinf, float) - Qrand
    err = np.asarray(Qinf_err, float)
    crossings = []
    for i in range(len(l)):                                  # exact touches (any index, incl. last)
        if e[i] - thr == 0.0:
            crossings.append((float(l[i]), min(i, len(l) - 2)))   # slope index clamped to a valid interval
    for i in range(len(l) - 1):                              # interior sign changes
        a, b = e[i] - thr, e[i + 1] - thr
        if a * b < 0:
            frac = a / (a - b)
            crossings.append((float(l[i] + frac * (l[i + 1] - l[i])), i))
    if not crossings:
        return {"xi": None, "dxi": None, "kind": "none"}
    def dxi_at(i):
        slope = abs((e[i + 1] - e[i]) / (l[i + 1] - l[i])) if i + 1 < len(l) else 1e-9
        slope = max(slope, 1e-9)
        eloc = 0.5 * (err[i] + err[min(i + 1, len(l) - 1)])
        return float(np.hypot(dQ_tol, eloc) / slope)
    if len(crossings) == 1:
        xi, i = crossings[0]
        return {"xi": float(xi), "dxi": dxi_at(i), "kind": "point"}
    xis = [c[0] for c in crossings]
    return {"xi": (float(min(xis)), float(max(xis))), "dxi": max(dxi_at(c[1]) for c in crossings),
            "kind": "range"}
