"""Why is the verbatim-BCY exchange DEAD (0 accepts in ~400 tries, 0 round trips)? Measure per-rung dlog =
-beta_a[U_a(x_b)-U_a(x_a)] - beta_b[U_b(x_a)-U_b(x_b)] on replicas equilibrated 300 sweeps at their own
(T,lam). dlog ~ -O(10): ladder coarse FOR OUR SYSTEM (insert rungs). dlog ~ -O(1000): implementation scale
bug. Uses the bcy Cavity class verbatim. cav 0, R=2.0."""
import sys, importlib.util, torch, statistics as st
spec = importlib.util.spec_from_file_location("bcy", "reports/logs-2026-07-14/bcy_shrinkage_pt.py")
# import module pieces without running main loop: read constants/classes by exec of the file top
src = open("reports/logs-2026-07-14/bcy_shrinkage_pt.py").read()
main_start = src.index("results = {}")
header = src[:main_start]
g_ns = {}
sys.argv = ["x", "2.0", "100"]
exec(header, g_ns)
torch = g_ns["torch"]; dev = g_ns["dev"]
carve, _mic = g_ns["carve"], g_ns["_mic"]
Cavity = g_ns["Cavity"]; LAMs, Ts, BETAs = g_ns["LAMs"], g_ns["Ts"], g_ns["BETAs"]
NR, NCH, R, L = g_ns["NR"], g_ns["NCH"], g_ns["R"], None
X, S, Lf = g_ns["X"], g_ns["S"], g_ns["L"]
STEP_MAX = g_ns["STEP_MAX"]

gen = torch.Generator(device=dev).manual_seed(0)
c = torch.rand(3, generator=gen, device=dev) * Lf
p = carve(X[0], S[0], c, R, Lf)
xin = _mic(p["x_in"], c, Lf); sin = p["s_in"]
xout = _mic(p["x_out"], c, Lf); bm = xout.norm(dim=-1) < (R + 2.5); bnd, sb = xout[bm], p["s_out"][bm]
n = xin.shape[0]
cav = Cavity(xin, sin, bnd, sb, n)
B = 2 * NR * NCH
Xm = xin[None].expand(B, n, 3).clone()
g = torch.Generator(device=dev).manual_seed(7)
U = cav.full_U(Xm)
print(f"equilibrating {B} replicas 300 sweeps at their own (T,lam)...", flush=True)
for sw in range(300):
    for i in torch.randperm(n, generator=g, device=dev).tolist():
        l = STEP_MAX * torch.rand(B, 1, device=dev, generator=g)
        nh = torch.randn(B, 3, device=dev, generator=g); nh = nh / nh.norm(dim=-1, keepdim=True)
        xi_new = Xm[:, i] + l * nh
        ok = xi_new.norm(dim=-1) < R
        dU = cav.row_U(Xm, i, xi_new) - cav.row_U(Xm, i, Xm[:, i])
        acc = ok & (torch.rand(B, device=dev, generator=g).log() < -cav.beta * dU)
        Xm[acc, i] = xi_new[acc]; U = torch.where(acc, U + dU, U)
print(f"{'pair':>6} {'lam_a->lam_b':>16} {'T_a->T_b':>12} | {'dlog (stackA)':>13} {'dlog (stackB)':>13}", flush=True)
Xr = Xm.view(2, NR, NCH, n, 3)
for r in range(NR - 1):
    xa = Xr[:, r].reshape(-1, n, 3); xb = Xr[:, r + 1].reshape(-1, n, 3)
    la_v = torch.full((xa.shape[0],), float(LAMs[r]), device=dev)
    lb_v = torch.full((xa.shape[0],), float(LAMs[r + 1]), device=dev)
    Uaa = cav.full_U(xa, la_v); Uab = cav.full_U(xb, la_v)
    Uba = cav.full_U(xa, lb_v); Ubb = cav.full_U(xb, lb_v)
    dlog = -BETAs[r] * (Uab - Uaa) - BETAs[r + 1] * (Uba - Ubb)
    print(f"{r:>6} {float(LAMs[r]):.4f}->{float(LAMs[r+1]):.4f} {float(Ts[r]):.3f}->{float(Ts[r+1]):.3f} | "
          f"{float(dlog[0]):>13.1f} {float(dlog[1]):>13.1f}", flush=True)
print("\ndlog ~ -10..-30: coarse ladder for our system (insert rungs). dlog ~ -100s+: scale bug.", flush=True)
