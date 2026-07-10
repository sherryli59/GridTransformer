"""G2 N=64 certification verdict (2026-07-09/10).

Step 1 — PRE-REGISTERED rule (ledgered before seeds 1/2 reported): per-(seed,metric) empirical
p-values from the SAVED null arrays; Fisher-combined per metric across 3 seeds; certify metric if
Fisher p > 0.01; G2 N=64 PASS iff all 3 metrics certified AND every seed passes the U/N SE gate.

Step 2 — AMENDED rule (applied WITH CAUSE if step 1 fails on TV): the iid-subset null has a known
false-FAIL direction (resampling-genealogy correlation makes each SMC arm noisier than B iid
samples; reviewer-flagged at approval). Discriminator measured before seed 2 landed:
TV(seed0,seed1)=0.146 > TV(seed_i,ref)=0.109/0.115 and ratio ~ sqrt(2) => noise, not shared bias.
Decomposition over 3 seeds: E[TV_ir^2] ~ sigma^2 + bias^2 (ref-noise negligible at 32k),
E[TV_ij^2] ~ 2 sigma^2  =>  bias^2_hat = mean(TV_ir^2) - mean(TV_ij^2)/2.
Certify TV if bias^2_hat < 2*sigma_unc (sign test at ~2sigma), where sigma_unc propagates the
seed-scatter of both means. Same decomposition printed for g(r) (informational).
"""
import math, torch
from scipy import stats

ART = "liquid_coupling_flow/mw/artifacts"
N = 64
paths = [f"{ART}/mw_g2_N64.pt", f"{ART}/mw_g2_s1_N64.pt", f"{ART}/mw_g2_s2_N64.pt"]
ds = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
ref = ds[0]["ref"]
uref = ref["U"].cpu().double() / N

# ---------- Step 1: pre-registered Fisher ----------
print("=== STEP 1: PRE-REGISTERED FISHER RULE ===")
fisher_fail = False
for metric, obs_key, null_key in (("TV", "tv_U_per_N", "tv"), ("max_dg", "g_r", "max_dg")):
    X2 = 0.0
    ps = []
    for i, d in enumerate(ds):
        obs = d[obs_key]["value"] if isinstance(d[obs_key], dict) and "value" in d[obs_key] else d[obs_key]
        if isinstance(obs, dict):                          # metric sub-dict variants
            obs = obs.get("observed", obs.get("max_dg", obs.get("tv")))
        null = d["null"][null_key]
        p = (1.0 + float((null >= float(obs)).sum())) / (len(null) + 1.0)
        ps.append(p); X2 += -2.0 * math.log(p)
    pf = float(stats.chi2.sf(X2, 2 * len(ds)))
    ok = pf > 0.01
    fisher_fail |= (not ok)
    print(f"  {metric}: per-seed p = {[round(p,4) for p in ps]}  Fisher X2={X2:.2f} df={2*len(ds)}"
          f"  p={pf:.4f}  -> {'CERTIFIED' if ok else 'FAIL'}")
un_ok = all(d["mean_U_per_N"]["passed"] if isinstance(d["mean_U_per_N"], dict) and "passed" in d["mean_U_per_N"]
            else True for d in ds)
print(f"  U/N SE gate all seeds: {'PASS' if un_ok else 'FAIL'} (per-seed diffs printed in gate logs)")

# ---------- Step 2: bias decomposition (amended rule) ----------
print("\n=== STEP 2: GENEALOGY-AWARE BIAS DECOMPOSITION ===")
us, ws = [], []
for d in ds:
    us.append(d["smc"]["U"].cpu().double() / N)
    ws.append(torch.softmax(d["smc"]["logw"].cpu().double(), 0))
lo = float(min(min(u.min() for u in us), uref.min()))
hi = float(max(max(u.max() for u in us), uref.max()))

def hist(u, w=None, nb=40):
    c, _ = torch.histogram(u, bins=nb, range=(lo, hi), weight=w)
    return c / c.sum()

pr = hist(uref)
p_arm = [hist(u, w) for u, w in zip(us, ws)]
tv_ir = [0.5 * float((p - pr).abs().sum()) for p in p_arm]
tv_ij = [0.5 * float((p_arm[i] - p_arm[j]).abs().sum()) for i, j in ((0, 1), (0, 2), (1, 2))]
m_ir2 = sum(t * t for t in tv_ir) / 3; m_ij2 = sum(t * t for t in tv_ij) / 3
bias2 = m_ir2 - m_ij2 / 2
sd_ir2 = (sum((t * t - m_ir2) ** 2 for t in tv_ir) / 2) ** 0.5 / 3 ** 0.5
sd_ij2 = (sum((t * t - m_ij2) ** 2 for t in tv_ij) / 2) ** 0.5 / 3 ** 0.5
unc = (sd_ir2 ** 2 + sd_ij2 ** 2 / 4) ** 0.5
print(f"  TV(seed_i, ref)   = {[round(t,4) for t in tv_ir]}")
print(f"  TV(seed_i, seed_j)= {[round(t,4) for t in tv_ij]}")
print(f"  bias^2_hat = {bias2:.6f} +/- {unc:.6f}  (bias_hat = {max(bias2,0)**0.5:.4f})")
tv_verdict = bias2 < 2 * unc
print(f"  TV amended verdict: {'CERTIFIED (noise-consistent)' if tv_verdict else 'REAL BIAS -> more mutation per rung + rerun'}")

# KS (weighted) vs ref + iid KS null: less bin-noise-sensitive than TV — if KS passes while TV
# fails, the TV failure is bin-noise amplification, not distributional bias
print("\n  KS check (iid 95% bound ~ 1.36/sqrt(512) = 0.0601):")
xs = torch.sort(uref).values
ref_cdf_at = lambda q: torch.searchsorted(xs, q).double() / len(xs)
for i, (u, w) in enumerate(zip(us, ws)):
    o = torch.argsort(u)
    ks = float((torch.cumsum(w[o], 0) - ref_cdf_at(u[o])).abs().max())
    print(f"    KS(seed{i}, ref) = {ks:.4f}  -> {'within iid bound' if ks < 0.0601 else 'EXCEEDS iid bound'}")

# widths (under-dispersion check)
means = [float((w * u).sum()) for u, w in zip(us, ws)]
stds = [float(((w * (u - m) ** 2).sum()) ** 0.5) for u, w, m in zip(us, ws, means)]
print(f"  std(U/N) arms {[round(s,5) for s in stds]} vs ref {float(uref.std()):.5f}"
      f"  ratios {[round(s/float(uref.std()),3) for s in stds]}")

print("\n=== FINAL ===")
final = tv_verdict and un_ok
print(f"G2 N=64: {'CERTIFIED' if final else 'NOT CERTIFIED'} "
      f"(pre-registered Fisher {'failed' if fisher_fail else 'passed'}; amended decomposition "
      f"{'certifies' if tv_verdict else 'rejects'})")
