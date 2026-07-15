# mW SMC Kernel Portfolio (suffix + two-blob + conveyor) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a portfolio of exact collective MH kernels (ordering-suffix resample, two-blob redistribution, P↔S conveyor) inside a tempered SMC bridge for the mW liquid, judged by total SW energy evaluations to matched equilibration depth vs plain single-site MC.

**Architecture:** Kernels live in a new `liquid_coupling_flow/mw/mw_kernels.py`; the suffix-conditional sampler is added to the v4-lineage `MWGlobalAR` (which `MWV4ToroidalResidual`/v10 inherits); a new `mw_smc_portfolio.py` runs the λ-bridge with a λ-scheduled kernel mixture, reusing `mw_smc.py`'s ESS/λ/resample machinery. Diagnostics D0–D2 gate design choices before the main ablation campaign.

**Tech Stack:** PyTorch (CUDA), existing mW modules (`mw_energy`, `mw_reference`, `mw_generator_v4/v10`, `mw_block_ar`, `mw_smc`, `mw_base`), `ka3d_smc_bridge.geometric_bridge_log_accept`, pytest.

**Spec:** `docs/superpowers/specs/2026-07-14-mw-smc-kernel-portfolio-design.md`

## Global Constraints

- **Metric currency = SW energy evaluations.** Every chain/bridge runs inside `with count_energy_evals() as counter:` (`mw_energy.py:53`); `counter.as_dict()["cost_units"]` is the ledger (1 unit = one particle-move-equivalent local eval; full `mw_energy` on `[B,N,3]` counts B·N units). Never inflate the ledger with evals on walkers whose proposal is already dead. Model forwards are free.
- **π_λ density convention:** q0(x) = **canonical-lift** density `q0_model.log_prob(x, L)` (`preordered=False`). It is permutation-invariant over unordered configs, so relabeling storage is always legal.
- **Storage invariant:** walker tensors are kept in canonical slot order; call `recanonicalize(x, L)` after any move family that can break it. `suffix_move` requires canonical storage on entry and rejects noncanonical regenerated tails (forward/reverse truncation normalizers share the same prefix and cancel exactly).
- **Blob selection exactness:** centers drawn state-independently; block = union of K-nearest sets; **reverse-check** — the same centers must select the same index block under the proposed config, else reject. The center-set selecting a given block is symmetric in (x, x′), so selection probabilities cancel.
- **CLAUDE.md directives:** long runs launched with `run_in_background: true`, both streams to a log (`> log.out 2>&1`); per-unit incremental saves (per-T\*, per-rung, per-arm); raw logs + one-off scripts committed to `reports/logs-<date>/` the moment a run finishes; every plot's full path printed as a clickable link; judge by full distributions, never single scalars; kill/check runs by exact PID, never `pgrep -f` self-matching patterns.
- **System constants:** N=64, ρ\* = `RHO_STAR` = 0.4564 → L = (N/ρ\*)^{1/3} ≈ 5.194; ambient T\* = `T_STAR` = 0.09632; β = 1/T\*. All from `liquid_coupling_flow/mw/mw_energy.py`.
- Python via the repo's usual interpreter; run pytest as `python -m pytest liquid_coupling_flow/tests/<file> -v` from `/mnt/ssd/GridTransformer`.

---

### Task 1: D0 — supercooled regime probe + banks

Pick T\*_work (hardest T\* where an extended plain-MC run still plateaus → we get a trustworthy reference bank) and T\*_hard (one notch colder; plain MC visibly fails). Watch for crystallization — mW is a fast crystallizer; a step-drop in U/N or a split second g(r) peak disqualifies that T\*.

**Files:**
- Create: `reports/logs-2026-07-14/d0_regime_probe.py`
- Create: `reports/logs-2026-07-14/d0_analyze.py`
- Output artifacts: `liquid_coupling_flow/mw/artifacts/d0_probe_T{tstar}_N64.pt` (per T\*, saved as each finishes), `liquid_coupling_flow/mw/artifacts/mw_bank_supercooled_N64.pt`

**Interfaces:**
- Consumes: `mw_reference.mc_run(N, L, beta, n_equil, n_collect, every, seed, B, track_every, ckpt_path, init_cfgs)` → dict with keys `cfgs [n,N,3] cpu`, `U [n]`, `traj [(sweep, U/N)]`, `acc`, `flat_budget`, `coll_drift`; `mw_reference.g_r(cfgs, L)`.
- Produces: chosen `TSTAR_WORK`, `TSTAR_HARD` (recorded in `reports/logs-2026-07-14/d0_verdict.md`); bank file with `{"cfgs": [n,64,3], "U": [n], "tstar": float, "L": float}`.

- [ ] **Step 1: Write the ladder script**

```python
"""D0: supercooled regime probe for the mW kernel campaign (spec 2026-07-14).
T* ladder, fixed-density N=64. Per-T* result saved THE MOMENT it finishes."""
import os, time, torch
from liquid_coupling_flow.mw.mw_energy import RHO_STAR, count_energy_evals
from liquid_coupling_flow.mw.mw_reference import mc_run, g_r

N = 64
L = (N / RHO_STAR) ** (1.0 / 3.0)
ART = "liquid_coupling_flow/mw/artifacts"
LADDER = [0.0963, 0.085, 0.075, 0.065, 0.055]

for tstar in LADDER:
    beta = 1.0 / tstar
    tag = f"{tstar:.4f}"
    ck = f"{ART}/d0_probe_T{tag}_N64_ckpt.pt"
    t0 = time.time()
    with count_energy_evals() as counter:
        res = mc_run(N, L, beta, n_equil=20_000, n_collect=20_000, every=100,
                     seed=0, B=8, track_every=100, ckpt_path=ck)
    r, g = g_r(res["cfgs"].cuda() if torch.cuda.is_available() else res["cfgs"], L)
    res.update({"tstar": tstar, "L": L, "gr_r": r.cpu(), "gr": g.cpu(),
                "evals": counter.as_dict(), "wall_s": time.time() - t0})
    torch.save(res, f"{ART}/d0_probe_T{tag}_N64.pt")
    print(f"T*={tag}: U/N final={float(res['U'].mean()/N):+.4f} acc={res['acc']:.3f} "
          f"coll_drift={res['coll_drift']:.4f} wall={res['wall_s']:.0f}s", flush=True)
print("D0 ladder DONE", flush=True)
```

- [ ] **Step 2: Launch in background with full logging**

Run (Bash, `run_in_background: true`):
`cd /mnt/ssd/GridTransformer && python reports/logs-2026-07-14/d0_regime_probe.py > reports/logs-2026-07-14/d0_regime_probe.out 2>&1`
Expected: per-T\* lines appearing in the log; per-T\* `.pt` files appearing as each rung finishes. Liveness = log mtime advancing, not pgrep.

- [ ] **Step 3: Write the analysis script (runs after the ladder completes)**

```python
"""D0 analysis: traces, distributions, crystallization check, T*_work verdict."""
import glob, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 64
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for f in sorted(glob.glob("liquid_coupling_flow/mw/artifacts/d0_probe_T*_N64.pt")):
    res = torch.load(f, map_location="cpu", weights_only=False)
    sw = [s for s, _ in res["traj"]]; uv = [u for _, u in res["traj"]]
    ax[0].plot(sw, uv, label=f"T*={res['tstar']:.4f}")
    ax[1].plot(res["gr_r"], res["gr"], label=f"T*={res['tstar']:.4f}")
    ax[2].hist((res["U"] / N).numpy(), bins=40, alpha=0.5, label=f"T*={res['tstar']:.4f}")
    # crystallization flag: step-drop = any 500-sweep window falling > 0.05/particle
    import numpy as np
    u = np.array(uv); w = 5
    drops = u[:-w] - u[w:]
    print(f"T*={res['tstar']:.4f} coll_drift={res['coll_drift']:.4f} "
          f"max_window_drop={drops.max() if len(drops) else 0:+.4f} "
          f"U/N final={float(res['U'].mean()/N):+.4f}")
ax[0].set(xlabel="sweep", ylabel="U/N", title="D0 traces"); ax[0].legend()
ax[1].set(xlabel="r", ylabel="g(r)", title="g(r) per T*"); ax[1].legend()
ax[2].set(xlabel="U/N", title="collected U/N distributions"); ax[2].legend()
fig.tight_layout(); out = "reports/logs-2026-07-14/d0_regime_probe.png"
fig.savefig(out, dpi=140); print(f"PLOT: /mnt/ssd/GridTransformer/{out}")
```

- [ ] **Step 4: Run analysis, decide T\*_work / T\*_hard, write verdict**

Run: `python reports/logs-2026-07-14/d0_analyze.py`
Decision rule: T\*_work = coldest T\* with `coll_drift` < 0.005/particle AND no crystallization flag; T\*_hard = one rung colder. Write both values + one-paragraph justification (citing the trace plot path) to `reports/logs-2026-07-14/d0_verdict.md`.

- [ ] **Step 5: Extended bank run at T\*_work**

Edit `d0_regime_probe.py`'s ladder to `LADDER = [TSTAR_WORK]` in a copy `d0_bank_run.py` with `n_equil=60_000, n_collect=40_000, every=200, seed=1`, saving `{"cfgs", "U", "tstar", "L"}` to `liquid_coupling_flow/mw/artifacts/mw_bank_supercooled_N64.pt`. Launch in background with logging as in Step 2. Expected: ~1600 bank configs.

- [ ] **Step 6: Commit scripts, logs, verdict, plots**

```bash
git add reports/logs-2026-07-14/d0_* liquid_coupling_flow/mw/artifacts/d0_probe_T*_N64.pt
git commit -m "measure(mw): D0 supercooled regime probe -- T*_work/T*_hard picked for kernel campaign"
```

---

### Task 2: D1 — block acceptance forecast (no chains)

Bound plain-MH block acceptance from the βΔU distribution of proposals; find K\*. Runs first on the ambient bank (immediate signal), rerun on the D0 bank when available.

**Files:**
- Create: `reports/logs-2026-07-14/d1_block_acc_forecast.py`
- Output: `reports/logs-2026-07-14/d1_forecast_{bank}.pt` + `.png`

**Interfaces:**
- Consumes: `mw_block_train.load_model(path, device)` → `(MWPeriodicBlockAR, ck)`; checkpoint `liquid_coupling_flow/mw/artifacts/mw_block_runs/mw_block_N64_mle.pt`; `MWPeriodicBlockAR.sample_block(x, block_idx, L, gen)` → `(x_new, log_q)`; `.block_log_prob(x, block_idx, L)`; `mw_block_ar.random_block(n, k, device, gen)`; `mw_energy(x, L)`.
- Produces: verdict `K*` at thresholds acc ≥ 0.1 / ≥ 0.01 per bank, recorded in `reports/logs-2026-07-14/d1_verdict.md`. Gates Task 7 (RL fine-tune) and the heat-bath row of the portfolio.

- [ ] **Step 1: Write the forecast script**

```python
"""D1: plain-MH block acceptance forecast. acc <= E[min(1, e^{-beta dU} * q_rev/q_fwd)]
computed exactly per proposal -- no chains. Distributions saved, not just means."""
import sys, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.mw.mw_energy import mw_energy, RHO_STAR
from liquid_coupling_flow.mw.mw_block_train import load_model
from liquid_coupling_flow.mw.mw_block_ar import random_block, wrap_pm

BANK = sys.argv[1]            # e.g. liquid_coupling_flow/mw/artifacts/mw_bank_supercooled_N64.pt
TSTAR = float(sys.argv[2])    # temperature to forecast at
DEV = "cuda"; N = 64; L = (N / RHO_STAR) ** (1.0 / 3.0); BETA = 1.0 / TSTAR
KS = [1, 2, 3, 4, 6, 8, 12, 16]; NTRIAL = 256

model, _ = load_model("liquid_coupling_flow/mw/artifacts/mw_block_runs/mw_block_N64_mle.pt", DEV)
model.eval()
bank = torch.load(BANK, map_location=DEV, weights_only=False)
cfgs = bank["cfgs"].to(DEV).float()
gen = torch.Generator(device=DEV).manual_seed(0)
cpu_gen = torch.Generator().manual_seed(0)

def knearest_union(x1, c, K):
    return wrap_pm(x1 - c[None], L).norm(dim=-1).topk(K, largest=False).indices

out = {}
with torch.no_grad():
    for K in KS:
        rows = []
        for tr in range(NTRIAL):
            x = cfgs[int(torch.randint(len(cfgs), (1,), generator=cpu_gen))][None]  # [1,N,3]
            for sel in ("random", "blob"):
                if sel == "random":
                    idx = random_block(N, K, DEV, gen=gen)
                else:
                    c = torch.rand(3, device=DEV, generator=gen) * L
                    idx = knearest_union(x[0], c, K)
                lq_rev = model.block_log_prob(x, idx, L)
                xp, lq_fwd = model.sample_block(x, idx, L, gen=gen)
                dU = (mw_energy(xp, L) - mw_energy(x, L))
                la = (-BETA * dU + lq_rev - lq_fwd).clamp(max=0.0)
                rows.append((sel, K, float(dU), float(la.exp())))
        out[K] = rows
        both = [r for r in rows]
        for sel in ("random", "blob"):
            accs = [a for s, _, _, a in both if s == sel]
            dus = [d for s, _, d, _ in both if s == sel]
            print(f"K={K:3d} sel={sel:6s} mean_acc={sum(accs)/len(accs):.3e} "
                  f"median beta*dU={BETA*sorted(dus)[len(dus)//2]:+.1f}")

torch.save({"rows": out, "tstar": TSTAR, "bank": BANK},
           f"reports/logs-2026-07-14/d1_forecast_T{TSTAR:.4f}.pt")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for K in KS:
    dus = [BETA * d for s, _, d, _ in out[K] if s == "blob"]
    ax[0].hist(dus, bins=50, histtype="step", label=f"K={K}")
    ax[1].plot([K], [sum(a for s, _, _, a in out[K] if s == "blob") /
                     max(1, sum(1 for s, _, _, _ in out[K] if s == "blob"))], "o")
ax[0].set(xlabel="beta*dU", title=f"blob proposals, T*={TSTAR}"); ax[0].legend()
ax[1].set(xlabel="K", ylabel="mean acceptance bound", yscale="log")
fig.tight_layout(); p = f"reports/logs-2026-07-14/d1_forecast_T{TSTAR:.4f}.png"
fig.savefig(p, dpi=140); print(f"PLOT: /mnt/ssd/GridTransformer/{p}")
```

- [ ] **Step 2: Run on the ambient bank now**

Ambient bank source: inspect `torch.load("liquid_coupling_flow/mw/artifacts/mw_bedrock.pt", map_location="cpu", weights_only=False).keys()` — if it holds a `cfgs`-like `[n,64,3]` tensor, pass that file; otherwise build one with `mc_run(N, L, 1/0.09632, n_equil=5_000, n_collect=10_000, every=100, seed=3, B=8)` and save `{"cfgs", "U", "tstar", "L"}` to `liquid_coupling_flow/mw/artifacts/mw_bank_ambient_N64.pt` (guard with `if not os.path.exists(...)` so reruns are free).
Run: `python reports/logs-2026-07-14/d1_block_acc_forecast.py <ambient bank> 0.09632 > reports/logs-2026-07-14/d1_ambient.out 2>&1`
Expected: per-K acceptance table; plot path printed.

- [ ] **Step 3: Re-run at T\*_work on the D0 bank (after Task 1 Step 5 finishes)**

Run: `python reports/logs-2026-07-14/d1_block_acc_forecast.py liquid_coupling_flow/mw/artifacts/mw_bank_supercooled_N64.pt <TSTAR_WORK> > reports/logs-2026-07-14/d1_supercooled.out 2>&1`

- [ ] **Step 4: Write verdict + commit**

`reports/logs-2026-07-14/d1_verdict.md`: K\* at acc ≥ 0.1 and ≥ 0.01 per bank/temperature; explicit gate outcomes: (a) heat-bath row lives? (K\*≥1 at 0.1) (b) Task 7 RL fine-tune needed? (acc at K=6 < 1e-3).

```bash
git add reports/logs-2026-07-14/d1_* && git commit -m "measure(mw): D1 block acceptance forecast -- K* at T* ambient/supercooled"
```

---

### Task 3: D2 — v10 shift-spread (re-rooted suffix gate)

**Files:**
- Create: `reports/logs-2026-07-14/d2_v10_shift_spread.py`
- Output: `reports/logs-2026-07-14/d2_shift_spread.pt` + `.png`, verdict in `reports/logs-2026-07-14/d2_verdict.md`

**Interfaces:**
- Consumes: `mw_generator_v10.load_generator_v10(path, device)`; checkpoint `liquid_coupling_flow/mw/artifacts/mw_gen_N64_v10_best_nll.pt`; `model.log_prob(x, L)` (canonical lift).
- Produces: GO/NO-GO for the re-rooted-suffix kernel variant (consumed by Task 5's optional extension; the base plan does NOT build re-rooting — this diagnostic just settles whether a follow-up is worth it).

- [ ] **Step 1: Write the script**

```python
"""D2: how shift-invariant is v10's canonical-lift log q0? Spread of log_prob under random
torus translations per config; the re-rooted-suffix MH ratio picks up exactly this spread."""
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.mw.mw_energy import RHO_STAR
from liquid_coupling_flow.mw.mw_generator_v10 import load_generator_v10

DEV = "cuda"; N = 64; L = (N / RHO_STAR) ** (1.0 / 3.0); M = 32; NCFG = 128
model = load_generator_v10("liquid_coupling_flow/mw/artifacts/mw_gen_N64_v10_best_nll.pt", DEV)
model = model[0] if isinstance(model, tuple) else model
model.eval()
bank = torch.load("liquid_coupling_flow/mw/artifacts/mw_bank_supercooled_N64.pt",
                  map_location=DEV, weights_only=False)
x = bank["cfgs"][:NCFG].to(DEV).float()
gen = torch.Generator(device=DEV).manual_seed(0)
lps = []
with torch.no_grad():
    for m in range(M):
        v = torch.rand(3, device=DEV, generator=gen) * L if m else torch.zeros(3, device=DEV)
        lps.append(model.log_prob(torch.remainder(x + v, L), L).cpu())
lp = torch.stack(lps)                       # [M, NCFG]
spread_std = lp.std(0); spread_rng = lp.max(0).values - lp.min(0).values
print(f"per-config log q0 spread: std median={spread_std.median():.3f} p90={spread_std.quantile(0.9):.3f}"
      f"  range median={spread_rng.median():.3f} p90={spread_rng.quantile(0.9):.3f} (nats)")
torch.save({"lp": lp, "spread_std": spread_std, "spread_rng": spread_rng},
           "reports/logs-2026-07-14/d2_shift_spread.pt")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(spread_std.numpy(), bins=40); ax[0].set(xlabel="std over shifts (nats)", title="log q0 shift spread")
ax[1].hist(spread_rng.numpy(), bins=40); ax[1].set(xlabel="range over shifts (nats)")
fig.tight_layout(); p = "reports/logs-2026-07-14/d2_shift_spread.png"
fig.savefig(p, dpi=140); print(f"PLOT: /mnt/ssd/GridTransformer/{p}")
```

- [ ] **Step 2: Run (needs the D0 bank; fall back to ambient bank to unblock)**

Run: `python reports/logs-2026-07-14/d2_v10_shift_spread.py > reports/logs-2026-07-14/d2.out 2>&1`
Expected: median/p90 spread printed; plot path printed. Fix the `load_generator_v10` return-shape guard to whatever the function actually returns (read `mw_generator_v10.py:259` when running).

- [ ] **Step 3: Verdict + commit**

GO for a future re-rooted-suffix variant iff median range ≲ 2 nats (comparable to per-move MH slack); record numbers either way in `d2_verdict.md`.

```bash
git add reports/logs-2026-07-14/d2_* && git commit -m "measure(mw): D2 v10 shift-spread -- re-rooted-suffix gate"
```

---

### Task 4: Suffix-conditional sampler on `MWGlobalAR` (v4 lineage)

**Files:**
- Modify: `liquid_coupling_flow/mw/mw_generator_v4.py` (add two methods to `MWGlobalAR`, after `sample`, ~line 206)
- Test: `liquid_coupling_flow/tests/test_mw_suffix.py`

**Interfaces:**
- Consumes: existing `MWGlobalAR._hidden(u, x, anchors, L)`, `self.head.log_prob(h, u, bound) -> [B,N]`, `self.head.sample(h, bound, gen) -> (u [B,3], lp [B])`, `mw_scaffold`, `wrap_pm` (already imported in the file).
- Produces (inherited by `MWV4ToroidalResidual`/v10 unchanged):
  - `MWGlobalAR.suffix_log_prob(x, m, L) -> Tensor[B]` — exact log-density of slots N−m..N−1 given the prefix; `x` is PREORDERED `[B,N,3]`.
  - `MWGlobalAR.sample_suffix(x, m, L, gen=None) -> (x_new [B,N,3], logq_fwd [B])` — redraw the last m slots given slots < N−m; prefix returned bit-identical.

- [ ] **Step 1: Write the failing tests**

```python
import torch
from liquid_coupling_flow.mw.mw_generator_v4 import MWGlobalAR

N, L = 27, 3.9  # 27 = 3^3 -> R=3 scaffold


def tiny_model():
    torch.manual_seed(0)
    return MWGlobalAR(d_model=32, n_layers=1, n_heads=2, n_mix=4, rail_k=4).eval()


def test_suffix_logprob_full_equals_preordered_logprob():
    m = tiny_model()
    with torch.no_grad():
        x, _ = m.sample(2, N, L, gen=torch.Generator().manual_seed(1))
        assert torch.allclose(m.suffix_log_prob(x, N, L),
                              m.log_prob(x, L, preordered=True), atol=1e-4)


def test_sample_suffix_prefix_frozen_and_score_consistent():
    m = tiny_model()
    with torch.no_grad():
        x, _ = m.sample(3, N, L, gen=torch.Generator().manual_seed(2))
        xs, lqf = m.sample_suffix(x, 10, L, gen=torch.Generator().manual_seed(3))
        assert torch.equal(xs[:, :N - 10], x[:, :N - 10])
        assert not torch.allclose(xs[:, N - 10:], x[:, N - 10:])
        assert torch.allclose(lqf, m.suffix_log_prob(xs, 10, L), atol=1e-4)


def test_sample_suffix_full_matches_sample_density():
    m = tiny_model()
    with torch.no_grad():
        x0 = torch.rand(2, N, 3) * L   # arbitrary prefix content is irrelevant at m=N
        xs, lqf = m.sample_suffix(x0, N, L, gen=torch.Generator().manual_seed(4))
        assert torch.allclose(lqf, m.log_prob(xs, L, preordered=True), atol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest liquid_coupling_flow/tests/test_mw_suffix.py -v`
Expected: FAIL with `AttributeError: 'MWGlobalAR' object has no attribute 'suffix_log_prob'`.

- [ ] **Step 3: Implement the two methods**

Add to `MWGlobalAR` in `mw_generator_v4.py` (after `sample`):

```python
    def suffix_log_prob(self, x, m, L):
        """Exact log q0 of slots N-m..N-1 given slots < N-m. x is PREORDERED [B,N,3]
        (slot j == AR step j, anchor t_j) -- the same convention as
        log_prob(..., preordered=True), of which this is the per-slot tail sum."""
        B, N, _ = x.shape
        x = torch.remainder(x, L)
        t, rank, R = mw_scaffold(N, L, x.device)
        s, bound = (L / R) / 2.0, float(R)
        u = wrap_pm(x - t[None], L) / s
        h = self._hidden(u, x, t, L)
        return self.head.log_prob(h, u, bound)[:, N - m:].sum(-1) - 3 * m * math.log(s)

    @torch.no_grad()
    def sample_suffix(self, x, m, L, gen=None):
        """Redraw slots N-m..N-1 from the exact AR conditional given slots < N-m.
        Mirrors sample()'s j-loop with the prefix pre-filled. Returns (x_new, logq_fwd);
        logq_fwd == suffix_log_prob(x_new, m, L) by construction (shared head/_hidden)."""
        device = next(self.parameters()).device
        B, N, _ = x.shape
        t, _, R = mw_scaffold(N, L, device)
        s, bound = (L / R) / 2.0, float(R)
        x = torch.remainder(x.to(device).clone(), L)
        u = wrap_pm(x - t[None], L) / s
        logq = torch.zeros(B, device=device)
        for j in range(N - m, N):
            h = self._hidden(u[:, :j + 1], x[:, :j + 1], t, L)[:, -1]
            uj, lp = self.head.sample(h, bound, gen)
            u[:, j] = uj
            x[:, j] = torch.remainder(t[j] + s * uj, L)
            logq = logq + lp - 3 * math.log(s)
        return x, logq
```

(`_hidden` only reads `u[:, :j]`/`x[:, :j]` for step j — the stale values at slot j are shifted out by `prev`/`key_pos`, identical to `sample()`'s convention.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest liquid_coupling_flow/tests/test_mw_suffix.py -v`
Expected: 3 passed.

- [ ] **Step 5: v10 integration check (GPU, real checkpoint)**

Add and run this test in the same file:

```python
import os, pytest

V10 = "liquid_coupling_flow/mw/artifacts/mw_gen_N64_v10_best_nll.pt"


@pytest.mark.skipif(not (torch.cuda.is_available() and os.path.exists(V10)), reason="needs GPU+ckpt")
def test_suffix_consistency_v10_checkpoint():
    from liquid_coupling_flow.mw.mw_generator_v10 import load_generator_v10
    from liquid_coupling_flow.mw.mw_energy import RHO_STAR
    model = load_generator_v10(V10, "cuda")
    model = model[0] if isinstance(model, tuple) else model
    model.eval()
    N = 64; L = (N / RHO_STAR) ** (1.0 / 3.0)
    with torch.no_grad():
        x, _ = model.sample(2, N, L, gen=torch.Generator(device="cuda").manual_seed(0))
        xs, lqf = model.sample_suffix(x, 24, L, gen=torch.Generator(device="cuda").manual_seed(1))
        assert torch.allclose(lqf, model.suffix_log_prob(xs, 24, L), atol=1e-3, rtol=0)
```

Run: `python -m pytest liquid_coupling_flow/tests/test_mw_suffix.py -v`
Expected: 4 passed. If `load_generator_v10` returns a tuple/dict, adapt the unwrap line (read `mw_generator_v10.py:259-268` — do the same in Task 3's script).

- [ ] **Step 6: Commit**

```bash
git add liquid_coupling_flow/mw/mw_generator_v4.py liquid_coupling_flow/tests/test_mw_suffix.py
git commit -m "feat(mw): exact suffix-conditional sampler/scorer on MWGlobalAR (v10 inherits)"
```

---

### Task 5: `mw_kernels.py` — exact collective kernels + exactness tests

**Files:**
- Create: `liquid_coupling_flow/mw/mw_kernels.py`
- Test: `liquid_coupling_flow/tests/test_mw_kernels.py`

**Interfaces:**
- Consumes: Task 4's `suffix_log_prob`/`sample_suffix`; `MWPeriodicBlockAR.sample_block/.block_log_prob`; `geometric_bridge_log_accept` (`ka3d_smc_bridge.py`); `mw_energy`; `mw_generator.mw_scaffold/canonical_order/wrap_pm`.
- Produces:
  - `recanonicalize(x, L) -> x` and `is_canonical(x, L) -> BoolTensor[B]`
  - `suffix_move(x, U, q0_model, m_lo, m_hi, lam, beta, L, gen) -> (x, U, stats)` — x must be canonical on entry and stays canonical-storage-compatible (accepted proposals passed `is_canonical`).
  - `conveyor_regions(N, L, device, m_frac) -> (cells_prefix [n1,3], cells_suffix [n2,3])`
  - `draw_centers(L, min_sep, gen, device, regions=None) -> (cA [3], cB [3])`
  - `two_blob_move(x, U, lq0, q0_model, block_model, K, lam, beta, L, min_sep, gen, regions=None) -> (x, U, lq0, stats)` — `lq0` = current canonical `q0_model.log_prob(x, L)` `[B]`, maintained through the move; pass `lq0=None` iff `lam == 1.0`.
  - `stats` dicts contain at least `{"acc": float, "reject_reverse": float, "reject_canon": float}`.

- [ ] **Step 1: Write the failing tests**

```python
import math, torch, pytest
from liquid_coupling_flow.mw.mw_generator_v4 import MWGlobalAR
from liquid_coupling_flow.mw.mw_block_ar import MWPeriodicBlockAR
from liquid_coupling_flow.mw.mw_energy import mw_energy
from liquid_coupling_flow.ka3d_smc_bridge import geometric_bridge_log_accept
from liquid_coupling_flow.mw.mw_kernels import (
    recanonicalize, is_canonical, suffix_move, two_blob_move, draw_centers, conveyor_regions)

N, L = 27, 3.9


def q0_tiny():
    torch.manual_seed(0)
    return MWGlobalAR(d_model=32, n_layers=1, n_heads=2, n_mix=4, rail_k=4).eval()


def blk_tiny():
    torch.manual_seed(1)
    return MWPeriodicBlockAR(d_model=32, n_rbf=6).eval()


def test_recanonicalize_makes_canonical_and_preserves_canonical_logq0():
    q0 = q0_tiny()
    x = torch.rand(4, N, 3) * L
    xc = recanonicalize(x, L)
    assert is_canonical(xc, L).all()
    with torch.no_grad():   # canonical lift is permutation-invariant
        assert torch.allclose(q0.log_prob(x, L), q0.log_prob(xc, L), atol=1e-4)


def test_suffix_move_lambda0_accepts_all_canonical():
    q0 = q0_tiny()
    with torch.no_grad():
        x = recanonicalize(q0.sample(8, N, L, gen=torch.Generator().manual_seed(2))[0], L)
        U = mw_energy(x, L)
        x2, U2, st = suffix_move(x, U, q0, 8, 12, lam=0.0, beta=10.0, L=L,
                                 gen=torch.Generator().manual_seed(3))
    assert st["acc"] == pytest.approx(1.0 - st["reject_canon"], abs=1e-9)
    assert torch.allclose(mw_energy(x2, L), U2, atol=1e-3)


def test_suffix_reduction_equals_general_bridge_formula():
    """-lam*(beta*dU + lqf - lqr) must equal the full geometric-bridge ratio with
    canonical-lift q0 whenever current AND proposed states are canonical."""
    q0 = q0_tiny()
    lam, beta = 0.37, 5.0
    with torch.no_grad():
        x = recanonicalize(q0.sample(16, N, L, gen=torch.Generator().manual_seed(4))[0], L)
        m = 9
        lqr = q0.suffix_log_prob(x, m, L)
        xp, lqf = q0.sample_suffix(x, m, L, gen=torch.Generator().manual_seed(5))
        ok = is_canonical(xp, L)
        assert ok.any(), "need at least one canonical proposal for the check"
        U, Up = mw_energy(x, L), mw_energy(xp, L)
        red = -lam * (beta * (Up - U) + (lqf - lqr))
        gen_form = geometric_bridge_log_accept(
            log_q0_current=q0.log_prob(x, L), log_q0_proposed=q0.log_prob(xp, L),
            energy_current=U, energy_proposed=Up,
            log_r_reverse=lqr, log_r_forward=lqf, lam=lam, beta=beta)
        assert torch.allclose(red[ok], gen_form[ok], atol=1e-3)


def test_two_blob_reverse_check_and_state_update():
    q0, blk = q0_tiny(), blk_tiny()
    with torch.no_grad():
        x = recanonicalize(torch.rand(6, N, 3) * L, L)
        U = mw_energy(x, L)
        lq0 = q0.log_prob(x, L)
        x2, U2, lq02, st = two_blob_move(x, U, lq0, q0, blk, K=3, lam=0.2, beta=2.0,
                                         L=L, min_sep=1.5,
                                         gen=torch.Generator().manual_seed(6))
    assert torch.allclose(mw_energy(x2, L), U2, atol=1e-3)
    with torch.no_grad():
        assert torch.allclose(q0.log_prob(x2, L), lq02, atol=1e-4)
    assert 0.0 <= st["acc"] <= 1.0 and 0.0 <= st["reject_reverse"] <= 1.0


def test_conveyor_regions_partition_and_center_draw():
    pre, suf = conveyor_regions(N, L, torch.device("cpu"), m_frac=0.4)
    assert pre.shape[0] + suf.shape[0] == N and suf.shape[0] == round(0.4 * N)
    cA, cB = draw_centers(L, 1.0, torch.Generator().manual_seed(7),
                          torch.device("cpu"), regions=(pre, suf))
    assert cA.shape == (3,) and cB.shape == (3,)


@pytest.mark.slow
def test_two_blob_stationarity_hot():
    """pi_1-invariance smoke at beta=2 (hot, so acceptance is non-trivial): U/N of a
    reference single-site chain must be statistically unchanged after 300 two-blob moves."""
    from liquid_coupling_flow.mw.mw_energy import RHO_STAR
    from liquid_coupling_flow.mw.mw_reference import mc_run
    q0, blk = q0_tiny(), blk_tiny()
    Ls = (N / RHO_STAR) ** (1.0 / 3.0)
    ref = mc_run(N, Ls, 2.0, n_equil=1500, n_collect=1500, every=50, seed=8, B=8)
    x = recanonicalize(ref["cfgs"][-8:].clone(), Ls)
    U = mw_energy(x, Ls)
    gen = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for _ in range(300):
            x, U, _, _ = two_blob_move(x, U, None, q0, blk, K=3, lam=1.0, beta=2.0,
                                       L=Ls, min_sep=1.5, gen=gen)
    u_ref = (ref["U"] / N)
    z = abs(float(U.mean() / N) - float(u_ref.mean())) / max(float(u_ref.std()) / math.sqrt(8), 1e-6)
    assert z < 4.0, f"two-blob chain drifted: z={z:.1f}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest liquid_coupling_flow/tests/test_mw_kernels.py -v -m "not slow"`
Expected: FAIL with `ModuleNotFoundError: liquid_coupling_flow.mw.mw_kernels`.

- [ ] **Step 3: Implement `mw_kernels.py`**

```python
"""Exact collective MH kernels for the mW lambda-bridge (spec 2026-07-14).

Conventions (load-bearing, see spec):
- pi_lambda uses the CANONICAL-LIFT q0: q0_model.log_prob(x, L) with preordered=False,
  permutation-invariant over unordered configs -> storage relabeling is always legal.
- suffix_move requires canonical storage on entry and rejects noncanonical regenerated
  tails; forward/reverse truncation normalizers share the same prefix and cancel, so the
  reduced ratio  -lam*(beta*dU + logq_fwd - logq_rev)  is exact
  (test_suffix_reduction_equals_general_bridge_formula is the guard).
- two_blob_move draws centers state-independently and REJECTS proposals whose union
  block is not re-selected by the same centers under the proposed config: the center-set
  selecting a given index block is symmetric in (x, x'), so selection probs cancel.
"""
from __future__ import annotations
import torch

from liquid_coupling_flow.mw.mw_energy import mw_energy
from liquid_coupling_flow.mw.mw_generator import mw_scaffold, canonical_order, wrap_pm
from liquid_coupling_flow.ka3d_smc_bridge import geometric_bridge_log_accept


def recanonicalize(x, L):
    """Sort each config into canonical slot order (pure relabeling)."""
    B, N, _ = x.shape
    x = torch.remainder(x, L)
    t, rank, R = mw_scaffold(N, L, x.device)
    perm = canonical_order(x, L, R, rank)
    return torch.gather(x, 1, perm[..., None].expand(-1, -1, 3))


def is_canonical(x, L):
    B, N, _ = x.shape
    t, rank, R = mw_scaffold(N, L, torch.remainder(x, L).device)
    perm = canonical_order(torch.remainder(x, L), L, R, rank)
    return (perm == torch.arange(N, device=x.device)[None]).all(1)


def suffix_move(x, U, q0_model, m_lo, m_hi, lam, beta, L, gen):
    """One suffix-resample MH move on every walker. x MUST be canonical storage."""
    B, N, _ = x.shape
    m = int(torch.randint(m_lo, m_hi + 1, (1,), device=x.device, generator=gen).item())
    with torch.no_grad():
        lq_rev = q0_model.suffix_log_prob(x, m, L)
        xp, lq_fwd = q0_model.sample_suffix(x, m, L, gen=gen)
    ok = is_canonical(xp, L)
    Up = U.clone()
    if ok.any():
        Up[ok] = mw_energy(xp[ok], L)          # evals only for live proposals
    la = -lam * (beta * (Up - U) + (lq_fwd - lq_rev))
    r = torch.rand(B, device=x.device, generator=gen).clamp_min(1e-38).log()
    acc = ok & (r < la)
    x = torch.where(acc[:, None, None], xp, x)
    U = torch.where(acc, Up, U)
    return x, U, {"acc": float(acc.float().mean()), "m": m,
                  "reject_canon": float((~ok).float().mean()), "reject_reverse": 0.0}


def conveyor_regions(N, L, device, m_frac=0.4):
    """(prefix cells, suffix cells) split of the curve-ordered anchor list -- fixed and
    state-independent, so conveyor center draws keep MH selection symmetric."""
    t, rank, R = mw_scaffold(N, L, device)
    cut = N - int(round(m_frac * N))
    return t[:cut], t[cut:]


def draw_centers(L, min_sep, gen, device, regions=None, max_tries=64):
    """Two centers, min-image separation >= min_sep. regions=None: both uniform in the box.
    regions=(cells_a, cells_b): random cell center + uniform jitter within the cell
    (conveyor placement) -- both distributions are fixed, so selection stays symmetric."""
    for _ in range(max_tries):
        if regions is None:
            cA = torch.rand(3, device=device, generator=gen) * L
            cB = torch.rand(3, device=device, generator=gen) * L
        else:
            n_cells = regions[0].shape[0] + regions[1].shape[0]
            cw = L / round(n_cells ** (1.0 / 3.0))          # cell width of the scaffold grid
            outs = []
            for cells in regions:
                i = int(torch.randint(cells.shape[0], (1,), device=device, generator=gen).item())
                jit = (torch.rand(3, device=device, generator=gen) - 0.5) * cw
                outs.append(torch.remainder(cells[i] + jit, L))
            cA, cB = outs
        if wrap_pm(cA - cB, L).norm() >= min_sep:
            return cA, cB
    raise RuntimeError(f"draw_centers: no pair with sep>={min_sep} after {max_tries} tries")


def _union_block(x1, cA, cB, K, L):
    """Union of the K nearest to each center; None if the two sets overlap."""
    iA = wrap_pm(x1 - cA[None], L).norm(dim=-1).topk(K, largest=False).indices
    iB = wrap_pm(x1 - cB[None], L).norm(dim=-1).topk(K, largest=False).indices
    idx = torch.cat([iA, iB]).unique().sort().values
    return idx if idx.numel() == 2 * K else None


def two_blob_move(x, U, lq0, q0_model, block_model, K, lam, beta, L, min_sep, gen,
                  regions=None):
    """One two-blob union-regen MH move per walker (walker loop: block model takes one
    shared index block). lq0 = canonical q0_model.log_prob(x, L) [B]; None iff lam==1."""
    if lam < 1.0 and lq0 is None:
        raise ValueError("lq0 required when lam < 1")
    B, N, _ = x.shape
    xp = x.clone()
    lq_f = torch.zeros(B, device=x.device)
    lq_r = torch.zeros(B, device=x.device)
    live = torch.zeros(B, dtype=torch.bool, device=x.device)
    n_rev = 0
    with torch.no_grad():
        for b in range(B):
            cA, cB = draw_centers(L, min_sep, gen, x.device, regions)
            idx = _union_block(x[b], cA, cB, K, L)
            if idx is None:
                continue
            xb, f = block_model.sample_block(x[b:b + 1], idx, L, gen=gen)
            idx2 = _union_block(xb[0], cA, cB, K, L)
            if idx2 is None or not torch.equal(idx2, idx):
                n_rev += 1
                continue
            lq_r[b] = block_model.block_log_prob(x[b:b + 1], idx, L)[0]
            xp[b] = xb[0]
            lq_f[b] = f[0]
            live[b] = True
        Up = U.clone()
        lq0p = lq0.clone() if lq0 is not None else None
        if live.any():
            Up[live] = mw_energy(xp[live], L)
            if lam < 1.0:
                lq0p[live] = q0_model.log_prob(xp[live], L)
    # geometric_bridge_log_accept natively handles lam==1 (q0 term dropped, lq0p may be None)
    la = geometric_bridge_log_accept(
        log_q0_current=lq0 if lq0 is not None else torch.zeros_like(U),
        log_q0_proposed=lq0p, energy_current=U, energy_proposed=Up,
        log_r_reverse=lq_r, log_r_forward=lq_f, lam=lam, beta=beta)
    r = torch.rand(B, device=x.device, generator=gen).clamp_min(1e-38).log()
    acc = live & (r < la)
    x = torch.where(acc[:, None, None], xp, x)
    U = torch.where(acc, Up, U)
    if lq0 is not None:
        lq0 = torch.where(acc, lq0p, lq0)
    return x, U, lq0, {"acc": float(acc.float().mean()),
                       "reject_reverse": n_rev / B, "reject_canon": 0.0,
                       "live": float(live.float().mean())}
```

**Note to implementer:** the dead `w = None` scaffolding inside `draw_centers` above is a writing artifact — implement the regions branch cleanly as: pick a random row of `cells`, jitter by `U(-cw/2, cw/2)` with `cw = L / round(N_cells_total ** (1/3))` where `N_cells_total = regions[0].shape[0] + regions[1].shape[0]`. Nothing else in the function changes.

- [ ] **Step 4: Run fast tests**

Run: `python -m pytest liquid_coupling_flow/tests/test_mw_kernels.py -v -m "not slow"`
Expected: 5 passed. Debug notes: `test_suffix_reduction...` failing at ~1e-1 scale means a `preordered` mix-up (the reduction only holds when both sides use canonical-lift q0 on canonical states); `two_blob` lq0 mismatch means a missed `torch.where` on the accept mask.

- [ ] **Step 5: Run the slow stationarity test**

Run: `python -m pytest liquid_coupling_flow/tests/test_mw_kernels.py -v -m slow`
Expected: 1 passed (z < 4). If it fails, STOP and debug the ratio — per the never-refute-bug-hypothesis directive, a stationarity failure is a bug until proven otherwise (usual suspects: reverse-check asymmetry, missing `lq_r` for dead walkers leaking into `la`, wrap errors at the box boundary).

- [ ] **Step 6: Suffix λ=0 stationarity (statistical, quick script not pytest)**

Add `reports/logs-2026-07-14/kernel_suffix_stationarity.py`: draw 64 samples from a tiny q0, apply 200 `suffix_move`s at `lam=0`, assert the mean canonical `log_prob` stays within 3·SEM of a fresh-sample batch (both estimate E_q0[log q0]); print both numbers. Run it; paste the two numbers into the commit message.

- [ ] **Step 7: Commit**

```bash
git add liquid_coupling_flow/mw/mw_kernels.py liquid_coupling_flow/tests/test_mw_kernels.py reports/logs-2026-07-14/kernel_suffix_stationarity.py
git commit -m "feat(mw): exact suffix/two-blob/conveyor MH kernels + stationarity gates"
```

---

### Task 6: `mw_smc_portfolio.py` — λ-scheduled portfolio bridge

**Files:**
- Create: `liquid_coupling_flow/mw/mw_smc_portfolio.py`
- Test: `liquid_coupling_flow/tests/test_mw_smc_portfolio.py`

**Interfaces:**
- Consumes: `mw_smc.ess/next_lambda/_resample/mutation_sweeps`; `mw_base.GeneratorBase` (verify at `mw_base.py:53` that `log_q` scores with `preordered=False`; if it scores preordered, wrap with a small `CanonicalQ0Base` in this file with `LOGQ_CONST=False`, `log_q(x) = model.log_prob(x, L)` chunked); Task 5 kernels; Task 4 suffix methods; `count_energy_evals`.
- Produces:
  - `smc_run_portfolio(q0_model, block_model, N, L, beta, B=64, ess_target=0.6, n_site_sweeps=2, suffix_cfg=(2, 0.30, 24, 48), twoblob_cfg=(2, 0.70, 6, 1.5), conveyor=False, m_frac=0.4, step=0.15, seed=0, save_tag="port") -> dict`
    - `suffix_cfg = (moves_per_rung, lam_max, m_lo, m_hi)`; `twoblob_cfg = (moves_per_rung, lam_max, K, min_sep)`.
    - Returns `{"x", "U", "logw", "logZ", "history", "evals", "wall_s"}`; `history` rows carry per-rung `lam`, ESS, per-kernel acceptance stats, cumulative `cost_units`.
    - Saves the running dict to `liquid_coupling_flow/mw/artifacts/mw_kportfolio_{save_tag}_N{N}.pt` **after every rung**.

- [ ] **Step 1: Write the failing smoke test**

```python
import os, torch
from liquid_coupling_flow.mw.mw_generator_v4 import MWGlobalAR
from liquid_coupling_flow.mw.mw_block_ar import MWPeriodicBlockAR
from liquid_coupling_flow.mw.mw_smc_portfolio import smc_run_portfolio

N, L = 27, 3.9


def test_portfolio_smoke_completes_and_ledgers():
    torch.manual_seed(0)
    q0 = MWGlobalAR(d_model=32, n_layers=1, n_heads=2, n_mix=4, rail_k=4).eval()
    blk = MWPeriodicBlockAR(d_model=32, n_rbf=6).eval()
    out = smc_run_portfolio(q0, blk, N, L, beta=2.0, B=8, ess_target=0.5,
                            n_site_sweeps=1, suffix_cfg=(1, 0.3, 8, 12),
                            twoblob_cfg=(1, 0.7, 3, 1.2), seed=0, save_tag="smoketest")
    assert out["history"][-1]["lam"] == 1.0
    assert out["evals"]["cost_units"] > 0
    lams = [h["lam"] for h in out["history"]]
    assert all(b >= a for a, b in zip(lams, lams[1:]))
    assert os.path.exists("liquid_coupling_flow/mw/artifacts/mw_kportfolio_smoketest_N27.pt")
    assert torch.isfinite(out["logZ"])
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest liquid_coupling_flow/tests/test_mw_smc_portfolio.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
"""lambda-bridge SMC with a scheduled collective-kernel portfolio (spec 2026-07-14).
pi_lambda proportional to q0_canonical^{1-lam} e^{-lam beta U}. Storage kept canonical.
Init: x ~ q0.sample (slot order), logw0 = logq0_canonical(x) - logq_sample(x) corrects
the (few-%) noncanonical-sample mismatch exactly. Energy ledger via count_energy_evals.
Per-rung persistence: the running result dict is saved after EVERY rung."""
from __future__ import annotations
import os, time, torch

from liquid_coupling_flow.mw.mw_energy import mw_energy, count_energy_evals
from liquid_coupling_flow.mw.mw_smc import ess, next_lambda, _resample, mutation_sweeps, LAM_FLOOR
from liquid_coupling_flow.mw.mw_kernels import (recanonicalize, suffix_move, two_blob_move,
                                                conveyor_regions)

ART = os.path.join(os.path.dirname(__file__), "artifacts")


class CanonicalQ0Base:
    """mutation_sweeps-compatible base scoring the CANONICAL-lift density."""
    LOGQ_CONST = False

    def __init__(self, model, N, L):
        self.model, self.N, self.L = model, N, L

    @torch.no_grad()
    def log_q(self, x, chunk=64):
        outs = [self.model.log_prob(x[i:i + chunk], self.L) for i in range(0, x.shape[0], chunk)]
        return torch.cat(outs, 0)


def smc_run_portfolio(q0_model, block_model, N, L, beta, B=64, ess_target=0.6,
                      n_site_sweeps=2, suffix_cfg=(2, 0.30, 24, 48),
                      twoblob_cfg=(2, 0.70, 6, 1.5), conveyor=False, m_frac=0.4,
                      step=0.15, seed=0, save_tag="port"):
    device = next(q0_model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    base = CanonicalQ0Base(q0_model, N, L)
    path = os.path.join(ART, f"mw_kportfolio_{save_tag}_N{N}.pt")
    regions = conveyor_regions(N, L, device, m_frac) if conveyor else None
    n_sfx, lam_sfx, m_lo, m_hi = suffix_cfg
    n_tb, lam_tb, K_tb, sep_tb = twoblob_cfg

    t0 = time.time()
    with count_energy_evals() as counter:
        with torch.no_grad():
            x, lq_samp = q0_model.sample(B, N, L, gen=gen)
            lq0 = base.log_q(x)
        logw = lq0 - lq_samp                       # exact init reweight (noncanonicality)
        U = mw_energy(x, L)
        lam, logZ, history = 0.0, 0.0, []
        while lam < 1.0:
            phi = -beta * U - lq0
            lam_new = next_lambda(logw, phi, lam, ess_target, B)
            logw = logw + (lam_new - lam) * phi
            lam = lam_new
            cur_ess = ess(logw)
            if cur_ess < ess_target * B:
                w = torch.softmax(logw, 0)
                logZ += float(torch.logsumexp(logw, 0)) - float(torch.log(torch.tensor(float(B))))
                x, U, lq0, logw = _resample(x, U, lq0, logw, gen)
            stats = {"lam": lam, "ess": cur_ess}
            # ---- mutations (all pi_lam-invariant) ----
            if lam < lam_sfx:
                x = recanonicalize(x, L)
                accs = []
                for _ in range(n_sfx):
                    x, U, st = suffix_move(x, U, q0_model, m_lo, m_hi, lam, beta, L, gen)
                    accs.append(st["acc"])
                lq0 = base.log_q(x)                # refresh after suffix batch
                stats["suffix_acc"] = sum(accs) / max(1, len(accs))
            if lam < lam_tb:
                accs = []
                for _ in range(n_tb):
                    x, U, lq0, st = two_blob_move(x, U, lq0 if lam < 1.0 else None,
                                                  q0_model, block_model, K_tb, lam, beta,
                                                  L, sep_tb, gen, regions=regions)
                    accs.append(st["acc"])
                stats["twoblob_acc"] = sum(accs) / max(1, len(accs))
            x, U, lq0, mstats = mutation_sweeps(x, base, lam, beta, L, n_site_sweeps, step, gen)
            x = recanonicalize(x, L)
            stats["site_acc"] = mstats["acc"]
            stats["cost_units"] = counter.as_dict()["cost_units"]
            history.append(stats)
            result = {"x": x.cpu(), "U": U.cpu(), "logw": logw.cpu(), "logZ": logZ,
                      "history": history, "evals": counter.as_dict(),
                      "wall_s": time.time() - t0}
            torch.save(result, path)               # per-rung persistence (CLAUDE.md)
        logZ += float(torch.logsumexp(logw, 0)) - float(torch.log(torch.tensor(float(B))))
        result = {"x": x, "U": U, "logw": logw, "logZ": logZ, "history": history,
                  "evals": counter.as_dict(), "wall_s": time.time() - t0}
        torch.save({**result, "x": x.cpu(), "U": U.cpu(), "logw": logw.cpu()}, path)
    return result
```

**Implementer checks while wiring:** (1) `mw_smc._resample(x, U, lq, logw, gen)` returns 4-tuple — signature at `mw_smc.py:38`; (2) `mutation_sweeps` recomputes `lq` itself for `LOGQ_CONST=False` bases and its per-site `base.log_q` is the dominant model-forward cost — fine, model forwards are free in the metric; (3) recanonicalize after `mutation_sweeps`/two-blob but NOT between two-blob repeats (lq0 stays valid under relabel-free moves); after any `recanonicalize`, `lq0` is unchanged (canonical lift is permutation-invariant) — no refresh needed; (4) logZ bookkeeping mirrors `mw_smc.smc_run` — read its rung loop (`mw_smc.py:69` on) and copy the exact resample-threshold/logZ accounting rather than the sketch above if they differ; (5) verify `mw_energy`/`du_move` actually increment the active `EnergyEvalCounter` (grep `active_energy_counter` in `mw_energy.py`) — if counting happens at call sites instead, add `counter.add_full(B, N)`-style calls next to every energy eval in `mw_kernels.py`/this file so the ledger stays honest.

- [ ] **Step 4: Run the smoke test**

Run: `python -m pytest liquid_coupling_flow/tests/test_mw_smc_portfolio.py -v`
Expected: PASS in a few minutes on GPU (tiny models). CPU fallback acceptable.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/mw/mw_smc_portfolio.py liquid_coupling_flow/tests/test_mw_smc_portfolio.py
git commit -m "feat(mw): lambda-scheduled kernel-portfolio SMC with exact init reweight + eval ledger"
```

---

### Task 7 (GATED by D1): block-model free-run energy fine-tune at T\*_work

**Skip entirely if D1 shows blob acceptance ≥ 1e-2 at K=6 at T\*_work.** Otherwise:

**Files:**
- Create: `reports/logs-2026-07-14/train_mw_block_rl.py` (pattern: `reports/logs-2026-07-13/train_capacity_rl.py`)
- Output: `liquid_coupling_flow/mw/artifacts/mw_block_runs/mw_block_N64_rl.pt`

**Interfaces:**
- Consumes: `mw_block_train.load_model` (warm start from `mw_block_N64_mle.pt`), D0 supercooled bank, `mw_energy`.
- Produces: fine-tuned checkpoint consumed by Task 8 via the same `load_model`.

- [ ] **Step 1: Write the trainer** — pattern `reports/logs-2026-07-13/train_capacity_rl.py` with mW energy swapped in and no species. Core loss per step (blocks: mixed `random_block`/`_union_block`, K ∈ {3,6}, batch 16, `lam_rl=0.1` after 500-step warmup, lr 1e-4, 8k steps):

```python
x = bank_cfgs[torch.randint(len(bank_cfgs), (16,), generator=cpu_gen)].to(DEV)
idx = pick_block(x, gen)                                   # random_block or union blob, K in {3,6}
nll = -model.block_log_prob(x, idx, L).mean()              # TF anchor on data blocks
with torch.no_grad():
    xp, _ = model.sample_block(x, idx, L, gen=gen)         # free-run proposal
    dU = mw_energy(xp, L) - mw_energy(x, L)                # reward evals: outside any counter
    adv = (dU.clamp(-10.0, 50.0) - dU.clamp(-10.0, 50.0).mean())
    adv = adv / adv.std().clamp_min(1e-6)
lq_f = model.block_log_prob(xp, idx, L)                    # exact, differentiable score
loss = nll + lam_rl * (adv * lq_f).mean()
```

Checkpoint best-by-val-acceptance-forecast every 500 steps (re-use the D1 forecast function on a held-out bank slice), `_last.pt` always saved.
- [ ] **Step 2: Launch in background with logging** (`> reports/logs-2026-07-14/train_mw_block_rl.out 2>&1`, `run_in_background: true`).
- [ ] **Step 3: On completion re-run D1** (Task 2 Step 3 command with the new checkpoint path added as an optional arg) — success gate: ≥10× acceptance at K=6 vs MLE checkpoint. Record in `d1_verdict.md`.
- [ ] **Step 4: Commit** trainer + log + verdict update.

---

### Task 8: Main campaign — bridge ablations vs plain-MC brackets

**Files:**
- Create: `reports/logs-<run-date>/mw_kernel_ablation.py` (use the actual date; referenced below as `logs-RUN/`)
- Create: `reports/logs-RUN/mw_kernel_ablation_analyze.py`
- Output: per-arm `.pt` under `liquid_coupling_flow/mw/artifacts/` (via `save_tag`), plots + `reports/<run-date>-mw-kernel-portfolio-results.md`

**Interfaces:**
- Consumes: `smc_run_portfolio` (Task 6), v10 checkpoint as `q0_model` (via `load_generator_v10`), best block checkpoint (MLE or RL per Task 7 gate), `mc_run`, D0 bank + verdict values.
- Produces: gate verdict per spec §5 (GO: some arm reaches a depth plain MC cannot at 3× evals, or matched depth at ≤½ evals, across ≥2 seeds; NO-GO otherwise), memory update.

- [ ] **Step 1: Write the campaign script**

```python
"""Kernel-portfolio ablation at T*_work (and T*_hard stress arm), evals-saved accounting.
Arms x seeds run sequentially; EACH arm's result saved on completion (never end-only)."""
import sys, torch
from liquid_coupling_flow.mw.mw_energy import RHO_STAR, count_energy_evals
from liquid_coupling_flow.mw.mw_reference import mc_run
from liquid_coupling_flow.mw.mw_generator_v10 import load_generator_v10
from liquid_coupling_flow.mw.mw_block_train import load_model
from liquid_coupling_flow.mw.mw_smc_portfolio import smc_run_portfolio

TSTAR = float(sys.argv[1]); BETA = 1.0 / TSTAR
N = 64; L = (N / RHO_STAR) ** (1.0 / 3.0); DEV = "cuda"
q0 = load_generator_v10("liquid_coupling_flow/mw/artifacts/mw_gen_N64_v10_best_nll.pt", DEV)
q0 = (q0[0] if isinstance(q0, tuple) else q0).eval()
blk, _ = load_model("liquid_coupling_flow/mw/artifacts/mw_block_runs/mw_block_N64_mle.pt", DEV)
blk.eval()   # swap to mw_block_N64_rl.pt if Task 7 ran and won

ARMS = {
    "site_only":  dict(suffix_cfg=(0, 0.0, 24, 48), twoblob_cfg=(0, 0.0, 6, 1.5)),
    "suffix":     dict(suffix_cfg=(2, 0.30, 24, 48), twoblob_cfg=(0, 0.0, 6, 1.5)),
    "twoblob":    dict(suffix_cfg=(0, 0.0, 24, 48), twoblob_cfg=(2, 0.70, 6, 1.5)),
    "both":       dict(suffix_cfg=(2, 0.30, 24, 48), twoblob_cfg=(2, 0.70, 6, 1.5)),
    "both_conv":  dict(suffix_cfg=(2, 0.30, 24, 48), twoblob_cfg=(2, 0.70, 6, 1.5),
                       conveyor=True),
}
for seed in (0, 1):
    for arm, kw in ARMS.items():
        tag = f"abl_{arm}_T{TSTAR:.4f}_s{seed}"
        out = smc_run_portfolio(q0, blk, N, L, BETA, B=64, seed=seed, save_tag=tag, **kw)
        print(f"{tag}: logZ={out['logZ']:.1f} U/N={float(out['U'].mean()/N):+.4f} "
              f"evals={out['evals']['cost_units']:.3e} wall={out['wall_s']:.0f}s", flush=True)
    # plain-MC brackets at MATCHED eval budgets (read the max portfolio ledger first)
    for tag, init in (("mc_cold", None),):
        with count_energy_evals() as c:
            res = mc_run(N, L, BETA, n_equil=0, n_collect=120_000, every=200,
                         seed=100 + seed, B=64,
                         ckpt_path=f"liquid_coupling_flow/mw/artifacts/abl_{tag}_T{TSTAR:.4f}_s{seed}_ckpt.pt")
        res["evals"] = c.as_dict()
        torch.save(res, f"liquid_coupling_flow/mw/artifacts/abl_{tag}_T{TSTAR:.4f}_s{seed}.pt")
        print(f"{tag} s{seed}: U/N={float(res['U'].mean()/N):+.4f} evals={res['evals']['cost_units']:.3e}",
              flush=True)
print("ABLATION DONE", flush=True)
```

- [ ] **Step 2: Pre-flight** — single arm, tiny budget (`B=16`, `ess_target=0.5`, one seed) to shake out shapes/OOM (chunk big evals — 10.4 GiB lesson). Run foreground with a 10-minute cap; fix anything; do not skip.
- [ ] **Step 3: Launch full campaign in background** (`python reports/logs-RUN/mw_kernel_ablation.py <TSTAR_WORK> > reports/logs-RUN/ablation_Twork.out 2>&1`, `run_in_background: true`). On completion, optionally repeat at `<TSTAR_HARD>`.
- [ ] **Step 4: Analysis script** — for each arm/seed: U/N-vs-cumulative-evals curves (from per-rung `history` + `traj`), final U/N *distributions* (violin per arm, both seeds), g(r) overlays vs the D0 bank, per-kernel acceptance-vs-λ traces, ESS trace; alien/data bracket framing (mc_cold = alien bracket, D0 bank statistics = data bracket). Every figure saved under `reports/logs-RUN/` with its full path printed.
- [ ] **Step 5: Results report** — `reports/<run-date>-mw-kernel-portfolio-results.md`: table of evals-to-depth per arm; spec §5 gate verdict with the plots linked; explicit per-kernel attribution (suffix vs two-blob vs conveyor deltas); caveats (N=64 box, seeds=2).
- [ ] **Step 6: Commit everything, update memory** — commit scripts/logs/report/artifacts; write/update an auto-memory file `mw-kernel-portfolio.md` (type: project) with the verdict and pointers, add the MEMORY.md line.

---

## Execution notes

- Task order: 1→2→3 are independent of 4→5→6 (two parallel streams); 7 gated by D1; 8 needs 1, 6, and (2 or 7).
- **Delayed-acceptance lever (spec §3): deliberately deferred, not dropped.** It only pays when the ablation ledger shows *rejected* collective moves dominating eval cost. Task 8's analysis must report the rejected-move share of `cost_units` per arm; if it exceeds ~50%, spin up a follow-up plan for two-stage MH (stage 1: 2-body-only ΔU surrogate; stage 2: full SW) — exactness-preserving, drops in at the `mw_kernels.py` acceptance sites.
- **Re-rooted suffix (spec §3): D2 produces the gate verdict only; building it is a follow-up plan if D2 says GO and Task 8 shows the conveyor arm is mixing-limited.
- D0's ladder is the long pole — launch Task 1 Step 2 first, then build Tasks 4–6 while it runs.
- Every failure of an exactness/stationarity test is treated as a real bug (never-refute-bug-hypothesis) — do not loosen tolerances to pass.
