# AR transformer vs eRSI on N=44 IPL — Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run our +A curve-flow AR generator on the paper's N=44 IPL system and compare against published eRSI on the same data/energy/observables, reporting discard fraction + ESS-vs-R (IS-free + IS-cost) as co-primary with reweighted observables.

**Architecture:** Reuse Grenioux et al.'s released energy/g(r) code (vendored, unmodified) and equilibrium dataset; train our `KACurveFlowModel` (the +A curve-flow head) adapted to IPL geometry (ρ=0.5, N=44, 50:50); compute self-normalized IS (`w = −βU − logq`) with our exact `logq`.

**Tech Stack:** Python, PyTorch, pytest. Their repo `github.com/h2o64/learndiffeq` (energy `SoftSphere.U`, `radial_distribution_function`); Zenodo 17966995 (`ipl44_T0.1` data); our `liquid_coupling_flow/ka_curveflow.py` (`KACurveFlowModel`).

## Global Constraints

- **System (match exactly):** N=44, 2D, **50:50 binary (22/22)**, IPL `r^-12` (BHHP soft-sphere), σ=[[1.0,1.2],[1.2,1.4]], ε=1, rcut=2.5σ shifted, **ρ=0.5 ⇒ L=√88≈9.3808**, **T=0.1 ⇒ β=10**.
- **Reuse, do not reimplement:** their energy `SoftSphere.U(a,x)`, their `radial_distribution_function(x,L)`, their equilibrium `ipl44_T0.1` data. The energy adapter must reproduce their energy on reference configs (Task-0 gate).
- **Exact-likelihood IS:** the IS weight is `w = −βU(species,positions) − logq(positions,species)` using the **full joint** `logq` (our `sample(return_logq=True)`/`log_prob`, which includes the arc-scale Jacobian). Species: fixed composition 22/22, same labels in `logq` and `U`.
- **Honest framing:** report discard fraction AND ESS R̄ as co-primary; report observables both raw and reweighted.
- **Scoped commits:** `git add` only the exact new files per task (never `git add -A`; the repo has many unrelated dirty files + parallel work streams). Commit messages end with `Co-Authored-By: Claude <noreply@anthropic.com>`.
- All new code under `liquid_coupling_flow/ipl44/`; tests under `liquid_coupling_flow/tests/`. Their cloned repo + the dataset are **gitignored** (don't commit large/third-party files).

---

### Task 0: Acquire their code + data + energy/g(r) adapters (energy-match gate)

**Files:**
- Create: `liquid_coupling_flow/ipl44/__init__.py` (empty), `liquid_coupling_flow/ipl44/ipl_energy.py`, `liquid_coupling_flow/ipl44/.gitignore`
- Test: `liquid_coupling_flow/tests/test_ipl_energy.py`
- Clone (gitignored): `liquid_coupling_flow/ipl44/learndiffeq/` ; Download (gitignored): `liquid_coupling_flow/ipl44/data/ipl44_T0.1_{positions,species}.pt`

**Interfaces:**
- Produces: `ipl_box() -> (N, L)` (=(44, sqrt(88))); `ipl_energy(positions, species) -> Tensor[B]` (their total U); `ipl_gr(positions, species, L, bins) -> (r, g)`; `load_ipl_reference() -> (positions[M,44,2], species[M,44])`.

- [ ] **Step 1: Clone their repo + download the dataset (gitignored)**

```bash
cd /mnt/ssd/GridTransformer/liquid_coupling_flow/ipl44
printf 'learndiffeq/\ndata/\n__pycache__/\n' > .gitignore
git clone --depth 1 https://github.com/h2o64/learndiffeq learndiffeq
mkdir -p data
curl -L --max-time 300 -o data/ipl44_T0.1_positions.pt "https://zenodo.org/api/records/17966995/files/ipl44_T0.1_positions.pt/content"
curl -L --max-time 300 -o data/ipl44_T0.1_species.pt   "https://zenodo.org/api/records/17966995/files/ipl44_T0.1_species.pt/content"
python -c "import torch; p=torch.load('data/ipl44_T0.1_positions.pt',weights_only=False); s=torch.load('data/ipl44_T0.1_species.pt',weights_only=False); print('positions',tuple(p.shape),p.dtype,'| species',tuple(s.shape),s.dtype,'| species counts',[int((s[0]==k).sum()) for k in (0,1)])"
```
Expected: positions `[M, 44, 2]`, species `[M, 44]` with 22/0 and 22/1 (record M, the dataset size). If clone/download fails, STOP and report — do not regenerate data.

- [ ] **Step 2: Write the failing test**

```python
# liquid_coupling_flow/tests/test_ipl_energy.py
import math, torch
from liquid_coupling_flow.ipl44.ipl_energy import ipl_box, ipl_energy, load_ipl_reference


def test_box_matches_density():
    N, L = ipl_box()
    assert N == 44 and abs(L - math.sqrt(88)) < 1e-6      # rho=0.5 -> L=sqrt(N/rho)=sqrt(88)


def test_energy_finite_and_equilibrium_on_reference():
    pos, sp = load_ipl_reference()
    U = ipl_energy(pos[:256], sp[:256])
    assert U.shape == (256,)
    assert torch.isfinite(U).all()                        # no NaN/inf from overlaps in equilibrium data
    # equilibrium configs have a tight energy band (low relative spread)
    assert float(U.std() / U.mean().abs()) < 0.2, float(U.std()/U.mean().abs())
    print("reference mean U =", float(U.mean()), " std =", float(U.std()))


if __name__ == "__main__":
    test_box_matches_density(); test_energy_finite_and_equilibrium_on_reference(); print("IPL ENERGY TESTS PASSED")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest liquid_coupling_flow/tests/test_ipl_energy.py -q`
Expected: FAIL with `ModuleNotFoundError: ... ipl44.ipl_energy`.

- [ ] **Step 4: Write the energy/g(r) adapter (reusing their code)**

```python
# liquid_coupling_flow/ipl44/ipl_energy.py
"""Thin adapters reusing Grenioux et al.'s released IPL energy + g(r) (vendored in ./learndiffeq, unmodified).
Do NOT reimplement the energy. N=44 IPL: 2D, 50:50, r^-12 BHHP, sigma=[[1,1.2],[1.2,1.4]], eps=1, rcut=2.5sigma,
rho=0.5 -> L=sqrt(88), T=0.1."""
from __future__ import annotations
import os, sys, math, torch

_HERE = os.path.dirname(__file__)
_REPO = os.path.join(_HERE, "learndiffeq")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)                              # import their package
N_IPL = 44
L_IPL = math.sqrt(N_IPL / 0.5)                             # rho = N/L^2 = 0.5
BETA_IPL = 10.0                                            # T = 0.1


def ipl_box():
    return N_IPL, L_IPL


def _energy_module():
    """Their SoftSphere instance (default sigma/eps/rcut match the IPL spec)."""
    from learndiffeq.particles.distributions.soft_spheres import SoftSphere
    return SoftSphere(n_particles=N_IPL, dim_phys=2, L=L_IPL)


_SS = None
def ipl_energy(positions, species):
    """Total IPL potential U (their SoftSphere.U). positions [B,44,2], species [B,44] int. -> U [B]."""
    global _SS
    if _SS is None:
        _SS = _energy_module()
    dev = positions.device
    return _SS.to(dev).U(species.long().to(dev), positions)


def ipl_gr(positions, species, L=None, bins=100):
    """Their radial distribution function. Returns (r_centers, g)."""
    from learndiffeq.particles.callbacks.utils import radial_distribution_function
    g = radial_distribution_function(positions, L or L_IPL)
    return g


def load_ipl_reference(device="cpu"):
    pos = torch.load(os.path.join(_HERE, "data", "ipl44_T0.1_positions.pt"), weights_only=False).to(device).float()
    sp = torch.load(os.path.join(_HERE, "data", "ipl44_T0.1_species.pt"), weights_only=False).to(device).long()
    return pos, sp
```

Note for the implementer: their package `__init__` may import heavy deps (e.g. pytorch-lightning). If `import learndiffeq...` fails on a missing dep, EITHER `pip install -r liquid_coupling_flow/ipl44/learndiffeq/requirements.txt`, OR (preferred, lighter) load the two pure-torch modules directly with `importlib.util.spec_from_file_location` from `learndiffeq/learndiffeq/particles/distributions/soft_spheres.py` (+ its `kob_andersen.py` and the `utils` it needs: `gram_torus`, `make_mask`) and `.../particles/callbacks/utils.py` (`radial_distribution_function`), avoiding the package `__init__`. Do NOT edit their files. If `radial_distribution_function`'s signature differs from `(x, L)`, adapt the call (read the function), do not reimplement g(r).

- [ ] **Step 5: Run test to verify it passes (the energy-match gate)**

Run: `python -m pytest liquid_coupling_flow/tests/test_ipl_energy.py -q -s`
Expected: PASS (2 passed); prints `reference mean U = <finite>`. **This is the gate:** equilibrium reference energy is finite with low spread. If it's NaN/huge or high-variance, the energy adapter (cutoff/shift/PBC) is wrong — fix before anything else.

- [ ] **Step 6: Commit (adapter + test only; NOT the gitignored repo/data)**

```bash
git add liquid_coupling_flow/ipl44/__init__.py liquid_coupling_flow/ipl44/ipl_energy.py liquid_coupling_flow/ipl44/.gitignore liquid_coupling_flow/tests/test_ipl_energy.py
git commit -m "feat(ipl): reuse Grenioux IPL energy + g(r) adapters + dataset loader (energy-match gate)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1: IPL geometry adaptation + support-coverage GO/NO-GO

**Files:**
- Create: `liquid_coupling_flow/ipl44/ipl_model.py`
- Test: `liquid_coupling_flow/tests/test_ipl_model.py`

**Interfaces:**
- Consumes: `KACurveFlowModel` (`ka_curveflow`), `ipl_box`/`load_ipl_reference` (Task 0), `_wrap_pm` (`ka_gridformer`).
- Produces: `make_ipl_model(num_bins=8, tail_bound=4.0, knn=16, device) -> KACurveFlowModel` (rho=0.5); `support_coverage(model, device) -> dict` (out-of-range fractions vs arc_range and tail_bound).

- [ ] **Step 1: Write the failing tests**

```python
# liquid_coupling_flow/tests/test_ipl_model.py
import math, torch
from liquid_coupling_flow.ipl44.ipl_model import make_ipl_model, support_coverage


def test_model_box_is_ipl():
    m = make_ipl_model(device="cpu")
    assert abs(m._Lof(44) - math.sqrt(88)) < 1e-5          # rho=0.5 box
    assert abs(m._arc_scale(44) - 44 ** (1 / 6)) < 1e-5


def test_support_coverage_passes_on_ipl_reference():
    m = make_ipl_model(device="cpu")
    cov = support_coverage(m, device="cpu")
    print("IPL support coverage:", cov)
    # the offset must fit inside BOTH the arc_range bins and the flow tail_bound (else the target is truncated)
    assert cov["oor_arc_range"] < 1e-3, cov
    assert cov["oor_tail_bound"] < 1e-3, cov


if __name__ == "__main__":
    test_model_box_is_ipl(); test_support_coverage_passes_on_ipl_reference(); print("IPL MODEL TESTS PASSED")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest liquid_coupling_flow/tests/test_ipl_model.py -q`
Expected: FAIL with `ModuleNotFoundError: ... ipl44.ipl_model`.

- [ ] **Step 3: Write the IPL model factory + support-coverage check**

```python
# liquid_coupling_flow/ipl44/ipl_model.py
"""IPL adaptation of the +A curve-flow generator (KACurveFlowModel). Only the geometry changes: rho=0.5
(box L=sqrt(N/0.5)), N=44, 50:50 species. Re-derives nothing about the head; re-validates that the IPL cage
fits the (a,b) offset support (arc_range + flow tail_bound) before training."""
from __future__ import annotations
import torch
from liquid_coupling_flow.ka_curveflow import KACurveFlowModel
from liquid_coupling_flow.ka_localframe import _wrap_pm
from liquid_coupling_flow.ipl44.ipl_energy import load_ipl_reference, ipl_box


def make_ipl_model(num_bins=8, tail_bound=4.0, knn=16, arc_range=3.0, device="cpu"):
    # rho=0.5 -> the geo builds L=sqrt(N/rho); arc_range/tail_bound re-checked by support_coverage
    m = KACurveFlowModel(rho=0.5, n_bins=192, knn=knn, arc_range=arc_range,
                         num_bins=num_bins, tail_bound=tail_bound).to(device)
    return m


@torch.no_grad()
def support_coverage(model, B=2048, device="cpu"):
    """Fraction of equilibrium (a,b) offsets that fall outside arc_range (the bin grid) and outside the flow
    tail_bound. Both must be ~0 or the target distribution is truncated."""
    N, L = ipl_box()
    pos, sp = load_ipl_reference(device); pos, sp = pos[:B], sp[:B]
    order = model.geo._curve_order(pos, N)
    xo = torch.gather(pos, 1, order[..., None].expand(-1, -1, 2))
    so = torch.gather(sp, 1, order)
    _, origin = model._local(xo, so, model.geo._scaffold(N, device), L, N)
    ab = _wrap_pm(xo - origin, L) / model._arc_scale(N)
    amax = ab.abs().max(-1).values                                       # per-particle max |offset|
    return {"max_abs_offset": float(amax.max()),
            "oor_arc_range": float((amax > model.arc_range).float().mean()),
            "oor_tail_bound": float((amax > model.flow.spline.tail_bound).float().mean())}
```

- [ ] **Step 4: Run tests to verify they pass (the support-coverage GO/NO-GO)**

Run: `python -m pytest liquid_coupling_flow/tests/test_ipl_model.py -q -s`
Expected: PASS (2 passed); prints the coverage dict. **GO/NO-GO:** if `oor_arc_range` or `oor_tail_bound` > 1e-3, the KA-tuned `arc_range=3.0`/`tail_bound=4.0` truncates the looser ρ=0.5 cage — increase them (pass larger values to `make_ipl_model`) until both are ~0, and record the chosen values. Every downstream number is on a truncated target otherwise. (If `model.flow.spline.tail_bound` is not the attribute name, read `transforms_spline.RQSplineElementwise` for the bound attribute and fix the check — do not skip it.)

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ipl44/ipl_model.py liquid_coupling_flow/tests/test_ipl_model.py
git commit -m "feat(ipl): IPL geometry adaptation of the +A curve-flow model + support-coverage gate

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Train on IPL (matched budget) + exactness gate

**Files:**
- Modify: `liquid_coupling_flow/ipl44/ipl_model.py` (append `train_ipl`, `__main__`)
- Test: `liquid_coupling_flow/tests/test_ipl_train.py`

**Interfaces:**
- Consumes: `make_ipl_model`, `load_ipl_reference`, `ipl_box`; `augment` (`ka_gridformer_train`).
- Produces: `train_ipl(steps, num_bins, tail_bound, arc_range, knn, out, device) -> ckpt_name`; checkpoint `ipl44_curveflow.pt` storing `{state_dict, num_bins, tail_bound, arc_range, knn, n_B, steps, epochs, n_params}`.

- [ ] **Step 1: Write the failing test (exactness gate on the IPL model)**

```python
# liquid_coupling_flow/tests/test_ipl_train.py
import torch
from liquid_coupling_flow.ipl44.ipl_model import make_ipl_model


def test_ipl_exactness_gate():
    # the IPL model must still satisfy sample logq == log_prob (required for the IS weight)
    torch.manual_seed(0)
    m = make_ipl_model(knn=8, device="cpu"); m.eval()       # knn<N=44 fine; small for speed
    pos, sp, logq = m.sample(4, 44, n_B=22, device="cpu", return_logq=True)
    assert torch.allclose(logq, m.log_prob(pos, sp), atol=1e-4)


if __name__ == "__main__":
    test_ipl_exactness_gate(); print("IPL TRAIN TEST PASSED")
```

- [ ] **Step 2: Run test to verify it passes (gate is inherited; confirms IPL wiring)**

Run: `python -m pytest liquid_coupling_flow/tests/test_ipl_train.py -q`
Expected: PASS (1 passed) — the exactness gate holds for the IPL-configured model (inherited from `KACurveFlowModel`). If it fails, the IPL geometry broke the sample/log_prob consistency — stop and report.

- [ ] **Step 3: Append the IPL trainer**

```python
# --- append to liquid_coupling_flow/ipl44/ipl_model.py ---
import os, time, math
ART = os.path.join(os.path.dirname(__file__), "data")


def train_ipl(steps=40000, num_bins=8, tail_bound=4.0, arc_range=3.0, knn=16, lr=3e-4,
              out="ipl44_curveflow.pt", device="cuda" if torch.cuda.is_available() else "cpu"):
    from liquid_coupling_flow.ka_gridformer_train import augment
    N, L = ipl_box()
    data, sp = load_ipl_reference(device)                                # [M,44,2],[M,44]
    nB = int((sp[0] == 1).sum())                                         # 22
    s_vec = sp[0]                                                        # shared composition vector (50:50)
    m = make_ipl_model(num_bins=num_bins, tail_bound=tail_bound, arc_range=arc_range, knn=knn, device=device)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4); B, t0 = 128, time.time(); warmup = 500
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / warmup)
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        loss = (-m.log_prob(augment(data[idx], L), sp[idx]) / N).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"NON-FINITE loss at step {step}")
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if step % 1000 == 0:
            print(f"  step {step:5d} nll/N {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    epochs = steps * B / data.shape[0]; n_params = sum(p.numel() for p in m.parameters())
    torch.save({"state_dict": m.state_dict(), "num_bins": num_bins, "tail_bound": tail_bound,
                "arc_range": arc_range, "knn": knn, "n_B": nB, "steps": steps, "epochs": epochs,
                "n_params": n_params}, os.path.join(ART, out))
    print(f"saved {out}  (epochs={epochs:.0f}, params={n_params/1e3:.0f}k vs eRSI 22k/580k)", flush=True)
    return out


if __name__ == "__main__":
    import sys
    train_ipl(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 40000)
```

- [ ] **Step 4: Smoke-run (200 steps) to verify the loop**

Run: `python -c "from liquid_coupling_flow.ipl44.ipl_model import train_ipl; train_ipl(steps=200)"`
Expected: decreasing `nll/N`, no `FloatingPointError`, saves `data/ipl44_curveflow.pt`, prints epochs + param count. (Controller runs the full matched-budget training in Step 5.)

- [ ] **Step 5: Full matched-budget training (controller experiment)**

Run: `python -m liquid_coupling_flow.ipl44.ipl_model 40000`
Expected: trains to convergence (nll/N decreasing, no NaN), saves the checkpoint with `epochs`/`n_params`. Report epochs (target ~1000 to match eRSI; adjust `steps` so `steps·128/M ≈ 1000`) and param count next to eRSI's 22k/580k — the fairness statement.

- [ ] **Step 6: Commit**

```bash
git add liquid_coupling_flow/ipl44/ipl_model.py liquid_coupling_flow/tests/test_ipl_train.py
git commit -m "feat(ipl): matched-budget IPL training for the +A curve-flow generator

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: The comparison — discard fraction, ESS-vs-R, reweighted observables + figure

**Files:**
- Create: `liquid_coupling_flow/ipl44/ipl_benchmark.py`
- Create: `reports/2026-06-26-ipl44-vs-ersi-results.md`

**Interfaces:**
- Consumes: `make_ipl_model`/`ipl_box`/`load_ipl_reference` (Task 0/1), `ipl_energy`/`ipl_gr` (Task 0), the trained `ipl44_curveflow.pt`.
- Produces: `benchmark(n_samples, device) -> dict` (discard_frac, ess_curve, U_raw/U_rw, cV_raw/cV_rw) + figure + report.

- [ ] **Step 1: Write the benchmark + figure script**

```python
# liquid_coupling_flow/ipl44/ipl_benchmark.py
"""The comparison vs eRSI (Grenioux 2025): IS-free discard fraction + ESS-vs-R (IS cost) co-primary with
reweighted U / c_V / g(r). Self-normalized IS weight w = -beta*U - logq (full joint logq). Honest framing:
report raw AND reweighted; discard fraction + ESS R-bar are the headline IS-free / IS-cost numbers."""
from __future__ import annotations
import os, math, torch, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.ipl44.ipl_model import make_ipl_model, ART
from liquid_coupling_flow.ipl44.ipl_energy import ipl_box, load_ipl_reference, ipl_energy, ipl_gr, BETA_IPL


@torch.no_grad()
def _gen(n, device):
    ck = torch.load(os.path.join(ART, "ipl44_curveflow.pt"), weights_only=False)
    m = make_ipl_model(num_bins=ck["num_bins"], tail_bound=ck["tail_bound"], arc_range=ck["arc_range"],
                       knn=ck["knn"], device=device); m.load_state_dict(ck["state_dict"]); m.eval()
    N, L = ipl_box(); xs, ss, lq = [], [], []
    for _ in range((n + 1023) // 1024):                                  # batch to fit memory
        x, s, q = m.sample(1024, N, n_B=ck["n_B"], device=device, return_logq=True)
        xs.append(x); ss.append(s); lq.append(q)
    return torch.cat(xs)[:n], torch.cat(ss)[:n], torch.cat(lq)[:n], ck


@torch.no_grad()
def benchmark(n_samples=200000, device="cuda" if torch.cuda.is_available() else "cpu"):
    N, L = ipl_box(); beta = BETA_IPL
    ref_pos, ref_sp = load_ipl_reference(device)
    U_ref = ipl_energy(ref_pos, ref_sp); U_ref_max = float(U_ref.max())
    x, s, logq, ck = _gen(n_samples, device)
    U = ipl_energy(x, s)
    # (1) discard fraction: energy > 2x reference max
    discard = float((U > 2 * U_ref_max).float().mean())
    # self-normalized IS weights
    log_w = (-beta * U - logq); log_w = torch.where(torch.isfinite(log_w), log_w, torch.full_like(log_w, -1e30))
    # (2) ESS vs R
    Rs = np.unique(np.geomspace(100, n_samples, 25).astype(int))
    ess = []
    for R in Rs:
        w = torch.softmax(log_w[:R], 0); ess.append(float(1.0 / (w * w).sum()))
    # (3) reweighted observables
    w_all = torch.softmax(log_w, 0)
    U_raw, U_rw = float(U.mean()), float((w_all * U).sum())
    cV_raw = float(beta ** 2 * U.var())
    cV_rw = float(beta ** 2 * ((w_all * U * U).sum() - (w_all * U).sum() ** 2))
    U_ref_mean = float(U_ref.mean())
    # g(r): target / raw / (reweighted via resampling by w)
    res = torch.multinomial(w_all, min(8192, n_samples), replacement=True)
    gr_t = ipl_gr(ref_pos[:8192], ref_sp[:8192], L); gr_raw = ipl_gr(x[:8192], s[:8192], L); gr_rw = ipl_gr(x[res], s[res], L)
    out = {"discard_frac": discard, "ess_R": list(zip(Rs.tolist(), ess)), "U_ref": U_ref_mean,
           "U_raw": U_raw, "U_rw": U_rw, "cV_raw": cV_raw, "cV_rw": cV_rw,
           "n_params": ck["n_params"], "epochs": ck["epochs"]}
    print(f"discard_frac {discard:.3f} (eRSI 0.03, eFM 0.84, RSI 1.0) | ESS@max {ess[-1]:.0f}/{Rs[-1]} | "
          f"U ref {U_ref_mean:.2f} raw {U_raw:.2f} rw {U_rw:.2f} | cV raw {cV_raw:.1f} rw {cV_rw:.1f} | "
          f"params {ck['n_params']/1e3:.0f}k epochs {ck['epochs']:.0f}", flush=True)
    # figure: g(r), p(U) vs q(U), ESS-vs-R
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    def _plot_gr(axi, gr, lab, c, lw):
        g = gr[1] if isinstance(gr, (tuple, list)) else gr; r = gr[0] if isinstance(gr, (tuple, list)) else np.arange(len(g))
        axi.plot(np.asarray(r), np.asarray(g), c, lw=lw, label=lab)
    _plot_gr(ax[0], gr_t, "target", "k", 2.4); _plot_gr(ax[0], gr_raw, "ours raw", "C0", 1.4); _plot_gr(ax[0], gr_rw, "ours reweighted", "C3", 1.6)
    ax[0].set_title("g(r)"); ax[0].set_xlabel("r"); ax[0].legend(fontsize=8)
    ax[1].hist((beta * U_ref).cpu().numpy(), 80, density=True, alpha=0.5, label="p(U) target (βU)")
    ax[1].hist((beta * U).cpu().numpy(), 80, density=True, alpha=0.5, label="q(U) ours (βU)")
    ax[1].set_title("energy histogram"); ax[1].set_xlabel("beta*U"); ax[1].legend(fontsize=8)
    ax[2].loglog(Rs, ess, "C0-o", label="ours"); ax[2].loglog(Rs, Rs, "k:", lw=1, label="ESS=R (ideal)")
    ax[2].set_title("ESS vs R"); ax[2].set_xlabel("R (samples)"); ax[2].set_ylabel("ESS"); ax[2].legend(fontsize=8)
    fig.suptitle(f"AR +A curve-flow vs eRSI, N=44 IPL T=0.1 (discard {discard:.2f}, {ck['n_params']/1e3:.0f}k params)", fontsize=13)
    fig.tight_layout(); p = os.path.join(ART, "ipl44_benchmark.png"); fig.savefig(p, dpi=120); print("saved", p, flush=True)
    return out


if __name__ == "__main__":
    import sys
    benchmark(n_samples=int(sys.argv[1]) if len(sys.argv) > 1 else 200000)
```

- [ ] **Step 2: Run the comparison (controller experiment)**

Run: `python -m liquid_coupling_flow.ipl44.ipl_benchmark 200000`
Expected: prints discard_frac (vs eRSI 0.03), ESS@max, U/c_V raw+reweighted vs reference, params/epochs; saves `data/ipl44_benchmark.png`. If `ipl_gr` returns a single array (not `(r,g)`), the `_plot_gr` guard handles it; if its shape differs, adapt the unpacking (don't reimplement g(r)).

- [ ] **Step 3: Write the results report**

Create `reports/2026-06-26-ipl44-vs-ersi-results.md` with: the system (N=44 IPL, ρ=0.5, T=0.1), the matched budget (epochs + our params vs eRSI 22k/580k), and the comparison table — **discard fraction (ours vs RSI 1.0 / eFM 0.84 / eRSI 0.03)**, **ESS R̄** (ours vs eRSI up to ~1.8e6), and **reweighted U / c_V / g(r)** vs target. State the honest verdict per the spec: if raw discard is worse than eRSI's 3% but reweighted observables match, the result is "raw generator worse (the AR half-cage core-fill, now in this system too), IS compensates at higher sample cost" — quantified by our ESS R̄ vs theirs. Include the figure.

- [ ] **Step 4: Commit**

```bash
git add liquid_coupling_flow/ipl44/ipl_benchmark.py reports/2026-06-26-ipl44-vs-ersi-results.md
git commit -m "feat(ipl): benchmark vs eRSI - discard/ESS/reweighted comparison + figure + report

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** §1 goal → Task 3. §2 system (IPL params, ρ=0.5, L=√88, T=0.1) → Global Constraints + Task 0/1. §3 reuse-their-code (energy/ESS/g(r) + data, energy-match gate) → Task 0. §4 exact-log-q IS weight (full joint logq, species consistent) → Task 2 exactness gate + Task 3 `w=−βU−logq`. §5 model = +A curve-flow re-trained on IPL + ρ=0.5 re-derivation + support-coverage gate → Task 1/2. §5 matched budget + param count → Task 2 Step 5. §6 comparison (discard / ESS-vs-R / reweighted / figure) → Task 3. §7 honest framing (discard+ESS co-primary, raw+reweighted) → Task 3 report. §8 scope (Option 1 only) → respected. ✓

**Placeholder scan:** the implementer-discretion notes (import mechanics in Task 0 Step 4; `tail_bound` attribute name in Task 1 Step 4; `ipl_gr` return shape in Task 3) are concrete fallbacks with named files to read, not "TBD". No "add error handling"/"similar to Task N". ✓

**Type consistency:** `ipl_box()->(N,L)`, `ipl_energy(positions,species)->U[B]`, `load_ipl_reference()->(pos,sp)`, `make_ipl_model(num_bins,tail_bound,arc_range,knn,device)`, checkpoint keys (`num_bins,tail_bound,arc_range,knn,n_B,n_params,epochs`) consistent across Tasks 0→3. The IS weight `w=−βU−logq` uses the full joint `logq` from `sample(return_logq=True)`/`log_prob` (per the species decision). ✓

**Implementer notes:** (1) Their g(r) function may compute total or partial g(r) — read it; report whatever it returns consistently for target/raw/reweighted (same function ⇒ fair). (2) If memory limits sampling 200k at once, the `_gen` batching handles it; reduce `n_samples` if needed and report the actual R. (3) `augment` (D4+translation) must be PBC-consistent at L=√88 — it operates per-config in the box, density-agnostic, so it carries over.
