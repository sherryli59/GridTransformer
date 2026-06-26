# Feature A — QueryAR/GPS curve conditioning on the spline-flow head — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the spline-flow placement head explicit curve ("GPS") conditioning — the scaffold position `s_j` + arc-length coordinate — and measure, as an increment over Phase-1 B-alone, whether it sharpens the conditional (position log-density) or the free-run g(r).

**Architecture:** A thin subclass `KACurveFlowModel(KAFlowHeadModel)` augments the flow's conditioning with a per-particle, deterministic **curve-feature** vector (periodic encoding of the scaffold position `s_j` from `geo._scaffold`, plus the arc-length coord `(j+0.5)/N`), added to the local-frame context before the spline flow. Zero-initialized so it starts identical to B-alone; warm-started from the trained B checkpoint to isolate the curve increment. Exactness preserved (the curve feature is part of the conditioning, identical in `log_prob` and `sample`).

**Tech Stack:** Python, PyTorch, pytest. Reuses `liquid_coupling_flow/ka_flowhead.py` (`KAFlowHeadModel`, `SplineFlowHead`), `ka_localframe.py` (`KALocalFrameModel.geo._scaffold`, `GEOM_PERIODS`), `ka_flowhead_eval.py` (g(r)).

## Global Constraints

- **Never edit `ka_localframe.py` or `ka_flowhead.py`** (the verified Phase-1 model; its negative verdict is measured on it). All new code is in a new file `liquid_coupling_flow/ka_curveflow.py` and `liquid_coupling_flow/tests/`. `KACurveFlowModel` overrides `log_prob`/`sample` (accepting the small duplication, deliberately, to keep Phase-1 untouched).
- **Exact likelihood invariant:** `sample(return_logq=True)` returns `log_prob(pos, sp)`; the curve feature is added to the conditioning identically in both paths ⇒ exactness gate must pass.
- **Increment discipline:** the curve feature is **zero at init** (final projection layer zero-initialized) ⇒ at init `KACurveFlowModel` ≡ `KAFlowHeadModel`; warm-start from `ka_flowhead_N100_k8_scratch.pt` so any change is attributable to the curve conditioning.
- **Scoped commits:** `git add` only the exact new files per task (never `git add -A`; shared dirty repo + parallel work streams).
- Target: causal `KALocalFrameModel` family, N=100, T*=0.5. `d_model=192`, `knn=16`, `num_bins=8`, `tail_bound=4.0`, `GEOM_PERIODS` has 5 entries. Tests: pytest, CPU, artifact-free, `knn<N`.
- **Verdict metric (the point of this plan):** flow normalized position log-density (B-alone = 2.704) and free-run g(r) (B-alone ≈ categorical: g_AB peak 2.44, core 0.32). Feature A is judged by whether it beats those.

---

### Task 1: `KACurveFlowModel` — curve-conditioned flow head

**Files:**
- Create: `liquid_coupling_flow/ka_curveflow.py`
- Test: `liquid_coupling_flow/tests/test_curveflow.py`

**Interfaces:**
- Consumes: `KAFlowHeadModel` (`.flow`, `._local`, `._step`, `.geo`, `._arc_scale`, `._Lof`, `.head_species`, `.n_species`, `.canonical`, `.d`, `.d_model`, `.periods`, `.frame_mode`) from `ka_flowhead`; `_wrap_pm` from `ka_gridformer`.
- Produces: `KACurveFlowModel(KAFlowHeadModel)` with `_curve_feat(N, device) -> Tensor[N, d_model]` (zero at init), overridden `log_prob`/`sample` that add `_curve_feat` to the conditioning. (Inherits `frame_mode` scaffold; this plan uses scaffold.)

- [ ] **Step 1: Write the failing tests**

```python
# liquid_coupling_flow/tests/test_curveflow.py
import math, torch
from liquid_coupling_flow.ka_curveflow import KACurveFlowModel


def _tiny(seed=0):
    torch.manual_seed(seed)
    m = KACurveFlowModel(rho=1.2, n_bins=192, knn=8); m.eval()   # knn<N=12
    return m


def test_curve_feat_zero_at_init():
    # Zero-initialized curve projection => curve feature is exactly zero => KACurveFlowModel == B-alone at init.
    m = _tiny(0)
    cf = m._curve_feat(12, "cpu")
    assert cf.shape == (12, m.d_model)
    assert cf.abs().max().item() == 0.0


def test_curve_feat_deterministic_per_index():
    # The curve feature depends only on (N, j) — same for every config (it's the GPS coordinate).
    m = _tiny(1)
    with torch.no_grad():                                        # make it nonzero
        for p in m.curve_proj.parameters():
            p.add_(0.1 * torch.randn_like(p))
    a = m._curve_feat(12, "cpu"); b = m._curve_feat(12, "cpu")
    assert torch.equal(a, b)
    assert not torch.allclose(a[3], a[7])                        # different curve positions differ


def test_exactness_gate_preserved():
    m = _tiny(2); N, B = 12, 4
    with torch.no_grad():
        for p in m.curve_proj.parameters():
            p.add_(0.05 * torch.randn_like(p))                  # nonzero curve conditioning
    pos, sp, logq = m.sample(B, N, n_B=4, device="cpu", return_logq=True)
    assert torch.allclose(logq, m.log_prob(pos, sp), atol=1e-4)


def test_log_prob_matches_b_alone_when_curve_zero():
    # With the curve feature zero (init), log_prob must equal the same computation without the curve term.
    import torch.nn.functional as F
    from liquid_coupling_flow.ka_localframe import _wrap_pm
    m = _tiny(3); N, B = 12, 3
    x = torch.rand(B, N, 2) * m._Lof(N); s = (torch.rand(B, N) < 0.35).long()
    order = m.geo._curve_order(x, N)
    xo = torch.gather(x, 1, order[..., None].expand(-1, -1, 2)); so = torch.gather(s, 1, order)
    ctx, origin = m._local(xo, so, m.geo._scaffold(N, x.device), m._Lof(N), N)
    ab = _wrap_pm(xo - origin, m._Lof(N)) / m._arc_scale(N)
    lp_ab = m.flow.log_prob(ctx, ab)                            # B-alone conditioning (curve feat is zero)
    sl = m.head_species(ctx)
    oh = F.one_hot(so, m.n_species).float(); rem = oh.sum(1, keepdim=True) - (oh.cumsum(1) - oh)
    lp_s = F.log_softmax(sl.masked_fill(rem <= 0, float("-inf")), -1).gather(-1, so[..., None]).squeeze(-1)
    expected = (lp_ab + lp_s).sum(1) - m.d * N * math.log(m._arc_scale(N))
    assert torch.allclose(m.log_prob(x, s), expected, atol=1e-4)


if __name__ == "__main__":
    test_curve_feat_zero_at_init()
    test_curve_feat_deterministic_per_index()
    test_exactness_gate_preserved()
    test_log_prob_matches_b_alone_when_curve_zero()
    print("CURVEFLOW TESTS PASSED")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest liquid_coupling_flow/tests/test_curveflow.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'liquid_coupling_flow.ka_curveflow'`.

- [ ] **Step 3: Write the curve-conditioned model**

```python
# liquid_coupling_flow/ka_curveflow.py
"""Feature A (GPS / QueryAR curve conditioning) for the spline-flow placement head. Adds a per-particle,
deterministic curve-feature (periodic encoding of the scaffold position s_j from geo._scaffold + the
arc-length coord (j+0.5)/N) to the local-frame context before the exact spline flow. Zero-initialized so
it starts identical to B-alone; warm-started from the trained B checkpoint to isolate the curve increment.
Exact likelihood preserved (the curve feature is part of the conditioning, identical in log_prob and
sample). TEST OF THE HYPOTHESIS that 'giving the curve' sharpens the conditional vs B-alone (2.704)."""
from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
from liquid_coupling_flow.ka_flowhead import KAFlowHeadModel
from liquid_coupling_flow.ka_localframe import _wrap_pm


class KACurveFlowModel(KAFlowHeadModel):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        enc = 2 * self.d * self.periods.numel()                          # sin/cos over d coords x K periods
        self.curve_proj = nn.Sequential(nn.Linear(enc + 1, self.d_model), nn.GELU(),
                                        nn.Linear(self.d_model, self.d_model))
        nn.init.zeros_(self.curve_proj[-1].weight); nn.init.zeros_(self.curve_proj[-1].bias)   # zero -> == B at init

    def _curve_feat(self, N, device):
        sc = self.geo._scaffold(N, device)                               # [N,2] curve position s_j (the GPS coord)
        a = 2 * math.pi * sc.unsqueeze(-1) / self.periods.to(device)     # [N,2,K]
        pe = torch.cat([torch.sin(a), torch.cos(a)], -1).flatten(1)      # [N, 2*d*K]
        arclen = ((torch.arange(N, device=device) + 0.5) / N)[:, None]   # [N,1] normalized arc-length
        return self.curve_proj(torch.cat([pe, arclen], -1))              # [N, d_model]

    def log_prob(self, x, s, canonical=None, preordered=False):
        B, N = x.shape[0], x.shape[1]; s = s.long(); s = s.expand(B, N).clone() if s.dim() == 1 else s
        L = self._Lof(N)
        if preordered:
            xo, so = x, s
        else:
            order = self.geo._curve_order(x, N)
            xo = torch.gather(x, 1, order[..., None].expand(-1, -1, 2)); so = torch.gather(s, 1, order)
        sc = self.geo._scaffold(N, x.device)
        context, origin = self._local(xo, so, sc, L, N)
        context = context + self._curve_feat(N, x.device)[None]          # <-- GPS curve conditioning
        ab = _wrap_pm(xo - origin, L) / self._arc_scale(N)
        lp_ab = self.flow.log_prob(context, ab)
        s_logits = self.head_species(context)
        if self.canonical if canonical is None else canonical:
            oh = F.one_hot(so, self.n_species).to(s_logits.dtype)
            rem = oh.sum(1, keepdim=True) - (oh.cumsum(1) - oh)
            s_logits = s_logits.masked_fill(rem <= 0, float("-inf"))
        lp_s = F.log_softmax(s_logits, -1).gather(-1, so[..., None]).squeeze(-1)
        return (lp_ab + lp_s).sum(1) - self.d * N * math.log(self._arc_scale(N))

    @torch.no_grad()
    def sample(self, B, N, n_B=None, device=None, return_logq=False):
        L = self._Lof(N); sc = self.geo._scaffold(N, device); arc = self._arc_scale(N)
        cf = self._curve_feat(N, device)                                 # [N, d_model] precomputed once
        pos = torch.zeros(B, N, 2, device=device); sp = torch.zeros(B, N, dtype=torch.long, device=device)
        rem = None
        if n_B is not None:
            rem = torch.zeros(B, self.n_species, device=device); rem[:, 0] = N - n_B; rem[:, 1] = n_B
        for j in range(N):
            h, origin = self._step(pos, sp, sc[j], j, L)
            h = h + cf[j]                                                 # <-- GPS curve conditioning (particle j)
            sj = torch.multinomial(F.softmax(self._species_logits(h, rem), -1), 1).squeeze(-1)
            ab, _ = self.flow.sample(h)
            if rem is not None:
                rem[torch.arange(B, device=device), sj] -= 1
            pos[:, j] = torch.remainder(origin + ab * arc, L); sp[:, j] = sj
        perm = self.geo._curve_order(pos, N)
        pos = torch.gather(pos, 1, perm[..., None].expand(-1, -1, 2)); sp = torch.gather(sp, 1, perm)
        if return_logq:
            return pos, sp, self.log_prob(pos, sp)
        return pos, sp
```

Note: `sample` adds `cf[j]` (the curve feature for particle j) to `h` BEFORE the species sampler and the flow, exactly mirroring `log_prob`'s `context + cf[None]` — so the conditioning is identical and exactness holds. The `frame_mode="inertial"` rotation is inherited but unused here (this plan runs scaffold); the inertial path is NOT re-implemented (out of scope).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest liquid_coupling_flow/tests/test_curveflow.py -q`
Expected: PASS (4 passed). If `test_log_prob_matches_b_alone_when_curve_zero` fails, `_curve_feat` is not exactly zero at init (check the final-layer zero-init) or the `log_prob` body diverged from `KAFlowHeadModel`'s formula.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_curveflow.py liquid_coupling_flow/tests/test_curveflow.py
git commit -m "feat(ka): KACurveFlowModel - GPS/QueryAR curve conditioning on the spline-flow head (zero-init, exact)"
```

---

### Task 2: Train Feature A (warm-start from B) + measure the increment

**Files:**
- Modify: `liquid_coupling_flow/ka_curveflow.py` (append `train` + `__main__`)
- Create: appends to `reports/2026-06-26-ka-spline-flow-head-results.md` (results addendum)

**Interfaces:**
- Consumes: `KACurveFlowModel`, `augment` (`ka_gridformer_train`), `load_compat` (`ka_exposure_lf`), `_categorical_pos_logdensity` (`ka_flowhead`).
- Produces: checkpoint `ka_curveflow_N100.pt`; printed position-log-density (vs B-alone 2.704) and g(r) increment.

- [ ] **Step 1: Append the training runner**

```python
# --- append to liquid_coupling_flow/ka_curveflow.py ---
import os, time
ART = os.path.join(os.path.dirname(__file__), "artifacts")


def train(steps=20000, train_N=100, warm="ka_flowhead_N100_k8_scratch.pt", lr=2e-4,
          device="cuda" if torch.cuda.is_available() else "cpu"):
    from liquid_coupling_flow.ka_gridformer_train import augment
    from liquid_coupling_flow.ka_exposure_lf import load_compat
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    data, sp, L = ref["x"].to(device), ref["s"].to(device).long(), ref["L"]; N = data.shape[1]
    ck = torch.load(os.path.join(ART, warm), map_location=device, weights_only=False)
    m = KACurveFlowModel(rho=1.2, n_bins=192, knn=16, num_bins=ck["num_bins"], tail_bound=ck["tail_bound"]).to(device)
    load_compat(m, ck["state_dict"])                                     # warm-start B (curve_proj stays zero-init)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4); B, t0 = 128, time.time(); warmup = 500
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / warmup)
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        loss = (-m.log_prob(augment(data[idx], L), sp) / N).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"NON-FINITE loss at step {step}")
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if step % 1000 == 0:
            print(f"  step {step:5d} nll/N {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    torch.save({"state_dict": m.state_dict(), "rho": 1.2, "n_bins": 192, "knn": 16,
                "num_bins": ck["num_bins"], "tail_bound": ck["tail_bound"], "step": steps},
               os.path.join(ART, "ka_curveflow_N100.pt"))
    print("saved ka_curveflow_N100.pt", flush=True)


@torch.no_grad()
def measure(train_N=100, device="cuda" if torch.cuda.is_available() else "cpu"):
    from liquid_coupling_flow.ka_flowhead import _categorical_pos_logdensity
    ck = torch.load(os.path.join(ART, "ka_curveflow_N100.pt"), map_location=device, weights_only=False)
    m = KACurveFlowModel(rho=1.2, n_bins=192, knn=16, num_bins=ck["num_bins"], tail_bound=ck["tail_bound"]).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    s, L, data = ref["s"].to(device).long(), ref["L"], ref["x"].to(device)[:128]; N = data.shape[1]
    o = m.geo._curve_order(data, N); xo = torch.gather(data, 1, o[..., None].expand(-1, -1, 2))
    so = torch.gather(s.expand(data.shape[0], N) if s.dim() == 1 else s[:128], 1, o)
    ctx, org = m._local(xo, so, m.geo._scaffold(N, device), L, N); ctx = ctx + m._curve_feat(N, device)[None]
    ab = _wrap_pm(xo - org, L) / m._arc_scale(N)
    print(f"Feature A flow pos log-density {float(m.flow.log_prob(ctx, ab).mean()):.3f}  "
          f"(B-alone 2.704, categorical {_categorical_pos_logdensity(train_N, device):.3f})", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "measure":
        measure()
    else:
        train(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
```

- [ ] **Step 2: Smoke-run (200 steps) to verify the loop**

Run: `python -c "from liquid_coupling_flow.ka_curveflow import train; train(steps=200)"`
Expected: decreasing `nll/N`, no `FloatingPointError`, saves `artifacts/ka_curveflow_N100.pt`. (Controller runs the full 20k.)

- [ ] **Step 3: Full training + the increment measurement (controller experiment)**

Run:
```bash
python -m liquid_coupling_flow.ka_curveflow 20000
python -m liquid_coupling_flow.ka_curveflow measure
python -m liquid_coupling_flow.ka_flowhead_eval ka_curveflow_N100.pt
```
Expected: the position log-density (vs B-alone 2.704) and the free-run g(r) (vs B-alone / categorical). **The verdict for Feature A:** if the curve conditioning raises the position log-density above 2.704 and/or sharpens g(r) toward data, GPS-guided helps; if it sits at 2.704 with unchanged g(r), the curve (deterministic, config-independent) adds nothing — confirming the conditional's broadness is a config-information limit, not a curve-awareness limit.

- [ ] **Step 4: Append the results addendum + commit**

Append a "## Feature A (curve/GPS conditioning) — increment over B" section to `reports/2026-06-26-ka-spline-flow-head-results.md` with the position log-density (A vs B 2.704 vs categorical 2.783), the g(r) table (A vs B vs data), and the verdict (did giving the curve move either metric?).

```bash
git add liquid_coupling_flow/ka_curveflow.py reports/2026-06-26-ka-spline-flow-head-results.md
git commit -m "feat(ka): Feature A curve-conditioning training + increment measurement vs B-alone"
```

---

## Self-Review

**Spec coverage (spec §5 Feature A):**
- QueryAR position query / "place-here" signal → the per-particle curve feature (`_curve_feat`) added to the conditioning (Task 1). The full *interleaved learnable query token inside `_local`* is not done (it would require editing `ka_localframe`); the post-context additive form is the faithful, exactness-safe realization — noted. ✓ (with that scoping)
- Scaffold position `s_j` + arc-length coords → `_curve_feat` (periodic encoding of `geo._scaffold` + `(j+0.5)/N`). ✓
- Δs range soft prior → folded into the arc-length coord (the per-index arc-length is the bounded-region signal); a separate Δs scalar is YAGNI given arc-length already encodes index position. Noted as a deliberate simplification.
- Hard Δs support (Approach 3) → explicitly out of scope (spec defers it). ✓
- "Measured as an increment over B-alone" → warm-start from B + zero-init curve_proj + Task 2 measurement. ✓
- Exactness preserved → curve feature identical in log_prob/sample; `test_exactness_gate_preserved`. ✓

**Placeholder scan:** no TBD/"add error handling"/"similar to Task N". The duplication of `log_prob`/`sample` from `KAFlowHeadModel` is deliberate and flagged (keeps the verified Phase-1 model untouched).

**Type consistency:** `_curve_feat(N, device) -> [N, d_model]` used identically in `log_prob` (`[None]` broadcast) and `sample` (`cf[j]`). `KACurveFlowModel.__init__(*args, **kw)` matches `KAFlowHeadModel`. Checkpoint keys (`num_bins`, `tail_bound`) read in `measure` match those written in `train` and in the Phase-1 `ka_flowhead.train`. `_categorical_pos_logdensity` imported from `ka_flowhead` (defined there in Phase 1).
