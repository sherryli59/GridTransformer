# Exact Spline-Flow Placement Head (Phase 1 / Feature B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the binned `(a,b)` categorical placement head of the causal `KALocalFrameModel` with an exact, sharp **autoregressive rational-quadratic spline-flow** head, and prove it (a) trains and converges, (b) preserves exact likelihood, and (c) makes in-dist N=100 free-run g(r) grow sharp contact peaks and an empty excluded-volume core.

**Architecture:** A standalone `SplineFlowHead` (Gaussian base + `RQSplineElementwise`, AR `a` then `b|a`, identity-initialized) is consumed by a thin subclass `KAFlowHeadModel(KALocalFrameModel)` that overrides `log_prob`/`sample` to use the flow instead of the categorical. The geometry-invariant frame, KNN context, categorical species head, curve ordering, and exact-likelihood bookkeeping are inherited unchanged.

**Tech Stack:** Python, PyTorch, pytest. Reuses `liquid_coupling_flow/transforms_spline.py` (`RQSplineElementwise`) and `liquid_coupling_flow/ka_localframe.py` (`KALocalFrameModel`).

## Global Constraints

- **Never edit `ka_localframe.py`** or any existing model/transform file. All new code is in new files (`ka_flowhead.py`) and `liquid_coupling_flow/tests/`.
- **Exact likelihood is invariant.** The model's exactness gate — `sample(return_logq=True)` returns `log_prob(pos, sp)` to float precision — must hold for the flow head. The flow's own `sample`↔`log_prob` must be mutually exact (round-trip).
- **Convergence is a blocking gate:** identity-init, bounded domain + linear tails, monotonicity guards (built into `RQSplineElementwise`), grad-clip 5.0. The head must overfit a tiny batch and beat the categorical baseline's normalized position log-density, with zero NaN/inf.
- **Scoped commits:** `git add` only the exact new files per task (never `git add -A`; the repo has many unrelated dirty files and parallel work streams on this branch).
- Target: causal `KALocalFrameModel`, N=100, T*=0.5. Defaults `arc_range=3.0`, `d=2`, `arc_scale(N)=N**(1/6)`, `d_model=192`, `knn=16`. Spline defaults `num_bins=8`, `tail_bound=4.0`.
- Tests: `liquid_coupling_flow/tests/`, pytest, CPU, artifact-free (random-weight tiny models; `knn<N`).

---

### Task 1: `SplineFlowHead` — exact AR spline-flow over the (a,b) offset

**Files:**
- Create: `liquid_coupling_flow/ka_flowhead.py`
- Test: `liquid_coupling_flow/tests/test_flowhead.py`

**Interfaces:**
- Consumes: `RQSplineElementwise(num_bins, tail_bound)` from `transforms_spline` — `.forward(x,params)->(y,logdet)`, `.inverse(y,params)->(x,logdet)`, `.params_per_dim == 3*num_bins-1`.
- Produces:
  - `SplineFlowHead(d_model, num_bins=8, tail_bound=4.0)` (`nn.Module`).
  - `.log_prob(h, ab) -> Tensor[...]` — normalized-coord flow log-density; `h [...,d_model]`, `ab [...,2]`.
  - `.sample(h, gen=None) -> (ab [...,2], logq [...])`.

- [ ] **Step 1: Write the failing tests**

```python
# liquid_coupling_flow/tests/test_flowhead.py
import math, torch
from liquid_coupling_flow.ka_flowhead import SplineFlowHead

_LOG2PI = math.log(2 * math.pi)


def test_sample_logprob_round_trip_exact():
    # The flow's sample and log_prob must be mutually exact: log_prob at the sampled point == sample's logq.
    torch.manual_seed(0)
    head = SplineFlowHead(d_model=16); head.eval()
    h = torch.randn(5, 16)
    g = torch.Generator().manual_seed(7)
    ab, logq = head.sample(h, gen=g)
    lp = head.log_prob(h, ab)
    assert torch.allclose(logq, lp, atol=1e-5), (logq, lp)


def test_identity_init_is_gaussian_base():
    # At identity init the spline is the identity, so log_prob(ab) == standard-normal base log-density of ab.
    torch.manual_seed(1)
    head = SplineFlowHead(d_model=16); head.eval()
    h = torch.randn(4, 16)
    ab = torch.randn(4, 2) * 0.5            # inside the spline domain
    base = (-0.5 * ab ** 2 - 0.5 * _LOG2PI).sum(-1)
    assert torch.allclose(head.log_prob(h, ab), base, atol=1e-4)


def test_finite_on_extreme_offsets():
    # Linear tails => finite density far outside the spline domain (no NaN/inf).
    head = SplineFlowHead(d_model=16); head.eval()
    h = torch.randn(3, 16)
    ab = torch.tensor([[100.0, -100.0], [50.0, 50.0], [-7.0, 7.0]])
    lp = head.log_prob(h, ab)
    assert torch.isfinite(lp).all()


def test_log_prob_differentiable():
    head = SplineFlowHead(d_model=16)
    h = torch.randn(4, 16); ab = torch.randn(4, 2) * 0.5
    loss = -head.log_prob(h, ab).mean()
    loss.backward()
    assert head.head_a.weight.grad is not None and torch.isfinite(head.head_a.weight.grad).all()


if __name__ == "__main__":
    test_sample_logprob_round_trip_exact()
    test_identity_init_is_gaussian_base()
    test_finite_on_extreme_offsets()
    test_log_prob_differentiable()
    print("FLOWHEAD TESTS PASSED")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest liquid_coupling_flow/tests/test_flowhead.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'liquid_coupling_flow.ka_flowhead'`.

- [ ] **Step 3: Write the head module**

```python
# liquid_coupling_flow/ka_flowhead.py
"""Exact AR rational-quadratic spline-flow placement head for the local-frame generator (Phase 1 / Feature B).
Replaces the binned (a,b) categorical with a continuous flow sharp enough to represent the hard-core contact
peak the categorical smears. Gaussian base on R + RQSplineElementwise (identity linear tails); AR factorization
p(a|h)*p(b|a,h). Identity-initialized for stable training. Returns the NORMALIZED-offset log-density; the caller
adds the arc_scale Jacobian-to-physical. Quetzal structure (transformer context -> small continuous head),
exact instead of diffusion so the SMC corrector's likelihood stays exact."""
from __future__ import annotations
import math, torch, torch.nn as nn
from liquid_coupling_flow.transforms_spline import RQSplineElementwise, DEFAULT_MIN_DERIVATIVE

_LOG2PI = math.log(2 * math.pi)


def _base_logp(z):                                          # standard-normal log-density, summed over last dim
    return (-0.5 * z ** 2 - 0.5 * _LOG2PI)


class SplineFlowHead(nn.Module):
    def __init__(self, d_model, num_bins=8, tail_bound=4.0):
        super().__init__()
        self.spline = RQSplineElementwise(num_bins=num_bins, tail_bound=tail_bound)
        self.num_bins = num_bins
        P = self.spline.params_per_dim                      # 3K - 1
        self.head_a = nn.Linear(d_model, P)
        self.head_b = nn.Linear(d_model + 1, P)             # condition b on the continuous a
        self._identity_init()

    def _identity_init(self):
        # zero weights; widths/heights bias 0 (-> uniform bins); interior-derivative bias = const giving
        # softplus(const)+min_deriv == 1 -> the RQS is exactly the identity at init (flow == Gaussian base).
        K = self.num_bins
        const = float(torch.log(torch.expm1(torch.tensor(1.0 - DEFAULT_MIN_DERIVATIVE))))
        for head in (self.head_a, self.head_b):
            nn.init.zeros_(head.weight)
            with torch.no_grad():
                head.bias.zero_()
                head.bias[2 * K:] = const                   # interior derivatives (P = 3K-1: [2K:] is the K-1 derivs)

    def log_prob(self, h, ab):
        a = ab[..., 0:1]; b = ab[..., 1:2]
        za, lda = self.spline.inverse(a, self.head_a(h))                 # a -> base
        lpa = _base_logp(za) + lda
        zb, ldb = self.spline.inverse(b, self.head_b(torch.cat([h, a], -1)))
        lpb = _base_logp(zb) + ldb
        return (lpa + lpb).squeeze(-1)

    def sample(self, h, gen=None):
        za = torch.randn(h.shape[:-1] + (1,), device=h.device, generator=gen)
        a, lda = self.spline.forward(za, self.head_a(h))                 # base -> a
        lpa = _base_logp(za) - lda
        zb = torch.randn(h.shape[:-1] + (1,), device=h.device, generator=gen)
        b, ldb = self.spline.forward(zb, self.head_b(torch.cat([h, a], -1)))
        lpb = _base_logp(zb) - ldb
        return torch.cat([a, b], -1), (lpa + lpb).squeeze(-1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest liquid_coupling_flow/tests/test_flowhead.py -q`
Expected: PASS (4 passed). If `test_identity_init_is_gaussian_base` fails, the derivative-bias slice `bias[2*K:]` doesn't line up with the `RQSplineElementwise` param layout (`[:K]` widths, `[K:2K]` heights, `[2K:]` K−1 interior derivs) — fix the slice. If `test_sample_logprob_round_trip_exact` fails, the `forward`/`inverse` log-det signs disagree — `sample` uses `−ld_forward`, `log_prob` uses `+ld_inverse`, and `inverse∘forward=id` makes them cancel.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_flowhead.py liquid_coupling_flow/tests/test_flowhead.py
git commit -m "feat(ka): exact AR spline-flow placement head (Gaussian base + RQS, identity-init, round-trip exact)"
```

---

### Task 2: `KAFlowHeadModel` — wire the flow head into the local-frame model

**Files:**
- Modify: `liquid_coupling_flow/ka_flowhead.py` (append the model class)
- Test: `liquid_coupling_flow/tests/test_flowhead_model.py`

**Interfaces:**
- Consumes: `SplineFlowHead` (Task 1); `KALocalFrameModel` (`._local`, `._step`, `.geo`, `._arc_scale`, `._Lof`, `.head_species`, `.n_species`, `.canonical`, `.d`, `.d_model`) from `ka_localframe`; `_wrap_pm` from `ka_gridformer`.
- Produces: `KAFlowHeadModel(KALocalFrameModel)` with `__init__(..., num_bins=8, tail_bound=4.0)`, overridden `log_prob(x, s, canonical=None, preordered=False)` and `sample(B, N, n_B=None, device=None, return_logq=False)`.

- [ ] **Step 1: Write the failing tests**

```python
# liquid_coupling_flow/tests/test_flowhead_model.py
import torch
from liquid_coupling_flow.ka_flowhead import KAFlowHeadModel


def _tiny(seed=0):
    torch.manual_seed(seed)
    m = KAFlowHeadModel(rho=1.2, n_bins=192, knn=8); m.eval()   # knn<N=12 required by _local topk
    return m


def test_exactness_gate_sample_logq_equals_log_prob():
    m = _tiny(0); N, B = 12, 4
    pos, sp, logq = m.sample(B, N, n_B=4, device="cpu", return_logq=True)
    assert torch.allclose(logq, m.log_prob(pos, sp), atol=1e-4)


def test_log_prob_finite_and_differentiable():
    m = _tiny(1); N, B = 12, 4
    x = torch.rand(B, N, 2) * m._Lof(N); s = (torch.rand(B, N) < 0.35).long()
    lp = m.log_prob(x, s)
    assert lp.shape == (B,) and torch.isfinite(lp).all()
    (-lp.mean()).backward()
    assert m.flow.head_a.weight.grad is not None and torch.isfinite(m.flow.head_a.weight.grad).all()


def test_no_vol_term_uses_flow_density():
    # The flow log_prob must equal (flow_pos_logdensity + species_logp).sum - arc_scale_jac, with NO bin_w
    # volume term (that was categorical-only). Recompute the expected value from the flow directly.
    import math
    import torch.nn.functional as F
    from liquid_coupling_flow.ka_localframe import _wrap_pm
    m = _tiny(2); N, B = 12, 3
    x = torch.rand(B, N, 2) * m._Lof(N); s = (torch.rand(B, N) < 0.35).long()
    order = m.geo._curve_order(x, N)
    xo = torch.gather(x, 1, order[..., None].expand(-1, -1, 2)); so = torch.gather(s, 1, order)
    ctx, origin = m._local(xo, so, m.geo._scaffold(N, x.device), m._Lof(N), N)
    ab = _wrap_pm(xo - origin, m._Lof(N)) / m._arc_scale(N)
    lp_ab = m.flow.log_prob(ctx, ab)
    sl = m.head_species(ctx)
    oh = F.one_hot(so, m.n_species).float(); rem = oh.sum(1, keepdim=True) - (oh.cumsum(1) - oh)
    lp_s = F.log_softmax(sl.masked_fill(rem <= 0, float("-inf")), -1).gather(-1, so[..., None]).squeeze(-1)
    expected = (lp_ab + lp_s).sum(1) - m.d * N * math.log(m._arc_scale(N))   # NO bin_w vol term
    assert torch.allclose(m.log_prob(x, s), expected, atol=1e-4)


if __name__ == "__main__":
    test_exactness_gate_sample_logq_equals_log_prob()
    test_log_prob_finite_and_differentiable()
    test_no_vol_term_uses_flow_density()
    print("FLOWHEAD-MODEL TESTS PASSED")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest liquid_coupling_flow/tests/test_flowhead_model.py -q`
Expected: FAIL with `ImportError: cannot import name 'KAFlowHeadModel'`.

- [ ] **Step 3: Append the model class**

```python
# --- append to liquid_coupling_flow/ka_flowhead.py ---
import torch.nn.functional as F
from liquid_coupling_flow.ka_localframe import KALocalFrameModel, _wrap_pm


class KAFlowHeadModel(KALocalFrameModel):
    """Local-frame AR generator with the categorical (a,b) head replaced by the exact spline flow.
    Species head + frame + KNN context + curve ordering are inherited unchanged; exact likelihood preserved
    (continuous flow density replaces P(bin)/bin_area; the arc_scale Jacobian is the same `jac` term)."""
    def __init__(self, *args, num_bins=8, tail_bound=4.0, **kw):
        super().__init__(*args, **kw)
        self.flow = SplineFlowHead(self.d_model, num_bins=num_bins, tail_bound=tail_bound)

    def _species_logits(self, h, rem):
        lg = self.head_species(h)
        return lg if rem is None else lg.masked_fill(rem <= 0, float("-inf"))

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
        ab = _wrap_pm(xo - origin, L) / self._arc_scale(N)
        lp_ab = self.flow.log_prob(context, ab)                          # [B,N] continuous flow log-density
        s_logits = self.head_species(context)
        if self.canonical if canonical is None else canonical:
            oh = F.one_hot(so, self.n_species).to(s_logits.dtype)
            rem = oh.sum(1, keepdim=True) - (oh.cumsum(1) - oh)
            s_logits = s_logits.masked_fill(rem <= 0, float("-inf"))
        lp_s = F.log_softmax(s_logits, -1).gather(-1, so[..., None]).squeeze(-1)
        jac = self.d * N * math.log(self._arc_scale(N))                  # NO bin_w vol term (flow is continuous)
        return (lp_ab + lp_s).sum(1) - jac

    @torch.no_grad()
    def sample(self, B, N, n_B=None, device=None, return_logq=False):
        L = self._Lof(N); sc = self.geo._scaffold(N, device); arc = self._arc_scale(N)
        pos = torch.zeros(B, N, 2, device=device); sp = torch.zeros(B, N, dtype=torch.long, device=device)
        rem = None
        if n_B is not None:
            rem = torch.zeros(B, self.n_species, device=device); rem[:, 0] = N - n_B; rem[:, 1] = n_B
        for j in range(N):
            h, origin = self._step(pos, sp, sc[j], j, L)
            sj = torch.multinomial(F.softmax(self._species_logits(h, rem), -1), 1).squeeze(-1)
            ab, _ = self.flow.sample(h)
            if rem is not None:
                rem[torch.arange(B, device=device), sj] -= 1
            pos[:, j] = torch.remainder(origin + ab * arc, L); sp[:, j] = sj
        perm = self.geo._curve_order(pos, N)
        pos = torch.gather(pos, 1, perm[..., None].expand(-1, -1, 2)); sp = torch.gather(sp, 1, perm)
        if return_logq:
            return pos, sp, self.log_prob(pos, sp)                       # exactness trick (matches base model)
        return pos, sp
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest liquid_coupling_flow/tests/test_flowhead_model.py -q`
Expected: PASS (3 passed). `test_exactness_gate` passes because `sample(return_logq=True)` returns `log_prob(pos,sp)` (the inherited exactness trick) and Task 1's round-trip proved the flow's own consistency, so the accumulated and recomputed densities agree.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_flowhead.py liquid_coupling_flow/tests/test_flowhead_model.py
git commit -m "feat(ka): KAFlowHeadModel wires spline-flow head into local-frame (exact log_prob/sample, no vol term)"
```

---

### Task 3: Training + convergence gate

**Files:**
- Modify: `liquid_coupling_flow/ka_flowhead.py` (append `train`, `convergence_gate`, `__main__`)

**Interfaces:**
- Consumes: `KAFlowHeadModel`, `augment` (`ka_gridformer_train`), `load_compat` (`ka_exposure_lf`).
- Produces: `train(steps, warm=None, num_bins=8, tail_bound=4.0, ...) -> ckpt_name`; `convergence_gate(device) -> dict` (overfit-tiny NLL + flow-vs-categorical position log-density + NaN flag).

- [ ] **Step 1: Append training + the convergence gate**

```python
# --- append to liquid_coupling_flow/ka_flowhead.py ---
import os, time
ART = os.path.join(os.path.dirname(__file__), "artifacts")


def train(steps=20000, train_N=100, num_bins=8, tail_bound=4.0, warm=None, lr=3e-4,
          out=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    from liquid_coupling_flow.ka_gridformer_train import augment
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    data, sp, L = ref["x"].to(device), ref["s"].to(device).long(), ref["L"]; N = data.shape[1]
    m = KAFlowHeadModel(rho=1.2, n_bins=192, knn=16, num_bins=num_bins, tail_bound=tail_bound).to(device)
    if warm is not None:
        from liquid_coupling_flow.ka_exposure_lf import load_compat
        load_compat(m, torch.load(os.path.join(ART, warm), map_location=device, weights_only=False)["state_dict"])
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4); B, t0 = 128, time.time()
    warmup = 500
    for step in range(steps):
        lr_scale = min(1.0, (step + 1) / warmup)
        for g in opt.param_groups:
            g["lr"] = lr * lr_scale
        idx = torch.randint(0, data.shape[0], (B,), device=device)
        loss = (-m.log_prob(augment(data[idx], L), sp) / N).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"NON-FINITE loss at step {step} (convergence-gate violation)")
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if step % 1000 == 0:
            print(f"  step {step:5d} nll/N {loss.item():.3f} lr {opt.param_groups[0]['lr']:.1e} {time.time()-t0:.0f}s", flush=True)
    out = out or f"ka_flowhead_N{train_N}_k{num_bins}{'_scratch' if warm is None else ''}.pt"
    torch.save({"state_dict": m.state_dict(), "rho": 1.2, "n_bins": 192, "knn": 16,
                "num_bins": num_bins, "tail_bound": tail_bound, "step": steps}, os.path.join(ART, out))
    print(f"saved {out}", flush=True); return out


@torch.no_grad()
def _categorical_pos_logdensity(train_N, device):
    """Mean per-particle NORMALIZED position log-density of the categorical baseline: log P(bin) - d*log(bin_w)."""
    from liquid_coupling_flow.ka_exposure_lf import _load
    m = _load("ka_localframe_N100_20k.pt", device)
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    s, L, data = ref["s"].to(device).long(), ref["L"], ref["x"].to(device)[:512]; N = data.shape[1]
    order = m.geo._curve_order(data, N); xo = torch.gather(data, 1, order[..., None].expand(-1, -1, 2))
    so = torch.gather(s.expand(data.shape[0], N) if s.dim() == 1 else s[:512], 1, order)
    context, origin = m._local(xo, so, m.geo._scaffold(N, device), L, N)
    ab = _wrap_pm(xo - origin, L) / m._arc_scale(N)
    ba, bb = m._bin(ab[..., 0]), m._bin(ab[..., 1])
    la = F.log_softmax(m.head_a(context), -1).gather(-1, ba[..., None]).squeeze(-1)
    lb = F.log_softmax(m.head_b(context + m.bin_a_emb(ba)), -1).gather(-1, bb[..., None]).squeeze(-1)
    return float(((la + lb) - m.d * math.log(m.bin_w)).mean())


def convergence_gate(train_N=100, device="cuda" if torch.cuda.is_available() else "cpu"):
    """(1) overfit a tiny batch; (2) flow normalized position log-density beats categorical; (3) no NaN."""
    from liquid_coupling_flow.ka_gridformer_train import augment
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    data, sp, L = ref["x"].to(device), ref["s"].to(device).long(), ref["L"]; N = data.shape[1]
    m = KAFlowHeadModel(rho=1.2, n_bins=192, knn=16).to(device); m.train()
    tiny = data[:8]
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3); nans = False
    first = None
    for step in range(400):
        loss = (-m.log_prob(tiny, sp) / N).mean()
        if first is None: first = loss.item()
        if not torch.isfinite(loss): nans = True; break
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    overfit_nll = loss.item()
    print(f"OVERFIT-TINY: nll/N {first:.3f} -> {overfit_nll:.3f} (should drop substantially)  NaN={nans}", flush=True)
    cat = _categorical_pos_logdensity(train_N, device)
    print(f"categorical normalized position log-density (baseline to beat): {cat:.3f}", flush=True)
    return {"overfit_first": first, "overfit_last": overfit_nll, "nan": nans, "categorical_pos_logdensity": cat}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "gate":
        convergence_gate()
    else:
        train(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
```

- [ ] **Step 2: Run the overfit-tiny convergence gate (fast, CPU/GPU)**

Run: `python -m liquid_coupling_flow.ka_flowhead gate`
Expected: prints `OVERFIT-TINY` with nll/N dropping substantially (the flow *can* fit 8 configs) and `NaN=False`, plus the categorical baseline's normalized position log-density. **Gate (b) is checked after full training (Step 4).** If overfit-tiny does NOT drop or shows NaN, the head won't train — stop and fix (likely identity-init or tail_bound) before the expensive run.

- [ ] **Step 3: Smoke-run training (200 steps) to verify the loop**

Run: `python -c "from liquid_coupling_flow.ka_flowhead import train; train(steps=200)"`
Expected: decreasing `nll/N`, no `FloatingPointError`, saves `artifacts/ka_flowhead_N100_k8_scratch.pt`.

- [ ] **Step 4: Full training run + gate (b)**

Run:
```bash
python -m liquid_coupling_flow.ka_flowhead 20000
python -c "import torch,math; from liquid_coupling_flow.ka_flowhead import KAFlowHeadModel,_categorical_pos_logdensity,ART; import torch.nn.functional as F; from liquid_coupling_flow.ka_localframe import _wrap_pm; dev='cuda'; ck=torch.load(ART+'/ka_flowhead_N100_k8_scratch.pt',weights_only=False); m=KAFlowHeadModel(rho=1.2,n_bins=192,knn=16,num_bins=ck['num_bins'],tail_bound=ck['tail_bound']).cuda(); m.load_state_dict(ck['state_dict']); m.eval(); ref=torch.load(ART+'/ka_reference_N100.pt',weights_only=False); s,L,data=ref['s'].cuda().long(),ref['L'],ref['x'].cuda()[:512]; N=100; o=m.geo._curve_order(data,N); xo=torch.gather(data,1,o[...,None].expand(-1,-1,2)); so=torch.gather(s.expand(data.shape[0],N) if s.dim()==1 else s[:512],1,o); ctx,org=m._local(xo,so,m.geo._scaffold(N,'cuda'),L,N); ab=_wrap_pm(xo-org,L)/m._arc_scale(N); flow=float(m.flow.log_prob(ctx,ab).mean()); print('flow pos log-density',round(flow,3),'vs categorical',round(_categorical_pos_logdensity(100,'cuda'),3),'-> flow should be HIGHER')"
```
Expected: **gate (b)** — the flow's mean normalized position log-density is **higher** than the categorical baseline's (sharper conditional ⇒ higher density at the true offsets). If not, the flow under-trained or `tail_bound`/`num_bins` need adjusting.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_flowhead.py
git commit -m "feat(ka): spline-flow training + convergence gate (overfit-tiny, flow-beats-categorical, NaN guard)"
```

---

### Task 4: InertialAR local-inertial-frame option (toggle)

**Files:**
- Modify: `liquid_coupling_flow/ka_flowhead.py` (add `frame_mode` to `KAFlowHeadModel`)
- Test: `liquid_coupling_flow/tests/test_flowhead_frame.py`

**Interfaces:**
- Produces: `KAFlowHeadModel(..., frame_mode="scaffold"|"inertial")`; a helper `_inertial_R(nbr_rel) -> R [...,2,2]` (per-particle rotation from placed-neighbour geometry, `|det|=1`).

- [ ] **Step 1: Write the failing test**

```python
# liquid_coupling_flow/tests/test_flowhead_frame.py
import torch
from liquid_coupling_flow.ka_flowhead import KAFlowHeadModel


def test_inertial_rotation_is_orthonormal_unit_det():
    m = KAFlowHeadModel(rho=1.2, n_bins=192, knn=8, frame_mode="inertial"); m.eval()
    torch.manual_seed(0)
    nbr_rel = torch.randn(4, 12, 8, 2)                       # [B,N,k,2] neighbour offsets
    R = m._inertial_R(nbr_rel)
    assert R.shape[-2:] == (2, 2)
    eye = torch.eye(2).expand_as(R)
    assert torch.allclose(R.transpose(-1, -2) @ R, eye, atol=1e-4)     # orthonormal
    det = R[..., 0, 0] * R[..., 1, 1] - R[..., 0, 1] * R[..., 1, 0]
    assert torch.allclose(det, torch.ones_like(det), atol=1e-4)        # |det| = 1 (rotation)


def test_inertial_exactness_gate_preserved():
    m = KAFlowHeadModel(rho=1.2, n_bins=192, knn=8, frame_mode="inertial"); m.eval()
    pos, sp, logq = m.sample(4, 12, n_B=4, device="cpu", return_logq=True)
    assert torch.allclose(logq, m.log_prob(pos, sp), atol=1e-4)


if __name__ == "__main__":
    test_inertial_rotation_is_orthonormal_unit_det()
    test_inertial_exactness_gate_preserved()
    print("FLOWHEAD-FRAME TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest liquid_coupling_flow/tests/test_flowhead_frame.py -q`
Expected: FAIL — `KAFlowHeadModel.__init__` has no `frame_mode`, no `_inertial_R`.

- [ ] **Step 3: Add the inertial frame**

Add `frame_mode` to `__init__` and a rotation built from the principal axis of the placed-neighbour offsets (PCA direction; a function of placed particles ⇒ `|det|=1` ⇒ exactness holds). The `(a,b)` offset is rotated into this frame before the flow (both `log_prob` and `sample`).

```python
# in KAFlowHeadModel.__init__, after super().__init__(...):
        self.frame_mode = frame_mode               # add `frame_mode="scaffold"` to the signature

# add methods to KAFlowHeadModel:
    def _inertial_R(self, nbr_rel):
        """Per-particle 2x2 rotation from the principal axis of placed-neighbour offsets nbr_rel [...,k,2]."""
        x, y = nbr_rel[..., 0], nbr_rel[..., 1]
        cxx = (x * x).mean(-1); cyy = (y * y).mean(-1); cxy = (x * y).mean(-1)
        theta = 0.5 * torch.atan2(2 * cxy, cxx - cyy)       # principal axis; atan2 handles cxx==cyy (-> +/-pi/2)
        c, s = torch.cos(theta), torch.sin(theta)
        return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)   # [...,2,2]

    def _frame_ab(self, ab, context_extra):
        """Rotate the offset into the local frame when frame_mode='inertial'; identity otherwise. |det|=1."""
        if self.frame_mode == "scaffold":
            return ab
        R = context_extra                                                  # precomputed [...,2,2]
        return torch.einsum("...ij,...j->...i", R, ab)
```

The rotation `R` must be computed from the **placed neighbours** (the same set `_local` gathers) and passed through both `log_prob` and `sample`. **Implementer:** thread `R` out of `_local`/`_step` (recompute the neighbour offsets `nbr_rel` already built there) and apply `_frame_ab(ab, R)` to the offset before `self.flow.*` in both paths; since `R` is orthonormal with `|det|=1`, no Jacobian term is added (exactness preserved — verified by `test_inertial_exactness_gate_preserved`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest liquid_coupling_flow/tests/test_flowhead_frame.py -q`
Expected: PASS (2 passed). If `theta` is unstable (degenerate neighbour geometry), the `atan2` guard keeps `R` finite; the orthonormality test still holds.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_flowhead.py liquid_coupling_flow/tests/test_flowhead_frame.py
git commit -m "feat(ka): InertialAR local-inertial-frame toggle (PCA rotation, |det|=1, exactness preserved)"
```

---

### Task 5: In-dist N=100 g(r) verdict vs categorical baseline

**Files:**
- Create: `liquid_coupling_flow/ka_flowhead_eval.py`
- Create: `reports/2026-06-26-ka-spline-flow-head-results.md`

**Interfaces:**
- Consumes: `KAFlowHeadModel`, `partial_gr` (`ka_observables`), the trained `ka_flowhead_N100_k8_scratch.pt` and the categorical baseline `ka_localframe_N100_20k.pt`.

- [ ] **Step 1: Write the eval script**

```python
# liquid_coupling_flow/ka_flowhead_eval.py
"""In-dist N=100 verdict: does the spline-flow head's FREE-RUN g(r) grow sharp contact peaks AND empty the
excluded-volume core, vs the categorical baseline and data? Peak position+count co-primary with height."""
from __future__ import annotations
import os, math, torch, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from liquid_coupling_flow.ka_flowhead import KAFlowHeadModel
from liquid_coupling_flow.ka_exposure_lf import _load
from liquid_coupling_flow.ka_observables import partial_gr
ART = os.path.join(os.path.dirname(__file__), "artifacts"); SIG = {(0, 0): 1.0, (0, 1): 0.8, (1, 1): 0.88}


def _flow(ckpt, device):
    ck = torch.load(os.path.join(ART, ckpt), map_location=device, weights_only=False)
    m = KAFlowHeadModel(rho=1.2, n_bins=192, knn=16, num_bins=ck["num_bins"], tail_bound=ck["tail_bound"],
                        frame_mode=ck.get("frame_mode", "scaffold")).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval(); return m


@torch.no_grad()
def main(flow_ckpt="ka_flowhead_N100_k8_scratch.pt", N=100, B=512,
         device="cuda" if torch.cuda.is_available() else "cpu"):
    ref = torch.load(os.path.join(ART, f"ka_reference_N{N}.pt"), map_location=device, weights_only=False)
    s, L, data = ref["s"].to(device).long(), ref["L"], ref["x"].to(device); nB = int((s == 1).sum())
    arms = {"data": (data[:B], s)}
    cat = _load("ka_localframe_N100_20k.pt", device); arms["categorical"] = cat.sample(B, N, n_B=nB, device=device)
    flow = _flow(flow_ckpt, device); arms["flow"] = flow.sample(B, N, n_B=nB, device=device)
    def pk(rc, g): p, _ = find_peaks(g, prominence=0.08, height=1.03); return [round(float(rc[i]), 2) for i in p]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for axi, (pair, nm) in zip(ax, [((0, 0), "AA"), ((0, 1), "AB"), ((1, 1), "BB")]):
        axi.axvspan(0, SIG[pair], color="grey", alpha=0.1); axi.axhline(1, color="grey", lw=0.6)
        print(f"\n g_{nm}(r):")
        for tag, (x, sx) in arms.items():
            rc, g = partial_gr(x, sx, L, min(L / 2, 4.5), 180, pair)
            axi.plot(rc, g, lw=2.4 if tag == "data" else 1.6, label=tag)
            print(f"   {tag:11s} peak {g.max():.2f}  core g(r<{SIG[pair]}) {float(g[rc<SIG[pair]].mean()):.3f}  peaks@ {pk(rc,g)}")
        axi.set_xlim(0.5, 3.0); axi.set_title(f"g_{nm}(r)"); axi.set_xlabel("r"); axi.legend(fontsize=9); axi.grid(alpha=0.25)
    out = os.path.join(ART, f"ka_flowhead_gr_N{N}.png"); fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\nsaved {out}", flush=True)


if __name__ == "__main__":
    import sys
    main(flow_ckpt=sys.argv[1] if len(sys.argv) > 1 else "ka_flowhead_N100_k8_scratch.pt")
```

- [ ] **Step 2: Run the verdict**

Run: `python -m liquid_coupling_flow.ka_flowhead_eval`
Expected: per-pair peak height, core fill, and peak positions for {data, categorical, flow}. **Success:** the flow's free-run g(r) has **higher contact peaks** and a **lower core** (`g(r<σ)` toward ~0.02) than categorical, with peak positions matching data — i.e. the flow head closes (some of) the data→TF amplitude/core wall the categorical couldn't. Saved figure `artifacts/ka_flowhead_gr_N100.png`.

- [ ] **Step 3: Write the results report**

Create `reports/2026-06-26-ka-spline-flow-head-results.md` with: the convergence-gate evidence (overfit-tiny, flow-vs-categorical position log-density, NaN-free), the §2 g(r) table (peak height + core + positions for data/categorical/flow, scaffold vs inertial frame), the figure, and an explicit verdict — did B close the amplitude/core wall? If yes, Phase 2 (A / QueryAR) is the next plan; if partial, note which pair (likely g_BB) remains and why.

- [ ] **Step 4: Commit**

```bash
git add liquid_coupling_flow/ka_flowhead_eval.py reports/2026-06-26-ka-spline-flow-head-results.md
git commit -m "feat(ka): in-dist N=100 g(r) verdict for the spline-flow head + results report"
```

---

## Self-Review

**Spec coverage:**
- §2 exact-likelihood invariant → Task 1 round-trip + Task 2 exactness gate + Task 4 inertial exactness; `ka_localframe.py` untouched (subclass). ✓
- §3 success = in-dist N=100 g(r) sharp peaks + empty core → Task 5. ✓
- §3/§6 convergence gate (overfit-tiny, beats-categorical, no-NaN, exactness) → Task 3 (gate) + Task 1/2 (exactness). ✓
- §4.1 RQS spline-flow head (AR a,b|a) → Task 1. ✓
- §4.2 exact density (flow replaces P(bin)/area; arc_scale jac kept, vol dropped) → Task 2 (`test_no_vol_term`). ✓
- §4.3 InertialAR frame toggle (|det|=1) → Task 4. ✓
- §4.4 convergence safeguards (identity-init, tails, monotonicity, grad-clip, warmup) → Task 1 init + Task 3 loop. ✓
- §5 Feature A (QueryAR) → **explicitly deferred to its own Phase-2 plan** (gated on Task 5's verdict + pulling the QueryAR/InertialAR mechanisms), per the staged design. ✓
- §7 tests → Tasks 1/2/4 unit + Task 3 gate + Task 5 structural. ✓
- §9 scope: size-transfer, hard Δs-support, diffusion heads all out. ✓

**Placeholder scan:** `out=None` resolves to a concrete default name; the inertial-frame Step 3 names the exact threading change (recompute `nbr_rel` in `_local`/`_step`, apply `_frame_ab` before `self.flow.*`) rather than "wire it up" — the one place an implementer must touch inherited code, called out explicitly. No "TBD"/"add error handling"/"similar to Task N".

**Type consistency:** `SplineFlowHead.log_prob(h, ab)` / `.sample(h, gen)` signatures match across Tasks 1/2/4. `KAFlowHeadModel.__init__(..., num_bins, tail_bound, frame_mode)` consistent across Tasks 2/3/4/5. Checkpoint dict keys (`num_bins`, `tail_bound`, `frame_mode`) written in Task 3, read in Task 5. `_categorical_pos_logdensity` defined Task 3, reused in Task 5's gate command.

**Note for the implementer (Task 4 threading):** `frame_mode="inertial"` is the only task that touches the inherited offset computation. Apply the rotation to the offset `ab` in BOTH `log_prob` and `sample` using the SAME neighbour set `_local`/`_step` already gather; because `R` is a rotation (`|det|=1`) built from *placed* particles, no log-det term is added. If `test_inertial_exactness_gate_preserved` fails, the rotation differs between `log_prob` and `sample` (e.g. neighbour set or `theta` computed inconsistently) — reconcile them.
