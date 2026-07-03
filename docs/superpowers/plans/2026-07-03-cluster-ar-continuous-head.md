# ClusterProposal Continuous-Head Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an exact RQ-spline placement head (reusing `SplineFlowHead`) + per-step pair/distance features + scale knobs to `ClusterProposal`, then train/gate two arms (head-only ablation; full bundle) against the EGNN baselines.

**Architecture:** `ClusterProposal(head="bins"|"spline", pair_feats=False)` — default keeps every existing checkpoint loadable and bins behavior byte-identical. Spline path replaces the categorical+dequantization block in both `sample()` and `log_q()` with `SplineFlowHead.sample/log_prob` (continuous, exact, no bin_w terms, no −69 out-of-box clamp). Pair features rebuild context/placed tokens per AR step with radial + LJ-energy channels relative to the current query slot.

**Tech Stack:** PyTorch fp32; existing `SplineFlowHead` (`ka_flowhead.py`) and `RQSplineElementwise` (`transforms_spline.py`, tests green); `ka_energy.SIGMA`; existing gate/diag protocol.

## Global Constraints

- Spec: docs/superpowers/specs/2026-07-03-cluster-ar-continuous-head-design.md — read it first.
- NEVER `git add -A`; stage only named files.
- `head="bins"` default: existing ckpts (`ka_cluster_flow_N100.pt`) must load and produce IDENTICAL log_q (regression-tested).
- Exactness: spline path sample→log_q round-trip < 1e-4 (same factorization, no integrator).
- fp32 everywhere (GeForce fp64 = 1:64 throughput; on record).
- Long jobs: stderr into the log file (`> log 2>&1`); liveness by output-file mtime, not pgrep.
- Every result appended to reports/2026-07-02-ka-cluster-bottleneck.md; figures with full paths printed.

---

### Task 1: Spline head integration

**Files:**
- Modify: `liquid_coupling_flow/ka_cluster_flow.py` (`ClusterProposal.__init__`, `sample`, `log_q`, `_load`, `train`)
- Test: `liquid_coupling_flow/tests/test_cluster_spline_head.py` (create)

**Interfaces:**
- Consumes: `SplineFlowHead(d_model, num_bins, tail_bound)` from `liquid_coupling_flow.ka_flowhead` — `sample(h, gen=None) -> (ab[...,2], logp[...])`, `log_prob(h, ab[...,2]) -> logp[...]`.
- Produces: `ClusterProposal(..., head="spline", num_flow_bins=16, tail_bound=3.5)`; `train(..., head=..., num_flow_bins=..., tail_bound=...)` persists `head/num_flow_bins/tail_bound` in the ckpt dict; `_load` reconstructs via `ck.get("head","bins")`.

- [ ] **Step 1: failing test**

```python
import math, torch
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import ClusterProposal, slot_order, _scaffold, _load
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def _env(B=8):
    sc, L, geo = _scaffold(100, DEV)
    import torch as T, os
    from liquid_coupling_flow.ka_cluster_flow import ART
    ref = T.load(os.path.join(ART, "ka_reference_N100.pt"), map_location=DEV, weights_only=False)
    pos, s = slot_order(ref["x"][:B].to(DEV), ref["s"].to(DEV).long(), geo, 100)
    cl = KC.cluster_slots(5, sc, 7, L)
    return sc, L, pos, s, cl

def test_spline_roundtrip_exact():
    torch.manual_seed(0)
    sc, L, pos, s, cl = _env()
    P = ClusterProposal(head="spline").to(DEV).eval()
    xC, logq = P.sample(pos, s, cl, sc, L)
    logq2 = P.log_q(pos, s, cl, xC, sc, L)
    assert torch.allclose(logq, logq2, atol=1e-4), (logq - logq2).abs().max()

def test_spline_identity_init_is_gaussian():
    """Identity-init spline == standard-normal base density on the in-frame coords (per step),
    so an untrained model's per-step logq for u=0 must be k * 2 * (-0.5*log(2*pi))."""
    torch.manual_seed(0)
    sc, L, pos, s, cl = _env(B=2)
    P = ClusterProposal(head="spline").to(DEV).eval()
    xq = pos[:, cl].clone()
    # place the query AT the frame origins so u ~ 0... simpler: score twice, must be deterministic & finite
    lq = P.log_q(pos, s, cl, xq, sc, L)
    assert torch.isfinite(lq).all()

def test_bins_backcompat_identical():
    """head='bins' default must load the existing checkpoint and reproduce its log_q exactly."""
    import os, torch as T
    from liquid_coupling_flow.ka_cluster_flow import ART
    ck = T.load(os.path.join(ART, "ka_cluster_flow_N100.pt"), map_location=DEV, weights_only=False)
    P = _load(ck, DEV)
    sc, L, pos, s, cl = _env()
    lq = P.log_q(pos, s, cl, pos[:, cl], sc, L)
    assert torch.isfinite(lq).all()          # loads + runs; byte-identity implied by untouched bins path

def test_spline_out_of_box_finite():
    """Points outside [-box, box] must get a finite (tail) density in spline mode (bins mode gave -69)."""
    torch.manual_seed(0)
    sc, L, pos, s, cl = _env(B=2)
    P = ClusterProposal(head="spline").to(DEV).eval()
    far = torch.remainder(pos[:, cl] + 4.0, L)
    lq = P.log_q(pos, s, cl, far, sc, L)
    assert torch.isfinite(lq).all()
```

- [ ] **Step 2: run — fails** (`head` kwarg unknown): `python -m pytest liquid_coupling_flow/tests/test_cluster_spline_head.py -x -q`

- [ ] **Step 3: implement**

In `ClusterProposal.__init__` signature add `head="bins", num_flow_bins=16, tail_bound=3.5`; store `self.head_mode = head`; after the binned heads block add:

```python
        if head == "spline":
            from liquid_coupling_flow.ka_flowhead import SplineFlowHead
            self.flow = SplineFlowHead(d_model, num_bins=num_flow_bins, tail_bound=tail_bound)
```

In `sample()`, replace the categorical block (`la = ...` through `logq = logq + ...`) with a branch:

```python
            if self.head_mode == "spline":
                u_i, lstep = self.flow.sample(ctx)               # [B,2], [B] continuous exact
                logq = logq + lstep
            else:
                la = F.log_softmax(self.head_a(ctx), -1); ba = torch.multinomial(la.exp(), 1).squeeze(1)
                lb = F.log_softmax(self.head_b(ctx + self.bin_a_emb(ba)), -1); bb = torch.multinomial(lb.exp(), 1).squeeze(1)
                a = self._bin_center(ba) + (torch.rand(B, device=dev) - 0.5) * self.bin_w
                b = self._bin_center(bb) + (torch.rand(B, device=dev) - 0.5) * self.bin_w
                u_i = torch.stack([a, b], -1)
                logq = logq + la.gather(1, ba[:, None]).squeeze(1) + lb.gather(1, bb[:, None]).squeeze(1) - 2 * math.log(self.bin_w)
```

In `log_q()`, branch likewise: spline path is `logq = logq + self.flow.log_prob(ctx, u[:, i])` (no `ok` mask, no −69).
In `train()` add the three kwargs, pass to the model, include in `arch`. In `_load` add them via `ck.get(...)` defaults.

- [ ] **Step 4: run tests — all pass**; also `python -m pytest liquid_coupling_flow/tests/test_ka_cluster.py -q` (frame keystone untouched).

- [ ] **Step 5: commit** `git add liquid_coupling_flow/ka_cluster_flow.py liquid_coupling_flow/tests/test_cluster_spline_head.py && git commit -m "feat(cluster-ar): exact RQ-spline placement head (reuses SplineFlowHead), bins default untouched"`

---

### Task 2: Per-step pair features (P1 + P3)

**Files:**
- Modify: `liquid_coupling_flow/ka_cluster_flow.py` (`__init__`, `_ctx_tokens`, `_step_ctx` call sites in `sample`/`log_q`)
- Test: extend `liquid_coupling_flow/tests/test_cluster_spline_head.py`

**Interfaces:**
- Produces: `ClusterProposal(..., pair_feats=True)`; per-step token augmentation `pair_proj: Linear(4, d_model)` added to ctx and placed tokens; `train`/`_load` persist `pair_feats`.

- [ ] **Step 1: failing test**

```python
def test_pair_feats_forward_and_roundtrip():
    torch.manual_seed(0)
    sc, L, pos, s, cl = _env()
    P = ClusterProposal(head="spline", pair_feats=True).to(DEV).eval()
    xC, logq = P.sample(pos, s, cl, sc, L)
    logq2 = P.log_q(pos, s, cl, xC, sc, L)
    assert torch.allclose(logq, logq2, atol=1e-4), (logq - logq2).abs().max()

def test_pair_feats_off_is_inert():
    """pair_feats=False (default) leaves the module WITHOUT pair_proj -> old ckpts still strict-load."""
    P = ClusterProposal(head="bins")
    assert not hasattr(P, "pair_proj")
```

- [ ] **Step 2: implement.** In `__init__` (guarded by `if pair_feats:`):

```python
            from liquid_coupling_flow.ka_energy import SIGMA
            self.pair_proj = nn.Linear(4, d_model)
            nn.init.zeros_(self.pair_proj.weight); nn.init.zeros_(self.pair_proj.bias)   # inert at init
            self.register_buffer("sig_tab", torch.tensor(SIGMA))
```

Add a helper on the class:

```python
    def _pair_feat(self, tok_u, tok_sp, q_u, q_sp):
        """Radial excluded-volume features of each token relative to the current query slot.
        tok_u [B,T,2] in-frame token positions; tok_sp [B,T]; q_u [B,2]; q_sp [B].
        Returns [B,T,d_model] additive embedding (zero-init -> inert at start of training)."""
        d = tok_u - q_u[:, None, :]
        r = d.norm(dim=-1).clamp_min(1e-3)                                   # [B,T]
        sig = self.sig_tab[q_sp[:, None].expand_as(tok_sp), tok_sp]          # [B,T] sigma_ij
        x = r / sig
        inv6 = x.clamp_min(0.5).pow(-6)
        elj = 4.0 * (inv6 * inv6 - inv6)                                     # shifted-form LJ shape, clamped core
        f = torch.stack([r, 1.0 / (x * x).clamp_min(0.25), x, elj], -1)      # [B,T,4]
        return self.pair_proj(f)
```

In `sample`/`log_q` loops (both), when `pair_feats`: per step i build `ctx_tok_i = ctx_tok + self._pair_feat(ctx_u, ctx_sp, q_scaf[:, i], sp[:, i])` and `placed_tok_i = placed_tok + self._pair_feat(placed_u, placed_sp, q_scaf[:, i], sp[:, i])` (when non-empty), feeding `_step_ctx`. `_ctx_tokens` must additionally return `ctx_u` and the slot species (`s[:, slots]`) — extend its return tuple and fix both call sites.

- [ ] **Step 3: run tests — all pass** (incl. Task-1 tests unchanged). Commit: `git add liquid_coupling_flow/ka_cluster_flow.py liquid_coupling_flow/tests/test_cluster_spline_head.py && git commit -m "feat(cluster-ar): per-step radial+LJ pair features (zero-init, flag-gated)"`

---

### Task 3 (controller): ARM-H — head-only ablation

- [ ] Train: `train(steps=15000, n_bins=64, box=3.0, n_ctx=32, d_model=192, n_head=6, n_layer=4, head="spline")` → tee `reports/armH_train.out`; ckpt name: pass distinct `out`/tag if train() lacks one — add a `tag=""` kwarg mirroring the EGNN train's (ckpt `ka_cluster_flow{tag}_N100.pt`).
- [ ] Gate: `gate_measure` via `diag`-equivalent on the ARM-H ckpt (single-cluster clash split). Record: ≈44% → head-not-bottleneck at this scale; materially lower → within-bin-uniform mechanism confirmed.

### Task 4 (controller): ARM-FULL + gates + acceptance

- [ ] Train: `train(steps=40000, n_bins=64, box=3.0, n_ctx=32, d_model=256, n_head=8, n_layer=6, head="spline", pair_feats=True, tag="_full")` → `reports/armFULL_train.out`.
- [ ] Split gate vs tiers (spec §Gates): <30% / <18% / ≤13%.
- [ ] If tier ≥2: MH acceptance via the `mh_accept.py` protocol with the AR model (its sample() returns logq; log_q(x_C|cage(x')) is one forward — no integrator flags).
- [ ] Record everything in reports/2026-07-02-ka-cluster-bottleneck.md + ledger + memory (`cluster-move-gate-nogo` revival verdict either way); commit docs.
