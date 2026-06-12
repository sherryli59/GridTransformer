# GraphormerAR Architecture Summary & Proposed Geometric Alternatives

**Date**: 2026-06-12  
**Branch**: `hilbert-arc-repr`  
**Scope**: `grid_transformer/models/transformer.py` (`GraphormerAR`) and its geometric
submodules (`EdgeBias`, `RBFEdgeBias`, `DeepIDABias`, `PeriodicIDA`, `CurveRailAttention`),
the MDN head (`training/ar.py`), and the arc-length representation.  
**Companion docs**: `reports/hilbert-arc-repr-plan.md` (arc repr spec),
`CODE_REVIEW_hilbert-arc-repr_2026-06-10.md` (state of the branch),
`CURVE_RAIL_REFACTOR_TODO.md` (rail config consolidation).

**Goal context**: a size-transferable AR transformer over Hilbert-sorted LJ particles whose
samples have small OT gap to the Boltzmann distribution (refinable by the EGNN flow at
`/mnt/ssd/flow_matching/learndiffeq-main`). In-distribution N=27 is essentially solved
(OTgap 0.12); zero-shot size transfer is not (OTgap > 1 at N=64/125). The bottleneck named
by the K=1 study is the learned *conditional*: the marginal features transfer, the local
next-particle conditional does not.

---

## Part I — Current architecture, with equations

### 1. Probabilistic setup

Particles $r_1,\dots,r_N \in [0,L)^3$ are sorted by their Hilbert code
$c_i = \mathcal{H}(\lfloor r_i / \ell \rfloor)$, $\ell = L/R$ (cell size, resolution $R$).
The model is an autoregressive factorization over the sorted order:

$$
p_\theta(r_{1:N}) \;=\; \prod_{t=1}^{N} p_\theta\big(\delta_t \,\big|\, \delta_{<t},\, r_{<t},\, \text{rail}_t\big)
\cdot \Big|\det \tfrac{\partial \delta}{\partial r}\Big|
$$

where the per-step target $\delta_t$ is, depending on flags:

| Mode | Target $\delta_t$ | Dim |
|---|---|---|
| `discrete` / `binned_discrete` | quantized cell/bin token | — (categorical over $K$) |
| continuous Δxyz (default head) | $\delta_t = r_t - r_{t-1}$ (min-image if `torus`) | 3 |
| `arc_repr` | $\delta_t = \big(\Delta s_t,\ f_{t,x},\ f_{t,y},\ f_{t,z}\big)$ | 4 |

with the arc-length representation (see `hilbert-arc-repr-plan.md`):

$$
\Delta s_t = \frac{c_t - c_{t-1}}{X}, \qquad X = \frac{R^3}{N}, \qquad
f_t = r_t - \underbrace{\big(\mathcal{H}^{-1}(c_t) + \tfrac12\big)\,\ell}_{\text{cell center}},
\quad |f| < \tfrac{\ell}{2}.
$$

By construction $\mathbb{E}[\Delta s] \approx 1$ for any $N, L$ at fixed density — the
scale-invariance that motivates the branch. (Known open item: the saved log-prob lacks the
arc→position Jacobian; review finding 3.6.)

### 2. Input embedding (token stream)

Position $t$'s residual-stream input $x_t^{(0)} \in \mathbb{R}^{d}$ ($d=512$ in the current
runs) is assembled additively:

$$
x_t^{(0)} =
\underbrace{\begin{cases}
E_{\text{tok}}[\text{SOS}] & t = 0\\[2pt]
W_{\text{in}}\,\delta_{t-1} & t \ge 1,\ \texttt{continuous\_input}\\[2pt]
E_{\text{tok}}[k_{t-1}] & \text{otherwise (discrete token)}
\end{cases}}_{\text{content}}
\;+\;
\underbrace{E_{\text{pos}}[t]}_{\text{opt.}}
\;+\;
\underbrace{W_{\rho}\,\rho}_{\text{opt. density}}
\;+\;
\underbrace{E_{\text{axis}}[t \bmod 3]}_{\text{opt. factorized}}
$$

The current best configs use `continuous_input=1`: the content is the *continuous* previous
displacement $W_{\text{in}}\delta_{t-1}$ ($W_{\text{in}} \in \mathbb{R}^{d\times 4}$ for
arc_repr), not the quantized token — this fixed the discrete bin-crossing pathology. Note
that **the model never sees its own absolute position in the residual stream**; absolute
geometry enters only through the attention bias (next section) and the rail.

### 3. Hilbert curve rail (CurveRailAttention) — applied **once**, pre-blocks

Each token $t$ gets $K$ deterministic look-ahead waypoints
$w_{t,1},\dots,w_{t,K} \in \mathbb{R}^3$ — min-image vectors from the particle to future
Hilbert cell centers (fixed-template: $\text{frac} \in \text{linspace}(-1,1,K)$ around
$j\cdot X$; K=1 lookahead: $\mathcal{H}^{-1}((j{+}1)X)$). Waypoints are featurized
geometrically and **ordered**:

$$
\phi_{t,k} = \text{MLP}\Big(\big[\underbrace{e^{-\gamma (\|w_{t,k}\| - \mu_1)^2},\dots,e^{-\gamma (\|w_{t,k}\| - \mu_{32})^2}}_{\text{RBF}(d)},\ \underbrace{w_{t,k}/\|w_{t,k}\|}_{\hat u}\big]\Big) + E_{\text{ord}}[k]
$$

then per-token cross-attention (no cross-token mixing, hence causal-safe):

$$
x_t \leftarrow x_t + W_O \sum_k \text{softmax}_k\!\Big(\tfrac{(W_Q \text{LN}(x_t))^\top (W_K \phi_{t,k})}{\sqrt{d_h}}\Big)\, W_V \phi_{t,k}
$$

**Key structural fact**: this runs exactly once, *before* the transformer blocks. The rail
signal must survive 4 blocks of mixing to influence the head.

### 4. Geometric attention bias (EdgeBias family) — shared across **all** layers

A single bias tensor $b \in \mathbb{R}^{B\times H\times T\times T}$ is computed from the
teacher-forced (train) or generated (sample) absolute coordinates and **reused by every
layer**. With min-image pair vectors $d_{ij} = \text{wrap}(r_i - r_j)$, $d = \|d_{ij}\|$:

**EdgeBias (default)** — log-binned distance embedding + direction MLP:

$$
b^h_{ij} = E_{\text{bin}}\Big[\Big\lfloor \tfrac{\log(1+d)}{\log(1+d_{\max})}(B_{\text{bins}}-1) \Big\rfloor\Big]_h
\;+\; \text{MLP}_{\text{dir}}\big(\hat d_{ij}\big)_h , \qquad B_{\text{bins}}=32,\ d_{\max}=4.
$$

**RBFEdgeBias** — smooth: $b^h_{ij} = \text{MLP}\big(\text{RBF}_{64}(d)\big)_h$.  
**DeepIDABias** — $b^h_{ij} = \text{MLP}_{3\text{-layer}}(d)_h$ (non-monotonic in $d$).  
**PeriodicIDA** (optional pre-block layer, off in current runs) — attention logits are
*inverse distances between learned coordinate offsets*:
$a^h_{ij} = \text{softplus}(s_h)\,\big/\big(\sqrt{3}\,\|\,(q^h_i + r_i) - (k^h_j + r_j)\,\|\big)$,
with vector values $v \in \mathbb{R}^3$ per head.

### 5. Transformer blocks (pre-LN, biased causal attention)

For layers $l = 1..L$ ($L=4$, $H=4$ heads in current runs):

$$
A^{(l)} = \text{softmax}\Big( \frac{Q^{(l)} K^{(l)\top}}{\sqrt{d_h}} + b + M_{\text{causal}} \Big),
\qquad
x \leftarrow x + W_O^{(l)} A^{(l)} V^{(l)}, \qquad
x \leftarrow x + \text{MLP}^{(l)}(\text{LN}(x))
$$

with $Q,K,V = W_{QKV}^{(l)}\,\text{LN}(x)$ and $\text{MLP} = W_2\,\text{GELU}(W_1 \cdot)$,
hidden $4d$. No RoPE (the `use_rope` flag is accepted and discarded). KV-cached generation
is exact because the bias row for the new query is recomputed each step
(`EdgeBias.bias_row`, $O(t)$).

### 6. MDN head and loss

$$
h_t = \text{LN}_f(x_t^{(L)}), \qquad
p(\delta_t \mid h_t) = \sum_{m=1}^{M} \pi_m(h_t)\; \mathcal{N}\big(\delta_t;\ \mu_m(h_t),\ \Sigma_m(h_t)\big)
$$

$$
\pi = \text{softmax}(W_\pi h), \quad \mu = W_\mu h \in \mathbb{R}^{M\times D}, \quad
\Sigma_m = \begin{cases}
\text{diag}\big(e^{2\,\text{clamp}(W_\sigma h, -10, 5)}\big) & \text{diagonal}\\
L_m L_m^\top,\ L_m = \text{tril}(W_L h) & \texttt{full\_covariance}
\end{cases}
$$

Current runs: $M=64$, full covariance, $D = 4$ (arc_repr). Loss is mean per-coordinate NLL,
optionally plus an importance-weight variance regularizer toward the Boltzmann target:

$$
\mathcal{L} = \frac{1}{B}\sum_b \frac{\text{NLL}_b}{n_b}
\;+\; \lambda_{\text{var}}\, \widehat{\text{Var}}\big(\log w\big), \qquad
\log w_b = -\tfrac{E_b}{k_BT} + \text{NLL}_b .
$$

### 7. Where the geometry actually lives (and doesn't)

```
δ_{t-1} ──W_in──► x ──[rail x-attn ×1]──► block1 ──► … ──► block4 ──► MDN(δ_t)
                              ▲                ▲ … ▲
                   waypoints (RBF+dir)     b_ij (RBF/bin of pair dist+dir)
                   Hilbert geometry        Euclidean geometry, SAME b all layers
```

Three observations that motivate Part II:

1. **Geometry is logit-only.** EdgeBias modulates *where* a token attends, but the value
   vectors $V$ carry no relative geometry — token $j$'s value is the same regardless of
   where $j$ sits relative to $i$. The network must reconstruct "neighbor at distance 1.1σ
   in direction $\hat u$" from the *content* stream (a chain of past deltas), which is
   exactly the brittle, order-dependent computation that fails to transfer.
2. **Euclidean and Hilbert structure never meet in attention.** The bias depends on
   $\|r_i - r_j\|$ only; the curve separation $|s_i - s_j|$ (equivalently $|i-j|$, or
   $c_i - c_j$) is invisible to it. But the *defining* geometric correlation of this data
   is the joint law of (Euclidean distance, curve separation): nearby-in-space pairs are
   usually nearby-on-curve, **except at Hilbert jumps**, which is precisely where samples
   go wrong (cf. `analyze_hilbert_lj_jumps.py`). The rail addresses the *forward* half of
   this (where the curve goes next) but nothing tells attention "this previously placed
   particle is spatially adjacent but curve-distant — it constrains me strongly."
3. **Direction features are in the global frame.** $\text{MLP}_{\text{dir}}(\hat d_{ij})$
   and the rail's $\hat u$ are absolute-frame vectors. The LJ energy is rotation-invariant;
   the Hilbert curve is not, but its local orientation is *known* at every step. Global-frame
   directions force the model to memorize the curve's orientation pattern per box size —
   one plausible contributor to the conditional not transferring.

---

## Part II — Proposed alternatives

Ordered by (expected impact on the geometric-correlation problem) × (implementation cost).
All keep the AR + Hilbert-order + MDN scaffold; they change how geometry enters.

### A. Joint (Euclidean × arc-length) edge bias — *cheapest, most targeted*

Replace the distance-only bias with a bias over the **pair** of Euclidean distance and
curve separation, the two coordinates whose correlation defines the Hilbert/LJ interaction:

$$
b^h_{ij} = \text{MLP}\Big( \big[\, \text{RBF}(\|d_{ij}\|)\ \big\|\ \text{RBF}_{\log}(\Delta s_{ij})\,\big] \Big)_h,
\qquad \Delta s_{ij} = \frac{c_i - c_j}{X} \;(\,\approx i - j \text{ for local steps}\,)
$$

using log-spaced RBF centers for $\Delta s$ (local steps $\approx 1..4$ need resolution;
jumps span decades). The MLP sees the *joint*, so it can express exactly the statement the
model currently cannot: *"spatially close ($d<1.5\sigma$) but curve-far ($\Delta s \gg 1$) ⇒
strong constraint, attend hard"* vs *"curve-adjacent ⇒ mostly redundant with my input
delta."* Using normalized $\Delta s$ (not raw token offset $i{-}j$) keeps it size-invariant
by the same argument as arc_repr.

- Cost: ~30 lines (a `JointEdgeBias` next to `RBFEdgeBias`); needs per-token Hilbert codes
  at attention time, which arc_repr caches already carry, and which the sampler already
  maintains (`prev_hilbert_codes`).
- KV-cache compatible: `bias_row` generalizes unchanged.
- Ablation reading: if this alone closes part of the held-out-size OTgap, the diagnosis
  (curve-geometry correlation is the missing signal) is confirmed cheaply.

### B. Geometric value path (vector messages), EGNN/PaiNN-flavored — *attack on "logit-only geometry"*

Let attention *transport geometry*, not just gate on it. Augment each block's value with a
pairwise geometric message:

$$
y_i = \sum_j A^h_{ij} \Big( V^h_j + W_g\,g(d_{ij}) \Big), \qquad
g(d_{ij}) = \big[\,\text{RBF}(\|d_{ij}\|)\ \big\|\ \hat d_{ij}\,\big]
$$

or the equivariant version that keeps a separate vector channel
$\vec v_i \in \mathbb{R}^{3\times d_v}$ updated as
$\vec v_i \leftarrow \vec v_i + \sum_j A_{ij}\,\big(\alpha_{ij} \vec v_j + \beta_{ij}\, \hat d_{ij}\big)$
with scalar gates $\alpha,\beta$ from RBF features (PaiNN-style), and invariant readout
$\|\vec v_i\|$ into the residual stream. Then the head can predict $\mu_m$ **in the local
neighbor basis**: $\mu_m = \sum_c \kappa_{mc}(h_i)\, \vec v_{i,c}$ — displacements expressed
as combinations of actual neighbor directions, which is the natural parameterization of
"place next particle in the pocket between these three neighbors" and is size-agnostic.

- This is the single biggest expressivity upgrade: the current model can *know* a neighbor
  is at 1.1σ but cannot *copy its direction* into the output without reconstructing it
  arithmetically from the delta chain.
- Cost: moderate — a new block type; keep the same `b` logits so it's additive to A.
  Per-pair value tensors are $O(T^2 d)$; with N≤125, fine.

### C. Rail in every layer (or rail tokens as keys/values) — *fix one-shot injection*

Two variants, same idea — the curve scaffold should be consultable *after* the model has
mixed in neighbor information, not only before block 1:

**C1 (per-layer rail)**: apply `CurveRailAttention` (shared or per-layer weights) after each
block's self-attention: $x \leftarrow x + \text{RailXAttn}^{(l)}(x, w)$. ~5 lines in
`forward`/`forward_step`; weight sharing keeps parameter count flat.

**C2 (rail tokens in the main attention)**: append the $K$ waypoints of token $t$ as extra
keys/values visible only to query $t$ (block-diagonal mask), with their EdgeBias computed
from the *actual* particle→waypoint vectors:

$$
A_t = \text{softmax}\big( q_t [K_{\text{tok},\le t};\ K_{\text{rail},t}]^\top / \sqrt{d_h} + [b_{t,\le t};\ b^{\text{rail}}_{t,1..K}] \big)
$$

C2 lets heads *trade off* curve guidance against neighbor constraints inside one softmax —
precisely the competition at Hilbert jump cells, where the right answer is "trust the
neighbors, the curve teleported." That trade-off is currently impossible: rail and edge bias
live in different mechanisms that never compete for the same attention mass.

### D. Local-frame (curve-gauge) direction features — *size/rotation transfer of the conditional*

Replace global-frame $\hat d_{ij}$ and waypoint directions with coordinates in a frame
defined by the local curve geometry. At step $t$, define the tangent frame from the known
scaffold: $\hat e_1 \propto w_{t,1}$ (direction to next expected cell),
$\hat e_2 \propto w_{t,2} - (\hat e_1^\top w_{t,2})\hat e_1$, $\hat e_3 = \hat e_1 \times \hat e_2$,
and feed $R_t^\top \hat d_{ij}$, $R_t^\top w_{t,k}$ with $R_t = [\hat e_1 \hat e_2 \hat e_3]$.
The conditional then sees "neighbor ahead-left along the curve" instead of "neighbor at
$(+x,-y)$", which is the same event in every octant of every box size. The MDN should
likewise predict $f_t$ in this frame (rotate back at decode). This directly targets the
K=1 finding that the audit-passing rail still fails zero-shot: the *features* matched
marginally but the absolute-frame conditional had to extrapolate.

- Caveat: the frame is deterministic from the scaffold, so train/sample parity is safe;
  but degenerate cases (collinear $w_1, w_2$ at some template positions) need a fallback
  (e.g., fixed canonical frame, flagged by an indicator feature).

### E. Two-track curve/space architecture — *bigger rewrite, cleanest factorization*

Make the implicit structure explicit: a **curve track** (1D transformer over the arc-length
sequence: inputs $\Delta s$, jump indicators, rail) and a **space track** (kNN geometric
attention over already-placed particles within cutoff $r_c$, equations as in B, no curve
information), exchanging information by cross-attention each layer:

$$
x^{\text{curve}} \leftrightarrow x^{\text{space}} \quad\text{(per-layer bidirectional x-attn)}
$$

The space track is manifestly size-transferable (local, distance-kernel, permutation-equiv
over neighbors); the curve track is size-invariant by arc normalization. Only the *coupling*
is learned. This is the architecture the diagnosis has been converging on (turn-token /
give-the-curve design ladder D1→D3), but it is a new model class — prototype only if A–D
plateau.

### F. Sharper distance resolution where LJ is stiff — *small fix, do alongside A*

The default `EdgeBias` log-bins put few bins near contact ($d \approx \sigma$) where
$\partial E_{\text{LJ}}/\partial r$ is enormous; bin quantization there injects exactly the
noise the MDN must average over. Either (i) switch default to `RBFEdgeBias` with centers
concentrated near $\sigma$ (e.g., centers at $\text{quantile}(d_{\text{train}})$), or
(ii) Bessel/sinc radial basis $ \sin(n\pi d/r_c)/d $ with polynomial cutoff envelope
(DimeNet-style), which is smooth, decays correctly, and is standard for LJ-like potentials.

### G. kNN-sparse attention with shared kernels — *throughput + inductive bias, enables N≫27*

Restrict each query to its $k$ nearest placed particles (min-image) plus the rail:
$A_{ij} = 0$ unless $j \in \text{kNN}(i)$. With B's value path this turns each layer into a
local geometric message pass; size transfer then can't fail due to attention-mass dilution
over longer prefixes (at N=125 the prefix is 5× longer but the *relevant* set is still ~12
neighbors). Cheap to test as a hard mask added to `b` before softmax.

---

## Part III — Recommendation

| Priority | Proposal | Why first | Risk |
|---|---|---|---|
| 1 | **A** joint (d, Δs) bias | Directly encodes the curve×geometry correlation; ~30 lines; clean ablation | low |
| 2 | **F** RBF/Bessel default + near-contact centers | Removes known resolution noise; trivially combinable | low |
| 3 | **C1/C2** rail every layer | One-shot injection is an obvious bottleneck; C2 gives jump-cell trade-off | low–med |
| 4 | **D** curve-gauge local frames | Targets the transfer failure mechanism identified by K=1 study | med (frame degeneracy) |
| 5 | **B** geometric value path + neighbor-basis MDN | Biggest expressivity win; needed if logit-only geometry is the ceiling | med |
| 6 | **G** kNN sparsity | Mainly matters for N≥125 and multi-size training | low |
| 7 | **E** two-track | Re-architecture; only if 1–5 plateau | high |

Suggested experiment sequence (consistent with the K=1 playbook and arc_repr validation
criteria in `hilbert-arc-repr-plan.md`): train each variant on N=27 first and require
NLL/g(r)-MAE parity with the current baseline (NLL −13.3, g(r) MAE 0.211, OTgap 0.12);
then the real test — multi-size train {L3, L5}, held-out L4, target OTgap ≤ 0.3. A and C
are checkpoint-compatible ablations of existing modules; D changes the data contract
(frame must be cached or recomputed identically at sample time — thread it through the
`CurveRailConfig` consolidation in `CURVE_RAIL_REFACTOR_TODO.md` rather than as more loose
scalars).
