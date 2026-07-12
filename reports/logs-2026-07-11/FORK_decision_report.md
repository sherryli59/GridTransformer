# Fork after the deterministic-corrector refutation — options weighed (2026-07-11)

**Decision:** how to make the large-K (K≈12) KA cavity block move propose *clean, near-equilibrium*
rearrangements for a basin-crossing jump, now that every position-space corrector on the half-cage AR base
is refuted (see `DIAGNOSIS_deterministic_corrector.md`). The root defect is that the AR generates block
positions with intra-block **half-cage blindness → wrong relative structure**; relaxation fixes overlaps,
not structure (best deterministic floor +6.9/particle, a strained non-basin; a real basin is +0.9).

## Binding criteria (what any option is judged against)

1. **Exact log_q** — the MTM/IS rigor depends on it (hard constraint, never relax).
2. **Correct relative structure** — the actual thing that broke; a proposal must place the K particles in a
   thermally-typical arrangement, not just overlap-free.
3. **No silent collapse** — a full-cage conditional iterated alone collapses (`full-cage-lever-needs-energy`);
   anything full-cage needs an energy guard.
4. **Cost** — new architecture + a multi-hour training run vs. reuse of existing, validated infrastructure.
5. **Research taste** — the user has resisted "just leaning on classic enhanced sampling / replicating the
   paper" (`transfer-needs-an-invariance`), and values exactness + size-transfer as the learned model's edge.

## Prior verdicts that condition this decision (campaign memory)

- `full-cage-lever-needs-energy`: the **non-causal full-cage conditional is CORRECT** — it lifts g_BB peak
  from 1.25 (half-cage) to 2.00 (data 2.40) and cuts spurious contacts 0.418→0.162. **But iterating it alone
  (pure Gibbs) COLLAPSES.** => full-cage placement is right; it just needs a guard.
- `coupling-flow-arm-fails-uniform-base` + `cluster-move-gate-nogo`: full-cage **flows** from a uniform base
  repeatedly hit an **expressiveness wall** on hard-core exclusion (KACouplingFlow plateaued; EGNN-big 13.3%).
- `traceable-egnn` / `tf2boltz-ersi`: an **exact-divergence full-cage EGNN flow** exists and is validated
  (analytic central-force divergence, periodic bugs fixed) — real reusable machinery for exact log_q.
- `scale-crossover-verdict` + `collective-move-selection-wall`: **no learned collective move has beaten
  PT/swap on depth** through N=576; the learned stack's proven wins are **exact zero-shot size-transfer** and
  **in-stack acceleration / SMC head-start** (`glass-smc-corrector-headstart`), NOT out-crossing classic MC.

The honest prior: a learned large-block move that beats classic sampling *on basin-crossing depth* is
historically unlikely; the learned proposal's durable value is being **exact, size-transferable, and a
warm-start**. This should temper how much to invest in any option whose only payoff is depth.

---

## Option A — Full-cage position *generator* + energy guard

**Mechanism:** a non-causal generator (each block particle conditioned on ALL others) or an exact-divergence
full-cage **flow** (EGNN-traceable) over the K positions, trained on cavity data, deployed as the block
proposal with an SMC/MH energy guard.

**Pros**
- Attacks the root directly; exactly the "clean low-energy alternative arrangements" the original goal named.
- Strong precedent that full-cage placement is *correct* (`full-cage-lever-needs-energy`).
- Exact-divergence EGNN flow infrastructure already exists (`traceable-egnn`) → exact log_q is reachable.

**Cons / risks**
- A non-causal *conditional* has **no tractable joint log_q** — exactness must come from either a flow
  (→ the expressiveness wall) or the guard (→ blurs into Option C's SMC).
- Full-cage **flows** have a track record of under-resolving hard-core structure here
  (`coupling-flow-arm-fails-uniform-base`). Building the base from AR-samples rather than uniform may dodge
  this, but that's unproven.
- Highest cost: new architecture + multi-hour training + a new exactness audit.

**Verdict:** highest upside, highest cost/risk; the exactness route (flow vs guard) must be nailed before
building, or it silently becomes Option C.

## Option B — Full-cage position *denoiser* on the (proven-fine) AR species

**Mechanism:** keep the AR base's species (measured fine: row C = +0.9/particle); learn a full-cage position
denoiser / probability-flow field that refines only the K positions to correct relative structure,
energy-guarded. A probability-flow ODE gives exact log_q via the divergence integral (same machinery as A's
flow, smaller conditioning problem).

**Pros**
- Reuses the one AR component that *works* (species), shrinking the learning problem to positions-given-species.
- "Contractive denoiser" is the exact path memory flagged alongside the guard (`full-cage-lever-needs-energy`).
- A probability-flow ODE keeps exact log_q without the guard having to supply it.
- Smaller/cheaper than A; most of A's upside at less cost.

**Cons / risks**
- Still a training run; still full-cage → needs the guard against collapse if run as a denoiser rather than a
  strict flow.
- Same hard-core expressiveness question as A (a denoiser must build the `r⁻¹²` correlation hole).
- Depends on the AR species staying good *conditioned on the new positions* — C fixed positions to data; a
  self-consistency check is needed.

**Verdict:** best cost/upside ratio; the most-reuse, root-attacking option. Recommended, gated by a cheap
pre-test (below).

## Option C — Annealed SMC mutation (no new model)

**Mechanism:** make the large block move an annealed, energy-guarded SMC mutation over the existing base.

**Pros**
- Cheapest (no training); exact by construction (SMC weights); robust — it's what already works at K=4.
- The learned proposal already gives a durable SMC head-start in the glass (`glass-smc-corrector-headstart`).

**Cons / risks**
- Directly the "classic enhanced sampling" the user pushed back on.
- Crossing a K=12 glassy basin by annealing may cost many sweeps — the very expense we're trying to remove;
  and `scale-crossover-verdict` says annealed SMC doesn't out-depth PT here.
- Produces sampling, not "a model that proposes clean large blocks."

**Verdict:** the safety net / fallback, not the headline. Keep as the guard *inside* A/B regardless.

## Option D — Rethink the K=12 target

**Mechanism:** treat the +6.9 strained floor as evidence that one-shot clean K=12 jumps are intrinsically
hard for this base; step back to moderate-K (K≈6–8) + guard, or re-scope what the block move must achieve
for the PTS measurement.

**Pros**
- Cheap; grounded in strong prior negatives (`collective-move-selection-wall` CLOSED; no crossover to N=576).
- For the actual PTS deliverable (Berthier-2016 overlap), independent equilibrium cavities can come from the
  learned generator's **exact size-transfer** strength rather than from out-crossing MC — plays to the
  model's proven edge, not its repeatedly-refuted one.

**Cons / risks**
- Reads as retreat from the stated goal; may re-confirm known walls without new insight.
- Moderate-K is a middle ground that may inherit half-cage structure error at smaller magnitude.

**Verdict:** valuable as a **framing correction** and as a cheap de-risking gate before any big build, not as
a standalone build.

---

## Recommendation

**Build B (full-cage position denoiser/flow on AR species, energy-guarded) — but gate it behind a cheap
pre-test drawn from D**, because a training run is only worth it if full-cage placement actually yields
*accepted* clean basin crossings here.

Concretely, staged:

1. **Cheap gate (hours, no training):** use the existing **non-causal full-cage conditional** (`ka3d`
   analog of `ka_noncausal`) as an energy-guarded block proposal — one Gibbs/heat-bath sweep over the K
   positions given AR species + full cage, then the exact MTM/SMC accept. Measure: does the guarded full-cage
   proposal land near-basin (≲ +1–2/particle) and produce *accepted* K=12 jumps that change the arrangement?
   - If **yes** → the structure fix is real; commit to training the exact position flow (B).
   - If **no** → the guard can't rescue it cheaply → fall to D (re-scope to the model's exact-transfer edge
     for PTS, with C as the sampling workhorse).

2. **If the gate passes:** train the full-cage position **probability-flow** on AR species (exact log_q via
   divergence; reuse `traceable-egnn` machinery), audit exactness (round-trip + divergence vs finite-diff),
   then run the basin-crossing + PTS gates.

This keeps every hard constraint (exact log_q throughout), reuses the most validated infrastructure (AR
species, EGNN traceable divergence, the SMC guard), attacks the actual root (relative structure), and — most
importantly — **spends the training budget only after a cheap experiment confirms full-cage-with-guard clears
the bar the deterministic corrector could not.**
