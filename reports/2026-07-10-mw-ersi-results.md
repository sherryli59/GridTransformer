# mW eRSI flow-base implementation review

## Implemented seam

The mW adapter, 3-D monatomic traceable-EGNN training entry point, fixed-RK4
deployment wrapper, and one-shot-IS measurement harness are now present.  The
vendored factory does not register `egnn_traceable`; the implementation uses a
vendored `RiemannianFlowMatching` shell and replaces its velocity child before
the optimizer is created.  Thus OT coupling, the flow-matching loss, and ODE
likelihood plumbing remain vendored, while the velocity and divergence are the
required traceable backend.

## Gates

| Gate | Result | Evidence |
| --- | --- | --- |
| G-a, 3-D analytical divergence | PASS | `forward_and_divergence` equals a direct autograd trace at K=4, 6, and 7; see `logs-2026-07-10/mw_ersi_ga.out`. |
| G-b(i), generated-score round trip | PASS (unit) | The same fixed RK4 solver is used forward and reverse; the N=8 smoke test is within 1e-3. |
| G-b(ii), normalization | **FAIL** | A nonzero N=2 flow gives a six-dimensional midpoint integral of **14.18645**, rather than 1.0; see `logs-2026-07-10/mw_ersi_gb.out`. |
| G-b(iii), solver convergence | FAIL | The score still moves by roughly 1e-3 to 2e-3 from 40 to 80 time points in the diagnostic. |

The original plan's N=8 normalization test is not valid: its 24-dimensional
integral cannot be obtained from the proposed coarse grid, and conditioning on
a prefix requires the unavailable marginal density.  The implemented check is
instead an actual N=2 full-joint integral.  It passes for a zero velocity field,
which validates the quadrature code, and fails for a trained traceable flow.

## Decision

G-b blocks G-c/G-d.  The flow may not be used to form `e^{-beta U}/q_flow`
weights, because the score is not a normalized proposal density.  The most
likely fault domain is the periodic/minimum-image, kNN traceable vector field's
global flow map (local divergence is certified by G-a, but a local trace does
not prove global invertibility or normalization across periodic cut loci).
Resolving that defect needs a separate design/fix before any N=64 training run.
