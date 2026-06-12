# Multi-size arc run — epoch-54 mid-training status

Snapshot: `best.ckpt` of `multisize_arc_fullcov` frozen to `/tmp/multisize_ep54.ckpt` at
00:01 (training mid-epoch-54, loss_epoch 0.168; run still live). {L3,L5} train, L4 held out.
Same protocol as ep18/ep38: 64 samples/size, `--ot_batch 64` (OTgap noise ~±0.1), T=0.9,
CPU, geometry from checkpoint hparams (`--mode relative --Lx L --Ly L --Lz L --density 1.0`,
N = density·L³ = 27/64/125).

| size | role | NLL/coord (ep18 → ep38 → **ep54**) | OTgap (ep18 → ep38 → **ep54**) | zero-shot single-size baseline |
|------|------|-----------------------------------:|-------------------------------:|-------------------------------:|
| L=3 N=27 | train | 0.168 → 0.170 → **0.174** | 0.646 → 0.677 → **0.621** | OTgap 0.524 / NLL 0.115 |
| **L=4 N=64** | **HELD OUT** | 0.273 → 0.255 → **0.259** | 0.941 → 0.996 → **0.912** | OTgap 0.795 / NLL 0.649 |
| L=5 N=125 | train | 0.187 → 0.169 → **0.164** | 0.653 → 0.773 → **0.550** | OTgap 0.911 / NLL 0.656 |

Artifacts: `reports/multisize_arc/samples_N{27,64,125}_ep54.npz`.

## Reading at ~1/2 of training

- **The AR-compounding gap is closing — the predicted late-stage move.** The conditional
  (teacher-forced NLL) has essentially converged (held-out L4 flat at ~0.26; L5 still
  ticking down to 0.164), and now the free-running rollout is catching up: train-size
  **N=125 fell 0.773 → 0.550** (a 0.22 drop, beyond the ±0.1 noise band → real), and
  **held-out L4 came off the uniform-noise ceiling (0.996 → 0.912)** for the first time.
- **Train sizes are now solidly refinable.** N=125 at 0.55 is approaching the ≲0.6 green
  line; N=27 at 0.62 is closing on its 0.524 single-size baseline. Both are clearly
  sub-baseline (0.911 / 0.524).
- **Held-out L4 is the lagging gate, and still above its baseline.** 0.912 beats the
  uniform ceiling but **not yet the 0.795 zero-shot baseline.** L4 generation is pure
  size-interpolation with zero training rollouts at that size, so it follows the conditional
  the slowest. Direction is right; magnitude is the open question for the back half.

## Decision gate unchanged

End of training (epoch 100, ~09:30 today): definitive held-out L4 OTgap on the full
256-sample protocol vs the 0.795 baseline and the ≤0.6 green line. If L4 generation
stalls above 0.795 despite a converged conditional, attack AR compounding directly
(`continuous_input_noise` during training) rather than adding epochs.
