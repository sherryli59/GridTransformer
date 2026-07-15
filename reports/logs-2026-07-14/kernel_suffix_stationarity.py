"""Suffix lambda=0 stationarity smoke (Task 5, spec 2026-07-14).

At lam=0 the bridge target pi_0 == q0 (canonical-lift). One suffix_move proposes the
tail from q0's own exact AR suffix conditional and, for canonical-storage proposals,
accepts with prob min(1, exp(0))=1 while rejecting noncanonical-storage proposals (a
self-loop). Detailed balance between two canonical states x,y differing only in the
suffix is exact:  pi(x)=P_prefix*q_suf(x_suf|prefix), accept(x->y)=min(1,1)=1, so
pi(x)T(x->y)=pi(y)T(y->x). Hence E_q0[log q0] is invariant.

Test: 64 walkers from q0, 200 suffix_moves at lam=0; the evolved batch's mean canonical
log_prob must sit within 3*SEM of an independent fresh q0 batch (both estimate
E_q0[log q0]). Prints both numbers -- paste into the commit message.

Run: CUDA_VISIBLE_DEVICES="" python reports/logs-2026-07-14/kernel_suffix_stationarity.py
"""
import math
import torch

from liquid_coupling_flow.mw.mw_generator_v4 import MWGlobalAR
from liquid_coupling_flow.mw.mw_energy import mw_energy
from liquid_coupling_flow.mw.mw_kernels import recanonicalize, suffix_move

N, L = 27, 3.9
Bsz = 64


def q0_tiny():
    torch.manual_seed(0)
    return MWGlobalAR(d_model=32, n_layers=1, n_heads=2, n_mix=4, rail_k=4).eval()


def main():
    q0 = q0_tiny()
    with torch.no_grad():
        # evolved chain: start from a q0 batch, run 200 suffix moves at lam=0
        x = recanonicalize(q0.sample(Bsz, N, L, gen=torch.Generator().manual_seed(11))[0], L)
        U = mw_energy(x, L)
        gen = torch.Generator().manual_seed(12)
        acc_hist = []
        for _ in range(200):
            x, U, st = suffix_move(x, U, q0, 1, 4, lam=0.0, beta=1.0, L=L, gen=gen)
            acc_hist.append(st["acc"])
        lp_evolved = q0.log_prob(x, L)

        # independent fresh q0 batch (same estimator E_q0[log q0])
        xf = recanonicalize(q0.sample(Bsz, N, L, gen=torch.Generator().manual_seed(13))[0], L)
        lp_fresh = q0.log_prob(xf, L)

    m_ev, m_fr = float(lp_evolved.mean()), float(lp_fresh.mean())
    sem = math.sqrt(float(lp_evolved.var(unbiased=True)) / Bsz
                    + float(lp_fresh.var(unbiased=True)) / Bsz)
    dz = abs(m_ev - m_fr) / max(sem, 1e-12)
    print(f"suffix lam=0 stationarity (E_q0[log q0], N={N}, B={Bsz}, 200 moves):")
    print(f"  mean log_prob evolved = {m_ev:.4f}")
    print(f"  mean log_prob fresh   = {m_fr:.4f}")
    print(f"  |diff| = {abs(m_ev - m_fr):.4f}   combined SEM = {sem:.4f}   z = {dz:.2f}")
    print(f"  mean per-move acceptance = {sum(acc_hist) / len(acc_hist):.4f}")
    assert dz < 3.0, f"suffix lam=0 chain drifted from q0: z={dz:.2f} (>3 SEM)"
    print("PASS: evolved mean within 3*SEM of fresh q0 (suffix move preserves q0).")


if __name__ == "__main__":
    main()
