"""Task 3 --- ordered-space independence-Metropolis-Hastings sampler.

State is the ORDERED tuple x_vec plus its cached log q~(x_vec) and U(Phi(x_vec)).
The ordering is part of the state and is never discarded during the chain.

Step (independence sampler; proposal does not depend on current state):
    propose x' ~ q~ with log q~(x');   accept with
        a = min(1, exp[ -beta (U(x') - U(x)) + log q~(x) - log q~(x') ]).

Target over ordered tuples: pi~(x_vec) ∝ e^{-beta U(Phi(x_vec))}.  Because U is
permutation-invariant, pi~ is flat over the N! orderings of each physical config,
so the projected (unordered) marginal is the physical Boltzmann distribution and
the pi~-average of any permutation-invariant observable equals its Boltzmann
average --- that is the exactness claim, checked independently in
test_ordmh_exactness.py.

Implementation guard (the standard silent bug): the REVERSE term log q~(x) in the
ratio is the CACHED value carried with the state --- i.e. evaluated on the
current ordering --- NOT recomputed by re-lifting a projected config.  With
cache_check_every>0 we assert the cached log q~ equals log_q_ordered re-evaluated
on the current ordered state.
"""
import numpy as np

from ordmh_energy import wca_energy


class OrderedMH:
    def __init__(self, N, L, beta, proposal, cache_check_every=0):
        self.N = N
        self.L = float(L)
        self.beta = float(beta)
        self.proposal = proposal
        self.cache_check_every = int(cache_check_every)

    def run(self, n_chains, n_steps, n_equil, thin=2, seed=0, collect=True):
        L, beta, prop = self.L, self.beta, self.proposal
        prop.rng = np.random.default_rng(seed)          # reseed proposal RNG
        rng = np.random.default_rng(seed + 1)

        # ---- initial ordered state (drawn from the proposal) ----
        x, logq = prop.sample_ordered(n_chains)         # (B,N,2), (B,)
        U = wca_energy(x, L)                            # (B,)

        n_acc = 0
        n_prop = 0
        collected = []
        acc_trace = []
        for step in range(n_steps):
            # ---- cached-ordering guard: reverse term is scored on CURRENT order ----
            if self.cache_check_every and step % self.cache_check_every == 0:
                logq_recompute = prop.log_q_ordered(x)
                assert np.allclose(logq, logq_recompute, atol=1e-8), (
                    "cached log q~ does not match the current ordered state --- "
                    "the reverse term must be evaluated on the current ordering, "
                    "not a re-lifted projected config")

            # ---- propose a fresh ordered tuple (independence sampler) ----
            xprop, logq_prop = prop.sample_ordered(n_chains)
            Uprop = wca_energy(xprop, L)

            log_a = -beta * (Uprop - U) + logq - logq_prop
            accept = np.log(rng.uniform(size=n_chains)) < log_a

            x = np.where(accept[:, None, None], xprop, x)
            U = np.where(accept, Uprop, U)
            logq = np.where(accept, logq_prop, logq)     # keep cache in sync w/ order

            n_acc += int(accept.sum())
            n_prop += n_chains
            acc_trace.append(accept.mean())

            if collect and step >= n_equil and (step - n_equil) % thin == 0:
                collected.append(x.copy())

        configs = np.concatenate(collected, axis=0) if collected else x.copy()

        # per-chain time-mean of U -> honest across-chain SEM (accounts for the
        # independence sampler's within-chain autocorrelation)
        if collected:
            stack = np.stack(collected, axis=0)          # (T, B, N, 2)
            U_series = wca_energy(stack.reshape(-1, self.N, 2), L).reshape(
                stack.shape[0], n_chains)                # (T, B)
            chain_meanU = U_series.mean(axis=0)          # (B,)
            mean_U = float(chain_meanU.mean())
            mean_U_err = float(chain_meanU.std(ddof=1) / np.sqrt(n_chains))
        else:
            chain_meanU = U
            mean_U = float(U.mean())
            mean_U_err = float(U.std(ddof=1) / np.sqrt(n_chains))

        return {
            "configs": configs,
            "acc_rate": n_acc / max(n_prop, 1),
            "acc_trace": np.asarray(acc_trace),
            "mean_U": mean_U,
            "mean_U_err": mean_U_err,
            "chain_meanU": chain_meanU,
            "n_chains": n_chains,
        }
