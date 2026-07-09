import torch
from liquid_coupling_flow.ka_pin import pin_mask, masked_mala, masked_swap, masked_block_relabel
from liquid_coupling_flow.ka_energy import ka_energy, ka_forces
from liquid_coupling_flow.ipl44.ipl_swap_smc import uniform_weight_fn

def _setup(B=4, N=64, dev="cpu", seed=0):
    torch.manual_seed(seed)
    L = (N / 1.2) ** 0.5
    x = torch.rand(B, N, 2, device=dev) * L
    s = (torch.rand(B, N, device=dev) < 0.37).long()
    return x, s, L

def test_pin_mask_count_and_frozen_fixed():
    x, s, L = _setup()
    mobile = pin_mask(4, 64, c=0.25, device="cpu", generator=torch.Generator().manual_seed(1))
    assert mobile.dtype == torch.bool and mobile.shape == (4, 64)
    # exactly ceil(c*N)=16 pinned per row
    assert int((~mobile[0]).sum()) == 16
    U = ka_energy(x, s, L)
    efn = lambda a, b: ka_energy(a, b, L); ffn = lambda a, b: ka_forces(a, b, L)
    x2, U2, acc = masked_mala(x, s, U, mobile, beta=2.0, L=L, dt=0.01, energy_fn=efn, force_fn=ffn)
    # frozen particles are bit-for-bit unchanged
    assert torch.equal(x2[~mobile], x[~mobile])
    # energy recomputed from x2 equals the tracked U2 (accept bookkeeping is exact)
    assert torch.allclose(ka_energy(x2, s, L), U2, atol=1e-4)
    assert 0.0 <= acc <= 1.0

def test_masked_mala_multi_step_frozen_invariance():
    """Regression test: frozen particles stay bit-for-bit fixed across repeated masked_mala calls."""
    x0, s, L = _setup()
    mobile = pin_mask(4, 64, c=0.25, device="cpu", generator=torch.Generator().manual_seed(1))
    U = ka_energy(x0, s, L)
    efn = lambda a, b: ka_energy(a, b, L); ffn = lambda a, b: ka_forces(a, b, L)

    # Run 20 iterations of masked_mala, threading x and U through
    x, U_curr = x0.clone(), U.clone()
    for step in range(20):
        x, U_curr, acc = masked_mala(x, s, U_curr, mobile, beta=2.0, L=L, dt=0.01, energy_fn=efn, force_fn=ffn)
        # Acceptance rate must be in [0, 1] every step
        assert 0.0 <= acc <= 1.0, f"Step {step}: acceptance rate {acc} out of bounds"

    # After 20 steps, frozen particles must be unchanged from initial
    assert torch.equal(x[~mobile], x0[~mobile]), "Frozen particles changed after multi-step loop"

def test_masked_species_moves_keep_frozen_and_composition():
    x, s, L = _setup(seed=2)
    mobile = pin_mask(4, 64, 0.25, "cpu", generator=torch.Generator().manual_seed(3))
    efn = lambda a, b: ka_energy(a, b, L)
    U = ka_energy(x, s, L)
    s0 = s.clone()
    s1, U1, acc = masked_swap(x, s, U, mobile, 2.0, efn, uniform_weight_fn)
    assert torch.equal(s1[~mobile], s0[~mobile])                 # frozen species untouched
    assert torch.equal(s1.sum(1), s0.sum(1))                     # swap conserves count
    assert torch.allclose(ka_energy(x, s1, L), U1, atol=1e-4)
    # block-relabel with a constant table (uncertain everywhere) -> exact, frozen fixed, count preserved
    table_fn = lambda xx: torch.full((xx.shape[0], xx.shape[1]), 0.4)
    s2, U2, acc2 = masked_block_relabel(x, s0, U, mobile, 2.0, efn, table_fn, k=4)
    assert torch.equal(s2[~mobile], s0[~mobile])
    assert torch.equal(s2.sum(1), s0.sum(1))
    assert torch.allclose(ka_energy(x, s2, L), U2, atol=1e-4)

def test_masked_swap_exact_with_nonuniform_weight():
    """Detailed-balance check for masked_swap with a non-uniform, s-independent weight_fn and frozen
    same-species particles present. Density 0.2 (not the KA-liquid 1.2) is used deliberately: with only
    4 particles, density 1.2 puts random configs deep in core-overlap (U ~ 1e7), which makes ratio_theory
    underflow to exactly 0 (math domain error on log) regardless of kernel correctness -- an artifact of
    the tiny particle count, not of the property under test. At density 0.2 the energy gap is O(1) so both
    states mix, and the test is decisive: the pre-fix unmasked-normalizer bug reproducibly gives
    |log(ratio_emp) - log(ratio_theory)| ~= 1.39 on this exact seed (mask-independent constant selection
    bias, verified by hand), while the fix gives ~= 0.007."""
    import math
    from collections import Counter
    torch.manual_seed(6)
    L = (4 / 0.2) ** 0.5
    x = torch.rand(1, 4, 2) * L
    mobile = torch.tensor([[True, True, False, False]])          # 1 mobile-A(0), 1 mobile-B(1); frozen A(2),B(3)
    w = torch.tensor([[0.8, 0.3, 0.6, 0.2]])                     # non-uniform, s-independent B-affinity
    weight_fn = lambda xx, ss: w.expand(xx.shape[0], -1).clone()
    efn = lambda a, b: ka_energy(a, b, L); beta = 1.5
    s = torch.tensor([[0, 1, 0, 1]]); U = efn(x, s); cnt = Counter()
    for _ in range(20000):
        s, U, _ = masked_swap(x, s, U, mobile, beta, efn, weight_fn)
        cnt[tuple(s[0].tolist())] += 1
    ratio_theory = math.exp(-beta * float(efn(x, torch.tensor([[1,0,0,1]])) - efn(x, torch.tensor([[0,1,0,1]]))))
    ratio_emp = cnt[(1,0,0,1)] / cnt[(0,1,0,1)]
    assert abs(math.log(ratio_emp) - math.log(ratio_theory)) < 0.1

def test_block_relabel_guards_k_gt_mobile():
    import pytest
    x, s, L = _setup(B=2, N=16, seed=4)
    mobile = torch.zeros(2, 16, dtype=torch.bool); mobile[:, :3] = True   # only 3 mobile
    efn = lambda a, b: ka_energy(a, b, L)
    table_fn = lambda xx: torch.full((xx.shape[0], xx.shape[1]), 0.4)
    with pytest.raises(AssertionError):
        masked_block_relabel(x, s, efn(x, s), mobile, 2.0, efn, table_fn, k=4)
