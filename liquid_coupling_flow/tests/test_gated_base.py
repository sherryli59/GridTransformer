"""Tests for ka3d_gated_base (truncated AR base for the cavity flow corrector).

Uses a stub base model (uniform block draws with a position-derived fake logq) so the exactness
bookkeeping is testable on CPU without checkpoints. The stub's "logq" is the block position sum,
which makes any row-misalignment across redraw scatter rounds an immediate test failure.
"""
import torch
import pytest

from liquid_coupling_flow.ka3d_gated_base import GatedARBase, gate_pass, sigma_min_ratio


class StubAR:
    """Uniform block draws in [-R, R]^3; logq := sum of the drawn block coords (an alignment tag)."""

    def __init__(self):
        self.calls = 0
        self.use_frame = False          # pass-through probe

    def sample_block_b(self, xo, so, block_mask, bnd, s_bnd, R, gen=None, pos_temp=1.0):
        self.calls += 1
        M = xo.shape[0]
        K = int(block_mask.sum())
        xo = xo.clone()
        blk = (torch.rand(M, K, 3, generator=gen) - 0.5) * 2 * R
        xo[:, block_mask] = blk
        return xo, so.clone(), blk.sum(dim=(1, 2))


def _setup(M=16, n=20, K=6, n_bnd=30, R=2.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    xo = (torch.rand(n, 3, generator=g) - 0.5) * 2 * R
    so = torch.randint(0, 2, (n,), generator=g)
    bnd = (torch.rand(n_bnd, 3, generator=g) - 0.5) * 2 * (R + 2.0)
    s_bnd = torch.randint(0, 2, (n_bnd,), generator=g)
    block_mask = torch.zeros(n, dtype=torch.bool)
    block_mask[:K] = True
    xo_b = xo[None].expand(M, n, 3).contiguous()
    so_b = so[None].expand(M, n).contiguous()
    return xo_b, so_b, block_mask, bnd, s_bnd, R, g


def _cage(xo_out, so_out, block_mask, bnd, s_bnd):
    M = xo_out.shape[0]
    return (torch.cat([bnd[None].expand(M, -1, -1), xo_out[:, ~block_mask]], 1),
            torch.cat([s_bnd[None].expand(M, -1), so_out[:, ~block_mask]], 1))


def test_all_returned_rows_pass_generous_cut():
    xo_b, so_b, bm, bnd, s_bnd, R, g = _setup()
    base = GatedARBase(StubAR(), cut=0.3, max_rounds=200)
    xo_out, so_out, lq = base.sample_block_b(xo_b, so_b, bm, bnd, s_bnd, R, gen=g)
    assert base.last_pass_mask.all()
    cx, cs = _cage(xo_out, so_out, bm, bnd, s_bnd)
    assert (sigma_min_ratio(xo_out[:, bm], so_out[:, bm], cx, cs) >= 0.3).all()


def test_logq_row_alignment_across_redraws():
    xo_b, so_b, bm, bnd, s_bnd, R, g = _setup()
    base = GatedARBase(StubAR(), cut=0.5, max_rounds=500)
    xo_out, _, lq = base.sample_block_b(xo_b, so_b, bm, bnd, s_bnd, R, gen=g)
    assert base.last_rounds > 0, "test needs at least one redraw round to exercise the scatter"
    assert torch.allclose(lq, xo_out[:, bm].sum(dim=(1, 2)), atol=1e-6), \
        "returned logq must belong to the returned (final) draw of each row"


def test_gated_distribution_is_truncated():
    xo_b, so_b, bm, bnd, s_bnd, R, g = _setup(M=64)
    stub = StubAR()
    raw_x, raw_s, _ = stub.sample_block_b(xo_b, so_b, bm, bnd, s_bnd, R, gen=g)
    cx, cs = _cage(raw_x, raw_s, bm, bnd, s_bnd)
    raw_ratio = sigma_min_ratio(raw_x[:, bm], raw_s[:, bm], cx, cs)

    base = GatedARBase(StubAR(), cut=0.5, max_rounds=500)
    gx, gs, _ = base.sample_block_b(xo_b, so_b, bm, bnd, s_bnd, R, gen=g)
    cx, cs = _cage(gx, gs, bm, bnd, s_bnd)
    gated_ratio = sigma_min_ratio(gx[:, bm], gs[:, bm], cx, cs)
    assert (gated_ratio >= 0.5).all()
    assert gated_ratio.mean() > raw_ratio.mean()


def test_impossible_cut_flags_rows_and_keeps_shapes():
    xo_b, so_b, bm, bnd, s_bnd, R, g = _setup()
    base = GatedARBase(StubAR(), cut=100.0, max_rounds=3)
    xo_out, so_out, lq = base.sample_block_b(xo_b, so_b, bm, bnd, s_bnd, R, gen=g)
    assert (~base.last_pass_mask).all()
    assert base.last_rounds == 3
    assert xo_out.shape == xo_b.shape and lq.shape == (xo_b.shape[0],)


def test_retained_slots_and_species_untouched():
    xo_b, so_b, bm, bnd, s_bnd, R, g = _setup()
    base = GatedARBase(StubAR(), cut=0.5, max_rounds=500)
    xo_out, so_out, _ = base.sample_block_b(xo_b, so_b, bm, bnd, s_bnd, R, gen=g)
    assert torch.equal(xo_out[:, ~bm], xo_b[:, ~bm])
    assert torch.equal(so_out, so_b)


def test_gate_is_deterministic_and_state_free():
    xo_b, so_b, bm, bnd, s_bnd, R, g = _setup()
    stub = StubAR()
    x, s, _ = stub.sample_block_b(xo_b, so_b, bm, bnd, s_bnd, R, gen=g)
    cx, cs = _cage(x, s, bm, bnd, s_bnd)
    m1 = gate_pass(x[:, bm], s[:, bm], cx, cs, cut=0.8)
    m2 = gate_pass(x[:, bm], s[:, bm], cx, cs, cut=0.8)
    assert torch.equal(m1, m2)


def test_attribute_passthrough():
    base = GatedARBase(StubAR(), cut=0.8)
    assert base.use_frame is False        # comes from the stub via __getattr__
    base.use_frame = True                 # sets on the wrapper, must not crash
    assert base.cut == 0.8
