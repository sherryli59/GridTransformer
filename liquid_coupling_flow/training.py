"""Reusable max-likelihood training loop for the toy flows."""

from __future__ import annotations

import torch


def train_flow(flow, target_sampler, steps: int = 3000, batch: int = 512,
               lr: float = 1e-3, device: str = "cpu", log_every: int = 250,
               grad_clip: float = 10.0):
    """Train `flow` to fit samples from `target_sampler(batch) -> [batch, dim]`
    by minimising E[-log p(x)]. Returns the (step, nll) history."""
    flow.to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    history = []
    for step in range(steps):
        x = target_sampler(batch).to(device)
        loss = flow.forward_kld(x)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), grad_clip)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            history.append((step, float(loss.item())))
            print(f"  step {step:5d}   nll {loss.item():.4f}")
    return history
