"""Train JointSpeciesFlow on IPL44 T=0.1 (10K Zenodo, 9000/1000 split). Warmup->1e-3 constant, EMA 0.999,
best-by-val-denoiser-acc + last checkpoints only. G1 numbers printed every eval.
CLI: python -m liquid_coupling_flow.ipl44.train_joint_flow TAG [steps] [hidden_nf] [n_layers] [lam] [ot=global|species]"""
import os, sys, time, copy, torch
from liquid_coupling_flow.ipl44.joint_flow import (JointSpeciesFlow, joint_loss, global_position_ot,
                                                   per_species_ot, random_22_labeling, denoiser_eval)
from liquid_coupling_flow.ipl44.ipl_energy import ipl_box

dev = "cuda" if torch.cuda.is_available() else "cpu"
N, L = ipl_box(); Lf = float(L)
tag = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
hidden = int(sys.argv[3]) if len(sys.argv) > 3 else 64
layers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
lam = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
ot = sys.argv[6] if len(sys.argv) > 6 else "global"
two_time = len(sys.argv) > 7 and sys.argv[7] == "tt"
B, lr, warmup, ema_decay = 256, 1e-3, 500, 0.999
D = "/mnt/ssd/GridTransformer/datasets"; ART = os.path.join(os.path.dirname(__file__), "data")

x = torch.remainder(torch.load(f"{D}/ipl44_T0.1_positions.pt", weights_only=False).float(), Lf)
sp = torch.load(f"{D}/ipl44_T0.1_species.pt", weights_only=False).long()
o = sp.argsort(-1); sp = torch.gather(sp, 1, o); x = torch.gather(x, 1, o.unsqueeze(-1).expand(-1, -1, 2))
xtr, str_, xva, sva = x[:9000].to(dev), sp[:9000].to(dev), x[9000:].to(dev), sp[9000:].to(dev)
nA = int((str_[0] == 0).sum())

m = JointSpeciesFlow(n_particles=N, L=Lf, hidden_nf=hidden, n_layers=layers, two_time=two_time).to(dev)
ema = copy.deepcopy(m).eval()
opt = torch.optim.Adam(m.parameters(), lr=lr)
npar = sum(p.numel() for p in m.parameters())
print(f"[{tag}] {npar/1e3:.1f}k params | hidden {hidden} layers {layers} lam {lam} ot {ot} two_time {two_time} | {steps} steps", flush=True)

def build_ot_bank():
    """Precompute one OT-aligned x0 per training target (refreshed periodically) — removes the ~160ms/step
    per-batch Hungarian from the training loop."""
    tb = time.time(); outs = []
    for s in range(0, xtr.shape[0], 512):
        x1c = xtr[s:s + 512]
        x0c = torch.rand(x1c.shape[0], N, 2, device=dev) * Lf
        outs.append(per_species_ot(x0c, x1c, nA, Lf) if ot == "species" else global_position_ot(x0c, x1c, Lf))
    print(f"  OT bank refreshed in {time.time()-tb:.0f}s", flush=True)
    return torch.cat(outs)


X0BANK = build_ot_bank()
best_acc, t0 = -1.0, time.time()
for step in range(steps):
    for g in opt.param_groups:
        g["lr"] = lr * min(1.0, (step + 1) / warmup)
    if step > 0 and step % 2000 == 0:
        X0BANK = build_ot_bank()
    idx = torch.randint(0, xtr.shape[0], (B,), device=dev)
    x1, s1 = xtr[idx], str_[idx]
    x0 = X0BANK[idx]
    s0 = random_22_labeling(B, N, N - nA, dev)
    loss, lp, ls = joint_loss(m, x0, x1, s0, s1, lam=lam)
    if not torch.isfinite(loss):
        print(f"NON-FINITE loss at {step}, skipping", flush=True); opt.zero_grad(); continue
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step()
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), m.parameters()):
            pe.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
        for be, b in zip(ema.buffers(), m.buffers()):
            be.copy_(b)
    if step % 1000 == 0 or step == steps - 1:
        ev = denoiser_eval(ema, xva[:512], sva[:512])
        ck = {"state_dict": ema.state_dict(), "raw_state_dict": m.state_dict(),
              "cfg": {"hidden_nf": hidden, "n_layers": layers, "lam": lam, "ot": ot, "two_time": two_time},
              "step": step, "val_acc": ev["acc"], "val_ece": ev["ece"]}
        torch.save(ck, f"{ART}/jf_{tag}_last.pt")
        star = ""
        if ev["acc"] > best_acc:
            best_acc = ev["acc"]; torch.save(ck, f"{ART}/jf_{tag}_best.pt"); star = " *BEST*"
        print(f"[{tag}] step {step:6d} loss {float(loss):.3f} (pos {float(lp):.3f} spec {float(ls):.3f}) "
              f"| val acc {ev['acc']:.4f} ece {ev['ece']:.3f}{star} | {time.time()-t0:.0f}s", flush=True)
print(f"[{tag}] DONE best val acc {best_acc:.4f}", flush=True)
