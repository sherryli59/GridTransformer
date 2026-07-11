"""Fine-tune the learned-pairwise-potential (EBM) local-frame head at N=100. Warm-start the shared parts
(transformer, head_a/head_b, embeddings) from the proven factorized checkpoint; phi is init ~0 so the model
starts IDENTICAL to the factorized head, then learns the excluded-volume tilt on b|a. Beat the factorized
suffix-blob baseline: k=4 30% MTM / 1.15 clashes, k=8 5% / 2.70 clashes."""
import os, sys, time, argparse
import torch
from liquid_coupling_flow.ka_localframe_ebm import KALocalFrameEBM
from liquid_coupling_flow.ka_gridformer_train import augment

ART = "liquid_coupling_flow/artifacts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warm", default=f"{ART}/ka_localframe_N100_20k.pt")
    ap.add_argument("--out", default=f"{ART}/ka_localframe_ebm_N100.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); dev = a.device
    ck = torch.load(a.warm, map_location=dev, weights_only=False)
    m = KALocalFrameEBM(rho=ck["rho"], n_bins=ck["n_bins"], knn=ck["knn"]).to(dev)
    miss, unexp = m.load_state_dict(ck["state_dict"], strict=False)
    print(f"warm-start {a.warm}: {len(miss)} new params (phi/pair_emb), {len(unexp)} unused", flush=True)
    ref = torch.load(f"{ART}/ka_reference_N100.pt", map_location=dev, weights_only=False)
    data, s, L = ref["x"].to(dev), ref["s"].to(dev).long(), ref["L"]; N = 100
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    t0 = time.time(); m.train()
    for step in range(a.steps + 1):
        idx = torch.randint(0, data.shape[0], (a.batch,), device=dev)
        loss = -(m.log_prob(augment(data[idx], L), s) / N).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if step % 500 == 0:
            print(f"  step {step:5d} -logq/N {loss.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
            torch.save({"state_dict": m.state_dict(), "rho": ck["rho"], "n_bins": ck["n_bins"],
                        "knn": ck["knn"], "step": step}, a.out)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
