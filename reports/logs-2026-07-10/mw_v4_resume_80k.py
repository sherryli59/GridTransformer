"""Continue v4 for 80k augmented MLE updates, preserving best and most-recent checkpoints."""
from liquid_coupling_flow.mw.mw_generator_v4 import train

train(steps=80_000, batch=32, lr=2e-4, val_every=500, seed=47,
      out="mw_gen_N64_v4.pt", warm="mw_gen_N64_v4.pt",
      extension="liquid_coupling_flow/mw/artifacts/mw_ref_N64_ext.pt")
