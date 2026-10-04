"""Sanity check: reproduce Shiu et al. Fig. 1 — sugar GRNs drive MN9.

Stimulate labellar sugar-sensing neurons at 200 Hz for 1 s and check that
the proboscis motor neuron MN9 fires, while it stays silent at baseline.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain.brain import Brain
from flybrain import neurons as nrn

TRIALS = 30

brain = Brain(batch=TRIALS, seed=0)
print(f"brain: {brain.n} neurons, {brain.W._nnz()} connections on {brain.device}")

sugar = brain.ids(nrn.SUGAR_GRNS)
mn9 = brain.ids(nrn.MN9)
print(f"sugar GRNs found: {len(sugar)}/{len(nrn.SUGAR_GRNS)}")

for label, hz in [("baseline", 0.0), ("sugar 200 Hz", 200.0)]:
    brain.reset()
    brain.clear_rates()
    brain.set_rate(sugar, hz)
    t0 = time.time()
    counts = brain.run(1000.0)  # 1 s, all neurons
    wall = time.time() - t0
    rate = counts.mean(dim=1)  # Hz, averaged over trials
    print(f"\n[{label}]  wall {wall:.1f}s for 1s x {TRIALS} trials")
    print(f"  MN9 rates (Hz): {rate[mn9].tolist()}")
    print(f"  neurons active (>0 Hz): {(rate > 0).sum().item()}")
