"""Event-driven propagation must match the sparse-matrix path exactly (same seed)."""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain.brain import Brain
from flybrain import neurons as nrn

br = Brain(batch=3, seed=0)
res = {}
for mode in (False, True):
    br.event_driven = mode
    br.gen.manual_seed(0); br.reset(); br.clear_rates()
    br.set_rate(br.ids(nrn.SUGAR_GRNS), 200.0)
    torch.cuda.synchronize(); t = time.time()
    c = br.run(300.0)
    torch.cuda.synchronize()
    res[mode] = c
    print(f"event_driven={mode}: {time.time() - t:.2f}s for 300 ms x3, spikes {c.sum().item():.0f}, MN9 {c[br.ids(nrn.MN9)].tolist()}")
print("max |diff| in spike counts:", (res[True] - res[False]).abs().max().item())
