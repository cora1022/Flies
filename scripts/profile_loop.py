"""Where does the time go in one closed-loop step? (brain batch B vs physics)"""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain.brain import Brain
from flybrain import experiment as ex

for B in (1, 2, 3):
    br = Brain(batch=B, seed=0)
    br.set_rate(br.ids(ex.resolve("JO")), 25.0)
    br.run(5); torch.cuda.synchronize(); t = time.time()
    for _ in range(1000): br.step()
    torch.cuda.synchronize(); print(f"brain B={B}: {(time.time() - t):.2f} s per 100 ms sim")
    t = time.time(); W = br.W
    x = torch.rand(br.n, B, device=br.device).lt(0.01).float()
    for _ in range(1000): y = torch.sparse.mm(W, x) if B > 1 else torch.mv(W, x[:, 0])
    torch.cuda.synchronize(); print(f"   spmm only: {(time.time() - t):.2f} s")
    del br; torch.cuda.empty_cache()

fly, sim, ctl = ex._make_fly_sim("fly", 0)
from flygym_demo.complex_terrain import HybridControllerObservation, apply_locomotion_action
import numpy as np
t = time.time()
for _ in range(1000):
    obs = HybridControllerObservation.from_sim(sim, fly.name)
    apply_locomotion_action(sim, fly.name, ctl.step(np.array([1.0, 1.0]), obs)); sim.step()
print(f"physics+ctl 1 fly: {(time.time() - t):.2f} s per 100 ms sim")
t = time.time()
for _ in range(1000): sim.step()
print(f"   physics only: {(time.time() - t):.2f} s")
