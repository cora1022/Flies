"""Probe: does one-sided odor input produce left/right asymmetry in descending neurons?

Stimulates attractive-odor ORNs (DM1, DM4, DM2, VM2, DL1 ... vinegar-like) on the
left side, right side, or both, and prints the firing of steering-related DNs.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain.brain import Brain
from flybrain import neurons as nrn

GLOMERULI = sys.argv[1].split(",") if len(sys.argv) > 1 else ["DM1", "DM4", "DM2", "VM2", "DL1", "DP1m", "VA2", "DM3"]
HZ = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
CONDS = ["none", "left", "right", "both"]

import os
from flybrain.brain import BrainParams
brain = Brain(BrainParams(w_syn=float(os.environ.get("W_SYN", 0.275))), batch=REPS * len(CONDS), seed=0)
orn = {s: brain.ids(nrn.by_type([f"ORN_{g}" for g in GLOMERULI], side=s)) for s in ("left", "right")}
print(f"ORNs: left {len(orn['left'])}, right {len(orn['right'])}  @ {HZ} Hz")
for c, cond in enumerate(CONDS):
    cols = slice(c * REPS, (c + 1) * REPS)
    for side in ("left", "right"):
        if cond in (side, "both"):
            brain.set_rate(orn[side], HZ, cols)

a = nrn.annotations()
dn = a[a.super_class == "descending"]
dn_idx = brain.ids(dn.root_id.tolist())
t0 = time.time()
counts = brain.run(1000.0, record=dn_idx)  # Hz over 1 s
print(f"wall {time.time() - t0:.1f}s")
rates = torch.stack([counts[:, c * REPS:(c + 1) * REPS].mean(1) for c in range(len(CONDS))], 1).cpu()

labels = dn.set_index("root_id").loc[[brain.flyids[i] for i in dn_idx.tolist()]]
names = (labels.cell_type.fillna("?") + "_" + labels.side.str[0]).tolist()
active = (rates.max(1).values > 0).nonzero().flatten().tolist()
print(f"DNs active in any condition: {len(active)} / {len(dn_idx)}")
print(f"{'DN':>14} " + " ".join(f"{c:>7}" for c in CONDS))
for i in sorted(active, key=lambda i: -(rates[i, 1] - rates[i, 2]).abs()):
    print(f"{names[i]:>14} " + " ".join(f"{r:7.1f}" for r in rates[i].tolist()))
for key in ["DNa01", "DNa02", "DNp09", "MDN"]:
    for i, nm in enumerate(names):
        if nm.startswith(key + "_"):
            print("  key", nm, [round(x, 1) for x in rates[i].tolist()])

# population readout: left-side DNs vs right-side DNs
import numpy as np
side = labels.side.to_numpy()
L, R = rates[side == "left"].sum(0), rates[side == "right"].sum(0)
print("population  L:", [round(x) for x in L.tolist()], " R:", [round(x) for x in R.tolist()])
print("(L-R)/(L+R):", [round(x, 4) for x in ((L - R) / (L + R + 1e-9)).tolist()])
Path("outputs").mkdir(exist_ok=True)
np.savez(f"outputs/probe_{HZ:g}.npz", rates=rates.numpy(), names=np.array(names), side=side)
