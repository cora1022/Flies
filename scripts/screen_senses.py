"""Screen every sensory class for left/right information reaching descending neurons.

For each sensory group, stimulate the left copy and the right copy (separate batch
columns) and measure a mirrored steering index per descending cell type:
    s = (DN_left - DN_right | left stim) - (DN_left - DN_right | right stim)
A good steering input gives large |s| on some DN types (ideally DNa01/DNa02).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain.brain import Brain
from flybrain import neurons as nrn

HZ, MS, REPS = 50.0, 500.0, 2
a = nrn.annotations().copy()
sub = pd.read_csv("data/flywire_annotations_783.tsv", sep="\t", usecols=["root_id", "cell_sub_class"])
a = a.merge(sub, on="root_id")
GROUPS = {
    "wind_gravity": a.cell_sub_class == "wind_gravity",
    "auditory": a.cell_sub_class == "auditory",
    "head_bristle": a.cell_sub_class == "head bristle",
    "ocellar": a.cell_sub_class == "ocellar",
    "bitter": a.cell_sub_class == "bitter",
    "sugar": a.cell_sub_class == "sugar",
    "water": a.cell_sub_class == "water",
    "high_salt": a.cell_sub_class == "high_salt/heavy_metal",
    "dry": a.cell_sub_class == "dry",
    "pheromone": a.cell_sub_class == "pheromone",
    "leg_SA_DMT": a.cell_sub_class == "SA_DMT_DMetaN",
    "AN_IPS_GNG": a.cell_sub_class.isin(["AN_IPS_GNG", "AN_GNG_IPS"]),
    "AN_multi": a.cell_sub_class == "AN_multi",
}
names = list(GROUPS)
brain = Brain(batch=len(names) * 2 * REPS, seed=1)
for g, name in enumerate(names):
    for s, side in enumerate(("left", "right")):
        ids = brain.ids(a.loc[GROUPS[name] & (a.side == side), "root_id"].tolist())
        col = (g * 2 + s) * REPS
        brain.set_rate(ids, HZ, slice(col, col + REPS))

dn = a[a.super_class == "descending"]
dn_idx = brain.ids(dn.root_id.tolist())
t0 = time.time()
counts = brain.run(MS, record=dn_idx).cpu().numpy() * (1000.0 / MS)
print(f"wall {time.time() - t0:.0f}s  batch {brain.batch}")
rates = counts.reshape(len(dn_idx), len(names), 2, REPS).mean(-1)  # dn, group, stimside

lab = dn.set_index("root_id").loc[[brain.flyids[i] for i in dn_idx.tolist()]].reset_index()
df = pd.DataFrame({"type": lab.cell_type.fillna("?"), "side": lab.side})
out = []
for g, name in enumerate(names):
    d = df.assign(rL=rates[:, g, 0], rR=rates[:, g, 1])
    piv = d.groupby(["type", "side"])[["rL", "rR"]].sum().unstack("side").fillna(0)
    if ("rL", "left") not in piv or ("rL", "right") not in piv:
        continue
    s = (piv[("rL", "left")] - piv[("rL", "right")]) - (piv[("rR", "left")] - piv[("rR", "right")])
    tot = rates[:, g].sum()
    top = s.abs().sort_values(ascending=False).head(5)
    turn = {k: round(float(s.get(k, 0)), 1) for k in ("DNa01", "DNa02")}
    out.append((name, float(s.abs().sum()), tot))
    print(f"\n[{name}] total DN Hz {tot:.0f}  |steer| {s.abs().sum():.0f}  turn-DNs {turn}")
    print("   " + ", ".join(f"{k}:{s[k]:+.0f}" for k in top.index))
np.savez("outputs/screen.npz", rates=rates, names=np.array(names))
