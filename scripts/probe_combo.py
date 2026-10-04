"""Does the wind (JO) steering signal survive on top of odor-driven arousal? Graded?"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain.brain import Brain
from flybrain import neurons as nrn

a = nrn.annotations()
sub = pd.read_csv("data/flywire_annotations_783.tsv", sep="\t", usecols=["root_id", "cell_sub_class"])
jo = a.merge(sub, on="root_id").query("cell_sub_class == 'wind_gravity'")
ODOR = [f"ORN_{g}" for g in ["DM1", "DM4", "DM2", "VM2", "DL1", "DP1m", "VA2", "DM3"]]
REPS = 2
# (odor Hz, wind-left Hz, wind-right Hz)
CONDS = [(0, 0, 0), (0, 50, 0), (0, 0, 50), (0, 15, 0), (0, 0, 15),
         (30, 0, 0), (30, 50, 0), (30, 0, 50), (30, 15, 0), (30, 0, 15)]
brain = Brain(batch=len(CONDS) * REPS, seed=2)
orn = brain.ids(nrn.by_type(ODOR))
joL = brain.ids(jo.query("side == 'left'").root_id.tolist())
joR = brain.ids(jo.query("side == 'right'").root_id.tolist())
for c, (o, wl, wr) in enumerate(CONDS):
    cols = slice(c * REPS, (c + 1) * REPS)
    for idx, hz in ((orn, o), (joL, wl), (joR, wr)):
        if hz: brain.set_rate(idx, hz, cols)
dn = a[a.super_class == "descending"]
dn = dn[dn.root_id.isin(brain.flyid2i)]
both = set(dn[dn.side == "left"].cell_type) & set(dn[dn.side == "right"].cell_type)
dn = dn[dn.cell_type.isin(both)]
L, R = brain.ids(dn.query("side=='left'").root_id.tolist()), brain.ids(dn.query("side=='right'").root_id.tolist())
t0 = time.time()
cnt = brain.run(500.0, record=torch.cat([L, R])).cpu().numpy() * 2
print(f"wall {time.time()-t0:.0f}s")
cl, cr = cnt[:len(L)].sum(0), cnt[len(L):].sum(0)
for c, cond in enumerate(CONDS):
    l, r = cl[c*REPS:(c+1)*REPS].mean(), cr[c*REPS:(c+1)*REPS].mean()
    print(f"odor {cond[0]:>2} windL {cond[1]:>2} windR {cond[2]:>2}:  L {l:7.0f}  R {r:7.0f}  L-R {l-r:+7.0f}")
