"""Does activity persist after the stimulus is removed? (runaway / bistability check)"""
import sys
from pathlib import Path
import pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain.brain import Brain
from flybrain import neurons as nrn

a = nrn.annotations()
sub = pd.read_csv("data/flywire_annotations_783.tsv", sep="\t", usecols=["root_id", "cell_sub_class"])
a = a.merge(sub, on="root_id")
stims = {
    "sugar57_200": (a[a.cell_sub_class.isin(["sugar", "sugar/low_salt"])].root_id.tolist(), 200),
    "sugar20_200": (nrn.SUGAR_GRNS, 200),
    "sugar57_50": (a[a.cell_sub_class.isin(["sugar", "sugar/low_salt"])].root_id.tolist(), 50),
    "odor_30": (nrn.by_type([f"ORN_{g}" for g in ["DM1", "DM4", "DM2", "VM2"]]), 30),
    "wind_50": (a[a.cell_sub_class == "wind_gravity"].root_id.tolist(), 50),
}
brain = Brain(batch=len(stims), seed=0)
for c, (ids, hz) in enumerate(stims.values()):
    brain.set_rate(brain.ids(ids), hz, slice(c, c + 1))
print(f"{'stim':>12}  " + "  ".join(f"{w:>9}" for w in ["on 0-300", "off 0-100", "off 400-700", "off 1.7-2s"]))
res = [brain.run(300.0).sum(0)]
brain.clear_rates()
res.append(brain.run(100.0).sum(0)); brain.run(300.0)
res.append(brain.run(300.0).sum(0)); brain.run(1000.0)
res.append(brain.run(300.0).sum(0))
for c, name in enumerate(stims):
    print(f"{name:>12}  " + "  ".join(f"{r[c].item():9.0f}" for r in res))
