"""Find an adaptation setting that stops runaway but keeps validated behaviours."""
import sys, time
from pathlib import Path
import pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain.brain import Brain, BrainParams
from flybrain import neurons as nrn

a = nrn.annotations()
sub = pd.read_csv("data/flywire_annotations_783.tsv", sep="\t", usecols=["root_id", "cell_sub_class"])
a = a.merge(sub, on="root_id")
sugar57 = a[a.cell_sub_class.isin(["sugar", "sugar/low_salt"])].root_id.tolist()
odor = nrn.by_type([f"ORN_{g}" for g in ["DM1", "DM4", "DM2", "VM2"]])
jo = a[a.cell_sub_class == "wind_gravity"]
dn = a[(a.super_class == "descending")]
both = set(dn[dn.side == "left"].cell_type.dropna()) & set(dn[dn.side == "right"].cell_type.dropna())
dn = dn[dn.cell_type.isin(both)]
# columns: 0 sugar57@200, 1 odor@30, 2 sugar20@200, 3 windL@50, 4 windR@50
settings = [(0.0, 300.0), (0.5, 300.0), (1.0, 300.0), (2.0, 300.0), (1.0, 1000.0)]
for b, tau in settings:
    br = Brain(BrainParams(adapt_b=b, adapt_tau=tau), batch=5, seed=0)
    for c, (ids, hz) in enumerate([(sugar57, 200), (odor, 30), (nrn.SUGAR_GRNS, 200),
                                   (jo[jo.side == "left"].root_id.tolist(), 50),
                                   (jo[jo.side == "right"].root_id.tolist(), 50)]):
        br.set_rate(br.ids(ids), hz, slice(c, c + 1))
    L = br.ids(dn[dn.side == "left"].root_id.tolist()); R = br.ids(dn[dn.side == "right"].root_id.tolist())
    mn9 = br.ids(nrn.MN9)
    on = br.run(1000.0)
    br.clear_rates(); br.run(500.0)
    off = br.run(500.0).sum(0) * 2
    tot = on.sum(0)
    print(f"b={b} tau={tau}: total Hz on [sug57 {tot[0]:.0f}, odor {tot[1]:.0f}]  off-after-0.5s [sug57 {off[0]:.0f}, odor {off[1]:.0f}]"
          f" | MN9 sug20 {on[mn9, 2].mean():.0f} Hz sug57 {on[mn9, 0].mean():.0f} Hz"
          f" | windL DN L/R {on[L, 3].sum():.0f}/{on[R, 3].sum():.0f}  windR {on[L, 4].sum():.0f}/{on[R, 4].sum():.0f}"
          f" | odor DN L/R {on[L, 1].sum():.0f}/{on[R, 1].sum():.0f}")
