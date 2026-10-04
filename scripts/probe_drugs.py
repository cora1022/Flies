"""What do 'drug' manipulations do? Wind from the left + a neuromodulator group."""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flybrain.brain import Brain
from flybrain import experiment as ex, neurons as nrn

C = [("대조군 (왼쪽 바람)", None, None), ("보상 도파민 100Hz", "activate", "보상 도파민"),
     ("처벌 도파민 100Hz", "activate", "처벌 도파민"), ("옥토파민 100Hz", "activate", "옥토파민"),
     ("세로토닌 100Hz", "activate", "세로토닌"), ("GABA 억제 끄기", "silence", "GABA 억제"),
     ("쓴맛 200Hz", "activate", "쓴맛"), ("물 200Hz", "activate", "물")]
br = Brain(batch=len(C), seed=0)
ids = lambda t, s="both": br.ids(ex.resolve(t, s))
br.set_rate(ids("JO", "left"), 40.0); br.set_rate(ids("JO", "right"), 10.0)
for c, (_, op, tgt) in enumerate(C):
    if op == "activate": br.set_rate(ids(tgt), 100.0 if "Hz" in C[c][0] and "200" not in C[c][0] else 200.0, slice(c, c + 1))
    if op == "silence": br.silence(ids(tgt), slice(c, c + 1))
a = ex.annotations()
dn = a[(a.super_class == "descending") & a.root_id.isin(br.flyid2i)]
both = set(dn[dn.side == "left"].cell_type.dropna()) & set(dn[dn.side == "right"].cell_type.dropna())
dn = dn[dn.cell_type.isin(both)]
L, R = br.ids(dn[dn.side == "left"].root_id.tolist()), br.ids(dn[dn.side == "right"].root_id.tolist())
t = time.time()
cnt = br.run(500.0) * 2
print(f"wall {time.time()-t:.0f}s")
for c, (name, *_) in enumerate(C):
    k = cnt[:, c]
    print(f"{name:>16}: total {k.sum():8.0f} Hz | DN L/R {k[L].sum():6.0f}/{k[R].sum():6.0f} | MDN {k[ids('MDN')].mean():5.1f} | P9 {k[ids('DNp09')].mean():5.1f} | MN9 {k[br.ids(nrn.MN9)].mean():5.1f}")
