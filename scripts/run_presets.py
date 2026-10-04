"""Queue every preset in viewer/presets.json on the running lab server.

    python scripts/run_presets.py                # all presets
    python scripts/run_presets.py 뒤바뀐 도파민   # only presets whose name contains one of these words
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "http://localhost:8765"


def fill(cfg: dict) -> dict:
    """Same defaults the viewer applies before running a preset."""
    c = json.loads(json.dumps(cfg))
    s = c["seconds"]

    def m_(m):
        m = {"side": "both", "t_on": 0, **m}
        m["t_off"] = min(m.get("t_off", s), s)
        if m["op"] == "activate":
            m.setdefault("hz", 150)
        return m
    c["foods"] = [{"fan": True, "effect_s": 0, **f} for f in c["foods"]]
    c["variants"] = [dict(name=v["name"], manips=[m_(m) for m in v.get("manips", [])], swap_jo=bool(v.get("swap_jo")),
                          twin=None if not v.get("twin") else dict(mode=v["twin"]["mode"], swap_jo=bool(v["twin"].get("swap_jo")),
                                                                   manips=[m_(m) for m in v["twin"].get("manips", [])]))
                     for v in c["variants"]]
    c.setdefault("wind_hz", 50)
    c.setdefault("arena", "sequence")
    c.setdefault("brain", {"w_syn": 0.275, "adapt_b": 0, "adapt_tau": 300})
    return c


presets = [p for g in json.loads((ROOT / "viewer/presets.json").read_text(encoding="utf8")) for p in g["items"]]
words = sys.argv[1:]
for p in presets:
    if words and not any(w in p["name"] for w in words):
        continue
    req = urllib.request.Request(URL + "/api/run", data=json.dumps(fill(p["cfg"])).encode(),
                                 headers={"Content-Type": "application/json"})
    print(p["name"], "->", urllib.request.urlopen(req).read().decode())
