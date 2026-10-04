"""Run a closed-loop brain + body experiment with one fly per variant.

All variants share one batched brain on the GPU (one batch column per fly) and
each has its own MuJoCo simulation. A variant can activate (Poisson drive, like
optogenetics) or silence (cut output synapses) any cell type or sensory group.

    cfg = {
      "seconds": 8, "wind_hz": 50, "foods": [[22, 14], [26, -12]],
      "brain": {"w_syn": 0.275, "adapt_b": 1.0, "adapt_tau": 300},
      "variants": [
        {"name": "대조군", "manips": []},
        {"name": "왼쪽 더듬이 절단", "manips": [{"op": "silence", "target": "JO", "side": "left"}]},
        {"name": "MDN 자극", "manips": [{"op": "activate", "target": "MDN", "side": "both",
                                         "hz": 100, "t_on": 2, "t_off": 4}]},
      ],
    }
"""
from __future__ import annotations

import base64
import math
import time
from functools import lru_cache

import mujoco as mj
import numpy as np
import pandas as pd
import torch

from .brain import Brain, BrainParams, DATA_DIR
from . import neurons as nrn

# named neuron groups addressable in experiments: name -> (description, selector on annotations)
def _sub(*vals):
    return lambda a: a.cell_sub_class.isin(vals)


def _type_re(pattern):
    return lambda a: a.cell_type.fillna("").str.contains(pattern, regex=True)


GROUPS = {
    "JO": ("바람·중력 감지 (더듬이 존스턴 기관)", _sub("wind_gravity")),
    "청각": ("소리 감지 (존스턴 기관)", _sub("auditory")),
    "설탕": ("설탕 미각 뉴런 전체", _sub("sugar", "sugar/low_salt")),
    "쓴맛": ("쓴맛 미각 뉴런", _sub("bitter")),
    "물": ("물 미각 뉴런", _sub("water")),
    "소금": ("짠맛·중금속 미각 뉴런", _sub("high_salt/heavy_metal")),
    "냄새": ("후각 수용 뉴런 전체 (ORN)", lambda a: a.cell_class == "olfactory"),
    "홑눈": ("홑눈 광수용 뉴런", _sub("ocellar")),
    "머리 감각모": ("머리 기계감각 뉴런", _sub("head bristle")),
    "보상 도파민": ("PAM 도파민 군집 (보상 학습)", _type_re(r"^PAM")),
    "처벌 도파민": ("PPL1 도파민 군집 (처벌 학습)", _type_re(r"^PPL1")),
    "도파민 전체": ("PAM·PPL1·PPL2·PPM 도파민 뉴런", _type_re(r"^(PAM|PPL1|PPL2|PPM)")),
    "옥토파민": ("OA 뉴런 (곤충의 아드레날린, 스트레스·각성)", _type_re(r"^OA-")),
    "세로토닌": ("5-HT 뉴런", _type_re(r"^(5-HT|CSD)")),
    "GABA 억제": ("GABA 억제 뉴런 전체 (예측 신뢰도 0.5 이상)",
                 lambda a: (a.top_nt == "gaba") & (a.top_nt_conf > 0.5)),
}
FOOD_RADIUS = 2.5
FEED_TIME = 1.5
SUGAR_HZ = 200.0  # paper's 20 sugar GRNs at 200 Hz: drives MN9 without runaway
READ_EVERY = 20          # steps between readouts (2 ms)
CTRL_EVERY = 5           # walking controller runs every 0.5 ms (same gait as every step, 3x cheaper)
TAU_RATE = 0.1           # s
MN9_FEED_HZ = 20.0
MAX_ACTIVE_PER_FRAME = 3000
CHOICE_FALLOFF_MM = 6.0  # wind weight exp(-distance / this) in the choice arena
VARIANT_COLORS = ["#ffb547", "#4fc3c9", "#f0806a", "#b59cf5"]


@lru_cache(maxsize=1)
def annotations() -> pd.DataFrame:
    a = pd.read_csv(DATA_DIR / "flywire_annotations_783.tsv", sep="\t",
                    usecols=["root_id", "super_class", "cell_class", "cell_sub_class", "cell_type",
                             "hemibrain_type", "side", "top_nt", "top_nt_conf", "pos_x", "pos_y"],
                    dtype={"cell_type": str, "hemibrain_type": str})
    return a


def resolve(target: str, side: str = "both") -> list[int]:
    """Root IDs for a group name or a cell type, on one or both sides."""
    a = annotations()
    if target in GROUPS:
        m = GROUPS[target][1](a)
    else:
        m = (a.cell_type == target) | (a.hemibrain_type == target)
    if side in ("left", "right"):
        m &= a.side == side
    return a.loc[m, "root_id"].tolist()


def catalog() -> dict:
    """Everything the UI can target: groups + all cell types with side counts."""
    a = annotations()
    t = a.dropna(subset=["cell_type"])
    g = t.groupby("cell_type").agg(
        sc=("super_class", "first"), nt=("top_nt", "first"),
        l=("side", lambda s: int((s == "left").sum())), r=("side", lambda s: int((s == "right").sum())),
        n=("root_id", "size"))
    types = [[k, row.sc, row.nt if isinstance(row.nt, str) else "", row.l, row.r, row.n]
             for k, row in g.iterrows()]
    groups = []
    for k, (desc, sel) in GROUPS.items():
        m = sel(a)
        groups.append([k, desc, int((m & (a.side == "left")).sum()), int((m & (a.side == "right")).sum())])
    return {"groups": groups, "types": types}


def _b64(x: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(x).tobytes()).decode()


# ---------------------------------------------------------------------------
_BRAIN_CACHE: dict = {}


def get_brain(params: BrainParams, batch: int, seed: int) -> Brain:
    """Reuse the loaded connectome when only the batch/state changes (loading takes ~10 s)."""
    key = (params.w_syn, params.adapt_b, params.adapt_tau, batch)
    br = _BRAIN_CACHE.get(key)
    if br is None:
        _BRAIN_CACHE.clear()
        torch.cuda.empty_cache()
        br = _BRAIN_CACHE[key] = Brain(params, batch=batch, seed=seed)
    br.gen.manual_seed(seed)
    br.clear_rates()
    br.silenced.zero_()
    br.reset()
    return br


def _make_fly_sim(name: str, seed: int):
    from flygym import Simulation
    from flygym.anatomy import ContactBodiesPreset
    from flygym.compose import FlatGroundWorld
    from flygym.utils.math import Rotation3D
    from flygym_demo.complex_terrain import (
        HybridTurningController, LocomotionAction, PreprogrammedSteps,
        apply_locomotion_action, make_locomotion_fly)

    fly = make_locomotion_fly(name=name, add_adhesion=True, colorize=True)
    world = FlatGroundWorld()
    world.add_fly(fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),
                  bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
                  add_ground_contact_sensors=False)
    sim = Simulation(world)
    steps = PreprogrammedSteps()
    order = fly.get_actuated_jointdofs_order("position")
    ctl = HybridTurningController(timestep=sim.timestep * CTRL_EVERY, preprogrammed_steps=steps,
                                  output_dof_order=order)
    sim.reset(); ctl.reset(seed=seed)
    apply_locomotion_action(sim, fly.name, LocomotionAction(
        joint_angles=steps.default_pose_by_dof_order(order),
        adhesion_onoff=np.ones(6, dtype=bool)))
    sim.warmup()
    return fly, sim, ctl


def body_meshes() -> dict:
    """Mesh geometry of the fly, shared by every run (export once)."""
    fly, sim, _ = _make_fly_sim("fly", 0)
    m = sim.mj_model
    out = []
    for g in range(m.ngeom):
        if m.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        out.append(dict(v=_b64(m.mesh_vert[va:va + vn].astype(np.float32)),
                        f=_b64(m.mesh_face[fa:fa + fn].astype(np.uint32)),
                        rgba=[round(float(x), 3) for x in rgba]))
    sim.close()
    return {"meshes": out}


def brain_map() -> dict:
    """Front-view neuron positions and labels for the brain activity map."""
    a = annotations().drop_duplicates("root_id").set_index("root_id")
    comp = pd.read_csv(DATA_DIR / "Completeness_783.csv", index_col=0)
    a = a.reindex(comp.index)
    xy = a[["pos_x", "pos_y"]].fillna(a[["pos_x", "pos_y"]].median()).to_numpy()
    span = xy.max(0) - xy.min(0)
    xy = (xy - xy.min(0)) / span * 65535
    return {"xy": _b64(xy.astype(np.uint16)), "aspect": float(span[0] / span[1]),
            "type": a.cell_type.fillna("").tolist(), "side": a.side.fillna("").str[:1].tolist(),
            "sc": a.super_class.fillna("").tolist()}


# ---------------------------------------------------------------------------
def _norm_foods(raw) -> list[dict]:
    """Foods may be [x, y] (sugar with a fan) or dicts with taste / fan / effect_s / hz."""
    out = []
    for f in raw:
        if isinstance(f, dict):
            out.append(dict(x=float(f["x"]), y=float(f["y"]), taste=f.get("taste", "설탕"),
                            side=f.get("side", "both"), fan=bool(f.get("fan", True)),
                            effect_s=float(f.get("effect_s", 0.0)), hz=float(f.get("hz", 200.0))))
        else:
            out.append(dict(x=float(f[0]), y=float(f[1]), taste="설탕", side="both", fan=True,
                            effect_s=0.0, hz=200.0))
    return out


def _drive(nL, nR, p9, mdn, turn_gain):
    """Descending-neuron readout -> [left, right] CPG drive (the hand-written part of the loop)."""
    walk = min(1.2, (nL + nR) / 2 + p9 / 50.0)
    asym = (nL - nR) / (nL + nR + 1e-6)  # + = left DNs more active -> turn left
    drive = walk * np.clip(np.array([1 - turn_gain * asym, 1 + turn_gain * asym]), -0.2, 1.4)
    backward = min(1.0, mdn / 40.0)  # MDN: moonwalker descending neuron
    if backward > 0.1:
        drive = -0.9 * backward * np.clip(np.array([1 + turn_gain * asym, 1 - turn_gain * asym]), 0.2, 1.4)
    return drive


def run_experiment(cfg: dict, progress=lambda frac, msg: None) -> dict:
    variants = cfg["variants"]
    seconds = float(cfg.get("seconds", 8.0))
    wind_hz = float(cfg.get("wind_hz", 50.0))
    foods = _norm_foods(cfg.get("foods", [[22, 14]]))
    choice = cfg.get("arena", "sequence") == "choice"  # all fans blow at once
    turn_gain = float(cfg.get("turn_gain", 1.5))
    fps = float(cfg.get("fps", 50.0))
    seed = int(cfg.get("seed", 0))
    bp = cfg.get("brain", {})
    params = BrainParams(w_syn=float(bp.get("w_syn", 0.275)),
                         adapt_b=float(bp.get("adapt_b", 0.0)),
                         adapt_tau=float(bp.get("adapt_tau", 300.0)))

    # brain columns: every variant has brain A; a twin variant also has brain B
    cols = []
    for vi, v in enumerate(variants):
        cols.append(dict(body=vi, role="A", manips=v.get("manips", []), swap_jo=bool(v.get("swap_jo", False))))
        tw = v.get("twin")
        if tw:
            cols.append(dict(body=vi, role="B", manips=tw.get("manips", []), swap_jo=bool(tw.get("swap_jo", False))))
    NB = len(cols)

    progress(0.0, "커넥톰 불러오는 중")
    brain = get_brain(params, NB, seed)
    ids = lambda target, side="both": brain.ids(resolve(target, side))
    joL, joR = ids("JO", "left"), ids("JO", "right")
    sugar = brain.ids(nrn.SUGAR_GRNS)  # the full 57-neuron sugar group can ignite runaway activity
    mn9 = brain.ids(nrn.MN9)
    a = annotations()
    dn = a[(a.super_class == "descending") & a.root_id.isin(brain.flyid2i)]
    paired = set(dn[dn.side == "left"].cell_type.dropna()) & set(dn[dn.side == "right"].cell_type.dropna())
    dn = dn[dn.cell_type.isin(paired)]
    dnL = brain.ids(dn[dn.side == "left"].root_id.tolist())
    dnR = brain.ids(dn[dn.side == "right"].root_id.tolist())
    mdn, p9 = ids("MDN"), ids("DNp09")
    key_types = ["DNa01", "DNa02", "DNp09", "MDN", "DNb06", "DNg29"]
    key_idx = {f"{t} {s[0].upper()}": ids(t, s) for t in key_types for s in ("left", "right")}
    key_idx = {k: v for k, v in key_idx.items() if len(v)}
    food_idx = []
    for f in foods:
        idx = sugar if f["taste"] == "설탕" else ids(f["taste"], f["side"])
        if not len(idx):
            raise ValueError(f"먹이 자극 '{f['taste']}' 에 해당하는 뉴런이 없어요")
        food_idx.append(idx)

    # ---- calibration on the unmanipulated brain: symmetric frontal wind -> zero turn
    progress(0.02, "좌우 영점 보정 중")
    brain.set_rate(joL, wind_hz / 2); brain.set_rate(joR, wind_hz / 2)
    brain.run(100.0)
    cnt = brain.run(300.0, record=torch.cat([dnL, dnR])).cpu().numpy() / 0.3
    refL = max(1.0, float(cnt[:len(dnL)].sum(0).mean()))
    refR = max(1.0, float(cnt[len(dnL):].sum(0).mean()))
    brain.clear_rates(); brain.reset()

    # ---- manipulations (all timed; silencing defaults to the whole run)
    act_list, sil_list = [], []
    for c, col in enumerate(cols):
        col["idx"] = {"activate": [], "silence": []}
        for mp in col["manips"]:
            idx = ids(mp["target"], mp.get("side", "both"))
            if not len(idx):
                raise ValueError(f"'{mp['target']}' 에 해당하는 뉴런이 없어요")
            t_on, t_off = float(mp.get("t_on", 0.0)), float(mp.get("t_off", seconds))
            if mp["op"] == "silence":
                sil_list.append((c, idx, t_on, t_off)); col["idx"]["silence"] += idx.tolist()
            else:
                act_list.append((c, idx, float(mp.get("hz", 100.0)), t_on, t_off))
                col["idx"]["activate"] += idx.tolist()

    # ---- bodies
    progress(0.04, "초파리 몸 만드는 중")
    from flygym_demo.complex_terrain import HybridControllerObservation, apply_locomotion_action
    bodies = []
    for vi, v in enumerate(variants):
        fly, sim, ctl = _make_fly_sim("fly", seed + vi)
        m = sim.mj_model
        bodies.append(dict(
            fly=fly, sim=sim, ctl=ctl, d=sim.mj_data, twin=(v.get("twin") or {}).get("mode"),
            cols=[c for c, col in enumerate(cols) if col["body"] == vi],
            thorax=mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "fly/c_thorax"),
            eye=mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "fly/l_eye"),
            geoms=[g for g in range(m.ngeom) if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH],
            action=None, eaten=[], eaten_set=set(), feed_t=[0.0] * len(foods),
            effect_until=[-1.0] * len(foods), feeding=False, drive=np.zeros(2), jo=(0.0, 0.0),
            fans=[], stim=[], dist=float("nan"), path=0.0, last_xy=None,
            frames=[], poses=[], active={c: [] for c in range(NB)}))
    dt = bodies[0]["sim"].timestep
    assert abs(dt * 1000 - brain.p.dt) < 1e-9
    rd = np.zeros((NB, 5))  # filtered readouts per brain column: rL, rR, mn9, mdn, p9

    n_steps = int(seconds / dt)
    frame_every = int(round(1.0 / fps / dt))
    alpha = (READ_EVERY * dt) / TAU_RATE
    win = READ_EVERY * dt
    acc_read = torch.zeros(brain.n, NB, device=brain.device)
    acc_frame = torch.zeros(brain.n, NB, device=brain.device)
    manual = torch.zeros(brain.n, NB, device=brain.device)
    sens = torch.zeros(brain.n, NB, device=brain.device)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    for step in range(n_steps):
        t = step * dt
        if step % READ_EVERY == 0:
            manual.zero_(); sens.zero_(); brain.silenced.zero_()
            for c, idx, hz, t_on, t_off in act_list:
                if t_on <= t < t_off:
                    manual[idx, c] = torch.clamp(manual[idx, c], min=hz)
            for c, idx, t_on, t_off in sil_list:
                if t_on <= t < t_off:
                    brain.silenced[idx, c] = True
            if step > 0:
                sums = torch.stack([acc_read[dnL].sum(0), acc_read[dnR].sum(0), acc_read[mn9].mean(0),
                                    acc_read[mdn].mean(0), acc_read[p9].mean(0)], 1).cpu().numpy() / win
                rd += alpha * (sums - rd)
                acc_read.zero_()

            for b in bodies:
                d = b["d"]
                fwd = d.xmat[b["thorax"]].reshape(3, 3)[:2, 0]
                fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
                head = d.xpos[b["eye"], :2]
                # --- food contact, taste stimulation and after-effects
                uneaten = [k for k in range(len(foods)) if k not in b["eaten_set"]]
                on, dists = [], {}
                for k in uneaten:
                    dists[k] = float(np.hypot(foods[k]["x"] - head[0], foods[k]["y"] - head[1]))
                    if dists[k] < FOOD_RADIUS:
                        on.append(k); b["effect_until"][k] = t + foods[k]["effect_s"]
                b["stim"] = [k for k in range(len(foods)) if k in on or b["effect_until"][k] > t]
                b["dist"] = min(dists.values()) if dists else float("nan")
                # --- wind from fans at uneaten foods (sequence mode: only the next one)
                srcs = [k for k in uneaten if foods[k]["fan"] and k not in on]
                if not choice:
                    nxt = min(uneaten) if uneaten else None
                    srcs = [k for k in srcs if k == nxt]
                if b["feeding"]:
                    srcs = []
                b["fans"] = srcs
                jl = jr = 0.0
                for k in srcs:
                    vec = np.array([foods[k]["x"], foods[k]["y"]]) - head
                    dist = float(np.linalg.norm(vec)); up = vec / (dist + 1e-9)
                    theta = math.atan2(fwd[0] * up[1] - fwd[1] * up[0], fwd @ up)
                    s = math.sin(theta / 2)
                    # choice arena: nearer fans dominate sharply, so a small lean toward one food
                    # becomes a commitment instead of walking the gap between two equal fans
                    w = 1.0 if not choice else math.exp(-dist / CHOICE_FALLOFF_MM)
                    jl += w * (1 + s) / 2; jr += w * (1 - s) / 2
                if choice and (jl + jr) > 0:
                    jl, jr = jl / (jl + jr), jr / (jl + jr)
                b["jo"] = (wind_hz * jl, wind_hz * jr)
                for c in b["cols"]:
                    l, r = (b["jo"][1], b["jo"][0]) if cols[c]["swap_jo"] else b["jo"]
                    sens[joL, c] = l; sens[joR, c] = r
                    for k in b["stim"]:
                        sens[food_idx[k], c] = torch.clamp(sens[food_idx[k], c], min=foods[k]["hz"])

                # --- brain(s) -> body
                if step > 0:
                    per = [(rd[c, 0] / refL, rd[c, 1] / refR, rd[c, 4], rd[c, 3]) for c in b["cols"]]
                    b["feeding"] = max(rd[c, 2] for c in b["cols"]) > MN9_FEED_HZ
                    if b["feeding"]:
                        drive = np.zeros(2)
                    elif b["twin"] == "split":  # brain A drives the left legs, brain B the right legs
                        drive = np.array([_drive(*per[0], turn_gain)[0], _drive(*per[1], turn_gain)[1]])
                    else:  # one brain, or two brains averaged
                        drive = _drive(*np.mean(per, axis=0), turn_gain)
                    b["drive"] = drive
                    for k in on:
                        if b["feeding"]:
                            b["feed_t"][k] += win
                            if b["feed_t"][k] >= FEED_TIME:
                                b["eaten"].append([round(t, 2), k]); b["eaten_set"].add(k)
            torch.maximum(sens, manual, out=brain.rate)
            brain._update_driven()

        spk = brain.step().float()
        acc_read += spk
        acc_frame += spk
        for b in bodies:
            if step % CTRL_EVERY == 0:
                obs = HybridControllerObservation.from_sim(b["sim"], b["fly"].name)
                b["action"] = b["ctl"].step(b["drive"], obs)
            apply_locomotion_action(b["sim"], b["fly"].name, b["action"])
            b["sim"].step()

        if step % frame_every == 0:
            fwin = frame_every * dt
            nact = acc_frame.gt(0).sum(0).tolist()
            for b in bodies:
                d = b["d"]
                xq = np.empty((len(b["geoms"]), 7), dtype=np.float32)
                q = np.empty(4)
                for j, g in enumerate(b["geoms"]):
                    mj.mju_mat2Quat(q, d.geom_xmat[g])
                    xq[j, :3] = d.geom_xpos[g]; xq[j, 3:] = q
                b["poses"].append(xq)
                sat = False
                for c in b["cols"]:
                    act = torch.nonzero(acc_frame[:, c]).flatten().cpu().numpy().astype(np.uint32)
                    if len(act) > MAX_ACTIVE_PER_FRAME:
                        act = np.sort(rng.choice(act, MAX_ACTIVE_PER_FRAME, replace=False)); sat = True
                    b["active"][c].append(act)
                xy = d.xpos[b["thorax"], :2].copy()
                if b["last_xy"] is not None:
                    b["path"] += float(np.linalg.norm(xy - b["last_xy"]))
                b["last_xy"] = xy
                cA = b["cols"][0]
                fr = dict(t=round(t, 3), jo=[round(x, 1) for x in b["jo"]],
                          dn=[round(rd[cA, 0] / refL, 3), round(rd[cA, 1] / refR, 3)],
                          drive=[round(float(x), 3) for x in b["drive"]],
                          mn9=round(float(rd[cA, 2]), 1), mdn=round(float(rd[cA, 3]), 1),
                          feeding=b["feeding"], eaten=sorted(b["eaten_set"]), fans=b["fans"], stim=b["stim"],
                          n_act=int(nact[cA]), sat=sat,
                          key={k: round(acc_frame[i, cA].sum().item() / fwin / len(i), 1)
                               for k, i in key_idx.items()})
                if len(b["cols"]) > 1:
                    cB = b["cols"][1]
                    fr.update(dn_b=[round(rd[cB, 0] / refL, 3), round(rd[cB, 1] / refR, 3)],
                              mn9_b=round(float(rd[cB, 2]), 1), mdn_b=round(float(rd[cB, 3]), 1),
                              n_act_b=int(nact[cB]))
                b["frames"].append(fr)
            acc_frame.zero_()
            if step % (frame_every * 5) == 0:
                el = time.time() - t0
                frac = (step + 1) / n_steps
                eta = el / frac * (1 - frac) if frac > 0.01 else float("nan")
                progress(0.05 + 0.93 * frac, f"t = {t:.1f} / {seconds:.0f} s · 남은 시간 약 {eta / 60:.0f}분"
                         if eta == eta else f"t = {t:.1f} s")

    progress(0.99, "결과 저장 중")

    def pack(lst):
        off = np.cumsum([0] + [len(x) for x in lst]).astype(np.uint32)
        return _b64(np.concatenate(lst) if lst else np.zeros(0, np.uint32)), _b64(off)

    out_vars = []
    for vi, (v, b) in enumerate(zip(variants, bodies)):
        fr = b["frames"]
        drives = np.array([f["drive"] for f in fr])
        turn = float(np.mean(drives[:, 1] - drives[:, 0])) if len(fr) else 0.0
        brains = []
        for c in b["cols"]:
            act, off = pack(b["active"][c])
            brains.append(dict(role=cols[c]["role"], active=act, active_off=off, manip_idx=cols[c]["idx"],
                               swap_jo=cols[c]["swap_jo"], manips=cols[c]["manips"]))
        out_vars.append(dict(
            name=v.get("name", f"실험군 {vi + 1}"), color=VARIANT_COLORS[vi % len(VARIANT_COLORS)],
            manips=v.get("manips", []), twin=v.get("twin"), swap_jo=bool(v.get("swap_jo", False)),
            frames=fr, poses=_b64(np.stack(b["poses"])), brains=brains,
            summary=dict(eaten=b["eaten"], first_eat=b["eaten"][0][0] if b["eaten"] else None,
                         path_mm=round(b["path"], 1),
                         final_dist=round(b["dist"], 1) if b["dist"] == b["dist"] else None,
                         turn_bias=round(turn, 3))))
        b["sim"].close()
    return dict(meta=dict(seconds=seconds, fps=fps, dt_ms=brain.p.dt, n_neurons=brain.n,
                          n_synapses=int(brain.W._nnz()), wind_hz=wind_hz, foods=foods,
                          arena="choice" if choice else "sequence",
                          food_radius=FOOD_RADIUS, calib=[refL, refR], brain=vars(params) | {},
                          n_geoms=len(bodies[0]["geoms"]), wall_s=round(time.time() - t0, 1),
                          key=list(key_idx), tags=dict(joL=joL.tolist(), joR=joR.tolist(),
                                                       dnL=dnL.tolist(), dnR=dnR.tolist(), mn9=mn9.tolist())),
                variants=out_vars, config=cfg)
