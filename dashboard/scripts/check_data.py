"""SupeRR Selector dashboard - data check. Must pass before any UI work.

Part A (must pass): every headline number in dashboard/public/data/*.json against superr/reports/final_v2/ (key_numbers.json,
final_top10.csv, alternates.csv, pressure_gap_*.csv, phase8_summary.json, verification.txt), reports/phase5_v2/phase6c_gap_forward.json,
outputs/phase7_exposure_floor/floor_sensitivity.csv and v1 reports/final/final_top10.csv; the simulator against the
handoff's hero_p.json with an independent copy of the exact engine; internal consistency of every deep dive.
Part B (must pass): port check - the ported definitions reproduce the example numbers on the approved design pages for
Tim David, Jasprit Bumrah and the 19th-over bowler list. Those numbers are parsed from the .dc.html files here, never
copied into the data.
"""
import itertools, json, os, re, sys
from functools import lru_cache

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); DASH = os.path.dirname(HERE); PROJECT = os.path.dirname(DASH)
SUPERR = os.path.join(PROJECT, "superr")
FIN2 = os.path.join(SUPERR, "reports", "final_v2"); FIN3 = os.path.join(SUPERR, "reports", "final_v3"); FIN1 = os.path.join(SUPERR, "reports", "final")
DATA = os.path.join(DASH, "public", "data"); HAND = os.path.join(DASH, "_handoff")
# the design pages and hero_p.json live in dashboard/_handoff/, which is not in the repository: without it the checks that read them are skipped
HAVE_DESIGN = all(os.path.exists(os.path.join(HAND, *p)) for p in (("data_builders", "hero_p.json"), ("design", "TimDavid.dc.html"), ("design", "Bumrah.dc.html"), ("design", "Main.dc.html")))
P7 = os.path.join(PROJECT, "outputs", "phase7_exposure_floor")
J = lambda n: json.load(open(os.path.join(DATA, n), encoding="utf-8"))
RES = {"A": [0, 0], "B": [0, 0]}; FAILS = []


def check(part, label, ok, detail=""):
    RES[part][1] += 1
    if ok: RES[part][0] += 1
    else: FAILS.append(f"[{part}] {label} {detail}")


def close(a, b, tol): return a is not None and b is not None and abs(float(a) - float(b)) <= tol


meta, picks, dives, sim, sb, mu, sp = (J(n) for n in ("meta.json", "picks.json", "players.json", "sim.json", "sandbox.json", "matchups.json", "specialists.json"))
kn = json.load(open(os.path.join(FIN3, "key_numbers.json")))
s8 = json.load(open(os.path.join(FIN3, "phase9_summary.json")))
fw6c = json.load(open(os.path.join(SUPERR, "reports", "phase5_v2", "phase6c_gap_forward.json")))
t2 = pd.read_csv(os.path.join(FIN3, "final_top10.csv")); a2 = pd.read_csv(os.path.join(FIN3, "alternates.csv"))
t2v = pd.read_csv(os.path.join(FIN2, "final_top10.csv"))   # the v2 list (history: v1 -> v2 -> v3)
t1 = pd.read_csv(os.path.join(FIN1, "final_top10.csv"))
pg = pd.read_csv(os.path.join(FIN3, "pressure_gap_top10.csv")); pa = pd.read_csv(os.path.join(FIN3, "pressure_gap_all_eligible.csv"))
fs = pd.read_csv(os.path.join(P7, "floor_sensitivity.csv"))
ver = [l.split("\t")[0] for l in open(os.path.join(FIN3, "verification.txt"), encoding="utf-8").read().splitlines() if l]
SIDE = {"batter": "bat", "bowler": "bowl"}
disp = lambda n: str(n).replace("Namboori ", "")

# ======================================================================== A1 picks
for role, R in picks["roles"].items():
    P = R["players"]; top = [p for p in P if p["group"] == "top 10"]; alt = [p for p in P if p["group"] == "alternate"]
    ref = t2[t2.role == role].sort_values("rank (eligible)")
    check("A", f"{role} top 10 names and order", [p["name_full"] for p in top] == list(ref.player))
    check("A", f"{role} top 10 = key_numbers final_top10", [p["name_full"] for p in top] == [x["player"] for x in kn["final_top10"][SIDE[role]]])
    for p, (_, x) in zip(top, ref.iterrows()):
        for k, c in (("cautious", "cautious rating (pts)"), ("best", "best estimate (pts)"), ("range_lo", "80% range low"),
                     ("range_hi", "80% range high"), ("pct_top10", "% of re-rankings in top 10"), ("strict_record", "strict-set pressure record (pts)")):
            check("A", f"{role} {p['name']} {k}", close(p[k], x[c], 1e-9), f"{p[k]} vs {x[c]}")
        check("A", f"{role} {p['name']} rank", p["rank"] == x["rank (eligible)"])
        check("A", f"{role} {p['name']} high-pressure balls = strict-set balls", p["high_pressure_balls"] == x["strict-set balls"])
        check("A", f"{role} {p['name']} high-pressure balls >= floor", p["high_pressure_balls"] >= 50)
    ka = kn["alternates"][SIDE[role]]
    check("A", f"{role} alternates = the next two eligible ranks (11, 12)", [p["name_full"] for p in alt] == [x["player"] for x in ka] == list(a2[a2.role == role].player) and [p["rank"] for p in alt] == [11, 12])
    for p, x in zip(alt, ka):
        check("A", f"{role} alternate {p['name']} rank / %", p["rank"] == x["rank"] and close(p["pct_top10"], x["pct_top10"], 1e-9))
    for p in P:
        g = pg[(pg.role == role) & (pg.player_id == p["player_id"])].iloc[0]
        q = p["pressure"]
        for k, c in (("normal", "normal_value"), ("pressure", "pressure_value"), ("gap", "gap"), ("lo80", "gap_lo80"), ("hi80", "gap_hi80")):
            check("A", f"{role} {p['name']} pressure {k}", close(q[k], g[c], 1e-12))
        check("A", f"{role} {p['name']} verdict", q["verdict"] == g.verdict)
        check("A", f"{role} {p['name']} verdict follows interval",
              q["verdict"] == ("better under pressure" if q["lo80"] > 0 else "worse under pressure" if q["hi80"] < 0 else "no clear difference"))
        check("A", f"{role} {p['name']} package_alternate flag", (g.package_alternate == "yes") == (p["group"] == "alternate"))
        check("A", f"{role} {p['name']} has initials and no photo", len(p["initials"]) == 2 and "photo" not in p)
    # field strip + verdict counts
    el = pa[(pa.role == role) & (pa.group == "eligible")]
    check("A", f"{role} field strip = all eligible", sorted(x["player_id"] for x in R["field"]) == sorted(el.player_id.astype(int)))
    check("A", f"{role} field strip size = eligibility count", len(R["field"]) == kn["eligibility"][SIDE[role]]["eligible"])
    fid = {x["player_id"]: x for x in R["field"]}
    check("A", f"{role} every pick sits in the field strip", all(close(fid[p["player_id"]]["gap"], p["pressure"]["gap"], 1e-12) for p in P))
    for grp, members in (("top 10", top), ("all eligible", R["field"])):
        cnt = pd.Series([(m["pressure"]["verdict"] if "pressure" in m else m["verdict"]) for m in members]).value_counts()
        for v, d_ in s8["verdicts"][role][grp].items():
            if v != "players":
                check("A", f"{role} {grp} verdict count '{v}'", int(cnt.get(v, 0)) == d_["n"] == meta["verdicts"][role][grp][v]["n"])
    lg = pa[(pa.role == role) & (pa.group == "league reference")].iloc[0]
    check("A", f"{role} league reference row", close(R["league"]["gap"], lg.gap, 1e-12) and R["league"]["verdict"] == lg.verdict)
check("A", "unit wording", picks["unit"] == meta["unit"] == "win-probability points per average-importance ball" == s8["unit"] == pg.unit.iloc[0])

# ======================================================================== A2 meta
for s in ("bat", "bowl"):
    check("A", f"eligibility {s}", meta["eligibility"][s] == kn["eligibility"][s])
    for k in ("eligible only, >= 20 each period", "pressure balls >= 20 each period (main)"):
        r = fw6c[s][k]; m = meta["forward_test"][{"bat": "batter", "bowl": "bowler"}[s]][k]
        check("A", f"forward test {s} {k}", m["players"] == r["players"] and all(close(a, b, 1e-12) for a, b in zip((m["rho"], m["lo95"], m["hi95"]), r["gap_rho"])))
    check("A", f"forward test {s} not confirmed", meta["forward_confirmed"][{"bat": "batter", "bowl": "bowler"}[s]] == fw6c[s]["confirmed"] == False)
check("A", "no-importance line matches the data", f"{meta['balls']['without_importance_pct']:.1f}%" in meta["how_it_works_lines"]["no_importance"] and 3 <= meta["balls"]["without_importance_pct"] <= 4)
check("A", "verification count", meta["verification"] == {"passed": sum(v == "PASS" for v in ver), "total": len(ver)} and meta["verification"]["passed"] == meta["verification"]["total"])
for role in ("batter", "bowler"):
    o, n = set(map(disp, t1[t1.role == role].player)), set(map(disp, t2v[t2v.role == role].player))
    sw = meta["changes"]["swaps"][role]
    check("A", f"{role} v1 -> v2 swaps", {x["name"] for x in sw["out"]} == o - n and {x["name"] for x in sw["in"]} == n - o)
    for x in sw["out"]: check("A", f"{role} {x['name']} left because under the floor", x["high_pressure_balls"] < 50)
for row in meta["changes"]["sensitivity"]:
    g = fs[fs.floor == row["floor"]]
    for role in ("batter", "bowler"):
        check("A", f"sensitivity floor {row['floor']} {role}", row[role]["eligible"] == (g.role == role).sum()
              and row[role]["top10"] == [disp(p) for p in g[g.role == role].sort_values("rank").head(10).player])
s75 = next(r for r in meta["changes"]["sensitivity"] if r["floor"] == 75)
check("A", "floor 75: Mohsin Khan leaves the bowling top 10 (no '50 = 75' claim possible)", "Mohsin Khan" in s75["bowler"]["out"])

# ======================================================================== A3 sandbox
for fl in sb["floors"]:
    for role in ("batter", "bowler"):
        L = sb["lists"][f"{role}|50|5|cautious|{fl}|0"]
        ref = fs[(fs.floor == fl) & (fs.role == role)].sort_values("rank")
        check("A", f"sandbox ours floor {fl} {role} = floor_sensitivity (top 12, order)", [x["player_id"] for x in L["top"]] == list(ref.player_id.head(12)))
        check("A", f"sandbox ours floor {fl} {role} eligible count", L["eligible"] == len(ref))
for role in ("batter", "bowler"):
    check("A", f"sandbox floor 50, rule off {role} = final_v2", [x["name"] for x in sb["lists"][f"{role}|50|5|cautious|50|0"]["top"][:10]] == list(map(disp, t2v[t2v.role == role].sort_values("rank (eligible)").player)))
    check("A", f"sandbox floor 50, rule on {role} = final_v3 (ours)", [x["name"] for x in sb["lists"][f"{role}|50|5|cautious|50|1"]["top"][:10]] == list(map(disp, t2[t2.role == role].sort_values("rank (eligible)").player))
          and [x["name"] for x in sb["lists"][f"{role}|50|5|cautious|50|1"]["top"][10:12]] == list(map(disp, a2[a2.role == role].sort_values("rank (eligible)").player))
          and sb["lists"][f"{role}|50|5|cautious|50|1"]["eligible"] == kn["eligibility"][SIDE[role]]["eligible"])
    check("A", f"sandbox floor 0, rule off {role} = v1 final", [x["name"] for x in sb["lists"][f"{role}|50|5|cautious|0|0"]["top"][:10]] == list(map(disp, t1[t1.role == role].sort_values("rank (eligible)").player)))
check("A", "sandbox ours = floor 50, rule on", sb["ours"] == {"own": 50, "m": 5, "key": "cautious", "floor": 50, "pp": 1})
check("A", "sandbox has every combination", len(sb["lists"]) == 2 * 3 * 3 * 2 * 5 * 2)

# ======================================================================== A4 simulator
hero = json.load(open(os.path.join(HAND, "data_builders", "hero_p.json"))) if HAVE_DESIGN else None
C = sim["outcomes"]; RUN = dict(zip(C[:6], sim["runs"]))


def chase(pX, lg, b0, w0, r0):  # independent copy of the exact engine (Main.dc.html / sim19.py)
    @lru_cache(None)
    def V(b, w, r):
        if r <= 0: return 1.0
        if b == 0 or w == 0: return 0.5 if r == 1 else 0.0
        p = pX if b > 6 else lg
        v = sum(p[i] * V(b - 1, w, r - RUN[C[i]]) for i in range(6))
        return v + p[6] * V(b - 1, w - 1, r) + (p[7] + p[8]) * V(b, w, r - 1)
    return V(b0, w0, r0)


lg = sim["league"]
if HAVE_DESIGN: check("A", "league death odds = hero_p.json", all(close(a, b, 1e-5) for a, b in zip(lg, hero["league"])))
base = 100 * (1 - chase(lg, lg, 12, 6, 18))
check("A", "average death bowler, 18 off 12, 6 wkts = 33.0%", round(base, 1) == 33.0 == sim["base_defend"]["18_6"], f"{base:.3f}")
if HAVE_DESIGN: check("A", "same with the published league odds = 33.0%", round(100 * (1 - chase(hero["league"], hero["league"], 12, 6, 18)), 1) == 33.0)
check("A", "simulator bowlers = top 10 + the two bowling alternates", [b["name"] for b in sim["bowlers"]] == [p["name"] for p in picks["roles"]["bowler"]["players"]])
for b in sim["bowlers"]:
    on_design = HAVE_DESIGN and b["name"] in hero["P"]       # bowlers added after the design pages (v3 alternates) have no published odds to port-check
    hp = hero["P"][b["name"]] if on_design else b["p"]        # without a published entry the defend % is checked on the stored odds
    if on_design: check("A", f"sim {b['name']} odds = hero_p.json (K = 120)", all(close(x, y, 1e-5) for x, y in zip(b["p"], hp)))
    n, cnt = b["n"], [b["counts"][k] for k in C]
    check("A", f"sim {b['name']} odds = shrinkage of counts", all(close(p_, (c + 120 * l) / (n + 120), 1e-5) for p_, c, l in zip(b["p"], cnt, lg)))
    for s in sim["situations"]:
        key = f"{s['need']}_{s['wk']}"
        pub = round(100 * (1 - chase(hp, hero["league"] if HAVE_DESIGN else lg, 12, s["wk"], s["need"])), 1)
        check("A", f"sim {b['name']} defend {key} = published", b["defend"][key] == pub, f"{b['defend'][key]} vs {pub}")

# ======================================================================== A5 deep dives
def check_grounds(nm, dv, low_is_good):
    """Grounds rule: 3 best against par + 1 worst against par among qualifying grounds; each shows its ball count."""
    G, n = dv["grounds"], dv["grounds_qualifying"]
    best = [g for g in G if g["tag"] == "best"]; worst = [g for g in G if g["tag"] == "worst"]
    good = (lambda g: -g["vs_par"]) if low_is_good else (lambda g: g["vs_par"])
    check("A", f"dive {nm} grounds: 3 best + 1 worst (or all if < 4)", len(best) == min(3, n) and len(worst) == (1 if n > 3 else 0) and [g["tag"] for g in G] == ["best"] * len(best) + ["worst"] * len(worst))
    check("A", f"dive {nm} grounds: best are ordered and beat the worst", all(good(a) >= good(b) for a, b in zip(best, best[1:])) and all(good(b) >= good(w) for b in best for w in worst))
    check("A", f"dive {nm} grounds: ball count shown", all(isinstance(g["balls"], int) and g["balls"] > 0 for g in G))



for role, R in picks["roles"].items():
    for p in R["players"]:
        dv = dives[str(p["player_id"])]; nm = p["name"]
        check("A", f"dive {nm} exists with right role", dv["role"] == role and dv["rank"] == p["rank"])
        reg = dv["regions"]
        check("A", f"dive {nm} has 9 regions", [r["name"] for r in reg] == ["Third man", "Point", "Cover", "Long-off", "Straight", "Long-on", "Midwicket", "Square leg", "Fine leg"])
        check("A", f"dive {nm} region boundary runs = 4x4s + 6x6s", all(r["boundary_runs"] == 4 * r["fours"] + 6 * r["sixes"] for r in reg))
        check("A", f"dive {nm} region caught <= outs", all(r["caught"] <= r["outs"] for r in reg))
        check("A", f"dive {nm} 20 overs", [o["over"] for o in dv["overs"]] == list(range(1, 21)))
        if role == "batter":
            tot = dv["region_totals"]
            check("A", f"dive {nm} region totals", tot["fours"] <= dv["fours"] and tot["sixes"] <= dv["sixes"] and tot["runs"] <= dv["runs"])
            bf = max(itertools.combinations(range(9), 5), key=lambda c: sum(reg[i]["boundary_runs"] for i in c))
            check("A", f"dive {nm} best field is the best 5 regions", dv["best_field"]["boundary_runs"] == sum(reg[i]["boundary_runs"] for i in bf)
                  == sum(r["boundary_runs"] for r in reg if r["name"] in dv["best_field"]["regions"]) and len(dv["best_field"]["regions"]) == 5)
            check("A", f"dive {nm} grounds >= 30 balls", all(g["balls"] >= 30 for g in dv["grounds"]))
            check_grounds(nm, dv, low_is_good=False)
            check("A", f"dive {nm} opponents >= 15 balls", all(o["balls"] >= 15 for o in dv["toughest"] + dv["easiest"]))
            check("A", f"dive {nm} 6 bowling styles", len(dv["styles"]) == 6)
        else:
            ov = dv["overs"]; sav = {o["over"]: o["saving"] for o in ov}
            for o in ov:
                ok = (o["saving"] == 0) if not o["balls"] else close(o["saving"], (o["par_econ"] - o["econ"]) * o["balls"] / (o["balls"] + 60), 0.02)
                check("A", f"dive {nm} over {o['over']} saving = (par - econ) x b/(b+60)", ok)
            best = max((c for c in itertools.combinations(range(1, 21), 4) if all(b - a > 1 for a, b in zip(c, c[1:]))), key=lambda c: sum(sav[o] for o in c))
            bp = dv["planner"]["best"]
            check("A", f"dive {nm} best plan is the best 4 non-adjacent overs", close(bp["saving"], sum(sav[o] for o in best), 1e-3)
                  and all(b - a > 1 for a, b in zip(bp["overs"], bp["overs"][1:])) and len(bp["overs"]) == 4)
            check("A", f"dive {nm} grounds >= 60 balls", all(g["balls"] >= 60 for g in dv["grounds"]))
            check_grounds(nm, dv, low_is_good=True)
            check("A", f"dive {nm} opponents >= 15 balls", all(o["balls"] >= 15 for o in dv["owns"] + dv["troubled_by"]))
            check("A", f"dive {nm} 5 delivery types, usage <= 100%", len(dv["deliveries"]) == 5 and sum(v["usage_pct"] for v in dv["deliveries"]) <= 100.5)
            check("A", f"dive {nm} RHB + LHB", [h["hand"] for h in dv["hands"]] == ["RHB", "LHB"])
            simb = next(b for b in sim["bowlers"] if b["player_id"] == p["player_id"])
            check("A", f"dive {nm} death balls = simulator death balls", dv["death"]["balls"] == simb["death_balls"])

# ======================================================================== A5b callouts, usual overs, flags (Phase 4)
from decimal import Decimal as _D, ROUND_HALF_UP as _HU
REQ = {"region": 30, "over": 24, "style": 30, "opponent": 15, "ground_batter": 30, "ground_bowler": 60, "outs": 15, "wickets": 15,
       "hand": 30, "delivery": 30, "advice": 100}                     # the minimum each kind of claim needs (user rule + existing minimums)
NUM = re.compile(r"\d+(?:\.\d+)?")


def _ev(P, expr):
    return eval(expr, {"__builtins__": {}, "P": P, "sum": sum, "len": len, "max": max, "min": min, "abs": abs})


def _fmt(v, f):
    return str(_D(repr(float(v))).quantize(_D({"i": "1", "1": "0.1", "2": "0.01"}[f]), rounding=_HU))


n_callouts = 0
for pid, P in dives.items():
    nm = P["name"]
    for c in P["callouts"]:
        n_callouts += 1
        found = NUM.findall(c["text"])
        want = [_fmt(_ev(P, n["expr"]), n["fmt"]) for n in c["nums"]]
        check("A", f"callout {nm} {c['id']}: every number equals players.json", found == want, f"text {found} vs data {want}: {c['text']}")
        for g in c["guards"]:
            check("A", f"callout {nm} {c['id']}: guard {g['kind']} >= {REQ[g['kind']]}", g["min"] >= REQ[g["kind"]] and _ev(P, g["expr"]) >= g["min"],
                  f"{g['expr']} = {_ev(P, g['expr'])}, min {g['min']}")
        kinds_needed = {"BAT_WHEEL": "region", "BOWL_WHEEL": "region", "BAT_OVER": "over", "BOWL_OVER": "over", "BAT_STYLE": "style",
                        "BAT_OPP": "opponent", "BOWL_OPP": "opponent", "GROUND": "ground_", "BAT_OUT": "outs", "BOWL_WT": "wickets",
                        "BOWL_DEL": "delivery", "BOWL_HANDS": "hand", "FIELD_BEST": "region", "PLAN_BEST": "over"}
        need = next(v for k, v in kinds_needed.items() if c["id"].startswith(k))
        check("A", f"callout {nm} {c['id']}: carries a {need} guard", any(g["kind"].startswith(need) for g in c["guards"]))
        if c["text"].startswith(("Bowl ", "Watch ")):
            check("A", f"callout {nm} {c['id']}: advice wording only with >= 100 balls", any(g["kind"] == "advice" and _ev(P, g["expr"]) >= 100 for g in c["guards"]))
        check("A", f"callout {nm} {c['id']}: opponent lines are descriptive", not re.search(r"struggles|owns|gets after|covered", c["text"]))
    slots = {c["slot"] for c in P["callouts"]}
    check("A", f"callouts {nm}: no empty placeholder text", all(c["text"].strip() and "{" not in c["text"] for c in P["callouts"]))
    if P["role"] == "bowler":
        ov = P["overs"]
        best_u = max((cmb for cmb in itertools.combinations(range(1, 21), 4) if all(b - a > 1 for a, b in zip(cmb, cmb[1:]))),
                     key=lambda cmb: (sum(ov[o - 1]["times"] for o in cmb), [-o for o in cmb]))
        u = P["planner"]["usual"]
        check("A", f"{nm} usual overs = 4 non-adjacent most-bowled overs", u["overs"] == list(best_u) and u["times"] == sum(ov[o - 1]["times"] for o in best_u))
    for k, fl in P["flags"].items():
        check("A", f"{nm} flag {k} = total < min", fl["few"] == (fl["total"] < fl["min"]))
print(f"  callouts checked: {n_callouts} across {len(dives)} players")


# ======================================================================== A5c badges (step A): every wedge and tick equals players.json
import math as _m


def _mix(a, b, t):
    t = max(0.0, min(1.0, t))
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]; cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(_m.floor(x + (y - x) * t + 0.5)):02x}" for x, y in zip(ca, cb))


SECTORS = {"Third man": (110, 160), "Point": (70, 110), "Cover": (35, 70), "Long-off": (11, 35), "Straight": (-11, 11),
           "Long-on": (-35, -11), "Midwicket": (-70, -35), "Square leg": (-110, -70), "Fine leg": (-160, -110)}
_top = max((r["run_share"] or 0) / 100 for d in dives.values() if d["role"] == "batter" for r in d["regions"])
for role, R in picks["roles"].items():
    for p in R["players"]:
        D = dives[str(p["player_id"])]; b = p["badge"]; nm = p["name"]
        check("A", f"badge {nm}: same in picks.json and players.json", b == D["badge"])
        check("A", f"badge {nm}: accessible name", b["label"] == f"{nm}'s badge: " + ("where he scores his runs" if role == "batter" else "which overs he bowls"))
        if role == "batter":
            K = (b["r_in"] and (48.0 - b["r_in"]) / _m.sqrt(_top))
            shares = [(r["run_share"] or 0) / 100 for r in D["regions"]]; lo, hi = min(shares), max(shares)
            ok = len(b["wedges"]) == 9
            for w, r, sh in zip(b["wedges"], D["regions"], shares):
                ok &= w["name"] == r["name"] and (w["from"], w["to"]) == SECTORS[r["name"]] and close(w["share"], sh, 1e-12)
                ok &= close(w["len"], b["r_in"] + K * _m.sqrt(sh), 1e-3) and w["color"] == _mix("#f6a8c2", "#e4007c", (sh - lo) / (hi - lo) if hi > lo else 1)
            check("A", f"badge {nm}: 9 wedges, sector angles, length = r_in + k*sqrt(share), pink scale", ok)
        else:
            balls = [o["balls"] for o in D["overs"]]; mx = max(balls); lo = min(x for x in balls if x)
            ok = len(b["ticks"]) == 20
            for t, o in zip(b["ticks"], D["overs"]):
                ok &= t["over"] == o["over"] and t["balls"] == o["balls"] and close(t["angle"], (o["over"] - 1) * 18, 1e-9)
                if o["balls"]:
                    ok &= not t["empty"] and close(t["len"], b["l_max"] * o["balls"] / mx, 1e-3) and t["color"] == _mix("#728dc8", "#254aa5", (o["balls"] - lo) / (mx - lo) if mx > lo else 1)
                else:
                    ok &= t["empty"] and t["color"] is None
            check("A", f"badge {nm}: 20 ticks, over 1 at top clockwise, length ~ balls, blue scale, outline when never bowled", ok)
sigs = [json.dumps([w["len"] for w in p["badge"].get("wedges", [])] + [t["len"] for t in p["badge"].get("ticks", [])]) for r in picks["roles"].values() for p in r["players"]]
check("A", "badges: no two players look the same", len(set(sigs)) == len(sigs))
check("A", "no photos shipped", not os.path.exists(os.path.join(DASH, "public", "photos")))


# ======================================================================== A6 head-to-head + specialists
for side, key in (("bat", "batters"), ("bowl", "bowlers")):
    T = pd.read_csv(os.path.join(SUPERR, "out", "model_v3", f"phase4_rankings_{side}_all_players_technical.csv"))
    ref = list(T[T.eligible].nsmallest(25, "rank_eligible").player_id)
    check("A", f"head-to-head {key} = top 25 of the v3 pool", [x["player_id"] for x in mu[key]] == ref)
    el = set(T[T.eligible].player_id)
    rows = [r for r in sp["rows"] if r["role"] == {"bat": "batter", "bowl": "bowler"}[side]]
    check("A", f"specialists {key} all v3 eligible", len(rows) > 0 and all(r["player_id"] in el for r in rows))
    check("A", f"specialists {key} meet each chip's minimum", all(r["balls"] >= r["min_balls"] for r in rows))
ids_b = {x["player_id"] for x in mu["batters"]}; ids_w = {x["player_id"] for x in mu["bowlers"]}
check("A", "head-to-head pairs within the rosters", all(p["batter_id"] in ids_b and p["bowler_id"] in ids_w for p in mu["pairs"]))
check("A", "head-to-head small-sample rule 12 balls", mu["min_balls"] == 12)

# ======================================================================== A7 the pressure-positive rule (v3), checked independently
PR = meta["changes"]["pressure_rule"]
for side, role in (("bat", "batter"), ("bowl", "bowler")):
    T2 = pd.read_csv(os.path.join(SUPERR, "out", "model_v2", f"phase4_rankings_{side}_all_players_technical.csv"), index_col=0)
    T3 = pd.read_csv(os.path.join(SUPERR, "out", "model_v3", f"phase4_rankings_{side}_all_players_technical.csv"), index_col=0)
    e2 = T2[T2.eligible == True]
    pv = 100 * e2.raw_value          # the strict-set pressure record: 100 x sum(v_adj) / sum(LI), LI >= 1.5 (Phase 6b definition)
    passes = pv > 0
    expect = e2[passes].sort_values("rank_eligible")     # the frozen v2 ordering with the rule applied and nothing else changed
    got = T3[T3.eligible == True].sort_values("rank_eligible")
    check("A", f"{role}s v3 pool = v2 floor-50 pool with pressure value > 0, same order", list(got.index) == list(expect.index) and list(got.rank_eligible) == list(range(1, len(got) + 1)))
    check("A", f"{role}s every v3 player has pressure value > 0 (picks.json too)", bool((100 * got.raw_value > 0).all()) and all(p["pressure"]["pressure"] > 0 for p in picks["roles"][role]["players"]))
    check("A", f"{role}s every player removed by the rule has pressure value <= 0", bool((pv[~passes] <= 0).all()) and len(pv[~passes]) == kn["eligibility"][side]["fail_pressure_rule"] == PR["failed"][role])
    check("A", f"{role}s rule applied to everyone alike: nobody outside the floor-50 pool is added", set(got.index) <= set(e2.index))
    check("A", f"{role}s pool sizes", PR["pool_before"][role] == len(e2) == kn["eligibility"][side]["eligible_floor50_before_pressure_rule"] and PR["pool_after"][role] == len(got))
    v2top = list(e2.sort_values("rank_eligible").index)[:10]
    v3top = list(got.index[:10])
    check("A", f"{role}s left / entered match the lists", [x["name"] for x in PR["left"][role]] == [disp(T2.loc[i, "name"]) for i in v2top if i not in v3top]
          and [x["name"] for x in PR["entered"][role]] == [disp(T3.loc[i, "name"]) for i in v3top if i not in v2top])
    check("A", f"{role}s 'with' and 'without' lists on the page", PR["lists"][role]["without_rule"]["top10"] == [disp(T2.loc[i, "name"]) for i in v2top]
          and PR["lists"][role]["with_rule"]["top10"] == [disp(T3.loc[i, "name"]) for i in v3top])
    for x in PR["left"][role]:
        check("A", f"{role} {x['name']} left because his pressure value is not above average", x["pressure_value"] <= 0)
    check("A", f"{role}s 12 picks, all above average under pressure", len(picks["roles"][role]["players"]) == 12 and all(p["pressure"]["pressure"] > 0 for p in picks["roles"][role]["players"]))
check("A", "Tilak Varma is still player 7897", any(p["player_id"] == 7897 and p["name"] == "Tilak Varma" for p in picks["roles"]["batter"]["players"]))
check("A", "nobody removed by the rule is still a pick", not ({x["name"] for r in PR["left"].values() for x in r} & {p["name"] for R in picks["roles"].values() for p in R["players"]}))
fw_ = meta["forward_test"]; FP = "eligible only, >= 20 each period"
check("A", "forward-test headline numbers (floor-50 pool, before the rule): batters +0.086 [-0.212, +0.344], bowlers -0.100 [-0.342, +0.139]",
      all(close(fw_[r][FP][k], v, 5e-4) for r, k, v in (("batter", "rho", 0.086), ("batter", "lo95", -0.212), ("batter", "hi95", 0.344), ("bowler", "rho", -0.100), ("bowler", "lo95", -0.342), ("bowler", "hi95", 0.139))))

# ======================================================================== B port check against the design pages
def js_array(html, name):
    m = re.search(r"const " + name + r" = (\[.*?\]);", html, re.S)
    return json.loads(m.group(1).replace("'", '"'))


if HAVE_DESIGN:
    td = open(os.path.join(HAND, "design", "TimDavid.dc.html"), encoding="utf-8").read()
    jb = open(os.path.join(HAND, "design", "Bumrah.dc.html"), encoding="utf-8").read()
    mn = open(os.path.join(HAND, "design", "Main.dc.html"), encoding="utf-8").read()
    TD = next(v for v in dives.values() if v["name"] == "Tim David"); JB = next(v for v in dives.values() if v["name"] == "Jasprit Bumrah")
    for s in js_array(td, "S"):
        r = next(x for x in TD["regions"] if x["name"] == s[0])
        check("B", f"Tim David wheel {s[0]}", (r["from"], r["to"], r["balls"], r["runs"], r["sixes"], r["outs"]) == tuple(s[1:7])
              and close(r["run_share"], s[7], 0.05) and close(r["sr"], s[8], 0.05) and close(r["league_sr"], s[9], 0.05), f"{r} vs {s}")
    for b in js_array(td, "B"):
        r = next(x for x in TD["regions"] if x["name"] == b[0])
        check("B", f"Tim David field setter {b[0]}", (r["fours"], r["sixes"], r["boundary_runs"], r["caught"]) == tuple(b[3:7]), f"{r} vs {b}")
    for o in js_array(td, "OV"):
        r = TD["overs"][o[0] - 1]
        check("B", f"Tim David over {o[0]}", r["balls"] == o[1] and (o[1] == 0 or (close(r["sr"], o[2], 0.05) and close(r["par_sr"], o[3], 0.05))), f"{r} vs {o}")
    tdtxt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", td))
    page_tot = [int(re.search(p, tdtxt).group(1).replace(",", "")) for p in (r"of ([0-9,]+) runs", r"/ ([0-9]+) fours", r"/ ([0-9]+) sixes", r"/ ([0-9]+) of his catches")]
    check("B", "Tim David field-setter totals", [TD["region_totals"][k] for k in ("boundary_runs", "fours", "sixes", "caught")] == page_tot, f"vs page {page_tot}")
    for s in js_array(jb, "S"):
        r = next(x for x in JB["regions"] if x["name"] == s[0])
        check("B", f"Bumrah wheel {s[0]}", (r["balls"], r["runs"], r["sixes"], r["outs"]) == tuple(s[3:7]) and close(r["run_share"], s[7], 0.05)
              and close(r["sr"], s[8], 0.05) and close(r["league_sr"], s[9], 0.05), f"{r} vs {s}")
    for o in js_array(jb, "OV"):
        r = JB["overs"][o[0] - 1]
        check("B", f"Bumrah over {o[0]}", r["balls"] == o[1] and r["wkts"] == o[4] and (o[1] == 0 or (close(r["econ"], o[2], 0.005) and close(r["par_econ"], o[3], 0.005))), f"{r} vs {o}")
    m = re.search(r"const BEST = (\[.*?\]), BEST_V = ([0-9.]+);", jb)
    check("B", "Bumrah best plan", JB["planner"]["best"]["overs"] == json.loads(m.group(1)) and round(JB["planner"]["best"]["saving"], 1) == float(m.group(2)),
          f"{JB['planner']['best']} vs {m.group(1)} {m.group(2)}")
    info = json.loads(re.search(r"const INFO = (\{.*?\});", mn, re.S).group(1).replace("'", '"'))
    check("B", "bowlers missing from the design page's 19th-over list are only v3 alternates", all(b["group"] == "alternate" for b in sim["bowlers"] if b["name"] not in info))
    for b in [b for b in sim["bowlers"] if b["name"] in info]:
        i = info.get(b["name"])
        check("B", f"19th-over list {b['name']}", i is not None and b["rank"] == i[0] and close(b["death_econ"], float(i[1]), 0.005) and close(b["dot_pct"], float(i[2]), 0.05)
              and close(b["boundary_pct"], float(i[3]), 0.05) and close(b["balls_per_wkt"], float(i[4]), 0.05) and b["death_balls"] == i[5], f"{b} vs {i}")

# ======================================================================== report
print("CHECK_DATA")
print(f"  Part A  headline numbers vs final_v3 / CSVs / engine / consistency: {RES['A'][0]} of {RES['A'][1]} pass")
print(f"  Part B  port check vs the design pages' example numbers:            {RES['B'][0]} of {RES['B'][1]} pass" if HAVE_DESIGN
      else "  Part B  port check: design files not present (dashboard/_handoff/ is not in the repository), skipped; hero_p.json comparisons skipped too")
print(f"  Simulator: average death bowler, 18 off 12, 6 wkts = {base:.2f}% -> {round(base, 1)}%")
for b in sim["bowlers"]:
    print(f"    {b['name']:20s} defend 18/6 {b['defend']['18_6']:5.1f}%  24/6 {b['defend']['24_6']:5.1f}%  14/4 {b['defend']['14_4']:5.1f}%  ({b['death_balls']} death balls)")
for f in FAILS[:60]:
    print("  FAIL", f)
ok = not FAILS
print("RESULT:", "PASS" if ok else f"FAIL ({len(FAILS)})")
sys.exit(0 if ok else 1)
