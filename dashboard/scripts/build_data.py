"""SupeRR Selector dashboard - data layer.

Reads the frozen v3 package (superr/reports/final_v3, superr/out/model_v3; v2 and v1 only for the change history), the ball file (superr/out/deliveries_clean.csv),
the Phase 3 ball values (superr/out/model/phase3_ball_values_adj.parquet), outputs/phase7_exposure_floor/floor_sensitivity.csv
Writes dashboard/public/data/*.json. Player badges are generated from each player's own numbers (scripts/badges.py); no images.

Definitions are ported from dashboard/_handoff/data_builders/ (design_stats.py, field_sim.py, sim19.py, build_matchups.py,
build_specialists.py) and the approved design pages:
  - field regions: from the stroke recorded on each ball (field_sim.py SEC map); defensive shots and leaves are left out
  - par: every player on the same kind of ball (same region / over / style / ground / delivery type / hand)
  - over by over: full 20-over innings only (scheduled_balls == 120), as on the design pages
  - small-sample rules: grounds >= 30 balls (batters) / >= 60 (bowlers); opponents >= 15 balls; matchup / heat cells < 12 balls
    are "too few"; over tiles outlined under 20 (batters) / 24 (bowlers) balls; planner "few balls" under 60
  - simulator: death-over outcome counts shrunk toward the league with K = 120; 19th by the bowler, 20th by the league
  - over planner saving: (par - his economy) x balls / (balls + 60); best plan = best 4 non-adjacent overs (exhaustive)
  - best field: the 5 regions holding most of his boundary runs (exhaustive over all 5-region sets)
Nothing here is copied from the design pages; their numbers are only used as a port check in check_data.py.
"""
import itertools, json, os, sys
from datetime import date
from functools import lru_cache

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from callouts import finish_dive  # noqa: E402  (role label, few-balls flags, callouts from templates)
from badges import bat_k, make_badge  # noqa: E402  (generated player badges)

HERE = os.path.dirname(os.path.abspath(__file__))
DASH = os.path.dirname(HERE)
PROJECT = os.path.dirname(DASH)
SUPERR = os.path.join(PROJECT, "superr")
FIN2 = os.path.join(SUPERR, "reports", "final_v2")
FIN3 = os.path.join(SUPERR, "reports", "final_v3")
FIN1 = os.path.join(SUPERR, "reports", "final")
MODEL3 = os.path.join(SUPERR, "out", "model_v3")
P7 = os.path.join(PROJECT, "outputs", "phase7_exposure_floor")
HANDOFF = os.path.join(DASH, "_handoff")
OUT = os.path.join(DASH, "public", "data")
ROLE = {"bat": "batter", "bowl": "bowler"}
UNIT = "win-probability points per average-importance ball"
K_SIM = 120
MIN_CELL = 12

# ---------------------------------------------------------------- definitions (ported)
REGIONS = [  # name, from, to (degrees; 0 = straight, clockwise = off side; drawn for a right-hander), strokes (field_sim.py)
    ("Third man", 110, 160, ["Late Cut", "Steer", "Reverse Paddle Scoop"]),
    ("Point", 70, 110, ["Square Cut", "Square drive", "UpperCut", "Reverse Sweep", "Switch"]),
    ("Cover", 35, 70, ["Cover Drive", "Inside out"]),
    ("Long-off", 11, 35, ["Off Drive"]),
    ("Straight", -11, 11, ["Straight Drive"]),
    ("Long-on", -35, -11, ["On drive"]),
    ("Midwicket", -70, -35, ["Flick", "Worked", "Slog Sweep", "Slog"]),
    ("Square leg", -110, -70, ["Pull shot", "Hook Shot", "Sweep Shot"]),
    ("Fine leg", -160, -110, ["Glance", "Leg Glance", "Paddle Sweep", "Paddle Scoop", "PickUp"]),
]
SEC = {s: r[0] for r in REGIONS for s in r[3]}
RNAMES = [r[0] for r in REGIONS]
STY = {"Right-arm Fast": "Right-arm pace", "Right-arm fast-medium": "Right-arm pace", "Right-arm medium-fast": "Right-arm pace",
       "Right-arm medium": "Right-arm pace", "Left-arm Fast": "Left-arm pace", "Left-arm fast-medium": "Left-arm pace",
       "Left-arm medium-fast": "Left-arm pace", "Left-arm medium": "Left-arm pace", "Right-arm offbreak": "Off-spin",
       "Right-arm Leg-Break": "Leg-spin", "Slow left-arm orthodox": "Left-arm finger spin", "Left-arm Chinaman": "Left-arm wrist spin"}
STYLES = ["Right-arm pace", "Left-arm pace", "Off-spin", "Leg-spin", "Left-arm finger spin", "Left-arm wrist spin"]
VAR = {"Yorker": "Yorker", "Wide Yorker": "Yorker", "Bouncer": "Bouncer", "Slow Bouncer": "Bouncer", "Full Toss": "Full toss",
       "Beamer": "Full toss", "Off Cutter": "Slower ball / cutter", "Leg Cut": "Slower ball / cutter", "Slow Ball": "Slower ball / cutter",
       "Knuckle": "Slower ball / cutter", "Back of Hand": "Slower ball / cutter", "Seamed In": "Stock: seam & swing",
       "Seamed Out": "Stock: seam & swing", "Inswing": "Stock: seam & swing", "Out Swing": "Stock: seam & swing",
       "Straight": "Stock: seam & swing", "Angled In": "Stock: seam & swing", "Angled Away": "Stock: seam & swing",
       "Reverse Swing": "Stock: seam & swing", "Reverse Swing In": "Stock: seam & swing", "Reverse Swing Out": "Stock: seam & swing",
       "Quicker Ball": "Stock: seam & swing"}
VARS = ["Stock: seam & swing", "Yorker", "Slower ball / cutter", "Bouncer", "Full toss"]
DISMISS = {"Caught": "Caught", "Bowled": "Bowled", "LBW": "LBW", "Stump": "Stumped", "Hit Wicket": "Hit wicket"}
SIM_C = ["0", "1", "2", "3", "4", "6", "W", "WD", "NB"]
RUN = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "6": 6}
SIM_SITUATIONS = [(18, 6), (24, 6), (14, 4)]


def disp(name):
    return str(name).replace("Namboori ", "")


def r1(x): return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 1)
def r2(x): return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 2)
def r4(x): return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)
def sr(runs, balls): return r1(100 * runs / balls) if balls else None
def econ(runs, balls): return r2(6 * runs / balls) if balls else None


def jdump(name, obj):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))


# ---------------------------------------------------------------- load
def load():
    cols = ["match_id", "innings", "innings_ball_num", "overid", "striker_id", "striker_name", "bowler_id", "bowler_name",
            "batting_type", "bowling_type", "pace_or_spin", "delivery_style", "stroke", "phase", "bat_runs", "team_runs",
            "batter_ball_faced", "is_legal", "is_wide", "is_noball", "is_wicket", "bowler_wicket", "dismissal_type",
            "dismissed_player_id", "bowler_runs_conceded", "is_super_over", "is_four", "is_six", "scheduled_balls"]
    d = pd.read_csv(os.path.join(SUPERR, "out", "deliveries_clean.csv"), usecols=cols)
    d = d[~d.is_super_over.astype(bool)].copy()
    for c in ["batter_ball_faced", "is_legal", "is_wide", "is_noball", "is_wicket", "bowler_wicket", "is_four", "is_six"]:
        d[c] = d[c].astype(bool)
    g = pd.read_csv(os.path.join(SUPERR, "out", "match_ground.csv"))[["match_id", "ground", "city"]]
    d = d.merge(g, on="match_id", how="left")
    a = pd.read_parquet(os.path.join(SUPERR, "out", "model", "phase3_ball_values_adj.parquet"),
                        columns=["match_id", "innings", "innings_ball_num", "li", "wpa"])
    n0 = len(d)
    d = d.merge(a, on=["match_id", "innings", "innings_ball_num"], how="left")
    assert len(d) == n0, "Phase 3 join duplicated rows"
    d["out_s"] = d.bowler_wicket & (d.dismissed_player_id == d.striker_id)
    d["caught"] = d.out_s & d.dismissal_type.astype(str).str.lower().str.contains("caught")
    d["style"] = d.bowling_type.map(STY)
    d["region"] = d.stroke.map(SEC)
    d["var"] = d.delivery_style.map(VAR)
    d["full20"] = d.scheduled_balls == 120
    return d, a


def lists():
    """v3 package (floor 50 + pressure value above average): top 10s, the next two eligible ranks as alternates, technical tables."""
    t2 = pd.read_csv(os.path.join(FIN3, "final_top10.csv"))
    a2 = pd.read_csv(os.path.join(FIN3, "alternates.csv"))
    tech = {s: pd.read_csv(os.path.join(MODEL3, f"phase4_rankings_{s}_all_players_technical.csv")) for s in ROLE}
    return t2, a2, tech


def initials(name):
    parts = [p for p in disp(name).split() if p]
    return (parts[0][0] + parts[-1][0]).upper() if len(parts) > 1 else parts[0][:2].upper()


# ---------------------------------------------------------------- picks + pressure
def picks_and_gaps(t2, a2, tech):
    pg = pd.read_csv(os.path.join(FIN3, "pressure_gap_top10.csv"))
    pa = pd.read_csv(os.path.join(FIN3, "pressure_gap_all_eligible.csv"))
    fs = pd.read_csv(os.path.join(P7, "floor_sensitivity.csv"))
    hp50 = fs[fs.floor == 50].set_index(["role", "player_id"]).high_pressure_balls
    out = {"unit": UNIT, "roles": {}}
    for side, role in ROLE.items():
        T = tech[side].set_index("player_id")
        rows = []
        for grp, src in (("top 10", t2[t2.role == role]), ("alternate", a2[a2.role == role])):
            for _, x in src.sort_values("rank (eligible)").iterrows():
                pid = int(T.index[(T.rank_eligible == x["rank (eligible)"]) & T.eligible].item())
                assert T.loc[pid, "name"] == x.player, (pid, x.player)
                g = pg[(pg.role == role) & (pg.player_id == pid)]
                assert len(g) == 1, (role, x.player)
                g = g.iloc[0]
                rows.append({
                    "player_id": pid, "name": disp(x.player), "name_full": x.player, "group": grp, "rank": int(x["rank (eligible)"]),
                    "initials": initials(x.player),
                    "cautious": float(x["cautious rating (pts)"]), "best": float(x["best estimate (pts)"]),
                    "range_lo": float(x["80% range low"]), "range_hi": float(x["80% range high"]),
                    "pct_top10": float(x["% of re-rankings in top 10"]), "median_rank": float(x["median rank (re-rankings)"]),
                    "own_pct": float(x["own-data %"]), "matches": int(x.matches), "matches_2025_26": int(x["matches 2025-26"]),
                    "high_pressure_balls": int(hp50[(role, pid)]), "strict_record": float(x["strict-set pressure record (pts)"]),
                    "share_pp": float(x["% strict-set LI powerplay"]), "share_mid": float(x["% middle"]), "share_death": float(x["% death"]),
                    "ipl_pct": float(x["% IPL"]), "missed_season": x["flag: missed a full season"],
                    "close_call": x["flag: close call (<50% of re-rankings in top 10)"] == "yes",
                    "pressure": {"normal": float(g.normal_value), "pressure": float(g.pressure_value), "gap": float(g.gap),
                                 "lo80": float(g.gap_lo80), "hi80": float(g.gap_hi80), "verdict": g.verdict,
                                 "balls_normal": int(g.balls_normal), "balls_pressure": int(g.balls_pressure)},
                })
        el = pa[(pa.role == role) & (pa.group == "eligible")]
        lg = pa[(pa.role == role) & (pa.group == "league reference")].iloc[0]
        field = [{"player_id": int(r.player_id), "name": disp(r.player), "gap": float(r.gap), "lo80": float(r.gap_lo80),
                  "hi80": float(r.gap_hi80), "verdict": r.verdict} for r in el.sort_values("gap").itertuples()]
        out["roles"][role] = {"players": rows, "field": field,
                              "league": {"normal": float(lg.normal_value), "pressure": float(lg.pressure_value), "gap": float(lg.gap),
                                         "lo80": float(lg.gap_lo80), "hi80": float(lg.gap_hi80), "verdict": lg.verdict,
                                         "players": int(len(el))}}
    return out


# ---------------------------------------------------------------- changes / meta
def changes_and_meta(tech):
    kn = json.load(open(os.path.join(FIN3, "key_numbers.json")))
    s8 = json.load(open(os.path.join(FIN3, "phase9_summary.json")))
    fs = pd.read_csv(os.path.join(P7, "floor_sensitivity.csv"))
    t1 = pd.read_csv(os.path.join(FIN1, "final_top10.csv")); t2 = pd.read_csv(os.path.join(FIN2, "final_top10.csv"))
    ver = [l.split("\t") for l in open(os.path.join(FIN3, "verification.txt"), encoding="utf-8").read().splitlines() if l]
    hp = fs.set_index(["floor", "role", "player"]).high_pressure_balls
    v23 = pd.read_csv(os.path.join(FIN3, "v2_vs_v3.csv"))
    swaps = {}
    for role in ROLE.values():
        o, n = set(t1[t1.role == role].player), set(t2[t2.role == role].player)
        swaps[role] = {
            "out": [{"name": disp(p), "v1_rank": int(t1[(t1.role == role) & (t1.player == p)]["rank (eligible)"].iloc[0]),
                     "high_pressure_balls": int(hp[(0, role, p)])} for p in sorted(o - n)],
            "in": [{"name": disp(p), "v2_rank": int(t2[(t2.role == role) & (t2.player == p)]["rank (eligible)"].iloc[0]),
                    "high_pressure_balls": int(hp[(50, role, p)])} for p in sorted(n - o)]}
    sens = []
    base = {r: list(fs[(fs.floor == 0) & (fs.role == r)].sort_values("rank").head(10).player) for r in ROLE.values()}
    for f, g in fs.groupby("floor"):
        row = {"floor": int(f)}
        for r in ROLE.values():
            top = list(g[g.role == r].sort_values("rank").head(10).player)
            row[r] = {"eligible": int((g.role == r).sum()), "top10": [disp(p) for p in top],
                      "out": [disp(p) for p in base[r] if p not in top], "in": [disp(p) for p in top if p not in base[r]]}
        sens.append(row)
    fw = {}
    for role, d_ in s8["forward_test"].items():
        fw[role] = {k: {"players": v["players"], "rho": v["gap_rho"][0], "lo95": v["gap_rho"][1], "hi95": v["gap_rho"][2]}
                    for k, v in d_.items()}
    verdicts = s8["verdicts"]
    pressure_rule = {"decided": "18 November", "left": {}, "entered": {}, "failed": {}, "pool_before": {}, "pool_after": {},
                     "lists": {r: {w: {k: [disp(n) for n in v] for k, v in s8["lists"][w][r].items()} for w in ("without_rule", "with_rule")} for r in ROLE.values()}}
    for side, role in ROLE.items():
        x = v23[v23.role == role]
        pressure_rule["left"][role] = [{"name": disp(r.player), "v2_rank": int(r.v2_rank), "pressure_value": float(r.pressure_value), "high_pressure_balls": int(r.high_pressure_balls)}
                                       for r in x[x.status == "left the top 10"].itertuples()]
        pressure_rule["entered"][role] = [{"name": disp(r.player), "v2_rank": int(r.v2_rank), "v3_rank": int(r.v3_rank), "pressure_value": float(r.pressure_value)}
                                          for r in x[x.status == "entered the top 10"].itertuples()]
        e = kn["eligibility"][side]
        pressure_rule["failed"][role] = e["fail_pressure_rule"]; pressure_rule["pool_before"][role] = e["eligible_floor50_before_pressure_rule"]
        pressure_rule["pool_after"][role] = e["eligible"]
    meta = {
        "title": "SupeRR Selector · Season 2, Round 1", "data_range": "2022–26", "last_built": date.today().isoformat(),
        "unit": UNIT, "rating_unit": "win-probability points per average ball", "floor": 50, "li_pressure": 1.5,
        "rules": {"own_share_min_pct": 50, "matches_2025_26_min": 5, "high_pressure_balls_min": 50, "pressure_value_above_average": True},
        "eligibility": kn["eligibility"],
        "verification": {"passed": sum(v[0] == "PASS" for v in ver), "total": len(ver)},
        "forward_test": fw, "forward_confirmed": {ROLE[s]: bool(kn["pressure_gap_forward_test"][s]["confirmed"]) for s in ROLE},
        "verdicts": verdicts, "chance_share_pct": 20,
        "changes": {"swaps": swaps, "sensitivity": sens, "pressure_rule": pressure_rule},
        "data_source": "Supplied SupeRR ball-by-ball data pack (2022–26, IPL and T20 internationals), matched to Cricsheet; "
                       "weather from Open-Meteo, grounds from GeoNames.",
    }
    return meta


# ---------------------------------------------------------------- deep dives
def wedge_rows(x, league_f, bowler):
    """x = faced balls of the player (batter) or against him (bowler), with region."""
    tot_runs = int(x.bat_runs.sum())
    xr = x.dropna(subset=["region"])
    lg = league_f.dropna(subset=["region"]).groupby("region").bat_runs.agg(["size", "sum"])
    rows = []
    for name, a, b, _ in REGIONS:
        y = xr[xr.region == name]
        fours, sixes = int(y.is_four.sum()), int(y.is_six.sum())
        rows.append({"name": name, "from": a, "to": b, "balls": int(len(y)), "runs": int(y.bat_runs.sum()), "fours": fours,
                     "sixes": sixes, "boundary_runs": 4 * fours + 6 * sixes, "outs": int(y.out_s.sum()), "caught": int(y.caught.sum()),
                     "run_share": r1(100 * y.bat_runs.sum() / tot_runs) if tot_runs else None,
                     "sr": sr(y.bat_runs.sum(), len(y)), "league_sr": sr(lg.loc[name, "sum"], lg.loc[name, "size"])})
    return rows


def pick_grounds(rows, low_is_good):
    """Grounds rule (same for every player): qualifying grounds only (>= 30 balls batters / >= 60 bowlers); the 3 best against
    par and the 1 worst against par. Fewer than 4 qualifying grounds: all of them as 'best', no separate 'worst'."""
    order = sorted(rows, key=lambda g: (g["vs_par"] if low_is_good else -g["vs_par"], -g["balls"]))
    best, rest = order[:3], order[3:]
    for g in best: g["tag"] = "best"
    worst = rest[-1:] if rest else []
    for g in worst: g["tag"] = "worst"
    return best + worst, len(rows)


def best_field(regions, k=5):
    best = max(itertools.combinations(range(len(regions)), k), key=lambda c: (sum(regions[i]["boundary_runs"] for i in c), [-i for i in c]))
    return [regions[i]["name"] for i in best], sum(regions[i]["boundary_runs"] for i in best)


def batter_dive(pid, d, f, fF):
    x = f[f.striker_id == pid]
    dx = x[x.phase == "Death"]; fd = f[f.phase == "Death"]
    R = {"player_id": int(pid), "role": "batter", "matches": int(x.match_id.nunique()), "balls": int(len(x)), "runs": int(x.bat_runs.sum()),
         "sr": sr(x.bat_runs.sum(), len(x)), "fours": int(x.is_four.sum()), "sixes": int(x.is_six.sum()), "outs": int(x.out_s.sum())}
    R["death"] = {"balls": int(len(dx)), "sr": sr(dx.bat_runs.sum(), len(dx)), "par_sr": sr(fd.bat_runs.sum(), len(fd)),
                  "boundary_pct": r1(100 * (dx.is_four | dx.is_six).mean()) if len(dx) else None,
                  "dot_pct": r1(100 * (dx.bat_runs == 0).mean()) if len(dx) else None,
                  "balls_per_out": r1(len(dx) / max(1, dx.out_s.sum()))}
    for lab, inn in (("setting", 1), ("chasing", 2)):
        y = x[x.innings == inn]; R[lab] = {"balls": int(len(y)), "sr": sr(y.bat_runs.sum(), len(y)), "outs": int(y.out_s.sum())}
    R["regions"] = wedge_rows(x, f, False)
    reg = R["regions"]
    R["region_totals"] = {"boundary_runs": sum(r["boundary_runs"] for r in reg), "fours": sum(r["fours"] for r in reg),
                          "sixes": sum(r["sixes"] for r in reg), "caught": sum(r["caught"] for r in reg),
                          "outs": sum(r["outs"] for r in reg), "runs": sum(r["runs"] for r in reg), "balls": sum(r["balls"] for r in reg)}
    bf, bfr = best_field(reg)
    R["best_field"] = {"regions": bf, "boundary_runs": bfr}
    # over by over (full 20-over innings)
    xf, ff = fF[fF.striker_id == pid], fF
    po = ff.groupby("overid").bat_runs.agg(["size", "sum"]); mo = xf.groupby("overid").bat_runs.agg(["size", "sum"])
    R["overs"] = [{"over": o, "balls": int(mo.loc[o, "size"]) if o in mo.index else 0,
                   "sr": sr(mo.loc[o, "sum"], mo.loc[o, "size"]) if o in mo.index else None,
                   "par_sr": sr(po.loc[o, "sum"], po.loc[o, "size"])} for o in range(1, 21)]
    R["overs_min_balls"] = 20
    # dismissals
    lo = f[f.out_s].dismissal_type.map(DISMISS).fillna("Other").value_counts()
    mo_ = x[x.out_s].dismissal_type.map(DISMISS).fillna("Other").value_counts()
    R["dismissals"] = [{"type": t, "n": int(mo_.get(t, 0)), "pct": r1(100 * mo_.get(t, 0) / max(1, mo_.sum())),
                        "league_pct": r1(100 * lo.get(t, 0) / lo.sum())} for t in ["Caught", "Bowled", "LBW", "Stumped", "Hit wicket", "Other"]
                       if mo_.get(t, 0) or t in ("Caught", "Bowled", "LBW")]
    # styles
    st = []
    for s in STYLES:
        y = x[x["style"] == s]; par = f[f["style"] == s]
        st.append({"style": s, "balls": int(len(y)), "runs": int(y.bat_runs.sum()), "sr": sr(y.bat_runs.sum(), len(y)), "outs": int(y.out_s.sum()),
                   "balls_per_out": r1(len(y) / y.out_s.sum()) if y.out_s.sum() else None, "par_sr": sr(par.bat_runs.sum(), len(par))})
    R["styles"] = st
    # grounds >= 30 balls
    gp = f.groupby("ground").bat_runs.agg(["size", "sum"])
    gr = x.groupby(["ground", "city"]).agg(balls=("bat_runs", "size"), runs=("bat_runs", "sum"), outs=("out_s", "sum"), matches=("match_id", "nunique")).reset_index()
    gr = gr[gr.balls >= 30].sort_values("balls", ascending=False)
    R["grounds"] = [{"ground": g.ground, "city": g.city, "balls": int(g.balls), "runs": int(g.runs), "outs": int(g.outs), "matches": int(g.matches),
                     "sr": sr(g.runs, g.balls), "par_sr": sr(gp.loc[g.ground, "sum"], gp.loc[g.ground, "size"]),
                     "vs_par": r1(sr(g.runs, g.balls) - sr(gp.loc[g.ground, "sum"], gp.loc[g.ground, "size"]))} for g in gr.itertuples()]
    R["grounds"], R["grounds_qualifying"] = pick_grounds(R["grounds"], low_is_good=False)
    R["grounds_min_balls"] = 30
    # opponents >= 15 balls
    op = x.groupby("bowler_id").agg(name=("bowler_name", lambda s: s.mode().iloc[0]), style=("style", lambda s: s.mode().iloc[0] if s.notna().any() else None),
                                   balls=("bat_runs", "size"), runs=("bat_runs", "sum"), outs=("out_s", "sum")).reset_index()
    op = op[op.balls >= 15].copy(); op["sr"] = (100 * op.runs / op.balls).round(1)
    rec = lambda o: [{"player_id": int(r.bowler_id), "name": disp(r.name), "style": r.style, "balls": int(r.balls), "runs": int(r.runs),
                      "outs": int(r.outs), "sr": float(r.sr)} for r in o.itertuples()]
    R["toughest"] = rec(op.sort_values(["sr", "balls"], ascending=[True, False]).head(3))
    R["easiest"] = rec(op.sort_values(["sr", "balls"], ascending=[False, False]).head(3))
    R["opponents_min_balls"] = 15
    R["phase_balls"] = {ph: int((x.phase == ph).sum()) for ph in ("Powerplay", "Middle", "Death")}
    R["max_ground_balls"] = int(x.groupby("ground").size().max()) if len(x) else 0
    R["max_opp_balls"] = int(x.groupby("bowler_id").size().max()) if len(x) else 0
    return R


def usual_overs(overs, n=4):
    """The n non-adjacent overs he has bowled most often (innings in which he bowled that over; full 20-over innings)."""
    best = max((c for c in itertools.combinations(range(1, 21), n) if all(b - a > 1 for a, b in zip(c, c[1:]))),
               key=lambda c: (sum(overs[o - 1]["times"] for o in c), [-o for o in c]))
    return {"overs": list(best), "times": sum(overs[o - 1]["times"] for o in best)}


def best_plan(overs, n=4):
    cand = [o["over"] for o in overs]
    sav = {o["over"]: o["saving"] for o in overs}
    best = None
    for c in itertools.combinations(cand, n):
        if any(b - a == 1 for a, b in zip(c, c[1:])):
            continue
        v = sum(sav[o] for o in c)
        if best is None or v > best[1] + 1e-12:
            best = (list(c), v)
    return best


def bowler_dive(pid, d, f, dF):
    y = d[d.bowler_id == pid]; yl = y[y.is_legal]
    R = {"player_id": int(pid), "role": "bowler", "matches": int(y.match_id.nunique()), "balls": int(len(yl)), "wickets": int(y.bowler_wicket.sum()),
         "runs": int(y.bowler_runs_conceded.sum()), "econ": econ(y.bowler_runs_conceded.sum(), len(yl)),
         "dot_pct": r1(100 * (yl.bowler_runs_conceded == 0).mean()), "balls_per_wkt": r1(len(yl) / max(1, y.bowler_wicket.sum()))}
    dy = y[y.phase == "Death"]; dyl = dy[dy.is_legal]; Ld = d[d.phase == "Death"]
    R["death"] = {"balls": int(len(dyl)), "econ": econ(dy.bowler_runs_conceded.sum(), len(dyl)),
                  "par_econ": econ(Ld.bowler_runs_conceded.sum(), Ld.is_legal.sum()),
                  "dot_pct": r1(100 * (dyl.bowler_runs_conceded == 0).mean()) if len(dyl) else None,
                  "boundary_pct": r1(100 * (dyl.is_four | dyl.is_six).mean()) if len(dyl) else None,
                  "balls_per_wkt": r1(len(dyl) / max(1, dy.bowler_wicket.sum())), "wickets": int(dy.bowler_wicket.sum()),
                  "yorker_pct": r1(100 * (dyl["var"] == "Yorker").mean()) if len(dyl) else None}
    fx = f[f.bowler_id == pid]
    R["regions"] = wedge_rows(fx, f, True)
    R["region_totals"] = {"runs_all": int(fx.bat_runs.sum()), "wickets_in_regions": sum(r["outs"] for r in R["regions"])}
    # over by over (full 20-over innings) + planner saving
    yF = dF[dF.bowler_id == pid]
    po = dF.groupby("overid").agg(r=("bowler_runs_conceded", "sum"), b=("is_legal", "sum"))
    mo = yF.groupby("overid").agg(r=("bowler_runs_conceded", "sum"), b=("is_legal", "sum"), w=("bowler_wicket", "sum"))
    times = yF.groupby("overid").apply(lambda g: g[["match_id", "innings"]].drop_duplicates().shape[0])
    ov = []
    for o in range(1, 21):
        b = int(mo.loc[o, "b"]) if o in mo.index else 0
        e = econ(mo.loc[o, "r"], b) if b else None
        p = 6 * po.loc[o, "r"] / po.loc[o, "b"]
        ov.append({"over": o, "balls": b, "econ": e, "par_econ": r2(p), "wkts": int(mo.loc[o, "w"]) if o in mo.index else 0,
                   "times": int(times.get(o, 0)),
                   "saving": float((p - 6 * mo.loc[o, "r"] / b) * b / (b + 60)) if b else 0.0})
    R["overs"] = ov
    R["overs_min_balls"] = 24
    bowled = [o for o in ov if o["balls"]]
    R["planner"] = {"shrink_balls": 60, "few_balls": 60, "avg_par_unbowled": r2(np.mean([6 * po.loc[o["over"], "r"] / po.loc[o["over"], "b"] for o in bowled]))}
    bp, bv = best_plan(ov)
    R["planner"]["best"] = {"overs": bp, "saving": round(bv, 3),
                            "his": r2(sum(6 * po.loc[o, "r"] / po.loc[o, "b"] for o in bp) - bv),
                            "avg": r2(sum(6 * po.loc[o, "r"] / po.loc[o, "b"] for o in bp))}
    R["planner"]["usual"] = usual_overs(ov)
    # wicket types
    lw = d[d.bowler_wicket].dismissal_type.map(DISMISS).fillna("Other").value_counts()
    mw = y[y.bowler_wicket].dismissal_type.map(DISMISS).fillna("Other").value_counts()
    R["wicket_types"] = [{"type": t, "n": int(mw.get(t, 0)), "pct": r1(100 * mw.get(t, 0) / max(1, mw.sum())),
                          "league_pct": r1(100 * lw.get(t, 0) / lw.sum())} for t in ["Caught", "Bowled", "LBW", "Stumped", "Hit wicket", "Other"]
                         if mw.get(t, 0) or t in ("Caught", "Bowled", "LBW")]
    # deliveries
    vr = []
    for v in VARS:
        z = y[y["var"] == v]; zl = z[z.is_legal]; par = d[d["var"] == v]
        vr.append({"type": v, "balls": int(len(zl)), "usage_pct": r1(100 * len(zl) / len(yl)), "econ": econ(z.bowler_runs_conceded.sum(), len(zl)),
                   "wkts": int(z.bowler_wicket.sum()), "balls_per_wkt": r1(len(zl) / z.bowler_wicket.sum()) if z.bowler_wicket.sum() else None,
                   "par_econ": econ(par.bowler_runs_conceded.sum(), par.is_legal.sum())})
    R["deliveries"] = vr
    R["deliveries_labelled_pct"] = r1(100 * yl["var"].notna().mean())
    hd = []
    for h in ("RHB", "LHB"):
        z = y[y.batting_type == h]; zl = z[z.is_legal]; par = d[d.batting_type == h]
        hd.append({"hand": h, "balls": int(len(zl)), "econ": econ(z.bowler_runs_conceded.sum(), len(zl)), "wkts": int(z.bowler_wicket.sum()),
                   "par_econ": econ(par.bowler_runs_conceded.sum(), par.is_legal.sum())})
    R["hands"] = hd
    # grounds >= 60 legal balls
    L = d[d.is_legal]
    gp_r = d.groupby("ground").bowler_runs_conceded.sum(); gp_b = L.groupby("ground").size()
    gr = y.groupby(["ground", "city"]).agg(runs=("bowler_runs_conceded", "sum"), wkts=("bowler_wicket", "sum"), matches=("match_id", "nunique"))
    gr = gr.join(yl.groupby(["ground", "city"]).size().rename("balls")).reset_index()
    gr = gr[gr.balls >= 60].sort_values("balls", ascending=False)
    R["grounds"] = [{"ground": g.ground, "city": g.city, "balls": int(g.balls), "runs": int(g.runs), "wkts": int(g.wkts), "matches": int(g.matches),
                     "econ": econ(g.runs, g.balls), "par_econ": econ(gp_r[g.ground], gp_b[g.ground]),
                     "vs_par": r2(econ(g.runs, g.balls) - econ(gp_r[g.ground], gp_b[g.ground]))} for g in gr.itertuples()]
    R["grounds"], R["grounds_qualifying"] = pick_grounds(R["grounds"], low_is_good=True)
    R["grounds_min_balls"] = 60
    op = fx.groupby("striker_id").agg(name=("striker_name", lambda s: s.mode().iloc[0]), balls=("bat_runs", "size"), runs=("bat_runs", "sum"),
                                     outs=("out_s", "sum")).reset_index()
    op = op[op.balls >= 15].copy(); op["sr"] = (100 * op.runs / op.balls).round(1)
    rec = lambda o: [{"player_id": int(r.striker_id), "name": disp(r.name), "balls": int(r.balls), "runs": int(r.runs), "outs": int(r.outs),
                      "sr": float(r.sr)} for r in o.itertuples()]
    R["owns"] = rec(op.sort_values(["outs", "sr"], ascending=[False, True]).head(3))
    R["troubled_by"] = rec(op.sort_values(["sr", "balls"], ascending=[False, False]).head(3))
    R["opponents_min_balls"] = 15
    R["phase_balls"] = {ph: int((yl.phase == ph).sum()) for ph in ("Powerplay", "Middle", "Death")}
    R["max_ground_balls"] = int(yl.groupby("ground").size().max()) if len(yl) else 0
    R["max_opp_balls"] = int(fx.groupby("striker_id").size().max()) if len(fx) else 0
    return R


# ---------------------------------------------------------------- simulator (sim19.py)
def sim_class(df):
    o = np.where(df.is_wide, "WD", np.where(df.is_noball, "NB", np.where(df.is_wicket, "W", "")))
    tr = df.team_runs.astype(int).clip(lower=0)
    runs = np.select([tr == 0, tr == 1, tr == 2, tr == 3, (tr == 4) | (tr == 5)], ["0", "1", "2", "3", "4"], "6")
    return pd.Series(np.where(o == "", runs, o), index=df.index)


def chase_win(pX, league, balls=12, wk=6, need=18):
    pX, league = dict(zip(SIM_C, pX)), dict(zip(SIM_C, league))

    @lru_cache(None)
    def V(b, w, r):
        if r <= 0: return 1.0
        if b == 0 or w == 0: return 0.5 if r == 1 else 0.0
        p = pX if b > 6 else league
        v = sum(p[o] * V(b - 1, w, r - RUN[o]) for o in ["0", "1", "2", "3", "4", "6"])
        return v + p["W"] * V(b - 1, w - 1, r) + (p["WD"] + p["NB"]) * V(b, w, r - 1)
    return V(balls, wk, need)


def simulator(d, a, bowlers):
    dd = d[d.phase == "Death"].copy()
    dd["o"] = sim_class(dd)
    cnt_l = dd.o.value_counts().reindex(SIM_C, fill_value=0)
    league = (cnt_l / cnt_l.sum()).values
    rows = []
    for b in bowlers:
        x = dd[dd.bowler_id == b["player_id"]]; n = len(x)
        cnt = x.o.value_counts().reindex(SIM_C, fill_value=0)
        p = (cnt.values + K_SIM * league) / (n + K_SIM)
        xl = x[x.is_legal]
        rows.append({"player_id": b["player_id"], "name": b["name"], "rank": b["rank"], "group": b["group"],
                     "counts": {k: int(v) for k, v in cnt.items()}, "n": int(n), "p": [round(float(v), 5) for v in p],
                     "death_balls": int(len(xl)), "death_econ": r2(6 * x.bowler_runs_conceded.sum() / max(1, len(xl))),
                     "dot_pct": r1(100 * (xl.bowler_runs_conceded == 0).mean()), "boundary_pct": r1(100 * (xl.is_four | xl.is_six).mean()),
                     "balls_per_wkt": r1(len(xl) / max(1, x.is_wicket.sum())),
                     "yorker_pct": r1(100 * xl.delivery_style.isin(["Yorker", "Wide Yorker"]).mean()),
                     "defend": {f"{need}_{wk}": round(100 * (1 - chase_win(p, league, 12, wk, need)), 1) for need, wk in SIM_SITUATIONS}})
    mean_abs = float(a.wpa.abs().mean())
    base = {f"{need}_{wk}": round(100 * (1 - chase_win(league, league, 12, wk, need)), 1) for need, wk in SIM_SITUATIONS}
    return {"outcomes": SIM_C, "runs": [0, 1, 2, 3, 4, 6], "K": K_SIM, "league": [round(float(v), 5) for v in league],
            "league_counts": {k: int(v) for k, v in cnt_l.items()}, "mean_abs_wpa": round(mean_abs, 6), "base_defend": base,
            "few_death_balls": 150, "situations": [{"need": n_, "wk": w_} for n_, w_ in SIM_SITUATIONS],
            "need_range": [4, 36], "wk_range": [1, 10], "bowlers": rows}


# ---------------------------------------------------------------- sandbox
def sandbox(tech):
    out = {"own_share_pct": [45, 50, 55], "matches_2025_26": [3, 5, 8], "keys": ["cautious", "projected"], "floors": [0, 30, 50, 75, 100],
           "pressure_rule": [0, 1], "ours": {"own": 50, "m": 5, "key": "cautious", "floor": 50, "pp": 1}, "lists": {}}
    for side, role in ROLE.items():
        T = tech[side]
        for own, m, key, fl, pp in itertools.product(out["own_share_pct"], out["matches_2025_26"], out["keys"], out["floors"], out["pressure_rule"]):
            el = T[(T.own_share >= own / 100) & (T.matches_2025_26 >= m) & (T.n_p >= fl) & ((T.pressure_value > 0) if pp else True)]
            order = ["cautious", "projected"] if key == "cautious" else ["projected", "cautious"]
            top = el.sort_values(order, ascending=False).head(12)
            out["lists"][f"{role}|{own}|{m}|{key}|{fl}|{pp}"] = {
                "eligible": int(len(el)),
                "top": [{"player_id": int(r.player_id), "name": disp(r.name), "value": round(100 * float(getattr(r, key)), 3),
                         "high_pressure_balls": int(r.n_p)} for r in top.itertuples()]}
    return out


# ---------------------------------------------------------------- head-to-head + specialists (build_matchups.py, build_specialists.py)
def matchups(d, tech):
    tb = tech["bat"][tech["bat"].eligible].nsmallest(25, "rank_eligible"); tw = tech["bowl"][tech["bowl"].eligible].nsmallest(25, "rank_eligible")
    bn = dict(zip(tb.player_id, tb.name.map(disp))); wn = dict(zip(tw.player_id, tw.name))
    d = d.copy(); d["press"] = d.li >= 1.5
    f = d[d.batter_ball_faced]
    m = f[f.striker_id.isin(bn) & f.bowler_id.isin(wn)].copy()
    m["p_run"] = np.where(m.press, m.bat_runs, 0)
    g = m.groupby(["striker_id", "bowler_id"]).agg(balls=("bat_runs", "size"), runs=("bat_runs", "sum"), outs=("out_s", "sum"),
                                                   dots=("bat_runs", lambda s: int((s == 0).sum())), fours=("is_four", "sum"), sixes=("is_six", "sum"),
                                                   p_balls=("press", "sum"), p_runs=("p_run", "sum"), matches=("match_id", "nunique")).reset_index()
    league_sr = r1(100 * f.bat_runs.sum() / len(f))
    pairs = [{"batter_id": int(r.striker_id), "bowler_id": int(r.bowler_id), "balls": int(r.balls), "runs": int(r.runs), "outs": int(r.outs),
              "dots": int(r.dots), "fours": int(r.fours), "sixes": int(r.sixes), "p_balls": int(r.p_balls), "p_runs": int(r.p_runs),
              "matches": int(r.matches)} for r in g.itertuples()]
    rows = []
    bf = f.groupby(["phase", "pace_or_spin"]).agg(b=("bat_runs", "size"), r=("bat_runs", "sum"))
    for pid in bn:
        x = f[f.striker_id == pid].groupby(["phase", "pace_or_spin"]).agg(balls=("bat_runs", "size"), runs=("bat_runs", "sum"), outs=("out_s", "sum"))
        for (ph, vs), r in bf.iterrows():
            v = x.loc[(ph, vs)] if (ph, vs) in x.index else None
            balls, runs = (int(v.balls), int(v.runs)) if v is not None else (0, 0)
            rate = r1(100 * runs / balls) if balls else None; base = r1(100 * r.r / r.b)
            rows.append({"player_id": int(pid), "role": "batter", "phase": ph, "vs": vs, "balls": balls, "runs": runs,
                         "outs": int(v.outs) if v is not None else 0, "rate": rate, "par": base,
                         "batter_edge": r1(100 * runs / balls - 100 * r.r / r.b) if balls else None})
    lg = d[d.is_legal]
    bw = d.groupby(["phase", "batting_type"]).agg(r=("bowler_runs_conceded", "sum")).join(lg.groupby(["phase", "batting_type"]).size().rename("b"))
    for pid in wn:
        y = d[d.bowler_id == pid]
        x = y.groupby(["phase", "batting_type"]).agg(runs=("bowler_runs_conceded", "sum"), outs=("bowler_wicket", "sum")).join(
            y[y.is_legal].groupby(["phase", "batting_type"]).size().rename("balls"))
        for (ph, vs), r in bw.iterrows():
            v = x.loc[(ph, vs)] if (ph, vs) in x.index else None
            balls = int(v.balls) if v is not None and pd.notna(v.balls) else 0
            runs = int(v.runs) if v is not None else 0
            rate = r2(6 * runs / balls) if balls else None; base = r2(6 * r.r / r.b)
            rows.append({"player_id": int(pid), "role": "bowler", "phase": ph, "vs": vs, "balls": balls, "runs": runs,
                         "outs": int(v.outs) if v is not None else 0, "rate": rate, "par": base,
                         "batter_edge": r1((6 * runs / balls - 6 * r.r / r.b) * 100 / 6) if balls else None})
    roster = lambda t: [{"player_id": int(r.player_id), "name": disp(r.name), "rank": int(r.rank_eligible)} for r in t.itertuples()]
    return {"min_balls": MIN_CELL, "league_sr": league_sr, "batters": roster(tb), "bowlers": roster(tw), "pairs": pairs, "splits": rows}


def specialists(d, tech):
    eb = tech["bat"][tech["bat"].eligible]; ew = tech["bowl"][tech["bowl"].eligible]
    f = d[d.batter_ball_faced]; rows = []

    def bat_role(key, label, mask, minb, metric="sr"):
        par_f = f[mask(f)]
        par = 100 * par_f.bat_runs.mean() if metric == "sr" else len(par_f) / par_f.out_s.sum()
        sub = par_f[par_f.striker_id.isin(eb.player_id)]
        g = sub.groupby("striker_id").agg(balls=("bat_runs", "size"), runs=("bat_runs", "sum"), outs=("out_s", "sum"))
        g = g[g.balls >= minb]
        g["value"] = 100 * g.runs / g.balls if metric == "sr" else g.balls / g.outs.clip(lower=1)
        for pid, r in g.sort_values("value", ascending=False).iterrows():
            p = eb[eb.player_id == pid].iloc[0]
            rows.append({"role": "batter", "chip": key, "label": label, "metric": "Strike rate" if metric == "sr" else "Balls per dismissal",
                         "player_id": int(pid), "name": disp(p["name"]), "value": r1(r.value), "par": r1(par), "edge": r1(r.value - par),
                         "balls": int(r.balls), "runs": int(r.runs), "outs": int(r.outs), "rank": int(p.rank_eligible), "min_balls": minb})

    def bowl_role(key, label, mask, minb, metric="econ"):
        par_d = d[mask(d)]; parL = par_d[par_d.is_legal]
        par = 6 * par_d.bowler_runs_conceded.sum() / len(parL) if metric == "econ" else len(parL) / par_d.bowler_wicket.sum()
        sub = par_d[par_d.bowler_id.isin(ew.player_id)]
        g = sub.groupby("bowler_id").agg(runs=("bowler_runs_conceded", "sum"), wkts=("bowler_wicket", "sum")).join(
            sub[sub.is_legal].groupby("bowler_id").size().rename("balls"))
        g = g[g.balls >= minb]
        g["value"] = 6 * g.runs / g.balls if metric == "econ" else g.balls / g.wkts.clip(lower=1)
        for pid, r in g.sort_values("value").iterrows():
            p = ew[ew.player_id == pid].iloc[0]
            rows.append({"role": "bowler", "chip": key, "label": label, "metric": "Economy" if metric == "econ" else "Balls per wicket",
                         "player_id": int(pid), "name": disp(p["name"]), "value": r2(r.value), "par": r2(par), "edge": r2(r.value - par),
                         "balls": int(r.balls), "runs": int(r.runs), "outs": int(r.wkts), "rank": int(p.rank_eligible), "min_balls": minb})

    bat_role("spin", "Spin hitters", lambda x: x.pace_or_spin == "Spin", 150)
    bat_role("pace", "Pace hitters", lambda x: x.pace_or_spin == "Pace", 250)
    bat_role("death", "Death hitters", lambda x: x.phase == "Death", 150)
    bat_role("powerplay", "Powerplay starters", lambda x: x.phase == "Powerplay", 150)
    bat_role("pressure", "Pressure-ball scorers", lambda x: x.li >= 1.5, 60)
    bat_role("anchor", "Hard to dismiss", lambda x: x.index == x.index, 400, metric="bpo")
    bowl_role("death", "Death bowlers", lambda x: x.phase == "Death", 150)
    bowl_role("powerplay", "Powerplay bowlers", lambda x: x.phase == "Powerplay", 150)
    bowl_role("middle", "Middle-overs squeeze", lambda x: x.phase == "Middle", 240)
    bowl_role("lhb", "Against left-handers", lambda x: x.batting_type == "LHB", 150)
    bowl_role("rhb", "Against right-handers", lambda x: x.batting_type == "RHB", 300)
    bowl_role("wickets", "Wicket-takers", lambda x: x.index == x.index, 400, metric="bpw")
    bowl_role("pressure", "Pressure-ball bowlers", lambda x: x.li >= 1.5, 60)
    return {"rows": rows}


# ---------------------------------------------------------------- main
def main():
    os.makedirs(OUT, exist_ok=True)
    for fn in os.listdir(OUT):
        if fn.endswith(".json"): os.remove(os.path.join(OUT, fn))
    d, a = load()
    print(f"balls {len(d)}; Phase 3 li joined {d.li.notna().mean():.4f}", flush=True)
    t2, a2, tech = lists()
    picks = picks_and_gaps(t2, a2, tech)
    meta = changes_and_meta(tech)
    no_li = float(d.li.isna().mean())
    meta["balls"] = {"total": int(len(d)), "without_importance": int(d.li.isna().sum()), "without_importance_pct": round(100 * no_li, 1)}
    meta["how_it_works_lines"] = {
        "no_importance": f"{100 * no_li:.1f}% of balls have no importance value: they come from rain-affected or no-result matches, "
                         "where there is no fair win probability to measure against. Those balls still count in the plain "
                         "counts (runs, wickets, field regions) but not in anything that uses importance."}
    jdump("meta.json", meta)
    f = d[d.batter_ball_faced]; dF = d[d.full20]; fF = dF[dF.batter_ball_faced]
    dives = {}
    for role, R in picks["roles"].items():
        for p in R["players"]:
            dv = batter_dive(p["player_id"], d, f, fF) if role == "batter" else bowler_dive(p["player_id"], d, f, dF)
            dv.update({"name": p["name"], "rank": p["rank"], "group": p["group"]})
            finish_dive(dv)
            dives[str(p["player_id"])] = dv
            print(f"  dive {role:6s} #{p['rank']:2d} {p['name']}", flush=True)
    k = bat_k(list(dives.values()))
    for role, R in picks["roles"].items():
        for p in R["players"]:
            b = make_badge(dives[str(p["player_id"])], k)
            dives[str(p["player_id"])]["badge"] = b; p["badge"] = b
    jdump("players.json", dives); jdump("picks.json", picks)
    bowlers = [p for p in picks["roles"]["bowler"]["players"]]
    jdump("sim.json", simulator(d, a, bowlers))
    jdump("sandbox.json", sandbox(tech))
    jdump("matchups.json", matchups(d, tech))
    jdump("specialists.json", specialists(d, tech))
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
