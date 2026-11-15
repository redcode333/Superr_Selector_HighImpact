"""Phase 8 - baseline vs pressure table and forward-test summary for the floor-50 (v2) package.

Reuses the Phase 6b code unchanged (player_match, summarise, gap_table: normal / pressure = 100 x sum(v_adj) / sum(LI)
over balls with LI < 1.5 / >= 1.5; gap = pressure - normal; 80% interval from 500 match resamples, default_rng(SEED + 61)).
Unit: win-probability points per average-importance ball.
League rows: all floor-50 eligible players of the role pooled into one record (their balls summed match by match), then
the same summarise + gap_table. Forward test: Phase 6c run on the floor-50 pool (read from reports/phase5_v2).
Reads only *_v2 outputs; writes reports/final_v2/pressure_gap_top10.csv, pressure_gap_all_eligible.csv,
phase8_summary.json, changes_v2.md.
"""
import json, os, sys
import numpy as np
import pandas as pd

os.environ.setdefault("SUPERR_FLOOR", "50")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variant import mpath, FLOOR, SUFFIX
from phase6b_gap import player_match, gap_table

assert FLOOR == 50 and SUFFIX == "_v2", (FLOOR, SUFFIX)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIN2 = os.path.join(ROOT, "reports", "final_v2")
REP5_2 = os.path.join(ROOT, "reports", "phase5_v2")
P7 = os.path.join(os.path.dirname(ROOT), "outputs", "phase7_exposure_floor")
UNIT = "win-probability points per average-importance ball"
ROLE = {"bat": "batter", "bowl": "bowler"}
COLS = ["role", "rank", "group", "player_id", "player", "balls_normal", "balls_pressure", "normal_value", "pressure_value",
        "gap", "gap_lo80", "gap_hi80", "verdict", "package_alternate"]


def verdict(lo, hi):
    if lo > 0: return "better under pressure"
    if hi < 0: return "worse under pressure"
    return "no clear difference"


def frame(G, tech, ids, side, group):
    g = G.reindex(ids)
    return pd.DataFrame({"role": ROLE[side], "rank": tech.loc[ids, "rank_eligible"].astype(int).values, "group": group, "player_id": ids,
                         "player": tech.loc[ids, "name"].values, "balls_normal": g.normal_balls.astype(int).values,
                         "balls_pressure": g.pressure_balls.astype(int).values, "normal_value": g.normal.values, "pressure_value": g.pressure.values,
                         "gap": g.gap.values, "gap_lo80": g.lo80.values, "gap_hi80": g.hi80.values,
                         "verdict": [verdict(a, b) for a, b in zip(g.lo80, g.hi80)]})


def league(pm, ids, side):
    p = pm[pm.player.isin(ids)]
    agg = p.groupby("match_id").agg(date=("date", "first"), **{c: (c, "sum") for c in ("Vp", "Lp", "np_", "Vn", "Ln", "nn")}).reset_index()
    agg["player"] = -1
    agg = agg.sort_values(["date", "match_id"]).reset_index(drop=True); agg["idx"] = np.arange(len(agg))
    g = gap_table(agg).loc[-1]
    return pd.DataFrame([{"role": ROLE[side], "rank": None, "group": "league reference", "player_id": None,
                          "player": f"Average eligible {ROLE[side]} (all {len(ids)} pooled)", "balls_normal": int(g.normal_balls),
                          "balls_pressure": int(g.pressure_balls), "normal_value": g.normal, "pressure_value": g.pressure, "gap": g.gap,
                          "gap_lo80": g.lo80, "gap_hi80": g.hi80, "verdict": verdict(g.lo80, g.hi80)}])


def shares(df):
    n = len(df); c = df.verdict.value_counts()
    return {v: {"n": int(c.get(v, 0)), "pct": round(100 * c.get(v, 0) / n, 1)} for v in
            ("better under pressure", "worse under pressure", "no clear difference")} | {"players": n}


def main():
    tops, alls, out = [], [], {"unit": UNIT, "verdicts": {}, "forward_test": {}}
    kn = json.load(open(os.path.join(FIN2, "key_numbers.json")))
    for side in ("bat", "bowl"):
        tech = pd.read_csv(mpath(f"phase4_rankings_{side}_all_players_technical.csv"), index_col=0)
        el = tech[tech.eligible == True].sort_values("rank_eligible")
        pm = player_match(side)
        G = gap_table(pm)                                       # identical call to Phase 6b
        # consistency with the v2 Phase 6b run (key_numbers pressure_gap top 10)
        for r in kn["pressure_gap"]["top10"][side]:
            pid = el.index[el.rank_eligible == r["rank"]][0]
            assert abs(G.loc[pid, "gap"] - r["gap"]) < 1e-9 and abs(G.loc[pid, "lo80"] - r["lo80"]) < 1e-9, (side, r["player"])
        top = frame(G, tech, list(el.index[:10]), side, "top 10")
        alt = frame(G, tech, list(el.index[10:12]), side, "alternate")
        allel = frame(G, tech, list(el.index), side, "eligible")
        lg = league(pm, list(el.index), side)
        pk = {a["player"] for a in kn["alternates"][side]}
        for d in (top, alt, allel, lg):
            d["package_alternate"] = ["yes" if p in pk else "no" for p in d.player]
        assert pk <= set(pd.concat([top, alt]).player), (side, pk)
        tops += [top, alt, lg]; alls += [allel, lg]
        out["verdicts"][ROLE[side]] = {"top 10": shares(top), "top 10 + 2 alternates": shares(pd.concat([top, alt])), "all eligible": shares(allel),
                                       "league reference": lg.verdict.iloc[0]}
        print(f"{ROLE[side]}s: {len(el)} eligible; package alternates (pct_top10 rule): {[a['player'] for a in kn['alternates'][side]]}")
    fw = json.load(open(os.path.join(REP5_2, "phase6c_gap_forward.json")))
    for side in ("bat", "bowl"):
        out["forward_test"][ROLE[side]] = {k: fw[side][k] for k in ("eligible only, >= 20 each period", "pressure balls >= 20 each period (main)")}
    rnd = lambda d: d.round({c: 4 for c in ("normal_value", "pressure_value", "gap", "gap_lo80", "gap_hi80")})
    for name, parts in (("pressure_gap_top10.csv", tops), ("pressure_gap_all_eligible.csv", alls)):
        rnd(pd.concat(parts))[COLS].assign(unit=UNIT).to_csv(os.path.join(FIN2, name), index=False)
    out["eligibility"] = kn["eligibility"]
    json.dump(out, open(os.path.join(FIN2, "phase8_summary.json"), "w"), indent=1, default=float)
    print(rnd(pd.concat(tops))[COLS].drop(columns="player_id").to_string(index=False))
    print(json.dumps(out["verdicts"], indent=1))
    for role, d in out["forward_test"].items():
        for k, v in d.items():
            print(role, k, v["players"], "gap rho %+.3f [%+.3f, %+.3f]" % tuple(v["gap_rho"]))


if __name__ == "__main__":
    main()
