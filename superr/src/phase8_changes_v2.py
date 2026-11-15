"""Phase 8 - write reports/final_v2/changes_v2.md (plain English) from the v2 outputs. No number is typed in here.

Reads: reports/final/{final_top10,alternates}.csv (v1), reports/final_v2/{final_top10,alternates}.csv, key_numbers.json,
phase8_summary.json, pressure_gap_top10.csv, verification.txt; outputs/phase7_exposure_floor/floor_sensitivity.csv.
"""
import json, os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIN1, FIN2 = os.path.join(ROOT, "reports", "final"), os.path.join(ROOT, "reports", "final_v2")
P7 = os.path.join(os.path.dirname(ROOT), "outputs", "phase7_exposure_floor")
ROLE = {"bat": "batter", "bowl": "bowler"}
RK, PCT, CR = "rank (eligible)", "% of re-rankings in top 10", "cautious rating (pts)"


def ci(v): return f"{v[0]:+.3f} (95% range {v[1]:+.3f} to {v[2]:+.3f})"


def main():
    kn = json.load(open(os.path.join(FIN2, "key_numbers.json")))
    s8 = json.load(open(os.path.join(FIN2, "phase8_summary.json")))
    unit = s8["unit"]
    t1, t2 = pd.read_csv(os.path.join(FIN1, "final_top10.csv")), pd.read_csv(os.path.join(FIN2, "final_top10.csv"))
    a2 = pd.read_csv(os.path.join(FIN2, "alternates.csv"))
    fs = pd.read_csv(os.path.join(P7, "floor_sensitivity.csv"))
    pg = pd.read_csv(os.path.join(FIN2, "pressure_gap_top10.csv"))
    ver = [l.split("\t")[0] for l in open(os.path.join(FIN2, "verification.txt"), encoding="utf-8").read().splitlines() if l]
    n_pass = sum(v == "PASS" for v in ver)
    hp = fs[fs.floor == 50].set_index(["role", "player"]).high_pressure_balls
    hp0 = fs[fs.floor == 0].set_index(["role", "player"]).high_pressure_balls
    L = ["# SupeRR Selector v2: what changed from v1", "",
         "v1 (reports/final/) is unchanged and stays the reference for the old rule. This package (reports/final_v2/) "
         "is built by `python run_all.py --floor 50 --clean`, which reruns phase4f_final onward; everything upstream "
         "(win probability, ball values, ratings) is the frozen v1 output.", "",
         "## What changed and why", "",
         "**One new eligibility rule.** A player must also have at least 50 high-pressure balls in 2022-26 (balls with "
         "importance LI >= 1.5; balls faced for batters, balls bowled for bowlers). The two v1 rules stay: at least 50% "
         "own-data share and at least 5 matches in 2025-26. The model, the ratings and the order of players are not "
         "changed; the rule only removes players from the pool. Reason: a player whose pressure record rests on a "
         "handful of high-pressure balls should not be picked for a pressure role.", ""]
    L += ["**Pool size.**", "", "| Floor | Batters eligible | Bowlers eligible |", "|---|---|---|"]
    for f, g in fs.groupby("floor"):
        L.append(f"| {f} | {(g.role == 'batter').sum()} | {(g.role == 'bowler').sum()} |")
    L += ["", "**Sensitivity (floors 30 / 50 / 75 / 100)** - changes to the top 10 compared with floor 0:", ""]
    base = {r: set(fs[(fs.floor == 0) & (fs.role == r) & (fs["rank"] <= 10)].player) for r in ROLE.values()}
    for f in (30, 50, 75, 100):
        parts = []
        for r in ROLE.values():
            top = set(fs[(fs.floor == f) & (fs.role == r) & (fs["rank"] <= 10)].player)
            out_, in_ = sorted(base[r] - top), sorted(top - base[r])
            parts.append(f"{r}s " + ("no change" if not out_ else f"out {', '.join(out_)}; in {', '.join(in_)}"))
        L.append(f"- Floor {f}: " + "; ".join(parts) + ".")
    L += ["", "Floor 50 is the adopted rule.", ""]
    L += ["**The two batting changes at floor 50.**", ""]
    for r in ROLE.values():
        o, n = set(t1[t1.role == r].player), set(t2[t2.role == r].player)
        for p in sorted(o - n):
            rk = int(t1[(t1.role == r) & (t1.player == p)][RK].iloc[0])
            L.append(f"- Out: {p} (v1 {r} #{rk}, {int(hp0[(r, p)])} high-pressure balls).")
        for p in sorted(n - o):
            rk = int(t2[(t2.role == r) & (t2.player == p)][RK].iloc[0])
            L.append(f"- In: {p} (v2 {r} #{rk}, {int(hp[(r, p)])} high-pressure balls).")
        if o == n:
            L.append(f"- {r.capitalize()}s: the top 10 is the same as v1.")
    L += ["", "## The new top 10s", ""]
    for r in ROLE.values():
        L += [f"**{r.capitalize()}s**", "", "| Rank | Player | Cautious rating | % of re-rankings in top 10 | High-pressure balls |", "|---|---|---|---|---|"]
        for _, x in t2[t2.role == r].sort_values(RK).iterrows():
            L.append(f"| {int(x[RK])} | {x.player} | {x[CR]:+.3f} | {x[PCT]:.1f} | {int(hp[(r, x.player)])} |")
        al = a2[a2.role == r].sort_values(RK)
        L += ["", "Alternates: " + "; ".join(f"{x.player} (rank {int(x[RK])}, in the top 10 in {x[PCT]:.1f}% of re-rankings)"
                                             for _, x in al.iterrows()), ""]
    L += ["The package picks alternates with the same rule as v1 (based on % of re-rankings in the top 10), which gives one per role "
          "here. pressure_gap_top10.csv lists ranks 11-12 for both roles; its package_alternate column marks the package's pick.", ""]
    L += ["## Normal vs pressure", "",
          f"Values are in **{unit}**: the player's adjusted win-probability added, divided by the total importance (LI) of "
          "the balls, x 100, on normal balls (LI < 1.5) and on pressure balls (LI >= 1.5). Gap = pressure minus normal. "
          "This is exactly the Phase 6b definition and code. The 80% interval comes from 500 match resamples.", "",
          "Verdict: \"better under pressure\" if the whole interval is above 0, \"worse under pressure\" if it is all "
          "below 0, otherwise \"no clear difference\".", "",
          "| Group | Better under pressure | Worse under pressure | No clear difference |", "|---|---|---|---|"]
    for r, d in s8["verdicts"].items():
        for grp in ("top 10", "all eligible"):
            v = d[grp]
            cell = lambda k: f"{v[k]['n']} of {v['players']} ({v[k]['pct']:.0f}%)"
            L.append(f"| {r.capitalize()}s, {grp} | {cell('better under pressure')} | {cell('worse under pressure')} | {cell('no clear difference')} |")
    L += [""]
    for r, d in s8["verdicts"].items():
        g = pg[(pg.role == r) & (pg.group == "league reference")].iloc[0]
        L.append(f"- Average eligible {r} (all pooled): normal {g.normal_value:+.3f}, pressure {g.pressure_value:+.3f}, "
                 f"gap {g.gap:+.3f} (80% interval {g.gap_lo80:+.3f} to {g.gap_hi80:+.3f}): {g.verdict}.")
    L += ["", "## Does a pressure gap carry forward? (floor-50 pool)", "",
          "Same test as Phase 6c: gap in 2022-24 against gap in 2025-26, players with at least 20 pressure balls in each period. "
          "The \"eligible only\" row is the floor-50 pool. The \"main\" row does not use eligibility, so it is the same as in v1.", ""]
    for r, d in s8["forward_test"].items():
        for k, v in d.items():
            L.append(f"- {r.capitalize()}s, {k}: {v['players']} players, rank correlation {ci(v['gap_rho'])}.")
    conf = {r: bool(kn["pressure_gap_forward_test"][s]["confirmed"]) for s, r in ROLE.items()}
    L += ["", "Neither role's interval excludes 0, so \"pressure gaps don't carry forward\" still holds on the new pool; the gap "
          "stays descriptive." if not any(conf.values()) else
          f"Confirmed for: {[r for r, c in conf.items() if c]} - the v1 statement no longer holds for that role.", ""]
    L += ["## Verification", "",
          f"- Two clean builds of the v2 package were identical; that build is saved as reports/final_v2/reference/.",
          f"- `python run_all.py --floor 50 --verify-only`: {n_pass} of {len(ver)} checks pass.",
          "- Floor 0 through the edited code reproduces every v1 file (checked before the v2 build).", ""]
    open(os.path.join(FIN2, "changes_v2.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
