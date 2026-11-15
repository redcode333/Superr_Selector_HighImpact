"""Phase 6c - confirmatory forward test of the normal-vs-pressure gap (descriptive; no change to scores or ranks).

Pre-stated criterion: the gap is confirmed as repeatable only if Spearman(gap 2022-24, gap 2025-26) > 0 with the 95%
player-bootstrap CI excluding 0 (main pool: players with >= 20 pressure balls (LI >= 1.5) in each period).
Also reported: pools with >= 40 / >= 60 pressure balls in each period, the eligible-only subset, and an incremental
check - OLS of 2025-26 pressure value on 2022-24 normal and 2022-24 gap (player-bootstrap CI of the gap coefficient).
Definitions as Phase 6b: normal / pressure = 100 x sum(v_adj) / sum(LI) over balls with LI < 1.5 / >= 1.5.
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_outcome import MODEL, SEED
from variant import mpath, vdir, FLOOR
from phase6b_gap import player_match, summarise

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP5 = vdir(os.path.join(ROOT, "reports", "phase5"))
N_BOOT = 1000


def boot_rho(x, y, rng):
    n = len(x); idx = rng.integers(0, n, (N_BOOT, n))
    reps = [spearmanr(x[i], y[i]).statistic for i in idx]
    return float(spearmanr(x, y).statistic), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def incremental(j, rng):
    X = np.column_stack([np.ones(len(j)), j.normal_tr, j.gap_tr]); y = j.pressure_te.values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    n = len(j); reps = []
    for i in rng.integers(0, n, (N_BOOT, n)):
        reps.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0][2])
    return float(beta[2]), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5)), float(beta[1])


def main():
    rng = np.random.default_rng(SEED + 71)
    out, rows = {}, []
    for side in ("bat", "bowl"):
        pm = player_match(side)
        pm["season"] = pd.to_datetime(pm.date).dt.year
        tr, te = summarise(pm[pm.season <= 2024]), summarise(pm[pm.season >= 2025])
        j0 = tr.join(te, lsuffix="_tr", rsuffix="_te", how="inner").dropna(subset=["gap_tr", "gap_te"])
        tech = pd.read_csv(mpath(f"phase4_rankings_{side}_all_players_technical.csv"), index_col=0)
        elig = set(tech[tech.eligible == True].index)
        res = {}
        for lab, sel in (("pressure balls >= 20 each period (main)", (j0.pressure_balls_tr >= 20) & (j0.pressure_balls_te >= 20)),
                         ("pressure balls >= 40 each period", (j0.pressure_balls_tr >= 40) & (j0.pressure_balls_te >= 40)),
                         ("pressure balls >= 60 each period", (j0.pressure_balls_tr >= 60) & (j0.pressure_balls_te >= 60)),
                         ("eligible only, >= 20 each period", (j0.pressure_balls_tr >= 20) & (j0.pressure_balls_te >= 20) & j0.index.isin(elig))):
            j = j0[sel].dropna(subset=["normal_tr", "pressure_te"])
            if len(j) < 15:
                res[lab] = {"players": int(len(j))}; continue
            g = boot_rho(j.gap_tr.values, j.gap_te.values, rng)
            nrm = boot_rho(j.normal_tr.values, j.normal_te.values, rng)
            inc = incremental(j, rng)
            res[lab] = {"players": int(len(j)), "gap_rho": g, "normal_rho": nrm, "gap_coef_for_future_pressure": inc[:3], "normal_coef": inc[3]}
            rows.append({"role": "batters" if side == "bat" else "bowlers", "pool": lab, "players": len(j),
                         "gap 22-24 vs gap 25-26": f"{g[0]:+.3f} [{g[1]:+.3f}, {g[2]:+.3f}]",
                         "normal 22-24 vs normal 25-26": f"{nrm[0]:+.3f} [{nrm[1]:+.3f}, {nrm[2]:+.3f}]",
                         "gap coefficient (future pressure ~ normal + gap)": f"{inc[0]:+.3f} [{inc[1]:+.3f}, {inc[2]:+.3f}]"})
        main_ = res["pressure balls >= 20 each period (main)"]
        res["confirmed"] = bool(main_.get("gap_rho") and main_["gap_rho"][0] > 0 and main_["gap_rho"][1] > 0)
        out[side] = res
    tab = pd.DataFrame(rows)
    L = ["", "# Phase 6c - confirmatory forward test of the pressure gap (descriptive; scores and ranks unchanged)", "",
         "Pre-stated criterion: confirmed only if Spearman(gap 2022-24, gap 2025-26) > 0 with the 95% player-bootstrap CI excluding 0 "
         "in the main pool (>= 20 pressure balls in each period). Gap coefficient: OLS of 2025-26 pressure value on 2022-24 normal "
         "and 2022-24 gap, player-bootstrap 95% CI.", "", tab.to_markdown(index=False), "",
         f"Confirmed: batters {out['bat']['confirmed']}, bowlers {out['bowl']['confirmed']}.", ""]
    with open(os.path.join(REP5, "phase5_summary.md"), "a", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    json.dump(out, open(os.path.join(REP5, "phase6c_gap_forward.json"), "w"), indent=1, default=float)
    import re
    fin = vdir(os.path.join(ROOT, "reports", "final"))
    kp = os.path.join(fin, "key_numbers.json"); kn = json.load(open(kp)); kn["pressure_gap_forward_test"] = out
    json.dump(kn, open(kp, "w"), indent=1, default=float)
    mp = os.path.join(fin, "method_log.md"); ml = open(mp, encoding="utf-8").read()
    ml = re.sub(r"\n21\. Confirmatory forward test.*\n?", "\n", ml).rstrip("\n")
    b, w = out["bat"]["pressure balls >= 20 each period (main)"], out["bowl"]["pressure balls >= 20 each period (main)"]
    conf = lambda s: "confirmed" if out[s]["confirmed"] else "not confirmed"
    ml += (f"\n21. Confirmatory forward test of the gap (2022-24 gap vs 2025-26 gap, >= 20 pressure balls in each period): batters "
           f"rho = {b['gap_rho'][0]:+.3f} [{b['gap_rho'][1]:+.3f}, {b['gap_rho'][2]:+.3f}] (n = {b['players']}, {conf('bat')}), bowlers "
           f"{w['gap_rho'][0]:+.3f} [{w['gap_rho'][1]:+.3f}, {w['gap_rho'][2]:+.3f}] (n = {w['players']}, {conf('bowl')}) "
           f"(pre-stated criterion: CI above 0). Gap not used in the ranking.\n")
    open(mp, "w", encoding="utf-8").write(ml)
    print(tab.to_string(index=False)); print({s: out[s]["confirmed"] for s in out})


if __name__ == "__main__":
    main()
