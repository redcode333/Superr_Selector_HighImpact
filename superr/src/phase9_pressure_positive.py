"""Phase 9 - pressure-positive eligibility rule (v3). Post-hoc, disclosed as such (decided 18 Nov after reviewing the list against the
problem statement, in the same way as the 50-ball floor of v2).

Rule, on top of the frozen floor-50 pool (>= 50 high-pressure balls, own-data >= 50%, >= 5 matches in 2025-26): a player is eligible only if
    pressure = 100 x sum(v_adj) / sum(LI) over balls with LI >= 1.5 (Phase 6b definition, same code) is > 0   (above average).
Ranking within the pool is the frozen cautious rating, exactly as before. Top 10 + the next 2 eligible ranks as alternates, per side.
Every player is treated identically; the model, the ratings and the ordering are not touched.

Reads only frozen outputs (out/model_v2, reports/final_v2, reports/phase5_v2, the Phase 3 ball values). Writes only
out/model_v3/ and reports/final_v3/. Nothing in v1 or v2 is opened for writing.

Stability: the saved Phase 5 re-rankings (500 match-bootstrap draws) hold ranks only, not per-draw pressure values, so the rule cannot be
applied inside each draw. Stability is therefore the saved draws re-ranked within the NEW eligible pool (the rule applied to the
point-estimate pool only). Restricting to a subset preserves each draw's order of the cautious ratings.
"""
import json, os, sys
import numpy as np
import pandas as pd

# Read-only use of the v2 modules: the downstream-variant switch must point at v2 so that their import-time checks pass.
os.environ["SUPERR_FLOOR"] = "50"; os.environ["SUPERR_SUFFIX"] = "_v2"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_outcome import SEED  # noqa: E402
from phase6b_gap import player_match, gap_table, repeatability, add_cols, THR  # noqa: E402
from phase6c_gap_forward import boot_rho, incremental  # noqa: E402
from phase6b_gap import summarise  # noqa: E402
from phase8_pressure_table import frame, league, shares, verdict, COLS, UNIT, ROLE  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

SUPERR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M2 = os.path.join(SUPERR, "out", "model_v2"); F2 = os.path.join(SUPERR, "reports", "final_v2"); P52 = os.path.join(SUPERR, "reports", "phase5_v2")
M3 = os.path.join(SUPERR, "out", "model_v3"); F3 = os.path.join(SUPERR, "reports", "final_v3")
SIDES = ("bat", "bowl")
RULE_TEXT = "pressure value above average: 100 x sum(v_adj) / sum(LI) over balls with LI >= 1.5 is > 0"


def row_table(t, side, rows):
    """Same columns and formulas as final_package.row_table (copied, not imported: that module writes on import)."""
    p = lambda x: round(100 * x, 3)
    out = pd.DataFrame({
        "role": ROLE[side], "rank (eligible)": rows.rank_eligible.astype(int), "player": rows["name"],
        "cautious rating (pts)": p(rows.cautious), "best estimate (pts)": p(rows.projected), "80% range low": p(rows.cautious), "80% range high": p(rows.hi80),
        "own-data %": (100 * rows.own_share).round(1), "% of re-rankings in top 10": rows.pct_top10.round(1), "median rank (re-rankings)": rows.median_rank_boot,
        "matches": rows.matches_all.astype(int), "matches 2025-26": rows.matches_2025_26.astype(int),
        "% strict-set LI powerplay": (100 * rows.share_Powerplay).round(1), "% middle": (100 * rows.share_Middle).round(1), "% death": (100 * rows.share_Death).round(1),
        "% IPL": (100 * rows.share_IPL).round(1), "strict-set pressure record (pts)": p(rows.raw_value), "strict-set balls": rows.n_p.astype(int)})
    ten = t[t.rank_eligible == 10].iloc[0]
    firsts = [next((y for y in range(2022, 2027) if r[f"Matches {y}"] > 0), 2022) for _, r in rows.iterrows()]
    out["flag: missed a full season"] = np.where(rows["Seasons with 0 matches since debut in data"] > 0,
                                                 ["yes (" + ", ".join(str(y) for y in range(2022, 2027) if r[f"Matches {y}"] == 0 and y >= first) + ")"
                                                  for (_, r), first in zip(rows.iterrows(), firsts)], "no")
    out["flag: negative strict record"] = np.where(rows.raw_value < 0, [f"yes ({100*v:+.3f} pts over {int(n)} balls)" for v, n in zip(rows.raw_value, rows.n_p)], "no")
    out["flag: close call (<50% of re-rankings in top 10)"] = np.where(rows.pct_top10 < 50, "yes", "no")
    out["% in top 10 minus final #10's"] = (rows.pct_top10 - ten.pct_top10).round(1)
    return out


def stability(R, pool):
    R = R[[p for p in R.columns if p in set(pool)]].rank(axis=1, method="first")
    return (R <= 10).mean() * 100, R.median()


def forward_eligible_row(side, elig, rng):
    """The 'eligible only, >= 20 each period' row of Phase 6c for a given pool (definitions and code of Phase 6c)."""
    pm = player_match(side); pm["season"] = pd.to_datetime(pm.date).dt.year
    tr, te = summarise(pm[pm.season <= 2024]), summarise(pm[pm.season >= 2025])
    j0 = tr.join(te, lsuffix="_tr", rsuffix="_te", how="inner").dropna(subset=["gap_tr", "gap_te"])
    j = j0[(j0.pressure_balls_tr >= 20) & (j0.pressure_balls_te >= 20) & j0.index.isin(elig)].dropna(subset=["normal_tr", "pressure_te"])
    g = boot_rho(j.gap_tr.values, j.gap_te.values, rng); nrm = boot_rho(j.normal_tr.values, j.normal_te.values, rng); inc = incremental(j, rng)
    return {"players": int(len(j)), "gap_rho": g, "normal_rho": nrm, "gap_coef_for_future_pressure": inc[:3], "normal_coef": inc[3]}


def main():
    os.makedirs(M3, exist_ok=True); os.makedirs(F3, exist_ok=True)
    kn2 = json.load(open(os.path.join(F2, "key_numbers.json")))
    top2 = pd.read_csv(os.path.join(F2, "final_top10.csv"))
    T2, T3, G, PM, ST, el3, pct3 = {}, {}, {}, {}, {}, {}, {}
    tops, alls, out_sum, lines, changes, excluded, gapstats = [], [], {"unit": UNIT, "rule": RULE_TEXT, "verdicts": {}, "forward_test": {}}, [], [], [], []
    for side in SIDES:
        role = ROLE[side]
        T = pd.read_csv(os.path.join(M2, f"phase4_rankings_{side}_all_players_technical.csv"), index_col=0)
        pm = player_match(side); g = gap_table(pm); PM[side], G[side] = pm, g
        gT = g.reindex(T.index)
        e2 = T.eligible == True
        # ---- checks against the frozen numbers (same definition as Phase 6b)
        assert (100 * T.raw_value[e2] - gT.pressure[e2]).abs().max() < 1e-6, "pressure recount differs from the strict-set record"
        assert (T.n_p[e2].astype(int) == gT.pressure_balls[e2].astype(int)).all(), "pressure ball counts differ"
        assert gT.pressure[e2].notna().all()
        for r in kn2["pressure_gap"]["top10"][side]:                      # Phase 6b numbers of the v2 top 10
            pid = T.index[(T.rank_eligible == r["rank"]) & e2][0]
            assert abs(g.loc[pid, "pressure"] - r["pressure"]) < 1e-9 and abs(g.loc[pid, "gap"] - r["gap"]) < 1e-9, r["player"]
        # ---- the rule
        rule = gT.pressure > 0
        e3 = e2 & rule
        T3s = T.copy()
        T3s["eligible_v2"], T3s["rank_eligible_v2"] = T.eligible, T.rank_eligible
        T3s["pressure_value"] = gT.pressure; T3s["passes_pressure_rule"] = np.where(e2, rule, np.nan)
        T3s["eligible"] = e3
        T3s["eligible_reason"] = [("yes" if e3.loc[i] else f"no: pressure value {gT.pressure.loc[i]:+.3f} is not above average")
                                  if e2.loc[i] else T.eligible_reason.loc[i] for i in T.index]
        el = T3s[e3].sort_values("rank_eligible_v2")
        T3s["rank_eligible"] = pd.Series(np.arange(1, len(el) + 1), index=el.index)
        # reference check: dropping the rule gives back the v2 lists exactly
        assert list(T[e2].sort_values("rank_eligible").index[:12]) == list(T3s[e2].sort_values("rank_eligible_v2").index[:12])
        # ---- stability (saved draws, re-ranked within the pools)
        R = pd.read_parquet(os.path.join(M2, f"phase5_rerankings_{side}.parquet")); R.columns = R.columns.astype(int)
        assert set(R.columns) == set(T.index[e2]), "re-rankings pool != v2 pool"
        p2, m2 = stability(R, list(T.index[e2]))
        pub = top2[top2.role == role].set_index("player")["% of re-rankings in top 10"]
        assert (p2.round(1).rename(T["name"]).reindex(pub.index) - pub).abs().max() < 0.051, "cannot reproduce the v2 stability percentages"
        p3, m3 = stability(R, list(el.index))
        T3s["pct_top10"] = p3; T3s["median_rank_boot"] = m3; T3s["pct_top10_v2"] = p2
        T3[side] = T3s; el3[side] = T3s[T3s.eligible == True].sort_values("rank_eligible")
        top, alt = el3[side].head(10), el3[side].iloc[10:12]
        ST[side] = T3s
        # ---- csv rows
        ids_all = list(el3[side].index)
        t_top = frame(g, T3s, list(top.index), side, "top 10"); t_alt = frame(g, T3s, list(alt.index), side, "alternate")
        t_all = frame(g, T3s, ids_all, side, "eligible"); lg = league(pm, ids_all, side)
        for d in (t_top, t_alt, t_all, lg):
            d["package_alternate"] = ["yes" if p in set(t_alt.player) else "no" for p in d.player]
        tops += [t_top, t_alt, lg]; alls += [t_all, lg]
        out_sum["verdicts"][role] = {"top 10": shares(t_top), "top 10 + 2 alternates": shares(pd.concat([t_top, t_alt])), "all eligible": shares(t_all),
                                     "league reference": lg.verdict.iloc[0]}
        # ---- final_top10 / alternates (v2 columns + Phase 6b columns)
        for name, rows in (("top", top), ("alt", alt)):
            tbl = row_table(T3s, side, rows)
            tbl = add_cols(tbl.reset_index(drop=True), g, pd.Series(rows.index))
            lines.append((name, tbl))
        # ---- v2 vs v3
        v2t, v3t = set(T[e2 & (T.rank_eligible <= 10)].index), set(top.index)
        v2l = set(T[e2 & (T.rank_eligible <= 12)].index); v3l = set(el3[side].head(12).index)
        for pid in sorted(v2l | v3l, key=lambda p: (min(T.rank_eligible.get(p, 99) if e2[p] else 99, 99))):
            r3 = T3s.loc[pid, "rank_eligible"]; r2 = T.loc[pid, "rank_eligible"]
            in3 = pid in v3t; in2 = pid in v2t
            if in2 and not in3:
                status, why = "left the top 10", f"pressure value {gT.pressure[pid]:+.3f} is not above average" if not rule[pid] else "pushed out by the entries"
            elif in3 and not in2:
                status, why = "entered the top 10", f"moved up from v2 rank {int(r2)}: {int((~rule[T.index[e2 & (T.rank_eligible < r2)]]).sum())} players above him in v2 failed the rule"
            elif in2 and in3:
                status, why = "stayed in the top 10", f"v2 rank {int(r2)} -> v3 rank {int(r3)}" if r2 != r3 else "unchanged"
            else:
                if pid in set(el3[side].index):
                    status, why = ("alternate (v3)" if pid in set(alt.index) else "outside top 10"), f"v2 rank {int(r2)} -> v3 rank {int(r3)}"
                else:
                    status, why = "failed the rule", f"pressure value {gT.pressure[pid]:+.3f} is not above average"
            changes.append({"role": role, "player_id": int(pid), "player": T.loc[pid, "name"], "v2_rank": int(r2) if e2[pid] else None,
                            "v3_rank": int(r3) if pd.notna(r3) else None, "status": status, "why": why,
                            "pressure_value": round(float(gT.pressure[pid]), 4), "high_pressure_balls": int(gT.pressure_balls[pid])})
        for pid in T.index[e2 & ~rule]:
            excluded.append({"role": role, "player_id": int(pid), "player": T.loc[pid, "name"], "v2_rank": int(T.loc[pid, "rank_eligible"]),
                             "pressure_value": round(float(gT.pressure[pid]), 4), "high_pressure_balls": int(gT.pressure_balls[pid]),
                             "normal_value": round(float(gT.normal[pid]), 4), "gap": round(float(gT.gap[pid]), 4)})
        # ---- gap statistics (v3 list, and the v2 list for comparison)
        for lab, tp_ids, pool_ids in (("v3 (with rule)", list(top.index), ids_all), ("v2 (without rule)", list(T[e2 & (T.rank_eligible <= 10)].index), list(T.index[e2]))):
            rest = [p for p in pool_ids if p not in set(tp_ids)]
            gs = g.reindex(tp_ids).gap; gr = g.reindex(rest).gap
            gapstats.append({"role": role, "list": lab, "pool_size": len(pool_ids), "mean_gap_top10": round(float(gs.mean()), 4), "mean_gap_rest_of_pool": round(float(gr.mean()), 4),
                             "top10_with_gap_above_0": int((gs > 0).sum()), "top10_with_pressure_above_0": int((g.reindex(tp_ids).pressure > 0).sum()),
                             "top10_80pct_range_excludes_0": int(((g.reindex(tp_ids).lo80 > 0) | (g.reindex(tp_ids).hi80 < 0)).sum())})
        failed = [f"{T.loc[p, 'name']} (v2 #{int(T.loc[p, 'rank_eligible'])}, {gT.pressure[p]:+.3f})" for p in T.index[e2 & ~rule]]
        print(f"{role}s: floor-50 pool {int(e2.sum())} -> {int(e3.sum())} eligible; failed the rule: {failed}", flush=True)

    # ---- repeatability (Phase 6b) and forward test (Phase 6c) on the v3 pool
    rep = {}
    for side in SIDES:
        r = repeatability(PM[side], set(el3[side].index))
        top20 = G[side].reindex(el3[side].head(20).index)
        rep[side] = {**r, "top20_ranges_excluding_0": int(((top20.lo80 > 0) | (top20.hi80 < 0)).sum()), "top20_expected_by_chance": 4.0}
    fw2 = json.load(open(os.path.join(P52, "phase6c_gap_forward.json")))
    fwd = {}
    for side in SIDES:
        e2set = set(ST[side].index[ST[side].eligible_v2 == True])
        chk = forward_eligible_row(side, e2set, np.random.default_rng(SEED + 71))      # not the shipped stream; sanity only (rho is stream-free)
        want = fw2[side]["eligible only, >= 20 each period"]
        assert abs(chk["gap_rho"][0] - want["gap_rho"][0]) < 1e-12 and chk["players"] == want["players"], "forward test not reproduced on the v2 pool"
        new = forward_eligible_row(side, set(el3[side].index), np.random.default_rng(SEED + 73))
        BIASED = "rule-passing pool, selected on pressure > 0 (selection-biased: the rule uses the same pressure data)"
        fwd[side] = {"eligible only, >= 20 each period": want, BIASED: new,
                     "pressure balls >= 20 each period (main)": fw2[side]["pressure balls >= 20 each period (main)"]}
        out_sum["forward_test"][ROLE[side]] = dict(fwd[side])

    # ---- write csv / tables
    rnd = lambda d: d.round({c: 4 for c in ("normal_value", "pressure_value", "gap", "gap_lo80", "gap_hi80")})
    for name, parts in (("pressure_gap_top10.csv", tops), ("pressure_gap_all_eligible.csv", alls)):
        rnd(pd.concat(parts))[COLS].assign(unit=UNIT).to_csv(os.path.join(F3, name), index=False)
    ft = pd.concat([t for n, t in lines if n == "top"]); al = pd.concat([t for n, t in lines if n == "alt"])
    ft.to_csv(os.path.join(F3, "final_top10.csv"), index=False); al.to_csv(os.path.join(F3, "alternates.csv"), index=False)
    pd.DataFrame({"role": ft.role, "rank": ft["rank (eligible)"], "player": ft.player, "selectable on Unstop (yes/no)": ""}).to_csv(os.path.join(F3, "unstop_check.csv"), index=False)
    pd.DataFrame(changes).to_csv(os.path.join(F3, "v2_vs_v3.csv"), index=False)
    pd.DataFrame(excluded).sort_values(["role", "v2_rank"]).to_csv(os.path.join(F3, "failed_pressure_rule.csv"), index=False)
    pd.DataFrame(gapstats).to_csv(os.path.join(F3, "gap_stats.csv"), index=False)
    for side in SIDES:
        t = T3[side]
        t.to_csv(os.path.join(M3, f"phase4_rankings_{side}_all_players_technical.csv"))
        t[t.eligible == True][["name", "rank_eligible", "rank_eligible_v2", "pct_top10", "pct_top10_v2", "pressure_value"]].sort_values("rank_eligible").to_csv(
            os.path.join(M3, f"stability_{side}.csv"))
        R = pd.read_parquet(os.path.join(M2, f"phase5_rerankings_{side}.parquet")); R.columns = R.columns.astype(int)
        keep = [p for p in R.columns if p in set(el3[side].index)]
        R[keep].rank(axis=1, method="first").to_parquet(os.path.join(M3, f"phase5_rerankings_{side}.parquet"))
    open(os.path.join(M3, "README_rankings.md"), "w", encoding="utf-8").write(
        "# v3 rankings\n\nSame model and ratings as v2. Eligibility adds one post-hoc rule on top of the floor-50 pool: " + RULE_TEXT + ".\n"
        "`eligible` / `rank_eligible` are the v3 values; `eligible_v2` / `rank_eligible_v2` keep the v2 ones. `pressure_value` is the Phase 6b pressure value.\n"
        "Re-rankings: the saved 500 draws re-ranked within the v3 pool (the rule is applied to the point-estimate pool only; the saved draws hold ranks, not per-draw pressure values).\n")

    # ---- key numbers (v2 structure; list-dependent parts replaced) and summary
    kn = json.loads(json.dumps(kn2))
    for side in SIDES:
        t = ST[side]
        kn["eligibility"][side].update({"eligible_floor50_before_pressure_rule": int((t.eligible_v2 == True).sum()), "eligible": int((t.eligible == True).sum()),
                                        "pressure_rule": RULE_TEXT, "fail_pressure_rule": int(((t.eligible_v2 == True) & (t.eligible != True)).sum())})
    ids = lambda rows, side: [{"rank": int(r.rank_eligible), "player": r["name"]} for _, r in rows.iterrows()]
    kn["final_top10"] = {s: [{"rank": int(r["rank (eligible)"]), "player": r.player, "cautious_pts": float(r["cautious rating (pts)"]),
                              "pct_top10": float(r["% of re-rankings in top 10"])} for _, r in ft[ft.role == ROLE[s]].iterrows()] for s in SIDES}
    kn["alternates"] = {s: [{"rank": int(r["rank (eligible)"]), "player": r.player, "pct_top10": float(r["% of re-rankings in top 10"])}
                            for _, r in al[al.role == ROLE[s]].iterrows()] for s in SIDES}
    kn["pressure_gap"] = {"definition": kn2["pressure_gap"]["definition"], "repeatability": rep,
                          "top10": {s: [{"rank": int(r["rank_eligible"]), "player": r["name"], "normal": float(G[s].loc[i, "normal"]), "pressure": float(G[s].loc[i, "pressure"]),
                                         "gap": float(G[s].loc[i, "gap"]), "lo80": float(G[s].loc[i, "lo80"]), "hi80": float(G[s].loc[i, "hi80"]),
                                         "pressure_balls": int(G[s].loc[i, "pressure_balls"])} for i, r in el3[s].head(10).iterrows()] for s in SIDES}}
    kn["pressure_gap_forward_test"] = {s: {**fw2[s], **{k: v for k, v in fwd[s].items() if k.startswith("rule-passing")}} for s in SIDES}
    kn["pressure_rule_v3"] = {"rule": RULE_TEXT, "status": "post-hoc; decided 18 Nov after reviewing the list against the problem statement, like the 50-ball floor",
                              "stability": "saved 500 re-rankings re-ranked within the v3 pool; rule applied to the point-estimate pool only (saved draws hold no per-draw pressure values)"}
    json.dump(kn, open(os.path.join(F3, "key_numbers.json"), "w"), indent=1, default=float)
    out_sum["eligibility"] = kn["eligibility"]
    out_sum["lists"] = {"with_rule": {ROLE[s]: {"top10": [r["name"] for _, r in el3[s].head(10).iterrows()], "alternates": [r["name"] for _, r in el3[s].iloc[10:12].iterrows()]} for s in SIDES},
                        "without_rule": {ROLE[s]: {"top10": [ST[s].loc[i, "name"] for i in ST[s][ST[s].eligible_v2 == True].sort_values("rank_eligible_v2").index[:10]],
                                                    "alternates": [ST[s].loc[i, "name"] for i in ST[s][ST[s].eligible_v2 == True].sort_values("rank_eligible_v2").index[10:12]]} for s in SIDES}}
    out_sum["gap_stats"] = gapstats
    json.dump(out_sum, open(os.path.join(F3, "phase9_summary.json"), "w"), indent=1, default=float)

    # ---- console report
    pd.set_option("display.width", 220)
    print(); print(pd.DataFrame(changes).to_string(index=False)); print(); print(pd.DataFrame(gapstats).to_string(index=False))
    for s in SIDES:
        print(); print(f"{ROLE[s]}s v3:"); print(ft[ft.role == ROLE[s]][["rank (eligible)", "player", "cautious rating (pts)", "% of re-rankings in top 10", "Pressure (balls >= 1.5x)", "Pressure gap", "Gap 80% range", "Pressure balls"]].to_string(index=False))
        print("alternates:"); print(al[al.role == ROLE[s]][["rank (eligible)", "player", "cautious rating (pts)", "% of re-rankings in top 10", "Pressure (balls >= 1.5x)", "Pressure gap"]].to_string(index=False))
    print(); print(json.dumps({s: fwd[s] for s in SIDES}, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
