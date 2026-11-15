"""Phase 7 - pressure-exposure eligibility floor (model unchanged; read-only on all existing outputs).

Adds a third eligibility rule to the Phase 4e rules (own-data share >= 50% AND >= 5 matches in 2025-26):
    high_pressure_balls >= FLOOR, high-pressure ball = LI >= 1.5 (same definition as strict_balls / n_p),
    batters: balls faced (rows with v_adj_bat, as in make_side), bowlers: balls bowled (all rows), full 2022-26 data.
The count is recomputed from the ball-value files and asserted equal to the existing n_p column.
Ranking within the eligible pool is the Phase 4e ordering (cautious, then projected, descending).
Stability: the saved Phase 5 re-rankings (500 match-bootstrap draws, SEED 42, eligibility = final list) are re-ranked
within the floor-50 pool. Restricting to a subset preserves the order of the cautious ratings in each draw, so this equals
re-running Phase 5 stability with the floor-50 pool and the same seed.
Writes only to <project>/outputs/phase7_exposure_floor/.
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4e_strict import SHARE_MIN, RECENT_MIN

SUPERR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.dirname(SUPERR)
MODEL = os.path.join(SUPERR, "out", "model")
FINAL = os.path.join(SUPERR, "reports", "final")
REF = os.path.join(FINAL, "reference")
PACK = os.path.join(PROJECT, "SupeRR_Data_Pack")
OUTDIR = os.path.join(PROJECT, "outputs", "phase7_exposure_floor")
FLOORS = [0, 30, 50, 75, 100]
LI_STRICT = 1.5
ROLE = {"bat": "batter", "bowl": "bowler"}


def high_pressure_balls():
    """Independent recount of LI >= 1.5 balls per player, mirroring build() + make_side()."""
    a = pd.read_parquet(os.path.join(MODEL, "phase3_ball_values_adj.parquet"),
                        columns=["match_id", "innings", "innings_ball_num", "season", "li", "striker_id", "bowler_id", "v_adj_bat", "batter_ball_faced"])
    assert a.season.between(2022, 2026).all()
    hp = a[a.li >= LI_STRICT]
    bat = hp[hp.v_adj_bat.notna()].groupby("striker_id").size()
    bowl = hp.groupby("bowler_id").size()
    faced_check = bool((a.v_adj_bat.notna() == a.batter_ball_faced.astype(bool)).all()) if "batter_ball_faced" in a else None
    return {"bat": bat, "bowl": bowl}, faced_check


def eligibility_floor(F, floor):
    """Phase 4e eligibility() + the exposure floor. No dropdown file exists (same as Phase 4e)."""
    F = F.copy()
    reasons = []
    for _, r in F.iterrows():
        why = []
        if not (r.own_share >= SHARE_MIN): why.append(f"own-data share {100 * r.own_share:.0f}% < 50%")
        if r.matches_2025_26 < RECENT_MIN: why.append(f"{int(r.matches_2025_26)} matches in 2025-26 < 5")
        if r.high_pressure_balls < floor: why.append(f"{int(r.high_pressure_balls)} high-pressure balls < {floor}")
        reasons.append("yes" if not why else "no: " + "; ".join(why))
    F["eligible"] = [x == "yes" for x in reasons]; F["eligible_reason"] = reasons
    F = F.sort_values(["cautious", "projected"], ascending=False)
    F["rank_all"] = np.arange(1, len(F) + 1)
    el = F[F.eligible]
    F["rank_eligible"] = pd.Series(np.arange(1, len(el) + 1), index=el.index)
    return F


def flatten(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items(): out.update(flatten(v, f"{pre}{k}/"))
    elif isinstance(d, list):
        for i, v in enumerate(d): out.update(flatten(v, f"{pre}{i}/"))
    else:
        out[pre[:-1]] = d
    return out


def verify_frozen(rtol=1e-6, atol=1e-9):
    """Same comparison as run_all.verify() but read-only (does not rewrite reports/final/verification.txt)."""
    rows = []
    a = pd.read_csv(os.path.join(REF, "final_top10.csv")); b = pd.read_csv(os.path.join(FINAL, "final_top10.csv"))
    for role in ("batter", "bowler"):
        ra = list(a[a.role == role].sort_values("rank (eligible)").player); rb = list(b[b.role == role].sort_values("rank (eligible)").player)
        rows.append((f"top 10 {role}s (names and order)", ra == rb))
    ka = flatten(json.load(open(os.path.join(REF, "key_numbers.json")))); kb = flatten(json.load(open(os.path.join(FINAL, "key_numbers.json"))))
    for k in sorted(set(ka) | set(kb)):
        va, vb = ka.get(k, "<missing>"), kb.get(k, "<missing>")
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
            ok = (va != va and vb != vb) or abs(va - vb) <= atol + rtol * abs(va)
        else:
            ok = va == vb
        rows.append((k, ok))
    return rows


def name_check():
    bbb = pd.read_csv(os.path.join(PACK, "ball_by_ball.csv"), usecols=["striker_id", "striker_name", "non_striker_id", "non_striker_name", "bowler_id", "bowler_name"])
    names = pd.concat([bbb[["striker_id", "striker_name"]].set_axis(["player_id", "name"], axis=1),
                       bbb[["non_striker_id", "non_striker_name"]].set_axis(["player_id", "name"], axis=1),
                       bbb[["bowler_id", "bowler_name"]].set_axis(["player_id", "name"], axis=1)]).drop_duplicates()
    mp = pd.read_csv(os.path.join(PACK, "competition_team_player_mapping.csv"), usecols=["player_id", "player_name"]).drop_duplicates()
    multi_in_ball = names.groupby("player_id").name.nunique()
    multi_in_ball = names[names.player_id.isin(multi_in_ball[multi_in_ball > 1].index)].sort_values("player_id")
    same_name_ids = names.groupby("name").player_id.nunique()
    same_name_ids = names[names.name.isin(same_name_ids[same_name_ids > 1].index)].sort_values("name")
    j = names.merge(mp, on="player_id", how="inner")
    diff_vs_mapping = j[j.name != j.player_name].drop_duplicates()
    return names, multi_in_ball, same_name_ids, diff_vs_mapping


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    log = []
    say = lambda s="": (print(s, flush=True), log.append(s))

    # ---- 1. count + assert
    hpb, faced_check = high_pressure_balls()
    tech = {}
    for side in ("bat", "bowl"):
        T = pd.read_csv(os.path.join(MODEL, f"phase4_rankings_{side}_all_players_technical.csv"), index_col=0)
        T["high_pressure_balls"] = hpb[side].reindex(T.index).fillna(0).astype(int)
        diff = T[T.high_pressure_balls != T.n_p.astype(int)]
        say(f"[1] {side}: high_pressure_balls vs existing strict_balls (n_p): {len(T)} players, {len(diff)} differ")
        if len(diff):
            say(diff[["name", "n_p", "high_pressure_balls"]].to_string())
            raise SystemExit("STOP: recount differs from strict_balls")
        tech[side] = T
    say(f"[1] check: batter 'balls faced' rows (v_adj_bat present) == batter_ball_faced flag: {faced_check}")

    # ---- 2. floors
    runs, rows = {}, []
    for side in ("bat", "bowl"):
        T = tech[side]
        for fl in FLOORS:
            F = eligibility_floor(T.drop(columns=["eligible", "eligible_reason", "rank_all", "rank_eligible"]), fl)
            if fl == 0:
                assert (F.rank_eligible.reindex(T.index).fillna(-1) == T.rank_eligible.fillna(-1)).all(), "floor 0 does not reproduce ranks"
                assert (F.eligible.reindex(T.index) == T.eligible).all()
                assert (F.rank_all.reindex(T.index) == T.rank_all).all()
                say(f"[2] {side}: FLOOR=0 reproduces current eligible set ({int(F.eligible.sum())}) and all ranks exactly - OK")
            runs[(side, fl)] = F
            el = F[F.eligible].sort_values("rank_eligible")
            for pid, r in el.iterrows():
                rows.append({"floor": fl, "role": ROLE[side], "rank": int(r.rank_eligible), "player_id": pid, "player": r["name"],
                             "cautious_rating_pts": round(100 * r.cautious, 4), "best_estimate_pts": round(100 * r.projected, 4),
                             "high_pressure_balls": int(r.high_pressure_balls),
                             "strict_record_pts": round(100 * r.raw_value, 4) if pd.notna(r.raw_value) else None,
                             "own_data_pct": round(100 * r.own_share, 1), "matches_2025_26": int(r.matches_2025_26),
                             "rank_at_floor0": int(T.loc[pid, "rank_eligible"])})
    fs = pd.DataFrame(rows)
    fs.to_csv(os.path.join(OUTDIR, "floor_sensitivity.csv"), index=False)

    summary = {}
    for side in ("bat", "bowl"):
        base = runs[(side, 0)]; base10 = list(base[base.rank_eligible <= 10].sort_values("rank_eligible").index)
        for fl in FLOORS:
            F = runs[(side, fl)]; el = F[F.eligible].sort_values("rank_eligible")
            top = list(el.index[:10]); alt = list(el.index[10:12])
            out_ = [p for p in base10 if p not in top]; in_ = [p for p in top if p not in base10]
            summary[(side, fl)] = dict(n_elig=len(el), top=top, alt=alt, out=out_, inn=in_)
            say(); say(f"[2] {ROLE[side].upper()}S  FLOOR={fl}  eligible={len(el)} (removed by floor: {len(base[base.eligible]) - len(el)})")
            say(f"    {'rk':>2}  {'player':<26}{'cautious':>9}{'hp_balls':>9}{'strict_rec':>11}{'rk@0':>6}")
            for p in top + alt:
                r = el.loc[p]
                tag = "" if p in top else "  (alternate)"
                say(f"    {int(r.rank_eligible):>2}  {r['name']:<26}{100 * r.cautious:>+9.3f}{int(r.high_pressure_balls):>9}"
                    f"{(100 * r.raw_value if pd.notna(r.raw_value) else float('nan')):>+11.3f}{int(tech[side].loc[p, 'rank_eligible']):>6}{tag}")
            nm = lambda ids: ", ".join(f"{F.loc[p, 'name']} ({int(F.loc[p, 'high_pressure_balls'])} hp balls)" for p in ids) or "-"
            say(f"    left top 10: {nm(out_)}"); say(f"    came in:     {nm(in_)}")

    # ---- 3. floor 50: stability + verification
    stab = {}
    for side in ("bat", "bowl"):
        R = pd.read_parquet(os.path.join(MODEL, f"phase5_rerankings_{side}.parquet"))
        R.columns = R.columns.astype(int)
        T = tech[side]
        assert set(R.columns) == set(T.index[T.eligible]), "re-rankings pool != current eligible pool"
        # reproduce published Phase 5 numbers at floor 0
        pub = pd.read_csv(os.path.join(FINAL, "final_top10.csv")); pub = pub[pub.role == ROLE[side]].set_index("player")["% of re-rankings in top 10"]
        b10 = T[T.rank_eligible <= 10]
        rep = pd.Series({T.loc[p, "name"]: round(100 * (R[p] <= 10).mean(), 1) for p in b10.index})
        assert (rep.reindex(pub.index) - pub).abs().max() < 0.051, "cannot reproduce Phase 5 stability"
        F = runs[(side, 50)]; pool = F.index[F.eligible]
        R50 = R[[p for p in R.columns if p in set(pool)]].rank(axis=1, method="first")
        pct = (R50 <= 10).mean() * 100
        stab[side] = pd.DataFrame({"player": F.loc[pct.index, "name"], "rank_floor50": F.loc[pct.index, "rank_eligible"].astype(int),
                                   "pct_top10_floor50": pct.round(1), "median_rank_floor50": R50.median().round(1),
                                   "pct_top10_floor0": ((R[pct.index] <= 10).mean() * 100).round(1)}).sort_values("rank_floor50")
        say(); say(f"[3] {ROLE[side]}s floor 50 stability (500 saved re-rankings, SEED 42; Phase 5 top-10 % reproduced at floor 0 - OK)")
        say(stab[side].head(14).to_string(index=False))

    vrows = verify_frozen()
    npass = sum(ok for _, ok in vrows)
    say(); say(f"[3] Phase 6 verification (frozen reports/final vs reference, read-only): {npass}/{len(vrows)} PASS")
    if npass != len(vrows):
        say("    FAIL keys: " + ", ".join(k for k, ok in vrows if not ok))
    # which checks the floor-50 list would change
    ref = flatten(json.load(open(os.path.join(REF, "key_numbers.json"))))
    list_keys = [k for k in ref if k.split("/")[0] in ("final_top10", "alternates", "eligibility") or k.startswith("pressure_gap/top10")]
    say(f"    of these, {len(list_keys) + 2} are list-dependent (top-10 names, alternates, eligibility counts, gap top-10); "
        f"the other {len(vrows) - len(list_keys) - 2} are model checks the floor cannot touch.")

    # ---- 4. files
    for side, suf in (("bat", "bat"), ("bowl", "bowl")):
        t = fs[(fs.floor == 50) & (fs.role == ROLE[side]) & (fs["rank"] <= 10)].drop(columns="floor").merge(
            stab[side][["pct_top10_floor50", "median_rank_floor50", "pct_top10_floor0"]], left_on="player_id", right_index=True, how="left")
        t.to_csv(os.path.join(OUTDIR, f"top10_floor50_{suf}.csv"), index=False)
        say(); say(f"[4] top10_floor50_{suf}.csv"); say(t.drop(columns=["player_id", "best_estimate_pts"]).to_string(index=False))

    # ---- 5. names
    names, multi, same, diffmap = name_check()
    tid = tech["bat"][tech["bat"]["name"].str.contains("Tilak", na=False)]
    say(); say("[5] Name check")
    for pid, r in tid.iterrows():
        say(f"    player_id {pid}: pipeline name '{r['name']}' (mode of striker_name in ball file); ball-file names for this id: "
            f"{sorted(names[names.player_id == pid].name.unique())}; current rank {int(r.rank_eligible)}, floor-50 rank "
            f"{runs[('bat', 50)].loc[pid, 'rank_eligible']}")
    say(f"    ids with >1 name inside ball_by_ball.csv: {multi.player_id.nunique()}")
    say(f"    names used by >1 player_id in ball_by_ball.csv: {same.name.nunique()}" + (f" -> {same.to_dict('records')}" if len(same) else ""))
    say(f"    ids whose ball-file name differs from competition_team_player_mapping.csv: {diffmap.player_id.nunique()}")
    if len(diffmap): say(diffmap.to_string(index=False))
    pd.DataFrame({"phase6_checks_pass": [npass], "phase6_checks_total": [len(vrows)]}).to_csv(os.path.join(OUTDIR, "verification_count.csv"), index=False)
    json.dump({"summary": {f"{s}|{f}": {k: [str(x) for x in v] if isinstance(v, list) else v for k, v in d.items()} for (s, f), d in summary.items()}},
              open(os.path.join(OUTDIR, "summary.json"), "w"), indent=1)
    for side in ("bat", "bowl"):
        stab[side].to_csv(os.path.join(OUTDIR, f"stability_floor50_{side}.csv"))
    open(os.path.join(OUTDIR, "console_summary.txt"), "w", encoding="utf-8").write("\n".join(map(str, log)) + "\n")


if __name__ == "__main__":
    main()
