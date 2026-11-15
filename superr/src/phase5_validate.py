"""Phase 5 - validation of the final (i*) rankings (no refitting of the choice) + availability columns.

1. Forward test: (i*) fitted on 2022-24 matches only; ranked by its cautious rating (mean - 1.2816 sd, posterior +
   hyper-parameter uncertainty, 200 draws) within the training-eligible pool (own-data share >= 50%, >= 5 matches in
   2023-24). Targets: (a) 2025-26 strict-set (LI >= 1.5) record, sum v_adj / sum LI, players with LI-sum >= 30 in 2025-26;
   (b) 2025-26 (i*) fitted on 2025-26 matches only, players with >= 5 matches in 2025-26. Spearman with 1000-player
   bootstrap CIs; top-10 hit rate (training top 10 in the test top 25 of the same pool) vs 1000 random draws of 10.
2. Baselines on the same players and targets: strike rate (batters) / minus economy (bowlers), total WPA, team win %
   when he plays (ties 0.5). Paired bootstrap differences vs (i*).
3. Rank stability: 500 re-rankings of the final model on match-resampled data (each player's matches resampled with
   replacement; (i*) with 20 inner draws per re-ranking; eligibility held at the final list).
4. Eligibility sensitivity: own-data share 45% / 55%, 2025-26 matches 3 / 8, rank by best estimate.
Availability columns (descriptive; scores and ranks unchanged) are appended to both rankings CSVs.
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_outcome import MODEL, OUT, SEED
from variant import mpath, vdir, FLOOR
from phase4_ratings import Z80
from phase4b_ratings import fast_stats, estimate, clip_vc, post_arrays, boot_vcs
from phase4c_projection import resample_weights
from phase4d_context import build
from phase4e_strict import make_side, player_match, features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP5 = vdir(os.path.join(ROOT, "reports", "phase5"))
N_BOOT, N_STAB, FWD_DRAWS, STAB_INNER = 1000, 500, 200, 20
SEASONS = [2022, 2023, 2024, 2025, 2026]


def istar(pm, draws, seed):
    F, _ = features(pm, istar_draws=draws, seed=seed)
    F["caut"] = F.Istar - Z80 * np.sqrt(F.Istar_var)
    return F


def strict_record(pm):
    g = pm.groupby("player").agg(H=("H", "sum"), L=("SL", "sum"))
    return pd.DataFrame({"rec": g.H / g.L.where(g.L > 0), "L": g.L})


def baselines(s, side, matches):
    """Training-period baselines per player (higher = better)."""
    if side == "bat":
        sr = 100 * s.groupby("player").bat_runs.sum() / s.groupby("player").size()
        wpa = s.groupby("player").wpa.sum()
        team = s[["player", "match_id", "batting_team_id"]].drop_duplicates(["player", "match_id"]).rename(columns={"batting_team_id": "team"})
    else:
        lg = s.groupby("player").is_legal.sum()
        sr = -(6 * s.groupby("player").bowler_runs_conceded.sum() / lg.where(lg > 0))
        wpa = -s.groupby("player").wpa.sum()
        team = s[["player", "match_id", "bowling_team_id"]].drop_duplicates(["player", "match_id"]).rename(columns={"bowling_team_id": "team"})
    t = team.merge(matches, on="match_id", how="left")
    t["won"] = np.where(t.tie, 0.5, (t.win_team_id == t.team).astype(float))
    tw = t[t.win_team_id.notna() | t.tie].groupby("player").won.mean()
    return pd.DataFrame({"strike_rate_or_neg_economy": sr, "total_wpa": wpa, "team_win_pct": tw})


def boot_spearman(df, cols, target, rng):
    n = len(df); idx = rng.integers(0, n, (N_BOOT, n))
    y = df[target].values; ry = np.argsort(np.argsort(y[idx], axis=1), axis=1)
    reps = {}
    for c in cols:
        rx = np.argsort(np.argsort(df[c].values[idx], axis=1), axis=1)
        reps[c] = np.array([np.corrcoef(rx[b], ry[b])[0, 1] for b in range(N_BOOT)])
    rows = []
    for c in cols:
        r = spearmanr(df[c], y).statistic
        lo, hi = np.percentile(reps[c], [2.5, 97.5])
        row = {"score": c, "n_players": n, "spearman": r, "lo95": lo, "hi95": hi}
        if c != "istar":
            dd = reps[c] - reps["istar"]
            row.update({"minus_istar": r - spearmanr(df["istar"], y).statistic, "diff_lo95": np.percentile(dd, 2.5), "diff_hi95": np.percentile(dd, 97.5)})
        rows.append(row)
    return pd.DataFrame(rows)


def hit_rate(df, score, target, rng, k=10, top=25):
    est = set(df.sort_values(score, ascending=False).index[:k]); tt = set(df.sort_values(target, ascending=False).index[:top])
    obs = len(est & tt)
    draws = np.array([len(set(rng.choice(df.index.values, k, replace=False)) & tt) for _ in range(1000)])
    return {"score": score, "pool": len(df), "hits_top10_in_test_top25": obs, "random_mean": draws.mean(),
            "random_95pct_interval": f"{np.percentile(draws, 2.5):.0f}-{np.percentile(draws, 97.5):.0f}", "P(random >= observed)": (draws >= obs).mean()}


def forward(s, side, matches, rng):
    pm = player_match(s)
    tr, te = pm[pm.season <= 2024], pm[pm.season >= 2025]
    Ft = istar(tr, FWD_DRAWS, SEED + 1)
    m2324 = tr[tr.season >= 2023].groupby("player").match_id.nunique()
    pool = Ft[(Ft.Istar_share >= 0.5) & (m2324.reindex(Ft.index).fillna(0) >= 5)]
    B = baselines(s[s.season <= 2024], side, matches)
    X = pd.DataFrame({"istar": pool.caut}).join(B, how="left")
    rec = strict_record(te)
    Fe = istar(te, FWD_DRAWS, SEED + 2)
    m_te = te.groupby("player").match_id.nunique()
    out = {}
    for lab, tgt, sel in (("2025-26 strict-set record (LI >= 1.5)", rec.rec, rec.L >= 30), ("2025-26 (i*) best estimate", Fe.Istar, m_te.reindex(Fe.index).fillna(0) >= 5)):
        j = X.join(tgt[sel].rename("target"), how="inner").dropna()
        cols = ["istar", "strike_rate_or_neg_economy", "total_wpa", "team_win_pct"]
        sp = boot_spearman(j, cols, "target", rng)
        hits = pd.DataFrame([hit_rate(j, c, "target", rng) for c in cols])
        out[lab] = (sp, hits, len(j))
    return out, len(pool)


def stability(s, tech, rng):
    pm = player_match(s)
    elig = tech[tech.eligible].index
    players = np.unique(pm.player.values); codes = np.searchsorted(players, pm.player.values); nP = len(players)
    ranks = []
    for b in range(N_STAB):
        w = resample_weights(pm, rng)
        g = fast_stats(codes, nP, w * pm.HA.values, w * pm.LHA.values, w * pm.LA.values, w * pm.LLA.values, np.bincount(codes, w, nP))
        post = [post_arrays(g.LH.values, g.LL.values, g.y1.values, g.y2.values, clip_vc(v)) for v in boot_vcs(g, STAB_INNER, SEED + 1000 + b)]
        ms, vs = np.array([p["m"] for p in post]), np.array([p["v"] for p in post])
        caut = pd.Series(ms.mean(0) - Z80 * np.sqrt(vs.mean(0) + ms.var(0)), index=players)
        ce = caut.reindex(elig).dropna()
        ranks.append(ce.rank(ascending=False, method="first"))
    R = pd.DataFrame(ranks)
    return R


def availability(s):
    pm = s[["player", "match_id", "season"]].drop_duplicates()
    t = pm.groupby(["player", "season"]).match_id.nunique().unstack("season").reindex(columns=SEASONS).fillna(0).astype(int)
    first = pm.groupby("player").season.min()
    zero = pd.Series([int(sum(t.loc[p, y] == 0 for y in SEASONS if y >= first[p])) for p in t.index], index=t.index)
    out = t.rename(columns={y: f"Matches {y}" for y in SEASONS})
    out["Seasons with 0 matches since debut in data"] = zero
    out["Matches 2025"], out["Matches 2026"] = t[2025], t[2026]
    return out


def main():
    t0 = time.perf_counter()
    os.makedirs(REP5, exist_ok=True)
    d, _ = build()
    mc = pd.read_csv(os.path.join(OUT, "match_context_enriched.csv"), usecols=["match_id", "win_team_id", "super_over", "result"])
    mc["tie"] = mc.super_over.astype(bool) | mc.result.fillna("").str.contains("tie", case=False)
    matches = mc[["match_id", "win_team_id", "tie"]]
    rng = np.random.default_rng(SEED)
    res = {}
    for side in ("bat", "bowl"):
        s = make_side(d, side, 1.5)
        tech = pd.read_csv(mpath(f"phase4_rankings_{side}_all_players_technical.csv"), index_col=0)
        tech = tech.drop(columns=[c for c in tech.columns if c.startswith("Matches 20") or c in
                                  ("Seasons with 0 matches since debut in data", "missed_full_season_top25")], errors="ignore")
        fwd, npool = forward(s, side, matches, rng)
        R = stability(s, tech, rng)
        Rs = R.copy(); Rs.columns = Rs.columns.astype(str)
        Rs.to_parquet(mpath(f"phase5_rerankings_{side}.parquet"), index=False)
        fw_rows = []
        for lab, (sp, hits, n) in fwd.items():
            fw_rows.append(sp.assign(side=side, target=lab))
        pd.concat(fw_rows).to_csv(os.path.join(REP5, f"phase5_forward_tests_{side}.csv"), index=False)
        top10 = tech[tech.rank_eligible <= 10].sort_values("rank_eligible")
        stab = pd.DataFrame({"final rank": top10.rank_eligible.astype(int).values, "player": top10["name"].values,
                             "% of draws in top 10": [100 * (R[p] <= 10).mean() for p in top10.index],
                             "median rank": [R[p].median() for p in top10.index],
                             "80% rank interval": [f"{R[p].quantile(.1):.0f}-{R[p].quantile(.9):.0f}" for p in top10.index]})
        pair = None
        if side == "bowl":
            ids = {n: tech.index[tech["name"] == n][0] for n in ("Mohsin Khan", "Bhuvneshwar Kumar") if (tech["name"] == n).any()}
            pair = pd.DataFrame([{"player": n, "final rank": int(tech.loc[i, "rank_eligible"]), "% of draws in top 10": 100 * (R[i] <= 10).mean(),
                                  "median rank": R[i].median(), "matches": int(tech.loc[i, "matches_all"]), "matches 2025-26": int(tech.loc[i, "matches_2025_26"]),
                                  "own-data %": round(100 * tech.loc[i, "own_share"], 1),
                                  "% draws Mohsin ranked above Bhuvneshwar": None} for n, i in ids.items()])
            if len(ids) == 2:
                pair["% draws Mohsin ranked above Bhuvneshwar"] = 100 * (R[ids["Mohsin Khan"]] < R[ids["Bhuvneshwar Kumar"]]).mean()
        # sensitivity
        base = set(tech[tech.rank_eligible <= 10].index)
        sens = []
        for lab, mask, key in (("final (share >= 50%, >= 5 matches 2025-26, cautious)", (tech.own_share >= .5) & (tech.matches_2025_26 >= 5) & (tech.n_p >= FLOOR), "cautious"),
                               ("own-data share >= 45%", (tech.own_share >= .45) & (tech.matches_2025_26 >= 5) & (tech.n_p >= FLOOR), "cautious"),
                               ("own-data share >= 55%", (tech.own_share >= .55) & (tech.matches_2025_26 >= 5) & (tech.n_p >= FLOOR), "cautious"),
                               (">= 3 matches 2025-26", (tech.own_share >= .5) & (tech.matches_2025_26 >= 3) & (tech.n_p >= FLOOR), "cautious"),
                               (">= 8 matches 2025-26", (tech.own_share >= .5) & (tech.matches_2025_26 >= 8) & (tech.n_p >= FLOOR), "cautious"),
                               ("rank by best estimate", (tech.own_share >= .5) & (tech.matches_2025_26 >= 5) & (tech.n_p >= FLOOR), "projected")):
            top = tech[mask].sort_values(key, ascending=False).head(10)
            sens.append({"variant": lab, "eligible": int(mask.sum()), "overlap with final top 10": len(base & set(top.index)),
                         "in": ", ".join(tech.loc[list(set(top.index) - base), "name"]) or "-", "out": ", ".join(tech.loc[list(base - set(top.index)), "name"]) or "-"})
        # availability columns
        av = availability(s)
        plain_path = mpath(f"phase4_rankings_{side}_all_players.csv")
        plain = pd.read_csv(plain_path)
        plain = plain[[c for c in plain.columns if not c.startswith("Matches 20") and c not in ("Seasons with 0 matches since debut in data", "Missed a full season (eligible top 25)")] + ["Matches 2025-26"]] if "Matches 2025-26" in plain else plain
        plain = plain.loc[:, ~plain.columns.duplicated()]
        name_to_id = tech.reset_index()[["player_id", "name", "rank_all"]]
        plain = plain.merge(name_to_id, left_on=["Player", "Rank (all players)"], right_on=["name", "rank_all"], how="left")
        av_cols = av.reindex(plain.player_id.values).reset_index(drop=True)
        plain = pd.concat([plain.drop(columns=["name", "rank_all"]).reset_index(drop=True), av_cols], axis=1)
        top25 = plain["Rank (eligible)"].notna() & (plain["Rank (eligible)"] <= 25)
        plain["Missed a full season (eligible top 25)"] = np.where(top25, np.where(plain["Seasons with 0 matches since debut in data"] > 0, "yes", "no"), "")
        plain.drop(columns=["player_id"]).to_csv(plain_path, index=False)
        tech2 = tech.join(av)
        tech2["missed_full_season_top25"] = (tech2.rank_eligible <= 25) & (tech2["Seasons with 0 matches since debut in data"] > 0)
        tech2.to_csv(mpath(f"phase4_rankings_{side}_all_players_technical.csv"))
        missed = tech2[tech2.missed_full_season_top25].sort_values("rank_eligible")
        missed_t = pd.DataFrame({"rank (eligible)": missed.rank_eligible.astype(int), "player": missed["name"],
                                 **{f"{y}": missed[f"Matches {y}"] for y in SEASONS},
                                 "seasons with 0 since debut": missed["Seasons with 0 matches since debut in data"]})
        res[side] = dict(fwd=fwd, npool=npool, stab=stab, pair=pair, sens=pd.DataFrame(sens), missed=missed_t)
        print(side, "done", flush=True)
    write_report(res, time.perf_counter() - t0)


def write_report(res, rt):
    f3 = lambda x: f"{x:.3f}" if isinstance(x, (float, np.floating)) else x
    L = ["# Phase 5 - validation of the final (i*) rankings", "",
         "No refitting of the choice. Forward test: (i*) fitted on 2022-24 matches only (200 hyper-parameter draws), ranked by "
         "its cautious rating within the training-eligible pool (own-data share >= 50%, >= 5 matches in 2023-24). 95% CIs from "
         "1000 player-bootstrap resamples. Baselines are computed on 2022-24 only; strike rate / economy and total WPA use all "
         "balls; team win % = share of his matches won by his team (ties 0.5).", ""]
    for side, R in res.items():
        role = "Batters" if side == "bat" else "Bowlers"
        L += [f"## {role}", "", f"Training-eligible pool: {R['npool']} players.", ""]
        for lab, (sp, hits, n) in R["fwd"].items():
            L += [f"### Forward test - target: {lab} ({n} players)", "", sp.map(f3).to_markdown(index=False), "",
                  "Top-10 hit rate (2022-24 top 10 in the 2025-26 top 25 of the same pool) vs 1000 random draws of 10:", "",
                  hits.map(f3).to_markdown(index=False), ""]
        L += ["### Rank stability (500 match-bootstrap re-rankings of the final model; eligibility fixed)", "",
              R["stab"].map(f3).to_markdown(index=False), ""]
        if R["pair"] is not None:
            L += ["Mohsin Khan vs Bhuvneshwar Kumar:", "", R["pair"].map(f3).to_markdown(index=False), ""]
        L += ["### Eligibility / ranking sensitivity", "", R["sens"].to_markdown(index=False), "",
              "### Eligible top-25 players who missed a full season (0 matches in a season since first appearing in the data)", "",
              (R["missed"].to_markdown(index=False) if len(R["missed"]) else "None."), ""]
    L += ["## Tested and rejected (recorded)", "",
          "- Recency weighting of matches: rejected - in forward tests 2024-26, equal weighting was best.",
          "- Availability penalty: rejected - effect of < 5 previous-season matches for bowlers +0.00003 [-0.00089, +0.00096].",
          "- Availability columns added to both rankings CSVs are descriptive only; scores and ranks are unchanged.", "",
          f"Runtime {rt:.0f} s.", ""]
    open(os.path.join(REP5, "phase5_summary.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
