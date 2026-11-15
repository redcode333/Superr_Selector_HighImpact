"""Phase 5b - validation test fixes only (scores, ranks and eligibility unchanged).

1. Hit rate: training top 10 in the test top 10, and in the test top third, vs 2000 random draws of 10.
2. Wider forward pools (validation only): players with leverage-sum (LI over balls with LI >= 1) >= 150 / 300 / 500 in
   BOTH 2022-24 and 2025-26; (i*) cautious rating from 2022-24 and the three baselines vs the two 2025-26 targets.
3. Rank stability for eligible ranks 11-25 from the SAME 500 re-rankings as Phase 5 (the Phase 5 random sequence is
   replayed: forward test first, then the re-rankings), plus every player outside the final top 10 whose share of
   draws in the top 10 exceeds that of final #10.
Appends a "Phase 5b" section to reports/phase5/phase5_summary.md.
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_outcome import MODEL, OUT, SEED
from variant import mpath, vdir, FLOOR
from phase4d_context import build
from phase4e_strict import make_side, player_match
from phase5_validate import istar, strict_record, baselines, boot_spearman, forward, stability, REP5, FWD_DRAWS

N_RAND = 2000
WIDE = []
POOLS = [150, 300, 500]
COLS = ["istar", "strike_rate_or_neg_economy", "total_wpa", "team_win_pct"]


def frames(s, side, matches):
    pm = player_match(s)
    tr, te = pm[pm.season <= 2024], pm[pm.season >= 2025]
    Ft = istar(tr, FWD_DRAWS, SEED + 1)
    Fe = istar(te, FWD_DRAWS, SEED + 2)
    B = baselines(s[s.season <= 2024], side, matches)
    X = pd.DataFrame({"istar": Ft.caut}).join(B, how="left")
    rec = strict_record(te)
    m2324 = tr[tr.season >= 2023].groupby("player").match_id.nunique()
    m_te = te.groupby("player").match_id.nunique()
    lev_tr, lev_te = tr.groupby("player").LHA.sum(), te.groupby("player").LHA.sum()
    targets = {"2025-26 strict-set record (LI >= 1.5)": (rec.rec, rec.L >= 30, rec.L > 0),
               "2025-26 (i*) best estimate": (Fe.Istar, m_te.reindex(Fe.index).fillna(0) >= 5, pd.Series(True, index=Fe.index))}
    orig_pool = Ft[(Ft.Istar_share >= 0.5) & (m2324.reindex(Ft.index).fillna(0) >= 5)].index
    return X, targets, orig_pool, lev_tr, lev_te


def hits(j, rng):
    rows = []
    n = len(j); third = int(np.ceil(n / 3))
    t10 = set(j.sort_values("target", ascending=False).index[:10]); t3 = set(j.sort_values("target", ascending=False).index[:third])
    draws = [set(rng.choice(j.index.values, 10, replace=False)) for _ in range(N_RAND)]
    r10 = np.array([len(d & t10) for d in draws]); r3 = np.array([len(d & t3) for d in draws])
    for c in COLS:
        est = set(j.sort_values(c, ascending=False).index[:10])
        o10, o3 = len(est & t10), len(est & t3)
        rows.append({"score": c, "pool": n, "top10_in_test_top10": o10, "random_mean_10": r10.mean(), "P(random >= obs) top10": (r10 >= o10).mean(),
                     f"top10_in_test_top_third ({third})": o3, "random_mean_third": r3.mean(), "P(random >= obs) third": (r3 >= o3).mean()})
    return pd.DataFrame(rows)


def main():
    t0 = time.perf_counter()
    d, _ = build()
    mc = pd.read_csv(os.path.join(OUT, "match_context_enriched.csv"), usecols=["match_id", "win_team_id", "super_over", "result"])
    mc["tie"] = mc.super_over.astype(bool) | mc.result.fillna("").str.contains("tie", case=False)
    matches = mc[["match_id", "win_team_id", "tie"]]
    rng = np.random.default_rng(SEED)              # replays the Phase 5 sequence
    rng2 = np.random.default_rng(SEED + 500)        # Phase 5b draws
    L = ["", "# Phase 5b - test fixes (scores, ranks and eligibility unchanged)", "",
         "The Phase 5 top-10-in-top-25 hit rate was uninformative: with pools of 31-41 players, a random draw of 10 already "
         "averaged 6-8 hits. It is replaced below by top 10 in the test top 10 and top 10 in the test top third "
         f"({N_RAND} random draws).", ""]
    for side in ("bat", "bowl"):
        role = "Batters" if side == "bat" else "Bowlers"
        s = make_side(d, side, 1.5)
        tech = pd.read_csv(mpath(f"phase4_rankings_{side}_all_players_technical.csv"), index_col=0)
        forward(s, side, matches, rng)              # consumes rng exactly as in Phase 5
        R = stability(s, tech, rng)                 # the same 500 re-rankings
        X, targets, orig_pool, lev_tr, lev_te = frames(s, side, matches)
        L += [f"## {role}", ""]
        # 1. hit rates on the original pools
        for lab, (tgt, sel, _) in targets.items():
            j = X.loc[X.index.intersection(orig_pool)].join(tgt[sel].rename("target"), how="inner").dropna()
            L += [f"### Hit rate - original pool - target: {lab}", "", hits(j, rng2).map(lambda x: f"{x:.3f}" if isinstance(x, float) else x).to_markdown(index=False), ""]
        # 2. wider pools
        for thr in POOLS:
            pool = lev_tr[lev_tr >= thr].index.intersection(lev_te[lev_te >= thr].index)
            for lab, (tgt, _, avail) in targets.items():
                j = X.loc[X.index.intersection(pool)].join(tgt[avail].rename("target"), how="inner").dropna()
                if len(j) < 10:
                    L += [f"### Wider pool leverage-sum >= {thr} in both periods - {lab}: {len(j)} players (too few)", ""]
                    continue
                sp = boot_spearman(j, COLS, "target", rng2)
                WIDE.append(sp.assign(side=side, pool=thr, target=lab))
                L += [f"### Wider pool: leverage-sum >= {thr} in both periods - target: {lab} ({len(j)} players)", "",
                      sp.map(lambda x: f"{x:.3f}" if isinstance(x, float) else x).to_markdown(index=False), ""]
        # 3. stability ranks 1-25
        el = tech[tech.eligible].sort_values("rank_eligible")
        pct = {p: 100 * (R[p] <= 10).mean() for p in R.columns}
        top10_check = pd.DataFrame({"rank": el.head(10).rank_eligible.astype(int).values, "player": el.head(10)["name"].values,
                                    "% in top 10 (replayed)": [pct[p] for p in el.head(10).index]})
        r1125 = el[(el.rank_eligible >= 11) & (el.rank_eligible <= 25)]
        st = pd.DataFrame({"final rank": r1125.rank_eligible.astype(int).values, "player": r1125["name"].values,
                           "% of draws in top 10": [pct[p] for p in r1125.index], "median rank": [R[p].median() for p in r1125.index],
                           "80% rank interval": [f"{R[p].quantile(.1):.0f}-{R[p].quantile(.9):.0f}" for p in r1125.index]})
        ten = el[el.rank_eligible == 10].index[0]
        above = el[(el.rank_eligible > 10) & el.index.map(lambda p: pct.get(p, 0) > pct[ten])]
        ab = pd.DataFrame({"final rank": above.rank_eligible.astype(int).values, "player": above["name"].values,
                           "% in top 10": [pct[p] for p in above.index], "median rank": [R[p].median() for p in above.index]})
        L += ["### Rank stability - eligible ranks 11-25 (same 500 re-rankings as Phase 5)", "",
              "Check that the replay reproduces Phase 5 (top 10):", "", top10_check.map(lambda x: f"{x:.1f}" if isinstance(x, float) else x).to_markdown(index=False), "",
              st.map(lambda x: f"{x:.1f}" if isinstance(x, float) else x).to_markdown(index=False), "",
              f"Players outside the final top 10 with a higher % of draws in the top 10 than final #10 "
              f"({el.loc[ten, 'name']}, {pct[ten]:.1f}%): " + (str(len(ab)) if len(ab) else "none") + ".", ""]
        if len(ab):
            L += [ab.map(lambda x: f"{x:.1f}" if isinstance(x, float) else x).to_markdown(index=False), ""]
        print(side, "done", flush=True)
    pd.concat(WIDE).to_csv(os.path.join(REP5, "phase5b_wider_pools.csv"), index=False)
    L += [f"Phase 5b runtime {time.perf_counter() - t0:.0f} s.", ""]
    with open(os.path.join(REP5, "phase5_summary.md"), "a", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
