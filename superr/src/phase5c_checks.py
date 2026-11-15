"""Phase 5c - extra checks kept in the repo (findings only; scores, ranks, eligibility unchanged).

a. Recency weighting: per-season forward tests T = 2024, 2025, 2026. Training seasons < T weighted 0.5^((T-1-season)/h),
   h in {inf, 3, 2, 1.5, 1}; rating = sum w v / (sum w LI + k), k = sH2 / tau_sum2 from the Phase 4 model on all data;
   target = season-T value per LI (sum v / sum LI, all balls); players with leverage-sum (LI over balls with LI >= 1)
   >= 150 in training and in season T. Spearman with 1000-player bootstrap CI per h, per season and pooled (mean of the
   three seasons).
b. Availability: season-T value ~ shrunk rating (seasons < T, equal weights) + 1[< 5 matches in T-1] + season dummies,
   T = 2023..2026, players with leverage-sum >= 150 before T and >= 50 in T; OLS with player-clustered SEs.
c. Strike rate vs (i*) for batters: per-season forward tests (T = 2024..2026; leverage-sum >= 150 before T and >= 50 in
   T) and pooled 2022-24 -> 2025-26 (>= 150 in both); targets = season-T strict-set record (LI >= 1.5, strict LI-sum >= 15)
   and season-T overall value per LI; paired player-bootstrap differences.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_outcome import MODEL, SEED
from variant import mpath, vdir, FLOOR
from phase4_ratings import player_stats, estimate
from phase4_ratings import player_match as pm_phase4
from phase4d_context import build
from phase4e_strict import make_side, player_match, features
from phase4_ratings import Z80

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP5 = vdir(os.path.join(ROOT, "reports", "phase5"))
HS = [np.inf, 3, 2, 1.5, 1]
N_BOOT = 1000


def ps_table(s):
    t = s.assign(LHA=np.where(s.li >= 1, s.li, 0.0), Hs=np.where(s.hi, s.v, 0.0), Ls=np.where(s.hi, s.li, 0.0))
    g = t.groupby(["player", "season"]).agg(V=("v", "sum"), L=("li", "sum"), LHA=("LHA", "sum"), Hs=("Hs", "sum"), Ls=("Ls", "sum"),
                                            M=("match_id", "nunique")).reset_index()
    return g


def sp_ci(x, y, rng):
    x, y = np.asarray(x), np.asarray(y); n = len(x)
    idx = rng.integers(0, n, (N_BOOT, n))
    reps = [spearmanr(x[i], y[i]).statistic for i in idx]
    return spearmanr(x, y).statistic, np.percentile(reps, 2.5), np.percentile(reps, 97.5), np.array(reps)


def main():
    t0 = time.perf_counter()
    d, _ = build()
    rng = np.random.default_rng(SEED + 900)
    res, L = {}, ["", "# Phase 5c - extra checks (findings only; scores, ranks and eligibility unchanged)", ""]
    for side in ("bat", "bowl"):
        role = "Batters" if side == "bat" else "Bowlers"
        s = make_side(d, side, 1.5)
        k = (lambda vc: vc["sH2"] / vc["tau_sum2"])(estimate(player_stats(pm_phase4(s.assign(hi=s.li >= 1)))))
        ps = ps_table(s)
        piv = {c: ps.pivot(index="player", columns="season", values=c).fillna(0) for c in ("V", "L", "LHA", "Hs", "Ls", "M")}
        # a. recency
        rows, reps_all = [], {}
        for T in (2024, 2025, 2026):
            tr = [y for y in piv["V"].columns if y < T]
            lha_tr = piv["LHA"][tr].sum(1); lha_T = piv["LHA"][T]
            pool = lha_tr.index[(lha_tr >= 150) & (lha_T >= 150)]
            y = (piv["V"][T] / piv["L"][T]).loc[pool]
            idx = rng.integers(0, len(pool), (N_BOOT, len(pool)))
            for h in HS:
                w = {yy: (1.0 if np.isinf(h) else 0.5 ** ((T - 1 - yy) / h)) for yy in tr}
                Vw = sum(piv["V"][yy] * w[yy] for yy in tr); Lw = sum(piv["L"][yy] * w[yy] for yy in tr)
                r = (Vw / (Lw + k)).loc[pool]
                rho = spearmanr(r, y).statistic
                rp = np.array([spearmanr(r.values[i], y.values[i]).statistic for i in idx])
                reps_all[(T, h)] = rp
                rows.append({"season T": T, "h": "inf (equal)" if np.isinf(h) else h, "players": len(pool), "spearman": rho,
                             "lo95": np.percentile(rp, 2.5), "hi95": np.percentile(rp, 97.5)})
        rec = pd.DataFrame(rows)
        pooled = []
        for h in HS:
            m = np.mean([reps_all[(T, h)] for T in (2024, 2025, 2026)], axis=0)
            pooled.append({"h": "inf (equal)" if np.isinf(h) else h,
                           "mean spearman (3 seasons)": rec[rec.h == ("inf (equal)" if np.isinf(h) else h)].spearman.mean(),
                           "lo95": np.percentile(m, 2.5), "hi95": np.percentile(m, 97.5),
                           "minus equal": rec[rec.h == ("inf (equal)" if np.isinf(h) else h)].spearman.mean() - rec[rec.h == "inf (equal)"].spearman.mean(),
                           "diff lo95": np.percentile(m - np.mean([reps_all[(T, np.inf)] for T in (2024, 2025, 2026)], axis=0), 2.5),
                           "diff hi95": np.percentile(m - np.mean([reps_all[(T, np.inf)] for T in (2024, 2025, 2026)], axis=0), 97.5)})
        pooled = pd.DataFrame(pooled)
        best_h = pooled.loc[pooled["mean spearman (3 seasons)"].idxmax(), "h"]
        # b. availability
        panel = []
        for T in (2023, 2024, 2025, 2026):
            tr = [yy for yy in piv["V"].columns if yy < T]
            lha_tr = piv["LHA"][tr].sum(1)
            pool = lha_tr.index[(lha_tr >= 150) & (piv["LHA"][T] >= 50)]
            r = (piv["V"][tr].sum(1) / (piv["L"][tr].sum(1) + k)).loc[pool]
            panel.append(pd.DataFrame({"player": pool, "T": T, "y": (piv["V"][T] / piv["L"][T]).loc[pool].values, "rating": r.values,
                                       "low_prev": (piv["M"][T - 1].loc[pool] < 5).astype(float).values}))
        pn = pd.concat(panel, ignore_index=True)
        X = pd.get_dummies(pn[["rating", "low_prev"]].assign(T=pn["T"].astype(str)), columns=["T"], drop_first=True).astype(float)
        fit = sm.OLS(pn.y.values, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(pn.player)[0]})
        lo, hi = fit.conf_int().loc["low_prev"]
        avail = {"coef_low_prev": float(fit.params["low_prev"]), "lo95": float(lo), "hi95": float(hi), "p": float(fit.pvalues["low_prev"]),
                 "coef_rating": float(fit.params["rating"]), "n_obs": int(len(pn)), "n_players": int(pn.player.nunique()),
                 "n_low_prev": int(pn.low_prev.sum())}
        L += [f"## {role}", "", f"Shrinkage constant k (Phase 4 model, all data): {k:.0f} units of leverage.", "",
              "### a. Recency weighting (per-season forward tests)", "", rec.map(lambda x: f"{x:.3f}" if isinstance(x, float) else x).to_markdown(index=False), "",
              pooled.map(lambda x: f"{x:.3f}" if isinstance(x, float) else x).to_markdown(index=False), "",
              f"Highest mean Spearman: h = {best_h}.", "",
              "### b. Availability (< 5 matches in the previous season)", "",
              pd.DataFrame([avail]).map(lambda x: f"{x:.5f}" if isinstance(x, float) else x).to_markdown(index=False), ""]
        res[side] = {"k": k, "recency_by_season": rec.to_dict("records"), "recency_pooled": pooled.to_dict("records"),
                     "recency_best_h": str(best_h), "availability": avail}
        # c. strike rate vs (i*) for batters
        if side == "bat":
            pmS = player_match(s)
            sr_ps = s.groupby(["player", "season"]).agg(runs=("bat_runs", "sum"), balls=("bat_runs", "size")).reset_index()
            crow = []
            def compare(train_seasons, test_seasons, pool, lab, seed):
                F = features(pmS[pmS.season.isin(train_seasons)], istar_draws=200, seed=seed)[0]
                ist = F.Istar - Z80 * np.sqrt(F.Istar_var)
                srt = sr_ps[sr_ps.season.isin(train_seasons)].groupby("player")[["runs", "balls"]].sum()
                sr = 100 * srt.runs / srt.balls
                Vt = piv["V"][test_seasons].sum(1); Lt = piv["L"][test_seasons].sum(1)
                Hs = piv["Hs"][test_seasons].sum(1); Ls = piv["Ls"][test_seasons].sum(1)
                for tlab, tgt, sel in (("strict-set record (LI >= 1.5)", Hs / Ls.where(Ls > 0), Ls >= 15), ("overall value per LI", Vt / Lt, Lt > 0)):
                    j = pd.DataFrame({"istar": ist, "sr": sr, "y": tgt[sel]}).loc[lambda x: x.index.isin(pool)].dropna()
                    if len(j) < 10:
                        continue
                    ri, li_, hi_, rpi = sp_ci(j.istar, j.y, np.random.default_rng(seed))
                    rs, ls, hs, rps = sp_ci(j.sr, j.y, np.random.default_rng(seed))
                    crow.append({"test": lab, "target": tlab, "players": len(j), "(i*)": ri, "(i*) lo95": li_, "(i*) hi95": hi_,
                                 "strike rate": rs, "SR lo95": ls, "SR hi95": hs, "SR - (i*)": rs - ri,
                                 "diff lo95": np.percentile(rps - rpi, 2.5), "diff hi95": np.percentile(rps - rpi, 97.5)})
            for T in (2024, 2025, 2026):
                tr = [yy for yy in piv["V"].columns if yy < T]
                lha_tr = piv["LHA"][tr].sum(1)
                pool = lha_tr.index[(lha_tr >= 150) & (piv["LHA"][T] >= 50)]
                compare(tr, [T], set(pool), f"seasons < {T} -> {T}", SEED + T)
            lha_a, lha_b = piv["LHA"][[2022, 2023, 2024]].sum(1), piv["LHA"][[2025, 2026]].sum(1)
            compare([2022, 2023, 2024], [2025, 2026], set(lha_a.index[(lha_a >= 150) & (lha_b >= 150)]), "2022-24 -> 2025-26 (pooled)", SEED + 7)
            ct = pd.DataFrame(crow)
            L += ["### c. Strike rate vs (i*) (finding only; the score is not changed)", "",
                  ct.map(lambda x: f"{x:.3f}" if isinstance(x, float) else x).to_markdown(index=False), ""]
            res[side]["strike_rate_vs_istar"] = ct.to_dict("records")
        print(side, "done", flush=True)
    L += [f"Phase 5c runtime {time.perf_counter() - t0:.0f} s.", ""]
    with open(os.path.join(REP5, "phase5_summary.md"), "a", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    json.dump(res, open(os.path.join(REP5, "phase5c_results.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
