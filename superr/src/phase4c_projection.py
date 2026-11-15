"""Phase 4b - projection model for pressure-ball value (replaces the uncertainty-averaged rating).

Unit of everything below: player-match aggregates of the corrected per-ball values v_adj (Phase 3).
Pressure ball = LI >= 1. Target = held-out H/LH (sum of v_adj over pressure balls / sum of their LI), weight = held-out LH.

Features (training matches only; rates shrunk by empirical Bayes, method of moments, towards the training mean):
  batters: pressure runs per ball faced; pressure dismissals per ball faced; overall rating A
  bowlers: pressure runs conceded per legal ball; pressure wickets per ball bowled; overall rating A
  Rate shrinkage: x_j = (S_j + k mu) / (n_j + k), k = sigma2 / tau2; sigma2 = pooled within-player per-ball variance,
  tau2 = between-player variance by moments (players with n >= 30). Rating A = Phase 4 posterior mean of theta
  (variance components re-estimated on the training matches; its shrinkage is built in, k = sH2 / tau_sum2).
Candidates: (i) A; (ii) the shrunk pressure scoring feature (batters: runs per ball; bowlers: minus runs conceded per
  legal ball); (iii) ridge on all three standardised features, lambda by 5-fold player-grouped CV on inner odd/even
  pairs built INSIDE the training set.
Splits: odd->even and even->odd (each player's matches numbered chronologically), train <= 2023 / 2024 / 2025 ->
  next season. Evaluation players: LH_train >= 100 and LH_test >= 50. Spearman, player bootstrap (1000).
Rule (pre-registered by the user): highest mean Spearman over the five splits; if the top two differ by < 0.02,
  the simpler (order ii, i, iii).
Final: chosen candidate on all 2022-26 data, mapped to target units (win-probability points per average ball) by the
  pooled out-of-sample weighted regression of target on score; uncertainty by resampling each player's matches
  (500 draws, features and shrinkage refitted each draw; the mapping is held fixed).
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_outcome import MODEL, SEED
from phase4_ratings import load, Z80, REP4
from phase4b_ratings import fast_stats, estimate, clip_vc, post_arrays

N_BOOT, N_DRAWS, MIN_N_K = 1000, 500, 30
SIM_REPS, SIM_DRAWS = 200, 100
STRONG, MODERATE, MIN_M = 400, 150, 15
LAMBDAS = np.logspace(-2, 3, 11)
SIMPLICITY = ["ii", "i", "iii"]
UNIT = ("# Ratings are in win-probability points per average ball: +0.10 means the player's side is projected to gain "
        "on average 0.10 percentage points of win probability per pressure ball of average importance, compared with a "
        "typical player in the same season, competition, innings and phase.")


# ------------------------------------------------------------------ data
def build_pm(side):
    s = load()[side]
    f = pd.read_parquet(os.path.join(MODEL, "phase1f_ball_values.parquet"),
                        columns=["match_id", "innings", "innings_ball_num", "dismissed_id", "bowler_wicket", "missing_reason"])
    f = f[f.missing_reason.isna()].drop(columns="missing_reason")
    s = s.merge(f, on=["match_id", "innings", "innings_ball_num"], how="left")
    hi = s.hi.values
    if side == "bat":
        y = s.bat_runs.values.astype(float); ev = (s.dismissed_id == s.striker_id).values.astype(float)
        nden = np.ones(len(s))
    else:
        y = s.bowler_runs_conceded.values.astype(float); ev = s.bowler_wicket.astype(bool).values.astype(float)
        nden = s.is_legal.astype(bool).values.astype(float)
    s = s.assign(n_p=hi.astype(float), den_p=hi * nden, y_p=hi * y, y2_p=hi * y ** 2, e_p=hi * ev,
                 H=np.where(hi, s.v, 0.0), LH=np.where(hi, s.li, 0.0), L=np.where(~hi, s.v, 0.0), LL=np.where(~hi, s.li, 0.0))
    pm = s.groupby(["player", "match_id"]).agg(season=("season", "first"), date=("match_date", "first"), n_p=("n_p", "sum"),
                                               den_p=("den_p", "sum"), y_p=("y_p", "sum"), y2_p=("y2_p", "sum"), e_p=("e_p", "sum"),
                                               H=("H", "sum"), LH=("LH", "sum"), L=("L", "sum"), LL=("LL", "sum")).reset_index()
    pm = pm.sort_values(["player", "date", "match_id"]).reset_index(drop=True)
    pm["idx"] = pm.groupby("player").cumcount()
    return s, pm


# ------------------------------------------------------------------ features
def shrink_rate(S, S2, n, den=None):
    """EB shrinkage of per-player rates S/den towards the pooled mean; returns shrunk rates, k, mu, sigma2, tau2.
    den = denominator (balls); S2 = sum of squares of the per-ball numerator (for the within-player variance)."""
    den = n if den is None else den
    N = den.sum(); mu = S.sum() / N
    ok = den >= 2
    x = np.where(den > 0, S / np.where(den > 0, den, 1), mu)
    sigma2 = (S2[ok] - S[ok] ** 2 / den[ok]).sum() / (den[ok] - 1).sum()
    e = den >= MIN_N_K
    Ne = den[e].sum()
    tau2 = ((den[e] * (x[e] - mu) ** 2).sum() - (e.sum() - 1) * sigma2) / (Ne - (den[e] ** 2).sum() / Ne)
    k = sigma2 / tau2 if tau2 > 0 else np.inf
    shr = (S + (k if np.isfinite(k) else 0) * mu) / (den + (k if np.isfinite(k) else 0)) if np.isfinite(k) else np.full(len(S), mu)
    return shr, k, mu, sigma2, tau2


def player_sums(pm, w=None):
    w = np.ones(len(pm)) if w is None else w
    cols = ["n_p", "den_p", "y_p", "y2_p", "e_p", "H", "LH", "L", "LL"]
    g = pd.DataFrame({c: pm[c].values * w for c in cols})
    g["player"] = pm.player.values; g["M"] = w
    return g.groupby("player").sum()


def rating_A(pm, w=None):
    """Phase 4 posterior mean of theta on the given player-matches (variance components re-estimated)."""
    w = np.ones(len(pm)) if w is None else w
    keep = w > 0
    rep = np.repeat(np.arange(len(pm))[keep], w[keep].astype(int))
    sub = pm.iloc[rep]
    players = np.unique(sub.player.values)
    codes = np.searchsorted(players, sub.player.values)
    M = np.bincount(codes, minlength=len(players))
    g = fast_stats(codes, len(players), sub.H.values, sub.LH.values, sub.L.values, sub.LL.values, M)
    vc = estimate(g)
    r = post_arrays(g.LH.values, g.LL.values, g.y1.values, g.y2.values, clip_vc(vc))
    return pd.Series(r["th"], index=players), vc


def features(pm, side, w=None):
    g = player_sums(pm, w)
    f = pd.DataFrame(index=g.index)
    den = g.den_p.values
    f["score_rate"], k1, mu1, s1, t1 = shrink_rate(g.y_p.values, g.y2_p.values, g.n_p.values, den)
    f["event_rate"], k2, mu2, s2, t2 = shrink_rate(g.e_p.values, g.e_p.values, g.n_p.values, g.n_p.values)
    A, vc = rating_A(pm, w)
    f["A"] = A.reindex(f.index).values
    f["LH"], f["n_p"], f["M"] = g.LH, g.n_p, g.M
    f["raw_H_per_LH"] = np.where(g.LH > 0, g.H / g.LH.where(g.LH > 0, 1), np.nan)
    f["raw_score_rate"] = np.where(den > 0, g.y_p / np.where(den > 0, den, 1), np.nan)
    f["raw_event_rate"] = np.where(g.n_p > 0, g.e_p / g.n_p.where(g.n_p > 0, 1), np.nan)
    # candidate (ii): higher is better
    f["cand_ii"] = f.score_rate if side == "bat" else -f.score_rate
    f["cand_i"] = f.A
    ks = {"score_rate": {"k": k1, "mu": mu1, "sigma2": s1, "tau2": t1}, "event_rate": {"k": k2, "mu": mu2, "sigma2": s2, "tau2": t2},
          "A": {"k_equiv": vc["sH2"] / max(vc["tau_sum2"], 1e-12), **vc}}
    return f, ks


def target(pm):
    g = pm.groupby("player").agg(H=("H", "sum"), LH=("LH", "sum"))
    return pd.DataFrame({"y": g.H / g.LH.where(g.LH > 0), "LH": g.LH})


# ------------------------------------------------------------------ ridge
FEATS = ["score_rate", "event_rate", "A"]


def ridge_fit(pm_tr, side, seed):
    """Inner odd/even pairs inside the training set; lambda by 5-fold player-grouped CV; returns model + scaler."""
    pm_tr = pm_tr.copy(); pm_tr["iidx"] = pm_tr.groupby("player").cumcount()
    X, Y, W, P = [], [], [], []
    for a, b in ((0, 1), (1, 0)):
        fa, _ = features(pm_tr[pm_tr.iidx % 2 == a], side)
        tb = target(pm_tr[pm_tr.iidx % 2 == b])
        j = fa[["score_rate", "event_rate", "A", "LH"]].rename(columns={"LH": "LH_feat"}).join(tb, how="inner")
        j = j[(j.LH_feat >= 50) & (j.LH >= 25)].dropna()
        X.append(j[FEATS].values); Y.append(j.y.values); W.append(j.LH.values); P.append(j.index.values)
    X, Y, W, P = np.vstack(X), np.concatenate(Y), np.concatenate(W), np.concatenate(P)
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    up = np.unique(P); rng = np.random.default_rng(seed); fold_of = dict(zip(up, rng.integers(0, 5, len(up))))
    fold = np.array([fold_of[p] for p in P])
    cv = {}
    for lam in LAMBDAS:
        err = 0.0
        for k in range(5):
            tr, te = fold != k, fold == k
            m = Ridge(alpha=lam).fit(Z[tr], Y[tr], sample_weight=W[tr])
            err += (W[te] * (Y[te] - m.predict(Z[te])) ** 2).sum()
        cv[lam] = err / W.sum()
    lam = min(cv, key=cv.get)
    m = Ridge(alpha=lam).fit(Z, Y, sample_weight=W)
    return m, mu, sd, lam, len(Y)


# ------------------------------------------------------------------ evaluation
def splits(pm):
    out = [("odd->even", pm.idx % 2 == 1, pm.idx % 2 == 0), ("even->odd", pm.idx % 2 == 0, pm.idx % 2 == 1)]
    for t in (2023, 2024, 2025):
        out.append((f"<={t} -> {t + 1}", pm.season <= t, pm.season == t + 1))
    return out


def evaluate(pm, side):
    rows, coefs, pooled = [], [], []
    rng_all = []
    for si, (name, trm, tem) in enumerate(splits(pm)):
        tr, te = pm[trm.values], pm[tem.values]
        f, ks = features(tr, side)
        m, mu, sd, lam, npairs = ridge_fit(tr, side, SEED + si)
        f["cand_iii"] = m.predict((f[FEATS].values - mu) / sd)
        coefs.append({"split": name, "lambda": lam, "n_pairs": npairs,
                      **{f"coef_{c} (per SD)": b for c, b in zip(FEATS, m.coef_)}, "intercept": m.intercept_,
                      **{f"k_{c}": ks[c].get("k", ks[c].get("k_equiv")) for c in ks}})
        t = target(te)
        j = f.join(t.rename(columns={"LH": "LH_test"}), how="inner")
        j = j[(j.LH >= 100) & (j.LH_test >= 50)].dropna(subset=["y"])
        pooled.append(j.assign(split=name))
        rng_all.append((name, j))
    # bootstrap (same player-resample index per split across candidates)
    cands = ["cand_i", "cand_ii", "cand_iii"]
    rng = np.random.default_rng(SEED)
    rep_rho = {c: np.zeros((N_BOOT, len(rng_all))) for c in cands}
    for s_i, (name, j) in enumerate(rng_all):
        n = len(j); y = j.y.values
        for c in cands:
            rho = spearmanr(j[c], y).statistic
            rows.append({"split": name, "candidate": c, "n_players": n, "spearman": rho})
        idx = rng.integers(0, n, (N_BOOT, n))
        ry = np.argsort(np.argsort(y[idx], axis=1), axis=1)
        for c in cands:
            x = j[c].values[idx]
            rx = np.argsort(np.argsort(x, axis=1), axis=1)
            rep_rho[c][:, s_i] = np.array([np.corrcoef(rx[b], ry[b])[0, 1] for b in range(N_BOOT)])
    res = pd.DataFrame(rows)
    for c in cands:
        for s_i, (name, _) in enumerate(rng_all):
            lo, hi = np.percentile(rep_rho[c][:, s_i], [2.5, 97.5])
            res.loc[(res.split == name) & (res.candidate == c), ["lo95", "hi95"]] = lo, hi
    means = []
    for c in cands:
        m = res[res.candidate == c].spearman.mean(); b = rep_rho[c].mean(1)
        means.append({"candidate": c, "mean_spearman_5_splits": m, "lo95": np.percentile(b, 2.5), "hi95": np.percentile(b, 97.5),
                      "positive_in_splits": int((res[res.candidate == c].spearman > 0).sum())})
    means = pd.DataFrame(means)
    diffs = []
    for a, b in (("cand_ii", "cand_i"), ("cand_iii", "cand_i"), ("cand_iii", "cand_ii")):
        d = rep_rho[a].mean(1) - rep_rho[b].mean(1)
        diffs.append({"difference": f"{a} - {b}", "mean": means.set_index("candidate").loc[a, "mean_spearman_5_splits"] -
                      means.set_index("candidate").loc[b, "mean_spearman_5_splits"], "lo95": np.percentile(d, 2.5), "hi95": np.percentile(d, 97.5)})
        for s_i, (name, _) in enumerate(rng_all):
            dd = rep_rho[a][:, s_i] - rep_rho[b][:, s_i]
            diffs.append({"difference": f"{a} - {b} [{name}]", "mean": res[(res.split == name) & (res.candidate == a)].spearman.iat[0]
                          - res[(res.split == name) & (res.candidate == b)].spearman.iat[0],
                          "lo95": np.percentile(dd, 2.5), "hi95": np.percentile(dd, 97.5)})
    return res, means, pd.DataFrame(diffs), pd.DataFrame(coefs), pd.concat(pooled)


def choose(means):
    m = means.assign(c=means.candidate.str.replace("cand_", "")).sort_values("mean_spearman_5_splits", ascending=False)
    top2 = m.head(2)
    if top2.mean_spearman_5_splits.iat[0] - top2.mean_spearman_5_splits.iat[1] < 0.02:
        pick = min(top2.c, key=SIMPLICITY.index)
        why = (f"top two ({top2.c.iat[0]} {top2.mean_spearman_5_splits.iat[0]:.3f}, {top2.c.iat[1]} "
               f"{top2.mean_spearman_5_splits.iat[1]:.3f}) differ by < 0.02 -> simpler: {pick}")
    else:
        pick = top2.c.iat[0]; why = f"highest mean ({pick} {top2.mean_spearman_5_splits.iat[0]:.3f}), margin >= 0.02"
    return pick, why


# ------------------------------------------------------------------ reliability
def odd_even_reliability(pm, side):
    fa, _ = features(pm[pm.idx % 2 == 0], side); fb, _ = features(pm[pm.idx % 2 == 1], side)
    ta, tb = target(pm[pm.idx % 2 == 0]), target(pm[pm.idx % 2 == 1])
    j = fa.join(fb, lsuffix="_a", rsuffix="_b", how="inner").join(ta.rename(columns={"y": "ya", "LH": "LHa"})).join(tb.rename(columns={"y": "yb", "LH": "LHb"}))
    j = j[(j.LHa >= 50) & (j.LHb >= 50)]
    sb = lambda r: 2 * r / (1 + r)
    out = {"n_players": len(j)}
    for c in ["score_rate", "event_rate", "A"]:
        out[c] = sb(pearsonr(j[c + "_a"], j[c + "_b"]).statistic)
    out["target_H_per_LH"] = sb(pearsonr(j.ya, j.yb).statistic)
    out["noise_ceiling_sqrt_target_rel"] = float(np.sqrt(max(out["target_H_per_LH"], 0)))
    return out


# ------------------------------------------------------------------ final rating + bootstrap
def score_of(f, pick, model=None):
    if pick == "i": return f.cand_i.values
    if pick == "ii": return f.cand_ii.values
    m, mu, sd = model
    return m.predict((f[FEATS].values - mu) / sd)


def resample_weights(pm, rng):
    """Resample each player's matches with replacement: counts per player-match row."""
    pl = pm.player.values
    starts = np.r_[0, np.flatnonzero(pl[1:] != pl[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(pm)])
    owner = np.repeat(np.arange(len(starts)), sizes)
    pick = starts[owner] + np.floor(rng.random(len(pm)) * sizes[owner]).astype(int)
    return np.bincount(pick, minlength=len(pm)).astype(float)


def final_rating(pm, side, pick, a, b):
    f, ks = features(pm, side)
    model = None
    if pick == "iii":
        m, mu, sd, lam, _ = ridge_fit(pm, side, SEED + 99); model = (m, mu, sd)
    base = a + b * score_of(f, pick, model)
    rng = np.random.default_rng(SEED + 5)
    D = np.full((len(f), N_DRAWS), np.nan)
    for d in range(N_DRAWS):
        w = resample_weights(pm, rng)
        fd, _ = features(pm, side, w)
        D[:, d] = (a + b * score_of(fd, pick, model)) if len(fd) == len(f) else pd.Series(a + b * score_of(fd, pick, model), index=fd.index).reindex(f.index).values
    f["projected"] = base
    f["proj_mean_boot"] = np.nanmean(D, 1)
    f["cautious"] = np.nanquantile(D, 0.1, axis=1); f["hi80"] = np.nanquantile(D, 0.9, axis=1)
    f["p_above_avg"] = (D > 0).mean(1)
    return f, ks, model


# ------------------------------------------------------------------ self-test (method test only)
def self_test(pm, side, scen_mult, seed):
    """Plant true pressure scoring rates on the real schedule; simulate per-match sums; run feature + shrinkage +
    match-resampling; metrics over eligible players. scen_mult inflates match-level noise (within-match correlation)."""
    g = player_sums(pm)
    _, k, mu, sigma2, tau2 = shrink_rate(g.y_p.values, g.y2_p.values, g.n_p.values, g.den_p.values)
    players = g.index.values; codes = np.searchsorted(players, pm.player.values)
    den = pm.den_p.values
    elig_mask = ((g.LH >= MODERATE) & (g.M >= MIN_M)).values
    e = np.where(elig_mask)[0]
    rng = np.random.default_rng(seed); rows = []
    sign = 1 if side == "bat" else -1
    for rep in range(SIM_REPS):
        r = rng.normal(mu, np.sqrt(tau2), len(players))
        S = den * r[codes] + np.sqrt(sigma2 * scen_mult * den) * rng.standard_normal(len(den))
        def est(w):
            Sj = np.bincount(codes, weights=w * S, minlength=len(players)); nj = np.bincount(codes, weights=w * den, minlength=len(players))
            S2 = np.bincount(codes, weights=w * (S ** 2 / np.where(den > 0, den, 1)), minlength=len(players))
            S2 = S2 + np.bincount(codes, weights=w * sigma2 * scen_mult * (den - 1).clip(0), minlength=len(players))
            return shrink_rate(Sj, S2, nj, nj)[0]
        point = est(np.ones(len(den)))
        sub = pm.assign(S=S)
        D = np.column_stack([est(resample_weights(sub, rng)) for _ in range(SIM_DRAWS)])
        lo, hi = np.quantile(D, 0.1, axis=1), np.quantile(D, 0.9, axis=1)
        caut = lo if sign > 0 else -hi          # cautious end in the "higher is better" direction

        tru, sc = sign * r[e], caut[e]
        order = e[np.argsort(-sc)]
        top10_true = set(e[np.argsort(-tru)[:10]])
        rows.append({"rep": rep, "spearman_all_eligible": spearmanr(tru, sc).statistic,
                     "spearman_est_top25": spearmanr(sign * r[order[:25]], caut[order[:25]]).statistic,
                     "true_top10_in_est_top10": len(top10_true & set(order[:10])) / 10,
                     "coverage_80": float(((lo[e] <= r[e]) & (r[e] <= hi[e])).mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    t0 = time.perf_counter()
    out = {}
    for side in ("bat", "bowl"):
        s, pm = build_pm(side)
        res, means, diffs, coefs, pooled = evaluate(pm, side)
        pick, why = choose(means)
        rel = odd_even_reliability(pm, side)
        col = f"cand_{pick}"
        wls = np.polyfit(pooled[col].values, pooled.y.values, 1, w=np.sqrt(pooled.LH_test.values))
        b, a = float(wls[0]), float(wls[1])
        f, ks, model = final_rating(pm, side, pick, a, b)
        # extra descriptive columns
        prof = s[s.hi].groupby("player")
        lh = prof.li.sum()
        f["name"] = s.groupby("player").name.agg(lambda x: x.mode().iat[0])
        for c, vals in (("tier", ["IPL", "full_member", "associate"]), ("phase", ["Powerplay", "Middle", "Death"]), ("innings", [1, 2])):
            t = s[s.hi].groupby(["player", c]).li.sum().unstack(c).reindex(columns=vals).fillna(0)
            for v in vals:
                f[f"share_LH_{v}"] = (t[v] / lh).reindex(f.index)
        hi_ = s[s.hi]
        if side == "bat":
            f["raw_pressure_strike_rate"] = (100 * hi_.groupby("player").bat_runs.sum() / hi_.groupby("player").size()).reindex(f.index)
        else:
            lg = hi_.groupby("player").is_legal.sum()
            f["raw_pressure_economy"] = (6 * hi_.groupby("player").bowler_runs_conceded.sum() / lg.where(lg > 0)).reindex(f.index)
        sH2 = ks["A"]["sH2"]
        f["raw_H_lo80"] = f.raw_H_per_LH - Z80 * np.sqrt(sH2 / f.LH.where(f.LH > 0))
        f["raw_H_hi80"] = f.raw_H_per_LH + Z80 * np.sqrt(sH2 / f.LH.where(f.LH > 0))
        f["evidence_level"] = np.where((f.LH >= STRONG) & (f.M >= MIN_M), "Strong", np.where((f.LH >= MODERATE) & (f.M >= MIN_M), "Moderate", "Limited"))
        f = f.sort_values(["cautious", "projected"], ascending=False)
        f["rank_all"] = np.arange(1, len(f) + 1)
        el = f[f.evidence_level != "Limited"]
        f["rank_eligible"] = pd.Series(np.arange(1, len(el) + 1), index=el.index)
        # self-test of the chosen method (only implemented for the rate feature; reported for (ii))
        st = pd.concat([self_test(pm, side, m_, SEED + 31 + i).assign(scenario=lab) for i, (lab, m_) in
                        enumerate((("per-ball noise as fitted", 1.0), ("match-level noise x2 (within-match correlation)", 2.0)))])
        st["side"] = side
        out[side] = dict(res=res, means=means, diffs=diffs, coefs=coefs, pick=pick, why=why, rel=rel, a=a, b=b, f=f, ks=ks, st=st,
                         n_pm=len(pm))
        write_files(f, side, pick)
        print(side, means.round(3).to_dict("records"), pick, why, flush=True)
    st_all = pd.concat([o["st"] for o in out.values()])
    st_all.to_csv(os.path.join(REP4, "phase4b_selftest_reps.csv"), index=False)
    for side in out:
        forest(out[side]["f"], side)
    write_report(out, st_all, time.perf_counter() - t0)
    write_howto(out)


def write_files(f, side, pick):
    p = lambda x: (100 * x).round(3)
    o = f
    first, second = ("% batting first", "% chasing") if side == "bat" else ("% bowling in 1st innings", "% bowling in chase")
    feat = {"bat": [("Pressure runs per ball (shrunk)", o.score_rate.round(3)), ("Pressure dismissals per ball (shrunk)", o.event_rate.round(4))],
            "bowl": [("Pressure runs conceded per legal ball (shrunk)", o.score_rate.round(3)), ("Pressure wickets per ball (shrunk)", o.event_rate.round(4))]}[side]
    plain = pd.DataFrame({
        "Rank (eligible)": o.rank_eligible.astype("Int64"), "Rank (all players)": o.rank_all, "Player": o.name, "Evidence level": o.evidence_level,
        "Cautious rating (used for rank)": p(o.cautious), "Projected pressure impact (best estimate)": p(o.projected),
        "Likely range low (80%)": p(o.cautious), "Likely range high (80%)": p(o.hi80),
        "Overall quality rating A (normal and pressure balls)": p(o.A),
        "Pressure record when it mattered": p(o.raw_H_per_LH), "Pressure record low (80%)": p(o.raw_H_lo80), "Pressure record high (80%)": p(o.raw_H_hi80),
        "Chance he is above average under pressure (%)": (100 * o.p_above_avg).round(1),
        **{k: v for k, v in feat},
        "Pressure exposure (sum of leverage)": o.LH.round(1), "Pressure balls": o.n_p.astype(int), "Matches": o.M.astype(int),
        "% of pressure from IPL": (100 * o.share_LH_IPL).round(1),
        "% of pressure from internationals": (100 * (o.share_LH_full_member + o.share_LH_associate)).round(1),
        "% pressure in powerplay": (100 * o.share_LH_Powerplay).round(1), "% pressure in middle overs": (100 * o.share_LH_Middle).round(1),
        "% pressure in death overs": (100 * o.share_LH_Death).round(1), first: (100 * o.share_LH_1).round(1), second: (100 * o.share_LH_2).round(1),
        ("Raw pressure strike rate" if side == "bat" else "Raw pressure economy"):
            (o.raw_pressure_strike_rate if side == "bat" else o.raw_pressure_economy).round(2)})
    tech = o.copy(); tech.index.name = "player_id"
    for path, frame, idx in ((os.path.join(MODEL, f"phase4_rankings_{side}_all_players.csv"), plain, False),
                             (os.path.join(MODEL, f"phase4_rankings_{side}_all_players_technical.csv"), tech, True)):
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(UNIT + (" Technical file: raw win-probability units (x100 for points)." if idx else "") + "\n")
            frame.to_csv(fh, index=idx)


def forest(f, side):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = f[f.rank_eligible <= 25].sort_values("rank_eligible").iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 9.2)); y = np.arange(len(t))
    ax.axvline(0, color="#bbbbbb", lw=1, zorder=0)
    ax.hlines(y, 100 * t.cautious, 100 * t.hi80, color="#3987e5", lw=2, zorder=1)
    ax.scatter(100 * t.projected, y, s=40, color="#1c5cab", zorder=2, edgecolor="white", linewidth=1.5, label="projected (best estimate)")
    ax.scatter(100 * t.cautious, y, s=40, marker="|", color="#0d366b", zorder=3, linewidth=2, label="cautious rating (ranking score)")
    ax.set_yticks(y); ax.set_yticklabels([f"{int(r)}. {n}  ({ev[0]})" for r, n, ev in zip(t.rank_eligible, t.name, t.evidence_level)], fontsize=8.5)
    ax.set_xlabel("projected pressure impact, win-probability points per average ball (80% likely range)")
    ax.set_title(f"Top 25 eligible {'batters' if side == 'bat' else 'bowlers'} - projected pressure impact, 2022-26\n(S = strong, M = moderate evidence)", fontsize=10.5)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", lw=0.8); ax.set_axisbelow(True); ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(REP4, f"phase4_forest_top25_{side}.png"), dpi=140); plt.close(fig)


def separability(f):
    el = f[f.rank_eligible.notna()].sort_values("rank_eligible")
    top = el.head(10)
    rows = []
    for _, r in top.iterrows():
        below = el[el.rank_eligible > r.rank_eligible]
        sep = below[below.hi80 < r.cautious]
        first_sep = int(sep.rank_eligible.min()) if len(sep) else None
        rows.append({"rank": int(r.rank_eligible), "player": r["name"], "80% range": f"{100*r.cautious:+.3f} to {100*r.hi80:+.3f}",
                     "eligible players clearly below (range entirely lower)": len(sep), "first clearly-below rank": first_sep,
                     "separable from next rank": bool(len(below) and below.iloc[0].hi80 < r.cautious)})
    return pd.DataFrame(rows)


def write_report(out, st_all, rt):
    f3 = lambda x: f"{x:.3f}" if isinstance(x, (float, np.floating)) else x
    names = {"cand_i": "(i) rating A", "cand_ii": "(ii) shrunk pressure scoring rate", "cand_iii": "(iii) ridge on all features"}
    L = ["# Phase 4b - projection model for pressure-ball value", "",
         "Target: each player's held-out pressure-ball value H/LH (win probability added per unit of leverage on balls with "
         "LI >= 1, corrected values v_adj). Evaluation players: LH_train >= 100 and LH_test >= 50. 95% CIs from 1000 player "
         "bootstrap resamples. Rule (pre-registered): highest mean Spearman over the five splits; if the top two differ by "
         "< 0.02, the simpler (ii, then i, then iii).", ""]
    for side, R in out.items():
        role = "Batters" if side == "bat" else "Bowlers"
        rel, ks = R["rel"], R["ks"]
        res = R["res"].assign(candidate=R["res"].candidate.map(names))
        piv = res.pivot(index="split", columns="candidate", values="spearman")
        cis = res.assign(ci=[f"{r.spearman:+.3f} [{r.lo95:+.3f}, {r.hi95:+.3f}]" for r in res.itertuples()]).pivot(index="split", columns="candidate", values="ci")
        nplayers = res.groupby("split").n_players.first()
        cis.insert(0, "players", nplayers)
        means = R["means"].assign(candidate=R["means"].candidate.map(names))
        f = R["f"]; el = f[f.rank_eligible.notna()]
        top = el.sort_values("rank_eligible").head(25)
        avg_rec = 0.0
        t = pd.DataFrame({"rank (eligible)": top.rank_eligible.astype(int), "rank (all)": top.rank_all, "player": top.name, "evidence": top.evidence_level,
                          "cautious": (100 * top.cautious).map(lambda x: f"{x:+.3f}"), "projected": (100 * top.projected).map(lambda x: f"{x:+.3f}"),
                          "80% range": [f"{100*a:+.3f} to {100*b:+.3f}" for a, b in zip(top.cautious, top.hi80)],
                          "pressure record": (100 * top.raw_H_per_LH).map(lambda x: f"{x:+.3f}"),
                          ("pressure SR" if side == "bat" else "pressure econ"): (top.raw_pressure_strike_rate if side == "bat" else top.raw_pressure_economy).round(2),
                          "P(above avg) %": (100 * top.p_above_avg).round(1), "sum LH": top.LH.round(0).astype(int), "matches": top.M.astype(int)})
        below = el[(el.rank_eligible <= 10) & (el.raw_H_per_LH < 0)]
        sep = separability(f)
        lev = f.evidence_level.value_counts().reindex(["Strong", "Moderate", "Limited"]).fillna(0).astype(int)
        kt = pd.DataFrame([{"feature": "pressure scoring rate", "k (balls)": ks["score_rate"]["k"], "mean": ks["score_rate"]["mu"],
                            "between-player SD": np.sqrt(max(ks["score_rate"]["tau2"], 0)), "odd/even reliability": rel["score_rate"]},
                           {"feature": "pressure dismissal/wicket rate", "k (balls)": ks["event_rate"]["k"], "mean": ks["event_rate"]["mu"],
                            "between-player SD": np.sqrt(max(ks["event_rate"]["tau2"], 0)), "odd/even reliability": rel["event_rate"]},
                           {"feature": "rating A (theta posterior mean)", "k (balls)": ks["A"]["k_equiv"], "mean": 0.0,
                            "between-player SD": np.sqrt(max(ks["A"]["tau_theta2"], 0)), "odd/even reliability": rel["A"]}])
        L += [f"## {role}", "",
              f"Players ranked: {len(f)} (Strong {lev['Strong']}, Moderate {lev['Moderate']}, Limited {lev['Limited']}).", "",
              "### Features (all 2022-26 data; k = shrinkage constant in balls, sum LH units for A)", "",
              kt.map(lambda x: f"{x:.4g}" if isinstance(x, (float, np.floating)) else x).to_markdown(index=False), "",
              f"Target (held-out H/LH) odd/even reliability: {rel['target_H_per_LH']:.3f} ({rel['n_players']} players with LH >= 50 "
              f"in each half) -> noise ceiling for any score's correlation with it about {rel['noise_ceiling_sqrt_target_rel']:.2f}.", "",
              "### Out-of-sample Spearman by split [95% CI]", "", cis.reset_index().to_markdown(index=False), "",
              means.map(f3).to_markdown(index=False), "",
              "Paired differences (bootstrap CI):", "", R["diffs"].map(f3).to_markdown(index=False), "",
              "Ridge coefficients per split (per SD of each feature; target units):", "",
              R["coefs"].map(lambda x: f"{x:.4g}" if isinstance(x, (float, np.floating)) else x).to_markdown(index=False), "",
              f"**Chosen: ({R['pick']})** - {R['why']}.", "",
              f"Mapping to target units (pooled out-of-sample weighted fit): projected = {R['a']:.5f} + {R['b']:.5f} x score.", "",
              "### Top 25 eligible (projected pressure impact, win-probability points per average ball)", "",
              t.to_markdown(index=False), "", f"![forest](phase4_forest_top25_{side}.png)", "",
              "### Which top-10 ranks are statistically separable (80% ranges)", "", sep.to_markdown(index=False), "",
              f"### Top-10 players whose raw pressure record is below average (< 0): {len(below)}", "",
              (pd.DataFrame({"rank": below.rank_eligible.astype(int), "player": below.name,
                             "pressure record": (100 * below.raw_H_per_LH).round(3), "sum LH": below.LH.round(0)}).to_markdown(index=False)
               if len(below) else ""), ""]
    sts = st_all.groupby(["side", "scenario"]).agg(spearman_all_eligible=("spearman_all_eligible", "mean"),
                                                 spearman_est_top25=("spearman_est_top25", "mean"),
                                                 true_top10_in_est_top10=("true_top10_in_est_top10", "mean"),
                                                 coverage_80=("coverage_80", "mean"),
                                                 coverage_80_p5=("coverage_80", lambda x: x.quantile(.05)),
                                                 coverage_80_p95=("coverage_80", lambda x: x.quantile(.95))).reset_index()
    sts.to_csv(os.path.join(REP4, "phase4b_selftest_summary.csv"), index=False)
    L += ["## Method self-test (simulation - no simulated number enters any ranking)", "",
          f"Real player-match schedule (pressure balls per match); true pressure scoring rates planted from the fitted "
          f"between-player spread; per-match sums generated with the fitted per-ball noise (scenario 2: match-level noise "
          f"doubled, to mimic within-match correlation); shrinkage refitted and each player's matches resampled "
          f"({SIM_DRAWS} draws) exactly as in the final rating; {SIM_REPS} repetitions. Truth = planted rate (sign so that "
          "higher is better); metrics over eligible players.", "",
          sts.map(f3).to_markdown(index=False), "",
          f"Runtime {rt:.0f} s.", ""]
    open(os.path.join(REP4, "phase4b_summary.md"), "w", encoding="utf-8").write("\n".join(L))


def write_howto(out):
    fb = out["bat"]["f"]; ex = fb[fb.rank_eligible == 1].iloc[0]
    pick = out["bat"]["pick"]; rb = out["bat"]["rel"]
    m_b = out["bat"]["means"].set_index("candidate").mean_spearman_5_splits
    fw = out["bowl"]["f"]; exw = fw[fw.rank_eligible == 1].iloc[0]
    pp = lambda x: f"{100 * x:+.2f}"
    feat_txt = {"ii": "the runs he scored per ball on pressure balls", "i": "his overall quality rating", "iii": "a weighted mix of his pressure scoring, dismissals and overall quality"}[pick]
    L = ["# How the ranking works", "",
         "1. **Give every ball an importance score.** A win-probability model built on ball-by-ball data says how much the "
         "match result could swing on that ball. An average ball scores 1; the last ball of a close chase can score 30. "
         "Balls scoring 1 or more are the **pressure balls**.",
         "2. **Measure what each pressure ball did for the player's side.** We record the change in his team's chance of "
         "winning, compared with a typical ball at the same stage of the same season and competition, corrected for the "
         "strength of the opponent and for matches against weaker associate teams. Per player, the total divided by the "
         "total importance is his **pressure record**.",
         "3. **Test what best predicts a player's future pressure record.** We hid part of the data five different ways "
         "(alternate matches, and each later season), and checked which measure from the visible part best predicted the "
         "hidden pressure record. The candidates were his overall quality rating, his pressure scoring rate, and a "
         f"combination. The winner by the rule we fixed in advance: **{feat_txt}**.",
         "4. **Turn it into a projection, with caution.** Short records are pulled towards average (more strongly the "
         "fewer balls). The measure is converted to the same scale as the pressure record, and we repeat the calculation "
         f"{N_DRAWS} times on reshuffled versions of each player's matches to see how much it could move.",
         "5. **Rank by the cautious rating** - the level the projection stays above in 90% of the reshuffles. Only players "
         f"with enough pressure exposure (total importance of pressure balls at least {MODERATE} and at least {MIN_M} matches) "
         "can enter the top 10.", "",
         f"## Worked example: {ex['name']} (batting rank 1)", "",
         f"- {int(ex.n_p)} pressure balls in {int(ex.M)} matches, total importance {ex.LH:.0f}.",
         f"- Pressure scoring: {ex.raw_score_rate:.2f} runs per ball raw ({ex.raw_pressure_strike_rate:.1f} strike rate), "
         f"{ex.score_rate:.2f} after pulling towards the average of {out['bat']['ks']['score_rate']['mu']:.2f}.",
         f"- Pressure record: {pp(ex.raw_H_per_LH)} win-probability points per average ball (range from those balls alone "
         f"{pp(ex.raw_H_lo80)} to {pp(ex.raw_H_hi80)}).",
         f"- Projected pressure impact: **{pp(ex.projected)}**, likely between {pp(ex.cautious)} and {pp(ex.hi80)}; cautious "
         f"rating **{pp(ex.cautious)}**; above an average player in {100 * ex.p_above_avg:.0f}% of reshuffles.",
         f"- Bowling rank 1: {exw['name']} (cautious rating {pp(exw.cautious)}, projected {pp(exw.projected)}, "
         f"pressure economy {exw.raw_pressure_economy:.2f}).", "",
         "> **What we found**",
         "> - Pressure-specific skill is not detectable: no player does measurably better under pressure than his general "
         "level predicts. Underlying quality carries into pressure.",
         f"> - Future pressure records are hard to predict: the best out-of-sample correlation for batters averaged "
         f"{m_b.max():.2f}, against a ceiling of about {rb['noise_ceiling_sqrt_target_rel']:.2f} set by how noisy the record itself is.",
         "> - No candidate was significantly better than another in any single test; the choice follows a rule fixed in advance.",
         "> - Neighbouring ranks often overlap: read the ranges, not just the order.", ""]
    open(os.path.join(REP4, "how_the_ranking_works.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
