"""Phase 4c - context-adjusted projection features, fixed small-sample uncertainty, same pre-registered rule.

Per pressure ball (LI >= 1), from the engine's LOSO outcome probabilities p_k at the state before the ball (season,
tier, innings, b, w, score / runs required; Phase 1b/1c tables):
  batters (balls faced): runs_above_expected = bat_runs - sum_{k in 0,1,2,3,4,6} p_k k
                         dismissal_above_expected = 1[striker dismissed] - p_W
  bowlers (balls bowled): conceded_below_expected = E[conceded] - bowler_runs_conceded,
                         E[conceded] = sum_k p_k k + p_WD mean(WD runs | phase) + p_NB mean(NB runs | phase)
                         wickets_above_expected = 1[bowler wicket] - p_W x P(wicket credited to bowler | phase)
                         (credit share from the pre-2022 history rows, so it is leak-free for every split)
Then, per feature: centred by season x tier x innings x phase (removes the engine's residual scoring drift, as for
wpa_centred); opponent correction f - beta_t q_opp (q_opp = the opponent quality used for v_adj; beta_t by OLS on
seasons < t, beta_2022 = 0); associate-tier correction f - gamma_t 1[associate] (gamma_t = within-player associate
gain on seasons < t, gamma_2022 = 0). Each feature is shrunk per player by empirical Bayes (method of moments).
Candidates: (i) rating A; (i*) uncertainty-averaged rating A (mixture mean over bootstrap draws of the variance
components); (ii') shrunk runs-above-expected (batters) / conceded-below-expected (bowlers); (iii') ridge on the
two corrected features plus A (lambda grid extended until the optimum is interior).
Rule (pre-registered): if no candidate beats (i*) by >= 0.02 in mean Spearman, the final score is (i*); otherwise the
highest mean, and if the top two differ by < 0.02 the simpler (ii' < i < i* < iii').
Uncertainty of the chosen score: match-bootstrap variance + empirical-Bayes posterior variance, through the mapping.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, norm
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_outcome import MODEL, SEED
from phase1b_wp import PHASE_B, BMAX, WMAX, RMAX
from phase1f_ball_values import load_table, SI, TI
from phase4_ratings import Z80, REP4
from phase4b_ratings import fast_stats, estimate, clip_vc, post_arrays, boot_vcs
from phase4c_projection import shrink_rate, rating_A, resample_weights, target

N_BOOT, N_DRAWS, ISTAR_DRAWS = 1000, 500, 200
SIM_REPS, SIM_DRAWS = 200, 100
STRONG, MODERATE, MIN_M = 400, 150, 15
SMAX = RMAX - 1
ORDER = ["ii", "i", "istar", "iii"]
NAMES = {"i": "(i) rating A", "istar": "(i*) uncertainty-averaged A", "ii": "(ii') shrunk context-adjusted scoring",
         "iii": "(iii') ridge: corrected features + A"}


# ------------------------------------------------------------------ per-ball features
def build():
    a = pd.read_parquet(os.path.join(MODEL, "phase3_ball_values_adj.parquet"))
    f = pd.read_parquet(os.path.join(MODEL, "phase1f_ball_values.parquet"),
                        columns=["match_id", "innings", "innings_ball_num", "balls_remaining", "wkts_in_hand", "score_before",
                                 "runs_required", "bat_runs", "bowler_runs_conceded", "is_legal", "dismissed_id", "bowler_wicket",
                                 "striker_name", "bowler_name", "missing_reason"])
    f = f[f.missing_reason.isna()].drop(columns="missing_reason")
    d = a.merge(f, on=["match_id", "innings", "innings_ball_num"], how="left", validate="1:1")
    VC, PC, classes = load_table(os.path.join(MODEL, "phase1b_wp_tables.parquet"), "r", "wp", RMAX + 1)
    _, P1, _ = load_table(os.path.join(MODEL, "phase1c_wp1_tables.parquet"), "s", "wp1", SMAX + 1)
    del VC
    si, ti = d.season.map(SI).values, d.tier.map(TI).values
    bi = d.balls_remaining.clip(0, BMAX).astype(int).values; wi = d.wkts_in_hand.clip(0, WMAX).astype(int).values
    s1 = np.round(d.score_before).clip(0, SMAX).astype(int).values; r2 = np.round(d.runs_required.fillna(0)).clip(0, RMAX).astype(int).values
    inn1 = (d.innings == 1).values
    p = np.where(inn1[:, None], P1[si, ti, bi, wi, s1], PC[si, ti, bi, wi, r2]).astype(float)
    del P1, PC
    ci = {c: i for i, c in enumerate(classes)}
    e_legal = sum(p[:, ci[c]] * int(c) for c in classes if c.isdigit())
    inp = json.load(open(os.path.join(MODEL, "phase1b_recursion_inputs.json")))
    mwd, mnb = np.zeros(len(d)), np.zeros(len(d))
    ph = PHASE_B[bi.clip(1, BMAX)]
    for s, dd in inp["run_distributions"].items():
        m = (d.season.values == int(s))
        j = np.arange(len(dd["fWD"][0]))
        mwd[m] = (np.array(dd["fWD"]) @ j)[ph[m]]; mnb[m] = (np.array(dd["fNB"]) @ j)[ph[m]]
    d["E_bat"] = e_legal
    d["E_conc"] = e_legal + p[:, ci["WD"]] * mwd + p[:, ci["NB"]] * mnb
    d["pW"] = p[:, ci["W"]]
    h = pd.read_parquet(os.path.join(MODEL, "history_deliveries.parquet"), columns=["is_wicket", "bowler_wicket", "phase", "innings", "is_super_over"])
    h = h[h.is_wicket.astype(bool) & (h.innings <= 2) & ~h.is_super_over.astype(bool)]
    credit = h.groupby("phase").bowler_wicket.mean().to_dict()
    d["credit"] = d.phase.map(credit)
    return d, credit


def correct(s, fcols, cell=("season", "tier", "innings", "phase")):
    """Centre, then opponent and associate-tier corrections (leak-free, earlier seasons). Returns frame + params."""
    hi = s[s.hi].copy()
    par = {}
    for c in fcols:
        hi[c] = hi[c] - hi.groupby(list(cell))[c].transform("mean")
        hi[c + "_centred"] = hi[c]
        b_t = {2022: 0.0}
        for t in (2023, 2024, 2025, 2026):
            e = hi[hi.season < t]
            X = np.column_stack([np.ones(len(e)), e.q.values]); b_t[t] = float(np.linalg.lstsq(X, e[c].values, rcond=None)[0][1])
        hi[c] = hi[c] - hi.season.map(b_t).values * hi.q.values
        g_t = {2022: 0.0}
        for t in (2023, 2024, 2025, 2026):
            e = hi[hi.season < t].assign(a=lambda x: x.tier == "associate")
            g = e.groupby(["player", "a"])[c].agg(["mean", "size"]).unstack("a")
            g = g[(g[("size", True)] >= 30) & (g[("size", False)] >= 30)]
            if len(g):
                wt = 1 / (1 / g[("size", True)] + 1 / g[("size", False)])
                g_t[t] = float(np.average(g[("mean", True)] - g[("mean", False)], weights=wt))
            else:
                g_t[t] = 0.0
        hi[c] = hi[c] - hi.season.map(g_t).values * (hi.tier == "associate").values
        par[c] = {"beta_by_season": b_t, "gamma_by_season": g_t}
    return hi, par


def sides(d):
    out = {}
    for side in ("bat", "bowl"):
        if side == "bat":
            s = d[d.v_adj_bat.notna()].copy()
            s["player"], s["name"], s["v"], s["q"] = s.striker_id, s.striker_name, s.v_adj_bat, s.q_opp_bat
            s["f1"] = s.bat_runs - s.E_bat
            s["f2"] = (s.dismissed_id == s.striker_id).astype(float) - s.pW
        else:
            s = d.copy()
            s["player"], s["name"], s["v"], s["q"] = s.bowler_id, s.bowler_name, s.v_adj_bowl, s.q_opp_bowl
            s["f1"] = s.E_conc - s.bowler_runs_conceded
            s["f2"] = s.bowler_wicket.astype(bool).astype(float) - s.pW * s.credit
        s["q"] = s.q.fillna(0.0)
        s["hi"] = s.li >= 1
        hi, par = correct(s, ["f1", "f2"])
        s = s.drop(columns=["f1", "f2"]).join(hi[["f1", "f2", "f1_centred", "f2_centred"]])
        for c in ("f1", "f2", "f1_centred", "f2_centred"):
            s[c] = s[c].fillna(0.0)
        out[side] = (s, par)
    return out


def tier_season_table(s):
    h = s[s.hi]
    rows = []
    for lab, cols in (("raw (before centring)", None), ("after centring", ["f1_centred", "f2_centred"]), ("after centring + opponent + tier", ["f1", "f2"])):
        if cols is None:
            continue
        t = h.groupby(["tier", "season"])[cols].mean().reset_index()
        t["stage"] = lab; t.columns = ["tier", "season", "f1", "f2", "stage"]
        rows.append(t)
    return pd.concat(rows)


def player_match(s):
    h = s.assign(n_p=s.hi.astype(float), f1s=np.where(s.hi, s.f1, 0), f1q=np.where(s.hi, s.f1 ** 2, 0),
                 f2s=np.where(s.hi, s.f2, 0), f2q=np.where(s.hi, s.f2 ** 2, 0),
                 H=np.where(s.hi, s.v, 0.0), LH=np.where(s.hi, s.li, 0.0), L=np.where(~s.hi, s.v, 0.0), LL=np.where(~s.hi, s.li, 0.0))
    pm = h.groupby(["player", "match_id"]).agg(season=("season", "first"), date=("match_date", "first"), n_p=("n_p", "sum"),
                                               f1s=("f1s", "sum"), f1q=("f1q", "sum"), f2s=("f2s", "sum"), f2q=("f2q", "sum"),
                                               H=("H", "sum"), LH=("LH", "sum"), L=("L", "sum"), LL=("LL", "sum")).reset_index()
    pm = pm.sort_values(["player", "date", "match_id"]).reset_index(drop=True)
    pm["idx"] = pm.groupby("player").cumcount()
    return pm


# ------------------------------------------------------------------ features per player
def features(pm, w=None, istar_draws=0, seed=SEED):
    w = np.ones(len(pm)) if w is None else w
    cols = ["n_p", "f1s", "f1q", "f2s", "f2q", "H", "LH", "L", "LL"]
    g = pd.DataFrame({c: pm[c].values * w for c in cols}); g["player"] = pm.player.values; g["M"] = w
    g = g.groupby("player").sum()
    F = pd.DataFrame(index=g.index)
    info = {}
    for c, sc, sq in (("f1", "f1s", "f1q"), ("f2", "f2s", "f2q")):
        shr, k, mu, s2, t2 = shrink_rate(g[sc].values, g[sq].values, g.n_p.values, g.n_p.values)
        F[c] = shr
        F[c + "_postvar"] = 1 / (g.n_p.values / s2 + (1 / t2 if t2 > 0 else np.inf)) if t2 > 0 else 0.0
        info[c] = {"k": k, "mu": mu, "sigma2": s2, "tau2": t2}
    A, vc = rating_A(pm, w)
    F["A"] = A.reindex(F.index).values
    info["A"] = {"k_equiv": vc["sH2"] / max(vc["tau_sum2"], 1e-12), **vc}
    if istar_draws:
        keep = w > 0
        rep = np.repeat(np.arange(len(pm))[keep], w[keep].astype(int)); sub = pm.iloc[rep]
        players = np.unique(sub.player.values); codes = np.searchsorted(players, sub.player.values)
        gs = fast_stats(codes, len(players), sub.H.values, sub.LH.values, sub.L.values, sub.LL.values, np.bincount(codes, minlength=len(players)))
        vcs = boot_vcs(gs, istar_draws, seed)
        ms = np.array([post_arrays(gs.LH.values, gs.LL.values, gs.y1.values, gs.y2.values, clip_vc(v))["m"] for v in vcs])
        vs = np.array([post_arrays(gs.LH.values, gs.LL.values, gs.y1.values, gs.y2.values, clip_vc(v))["v"] for v in vcs])
        F["Istar"] = pd.Series(ms.mean(0), index=players).reindex(F.index).values
        F["Istar_var"] = pd.Series(vs.mean(0) + ms.var(0), index=players).reindex(F.index).values
        rA = post_arrays(gs.LH.values, gs.LL.values, gs.y1.values, gs.y2.values, clip_vc(vc))
        F["A_postvar"] = pd.Series(rA["vth"], index=players).reindex(F.index).values
    F["n_p"], F["LH"], F["M"] = g.n_p, g.LH, g.M
    F["raw_H_per_LH"] = np.where(g.LH > 0, g.H / g.LH.where(g.LH > 0, 1), np.nan)
    F["raw_f1"] = np.where(g.n_p > 0, g.f1s / g.n_p.where(g.n_p > 0, 1), np.nan)
    F["raw_f2"] = np.where(g.n_p > 0, g.f2s / g.n_p.where(g.n_p > 0, 1), np.nan)
    return F, info


FE = ["f1", "f2", "A"]


def ridge_fit(pm_tr, seed):
    pm_tr = pm_tr.copy(); pm_tr["iidx"] = pm_tr.groupby("player").cumcount()
    X, Y, W, P = [], [], [], []
    for a, b in ((0, 1), (1, 0)):
        fa, _ = features(pm_tr[pm_tr.iidx % 2 == a]); tb = target(pm_tr[pm_tr.iidx % 2 == b])
        j = fa[FE + ["LH"]].rename(columns={"LH": "LH_feat"}).join(tb, how="inner")
        j = j[(j.LH_feat >= 50) & (j.LH >= 25)].dropna()
        X.append(j[FE].values); Y.append(j.y.values); W.append(j.LH.values); P.append(j.index.values)
    X, Y, W, P = np.vstack(X), np.concatenate(Y), np.concatenate(W), np.concatenate(P)
    mu, sd = X.mean(0), X.std(0); Z = (X - mu) / sd
    up = np.unique(P); fold_of = dict(zip(up, np.random.default_rng(seed).integers(0, 5, len(up)))); fold = np.array([fold_of[p] for p in P])

    def cv(lam):
        err = 0.0
        for k in range(5):
            tr, te = fold != k, fold == k
            m = Ridge(alpha=lam).fit(Z[tr], Y[tr], sample_weight=W[tr]); err += (W[te] * (Y[te] - m.predict(Z[te])) ** 2).sum()
        return err / W.sum()
    grid = list(np.logspace(-2, 4, 13))
    res = {l: cv(l) for l in grid}
    for _ in range(12):                                  # extend until the optimum is interior
        best = min(res, key=res.get)
        if best == max(res):
            for l in np.logspace(np.log10(best) + 0.5, np.log10(best) + 2, 4): res[l] = cv(l)
        elif best == min(res):
            for l in np.logspace(np.log10(best) - 2, np.log10(best) - 0.5, 4): res[l] = cv(l)
        else:
            break
    lam = min(res, key=res.get); interior = (lam != max(res)) and (lam != min(res))
    m = Ridge(alpha=lam).fit(Z, Y, sample_weight=W)
    return m, mu, sd, lam, interior, len(Y), (min(res), max(res))


# ------------------------------------------------------------------ evaluation
def splits(pm):
    out = [("odd->even", pm.idx % 2 == 1, pm.idx % 2 == 0), ("even->odd", pm.idx % 2 == 0, pm.idx % 2 == 1)]
    for t in (2023, 2024, 2025):
        out.append((f"<={t} -> {t + 1}", pm.season <= t, pm.season == t + 1))
    return out


def evaluate(pm):
    rows, coefs, pooled, sets = [], [], [], []
    cands = ["i", "istar", "ii", "iii"]
    for si_, (name, trm, tem) in enumerate(splits(pm)):
        tr, te = pm[trm.values], pm[tem.values]
        F, info = features(tr, istar_draws=ISTAR_DRAWS, seed=SEED + si_)
        m, mu, sd, lam, interior, npairs, rng_ = ridge_fit(tr, SEED + si_)
        F["c_i"], F["c_istar"], F["c_ii"] = F.A, F.Istar, F.f1
        F["c_iii"] = m.predict((F[FE].values - mu) / sd)
        coefs.append({"split": name, "lambda": lam, "interior": interior, "grid_searched": f"{rng_[0]:.3g}-{rng_[1]:.3g}", "n_pairs": npairs,
                      **{f"coef_{c} (per SD)": b for c, b in zip(FE, m.coef_)}, "intercept": m.intercept_,
                      "k_f1": info["f1"]["k"], "k_f2": info["f2"]["k"], "k_A": info["A"]["k_equiv"]})
        j = F.join(target(te).rename(columns={"LH": "LH_test"}), how="inner")
        j = j[(j.LH >= 100) & (j.LH_test >= 50)].dropna(subset=["y"])
        pooled.append(j.assign(split=name)); sets.append((name, j))
    rng = np.random.default_rng(SEED)
    rep = {c: np.zeros((N_BOOT, len(sets))) for c in cands}
    for s_i, (name, j) in enumerate(sets):
        n = len(j); y = j.y.values
        idx = rng.integers(0, n, (N_BOOT, n)); ry = np.argsort(np.argsort(y[idx], axis=1), axis=1)
        for c in cands:
            rows.append({"split": name, "candidate": c, "n_players": n, "spearman": spearmanr(j[f"c_{c}"], y).statistic})
            rx = np.argsort(np.argsort(j[f"c_{c}"].values[idx], axis=1), axis=1)
            rep[c][:, s_i] = [np.corrcoef(rx[b], ry[b])[0, 1] for b in range(N_BOOT)]
    res = pd.DataFrame(rows)
    for c in cands:
        for s_i, (name, _) in enumerate(sets):
            res.loc[(res.split == name) & (res.candidate == c), ["lo95", "hi95"]] = np.percentile(rep[c][:, s_i], [2.5, 97.5])
    means = pd.DataFrame([{"candidate": c, "mean_spearman_5_splits": res[res.candidate == c].spearman.mean(),
                           "lo95": np.percentile(rep[c].mean(1), 2.5), "hi95": np.percentile(rep[c].mean(1), 97.5),
                           "positive_in_splits": int((res[res.candidate == c].spearman > 0).sum())} for c in cands])
    diffs = []
    mm = means.set_index("candidate").mean_spearman_5_splits
    for c in ("i", "ii", "iii"):
        d_ = rep[c].mean(1) - rep["istar"].mean(1)
        diffs.append({"difference": f"{c} - i*", "mean": mm[c] - mm["istar"], "lo95": np.percentile(d_, 2.5), "hi95": np.percentile(d_, 97.5)})
        for s_i, (name, _) in enumerate(sets):
            dd = rep[c][:, s_i] - rep["istar"][:, s_i]
            diffs.append({"difference": f"{c} - i* [{name}]",
                          "mean": res[(res.split == name) & (res.candidate == c)].spearman.iat[0] - res[(res.split == name) & (res.candidate == "istar")].spearman.iat[0],
                          "lo95": np.percentile(dd, 2.5), "hi95": np.percentile(dd, 97.5)})
    return res, means, pd.DataFrame(diffs), pd.DataFrame(coefs), pd.concat(pooled)


def choose(means):
    mm = means.set_index("candidate").mean_spearman_5_splits
    others = mm.drop("istar")
    if others.max() - mm["istar"] < 0.02:
        return "istar", f"no candidate beats (i*) by >= 0.02 (best other {others.idxmax()} {others.max():.3f} vs i* {mm['istar']:.3f}) -> (i*)"
    top = mm.sort_values(ascending=False).head(2)
    if top.iloc[0] - top.iloc[1] < 0.02:
        pick = min(top.index, key=ORDER.index)
        return pick, f"{top.index[0]} {top.iloc[0]:.3f} beats i* by >= 0.02 but top two differ by < 0.02 -> simpler: {pick}"
    return top.index[0], f"highest mean {top.index[0]} {top.iloc[0]:.3f}, beats i* ({mm['istar']:.3f}) by >= 0.02 and the runner-up by >= 0.02"


def reliability(pm):
    fa, _ = features(pm[pm.idx % 2 == 0]); fb, _ = features(pm[pm.idx % 2 == 1])
    ta, tb = target(pm[pm.idx % 2 == 0]), target(pm[pm.idx % 2 == 1])
    j = fa.join(fb, lsuffix="_a", rsuffix="_b", how="inner").join(ta.rename(columns={"y": "ya", "LH": "LHa"})).join(tb.rename(columns={"y": "yb", "LH": "LHb"}))
    j = j[(j.LHa >= 50) & (j.LHb >= 50)]
    sb = lambda r: 2 * r / (1 + r)
    out = {"n_players": len(j), **{c: sb(pearsonr(j[c + "_a"], j[c + "_b"]).statistic) for c in FE},
           "target": sb(pearsonr(j.ya, j.yb).statistic)}
    out["noise_ceiling"] = float(np.sqrt(max(out["target"], 0)))
    return out


# ------------------------------------------------------------------ final score + uncertainty
def score_and_var(F, pick, model):
    """Point score and its posterior (EB) variance on the score scale."""
    if pick == "ii":
        return F.f1.values, F.f1_postvar.values
    if pick in ("i", "istar"):
        return (F.Istar.values, F.Istar_var.values) if pick == "istar" else (F.A.values, F.A_postvar.values)
    m, mu, sd = model
    pv = (m.coef_[0] / sd[0]) ** 2 * F.f1_postvar.values + (m.coef_[1] / sd[1]) ** 2 * F.f2_postvar.values + (m.coef_[2] / sd[2]) ** 2 * F.A_postvar.values
    return m.predict((F[FE].values - mu) / sd), pv


def final_rating(pm, pick, a, b, use_boot=True):
    F, info = features(pm, istar_draws=1000 if pick == "istar" else ISTAR_DRAWS, seed=SEED + 77)
    model = None
    if pick == "iii":
        m, mu, sd, lam, _, _, _ = ridge_fit(pm, SEED + 99); model = (m, mu, sd)
    sc, pv = score_and_var(F, pick, model)
    rng = np.random.default_rng(SEED + 5)
    D = np.zeros((len(F), N_DRAWS))
    for dd in range(N_DRAWS):
        Fd, _ = features(pm, resample_weights(pm, rng))
        if pick == "ii":
            D[:, dd] = Fd.f1.values
        elif pick in ("i", "istar"):
            D[:, dd] = Fd.A.values               # match-bootstrap component of A (point estimate per draw)
        else:
            D[:, dd] = model[0].predict((Fd[FE].values - model[1]) / model[2])
    bv = D.var(1)
    F["score"] = sc; F["boot_var"] = bv; F["post_var"] = pv
    F["projected"] = a + b * sc
    F["sd_total"] = np.abs(b) * np.sqrt((bv if use_boot else 0) + pv)
    F["cautious"] = F.projected - Z80 * F.sd_total; F["hi80"] = F.projected + Z80 * F.sd_total
    F["p_above_avg"] = norm.cdf(F.projected / F.sd_total)
    return F, info


# ------------------------------------------------------------------ self-test (method test only)
def self_test(pm, pick, seed):
    """Plant true abilities on the real schedule, regenerate per-match sums with the fitted noise, re-run the
    chosen score's estimator with BOTH variance versions; coverage overall and for players with < 15 matches."""
    rng = np.random.default_rng(seed)
    players = np.unique(pm.player.values); codes = np.searchsorted(players, pm.player.values)
    M = np.bincount(codes, minlength=len(players))
    small = M < MIN_M
    rows = []
    if pick == "ii":
        F0, info = features(pm)
        mu, s2, t2 = info["f1"]["mu"], info["f1"]["sigma2"], info["f1"]["tau2"]
        n = pm.n_p.values
        has = np.bincount(codes, weights=n, minlength=len(players)) > 0
        for rep in range(SIM_REPS):
            r = rng.normal(mu, np.sqrt(t2), len(players))
            S = n * r[codes] + np.sqrt(s2 * n) * rng.standard_normal(len(n))
            Q = S ** 2 / np.where(n > 0, n, 1) + s2 * (n - 1).clip(0)
            def est(w):
                Sj = np.bincount(codes, weights=w * S, minlength=len(players)); nj = np.bincount(codes, weights=w * n, minlength=len(players))
                Qj = np.bincount(codes, weights=w * Q, minlength=len(players))
                shr, k, m_, s2h, t2h = shrink_rate(Sj, Qj, nj, nj)
                pv = 1 / (nj / s2h + 1 / t2h) if t2h > 0 else np.zeros(len(nj))
                return shr, pv
            e0, pv = est(np.ones(len(n)))
            sub = pm[["player"]].copy()
            B = np.column_stack([est(resample_weights(sub, rng))[0] for _ in range(SIM_DRAWS)])
            bv = B.var(1)
            for lab, var in (("bootstrap + posterior (specified)", bv + pv), ("posterior only", pv), ("bootstrap only (run 1)", bv)):
                sd = np.sqrt(var); cov = (np.abs(r - e0) <= Z80 * sd)
                rows.append({"rep": rep, "variance": lab, "coverage_all": cov[has].mean(), "coverage_lt15_matches": cov[has & small].mean(),
                             "coverage_ge15_matches": cov[has & ~small].mean(),
                             "spearman_eligible": spearmanr(r[has & ~small], e0[has & ~small]).statistic})
    else:   # rating A / uncertainty-averaged A on player-match leverage sums
        g0 = fast_stats(codes, len(players), pm.H.values, pm.LH.values, pm.L.values, pm.LL.values, M)
        vc = estimate(g0)
        LH, LL = pm.LH.values, pm.LL.values
        has = (np.bincount(codes, weights=LH + LL, minlength=len(players)) > 0)
        for rep in range(SIM_REPS):
            th = rng.normal(0, np.sqrt(max(vc["tau_theta2"], 1e-12)), len(players))
            H = LH * th[codes] + np.sqrt(vc["sH2"] * LH) * rng.standard_normal(len(LH))
            L = LL * th[codes] + np.sqrt(vc["sL2"] * LL) * rng.standard_normal(len(LL))
            g = fast_stats(codes, len(players), H, LH, L, LL, M)
            vcs = boot_vcs(g, SIM_DRAWS, seed + rep + 1)
            post = [post_arrays(g.LH.values, g.LL.values, g.y1.values, g.y2.values, clip_vc(v)) for v in vcs]
            ms = np.array([p_["m"] for p_ in post]); vs = np.array([p_["v"] for p_ in post])
            mean, pv = ms.mean(0), vs.mean(0) + ms.var(0)
            B = []
            for _ in range(SIM_DRAWS // 2):
                w = resample_weights(pm[["player"]], rng)
                gb = fast_stats(codes, len(players), w * H, w * LH, w * L, w * LL, np.bincount(codes, weights=w, minlength=len(players)))
                B.append(post_arrays(gb.LH.values, gb.LL.values, gb.y1.values, gb.y2.values, clip_vc(estimate(gb)))["m"])
            bv = np.var(np.array(B), axis=0)
            for lab, var in (("bootstrap + posterior (specified)", bv + pv), ("posterior only", pv)):
                cov = np.abs(th - mean) <= Z80 * np.sqrt(var)
                rows.append({"rep": rep, "variance": lab, "coverage_all": cov[has].mean(), "coverage_lt15_matches": cov[has & small].mean(),
                             "coverage_ge15_matches": cov[has & ~small].mean(), "spearman_eligible": spearmanr(th[has & ~small], mean[has & ~small]).statistic})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    t0 = time.perf_counter()
    d, credit = build()
    S = sides(d)
    out = {}
    for side in ("bat", "bowl"):
        s, par = S[side]
        tsea = tier_season_table(s)
        pm = player_match(s)
        res, means, diffs, coefs, pooled = evaluate(pm)
        pick, why = choose(means)
        rel = reliability(pm)
        col = f"c_{pick}"
        wls = np.polyfit(pooled[col].values, pooled.y.values, 1, w=np.sqrt(pooled.LH_test.values))
        b, a = float(wls[0]), float(wls[1])
        st = self_test(pm, pick, SEED + (41 if side == "bat" else 43))
        sts = st.groupby("variance")[["coverage_all", "coverage_lt15_matches", "coverage_ge15_matches", "spearman_eligible"]].mean()
        ok = lambda v: 0.75 <= sts.loc[v, "coverage_all"] <= 0.85 and 0.75 <= sts.loc[v, "coverage_lt15_matches"] <= 0.85
        spec = "bootstrap + posterior (specified)"
        if ok(spec):
            var_choice, var_note = spec, "specified total variance passes the coverage acceptance"
        elif ok("posterior only"):
            var_choice, var_note = "posterior only", ("specified total variance FAILS the coverage acceptance (it double-counts sampling noise, "
                                                      "which the posterior variance already contains); fixed by using the posterior variance alone, which passes")
        else:
            var_choice, var_note = spec, "NEITHER variance version passes the coverage acceptance - ranking not trustworthy"
        F, info = final_rating(pm, pick, a, b, use_boot=(var_choice == spec))
        prof = describe(s, F)
        F = F.join(prof)
        F["evidence_level"] = np.where((F.LH >= STRONG) & (F.M >= MIN_M), "Strong", np.where((F.LH >= MODERATE) & (F.M >= MIN_M), "Moderate", "Limited"))
        F = F.sort_values(["cautious", "projected"], ascending=False)
        F["rank_all"] = np.arange(1, len(F) + 1)
        el = F[F.evidence_level != "Limited"]
        F["rank_eligible"] = pd.Series(np.arange(1, len(el) + 1), index=el.index)
        out[side] = dict(res=res, means=means, diffs=diffs, coefs=coefs, pick=pick, why=why, rel=rel, a=a, b=b, F=F, info=info,
                         par=par, tsea=tsea, st=st, sts=sts, var_choice=var_choice, var_note=var_note)
        write_files(F, side, pick)
        print(side, means.round(3).to_dict("records"), pick, why, sts.round(3).to_dict(), var_choice, flush=True)
    pd.concat([o["st"].assign(side=k) for k, o in out.items()]).to_csv(os.path.join(REP4, "phase4c_selftest_reps.csv"), index=False)
    for side in out:
        forest(out[side]["F"], side)
    write_readme(out)
    write_report(out, credit, time.perf_counter() - t0)
    write_howto(out)


def describe(s, F):
    h = s[s.hi]; lh = h.groupby("player").li.sum()
    o = pd.DataFrame(index=F.index)
    o["name"] = s.groupby("player").name.agg(lambda x: x.mode().iat[0]).reindex(F.index)
    for c, vals in (("tier", ["IPL", "full_member", "associate"]), ("phase", ["Powerplay", "Middle", "Death"]), ("innings", [1, 2])):
        t = h.groupby(["player", c]).li.sum().unstack(c).reindex(columns=vals).fillna(0)
        for v in vals:
            o[f"share_LH_{v}"] = (t[v] / lh).reindex(F.index)
    if "bat_runs" in h and (s.v.name if hasattr(s.v, "name") else None):
        pass
    side_bat = (s.v_adj_bat.notna().all())
    if side_bat:
        o["raw_pressure_strike_rate"] = (100 * h.groupby("player").bat_runs.sum() / h.groupby("player").size()).reindex(F.index)
    else:
        lg = h.groupby("player").is_legal.sum()
        o["raw_pressure_economy"] = (6 * h.groupby("player").bowler_runs_conceded.sum() / lg.where(lg > 0)).reindex(F.index)
    return o


def plain_frame(F, side, pick):
    p = lambda x: (100 * x).round(3)
    first, second = ("% batting first", "% chasing") if side == "bat" else ("% bowling in 1st innings", "% bowling in chase")
    f1n, f2n = (("Runs above expected per pressure ball (shrunk)", "Dismissals above expected per pressure ball (shrunk)") if side == "bat"
                else ("Runs conceded below expected per pressure ball (shrunk)", "Wickets above expected per pressure ball (shrunk)"))
    return pd.DataFrame({
        "Rank (eligible)": F.rank_eligible.astype("Int64"), "Rank (all players)": F.rank_all, "Player": F.name, "Evidence level": F.evidence_level,
        "Cautious rating (used for rank)": p(F.cautious), "Projected pressure impact (best estimate)": p(F.projected),
        "Likely range low (80%)": p(F.cautious), "Likely range high (80%)": p(F.hi80),
        "Overall quality rating A": p(F.A), "Pressure record when it mattered": p(F.raw_H_per_LH),
        "Chance he is above average under pressure (%)": (100 * F.p_above_avg).round(1),
        f1n: F.f1.round(4), f2n: F.f2.round(5),
        "Pressure exposure (sum of leverage)": F.LH.round(1), "Pressure balls": F.n_p.astype(int), "Matches": F.M.astype(int),
        "% of pressure from IPL": (100 * F.share_LH_IPL).round(1),
        "% of pressure from internationals": (100 * (F.share_LH_full_member + F.share_LH_associate)).round(1),
        "% pressure in powerplay": (100 * F.share_LH_Powerplay).round(1), "% pressure in middle overs": (100 * F.share_LH_Middle).round(1),
        "% pressure in death overs": (100 * F.share_LH_Death).round(1), first: (100 * F.share_LH_1).round(1), second: (100 * F.share_LH_2).round(1),
        ("Raw pressure strike rate" if side == "bat" else "Raw pressure economy"):
            (F.raw_pressure_strike_rate if side == "bat" else F.raw_pressure_economy).round(2)})


def write_files(F, side, pick):
    plain_frame(F, side, pick).to_csv(os.path.join(MODEL, f"phase4_rankings_{side}_all_players.csv"), index=False)
    t = F.copy(); t.index.name = "player_id"
    t.to_csv(os.path.join(MODEL, f"phase4_rankings_{side}_all_players_technical.csv"))


def forest(F, side):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = F[F.rank_eligible <= 25].sort_values("rank_eligible").iloc[::-1]
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


def write_readme(out):
    L = ["# Rankings files", "",
         "`out/model/phase4_rankings_bat_all_players.csv` and `..._bowl_all_players.csv` (plain English, one header row); "
         "`..._technical.csv` (all model columns, raw units).", "",
         "**Unit.** All ratings are in win-probability points per average ball: +0.10 means the player's side is projected "
         "to gain on average 0.10 percentage points of win probability per pressure ball of average importance, compared "
         "with a typical player in the same season, competition, innings and phase.", "",
         "**Columns.**", "",
         "- Rank (eligible): rank among players with Strong or Moderate evidence; the submission top 10 is ranks 1-10.",
         "- Rank (all players): rank among everyone who faced / bowled at least one valid ball.",
         "- Evidence level: Strong = pressure exposure >= 400 and >= 15 matches; Moderate = >= 150 and >= 15 matches; Limited otherwise.",
         "- Cautious rating: the 10th percentile of the projection - the ranking score.",
         "- Projected pressure impact: the best estimate; Likely range: 80% interval (match resampling + small-sample uncertainty).",
         "- Overall quality rating A: leverage-weighted value over all his balls (Phase 4).",
         "- Pressure record when it mattered: his actual win probability added per unit of leverage on pressure balls (no shrinkage).",
         "- Runs above / conceded below expected: per pressure ball, against the engine's expectation for that exact match "
         "situation, centred by season x competition x innings x phase and corrected for opponent strength and associate matches.",
         "- Pressure exposure: total leverage of his pressure balls (an average ball = 1).", ""]
    for side, R in out.items():
        L.append(f"- {'Batters' if side == 'bat' else 'Bowlers'}: final score = {NAMES[R['pick']]}; uncertainty = {R['var_choice']}.")
    open(os.path.join(MODEL, "README_rankings.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def write_report(out, credit, rt):
    f3 = lambda x: f"{x:.3f}" if isinstance(x, (float, np.floating)) else x
    g4 = lambda x: f"{x:.4g}" if isinstance(x, (float, np.floating)) else x
    L = ["# Phase 4c - context-adjusted projection (final method)", "",
         "## Run 1 (Phase 4b) - rejected", "",
         "Run 1 chose (ii) raw shrunk pressure runs per ball for both roles. It was rejected on inspection before final "
         "selection: (1) environment confound - pressure runs per ball range from 1.21 (associate) to 1.70 (IPL 2026), "
         "about 3 between-player SDs for bowlers, and 6 of the bowling top 10 had 0% IPL exposure (Delany's own pressure "
         "record -0.20); (2) uncertainty bug - the match bootstrap gave zero spread to players with 1 or few matches "
         "(e.g. a 1-match bowler ranked 5th of all bowlers at '100% above average'), and self-test coverage was 0.70-0.72. "
         "Its results are kept in `reports/phase4/phase4b_summary.md`.", "",
         "## Run 2 (this run)", "",
         f"Bowler-credit share of wickets by phase (pre-2022 history): {json.dumps({k: round(v, 4) for k, v in credit.items()})}.", ""]
    for side, R in out.items():
        role = "Batters" if side == "bat" else "Bowlers"
        f1n, f2n = (("runs above expected", "dismissals above expected") if side == "bat" else ("conceded below expected", "wickets above expected"))
        info, rel, par = R["info"], R["rel"], R["par"]
        kt = pd.DataFrame([{"feature": f1n, "k (pressure balls)": info["f1"]["k"], "between-player SD": np.sqrt(max(info["f1"]["tau2"], 0)),
                            "odd/even reliability": rel["f1"]},
                           {"feature": f2n, "k (pressure balls)": info["f2"]["k"], "between-player SD": np.sqrt(max(info["f2"]["tau2"], 0)),
                            "odd/even reliability": rel["f2"]},
                           {"feature": "rating A", "k (pressure balls)": info["A"]["k_equiv"], "between-player SD": np.sqrt(max(info["A"]["tau_theta2"], 0)),
                            "odd/even reliability": rel["A"]}])
        ts = R["tsea"].pivot_table(index=["stage", "tier"], columns="season", values="f1").reset_index()
        ts2 = R["tsea"].pivot_table(index=["stage", "tier"], columns="season", values="f2").reset_index()
        res = R["res"]
        cis = res.assign(ci=[f"{r.spearman:+.3f} [{r.lo95:+.3f}, {r.hi95:+.3f}]" for r in res.itertuples()]).pivot(index="split", columns="candidate", values="ci")
        cis = cis.rename(columns=NAMES); cis.insert(0, "players", res.groupby("split").n_players.first())
        means = R["means"].assign(candidate=R["means"].candidate.map(NAMES))
        F = R["F"]; el = F[F.rank_eligible.notna()].sort_values("rank_eligible")
        top = el.head(25)
        tt = pd.DataFrame({"rank (eligible)": top.rank_eligible.astype(int), "rank (all)": top.rank_all, "player": top.name, "evidence": top.evidence_level,
                           "cautious": (100 * top.cautious).round(3), "projected": (100 * top.projected).round(3),
                           "80% range": [f"{100*a:+.3f} to {100*b:+.3f}" for a, b in zip(top.cautious, top.hi80)],
                           "% IPL": (100 * top.share_LH_IPL).round(1), "pressure record": (100 * top.raw_H_per_LH).round(3),
                           f1n + " (shrunk)": top.f1.round(4), "A": (100 * top.A).round(3), "pressure balls": top.n_p.astype(int),
                           "sum LH": top.LH.round(0).astype(int), "matches": top.M.astype(int)})
        flags = el[(el.rank_eligible <= 10) & ((el.raw_H_per_LH < 0) | (el.share_LH_IPL == 0))]
        flag_lines = []
        for pid, r in flags.iterrows():
            why = []
            if r.raw_H_per_LH < 0: why.append(f"pressure record {100*r.raw_H_per_LH:+.3f}")
            if r.share_LH_IPL == 0: why.append("0% IPL exposure")
            flag_lines.append(f"- Rank {int(r.rank_eligible)} {r['name']} ({', '.join(why)}): ranks here because its score = "
                              f"{('shrunk ' + f1n + f' {r.f1:+.4f} per ball (raw {r.raw_f1:+.4f} over {int(r.n_p)} pressure balls, {int(r.M)} matches)') if R['pick'] == 'ii' else f'rating {100*r.score:+.3f}'}"
                              f" -> projected {100*r.projected:+.3f}, 80% range {100*r.cautious:+.3f} to {100*r.hi80:+.3f}; overall rating A {100*r.A:+.3f}.")
        sep = []
        for _, r in el.head(10).iterrows():
            below = el[el.rank_eligible > r.rank_eligible]; clear = below[below.hi80 < r.cautious]
            sep.append({"rank": int(r.rank_eligible), "player": r["name"], "80% range": f"{100*r.cautious:+.3f} to {100*r.hi80:+.3f}",
                        "first rank clearly below": int(clear.rank_eligible.min()) if len(clear) else None,
                        "separable from next": bool(len(below) and below.iloc[0].hi80 < r.cautious)})
        small = F[F.M < MIN_M]
        sm = pd.DataFrame({"matches": ["1", "2-4", "5-14", ">=15"],
                           "players": [int((F.M == 1).sum()), int(((F.M >= 2) & (F.M <= 4)).sum()), int(((F.M >= 5) & (F.M < 15)).sum()), int((F.M >= 15).sum())],
                           "median 80% width (pts)": [100 * (F.hi80 - F.cautious)[m].median() for m in (F.M == 1, (F.M >= 2) & (F.M <= 4), (F.M >= 5) & (F.M < 15), F.M >= 15)],
                           "best rank_all in group": [int(F.rank_all[m].min()) if m.any() else None for m in (F.M == 1, (F.M >= 2) & (F.M <= 4), (F.M >= 5) & (F.M < 15), F.M >= 15)]})
        L += [f"## {role}", "",
              "### Corrected features: mean per pressure ball by tier x season", "",
              f"{f1n}:", "", ts.map(g4).to_markdown(index=False), "", f"{f2n}:", "", ts2.map(g4).to_markdown(index=False), "",
              "After centring every cell is 0 to rounding. The tier correction then deliberately shifts associate-tier balls by "
              f"-gamma_t (the within-player associate gain): {f1n} gamma {json.dumps({k: round(v, 4) for k, v in par['f1']['gamma_by_season'].items()})}; "
              f"opponent beta {json.dumps({k: round(v, 4) for k, v in par['f1']['beta_by_season'].items()})}.", "",
              "### Features (all data)", "", kt.map(g4).to_markdown(index=False), "",
              f"Target odd/even reliability {rel['target']:.3f} ({rel['n_players']} players) -> noise ceiling about {rel['noise_ceiling']:.2f}.", "",
              "### Out-of-sample Spearman by split [95% CI]", "", cis.reset_index().to_markdown(index=False), "",
              means.map(f3).to_markdown(index=False), "", "Paired differences vs (i*):", "", R["diffs"].map(f3).to_markdown(index=False), "",
              "Ridge per split (lambda grid extended until interior):", "", R["coefs"].map(g4).to_markdown(index=False), "",
              f"**Chosen: {NAMES[R['pick']]}** - {R['why']}.", "",
              f"Mapping to win-probability points (pooled out-of-sample weighted fit): projected = {R['a']:.5f} + {R['b']:.5f} x score.", "",
              "### Self-test of the uncertainty (method test only; 200 repetitions)", "",
              R["sts"].reset_index().map(f3).to_markdown(index=False), "",
              f"Acceptance: coverage 0.75-0.85 overall and for < 15 matches. **{R['var_choice']}** - {R['var_note']}.", "",
              "### Small samples now get wide ranges", "", sm.map(f3).to_markdown(index=False), "",
              "### Top 25 eligible", "", tt.to_markdown(index=False), "", f"![forest](phase4_forest_top25_{side}.png)", "",
              f"### Eligible top-10 flags (negative pressure record or 0% IPL): {len(flags)}", "", "\n".join(flag_lines) if flag_lines else "None.", "",
              "### Separability of the top 10 (80% ranges)", "", pd.DataFrame(sep).to_markdown(index=False), ""]
    L += [f"Runtime {rt:.0f} s.", ""]
    open(os.path.join(REP4, "phase4c_summary.md"), "w", encoding="utf-8").write("\n".join(L))


def write_howto(out):
    p = lambda x: f"{100 * x:+.2f}"
    Rb, Rw = out["bat"], out["bowl"]
    ex = Rb["F"][Rb["F"].rank_eligible == 1].iloc[0]; exw = Rw["F"][Rw["F"].rank_eligible == 1].iloc[0]
    desc = {"ii": "how many more runs than expected he scored (batters) or how many fewer than expected he conceded (bowlers) on pressure balls",
            "i": "his overall quality rating", "istar": "his overall quality rating, averaged over the model's uncertainty",
            "iii": "a weighted mix of his runs above expected, dismissals or wickets above expected, and overall quality"}
    L = ["# How the ranking works", "",
         "1. **Give every ball an importance score.** A win-probability model built on ball-by-ball data says how much the "
         "match result could swing on that ball (average ball = 1; the last ball of a close chase can reach 30). Balls "
         "scoring 1 or more are the **pressure balls**.",
         "2. **Compare each pressure ball with what was expected in that exact situation.** The same model gives the "
         "expected runs and the chance of a wicket for that score, balls left, wickets left, season and competition. "
         "We record runs above (or below) expected and wickets above expected, compare them with the average for that "
         "season, competition, innings and phase, and correct for the opponent's strength and for matches against weaker "
         "associate teams.",
         "3. **Test which measure best predicts a player's future pressure record.** We hid part of the data five ways "
         "(alternate matches, and each later season) and used a rule fixed in advance to pick the measure. Batters: "
         f"{desc[Rb['pick']]}. Bowlers: {desc[Rw['pick']]}.",
         "4. **Turn it into a projection with honest uncertainty.** Short records are pulled towards average, and the "
         "likely range combines how much the result moves when his matches are reshuffled with how little we know from a "
         "short record - so a player with one match gets a wide range.",
         "5. **Rank by the cautious rating** (the bottom of the 80% range). Only players with pressure exposure of at least "
         f"{MODERATE} and at least {MIN_M} matches can enter the top 10.", "",
         f"## Worked example: {ex['name']} (batting rank 1)", "",
         f"- {int(ex.n_p)} pressure balls in {int(ex.M)} matches; pressure exposure {ex.LH:.0f}; {100 * ex.share_LH_IPL:.0f}% of it in the IPL.",
         f"- Runs above expected: {ex.raw_f1:+.3f} per pressure ball raw, {ex.f1:+.3f} after pulling towards average "
         f"(pressure strike rate {ex.raw_pressure_strike_rate:.1f}).",
         f"- Pressure record (win probability added per unit of importance): {p(ex.raw_H_per_LH)} points.",
         f"- Projected pressure impact {p(ex.projected)}, likely range {p(ex.cautious)} to {p(ex.hi80)}; cautious rating **{p(ex.cautious)}**.",
         f"- Bowling rank 1: {exw['name']} - {int(exw.n_p)} pressure balls, {int(exw.M)} matches, projected {p(exw.projected)}, cautious {p(exw.cautious)}.", "",
         "> **What we found**",
         "> - Pressure-specific skill is not detectable: no player does measurably better under pressure than his general "
         "level predicts. Underlying quality carries into pressure.",
         f"> - Future pressure records are hard to predict: best mean out-of-sample correlation {Rb['means'].mean_spearman_5_splits.max():.2f} "
         f"(batters) and {Rw['means'].mean_spearman_5_splits.max():.2f} (bowlers), against ceilings of about "
         f"{Rb['rel']['noise_ceiling']:.2f} and {Rw['rel']['noise_ceiling']:.2f}.",
         "> - Raw runs per ball mostly measure where and when a player played; comparing with the expected value for each "
         "situation removes that.",
         "> - Neighbouring ranks overlap: read the ranges, not just the order.", ""]
    open(os.path.join(REP4, "how_the_ranking_works.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
