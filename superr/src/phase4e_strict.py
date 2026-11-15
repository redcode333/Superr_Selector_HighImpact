"""Phase 4d - strict, LI-weighted pressure definition (batters and bowlers).

Pressure set (pre-registered): primary LI >= 1.5; sensitivity LI >= 2.0 and global top 10% of LI (LI >= p90).
Within the set every ball is weighted by its LI. Selection target = held-out sum(v_adj) / sum(LI) on the same set.
Candidates: (i*) uncertainty-averaged overall rating A (all balls; unchanged baseline); (P1) shrunk LI-weighted runs
above expected (batters) / conceded below expected (bowlers) on the set; (P2) shrunk LI-weighted win-probability value
on the set; (R) ridge on A + P1 + LI-weighted dismissals / wickets above expected (lambda extended until interior).
Rule: the best candidate is chosen only if it beats (i*) by >= 0.02 and the runner-up by >= 0.02, else the simpler
(P1 < P2 < i* < R); if nothing beats (i*) by >= 0.02, the final score is (i*).
Shrinkage: P1 / dismissal features - LI-weighted mean with effective sample size n_eff = (sum LI)^2 / sum LI^2, between-
player variance by method of moments; P2 - Phase 4 player-match residual variance sH2 and k = sH2 / tau2.
Uncertainty: posterior variance, plus match-bootstrap variance only if the self-test passes with it.
Eligibility: own-data share (1 - posterior var / prior var of the final score) >= 0.50, >= 5 matches in 2025-26,
and the Unstop dropdown list if data/unstop_dropdown_players.csv exists.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, norm
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_outcome import MODEL, SEED
from variant import mpath, vdir, FLOOR
from phase4_ratings import Z80, REP4
from phase4b_ratings import fast_stats, estimate, clip_vc, post_arrays, boot_vcs
from phase4c_projection import resample_weights
from phase4d_context import build, correct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_BOOT, ISTAR_DRAWS, SIM_REPS, SIM_DRAWS = 1000, 200, 200, 100
ORDER = ["P1", "P2", "istar", "R"]
NAMES = {"istar": "(i*) uncertainty-averaged A", "P1": "(P1) LI-weighted runs vs expected", "P2": "(P2) LI-weighted WP value",
         "R": "(R) ridge A + P1 + dismissals/wickets"}
SHARE_MIN, RECENT_MIN = 0.50, 5
PHASES = ["Powerplay", "Middle", "Death"]


# ------------------------------------------------------------------ data
def make_side(d, side, thr):
    if side == "bat":
        s = d[d.v_adj_bat.notna()].copy()
        s["player"], s["name"], s["v"], s["q"] = s.striker_id, s.striker_name, s.v_adj_bat, s.q_opp_bat
        s["f1"] = s.bat_runs - s.E_bat
        s["f2"] = -((s.dismissed_id == s.striker_id).astype(float) - s.pW)          # higher = better (fewer dismissals)
    else:
        s = d.copy()
        s["player"], s["name"], s["v"], s["q"] = s.bowler_id, s.bowler_name, s.v_adj_bowl, s.q_opp_bowl
        s["f1"] = s.E_conc - s.bowler_runs_conceded
        s["f2"] = s.bowler_wicket.astype(bool).astype(float) - s.pW * s.credit
    s["q"] = s.q.fillna(0.0)
    s["hi"] = s.li >= thr
    hi, par = correct(s, ["f1", "f2"])
    s = s.drop(columns=["f1", "f2"]).join(hi[["f1", "f2"]])
    s[["f1", "f2"]] = s[["f1", "f2"]].fillna(0.0)
    s["hiA"] = s.li >= 1                       # rating A keeps its own (unchanged) all-ball split
    return s


def player_match(s):
    li, hi = s.li.values, s.hi.values
    h = pd.DataFrame({"player": s.player.values, "match_id": s.match_id.values, "season": s.season.values, "date": s.match_date.values,
                      "n_p": hi.astype(float), "SL": hi * li, "SL2": hi * li ** 2,
                      "Sf1": hi * li * s.f1.values, "Sf1q": hi * li * s.f1.values ** 2, "Sf2": hi * li * s.f2.values, "Sf2q": hi * li * s.f2.values ** 2,
                      "H": hi * s.v.values,
                      "HA": np.where(s.hiA, s.v, 0.0), "LHA": np.where(s.hiA, li, 0.0), "LA": np.where(~s.hiA, s.v, 0.0), "LLA": np.where(~s.hiA, li, 0.0)})
    pm = h.groupby(["player", "match_id"]).agg(season=("season", "first"), date=("date", "first"),
                                               **{c: (c, "sum") for c in ["n_p", "SL", "SL2", "Sf1", "Sf1q", "Sf2", "Sf2q", "H", "HA", "LHA", "LA", "LLA"]}).reset_index()
    pm = pm.sort_values(["player", "date", "match_id"]).reset_index(drop=True)
    pm["idx"] = pm.groupby("player").cumcount()
    return pm


# ------------------------------------------------------------------ estimators
def shrink_w(SL, Sf, Sfq, SL2, min_neff=30):
    mu = Sf.sum() / SL.sum()
    x = np.where(SL > 0, Sf / np.where(SL > 0, SL, 1), mu)
    neff = np.where(SL2 > 0, SL ** 2 / np.where(SL2 > 0, SL2, 1), 0.0)
    ok = SL > 0
    den = (SL[ok] - SL2[ok] / SL[ok]).sum()
    s2 = (Sfq[ok] - Sf[ok] ** 2 / SL[ok]).sum() / den
    e = neff >= min_neff; N = neff[e].sum()
    t2 = ((neff[e] * (x[e] - mu) ** 2).sum() - (e.sum() - 1) * s2) / (N - (neff[e] ** 2).sum() / N)
    if t2 <= 0:
        return np.full(len(SL), mu), np.zeros(len(SL)), np.zeros(len(SL)), {"k": np.inf, "mu": mu, "sigma2": s2, "tau2": t2}
    k = s2 / t2
    shr = (neff * x + k * mu) / (neff + k)
    pv = 1 / (neff / s2 + 1 / t2)
    return shr, pv, neff / (neff + k), {"k": k, "mu": mu, "sigma2": s2, "tau2": t2}


def shrink_value(codes, nP, H, LH, M):
    """P2: player-match residual variance (Phase 4 style), MoM tau2, shrink to 0."""
    Hj, LHj = np.bincount(codes, H, nP), np.bincount(codes, LH, nP)
    y = np.where(LHj > 0, Hj / np.where(LHj > 0, LHj, 1), 0.0)
    num = np.bincount(codes, (H - LH * y[codes]) ** 2, nP)
    den = np.bincount(codes, np.where(LHj[codes] > 0, LH - LH ** 2 / np.where(LHj[codes] > 0, LHj[codes], 1), 0), nP)
    m2 = M >= 2
    s2 = num[m2].sum() / den[m2].sum()
    e = LHj >= 20
    t2 = (LHj[e] * y[e] ** 2 - s2).sum() / LHj[e].sum()
    if t2 <= 0:
        return np.zeros(nP), np.zeros(nP), np.zeros(nP), {"k": np.inf, "sigma2": s2, "tau2": t2}
    k = s2 / t2
    return Hj / (LHj + k), 1 / (LHj / s2 + 1 / t2), LHj / (LHj + k), {"k": k, "sigma2": s2, "tau2": t2}


def features(pm, w=None, istar_draws=0, seed=SEED):
    w = np.ones(len(pm)) if w is None else w
    keep = w > 0
    rep = np.repeat(np.arange(len(pm))[keep], w[keep].astype(int))
    sub = pm.iloc[rep]
    players = np.unique(sub.player.values); codes = np.searchsorted(players, sub.player.values); nP = len(players)
    M = np.bincount(codes, minlength=nP).astype(float)
    sm = lambda c: np.bincount(codes, sub[c].values, nP)
    F = pd.DataFrame(index=players)
    SL, SL2 = sm("SL"), sm("SL2")
    F["P1"], F["P1_pv"], F["P1_share"], i1 = shrink_w(SL, sm("Sf1"), sm("Sf1q"), SL2)
    F["F2"], F["F2_pv"], _, i2 = shrink_w(SL, sm("Sf2"), sm("Sf2q"), SL2)
    F["P2"], F["P2_pv"], F["P2_share"], i3 = shrink_value(codes, nP, sub.H.values, sub.SL.values, M)
    g = fast_stats(codes, nP, sub.HA.values, sub.LHA.values, sub.LA.values, sub.LLA.values, M.astype(int))
    vc = estimate(g); rA = post_arrays(g.LHA.values if "LHA" in g else g.LH.values, g.LL.values, g.y1.values, g.y2.values, clip_vc(vc))
    F["A"], F["A_pv"] = rA["th"], rA["vth"]
    if istar_draws:
        vcs = boot_vcs(g, istar_draws, seed)
        post = [post_arrays(g.LH.values, g.LL.values, g.y1.values, g.y2.values, clip_vc(v)) for v in vcs]
        ms, vs = np.array([p["m"] for p in post]), np.array([p["v"] for p in post])
        F["Istar"], F["Istar_var"] = ms.mean(0), vs.mean(0) + ms.var(0)
        cv = clip_vc(vc); F["Istar_share"] = 1 - F.Istar_var / (cv["tau_theta2"] + cv["tau_delta2"])
    F["n_p"], F["SL"], F["M"] = sm("n_p"), SL, M
    F["raw_value"] = np.where(SL > 0, sm("H") / np.where(SL > 0, SL, 1), np.nan)
    F["raw_P1"] = np.where(SL > 0, sm("Sf1") / np.where(SL > 0, SL, 1), np.nan)
    return F, {"P1": i1, "F2": i2, "P2": i3, "A": {"k_equiv": vc["sH2"] / max(vc["tau_sum2"], 1e-12), **vc}}


def target(pm):
    g = pm.groupby("player").agg(H=("H", "sum"), LH=("SL", "sum"))
    return pd.DataFrame({"y": g.H / g.LH.where(g.LH > 0), "LH": g.LH})


FE = ["A", "P1", "F2"]


def ridge_fit(pm_tr, seed):
    pm_tr = pm_tr.copy(); pm_tr["iidx"] = pm_tr.groupby("player").cumcount()
    X, Y, W, P = [], [], [], []
    for a, b in ((0, 1), (1, 0)):
        fa, _ = features(pm_tr[pm_tr.iidx % 2 == a]); tb = target(pm_tr[pm_tr.iidx % 2 == b])
        j = fa[FE + ["SL"]].join(tb, how="inner"); j = j[(j.SL >= 30) & (j.LH >= 15)].dropna()
        X.append(j[FE].values); Y.append(j.y.values); W.append(j.LH.values); P.append(j.index.values)
    X, Y, W, P = np.vstack(X), np.concatenate(Y), np.concatenate(W), np.concatenate(P)
    mu, sd = X.mean(0), X.std(0); Z = (X - mu) / sd
    up = np.unique(P); fold_of = dict(zip(up, np.random.default_rng(seed).integers(0, 5, len(up)))); fold = np.array([fold_of[p] for p in P])
    def cv(lam):
        e = 0.0
        for k in range(5):
            tr, te = fold != k, fold == k
            m = Ridge(alpha=lam).fit(Z[tr], Y[tr], sample_weight=W[tr]); e += (W[te] * (Y[te] - m.predict(Z[te])) ** 2).sum()
        return e / W.sum()
    res = {l: cv(l) for l in np.logspace(-2, 4, 13)}
    for _ in range(12):
        best = min(res, key=res.get)
        if best == max(res):
            for l in np.logspace(np.log10(best) + 0.5, np.log10(best) + 2, 4): res[l] = cv(l)
        elif best == min(res):
            for l in np.logspace(np.log10(best) - 2, np.log10(best) - 0.5, 4): res[l] = cv(l)
        else:
            break
    lam = min(res, key=res.get)
    return Ridge(alpha=lam).fit(Z, Y, sample_weight=W), mu, sd, lam, (lam != max(res)) and (lam != min(res)), len(Y)


# ------------------------------------------------------------------ evaluation
def splits(pm):
    out = [("odd->even", pm.idx % 2 == 1, pm.idx % 2 == 0), ("even->odd", pm.idx % 2 == 0, pm.idx % 2 == 1)]
    for t in (2023, 2024, 2025):
        out.append((f"<={t} -> {t + 1}", pm.season <= t, pm.season == t + 1))
    return out


def evaluate(pm, istar_cache, lh_tr, lh_te):
    cands = ["istar", "P1", "P2", "R"]
    rows, coefs, pooled, sets = [], [], [], []
    for si_, (name, trm, tem) in enumerate(splits(pm)):
        tr, te = pm[trm.values], pm[tem.values]
        F, info = features(tr)
        if name not in istar_cache:
            istar_cache[name] = features(tr, istar_draws=ISTAR_DRAWS, seed=SEED + si_)[0].Istar
        F["c_istar"] = istar_cache[name].reindex(F.index).values
        m, mu, sd, lam, interior, npairs = ridge_fit(tr, SEED + si_)
        F["c_P1"], F["c_P2"], F["c_R"] = F.P1, F.P2, m.predict((F[FE].values - mu) / sd)
        coefs.append({"split": name, "lambda": lam, "interior": interior, "n_pairs": npairs,
                      **{f"coef_{c} (per SD)": b for c, b in zip(FE, m.coef_)}, "k_P1": info["P1"]["k"], "k_P2": info["P2"]["k"], "k_F2": info["F2"]["k"]})
        j = F.join(target(te).rename(columns={"LH": "LH_test"}), how="inner")
        j = j[(j.SL >= lh_tr) & (j.LH_test >= lh_te)].dropna(subset=["y"])
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
    means = pd.DataFrame([{"candidate": c, "mean": res[res.candidate == c].spearman.mean(), "lo95": np.percentile(rep[c].mean(1), 2.5),
                           "hi95": np.percentile(rep[c].mean(1), 97.5), "positive_splits": int((res[res.candidate == c].spearman > 0).sum())} for c in cands])
    mm = means.set_index("candidate")["mean"]
    diffs = []
    for c in ("P1", "P2", "R"):
        d_ = rep[c].mean(1) - rep["istar"].mean(1)
        diffs.append({"difference": f"{c} - i*", "mean": mm[c] - mm["istar"], "lo95": np.percentile(d_, 2.5), "hi95": np.percentile(d_, 97.5),
                      "splits_CI_excl_0": int(sum((np.percentile(rep[c][:, k] - rep["istar"][:, k], 2.5) > 0) or
                                                  (np.percentile(rep[c][:, k] - rep["istar"][:, k], 97.5) < 0) for k in range(len(sets))))})
    top2 = mm.sort_values(ascending=False).index[:2]
    d12 = rep[top2[0]].mean(1) - rep[top2[1]].mean(1)
    diffs.append({"difference": f"{top2[0]} - {top2[1]} (top two)", "mean": mm[top2[0]] - mm[top2[1]], "lo95": np.percentile(d12, 2.5),
                  "hi95": np.percentile(d12, 97.5), "splits_CI_excl_0": None})
    return res, means, pd.DataFrame(diffs), pd.DataFrame(coefs), pd.concat(pooled)


def choose(means):
    mm = means.set_index("candidate")["mean"]
    others = mm.drop("istar")
    if others.max() - mm["istar"] < 0.02:
        return "istar", f"nothing beats (i*) by >= 0.02 (best other {others.idxmax()} {others.max():.3f} vs i* {mm['istar']:.3f}) -> (i*)"
    top = mm.sort_values(ascending=False).head(2)
    if top.iloc[0] - top.iloc[1] < 0.02:
        pick = min(top.index, key=ORDER.index)
        return pick, f"{top.index[0]} {top.iloc[0]:.3f} and {top.index[1]} {top.iloc[1]:.3f} differ by < 0.02 -> simpler: {pick}"
    return top.index[0], f"{top.index[0]} {top.iloc[0]:.3f} beats i* ({mm['istar']:.3f}) and the runner-up by >= 0.02"


def reliability_target(pm):
    ta, tb = target(pm[pm.idx % 2 == 0]), target(pm[pm.idx % 2 == 1])
    j = ta.join(tb, lsuffix="a", rsuffix="b", how="inner"); j = j[(j.LHa >= 30) & (j.LHb >= 30)].dropna()
    r = pearsonr(j.ya, j.yb).statistic; sb = 2 * r / (1 + r)
    return {"n_players": len(j), "target_reliability": sb, "noise_ceiling": float(np.sqrt(max(sb, 0)))}


# ------------------------------------------------------------------ final + self-test
def score_var(F, pick, model):
    if pick == "P1": return F.P1.values, F.P1_pv.values, F.P1_share.values
    if pick == "P2": return F.P2.values, F.P2_pv.values, F.P2_share.values
    if pick == "istar": return F.Istar.values, F.Istar_var.values, F.Istar_share.values
    m, mu, sd = model
    c = m.coef_ / sd
    pv = c[0] ** 2 * F.A_pv.values + c[1] ** 2 * F.P1_pv.values + c[2] ** 2 * F.F2_pv.values
    sc = m.predict((F[FE].values - mu) / sd)
    prior = c[0] ** 2 * F.A.var() + c[1] ** 2 * F.P1.var() + c[2] ** 2 * F.F2.var()
    return sc, pv, 1 - pv / max(prior + pv.mean(), 1e-18)


def self_test(pm, pick, seed):
    rng = np.random.default_rng(seed)
    players = np.unique(pm.player.values); codes = np.searchsorted(players, pm.player.values); nP = len(players)
    M = np.bincount(codes, minlength=nP); small = M < 15
    F0, info = features(pm)
    rows = []
    for rep_ in range(SIM_REPS):
        if pick in ("P1", "R"):
            i = info["P1"]; SL, SL2 = pm.SL.values, pm.SL2.values
            truth = rng.normal(i["mu"], np.sqrt(max(i["tau2"], 1e-12)), nP)
            Sf = SL * truth[codes] + np.sqrt(i["sigma2"] * SL2) * rng.standard_normal(len(SL))
            Sfq = np.where(SL > 0, Sf ** 2 / np.where(SL > 0, SL, 1), 0) + i["sigma2"] * np.where(SL > 0, SL - SL2 / np.where(SL > 0, SL, 1), 0)
            def est(w):
                b = lambda x: np.bincount(codes, w * x, nP)
                shr, pv, _, _ = shrink_w(b(SL), b(Sf), b(Sfq), b(SL2)); return shr, pv
            has = np.bincount(codes, SL, nP) > 0
        elif pick == "P2":
            i = info["P2"]; LH = pm.SL.values
            truth = rng.normal(0, np.sqrt(max(i["tau2"], 1e-12)), nP)
            H = LH * truth[codes] + np.sqrt(i["sigma2"] * LH) * rng.standard_normal(len(LH))
            def est(w):
                keep = w > 0; r_ = np.repeat(np.arange(len(LH))[keep], w[keep].astype(int))
                shr, pv, _, _ = shrink_value(codes[r_], nP, H[r_], LH[r_], np.bincount(codes[r_], minlength=nP)); return shr, pv
            has = np.bincount(codes, LH, nP) > 0
        else:
            g0 = fast_stats(codes, nP, pm.HA.values, pm.LHA.values, pm.LA.values, pm.LLA.values, M); vc = estimate(g0)
            LHA, LLA = pm.LHA.values, pm.LLA.values
            truth = rng.normal(0, np.sqrt(max(vc["tau_theta2"], 1e-12)), nP)
            HA = LHA * truth[codes] + np.sqrt(vc["sH2"] * LHA) * rng.standard_normal(len(LHA))
            LA = LLA * truth[codes] + np.sqrt(vc["sL2"] * LLA) * rng.standard_normal(len(LLA))
            def est(w, draws=0):
                g = fast_stats(codes, nP, w * HA, w * LHA, w * LA, w * LLA, np.bincount(codes, w, nP))
                if draws:
                    post = [post_arrays(g.LH.values, g.LL.values, g.y1.values, g.y2.values, clip_vc(v)) for v in boot_vcs(g, draws, seed + rep_)]
                    ms = np.array([p["m"] for p in post]); return ms.mean(0), np.array([p["v"] for p in post]).mean(0) + ms.var(0)
                r_ = post_arrays(g.LH.values, g.LL.values, g.y1.values, g.y2.values, clip_vc(estimate(g))); return r_["m"], r_["v"]
            has = np.bincount(codes, LHA + LLA, nP) > 0
        e0, pv = est(np.ones(len(pm)), SIM_DRAWS) if pick == "istar" else est(np.ones(len(pm)))
        B = np.column_stack([est(resample_weights(pm[["player"]], rng))[0] for _ in range(SIM_DRAWS if pick != "istar" else SIM_DRAWS // 2)])
        bv = B.var(1)
        for lab, var in (("posterior only", pv), ("posterior + bootstrap", pv + bv)):
            cov = np.abs(truth - e0) <= Z80 * np.sqrt(var)
            rows.append({"rep": rep_, "variance": lab, "coverage_all": cov[has].mean(), "coverage_lt15": cov[has & small].mean(),
                         "coverage_ge15": cov[has & ~small].mean()})
    return pd.DataFrame(rows)


def run_definition(d, side, defn, thr, istar_cache, with_selftest, var_override=None):
    s = make_side(d, side, thr)
    pm = player_match(s)
    res, means, diffs, coefs, pooled = evaluate(pm, istar_cache, lh_tr=60, lh_te=30)
    pick, why = choose(means)
    rel = reliability_target(pm)
    wls = np.polyfit(pooled[f"c_{pick}"].values, pooled.y.values, 1, w=np.sqrt(pooled.LH_test.values))
    b, a = float(wls[0]), float(wls[1])
    sts, var_choice, var_note = None, var_override or "posterior only", "variance choice taken from the primary definition"
    if with_selftest:
        st = self_test(pm, pick, SEED + (51 if side == "bat" else 53))
        sts = st.groupby("variance")[["coverage_all", "coverage_lt15", "coverage_ge15"]].mean()
        ok = lambda v: 0.75 <= sts.loc[v, "coverage_all"] <= 0.85 and 0.75 <= sts.loc[v, "coverage_lt15"] <= 0.85
        if ok("posterior + bootstrap"):
            var_choice, var_note = "posterior + bootstrap", "passes the coverage acceptance with the bootstrap added"
        elif ok("posterior only"):
            var_choice, var_note = "posterior only", "posterior variance passes; adding the bootstrap does not"
        else:
            var_choice, var_note = "posterior only", "NEITHER version passes the coverage acceptance - ranges not trustworthy"
    F, info = features(pm, istar_draws=1000 if pick == "istar" else 0, seed=SEED + 77)
    model = None
    if pick == "R":
        m, mu, sd, *_ = ridge_fit(pm, SEED + 99); model = (m, mu, sd)
    sc, pv, share = score_var(F, pick, model)
    bv = np.zeros(len(F))
    if var_choice == "posterior + bootstrap":
        rng = np.random.default_rng(SEED + 5); D = []
        for _ in range(500):
            Fd, _ = features(pm, resample_weights(pm, rng))
            D.append(score_var(Fd.reindex(F.index), pick if pick != "istar" else "P2", model)[0] if pick != "istar" else Fd.A.reindex(F.index).values)
        bv = np.nanvar(np.array(D), axis=0)
    F["score"], F["post_var"], F["boot_var"], F["own_share"] = sc, pv, bv, share
    F["projected"] = a + b * sc
    F["sd_total"] = abs(b) * np.sqrt(pv + bv)
    F["cautious"], F["hi80"] = F.projected - Z80 * F.sd_total, F.projected + Z80 * F.sd_total
    F["p_above_avg"] = norm.cdf(F.projected / F.sd_total.where(F.sd_total > 0))
    # descriptives
    h = s[s.hi]
    lh = h.groupby("player").li.sum()
    for c, vals in (("phase", PHASES), ("innings", [1, 2]), ("tier", ["IPL", "full_member", "associate"])):
        t = h.groupby(["player", c]).li.sum().unstack(c).reindex(columns=vals).fillna(0)
        for v in vals:
            F[f"share_{v}"] = (t[v] / lh).reindex(F.index)
    F["name"] = s.groupby("player").name.agg(lambda x: x.mode().iat[0]).reindex(F.index)
    F["matches_all"] = s.groupby("player").match_id.nunique().reindex(F.index)
    F["matches_2025_26"] = s[s.season >= 2025].groupby("player").match_id.nunique().reindex(F.index).fillna(0).astype(int)
    F["strict_balls_per_match"] = F.n_p / F.matches_all
    if side == "bat":
        F["raw_pressure_strike_rate"] = (100 * h.groupby("player").bat_runs.sum() / h.groupby("player").size()).reindex(F.index)
    else:
        lg = h.groupby("player").is_legal.sum()
        F["raw_pressure_economy"] = (6 * h.groupby("player").bowler_runs_conceded.sum() / lg.where(lg > 0)).reindex(F.index)
    return dict(s=s, pm=pm, res=res, means=means, diffs=diffs, coefs=coefs, pick=pick, why=why, rel=rel, a=a, b=b, F=F, info=info,
                sts=sts, var_choice=var_choice, var_note=var_note)


def eligibility(F, dropdown):
    reasons = []
    for pid, r in F.iterrows():
        why = []
        if not (r.own_share >= SHARE_MIN): why.append(f"own-data share {100 * r.own_share:.0f}% < 50%")
        if r.matches_2025_26 < RECENT_MIN: why.append(f"{int(r.matches_2025_26)} matches in 2025-26 < 5")
        if FLOOR and r.n_p < FLOOR: why.append(f"{int(r.n_p)} high-pressure balls (LI >= 1.5) < {FLOOR}")
        if dropdown is not None and r["name"] not in dropdown: why.append("not in Unstop dropdown list")
        reasons.append("yes" if not why else "no: " + "; ".join(why))
    F["eligible"] = [x == "yes" for x in reasons]; F["eligible_reason"] = reasons
    F = F.sort_values(["cautious", "projected"], ascending=False)
    F["rank_all"] = np.arange(1, len(F) + 1)
    el = F[F.eligible]
    F["rank_eligible"] = pd.Series(np.arange(1, len(el) + 1), index=el.index)
    return F


def composition(d, thr):
    v = d.assign(inset=d.li >= thr)
    g = v.groupby(["innings", "phase"]).agg(balls=("li", "size"), set_balls=("inset", "sum"),
                                           set_LI=("li", lambda x: x[v.loc[x.index, "inset"]].sum())).reset_index()
    g["share_of_cell_in_set"] = g.set_balls / g.balls
    g["share_of_set_balls"] = g.set_balls / g.set_balls.sum()
    g["LI_weighted_share_of_set"] = g.set_LI / g.set_LI.sum()
    g["phase"] = pd.Categorical(g.phase, PHASES)
    return g.sort_values(["innings", "phase"]), float(v.inset.mean())


def bias(F):
    el = F[F.eligible].sort_values("rank_eligible")
    rows = []
    for lab, grp in (("eligible top 10", el.head(10)), ("eligible top 25", el.head(25)), ("all eligible", el)):
        rows.append({"group": lab, "n": len(grp), **{f"mean {p} share": grp[f"share_{p}"].mean() for p in PHASES},
                     "strict balls per match": grp.strict_balls_per_match.mean()})
    t = pd.DataFrame(rows)
    rho_pp = spearmanr(el.score, el.share_Powerplay, nan_policy="omit").statistic
    rho_de = spearmanr(el.score, el.share_Death, nan_policy="omit").statistic
    gap = {p: float(t.iloc[0][f"mean {p} share"] - t.iloc[2][f"mean {p} share"]) for p in PHASES}
    return t, rho_pp, rho_de, gap


# ------------------------------------------------------------------ main
def main():
    t0 = time.perf_counter()
    d, credit = build()
    p90 = float(d.li.quantile(0.9))
    defs = [("LI >= 1.5 (primary)", 1.5), ("LI >= 2.0", 2.0), (f"top 10% (LI >= {p90:.3f})", p90)]
    dd_path = [p for p in (os.path.join(ROOT, "data", "unstop_dropdown_players.csv"), os.path.join(os.path.dirname(ROOT), "data", "unstop_dropdown_players.csv")) if os.path.exists(p)]
    dropdown = set(pd.read_csv(dd_path[0]).iloc[:, 0].astype(str)) if dd_path else None
    comp = {lab: composition(d, thr) for lab, thr in defs}
    out = {}
    for side in ("bat", "bowl"):
        cache = {}
        out[side] = {}
        prim = None
        for k, (lab, thr) in enumerate(defs):
            R = run_definition(d, side, lab, thr, cache, with_selftest=(k == 0), var_override=None if k == 0 else prim["var_choice"])
            R["F"] = eligibility(R["F"], dropdown)
            R["bias"] = bias(R["F"])
            out[side][lab] = R
            if k == 0:
                prim = R
            print(side, lab, R["means"].round(3).to_dict("records"), R["pick"], R["var_choice"], flush=True)
        write_files(prim["F"], side)
        forest(prim["F"], side)
    write_readme(out, defs, dropdown)
    write_report(out, defs, comp, credit, dropdown, time.perf_counter() - t0)
    write_howto(out, defs)


def plain(F, side):
    p = lambda x: (100 * x).round(3)
    first, second = ("% batting first", "% chasing") if side == "bat" else ("% bowling in 1st innings", "% bowling in chase")
    return pd.DataFrame({
        "Rank (eligible)": F.rank_eligible.astype("Int64"), "Rank (all players)": F.rank_all, "Player": F["name"],
        "Eligible": F.eligible_reason, "Cautious rating (used for rank)": p(F.cautious), "Projected pressure impact (best estimate)": p(F.projected),
        "Likely range low (80%)": p(F.cautious), "Likely range high (80%)": p(F.hi80), "Own-data share (%)": (100 * F.own_share).round(1),
        "Pressure record (strict set, LI-weighted)": p(F.raw_value), "Chance he is above average under pressure (%)": (100 * F.p_above_avg).round(1),
        ("Runs above expected per unit leverage (strict set)" if side == "bat" else "Runs conceded below expected per unit leverage (strict set)"): F.raw_P1.round(4),
        "Overall quality rating A": p(F.A), "Strict pressure balls": F.n_p.astype(int), "Matches": F.matches_all.astype(int),
        "Matches 2025-26": F.matches_2025_26, "Strict pressure balls per match": F.strict_balls_per_match.round(2),
        "% of pressure from IPL": (100 * F.share_IPL).round(1), "% of pressure from internationals": (100 * (F.share_full_member + F.share_associate)).round(1),
        "% pressure in powerplay": (100 * F.share_Powerplay).round(1), "% pressure in middle overs": (100 * F.share_Middle).round(1),
        "% pressure in death overs": (100 * F.share_Death).round(1), first: (100 * F.share_1).round(1), second: (100 * F.share_2).round(1),
        ("Raw pressure strike rate" if side == "bat" else "Raw pressure economy"): (F.raw_pressure_strike_rate if side == "bat" else F.raw_pressure_economy).round(2)})


def write_files(F, side):
    plain(F, side).to_csv(mpath(f"phase4_rankings_{side}_all_players.csv"), index=False)
    t = F.copy(); t.index.name = "player_id"; t.to_csv(mpath(f"phase4_rankings_{side}_all_players_technical.csv"))


def forest(F, side):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = F[F.rank_eligible <= 25].sort_values("rank_eligible").iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 9.2)); y = np.arange(len(t))
    ax.axvline(0, color="#bbbbbb", lw=1, zorder=0)
    ax.hlines(y, 100 * t.cautious, 100 * t.hi80, color="#3987e5", lw=2, zorder=1)
    ax.scatter(100 * t.projected, y, s=40, color="#1c5cab", zorder=2, edgecolor="white", linewidth=1.5, label="projected (best estimate)")
    ax.scatter(100 * t.cautious, y, s=40, marker="|", color="#0d366b", zorder=3, linewidth=2, label="cautious rating (ranking score)")
    ax.set_yticks(y); ax.set_yticklabels([f"{int(r)}. {n}" for r, n in zip(t.rank_eligible, t["name"])], fontsize=8.5)
    ax.set_xlabel("final rating (i*), win-probability points per average ball (80% likely range)")
    ax.set_title(f"Top 25 eligible {'batters' if side == 'bat' else 'bowlers'} - final rating, 2022-26", fontsize=10.5)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", lw=0.8); ax.set_axisbelow(True); ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(REP4, f"phase4_forest_top25_{side}.png"), dpi=140); plt.close(fig)


def write_readme(out, defs, dropdown):
    prim = defs[0][0]
    L = ["# Rankings files", "",
         "`phase4_rankings_bat_all_players.csv`, `phase4_rankings_bowl_all_players.csv`: plain English, one header row. "
         "`*_technical.csv`: all model columns, raw units.", "",
         "**Unit.** Ratings are in win-probability points per average ball: +0.10 means the player's side is projected to "
         "gain on average 0.10 percentage points of win probability per unit of leverage on pressure balls, compared with "
         "a typical player in the same season, competition, innings and phase.", "",
         "**Pressure balls.** Balls with leverage index (LI) >= 1.5 - the expected swing in win probability on that ball "
         "is at least 1.5 times the average ball's. Inside that set each ball counts in proportion to its LI.", "",
         "**Eligible.** Own-data share >= 50% (at most half of the rating comes from the pull towards average), at least 5 "
         "matches in 2025-26" + ("" if dropdown is not None else "; WARNING: `data/unstop_dropdown_players.csv` was not found, so "
                                 "names were NOT checked against the Unstop dropdown") + ".", "",
         "**Columns.** Rank (eligible) - submission top 10 = 1-10; Rank (all players); Eligible (yes / no with reason); Cautious "
         "rating = bottom of the 80% range (ranking score); Projected pressure impact = best estimate; Own-data share; "
         "Pressure record = actual LI-weighted win probability added on strict pressure balls (no shrinkage); runs above / "
         "conceded below expected per unit leverage on strict pressure balls; Overall quality rating A; Strict pressure "
         "balls; Matches; Matches 2025-26; exposure shares by competition, phase and innings; raw pressure strike rate / economy.", ""]
    for side in out:
        R = out[side][prim]
        L.append(f"- {'Batters' if side == 'bat' else 'Bowlers'}: final score = {NAMES[R['pick']]}; uncertainty = {R['var_choice']}.")
    open(mpath("README_rankings.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def write_report(out, defs, comp, credit, dropdown, rt):
    f3 = lambda x: f"{x:.3f}" if isinstance(x, (float, np.floating)) else x
    g4 = lambda x: f"{x:.4g}" if isinstance(x, (float, np.floating)) else x
    prim = defs[0][0]
    L = ["# Phase 4d - strict, LI-weighted pressure definition", "",
         "Run 1 (Phase 4b) - rejected: environment confound (raw runs per ball) + uncertainty bug. Run 2 (Phase 4c) - "
         "rejected: pressure definition too loose (LI >= 1: 40% of balls, 91% of first-innings powerplay balls, each ball "
         "counted equally) and circular target (same loose set). Their reports are kept (`phase4b_summary.md`, "
         "`phase4c_summary.md`).", "",
         "Dropdown list: " + ("applied" if dropdown is not None else "**WARNING - `data/unstop_dropdown_players.csv` not found; names not checked**") + ".",
         "Evaluation players per split: strict-set leverage >= 60 in training and >= 30 in the test part (the 4c thresholds "
         "of 100 / 50 were for the looser set; scaled to keep a comparable player count).", "",
         "## Composition of each pressure set (all valid balls)", ""]
    for lab, (g, frac) in comp.items():
        L += [f"**{lab}** - {100 * frac:.1f}% of all balls.", "", g.map(f3).to_markdown(index=False), ""]
    for side in out:
        role = "Batters" if side == "bat" else "Bowlers"
        L += [f"## {role}", ""]
        for lab, _ in defs:
            R = out[side][lab]; F = R["F"]
            res = R["res"]
            cis = res.assign(ci=[f"{r.spearman:+.3f} [{r.lo95:+.3f}, {r.hi95:+.3f}]" for r in res.itertuples()]).pivot(index="split", columns="candidate", values="ci").rename(columns=NAMES)
            cis.insert(0, "players", res.groupby("split").n_players.first())
            means = R["means"].assign(candidate=R["means"].candidate.map(NAMES))
            diffs = R["diffs"]
            sep = not any((diffs.lo95 > 0) | (diffs.hi95 < 0))
            L += [f"### {lab}", "",
                  f"Target (held-out LI-weighted value on this set) odd/even reliability {R['rel']['target_reliability']:.3f} "
                  f"({R['rel']['n_players']} players) -> noise ceiling {R['rel']['noise_ceiling']:.2f}.", "",
                  cis.reset_index().to_markdown(index=False), "", means.map(f3).to_markdown(index=False), "",
                  diffs.map(f3).to_markdown(index=False), "",
                  ("**The test cannot tell the candidates apart: no paired difference has a CI excluding 0.**" if sep else
                   "At least one paired difference has a CI excluding 0 (see table)."), "",
                  "Ridge per split:", "", R["coefs"].map(g4).to_markdown(index=False), "",
                  f"**Chosen: {NAMES[R['pick']]}** - {R['why']}. Mapping: projected = {R['a']:.5f} + {R['b']:.5f} x score.", ""]
            if R["sts"] is not None:
                L += ["Self-test of the uncertainty (200 repetitions; method test only):", "", R["sts"].reset_index().map(f3).to_markdown(index=False), "",
                      f"Acceptance 0.75-0.85 overall and for < 15 matches -> **{R['var_choice']}** ({R['var_note']}).", ""]
            el = F[F.eligible]
            sh = F.own_share
            near = F[(sh >= 0.45) & (sh <= 0.55)]
            L += [f"Eligibility: {int(F.eligible.sum())} of {len(F)} eligible. Own-data share >= 50% fails for {int((~(sh >= SHARE_MIN)).sum())}; "
                  f"< 5 matches in 2025-26 for {int((F.matches_2025_26 < RECENT_MIN).sum())}. Players with own-data share 45-55% have median "
                  f"{near.matches_all.median() if len(near) else float('nan'):.0f} matches and {near.n_p.median() if len(near) else float('nan'):.0f} strict pressure balls.", ""]
            if lab == prim:
                top = el.sort_values("rank_eligible").head(25)
                tt = pd.DataFrame({"rank (eligible)": top.rank_eligible.astype(int), "rank (all)": top.rank_all, "player": top["name"],
                                   "cautious": (100 * top.cautious).round(3), "projected": (100 * top.projected).round(3),
                                   "80% range": [f"{100*a:+.3f} to {100*b:+.3f}" for a, b in zip(top.cautious, top.hi80)],
                                   "own-data %": (100 * top.own_share).round(0), "pressure record": (100 * top.raw_value).round(3),
                                   "% IPL": (100 * top.share_IPL).round(0), "% PP": (100 * top.share_Powerplay).round(0),
                                   "% mid": (100 * top.share_Middle).round(0), "% death": (100 * top.share_Death).round(0),
                                   "strict balls": top.n_p.astype(int), "per match": top.strict_balls_per_match.round(2),
                                   "matches": top.matches_all.astype(int), "2025-26": top.matches_2025_26})
                bt, rpp, rde, gap = R["bias"]
                flag = [p for p, gv in gap.items() if abs(gv) > 0.20]
                neg = el[(el.rank_eligible <= 10) & ((el.raw_value < 0) | (el.share_IPL == 0))]
                L += ["#### Top 25 eligible (primary definition)", "", tt.to_markdown(index=False), "", f"![forest](phase4_forest_top25_{side}.png)", "",
                      "#### Bias checks", "", bt.map(f3).to_markdown(index=False), "",
                      f"Spearman of final score vs powerplay share {rpp:+.3f}; vs death share {rde:+.3f} (eligible players).", "",
                      ("Top 10 differs from the eligible pool by > 20 points in: " + ", ".join(f"{p} ({100*gap[p]:+.0f} pts)" for p in flag)
                       + ". Why: see the per-player phase shares and strict balls per match in the top-25 table." if flag
                       else "Top 10 is within 20 points of the eligible pool in every phase share."), "",
                      f"Eligible top-10 players with negative pressure record or 0% IPL: {len(neg)}", ""]
                for _, r in neg.iterrows():
                    L.append(f"- {r['name']} (rank {int(r.rank_eligible)}): pressure record {100*r.raw_value:+.3f}, IPL {100*r.share_IPL:.0f}%, "
                             f"score {r.score:+.4f} (raw P1 {r.raw_P1:+.4f} over {int(r.n_p)} strict balls), projected {100*r.projected:+.3f}, "
                             f"A {100*r.A:+.3f}, own-data share {100*r.own_share:.0f}%.")
                L.append("")
        # side by side top 10
        tops = {lab: out[side][lab]["F"] for lab, _ in defs}
        sbs = pd.DataFrame({lab: list(F[F.rank_eligible <= 10].sort_values("rank_eligible")["name"]) + [""] * (10 - int((F.rank_eligible <= 10).sum()))
                            for lab, F in tops.items()}, index=range(1, 11))
        sets = {lab: set(F[F.rank_eligible <= 10].index) for lab, F in tops.items()}
        ov = [f"{prim} vs {lab}: {len(sets[prim] & sets[lab])} / 10" for lab, _ in defs[1:]]
        L += ["### Top 10 under the three definitions", "", sbs.to_markdown(), "", "Overlap: " + "; ".join(ov) + ".", ""]
    L += [f"Runtime {rt:.0f} s.", ""]
    open(os.path.join(REP4, "phase4d_summary.md"), "w", encoding="utf-8").write("\n".join(L))


def write_howto(out, defs):
    prim = defs[0][0]
    Rb, Rw = out["bat"][prim], out["bowl"][prim]
    Fb = Rb["F"]; el = Fb[Fb.eligible].sort_values("rank_eligible").head(25)
    fin = el.sort_values("share_Death", ascending=False).iloc[0]
    Fw = Rw["F"]; exw = Fw[Fw.rank_eligible == 1].iloc[0]
    p = lambda x: f"{100 * x:+.2f}"
    desc = {"P1": "runs above expected (batters) or below expected (bowlers), weighted by importance",
            "P2": "win probability added, weighted by importance", "istar": "overall quality across all balls",
            "R": "a weighted mix of overall quality, runs versus expected and dismissals / wickets versus expected"}
    L = ["# How the ranking works", "",
         "1. **Score every ball's importance.** A win-probability model gives each ball a leverage index (LI): how much the "
         "result could swing on it, with an average ball = 1. **Pressure balls are those with LI of 1.5 or more**, and inside "
         "that set a ball with LI 6 counts six times as much as a ball with LI 1.",
         "2. **Compare each pressure ball with what was expected in that exact situation** (score, balls and wickets left, "
         "season, competition), against the average for that stage, corrected for opponent strength and for matches against "
         "weaker associate teams.",
         "3. **Pick the measure that best predicts future pressure performance**, using a rule fixed in advance and five "
         f"held-out tests. Batters: {desc[Rb['pick']]}. Bowlers: {desc[Rw['pick']]}.",
         "4. **Weigh evidence with caution.** Short records are pulled towards average; the likely range reflects how much "
         "we know about each player.",
         "5. **Rank by the cautious rating** (bottom of the 80% range), among players whose rating is at least half their own "
         "data and who played at least 5 matches in 2025-26.", "",
         f"## Worked example: {fin['name']} (batting rank {int(fin.rank_eligible)}, {100 * fin.share_Death:.0f}% of his pressure in the death overs)", "",
         f"- {int(fin.n_p)} strict pressure balls in {int(fin.matches_all)} matches ({fin.strict_balls_per_match:.2f} per match), "
         f"{int(fin.matches_2025_26)} matches in 2025-26.",
         f"- On those balls, importance-weighted: {fin.raw_P1:+.3f} runs above expected per unit of leverage; win probability added "
         f"{p(fin.raw_value)} points per unit of leverage.",
         f"- Projected pressure impact {p(fin.projected)}, likely range {p(fin.cautious)} to {p(fin.hi80)}; own-data share "
         f"{100 * fin.own_share:.0f}%; cautious rating **{p(fin.cautious)}**.",
         f"- Bowling rank 1: {exw['name']} - {int(exw.n_p)} strict pressure balls, cautious {p(exw.cautious)}.", "",
         "> **What we found**",
         "> - Pressure-specific skill is not detectable; underlying quality carries into pressure.",
         f"> - Best mean out-of-sample correlation with future strict-set pressure value: batters {Rb['means']['mean'].max():.2f} "
         f"(ceiling {Rb['rel']['noise_ceiling']:.2f}), bowlers {Rw['means']['mean'].max():.2f} (ceiling {Rw['rel']['noise_ceiling']:.2f}).",
         "> - Neighbouring ranks overlap: read the ranges, not just the order.", ""]
    open(os.path.join(REP4, "how_the_ranking_works.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
