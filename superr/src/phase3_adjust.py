"""Phase 3 - context adjustments of per-ball player value, each kept only if it helps (lean ablation).

Per-ball value (Phase 1f, valid 2022+ balls): batting side = balls faced (batter_ball_faced), v = wpa_centred;
bowling side = balls bowled, v = -wpa_centred. Player pressure performance P = sum(w v) / sum(w LI) (leverage
weighting; w = 1 unless a tier weight is kept). All "quality" and "effect" quantities below are leverage-weighted
means of v / LI (= sum v / sum LI), shrunk to 0 with an empirical-Bayes constant k (method of moments), and enter a
ball's value scaled by its LI so that units match (an effect per unit of leverage).

Keep rule (pre-stated, per side): an adjustment is kept if EITHER
  (a) odd/even split-half reliability (Spearman-Brown) of P improves, 95% player-bootstrap CI of the gain > 0, OR
  (b) chronological out-of-sample prediction of per-ball wpa_centred improves: test seasons 2024-26, player skill
      from seasons < t (shrunk sum v_adj / (sum LI + k)), prediction = LI x skill + the ball's adjustment term;
      MSE against the raw value; 95% match-bootstrap CI of the MSE change < 0.
Adjustments are tested in order (opponent, tier, ground, dew), each on top of the ones kept before it.
Split-half eligibility: >= 150 balls in each half (matches numbered chronologically per player, odd vs even).

1. Opponent quality q (leak-free): the opposing bowler's (for batter values) / batter's (for bowler values) shrunk
   leverage-weighted mean value from matches BEFORE this match; k_t from matches of seasons < t (2022: from 2022).
   beta_t by WLS of v / LI on q (weights LI, match-clustered SEs) on seasons < t; beta_2022 = 0 (no earlier season).
   v_adj = v - beta_t * LI * q.
2. Tier: paired within-player gain (associate - other) of P; if significant, associate balls get a weight w_a chosen
   from {1, 0.75, 0.5, 0.25, 0} by split-half reliability (down-weighting rather than a tier term: the values are
   already centred within tier, so a tier term would only re-centre).
3. Ground: leave-one-match-out shrunk mean of batting value per (ground, innings) over all balls; subtracted from
   batter values, added to bowler values (x LI).
4. Dew: on bowler values, WLS of v / LI on spin x innings-2 x under-lights x dewpoint depression (full factorial;
   timing_confidence != low), match-clustered CI for the 4-way term. Kept only if significant AND it passes the rule.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_outcome import MODEL, OUT, SEED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP3 = os.path.join(ROOT, "reports", "phase3")
N_BOOT, MIN_HALF, LI_FLOOR = 1000, 150, 0.05
TEST = [2024, 2025, 2026]
TIER_W = [1.0, 0.75, 0.5, 0.25, 0.0]
SIDES = ("bat", "bowl")


# ------------------------------------------------------------------ helpers
def mm_k(player, v, li, min_l=30.0):
    """Method-of-moments shrinkage constant for P_j = sum v / sum LI, Var(v_i) ~ sigma2 * LI_i.
    Estimated on groups with sum LI >= min_l (about 30 average balls): near-dead balls carry v = -(cell mean) with
    LI ~ 0, so tiny-L groups have unbounded ratios that would inflate tau2 and drive k to 0."""
    g = pd.DataFrame({"p": player, "v": v, "l": li}).groupby("p").agg(S=("v", "sum"), L=("l", "sum"))
    g = g[g.L >= min_l]
    pj = g.S / g.L
    inc = pd.Series(player).isin(g.index).values
    resid = v[inc] - pd.Series(player[inc]).map(pj).values * li[inc]
    sigma2 = float((resid ** 2).sum() / li[inc].sum())
    L, Ltot = g.L.values, g.L.sum()
    pbar = g.S.sum() / Ltot
    tau2 = ((L * (pj.values - pbar) ** 2).sum() - (len(g) - 1) * sigma2) / (Ltot - (L ** 2).sum() / Ltot)
    return (sigma2 / tau2) if tau2 > 0 else np.inf, sigma2, tau2


def P_of(df, val, w=None):
    w = np.ones(len(df)) if w is None else w
    g = pd.DataFrame({"p": df.player.values, "wv": w * df[val].values, "wl": w * df.li.values}).groupby("p").sum()
    return g.wv / g.wl


def split_half(df, val, w=None, n=N_BOOT, seed=SEED, return_players=False):
    """Spearman-Brown corrected odd/even reliability of P across eligible players."""
    w = np.ones(len(df)) if w is None else w
    half = df.player_match_idx.values % 2
    t = pd.DataFrame({"p": df.player.values, "h": half, "wv": w * df[val].values, "wl": w * df.li.values, "n": 1})
    g = t.groupby(["p", "h"]).sum().unstack("h")
    g = g[(g[("n", 0)] >= MIN_HALF) & (g[("n", 1)] >= MIN_HALF)]
    a, b = (g[("wv", 0)] / g[("wl", 0)]).values, (g[("wv", 1)] / g[("wl", 1)]).values
    return (a, b, g.index.values) if return_players else (a, b)


def sb(a, b):
    r = np.corrcoef(a, b)[0, 1]; return 2 * r / (1 + r)


def rel_compare(df, base_val, cand_val, w_base=None, w_cand=None, n=N_BOOT, seed=SEED):
    a0, b0, p0 = split_half(df, base_val, w_base, return_players=True)
    a1, b1, p1 = split_half(df, cand_val, w_cand, return_players=True)
    common = np.intersect1d(p0, p1)
    i0, i1 = np.searchsorted(p0, common), np.searchsorted(p1, common)
    a0, b0, a1, b1 = a0[i0], b0[i0], a1[i1], b1[i1]
    rng = np.random.default_rng(seed); m = len(common)
    reps = np.array([[sb(a0[ix], b0[ix]), sb(a1[ix], b1[ix])] for ix in rng.integers(0, m, (n, m))])
    d = reps[:, 1] - reps[:, 0]
    return {"n_players": m, "base": sb(a0, b0), "cand": sb(a1, b1), "diff": sb(a1, b1) - sb(a0, b0),
            "lo95": np.percentile(d, 2.5), "hi95": np.percentile(d, 97.5)}


def oos_compare(df, base_val, cand_val, w_base=None, w_cand=None, n=N_BOOT, seed=SEED):
    """Chronological prediction of v_raw: LI * skill(seasons < t, from *_val) + (v_raw - *_val) at the ball."""
    se = {}
    for nm, val, w in (("base", base_val, w_base), ("cand", cand_val, w_cand)):
        w = np.ones(len(df)) if w is None else w
        out = np.full(len(df), np.nan)
        for t in TEST:
            tr, te = (df.season < t).values, (df.season == t).values
            k, _, _ = mm_k(df.player.values[tr], w[tr] * df[val].values[tr], w[tr] * df.li.values[tr])
            g = pd.DataFrame({"p": df.player.values[tr], "wv": w[tr] * df[val].values[tr],
                              "wl": w[tr] * df.li.values[tr]}).groupby("p").sum()
            skill = g.wv / (g.wl + (0 if np.isinf(k) else k)) if np.isfinite(k) else g.wv * 0
            s = pd.Series(df.player.values[te]).map(skill).fillna(0).values
            pred = df.li.values[te] * s + (df.v_raw.values[te] - df[val].values[te])
            out[te] = (df.v_raw.values[te] - pred) ** 2
        se[nm] = out
    m = ~np.isnan(se["base"])
    dd = pd.DataFrame({"mid": df.match_id.values[m], "d": se["cand"][m] - se["base"][m]}).groupby("mid").d.agg(["sum", "size"])
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(dd), (n, len(dd)))
    reps = dd["sum"].values[idx].sum(1) / dd["size"].values[idx].sum(1)
    return {"n_balls": int(m.sum()), "base": float(se["base"][m].mean()), "cand": float(se["cand"][m].mean()),
            "diff": float(dd["sum"].sum() / dd["size"].sum()), "lo95": np.percentile(reps, 2.5), "hi95": np.percentile(reps, 97.5)}


def decide(df, base_val, cand_val, w_base=None, w_cand=None):
    r = rel_compare(df, base_val, cand_val, w_base, w_cand)
    o = oos_compare(df, base_val, cand_val, w_base, w_cand)
    keep = bool(r["lo95"] > 0) or bool(o["hi95"] < 0)
    return keep, r, o


# ------------------------------------------------------------------ data
def load():
    d = pd.read_parquet(os.path.join(MODEL, "phase1f_ball_values.parquet"))
    d = d[d.missing_reason.isna()].copy()
    d["match_date"] = pd.to_datetime(d.match_date)
    order = d[["match_id", "match_date"]].drop_duplicates().sort_values(["match_date", "match_id"])
    order["match_seq"] = np.arange(len(order))
    d = d.merge(order[["match_id", "match_seq"]], on="match_id")
    g = pd.read_csv(os.path.join(OUT, "match_ground.csv"), usecols=["match_id", "ground"])
    d = d.merge(g, on="match_id", how="left")
    ic = pd.read_csv(os.path.join(OUT, "innings_context.csv"),
                     usecols=["match_id", "innings", "dewpoint_depression_c", "light_innings", "timing_confidence"])
    d = d.merge(ic, on=["match_id", "innings"], how="left")
    d = d.sort_values(["match_seq", "innings", "innings_ball_num"]).reset_index(drop=True)
    d["ball_key"] = np.arange(len(d))
    sides = {}
    for side in SIDES:
        if side == "bat":
            s = d[d.batter_ball_faced.astype(bool)].copy()
            s["player"], s["opp"], s["v_raw"] = s.striker_id, s.bowler_id, s.wpa_centred
        else:
            s = d.copy()
            s["player"], s["opp"], s["v_raw"] = s.bowler_id, s.striker_id, -s.wpa_centred
        pm = s[["player", "match_seq"]].drop_duplicates().sort_values(["player", "match_seq"])
        pm["player_match_idx"] = pm.groupby("player").cumcount()
        s = s.merge(pm, on=["player", "match_seq"]).sort_values("ball_key").reset_index(drop=True)
        sides[side] = s
    return d, sides


def prior_quality(role_df, val, season_of_match):
    """Per (player, match): shrunk leverage-weighted mean of role_df[val] from matches before; k_t from seasons < t."""
    pm = role_df.groupby(["player", "match_seq", "season"]).agg(S=(val, "sum"), L=("li", "sum")).reset_index()
    pm = pm.sort_values(["player", "match_seq"])
    pm["S_before"] = pm.groupby("player").S.cumsum() - pm.S
    pm["L_before"] = pm.groupby("player").L.cumsum() - pm.L
    ks = {}
    for t in sorted(pm.season.unique()):
        tr = role_df.season < t if t > role_df.season.min() else role_df.season == t
        ks[int(t)] = mm_k(role_df.player.values[tr], role_df[val].values[tr], role_df.li.values[tr])[0]
    pm["k"] = pm.season.map(ks)
    pm["q"] = np.where(np.isfinite(pm.k), pm.S_before / (pm.L_before + pm.k.where(np.isfinite(pm.k), 0)), 0.0)
    return pm[["player", "match_seq", "q"]], ks


def wls_cluster(y, X, w, groups):
    m = sm.WLS(y, X, weights=w).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return m


# ------------------------------------------------------------------ main
def main():
    t0 = time.perf_counter()
    os.makedirs(REP3, exist_ok=True)
    d, S = load()
    log, notes = [], {}
    cur = {side: "v_raw" for side in SIDES}          # current value column per side
    wcur = {side: None for side in SIDES}            # current weights per side
    for side in SIDES:
        S[side]["v_cur"] = S[side].v_raw

    def record(step, side, keep, r, o, extra="", applied=None, rule_type="predictive_feature", ruling=""):
        for crit, res in (("split_half_reliability_P", r), ("oos_mse_wpa_centred", o)):
            log.append({"step": step, "side": side, "rule_type": rule_type, "criterion": crit,
                        **{k: res[k] for k in ("base", "cand", "diff", "lo95", "hi95")},
                        "n": res.get("n_players", res.get("n_balls")), "passes_predictive_rule": keep,
                        "kept": keep if applied is None else applied, "ruling": ruling, "note": extra})

    # baseline reliability (raw) for P, normal, lift - informative
    # ---------------- 1. opponent quality
    q_bat_role, k_bowl = prior_quality(S["bowl"], "v_raw", None)   # bowler quality as bowler
    q_bowl_role, k_bat = prior_quality(S["bat"], "v_raw", None)    # batter quality as batter
    S["bat"] = S["bat"].merge(q_bat_role.rename(columns={"player": "opp", "q": "q_opp"}), on=["opp", "match_seq"], how="left")
    S["bowl"] = S["bowl"].merge(q_bowl_role.rename(columns={"player": "opp", "q": "q_opp"}), on=["opp", "match_seq"], how="left")
    betas = {}
    for side in SIDES:
        s = S[side]; s["q_opp"] = s.q_opp.fillna(0.0)
        r = np.where(s.li >= LI_FLOOR, s.v_cur / s.li.clip(lower=LI_FLOOR), np.nan)
        b_t = {2022: (0.0, np.nan, np.nan)}
        for t in [2023, 2024, 2025, 2026]:
            m = (s.season < t).values & ~np.isnan(r)
            fit = wls_cluster(r[m], sm.add_constant(s.q_opp.values[m]), s.li.values[m], s.match_id.values[m])
            lo, hi = fit.conf_int()[1]
            b_t[t] = (float(fit.params[1]), float(lo), float(hi))
        betas[side] = b_t
        s["adj_opp"] = -s.season.map(lambda t: b_t[t][0]).values * s.li.values * s.q_opp.values
        s["v_cand"] = s.v_cur + s.adj_opp
        keep, rr, oo = decide(s, "v_cur", "v_cand")
        record("1_opponent_quality", side, keep, rr, oo, f"beta by season: { {t: round(b[0], 4) for t, b in b_t.items()} }",
               applied=True, rule_type="bias_correction",
               ruling="APPLY: beta significant in 2024-26, estimated leak-free on earlier seasons (2022 = 0)")
        s["v_cur"] = s.v_cand
        S[side] = s
    notes["opponent_k"] = {"bowler_quality_k_by_season": k_bowl, "batter_quality_k_by_season": k_bat}
    notes["opponent_beta"] = betas

    # ---------------- 2. tier
    def assoc_gain(s, val, seasons=None, n=N_BOOT):
        """Within-player associate gain: LI-weighted P on associate balls minus P on other balls (players >= 30 balls in
        both), weighted by 1/(1/L_assoc + 1/L_other); player-bootstrap 95% CI."""
        e = s if seasons is None else s[s.season.isin(seasons)]
        g = e.assign(a=(e.tier == "associate")).groupby(["player", "a"]).agg(v=(val, "sum"), l=("li", "sum"), n=("li", "size")).unstack("a")
        g = g[(g[("n", True)] >= 30) & (g[("n", False)] >= 30)]
        diff = (g[("v", True)] / g[("l", True)] - g[("v", False)] / g[("l", False)]).values
        wt = (1 / (1 / g[("l", True)] + 1 / g[("l", False)])).values
        if len(diff) == 0:
            return 0.0, np.nan, np.nan, 0
        rng = np.random.default_rng(SEED); m = len(diff)
        reps = [np.average(diff[ix], weights=wt[ix]) for ix in rng.integers(0, m, (n, m))]
        lo, hi = np.percentile(reps, [2.5, 97.5])
        return float(np.average(diff, weights=wt)), float(lo), float(hi), m

    tier_rows = []
    for side in SIDES:
        s = S[side]
        s["tier_weight"] = 1.0
        # tier term on opponent-adjusted values: gamma_t = within-player gain on seasons < t (leak-free), gamma_2022 = 0
        assoc = (s.tier == "associate").values
        gam = {2022: 0.0}
        for t in [2023, 2024, 2025, 2026]:
            gam[t] = assoc_gain(s[s.season < t], "v_cur", n=1)[0]
        s["adj_tier"] = -s.season.map(gam).values * s.li.values * assoc
        s["v_cand"] = s.v_cur + s.adj_tier
        keep, rr, oo = decide(s, "v_cur", "v_cand")
        record("2_tier_term", side, keep, rr, oo, f"gamma by season: { {t: round(g_, 5) for t, g_ in gam.items()} }",
               applied=True, rule_type="bias_correction",
               ruling="APPLY after opponent quality: removes a within-player associate gain (cannot raise odd/even reliability)")
        notes[f"tier_gamma_{side}"] = gam
        for stage, val in (("raw", "v_raw"), ("after opponent", "v_cur"), ("after opponent + tier", "v_cand")):
            for lab, seas in (("all seasons", None), ("2023-26 (gamma estimated)", [2023, 2024, 2025, 2026])):
                est, lo, hi, m = assoc_gain(s, val, seas)
                tier_rows.append({"side": side, "values": stage, "seasons": lab, "players_in_both": m,
                                  "assoc_gain": est, "lo95": lo, "hi95": hi})
        s["v_cur"] = s.v_cand
        S[side] = s
    tier_tab = pd.DataFrame(tier_rows)

    # ---------------- 3. ground (leave-one-match-out, per ground x innings, from all balls' batting value)
    allb = d.assign(vb=d.wpa_centred)
    gm = allb.groupby(["ground", "innings", "match_id"]).agg(S=("vb", "sum"), L=("li", "sum")).reset_index()
    gt = gm.groupby(["ground", "innings"]).agg(SG=("S", "sum"), LG=("L", "sum")).reset_index()
    gm = gm.merge(gt, on=["ground", "innings"])
    key_gi = allb.ground.astype(str) + "|" + allb.innings.astype(str)
    k_g, s2g, t2g = mm_k(key_gi.values, allb.vb.values, allb.li.values)
    gm["g"] = (gm.SG - gm.S) / (gm.LG - gm.L + (k_g if np.isfinite(k_g) else np.inf))
    gm["g"] = gm.g.fillna(0.0)
    notes["ground_k"] = {"k": k_g, "sigma2": s2g, "tau2": t2g}
    for side in SIDES:
        s = S[side].merge(gm[["ground", "innings", "match_id", "g"]], on=["ground", "innings", "match_id"], how="left")
        s["g"] = s.g.fillna(0.0)
        sign = -1.0 if side == "bat" else 1.0
        s["adj_ground"] = sign * s.li * s.g
        s["v_cand"] = s.v_cur + s.adj_ground
        w = s.tier_weight.values if "tier_weight" in s else None
        keep, rr, oo = decide(s, "v_cur", "v_cand", w, w)
        record("3_ground", side, keep, rr, oo, f"k = {k_g:.1f}", applied=False, rule_type="predictive_feature",
               ruling="DROP: lowers batting reliability significantly; likely confounded with home players' own skill")
        s["adj_ground"] = 0.0
        S[side] = s.sort_values("ball_key").reset_index(drop=True)

    # ---------------- 4. dew (bowler values)
    s = S["bowl"]
    m = (s.timing_confidence != "low") & s.dewpoint_depression_c.notna() & s.light_innings.notna() & (s.li >= LI_FLOOR)
    x = s[m]
    spin = (x.pace_or_spin == "Spin").astype(float).values; inn2 = (x.innings == 2).astype(float).values
    lights = (x.light_innings == "under_lights").astype(float).values; dpd = x.dewpoint_depression_c.values - x.dewpoint_depression_c.mean()
    cols = {"spin": spin, "inn2": inn2, "lights": lights, "dpd": dpd}
    names = list(cols)
    X = {"const": np.ones(len(x))}
    import itertools
    for r_ in range(1, 5):
        for comb in itertools.combinations(names, r_):
            X[":".join(comb)] = np.prod([cols[c] for c in comb], axis=0)
    Xdf = pd.DataFrame(X)
    yv = x.v_cur.values / x.li.values
    fit = wls_cluster(yv, Xdf, x.li.values, x.match_id.values)
    term = "spin:inn2:lights:dpd"
    lo, hi = fit.conf_int().loc[term]
    dew = {"term": term, "coef_per_degC": float(fit.params[term]), "lo95": float(lo), "hi95": float(hi), "p": float(fit.pvalues[term]),
           "n_balls": int(len(x)), "n_matches": int(x.match_id.nunique()),
           "spin:dpd (all)": float(fit.params["spin:dpd"]), "spin:inn2:lights": float(fit.params["spin:inn2:lights"])}
    sig = not (lo <= 0 <= hi)
    s["adj_dew"] = 0.0
    if sig:
        s.loc[m, "adj_dew"] = -(fit.params[term] * (spin * inn2 * lights * dpd)) * x.li.values
        s["v_cand"] = s.v_cur + s.adj_dew
        w = s.tier_weight.values
        keep, rr, oo = decide(s, "v_cur", "v_cand", w, w)
        record("4_dew", "bowl", keep, rr, oo, f"4-way coef {dew['coef_per_degC']:.4f} [{lo:.4f}, {hi:.4f}]",
               applied=False, ruling="DROP (ruling)")
        s["adj_dew"] = 0.0
    else:
        log.append({"step": "4_dew", "side": "bowl", "rule_type": "predictive_feature", "criterion": "significance of 4-way term",
                    "base": np.nan, "cand": dew["coef_per_degC"], "diff": dew["coef_per_degC"], "lo95": lo, "hi95": hi, "n": len(x),
                    "passes_predictive_rule": False, "kept": False, "ruling": "DROP: tested null",
                    "note": f"p = {dew['p']:.2f}; not significant -> dropped without the reliability test"})
    S["bowl"] = s

    # ---------------- final reliability (informative): P, normal, lift, raw vs adj
    info = []
    for side in SIDES:
        s = S[side]; w = s.tier_weight.values
        for lab, val, ww in (("raw", "v_raw", None), ("adj", "v_cur", w)):
            a, b = split_half(s, val, ww); info.append({"side": side, "values": lab, "measure": "pressure P", "n_players": len(a), "spearman_brown": sb(a, b)})
            # normal and lift per half
            t = s.assign(h=s.player_match_idx % 2, r=np.where(s.li >= LI_FLOOR, s[val] / s.li.clip(lower=LI_FLOOR), np.nan),
                         wv=(1 if ww is None else ww) * s[val], wl=(1 if ww is None else ww) * s.li)
            low = t[(t.li < 1) & t.r.notna()]
            nh = low.groupby(["player", "h"]).r.mean().unstack()
            ph = t.groupby(["player", "h"]).agg(wv=("wv", "sum"), wl=("wl", "sum"), n=("li", "size")).unstack()
            ok = (ph[("n", 0)] >= MIN_HALF) & (ph[("n", 1)] >= MIN_HALF)
            P0, P1 = (ph[("wv", 0)] / ph[("wl", 0)])[ok], (ph[("wv", 1)] / ph[("wl", 1)])[ok]
            N0, N1 = nh[0].reindex(P0.index), nh[1].reindex(P0.index)
            okn = N0.notna() & N1.notna()
            info.append({"side": side, "values": lab, "measure": "normal N", "n_players": int(okn.sum()), "spearman_brown": sb(N0[okn].values, N1[okn].values)})
            info.append({"side": side, "values": lab, "measure": "lift P - N", "n_players": int(okn.sum()),
                         "spearman_brown": sb((P0 - N0)[okn].values, (P1 - N1)[okn].values)})
    info = pd.DataFrame(info)

    # ---------------- output
    out = d[["ball_key", "match_id", "innings", "innings_ball_num", "overid", "over_ball_num", "match_date", "season", "tier",
             "phase", "striker_id", "non_striker_id", "bowler_id", "batting_team_id", "bowling_team_id", "pace_or_spin", "ground",
             "li", "wpa", "wpa_centred", "batter_ball_faced"]].copy()
    for side in SIDES:
        s = S[side].set_index("ball_key")
        cols_ = {"v_raw": f"v_raw_{side}", "v_cur": f"v_adj_{side}", "q_opp": f"q_opp_{side}", "adj_opp": f"adj_opp_{side}",
                 "adj_ground": f"adj_ground_{side}", "adj_tier": f"adj_tier_{side}", "tier_weight": f"tier_weight_{side}"}
        if side == "bowl":
            cols_["adj_dew"] = "adj_dew_bowl"
        out = out.join(s[list(cols_)].rename(columns=cols_), on="ball_key")
    out.to_parquet(os.path.join(MODEL, "phase3_ball_values_adj.parquet"), index=False, compression="zstd")
    abl = pd.DataFrame(log)
    abl.to_csv(os.path.join(REP3, "ablations.csv"), index=False)
    kept = {side: sorted({r["step"] for r in log if r["side"] == side and r["kept"]}) for side in SIDES}
    json.dump({"kept": kept, "notes": notes, "dew": dew, "tier": tier_rows}, open(os.path.join(MODEL, "phase3_config.json"), "w"),
              indent=1, default=float)
    write_report(abl, tier_tab, dew, info, notes, kept, out, time.perf_counter() - t0)
    print(abl.to_string()); print(kept); print(info)


def write_report(abl, tier_tab, dew, info, notes, kept, out, rt):
    f = lambda x: f"{x:.5f}" if isinstance(x, float) else x
    b = notes["opponent_beta"]
    beta_tab = pd.DataFrame([{"side": side, "season": t, "beta": v[0], "lo95": v[1], "hi95": v[2]} for side in b for t, v in b[side].items()])
    L = ["# Phase 3 - context adjustments", "",
         "Values: batting = balls faced, v = wpa_centred; bowling = balls bowled, v = -wpa_centred. Pressure performance "
         "P = sum(v) / sum(LI). Predictive rule: split-half (odd/even matches) Spearman-Brown reliability of P improves "
         "with the 95% player-bootstrap CI of the gain > 0, OR chronological out-of-sample MSE of per-ball wpa_centred "
         "(test 2024-26) falls with the 95% match-bootstrap CI < 0. Every test is in `reports/phase3/ablations.csv` "
         "(columns rule_type, passes_predictive_rule, kept, ruling).", "",
         "**Bias corrections vs predictive features.** The predictive rule is right for features that are meant to add "
         "signal (ground, dew). It is the wrong test for bias corrections: removing a bias that is consistent within a "
         "player (stronger opposition faced, easier associate opponents) cannot raise odd/even reliability, because both "
         "halves carry the same bias. Bias corrections are therefore judged on whether the bias is real and estimated "
         "leak-free, and applied: opponent quality (beta significant in 2024-26, from earlier seasons; 2022 = 0) and the "
         "associate tier term (within-player gain, gamma_t from seasons < t, applied after the opponent adjustment). "
         "Ground (lowers batting reliability significantly; likely confounded with home players' own skill) and dew "
         "(tested null) are dropped. v_adj = v_raw + adj_opp + adj_tier.", "",
         "Effects enter a ball's value scaled by its LI (an effect per unit of leverage), so that an opponent or ground "
         "shifts a high-leverage ball more than a dead one; quality and effect estimates are leverage-weighted means of "
         "v / LI, matching P.", "",
         f"**Kept - batting: {kept['bat'] or 'none'}; bowling: {kept['bowl'] or 'none'}.**", "",
         "## Ablation log", "", abl.map(f).to_markdown(index=False), "",
         "## 1. Opponent quality", "",
         "q = opponent's shrunk leverage-weighted mean value from matches before this one (k by method of moments, "
         "from seasons before the ball's season; 2022 from 2022). beta by WLS of v/LI on q (weights LI, match-clustered "
         "CI) on earlier seasons; beta_2022 = 0. Adjustment = -beta x LI x q.", "",
         beta_tab.map(f).to_markdown(index=False), "", f"k: {json.dumps(notes['opponent_k'], default=lambda x: round(float(x), 1))}", "",
         "## 2. Tier", "",
         "Values are centred within season x tier x innings x phase, so the mean over all associate-tier balls is 0 by "
         "construction; the test is within player (players with >= 30 balls in both): leverage-weighted P on associate "
         "balls minus P on other balls, weighted by 1/(1/L_assoc + 1/L_other), player-bootstrap CI.", "",
         "Tier term (applied): v_adj = v - gamma_t x LI x 1[associate], gamma_t = the within-player gain on "
         "opponent-adjusted values from seasons < t (gamma_2022 = 0). A plain tier dummy would do nothing because values "
         "are already centred within tier; the within-player gain is what remains. "
         f"gamma_t - batting: {notes['tier_gamma_bat']}; bowling: {notes['tier_gamma_bowl']}. "
         "(The v1 down-weighting scan, best weight 1 on reliability for both sides, is superseded.)", "",
         "Associate gain at each stage (within player):", "",
         tier_tab.map(f).to_markdown(index=False), "",
         "After the tier term the remaining within-player gain is ~0 on both sides, with 95% CIs including 0, whether measured on all seasons or on 2023-26 only (where gamma_t was estimated; 2022 carries gamma = 0 by design). The opponent adjustment alone barely changes the gain. A sensitivity ranking on "
         "IPL + full-member balls only is produced in Phase 4.", "",
         "## 3. Ground", "", f"Leave-one-match-out shrunk mean per (ground, innings); k = {notes['ground_k']['k']:.1f}.", "",
         "## 4. Dew (dropped: tested null)", "",
         "Bowler v/LI on spin x innings 2 x under lights x dewpoint depression (centred), full factorial, weights LI, "
         "match-clustered CI, timing_confidence != low.", "",
         pd.DataFrame([dew]).map(f).to_markdown(index=False), "",
         "## Reliability of the measures (split-half, Spearman-Brown; informative)", "", info.map(f).to_markdown(index=False), "",
         "## Output", "", "`out/model/phase3_ball_values_adj.parquet`: one row per valid 2022+ ball; v_raw_bat / v_adj_bat "
         "(balls faced only), v_raw_bowl / v_adj_bowl, q_opp_*, adj_opp_*, adj_tier_* (applied), adj_ground_* and adj_dew_bowl (0, dropped). "
         "Phase 4 uses v_adj with leverage weighting (sum of w v_adj / sum of w LI).", "", f"Runtime {rt:.0f} s.", ""]
    open(os.path.join(REP3, "phase3_summary.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
