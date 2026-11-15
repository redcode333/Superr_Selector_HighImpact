"""Phase 4 - player pressure ratings (empirical Bayes, closed form). Batters on balls faced, bowlers on balls bowled,
values v_adj from Phase 3 (opponent + tier corrected), leverage-weighted.

Per player-match m of player j: H = sum v over balls with LI >= 1, LH = sum LI over them; L, LL likewise for LI < 1.
  H_jm ~ N(LH_jm (theta_j + delta_j), sH^2 LH_jm),   L_jm ~ N(LL_jm theta_j, sL^2 LL_jm),
  theta_j ~ N(0, tau_theta^2), delta_j ~ N(0, tau_delta^2), independent.
Player totals are sufficient: y1 = H_j / LH_j ~ N(theta + delta, sH^2 / LH_j), y2 = L_j / LL_j ~ N(theta, sL^2 / LL_j).
Variance components:
  sH^2 = sum_jm (H_jm - LH_jm y1_j)^2 / sum_jm (LH_jm - LH_jm^2 / LH_j)   (player-match residuals: the match is the
         unit, so within-match correlation of balls is absorbed; players with >= 2 matches), same for sL^2;
  tau_theta^2 = sum_j (LL_j y2_j^2 - sL^2) / sum_j LL_j,  tau_theta^2 + tau_delta^2 = sum_j (LH_j y1_j^2 - sH^2) / sum_j LH_j
         (method of moments on players with LH_j >= 20 and LL_j >= 20; near-dead balls make tiny-LL ratios unstable).
  Bootstrap CIs: 1000 resamples of players. Cross-check: tau_theta^2 also = E[w y1 y2] (covariance), reported.
Posterior: exact bivariate Normal for (theta, delta). pressure = theta + delta, normal = theta, lift = delta.
Units in the report: x100 = win-probability percentage points per average-leverage ball (LI = 1).
Ranking score = posterior 10th percentile of pressure. Eligibility: sum LH >= 150 and >= 15 matches (sensitivity:
sum LH >= 100 and >= 200). Tie-break when 80% intervals overlap heavily (p10 within 5% of the median 80% interval
width): higher lift mean, then larger sum LH.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_outcome import MODEL, SEED
from variant import vdir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP4 = vdir(os.path.join(ROOT, "reports", "phase4"))
N_BOOT, Z80 = 1000, norm.ppf(0.9)
ELIG = {"main": (150, 15), "lh100": (100, 15), "lh200": (200, 15)}
MIN_TAU = 20.0
PHASES = ["Powerplay", "Middle", "Death"]
TIERS = ["IPL", "full_member", "associate"]


def load():
    a = pd.read_parquet(os.path.join(MODEL, "phase3_ball_values_adj.parquet"))
    f = pd.read_parquet(os.path.join(MODEL, "phase1f_ball_values.parquet"),
                        columns=["match_id", "innings", "innings_ball_num", "striker_name", "bowler_name", "bat_runs",
                                 "bowler_runs_conceded", "is_legal", "is_wide", "is_noball", "missing_reason"])
    f = f[f.missing_reason.isna()].drop(columns="missing_reason")
    a = a.merge(f, on=["match_id", "innings", "innings_ball_num"], how="left", validate="1:1")
    sides = {}
    for side in ("bat", "bowl"):
        if side == "bat":
            s = a[a.v_adj_bat.notna()].copy()
            s["player"], s["name"], s["v"], s["v_raw"] = s.striker_id, s.striker_name, s.v_adj_bat, s.v_raw_bat
        else:
            s = a.copy()
            s["player"], s["name"], s["v"], s["v_raw"] = s.bowler_id, s.bowler_name, s.v_adj_bowl, s.v_raw_bowl
        s["hi"] = s.li >= 1
        sides[side] = s
    return sides


def player_match(s):
    s = s.assign(Hv=np.where(s.hi, s.v, 0.0), LHv=np.where(s.hi, s.li, 0.0), Lv=np.where(~s.hi, s.v, 0.0), LLv=np.where(~s.hi, s.li, 0.0))
    return s.groupby(["player", "match_id"]).agg(H=("Hv", "sum"), LH=("LHv", "sum"), L=("Lv", "sum"), LL=("LLv", "sum")).reset_index()


def player_stats(pm):
    """Per-player sufficient statistics for the variance-component estimators."""
    g = pm.groupby("player").agg(H=("H", "sum"), LH=("LH", "sum"), L=("L", "sum"), LL=("LL", "sum"), M=("match_id", "size"))
    pm = pm.join(g[["H", "LH", "L", "LL"]].rename(columns=lambda c: c + "_j"), on="player")
    y1 = np.where(pm.LH_j > 0, pm.H_j / pm.LH_j.where(pm.LH_j > 0, 1), 0.0)
    y2 = np.where(pm.LL_j > 0, pm.L_j / pm.LL_j.where(pm.LL_j > 0, 1), 0.0)
    pm["numH"] = (pm.H - pm.LH * y1) ** 2
    pm["denH"] = np.where(pm.LH_j > 0, pm.LH - pm.LH ** 2 / pm.LH_j.where(pm.LH_j > 0, 1), 0.0)
    pm["numL"] = (pm.L - pm.LL * y2) ** 2
    pm["denL"] = np.where(pm.LL_j > 0, pm.LL - pm.LL ** 2 / pm.LL_j.where(pm.LL_j > 0, 1), 0.0)
    r = pm.groupby("player")[["numH", "denH", "numL", "denL"]].sum()
    g = g.join(r)
    g["y1"] = np.where(g.LH > 0, g.H / g.LH.where(g.LH > 0, 1), np.nan)
    g["y2"] = np.where(g.LL > 0, g.L / g.LL.where(g.LL > 0, 1), np.nan)
    return g


def estimate(g, w=None):
    """Variance components from player stats; w = bootstrap multiplicities (None = 1)."""
    w = np.ones(len(g)) if w is None else w
    m2 = (g.M.values >= 2)
    sH2 = (w * g.numH.values * m2).sum() / (w * g.denH.values * m2).sum()
    sL2 = (w * g.numL.values * m2).sum() / (w * g.denL.values * m2).sum()
    e = (g.LH.values >= MIN_TAU) & (g.LL.values >= MIN_TAU)
    LH, LL, y1, y2 = g.LH.values[e], g.LL.values[e], g.y1.values[e], g.y2.values[e]
    we = w[e]
    t_th = (we * (LL * y2 ** 2 - sL2)).sum() / (we * LL).sum()
    t_sum = (we * (LH * y1 ** 2 - sH2)).sum() / (we * LH).sum()
    hm = 1 / (1 / LH + 1 / LL)
    t_cov = (we * hm * y1 * y2).sum() / (we * hm).sum()
    return {"sH2": sH2, "sL2": sL2, "tau_theta2": t_th, "tau_delta2": t_sum - t_th, "tau_sum2": t_sum, "tau_theta2_cov": t_cov}


def posterior(g, vc):
    """Exact bivariate Normal posterior of (theta, delta) per player."""
    tt, td = max(vc["tau_theta2"], 1e-12), max(vc["tau_delta2"], 1e-12)
    a1 = np.where(g.LH > 0, g.LH / vc["sH2"], 0.0)      # precision weight of y1 (theta + delta)
    a2 = np.where(g.LL > 0, g.LL / vc["sL2"], 0.0)      # precision weight of y2 (theta)
    y1, y2 = np.nan_to_num(g.y1.values), np.nan_to_num(g.y2.values)
    # precision matrix: [[1/tt + a1 + a2, a1], [a1, 1/td + a1]]; rhs: [a1 y1 + a2 y2, a1 y1]
    P11, P12, P22 = 1 / tt + a1 + a2, a1, 1 / td + a1
    det = P11 * P22 - P12 ** 2
    S11, S12, S22 = P22 / det, -P12 / det, P11 / det
    b1, b2 = a1 * y1 + a2 * y2, a1 * y1
    th, de = S11 * b1 + S12 * b2, S12 * b1 + S22 * b2
    out = pd.DataFrame(index=g.index)
    out["theta"], out["delta"] = th, de
    out["sd_theta"], out["sd_delta"] = np.sqrt(S11), np.sqrt(S22)
    out["pressure"] = th + de
    out["sd_pressure"] = np.sqrt(S11 + S22 + 2 * S12)
    out["corr_theta_delta"] = S12 / np.sqrt(S11 * S22)
    return out


def profile(s):
    """Share of LH (LI >= 1 leverage) by tier, phase, innings; matches; raw rates."""
    h = s[s.hi]
    lh = h.groupby("player").li.sum()
    out = pd.DataFrame({"sum_LH": lh})
    for col, cats, pre in (("tier", TIERS, "share_LH_"), ("phase", PHASES, "share_LH_"), ("innings", [1, 2], "share_LH_inn")):
        t = h.groupby(["player", col]).li.sum().unstack(col).reindex(columns=cats).fillna(0)
        for c in cats:
            out[f"{pre}{c}"] = t[c] / lh
    out["matches"] = s.groupby("player").match_id.nunique()
    out["name"] = s.groupby("player").name.agg(lambda x: x.mode().iat[0] if len(x.mode()) else x.iat[0])
    out["total_wpa"] = s.groupby("player").v_raw.sum()
    return out


def raw_rates(s, side):
    h = s[s.hi]
    if side == "bat":
        return (100 * h.groupby("player").bat_runs.sum() / h.groupby("player").size()).rename("raw_pressure_strike_rate")
    legal = h.groupby("player").is_legal.sum()
    return (6 * h.groupby("player").bowler_runs_conceded.sum() / legal.where(legal > 0)).rename("raw_pressure_economy")


def rank(df, lh_min, m_min, score="p10_pressure"):
    """Sort by score; adjacent players whose scores differ by < tol (5% of the median 80% interval width) are
    re-ordered by higher lift mean, then larger sum LH (one bubble pass, so no grid-boundary artifacts)."""
    e = df[(df.sum_LH >= lh_min) & (df.matches >= m_min)].sort_values(score, ascending=False).copy()
    tol = 0.05 * (2 * Z80 * e.sd_pressure).median()
    idx = list(e.index); sc = e[score].to_dict(); key = {i: (round(e.at[i, "lift"], 12), e.at[i, "sum_LH"]) for i in idx}
    for i in range(len(idx) - 1):
        a, b = idx[i], idx[i + 1]
        if sc[a] - sc[b] < tol and key[b] > key[a]:
            idx[i], idx[i + 1] = b, a
    e = e.loc[idx]
    e["rank"] = np.arange(1, len(e) + 1)
    return e, tol


def rate_side(s, side, boot=True):
    pm = player_match(s)
    g = player_stats(pm)
    vc = estimate(g)
    ci = {}
    if boot:
        rng = np.random.default_rng(SEED)
        reps = [estimate(g, np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g)).astype(float)) for _ in range(N_BOOT)]
        R = pd.DataFrame(reps)
        ci = {k: (float(R[k].quantile(.025)), float(R[k].quantile(.975))) for k in R}
    post = posterior(g, vc)
    df = post.join(g[["H", "LH", "L", "LL", "M", "y1", "y2"]]).join(profile(s)).join(raw_rates(s, side))
    df["p10_pressure"] = df.pressure - Z80 * df.sd_pressure
    df["pressure_lo80"], df["pressure_hi80"] = df.p10_pressure, df.pressure + Z80 * df.sd_pressure
    df["normal"] = df.theta
    df["lift"], df["lift_lo80"], df["lift_hi80"] = df.delta, df.delta - Z80 * df.sd_delta, df.delta + Z80 * df.sd_delta
    if vc["tau_delta2"] <= 0:                      # no detectable between-player lift: delta is exactly 0
        df[["lift", "lift_lo80", "lift_hi80", "delta"]] = 0.0
    df["P_pressure_gt_0"] = norm.cdf(df.pressure / df.sd_pressure)
    df["raw_H_per_LH"], df["raw_L_per_LL"] = df.y1, df.y2
    # comparison only: pressure balls alone (LI >= 1), shrunk with prior variance tau_sum2
    k = vc["sH2"] / max(vc["tau_sum2"], 1e-12)
    df["pressure_H_only"] = df.H / (df.LH + k)
    df["sd_H_only"] = np.sqrt(1 / (df.LH / vc["sH2"] + 1 / max(vc["tau_sum2"], 1e-12)))
    df["p10_H_only"] = df.pressure_H_only - Z80 * df.sd_H_only
    return df, vc, ci


def equal_phase_pressure(s, vc):
    """Equal-phase-weighted pressure: mean over phases of the shrunk per-phase H/LH (k = sH2 / tau_sum2)."""
    k = vc["sH2"] / max(vc["tau_sum2"], 1e-12)
    h = s[s.hi]
    t = h.groupby(["player", "phase"]).agg(v=("v", "sum"), l=("li", "sum")).reset_index()
    t["p"] = t.v / (t.l + k)
    return t.pivot(index="player", columns="phase", values="p").reindex(columns=PHASES).mean(axis=1, skipna=True)


def forest(top, side, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = top.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 9))
    y = np.arange(len(t))
    ax.axvline(0, color="#bbbbbb", lw=1, zorder=0)
    ax.hlines(y, 100 * t.pressure_lo80, 100 * t.pressure_hi80, color="#3987e5", lw=2, zorder=1)
    ax.scatter(100 * t.pressure, y, s=40, color="#1c5cab", zorder=2, edgecolor="white", linewidth=1.5, label="posterior mean")
    ax.scatter(100 * t.p10_pressure, y, s=40, marker="|", color="#0d366b", zorder=3, linewidth=2, label="10th percentile (ranking score)")
    ax.set_yticks(y); ax.set_yticklabels([f"{int(r)}. {n}" for r, n in zip(t["rank_main"], t["name"])], fontsize=8.5)
    ax.set_xlabel("pressure performance, win-probability points per average-leverage ball (80% interval)")
    role = "batters (balls faced)" if side == "bat" else "bowlers (balls bowled)"
    ax.set_title(f"Top 25 {role} by pressure rating, 2022-26", fontsize=11)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    t0 = time.perf_counter()
    os.makedirs(REP4, exist_ok=True)
    S = load()
    res = {}
    for side in ("bat", "bowl"):
        s = S[side]
        df, vc, ci = rate_side(s, side)
        main_rank, tol = rank(df, *ELIG["main"])
        df["rank_main"] = main_rank["rank"]
        for k in ("lh100", "lh200"):
            df[f"rank_{k}"] = rank(df, *ELIG[k])[0]["rank"]
        # (b) raw H/LH ranking
        raw = df[(df.sum_LH >= ELIG["main"][0]) & (df.matches >= ELIG["main"][1])].sort_values("raw_H_per_LH", ascending=False)
        raw = raw.assign(rank_raw=np.arange(1, len(raw) + 1))
        df["rank_raw_H_per_LH"] = raw["rank_raw"]
        # (d) equal-phase-weighted pressure
        eq = equal_phase_pressure(s, vc)
        df["equal_phase_pressure"] = eq
        el = df[df.rank_main.notna()].copy()
        el["rank_equal_phase"] = el.equal_phase_pressure.rank(ascending=False, method="first")
        df["rank_equal_phase"] = el.rank_equal_phase
        # (f) IPL + full-member-only
        s2 = s[s.tier != "associate"]
        df2, vc2, _ = rate_side(s2, side, boot=False)
        r2, _ = rank(df2, *ELIG["main"])
        df["rank_ipl_fm_only"] = r2["rank"]
        el_ = df[df.rank_main.notna()]
        df["rank_H_only"] = el_.p10_H_only.rank(ascending=False, method="first")
        res[side] = dict(df=df, vc=vc, ci=ci, tol=tol, vc2=vc2, n_elig=int(df.rank_main.notna().sum()))
        cols = ["rank_main", "name", "p10_pressure", "pressure", "pressure_lo80", "pressure_hi80", "normal", "lift", "lift_lo80",
                "lift_hi80", "P_pressure_gt_0", "raw_H_per_LH", "raw_L_per_LL", "sum_LH", "LL", "matches",
                "share_LH_IPL", "share_LH_full_member", "share_LH_associate", "share_LH_Powerplay", "share_LH_Middle",
                "share_LH_Death", "share_LH_inn1", "share_LH_inn2", "raw_pressure_strike_rate" if side == "bat" else "raw_pressure_economy",
                "total_wpa", "sd_pressure", "sd_theta", "sd_delta", "corr_theta_delta", "equal_phase_pressure",
                "rank_lh100", "rank_lh200", "rank_raw_H_per_LH", "rank_equal_phase", "rank_ipl_fm_only",
                "pressure_H_only", "p10_H_only", "rank_H_only"]
        df.index.name = "player_id"
        df.sort_values(["rank_main", "sum_LH"], ascending=[True, False])[cols].to_csv(
            os.path.join(MODEL, f"phase4_player_ratings_{side}.csv"))
        forest(df[df.rank_main <= 25].sort_values("rank_main"), side, os.path.join(REP4, f"phase4_forest_top25_{side}.png"))
    write_report(res, time.perf_counter() - t0)


def write_report(res, rt):
    pp = lambda x: f"{100 * x:+.2f}" if isinstance(x, float) else x
    f3 = lambda x: f"{x:.3f}" if isinstance(x, float) else x
    L = ["# Phase 4 - player pressure ratings", "",
         "Empirical-Bayes model on player-match leverage sums (see `src/phase4_ratings.py` docstring). pressure = theta + "
         "delta, normal = theta, lift = delta. **Units: win-probability percentage points per average-leverage ball** "
         "(x100 of v / LI). Ranking score = posterior 10th percentile of pressure; eligibility sum LH >= 150 and >= 15 "
         "matches. Values v_adj (opponent + associate-tier corrected, Phase 3).", ""]
    for side, R in res.items():
        df, vc, ci = R["df"], R["vc"], R["ci"]
        role = "Batters (balls faced)" if side == "bat" else "Bowlers (balls bowled)"
        sr = "raw_pressure_strike_rate" if side == "bat" else "raw_pressure_economy"
        el = df[df.rank_main.notna()]
        vct = pd.DataFrame([{"component": k, "estimate": vc[k], "lo95": ci[k][0], "hi95": ci[k][1],
                             "sd_x100": 100 * np.sqrt(max(vc[k], 0))} for k in ("sH2", "sL2", "tau_theta2", "tau_delta2", "tau_sum2", "tau_theta2_cov")])
        top = el.sort_values("rank_main").head(25)
        a_cols = ["rank_main", "name", "p10_pressure", "pressure", "pressure_lo80", "pressure_hi80", "normal", "lift", "lift_lo80",
                  "lift_hi80", "P_pressure_gt_0", "raw_H_per_LH", "raw_L_per_LL", "sum_LH", "matches", "share_LH_IPL",
                  "share_LH_full_member", "share_LH_associate", "share_LH_Powerplay", "share_LH_Middle", "share_LH_Death",
                  "share_LH_inn1", "share_LH_inn2"]
        ta = top[a_cols].copy()
        for c in ["p10_pressure", "pressure", "pressure_lo80", "pressure_hi80", "normal", "lift", "lift_lo80", "lift_hi80",
                  "raw_H_per_LH", "raw_L_per_LL"]:
            ta[c] = ta[c].map(pp)
        for c in [c for c in a_cols if c.startswith("share") or c == "P_pressure_gt_0"]:
            ta[c] = ta[c].map(lambda x: f"{x:.2f}")
        ta["sum_LH"] = ta.sum_LH.round(0).astype(int); ta["rank_main"] = ta.rank_main.astype(int)
        rawtop = el.sort_values("rank_raw_H_per_LH").head(25)[["rank_raw_H_per_LH", "name", "raw_H_per_LH", "sum_LH", "matches", "rank_main", "pressure", "p10_pressure"]]
        rawtop = rawtop.assign(raw_H_per_LH=rawtop.raw_H_per_LH.map(pp), pressure=rawtop.pressure.map(pp),
                               p10_pressure=rawtop.p10_pressure.map(pp), sum_LH=rawtop.sum_LH.round(0))
        moved_out = int((rawtop.rank_main > 25).sum())
        rho_sr = spearmanr(el.pressure, el[sr], nan_policy="omit")
        rho_wpa = spearmanr(el.pressure, el.total_wpa)
        rho_p10_raw = spearmanr(el.p10_pressure, el.raw_H_per_LH)
        death = el[(el.rank_main <= 10) & (el.share_LH_Death > 0.6)][["rank_main", "name", "share_LH_Death", "equal_phase_pressure", "rank_equal_phase"]]
        death = death.assign(equal_phase_pressure=death.equal_phase_pressure.map(pp))
        assoc25 = top[top.share_LH_associate > 0.5][["rank_main", "name", "share_LH_associate"]]
        ipl10 = set(el[el.rank_ipl_fm_only <= 10].index) if el.rank_ipl_fm_only.notna().any() else set()
        main10 = set(el[el.rank_main <= 10].index)
        cmp10 = el[el.index.isin(main10 | ipl10)].sort_values("rank_main")[["rank_main", "rank_ipl_fm_only", "name", "share_LH_associate"]]
        sens = el[el.rank_main <= 10].sort_values("rank_main")[["rank_main", "name", "rank_lh100", "rank_lh200"]]
        honly = el.sort_values("rank_H_only").head(10)[["rank_H_only", "name", "p10_H_only", "pressure_H_only", "raw_H_per_LH", "sum_LH", "rank_main"]]
        honly = honly.assign(p10_H_only=honly.p10_H_only.map(pp), pressure_H_only=honly.pressure_H_only.map(pp),
                             raw_H_per_LH=honly.raw_H_per_LH.map(pp), sum_LH=honly.sum_LH.round(0))
        n_neg_raw_top10 = int(((el.rank_main <= 10) & (el.raw_H_per_LH < 0)).sum())
        k_h = vc["sH2"] / max(vc["tau_sum2"], 1e-12)
        L += [f"## {role}", "",
              f"Eligible players: {R['n_elig']} (sum LH >= 150, >= 15 matches). Tie tolerance on p10: {100 * R['tol']:.3f} pts.", "",
              "### Variance components (1000 player-bootstrap resamples)", "",
              vct.map(lambda x: f"{x:.3e}" if isinstance(x, float) else x).to_markdown(index=False), "",
              "sd_x100 = square root in win-probability points per average ball. tau_theta2_cov is the covariance-based "
              "cross-check of tau_theta2.", "",
              "### (a) Top 25 (pressure, normal, lift in pts per average ball; intervals 80%)", "",
              ta.to_markdown(index=False), "",
              f"![forest](phase4_forest_top25_{side}.png)", "",
              "### (b) Top 25 by raw H/LH (no shrinkage)", "",
              f"{moved_out} of the raw top 25 are outside the shrunk top 25.", "",
              rawtop.to_markdown(index=False), "",
              "### (c) Spearman correlations (eligible players)", "",
              f"- pressure vs raw pressure {'strike rate' if side == 'bat' else 'economy (lower is better)'} on LI >= 1 balls: "
              f"{rho_sr.statistic:+.3f}",
              f"- pressure vs total WPA (raw, all balls): {rho_wpa.statistic:+.3f}",
              f"- ranking score (p10) vs raw H/LH: {rho_p10_raw.statistic:+.3f}", "",
              "### (d) Death specialists in the top 10 (> 60% of LH in death overs)", "",
              (death.to_markdown(index=False) if len(death) else "None.") , "",
              "Equal-phase-weighted pressure = mean over powerplay / middle / death of the shrunk per-phase H/LH.", "",
              f"### (e) Top-25 players with > 50% of LH from associate matches: {len(assoc25)}", "",
              (assoc25.to_markdown(index=False) if len(assoc25) else ""), "",
              f"### (f) Main vs IPL + full-member-only ranking: top-10 overlap {len(main10 & ipl10)} / 10", "",
              cmp10.to_markdown(index=False), "",
              "### Eligibility sensitivity (top 10 under sum LH >= 100 and >= 200)", "", sens.to_markdown(index=False), "",
              "### Lift and what the main score measures", "",
              f"tau_delta2 = {vc['tau_delta2']:.2e} (95% CI {ci['tau_delta2'][0]:.2e} to {ci['tau_delta2'][1]:.2e}): no detectable "
              "between-player variation in pressure-specific performance, so every lift is shrunk to exactly 0 and pressure = "
              "theta. theta is informed by both kinds of ball with precision LH/sH2 and LL/sL2; normal balls are less noisy "
              f"(sL2/sH2 = {vc['sL2'] / vc['sH2']:.2f}) and more numerous, so the main score is driven mostly by normal-ball "
              f"performance. Top-10 players with negative raw H/LH: {n_neg_raw_top10}.", "",
              f"Comparison only - pressure balls alone (H/LH shrunk with prior variance tau_sum2, k = {k_h:.0f} LI units), top 10:", "",
              honly.to_markdown(index=False), ""]
    L += [f"Runtime {rt:.0f} s.", ""]
    open(os.path.join(REP4, "phase4_summary.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
