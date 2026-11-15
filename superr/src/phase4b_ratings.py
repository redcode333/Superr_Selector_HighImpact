"""Phase 4 (revised) - uncertainty-averaged pressure ratings, all-player rankings, method self-test.

Model as Phase 4 (src/phase4_ratings.py): per player-match leverage sums H, LH (LI >= 1) and L, LL (LI < 1);
y1 = H/LH ~ N(theta + delta, sH2/LH), y2 = L/LL ~ N(theta, sL2/LL), theta ~ N(0, tau_theta2), delta ~ N(0, tau_delta2).

Uncertainty-averaged rating (replaces plugging in tau_delta2 = 0): for each of 1000 player-bootstrap draws of
(sH2, sL2, tau_theta2, tau_delta2), tau_delta2 clipped at 0 (and tau_theta2 at a tiny positive floor), every player's
exact posterior for pressure = theta + delta is computed on the ORIGINAL data; the draws are combined as a mixture
(mean = average of draw means, variance = average draw variance + variance of draw means). The cautious rating is the
10th percentile of the mixture, from 10 samples per draw (10,000 per player; no Normal assumption).
rating_A = the earlier score (point estimates, tau_delta2 = 0, Normal 10th percentile), kept for comparison.

Weight of pressure balls: the posterior mean of pressure is linear, w_H y1 + w_L y2; the pressure-ball weight share is
w_H / (w_H + w_L), compared with the pressure balls' share of leverage LH / (LH + LL).
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_outcome import MODEL, SEED
from phase4_ratings import load, player_match, player_stats, estimate, posterior, profile, raw_rates, Z80, REP4

N_DRAWS, N_SAMP = 1000, 10
SIM_REPS, SIM_DRAWS, SIM_SAMP = 200, 200, 10
STRONG, MODERATE, MIN_M = 400, 150, 15
UNIT = ("# Ratings are in win-probability points per average ball: +0.10 means the player's side gained on average "
        "0.10 percentage points of win probability per ball of average match importance, compared with a typical "
        "player in the same season, competition, innings and phase.")


# ------------------------------------------------------------------ core
def clip_vc(vc):
    v = dict(vc)
    v["tau_delta2"] = max(v["tau_delta2"], 0.0)
    v["tau_theta2"] = max(v["tau_theta2"], 1e-12)
    return v


def post_arrays(LH, LL, y1, y2, vc):
    """Vectorised exact posterior; returns means/vars of pressure, theta, delta and the linear weights."""
    tt, td = vc["tau_theta2"], max(vc["tau_delta2"], 1e-14)
    a1 = np.where(LH > 0, LH / vc["sH2"], 0.0); a2 = np.where(LL > 0, LL / vc["sL2"], 0.0)
    y1, y2 = np.nan_to_num(y1), np.nan_to_num(y2)
    P11, P12, P22 = 1 / tt + a1 + a2, a1, 1 / td + a1
    det = P11 * P22 - P12 ** 2
    S11, S12, S22 = P22 / det, -P12 / det, P11 / det
    b1, b2 = a1 * y1 + a2 * y2, a1 * y1
    th, de = S11 * b1 + S12 * b2, S12 * b1 + S22 * b2
    wH, wL = a1 * (S11 + 2 * S12 + S22), a2 * (S11 + S12)
    return {"m": th + de, "v": S11 + S22 + 2 * S12, "th": th, "vth": S11, "de": de, "vde": S22, "wH": wH, "wL": wL}


def mixture(LH, LL, y1, y2, vcs, n_samp, rng, keep_weights=False):
    """Combine posteriors over draws of the variance components; sample the mixture for quantiles."""
    n = len(LH); D = len(vcs)
    ms, vs, ths, des = (np.zeros((D, n)) for _ in range(4))
    samp_p = np.empty((n, D * n_samp), dtype=np.float32); samp_d = np.empty((n, D * n_samp), dtype=np.float32)
    wsh = np.zeros((D, n))
    for i, vc in enumerate(vcs):
        r = post_arrays(LH, LL, y1, y2, clip_vc(vc))
        ms[i], vs[i], ths[i], des[i] = r["m"], r["v"], r["th"], r["de"]
        z = rng.standard_normal((n, n_samp))
        samp_p[:, i * n_samp:(i + 1) * n_samp] = r["m"][:, None] + np.sqrt(r["v"])[:, None] * z
        samp_d[:, i * n_samp:(i + 1) * n_samp] = r["de"][:, None] + np.sqrt(r["vde"])[:, None] * rng.standard_normal((n, n_samp))
        if keep_weights:
            tot = r["wH"] + r["wL"]
            wsh[i] = np.where(tot > 0, r["wH"] / np.where(tot > 0, tot, 1), np.nan)
    out = {"mean": ms.mean(0), "var": vs.mean(0) + ms.var(0), "theta": ths.mean(0), "lift": des.mean(0),
           "p10": np.quantile(samp_p, 0.1, axis=1), "p90": np.quantile(samp_p, 0.9, axis=1),
           "p_gt0": (samp_p > 0).mean(1), "lift_p10": np.quantile(samp_d, 0.1, axis=1), "lift_p90": np.quantile(samp_d, 0.9, axis=1)}
    if keep_weights:
        out["w_share_draws"] = wsh
    return out


def boot_vcs(g, n, seed):
    rng = np.random.default_rng(seed)
    return [estimate(g, np.bincount(rng.integers(0, len(g), len(g)), minlength=len(g)).astype(float)) for _ in range(n)]


def tie_rank(e, score, sd_width):
    """Sort by score; adjacent players closer than 5% of the median 80% width are re-ordered by lift, then sum LH."""
    e = e.sort_values(score, ascending=False)
    tol = 0.05 * np.median(sd_width[e.index]) if len(e) else 0
    idx = list(e.index); sc = e[score].to_dict(); key = {i: (round(e.at[i, "lift"], 12), e.at[i, "sum_LH"]) for i in idx}
    for i in range(len(idx) - 1):
        a, b = idx[i], idx[i + 1]
        if sc[a] - sc[b] < tol and key[b] > key[a]:
            idx[i], idx[i + 1] = b, a
    return pd.Series(np.arange(1, len(idx) + 1), index=idx), tol


# ------------------------------------------------------------------ self-test (method test only)
def fast_stats(codes, n_players, H, LH, L, LL, M):
    s = lambda x: np.bincount(codes, weights=x, minlength=n_players)
    Hj, LHj, Lj, LLj = s(H), s(LH), s(L), s(LL)
    y1 = np.where(LHj > 0, Hj / np.where(LHj > 0, LHj, 1), np.nan); y2 = np.where(LLj > 0, Lj / np.where(LLj > 0, LLj, 1), np.nan)
    y1m, y2m = np.nan_to_num(y1)[codes], np.nan_to_num(y2)[codes]
    LHjm, LLjm = LHj[codes], LLj[codes]
    numH = s((H - LH * y1m) ** 2); denH = s(np.where(LHjm > 0, LH - LH ** 2 / np.where(LHjm > 0, LHjm, 1), 0))
    numL = s((L - LL * y2m) ** 2); denL = s(np.where(LLjm > 0, LL - LL ** 2 / np.where(LLjm > 0, LLjm, 1), 0))
    return pd.DataFrame({"H": Hj, "LH": LHj, "L": Lj, "LL": LLj, "M": M, "y1": y1, "y2": y2,
                         "numH": numH, "denH": denH, "numL": numL, "denL": denL})


def self_test(pm, players, elig, vc, tau_delta2_true, seed):
    """Plant abilities on the real schedule, simulate player-match sums, run both rating versions."""
    rng = np.random.default_rng(seed)
    codes = pd.Categorical(pm.player, categories=players).codes
    n = len(players)
    LH, LL = pm.LH.values, pm.LL.values
    M = np.bincount(codes, minlength=n)
    e = np.where(elig)[0]
    rows = []
    for rep in range(SIM_REPS):
        th = rng.normal(0, np.sqrt(vc["tau_theta2"]), n); de = rng.normal(0, np.sqrt(tau_delta2_true), n) if tau_delta2_true > 0 else np.zeros(n)
        true = th + de
        H = LH * true[codes] + np.sqrt(vc["sH2"] * LH) * rng.standard_normal(len(LH))
        L = LL * th[codes] + np.sqrt(vc["sL2"] * LL) * rng.standard_normal(len(LL))
        g = fast_stats(codes, n, H, LH, L, LL, M)
        vhat = estimate(g)
        rA = post_arrays(g.LH.values, g.LL.values, g.y1.values, g.y2.values, clip_vc(vhat))
        sdA = np.sqrt(rA["v"])
        A = {"score": rA["m"] - Z80 * sdA, "lo": rA["m"] - Z80 * sdA, "hi": rA["m"] + Z80 * sdA}
        vcs = boot_vcs(g, SIM_DRAWS, seed + rep + 1)
        mU = mixture(g.LH.values, g.LL.values, g.y1.values, g.y2.values, vcs, SIM_SAMP, rng)
        U = {"score": mU["p10"], "lo": mU["p10"], "hi": mU["p90"]}
        t_e = true[e]
        top10_true = set(e[np.argsort(-t_e)[:10]])
        for name, R in (("rating_A", A), ("uncertainty_averaged", U)):
            s_e = R["score"][e]
            order = e[np.argsort(-s_e)]
            top25 = order[:25]
            rows.append({"rep": rep, "version": name, "spearman_all_eligible": spearmanr(t_e, s_e).statistic,
                         "spearman_est_top25": spearmanr(true[top25], R["score"][top25]).statistic,
                         "true_top10_in_est_top10": len(top10_true & set(order[:10])) / 10,
                         "coverage_80": float(((R["lo"][e] <= t_e) & (t_e <= R["hi"][e])).mean()),
                         "tau_delta2_hat_pos": vhat["tau_delta2"] > 0})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    t0 = time.perf_counter()
    S = load()
    results, sim_all = {}, []
    for side in ("bat", "bowl"):
        s = S[side]
        pm = player_match(s)
        g = player_stats(pm)
        vc = estimate(g)
        vcs = boot_vcs(g, N_DRAWS, SEED)
        V = pd.DataFrame(vcs)
        ci = {k: (float(V[k].quantile(.025)), float(V[k].quantile(.975))) for k in V}
        n_pos = int((V.tau_delta2 > 0).sum())
        rng = np.random.default_rng(SEED + 7)
        mx = mixture(g.LH.values, g.LL.values, g.y1.values, g.y2.values, vcs, N_SAMP, rng, keep_weights=True)
        # rating_A (point estimates, tau_delta2 clipped, Normal)
        rA = post_arrays(g.LH.values, g.LL.values, g.y1.values, g.y2.values, clip_vc(vc))
        prof = profile(s).reindex(g.index)
        df = pd.DataFrame(index=g.index)
        df["name"] = prof.name
        df["cautious"] = mx["p10"]; df["pressure"] = mx["mean"]; df["sd_mix"] = np.sqrt(mx["var"])
        df["lo80"], df["hi80"] = mx["p10"], mx["p90"]
        df["normal"] = mx["theta"]; df["lift"] = mx["lift"]; df["lift_lo80"], df["lift_hi80"] = mx["lift_p10"], mx["lift_p90"]
        df["p_above_avg"] = mx["p_gt0"]
        df["rating_A"] = rA["m"] - Z80 * np.sqrt(rA["v"]); df["pressure_A"] = rA["m"]; df["sd_A"] = np.sqrt(rA["v"])
        df["raw_H_per_LH"] = g.y1; df["raw_L_per_LL"] = g.y2
        df["raw_H_lo80"] = g.y1 - Z80 * np.sqrt(vc["sH2"] / g.LH.where(g.LH > 0))
        df["raw_H_hi80"] = g.y1 + Z80 * np.sqrt(vc["sH2"] / g.LH.where(g.LH > 0))
        df["sum_LH"] = g.LH; df["sum_LL"] = g.LL
        df["pressure_balls"] = s[s.hi].groupby("player").size().reindex(g.index).fillna(0).astype(int)
        df["balls"] = s.groupby("player").size().reindex(g.index)
        df["matches"] = prof.matches
        for c in ["share_LH_IPL", "share_LH_full_member", "share_LH_associate", "share_LH_Powerplay", "share_LH_Middle",
                  "share_LH_Death", "share_LH_inn1", "share_LH_inn2", "total_wpa"]:
            df[c] = prof[c]
        rr = raw_rates(s, side); df[rr.name] = rr
        wsh = mx["w_share_draws"]
        df["pressure_weight_share"] = np.nanmean(wsh, axis=0)
        df["pressure_leverage_share"] = g.LH / (g.LH + g.LL)
        df["evidence_level"] = np.where((df.sum_LH >= STRONG) & (df.matches >= MIN_M), "Strong",
                                        np.where((df.sum_LH >= MODERATE) & (df.matches >= MIN_M), "Moderate", "Limited"))
        width = df.hi80 - df.lo80
        r_all, tol = tie_rank(df, "cautious", width)
        df["rank_all"] = r_all
        el = df[df.evidence_level != "Limited"]
        r_el, _ = tie_rank(el, "cautious", width)
        df["rank_eligible"] = r_el
        widthA = 2 * Z80 * df.sd_A
        rA_el, _ = tie_rank(el.assign(lift=0.0), "rating_A", widthA)
        df["rank_eligible_A"] = rA_el
        # weights summary over eligible players, by draw
        e_mask = (df.evidence_level != "Limited").values
        w_by_draw = np.nanmean(wsh[:, e_mask], axis=1)
        wsum = {"draws_tau_delta2_gt_0": n_pos, "of_draws": N_DRAWS,
                "mean_pressure_weight_share_eligible": float(np.nanmean(w_by_draw)),
                "mean_pressure_weight_share_when_tau_delta2_gt_0": float(np.nanmean(w_by_draw[V.tau_delta2.values > 0])) if n_pos else None,
                "mean_pressure_weight_share_when_tau_delta2_eq_0": float(np.nanmean(w_by_draw[V.tau_delta2.values <= 0])),
                "mean_pressure_leverage_share_eligible": float(df.loc[e_mask, "pressure_leverage_share"].mean())}
        # self-test (method test only)
        elig = e_mask
        players = list(g.index)
        for scen, td in (("no lift (fitted)", 0.0), ("lift spread at upper 95% CI", max(ci["tau_delta2"][1], 0.0))):
            st = self_test(pm, players, elig, vc, td, SEED + (11 if side == "bat" else 23) + (0 if td == 0 else 1000))
            st["side"], st["scenario"] = side, scen
            sim_all.append(st)
            print(side, scen, st.groupby("version")[["spearman_all_eligible", "true_top10_in_est_top10", "coverage_80"]].mean().round(3).to_dict(), flush=True)
        results[side] = dict(df=df, vc=vc, ci=ci, wsum=wsum, tol=tol)
        write_files(df, side)
    sim = pd.concat(sim_all)
    sim.to_csv(os.path.join(REP4, "phase4_selftest_reps.csv"), index=False)
    simsum = sim.groupby(["side", "scenario", "version"]).agg(
        spearman_all_eligible=("spearman_all_eligible", "mean"), spearman_est_top25=("spearman_est_top25", "mean"),
        true_top10_in_est_top10=("true_top10_in_est_top10", "mean"), coverage_80=("coverage_80", "mean"),
        coverage_80_p5=("coverage_80", lambda x: x.quantile(.05)), coverage_80_p95=("coverage_80", lambda x: x.quantile(.95)),
        share_reps_tau_delta2_hat_pos=("tau_delta2_hat_pos", "mean")).reset_index()
    simsum.to_csv(os.path.join(REP4, "phase4_selftest_summary.csv"), index=False)
    for side in results:
        forest_u(results[side]["df"], side)
    write_report(results, simsum, time.perf_counter() - t0)
    write_howto(results)


def write_files(df, side):
    pp = lambda x: 100 * x
    rs = "Raw pressure strike rate" if side == "bat" else "Raw pressure economy"
    rcol = "raw_pressure_strike_rate" if side == "bat" else "raw_pressure_economy"
    first, second = ("% batting first", "% chasing") if side == "bat" else ("% bowling in 1st innings", "% bowling in chase")
    o = df.sort_values("rank_all")
    plain = pd.DataFrame({
        "Rank (eligible)": o.rank_eligible.astype("Int64"), "Rank (all players)": o.rank_all.astype(int), "Player": o.name,
        "Evidence level": o.evidence_level, "Cautious rating (used for rank)": pp(o.cautious).round(3),
        "Pressure rating (best estimate)": pp(o.pressure).round(3), "Likely range low (80%)": pp(o.lo80).round(3),
        "Likely range high (80%)": pp(o.hi80).round(3), "Normal-situation rating": pp(o.normal).round(3),
        "Extra under pressure (estimate)": pp(o.lift).round(3), "Extra under pressure low (80%)": pp(o.lift_lo80).round(3),
        "Extra under pressure high (80%)": pp(o.lift_hi80).round(3),
        "Pressure record when it mattered": pp(o.raw_H_per_LH).round(3), "Pressure record low (80%)": pp(o.raw_H_lo80).round(3),
        "Pressure record high (80%)": pp(o.raw_H_hi80).round(3),
        "Chance he is above average under pressure (%)": (100 * o.p_above_avg).round(1),
        "Pressure exposure (sum of leverage)": o.sum_LH.round(1), "Pressure balls": o.pressure_balls, "Matches": o.matches,
        "% of pressure from IPL": (100 * o.share_LH_IPL).round(1),
        "% of pressure from internationals": (100 * (o.share_LH_full_member + o.share_LH_associate)).round(1),
        "% pressure in powerplay": (100 * o.share_LH_Powerplay).round(1), "% pressure in middle overs": (100 * o.share_LH_Middle).round(1),
        "% pressure in death overs": (100 * o.share_LH_Death).round(1),
        first: (100 * o.share_LH_inn1).round(1), second: (100 * o.share_LH_inn2).round(1), rs: o[rcol].round(2)})
    tech = o.copy(); tech.index.name = "player_id"
    for path, frame in ((os.path.join(MODEL, f"phase4_rankings_{side}_all_players.csv"), plain),
                        (os.path.join(MODEL, f"phase4_rankings_{side}_all_players_technical.csv"), tech)):
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(UNIT + (" Technical file: values in raw win-probability units (multiply by 100 for points)." if "technical" in path else "") + "\n")
            frame.to_csv(fh, index="technical" in path)


def forest_u(df, side):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = df[df.rank_eligible <= 25].sort_values("rank_eligible").iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 9.2))
    y = np.arange(len(t))
    ax.axvline(0, color="#bbbbbb", lw=1, zorder=0)
    ax.hlines(y, 100 * t.lo80, 100 * t.hi80, color="#3987e5", lw=2, zorder=1)
    ax.scatter(100 * t.pressure, y, s=40, color="#1c5cab", zorder=2, edgecolor="white", linewidth=1.5, label="best estimate")
    ax.scatter(100 * t.cautious, y, s=40, marker="|", color="#0d366b", zorder=3, linewidth=2, label="cautious rating (ranking score)")
    ax.set_yticks(y); ax.set_yticklabels([f"{int(r)}. {n}  ({ev[0]})" for r, n, ev in zip(t.rank_eligible, t.name, t.evidence_level)], fontsize=8.5)
    ax.set_xlabel("pressure rating, win-probability points per average ball (80% likely range)")
    role = "batters" if side == "bat" else "bowlers"
    ax.set_title(f"Top 25 eligible {role} - uncertainty-averaged pressure rating, 2022-26\n(S = strong, M = moderate evidence)", fontsize=10.5)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", lw=0.8); ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(REP4, f"phase4_forest_top25_{side}.png"), dpi=140); plt.close(fig)


def write_report(results, simsum, rt):
    pp = lambda x: f"{100 * x:+.3f}" if isinstance(x, (float, np.floating)) else x
    L = ["# Phase 4 (revised) - uncertainty-averaged pressure ratings", "",
         "Ratings in **win-probability points per average ball** (x100 of value per unit leverage). Cautious rating = "
         "10th percentile of the uncertainty-averaged posterior (1000 bootstrap draws of the variance components, "
         "lift spread clipped at 0, 10 samples per draw). rating_A = earlier score (point estimates, lift spread = 0).", "",
         "Plain files: `out/model/phase4_rankings_{bat,bowl}_all_players.csv`; technical copies `*_technical.csv`. "
         "Submission top 10 = Rank (eligible) 1-10.", ""]
    for side, R in results.items():
        df, vc, ci, w = R["df"], R["vc"], R["ci"], R["wsum"]
        role = "Batters (balls faced)" if side == "bat" else "Bowlers (balls bowled)"
        lev = df.evidence_level.value_counts().reindex(["Strong", "Moderate", "Limited"]).fillna(0).astype(int)
        top = df[df.rank_eligible <= 25].sort_values("rank_eligible")
        t = pd.DataFrame({"rank (eligible)": top.rank_eligible.astype(int), "rank (all)": top.rank_all.astype(int),
                          "player": top.name, "evidence": top.evidence_level, "cautious": top.cautious.map(pp),
                          "best estimate": top.pressure.map(pp), "range": [f"{100*a:+.3f} to {100*b:+.3f}" for a, b in zip(top.lo80, top.hi80)],
                          "normal": top.normal.map(pp), "extra under pressure": top.lift.map(pp),
                          "pressure record": top.raw_H_per_LH.map(pp), "P(above avg)": (100 * top.p_above_avg).round(1),
                          "sum LH": top.sum_LH.round(0).astype(int), "matches": top.matches,
                          "rank_A": top.rank_eligible_A.astype("Int64")})
        neg = df[(df.rank_eligible <= 25) & (df.raw_H_per_LH < 0)].sort_values("rank_eligible")
        negt = pd.DataFrame({"player": neg.name, "pressure record": neg.raw_H_per_LH.map(pp), "rank_A (eligible)": neg.rank_eligible_A.astype("Int64"),
                             "rank uncertainty-averaged (eligible)": neg.rank_eligible.astype(int), "sum LH": neg.sum_LH.round(0)})
        allneg = df[(df.rank_eligible_A <= 25) & (df.raw_H_per_LH < 0)]
        allnegt = pd.DataFrame({"player": allneg.name, "pressure record": allneg.raw_H_per_LH.map(pp),
                                "rank_A (eligible)": allneg.rank_eligible_A.astype("Int64"), "rank uncertainty-averaged (eligible)": allneg.rank_eligible.astype("Int64")})
        el = df[df.evidence_level != "Limited"]
        rho = spearmanr(el.cautious, el.rating_A).statistic
        top10_overlap = len(set(df[df.rank_eligible <= 10].index) & set(df[df.rank_eligible_A <= 10].index))
        L += [f"## {role}", "",
              f"Players ranked: {len(df)} (Strong {lev['Strong']}, Moderate {lev['Moderate']}, Limited {lev['Limited']}).", "",
              "### Uncertainty averaging", "",
              f"- Draws with lift spread > 0: **{w['draws_tau_delta2_gt_0']} of {w['of_draws']}** "
              f"(tau_delta2 point {vc['tau_delta2']:.2e}, 95% CI {ci['tau_delta2'][0]:.2e} to {ci['tau_delta2'][1]:.2e}).",
              f"- Average weight of pressure balls in the pressure rating (eligible players): **{100*w['mean_pressure_weight_share_eligible']:.1f}%** "
              f"(draws with lift spread > 0: {100*w['mean_pressure_weight_share_when_tau_delta2_gt_0']:.1f}%; "
              f"draws with 0: {100*w['mean_pressure_weight_share_when_tau_delta2_eq_0']:.1f}%), against a "
              f"{100*w['mean_pressure_leverage_share_eligible']:.1f}% share of their leverage.",
              f"- Cautious rating vs rating_A (eligible): Spearman {rho:.3f}; top-10 overlap {top10_overlap} / 10.", "",
              "### Top 25 eligible (uncertainty-averaged)", "", t.to_markdown(index=False), "",
              f"![forest](phase4_forest_top25_{side}.png)", "",
              "### Top-25 players with a pressure record below 0 (uncertainty-averaged top 25)", "",
              (negt.to_markdown(index=False) if len(negt) else "None."), "",
              "Same check on the rating_A top 25:", "", (allnegt.to_markdown(index=False) if len(allnegt) else "None."), ""]
    L += ["## Method self-test (simulation - no simulated number enters any ranking)", "",
          f"Real player-match schedule and leverage; true abilities planted from the fitted spread (scenario 1: no lift; "
          f"scenario 2: lift spread at the upper end of its 95% CI); player-match values generated with the fitted noise "
          f"(sH2, sL2); the full pipeline re-run {SIM_REPS} times (uncertainty-averaged version with {SIM_DRAWS} bootstrap "
          f"draws x {SIM_SAMP} samples per repetition, for speed; tie-break not applied). Truth = planted pressure ability "
          "(theta + delta); metrics over eligible players.", "",
          simsum.map(lambda x: f"{x:.3f}" if isinstance(x, float) else x).to_markdown(index=False), "",
          "## Pre-registered for Phase 5", "",
          "The final score is the **uncertainty-averaged cautious rating** if its 2022-24 -> 2025-26 prediction of each "
          "player's pressure-ball value (H/LH) is not significantly worse than rating_A (paired bootstrap CI of the "
          "difference in Spearman correlation, resampling players, not entirely below 0); otherwise rating_A.", "",
          f"Runtime {rt:.0f} s.", ""]
    open(os.path.join(REP4, "phase4_summary.md"), "w", encoding="utf-8").write("\n".join(L))


def write_howto(results):
    df = results["bat"]["df"]
    ex = df[df.rank_eligible == 1].iloc[0]
    bw = results["bowl"]["df"]; exb = bw[bw.rank_eligible == 1].iloc[0]
    w = results["bat"]["wsum"]
    p = lambda x: f"{100 * x:+.2f}"
    L = ["# How the ranking works", "",
         "1. **Give every ball an importance score.** From a win-probability model built on ball-by-ball data, we "
         "work out how much the match result could swing on that ball. An average ball scores 1; the last ball of a close "
         "chase can score 30; a ball early in a one-sided game scores close to 0. Balls scoring 1 or more are the "
         "**pressure balls**.",
         "2. **Measure what each ball did for the player's side.** For each ball we record the change in his team's "
         "chance of winning, compared with a typical ball at the same stage of the same season and competition. We then "
         "correct for the strength of the opponent (facing a top bowler is harder) and for matches against weaker "
         "associate teams (where everyone's numbers look better).",
         "3. **Add it up per player.** For pressure balls, we divide the total win probability he added by the total "
         "importance of those balls: his **pressure record**. We do the same for the other balls: his **normal record**.",
         "4. **Weigh the evidence with caution.** A short record can be luck, so every player is pulled towards average, "
         "and pulled harder the less he has played. Because we found no sign that some players have a special pressure "
         "skill beyond their general quality, a player's normal balls also tell us about his pressure balls. We repeat "
         "this with 1000 plausible versions of the model's settings and average the answers.",
         "5. **Rank by the cautious rating.** This is the level we are 90% confident the player is above. Only players "
         f"with enough pressure exposure (total importance of pressure balls at least {MODERATE} and at least {MIN_M} "
         "matches) can enter the top 10.", "",
         f"## Worked example: {ex['name']} (batting rank 1)", "",
         f"- He faced **{int(ex.pressure_balls)} pressure balls** across {int(ex.matches)} matches, with a total importance of "
         f"**{ex.sum_LH:.0f}** (about {ex.sum_LH / max(ex.pressure_balls, 1):.1f} average balls' worth each).",
         f"- On those balls his side gained **{p(ex.raw_H_per_LH)} win-probability points per average ball** (his pressure "
         f"record; plausible range from those balls alone {p(ex.raw_H_lo80)} to {p(ex.raw_H_hi80)}). On his other balls "
         f"the figure was {p(ex.raw_L_per_LL)}.",
         f"- After weighing the evidence (step 4), his best estimate is **{p(ex.pressure)}**, likely between "
         f"{p(ex.lo80)} and {p(ex.hi80)}. The pressure balls carry {100 * ex.pressure_weight_share:.0f}% of that estimate.",
         f"- His **cautious rating is {p(ex.cautious)}**: we are 90% confident his true level is above this. The chance he "
         f"is above an average player under pressure is {100 * ex.p_above_avg:.0f}%.",
         f"- For bowlers the same steps give {exb['name']} rank 1 (cautious rating {p(exb.cautious)}, pressure record "
         f"{p(exb.raw_H_per_LH)} over {exb.sum_LH:.0f} importance).", "",
         "> **What we found**",
         "> - Pressure-specific skill is not detectable: no player stands out for doing better under pressure than his "
         "normal level would predict. Underlying quality carries into pressure.",
         f"> - So the best guide to who performs under pressure is overall quality, weighted towards the balls that "
         f"matter; on average the pressure balls carry {100 * w['mean_pressure_weight_share_eligible']:.0f}% of a batter's rating.",
         "> - Facing weaker associate teams inflates numbers; we remove that within each player.",
         "> - Uncertainty is large: neighbouring ranks often overlap, so read the ranges, not just the order.", ""]
    open(os.path.join(REP4, "how_the_ranking_works.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
