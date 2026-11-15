"""Phase 2 - pressure definition and validation (v2: leverage weighting is the main measure; flags are descriptive).

Pressure ball (main definition): LI >= global p80 of LI over all valid 2022+ main-innings balls (Phase 1f).
LI is the expected absolute swing in the batting team's win probability on this ball, scaled to mean 1 - the
situations where the ball matters most to the result. It combines required rate, wickets in hand, balls left,
stage of the innings and closeness of the contest through the WP model, rather than through hand-set weights.
Sensitivity definition (built only if innings 1 holds < 15% of main pressure balls): top 20% of LI within each
innings number (separate p80 for innings 1 and innings 2).
Validation: share of pressure balls by innings x phase and tier; pressure vs ordinary outcome rates with
match-bootstrap CIs; LI heatmaps over (over x wickets lost); the five highest-LI balls.
"""
import json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_outcome import MODEL, REP, SEED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(os.path.dirname(ROOT), "SupeRR_Data_Pack")
N_BOOT = 1000
PHASES = ["Powerplay", "Middle", "Death"]
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]


def outcome_stats(v, flag, by=None, n=N_BOOT, seed=SEED):
    """Dot %, boundary %, wicket rate, run rate for pressure vs ordinary balls, with match-bootstrap 95% CIs."""
    v = v.assign(legal=v.is_legal.astype(bool), dot=(v.is_legal.astype(bool) & (v.team_runs == 0)),
                 bnd=(v.is_four.astype(bool) | v.is_six.astype(bool)), wk=v.is_wicket.astype(bool), P=v[flag])
    groups = [("all", v)] if by is None else [(f"innings {k}", g) for k, g in v.groupby(by)]
    rows = []
    rng = np.random.default_rng(seed)
    for lab, g in groups:
        agg = g.groupby(["match_id", "P"]).agg(legal=("legal", "sum"), balls=("legal", "size"), dot=("dot", "sum"),
                                              bnd=("bnd", "sum"), wk=("wk", "sum"), runs=("team_runs", "sum"))
        agg = agg.unstack("P", fill_value=0)
        M = len(agg); idx = rng.integers(0, M, (n, M))
        metr = {"dot_pct": ("dot", "legal", 100), "boundary_pct": ("bnd", "legal", 100),
                "wickets_per_100_balls": ("wk", "balls", 100), "run_rate": ("runs", "legal", 6)}
        for name, (num, den, k) in metr.items():
            est = {}
            for p in (True, False):
                a, b = agg[(num, p)].values, agg[(den, p)].values
                est[p] = (k * a.sum() / b.sum(), k * a[idx].sum(1) / b[idx].sum(1))
            diff = est[True][1] - est[False][1]
            rows.append({"group": lab, "metric": name, "pressure": est[True][0],
                         "pressure_lo95": np.percentile(est[True][1], 2.5), "pressure_hi95": np.percentile(est[True][1], 97.5),
                         "ordinary": est[False][0], "ordinary_lo95": np.percentile(est[False][1], 2.5),
                         "ordinary_hi95": np.percentile(est[False][1], 97.5), "diff": est[True][0] - est[False][0],
                         "diff_lo95": np.percentile(diff, 2.5), "diff_hi95": np.percentile(diff, 97.5),
                         "n_pressure_balls": int(agg[("balls", True)].sum()), "n_ordinary_balls": int(agg[("balls", False)].sum())})
    return pd.DataFrame(rows)


def shares(v, flag):
    out = {}
    for key in (["innings", "phase"], ["tier"], ["innings"]):
        g = v.groupby(key).agg(balls=(flag, "size"), pressure_balls=(flag, "sum")).reset_index()
        g["share_of_all_pressure_balls"] = g.pressure_balls / g.pressure_balls.sum()
        g["pressure_rate_in_cell"] = g.pressure_balls / g.balls
        if "phase" in key:
            g["phase"] = pd.Categorical(g.phase, PHASES); g = g.sort_values(key)
        out["x".join(key)] = g
    return out


def heatmaps(v):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list("blue", RAMP); cmap.set_bad("#f0efec")
    v = v.assign(wl=10 - v.wkts_in_hand)
    tabs = {}
    for inn in (1, 2):
        g = v[v.innings == inn].groupby(["wl", "overid"]).li.agg(["mean", "size"])
        m = g["mean"].unstack("overid").reindex(index=range(10), columns=range(1, 21))
        nn = g["size"].unstack("overid").reindex(index=range(10), columns=range(1, 21)).fillna(0)
        tabs[inn] = (m.where(nn >= 30), nn)
    vals = pd.concat([t[0].stack() for t in tabs.values()])
    norm = LogNorm(vmin=max(vals.min(), 0.2), vmax=vals.max())
    paths = []
    for inn, (m, nn) in tabs.items():
        fig, ax = plt.subplots(figsize=(12, 5.2))
        im = ax.imshow(np.ma.masked_invalid(m.values), cmap=cmap, norm=norm, aspect="auto", origin="lower")
        for (i, j), val in np.ndenumerate(m.values):
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}" if val < 10 else f"{val:.0f}", ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if norm(val) > 0.55 else "#1f1f1f")
        ax.set_xticks(range(20)); ax.set_xticklabels(range(1, 21)); ax.set_yticks(range(10))
        ax.set_xlabel("over"); ax.set_ylabel("wickets lost before the ball")
        side = "batting first (WP1)" if inn == 1 else "chasing (chase WP)"
        ax.set_title(f"Mean leverage index by over and wickets lost - innings {inn}, {side}, 2022-26 "
                     "(grey: fewer than 30 balls; colour scale shared with the other innings, log)", fontsize=9.5)
        for s in ax.spines.values(): s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("mean LI (1 = average ball)")
        tk = [t for t in (0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20) if norm.vmin <= t <= norm.vmax]
        cb.set_ticks(tk); cb.set_ticklabels([f"{t:g}" for t in tk]); cb.minorticks_off()
        cb.outline.set_visible(False)
        fig.tight_layout()
        p = os.path.join(REP, f"phase2_li_heatmap_innings{inn}.png"); fig.savefig(p, dpi=140); plt.close(fig)
        paths.append(p)
    return paths


def examples(v, k=5):
    teams = pd.read_csv(os.path.join(PACK, "teams.csv")).drop_duplicates("team_id").set_index("team_id").team_name
    top = v.sort_values("li", ascending=False).head(k)
    rows = []
    for _, r in top.iterrows():
        over = f"{int(r.overid)}.{int(r.over_ball_num)}"
        bat, bowl = teams.get(r.batting_team_id, r.batting_team_id), teams.get(r.bowling_team_id, r.bowling_team_id)
        if r.innings == 2:
            sit = (f"{bat} need {int(r.runs_required)} off {int(r.balls_remaining)} balls, {int(r.wkts_in_hand)} wickets "
                   f"in hand (score {int(round(r.score_before))}/{10 - int(r.wkts_in_hand)})")
        else:
            sit = f"{bat} {int(round(r.score_before))}/{10 - int(r.wkts_in_hand)}, {int(r.balls_remaining)} balls left (batting first)"
        what = []
        if r.is_wide: what.append("wide")
        if r.is_noball: what.append("no-ball")
        what.append(f"{int(r.bat_runs)} off the bat" + (f", {int(r.team_runs)} total" if r.team_runs != r.bat_runs else ""))
        if r.is_wicket: what.append(f"WICKET ({r.dismissal_type})")
        rows.append({"match": f"{r.comp_name}, {r.match_date}: {bat} v {bowl}", "innings": int(r.innings), "over": over,
                     "situation": sit, "striker": r.striker_name, "bowler": r.bowler_name, "what happened": "; ".join(what),
                     "LI": r.li, "WP before": r.wp_before, "WP after": r.wp_after, "WPA (batting team)": r.wpa})
    return pd.DataFrame(rows)


def li_by_wp(v):
    band = pd.cut(v.wp_before, [0, .05, .2, .35, .5, .65, .8, .95, 1], include_lowest=True)
    return v.groupby(["innings", band], observed=True).agg(balls=("li", "size"), mean_li=("li", "mean"),
                                                            pressure_rate=("pressure_main", "mean")).reset_index()


def li_share(v):
    g = v.groupby(["innings", "phase"]).agg(balls=("li", "size"), sum_li=("li", "sum")).reset_index()
    g["share_of_balls"] = g.balls / g.balls.sum()
    g["share_of_total_leverage"] = g.sum_li / g.sum_li.sum()
    g["phase"] = pd.Categorical(g.phase, PHASES)
    return g.sort_values(["innings", "phase"])


def main():
    t0 = time.perf_counter()
    d = pd.read_parquet(os.path.join(MODEL, "phase1f_ball_values.parquet"))
    v = d[d.missing_reason.isna()].copy()
    thr, thr90 = float(v.li.quantile(0.8)), float(v.li.quantile(0.9))
    thr_inn = {int(k): float(x) for k, x in v.groupby("innings").li.quantile(0.8).items()}
    v["pressure_main"] = v.li >= thr
    v["pressure_p90"] = v.li >= thr90
    v["pressure_per_innings"] = v.li >= v.innings.map(thr_inn)
    flags = ["pressure_main", "pressure_p90", "pressure_per_innings"]
    inn1_share = float(v.loc[v.innings == 1, "pressure_main"].sum() / v.pressure_main.sum())
    sh = {f: shares(v, f) for f in flags}
    stats = {f: pd.concat([outcome_stats(v, f), outcome_stats(v, f, by="innings")]) for f in flags}
    lis = li_share(v)
    bnd = v[v.is_legal.astype(bool)].groupby("phase").agg(fours_pct=("is_four", "mean"), sixes_pct=("is_six", "mean")) * 100
    li_inn = v.groupby("innings").li.agg(p1=lambda x: x.quantile(.01), median="median", p99=lambda x: x.quantile(.99), max="max")
    paths = heatmaps(v)
    ex = examples(v)
    lw = li_by_wp(v)
    keys = ["match_id", "innings", "innings_ball_num"]
    out = d[keys].merge(v[keys + ["li"] + flags], on=keys, how="left")
    out.to_parquet(os.path.join(MODEL, "phase2_pressure_flags.parquet"), index=False)
    cfg = {"main_measure": "leverage weighting, no cutoff: pressure = sum(v)/sum(LI); normal = mean(v/LI | LI < 1); lift = pressure - normal",
           "flags_descriptive_only": {"pressure_main": f"li >= global p80 = {thr:.5f}", "pressure_p90": f"li >= global p90 = {thr90:.5f}",
                                      "pressure_per_innings": {"rule": "li >= p80 of its innings number", "thresholds": thr_inn}},
           "inn1_share_of_p80_pressure_balls": inn1_share}
    json.dump(cfg, open(os.path.join(MODEL, "phase2_pressure_config.json"), "w"), indent=1, default=float)
    write_report(cfg, thr, thr90, thr_inn, sh, stats, ex, lw, lis, bnd, li_inn, paths, time.perf_counter() - t0)
    print(json.dumps(cfg, indent=1, default=float))


def write_report(cfg, thr, thr90, thr_inn, sh, stats, ex, lw, lis, bnd, li_inn, paths, rt):
    f = lambda x: f"{x:.4f}" if isinstance(x, float) else x
    f2 = lambda x: f"{x:.2f}" if isinstance(x, float) else x
    L = ["# Phase 2 - pressure definition and validation", "",
         "## Main measure: leverage weighting (no cutoff)", "",
         "Every ball carries its leverage index LI (Phase 1f): the expected absolute change in the batting team's win "
         "probability on that ball, scaled to mean 1 over all valid 2022-26 balls. LI is high exactly when one ball can "
         "move the result - a close contest, late in a chase, few balls or wickets left, a required rate near what is "
         "achievable - and the required rate, wickets, balls, phase and closeness enter through the win-probability "
         "model, not through hand-set weights. The same LI applies to the batter and the bowler of the ball.", "",
         "For a player, with v = wpa_centred from his side (batting: balls faced, v = wpa_centred; bowling: balls bowled, "
         "v = -wpa_centred):", "",
         "- **pressure performance** = sum(v) / sum(LI) over all his valid balls - every ball counts, weighted by how much "
         "it mattered (= the LI-weighted mean of v / LI);",
         "- **normal performance** = mean(v / LI) over his balls with LI < 1 (ratio computed for LI >= 0.05; the ~5% of "
         "near-dead balls below that carry no information and make the ratio unstable);",
         "- **lift** = pressure - normal.", "",
         "Hard flags are kept only for description, the dashboard and sensitivity: LI >= global p80 "
         f"({thr:.3f}), LI >= global p90 ({thr90:.3f}), and LI >= the p80 of its innings number "
         f"(innings 1: {thr_inn[1]:.3f}, innings 2: {thr_inn[2]:.3f}). Columns pressure_main, pressure_p90, "
         "pressure_per_innings in `out/model/phase2_pressure_flags.parquet`.", "",
         "## Why a cutoff fails for innings 1 (finding)", "",
         f"- The global p80 cutoff ({thr:.3f}) sits on the innings-1 powerplay median LI (1.29), so whether an innings-1 "
         "ball is a pressure ball is knife-edge: 53% of innings-1 powerplay balls clear it and essentially no innings-1 "
         "middle or death ball does.",
         "- The over 6 -> 7 cliff in innings-1 LI is real, not a bug: fielding restrictions end. Share of fours per legal "
         f"ball 2022-26: powerplay {bnd.loc['Powerplay', 'fours_pct']:.1f}% vs middle {bnd.loc['Middle', 'fours_pct']:.1f}% "
         "(all seasons, Phase 1a: 14.8% vs 8.4%), so a powerplay ball has a wider spread of outcomes and of WP change.",
         f"- First-innings LI spans about {li_inn.loc[1, 'p1']:.1f}-{li_inn.loc[1, 'max']:.1f}, chase LI about "
         f"{li_inn.loc[2, 'p1']:.2f}-{li_inn.loc[2, 'max']:.0f}: a single first-innings ball rarely swings a T20 match, "
         "while the last balls of a close chase can decide it. Leverage weighting uses this range directly instead of "
         "forcing a binary split.", "",
         "LI by innings:", "", li_inn.reset_index().map(f).to_markdown(index=False), "",
         "## Share of total leverage (sum of LI) by innings x phase", "", lis.map(f).to_markdown(index=False), "",
         f"Innings 1 share of p80 pressure balls: {cfg['inn1_share_of_p80_pressure_balls']:.1%}.", ""]
    for fl, s in sh.items():
        L += [f"## Where flagged pressure balls fall - `{fl}`", ""]
        for k, t in s.items():
            L += [f"By {k}:", "", t.map(f).to_markdown(index=False), ""]
    for fl, s in stats.items():
        L += [f"## Pressure vs ordinary balls - `{fl}`", "",
              "Dot % and boundary % per legal ball; wickets per 100 deliveries; run rate = 6 x team runs / legal balls. "
              "95% CIs from 1000 match-bootstrap resamples.", "",
              s.map(f2).to_markdown(index=False), ""]
    L += ["## Mean LI by win probability before the ball", "", lw.astype({"wp_before": str}).map(f).to_markdown(index=False), "",
          "## Heatmaps", ""] + [f"![{os.path.basename(p)}]({os.path.basename(p)})" for p in paths] + ["",
          "## The five highest-LI balls, 2022-26", "", ex.map(f).to_markdown(index=False), "", f"Runtime {rt:.0f} s.", ""]
    open(os.path.join(REP, "phase2_summary.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
