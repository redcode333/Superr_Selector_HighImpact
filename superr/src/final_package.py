"""Phase 6 - freeze and package. Reads the frozen outputs (no scores, ranks or eligibility are changed) and writes
reports/final/: final_top10.csv, alternates.csv, key_numbers.json, unstop_check.csv, method_log.md, figures/.

Close call flag: share of the 500 match-bootstrap re-rankings that put the player in the top 10 < 50%.
"""
import json, os, re, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_outcome import MODEL, OUT
from variant import mpath, vdir, FLOOR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
REP5 = vdir(os.path.join(REP, "phase5"))
FIN = vdir(os.path.join(REP, "final")); FIG = os.path.join(FIN, "figures")
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#b5b4ad", "#1f1f1e", "#6b6a64"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
PHASES = ["Powerplay", "Middle", "Death"]


# ------------------------------------------------------------------ tables
def load_role(side):
    t = pd.read_csv(mpath(f"phase4_rankings_{side}_all_players_technical.csv"), index_col=0)
    R = pd.read_parquet(mpath(f"phase5_rerankings_{side}.parquet"))
    R.columns = R.columns.astype(int)
    t["pct_top10"] = pd.Series({p: 100 * (R[p] <= 10).mean() for p in R.columns})
    t["median_rank_boot"] = pd.Series({p: R[p].median() for p in R.columns})
    return t


def row_table(t, side, rows):
    p = lambda x: round(100 * x, 3)
    out = pd.DataFrame({
        "role": "batter" if side == "bat" else "bowler", "rank (eligible)": rows.rank_eligible.astype(int), "player": rows["name"],
        "cautious rating (pts)": p(rows.cautious), "best estimate (pts)": p(rows.projected), "80% range low": p(rows.cautious), "80% range high": p(rows.hi80),
        "own-data %": (100 * rows.own_share).round(1), "% of re-rankings in top 10": rows.pct_top10.round(1), "median rank (re-rankings)": rows.median_rank_boot,
        "matches": rows.matches_all.astype(int), "matches 2025-26": rows.matches_2025_26.astype(int),
        "% strict-set LI powerplay": (100 * rows.share_Powerplay).round(1), "% middle": (100 * rows.share_Middle).round(1), "% death": (100 * rows.share_Death).round(1),
        "% IPL": (100 * rows.share_IPL).round(1), "strict-set pressure record (pts)": p(rows.raw_value), "strict-set balls": rows.n_p.astype(int)})
    ten = t[t.rank_eligible == 10].iloc[0]
    out["flag: missed a full season"] = np.where(rows["Seasons with 0 matches since debut in data"] > 0,
                                                 ["yes (" + ", ".join(str(y) for y in range(2022, 2027) if r[f"Matches {y}"] == 0 and y >= first) + ")"
                                                  for (_, r), first in zip(rows.iterrows(), [next((y for y in range(2022, 2027) if r[f"Matches {y}"] > 0), 2022) for _, r in rows.iterrows()])], "no")
    out["flag: negative strict record"] = np.where(rows.raw_value < 0, [f"yes ({100*v:+.3f} pts over {int(n)} balls)" for v, n in zip(rows.raw_value, rows.n_p)], "no")
    out["flag: close call (<50% of re-rankings in top 10)"] = np.where(rows.pct_top10 < 50, "yes", "no")
    out["% in top 10 minus final #10's"] = (rows.pct_top10 - ten.pct_top10).round(1)
    return out


def tables():
    tops, alts, ts = [], [], {}
    for side in ("bat", "bowl"):
        t = load_role(side); ts[side] = t
        el = t[t.eligible == True].sort_values("rank_eligible")
        tops.append(row_table(t, side, el.head(10)))
        ten = el[el.rank_eligible == 10].iloc[0]
        alt = el[(el.rank_eligible > 10) & (el.pct_top10 > ten.pct_top10)]
        alts.append(row_table(t, side, alt))
    top = pd.concat(tops); alt = pd.concat(alts)
    top.to_csv(os.path.join(FIN, "final_top10.csv"), index=False)
    alt.to_csv(os.path.join(FIN, "alternates.csv"), index=False)
    pd.DataFrame({"role": top.role, "rank": top["rank (eligible)"], "player": top.player, "selectable on Unstop (yes/no)": ""}).to_csv(
        os.path.join(FIN, "unstop_check.csv"), index=False)
    return top, alt, ts


# ------------------------------------------------------------------ key numbers
def metrics(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12); nt = y != 0.5
    return {"brier": float(((p - y) ** 2).mean()), "logloss": float((-(y * np.log(p) + (1 - y) * np.log(1 - p))).mean()),
            "auc_ties_excluded": float(roc_auc_score(y[nt], p[nt]))}


def grab(path, pattern, cast=float, n=1):
    txt = open(path, encoding="utf-8").read()
    m = re.findall(pattern, txt)
    return [cast(x) for x in m[:n]] if m else None


def key_numbers(top, alt, ts):
    cfg1a = json.load(open(mpath("phase1a_config.json")))
    fin1a = json.load(open(mpath("phase1a_final.json")))
    inp1b = json.load(open(mpath("phase1b_recursion_inputs.json")))
    chk1f = json.load(open(mpath("phase1f_checks.json")))
    cfg2 = json.load(open(mpath("phase2_pressure_config.json")))
    cfg3 = json.load(open(mpath("phase3_config.json")))
    p5c = json.load(open(os.path.join(REP5, "phase5c_results.json")))
    ce = pd.read_parquet(mpath("phase1b_chase_eval.parquet"))
    ie = pd.read_parquet(mpath("phase1c_inn1_eval.parquet"))
    adj = pd.read_parquet(mpath("phase3_ball_values_adj.parquet"), columns=["li", "innings", "phase"])
    dc = pd.read_csv(os.path.join(OUT, "deliveries_clean.csv"), usecols=["match_id"])
    mc = pd.read_csv(os.path.join(OUT, "match_context_enriched.csv"), usecols=["match_id", "has_ball_data"])
    comp = {}
    for thr in (1.0, 1.5, 2.0):
        inset = adj.li >= thr
        g = adj[inset].groupby(["innings", "phase"]).li.agg(["size", "sum"])
        comp[f"LI>={thr}"] = {"share_of_all_balls": float(inset.mean()),
                              **{f"inn{i}_{ph}_LI_share": float(g.loc[(i, ph), "sum"] / g["sum"].sum()) if (i, ph) in g.index else 0.0
                                 for i in (1, 2) for ph in PHASES}}
    fw = pd.concat([pd.read_csv(os.path.join(REP5, f"phase5_forward_tests_{s}.csv")) for s in ("bat", "bowl")])
    wide = pd.read_csv(os.path.join(REP5, "phase5b_wider_pools.csv"))
    fwd = {f"{r.side}|{r.target}|{r.score}": {"n": int(r.n_players), "spearman": float(r.spearman), "lo95": float(r.lo95), "hi95": float(r.hi95)}
           for r in fw.itertuples()}
    wid = {f"{r.side}|pool>={r.pool}|{r.target}|{r.score}": {"n": int(r.n_players), "spearman": float(r.spearman), "lo95": float(r.lo95), "hi95": float(r.hi95)}
           for r in wide.itertuples()}
    elig = {side: {"ranked": int(len(t)), "eligible": int((t.eligible == True).sum()),
                   "fail_own_share": int((~(t.own_share >= 0.5)).sum()), "fail_matches_2025_26": int((t.matches_2025_26 < 5).sum()),
                   **({"exposure_floor_high_pressure_balls": FLOOR, "fail_exposure_floor": int((t.n_p < FLOOR).sum())} if FLOOR else {})} for side, t in ts.items()}
    p4d = os.path.join(REP, "phase4", "phase4d_summary.md")
    kn = {
        "data": {"datapack_2022plus_deliveries": int(len(dc)), "datapack_matches": int(len(mc)),
                 "model_rows_phase1a": cfg1a["row_log"], "valid_2022plus_balls_with_wpa": chk1f["valid_balls"], "null_balls": chk1f["null_by_reason"]},
        "outcome_model": {"phase1a_chosen": cfg1a["chosen_model"], "pooled_logloss_B_minus_A": cfg1a["pooled_B_minus_A"],
                          "phase1a_fix_B2_minus_B": fin1a["pooled_B2_minus_B"], "phase1a_fix_Brecal_minus_B": fin1a["pooled_Brecal_minus_B"],
                          "final_outcome_model": "plain B (Phase 1b v2: Platt vs plain inconclusive -> plain)"},
        "chase_wp": {"final": inp1b.get("final_wp"), "raw": metrics(ce["raw"].values, ce.y.values), "smooth_g": metrics(ce["smooth"].values, ce.y.values),
                     "n_balls": int(len(ce))},
        "first_innings_wp": {"wp1": metrics(ie.wp1.values, ie.y.values), "logistic_baseline": metrics(ie.logistic.values, ie.y.values), "n_balls": int(len(ie))},
        "per_ball_checks": {"max_abs_sum_wpa_minus_end_start": chk1f["max_abs_sum_wpa_minus_(end-start)"], "li_raw_mean": chk1f["li_raw_mean_scale"]},
        "pressure": {"li_p80": float(cfg2["flags_descriptive_only"]["pressure_main"].split("=")[-1]),
                     "li_p90": float(cfg2["flags_descriptive_only"]["pressure_p90"].split("=")[-1]), "composition": comp},
        "bias_corrections": {"opponent_beta": cfg3["notes"]["opponent_beta"], "tier_gamma_bat": cfg3["notes"].get("tier_gamma_bat"),
                             "tier_gamma_bowl": cfg3["notes"].get("tier_gamma_bowl"), "dew_4way": cfg3["dew"]},
        "eligibility": elig,
        "final_top10": {side: [{"rank": int(r["rank (eligible)"]), "player": r.player, "cautious_pts": float(r["cautious rating (pts)"]),
                                "pct_top10": float(r["% of re-rankings in top 10"])} for _, r in top[top.role == ("batter" if side == "bat" else "bowler")].iterrows()]
                        for side in ("bat", "bowl")},
        "alternates": {side: [{"rank": int(r["rank (eligible)"]), "player": r.player, "pct_top10": float(r["% of re-rankings in top 10"])}
                              for _, r in alt[alt.role == ("batter" if side == "bat" else "bowler")].iterrows()] for side in ("bat", "bowl")},
        "forward_test_phase5": fwd, "forward_test_phase5b_wider": wid,
        "phase5c": p5c,
        "rejected_runs": {
            "4d_bat_eligible_under_P1": grab(p4d, r"Eligibility: (\d+) of 881 eligible", int),
            "4d_bowl_strict_target_reliability": grab(p4d, r"Target \(held-out LI-weighted value on this set\) odd/even reliability (-?[\d.]+) \(127 players\)"),
            "availability_effect_user_reported_bowlers": [0.00003, -0.00089, 0.00096]},
    }
    json.dump(kn, open(os.path.join(FIN, "key_numbers.json"), "w"), indent=1, default=float)
    return kn


# ------------------------------------------------------------------ figures
def style(ax):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(MUTED); ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8.5)
    ax.grid(axis="x", color="#ececea", lw=0.8); ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300, bbox_inches="tight")


def figures(ts, kn):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    # fig1 pipeline
    steps = ["Ball-by-ball data\n(2005-26, 678k balls)", "Outcome model\n(LightGBM, 9 outcomes)", "Win probability\n(recursion + calibration)",
             "Importance (LI)\nexpected WP swing", "Value per ball\n(WPA, centred,\nopponent + tier corrected)", "Player rating\n(empirical Bayes,\nuncertainty-averaged)", "Rank\n(cautious rating,\neligibility)"]
    fig, ax = plt.subplots(figsize=(13, 2.6)); ax.axis("off")
    w, gap = 1.55, 0.35
    for i, s_ in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), w, 1, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#eef4fc" if i < 6 else BLUE, ec=BLUE, lw=1.2))
        ax.text(x + w / 2, 0.5, s_, ha="center", va="center", fontsize=8.3, color="white" if i == 6 else INK)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.03, 0.5), xytext=(x + w + 0.03, 0.5), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))
    ax.set_xlim(-0.1, len(steps) * (w + gap)); ax.set_ylim(-0.15, 1.15)
    save(fig, "fig1_pipeline"); plt.close(fig)
    # fig2 LI distribution
    adj = pd.read_parquet(mpath("phase3_ball_values_adj.parquet"), columns=["li", "innings", "phase"])
    cells = [(i, ph) for i in (1, 2) for ph in PHASES]
    data = [np.clip(adj[(adj.innings == i) & (adj.phase == ph)].li.values, 0.01, None) for i, ph in cells]
    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot(data, vert=False, whis=(1, 99), showfliers=False, widths=0.55, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    for b, (i, _) in zip(bp["boxes"], cells):
        b.set(facecolor=BLUE if i == 2 else "#9ec5f4", edgecolor=BLUE)
    ax.set_xscale("log"); ax.set_yticks(range(1, 7)); ax.set_yticklabels([f"{'1st innings' if i == 1 else 'Chase'} - {ph}" for i, ph in cells])
    ax.axvline(1, color=MUTED, ls="--", lw=0.8); ax.text(1.03, 6.45, "average ball (LI = 1)", fontsize=7.5, color=MUTED)
    ax.set_xlabel("leverage index (log scale; box = middle 50%, whiskers = 1st-99th percentile)")
    ax.set_title("Ball importance by innings and phase, 2022-26", fontsize=10.5, color=INK, loc="left"); style(ax)
    save(fig, "fig2_li_by_phase"); plt.close(fig)
    # fig3 pressure definition
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    labels = [f"{'1st inn' if i == 1 else 'Chase'} {ph}" for i, ph in cells]
    for row, thr in enumerate((1.0, 1.5, 2.0)):
        inset = adj.li >= thr
        cnt = adj[inset].groupby(["innings", "phase"]).size()
        sh = np.array([cnt.get((i, ph), 0) for i, ph in cells], float); sh = sh / sh.sum()
        left = 0
        for k_, v in enumerate(sh):
            ax.barh(row, v, left=left, color=CAT[k_], edgecolor="white", lw=1.5, height=0.6, label=labels[k_] if row == 0 else None)
            if v > 0.06:
                ax.text(left + v / 2, row, f"{100*v:.0f}%", ha="center", va="center", fontsize=8, color="white")
            left += v
    ax.set_yticks(range(3)); ax.set_yticklabels([f"LI >= {t:g}\n({100*kn['pressure']['composition'][f'LI>={t}']['share_of_all_balls']:.0f}% of balls)" for t in (1.0, 1.5, 2.0)])
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("share of balls in the pressure set")
    ax.legend(ncol=6, fontsize=7.3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.set_title("Where 'pressure balls' come from under three definitions", fontsize=10.5, color=INK, loc="left"); style(ax)
    save(fig, "fig3_pressure_definition"); plt.close(fig)
    # fig4 forest
    for side, t in ts.items():
        el = t[t.eligible == True].sort_values("rank_eligible")
        ten = el[el.rank_eligible == 10].iloc[0]
        top = el.head(25).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8.5, 8.8)); y = np.arange(len(top))
        for yy, (_, r) in zip(y, top.iterrows()):
            alt = r.rank_eligible > 10 and r.pct_top10 > ten.pct_top10
            c = BLUE if r.rank_eligible <= 10 else (ORANGE if alt else GREY)
            ax.hlines(yy, 100 * r.cautious, 100 * r.hi80, color=c, lw=2.2)
            ax.scatter(100 * r.projected, yy, s=34, color=c, edgecolor="white", lw=1.2, zorder=3)
            ax.scatter(100 * r.cautious, yy, s=40, marker="|", color=INK, lw=1.8, zorder=4)
        ax.set_yticks(y); ax.set_yticklabels([f"{int(r)}. {n}" + ("  (alternate)" if (r > 10 and p_ > ten.pct_top10) else "")
                                              for r, n, p_ in zip(top.rank_eligible, top["name"], top.pct_top10)], fontsize=8.5)
        ax.axhline(len(top) - 10.5, color=MUTED, lw=0.8, ls="--")
        ax.text(ax.get_xlim()[1] if False else 100 * top.hi80.max(), len(top) - 10.3, "top 10", ha="right", fontsize=8, color=MUTED)
        ax.set_xlabel("win-probability points per average ball\n(dot = best estimate, bar = 80% range, tick = cautious rating used to rank)")
        ax.set_title(f"Top 25 eligible {'batters' if side == 'bat' else 'bowlers'}: final rating, 2022-26", fontsize=10.5, color=INK, loc="left"); style(ax)
        save(fig, f"fig4_forest_{side}"); plt.close(fig)
    # fig5 stability
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, (side, t) in zip(axs, ts.items()):
        el = t[t.eligible == True].sort_values("rank_eligible").head(15)
        ten = el[el.rank_eligible == 10].iloc[0]
        cols = [BLUE if r <= 10 else (ORANGE if p_ > ten.pct_top10 else GREY) for r, p_ in zip(el.rank_eligible, el.pct_top10)]
        ax.bar(range(len(el)), el.pct_top10, color=cols, edgecolor="white", lw=1)
        ax.axhline(ten.pct_top10, color=MUTED, ls="--", lw=0.8)
        ax.set_xticks(range(len(el))); ax.set_xticklabels([f"{int(r)}. {n}" for r, n in zip(el.rank_eligible, el["name"])], rotation=55, ha="right", fontsize=7.8)
        ax.set_title("Batters" if side == "bat" else "Bowlers", fontsize=10, color=INK, loc="left"); style(ax); ax.grid(axis="y", color="#ececea"); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("% of 500 re-rankings in the top 10")
    fig.suptitle("Rank stability: blue = final top 10, orange = alternates (above final #10's share), dashed = final #10", fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout(); save(fig, "fig5_stability"); plt.close(fig)
    # fig6 forward test (wider pool >= 150)
    wide = pd.read_csv(os.path.join(REP5, "phase5b_wider_pools.csv"))
    wide = wide[wide["pool"] == 150]
    names = {"istar": "final rating (i*)", "strike_rate_or_neg_economy": "strike rate / economy", "total_wpa": "total WPA", "team_win_pct": "team win %"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharex=True)
    for ax, side in zip(axs, ("bat", "bowl")):
        w_ = wide[wide.side == side]
        for k_, (tgt, col) in enumerate(zip(sorted(w_.target.unique()), (BLUE, ORANGE))):
            ww = w_[w_.target == tgt].set_index("score").loc[list(names)]
            yy = np.arange(len(ww)) + (k_ - 0.5) * 0.28
            ax.hlines(yy, ww.lo95, ww.hi95, color=col, lw=2); ax.scatter(ww.spearman, yy, color=col, s=32, zorder=3, edgecolor="white",
                                                                       label=f"{tgt} (n={int(ww.n_players.iloc[0])})")
        ax.axvline(0, color=MUTED, lw=0.8); ax.set_yticks(range(len(names))); ax.set_yticklabels(list(names.values())); ax.invert_yaxis()
        ax.set_title("Batters" if side == "bat" else "Bowlers", fontsize=10, color=INK, loc="left"); style(ax)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    axs[0].set_xlabel("Spearman, 2022-24 score vs 2025-26 target (95% CI)"); axs[1].set_xlabel("Spearman, 2022-24 score vs 2025-26 target (95% CI)")
    fig.suptitle("Forward test (players with leverage-sum >= 150 in both periods)", fontsize=10.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(); save(fig, "fig6_forward_test"); plt.close(fig)
    # fig7 shrinkage
    a = pd.read_parquet(mpath("phase3_ball_values_adj.parquet"), columns=["striker_id", "bowler_id", "v_adj_bat", "v_adj_bowl", "li", "match_id"])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, side in zip(axs, ("bat", "bowl")):
        t = ts[side]
        pid, v = ("striker_id", "v_adj_bat") if side == "bat" else ("bowler_id", "v_adj_bowl")
        sub = a[a[v].notna()]
        g = sub.groupby(pid).agg(V=(v, "sum"), L=("li", "sum"))
        raw = (g.V / g.L.where(g.L > 0)).reindex(t.index)
        m = t.matches_all
        ax.scatter(m, (100 * raw).clip(-1.0, 1.0), s=7, color=GREY, alpha=0.6, lw=0, label="raw (own data only; clipped at +/-1)")
        ax.scatter(m, 100 * t.projected, s=7, color=BLUE, alpha=0.8, lw=0, label="final best estimate")
        ax.set_xscale("log"); ax.axhline(0, color=MUTED, lw=0.8)
        for nm in ("Mohsin Khan", "Bhuvneshwar Kumar") if side == "bowl" else ():
            if (t["name"] == nm).any():
                i = t.index[t["name"] == nm][0]
                ax.annotate("", xy=(m[i], 100 * t.projected[i]), xytext=(m[i], float(np.clip(100 * raw[i], -1, 1))), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2))
                ax.annotate(nm, (m[i], 100 * t.projected[i]), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK)
        ax.set_xlabel("matches (log scale)"); ax.set_title("Batters" if side == "bat" else "Bowlers", fontsize=10, color=INK, loc="left"); style(ax)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right", markerscale=2)
    axs[0].set_ylabel("rating, win-probability points per average ball")
    for a_ in axs:
        a_.set_ylim(-1.08, 1.08)
    fig.suptitle("Short records are pulled towards average: raw value vs final rating", fontsize=10.5, x=0.01, ha="left", color=INK)
    fig.tight_layout(); save(fig, "fig7_shrinkage"); plt.close(fig)


# ------------------------------------------------------------------ method log
def method_log(kn):
    p5c = kn["phase5c"]
    bat_rec = {str(r["h"]): r for r in p5c["bat"]["recency_pooled"]}; bowl_rec = {str(r["h"]): r for r in p5c["bowl"]["recency_pooled"]}
    pooled_sr = [r for r in p5c["bat"]["strike_rate_vs_istar"] if r["test"].startswith("2022-24")]
    L = ["# Method log (one line per run, in order)", "",
         "1. Phase 0 - history rebuilt from Cricsheet 2005-21 for training; kept.",
         f"2. Phase 1a - outcome model: LightGBM (B) vs multinomial logistic (A): pooled log loss B - A = {kn['outcome_model']['pooled_logloss_B_minus_A'][0]:+.5f} "
         f"[{kn['outcome_model']['pooled_logloss_B_minus_A'][1]:+.5f}, {kn['outcome_model']['pooled_logloss_B_minus_A'][2]:+.5f}]; B kept.",
         f"3. Phase 1a-fix - impact-player flag + recency weights (B2): B2 - B = {kn['outcome_model']['phase1a_fix_B2_minus_B'][0]:+.5f} "
         f"[{kn['outcome_model']['phase1a_fix_B2_minus_B'][1]:+.5f}, {kn['outcome_model']['phase1a_fix_B2_minus_B'][2]:+.5f}]; rejected. Platt recalibration: "
         f"{kn['outcome_model']['phase1a_fix_Brecal_minus_B'][0]:+.5f} [{kn['outcome_model']['phase1a_fix_Brecal_minus_B'][1]:+.5f}, {kn['outcome_model']['phase1a_fix_Brecal_minus_B'][2]:+.5f}]; later dropped (Phase 1b v2: calibrated chase log loss Platt - plain +0.00454 [-0.00076, +0.00961]).",
         "4. Phase 1b v1 - chase WP recursion; tie valued 0 (bug); rejected and fixed (tie = 0.5; 1 off 1 with 1 wicket 0.593 -> 0.800).",
         "5. Phase 1b v2 - banded calibration map; rejected (jumps up to 0.044 at band edges; martingale gap up to 0.087 at b = 1).",
         f"6. Phase 1b final - smooth calibration map: chase Brier raw {kn['chase_wp']['raw']['brier']:.4f} -> {kn['chase_wp']['smooth_g']['brier']:.4f}, log loss "
         f"{kn['chase_wp']['raw']['logloss']:.4f} -> {kn['chase_wp']['smooth_g']['logloss']:.4f}; kept.",
         f"7. Phase 1c - first-innings WP: Brier {kn['first_innings_wp']['wp1']['brier']:.4f} vs logistic {kn['first_innings_wp']['logistic_baseline']['brier']:.4f}; kept.",
         f"8. Phase 1f - per-ball WPA and LI: per-innings sum check max error {kn['per_ball_checks']['max_abs_sum_wpa_minus_end_start']:.1e}; kept.",
         f"9. Phase 2 - pressure flags at LI p80 = {kn['pressure']['li_p80']:.3f}: innings-1 pressure balls almost all powerplay; replaced by leverage weighting.",
         "10. Phase 3 v1 - adjustments judged by the predictive rule: none kept. Revised: opponent quality and associate tier applied as bias corrections; ground (batting reliability -0.058 [-0.101, -0.027]) and dew (4-way p = 0.67) dropped.",
         "11. Phase 4 - bivariate EB model: lift variance <= 0 for both roles (bat -1.05e-07 [-2.10e-06, +1.39e-06]; bowl -6.55e-07 [-1.63e-06, +3.43e-07]); rating A kept.",
         "12. Phase 4 revised - uncertainty-averaged A (i*); cautious rating = 10th percentile; kept.",
         "13. Phase 4b (run 1) - raw pressure runs per ball chosen by rule; rejected: environment confound (pressure runs per ball 1.21 associate to 1.70 IPL 2026) and bootstrap gave zero spread to 1-match players (coverage 0.70-0.72).",
         "14. Phase 4c (run 2) - context-adjusted features; rejected: pressure set LI >= 1 (40% of balls, 91% of first-innings powerplay) and target on the same set (circular).",
         f"15. Phase 4d (run 3) - strict set LI >= 1.5: batting P1 - i* = +0.049 [-0.085, +0.174], eligible batters {kn['rejected_runs']['4d_bat_eligible_under_P1']}; bowling target reliability "
         f"{kn['rejected_runs']['4d_bowl_strict_target_reliability']}; not usable.",
         f"16. Phase 4e (final) - (i*) for both roles; eligible {kn['eligibility']['bat']['eligible']} batters, {kn['eligibility']['bowl']['eligible']} bowlers; kept."
         + (f" v2: exposure floor adopted - eligible also needs >= {FLOOR} high-pressure balls (LI >= 1.5) in 2022-26 (sensitivity 0/30/50/75/100 in outputs/phase7_exposure_floor/)." if FLOOR else ""),
         f"17. Recency weighting - pooled 3-season Spearman: batters equal {bat_rec['inf (equal)']['mean spearman (3 seasons)']:.3f}, h = 1 {bat_rec['1']['mean spearman (3 seasons)']:.3f} "
         f"(diff {bat_rec['1']['minus equal']:+.3f} [{bat_rec['1']['diff lo95']:+.3f}, {bat_rec['1']['diff hi95']:+.3f}]); bowlers equal {bowl_rec['inf (equal)']['mean spearman (3 seasons)']:.3f}, "
         f"h = 1 {bowl_rec['1']['mean spearman (3 seasons)']:.3f}; rejected (no difference with CI excluding 0).",
         f"18. Availability penalty (< 5 matches previous season) - batters {p5c['bat']['availability']['coef_low_prev']:+.5f} [{p5c['bat']['availability']['lo95']:+.5f}, {p5c['bat']['availability']['hi95']:+.5f}], "
         f"bowlers {p5c['bowl']['availability']['coef_low_prev']:+.5f} [{p5c['bowl']['availability']['lo95']:+.5f}, {p5c['bowl']['availability']['hi95']:+.5f}] (earlier user test: +0.00003 [-0.00089, +0.00096]); rejected.",
         (f"19. Strike rate vs (i*) for batters - pooled 2022-24 -> 2025-26: strict-set target SR - (i*) = {pooled_sr[0]['SR - (i*)']:+.3f} [{pooled_sr[0]['diff lo95']:+.3f}, {pooled_sr[0]['diff hi95']:+.3f}]; "
          f"overall-value target {pooled_sr[1]['SR - (i*)']:+.3f} [{pooled_sr[1]['diff lo95']:+.3f}, {pooled_sr[1]['diff hi95']:+.3f}]; recorded as a finding; score unchanged." if len(pooled_sr) >= 2 else "19. Strike rate vs (i*): see Phase 5c."),
         ""]
    open(os.path.join(FIN, "method_log.md"), "w", encoding="utf-8").write("\n".join(L))


def main():
    os.makedirs(FIG, exist_ok=True)
    top, alt, ts = tables()
    kn = key_numbers(top, alt, ts)
    figures(ts, kn)
    method_log(kn)
    print(top[["role", "rank (eligible)", "player", "cautious rating (pts)", "% of re-rankings in top 10"]].to_string(index=False))
    print(alt[["role", "rank (eligible)", "player", "% of re-rankings in top 10"]].to_string(index=False))


if __name__ == "__main__":
    main()
