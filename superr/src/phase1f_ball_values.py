"""Phase 1f - per-ball WPA and leverage index (LI) for every 2022+ main-innings delivery.

WP is always from the BATTING team's view:
  innings 1: WP1 (Phase 1c, raw - the optional WP1 map was not fitted, see report), state (b, w, score);
  innings 2: final chase WP (Phase 1b `wp` = smooth g), state (b, w, runs required).
wp_before = table value at the state before the ball. wp_after = wp_before of the next ball in the same innings;
after the last ball of innings 1: 1 - C(final total + 1) (C = chase WP at b = 120, w = 10); after the last chase
ball: the result (1 / 0.5 / 0). WPA = wp_after - wp_before, so per-innings sums telescope exactly.
LI_raw = E_o[|WP(s_o) - WP(s_now)|] under the outcome probabilities and run distributions of the table;
LI = LI_raw / mean(LI_raw over all valid 2022+ balls) (one scale for both innings).
wpa_centred = WPA - mean(WPA | season, tier, innings, phase) over valid balls (removes the run / six drift).
DLS-affected, conflict (17765) and no-result matches: WP, WPA, LI = null with missing_reason.
"""
import json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_outcome import FULL_MEMBERS, CONFLICT_MATCHES, TIERS, MODEL, REP, OUT
from phase1b_wp import SEASONS, BMAX, WMAX, RMAX, JMAX, PHASE_B

SMAX = RMAX - 1
SI = {s: i for i, s in enumerate(SEASONS)}; TI = {t: i for i, t in enumerate(TIERS)}


def load_table(path, key, val, n_key):
    T = pd.read_parquet(path)
    classes = [c[2:] for c in T.columns if c.startswith("p_")]
    shp = (len(SEASONS), len(TIERS), BMAX + 1, WMAX + 1, n_key)
    V = np.zeros(shp); P = np.zeros(shp + (len(classes),), dtype=np.float32)
    T = T.sort_values(["season", "tier", "b", "w", key])
    for (s, t), g in T.groupby(["season", "tier"], observed=True):
        V[SI[int(s)], TI[t]] = g[val].values.reshape(shp[2:])
        P[SI[int(s)], TI[t]] = g[[f"p_{c}" for c in classes]].values.reshape(shp[2:] + (len(classes),))
    return V, P, classes


def li_raw(V, P, classes, dists, si, ti, bi, wi, xi, sign, xmax):
    """E_o |V(next) - V(now)|; sign +1 (innings 1: score goes up) or -1 (chase: runs required go down)."""
    cidx = {c: i for i, c in enumerate(classes)}
    nxt = lambda x, j: np.clip(x + sign * j, 0, xmax)
    now = V[si, ti, bi, wi, xi]
    p = P[si, ti, bi, wi, xi].astype(float); ph = PHASE_B[bi]
    f = {k: np.stack([dists[SEASONS[i]][k] for i in si])[np.arange(len(si)), ph] for k in ("fWD", "fNB", "fWL", "fWI")}
    qL = np.array([dists[SEASONS[i]]["qL"] for i in si])[np.arange(len(si)), ph]
    out = np.zeros(len(si))
    for c in classes:
        if c.isdigit():
            out += p[:, cidx[c]] * np.abs(V[si, ti, bi - 1, wi, nxt(xi, int(c))] - now)
    for j in range(JMAX + 1):
        out += p[:, cidx["W"]] * (qL * f["fWL"][:, j] * np.abs(V[si, ti, bi - 1, wi - 1, nxt(xi, j)] - now)
                                  + (1 - qL) * f["fWI"][:, j] * np.abs(V[si, ti, bi, wi - 1, nxt(xi, j)] - now))
        out += (p[:, cidx["WD"]] * f["fWD"][:, j] + p[:, cidx["NB"]] * f["fNB"][:, j]) * np.abs(V[si, ti, bi, wi, nxt(xi, j)] - now)
    return out


def main():
    t0 = time.perf_counter()
    dcols = ["match_id", "innings", "overid", "over_ball_num", "innings_ball_num", "batting_team_id", "bowling_team_id",
             "striker_id", "striker_name", "non_striker_id", "non_striker_name", "bowler_id", "bowler_name", "batting_type",
             "bowling_type", "pace_or_spin", "batting_order", "bat_runs", "is_four", "is_six", "is_wicket", "bowler_wicket",
             "dismissal_type", "dismissed_player_id", "extras_type", "is_legal", "is_wide", "is_noball", "is_bye",
             "batter_ball_faced", "extras_runs", "team_runs", "bowler_runs_conceded", "is_super_over", "phase",
             "balls_remaining", "wkts_in_hand", "score_before", "runs_required", "legal_balls_before", "scheduled_balls"]
    d = pd.read_csv(os.path.join(OUT, "deliveries_clean.csv"), usecols=dcols, low_memory=False)
    mc = pd.read_csv(os.path.join(OUT, "match_context_enriched.csv"),
                     usecols=["match_id", "comp_name", "match_date", "home_team", "away_team", "dls_affected",
                              "win_team_id", "super_over", "result"])
    ipl = mc.comp_name.str.startswith("IPL")
    full = mc.home_team.isin(FULL_MEMBERS) & mc.away_team.isin(FULL_MEMBERS)
    mc["tier"] = np.where(ipl, "IPL", np.where(full, "full_member", "associate"))
    mc["season"] = pd.to_datetime(mc.match_date).dt.year
    mc["tie"] = mc.super_over.astype(bool) | mc.result.fillna("").str.contains("tie", case=False)
    d = d[(d.innings <= 2) & ~d.is_super_over.astype(bool)].merge(mc, on="match_id", how="left")
    d = d.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    reason = np.select([d.dls_affected.astype(bool), d.match_id.isin(CONFLICT_MATCHES),
                        d.win_team_id.isna() & ~d.tie, ~d.season.isin(SEASONS)],
                       ["dls_affected", "conflict_match", "no_result", "season_outside_tables"], "")
    d["missing_reason"] = np.where(reason == "", None, reason)
    ok = d.missing_reason.isna().values

    # tables
    VC, PC, classes = load_table(os.path.join(MODEL, "phase1b_wp_tables.parquet"), "r", "wp", RMAX + 1)
    V1, P1, classes1 = load_table(os.path.join(MODEL, "phase1c_wp1_tables.parquet"), "s", "wp1", SMAX + 1)
    assert classes == classes1
    inp = json.load(open(os.path.join(MODEL, "phase1b_recursion_inputs.json")))
    dists = {int(s): {k: np.array(v) for k, v in dd.items() if k != "n"} for s, dd in inp["run_distributions"].items()}

    si = d.season.map(SI).fillna(0).astype(int).values; ti = d.tier.map(TI).fillna(0).astype(int).values
    bi = d.balls_remaining.clip(0, BMAX).astype(int).values; wi = d.wkts_in_hand.clip(0, WMAX).astype(int).values
    s1 = np.round(d.score_before).clip(0, SMAX).astype(int).values
    r2 = np.round(d.runs_required.fillna(0)).clip(0, RMAX).astype(int).values
    inn1 = (d.innings == 1).values
    wpb = np.where(inn1, V1[si, ti, bi, wi, s1], VC[si, ti, bi, wi, r2])

    # LI (live states only: b >= 1, w >= 1, and r >= 1 for the chase)
    li = np.full(len(d), np.nan)
    live = ok & (bi >= 1) & (wi >= 1) & (inn1 | (r2 >= 1))
    m = live & inn1
    li[m] = li_raw(V1, P1, classes, dists, si[m], ti[m], bi[m], wi[m], s1[m], +1, SMAX)
    m = live & ~inn1
    li[m] = li_raw(VC, PC, classes, dists, si[m], ti[m], bi[m], wi[m], r2[m], -1, RMAX)
    li[ok & ~live] = 0.0

    # wp_after
    d["wp_before"] = wpb
    nxt = d.groupby(["match_id", "innings"]).wp_before.shift(-1)
    last = nxt.isna().values
    final = np.round(d.score_before + d.team_runs).clip(0, SMAX).astype(int).values
    end1 = 1 - VC[si, ti, BMAX, WMAX, np.minimum(final + 1, RMAX)]
    res2 = np.where(d.tie, 0.5, (d.win_team_id == d.batting_team_id).astype(float))
    d["wp_after"] = np.where(~last, nxt, np.where(inn1, end1, res2))
    d["wpa"] = d.wp_after - d.wp_before
    d["li_raw"] = li
    for c in ("wp_before", "wp_after", "wpa", "li_raw"):
        d.loc[~ok, c] = np.nan
    li_mean = float(d.loc[ok, "li_raw"].mean())
    d["li"] = d.li_raw / li_mean
    cell = ["season", "tier", "innings", "phase"]
    cm = d[ok].groupby(cell).wpa.agg(n="size", mean_wpa="mean").reset_index()
    d = d.merge(cm[cell + ["mean_wpa"]], on=cell, how="left")
    d["wpa_centred"] = d.wpa - d.mean_wpa
    d = d.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

    # checks
    v = d[d.missing_reason.isna()]
    g = v.groupby(["match_id", "innings"])
    tele = (g.wpa.sum() - (g.wp_after.last() - g.wp_before.first())).abs()
    cen = v.groupby(cell).wpa_centred.mean().abs()
    inn_break = (v[v.innings == 1].groupby("match_id").wp_after.last().rename("a")
                 .to_frame().join(v[v.innings == 2].groupby("match_id").wp_before.first().rename("c"), how="inner"))
    gap = (1 - inn_break.a - inn_break.c).abs()
    qs = lambda x: pd.Series({"n": len(x), "median": x.median(), "p80": x.quantile(.8), "p90": x.quantile(.9),
                              "p99": x.quantile(.99), "max": x.max(), "mean": x.mean()})
    li_all = qs(v.li).to_frame("all").T
    li_ip = v.groupby(["innings", "phase"]).li.apply(qs).unstack().reset_index()
    checks = {"valid_balls": int(len(v)), "null_balls": int((~d.missing_reason.isna()).sum()),
              "null_by_reason": d.missing_reason.value_counts().to_dict(),
              "innings_checked": int(len(tele)), "max_abs_sum_wpa_minus_(end-start)": float(tele.max()),
              "max_abs_mean_wpa_centred_by_cell": float(cen.max()), "li_raw_mean_scale": li_mean,
              "innings_break_matches": int(len(gap)),
              "innings_break_gap_|1-WP1_end - WPchase_start|_median": float(gap.median()),
              "innings_break_gap_p99": float(gap.quantile(.99)), "innings_break_gap_max": float(gap.max()),
              "innings_break_gap_>0.01": int((gap > 0.01).sum())}

    out_cols = ["match_id", "innings", "overid", "over_ball_num", "innings_ball_num", "batting_team_id", "bowling_team_id",
                "striker_id", "striker_name", "non_striker_id", "non_striker_name", "bowler_id", "bowler_name",
                "batting_type", "bowling_type", "pace_or_spin", "batting_order", "phase", "season", "tier", "comp_name",
                "match_date", "scheduled_balls", "balls_remaining", "wkts_in_hand", "score_before", "runs_required",
                "wp_before", "wp_after", "wpa", "wpa_centred", "li", "li_raw", "bat_runs", "team_runs", "extras_runs",
                "bowler_runs_conceded", "is_legal", "is_wide", "is_noball", "is_bye", "batter_ball_faced", "is_four",
                "is_six", "is_wicket", "bowler_wicket", "dismissal_type", "dismissed_player_id", "extras_type",
                "missing_reason"]
    d[out_cols].rename(columns={"dismissed_player_id": "dismissed_id"}).to_parquet(
        os.path.join(MODEL, "phase1f_ball_values.parquet"), index=False, compression="zstd")
    json.dump(checks, open(os.path.join(MODEL, "phase1f_checks.json"), "w"), indent=1, default=float)
    write_report(checks, cm, li_all, li_ip, time.perf_counter() - t0)
    print(json.dumps(checks, indent=1, default=float))


def write_report(checks, cm, li_all, li_ip, rt):
    f = lambda x: f"{x:.5f}" if isinstance(x, float) else x
    cmw = cm.assign(mean_wpa=cm.mean_wpa.map(lambda x: f"{x:+.5f}"))
    pivot = cm.pivot_table(index=["tier", "innings"], columns="phase", values="mean_wpa", aggfunc="mean")[["Powerplay", "Middle", "Death"]]
    pooled = cm.assign(t=cm.n * cm.mean_wpa).groupby(["innings", "phase"]).apply(
        lambda g: pd.Series({"n": g.n.sum(), "mean_wpa": g.t.sum() / g.n.sum()}), include_groups=False).reset_index()
    L = ["# Phase 1f - per-ball WPA and leverage", "",
         "Output: `out/model/phase1f_ball_values.parquet` (one row per 2022+ main-innings delivery). WP from the batting "
         "team's view: innings 1 = WP1 (raw; optional WP1 map not fitted, see below), innings 2 = chase WP (smooth g). "
         "LI = E_o|WP(next) - WP(now)| / its mean over all valid 2022+ balls. wpa_centred = WPA - mean(WPA | season, tier, "
         "innings, phase).", "",
         "Optional WP1 smooth map: **not fitted.** Fitting it on earlier seasons needs out-of-fold WP1 on 2008-21 first "
         "innings, whose terminal value is the calibrated chase WP of that season; no calibrated chase tables exist "
         "before 2022 (the chase map itself needs earlier seasons), so a like-for-like fit is not possible within the "
         "30-minute cap. WP1 stays raw.", "",
         "## Checks", "", pd.DataFrame([{"check": k, "value": v} for k, v in checks.items()]).map(f).to_markdown(index=False), "",
         "The innings-break gap is between 1 - (WP1 after the last first-innings ball) and the chase WP before the first "
         "chase ball; it is non-zero only where the recorded target differs from final total + 1 (estimated extras, "
         "penalty runs) - per-innings sums are unaffected.", "",
         "## Mean WPA per cell before centring (season x tier x innings x phase)", "",
         "Pooled over seasons and tiers:", "", pooled.map(f).to_markdown(index=False), "",
         "Mean of the season cells by tier x innings:", "", pivot.map(lambda x: f"{x:+.5f}").reset_index().to_markdown(index=False), "",
         "All cells:", "", cmw.to_markdown(index=False), "",
         "## LI distribution", "", li_all.map(f).to_markdown(), "", li_ip.map(f).to_markdown(index=False), "",
         f"Runtime {rt:.0f} s.", ""]
    open(os.path.join(REP, "phase1f_summary.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
