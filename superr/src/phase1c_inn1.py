"""Phase 1c - first-innings win probability (team batting first) by recursion, linked to the final chase WP.

State (b, w, s): legal balls remaining, wickets in hand, score so far, innings 1.
Terminal (b = 0 or w = 0): V(s) = 1 - C_{season,tier}(s + 1), C(t) = final chase WP (Phase 1b `wp` column) at the
start of the chase (b = 120, w = 10, r = t). Ties carry through: C already values a tied chase at 0.5.
Live: V(b,w,s) = sum_{k in 0,1,2,3,4,6} p_k V(b-1, w, s+k)
               + p_W [ q_legal sum_j fWL(j) V(b-1, w-1, s+j) + (1 - q_legal) sum_j fWI(j) V(b, w-1, s+j) ]
               + p_WD sum_j fWD(j) V(b, w, s+j) + p_NB sum_j fNB(j) V(b, w, s+j)
p = plain-B outcome probabilities at (innings 1, b, w, r = s, rate = current run rate 6 s / max(120 - b, 1) capped at
36, phase from b, tier, season), from the same leave-one-season-out models as Phase 1b (season s: trained on all
seasons except s); f* and q_legal = the empirical run distributions of those training rows, by phase.
Scores are capped at SMAX = 309 (target 310 = chase table limit). Same-b terms point to higher s (or w-1), so the
order b up, w up, s down solves it exactly; clamping at SMAX and zero-run wides are self-loops, handled by 1/(1 - p_self).
The same recursion with V_terminal(s) = s gives the expected final total.

Limitations: as Phase 1b (free hits ignored, 4 = 4 runs, 6 = 6 runs, 20-over phases); the chase that follows is
always taken as a full 20 overs; WP1 is uncalibrated (no map in this phase).
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from numba import njit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_outcome import load_balls, merge_rare, FULL_MEMBERS, CONFLICT_MATCHES, PHASES, TIERS, SEED, MODEL, REP, OUT
from phase1a_fix import train, frame, FEATS_B
from phase1b_wp import (SEASONS, BMAX, WMAX, RMAX, JMAX, PHASE_B, phase_of_b, run_dists, boot_pairs, reliability, bll,
                        brier)

SMAX = RMAX - 1
N_BOOT = 1000
EXAMPLES = [("50/0 after 6 overs", 84, 10, 50), ("120/3 after 15 overs", 30, 7, 120),
            ("160/7 after 18 overs", 12, 3, 160), ("190/2 after 19 overs", 6, 8, 190)]


@njit(cache=True)
def recurse1(P, term, runs_legal, iW, iWD, iNB, fWD, fNB, fWL, fWI, qL, phase_b):
    """P[b, w, s, k] (b 1..BMAX, w 1..WMAX, s 0..SMAX). Returns V[b, w, s]."""
    B, W, S = P.shape[0], P.shape[1], P.shape[2]
    smax = S - 1
    V = np.zeros((B, W, S))
    for s in range(S):
        for w in range(W):
            V[0, w, s] = term[s]
        for b in range(B):
            V[b, 0, s] = term[s]
    J = fWD.shape[1]
    for b in range(1, B):
        ph = phase_b[b]
        for w in range(1, W):
            for s in range(smax, -1, -1):
                p = P[b, w, s]
                v = 0.0
                for k in range(runs_legal.shape[0]):
                    v += p[k] * V[b - 1, w, min(s + runs_legal[k], smax)]
                swl, swi, same, self_p = 0.0, 0.0, 0.0, 0.0
                for j in range(J):
                    idx = min(s + j, smax)
                    swl += fWL[ph, j] * V[b - 1, w - 1, idx]
                    swi += fWI[ph, j] * V[b, w - 1, idx]
                    pj = p[iWD] * fWD[ph, j] + p[iNB] * fNB[ph, j]
                    if idx == s:
                        self_p += pj
                    else:
                        same += pj * V[b, w, idx]
                v += p[iW] * (qL[ph] * swl + (1.0 - qL[ph]) * swi) + same
                V[b, w, s] = v / (1.0 - self_p)
    return V


def grid1(season, tier):
    b, w, s = np.meshgrid(np.arange(1, BMAX + 1), np.arange(1, WMAX + 1), np.arange(0, SMAX + 1), indexing="ij")
    g = pd.DataFrame({"b": b.ravel().astype(float), "w": w.ravel(), "r": s.ravel().astype(float)})
    g["innings"] = 1
    g["rate"] = np.clip(6 * g.r / np.maximum(BMAX - g.b, 1), 0, 36)
    g["phase"] = np.array(PHASES)[phase_of_b(g.b.values.astype(int))]
    g["format_tier"] = tier
    g["season"] = season
    return g


def inn1_rows_2022():
    d = pd.read_csv(os.path.join(OUT, "deliveries_clean.csv"), low_memory=False,
                    usecols=["match_id", "innings", "innings_ball_num", "is_super_over", "batting_team_id",
                             "balls_remaining", "wkts_in_hand", "score_before", "legal_balls_before", "team_runs"])
    mc = pd.read_csv(os.path.join(OUT, "match_context_enriched.csv"),
                     usecols=["match_id", "comp_name", "match_date", "home_team", "away_team", "dls_affected",
                              "win_team_id", "super_over", "result"])
    ipl = mc.comp_name.str.startswith("IPL")
    full = mc.home_team.isin(FULL_MEMBERS) & mc.away_team.isin(FULL_MEMBERS)
    mc["format_tier"] = np.where(ipl, "IPL", np.where(full, "full_member", "associate"))
    mc["season"] = pd.to_datetime(mc.match_date).dt.year
    mc["tie"] = mc.super_over.astype(bool) | mc.result.fillna("").str.contains("tie", case=False)
    d = d[(d.innings == 1) & ~d.is_super_over.astype(bool)].merge(mc, on="match_id", how="left")
    log = {"inn1_deliveries_2022plus": len(d)}
    m = ~d.dls_affected.astype(bool) & ~d.match_id.isin(CONFLICT_MATCHES); log["excluded_dls_or_conflict"] = int((~m).sum()); d = d[m]
    d = d.assign(y=np.where(d.tie, 0.5, np.where(d.win_team_id.notna(),
                                                  (d.win_team_id == d.batting_team_id).astype(float), np.nan)))
    m = d.y.notna(); log["excluded_no_result"] = int((~m).sum()); d = d[m]
    m = (d.balls_remaining >= 1) & (d.wkts_in_hand >= 1); log["excluded_b_or_w_0"] = int((~m).sum()); d = d[m]
    d = d[d.season.isin(SEASONS)].copy()
    d["b"] = d.balls_remaining.astype(int).clip(1, BMAX)
    d["w"] = d.wkts_in_hand.astype(int)
    d["s"] = np.round(d.score_before).astype(int).clip(0, SMAX)
    d["crr"] = np.clip(6 * d.score_before / np.maximum(d.legal_balls_before, 1), 0, 36)
    d = d.sort_values(["match_id", "innings_ball_num"]).reset_index(drop=True)
    log["rows_evaluated"] = len(d); log["matches_evaluated"] = int(d.match_id.nunique())
    log["tie_matches"] = int(d[d.y == 0.5].match_id.nunique())
    return d, log


def hist_inn1_rows():
    h = pd.read_parquet(os.path.join(MODEL, "history_deliveries.parquet"),
                        columns=["innings", "is_super_over", "dls_affected", "season", "balls_remaining", "wkts_in_hand",
                                 "score_before", "legal_balls_before", "batting_team_won"])
    h = h[(h.innings == 1) & ~h.is_super_over.astype(bool) & ~h.dls_affected.astype(bool) & h.batting_team_won.notna()
          & (h.balls_remaining >= 1) & (h.wkts_in_hand >= 1)]
    return pd.DataFrame({"b": h.balls_remaining.values, "w": h.wkts_in_hand.values, "s": h.score_before.values,
                         "crr": np.clip(6 * h.score_before.values / np.maximum(h.legal_balls_before.values, 1), 0, 36),
                         "y": h.batting_team_won.values, "season": h.season.values})


def logistic_loso(ev, hist):
    X = lambda d: np.column_stack([d.b.values, d.w.values, d.s.values, d.crr.values]).astype(float)
    out = np.zeros(len(ev))
    for s in SEASONS:
        tr = pd.concat([hist, ev.loc[ev.season != s, ["b", "w", "s", "crr", "y", "season"]]])
        x, y = X(tr), tr.y.values; tie = y == 0.5
        XX = np.vstack([x[~tie], x[tie], x[tie]]); Y = np.concatenate([y[~tie], np.zeros(tie.sum()), np.ones(tie.sum())])
        Wt = np.concatenate([np.ones((~tie).sum()), np.full(2 * tie.sum(), 0.5)])
        sc = StandardScaler().fit(XX)
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(XX), Y, sample_weight=Wt)
        sel = (ev.season == s).values
        out[sel] = m.predict_proba(sc.transform(X(ev[sel])))[:, 1]
    return out


def expected_next1(Vd, Pd, dists, rows, classes):
    cidx = {c: i for i, c in enumerate(classes)}
    runs_legal = [int(c) for c in classes if c.isdigit()]
    e = np.zeros(len(rows))
    for (s_, t), ix in rows.groupby(["season", "format_tier"]).groups.items():
        ix = np.asarray(ix); V, P, d = Vd[(s_, t)], Pd[(s_, t)], dists[s_]
        bi, wi, si = rows.b.values[ix], rows.w.values[ix], rows.s.values[ix]
        p = P[bi, wi, si].astype(float); ph = PHASE_B[bi]
        cl = lambda x: np.minimum(x, SMAX)
        v = sum(p[:, cidx[str(c)]] * V[bi - 1, wi, cl(si + c)] for c in runs_legal)
        swl = sum(d["fWL"][ph, j] * V[bi - 1, wi - 1, cl(si + j)] for j in range(JMAX + 1))
        swi = sum(d["fWI"][ph, j] * V[bi, wi - 1, cl(si + j)] for j in range(JMAX + 1))
        swd = sum(d["fWD"][ph, j] * V[bi, wi, cl(si + j)] for j in range(JMAX + 1))
        snb = sum(d["fNB"][ph, j] * V[bi, wi, cl(si + j)] for j in range(JMAX + 1))
        e[ix] = v + p[:, cidx["W"]] * (d["qL"][ph] * swl + (1 - d["qL"][ph]) * swi) + p[:, cidx["WD"]] * swd + p[:, cidx["NB"]] * snb
    return e


def main():
    t0 = time.perf_counter()
    cfg_b = json.load(open(os.path.join(MODEL, "phase1a_final.json")))["B_config"]
    inp = json.load(open(os.path.join(MODEL, "phase1b_recursion_inputs.json")))
    final_wp = inp["final_wp"]
    C = pd.read_parquet(os.path.join(MODEL, "phase1b_wp_tables.parquet"), columns=["season", "tier", "b", "w", "r", "wp"])
    C = C[(C.b == BMAX) & (C.w == WMAX)]
    Cd = {(int(s), t): g.sort_values("r").wp.values for (s, t), g in C.groupby(["season", "tier"], observed=True)}

    b, _ = load_balls(); b, classes, _ = merge_rare(b)
    K = len(classes); cidx = {c: i for i, c in enumerate(classes)}
    y = b.outcome.map(cidx).values
    runs_legal = np.array([int(c) for c in classes if c.isdigit()], dtype=np.int64)
    legal_idx = [cidx[str(c)] for c in runs_legal]; nl = len(legal_idx)

    Vd, Ed, Pd, dists = {}, {}, {}, {}
    for s in SEASONS:
        ts = time.perf_counter()
        tr = (b.season != s).values
        bst = train(b[tr], y[tr], FEATS_B, cfg_b)
        dist = run_dists(b[tr]); dists[s] = dist
        for t in TIERS:
            p = bst.predict(frame(grid1(s, t), FEATS_B))
            P = np.zeros((BMAX + 1, WMAX + 1, SMAX + 1, K))
            P[1:, 1:, :] = p.reshape(BMAX, WMAX, SMAX + 1, K)
            P_re = np.concatenate([P[..., legal_idx], P[..., [cidx["W"], cidx["WD"], cidx["NB"]]]], axis=-1)
            args = (runs_legal, nl, nl + 1, nl + 2, dist["fWD"], dist["fNB"], dist["fWL"], dist["fWI"], dist["qL"], PHASE_B)
            term = 1.0 - Cd[(s, t)][np.minimum(np.arange(SMAX + 1) + 1, RMAX)]
            Vd[(s, t)] = recurse1(P_re, term, *args)
            Ed[(s, t)] = recurse1(P_re, np.arange(SMAX + 1).astype(float), *args)
            Pd[(s, t)] = P.astype(np.float32)
        print(f"season {s}: {time.perf_counter()-ts:.0f}s", flush=True)

    # store
    bb, ww, ss = np.meshgrid(np.arange(BMAX + 1), np.arange(WMAX + 1), np.arange(SMAX + 1), indexing="ij")
    parts = []
    for (s, t) in Vd:
        T = pd.DataFrame({"season": np.int16(s), "tier": t, "b": bb.ravel().astype(np.int16), "w": ww.ravel().astype(np.int8),
                          "s": ss.ravel().astype(np.int16), "wp1": Vd[(s, t)].ravel(), "exp_final_score": Ed[(s, t)].ravel()})
        for k, c in enumerate(classes):
            T[f"p_{c}"] = Pd[(s, t)][..., k].ravel()
        parts.append(T)
    T = pd.concat(parts, ignore_index=True); T["tier"] = T.tier.astype("category")
    T.to_parquet(os.path.join(MODEL, "phase1c_wp1_tables.parquet"), index=False, compression="zstd")
    del T, parts

    # checks: terminal + monotonicity
    A = np.stack([Vd[k] for k in Vd])
    term_ok = int(sum((np.abs(Vd[k][0] - (1 - Cd[k][np.minimum(np.arange(SMAX + 1) + 1, RMAX)])[None, :]) > 1e-12).sum()
                      for k in Vd))
    mono = []
    for name, D, bad in (("increasing_in_s", np.diff(A[:, 1:, 1:, :], axis=3), lambda x: x < -1e-12),
                         ("increasing_in_w", np.diff(A[:, 1:, :, :], axis=2), lambda x: x < -1e-12),
                         ("increasing_in_b", np.diff(A[:, :, 1:, :], axis=1), lambda x: x < -1e-12)):
        v = bad(D); mag = np.abs(np.where(v, D, 0))
        mono.append({"check": name, "pairs": int(v.size), "violations": int(v.sum()),
                     "violations_>1e-3": int((mag > 1e-3).sum()), "max_violation": float(mag.max())})
    mono = pd.DataFrame(mono)

    # evaluation
    ev, elog = inn1_rows_2022()
    v = np.empty(len(ev)); ef = np.empty(len(ev))
    for (s, t), ix in ev.groupby(["season", "format_tier"]).groups.items():
        ix = np.asarray(ix); v[ix] = Vd[(s, t)][ev.b.values[ix], ev.w.values[ix], ev.s.values[ix]]
        ef[ix] = Ed[(s, t)][ev.b.values[ix], ev.w.values[ix], ev.s.values[ix]]
    ev["wp1"], ev["exp_final"] = v, ef
    # martingale
    e = expected_next1(Vd, Pd, dists, ev, classes)
    d = np.abs(e - ev.wp1.values)
    nxt = ev.groupby("match_id").wp1.shift(-1)
    final_score = np.round(ev.score_before + ev.team_runs).astype(int).clip(0, SMAX)
    term_real = np.array([1 - Cd[(s, t)][min(fs + 1, RMAX)] for s, t, fs in zip(ev.season, ev.format_tier, final_score)])
    real = np.where(nxt.isna(), term_real, nxt) - ev.wp1.values
    g = pd.DataFrame({"m": ev.match_id, "d": real}).groupby("m").d.agg(["sum", "size"])
    rng = np.random.default_rng(SEED); idx = rng.integers(0, len(g), (N_BOOT, len(g)))
    reps = g["sum"].values[idx].sum(1) / g["size"].values[idx].sum(1)
    mart = {"balls": len(ev), "mean_abs_E[V_next]-V_now": float(d.mean()), "p99": float(np.percentile(d, 99)),
            "max": float(d.max()), "realised_mean_next_minus_now": float(real.mean()),
            "realised_CI95": [float(x) for x in np.percentile(reps, [2.5, 97.5])]}
    # baseline + metrics
    ev["logistic"] = logistic_loso(ev, hist_inn1_rows())
    met = boot_pairs(ev, [("wp1", "logistic")])
    rel = pd.concat([reliability(ev, "wp1").assign(model="WP1 recursion"),
                     reliability(ev, "logistic").assign(model="logistic (b, w, score, crr)")])
    by_tier = ev.groupby("format_tier").apply(lambda q: pd.Series({
        "n_balls": len(q), "n_matches": q.match_id.nunique(), "brier_wp1": brier(q.wp1, q.y).mean(),
        "brier_logit": brier(q.logistic, q.y).mean(), "logloss_wp1": bll(q.wp1.values, q.y.values).mean(),
        "logloss_logit": bll(q.logistic.values, q.y.values).mean()}), include_groups=False).reset_index()
    ov = pd.cut((BMAX - ev.b) // 6 + 1, [0, 6, 10, 15, 20], labels=["1-6", "7-10", "11-15", "16-20"])
    by_over = ev.assign(overs=ov).groupby("overs", observed=True).apply(
        lambda q: pd.Series({"n": len(q), "brier_wp1": brier(q.wp1, q.y).mean(), "brier_logit": brier(q.logistic, q.y).mean()}),
        include_groups=False).reset_index()
    ev[["match_id", "innings_ball_num", "season", "format_tier", "b", "w", "s", "y", "wp1", "exp_final", "logistic"]].to_parquet(
        os.path.join(MODEL, "phase1c_inn1_eval.parquet"), index=False)

    # worked examples (IPL 2026)
    key = (2026, "IPL"); V, P, dd, Ex = Vd[key], Pd[key], dists[2026], Ed[key]
    exs = []
    for name, b_, w_, s_ in EXAMPLES:
        p = P[b_, w_, s_].astype(float); ph = PHASE_B[b_]; cl = lambda x: min(x, SMAX)
        lines = [(c, p[cidx[c]], f"V({b_-1},{w_},{s_+int(c)})", V[b_ - 1, w_, cl(s_ + int(c))]) for c in classes if c.isdigit()]
        vW = (dd["qL"][ph] * sum(dd["fWL"][ph, j] * V[b_ - 1, w_ - 1, cl(s_ + j)] for j in range(JMAX + 1))
              + (1 - dd["qL"][ph]) * sum(dd["fWI"][ph, j] * V[b_, w_ - 1, cl(s_ + j)] for j in range(JMAX + 1)))
        lines.append(("W", p[cidx["W"]], f"legal (q={dd['qL'][ph]:.4f}) V({b_-1},{w_-1},{s_}+j); illegal V({b_},{w_-1},{s_}+j)", vW))
        for c, kf in (("WD", "fWD"), ("NB", "fNB")):
            lines.append((c, p[cidx[c]], f"E_j V({b_},{w_},{s_}+j), j~{kf}",
                          sum(dd[kf][ph, j] * V[b_, w_, cl(s_ + j)] for j in range(JMAX + 1))))
        exs.append({"name": name, "b": b_, "w": w_, "s": s_, "phase": PHASES[ph], "wp1": V[b_, w_, s_],
                    "exp_final": Ex[b_, w_, s_], "lines": lines, "sum": sum(x[1] * x[3] for x in lines)})
    defend = pd.DataFrame({"first_innings_total": [140, 160, 180, 200, 220, 240],
                           "P(batting first wins) = 1 - C(total+1)": [1 - Cd[key][t + 1] for t in [140, 160, 180, 200, 220, 240]]})
    write_report(final_wp, elog, term_ok, mono, mart, met, rel, by_tier, by_over, exs, defend, time.perf_counter() - t0)


def write_report(final_wp, elog, term_ok, mono, mart, met, rel, by_tier, by_over, exs, defend, rt):
    f = lambda x: f"{x:.4f}" if isinstance(x, float) else x
    L = ["# Phase 1c - first-innings win probability", "",
         f"WP1 = P(team batting first wins) by recursion over plain-B outcome probabilities (LOSO per season), with "
         f"terminal value 1 - C(score + 1), C = final chase WP ({final_wp}) at b = 120, w = 10. Tables: "
         "`out/model/phase1c_wp1_tables.parquet` keyed (season, tier, b, w, s) with wp1, exp_final_score and outcome "
         "probabilities.", "",
         "IPL 2026 defend curve (terminal value):", "", defend.map(f).to_markdown(index=False), "",
         "## Checks", "", f"Terminal values (b = 0 or w = 0) equal to 1 - C(s+1): mismatches = {term_ok}.", "",
         mono.map(lambda x: f"{x:.2e}" if isinstance(x, float) else x).to_markdown(index=False), "",
         "## Martingale (innings 1)", "",
         f"{mart['balls']:,} first-innings balls 2022-26: mean |E_o[V(next)] - V(now)| = {mart['mean_abs_E[V_next]-V_now']:.2e}, "
         f"p99 {mart['p99']:.2e}, max {mart['max']:.2e}.", "",
         f"Realised path (next ball's WP1, or 1 - C(final total + 1) after the last ball): mean change "
         f"{mart['realised_mean_next_minus_now']:+.5f} (95% match-bootstrap CI {mart['realised_CI95'][0]:+.5f} to "
         f"{mart['realised_CI95'][1]:+.5f}).", "",
         "## Calibration 2022-26 (WP1 vs LOSO logistic on b, w, score, crr)", "",
         "Rows: " + "; ".join(f"{k} {v:,}" for k, v in elog.items()), "",
         "Ties: 0.5, in Brier and log loss, excluded from AUC; 1000 match-bootstrap resamples.", "",
         met.map(lambda x: f"{x:.5f}" if isinstance(x, float) else x).to_markdown(index=False), "",
         "Reliability by decile:", "", rel.map(f).to_markdown(index=False), "",
         "By tier:", "", by_tier.map(f).to_markdown(index=False), "",
         "Brier by overs bowled:", "", by_over.astype({"overs": str}).map(f).to_markdown(index=False), "",
         "## Worked examples (IPL 2026)", ""]
    for e in exs:
        L += [f"### {e['name']} (b={e['b']}, w={e['w']}, s={e['s']}, phase {e['phase']}): WP1 = {e['wp1']:.4f}, "
              f"expected total {e['exp_final']:.1f}", "",
              "| outcome | p | continuation | value | p x value |", "|---|---|---|---|---|"]
        for c, p, desc, val in e["lines"]:
            L.append(f"| {c} | {p:.4f} | {desc} | {val:.4f} | {p*val:.4f} |")
        L += [f"| **sum** | {sum(x[1] for x in e['lines']):.4f} | | | **{e['sum']:.4f}** |", ""]
    L += ["## Limitations", "",
          "- As Phase 1b: free hits ignored; class 4 = 4 runs, 6 = 6 runs; phases assume 20 overs.",
          "- The chase after innings 1 is always taken as 20 overs (reduced non-DLS matches are evaluated as if full).",
          "- WP1 carries no calibration map of its own; it inherits the chase map only through the terminal value.",
          "- Plain B under-predicts sixes in 2025-26; Phase 4 per-ball value will be centred by season x phase.",
          '- OPEN (accepted, not tuned): the upper chase tail is under-predicted after g (raw-WP decile 8: g 0.768 vs actual 0.824; decile 9: 0.900 vs 0.930).',
          '- OPEN (accepted, not tuned): first-innings run bias - projected totals at the first ball are 5-11 runs short (IPL 175 vs 186, full member 168 vs 173); realised WP1 drifts +0.00042 per ball [+0.00030, +0.00055]. Player credit uses season x tier x innings x phase centred WPA (Phase 1f).', "",
          f"Runtime {rt:.0f} s.", ""]
    open(os.path.join(REP, "phase1c_summary.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
