"""Phase 1b (final) - replace the banded calibration map with one smooth map that is the identity at b = 0.

  logit g = logit(WP) + a1 x + a2 x^2 + c1 x logit(WP) + c2 x^2 logit(WP),   x = b / 120
Fitted as a binomial GLM (statsmodels) with offset logit(WP) and no intercept; logit(WP) clipped to [-8, 8] for
the fit (offset and interactions). When applied, the offset is the unclipped logit(WP) and the interactions use the
clipped one, so g -> WP as b -> 0 for every WP (differs from the fit only where WP < 3.4e-4 or > 0.99966).
Ties: label 0.5 entered twice (0 and 1) at half weight. Season s: OOF chase balls of seasons < s (2008+), recency
weight 0.5 ** ((s - 1 - season) / H), H in {4, 8, inf} chosen by season s-1 log loss (fitted on < s-1 with weights
relative to s-1). Raw WP = plain-B LOSO recursion (Phase 1b v2). Terminal states stay exact.

Decision: smooth g if acceptance 1 (no band-type jumps) and 2 (martingale) pass and its pooled 2022-26 log loss is
not significantly worse than raw (CI of smooth - raw not entirely above 0); otherwise raw WP.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase1b_wp as W
from phase1a_outcome import load_balls, merge_rare, TIERS, MODEL, REP, OUT
from phase1b_wp import (SEASONS, BMAX, WMAX, RMAX, JMAX, prep_rows, chase_rows_history, chase_rows_2022,
                        terminal_table, expected_next, boot_pairs, bll, brier)

HIST_SEASONS = list(range(2008, 2022))
H_GRID = [4, 8, np.inf]
LCLIP = 8.0
TOL_JUMP = 0.002


def logit_raw(wp):
    p = np.clip(wp, 1e-12, 1 - 1e-12); return np.log(p / (1 - p))


def design(b, L):
    x = np.asarray(b, float) / BMAX; Lc = np.clip(L, -LCLIP, LCLIP)
    return np.column_stack([x, x ** 2, x * Lc, x ** 2 * Lc]), Lc


def fit_glm(rows, ref, H):
    w_rec = np.ones(len(rows)) if np.isinf(H) else 0.5 ** ((ref - rows.season.values) / H)
    L = logit_raw(rows.wp.values); X, Lc = design(rows.b.values, L)
    y = rows.y.values; tie = y == 0.5
    XX = np.vstack([X[~tie], X[tie], X[tie]]); off = np.concatenate([Lc[~tie], Lc[tie], Lc[tie]])
    Y = np.concatenate([y[~tie], np.zeros(tie.sum()), np.ones(tie.sum())])
    Wt = np.concatenate([w_rec[~tie], 0.5 * w_rec[tie], 0.5 * w_rec[tie]])
    m = sm.GLM(Y, XX, family=sm.families.Binomial(), offset=off, freq_weights=Wt).fit()
    return np.asarray(m.params)


def apply_g(theta, b, wp):
    L = logit_raw(wp); X, _ = design(b, L)
    z = L + X @ theta
    g = 1 / (1 + np.exp(-z))
    return np.where((wp <= 0) | (wp >= 1), wp, g)


def fit_season(rows, s):
    tr, va = rows[rows.season < s - 1], rows[rows.season == s - 1]
    sel = {}
    for H in H_GRID:
        th = fit_glm(tr, s - 2, H)
        sel[H] = float(bll(apply_g(th, va.b.values, va.wp.values), va.y.values).mean())
    H = min(sel, key=sel.get)
    th = fit_glm(rows[rows.season < s], s - 1, H)
    return th, {"season": s, "H": "inf" if np.isinf(H) else int(H), **{f"prevLL_H{h}": v for h, v in sel.items()},
                "a1": th[0], "a2": th[1], "c1": th[2], "c2": th[3], "n_fit_balls": int((rows.season < s).sum())}


def table_g(theta, WP):
    G = terminal_table()
    bb = np.broadcast_to(np.arange(BMAX + 1)[:, None, None], WP.shape)
    live = (slice(1, None), slice(1, None), slice(1, None))
    G[live] = apply_g(theta, bb[live].ravel(), WP[live].ravel()).reshape(BMAX, WMAX, RMAX)
    return G


def history_raw_wp(cache):
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    b, _ = load_balls(); b, classes, _ = merge_rare(b)
    y = b.outcome.map({c: i for i, c in enumerate(classes)}).values
    cfg_b = json.load(open(os.path.join(MODEL, "phase1a_final.json")))["B_config"]
    W.ALL_SEASONS = HIST_SEASONS
    WPd, _, _, _ = W.build_variant(b, y, classes, cfg_b, "plain")
    h = prep_rows(chase_rows_history())
    h = h[h.season.isin(HIST_SEASONS)].reset_index(drop=True)
    wp = np.empty(len(h))
    for (s, t), ix in h.groupby(["season", "format_tier"]).groups.items():
        ix = np.asarray(ix); wp[ix] = WPd[(s, t)][h.b.values[ix], h.w.values[ix], h.r.values[ix]]
    h["wp"] = wp
    h.to_parquet(cache, index=False)
    return h


def load_tables():
    T = pd.read_parquet(os.path.join(MODEL, "phase1b_wp_tables_plain.parquet"))
    classes = [c[2:] for c in T.columns if c.startswith("p_")]
    WPd, Bd, Pd = {}, {}, {}
    shp = (BMAX + 1, WMAX + 1, RMAX + 1)
    for (s, t), g in T.groupby(["season", "tier"], observed=True):
        g = g.sort_values(["b", "w", "r"])
        WPd[(s, t)] = g.wp_raw.values.reshape(shp); Bd[(s, t)] = g.wp.values.reshape(shp)
        Pd[(s, t)] = g[[f"p_{c}" for c in classes]].values.reshape(shp + (len(classes),)).astype(np.float32)
    inp = json.load(open(os.path.join(MODEL, "phase1b_recursion_inputs.json")))
    assert inp["chosen_variant"] == "plain"
    dists = {int(s): {k: np.array(v) for k, v in d.items() if k != "n"} for s, d in inp["run_distributions"].items()}
    return WPd, Bd, Pd, dists, classes, inp


def jumps(TAB):
    """Adjacent-b |difference| at fixed (w, r) over live pairs (b-1 >= 1, w >= 1, r >= 1)."""
    A = np.stack([TAB[(s, t)] for s in SEASONS for t in TIERS])[:, 1:, 1:, 1:]
    D = np.abs(np.diff(A, axis=1))                     # pair index i = (b = i+1, b = i+2)
    return D


def main():
    t0 = time.perf_counter()
    WPd, Bd, Pd, dists, classes, inp = load_tables()
    hist = history_raw_wp(os.path.join(MODEL, "phase1b_history_chase_wp_raw.parquet"))
    ev = prep_rows(chase_rows_2022()[0])
    for col, D in (("wp", WPd), ("wp_banded", Bd)):
        v = np.empty(len(ev))
        for (s, t), ix in ev.groupby(["season", "format_tier"]).groups.items():
            ix = np.asarray(ix); v[ix] = D[(s, t)][ev.b.values[ix], ev.w.values[ix], ev.r.values[ix]]
        ev[col] = v
    rows = pd.concat([hist, ev[hist.columns]], ignore_index=True)

    params, Gd = [], {}
    sm_ = np.empty(len(ev))
    for s in SEASONS:
        th, info = fit_season(rows, s); params.append(info)
        m = (ev.season == s).values
        sm_[m] = apply_g(th, ev.b.values[m], ev.wp.values[m])
        for t in TIERS:
            Gd[(s, t)] = table_g(th, WPd[(s, t)])
    ev["wp_smooth"] = sm_
    params = pd.DataFrame(params)
    print(params, flush=True)

    # acceptance 1: jumps
    Dr, Ds, Db = jumps(WPd), jumps(Gd), jumps(Bd)
    i3031 = 29                                          # pair (30, 31)
    acc1 = pd.DataFrame([{"table": n, "max_adjacent_b_diff": float(D.max()),
                          "max_at_b30_31": float(D[:, i3031].max()),
                          "max_excess_over_raw_same_cell": float((D - Dr).max())}
                         for n, D in (("raw", Dr), ("banded g", Db), ("smooth g", Ds))])
    pass1 = bool(Ds.max() <= Dr.max() + TOL_JUMP)

    # acceptance 2: martingale
    mart = []
    for n, TAB, col in (("raw", WPd, "wp"), ("banded g", Bd, "wp_banded"), ("smooth g", Gd, "wp_smooth")):
        d = np.abs(expected_next(TAB, Pd, dists, ev, classes) - ev[col].values)
        for lab, sel in (("all", np.ones(len(ev), bool)), ("b 1-6", ev.b.values <= 6)):
            mart.append({"table": n, "balls": lab, "n": int(sel.sum()), "mean": d[sel].mean(),
                         "p99": np.percentile(d[sel], 99), "max": d[sel].max()})
        if n == "smooth g":
            by_b = pd.DataFrame({"b": ev.b, "d": d})[ev.b <= 6].groupby("b").d.agg(["size", "mean", "max"]).reset_index()
    mart = pd.DataFrame(mart)
    ms = mart[mart.table == "smooth g"].set_index("balls")
    pass2 = bool(ms.loc["all", "mean"] < 0.001 and ms.loc["b 1-6", "max"] < 0.087 / 4)

    # acceptance 3: calibration
    E = ev.rename(columns={"wp": "raw", "wp_banded": "banded", "wp_smooth": "smooth"})
    met = boot_pairs(E, [("smooth", "raw"), ("smooth", "banded"), ("banded", "raw")])
    dll = met[(met.metric == "logloss") & (met.model == "smooth - raw")].iloc[0]
    pass3 = bool(not (dll.lo95 > 0))
    dec = pd.qcut(E.raw.rank(method="first"), 10, labels=False) + 1
    rel = E.assign(decile=dec).groupby("decile").agg(n=("y", "size"), raw=("raw", "mean"), banded=("banded", "mean"),
                                                    smooth=("smooth", "mean"), actual=("y", "mean")).reset_index()
    dec_s = pd.qcut(E.smooth.rank(method="first"), 10, labels=False) + 1
    rel_s = E.assign(decile=dec_s).groupby("decile").agg(smooth=("smooth", "mean"), actual=("y", "mean")).reset_index()
    use_smooth = pass1 and pass2 and pass3
    final = "smooth g" if use_smooth else "raw"
    print(acc1, mart, met, final, sep="\n", flush=True)

    plot_g(params)

    # canonical tables
    T = pd.read_parquet(os.path.join(MODEL, "phase1b_wp_tables_plain.parquet")).rename(columns={"wp": "wp_banded"})
    T = T.sort_values(["season", "tier", "b", "w", "r"]).reset_index(drop=True)
    sm_col = np.empty(len(T))
    for (s, t), ix in T.groupby(["season", "tier"], observed=True).groups.items():
        ix = np.asarray(ix)
        sm_col[ix] = Gd[(s, t)][T.b.values[ix], T.w.values[ix], T.r.values[ix]]
    T["wp_smooth"] = sm_col
    T["wp"] = T.wp_smooth if use_smooth else T.wp_raw
    lead = ["season", "tier", "b", "w", "r", "wp", "wp_raw", "wp_smooth", "wp_banded"]
    T[lead + [c for c in T.columns if c not in lead]].to_parquet(os.path.join(MODEL, "phase1b_wp_tables.parquet"),
                                                                 index=False, compression="zstd")
    inp.update({"final_wp": final, "wp_column_for_WPA_LI": "wp", "smooth_g": {
        "form": "logit g = logit WP + a1 x + a2 x^2 + c1 x clip(logit WP) + c2 x^2 clip(logit WP), x = b/120",
        "params_by_season": params.to_dict("records")}})
    json.dump(inp, open(os.path.join(MODEL, "phase1b_recursion_inputs.json"), "w"), indent=1, default=float)
    E.to_parquet(os.path.join(MODEL, "phase1b_chase_eval.parquet"), index=False)
    write_report(params, acc1, pass1, mart, by_b, pass2, met, pass3, rel, rel_s, final, time.perf_counter() - t0)


def plot_g(params):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    wp = np.linspace(0.001, 0.999, 400)
    fig, ax = plt.subplots(1, 5, figsize=(17, 3.8), sharey=True)
    for a, bb in zip(ax, [6, 30, 60, 90, 120]):
        a.plot([0, 1], [0, 1], color="#bbb", ls="--", lw=1)
        for _, r in params.iterrows():
            th = r[["a1", "a2", "c1", "c2"]].values.astype(float)
            a.plot(wp, apply_g(th, np.full(len(wp), bb), wp), lw=2 if r.season == 2026 else 0.8,
                   alpha=1 if r.season == 2026 else 0.5, label=str(int(r.season)))
        a.set_title(f"b = {bb}"); a.set_xlabel("raw WP")
    ax[0].set_ylabel("smooth g(WP)"); ax[0].legend(fontsize=7)
    fig.suptitle("Phase 1b: smooth calibration map g by balls remaining (one curve per test season; 2026 bold)")
    fig.tight_layout(); fig.savefig(os.path.join(REP, "phase1b_smooth_g.png"), dpi=130); plt.close(fig)


def write_report(params, acc1, pass1, mart, by_b, pass2, met, pass3, rel, rel_s, final, rt):
    f = lambda x: f"{x:.4f}" if isinstance(x, float) else x
    f5 = lambda x: f"{x:.5f}" if isinstance(x, float) else x
    pf = lambda b: "PASS" if b else "FAIL"
    L = ["# Phase 1b (final) - smooth calibration map", "",
         f"**Final chase WP: {final}.** Canonical table `out/model/phase1b_wp_tables.parquet`: `wp` (final, used by "
         "WPA and LI), `wp_raw`, `wp_smooth`, `wp_banded` (v2 map, kept for the record), outcome probabilities.", "",
         "Map: logit g = logit WP + a1 x + a2 x^2 + c1 x L + c2 x^2 L, x = b/120, L = logit WP clipped to [-8, 8]; "
         "binomial GLM with offset, no intercept, ties at half weight; season s fitted on OOF chase balls of seasons < s "
         "(2008+) with recency weight 0.5^((s-1-season)/H), H by season s-1 log loss.", "",
         "## 4. Fitted parameters", "", params.map(f5).to_markdown(index=False), "",
         "![g](phase1b_smooth_g.png)", "",
         f"## 1. Jumps in b - {pf(pass1)}", "",
         f"Rule: max |g(b) - g(b-1)| (live cells, fixed w, r) <= raw max + {TOL_JUMP}.", "",
         acc1.map(f5).to_markdown(index=False), "",
         f"## 2. Martingale after g - {pf(pass2)}", "",
         "Target: mean < 0.001 overall; max at b <= 6 well below 0.087 (taken as < 0.087/4).", "",
         mart.map(lambda x: f"{x:.2e}" if isinstance(x, float) else x).to_markdown(index=False), "",
         "Smooth g, by b:", "", by_b.map(lambda x: f"{x:.2e}" if isinstance(x, float) else x).to_markdown(index=False), "",
         f"## 3. Calibration 2022-26 - smooth log loss not significantly worse than raw: {pf(pass3)}", "",
         "Ties: 0.5, in Brier and log loss, excluded from AUC; 1000 match-bootstrap resamples.", "",
         met.map(f5).to_markdown(index=False), "",
         "Reliability by decile of raw WP (same balls):", "", rel.map(f).to_markdown(index=False), "",
         "Smooth g by its own deciles:", "", rel_s.map(f).to_markdown(index=False), "",
         "## Decision", "",
         f"Acceptance 1 {pf(pass1)}, 2 {pf(pass2)}, log-loss rule {pf(pass3)} -> **{final}**.", "",
         "## Limitations (Phase 1b)", "",
         "- Free hits ignored in the recursion.",
         "- Class 4 = 4 runs (absorbs 5s), class 6 = 6 runs (absorbs 7s); phase assumes a 20-over innings.",
         "- Raw WP is not monotone in b in about 6% of adjacent cells (max ~0.005, mostly early innings, high WP); accepted.",
         "- Plain B under-predicts sixes (and over-predicts dots) in 2025-26 (scoring drift). Phase 4 per-ball player "
         "value will be centred by season x phase.",
         "- History WP used to fit g comes from LOSO models that saw the test season's ball outcomes (never its results).",
         "- After g the table is not an exact martingale (values above).",
         '- OPEN (accepted, not tuned): the upper chase tail is under-predicted after g (raw-WP decile 8: g 0.768 vs actual 0.824; decile 9: 0.900 vs 0.930).',
         '- OPEN (accepted, not tuned): first-innings run bias - projected totals at the first ball are 5-11 runs short (IPL 175 vs 186, full member 168 vs 173); realised WP1 drifts +0.00042 per ball [+0.00030, +0.00055]. Player credit uses season x tier x innings x phase centred WPA (Phase 1f).', "", f"Runtime {rt:.0f} s.", ""]
    open(os.path.join(REP, "phase1b_final_summary.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
