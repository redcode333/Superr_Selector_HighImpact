"""Phase 1a-fix - scoring drift (sixes under-, dots over-predicted on 2025-26).

B2 = B + feature impact_player_era (1 for IPL seasons >= 2023, else 0) + recency sample weights
     w = 0.5 ** ((test_season - 1 - season) / H), H in {2, 4, 8, none}, chosen on the 2021 validation only
     (train < 2021, validate 2021, B's fixed num_leaves / min_child_samples, early stopping picks the rounds).
Same expanding-window folds (2022..2026) and the same decision rule as 1a: keep B2 only if pooled 2022-26
log loss B2 - B < 0 with the 95% match-bootstrap CI excluding 0.

Six-bias "fixed" (stated before looking at the result): on 2025-26 pooled, for both class 6 and class 0, the
mean predicted probability lies inside the 95% match-bootstrap CI of the actual rate, AND the sign of
(predicted - actual) is not the same in all 10 deciles.

If neither B nor B2 fixes it: keep B and recalibrate per class with Platt scaling (one-vs-rest logistic on
logit p_k) fitted on the PREVIOUS season inside each fold (fold t: B trained on seasons < t-1, predicted on t-1,
Platt fitted there; applied to the fold-t predictions of B trained on < t), then renormalised to sum 1.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_outcome import (load_balls, merge_rare, lgb_params, lgb_frame, ball_logloss, boot_diff, FEATS_B,
                             PHASES, TIERS, SEED, MODEL, REP)

H_GRID = [2, 4, 8, None]
FOLDS = [2021, 2022, 2023, 2024, 2025, 2026]          # 2021 only feeds the recalibration of fold 2022
TEST = FOLDS[1:]
FEATS_B2 = FEATS_B + ["impact_player_era"]


def add_era(b):
    b["impact_player_era"] = ((b.format_tier == "IPL") & (b.season >= 2023)).astype(int)
    return b


def frame(df, feats):
    x = lgb_frame(df)
    if "impact_player_era" in feats:
        x["impact_player_era"] = df.impact_player_era.values
    return x[feats]


def weights(season, ref_season, H):
    """Recency weight; ref_season = test_season - 1 (expanding) or the LOSO season (distance-based, see 1b)."""
    if H is None:
        return None
    return 0.5 ** ((ref_season - season) / H)


def train(df, y, feats, cfg, w=None, n_class=9):
    import lightgbm as lgb
    p = lgb_params(int(cfg["num_leaves"]), int(cfg["min_child_samples"]), n_class)
    ds = lgb.Dataset(frame(df, feats), y, weight=w, categorical_feature=["phase_c", "tier_c"])
    return lgb.train(p, ds, int(cfg["best_iteration"]))


def choose_H(b, y, cfg_b, K):
    import lightgbm as lgb
    tr, va = (b.season < 2021).values, (b.season == 2021).values
    res = []
    for H in H_GRID:
        p = lgb_params(int(cfg_b["num_leaves"]), int(cfg_b["min_child_samples"]), K)
        w = weights(b.season.values[tr], 2020, H)
        dtr = lgb.Dataset(frame(b[tr], FEATS_B2), y[tr], weight=w, categorical_feature=["phase_c", "tier_c"])
        dva = lgb.Dataset(frame(b[va], FEATS_B2), y[va], reference=dtr)
        bst = lgb.train(p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        res.append({"H": "none" if H is None else H, "best_iteration": bst.best_iteration,
                    "valid_logloss_2021": bst.best_score["valid_0"]["multi_logloss"]})
    res = pd.DataFrame(res).sort_values("valid_logloss_2021").reset_index(drop=True)
    best = res.iloc[0]
    H = None if best.H == "none" else int(best.H)
    return H, int(best.best_iteration), res


class Platt:
    """Per-class one-vs-rest logistic on logit(p_k); renormalised after mapping."""
    def fit(self, p, y):
        self.m = []
        for k in range(p.shape[1]):
            z = np.log(np.clip(p[:, k], 1e-6, 1 - 1e-6) / np.clip(1 - p[:, k], 1e-6, 1))[:, None]
            self.m.append(LogisticRegression(C=1e6, max_iter=1000).fit(z, (y == k).astype(int)))
        return self

    def transform(self, p):
        q = np.column_stack([m.predict_proba(np.log(np.clip(p[:, k], 1e-6, 1 - 1e-6) /
                             np.clip(1 - p[:, k], 1e-6, 1))[:, None])[:, 1] for k, m in enumerate(self.m)])
        return q / q.sum(axis=1, keepdims=True)

    def params(self, classes):
        return {c: [float(m.intercept_[0]), float(m.coef_[0, 0])] for c, m in zip(classes, self.m)}


def calib_table(df, models, classes_show, cname):
    out = []
    for m in models:
        for c in classes_show:
            p = df[f"p{m}_{c}"].astype(float); a = (df.outcome == c).astype(float)
            dec = pd.qcut(p.rank(method="first"), 10, labels=False) + 1
            g = pd.DataFrame({"decile": dec, "p": p, "a": a}).groupby("decile").agg(
                n=("a", "size"), mean_predicted=("p", "mean"), actual_rate=("a", "mean")).reset_index()
            g["model"], g["class"] = m, c
            out.append(g)
    t = pd.concat(out)[["model", "class", "decile", "n", "mean_predicted", "actual_rate"]]
    t["gap_pred_minus_actual"] = t.mean_predicted - t.actual_rate
    return t


def ece(t):
    t = t.assign(w=t.n * t.gap_pred_minus_actual.abs())
    e = t.groupby(["model", "class"]).apply(lambda g: g.w.sum() / g.n.sum()).unstack("class")
    return e


def aggregate_bias(df, models, classes_show, n=1000, seed=SEED):
    """Pooled mean predicted vs actual rate with a match-bootstrap CI for the actual rate."""
    rng = np.random.default_rng(seed)
    mids = df.match_id.values
    um, inv = np.unique(mids, return_inverse=True)
    idx = rng.integers(0, len(um), size=(n, len(um)))
    cnt = np.bincount(inv, minlength=len(um)).astype(float)
    rows = []
    for c in classes_show:
        a = np.bincount(inv, weights=(df.outcome.values == c).astype(float), minlength=len(um))
        reps = a[idx].sum(1) / cnt[idx].sum(1)
        lo, hi = np.percentile(reps, [2.5, 97.5])
        for m in models:
            p = df[f"p{m}_{c}"].values.astype(float)
            rows.append({"model": m, "class": c, "mean_predicted": p.mean(), "actual": a.sum() / cnt.sum(),
                         "actual_lo95": lo, "actual_hi95": hi, "pred_in_CI": bool(lo <= p.mean() <= hi)})
    return pd.DataFrame(rows)


def bias_fixed(agg, cal, m):
    ok = True
    for c in ("6", "0"):
        inci = bool(agg[(agg.model == m) & (agg["class"] == c)].pred_in_CI.iloc[0])
        s = np.sign(cal[(cal.model == m) & (cal["class"] == c)].gap_pred_minus_actual.values)
        mixed = not (np.all(s > 0) or np.all(s < 0))
        ok &= inci and mixed
    return ok


def main():
    t0 = time.perf_counter()
    b, log = load_balls()
    b, classes, rule = merge_rare(b)
    b = add_era(b)
    K = len(classes); cidx = {c: i for i, c in enumerate(classes)}
    y = b.outcome.map(cidx).values
    cfg1a = json.load(open(os.path.join(MODEL, "phase1a_config.json")))
    cfg_b = cfg1a["lgb_fixed_config_chosen_on_2021"]

    H, it2, hgrid = choose_H(b, y, cfg_b, K)
    cfg_b2 = {**cfg_b, "best_iteration": it2, "H": H}
    print("H grid\n", hgrid, flush=True)

    preds = {}
    for t in FOLDS:
        tr, te = (b.season < t).values, (b.season == t).values
        pb = train(b[tr], y[tr], FEATS_B, cfg_b).predict(frame(b[te], FEATS_B))
        w = weights(b.season.values[tr], t - 1, H)
        pb2 = train(b[tr], y[tr], FEATS_B2, cfg_b2, w).predict(frame(b[te], FEATS_B2))
        preds[t] = (te, pb, pb2)
        print(f"fold {t}: B={ball_logloss(pb, y[te]).mean():.5f} B2={ball_logloss(pb2, y[te]).mean():.5f}", flush=True)

    # Platt recalibration of B from the previous season (computed always; used only if neither fixes the bias)
    oof, platt_params = [], {}
    for t in TEST:
        te, pb, pb2 = preds[t]
        te_prev, pb_prev, _ = preds[t - 1]
        pl = Platt().fit(pb_prev, y[te_prev])
        pbr = pl.transform(pb)
        platt_params[t] = pl.params(classes)
        f = b.loc[te, ["match_id", "season", "format_tier", "phase", "innings", "outcome"]].copy()
        for name, p in (("B", pb), ("B2", pb2), ("Br", pbr)):
            for k, c in enumerate(classes):
                f[f"p{name}_{c}"] = p[:, k].astype(np.float32)
            f[f"loss_{name}"] = ball_logloss(p, y[te])
        oof.append(f)
    oof = pd.concat(oof, ignore_index=True)

    rows = []
    for t in TEST + ["2022-26 pooled"]:
        g = oof if t == "2022-26 pooled" else oof[oof.season == t]
        r = {"test_season": t, "n_balls": len(g), "logloss_B": g.loss_B.mean(), "logloss_B2": g.loss_B2.mean(),
             "logloss_Brecal": g.loss_Br.mean()}
        for nm, (c1, c2) in {"B2_minus_B": ("loss_B", "loss_B2"), "Brecal_minus_B": ("loss_B", "loss_Br")}.items():
            r[nm], r[nm + "_lo95"], r[nm + "_hi95"] = boot_diff(g[c1].values, g[c2].values, g.match_id.values)
        rows.append(r)
    res = pd.DataFrame(rows)
    pooled = res.iloc[-1]
    keep_b2 = bool(pooled.B2_minus_B < 0 and pooled.B2_minus_B_hi95 < 0)

    late = oof[oof.season.isin([2025, 2026])]
    cal = calib_table(late, ["B", "B2", "Br"], ["6", "0"], "")
    cal_all = calib_table(late, ["B", "B2", "Br"], classes, "")
    ece_t = ece(cal_all)[classes]
    agg = aggregate_bias(late, ["B", "B2", "Br"], classes)
    fixed = {m: bias_fixed(agg, cal, m) for m in ("B", "B2", "Br")}

    if keep_b2 and fixed["B2"]:
        final, why = "B2", "B2 beats B on log loss (CI excludes 0) and passes the six/dot bias check"
    elif fixed["B"]:
        final, why = "B", "B already passes the six/dot bias check"
    else:
        final = "B+platt"
        why = ("B2 " + ("beats B on log loss but " if keep_b2 else "does not beat B on log loss and ")
               + f"{'passes' if fixed['B2'] else 'does not fix'} the six/dot bias; per instruction B is kept and "
               "Platt-recalibrated per class on the previous season")
    print("keep_b2:", keep_b2, "fixed:", fixed, "final:", final, flush=True)

    os.makedirs(REP, exist_ok=True)
    res.to_csv(os.path.join(REP, "phase1a_fix_fold_logloss.csv"), index=False)
    cal.to_csv(os.path.join(REP, "phase1a_fix_calibration_6_0_2025_26.csv"), index=False)
    ece_t.reset_index().to_csv(os.path.join(REP, "phase1a_fix_ece_2025_26.csv"), index=False)
    agg.to_csv(os.path.join(REP, "phase1a_fix_aggregate_bias_2025_26.csv"), index=False)
    hgrid.to_csv(os.path.join(REP, "phase1a_fix_H_tuning_2021.csv"), index=False)
    oof.to_parquet(os.path.join(MODEL, "phase1a_fix_oof.parquet"), index=False)
    final_cfg = {"final_model": final, "reason": why, "keep_B2_by_logloss_rule": keep_b2, "bias_fixed": fixed,
                 "B_config": cfg_b, "B2_config": {**cfg_b2, "H": H, "features": FEATS_B2},
                 "pooled_B2_minus_B": [pooled.B2_minus_B, pooled.B2_minus_B_lo95, pooled.B2_minus_B_hi95],
                 "pooled_Brecal_minus_B": [pooled.Brecal_minus_B, pooled.Brecal_minus_B_lo95, pooled.Brecal_minus_B_hi95],
                 "platt_params_by_fold": platt_params, "classes": classes}
    json.dump(final_cfg, open(os.path.join(MODEL, "phase1a_final.json"), "w"), indent=2, default=float)
    write_report(hgrid, H, it2, res, cal, ece_t, agg, fixed, keep_b2, final, why, time.perf_counter() - t0)


def write_report(hgrid, H, it2, res, cal, ece_t, agg, fixed, keep_b2, final, why, rt):
    f = lambda x: f"{x:.5f}" if isinstance(x, float) else x
    p = res.iloc[-1]
    piv = cal.pivot_table(index=["class", "decile"], columns="model",
                          values=["mean_predicted", "actual_rate"]).round(4)
    piv.columns = [f"{a}_{m}" for a, m in piv.columns]
    piv = piv[[c for c in piv.columns if c.startswith("mean_predicted")] + ["actual_rate_B"]].rename(
        columns={"actual_rate_B": "actual"})
    L = ["# Phase 1a-fix - scoring drift", "",
         "B2 = B + impact_player_era (IPL & season >= 2023) + recency weights 0.5^((t-1-season)/H). "
         "Br = B with per-class Platt recalibration fitted on the previous season inside each fold.", "",
         "## H chosen on 2021 validation (train < 2021)", "", hgrid.map(f).to_markdown(index=False), "",
         f"Chosen H = {H if H else 'none'}, rounds = {it2}.", "",
         "## Out-of-sample log loss (same expanding folds)", "", res.map(f).to_markdown(index=False), "",
         f"Rule: keep B2 only if pooled B2 - B < 0 with CI excluding 0 -> **{'B2 kept' if keep_b2 else 'B2 rejected'}** "
         f"(B2 - B = {p.B2_minus_B:+.5f} [{p.B2_minus_B_lo95:+.5f}, {p.B2_minus_B_hi95:+.5f}]).", "",
         "## Six / dot calibration by decile, 2025-26", "", piv.reset_index().to_markdown(index=False), "",
         "## Pooled 2025-26: mean predicted vs actual (match-bootstrap 95% CI of actual)", "",
         agg.map(f).to_markdown(index=False), "",
         "## ECE by class, 2025-26 (weighted mean |pred - actual| over deciles)", "",
         ece_t.reset_index().map(f).to_markdown(index=False), "",
         "Bias-fixed check (class 6 and 0: pooled prediction inside the actual-rate CI and decile gaps not all one sign): "
         + ", ".join(f"{k}: {v}" for k, v in fixed.items()), "",
         f"**Final outcome model: {final}** - {why}.", "", f"Runtime {rt:.0f} s.", ""]
    open(os.path.join(REP, "phase1a_fix_summary.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
