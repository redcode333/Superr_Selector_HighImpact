"""Phase 1a - ball-outcome model p(o | s, x).

Outcome o per delivery (main innings only), in priority order:
  W   any wicket counted in wkts_before (retired hurt / retired not out are not wickets), on any delivery type
  WD  wide (no wicket);  NB  no-ball (no wicket)
  0..6  total runs off a legal, non-wicket delivery (bat + byes + leg-byes + penalty).
Rare-class rule (data-driven): a runs class with share < MERGE_SHARE of all deliveries is merged into the next
lower kept class (5 -> 4, 7 -> 6). Runs on W / WD / NB balls are not part of the class (summarised for Phase 1b).
Rows with unknown runs (estimated bye runs in data-pack-only matches) are excluded with a reason.

State / context features (NO player identity or attributes, no pace_or_spin):
  b = balls_remaining, w = wkts_in_hand, innings, r = runs_required (inns 2) or score_before (inns 1),
  rate = 6 r / max(b, 1) (inns 2, required) or 6 score / max(legal balls, 1) (inns 1, current), capped at 36,
  phase, format_tier (IPL / full_member / associate; A-team matches -> associate), season (calendar year).

Models:
  A  multinomial logistic:  softmax(beta_o . x); x = cubic B-splines (6 knots) of b, of r and of rate - the r and rate
     splines separately per innings - plus one-hot w, phase, tier, innings, and linear season. Design columns are
     standardised before fitting (numerical conditioning only); lbfgs, C=1, tol=1e-5, convergence recorded per fold.
  B  LightGBM multiclass on the raw features; ONE fixed configuration chosen on 2021 (train < 2021, validate 2021,
     early stopping) and reused unchanged in every fold.
  C  baseline (no learning): class frequencies by phase x innings in the training rows (+0.5 smoothing).
  D  comparison only: sklearn RandomForestClassifier, 300 trees, min_samples_leaf=50, same features as B
     (probability = mean of per-tree leaf class frequencies). D can be selected only if it beats BOTH A and B out of
     sample (pooled D - A < 0 and D - B < 0, both 95% CIs excluding 0).
Evaluation: expanding window. For test season t in 2022..2026: train on all seasons < t (history included), test on t.
Metric: multiclass log loss = -mean log p(o_true). B - A difference with a 95% CI from a bootstrap over matches
(1000 resamples, seed 42). Decision rule: keep B only if it beats A out of sample, i.e. the pooled 2022-26 B - A
log-loss difference is < 0 and its 95% CI excludes 0; otherwise A.
"""
import json, os, time
import numpy as np
import pandas as pd
import psutil
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT, MODEL = os.path.join(ROOT, "out"), os.path.join(ROOT, "out", "model")
REP = os.path.join(ROOT, "reports", "phase1")
SEED, MERGE_SHARE, N_BOOT, MAX_ITER_A = 42, 0.001, 1000, 5000
TEST_SEASONS = [2022, 2023, 2024, 2025, 2026]
CONFLICT_MATCHES = {17765}
FULL_MEMBERS = {"Australia", "Bangladesh", "England", "India", "New Zealand", "Pakistan", "South Africa",
                "Sri Lanka", "West Indies", "Zimbabwe", "Ireland", "Afghanistan"}     # all full members since 2017
PHASES, TIERS = ["Powerplay", "Middle", "Death"], ["IPL", "full_member", "associate"]


# ------------------------------------------------------------------ data
def outcome_class(df):
    """Raw outcome label before rare-class merging; None where runs are unknown."""
    runs = df.team_runs
    known = np.isclose(runs, np.round(runs))
    lab = np.where(df.is_wicket, "W", np.where(df.is_wide, "WD", np.where(df.is_noball, "NB",
                   np.where(known, np.round(runs).astype(int).astype(str), None))))
    return pd.Series(lab, index=df.index, dtype=object)


def load_balls():
    """All modelling rows (history + 2022+), with outcome, features and exclusion log."""
    cols = ["match_id", "innings", "innings_ball_num", "is_super_over", "is_legal", "is_wide", "is_noball", "is_wicket",
            "team_runs", "bat_runs", "balls_remaining", "wkts_in_hand", "score_before", "runs_required",
            "legal_balls_before", "phase"]
    h = pd.read_parquet(os.path.join(MODEL, "history_deliveries.parquet"),
                        columns=cols + ["season", "format_tier", "dls_affected"])
    h["source"] = "history"
    d = pd.read_csv(os.path.join(OUT, "deliveries_clean.csv"), usecols=cols + ["extras_source"], low_memory=False)
    mc = pd.read_csv(os.path.join(OUT, "match_context_enriched.csv"),
                     usecols=["match_id", "comp_name", "match_date", "home_team", "away_team", "dls_affected"])
    ipl = mc.comp_name.str.startswith("IPL")
    full = mc.home_team.isin(FULL_MEMBERS) & mc.away_team.isin(FULL_MEMBERS)
    mc["format_tier"] = np.where(ipl, "IPL", np.where(full, "full_member", "associate"))
    mc["season"] = pd.to_datetime(mc.match_date).dt.year
    d = d.merge(mc[["match_id", "season", "format_tier", "dls_affected"]], on="match_id", how="left")
    d["source"] = "datapack_2022plus"
    b = pd.concat([h, d], ignore_index=True)
    log = {"all_deliveries": len(b)}
    keep = (b.innings <= 2) & ~b.is_super_over.astype(bool)
    log["excluded_super_over"] = int((~keep).sum())
    m = keep & ~b.dls_affected.astype(bool); log["excluded_dls_affected"] = int((keep & ~m).sum()); keep = m
    m = keep & ~b.match_id.isin(CONFLICT_MATCHES); log["excluded_conflict_17765"] = int((keep & ~m).sum()); keep = m
    b = b[keep].copy()
    b["outcome_raw"] = outcome_class(b)
    unk = b.outcome_raw.isna()
    log["excluded_unknown_runs_estimated_byes"] = int(unk.sum())
    b = b[~unk].copy()
    # features
    b["b"] = b.balls_remaining.astype(float)
    b["w"] = b.wkts_in_hand.astype(int)
    b["r"] = np.where(b.innings == 1, b.score_before, b.runs_required).astype(float)
    rate = np.where(b.innings == 1, 6 * b.score_before / np.maximum(b.legal_balls_before, 1),
                    6 * b.runs_required / np.maximum(b.balls_remaining, 1))
    b["rate"] = np.clip(rate, 0, 36)
    log["rows_used"] = len(b)
    return b.reset_index(drop=True), log


def merge_rare(b):
    """Merge runs classes with share < MERGE_SHARE into the next lower kept runs class."""
    share = b.outcome_raw.value_counts(normalize=True)
    runs_cls = sorted(int(c) for c in share.index if c.isdigit())
    kept = [c for c in runs_cls if share[str(c)] >= MERGE_SHARE]
    rule = {}
    for c in runs_cls:
        if c not in kept:
            rule[str(c)] = str(max(k for k in kept if k < c))
    b["outcome"] = b.outcome_raw.replace(rule)
    classes = [str(c) for c in kept] + ["W", "WD", "NB"]
    return b, classes, rule


# ------------------------------------------------------------------ models
class ModelA:
    """Multinomial logistic with spline bases (see module docstring)."""
    def __init__(self):
        self.sp_b = SplineTransformer(n_knots=6, degree=3, include_bias=False)
        self.sp_r = {k: SplineTransformer(n_knots=6, degree=3, include_bias=False) for k in (1, 2)}
        self.sp_rate = {k: SplineTransformer(n_knots=6, degree=3, include_bias=False) for k in (1, 2)}

    def _x(self, df, fit):
        f = (lambda t, v: t.fit_transform(v)) if fit else (lambda t, v: t.transform(v))
        parts = [f(self.sp_b, df[["b"]].clip(0, 120).values)]
        for k in (1, 2):
            ind = (df.innings.values == k)[:, None]
            parts.append(f(self.sp_r[k], df[["r"]].clip(0, 260).values) * ind)
            parts.append(f(self.sp_rate[k], df[["rate"]].values) * ind)
        parts.append(np.eye(11)[df.w.clip(0, 10).values][:, 1:])
        parts.append((df.innings.values == 2)[:, None].astype(float))
        parts.append(np.column_stack([(df.phase.values == p) for p in PHASES[1:]]).astype(float))
        parts.append(np.column_stack([(df.format_tier.values == t) for t in TIERS[1:]]).astype(float))
        parts.append(((df.season.values - 2015) / 10.0)[:, None])
        return np.hstack(parts)

    def fit(self, df, y):
        """Columns standardised (conditioning only; the model is unchanged), lbfgs to convergence; n_iter recorded."""
        x = self._x(df, True)
        self.sc = StandardScaler().fit(x)
        self.m = LogisticRegression(C=1.0, max_iter=MAX_ITER_A, tol=1e-5, random_state=SEED)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            self.m.fit(self.sc.transform(x), y)
        self.n_iter = int(np.max(self.m.n_iter_))
        self.converged = not any(issubclass(x_.category, ConvergenceWarning) for x_ in w)
        return self

    def predict_proba(self, df):
        return self.m.predict_proba(self.sc.transform(self._x(df, False)))


FEATS_B = ["b", "w", "innings", "r", "rate", "phase_c", "tier_c", "season"]


def lgb_frame(df):
    x = df[["b", "w", "innings", "r", "rate", "season"]].copy()
    x["phase_c"] = pd.Categorical(df.phase, categories=PHASES).codes
    x["tier_c"] = pd.Categorical(df.format_tier, categories=TIERS).codes
    return x[FEATS_B]


def lgb_params(num_leaves, min_child, n_class):
    return {"objective": "multiclass", "num_class": n_class, "learning_rate": 0.05, "num_leaves": num_leaves,
            "min_child_samples": min_child, "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l2": 1.0, "seed": SEED, "deterministic": True, "force_row_wise": True, "num_threads": 4,
            "verbose": -1}


def tune_lgb_on_2021(b, y, n_class):
    """Fixed configuration: train seasons < 2021, validate on 2021, early stopping; grid over leaves x min_child."""
    import lightgbm as lgb
    tr, va = b.season < 2021, b.season == 2021
    res = []
    for nl in (15, 31, 63):
        for mc in (100, 1000):
            p = lgb_params(nl, mc, n_class)
            dtr = lgb.Dataset(lgb_frame(b[tr]), y[tr], categorical_feature=["phase_c", "tier_c"])
            dva = lgb.Dataset(lgb_frame(b[va]), y[va], reference=dtr)
            bst = lgb.train(p, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
            res.append({"num_leaves": nl, "min_child_samples": mc, "best_iteration": bst.best_iteration,
                        "valid_logloss_2021": bst.best_score["valid_0"]["multi_logloss"]})
    res = pd.DataFrame(res).sort_values("valid_logloss_2021")
    return res.iloc[0].to_dict(), res


def fit_lgb(cfg, df, y, n_class):
    import lightgbm as lgb
    p = lgb_params(int(cfg["num_leaves"]), int(cfg["min_child_samples"]), n_class)
    return lgb.train(p, lgb.Dataset(lgb_frame(df), y, categorical_feature=["phase_c", "tier_c"]),
                     int(cfg["best_iteration"]))


def fit_predict_rf(tr, ytr, te, n_class):
    """Model D: random forest (comparison only). Returns probabilities aligned to classes 0..n_class-1."""
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=50, n_jobs=8, random_state=SEED)
    rf.fit(lgb_frame(tr).values.astype(np.float32), ytr)
    p = np.zeros((len(te), n_class))
    p[:, rf.classes_] = rf.predict_proba(lgb_frame(te).values.astype(np.float32))
    del rf
    return p


def baseline_c(tr, ytr, te, n_class):
    """Class frequencies by phase x innings from training rows (+0.5 smoothing)."""
    key_tr = tr.phase + "|" + tr.innings.astype(str)
    cnt = pd.crosstab(key_tr, ytr).reindex(columns=range(n_class), fill_value=0) + 0.5
    prob = cnt.div(cnt.sum(axis=1), axis=0)
    key_te = te.phase + "|" + te.innings.astype(str)
    return prob.reindex(key_te).values


# ------------------------------------------------------------------ metrics
def ball_logloss(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))


def boot_diff(loss_a, loss_b, match_ids, n=N_BOOT, seed=SEED):
    """Mean of (loss_b - loss_a) per ball, with a 95% percentile CI from resampling matches."""
    g = pd.DataFrame({"m": match_ids, "d": loss_b - loss_a}).groupby("m").d.agg(["sum", "size"])
    s, c = g["sum"].values, g["size"].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    reps = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return s.sum() / c.sum(), np.percentile(reps, 2.5), np.percentile(reps, 97.5)


# ------------------------------------------------------------------ main
def main():
    t0 = time.perf_counter(); proc = psutil.Process()
    os.makedirs(REP, exist_ok=True); os.makedirs(MODEL, exist_ok=True)
    b, log = load_balls()
    b, classes, rule = merge_rare(b)
    cidx = {c: i for i, c in enumerate(classes)}
    y = b.outcome.map(cidx).values
    K = len(classes)

    # 1. class counts
    cc = b.outcome.value_counts().reindex(classes)
    counts = pd.DataFrame({"class": classes, "n": cc.values, "share": (cc / cc.sum()).values})
    by_phase = pd.crosstab(b.outcome, b.phase, normalize="columns").reindex(index=classes, columns=PHASES)
    by_phase_n = pd.crosstab(b.outcome, b.phase).reindex(index=classes, columns=PHASES)
    raw_counts = b.outcome_raw.value_counts()
    counts.to_csv(os.path.join(REP, "phase1a_class_counts.csv"), index=False)
    by_phase.reset_index().to_csv(os.path.join(REP, "phase1a_class_share_by_phase.csv"), index=False)

    # 2. LightGBM fixed config chosen on 2021
    cfg, grid = tune_lgb_on_2021(b, y, K)
    grid.to_csv(os.path.join(REP, "phase1a_lgb_tuning_2021.csv"), index=False)

    # 3. expanding-window folds
    rows, oof = [], []
    for t in TEST_SEASONS:
        tr, te = (b.season < t).values, (b.season == t).values
        if te.sum() == 0: continue
        ta = time.perf_counter(); ma = ModelA().fit(b[tr], y[tr]); pa = ma.predict_proba(b[te])
        print(f"fold {t}: A fitted in {time.perf_counter()-ta:.0f}s, n_iter={ma.n_iter}, converged={ma.converged}", flush=True)
        pb = fit_lgb(cfg, b[tr], y[tr], K).predict(lgb_frame(b[te]))
        pc = baseline_c(b[tr], y[tr], b[te], K)
        tf = time.perf_counter(); pdd = fit_predict_rf(b[tr], y[tr], b[te], K); rf_s = time.perf_counter() - tf
        la, lb_, lc, ld = (ball_logloss(p, y[te]) for p in (pa, pb, pc, pdd))
        mids = b.match_id.values[te]
        row = {"test_season": t, "n_train_balls": int(tr.sum()), "n_balls": int(te.sum()),
               "n_matches": int(b[te].match_id.nunique()), "logloss_A": la.mean(), "logloss_B": lb_.mean(),
               "logloss_C": lc.mean(), "logloss_D": ld.mean(), "A_converged": ma.converged, "A_n_iter": ma.n_iter,
               "rf_fit_predict_s": round(rf_s, 1),
               "D_zero_prob_on_true_class": int((pdd[np.arange(te.sum()), y[te]] == 0).sum())}
        for nm, (x1, x2) in {"B_minus_A": (la, lb_), "D_minus_A": (la, ld), "D_minus_B": (lb_, ld)}.items():
            row[nm], row[nm + "_lo95"], row[nm + "_hi95"] = boot_diff(x1, x2, mids)
        rows.append(row)
        f = b.loc[te, ["match_id", "innings", "innings_ball_num", "season", "format_tier", "phase", "outcome"]].copy()
        for name, p in (("A", pa), ("B", pb), ("C", pc), ("D", pdd)):
            for k, c in enumerate(classes):
                f[f"p{name}_{c}"] = p[:, k].astype(np.float32)
        f["loss_A"], f["loss_B"], f["loss_C"], f["loss_D"] = la, lb_, lc, ld
        oof.append(f); del pa, pb, pc, pdd
        print(f"fold {t}: n={te.sum()} A={la.mean():.4f} B={lb_.mean():.4f} C={lc.mean():.4f} D={ld.mean():.4f} "
              f"B-A={row['B_minus_A']:+.4f} [{row['B_minus_A_lo95']:+.4f},{row['B_minus_A_hi95']:+.4f}] "
              f"rss={proc.memory_info().rss/2**20:.0f}MB", flush=True)
    res = pd.DataFrame(rows)
    oof = pd.concat(oof, ignore_index=True)
    pooled = {"test_season": "2022-26 pooled", "n_balls": len(oof), "n_matches": oof.match_id.nunique(),
              "logloss_A": oof.loss_A.mean(), "logloss_B": oof.loss_B.mean(), "logloss_C": oof.loss_C.mean(),
              "logloss_D": oof.loss_D.mean(), "D_zero_prob_on_true_class": int(res.D_zero_prob_on_true_class.sum())}
    for nm, (c1, c2) in {"B_minus_A": ("loss_A", "loss_B"), "D_minus_A": ("loss_A", "loss_D"),
                         "D_minus_B": ("loss_B", "loss_D")}.items():
        pooled[nm], pooled[nm + "_lo95"], pooled[nm + "_hi95"] = boot_diff(oof[c1].values, oof[c2].values, oof.match_id.values)
    res = pd.concat([res, pd.DataFrame([pooled])], ignore_index=True)
    res.to_csv(os.path.join(REP, "phase1a_fold_logloss.csv"), index=False)

    # 4. decision rule: B only if it beats A; D only if it beats both A and B (all out of sample, CI excludes 0)
    d, lo, hi = pooled["B_minus_A"], pooled["B_minus_A_lo95"], pooled["B_minus_A_hi95"]
    chosen = "B" if (d < 0 and hi < 0) else "A"
    if pooled["D_minus_A_hi95"] < 0 and pooled["D_minus_B_hi95"] < 0:
        chosen = "D"

    # 5. per-class calibration on 2025-26 for every model (deciles of predicted prob), classes W, 4, 6, 0
    cal = []
    late = oof[oof.season.isin([2025, 2026])]
    for m in ("A", "B", "C", "D"):
        for c in ["W", "4", "6", "0"]:
            p = late[f"p{m}_{c}"].astype(float); act = (late.outcome == c).astype(float)
            dec = pd.qcut(p.rank(method="first"), 10, labels=False) + 1
            g = pd.DataFrame({"decile": dec, "p": p, "a": act}).groupby("decile").agg(
                n=("a", "size"), mean_predicted=("p", "mean"), actual_rate=("a", "mean"))
            g["class"], g["model"] = c, m; cal.append(g.reset_index())
    cal = pd.concat(cal)[["model", "class", "decile", "n", "mean_predicted", "actual_rate"]]
    cal["abs_gap"] = (cal.mean_predicted - cal.actual_rate).abs()
    cal_summary = (cal.assign(w=cal.n * cal.abs_gap).groupby(["model", "class"])
                   .apply(lambda g: g.w.sum() / g.n.sum()).unstack("class")[["W", "4", "6", "0"]])
    cal_summary.columns = [f"ECE_{c}" for c in cal_summary.columns]
    cal_summary.reset_index().to_csv(os.path.join(REP, "phase1a_calibration_ece_2025_26.csv"), index=False)
    cal.to_csv(os.path.join(REP, "phase1a_calibration_2025_26.csv"), index=False)
    plot_calibration(cal, chosen)

    oof.to_parquet(os.path.join(MODEL, "phase1a_oof_outcome_probs.parquet"), index=False)
    extra_runs = (b[b.outcome.isin(["WD", "NB", "W"])].groupby("outcome").team_runs.agg(["mean", "size"]))
    cfg_out = {"classes": classes, "merge_rule": rule, "merge_share_threshold": MERGE_SHARE, "chosen_model": chosen,
               "decision_rule": "B only if pooled 2022-26 log loss B - A < 0 with 95% match-bootstrap CI excluding 0; "
                                "D (comparison only) only if it beats both A and B the same way",
               "pooled_B_minus_A": [d, lo, hi],
               "pooled_D_minus_A": [pooled["D_minus_A"], pooled["D_minus_A_lo95"], pooled["D_minus_A_hi95"]],
               "pooled_D_minus_B": [pooled["D_minus_B"], pooled["D_minus_B_lo95"], pooled["D_minus_B_hi95"]],
               "rf_config": {"n_estimators": 300, "min_samples_leaf": 50, "random_state": SEED},
               "lgb_fixed_config_chosen_on_2021": cfg,
               "mean_team_runs_on_W_WD_NB_balls": extra_runs["mean"].round(3).to_dict(), "row_log": log}
    json.dump(cfg_out, open(os.path.join(MODEL, "phase1a_config.json"), "w"), indent=2, default=float)

    runtime = time.perf_counter() - t0
    mem = proc.memory_info()
    peak = getattr(mem, "peak_wset", None) or mem.rss
    perf = {"runtime_s": round(runtime, 1), "peak_memory_mb": round(peak / 2**20, 1)}
    write_report(log, counts, by_phase, by_phase_n, raw_counts, rule, res, cal, chosen, cfg, grid, perf, extra_runs,
                 cal_summary)
    write_dictionary(rule)
    return log, counts, by_phase, res, cal, chosen, cfg, perf, rule, raw_counts, cal_summary


def plot_calibration(cal, chosen):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    colors = {"A": "#2a6fdb", "B": "#e07b00", "C": "#888888", "D": "#2e9e5b"}
    for a, c in zip(ax, ["W", "4", "6", "0"]):
        g = cal[cal["class"] == c]
        lim = max(g.mean_predicted.max(), g.actual_rate.max()) * 1.08
        a.plot([0, lim], [0, lim], color="#bbb", lw=1, ls="--")
        for m, gm in g.groupby("model"):
            a.plot(gm.mean_predicted, gm.actual_rate, "o-", color=colors[m], ms=3, lw=1.6 if m == chosen else 1,
                   label=f"{m}{' (chosen)' if m == chosen else ''}")
        a.set_title(f"class {c}"); a.set_xlabel("mean predicted"); a.set_xlim(0, lim); a.set_ylim(0, lim)
    ax[0].set_ylabel("actual rate"); ax[0].legend(fontsize=8)
    fig.suptitle("Phase 1a: calibration by decile of predicted probability, test seasons 2025-26 (out of sample)")
    fig.tight_layout(); fig.savefig(os.path.join(REP, "phase1a_calibration_2025_26.png"), dpi=130); plt.close(fig)


def write_report(log, counts, by_phase, by_phase_n, raw_counts, rule, res, cal, chosen, cfg, grid, perf, extra_runs,
                 cal_summary):
    f = lambda x: f"{x:.4f}" if isinstance(x, float) else x
    pooled = res.iloc[-1]
    ll = res[["test_season", "n_balls", "n_matches", "logloss_A", "logloss_B", "logloss_C", "logloss_D",
              "A_converged", "A_n_iter"]]
    diffs = res[["test_season"] + [c for c in res.columns if "_minus_" in c]]
    L = ["# Phase 1a - ball-outcome model", "",
         "## Phase 1 metric conventions",
         "Tied matches (WP label 0.5) are excluded from AUC and kept in Brier score and log loss (Phases 1b-1d).", "",
         "## Rows", "\n".join(f"- {k}: {v:,}" for k, v in log.items()), "",
         "## Outcome classes", f"Rare-class rule: a runs class with share < {MERGE_SHARE:.1%} of deliveries is merged into "
         f"the next lower kept runs class. Applied: {rule or 'none'}. Raw counts of merged classes: "
         + ", ".join(f"{k}: {int(raw_counts.get(k, 0))}" for k in rule), "",
         counts.assign(share=counts.share.map(lambda x: f"{x:.2%}")).to_markdown(index=False), "",
         "Share by phase:", "", (by_phase.map(lambda x: f"{x:.2%}")).reset_index().to_markdown(index=False), "",
         "Mean team runs on W / WD / NB balls (for Phase 1b): " + str(extra_runs["mean"].round(3).to_dict()), "",
         "## Out-of-sample log loss (expanding window)",
         "A = multinomial logistic with splines; B = LightGBM; C = phase x innings class frequencies; "
         "D = random forest (300 trees, min_samples_leaf=50; comparison only). X - Y < 0 means X is better; "
         "CIs from 1000 match-bootstrap resamples.", "",
         ll.map(f).to_markdown(index=False), "", diffs.map(f).to_markdown(index=False), "",
         f"Random-forest balls whose true class got probability 0 (log loss clipped at 1e-15): "
         f"{int(pooled.D_zero_prob_on_true_class)}", "",
         f"LightGBM fixed configuration, chosen on 2021 (train < 2021, validate 2021): {cfg}", "",
         grid.map(f).to_markdown(index=False), "",
         "## Decision", "Rule: keep B only if it beats A out of sample (pooled B - A < 0 and 95% CI excludes 0); "
         "D can be selected only if it beats both A and B the same way.", "",
         f"- B - A = {pooled.B_minus_A:+.5f} [{pooled.B_minus_A_lo95:+.5f}, {pooled.B_minus_A_hi95:+.5f}]",
         f"- D - A = {pooled.D_minus_A:+.5f} [{pooled.D_minus_A_lo95:+.5f}, {pooled.D_minus_A_hi95:+.5f}]",
         f"- D - B = {pooled.D_minus_B:+.5f} [{pooled.D_minus_B_lo95:+.5f}, {pooled.D_minus_B_hi95:+.5f}]", "",
         f"**Chosen: {chosen}.**", "",
         "## Calibration, test seasons 2025-26 (all models)",
         "Weighted mean |predicted - actual| over deciles (ECE) per class:", "",
         cal_summary.reset_index().map(f).to_markdown(index=False), "",
         "By decile:", "", cal.map(f).to_markdown(index=False), "",
         "![calibration](phase1a_calibration_2025_26.png)", "",
         f"## Runtime: {perf['runtime_s']} s; peak memory {perf['peak_memory_mb']} MB", ""]
    open(os.path.join(REP, "phase1a_summary.md"), "w", encoding="utf-8").write("\n".join(L))


def write_dictionary(rule):
    p = os.path.join(OUT, "data_dictionary.csv")
    d = pd.read_csv(p)
    d = d[~d.table.astype(str).str.startswith("model/phase1a")]
    new = pd.DataFrame([
        ("model/phase1a_oof_outcome_probs", "outcome", "category",
         "W (any wicket in wkts_before) > WD > NB > total runs off a legal non-wicket ball; merged: " + json.dumps(rule),
         "exact (estimated-bye rows excluded)", "ball-outcome model target"),
        ("model/phase1a_oof_outcome_probs", "pA_* / pB_* / pC_*", "probability",
         "Out-of-sample class probabilities: expanding window, trained on seasons < test season", "estimate", "model evaluation"),
        ("model/phase1a_oof_outcome_probs", "format_tier (2022+)", "category",
         "IPL; full_member if both teams are ICC full members; else associate (A-team matches mapped to associate)", "inferred", "model feature"),
    ], columns=["table", "field", "unit", "definition", "status", "allowed_use"])
    pd.concat([d, new], ignore_index=True).to_csv(p, index=False)


if __name__ == "__main__":
    log, counts, by_phase, res, cal, chosen, cfg, perf, rule, raw, cal_summary = main()
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(log); print(counts.to_string(index=False)); print(by_phase.round(4).to_string())
    print(res.round(5).to_string(index=False)); print("chosen:", chosen, cfg)
    print(cal_summary.round(5).to_string()); print(perf)
