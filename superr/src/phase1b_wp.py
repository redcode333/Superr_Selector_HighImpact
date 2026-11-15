"""Phase 1b (v2) - chase win probability by recursion over the outcome model, with a WP calibration map.

State (b, w, r): legal balls remaining, wickets in hand, runs required.
Terminal values: WP(r <= 0) = 1; innings over (b = 0 or w = 0) with r = 1 (scores level, super over ~ coin
flip) -> 0.5; innings over with r >= 2 -> 0.   [v1 valued the tie as 0: fixed]
Live states (b >= 1, w >= 1, r >= 1), p = outcome probabilities at (b, w, r) (innings 2, phase from b on a
20-over innings, rate = 6 r / b capped at 36, the table's tier and season):
  WP(b,w,r) = sum_{k in 0,1,2,3,4,6} p_k WP(b-1, w, r-k)
            + p_W [ q_legal(ph) sum_j fWL_ph(j) WP(b-1, w-1, r-j) + (1-q_legal(ph)) sum_j fWI_ph(j) WP(b, w-1, r-j) ]
            + p_WD sum_j fWD_ph(j) WP(b, w, r-j) + p_NB sum_j fNB_ph(j) WP(b, w, r-j)
f* = empirical team-run distributions on wide / no-ball / legal-wicket / illegal-wicket balls from the training
rows, by phase; illegal balls do not consume a ball. Same-b terms point to smaller r (or to w-1), so iterating
b up, w up, r up solves it exactly; the rare zero-run wide (< 0.02%) is a self-loop handled by 1/(1 - p_self).

Tables: leave-one-season-out (LOSO) per (season, tier). Season t's outcome model = LightGBM B trained on all
seasons except t. Variant 'platt' adds per-class Platt maps fitted on season t-1, predicted by B trained on all
seasons except t and t-1; variant 'plain' is B alone. Built for every season 2008..2026 (2008-21 only to give
out-of-fold WP on history chase balls for the calibration map).

Calibration map g (per test season s, per balls-remaining band 1-30 / 31-60 / 61-90 / 91-120), monotone in
logit(WP), fitted ONLY on OOF chase balls of seasons < s (2008+). Candidates: isotonic, or 2-parameter logistic
sigmoid(a + c logit WP). One method per season, chosen by log loss on season s-1 with both fitted on seasons < s-1;
the chosen method is then refitted on seasons < s. g applies to live states; terminal values stay exact.
Note: the WP of an earlier season t comes from the LOSO model of t, which was trained on season s's ball outcomes
(not its results); the map never sees season s's chase results.

Variant choice: the one whose calibrated chase WP has lower pooled 2022-26 log loss with the 95% match-bootstrap CI
of the difference excluding 0; otherwise plain B (simpler).

Limitations: free hits ignored; class 4 = 4 runs (absorbs 5s), 6 = 6 runs (absorbs 7s); phase assumes 20 overs;
raw WP is not monotone in b everywhere (LightGBM steps in b and rate; accepted, see report).
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from numba import njit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_outcome import load_balls, merge_rare, FULL_MEMBERS, CONFLICT_MATCHES, PHASES, TIERS, SEED, MODEL, REP, OUT
from phase1a_fix import train, frame, Platt, FEATS_B

SEASONS = [2022, 2023, 2024, 2025, 2026]
ALL_SEASONS = list(range(2008, 2027))
RMAX, BMAX, WMAX, JMAX, OFF = 310, 120, 10, 8, 8
BANDS = [(1, 30), (31, 60), (61, 90), (91, 120)]
N_BOOT = 1000
EPS = 1e-6
V1_TIEBUG = {"brier": 0.1280, "logloss": 0.4019, "auc": 0.9027}      # v1 report, Platt raw WP with tie = 0


def phase_of_b(b):
    bowled = BMAX - np.asarray(b)
    return np.where(bowled < 36, 0, np.where(bowled >= 90, 2, 1))


PHASE_B = np.zeros(BMAX + 1, dtype=np.int64); PHASE_B[1:] = phase_of_b(np.arange(1, BMAX + 1))
BAND_B = np.zeros(BMAX + 1, dtype=np.int64)
for _i, (_lo, _hi) in enumerate(BANDS):
    BAND_B[_lo:_hi + 1] = _i


def run_dists(tr):
    out = {"fWD": np.zeros((3, JMAX + 1)), "fNB": np.zeros((3, JMAX + 1)), "fWL": np.zeros((3, JMAX + 1)),
           "fWI": np.zeros((3, JMAX + 1)), "qL": np.zeros(3), "n": {}}
    runs = np.clip(np.round(tr.team_runs.values).astype(int), 0, JMAX)
    o, legal, ph_ = tr.outcome.values, tr.is_legal.values.astype(bool), tr.phase.values
    for i, ph in enumerate(PHASES):
        m = ph_ == ph
        for key, sel in (("fWD", o == "WD"), ("fNB", o == "NB"), ("fWL", (o == "W") & legal), ("fWI", (o == "W") & ~legal)):
            c = np.bincount(runs[m & sel], minlength=JMAX + 1).astype(float)
            out[key][i] = c / c.sum(); out["n"][f"{key}_{ph}"] = int(c.sum())
        out["qL"][i] = legal[m & (o == "W")].mean()
    return out


def terminal_table():
    """[b, w, r] array with exact terminal values; live cells 0 (filled later)."""
    T = np.zeros((BMAX + 1, WMAX + 1, RMAX + 1))
    T[:, :, 0] = 1.0
    T[0, :, 1] = 0.5
    T[:, 0, 1] = 0.5
    return T


@njit(cache=True)
def recurse(P, runs_legal, iW, iWD, iNB, fWD, fNB, fWL, fWI, qL, phase_b):
    """P[b, w, r, k] (b 1..BMAX, w 1..WMAX, r 1..RMAX). Returns WP[b, w, r + OFF] (index OFF is r = 0)."""
    B, W, R = P.shape[0], P.shape[1], P.shape[2]
    WP = np.zeros((B, W, R + OFF))
    WP[:, :, :OFF + 1] = 1.0                 # r <= 0: won
    WP[0, :, OFF + 1] = 0.5                  # balls gone, scores level: tie
    WP[:, 0, OFF + 1] = 0.5                  # all out, scores level: tie
    J = fWD.shape[1]
    for b in range(1, B):
        ph = phase_b[b]
        for w in range(1, W):
            for r in range(1, R):
                ri = r + OFF
                p = P[b, w, r]
                v = 0.0
                for k in range(runs_legal.shape[0]):
                    v += p[k] * WP[b - 1, w, ri - runs_legal[k]]
                sw_legal, sw_ill, s_wd, s_nb, self_p = 0.0, 0.0, 0.0, 0.0, 0.0
                for j in range(J):
                    sw_legal += fWL[ph, j] * WP[b - 1, w - 1, ri - j]
                    sw_ill += fWI[ph, j] * WP[b, w - 1, ri - j]
                    if j == 0:
                        self_p += p[iWD] * fWD[ph, 0] + p[iNB] * fNB[ph, 0]
                    else:
                        s_wd += fWD[ph, j] * WP[b, w, ri - j]
                        s_nb += fNB[ph, j] * WP[b, w, ri - j]
                v += p[iW] * (qL[ph] * sw_legal + (1.0 - qL[ph]) * sw_ill)
                v += p[iWD] * s_wd + p[iNB] * s_nb
                WP[b, w, ri] = v / (1.0 - self_p)
    return WP


def state_grid(season, tier):
    b, w, r = np.meshgrid(np.arange(1, BMAX + 1), np.arange(1, WMAX + 1), np.arange(1, RMAX + 1), indexing="ij")
    g = pd.DataFrame({"b": b.ravel().astype(float), "w": w.ravel(), "r": r.ravel().astype(float)})
    g["innings"] = 2
    g["rate"] = np.clip(6 * g.r / g.b, 0, 36)
    g["phase"] = np.array(PHASES)[phase_of_b(g.b.values.astype(int))]
    g["format_tier"] = tier
    g["season"] = season
    return g


# ------------------------------------------------------------------ chase rows
def chase_rows_2022():
    d = pd.read_csv(os.path.join(OUT, "deliveries_clean.csv"), low_memory=False,
                    usecols=["match_id", "innings", "innings_ball_num", "is_super_over", "batting_team_id",
                             "balls_remaining", "wkts_in_hand", "runs_required"])
    mc = pd.read_csv(os.path.join(OUT, "match_context_enriched.csv"),
                     usecols=["match_id", "comp_name", "match_date", "home_team", "away_team", "dls_affected",
                              "win_team_id", "super_over", "result"])
    ipl = mc.comp_name.str.startswith("IPL")
    full = mc.home_team.isin(FULL_MEMBERS) & mc.away_team.isin(FULL_MEMBERS)
    mc["format_tier"] = np.where(ipl, "IPL", np.where(full, "full_member", "associate"))
    mc["season"] = pd.to_datetime(mc.match_date).dt.year
    mc["tie"] = mc.super_over.astype(bool) | mc.result.fillna("").str.contains("tie", case=False)
    d = d[(d.innings == 2) & ~d.is_super_over.astype(bool)].merge(mc, on="match_id", how="left")
    log = {"chase_deliveries_2022plus": len(d)}
    m = ~d.dls_affected.astype(bool) & ~d.match_id.isin(CONFLICT_MATCHES); log["excluded_dls_or_conflict"] = int((~m).sum()); d = d[m]
    d = d.assign(y=np.where(d.tie, 0.5, np.where(d.win_team_id.notna(),
                                                  (d.win_team_id == d.batting_team_id).astype(float), np.nan)))
    m = d.y.notna(); log["excluded_no_result"] = int((~m).sum()); d = d[m]
    m = (d.runs_required > 0) & (d.balls_remaining >= 1) & (d.wkts_in_hand >= 1)
    log["excluded_state_r_le_0_or_b_or_w_0"] = int((~m).sum()); d = d[m]
    d = d[d.season.isin(SEASONS)]
    return d[["match_id", "innings_ball_num", "season", "format_tier", "balls_remaining", "wkts_in_hand",
              "runs_required", "y"]], log


def chase_rows_history():
    h = pd.read_parquet(os.path.join(MODEL, "history_deliveries.parquet"),
                        columns=["match_id", "innings_ball_num", "innings", "is_super_over", "dls_affected", "season",
                                 "format_tier", "balls_remaining", "wkts_in_hand", "runs_required", "batting_team_won"])
    h = h[(h.innings == 2) & ~h.is_super_over.astype(bool) & ~h.dls_affected.astype(bool) & h.batting_team_won.notna()
          & (h.runs_required > 0) & (h.balls_remaining >= 1) & (h.wkts_in_hand >= 1) & h.season.isin(ALL_SEASONS)]
    return h.rename(columns={"batting_team_won": "y"})[["match_id", "innings_ball_num", "season", "format_tier",
                                                       "balls_remaining", "wkts_in_hand", "runs_required", "y"]]


def prep_rows(d):
    d = d.copy()
    d["b"] = d.balls_remaining.astype(int).clip(1, BMAX)
    d["w"] = d.wkts_in_hand.astype(int)
    d["r"] = np.round(d.runs_required).astype(int).clip(1, RMAX)
    d["band"] = BAND_B[d.b.values]
    return d.sort_values(["match_id", "innings_ball_num"]).reset_index(drop=True)


# ------------------------------------------------------------------ calibration map g
def lgt(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))


def bll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier(p, y): return (p - y) ** 2


class BandMap:
    """Monotone map of logit(WP) per balls-remaining band; method 'isotonic' or 'logistic'."""
    def __init__(self, method): self.method = method

    def fit(self, wp, y, band):
        self.m = {}
        x = lgt(wp)
        for k in range(len(BANDS)):
            s = band == k
            if self.method == "isotonic":
                self.m[k] = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(x[s], y[s])
            else:
                xx, yy = x[s], y[s]; tie = yy == 0.5
                X = np.concatenate([xx[~tie], xx[tie], xx[tie]])[:, None]
                Y = np.concatenate([yy[~tie], np.zeros(tie.sum()), np.ones(tie.sum())])
                Wt = np.concatenate([np.ones((~tie).sum()), np.full(2 * tie.sum(), 0.5)])
                self.m[k] = LogisticRegression(C=1e6, max_iter=1000).fit(X, Y, sample_weight=Wt)
        return self

    def __call__(self, wp, band):
        x = lgt(wp); out = np.empty(len(x))
        for k in range(len(BANDS)):
            s = band == k
            if s.any():
                out[s] = (self.m[k].predict(x[s]) if self.method == "isotonic"
                          else self.m[k].predict_proba(x[s][:, None])[:, 1])
        return np.clip(out, 1e-4, 1 - 1e-4)

    def params(self):
        if self.method == "logistic":
            return {BANDS[k]: [float(m.intercept_[0]), float(m.coef_[0, 0])] for k, m in self.m.items()}
        return {BANDS[k]: {"n_thresholds": int(len(m.X_thresholds_))} for k, m in self.m.items()}


def fit_g(rows, s):
    """Choose isotonic vs logistic on season s-1 (fitted on < s-1), refit chosen on < s."""
    tr, va = rows[rows.season < s - 1], rows[rows.season == s - 1]
    sel = {}
    for meth in ("isotonic", "logistic"):
        g = BandMap(meth).fit(tr.wp.values, tr.y.values, tr.band.values)
        sel[meth] = float(bll(g(va.wp.values, va.band.values), va.y.values).mean())
    meth = min(sel, key=sel.get)
    tr = rows[rows.season < s]
    g = BandMap(meth).fit(tr.wp.values, tr.y.values, tr.band.values)
    return g, {"season": s, "chosen": meth, "prev_season_logloss_isotonic": sel["isotonic"],
               "prev_season_logloss_logistic": sel["logistic"], "n_fit_balls": len(tr),
               "logistic_slope_min": (min(v[1] for v in g.params().values()) if meth == "logistic" else None)}


def apply_g_table(g, WP):
    """g on live cells of a [b, w, r] table; terminal cells keep exact values."""
    G = terminal_table()
    bb, ww, rr = np.meshgrid(np.arange(1, BMAX + 1), np.arange(1, WMAX + 1), np.arange(1, RMAX + 1), indexing="ij")
    G[1:, 1:, 1:] = g(WP[1:, 1:, 1:].ravel(), BAND_B[bb.ravel()]).reshape(BMAX, WMAX, RMAX)
    return G


# ------------------------------------------------------------------ table building
def build_variant(b, y, classes, cfg_b, variant):
    """LOSO tables for every season in ALL_SEASONS. Returns WP[season][tier] arrays, P for 2022-26, dists."""
    K = len(classes); cidx = {c: i for i, c in enumerate(classes)}
    runs_legal = np.array([int(c) for c in classes if c.isdigit()], dtype=np.int64)
    legal_idx = [cidx[str(c)] for c in runs_legal]
    WPd, Pd, dists, platt = {}, {}, {}, {}
    for s in ALL_SEASONS:
        ts = time.perf_counter()
        tr = (b.season != s).values
        bst = train(b[tr], y[tr], FEATS_B, cfg_b)
        pl = None
        if variant == "platt":
            tr2, prev = ((b.season != s) & (b.season != s - 1)).values, (b.season == s - 1).values
            pl = Platt().fit(train(b[tr2], y[tr2], FEATS_B, cfg_b).predict(frame(b[prev], FEATS_B)), y[prev])
            platt[s] = pl.params(classes)
        dist = run_dists(b[tr]); dists[s] = dist
        for tier in TIERS:
            p = bst.predict(frame(state_grid(s, tier), FEATS_B))
            if pl is not None:
                p = pl.transform(p)
            P = np.zeros((BMAX + 1, WMAX + 1, RMAX + 1, K))
            P[1:, 1:, 1:] = p.reshape(BMAX, WMAX, RMAX, K)
            nl = len(legal_idx)
            P_re = np.concatenate([P[..., legal_idx], P[..., [cidx["W"], cidx["WD"], cidx["NB"]]]], axis=-1)
            WP = recurse(P_re, runs_legal, nl, nl + 1, nl + 2, dist["fWD"], dist["fNB"], dist["fWL"], dist["fWI"],
                         dist["qL"], PHASE_B)
            WPd[(s, tier)] = WP[:, :, OFF:].copy()
            if s in SEASONS:
                Pd[(s, tier)] = P.astype(np.float32)
        print(f"[{variant}] season {s}: {time.perf_counter()-ts:.0f}s", flush=True)
    return WPd, Pd, dists, platt


def expected_next(TAB, Pd, dists, rows, classes):
    """E_o[TAB(next state)] for each row, using the outcome probabilities and run distributions of its table."""
    cidx = {c: i for i, c in enumerate(classes)}
    runs_legal = [int(c) for c in classes if c.isdigit()]
    e = np.zeros(len(rows))
    for (s, tier), grp in rows.groupby(["season", "format_tier"]).groups.items():
        ix = np.asarray(grp)
        T, P, d = TAB[(s, tier)], Pd[(s, tier)], dists[s]
        bi, wi, ri = rows.b.values[ix], rows.w.values[ix], rows.r.values[ix]
        p = P[bi, wi, ri].astype(float); ph = PHASE_B[bi]
        v = np.zeros(len(ix))
        for c in runs_legal:
            v += p[:, cidx[str(c)]] * T[bi - 1, wi, np.maximum(ri - c, 0)]
        sWL = sum(d["fWL"][ph, j] * T[bi - 1, wi - 1, np.maximum(ri - j, 0)] for j in range(JMAX + 1))
        sWI = sum(d["fWI"][ph, j] * T[bi, wi - 1, np.maximum(ri - j, 0)] for j in range(JMAX + 1))
        sWD = sum(d["fWD"][ph, j] * T[bi, wi, np.maximum(ri - j, 0)] for j in range(JMAX + 1))
        sNB = sum(d["fNB"][ph, j] * T[bi, wi, np.maximum(ri - j, 0)] for j in range(JMAX + 1))
        v += p[:, cidx["W"]] * (d["qL"][ph] * sWL + (1 - d["qL"][ph]) * sWI) + p[:, cidx["WD"]] * sWD + p[:, cidx["NB"]] * sNB
        e[ix] = v
    return e


def checks(TAB, label):
    """Terminal-value and monotonicity checks over the 15 tables of 2022-26."""
    A = np.stack([TAB[(s, t)] for s in SEASONS for t in TIERS])
    chk = {"table": label,
           "WP(r<=0)!=1": int((np.abs(A[:, :, :, 0] - 1) > 1e-12).sum()),
           "WP(b=0,r=1)!=0.5": int((np.abs(A[:, 0, :, 1] - 0.5) > 1e-12).sum()),
           "WP(w=0,r=1)!=0.5": int((np.abs(A[:, :, 0, 1] - 0.5) > 1e-12).sum()),
           "WP(b=0,r>=2)!=0": int((np.abs(A[:, 0, :, 2:]) > 1e-12).sum()),
           "WP(w=0,r>=2)!=0": int((np.abs(A[:, :, 0, 2:]) > 1e-12).sum()),
           "out_of_[0,1]": int(((A < -1e-12) | (A > 1 + 1e-12)).sum())}
    mono = []
    for name, D, bad in (("increasing_in_b", np.diff(A[:, :, 1:, 1:], axis=1), lambda x: x < -1e-12),
                         ("increasing_in_w", np.diff(A[:, 1:, :, 1:], axis=2), lambda x: x < -1e-12),
                         ("decreasing_in_r", np.diff(A[:, 1:, 1:, :], axis=3), lambda x: x > 1e-12)):
        v = bad(D); mag = np.abs(np.where(v, D, 0))
        mono.append({"table": label, "check": name, "pairs_checked": int(v.size), "violations": int(v.sum()),
                     "violations_>1e-4": int((mag > 1e-4).sum()), "max_violation": float(mag.max())})
    return chk, mono


# ------------------------------------------------------------------ metrics
def boot_pairs(df, pairs, n=N_BOOT, seed=SEED):
    """Point and 95% match-bootstrap CI for Brier, log loss, AUC (ties excluded) of every column and pair diff."""
    cols = sorted({c for p in pairs for c in p})
    rng = np.random.default_rng(seed)
    um, inv = np.unique(df.match_id.values, return_inverse=True)
    order = np.argsort(inv, kind="stable"); starts = np.searchsorted(inv[order], np.arange(len(um)))
    ends = np.append(starts[1:], len(inv))
    y = df.y.values; V = {c: df[c].values for c in cols}

    def met(idx):
        yy = y[idx]; nt = yy != 0.5; o = {}
        for c in cols:
            p = V[c][idx]
            o[("brier", c)] = brier(p, yy).mean(); o[("logloss", c)] = bll(p, yy).mean()
            o[("auc", c)] = roc_auc_score(yy[nt], p[nt])
        return o
    pt = met(np.arange(len(df)))
    R = pd.DataFrame([met(np.concatenate([order[starts[i]:ends[i]] for i in rng.integers(0, len(um), len(um))]))
                      for _ in range(n)])
    rows = []
    for m in ("brier", "logloss", "auc"):
        for c in cols:
            lo, hi = np.percentile(R[(m, c)], [2.5, 97.5])
            rows.append({"metric": m, "model": c, "value": pt[(m, c)], "lo95": lo, "hi95": hi})
        for a, c in pairs:
            lo, hi = np.percentile(R[(m, a)] - R[(m, c)], [2.5, 97.5])
            rows.append({"metric": m, "model": f"{a} - {c}", "value": pt[(m, a)] - pt[(m, c)], "lo95": lo, "hi95": hi})
    return pd.DataFrame(rows).drop_duplicates(["metric", "model"])


def reliability(df, col):
    dec = pd.qcut(df[col].rank(method="first"), 10, labels=False) + 1
    return (pd.DataFrame({"decile": dec, "p": df[col], "y": df.y, "m": df.match_id}).groupby("decile")
            .agg(n_balls=("y", "size"), n_matches=("m", "nunique"), mean_predicted=("p", "mean"),
                 actual_win_rate=("y", "mean")).reset_index())


def logit_baseline(ev, hist):
    def X(d):
        bb, w, r = d.b.values.astype(float), d.w.values.astype(float), d.r.values.astype(float)
        return np.column_stack([bb, w, r, np.clip(6 * r / bb, 0, 36)])
    out = np.zeros(len(ev))
    for s in SEASONS:
        tr = pd.concat([hist, ev[ev.season != s]])
        x, yv = X(tr), tr.y.values; tie = yv == 0.5
        XX = np.vstack([x[~tie], x[tie], x[tie]]); Y = np.concatenate([yv[~tie], np.zeros(tie.sum()), np.ones(tie.sum())])
        Wt = np.concatenate([np.ones((~tie).sum()), np.full(2 * tie.sum(), 0.5)])
        sc = StandardScaler().fit(XX)
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(XX), Y, sample_weight=Wt)
        sel = (ev.season == s).values
        out[sel] = m.predict_proba(sc.transform(X(ev[sel])))[:, 1]
    return out


# ------------------------------------------------------------------ worked examples
def example(WP, G, P, d, classes, bb, ww, rr):
    cidx = {c: i for i, c in enumerate(classes)}
    p = P[bb, ww, rr].astype(float); ph = PHASE_B[bb]
    lines = []
    for c in [c for c in classes if c.isdigit()]:
        k = int(c); nxt = (bb - 1, ww, max(rr - k, 0))
        lines.append((c, p[cidx[c]], f"WP{nxt}", WP[nxt]))
    wl = [(j, d["fWL"][ph, j], WP[bb - 1, ww - 1, max(rr - j, 0)]) for j in range(JMAX + 1) if d["fWL"][ph, j] > 0]
    wi = [(j, d["fWI"][ph, j], WP[bb, ww - 1, max(rr - j, 0)]) for j in range(JMAX + 1) if d["fWI"][ph, j] > 0]
    vW = d["qL"][ph] * sum(f * v for _, f, v in wl) + (1 - d["qL"][ph]) * sum(f * v for _, f, v in wi)
    lines.append(("W", p[cidx["W"]], f"legal (q={d['qL'][ph]:.4f}): " + ", ".join(
        f"j={j}: {f:.4f}x{v:.3f}" for j, f, v in wl) + f"; illegal: " + ", ".join(
        f"j={j}: {f:.4f}x{v:.3f}" for j, f, v in wi), vW))
    for c, key in (("WD", "fWD"), ("NB", "fNB")):
        val = sum(d[key][ph, j] * WP[bb, ww, max(rr - j, 0)] for j in range(JMAX + 1))
        lines.append((c, p[cidx[c]], f"E_j WP({bb},{ww},{rr}-j), j~{key}", val))
    return {"b": bb, "w": ww, "r": rr, "phase": PHASES[ph], "wp": WP[bb, ww, rr], "g": G[bb, ww, rr], "lines": lines,
            "sum": sum(x[1] * x[3] for x in lines)}


# ------------------------------------------------------------------ main
def main():
    t0 = time.perf_counter()
    fin = json.load(open(os.path.join(MODEL, "phase1a_final.json")))
    cfg_b = fin["B_config"]
    b, _ = load_balls()
    b, classes, _ = merge_rare(b)
    y = b.outcome.map({c: i for i, c in enumerate(classes)}).values

    ev_raw, elog = chase_rows_2022()
    hist = prep_rows(chase_rows_history())
    ev = prep_rows(ev_raw)
    allrows = pd.concat([hist, ev], ignore_index=True)
    elog.update({"rows_evaluated_2022_26": len(ev), "matches_evaluated": int(ev.match_id.nunique()),
                 "tie_matches": int(ev[ev.y == 0.5].match_id.nunique()),
                 "history_chase_balls_2008_21_for_g": len(hist)})

    res = {}
    for variant in ("plain", "platt"):
        WPd, Pd, dists, platt = build_variant(b, y, classes, cfg_b, variant)
        key = list(zip(allrows.season, allrows.format_tier))
        wp = np.empty(len(allrows))
        for (s, t), ix in allrows.groupby(["season", "format_tier"]).groups.items():
            ix = np.asarray(ix)
            wp[ix] = WPd[(s, t)][allrows.b.values[ix], allrows.w.values[ix], allrows.r.values[ix]]
        rows = allrows.assign(wp=wp)
        gsel, Gd, gmaps = [], {}, {}
        e_rows = rows[rows.season.isin(SEASONS)].reset_index(drop=True)
        cal = np.empty(len(e_rows))
        for s in SEASONS:
            g, info = fit_g(rows, s); gsel.append(info); gmaps[s] = g
            m = (e_rows.season == s).values
            cal[m] = g(e_rows.wp.values[m], e_rows.band.values[m])
            for t in TIERS:
                Gd[(s, t)] = apply_g_table(g, WPd[(s, t)])
        e_rows["wp_cal"] = cal
        # martingale: raw (exact) and after g
        e_raw = expected_next(WPd, Pd, dists, e_rows, classes)
        e_cal = expected_next(Gd, Pd, dists, e_rows, classes)
        mart = {"variant": variant, "balls": len(e_rows),
                "raw_mean_abs": float(np.abs(e_raw - e_rows.wp).mean()), "raw_max_abs": float(np.abs(e_raw - e_rows.wp).max()),
                "g_mean_abs": float(np.abs(e_cal - e_rows.wp_cal).mean()),
                "g_p99_abs": float(np.percentile(np.abs(e_cal - e_rows.wp_cal), 99)),
                "g_max_abs": float(np.abs(e_cal - e_rows.wp_cal).max()),
                "g_mean_signed": float((e_cal - e_rows.wp_cal).mean())}
        by_band = pd.DataFrame({"band": e_rows.band.map(lambda k: f"{BANDS[k][0]}-{BANDS[k][1]}"),
                                "d": np.abs(e_cal - e_rows.wp_cal)}).groupby("band").d.agg(["mean", "max"]).reset_index()
        chk_raw, mono_raw = checks(WPd, f"{variant} raw")
        chk_g, mono_g = checks(Gd, f"{variant} g")
        res[variant] = dict(WPd={k: v for k, v in WPd.items() if k[0] in SEASONS}, Gd=Gd, Pd=Pd, dists=dists,
                            platt=platt, gsel=pd.DataFrame(gsel), gmaps=gmaps, e_rows=e_rows, mart=mart,
                            mart_band=by_band, chk=[chk_raw, chk_g], mono=mono_raw + mono_g)
        del WPd

    # evaluation frame
    E = res["plain"]["e_rows"][["match_id", "innings_ball_num", "season", "format_tier", "b", "w", "r", "band", "y"]].copy()
    for v in ("plain", "platt"):
        r_ = res[v]["e_rows"]
        assert (r_.match_id.values == E.match_id.values).all()
        E[f"{v}_raw"], E[f"{v}_g"] = r_.wp.values, r_.wp_cal.values
    E["logistic"] = logit_baseline(E, hist[hist.season <= 2021][["b", "w", "r", "y", "season"]])
    met = boot_pairs(E, [("plain_g", "plain_raw"), ("platt_g", "platt_raw"), ("platt_g", "plain_g"),
                         ("plain_g", "logistic"), ("platt_g", "logistic")])
    d = met[(met.metric == "logloss") & (met.model == "platt_g - plain_g")].iloc[0]
    chosen = "platt" if (d.value < 0 and d.hi95 < 0) else "plain"
    why = (f"calibrated log loss platt - plain = {d.value:+.5f} [{d.lo95:+.5f}, {d.hi95:+.5f}]; "
           + ("Platt lower with CI excluding 0" if chosen == "platt" else
              "not lower with CI excluding 0 -> plain B (simpler)"))
    print("chosen:", chosen, why, flush=True)

    # store tables (both variants; chosen one also as the canonical file)
    for v in ("plain", "platt"):
        R = res[v]; parts = []
        bb, ww, rr = np.meshgrid(np.arange(BMAX + 1), np.arange(WMAX + 1), np.arange(RMAX + 1), indexing="ij")
        for s in SEASONS:
            for t in TIERS:
                T = pd.DataFrame({"season": np.int16(s), "tier": t, "b": bb.ravel().astype(np.int16),
                                  "w": ww.ravel().astype(np.int8), "r": rr.ravel().astype(np.int16),
                                  "wp_raw": R["WPd"][(s, t)].ravel(), "wp": R["Gd"][(s, t)].ravel()})
                for k, c in enumerate(classes):
                    T[f"p_{c}"] = R["Pd"][(s, t)][..., k].ravel()
                parts.append(T)
        T = pd.concat(parts, ignore_index=True); T["tier"] = T.tier.astype("category")
        T.to_parquet(os.path.join(MODEL, f"phase1b_wp_tables_{v}.parquet"), index=False, compression="zstd")
        if v == chosen:
            T.to_parquet(os.path.join(MODEL, "phase1b_wp_tables.parquet"), index=False, compression="zstd")
        del T, parts
    R = res[chosen]
    json.dump({"chosen_variant": chosen, "reason": why, "wp_column_for_WPA_LI": "wp (= g(wp_raw); terminal states exact)",
               "bands": BANDS, "g_selection": R["gsel"].to_dict("records"),
               "g_params": {s: {str(k): v for k, v in g.params().items()} for s, g in R["gmaps"].items()},
               "g_method_by_season": {s: g.method for s, g in R["gmaps"].items()},
               "platt_by_season": {s: p for s, p in R["platt"].items() if s in SEASONS},
               "run_distributions": {s: {k: (v.round(6).tolist() if isinstance(v, np.ndarray) else v)
                                         for k, v in R["dists"][s].items()} for s in SEASONS}},
              open(os.path.join(MODEL, "phase1b_recursion_inputs.json"), "w"), indent=1, default=float)
    E.to_parquet(os.path.join(MODEL, "phase1b_chase_eval.parquet"), index=False)

    # worked examples on IPL 2026, chosen variant (and the 1-off-1 for the other variant)
    ex_states = [("12 off 6, 4 wkts", 6, 4, 12), ("40 off 24, 7 wkts", 24, 7, 40), ("8 off 12, 2 wkts", 12, 2, 8),
                 ("100 off 60, 9 wkts", 60, 9, 100), ("1 off 1, 1 wkt", 1, 1, 1)]
    exs = []
    for nm, bb_, ww_, rr_ in ex_states:
        e_ = example(R["WPd"][(2026, "IPL")], R["Gd"][(2026, "IPL")], R["Pd"][(2026, "IPL")], R["dists"][2026],
                     classes, bb_, ww_, rr_)
        e_["name"] = nm; exs.append(e_)
    other = "platt" if chosen == "plain" else "plain"
    o = res[other]
    ex_other = example(o["WPd"][(2026, "IPL")], o["Gd"][(2026, "IPL")], o["Pd"][(2026, "IPL")], o["dists"][2026], classes, 1, 1, 1)
    write_report(res, met, E, chosen, why, exs, ex_other, other, elog, time.perf_counter() - t0)


def write_report(res, met, E, chosen, why, exs, ex_other, other, elog, rt):
    f = lambda x: f"{x:.4f}" if isinstance(x, float) else x
    R = res[chosen]
    chk = pd.DataFrame([c for v in res for c in res[v]["chk"]])
    mono = pd.DataFrame([m for v in res for m in res[v]["mono"]])
    mart = pd.DataFrame([res[v]["mart"] for v in res])
    gsel = pd.concat([res[v]["gsel"].assign(variant=v) for v in res])
    rel = []
    for c, lab in ((f"{chosen}_raw", "raw WP"), (f"{chosen}_g", "g(WP)"), ("logistic", "logistic")):
        rel.append(reliability(E, c).rename(columns={"mean_predicted": f"pred_{lab}", "actual_win_rate": f"actual_{lab}"})
                   .drop(columns=["n_balls", "n_matches"]))
    relt = rel[0].merge(rel[1], on="decile").merge(rel[2], on="decile")
    # deciles defined on each model's own ranking; also g on raw-WP deciles
    dec = pd.qcut(E[f"{chosen}_raw"].rank(method="first"), 10, labels=False) + 1
    same = E.assign(dec=dec).groupby("dec").agg(n=("y", "size"), raw=(f"{chosen}_raw", "mean"),
                                                g=(f"{chosen}_g", "mean"), actual=("y", "mean")).reset_index()
    def ex_md(e, title):
        L = [f"### {title}  (b={e['b']}, w={e['w']}, r={e['r']}, phase {e['phase']}): raw WP = {e['wp']:.4f}, g(WP) = {e['g']:.4f}", "",
             "| outcome | p | continuation | value | p x value |", "|---|---|---|---|---|"]
        for c, p, desc, val in e["lines"]:
            L.append(f"| {c} | {p:.4f} | {desc} | {val:.4f} | {p*val:.4f} |")
        L += [f"| **sum** | {sum(x[1] for x in e['lines']):.4f} | | | **{e['sum']:.4f}** |", ""]
        return L
    L = ["# Phase 1b (v2) - chase win probability: tie fix, calibration map g, Platt vs plain B", "",
         f"**Decision: {chosen} B** ({why}). WPA and LI use `wp` = g(raw WP) from "
         "`out/model/phase1b_wp_tables.parquet` (also stored: `phase1b_wp_tables_plain.parquet`, `_platt.parquet`; "
         "columns wp_raw, wp, outcome probabilities).", "",
         "## Fix 1: tie boundary", "",
         "Innings over (b = 0 or w = 0) with r = 1 is now 0.5 (v1: 0). All 15 tables x 2 variants rebuilt.", "",
         *ex_md(exs[-1], f"Need 1 off 1, 1 wicket ({chosen}, IPL 2026)"),
         "v1 (tie = 0) gave 0.5929: the dot ball (now WP(0,1,1) = 0.5) and the legal wicket with no run "
         "(now WP(0,0,1) = 0.5) were valued 0.", "",
         *ex_md(ex_other, f"Need 1 off 1, 1 wicket ({other}, IPL 2026)"),
         "## Checks after the fix (2022-26 tables)", "",
         chk.to_markdown(index=False), "",
         mono.map(lambda x: f"{x:.2e}" if isinstance(x, float) else x).to_markdown(index=False), "",
         "Monotonicity in b: accepted as is (LightGBM steps in b and rate; largest raw violation ~0.005). "
         "g is monotone within a band, so it cannot add w or r violations but can shift values at band edges "
         "(b = 30/31, 60/61, 90/91).", "",
         "## Fix 2: calibration map g", "",
         "Per season s and band (b 1-30, 31-60, 61-90, 91-120), fitted on OOF chase balls of seasons < s "
         f"(2008+, {elog['history_chase_balls_2008_21_for_g']:,} history balls plus earlier 2022-26 seasons). "
         "Method chosen per season by season s-1 log loss:", "",
         gsel.map(lambda x: f"{x:.5f}" if isinstance(x, float) else x).to_markdown(index=False), "",
         "### Martingale", "",
         mart.map(lambda x: f"{x:.2e}" if isinstance(x, float) else x).to_markdown(index=False), "",
         f"After g by band ({chosen}):", "", R["mart_band"].map(lambda x: f"{x:.2e}" if isinstance(x, float) else x).to_markdown(index=False), "",
         "### Chase calibration 2022-26: before / after", "",
         "Rows: " + "; ".join(f"{k} {v:,}" for k, v in elog.items()), "",
         f"v1 (tie bug, Platt raw): Brier {V1_TIEBUG['brier']}, log loss {V1_TIEBUG['logloss']}, AUC {V1_TIEBUG['auc']}.", "",
         "Ties: label 0.5, kept in Brier and log loss, excluded from AUC. 95% CIs from 1000 match-bootstrap resamples.", "",
         met.map(lambda x: f"{x:.5f}" if isinstance(x, float) else x).to_markdown(index=False), "",
         f"Reliability by decile ({chosen}; each column's deciles on its own ranking):", "",
         relt.map(f).to_markdown(index=False), "",
         "Same balls, deciles of raw WP:", "", same.map(f).to_markdown(index=False), "",
         "## Worked examples (IPL 2026, " + chosen + ")", ""]
    for e in exs:
        L += ex_md(e, e["name"])
    L += ["## Limitations", "",
          "- Free hits ignored. Class 4 = 4 runs (absorbs 5s), 6 = 6 runs (absorbs 7s). Phase assumes 20 overs.",
          "- Raw WP is not monotone in b in about 6% of neighbouring cells (max ~0.005, mostly early-innings b and high WP); accepted, no fix.",
          "- g is fitted on history WP from LOSO models that saw the test season's ball outcomes (never its results).",
          "- After g the table is no longer an exact martingale (see above).", "", f"Runtime {rt:.0f} s.", ""]
    open(os.path.join(REP, "phase1b_summary.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
