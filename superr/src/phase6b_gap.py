"""Phase 6b - "normal vs pressure" gap as a DESCRIPTIVE column (scores, ranks and eligibility unchanged).

normal   = 100 * sum(v_adj) / sum(LI) over balls with LI < 1.5
pressure = 100 * sum(v_adj) / sum(LI) over balls with LI >= 1.5
gap = pressure - normal; 80% range from 500 resamples of each player's matches (with replacement).
v_adj = Phase 3 corrected ball value (opponent + associate tier); batters on balls faced, bowlers on balls bowled.
Repeatability: each player's matches numbered chronologically, odd vs even halves; eligible players with >= 20 pressure
balls in each half; Spearman of gap(odd) vs gap(even) and of normal(odd) vs normal(even), 1000-player bootstrap CI.
"""
import json, os, re, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase1a_outcome import MODEL, SEED
from variant import mpath, vdir, FLOOR
from phase4c_projection import resample_weights

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIN = vdir(os.path.join(ROOT, "reports", "final")); FIG = os.path.join(FIN, "figures")
REP5 = vdir(os.path.join(ROOT, "reports", "phase5")); DASH = vdir(os.path.join(ROOT, "dash"))
THR, N_BOOT_M, N_BOOT_P, MIN_HALF, CAREFUL = 1.5, 500, 1000, 20, 60
COLS = ["Normal (win % per average ball)", "Pressure (balls >= 1.5x)", "Pressure gap", "Gap 80% range", "Pressure balls",
        "Normal balls", "Gap flag"]
BLUE, GREY, INK, MUTED, BAND = "#2a78d6", "#8d8c85", "#1f1f1e", "#6b6a64", "#cde2fb"


def player_match(side):
    a = pd.read_parquet(mpath("phase3_ball_values_adj.parquet"),
                        columns=["match_id", "match_date", "striker_id", "bowler_id", "v_adj_bat", "v_adj_bowl", "li"])
    pid, v = ("striker_id", "v_adj_bat") if side == "bat" else ("bowler_id", "v_adj_bowl")
    a = a[a[v].notna()]
    hi = a.li.values >= THR
    t = pd.DataFrame({"player": a[pid].values, "match_id": a.match_id.values, "date": a.match_date.values,
                      "Vp": np.where(hi, a[v], 0.0), "Lp": np.where(hi, a.li, 0.0), "np_": hi.astype(int),
                      "Vn": np.where(~hi, a[v], 0.0), "Ln": np.where(~hi, a.li, 0.0), "nn": (~hi).astype(int)})
    pm = t.groupby(["player", "match_id"]).agg(date=("date", "first"), Vp=("Vp", "sum"), Lp=("Lp", "sum"), np_=("np_", "sum"),
                                               Vn=("Vn", "sum"), Ln=("Ln", "sum"), nn=("nn", "sum")).reset_index()
    pm = pm.sort_values(["player", "date", "match_id"]).reset_index(drop=True)
    pm["idx"] = pm.groupby("player").cumcount()
    return pm


def summarise(pm, w=None):
    w = np.ones(len(pm)) if w is None else w
    g = pd.DataFrame({c: pm[c].values * w for c in ("Vp", "Lp", "np_", "Vn", "Ln", "nn")}); g["player"] = pm.player.values
    g = g.groupby("player").sum()
    normal = 100 * g.Vn / g.Ln.where(g.Ln > 0); pressure = 100 * g.Vp / g.Lp.where(g.Lp > 0)
    return pd.DataFrame({"normal": normal, "pressure": pressure, "gap": pressure - normal, "pressure_balls": g.np_, "normal_balls": g.nn})


def gap_table(pm):
    base = summarise(pm)
    rng = np.random.default_rng(SEED + 61)
    D = np.column_stack([summarise(pm, resample_weights(pm, rng)).gap.reindex(base.index).values for _ in range(N_BOOT_M)])
    base["lo80"], base["hi80"] = np.nanquantile(D, 0.1, axis=1), np.nanquantile(D, 0.9, axis=1)
    return base


def repeatability(pm, eligible):
    a, b = summarise(pm[pm.idx % 2 == 1]), summarise(pm[pm.idx % 2 == 0])
    j = a.join(b, lsuffix="_odd", rsuffix="_even", how="inner")
    j = j[j.index.isin(eligible) & (j.pressure_balls_odd >= MIN_HALF) & (j.pressure_balls_even >= MIN_HALF)].dropna()
    rng = np.random.default_rng(SEED + 62); n = len(j); idx = rng.integers(0, n, (N_BOOT_P, n))
    out = {"players": n}
    for c in ("gap", "normal", "pressure"):
        x, y = j[f"{c}_odd"].values, j[f"{c}_even"].values
        reps = [spearmanr(x[i], y[i]).statistic for i in idx]
        out[c] = {"rho": float(spearmanr(x, y).statistic), "lo95": float(np.percentile(reps, 2.5)), "hi95": float(np.percentile(reps, 97.5))}
    return out


def fmt_range(lo, hi):
    return "" if pd.isna(lo) else f"{lo:+.2f} to {hi:+.2f}"


def add_cols(df, G, id_series):
    g = G.reindex(id_series.values)
    df = df.drop(columns=[c for c in COLS if c in df.columns])
    df["Normal (win % per average ball)"] = g.normal.round(3).values
    df["Pressure (balls >= 1.5x)"] = g.pressure.round(3).values
    df["Pressure gap"] = g.gap.round(3).values
    df["Gap 80% range"] = [fmt_range(lo, hi) for lo, hi in zip(g.lo80, g.hi80)]
    df["Pressure balls"] = g.pressure_balls.fillna(0).astype(int).values
    df["Normal balls"] = g.normal_balls.fillna(0).astype(int).values
    df["Gap flag"] = np.where(g.pressure_balls.fillna(0).values < CAREFUL, "fewer than 60 pressure balls - read with care", "")
    return df


def figure(tops, rep):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    for ax, (side, t) in zip(axs, tops.items()):
        t = t.iloc[::-1]; y = np.arange(len(t))
        for yy, (_, r) in zip(y, t.iterrows()):
            ax.add_patch(plt.Rectangle((r.normal + r.lo80, yy - 0.28), r.hi80 - r.lo80, 0.56, color=BAND, lw=0, zorder=0))
            ax.plot([r.normal, r.pressure], [yy, yy], color=MUTED, lw=1.2, zorder=1)
        ax.scatter(t.normal, y, s=42, color=GREY, zorder=3, edgecolor="white", lw=1.2, label="normal (LI < 1.5)")
        ax.scatter(t.pressure, y, s=42, color=BLUE, zorder=4, edgecolor="white", lw=1.2, label="pressure (LI >= 1.5)")
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels([f"{int(r)}. {n}" + (" *" if pb < CAREFUL else "") for r, n, pb in zip(t["rank"], t["name"], t.pressure_balls)], fontsize=8.5)
        ax.set_title("Batters" if side == "bat" else "Bowlers", fontsize=10, color=INK, loc="left")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.grid(axis="x", color="#ececea", lw=0.8); ax.set_axisbelow(True)
        ax.set_xlabel("win-probability points per average ball")
    axs[0].legend(fontsize=8, frameon=False, loc="upper right")
    rb, rw = rep["bat"], rep["bowl"]
    fig.suptitle("Normal vs pressure value, final top 10 (2022-26). Band = 80% range of where the pressure dot could sit given the gap's uncertainty.",
                 fontsize=10.5, x=0.01, ha="left", color=INK)
    fig.text(0.01, -0.04, (f"Repeatability (odd vs even matches): gap rho = {rb['gap']['rho']:+.2f} [{rb['gap']['lo95']:+.2f}, {rb['gap']['hi95']:+.2f}] batters "
                           f"(n = {rb['players']}), {rw['gap']['rho']:+.2f} [{rw['gap']['lo95']:+.2f}, {rw['gap']['hi95']:+.2f}] bowlers (n = {rw['players']}); "
                           f"normal rho = {rb['normal']['rho']:+.2f} batters, {rw['normal']['rho']:+.2f} bowlers. The gap is descriptive and not used in the ranking. "
                           "* fewer than 60 pressure balls."), fontsize=8.3, color=MUTED, wrap=True)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, f"fig8_pressure_gap.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(DASH, exist_ok=True)
    res, tops, dash_rows, G_all = {}, {}, [], {}
    for side in ("bat", "bowl"):
        tech_p = mpath(f"phase4_rankings_{side}_all_players_technical.csv")
        tech = pd.read_csv(tech_p, index_col=0)
        pm = player_match(side)
        G = gap_table(pm); G_all[side] = G
        el = tech[tech.eligible == True].sort_values("rank_eligible")
        rep = repeatability(pm, set(el.index))
        top20 = G.reindex(el.head(20).index)
        excl = int(((top20.lo80 > 0) | (top20.hi80 < 0)).sum())
        res[side] = {**rep, "top20_ranges_excluding_0": excl, "top20_expected_by_chance": 0.2 * 20}
        # technical
        tech = tech.drop(columns=[c for c in tech.columns if c.startswith("gap_")], errors="ignore")
        g = G.reindex(tech.index)
        for c in ("normal", "pressure", "gap", "lo80", "hi80", "pressure_balls", "normal_balls"):
            tech[f"gap_{c}"] = g[c].values
        tech.to_csv(tech_p)
        # plain
        plain_p = mpath(f"phase4_rankings_{side}_all_players.csv")
        plain = pd.read_csv(plain_p)
        ids = plain.merge(tech.reset_index()[["player_id", "name", "rank_all"]], left_on=["Player", "Rank (all players)"],
                          right_on=["name", "rank_all"], how="left").player_id
        add_cols(plain, G, ids).to_csv(plain_p, index=False)
        t10 = G.reindex(el.head(10).index).assign(rank=el.head(10).rank_eligible.astype(int).values, name=el.head(10)["name"].values)
        tops[side] = t10
        for pid, r in el.iterrows():
            gg = G.loc[pid] if pid in G.index else None
            if gg is None: continue
            dash_rows.append({"player": r["name"], "role": "batter" if side == "bat" else "bowler", "rank": int(r.rank_eligible),
                              "normal": round(gg.normal, 4), "pressure": round(gg.pressure, 4), "gap": round(gg.gap, 4),
                              "lo80": round(gg.lo80, 4), "hi80": round(gg.hi80, 4), "pressure_balls": int(gg.pressure_balls)})
    pd.DataFrame(dash_rows).to_csv(os.path.join(DASH, "pressure_gap.csv"), index=False)
    # final_top10.csv
    ft_p = os.path.join(FIN, "final_top10.csv"); ft = pd.read_csv(ft_p)
    parts = []
    for side, role in (("bat", "batter"), ("bowl", "bowler")):
        tech = pd.read_csv(mpath(f"phase4_rankings_{side}_all_players_technical.csv"), index_col=0)
        sub = ft[ft.role == role]
        ids = sub.merge(tech.reset_index()[["player_id", "name", "rank_eligible"]], left_on=["player", "rank (eligible)"],
                        right_on=["name", "rank_eligible"], how="left").player_id
        parts.append(add_cols(sub.reset_index(drop=True), G_all[side], ids))
    pd.concat(parts).to_csv(ft_p, index=False)
    # key numbers
    kp = os.path.join(FIN, "key_numbers.json"); kn = json.load(open(kp))
    kn["pressure_gap"] = {"definition": "pressure (LI >= 1.5) minus normal (LI < 1.5), 100 x sum v_adj / sum LI; descriptive only",
                          "repeatability": res,
                          "top10": {s: [{"rank": int(r["rank"]), "player": r["name"], "normal": float(r.normal), "pressure": float(r.pressure),
                                         "gap": float(r.gap), "lo80": float(r.lo80), "hi80": float(r.hi80), "pressure_balls": int(r.pressure_balls)}
                                        for _, r in t.iterrows()] for s, t in tops.items()}}
    json.dump(kn, open(kp, "w"), indent=1, default=float)
    figure(tops, res)
    # reports
    rb, rw = res["bat"], res["bowl"]
    def tbl(t):
        return pd.DataFrame({"rank": t["rank"], "player": t["name"], "normal": t.normal.round(3), "pressure": t.pressure.round(3),
                             "gap": t.gap.round(3), "gap 80% range": [fmt_range(a, b) for a, b in zip(t.lo80, t.hi80)],
                             "pressure balls": t.pressure_balls.astype(int), "flag": np.where(t.pressure_balls < CAREFUL, "< 60 pressure balls", "")})
    L = ["# Phase 6b - normal vs pressure gap (descriptive; scores, ranks and eligibility unchanged)", "",
         "normal = 100 x sum(v_adj) / sum(LI) over balls with LI < 1.5; pressure = same over LI >= 1.5; gap = pressure - normal; "
         "80% range from 500 resamples of each player's matches.", "",
         "Repeatability (odd vs even matches; eligible players with >= 20 pressure balls in each half; 1000-player bootstrap):", "",
         pd.DataFrame([{"role": r_, "players": d["players"], **{f"{c} rho": f"{d[c]['rho']:+.3f} [{d[c]['lo95']:+.3f}, {d[c]['hi95']:+.3f}]" for c in ("gap", "normal", "pressure")},
                        "top-20 ranges excluding 0": d["top20_ranges_excluding_0"], "expected by chance": d["top20_expected_by_chance"]}
                       for r_, d in (("batters", rb), ("bowlers", rw))]).to_markdown(index=False), "",
         "Top 10 batters:", "", tbl(tops["bat"]).to_markdown(index=False), "", "Top 10 bowlers:", "", tbl(tops["bowl"]).to_markdown(index=False), ""]
    sp = os.path.join(REP5, "phase5_summary.md"); txt = open(sp, encoding="utf-8").read()
    txt = txt.split("\n# Phase 6b")[0].rstrip("\n") + "\n\n" + "\n".join(L)
    open(sp, "w", encoding="utf-8").write(txt)
    mp = os.path.join(FIN, "method_log.md"); ml = open(mp, encoding="utf-8").read()
    ml = re.sub(r"\n20\. Pressure-specific lift.*\n?", "\n", ml).rstrip("\n")
    def verdict(d, role):
        rep_ = d["gap"]["lo95"] > 0 or d["gap"]["hi95"] < 0
        return (f"{role}: the gap {'repeats' if rep_ else 'does not repeat'} between halves of a career (rho = {d['gap']['rho']:+.3f} "
                f"[{d['gap']['lo95']:+.3f}, {d['gap']['hi95']:+.3f}], n = {d['players']})")
    ml += ("\n20. Pressure-specific lift estimated in Phase 4 (between-player variance <= 0, shrunk to 0) and re-tested here with LI >= 1.5 vs < 1.5: "
           + verdict(rb, "batters") + "; " + verdict(rw, "bowlers")
           + f"; top-20 gap ranges excluding 0: batters {rb['top20_ranges_excluding_0']}, bowlers {rw['top20_ranges_excluding_0']} (4 expected by chance). "
           "Shown as a description of 2022-26, not used in the ranking.\n")
    open(mp, "w", encoding="utf-8").write(ml)
    print(json.dumps(res, indent=1, default=float))
    print(tbl(tops["bat"]).to_string(index=False)); print(tbl(tops["bowl"]).to_string(index=False))


if __name__ == "__main__":
    main()
