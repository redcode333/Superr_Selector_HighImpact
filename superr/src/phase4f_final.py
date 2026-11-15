"""Phase 4e - final selection: (i*) uncertainty-averaged overall rating A for BOTH roles (documented deviation).

Reuses the Phase 4d pipeline (src/phase4e_strict.py) with the candidate fixed to (i*). The five-split evaluation is
still run - it provides the pooled out-of-sample mapping to win-probability points and the comparison table.
Strict-set (LI >= 1.5) pressure record is a descriptive column only.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase4e_strict as P
from phase4e_strict import build, run_definition, eligibility, bias, write_files, forest, NAMES, ROOT, PHASES
from phase4_ratings import REP4
from phase1a_outcome import MODEL
from variant import mpath, vdir, FLOOR
from variant import mpath, vdir, FLOOR

PRIMARY = ("LI >= 1.5 (strict set, descriptive)", 1.5)
DEVIATION = ("Documented deviation from the Phase 4d rule: final score = (i*) for both roles. Batting P1 won only on "
             "point estimate (P1 - i* = +0.049 [-0.085, +0.174], no split CI excludes 0) and left 3 of 881 batters "
             "eligible (own-data share median 2%); both sensitivity definitions chose (i*); the bowling strict-set target "
             "reliability is negative (-0.127). The strict-set (LI >= 1.5) pressure record stays as a descriptive column.")


def main():
    t0 = time.perf_counter()
    P.choose = lambda means: ("istar", "fixed to (i*) - documented deviation (see top of report)")
    d, credit = build()
    dd = [p for p in (os.path.join(ROOT, "data", "unstop_dropdown_players.csv"),
                      os.path.join(os.path.dirname(ROOT), "data", "unstop_dropdown_players.csv")) if os.path.exists(p)]
    dropdown = set(pd.read_csv(dd[0]).iloc[:, 0].astype(str)) if dd else None
    out = {}
    for side in ("bat", "bowl"):
        R = run_definition(d, side, PRIMARY[0], PRIMARY[1], {}, with_selftest=True)
        R["F"] = eligibility(R["F"], dropdown)
        R["bias"] = bias(R["F"])
        out[side] = R
        write_files(R["F"], side); forest(R["F"], side)
        print(side, R["var_choice"], R["sts"].round(3).to_dict(), int(R["F"].eligible.sum()), flush=True)
    write_readme(out, dropdown)
    write_report(out, dropdown, time.perf_counter() - t0)
    write_howto(out)


def top10_table(F):
    el = F[F.eligible].sort_values("rank_eligible")
    rows = []
    for _, r in el.head(10).iterrows():
        below = el[el.rank_eligible > r.rank_eligible]
        clear = below[below.hi80 < r.cautious]
        rows.append({"rank": int(r.rank_eligible), "player": r["name"], "cautious": round(100 * r.cautious, 3),
                     "80% range": f"{100 * r.cautious:+.3f} to {100 * r.hi80:+.3f}",
                     "first rank clearly below": int(clear.rank_eligible.min()) if len(clear) else None,
                     "own-data %": round(100 * r.own_share, 1), "% PP": round(100 * r.share_Powerplay, 0) if pd.notna(r.share_Powerplay) else None,
                     "% mid": round(100 * r.share_Middle, 0) if pd.notna(r.share_Middle) else None,
                     "% death": round(100 * r.share_Death, 0) if pd.notna(r.share_Death) else None,
                     "% IPL": round(100 * r.share_IPL, 0) if pd.notna(r.share_IPL) else None,
                     "strict-set pressure record": round(100 * r.raw_value, 3) if pd.notna(r.raw_value) else None,
                     "matches": int(r.matches_all), "2025-26": int(r.matches_2025_26)})
    return pd.DataFrame(rows)


def explain(F, name):
    m = F[F["name"] == name]
    if not len(m):
        return f"- {name}: not found."
    r = m.iloc[0]
    el = F[F.eligible]
    tenth = el[el.rank_eligible == 10].iloc[0] if (el.rank_eligible == 10).any() else None
    s = (f"- {name}: overall rating A {100 * r.A:+.3f} (rank {int(F.A.rank(ascending=False)[m.index[0]])} of {len(F)} on A); "
         f"best estimate {100 * r.projected:+.3f}, cautious {100 * r.cautious:+.3f} (rank {int(r.rank_all)} of all), own-data share "
         f"{100 * r.own_share:.0f}%, {int(r.matches_all)} matches ({int(r.matches_2025_26)} in 2025-26). Eligible: {r.eligible_reason}.")
    if r.eligible:
        s += f" Eligible rank {int(r.rank_eligible)}."
    if tenth is not None:
        s += f" Eligible rank 10 has cautious {100 * tenth.cautious:+.3f}."
    return s


def write_readme(out, dropdown):
    L = ["# Rankings files", "",
         "`phase4_rankings_bat_all_players.csv`, `phase4_rankings_bowl_all_players.csv`: plain English, one header row. "
         "`*_technical.csv`: all model columns, raw units.", "",
         "**Unit.** Ratings are in win-probability points per average ball: +0.10 means the player's side gains on average "
         "0.10 percentage points of win probability per ball of average match importance, compared with a typical player in "
         "the same season, competition, innings and phase (corrected for opponent strength and associate matches).", "",
         "**Final score (both roles).** The uncertainty-averaged overall rating (i*): leverage-weighted value over all of "
         "the player's balls, with pressure balls (LI >= 1) and other balls combined by the empirical-Bayes model, averaged "
         "over 1000 plausible settings of that model, mapped to win-probability points by the out-of-sample fit.", "",
         "**Pressure record (strict set)** is descriptive only: actual LI-weighted win probability added on balls with LI >= 1.5.", "",
         "**Eligible.** Own-data share of the final score >= 50% and at least 5 matches in 2025-26"
         + (f" and at least {FLOOR} high-pressure balls (LI >= 1.5) in 2022-26" if FLOOR else "")
         + ("" if dropdown is not None else "; WARNING: `data/unstop_dropdown_players.csv` not found, names NOT checked against the Unstop dropdown") + ".", "",
         DEVIATION, ""]
    for side, R in out.items():
        L.append(f"- {'Batters' if side == 'bat' else 'Bowlers'}: uncertainty = {R['var_choice']}; {int(R['F'].eligible.sum())} eligible of {len(R['F'])}.")
    open(mpath("README_rankings.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def write_report(out, dropdown, rt):
    f3 = lambda x: f"{x:.3f}" if isinstance(x, (float, np.floating)) else x
    L = ["# Phase 4e - final selection", "", DEVIATION, "",
         "Previous runs kept for the record: run 1 (Phase 4b) rejected - environment confound + uncertainty bug; run 2 "
         "(Phase 4c) rejected - pressure definition too loose (LI >= 1), circular target; run 3 (Phase 4d) - strict "
         "definition, rule outcome not usable (3 eligible batters; negative bowling target reliability).", "",
         "Dropdown list: " + ("applied" if dropdown is not None else "**WARNING - `data/unstop_dropdown_players.csv` not found; names not checked**") + ".", ""]
    for side, R in out.items():
        F = R["F"]; el = F[F.eligible]
        role = "Batters" if side == "bat" else "Bowlers"
        m = R["means"].assign(candidate=R["means"].candidate.map(NAMES))
        rho_ipl = spearmanr(el.score, el.share_IPL, nan_policy="omit").statistic
        rho_de = spearmanr(el.score, el.share_Death, nan_policy="omit").statistic
        bt, rpp, _, gap = R["bias"]
        near = F[(F.own_share >= 0.45) & (F.own_share <= 0.55)]
        L += [f"## {role}", "",
              "Comparison on the strict-set target (for the record; not used to choose):", "", m.map(f3).to_markdown(index=False), "",
              f"Mapping to win-probability points: projected = {R['a']:.5f} + {R['b']:.5f} x (i*).", "",
              "Self-test of the uncertainty (200 repetitions; method test only):", "", R["sts"].reset_index().map(f3).to_markdown(index=False), "",
              f"Acceptance 0.75-0.85 overall and for < 15 matches -> **{R['var_choice']}** ({R['var_note']}).", "",
              f"Eligibility: {int(F.eligible.sum())} of {len(F)}. Own-data share < 50%: {int((~(F.own_share >= 0.5)).sum())}; "
              f"< 5 matches in 2025-26: {int((F.matches_2025_26 < 5).sum())}. Players with own-data share 45-55% have median "
              f"{near.matches_all.median():.0f} matches.", "",
              "### Eligible top 10", "", top10_table(F).to_markdown(index=False), "",
              f"Spearman (eligible players) of final score vs IPL share {rho_ipl:+.3f}; vs death share {rho_de:+.3f}; vs powerplay share {rpp:+.3f}.", "",
              "Phase shares (strict-set LI) - top 10 / top 25 vs all eligible:", "", bt.map(f3).to_markdown(index=False), "",
              ("Gaps > 20 points: " + ", ".join(f"{p} {100*g:+.0f}" for p, g in gap.items() if abs(g) > 0.2)) if any(abs(g) > 0.2 for g in gap.values())
              else "No phase-share gap between the top 10 and the eligible pool exceeds 20 points.", "",
              f"![forest](phase4_forest_top25_{side}.png)", ""]
        if side == "bat":
            L += ["### Glenn Maxwell and Reeza Hendricks", "", explain(F, "Glenn Maxwell"), explain(F, "Reeza Hendricks"), ""]
    L += [f"Runtime {rt:.0f} s.", ""]
    open(os.path.join(REP4, "phase4e_summary.md"), "w", encoding="utf-8").write("\n".join(L))


def write_howto(out):
    Rb, Rw = out["bat"], out["bowl"]
    Fb = Rb["F"]; el = Fb[Fb.eligible].sort_values("rank_eligible")
    ex = el.iloc[0]
    fin = el.head(10).sort_values("share_Death", ascending=False).iloc[0]
    exw = Rw["F"][Rw["F"].rank_eligible == 1].iloc[0]
    p = lambda x: f"{100 * x:+.2f}"
    L = ["# How the ranking works", "",
         "1. **Score every ball's importance.** A win-probability model gives each ball a leverage index (LI): how much the "
         "result could swing on it, with an average ball = 1. The last ball of a close chase can reach 30.",
         "2. **Measure what each ball did for the player's side**: the change in his team's chance of winning, compared with "
         "a typical ball at the same stage of the same season and competition, corrected for opponent strength and for "
         "matches against weaker associate teams.",
         "3. **Combine all his balls, weighted by importance.** Balls with LI of 1 or more and the rest are combined by a "
         "statistical model that also pulls short records towards average. We tested whether some players are better under "
         "pressure than their general level predicts: no such effect was detectable, so a player's ordinary balls also tell "
         "us about his pressure balls. The calculation is repeated over 1000 plausible versions of the model and averaged.",
         "4. **Honest ranges.** Each rating comes with an 80% likely range; a short record gives a wide range.",
         "5. **Rank by the cautious rating** (bottom of the 80% range), among players whose rating comes at least half from "
         "their own data and who played at least 5 matches in 2025-26.", "",
         f"## Worked example: {fin['name']} (batting rank {int(fin.rank_eligible)}; "
         f"{100 * fin.share_Death:.0f}% of his strict-set pressure in death overs)", "",
         f"- {int(fin.matches_all)} matches ({int(fin.matches_2025_26)} in 2025-26), {int(fin.n_p)} balls with LI >= 1.5 "
         f"({fin.strict_balls_per_match:.2f} per match).",
         f"- Overall rating {p(fin.A)} win-probability points per average ball; on the LI >= 1.5 balls alone he added "
         f"{p(fin.raw_value)} per unit of importance (descriptive).",
         f"- Final rating {p(fin.projected)}, likely range {p(fin.cautious)} to {p(fin.hi80)}; {100 * fin.own_share:.0f}% of the "
         f"rating comes from his own data; cautious rating **{p(fin.cautious)}**.",
         f"- Batting rank 1: {ex['name']} (cautious {p(ex.cautious)}). Bowling rank 1: {exw['name']} (cautious {p(exw.cautious)}).", "",
         "> **What we found**",
         "> - Pressure-specific skill is not detectable; underlying quality carries into pressure.",
         "> - Measures built only from the highest-pressure balls did not predict future pressure performance better than "
         "overall quality, and are too noisy to rank most players.",
         "> - Neighbouring ranks overlap: read the ranges, not just the order.", ""]
    open(os.path.join(REP4, "how_the_ranking_works.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
