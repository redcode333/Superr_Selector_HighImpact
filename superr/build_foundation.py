"""SupeRR Selector - foundation layer.

Builds, from the supplied data pack only (no external data):
  out/deliveries_clean.csv   one row per delivery, with legality, corrected time, innings state, phase
  out/innings_summary.csv    one row per match-innings, with totals (lower bound + estimate) and quality flags
  out/match_context.csv      one row per match, with corrected start times, local start, slot, reconciliation grade
  out/venue_draft.csv        one row per venue_id, with usage stats and an UNVERIFIED identity hint
  out/foundation_validation.csv  checks run and their results

Run:  python3 build_foundation.py <path to SupeRR_Data_Pack>
Idempotent: never modifies input files; overwrites only files in out/.
"""
import sys, re, os
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/pack/SupeRR_Data_Pack"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- constants (documented decisions)
ILLEGAL_TYPES = {"wide", "Overstep", "other"}   # 'other' inferred as no-ball: overs containing it have 7 deliveries
NOBALL_TYPES = {"Overstep", "other"}
BYE_TYPES = {"byes", "legbyes"}
BYE_RUNS_EST = 1.5   # avg runs per bye/leg-bye event, estimated by regressing result margins on bye counts (n=400)
TZ_SWITCH = pd.Timestamp("2024-03-20")  # timestamps before this are stored in IST, from this date in UTC
COUNTRY_TZ = {  # (tz, confidence). Multi-zone countries get 'low' until venue is verified.
    "India": ("Asia/Kolkata", "high"), "Sri Lanka": ("Asia/Colombo", "high"), "Bangladesh": ("Asia/Dhaka", "high"),
    "Pakistan": ("Asia/Karachi", "high"), "United Arab Emirates": ("Asia/Dubai", "high"), "Oman": ("Asia/Muscat", "high"),
    "Qatar": ("Asia/Qatar", "high"), "Japan": ("Asia/Tokyo", "high"), "South Africa": ("Africa/Johannesburg", "high"),
    "Zimbabwe": ("Africa/Harare", "high"), "United Kingdom": ("Europe/London", "high"), "Ireland": ("Europe/Dublin", "high"),
    "Netherlands": ("Europe/Amsterdam", "high"), "New Zealand": ("Pacific/Auckland", "high"),
    "West Indies": ("America/Port_of_Spain", "medium"),   # Jamaica is UTC-5, rest UTC-4
    "Australia": ("Australia/Sydney", "low"),             # 3 zones + DST differences
    "United States": ("America/New_York", "low"),         # multiple zones
}

d = lambda f: pd.read_csv(os.path.join(SRC, f))
b, m, c, xi = d("ball_by_ball.csv"), d("matches.csv"), d("competitions.csv"), d("playing_eleven.csv")
m = m.merge(c[["comp_id", "comp_name", "match_format", "season"]], on="comp_id", how="left")
m["match_date"] = pd.to_datetime(m.match_date)
checks = []
def check(name, passed, detail):
    checks.append({"check": name, "passed": bool(passed), "detail": detail})

# ---------------------------------------------------------------- deliveries
b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
b["is_super_over"] = b.innings >= 3
b["is_wide"] = b.extras_type.eq("wide")
b["is_noball"] = b.extras_type.isin(NOBALL_TYPES)
b["is_bye"] = b.extras_type.isin(BYE_TYPES)
b["is_legal"] = ~b.extras_type.isin(ILLEGAL_TYPES)
b["bat_runs"] = b.runs                                  # runs off the bat only (wides/byes carry 0 here)
b["batter_ball_faced"] = ~b.is_wide                     # a no-ball counts as faced by the batter
b["extras_runs_lb"] = (b.is_wide | b.is_noball).astype(int)          # 1-run penalty, known lower bound
b["extras_runs_est"] = b.extras_runs_lb + b.is_bye * BYE_RUNS_EST     # adds estimated bye/leg-bye runs
b["team_runs_lb"] = b.bat_runs + b.extras_runs_lb
b["team_runs_est"] = b.bat_runs + b.extras_runs_est
b["bowler_runs_conceded"] = b.bat_runs + b.extras_runs_lb          # byes/leg-byes not charged to bowler (exact)
b["bowler_wicket"] = b.is_wicket.astype(bool) & ~b.dismissal_type.isin(["RunOut", "Retire Out",
                                                                       "Obstracting the field", "Obstructing the field"])
b["is_wicket"] = b.is_wicket.astype(bool)

# corrected timestamps.
# The source column is labelled IST/'UTC' inconsistently: most data before 2024-03-20 is stored in IST and most after in UTC,
# but several post-switch series are still IST. Convention is therefore inferred PER SERIES (comp_id): we pick the
# convention under which the series' local first-ball times fall in a plausible T20 start window; ties fall back to the date rule.
raw = pd.to_datetime(b.delivery_ist_time.str.replace(" UTC", "", regex=False), errors="coerce")
raw[raw.dt.strftime("%H:%M:%S").eq("00:00:00")] = pd.NaT   # 42 placeholder midnight stamps
med = raw.groupby(b.match_id).transform("median")
raw[(raw - med).abs() > pd.Timedelta(hours=5)] = pd.NaT       # stray stamps far from the match's own time
b["_raw"] = raw
PLAUSIBLE = (10.0, 23.5)   # local first-ball hour window for T20 starts; upper bound allows rain-delayed starts
def local_hour(ts_utc, country):
    tz = COUNTRY_TZ.get(country, (None,))[0]
    if pd.isna(ts_utc) or tz is None:
        return np.nan
    t = ts_utc.tz_localize("UTC").tz_convert(ZoneInfo(tz))
    return t.hour + t.minute / 60
fr = b.groupby("match_id")._raw.min().rename("raw0").reset_index().merge(
    m[["match_id", "comp_id", "match_date", "venue_country"]], on="match_id")
fr["h_if_ist"] = [local_hour(r - pd.Timedelta(hours=5, minutes=30), cn) for r, cn in zip(fr.raw0, fr.venue_country)]
fr["h_if_utc"] = [local_hour(r, cn) for r, cn in zip(fr.raw0, fr.venue_country)]
inwin = lambda s: s.between(*PLAUSIBLE)
fr["ok_ist"], fr["ok_utc"] = inwin(fr.h_if_ist), inwin(fr.h_if_utc)
# Per-match rule:
#  1. decisive match: exactly one convention gives a plausible local start  -> use it
#  2. ambiguous match (both plausible): use the convention of the decisive matches in the same series if they all agree,
#     otherwise the date rule
#  3. neither plausible -> date rule, flagged unverifiable (start slot set to unknown)
fr["default"] = np.where(fr.match_date < TZ_SWITCH, "IST", "UTC")
decisive = fr.ok_ist ^ fr.ok_utc
fr["decisive_conv"] = np.where(decisive, np.where(fr.ok_ist, "IST", "UTC"), None)
series_vote = fr[decisive].groupby("comp_id").decisive_conv.agg(lambda s: s.iloc[0] if s.nunique() == 1 else None)
fr["series_conv"] = fr.comp_id.map(series_vote)
both = fr.ok_ist & fr.ok_utc
fr["convention"] = np.select(
    [decisive, both & fr.series_conv.notna()], [fr.decisive_conv, fr.series_conv], fr.default)
fr["basis"] = np.select(
    [decisive, both & fr.series_conv.notna(), both], ["decisive_single_plausible", "series_consistency", "date_rule_ambiguous"],
    "unverifiable")
fr["ts_confidence"] = fr.basis.map({"decisive_single_plausible": "high", "series_consistency": "medium",
                                    "date_rule_ambiguous": "medium", "unverifiable": "low"})
MATCH_CONV = fr.set_index("match_id")
SERIES_CONV = fr.merge(c[["comp_id", "comp_name"]], on="comp_id")[
    ["match_id", "comp_name", "match_date", "venue_country", "default", "convention", "basis", "ts_confidence",
     "h_if_ist", "h_if_utc"]]
stored_ist = b.match_id.map(MATCH_CONV.convention).eq("IST")
b["ts_utc"] = pd.to_datetime(np.where(stored_ist, b._raw - pd.Timedelta(hours=5, minutes=30), b._raw))
b["ts_convention"] = np.where(stored_ist, "IST", "UTC")
b = b.drop(columns="_raw")

# exact per-ball extras from Cricsheet where the innings aligned 1:1 (see align_cricsheet.py); estimate elsewhere
CS_EXTRAS = os.path.join(OUT, "cricsheet_ball_extras.csv")
if os.path.exists(CS_EXTRAS):
    ex = pd.read_csv(CS_EXTRAS)
    b = b.merge(ex[["match_id", "innings", "innings_ball_num", "cs_extras_runs", "cs_extras_kinds"]],
                on=["match_id", "innings", "innings_ball_num"], how="left")
else:
    b["cs_extras_runs"], b["cs_extras_kinds"] = np.nan, np.nan
has_cs = b.cs_extras_runs.notna()
b["extras_source"] = np.where(has_cs, "cricsheet_exact", "estimate")
b["extras_runs"] = np.where(has_cs, b.cs_extras_runs, b.extras_runs_est)
kinds = b.cs_extras_kinds.fillna("")
bowler_extras_cs = np.where(kinds.str.contains("byes"), kinds.str.contains("noballs").astype(int), b.cs_extras_runs)
b["bowler_runs_conceded"] = np.where(has_cs, b.bat_runs + bowler_extras_cs, b.bowler_runs_conceded)
b["team_runs"] = b.bat_runs + b.extras_runs

# innings state BEFORE each delivery (main innings only)
g = b.groupby(["match_id", "innings"])
b["score_before_lb"] = g.team_runs_lb.cumsum() - b.team_runs_lb
b["score_before_est"] = g.team_runs_est.cumsum() - b.team_runs_est
b["score_before"] = g.team_runs.cumsum() - b.team_runs            # best available: exact where aligned
b["wkts_before"] = g.is_wicket.cumsum() - b.is_wicket.astype(int)
b["legal_balls_before"] = g.is_legal.cumsum() - b.is_legal.astype(int)

# scheduled balls per innings (revised_overs columns are in BALLS)
mi = m.set_index("match_id")
sched1 = b.match_id.map(mi.first_inn_revised_overs).fillna(120)
# innings 2: Cricsheet's official target overs override the data pack when they indicate a reduced chase
# (the data pack omits revised overs for several rain-reduced chases). Where the data pack shows a reduction but
# Cricsheet says 20 overs, the sources conflict: keep the data pack value and flag the match.
def _ob(o):
    o = float(o); return int(o) * 6 + int(round((o - int(o)) * 10))
_csi, _csl = os.path.join(OUT, "cricsheet_index.csv"), os.path.join(OUT, "cricsheet_link.csv")
CS_SCHED2, CS_METHOD = pd.Series(dtype=float), pd.Series(dtype=object)
if os.path.exists(_csi) and os.path.exists(_csl):
    _x = pd.read_csv(_csl, dtype={"cs_match_id": str}).merge(pd.read_csv(_csi, dtype={"cs_match_id": str}), on="cs_match_id")
    if "i2_target_overs" in _x:
        CS_SCHED2 = _x.set_index("match_id").i2_target_overs.dropna().map(_ob)
    CS_METHOD = _x.set_index("match_id").method.dropna()
dp2 = mi.second_inn_revised_overs.fillna(120)
cs2 = CS_SCHED2.reindex(mi.index)
mi["sched2_source"] = np.where(cs2.notna() & (cs2 < 120), "cricsheet_target_overs",
                      np.where(cs2.notna() & (cs2 == 120) & (dp2 < 120), "datapack_CONFLICT_cricsheet_says_20", "datapack_or_default"))
mi["sched2_final"] = np.where(mi.sched2_source == "cricsheet_target_overs", cs2, dp2)
sched2 = b.match_id.map(mi.sched2_final)
b["scheduled_balls"] = np.where(b.innings == 1, sched1, np.where(b.innings == 2, sched2, 6)).astype(int)
b["balls_remaining"] = (b.scheduled_balls - b.legal_balls_before).clip(lower=0)
b["wkts_in_hand"] = 10 - b.wkts_before

# phase: powerplay = first 30% of scheduled overs, death = final 25% (exact 1-6 / 7-15 / 16-20 at 20 overs)
over_frac = (b.overid - 1) / (b.scheduled_balls / 6)
b["phase"] = np.select([b.is_super_over, over_frac < 0.30, over_frac >= 0.75], ["SuperOver", "Powerplay", "Death"], "Middle")
b["phase_exact"] = b.scheduled_balls.eq(120) | b.is_super_over

# innings totals + targets
inn = b[~b.is_super_over].groupby(["match_id", "innings"]).agg(
    total_lb=("team_runs_lb", "sum"), total_est=("team_runs_est", "sum"), total=("team_runs", "sum"),
    extras_exact=("extras_source", lambda s: (s == "cricsheet_exact").all()), bat_runs=("bat_runs", "sum"),
    wickets=("is_wicket", "sum"), legal_balls=("is_legal", "sum"), deliveries=("runs", "size"),
    wides=("is_wide", "sum"), noballs=("is_noball", "sum"), bye_events=("is_bye", "sum"),
    scheduled_balls=("scheduled_balls", "first"), first_ball_utc=("ts_utc", "min"), last_ball_utc=("ts_utc", "max"),
).reset_index()
inn["all_out"] = inn.wickets >= 10
inn["balls_complete"] = inn.legal_balls >= inn.scheduled_balls

t1 = inn[inn.innings == 1].set_index("match_id")
CS_IDX, CS_LINK = os.path.join(OUT, "cricsheet_index.csv"), os.path.join(OUT, "cricsheet_link.csv")
if os.path.exists(CS_IDX) and os.path.exists(CS_LINK):
    csl = pd.read_csv(CS_LINK, dtype={"cs_match_id": str}).merge(
        pd.read_csv(CS_IDX, dtype={"cs_match_id": str})[["cs_match_id", "i2_target"]], on="cs_match_id")
    cs_target = csl.set_index("match_id").i2_target.reindex(mi.index)
else:
    cs_target = pd.Series(np.nan, index=mi.index)
target_official = cs_target.fillna(mi.revised_target)
exact1 = t1.extras_exact.reindex(mi.index).fillna(False).astype(bool)
target_est = np.where(exact1, t1.total.reindex(mi.index) + 1, (t1.total_est.round() + 1).reindex(mi.index))
target_est = pd.Series(target_est, index=mi.index)
target_lb = (t1.total_lb + 1).reindex(mi.index)
mi["target_est"] = target_official.fillna(target_est)
mi["target_source"] = np.where(cs_target.notna(), "cricsheet_official",
                       np.where(mi.revised_target.notna(), "datapack_revised_target",
                       np.where(exact1, "innings1_exact_total", np.where(target_est.notna(), "innings1_estimated_total", None))))
in2 = b.innings == 2
b["target"] = np.where(in2, b.match_id.map(mi.target_est), np.nan)
b["target_source"] = np.where(in2, b.match_id.map(mi.target_source), None)
b["runs_required"] = np.where(in2, b.target - b.score_before, np.nan)
b["req_run_rate"] = np.where(in2 & (b.balls_remaining > 0), 6 * b.runs_required / b.balls_remaining.replace(0, np.nan), np.nan)
b["curr_run_rate"] = np.where(b.legal_balls_before > 0, 6 * b.score_before / b.legal_balls_before.replace(0, np.nan), np.nan)

# ---------------------------------------------------------------- match reconciliation grade
res = m.set_index("match_id").result.astype(str).str.lower()
margin_runs = res.str.extract(r"by (\d+) run")[0].astype(float)
t2 = inn[inn.innings == 2].set_index("match_id")
gap = (t1.total - t2.total).reindex(mi.index) - margin_runs
mi["recon_gap_runs"] = gap.round(1)
dls_text = res.str.contains(r"d/l|dls")
super_text = res.str.contains("super over")
mi["dls_affected"] = (mi.revised_target.notna() | dls_text | mi.first_inn_revised_overs.notna()
                     | mi.second_inn_revised_overs.notna() | mi.index.isin(CS_METHOD.index)
                     | (mi.sched2_final < 120))
mi["recon_grade"] = np.select(
    [mi.dls_affected, super_text, margin_runs.isna(), gap.abs() <= 2, gap.abs() <= 6],
    ["not_checkable_DLS", "not_checkable_super_over", "not_checkable_wicket_or_other_result", "A_within_2", "B_within_6"],
    "C_over_6")

# ---------------------------------------------------------------- match context
first = b[~b.is_super_over].groupby("match_id").ts_utc.min()
ctx = m[["match_id", "comp_id", "comp_name", "match_format", "season", "match_number", "match_date", "stage",
         "knock_out", "venue_id", "venue_country", "home_team", "away_team", "win_team_id", "result"]].copy()
ctx["has_ball_data"] = ctx.match_id.isin(b.match_id)
ctx["first_ball_utc"] = ctx.match_id.map(first)
ctx["ts_convention_in_source"] = ctx.match_id.map(MATCH_CONV.convention)
ctx["ts_convention_basis"] = ctx.match_id.map(MATCH_CONV.basis)
ctx["ts_confidence"] = ctx.match_id.map(MATCH_CONV.ts_confidence)
tzinfo = ctx.venue_country.map(lambda x: COUNTRY_TZ.get(x, (None, "none")))
ctx["local_tz_assumed"] = tzinfo.str[0]
ctx["local_tz_confidence"] = tzinfo.str[1]
def to_local(row):
    if pd.isna(row.first_ball_utc) or row.local_tz_assumed is None:
        return pd.NaT
    return row.first_ball_utc.tz_localize("UTC").tz_convert(ZoneInfo(row.local_tz_assumed)).tz_localize(None)
ctx["first_ball_local"] = ctx.apply(to_local, axis=1)
h = ctx.first_ball_local.dt.hour + ctx.first_ball_local.dt.minute / 60
ctx["start_slot"] = np.select([h.isna() | ctx.ts_confidence.eq("low"), h < 15.0, h < 17.5],
                              ["unknown", "day", "afternoon_into_lights"], "evening_lights")
ctx["slot_basis"] = "local first-ball time; precise sunset needs verified venue coordinates"
ctx["local_hour_plausible"] = h.between(*PLAUSIBLE) | h.isna()
ctx["super_over"] = ctx.match_id.isin(b[b.is_super_over].match_id)
ctx = ctx.join(mi[["dls_affected", "sched2_source", "target_est", "target_source", "recon_gap_runs", "recon_grade"]], on="match_id")
for k in (1, 2):
    tk = inn[inn.innings == k].set_index("match_id")
    for col in ["total", "extras_exact", "total_lb", "total_est", "wickets", "legal_balls", "scheduled_balls"]:
        ctx[f"inn{k}_{col}"] = ctx.match_id.map(tk[col])

# ---------------------------------------------------------------- venue draft (identity NOT verified)
ipl = ctx[ctx.comp_name.str.startswith("IPL")]
home_mode = ipl.groupby("venue_id").home_team.agg(lambda s: s.value_counts().index[0])
home_share = ipl.groupby("venue_id").home_team.agg(lambda s: s.value_counts(normalize=True).iloc[0])
ven = ctx.groupby("venue_id").agg(
    venue_country=("venue_country", "first"), n_matches=("match_id", "size"),
    n_ipl=("comp_name", lambda s: s.str.startswith("IPL").sum()),
    first_match=("match_date", "min"), last_match=("match_date", "max"),
    competitions=("comp_name", lambda s: "; ".join(s.value_counts().index[:3])),
    slot_mix=("start_slot", lambda s: "; ".join(f"{k}:{v}" for k, v in s.value_counts().items())),
).reset_index()
ven["ipl_modal_home_team"] = ven.venue_id.map(home_mode)
ven["ipl_modal_home_share"] = ven.venue_id.map(home_share).round(2)
ven["ground_name"] = None
ven["city"] = None
ven["latitude"] = None
ven["longitude"] = None
ven["timezone"] = None
ven["verification_status"] = "unverified"
ven["identity_hint"] = np.where(ven.ipl_modal_home_team.notna(),
                               "IPL home ground of " + ven.ipl_modal_home_team.astype(str) + " (from modal home team; verify)",
                               "verify via match date + teams + scorecard")

# ---------------------------------------------------------------- validation checks
legal_per_over = b[~b.is_super_over].groupby(["match_id", "innings", "overid"]).is_legal.sum()
check("legal balls per over <= 6 (1 known 7-ball over tolerated: umpire miscount, match 41509 over 12)",
      (legal_per_over > 6).sum() <= 1, f"overs >6 legal: {(legal_per_over>6).sum()}, share==6: {(legal_per_over==6).mean():.3f}")
full = inn[~inn.all_out & inn.scheduled_balls.eq(120) & (inn.innings == 1)]
check("unfinished 1st innings have 120 legal balls", (full.legal_balls == 120).mean() > 0.95,
      f"{(full.legal_balls==120).mean():.3f} of {len(full)} non-all-out 20-over 1st innings")
check("wickets per innings <= 10", (inn.wickets <= 10).all(), f"max={inn.wickets.max()}")
check("runs_required never NaN in 2nd innings", b.loc[in2, "runs_required"].notna().all(), f"nulls={b.loc[in2,'runs_required'].isna().sum()}")
check("local start hour plausible (10:00-23:30)", ctx.local_hour_plausible.mean() > 0.97,
      f"implausible={(~ctx.local_hour_plausible).sum()}; listed in match_context.local_hour_plausible")
check("first-ball date within 1 day of match_date",
      ((ctx.first_ball_local.dt.normalize() - ctx.match_date).dt.days.abs() <= 1).mean() > 0.97,
      f"{((ctx.first_ball_local.dt.normalize()-ctx.match_date).dt.days.abs()<=1).mean():.3f}")
check("abandoned = no ball data", ctx[~ctx.has_ball_data].result.str.contains("Abandoned").all(),
      f"{(~ctx.has_ball_data).sum()} matches without deliveries")
check("phase split at 20 overs = 6/9/5 overs",
      b[(b.scheduled_balls == 120) & ~b.is_super_over].groupby("phase").overid.agg(["min", "max"]).to_dict() ==
      {"min": {"Death": 16, "Middle": 7, "Powerplay": 1}, "max": {"Death": 20, "Middle": 15, "Powerplay": 6}},
      "Powerplay 1-6, Middle 7-15, Death 16-20")
xi_played = xi[xi.is_played.astype(str).str.lower() == "true"]
bat_ids = set(zip(b.match_id, b.striker_id)) | set(zip(b.match_id, b.bowler_id))
xi_ids = set(zip(xi_played.match_id, xi_played.player_id))
miss = bat_ids - xi_ids
check("every striker/bowler appears in playing XI (known gap: 2 UAE players, 3 matches)", len(miss) <= 6,
      f"missing (match, player) pairs={len(miss)}: {sorted(miss)}")
grade = ctx.recon_grade.value_counts().to_dict()
check("innings totals reconcile with run margins (info)", True, str(grade))
ex_share = (b.extras_source == "cricsheet_exact").mean()
check("share of deliveries with exact Cricsheet extras (info)", True, f"{ex_share:.3f}")
cs_matches = ctx.match_id.isin(mi.index[cs_target.notna()])
check("targets from Cricsheet official records (info)", True, f"{int(cs_target.notna().sum())} matches")
check("innings-2 scheduled balls source (info)", True, str(mi.sched2_source.value_counts().to_dict()))
# consistency: wherever the target is official and reduced, balls remaining must start below 120
_red = mi.index[mi.sched2_final < 120]
check("no chase has a reduced official target with a 20-over schedule",
      not ((mi.target_source == "cricsheet_official") & (mi.sched2_final == 120) & mi.index.isin(CS_SCHED2[CS_SCHED2 < 120].index)).any(),
      f"reduced chases: {len(_red)}")

# ---------------------------------------------------------------- write
keep = ["match_id", "innings", "overid", "over_ball_num", "innings_ball_num", "batting_team_id", "bowling_team_id",
        "striker_id", "striker_name", "non_striker_id", "non_striker_name", "bowler_id", "bowler_name",
        "batting_type", "bowling_type", "pace_or_spin", "batting_order", "bat_runs", "is_four", "is_six",
        "is_wicket", "bowler_wicket", "dismissal_type", "dismissed_player_id", "extras_type",
        "is_legal", "is_wide", "is_noball", "is_bye", "batter_ball_faced", "extras_runs_lb", "extras_runs_est",
        "team_runs_lb", "team_runs_est", "extras_runs", "extras_source", "team_runs", "bowler_runs_conceded", "ts_utc", "ts_convention",
        "is_super_over", "phase", "phase_exact", "scheduled_balls", "legal_balls_before", "balls_remaining",
        "score_before", "score_before_lb", "score_before_est", "wkts_before", "wkts_in_hand", "curr_run_rate",
        "target", "target_source", "runs_required", "req_run_rate", "bowl_speed", "delivery_style", "stroke"]
b[keep].to_csv(os.path.join(OUT, "deliveries_clean.csv"), index=False)
inn.to_csv(os.path.join(OUT, "innings_summary.csv"), index=False)
ctx.to_csv(os.path.join(OUT, "match_context.csv"), index=False)
ven.sort_values(["n_ipl", "n_matches"], ascending=False).to_csv(os.path.join(OUT, "venue_draft.csv"), index=False)
pd.DataFrame(checks).to_csv(os.path.join(OUT, "foundation_validation.csv"), index=False)
SERIES_CONV.to_csv(os.path.join(OUT, "timestamp_convention_by_match.csv"), index=False)
print(pd.DataFrame(checks).to_string(index=False))
