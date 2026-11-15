"""Phase 0 - training-only history: parse PRE-2022 men's IPL and T20I Cricsheet JSON into the deliveries_clean schema.

Outputs (out/model/):
  history_deliveries.csv    one row per delivery, split = "history_training_only"
  history_matches.csv       one row per match (teams, winner, result, method, format tier, dls_affected)
  phase0_state_check.csv    per-field agreement of this parser vs out/deliveries_clean.csv on 2022+ linked matches
  phase0_rowcounts.csv      rows by season x source
Appends phase-0 fields to out/data_dictionary.csv (existing rows untouched; idempotent).

State fields - same formulas as build_foundation.py:
  is_legal           = not (wide or no-ball)
  legal_balls_before = cumulative legal balls in the innings before this delivery
  score_before       = cumulative team runs (runs.total, incl. extras and penalty runs) before this delivery
  wkts_before        = cumulative balls with >= 1 wicket before this delivery; "retired hurt" / "retired not out"
                       are NOT wickets (data-pack convention), "retired out" is
  scheduled_balls    = 6 * overs: innings 2 uses innings[1].target.overs when present, else info.overs;
                       innings 1 uses 6 x info.overs (Cricsheet does not record a reduced first innings, and the
                       data pack's own first_inn_revised_overs is inconsistent, so it cannot be reproduced); super over = 6.
                       Files whose info.overs != 20 but whose overs never exceed 20 are treated as 20 overs (flagged).
  balls_remaining    = max(scheduled_balls - legal_balls_before, 0);  wkts_in_hand = 10 - wkts_before
  phase              = Powerplay if (overid-1)/(scheduled_balls/6) < 0.30, Death if >= 0.75, else Middle; SuperOver
  target             = innings[1].target.runs (official, incl. DLS), else innings-1 total + 1
  runs_required      = target - score_before;  req_run_rate = 6 * runs_required / balls_remaining
  curr_run_rate      = 6 * score_before / legal_balls_before
Format tier: IPL; T20I "full_member" if both teams were ICC full members on the match date
(Ireland and Afghanistan from 2017-06-22), else "associate" (is_associate = True).
pace_or_spin / batting_type / bowling_type are not in Cricsheet: taken from the data pack for players linked through
ball-by-ball aligned 2022+ innings (status inferred); otherwise null with missing_reason.
"""
import glob, json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))           # superr/
CS_ROOT = os.environ.get("CRICSHEET_DIR", os.path.join(os.path.dirname(ROOT), "data", "cricsheet"))
OUT, MODEL = os.path.join(ROOT, "out"), os.path.join(ROOT, "out", "model")
HISTORY_END = "2022-01-01"            # history = matches whose first day is before this date
SPLIT = "history_training_only"
FULL_MEMBERS = {"Australia", "Bangladesh", "England", "India", "New Zealand", "Pakistan", "South Africa",
                "Sri Lanka", "West Indies", "Zimbabwe"}
FULL_FROM_2017 = {"Ireland", "Afghanistan"}                                    # full members from 2017-06-22
NOT_WICKET = {"retired hurt", "retired not out"}
NOT_BOWLER = {"run out", "retired hurt", "retired not out", "retired out", "obstructing the field",
              "hit the ball twice", "handled the ball", "timed out"}


def overs_to_balls(o):
    """Cricsheet overs (e.g. 20, 5.4) -> balls."""
    o = float(o); whole = int(o)
    return whole * 6 + int(round((o - whole) * 10))


def is_full_member(team, date):
    return team in FULL_MEMBERS or (team in FULL_FROM_2017 and date >= "2017-06-22")


def parse_match(path, source):
    """Return (match_row, list of delivery rows) for one Cricsheet JSON file, or (None, []) if out of scope."""
    j = json.load(open(path, encoding="utf-8")); i = j["info"]
    if i.get("gender") != "male" or i.get("match_type") != "T20":
        return None, []
    max_over = max((ov["over"] for inn in j.get("innings", []) for ov in inn.get("overs", [])), default=0)
    overs_flag, reduced_info = None, False
    if i.get("overs") is None or i["overs"] > 20:            # mislabelled (e.g. 50-over tag on a T20I)
        if max_over > 19:
            return None, []
        overs_flag = f"info.overs={i.get('overs')} but max over index {max_over}: treated as 20 overs"
        i["overs"] = 20
    elif i["overs"] < 20:                                    # genuinely reduced match: keep, mark as reduced
        overs_flag, reduced_info = f"info.overs={i['overs']} < 20: kept as scheduled, dls_affected", True
    cs_id = int(os.path.basename(path)[:-5]); date = i["dates"][0]
    reg = (i.get("registry") or {}).get("people", {})
    out = i.get("outcome") or {}
    teams = i["teams"]
    tier = "IPL" if source == "ipl" else (
        "full_member" if all(is_full_member(t, date) for t in teams) else "associate")
    inns = j.get("innings", [])
    tgt = next((inn["target"] for inn in inns if inn.get("target") and not inn.get("super_over")), None)
    match = {"match_id": cs_id, "cs_match_id": cs_id, "source": source, "match_date": date, "season": int(date[:4]),
             "cs_season": str(i.get("season")), "event_name": (i.get("event") or {}).get("name"),
             "stage": (i.get("event") or {}).get("stage"), "team1": teams[0], "team2": teams[1],
             "venue": i.get("venue"), "city": i.get("city"), "format_tier": tier, "is_associate": tier == "associate",
             "winner": out.get("winner"), "result": out.get("result"), "eliminator": out.get("eliminator"),
             "method": out.get("method"), "by_runs": (out.get("by") or {}).get("runs"),
             "by_wickets": (out.get("by") or {}).get("wickets"),
             "target_runs": tgt["runs"] if tgt else None, "target_overs": tgt.get("overs") if tgt else None,
             "dls_affected": reduced_info or bool(out.get("method")) or bool(tgt and tgt.get("overs") not in (None, i["overs"])),
             "info_overs_flag": overs_flag, "split": SPLIT}
    # innings-1 scheduled balls: Cricsheet does not record a reduced first innings -> 6 x info.overs (no hindsight).
    # Matches with any reduction are dls_affected and excluded from model training.
    sched1, sched1_src = 6 * int(i["overs"]), "cricsheet_info_overs"
    rows, k_main, k_super = [], 0, 2
    for inn in inns:
        sup = bool(inn.get("super_over"))
        if sup: k_super += 1; k = k_super
        else:
            k_main += 1; k = k_main
            if k > 2: continue
        bat = inn["team"]; bowl = teams[1] if bat == teams[0] else teams[0]
        if sup: sched = 6
        elif k == 2 and inn.get("target", {}).get("overs") is not None: sched = overs_to_balls(inn["target"]["overs"])
        elif k == 1: sched = sched1
        else: sched = 6 * int(i["overs"])
        sched_src = "super_over" if sup else ("cricsheet_target_overs" if (k == 2 and inn.get("target", {}).get("overs") is not None)
                                             else sched1_src if k == 1 else "cricsheet_info_overs")
        ibn = 0
        for ov in inn.get("overs", []):
            for obn, x in enumerate(ov["deliveries"], start=1):
                ibn += 1
                ex = x.get("extras", {}); w = [wk for wk in x.get("wickets", []) if wk["kind"] not in NOT_WICKET]
                w0 = w[0] if w else (x.get("wickets") or [{}])[0]
                kind = w[0]["kind"] if w else None
                is_wide, is_nb = "wides" in ex, "noballs" in ex
                rows.append({
                    "match_id": cs_id, "innings": k, "overid": ov["over"] + 1, "over_ball_num": obn,
                    "innings_ball_num": ibn, "batting_team": bat, "bowling_team": bowl,
                    "striker_cs_id": reg.get(x["batter"]), "striker_name": x["batter"],
                    "non_striker_cs_id": reg.get(x["non_striker"]), "non_striker_name": x["non_striker"],
                    "bowler_cs_id": reg.get(x["bowler"]), "bowler_name": x["bowler"],
                    "bat_runs": x["runs"]["batter"],
                    "is_four": x["runs"]["batter"] == 4 and not x["runs"].get("non_boundary"),
                    "is_six": x["runs"]["batter"] == 6 and not x["runs"].get("non_boundary"),
                    "is_wicket": bool(w), "bowler_wicket": bool(w) and kind not in NOT_BOWLER,
                    "dismissal_type": kind, "dismissed_cs_id": reg.get(w[0]["player_out"]) if w else None,
                    "dismissed_name": w[0]["player_out"] if w else None,
                    "n_wickets_on_ball": len(w), "retired_not_wicket": bool(x.get("wickets")) and not w,
                    "extras_type": ",".join(sorted(ex)) or None,
                    "is_legal": not (is_wide or is_nb), "is_wide": is_wide, "is_noball": is_nb,
                    "is_bye": "byes" in ex or "legbyes" in ex, "batter_ball_faced": not is_wide,
                    "extras_runs": x["runs"]["extras"], "extras_source": "cricsheet_exact",
                    "team_runs": x["runs"]["total"],
                    "bowler_runs_conceded": x["runs"]["batter"] + ex.get("wides", 0) + ex.get("noballs", 0),
                    "is_super_over": sup, "scheduled_balls": sched, "scheduled_balls_source": sched_src,
                    "target_official": inn["target"]["runs"] if (k == 2 and inn.get("target")) else None})
    return match, rows


def add_state(b, match_target):
    """Vectorised state fields (same formulas as build_foundation.py). match_target: Series match_id -> official target."""
    b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    g = b.groupby(["match_id", "innings"])
    b["score_before"] = g.team_runs.cumsum() - b.team_runs
    b["wkts_before"] = g.is_wicket.cumsum() - b.is_wicket.astype(int)
    b["legal_balls_before"] = g.is_legal.cumsum() - b.is_legal.astype(int)
    b["balls_remaining"] = (b.scheduled_balls - b.legal_balls_before).clip(lower=0)
    b["wkts_in_hand"] = 10 - b.wkts_before
    frac = (b.overid - 1) / (b.scheduled_balls / 6)
    b["phase"] = np.select([b.is_super_over, frac < 0.30, frac >= 0.75], ["SuperOver", "Powerplay", "Death"], "Middle")
    b["phase_exact"] = b.scheduled_balls.eq(120) | b.is_super_over
    inn1 = b[b.innings == 1].groupby("match_id").team_runs.sum()
    tgt = match_target.reindex(inn1.index.union(match_target.index))
    in2 = b.innings == 2
    off = b.match_id.map(match_target)
    b["target"] = np.where(in2, off.fillna(b.match_id.map(inn1 + 1)), np.nan)
    b["target_source"] = np.where(in2, np.where(off.notna(), "cricsheet_official", "innings1_exact_total"), None)
    b["runs_required"] = np.where(in2, b.target - b.score_before, np.nan)
    br = b.balls_remaining.replace(0, np.nan)
    b["req_run_rate"] = np.where(in2 & (b.balls_remaining > 0), 6 * b.runs_required / br, np.nan)
    b["curr_run_rate"] = np.where(b.legal_balls_before > 0, 6 * b.score_before / b.legal_balls_before.replace(0, np.nan), np.nan)
    return b


def parse_files(files):
    ms, rs = [], []
    for path, src in files:
        m, r = parse_match(path, src)
        if m: ms.append(m); rs.extend(r)
    return pd.DataFrame(ms), pd.DataFrame(rs)


def all_files():
    return [(f, src) for src in ("ipl", "t20s") for f in sorted(glob.glob(os.path.join(CS_ROOT, src, "*.json")))]


def player_type_map(link_balls, ours):
    """Map Cricsheet person id -> data-pack player id + types, from ball-by-ball aligned 2022+ innings (majority vote)."""
    j = link_balls.merge(ours, on=["dp_match_id", "innings", "innings_ball_num"])
    maps = {}
    for cs_col, dp_col, cols in [("bowler_cs_id", "bowler_id", ["pace_or_spin", "bowling_type"]),
                                 ("striker_cs_id", "striker_id", ["batting_type"])]:
        v = j.groupby([cs_col, dp_col]).size().rename("n").reset_index()
        tot = v.groupby(cs_col).n.transform("sum")
        v = v[v.n / tot >= 0.95].drop_duplicates(cs_col)                       # unambiguous links only
        attrs = j.groupby(dp_col)[cols].agg(lambda s: s.mode().iloc[0] if s.notna().any() else None)
        maps[cs_col] = v.merge(attrs, left_on=dp_col, right_index=True).set_index(cs_col)
    return maps


def main():
    os.makedirs(MODEL, exist_ok=True)
    files = all_files()
    # ---- history (pre-2022)
    idx = []
    for f, src in files:
        with open(f, encoding="utf-8") as fh:
            d0 = json.load(fh)["info"]["dates"][0]
        idx.append((f, src, d0))
    hist_files = [(f, s) for f, s, d0 in idx if d0 < HISTORY_END]
    hm, hb = parse_files(hist_files)
    hb = hb.merge(hm[["match_id", "source", "match_date", "season", "format_tier", "is_associate", "dls_affected", "split"]],
                  on="match_id")
    hb = add_state(hb, hm.set_index("match_id").target_runs.dropna())

    # ---- 2022+ linked matches: parse with the SAME code, compare state fields ball by ball on aligned innings
    link = pd.read_csv(os.path.join(OUT, "cricsheet_link.csv"))
    st = pd.read_csv(os.path.join(OUT, "cricsheet_alignment_status.csv"))
    fmap = {int(os.path.basename(f)[:-5]): (f, s) for f, s in files}
    lm, lb = parse_files([fmap[c] for c in link.cs_match_id if c in fmap])
    lb = add_state(lb, lm.set_index("match_id").target_runs.dropna())
    lb = lb.merge(link[["match_id", "cs_match_id"]].rename(columns={"match_id": "dp_match_id", "cs_match_id": "match_id"}),
                  on="match_id")
    lb = lb[~lb.is_super_over]
    cols = ["is_legal", "is_wide", "is_noball", "bat_runs", "team_runs", "is_wicket", "legal_balls_before",
            "balls_remaining", "scheduled_balls", "score_before", "wkts_before", "wkts_in_hand", "target",
            "runs_required", "req_run_rate", "phase"]
    ours = pd.read_csv(os.path.join(OUT, "deliveries_clean.csv"), low_memory=False,
                       usecols=["match_id", "innings", "innings_ball_num", "is_super_over", "striker_id", "bowler_id",
                                "pace_or_spin", "batting_type", "bowling_type"] + cols)
    ours = ours[~ours.is_super_over].rename(columns={"match_id": "dp_match_id"})
    aligned = st[st.aligned][["match_id", "innings"]].rename(columns={"match_id": "dp_match_id"})
    lb_al = lb.merge(aligned, on=["dp_match_id", "innings"])
    cmp = lb_al.merge(ours, on=["dp_match_id", "innings", "innings_ball_num"], suffixes=("_cs", "_dp"))
    ok = pd.DataFrame(index=cmp.index)
    for c in cols:
        a, b_ = cmp[c + "_cs"], cmp[c + "_dp"]
        if c == "phase": ok[c] = a.astype(str) == b_.astype(str)
        else: ok[c] = np.isclose(a.astype(float), b_.astype(float), equal_nan=True)
    state_cols = ["is_legal", "legal_balls_before", "balls_remaining", "score_before", "wkts_before", "wkts_in_hand",
                  "target", "runs_required", "phase"]
    ok["ALL_STATE_FIELDS"] = ok[state_cols].all(axis=1)
    mctx = pd.read_csv(os.path.join(OUT, "match_context.csv"))
    dls = cmp.dp_match_id.isin(mctx.query("dls_affected").match_id)
    ok["ALL_STATE_FIELDS_nonDLS_foundation"] = ok.ALL_STATE_FIELDS.where(~dls)
    # training population: neither source marks the match reduced / rain-affected, and a result was reached
    reduced_cs = cmp.match_id.isin(lm[lm.dls_affected | lm.winner.isna()].match_id)
    no_result_dp = cmp.dp_match_id.isin(mctx[mctx.result.astype(str).str.contains("No Result|Abandon", case=False)].match_id)
    ok["ALL_STATE_FIELDS_unreduced_both_sources"] = ok.ALL_STATE_FIELDS.where(~(dls | reduced_cs | no_result_dp))
    check = pd.DataFrame({"field": ok.columns, "n_balls": [ok[c].notna().sum() for c in ok.columns],
                          "match_rate": [ok[c].dropna().astype(bool).mean() for c in ok.columns]})
    check["n_mismatch"] = [int((~ok[c].dropna().astype(bool)).sum()) for c in ok.columns]
    meta = {"linked_matches": len(link), "parsed_linked_matches": lm.shape[0],
            "aligned_innings": len(aligned), "innings_compared": cmp.groupby(["dp_match_id", "innings"]).ngroups,
            "balls_compared": len(cmp)}
    check.to_csv(os.path.join(MODEL, "phase0_state_check.csv"), index=False)
    bad = cmp.loc[~ok.ALL_STATE_FIELDS, ["dp_match_id", "match_id", "innings", "innings_ball_num"]].assign(
        fields=ok.loc[~ok.ALL_STATE_FIELDS, state_cols].apply(lambda r: ",".join(c for c in state_cols if not r[c]), axis=1))
    # cause of each mismatching innings
    sch = cmp.groupby(["dp_match_id", "innings"]).agg(s_cs=("scheduled_balls_cs", "first"), s_dp=("scheduled_balls_dp", "first"))
    def cause(r):
        f = set(r.fields.split(","))
        s = sch.loc[(r.dp_match_id, r.innings)]
        if "is_legal" in f or "legal_balls_before" in f: return "wide recorded on an adjacent ball (ordering)"
        if f & {"wkts_before", "wkts_in_hand"}: return "wicket recorded in Cricsheet but not in data pack"
        if r.innings == 2 and s.s_cs < s.s_dp: return "data pack lacks revised overs; Cricsheet official target overs"
        if r.innings == 1 and s.s_cs != s.s_dp: return "reduced first innings not recorded by Cricsheet"
        if r.innings == 2 and s.s_cs != s.s_dp: return "Cricsheet target overs != data-pack revised overs"
        return "other"
    bad["cause"] = bad.apply(cause, axis=1) if len(bad) else []
    bad.to_csv(os.path.join(MODEL, "phase0_state_mismatches.csv"), index=False)
    causes = (bad.groupby("cause").agg(balls=("cause", "size"), innings=("dp_match_id", lambda s: 0))
              .drop(columns="innings") if len(bad) else pd.DataFrame())
    if len(bad):
        causes["innings"] = bad.drop_duplicates(["dp_match_id", "innings"]).groupby("cause").size()
        causes["dp_match_ids"] = bad.groupby("cause").dp_match_id.agg(lambda s: " ".join(map(str, sorted(set(s)))))
        causes.reset_index().to_csv(os.path.join(MODEL, "phase0_mismatch_causes.csv"), index=False)
    meta["mismatch_causes"] = causes

    # ---- player types for history via linked people (inferred)
    maps = player_type_map(lb_al[["dp_match_id", "innings", "innings_ball_num", "bowler_cs_id", "striker_cs_id"]], ours)
    bm, sm = maps["bowler_cs_id"], maps["striker_cs_id"]
    hb["bowler_id_datapack"] = hb.bowler_cs_id.map(bm.bowler_id)
    hb["striker_id_datapack"] = hb.striker_cs_id.map(sm.striker_id)
    hb["pace_or_spin"] = hb.bowler_cs_id.map(bm.pace_or_spin)
    hb["bowling_type"] = hb.bowler_cs_id.map(bm.bowling_type)
    hb["batting_type"] = hb.striker_cs_id.map(sm.batting_type)
    hb["pace_or_spin_source"] = np.where(hb.pace_or_spin.notna(), "datapack_via_cricsheet_registry_link", None)
    hb["pace_or_spin_missing_reason"] = np.where(hb.pace_or_spin.isna(), "bowler not linked to a 2022+ data-pack player", None)
    hb["batting_type_missing_reason"] = np.where(hb.batting_type.isna(), "batter not linked to a 2022+ data-pack player", None)

    # ---- result label (exact): did the batting team win? ties = 0.5 for both sides (super over ignored)
    win = hb.match_id.map(hm.set_index("match_id").winner)
    res = hb.match_id.map(hm.set_index("match_id").result)
    tie = res.eq("tie")
    hb["batting_team_won"] = np.where(tie, 0.5, np.where(win.notna(), (win == hb.batting_team).astype(float), np.nan))
    hb.loc[hb.is_super_over, "batting_team_won"] = np.nan
    hb["result_missing_reason"] = np.where(hb.is_super_over, "super-over delivery (excluded)",
                                  np.where(hb.batting_team_won.isna(), "no winner recorded (" + res.fillna("unknown") + ")", None))

    order = ["match_id", "cs_match_id", "split", "source", "match_date", "season", "format_tier", "is_associate",
             "dls_affected", "innings", "overid", "over_ball_num", "innings_ball_num", "batting_team", "bowling_team",
             "striker_cs_id", "striker_name", "striker_id_datapack", "non_striker_cs_id", "non_striker_name",
             "bowler_cs_id", "bowler_name", "bowler_id_datapack", "batting_type", "bowling_type", "pace_or_spin",
             "pace_or_spin_source", "pace_or_spin_missing_reason", "batting_type_missing_reason",
             "bat_runs", "is_four", "is_six", "is_wicket", "bowler_wicket", "dismissal_type", "dismissed_cs_id",
             "dismissed_name", "n_wickets_on_ball", "retired_not_wicket", "extras_type", "is_legal", "is_wide",
             "is_noball", "is_bye", "batter_ball_faced", "extras_runs", "extras_source", "team_runs",
             "bowler_runs_conceded", "is_super_over", "phase", "phase_exact", "scheduled_balls",
             "scheduled_balls_source", "legal_balls_before", "balls_remaining", "score_before", "wkts_before",
             "wkts_in_hand", "curr_run_rate", "target", "target_source", "runs_required", "req_run_rate",
             "batting_team_won", "result_missing_reason"]
    hb["cs_match_id"] = hb.match_id
    hb[order].to_parquet(os.path.join(MODEL, "history_deliveries.parquet"), index=False)
    old_csv = os.path.join(MODEL, "history_deliveries.csv")       # superseded by the Parquet file
    if os.path.exists(old_csv): os.remove(old_csv)
    hm.to_csv(os.path.join(MODEL, "history_matches.csv"), index=False)
    meta["ties"] = {"tied_matches": int(hm.result.eq("tie").sum()),
                    "tied_with_super_over": int((hm.result.eq("tie") & hm.eliminator.notna()).sum()),
                    "main_innings_balls_labelled_0.5": int((hb.batting_team_won == 0.5).sum())}

    main_b = hb[~hb.is_super_over]
    rc = (main_b.groupby(["season", "source", "format_tier"])
          .agg(matches=("match_id", "nunique"), deliveries=("match_id", "size"), legal_balls=("is_legal", "sum"))
          .reset_index())
    rc.to_csv(os.path.join(MODEL, "phase0_rowcounts.csv"), index=False)
    write_report(check, meta)
    write_dictionary()
    return hm, hb, rc, check, meta, bad


DICT_ROWS = [
    ("model/history_deliveries", "split", "category", "history_training_only: pre-2022 Cricsheet matches, used ONLY to train WP / outcome models; never ranked", "exact", "model training only"),
    ("model/history_deliveries", "match_id / cs_match_id", "id", "Cricsheet match id (>= 211028; no overlap with data-pack ids 13511-55561)", "exact", "join key"),
    ("model/history_deliveries", "format_tier / is_associate", "category", "IPL; T20I full_member if both teams ICC full members on the date (Ireland, Afghanistan from 2017-06-22), else associate", "inferred", "model feature"),
    ("model/history_deliveries", "is_legal / is_wide / is_noball", "bool", "Legal unless the Cricsheet delivery has a wides or noballs extra (same rule as deliveries_clean)", "exact", "match state"),
    ("model/history_deliveries", "is_wicket / wkts_before / wkts_in_hand", "wickets", "Ball has >=1 wicket other than retired hurt / retired not out (data-pack convention); cumulative before the ball", "exact", "match state"),
    ("model/history_deliveries", "score_before / team_runs", "runs", "Cricsheet runs.total incl. extras and penalty runs; cumulative before the ball", "exact", "match state"),
    ("model/history_deliveries", "scheduled_balls / scheduled_balls_source", "balls", "Innings 2: 6 x innings[1].target.overs if present, else 120. Innings 1: 6 x info.overs (a reduced first innings is not recorded by Cricsheet, so it is 120 even when rain-reduced). Files with info.overs != 20 but <= 20 overs bowled are treated as 20 (info_overs_flag)", "exact except rain-reduced innings 1", "match state; exclude dls_affected when training"),
    ("model/history_deliveries", "legal_balls_before / balls_remaining", "balls", "Cumulative legal balls before the ball; max(scheduled_balls - legal_balls_before, 0)", "exact", "match state"),
    ("model/history_deliveries", "phase", "category", "Powerplay if (overid-1)/(scheduled_balls/6) < 0.30, Death if >= 0.75, else Middle (same as deliveries_clean)", "exact at 20 overs", "phase splits"),
    ("model/history_deliveries", "target / runs_required / req_run_rate", "runs", "Innings 2: official innings[1].target.runs, else innings-1 total + 1; runs_required = target - score_before", "exact", "chase state"),
    ("model/history_deliveries", "dls_affected", "bool", "outcome.method present (D/L, VJD, ...) or target overs != 20", "exact", "exclude from chase-model training"),
    ("model/history_deliveries", "pace_or_spin / bowling_type / batting_type", "category", "Not in Cricsheet. Copied from the data pack for players linked by ball-by-ball aligned 2022+ innings (>=95% consistent link); else null with missing_reason", "inferred", "NOT a Phase 1 (outcome/WP) feature; pace/spin is a Phase 3 candidate trained on 2022+ only"),
    ("model/history_deliveries", "batting_team_won", "0/0.5/1", "1 if the batting team won (Cricsheet outcome.winner); 0.5 for both sides of a tie (outcome.result == 'tie', super-over result ignored); null for no result and for super-over deliveries", "exact", "WP training label"),
    ("model/history_deliveries", "info_overs_flag", "text", "info.overs > 20 with <= 20 overs bowled: treated as 20 (mislabelled); info.overs < 20: kept as scheduled and dls_affected=True", "exact", "audit"),
    ("match_context", "sched2_source", "category", "innings-2 scheduled balls: cricsheet_target_overs when Cricsheet shows a reduced chase, else data pack; datapack_CONFLICT_cricsheet_says_20 for match 17765", "exact", "match state provenance"),
    ("model/history_deliveries", "bowler_runs_conceded", "runs", "bat runs + wides + no-ball runs (byes/leg-byes excluded)", "exact", "bowling economy"),
]


ACCEPTED = {"date": "2026-11-13", "foundation": "pre-patch (superr_foundation.zip)", "overall": 0.98767,
            "unreduced_both_sources": 0.99981, "balls": 179906}


def write_report(check, meta):
    """reports/phase0/: state-check tables + summary with the accepted (pre-patch) and current numbers."""
    rep = os.path.join(ROOT, "reports", "phase0"); os.makedirs(rep, exist_ok=True)
    check.to_csv(os.path.join(rep, "state_check.csv"), index=False)
    causes = meta.get("mismatch_causes")
    if causes is not None and len(causes): causes.reset_index().to_csv(os.path.join(rep, "mismatch_causes.csv"), index=False)
    r = check.set_index("field").match_rate
    rc = pd.read_csv(os.path.join(MODEL, "phase0_rowcounts.csv"))
    L = ["# Phase 0 - training-only history: acceptance record", "",
         "## Accepted result", f"Accepted on {ACCEPTED['date']} against the {ACCEPTED['foundation']} foundation "
         f"({ACCEPTED['balls']:,} balls in ball-by-ball aligned innings of 2022+ linked matches):", "",
         f"- all state fields, all balls: **{ACCEPTED['overall']:.2%}**",
         f"- all state fields, matches neither source marks reduced / rain-affected / no result: **{ACCEPTED['unreduced_both_sources']:.2%}**", "",
         "Residual mismatches were source disagreements (unrecorded reduced first innings, missing revised overs in the data pack, "
         "2 wickets absent from the data pack, 2 wide-order swaps), not parser errors.", "",
         "## Current result (after the DLS-fix foundation patch)", f"Balls compared: {meta['balls_compared']:,}; innings: {meta['innings_compared']}", "",
         f"- all state fields, all balls: **{r['ALL_STATE_FIELDS']:.2%}**",
         f"- excluding foundation dls_affected matches: **{r['ALL_STATE_FIELDS_nonDLS_foundation']:.2%}**",
         f"- unreduced in both sources: **{r['ALL_STATE_FIELDS_unreduced_both_sources']:.2%}**", "",
         check.to_markdown(index=False), "",
         "Mismatch causes:", "", causes.reset_index().to_markdown(index=False) if causes is not None and len(causes) else "_none_", "",
         "## History rows", f"{rc.matches.sum():,} matches, {rc.deliveries.sum():,} main-innings deliveries "
         "(see out/model/phase0_rowcounts.csv). Ties: " + str(meta.get("ties")), ""]
    open(os.path.join(rep, "phase0_summary.md"), "w", encoding="utf-8").write("\n".join(L))


def write_dictionary():
    """Append phase-0 rows to out/data_dictionary.csv; existing rows untouched; re-runs replace only phase-0 rows."""
    p = os.path.join(OUT, "data_dictionary.csv")
    d = pd.read_csv(p)
    d = d[~d.table.astype(str).str.startswith("model/history_") &
          ~((d.table == "match_context") & (d.field == "sched2_source"))]
    new = pd.DataFrame(DICT_ROWS, columns=["table", "field", "unit", "definition", "status", "allowed_use"])
    pd.concat([d, new], ignore_index=True).to_csv(p, index=False)


if __name__ == "__main__":
    hm, hb, rc, check, meta, bad = main()
    pd.set_option("display.width", 200)
    print("PHASE 0 rows by season (main innings):"); print(rc.to_string(index=False))
    print("\nState-field check vs deliveries_clean (2022+ linked, aligned innings):", meta)
    print(check.to_string(index=False))
