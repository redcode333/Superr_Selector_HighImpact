"""Link SupeRR matches to Cricsheet matches.
Candidates: same team pair, date within +-1 day. Scored by date lag and by agreement of runs-off-the-bat per innings
(recorded independently by both sources). Each Cricsheet match used at most once.
Usage: python3 match_cricsheet.py <data_pack> <out_dir>"""
import sys, os, re
import pandas as pd, numpy as np
pack, out = sys.argv[1], sys.argv[2]
m = pd.read_csv(os.path.join(pack, "matches.csv")).merge(pd.read_csv(os.path.join(pack, "competitions.csv"))[["comp_id", "comp_name"]], on="comp_id")
cs = pd.read_csv(os.path.join(out, "cricsheet_index.csv"))
inn = pd.read_csv(os.path.join(out, "innings_summary.csv"))
b = pd.read_csv(os.path.join(pack, "ball_by_ball.csv"), usecols=["match_id", "innings", "batting_team_id"]).drop_duplicates(["match_id", "innings"])
tn = pd.read_csv(os.path.join(pack, "teams.csv")).drop_duplicates("team_id").set_index("team_id").team_name
ALIAS = {"royal challengers bangalore": "royal challengers bengaluru", "uae": "united arab emirates",
         "usa": "united states of america", "united states": "united states of america"}
norm = lambda t: ALIAS.get(re.sub(r"\s+", " ", str(t).strip().lower()), re.sub(r"\s+", " ", str(t).strip().lower()))
ours = inn[inn.innings <= 2].merge(b, on=["match_id", "innings"])
ours["team"] = ours.batting_team_id.map(tn).map(norm)
ob = ours.pivot(index="match_id", columns="innings", values=["bat_runs", "team"])
ob.columns = [f"{a}{k}" for a, k in ob.columns]
for k in (1, 2):
    extras = sum(cs[f"i{k}_{x}"].fillna(0) for x in ["wide_runs", "nb_runs", "bye_runs", "legbye_runs", "pen_runs"])
    cs[f"bat{k}"] = cs[f"i{k}_runs"] - extras
    cs[f"t{k}"] = cs[f"i{k}_team"].map(norm)
m["key_teams"] = [frozenset({norm(a), norm(c)}) for a, c in zip(m.home_team, m.away_team)]
cs["key_teams"] = [frozenset({norm(a), norm(c)}) for a, c in zip(cs.team1, cs.team2)]
m["d"] = pd.to_datetime(m.match_date); cs["d"] = pd.to_datetime(cs.date)
cand = m[["match_id", "d", "key_teams"]].merge(cs[["cs_match_id", "d", "key_teams", "bat1", "bat2", "t1"]], on="key_teams", suffixes=("", "_cs"))
cand["lag"] = (cand.d_cs - cand.d).dt.days
cand = cand[cand.lag.abs() <= 1].join(ob, on="match_id")
same_order = cand.t1 == cand.team1
cand["bat_diff"] = np.where(same_order, (cand.bat_runs1 - cand.bat1).abs() + (cand.bat_runs2 - cand.bat2).abs(), np.nan)
cand["score"] = cand.lag.abs() * 5 + cand.bat_diff.fillna(50)
cand = cand.sort_values("score")
used, rows = set(), []
for _, r in cand.iterrows():                 # greedy one-to-one assignment, best score first
    if r.match_id in {x["match_id"] for x in rows} or r.cs_match_id in used:
        continue
    used.add(r.cs_match_id); rows.append(r)
link = pd.DataFrame(rows)[["match_id", "cs_match_id", "lag", "bat_diff", "score"]]
link["link_quality"] = np.select([link.bat_diff.eq(0), link.bat_diff.le(4)], ["exact_bat_runs", "close_bat_runs"],
                                 np.where(link.bat_diff.isna(), "teams_dates_only", "weak"))
link.to_csv(os.path.join(out, "cricsheet_link.csv"), index=False)
um = m[~m.match_id.isin(link.match_id)]
print("linked", len(link), "of", len(m), "|", link.link_quality.value_counts().to_dict())
cat = np.select([um.result.str.contains("Abandoned"), um.home_team.str.contains("Afghanistan") | um.away_team.str.contains("Afghanistan"),
                 um.home_team.str.endswith(" A") | um.away_team.str.endswith(" A")],
                ["abandoned", "afghanistan (absent from cricsheet)", "A-team (not international)"], "other")
print("unlinked:", pd.Series(cat).value_counts().to_dict())
print(um[cat == "other"][["match_id", "match_date", "home_team", "away_team", "comp_name"]].to_string(index=False))
