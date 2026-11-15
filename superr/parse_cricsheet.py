"""Parse Cricsheet JSON (men's T20I + IPL, 2022+) into a match-level index with venue, city and exact innings totals.
Usage: python3 parse_cricsheet.py <cricsheet_root_dir> <out_csv>"""
import json, glob, sys, os
import pandas as pd
root, out = sys.argv[1], sys.argv[2]
rows = []
for f in glob.glob(os.path.join(root, "*", "*.json")):
    d = json.load(open(f)); i = d["info"]
    if i.get("gender") != "male" or i["dates"][0] < "2021-12-25":
        continue
    r = {"cs_match_id": os.path.basename(f)[:-5], "cs_source": f.split(os.sep)[-2], "date": i["dates"][0],
         "team1": i["teams"][0], "team2": i["teams"][1], "venue": i.get("venue"), "city": i.get("city"),
         "event": (i.get("event") or {}).get("name"), "winner": (i.get("outcome") or {}).get("winner"),
         "by_runs": ((i.get("outcome") or {}).get("by") or {}).get("runs"),
         "by_wkts": ((i.get("outcome") or {}).get("by") or {}).get("wickets"),
         "result_type": (i.get("outcome") or {}).get("result"), "method": (i.get("outcome") or {}).get("method")}
    k = 0
    for inn in d.get("innings", []):
        if inn.get("super_over"):
            continue
        k += 1
        if k > 2:
            break
        runs = wk = legal = wides = nb = byes = lb = pen = 0
        for ov in inn.get("overs", []):
            for dl in ov["deliveries"]:
                runs += dl["runs"]["total"]; wk += len(dl.get("wickets", []))
                ex = dl.get("extras", {})
                wides += ex.get("wides", 0); nb += ex.get("noballs", 0); byes += ex.get("byes", 0)
                lb += ex.get("legbyes", 0); pen += ex.get("penalty", 0)
                legal += 0 if ("wides" in ex or "noballs" in ex) else 1
        r.update({f"i{k}_team": inn["team"], f"i{k}_runs": runs, f"i{k}_pen_runs": pen, f"i{k}_wkts": wk, f"i{k}_legal": legal,
                  f"i{k}_wide_runs": wides, f"i{k}_nb_runs": nb, f"i{k}_bye_runs": byes, f"i{k}_legbye_runs": lb,
                  f"i{k}_target": (inn.get("target") or {}).get("runs"),
                  f"i{k}_target_overs": (inn.get("target") or {}).get("overs")})
    rows.append(r)
df = pd.DataFrame(rows).sort_values("date")
df.to_csv(out, index=False)
print(len(df), df.cs_source.value_counts().to_dict(), df.date.min(), df.date.max())
