"""Align Cricsheet deliveries 1:1 with data-pack deliveries for linked matches, to take EXACT extras per ball.
Alignment accepted for an innings only if delivery counts match AND runs-off-the-bat agree on every ball.
Usage: python3 align_cricsheet.py <cricsheet_root> <out_dir>"""
import json, os, sys, glob
import pandas as pd, numpy as np
root, out = sys.argv[1], sys.argv[2]
L = pd.read_csv(os.path.join(out, "cricsheet_link.csv"), dtype={"cs_match_id": str})
cs = pd.read_csv(os.path.join(out, "cricsheet_index.csv"), dtype={"cs_match_id": str}).set_index("cs_match_id")
ours = pd.read_csv(os.path.join(out, "deliveries_clean.csv"), usecols=["match_id", "innings", "innings_ball_num", "bat_runs",
                   "extras_type", "is_super_over"], low_memory=False)
ours = ours[~ours.is_super_over].sort_values(["match_id", "innings", "innings_ball_num"])
files = {os.path.basename(f)[:-5]: f for f in glob.glob(os.path.join(root, "*", "*.json"))}
rows, status = [], []
for _, l in L.iterrows():
    d = json.load(open(files[l.cs_match_id]))
    main = [i for i in d.get("innings", []) if not i.get("super_over")][:2]
    for k, inn in enumerate(main, start=1):
        dl = [x for ov in inn["overs"] for x in ov["deliveries"]]
        o = ours[(ours.match_id == l.match_id) & (ours.innings == k)]
        cs_bat = np.array([x["runs"]["batter"] for x in dl])
        ok = len(dl) == len(o) and np.array_equal(cs_bat, o.bat_runs.values)
        status.append({"match_id": l.match_id, "innings": k, "n_ours": len(o), "n_cs": len(dl), "aligned": ok})
        if ok:
            for ibn, x in zip(o.innings_ball_num.values, dl):
                e = x["runs"]["extras"]
                rows.append({"match_id": l.match_id, "innings": k, "innings_ball_num": ibn, "cs_extras_runs": e, "cs_total_runs": x["runs"]["total"],
                             "cs_extras_kinds": ",".join(sorted(x.get("extras", {}).keys()))})
st = pd.DataFrame(status); st.to_csv(os.path.join(out, "cricsheet_alignment_status.csv"), index=False)
pd.DataFrame(rows).to_csv(os.path.join(out, "cricsheet_ball_extras.csv"), index=False)
print("innings aligned:", st.aligned.sum(), "of", len(st), "| matches fully aligned:", st.groupby("match_id").aligned.all().sum())
print("misaligned examples:"); print(st[~st.aligned].head(8).to_string(index=False))
