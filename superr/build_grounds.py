"""Assign a verified ground to every match.
Primary source: Cricsheet venue for linked matches (ODC-By, cricsheet.org).
Fallback for unlinked matches (Afghanistan matches are absent from Cricsheet; A-team matches are not internationals):
the most common Cricsheet ground for the same data-pack venue_id, confidence reduced, and 'low' when that venue_id
is known to map to more than one ground.
Usage: python3 build_grounds.py <data_pack> <out_dir>"""
import sys, os
import pandas as pd, numpy as np
pack, out = sys.argv[1], sys.argv[2]
ALIAS = {  # renamed or re-labelled grounds -> one canonical name (same physical ground)
    "Hnry Stadium, Wellington": "Sky Stadium, Wellington",
    "Bir Sreshtho Flight Lieutenant Matiur Rahman Stadium, Chattogram": "Zahur Ahmed Chowdhury Stadium, Chattogram",
    "Niranjan Shah Stadium, Rajkot": "Saurashtra Cricket Association Stadium, Rajkot",
    "People First Stadium, Carrara": "Metricon Stadium, Carrara",
}
CITY_FIX = {"Dr DY Patil Sports Academy, Navi Mumbai": "Navi Mumbai",
            "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh": "New Chandigarh"}
m = pd.read_csv(os.path.join(out, "match_context.csv"))
L = pd.read_csv(os.path.join(out, "cricsheet_link.csv"), dtype={"cs_match_id": str})
cs = pd.read_csv(os.path.join(out, "cricsheet_index.csv"), dtype={"cs_match_id": str})
x = m[["match_id", "venue_id", "venue_country", "comp_name", "has_ball_data"]].merge(
    L.merge(cs[["cs_match_id", "venue", "city"]], on="cs_match_id"), on="match_id", how="left")
x["ground"] = x.venue.replace(ALIAS)
x["city"] = np.where(x.ground.isin(CITY_FIX), x.ground.map(CITY_FIX), x.city)
x["ground_source"] = np.where(x.ground.notna(), "cricsheet_match_record", None)
x["ground_confidence"] = np.where(x.ground.notna(), "high", None)

# venue_id -> ground mapping learned from linked matches
vmap = x[x.ground.notna()].groupby("venue_id").ground.agg(lambda s: s.value_counts().index[0])
vcity = x[x.ground.notna()].groupby("venue_id").city.agg(lambda s: s.value_counts().index[0])
vn = x[x.ground.notna()].groupby("venue_id").ground.nunique()
fill = x.ground.isna() & x.venue_id.isin(vmap.index)
x.loc[fill, "ground"] = x.loc[fill, "venue_id"].map(vmap)
x.loc[fill, "city"] = x.loc[fill, "venue_id"].map(vcity)
x.loc[fill, "ground_source"] = "inferred_from_venue_id"
x.loc[fill, "ground_confidence"] = np.where(x.loc[fill, "venue_id"].map(vn) > 1, "low", "medium")
x["ground_confidence"] = x.ground_confidence.fillna("none")

# conflicts between data-pack venue_id and the per-match Cricsheet ground
x["venue_id_conflict"] = (x.ground_source == "cricsheet_match_record") & (x.ground != x.venue_id.map(vmap))
x[["match_id", "venue_id", "ground", "city", "venue_country", "ground_source", "ground_confidence", "venue_id_conflict",
   "cs_match_id", "link_quality"]].to_csv(os.path.join(out, "match_ground.csv"), index=False)

g = x[x.ground.notna()].groupby("ground").agg(
    city=("city", "first"), country=("venue_country", lambda s: s.value_counts().index[0]),
    n_matches=("match_id", "size"), n_ipl=("comp_name", lambda s: s.str.startswith("IPL").sum()),
    venue_ids=("venue_id", lambda s: ",".join(map(str, sorted(s.unique()))))).reset_index()
g["aliases"] = g.ground.map(lambda n: "; ".join(k for k, v in ALIAS.items() if v == n))
for col in ["latitude", "longitude", "timezone", "coord_source", "coord_level"]:
    g[col] = None
g.sort_values("n_matches", ascending=False).to_csv(os.path.join(out, "grounds.csv"), index=False)

played = x[x.has_ball_data]
print("grounds:", len(g), "| played matches by ground confidence:", played.ground_confidence.value_counts().to_dict())
print("venue_id conflicts (Cricsheet ground != venue_id's usual ground):", int(x.venue_id_conflict.sum()))
print(x[x.venue_id_conflict][["match_id", "venue_id", "ground", "comp_name"]].merge(
    vmap.rename("venue_id_usual_ground"), left_on="venue_id", right_index=True).to_string(index=False))
print("no ground at all:", played[played.ground.isna()][["match_id", "venue_id", "venue_country", "comp_name"]].to_string(index=False))
