"""Attach ground coordinates, true local time, sunset-based light condition and weather.
Inputs (out/): match_context.csv, match_ground.csv, grounds_geocoded.csv, innings_weather.csv, innings_windows.csv
Outputs (out/): match_context_enriched.csv (match level), innings_context.csv (innings level)
Usage: python3 enrich_context.py <out_dir>

Definitions
- first_ball_local_ground: first ball in the ground's own IANA time zone (from geocoding), replacing the
  one-zone-per-country assumption in match_context.csv (wrong for Australia and the West Indies).
- light_innings: share of the innings bowled after local sunset: <0.25 daylight, 0.25-0.75 transition, >0.75 under_lights.
- light_match: night if innings 1 is mostly after sunset; day_night if only innings 2 is; otherwise day.
- dew_band_descriptive: ONLY for innings mostly under lights; dew-point depression (air temp minus dew point)
  <=3 C 'high', 3-6 C 'moderate', >6 C 'low'. A heuristic for description and the dashboard; the rating models
  use the continuous value, and only if it passes the Phase 3 ablation.
- Weather is ERA5 reanalysis at city coordinates (Open-Meteo), not observations at the ground.
"""
import sys, os
import numpy as np, pandas as pd
from zoneinfo import ZoneInfo

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
rd = lambda f, **k: pd.read_csv(os.path.join(OUT, f), **k)
ctx = rd("match_context.csv", parse_dates=["first_ball_utc"])
mg = rd("match_ground.csv")
geo = rd("grounds_geocoded.csv")
wx = rd("innings_weather.csv", parse_dates=["sunset_utc"])
win = rd("innings_windows.csv", parse_dates=["first_ball_utc", "last_ball_utc"])

g = geo[["ground", "latitude", "longitude", "elevation_m", "timezone", "coord_level"]].rename(columns={"timezone": "ground_tz"})
m = ctx.merge(mg[["match_id", "ground", "city", "ground_source", "ground_confidence", "venue_id_conflict"]], on="match_id", how="left") \
       .merge(g, on="ground", how="left")

def local(ts, tz):
    if pd.isna(ts) or pd.isna(tz):
        return pd.NaT
    return ts.tz_localize("UTC").tz_convert(ZoneInfo(tz)).tz_localize(None)
m["first_ball_local_ground"] = [local(t, z) for t, z in zip(m.first_ball_utc, m.ground_tz)]
m["local_time_changed_vs_country_tz"] = (m.first_ball_local_ground != pd.to_datetime(m.first_ball_local)) & m.first_ball_local_ground.notna()
m["local_tz_source"] = np.where(m.ground_tz.notna(), "ground_geocode", "none")

# innings level
inn = win[["match_id", "innings", "first_ball_utc", "last_ball_utc"]].merge(wx, on=["match_id", "innings"], how="left")
inn = inn.merge(m[["match_id", "ground_tz", "ground", "ts_confidence", "ground_confidence"]], on="match_id", how="left")
inn["innings_start_local"] = [local(t, z) for t, z in zip(inn.first_ball_utc, inn.ground_tz)]
inn["sunset_local"] = [local(t, z) for t, z in zip(inn.sunset_utc, inn.ground_tz)]
s = inn.share_of_innings_after_sunset
inn["light_innings"] = np.select([s.isna(), s < 0.25, s <= 0.75], ["unknown", "daylight", "transition"], "under_lights")
dd = inn.dewpoint_depression_c
inn["dew_band_descriptive"] = np.where(inn.light_innings != "under_lights", "not_applicable",
                              np.select([dd.isna(), dd <= 3, dd <= 6], ["unknown", "high", "moderate"], "low"))
# timing reliability: weather + light inherit the timestamp confidence of the match
inn["timing_confidence"] = inn.ts_confidence
inn = inn.drop(columns=["ground_tz", "ts_confidence"])

# match level light condition
sh = inn.pivot(index="match_id", columns="innings", values="share_of_innings_after_sunset")
sh.columns = [f"after_sunset_share_inn{c}" for c in sh.columns]
m = m.join(sh, on="match_id")
a1, a2 = m.after_sunset_share_inn1, m.after_sunset_share_inn2
m["light_match"] = np.select([a1.isna(), a1 >= 0.5, a2 >= 0.5], ["unknown", "night", "day_night"], "day")
for v in ["temperature_2m_mean", "relative_humidity_2m_mean", "dewpoint_depression_c", "precipitation_sum_mm"]:
    p = inn.pivot(index="match_id", columns="innings", values=v)
    p.columns = [f"{v}_inn{c}" for c in p.columns]
    m = m.join(p, on="match_id")

m.to_csv(os.path.join(OUT, "match_context_enriched.csv"), index=False)
inn.to_csv(os.path.join(OUT, "innings_context.csv"), index=False)

played = m[m.has_ball_data]
print("matches:", len(m), "| played with ground tz:", played.ground_tz.notna().sum(), "of", len(played))
print("local start changed vs country-level tz:", int(played.local_time_changed_vs_country_tz.sum()),
      played[played.local_time_changed_vs_country_tz].venue_country.value_counts().to_dict())
print("light_match:", played.light_match.value_counts().to_dict())
print(pd.crosstab(played.match_format, played.light_match))
print("old start_slot vs new light_match (played):"); print(pd.crosstab(played.start_slot, played.light_match))
print("dew band (innings under lights):", inn[inn.light_innings == "under_lights"].dew_band_descriptive.value_counts().to_dict())
