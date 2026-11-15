"""SupeRR Selector - fetch ground coordinates, time zones, hourly weather and sunset from Open-Meteo.

Run on a machine with internet access:
    pip install requests pandas
    python fetch_context.py              # full run (~1-3 minutes per 100 matches; all responses cached)
    python fetch_context.py --test       # geocode 3 grounds + weather for 2 matches, print raw responses

Inputs  (same folder):  grounds.csv, innings_windows.csv
Outputs (same folder):  grounds_geocoded.csv, innings_weather.csv, cache/  (raw JSON for every request)

Sources (store in the report):
  Geocoding: Open-Meteo Geocoding API (GeoNames data, CC-BY 4.0)  https://open-meteo.com/en/docs/geocoding-api
  Weather:   Open-Meteo Historical Weather API (ERA5 / ERA5-Land reanalysis, CC-BY 4.0)  https://open-meteo.com/en/docs/historical-weather-api
Weather values are REANALYSIS on a ~9-25 km grid, at CITY coordinates - not ground observations. Flags record this.
"""
import json, os, sys, time, hashlib
from datetime import timedelta
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
TEST = "--test" in sys.argv
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
WX_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation", "cloud_cover", "wind_speed_10m"]
COUNTRY_CODES = {
    "India": ["IN"], "Sri Lanka": ["LK"], "Bangladesh": ["BD"], "Pakistan": ["PK"], "United Arab Emirates": ["AE"],
    "Oman": ["OM"], "Qatar": ["QA"], "Japan": ["JP"], "South Africa": ["ZA"], "Zimbabwe": ["ZW"],
    "United Kingdom": ["GB"], "Ireland": ["IE", "GB"],          # Belfast/Bready are in Northern Ireland (GB)
    "Netherlands": ["NL"], "New Zealand": ["NZ"], "Australia": ["AU"], "United States": ["US"],
    "West Indies": ["TT", "BB", "JM", "GY", "LC", "VC", "GD", "KN", "DM", "AG"],
}

WI_CITY_CODES = {  # "West Indies" spans many countries: restrict each city to its own country code
    "Bridgetown": ["BB"], "Providence": ["GY"], "Georgetown": ["GY"], "Tarouba": ["TT"], "San Fernando": ["TT"],
    "Port of Spain": ["TT"], "Kingston": ["JM"], "Gros Islet": ["LC"], "Kingstown": ["VC"], "Basseterre": ["KN"],
    "Roseau": ["DM"], "North Sound": ["AG"], "St John's": ["AG"], "St George's": ["GD"]}

ALT_QUERY = {  # fallback place names if GeoNames lacks the city label used by Cricsheet (flagged via coord_query)
    "New Chandigarh": ["Mullanpur"], "Tarouba": ["San Fernando"], "Bready": ["Strabane"], "Al Amarat": ["Muscat"],
    "Providence": ["Georgetown"], "Carrara": ["Gold Coast"], "St George's": ["Saint George's", "St. George's"]}

def get(url, params, tag):
    """GET with on-disk cache and polite retry. Returns parsed JSON."""
    key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:16]
    path = os.path.join(CACHE, f"{tag}_{key}.json")
    if os.path.exists(path):
        return json.load(open(path))
    for attempt in range(5):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1)); continue
        r.raise_for_status()
        data = r.json()
        json.dump({"_request": {"url": url, "params": params, "retrieved_utc": pd.Timestamp.now("UTC").isoformat()},
                   **data}, open(path, "w"))
        time.sleep(0.25)
        return data
    raise RuntimeError(f"rate limited: {url} {params}")

# ------------------------------------------------------------------ 1. geocode grounds (city level)
grounds = pd.read_csv(os.path.join(HERE, "grounds.csv"))
if TEST:
    grounds = grounds[grounds.city.isin(["Jaipur", "Mount Maunganui", "Lauderhill"])]
unmapped = sorted(set(grounds[(grounds.country == "West Indies") & ~grounds.city.isin(WI_CITY_CODES)].city))
if unmapped:
    sys.exit(f"West Indies cities missing from WI_CITY_CODES (add them, do not guess): {unmapped}")
rows = []
for _, g in grounds.iterrows():
    codes = WI_CITY_CODES[g.city] if g.country == "West Indies" else COUNTRY_CODES.get(g.country, [])
    hit, query_used = None, None
    queries = [g.city] + ALT_QUERY.get(g.city, []) + [g.ground.split(",")[0]]
    for q in queries:                                   # city, then a nearby-town fallback, then the ground name
        res = get(GEO_URL, {"name": q, "count": 10, "language": "en", "format": "json"}, "geo").get("results", [])
        res = [x for x in res if not codes or x.get("country_code") in codes]
        if res:
            hit, query_used = res[0], q; break
    if TEST:
        print("GEOCODE", g.ground, "->", json.dumps(hit)[:300])
    rows.append({**g.to_dict(),
                 "latitude": hit and hit["latitude"], "longitude": hit and hit["longitude"],
                 "elevation_m": hit and hit.get("elevation"), "timezone": hit and hit.get("timezone"),
                 "geo_name": hit and hit.get("name"), "geo_admin1": hit and hit.get("admin1"),
                 "geo_country_code": hit and hit.get("country_code"), "geo_id": hit and hit.get("id"),
                 "coord_level": "city" if hit else None, "coord_query": query_used,
                 "coord_source": "Open-Meteo Geocoding API (GeoNames)" if hit else None,
                 "missing_reason": None if hit else "no geocoding result in expected country"})
geo = pd.DataFrame(rows)
geo.to_csv(os.path.join(HERE, "grounds_geocoded.csv"), index=False)
print(f"geocoded {geo.latitude.notna().sum()}/{len(geo)} grounds; missing: {list(geo[geo.latitude.isna()].ground)}")

# ------------------------------------------------------------------ 2. weather per innings
win = pd.read_csv(os.path.join(HERE, "innings_windows.csv"), parse_dates=["first_ball_utc", "last_ball_utc"])
win = win.merge(geo[["ground", "latitude", "longitude", "timezone", "elevation_m"]], on="ground", how="left")
if TEST:
    win = win[win.match_id.isin(win.dropna(subset=["latitude"]).match_id.unique()[:2])]
out = []
for mid, w in win.groupby("match_id"):
    lat, lon = w.latitude.iloc[0], w.longitude.iloc[0]
    if pd.isna(lat) or w.first_ball_utc.isna().all():
        for _, r in w.iterrows():
            out.append({"match_id": mid, "innings": r.innings,
                        "missing_reason": "no coordinates" if pd.isna(lat) else "no timestamps"})
        continue
    start = (w.first_ball_utc.min() - timedelta(days=1)).date().isoformat()
    end = (w.last_ball_utc.max() + timedelta(days=1)).date().isoformat()
    data = get(WX_URL, {"latitude": round(lat, 4), "longitude": round(lon, 4), "start_date": start, "end_date": end,
                        "hourly": ",".join(HOURLY), "daily": "sunset", "timezone": "GMT"}, "wx")
    if TEST:
        print("WEATHER", mid, {k: data[k] for k in data if k not in ("hourly", "daily")})
        print("  hourly sample:", {k: v[:2] for k, v in data["hourly"].items()}, "| daily:", data.get("daily"))
    h = pd.DataFrame(data["hourly"]); h["time"] = pd.to_datetime(h.time)
    sunsets = pd.to_datetime(pd.Series(data["daily"]["sunset"]))
    for _, r in w.iterrows():
        a, z = r.first_ball_utc.floor("h"), r.last_ball_utc.ceil("h")
        seg = h[(h.time >= a) & (h.time <= z)]
        ss = sunsets.iloc[(sunsets - r.first_ball_utc).abs().argmin()]          # sunset nearest to this innings
        dur = (r.last_ball_utc - r.first_ball_utc).total_seconds()
        after = max(0.0, (r.last_ball_utc - max(ss, r.first_ball_utc)).total_seconds())
        out.append({
            "match_id": mid, "innings": r.innings, "n_hours": len(seg),
            **{f"{v}_mean": seg[v].mean() for v in HOURLY if v != "precipitation"},
            "precipitation_sum_mm": seg.precipitation.sum(),
            "dewpoint_depression_c": (seg.temperature_2m - seg.dew_point_2m).mean(),
            "sunset_utc": ss, "start_minus_sunset_min": (r.first_ball_utc - ss).total_seconds() / 60,
            "share_of_innings_after_sunset": (after / dur) if dur > 0 else None,
            "wx_source": "Open-Meteo Historical Weather API (ERA5 reanalysis)", "wx_is_reanalysis": True,
            "wx_coord_level": "city", "wx_grid_lat": data.get("latitude"), "wx_grid_lon": data.get("longitude"),
            "missing_reason": None if len(seg) else "no hourly rows in innings window"})
res = pd.DataFrame(out)
res.to_csv(os.path.join(HERE, "innings_weather.csv"), index=False)
print(f"weather rows: {len(res)}; with data: {res.n_hours.fillna(0).gt(0).sum() if 'n_hours' in res else 0}")
print("Done. Upload grounds_geocoded.csv and innings_weather.csv (the cache/ folder is optional, for provenance).")
