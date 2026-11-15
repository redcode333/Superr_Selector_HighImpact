"""Player badges (STEPS_ABC.md, step A): a picture of each player's own data, computed at build time from players.json.

Batter: 9 wedges in field orientation (straight up, off side clockwise), the deep-dive field-map sector angles.
        length = R_IN + K x sqrt(share), share = his share of runs in that region (players.json regions[i].run_share / 100);
        colour on a pink scale from #f6a8c2 (his smallest share) to #e4007c (his largest).
Bowler: 20 ticks, over 1 at the top, clockwise. length = L_MAX x balls / his busiest over's balls;
        an over he never bowled gets a faint outline tick; colour on a blue scale from #728dc8 (fewest balls) to #254aa5 (most).
Geometry is in a 100 x 100 box, centre (50, 50). K is one constant for all batters, so wedge lengths compare across badges.
check_data.py re-derives every value from players.json and asserts equality.
"""
import math

C = 50.0
DISC_R = 18.5            # navy disc with initials
BAT_R_IN, BAT_R_MAX = 20.5, 48.0
TICK_R_IN, TICK_L_MAX, TICK_EMPTY = 23.5, 24.0, 5.0
PINK = ("#f6a8c2", "#e4007c")
BLUE = ("#728dc8", "#254aa5")


def mix(a, b, t):
    """Linear interpolation between two hex colours, t in [0, 1]; rounded to the nearest integer channel."""
    t = max(0.0, min(1.0, t))
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]; cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(math.floor(x + (y - x) * t + 0.5)):02x}" for x, y in zip(ca, cb))


def bat_k(dives):
    """One K for every batter: the largest region share in the pool reaches BAT_R_MAX."""
    top = max((r["run_share"] or 0) / 100 for d in dives if d["role"] == "batter" for r in d["regions"])
    return (BAT_R_MAX - BAT_R_IN) / math.sqrt(top)


def make_badge(d, k):
    if d["role"] == "batter":
        shares = [(r["run_share"] or 0) / 100 for r in d["regions"]]
        lo, hi = min(shares), max(shares)
        wedges = [{"name": r["name"], "from": r["from"], "to": r["to"], "share": s,
                   "len": round(BAT_R_IN + k * math.sqrt(s), 4),
                   "color": mix(*PINK, (s - lo) / (hi - lo) if hi > lo else 1.0)} for r, s in zip(d["regions"], shares)]
        return {"kind": "batter", "k": round(k, 6), "r_in": BAT_R_IN, "disc_r": DISC_R, "wedges": wedges,
                "label": f"{d['name']}'s badge: where he scores his runs"}
    balls = [o["balls"] for o in d["overs"]]
    mx = max(balls); bowled = [b for b in balls if b]; lo = min(bowled)
    ticks = []
    for o in d["overs"]:
        b = o["balls"]
        ticks.append({"over": o["over"], "balls": b, "angle": (o["over"] - 1) * 18.0,
                      "len": round(TICK_L_MAX * b / mx, 4) if b else TICK_EMPTY, "empty": not b,
                      "color": mix(*BLUE, (b - lo) / (mx - lo) if mx > lo else 1.0) if b else None})
    return {"kind": "bowler", "r_in": TICK_R_IN, "l_max": TICK_L_MAX, "disc_r": DISC_R, "ticks": ticks,
            "label": f"{d['name']}'s badge: which overs he bowls"}
