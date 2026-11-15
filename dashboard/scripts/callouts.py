"""Deep-dive callouts from templates, plus the role label and the per-section "few balls" flags.

Every callout is stored with:
  text    the sentence shown on the page
  nums    for each number in the sentence, in order: an expression over the player's players.json record (P) and a format
  guards  the sample checks that let the template fire: {kind, expr, min}
check_data.py re-evaluates every expression against players.json, extracts the numbers from the text and asserts they match,
and asserts every guard against the minimums below. A template fires only when all its guards pass; if no template
qualifies for a slot, the slot is left empty (never padded).
"""
import re
from decimal import Decimal, ROUND_HALF_UP

MINS = {"region": 30, "over": 24, "style": 30, "opponent": 15, "ground_batter": 30, "ground_bowler": 60,
        "outs": 15, "wickets": 15, "hand": 30, "delivery": 30, "advice": 100}   # advice wording ("Bowl ...", "Watch ...") needs >= 100 balls
FMT = {"i": "1", "1": "0.1", "2": "0.01"}      # decimal places; rounding is half up (as on the design pages and in JS)
GAP_SR, GAP_ECON = 5, 0.25                      # a "better / worse than par" claim needs at least this gap
PHRASE = {"Third man": "to third man", "Point": "through point", "Cover": "through cover", "Long-off": "to long-off",
          "Straight": "straight down the ground", "Long-on": "to long-on", "Midwicket": "through midwicket",
          "Square leg": "to square leg", "Fine leg": "to fine leg"}
DIGIT = re.compile(r"\d")
DEL_LABEL = {"Stock: seam & swing": "stock ball", "Yorker": "yorker", "Slower ball / cutter": "slower ball",
             "Bouncer": "bouncer", "Full toss": "full toss"}
STYLE_LABEL = {"Right-arm pace": "right-arm pace", "Left-arm pace": "left-arm pace", "Off-spin": "off-spin", "Leg-spin": "leg-spin",
               "Left-arm finger spin": "left-arm finger spin", "Left-arm wrist spin": "left-arm wrist spin"}


def ev(P, expr):
    return eval(expr, {"__builtins__": {}, "P": P, "sum": sum, "len": len, "max": max, "min": min, "abs": abs})  # noqa: S307 (own expressions)


def fmtv(v, f):
    return str(Decimal(repr(float(v))).quantize(Decimal(FMT[f]), rounding=ROUND_HALF_UP))


class Out:
    def __init__(self, P):
        self.P, self.items = P, []

    def add(self, slot, tid, tmpl, nums, guards, **words):
        for w in words.values():
            assert not DIGIT.search(w), (tid, w)          # numbers may only come from nums
        if not all(ev(self.P, g[1]) >= g[2] for g in guards):
            return
        vals = [ev(self.P, e) for e, _ in nums]
        assert all(v >= 0 for v in vals), (tid, vals)     # sentences never carry a sign
        text = tmpl.format(*[fmtv(v, f) for v, (_, f) in zip(vals, nums)], **words)
        self.items.append({"slot": slot, "id": tid, "text": text, "nums": [{"expr": e, "fmt": f} for e, f in nums],
                           "guards": [{"kind": k, "expr": e, "min": m} for k, e, m in guards]})


def plural(n, one, many): return one if n == 1 else many


# ------------------------------------------------------------------------------------------------ batters
def batter_callouts(P):
    o = Out(P); reg = P["regions"]; R = "P['regions']"
    q = [i for i, r in enumerate(reg) if r["balls"] >= MINS["region"]]
    g_reg = lambda i: ("region", f"{R}[{i}]['balls']", MINS["region"])
    # wheel
    a, b = sorted(range(9), key=lambda i: (-reg[i]["runs"], i))[:2]
    ra, rb = reg[a]["name"].lower(), reg[b]["name"].lower()
    o.add("wheel", "BAT_WHEEL_MAIN", "{0}% of his runs go to {ra} and {rb}, at strike rates of {1} and {2} (par {3} and {4}).",
          [(f"100*({R}[{a}]['runs']+{R}[{b}]['runs'])/P['runs']", "i"), (f"{R}[{a}]['sr']", "i"), (f"{R}[{b}]['sr']", "i"),
           (f"{R}[{a}]['league_sr']", "i"), (f"{R}[{b}]['league_sr']", "i")], [g_reg(a), g_reg(b)], ra=ra, rb=rb)
    if reg[a]["sixes"] + reg[b]["sixes"] >= 1:
        o.add("wheel", "BAT_WHEEL_SIXES", "{0} of his {1} sixes land in those two regions.",
              [(f"{R}[{a}]['sixes']+{R}[{b}]['sixes']", "i"), ("P['sixes']", "i")], [g_reg(a), g_reg(b)])
    oa, ob = reg[a]["outs"], reg[b]["outs"]
    if oa + ob >= 3 and oa + ob >= 0.4 * P["outs"]:
        o.add("wheel", "BAT_WHEEL_OUTS", "{0} of his {1} dismissals come on those same shots: {2} towards {ra} and {3} towards {rb}. "
              "His strength is also where he gets out.",
              [(f"{R}[{a}]['outs']+{R}[{b}]['outs']", "i"), ("P['outs']", "i"), (f"{R}[{a}]['outs']", "i"), (f"{R}[{b}]['outs']", "i")],
              [g_reg(a), g_reg(b)], ra=ra, rb=rb)
    below = [i for i in q if reg[i]["sr"] < reg[i]["league_sr"]]
    if below:
        w = min(below, key=lambda i: (reg[i]["sr"] - reg[i]["league_sr"], i))
        if reg[w]["league_sr"] - reg[w]["sr"] >= GAP_SR:
            ph = PHRASE[reg[w]["name"]]; ph = ph[0].upper() + ph[1:]
            if len(below) == 1:
                o.add("wheel", "BAT_WHEEL_WEAK_ONLY", "{r} he scores {0} against a par of {1}: the only region with at least {2} balls where he is below par.",
                      [(f"{R}[{w}]['sr']", "i"), (f"{R}[{w}]['league_sr']", "i"), ("P['mins']['region']", "i")], [g_reg(w)], r=ph)
            else:
                o.add("wheel", "BAT_WHEEL_WEAK", "{r} he scores {0} against a par of {1}: his weakest region against par.",
                      [(f"{R}[{w}]['sr']", "i"), (f"{R}[{w}]['league_sr']", "i")], [g_reg(w)], r=ph)
    # overs
    ov = P["overs"]; O = "P['overs']"
    qo = [i for i, x in enumerate(ov) if x["balls"] >= MINS["over"]]
    g_ov = lambda i: ("over", f"{O}[{i}]['balls']", MINS["over"])
    if qo:
        bi = max(qo, key=lambda i: (ov[i]["sr"] - ov[i]["par_sr"], -i))
        if ov[bi]["sr"] - ov[bi]["par_sr"] >= GAP_SR:
            o.add("overs", "BAT_OVER_BEST", "His best over against par is over {0}: a strike rate of {1} against {2}.",
                  [(f"{O}[{bi}]['over']", "i"), (f"{O}[{bi}]['sr']", "i"), (f"{O}[{bi}]['par_sr']", "i")], [g_ov(bi)])
        wi = min(qo, key=lambda i: (ov[i]["sr"] - ov[i]["par_sr"], i))
        if ov[wi]["par_sr"] - ov[wi]["sr"] >= GAP_SR:
            o.add("overs", "BAT_OVER_WEAK", "His slowest against par is over {0}: {1} against {2}.",
                  [(f"{O}[{wi}]['over']", "i"), (f"{O}[{wi}]['sr']", "i"), (f"{O}[{wi}]['par_sr']", "i")], [g_ov(wi)])
        dq = [i for i in qo if ov[i]["over"] >= 16]
        if len(dq) >= 3 and all(ov[i]["sr"] > ov[i]["par_sr"] for i in dq):
            o.add("overs", "BAT_OVER_DEATH", "Above par in all {0} death overs where he has at least {1} balls.",
                  [(f"sum(1 for x in {O} if x['over'] >= 16 and x['balls'] >= P['mins']['over'])", "i"), ("P['mins']['over']", "i")],
                  [g_ov(i) for i in dq])
    # dismissals
    D = P["dismissals"]; ix = {d["type"]: i for i, d in enumerate(D)}
    g_out = ("outs", "P['outs']", MINS["outs"])
    c, bw, lb = ix.get("Caught"), ix.get("Bowled"), ix.get("LBW")
    if c is not None and bw is not None and D[c]["pct"] >= D[c]["league_pct"] + 5 and D[bw]["pct"] <= D[bw]["league_pct"]:
        o.add("dismissals", "BAT_OUT_AIR", "{0}% of his dismissals are caught, against {1}% for all batters: he gets out in the air, not to the stumps.",
              [(f"P['dismissals'][{c}]['pct']", "i"), (f"P['dismissals'][{c}]['league_pct']", "i")], [g_out])
    if bw is not None and lb is not None and D[bw]["pct"] + D[lb]["pct"] >= D[bw]["league_pct"] + D[lb]["league_pct"] + 5:
        o.add("dismissals", "BAT_OUT_STUMPS", "{0}% of his dismissals are bowled or lbw, against {1}% for all batters: bowled or lbw more often than most.",
              [(f"P['dismissals'][{bw}]['pct']+P['dismissals'][{lb}]['pct']", "i"),
               (f"P['dismissals'][{bw}]['league_pct']+P['dismissals'][{lb}]['league_pct']", "i")], [g_out])
    # styles
    st = P["styles"]; S = "P['styles']"
    qs = [i for i, x in enumerate(st) if x["balls"] >= MINS["style"]]
    g_st = lambda i: ("style", f"{S}[{i}]['balls']", MINS["style"])
    bel = [i for i in qs if st[i]["sr"] < st[i]["par_sr"]]
    w = min(bel, key=lambda i: (st[i]["sr"] - st[i]["par_sr"], i)) if bel else None
    if bel and st[w]["par_sr"] - st[w]["sr"] >= GAP_SR:
        only = len(bel) == 1
        tail = "; the only type with at least {3} balls where he is below par." if only else "."
        nums = [(f"{S}[{w}]['sr']", "i"), (f"{S}[{w}]['par_sr']", "i"), (f"{S}[{w}]['balls']", "i")] + ([("P['mins']['style']", "i")] if only else [])
        if st[w]["balls"] >= MINS["advice"]:      # advice only on a big sample
            o.add("styles", "BAT_STYLE_ADVICE" + ("_ONLY" if only else ""), "Bowl {s} at him: strike rate {0}, par {1}, {2} balls" + tail, nums,
                  [g_st(w), ("advice", f"{S}[{w}]['balls']", MINS["advice"])], s=STYLE_LABEL[st[w]["style"]])
        else:
            o.add("styles", "BAT_STYLE_WEAK" + ("_ONLY" if only else ""), "Weakest against {s}: strike rate {0}, par {1}, {2} balls" + tail, nums,
                  [g_st(w)], s=STYLE_LABEL[st[w]["style"]])
    elif not bel and len(qs) >= 3:
        o.add("styles", "BAT_STYLE_ALL", "He scores above par against all {0} bowling types he has faced for at least {1} balls.",
              [(f"sum(1 for x in {S} if x['balls'] >= P['mins']['style'])", "i"), ("P['mins']['style']", "i")], [g_st(i) for i in qs])
    # opponents
    if P["toughest"]:
        t = P["toughest"][0]
        o.add("opponents", "BAT_OPP_LOWEST_SR", "{n}: {0} off {1} against him, out {2}.",
              [("P['toughest'][0]['runs']", "i"), ("P['toughest'][0]['balls']", "i"), ("P['toughest'][0]['outs']", "i")],
              [("opponent", "P['toughest'][0]['balls']", MINS["opponent"])], n=t["name"])
    # grounds
    _grounds(o, P, "sr", "par_sr", "a strike rate of ", "", MINS["ground_batter"], higher_is_good=True)
    # field setter: what the computed best field covers
    bf = [i for i, r in enumerate(reg) if r["name"] in P["best_field"]["regions"]]
    o.add("field", "FIELD_BEST", "The best five regions for him hold {0} of his {1} boundary runs ({2}%).",
          [("P['best_field']['boundary_runs']", "i"), ("P['region_totals']['boundary_runs']", "i"),
           ("100*P['best_field']['boundary_runs']/P['region_totals']['boundary_runs']", "i")], [g_reg(i) for i in bf])
    return o.items


# ------------------------------------------------------------------------------------------------ bowlers
def bowler_callouts(P):
    o = Out(P); reg = P["regions"]; R = "P['regions']"
    q = [i for i, r in enumerate(reg) if r["balls"] >= MINS["region"]]
    g_reg = lambda i: ("region", f"{R}[{i}]['balls']", MINS["region"])
    if q:
        i = max(q, key=lambda i: (reg[i]["outs"], -i))
        if reg[i]["outs"] >= 3:
            o.add("wheel", "BOWL_WHEEL_WKTS", "{0} of his {1} wickets come from batters hitting towards {r}, where they score {2} against a par of {3}.",
                  [(f"{R}[{i}]['outs']", "i"), ("P['wickets']", "i"), (f"{R}[{i}]['sr']", "i"), (f"{R}[{i}]['league_sr']", "i")], [g_reg(i)],
                  r=reg[i]["name"].lower())
        above = [j for j in q if reg[j]["sr"] > reg[j]["league_sr"]]
        j = max(above, key=lambda j: (reg[j]["sr"] - reg[j]["league_sr"], -j)) if above else None
        if above and reg[j]["sr"] - reg[j]["league_sr"] >= GAP_SR:
            o.add("wheel", "BOWL_WHEEL_LEAK_ONLY" if len(above) == 1 else "BOWL_WHEEL_LEAK",
                  "Batters beat him {r}: a strike rate of {0} against a par of {1}" + (", the one region where they do." if len(above) == 1 else "."),
                  [(f"{R}[{j}]['sr']", "i"), (f"{R}[{j}]['league_sr']", "i")], [g_reg(j)], r=PHRASE[reg[j]["name"]])
        under = [j for j in q if reg[j]["league_sr"] - reg[j]["sr"] >= GAP_SR]
        if under:
            j = max(under, key=lambda j: (reg[j]["league_sr"] - reg[j]["sr"], -j))
            o.add("wheel", "BOWL_WHEEL_SHUT", "Batters are shut down {r}: they score {0} there against him, {1} under par.",
                  [(f"{R}[{j}]['sr']", "i"), (f"{R}[{j}]['league_sr']-{R}[{j}]['sr']", "i")], [g_reg(j)], r=PHRASE[reg[j]["name"]])
    ov = P["overs"]; O = "P['overs']"
    qo = [i for i, x in enumerate(ov) if x["balls"] >= MINS["over"]]
    g_ov = lambda i: ("over", f"{O}[{i}]['balls']", MINS["over"])
    if len(qo) >= 3:
        o.add("overs", "BOWL_OVER_COUNT", "Cheaper than par in {0} of the {1} overs where he has bowled at least {2} balls.",
              [(f"sum(1 for x in {O} if x['balls'] >= P['mins']['over'] and x['econ'] < x['par_econ'])", "i"),
               (f"sum(1 for x in {O} if x['balls'] >= P['mins']['over'])", "i"), ("P['mins']['over']", "i")], [g_ov(i) for i in qo])
    if qo:
        bi = max(qo, key=lambda i: (ov[i]["par_econ"] - ov[i]["econ"], -i))
        if ov[bi]["par_econ"] - ov[bi]["econ"] >= GAP_ECON:
            o.add("overs", "BOWL_OVER_BEST", "Over {0} is his best against par: {1} balls at {2} an over against {3}, with {4} {w}.",
                  [(f"{O}[{bi}]['over']", "i"), (f"{O}[{bi}]['balls']", "i"), (f"{O}[{bi}]['econ']", "2"), (f"{O}[{bi}]['par_econ']", "2"),
                   (f"{O}[{bi}]['wkts']", "i")], [g_ov(bi)], w=plural(ov[bi]["wkts"], "wicket", "wickets"))
    W = P["wicket_types"]; ix = {d["type"]: i for i, d in enumerate(W)}
    g_w = ("wickets", "P['wickets']", MINS["wickets"])
    if "Bowled" in ix and W[ix["Bowled"]]["pct"] >= W[ix["Bowled"]]["league_pct"] + 5:
        k = ix["Bowled"]
        o.add("wicket_types", "BOWL_WT_STUMPS", "{0}% of his wickets are bowled, against {1}% for all bowlers: he hits the stumps.",
              [(f"P['wicket_types'][{k}]['pct']", "i"), (f"P['wicket_types'][{k}]['league_pct']", "i")], [g_w])
    if "Caught" in ix and W[ix["Caught"]]["pct"] >= W[ix["Caught"]]["league_pct"] + 5:
        k = ix["Caught"]
        o.add("wicket_types", "BOWL_WT_CAUGHT", "{0}% of his wickets are caught, against {1}% for all bowlers.",
              [(f"P['wicket_types'][{k}]['pct']", "i"), (f"P['wicket_types'][{k}]['league_pct']", "i")], [g_w])
    V = P["deliveries"]; Dv = "P['deliveries']"
    qd = [i for i, x in enumerate(V) if x["balls"] >= MINS["delivery"] and x["usage_pct"] >= 5]
    g_d = lambda i: ("delivery", f"{Dv}[{i}]['balls']", MINS["delivery"])
    good = [i for i in qd if V[i]["par_econ"] - V[i]["econ"] >= GAP_ECON]
    if good:
        i = max(good, key=lambda i: (V[i]["par_econ"] - V[i]["econ"], -i))
        o.add("deliveries", "BOWL_DEL_BEST", "His {t} goes for {0} an over against a par of {1}.",
              [(f"{Dv}[{i}]['econ']", "2"), (f"{Dv}[{i}]['par_econ']", "2")], [g_d(i)], t=DEL_LABEL[V[i]["type"]])
    bad = [i for i in qd if V[i]["econ"] - V[i]["par_econ"] >= GAP_ECON]
    if bad:
        i = max(bad, key=lambda i: (V[i]["econ"] - V[i]["par_econ"], -i))
        o.add("deliveries", "BOWL_DEL_WORST", "His {t} costs {0} an over against a par of {1}: the ball batters get after.",
              [(f"{Dv}[{i}]['econ']", "2"), (f"{Dv}[{i}]['par_econ']", "2")], [g_d(i)], t=DEL_LABEL[V[i]["type"]])
    H = P["hands"]
    if all(h["balls"] >= MINS["hand"] for h in H) and abs(H[0]["econ"] - H[1]["econ"]) >= 0.5:
        o.add("hands", "BOWL_HANDS", "Right-handers score {0} an over against him, left-handers {1} (par {2} and {3}).",
              [("P['hands'][0]['econ']", "2"), ("P['hands'][1]['econ']", "2"), ("P['hands'][0]['par_econ']", "2"), ("P['hands'][1]['par_econ']", "2")],
              [("hand", "P['hands'][0]['balls']", MINS["hand"]), ("hand", "P['hands'][1]['balls']", MINS["hand"])])
    if P["owns"] and P["owns"][0]["outs"] >= 2:
        t = P["owns"][0]
        o.add("opponents", "BOWL_OPP_MOST_OUTS", "{n}: {0} off {1} against him, out {2}.",
              [("P['owns'][0]['runs']", "i"), ("P['owns'][0]['balls']", "i"), ("P['owns'][0]['outs']", "i")],
              [("opponent", "P['owns'][0]['balls']", MINS["opponent"])], n=t["name"])
    if P["troubled_by"]:
        o.add("opponents", "BOWL_OPP_HIGHEST_SR", "{n}: {0} off {1} against him, out {2}.",
              [("P['troubled_by'][0]['runs']", "i"), ("P['troubled_by'][0]['balls']", "i"), ("P['troubled_by'][0]['outs']", "i")],
              [("opponent", "P['troubled_by'][0]['balls']", MINS["opponent"])], n=P["troubled_by"][0]["name"])
    _grounds(o, P, "econ", "par_econ", "", " an over", MINS["ground_bowler"], higher_is_good=False)
    # planner: which overs of the computed best plan do most of the saving
    bp = P["planner"]["best"]["overs"]
    top2 = sorted(bp, key=lambda n: (-ov[n - 1]["saving"], n))[:2]
    a, b = sorted(top2)
    o.add("planner", "PLAN_BEST", "Overs {0} and {1} do most of the work in the best plan: {2} of the {3} runs it saves.",
          [(f"{O}[{a - 1}]['over']", "i"), (f"{O}[{b - 1}]['over']", "i"), (f"{O}[{a - 1}]['saving']+{O}[{b - 1}]['saving']", "1"),
           ("P['planner']['best']['saving']", "1")], [g_ov(a - 1), g_ov(b - 1)])
    return o.items


def _grounds(o, P, val, par, pre, post, mn, higher_is_good):
    G = P["grounds"]; kind = "ground_batter" if higher_is_good else "ground_bowler"
    f = "i" if higher_is_good else "2"
    for i, g in enumerate(G):
        if DIGIT.search(g["ground"]):
            continue
        gap = GAP_SR if higher_is_good else GAP_ECON
        better = g["vs_par"] >= gap if higher_is_good else g["vs_par"] <= -gap
        worse = g["vs_par"] <= -gap if higher_is_good else g["vs_par"] >= gap
        if g["tag"] == "best" and i == 0 and better:
            o.add("grounds", "GROUND_BEST", "Best ground against par: {g}, " + pre + "{0}" + post + " against {1} from {2} balls.",
                  [(f"P['grounds'][{i}]['{val}']", f), (f"P['grounds'][{i}]['{par}']", f), (f"P['grounds'][{i}]['balls']", "i")],
                  [(kind, f"P['grounds'][{i}]['balls']", mn)], g=g["ground"])
        if g["tag"] == "worst" and worse:
            nums = [(f"P['grounds'][{i}]['{val}']", f), (f"P['grounds'][{i}]['{par}']", f), (f"P['grounds'][{i}]['balls']", "i")]
            pre = pre.replace("a strike rate of ", "strike rate ")
            if g["balls"] >= MINS["advice"]:      # advice only on a big sample
                o.add("grounds", "GROUND_WORST_ADVICE", "Watch {g}: " + pre + "{0}" + post + ", par {1}, {2} balls.", nums,
                      [(kind, f"P['grounds'][{i}]['balls']", mn), ("advice", f"P['grounds'][{i}]['balls']", MINS["advice"])], g=g["ground"])
            else:
                o.add("grounds", "GROUND_WORST", "Weakest ground against par: {g}, " + pre + "{0}" + post + ", par {1}, {2} balls.", nums,
                      [(kind, f"P['grounds'][{i}]['balls']", mn)], g=g["ground"])


# ------------------------------------------------------------------------------------------------ role label + flags
ROLE_BAT = {"Powerplay": "Top order", "Middle": "Middle order", "Death": "Finisher"}
ROLE_BOWL = {"Powerplay": "New ball", "Middle": "Middle overs", "Death": "Death"}


def role_label(P):
    pb = P["phase_balls"]; tot = sum(pb.values()) or 1
    if P["role"] == "batter":
        return ROLE_BAT[max(pb, key=lambda k: (pb[k], k))]
    top = [k for k in sorted(pb, key=lambda k: -pb[k]) if pb[k] / tot >= 0.25][:2]
    return " and ".join(ROLE_BOWL[k] for k in top).capitalize()


def flags(P):
    """A 'few balls' flag per section: the player's total behind the section is under the section's minimum."""
    bat = P["role"] == "batter"
    F = {"wheel": (P["region_totals"]["balls"] if bat else sum(r["balls"] for r in P["regions"]), MINS["region"], "balls in the field regions"),
         "overs": (max(o["balls"] for o in P["overs"]), MINS["over"], "balls in his busiest over"),
         "grounds": (P["max_ground_balls"], MINS["ground_batter" if bat else "ground_bowler"], "balls at his most-played ground"),
         "opponents": (P["max_opp_balls"], MINS["opponent"], "balls against his most-faced opponent")}
    if bat:
        F.update({"field": F["wheel"], "styles": (max(s["balls"] for s in P["styles"]), MINS["style"], "balls against his most-faced type"),
                  "dismissals": (P["outs"], MINS["outs"], "dismissals")})
    else:
        F.update({"planner": F["overs"], "wicket_types": (P["wickets"], MINS["wickets"], "wickets"),
                  "deliveries": (max(v["balls"] for v in P["deliveries"]), MINS["delivery"], "balls of his most-used type"),
                  "hands": (min(h["balls"] for h in P["hands"]), MINS["hand"], "balls to the less-faced hand")})
    return {k: {"few": v[0] < v[1], "total": v[0], "min": v[1], "what": v[2]} for k, v in F.items()}


def finish_dive(P):
    P["mins"] = {k: v for k, v in MINS.items() if not k.startswith("ground_")} | {"ground": MINS["ground_batter" if P["role"] == "batter" else "ground_bowler"]}
    P["role_label"] = role_label(P)
    P["flags"] = flags(P)
    P["callouts"] = batter_callouts(P) if P["role"] == "batter" else bowler_callouts(P)
