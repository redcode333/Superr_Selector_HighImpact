"""SupeRR Selector - rebuild everything from the raw inputs, in order, and verify against the frozen results.

    python run_all.py                 # full run (foundation -> phase 0 ... phase 5c -> final package) + verification
    python run_all.py --from phase1a_outcome   # resume from a step (earlier outputs must exist)
    python run_all.py --verify-only   # only compare reports/final with the frozen reference
    python run_all.py --floor 50 [--clean]       # v2: exposure-floor eligibility (>= 50 balls with LI >= 1.5); reruns
                                                 # phase4f_final onward only (upstream frozen), outputs in *_v2 folders
    python run_all.py --floor 50 --verify-only   # compare reports/final_v2 with reports/final_v2/reference
    python run_all.py --floor 50 --pressure-positive [--clean]   # v3: post-hoc rule on top of the floor-50 pool (pressure value > 0);
                                                 # reads the frozen v2 outputs only, writes out/model_v3 and reports/final_v3
    python run_all.py --floor 50 --pressure-positive --verify-only   # compare reports/final_v3 with reports/final_v3/reference

Inputs: SupeRR_Data_Pack/ (data pack), data/cricsheet/{ipl,t20s}/ (Cricsheet JSON), superr/out/cache/ (Open-Meteo /
GeoNames responses; fetch_context.py only calls the network for requests missing from the cache).
Seeds are fixed in every script (SEED = 42 and fixed offsets). Each step runs as a separate process; the run stops on
the first error. Verification compares the regenerated reports/final/key_numbers.json and final_top10.csv with the
frozen copies in reports/final/reference/ (created from the current files on the first run if absent).
"""
import json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SUPERR = os.path.join(HERE, "superr")
PACK = os.path.join(HERE, "SupeRR_Data_Pack")
CRICSHEET = os.path.join(HERE, "data", "cricsheet")
FINAL = os.path.join(SUPERR, "reports", "final")
REF = os.path.join(FINAL, "reference")
FLOOR_FIRST_STEP = "phase4f_final"      # the exposure floor only changes eligibility -> everything before is unchanged


def variant_dirs(suffix):
    """Every folder the downstream steps write to (see superr/src/variant.py)."""
    return [os.path.join(SUPERR, "out", "model" + suffix), os.path.join(SUPERR, "reports", "final" + suffix),
            os.path.join(SUPERR, "reports", "phase4" + suffix), os.path.join(SUPERR, "reports", "phase5" + suffix),
            os.path.join(SUPERR, "dash" + suffix)]
PY = sys.executable
V3_STEP = ("phase9_pressure_positive", ["src/phase9_pressure_positive.py"])   # needs the frozen v2 outputs; nothing upstream is rerun

STEPS = [
    ("parse_cricsheet", ["parse_cricsheet.py", CRICSHEET, "out/cricsheet_index.csv"]),
    ("build_foundation_pass1", ["build_foundation.py", PACK]),
    ("match_cricsheet", ["match_cricsheet.py", PACK, "out"]),
    ("align_cricsheet", ["align_cricsheet.py", CRICSHEET, "out"]),
    ("build_foundation_pass2", ["build_foundation.py", PACK]),
    ("build_grounds", ["build_grounds.py", PACK, "out"]),
    ("fetch_context", ["out/fetch_context.py"]),
    ("enrich_context", ["enrich_context.py", "out"]),
] + [(m, [f"src/{m}.py"]) for m in [
    "phase0_history", "phase1a_outcome", "phase1a_fix", "phase1b_wp", "phase1b_smooth", "phase1c_inn1", "phase1f_ball_values",
    "phase2_pressure", "phase3_adjust", "phase4_ratings", "phase4b_ratings", "phase4c_projection", "phase4d_context",
    "phase4e_strict", "phase4f_final", "phase5_validate", "phase5b_tests", "phase5c_checks", "final_package", "phase6b_gap", "phase6c_gap_forward"]]

DERIVED_CACHES = [os.path.join(SUPERR, "out", "model", "phase1b_history_chase_wp_raw.parquet")]


def snapshot_reference():
    if os.path.exists(os.path.join(REF, "key_numbers.json")):
        return False
    os.makedirs(REF, exist_ok=True)
    for f in ("key_numbers.json", "final_top10.csv"):
        shutil.copy2(os.path.join(FINAL, f), os.path.join(REF, f))
    return True


def flatten(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{pre}{k}/"))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flatten(v, f"{pre}{i}/"))
    else:
        out[pre[:-1]] = d
    return out


def verify(rtol=1e-6, atol=1e-9):
    import pandas as pd
    rows = []
    a = pd.read_csv(os.path.join(REF, "final_top10.csv")); b = pd.read_csv(os.path.join(FINAL, "final_top10.csv"))
    for role in ("batter", "bowler"):
        ra = list(a[a.role == role].sort_values("rank (eligible)").player); rb = list(b[b.role == role].sort_values("rank (eligible)").player)
        rows.append((f"top 10 {role}s (names and order)", ra == rb, f"{rb}" if ra != rb else ""))
    ka = flatten(json.load(open(os.path.join(REF, "key_numbers.json")))); kb = flatten(json.load(open(os.path.join(FINAL, "key_numbers.json"))))
    for k in sorted(set(ka) | set(kb)):
        va, vb = ka.get(k, "<missing>"), kb.get(k, "<missing>")
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
            ok = (va != va and vb != vb) or abs(va - vb) <= atol + rtol * abs(va)
        else:
            ok = va == vb
        rows.append((k, ok, "" if ok else f"reference {va} -> new {vb}"))
    n_fail = sum(not ok for _, ok, _ in rows)
    print(f"\nVERIFICATION: {len(rows) - n_fail} PASS, {n_fail} FAIL (numbers: rtol {rtol}, atol {atol})")
    for k, ok, note in rows:
        if not ok or k.startswith("top 10"):
            print(f"  {'PASS' if ok else 'FAIL'}  {k}  {note}")
    with open(os.path.join(FINAL, "verification.txt"), "w", encoding="utf-8") as fh:
        for k, ok, note in rows:
            fh.write(f"{'PASS' if ok else 'FAIL'}\t{k}\t{note}\n")
    return n_fail


def main():
    global FINAL, REF
    args = sys.argv[1:]
    floor = int(args[args.index("--floor") + 1]) if "--floor" in args else 0
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    pp = "--pressure-positive" in args
    if pp and floor != 50:
        sys.exit("--pressure-positive is defined on top of the floor-50 pool: use --floor 50 --pressure-positive")
    if floor:
        suffix = "_v3" if pp else os.environ.get("SUPERR_SUFFIX", "_v2")
        assert suffix, "a floor run must not write to the v1 folders"
        env.update({"SUPERR_FLOOR": str(floor), "SUPERR_SUFFIX": suffix})
        FINAL = os.path.join(SUPERR, "reports", "final" + suffix); REF = os.path.join(FINAL, "reference")
    if "--verify-only" in args:
        sys.exit(1 if verify() else 0)
    start = 0
    steps = [V3_STEP] if pp else STEPS
    if "--from" in args:
        name = args[args.index("--from") + 1]
        start = [s for s, _ in STEPS].index(name)
    elif floor and not pp:
        start = [s for s, _ in STEPS].index(FLOOR_FIRST_STEP)
    if floor and not pp and start < [s for s, _ in STEPS].index(FLOOR_FIRST_STEP):
        sys.exit(f"--floor runs start at {FLOOR_FIRST_STEP} or later (upstream outputs are shared with v1)")
    if floor and "--clean" in args:
        for d in variant_dirs(env["SUPERR_SUFFIX"]):
            if os.path.isdir(d):
                keep = os.path.join(d, "reference")
                for name_ in os.listdir(d):
                    p = os.path.join(d, name_)
                    if p != keep:
                        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        print(f"Cleaned the {env['SUPERR_SUFFIX']} output folders (reference kept).")
    if not floor and snapshot_reference():
        print("Frozen reference created from the current reports/final files.")
    if start == 0:
        for c in DERIVED_CACHES:
            if os.path.exists(c):
                os.remove(c)
    t_all = time.time()
    for name, cmd in steps[start:]:
        t = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {name} ...", flush=True)
        r = subprocess.run([PY] + cmd, cwd=SUPERR, env=env)
        if r.returncode != 0:
            print(f"STOPPED: step '{name}' failed with exit code {r.returncode} after {time.time() - t:.0f}s")
            sys.exit(r.returncode)
        print(f"    done in {time.time() - t:.0f}s", flush=True)
    print(f"All steps done in {(time.time() - t_all) / 60:.1f} min")
    if floor and not os.path.exists(os.path.join(REF, "key_numbers.json")):
        print(f"No reference in {REF} yet - create it after the two-run identity check (not created automatically for floor runs).")
        sys.exit(0)
    sys.exit(1 if verify() else 0)


if __name__ == "__main__":
    main()
