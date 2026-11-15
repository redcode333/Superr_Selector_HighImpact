"""Phase 8 driver - build the floor-50 package twice from clean and check reproducibility; optional v1 regression check.

    python superr/src/phase8_v2_build.py v1check [--keep]  # floor 0 into *_v1check folders, compared with v1 (then removed)
    python superr/src/phase8_v2_build.py twice             # run_all --floor 50 --clean twice; assert identical; save reference

"Identical" = byte-identical, with three documented exceptions that carry no results:
  - SVG figures: matplotlib's build-date line and its random clip-path ids;
  - .md reports: "Runtime N s" lines;
  - CSV numbers: equal within rtol 1e-9 (floating-point summation-order noise; observed max 6.6e-23 on values ~1e-7).
"""
import os, re, shutil, subprocess, sys, time, filecmp

HERE = os.path.dirname(os.path.abspath(__file__))
SUPERR = os.path.dirname(HERE); PROJECT = os.path.dirname(SUPERR)
sys.path.insert(0, PROJECT)
from run_all import STEPS, variant_dirs, FLOOR_FIRST_STEP  # noqa: E402

PY = sys.executable
SCRATCH = os.environ.get("PHASE8_SCRATCH", os.path.join(PROJECT, "_phase8_scratch"))
NL = chr(10)
SVG_ID = re.compile(r"(?<![0-9A-Za-z])[pm][0-9a-f]{10}(?![0-9a-f])")
RUNTIME = re.compile(r"[Rr]untime [0-9]+ s")


def files_under(d):
    out = {}
    for root, _, fs in os.walk(d):
        if "reference" in os.path.relpath(root, d).split(os.sep):
            continue
        for f in fs:
            p = os.path.join(root, f); out[os.path.relpath(p, d)] = p
    return out


def lines(p):
    return open(p, encoding="utf-8").read().splitlines()


def same(a, b):
    if filecmp.cmp(a, b, shallow=False):
        return True
    if a.endswith(".svg"):
        norm = lambda p: SVG_ID.sub("pID", NL.join(l for l in lines(p) if "<dc:date>" not in l))
        return norm(a) == norm(b)
    if a.endswith(".md"):
        norm = lambda p: [l for l in lines(p) if not RUNTIME.search(l)]
        return norm(a) == norm(b)
    if a.endswith(".csv"):
        import numpy as np, pandas as pd
        x, y = pd.read_csv(a), pd.read_csv(b)
        if list(x.columns) != list(y.columns) or x.shape != y.shape:
            return False
        for c in x.columns:
            if x[c].dtype.kind in "fi" and y[c].dtype.kind in "fi":
                if not np.allclose(x[c].values, y[c].values, rtol=1e-9, atol=1e-15, equal_nan=True):
                    return False
            elif not x[c].equals(y[c]):
                return False
        return True
    return False


def compare(dir_a, dir_b, only_a=False):
    fa, fb = files_under(dir_a), files_under(dir_b)
    keys = sorted(fa) if only_a else sorted(set(fa) | set(fb))
    bad = [k for k in keys if k not in fa or k not in fb or not same(fa[k], fb[k])]
    exact = sum(1 for k in keys if k in fa and k in fb and filecmp.cmp(fa[k], fb[k], shallow=False))
    return len(keys), bad, exact


def run_steps(env):
    first = [s for s, _ in STEPS].index(FLOOR_FIRST_STEP)
    for name, cmd in STEPS[first:]:
        t = time.time(); print(f"  [{time.strftime('%H:%M:%S')}] {name} ...", flush=True)
        r = subprocess.run([PY] + cmd, cwd=SUPERR, env=env, stdout=subprocess.DEVNULL)
        if r.returncode:
            sys.exit(f"STOPPED: {name} failed ({r.returncode})")
        print(f"      {time.time() - t:.0f}s", flush=True)


def report(label, pairs):
    total, bad_all, exact = 0, [], 0
    for a, b, only_a in pairs:
        n, bad, ex = compare(a, b, only_a)
        total += n; exact += ex; bad_all += [os.path.join(os.path.basename(a), x) for x in bad]
    print(f"{label}: {total - len(bad_all)}/{total} files identical ({exact} byte-identical, {total - len(bad_all) - exact} "
          f"identical after the documented normalisation)" + (f"; DIFFER: {bad_all}" if bad_all else ""), flush=True)
    return bad_all


def v1check():
    sfx = "_v1check"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "SUPERR_FLOOR": "0", "SUPERR_SUFFIX": sfx}
    if "--reuse" not in sys.argv:
        for d in variant_dirs(sfx):
            shutil.rmtree(d, ignore_errors=True)
        run_steps(env)
    bad = report("V1 REGRESSION (floor 0 vs frozen v1 files)", [(d, d[: -len(sfx)], True) for d in variant_dirs(sfx)])
    if not bad or "--keep" not in sys.argv:
        for d in variant_dirs(sfx):
            shutil.rmtree(d, ignore_errors=True)
    return not bad


def twice():
    run = lambda: subprocess.run([PY, os.path.join(PROJECT, "run_all.py"), "--floor", "50", "--clean"], cwd=PROJECT,
                                 env={**os.environ, "PYTHONIOENCODING": "utf-8"}, stdout=subprocess.DEVNULL).returncode
    snap = os.path.join(SCRATCH, "run1")
    shutil.rmtree(snap, ignore_errors=True)
    print("v2 run 1 ...", flush=True)
    if run(): sys.exit("STOP: run 1 failed")
    for d in variant_dirs("_v2"):
        shutil.copytree(d, os.path.join(snap, os.path.basename(d)), ignore=shutil.ignore_patterns("reference"))
    print("v2 run 2 ...", flush=True)
    if run(): sys.exit("STOP: run 2 failed")
    bad = report("RUN1 vs RUN2", [(os.path.join(snap, os.path.basename(d)), d, False) for d in variant_dirs("_v2")])
    if bad:
        sys.exit("STOP: the two runs differ - reference NOT saved")
    fin = os.path.join(SUPERR, "reports", "final_v2"); ref = os.path.join(fin, "reference")
    os.makedirs(ref, exist_ok=True)
    for f in ("key_numbers.json", "final_top10.csv"):
        shutil.copy2(os.path.join(fin, f), os.path.join(ref, f))
    print(f"Reference saved: {ref}", flush=True)
    shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "v1check":
        sys.exit(0 if v1check() else 1)
    twice()
