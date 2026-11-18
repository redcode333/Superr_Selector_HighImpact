"""Phase 9 driver - build the v3 package twice and check that the two runs are identical; then save the reference.

    python superr/src/phase9_v3_build.py twice

v3 is a deterministic function of frozen outputs (out/model_v2, reports/final_v2, reports/phase5_v2, the Phase 3 ball values), so the
check runs the phase itself twice from clean and compares every file (same comparison rules as phase8_v2_build.py: byte-identical,
except SVG build dates / clip ids, 'Runtime N s' lines, and CSV numbers within rtol 1e-9). Parquet files are compared by content.
"""
import os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); SUPERR = os.path.dirname(HERE); PROJECT = os.path.dirname(SUPERR)
sys.path.insert(0, HERE)
from phase8_v2_build import compare  # noqa: E402
import pandas as pd  # noqa: E402

PY = sys.executable
DIRS = [os.path.join(SUPERR, "out", "model_v3"), os.path.join(SUPERR, "reports", "final_v3")]
SCRATCH = os.environ.get("PHASE9_SCRATCH", os.path.join(PROJECT, "_phase9_scratch"))


def run():
    return subprocess.run([PY, os.path.join(PROJECT, "run_all.py"), "--floor", "50", "--pressure-positive", "--clean"], cwd=PROJECT,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"}, stdout=subprocess.DEVNULL).returncode


def parquet_same(a, b):
    x, y = pd.read_parquet(a), pd.read_parquet(b)
    return x.equals(y)


def main():
    snap = os.path.join(SCRATCH, "run1"); shutil.rmtree(SCRATCH, ignore_errors=True)
    if run(): sys.exit("STOP: run 1 failed")
    for d in DIRS: shutil.copytree(d, os.path.join(snap, os.path.basename(d)), ignore=shutil.ignore_patterns("reference"))
    if run(): sys.exit("STOP: run 2 failed")
    total, bad, exact = 0, [], 0
    for d in DIRS:
        a = os.path.join(snap, os.path.basename(d))
        n, b, ex = compare(a, d); total += n; exact += ex
        for k in list(b):
            if k.endswith(".parquet") and parquet_same(os.path.join(a, k), os.path.join(d, k)): b.remove(k)
        bad += [os.path.join(os.path.basename(d), x) for x in b]
    print(f"RUN1 vs RUN2: {total - len(bad)}/{total} files identical ({exact} byte-identical)" + (f"; DIFFER: {bad}" if bad else ""))
    if bad: sys.exit("STOP: the two runs differ - reference NOT saved")
    fin = DIRS[1]; ref = os.path.join(fin, "reference"); os.makedirs(ref, exist_ok=True)
    for f in ("key_numbers.json", "final_top10.csv"): shutil.copy2(os.path.join(fin, f), os.path.join(ref, f))
    print("Reference saved:", ref); shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    main()
