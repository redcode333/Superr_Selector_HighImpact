"""Output variant for the downstream steps (phase4f_final onward).

SUPERR_FLOOR  (default 0)  : exposure-floor eligibility rule, high-pressure (LI >= 1.5) balls >= FLOOR.
SUPERR_SUFFIX (default "" when FLOOR == 0, "_v2" otherwise): appended to every downstream output folder
               (out/model -> out/model_v2, reports/final -> reports/final_v2, reports/phase4 -> reports/phase4_v2, ...).
With neither variable set every path and rule is exactly the v1 one, so v1 behaviour is unchanged.
Upstream inputs (phase 0-4e products) are always read from the v1 folders; only files that phase4f+ write are
routed to the variant folder.
"""
import os

FLOOR = int(os.environ.get("SUPERR_FLOOR", "0"))
SUFFIX = os.environ.get("SUPERR_SUFFIX", "" if FLOOR == 0 else "_v2")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_V1 = os.path.join(_ROOT, "out", "model")
DOWNSTREAM_MODEL_FILES = ("phase4_rankings_", "phase5_rerankings_", "README_rankings")


def vdir(path):
    """Variant output folder for a v1 output folder (created when it is a variant folder)."""
    if not SUFFIX:
        return path
    d = path.rstrip("\\/") + SUFFIX
    os.makedirs(d, exist_ok=True)
    return d


def mpath(fname):
    """Path in out/model: downstream products go to the variant folder, upstream inputs stay in v1."""
    if SUFFIX and fname.startswith(DOWNSTREAM_MODEL_FILES):
        return os.path.join(vdir(MODEL_V1), fname)
    return os.path.join(MODEL_V1, fname)
