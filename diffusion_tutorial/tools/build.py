"""Convert the plain-Python notebook sources in src/ into .ipynb files and (optionally) run them.

    python tools/build.py                 # convert all src/*.py -> *.ipynb (no execution)
    python tools/build.py --execute       # convert + execute every notebook in order
    python tools/build.py 03 05 --execute # only notebooks whose name starts with 03 or 05

Set DM_FAST=1 in the environment for a quick smoke test of every notebook.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()
    srcs = sorted((ROOT / "src").glob("[0-9][0-9]_*.py"))
    if args.only:
        srcs = [s for s in srcs if any(s.name.startswith(o) for o in args.only)]
    for src in srcs:
        nb = ROOT / (src.stem + ".ipynb")
        subprocess.run([sys.executable, "-m", "jupytext", "--to", "ipynb", "--output", str(nb), str(src)],
                       check=True, cwd=ROOT)
        if args.execute:
            t0 = time.time()
            print(f"executing {nb.name} ...", flush=True)
            subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                            "--inplace", f"--ExecutePreprocessor.timeout={args.timeout}", str(nb)],
                           check=True, cwd=ROOT)
            print(f"  done in {(time.time() - t0) / 60:.1f} min", flush=True)
