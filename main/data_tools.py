"""Data handling for the repository: main/DATA (~200 GB) is not in git. Three tiers:

  1. archive (~107 GB): every checkpoint (*.pth / *.pt: 101 GB; the trained networks, saved at ~114 points through
     training for the training-dynamics runs; days of GPU to retrain), the training runs' logs and tests (5.4 GB) and
     the probe set (analysis/probes.npz) -> an external archive (a Hugging Face dataset repo, DATA_REPO below),
     uploaded with `upload`, fetched with `fetch`. Every cache below is a forward pass over these checkpoints;
  2. caches (everything else in main/DATA): regenerated from the checkpoints by the analysis modules; every module's
     docstring names its caches and the command (and runtime) that rebuilds them;
  3. light (in git): main/DATA_light/ mirrors main/DATA for the small figure-level caches (named summary caches the
     scripts read, <= 5 MB each, plus a few recent ones; ~100 MB), so figures redraw without recomputing.
  main/DATA_MANIFEST.tsv lists every file of main/DATA (path, bytes, tier); main/DATA_MANIFEST.md summarises the folders
  (size, files, checkpoints, the modules that mention them).

Commands (run from the repository root with the repo's python, /home/johns/anaconda3/envs/eureka/bin/python):
  python main/data_tools.py manifest        # rewrite DATA_MANIFEST.tsv / .md from main/DATA
  python main/data_tools.py light-build     # refresh main/DATA_light from main/DATA (the curated list)
  python main/data_tools.py light-restore   # copy main/DATA_light into main/DATA (missing files only)
  python main/data_tools.py upload [glob]   # checkpoints (or files matching glob) -> the archive (needs a write token)
  python main/data_tools.py fetch <glob>    # archive -> main/DATA, e.g. 'experimental/training_dynamics/trained_s*/*'
For another agent: clone the repo, run light-restore for the figures, fetch the checkpoints a module needs, then follow
that module's docstring to regenerate its caches."""
import os
import re
import sys
import glob
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "main", "DATA")
LIGHT = os.path.join(ROOT, "main", "DATA_light")
CODE = [os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
DATA_REPO = os.environ.get("CGW_DATA_REPO", "")               # e.g. "<hf-user>/cognitive-gridworld-data" (dataset)
LIGHT_CAP = 5_000_000                                          # bytes per file for the named summary caches
LIGHT_EXTRA = ("experimental/training_dynamics/through_training/chaos/pairs_ratio.npz",
               "experimental/training_dynamics/through_training/flips/ratio/*.npz")
CKPT = (".pth", ".pt")


def _files():
    for d, _, fs in os.walk(DATA):
        for f in fs:
            p = os.path.join(d, f)
            if os.path.isfile(p):
                yield os.path.relpath(p, DATA), os.path.getsize(p)


def _code_text():
    txt = []
    for d in CODE:
        for f in glob.glob(os.path.join(d, "*.py")):
            with open(f, errors="ignore") as fh:
                txt.append((os.path.relpath(f, ROOT), fh.read()))
    return txt


def light_list():
    """the curated light files (paths relative to main/DATA): summary caches named literally in the analysis scripts
    (not per-checkpoint ck_*), <= LIGHT_CAP bytes, plus LIGHT_EXTRA"""
    names = set()
    for _, t in _code_text():
        names.update(re.findall(r"[\"']([A-Za-z0-9_.\-]+\.(?:npz|json|tsv|csv))[\"']", t))
    out = sorted(p for p, s in _files() if os.path.basename(p) in names and s <= LIGHT_CAP and
                 not re.match(r"ck_\d+\.npz$", os.path.basename(p)))
    for g in LIGHT_EXTRA:
        out += sorted(os.path.relpath(p, DATA) for p in glob.glob(os.path.join(DATA, g)))
    return sorted(set(out))


RUN_DIR = re.compile(r"(^|/)[A-Za-z0-9_]+_s\d{8}/")               # a training run's folder (weights, logs, tests)
ARCHIVE_FILES = ("experimental/training_dynamics/analysis/probes.npz",)   # the probe set every analysis reads
ARCHIVE_PATTERNS = ["**/*.pth", "**/*.pt", "**/*_s[0-9]*/**", *ARCHIVE_FILES]


def tier(path, light):
    """archive: checkpoints, everything in a training run's folder, the probe set; light: in git; cache: regenerable"""
    if path.endswith(CKPT) or RUN_DIR.search(path) or path in ARCHIVE_FILES:
        return "archive"
    return "light" if path in light else "cache"


def manifest():
    light = set(light_list())
    rows = sorted(_files())
    with open(os.path.join(ROOT, "main", "DATA_MANIFEST.tsv"), "w") as f:
        f.write("path\tbytes\ttier\n")
        for p, s in rows:
            f.write(f"{p}\t{s}\t{tier(p, light)}\n")
    agg = {}
    for p, s in rows:
        parts = p.split("/")
        for depth in range(1, min(4, len(parts))):
            k = "/".join(parts[:depth])
            a = agg.setdefault(k, [0, 0, 0, 0])
            a[0] += s
            a[1] += 1
            a[2] += s if tier(p, light) == "archive" else 0
            a[3] += s if p in light else 0
    code = _code_text()
    lines = ["# main/DATA manifest", "",
             f"{len(rows):,} files, {sum(s for _, s in rows) / 1e9:.1f} GB. Tiers: archive (checkpoints, run logs, the probe set: `python "
             "main/data_tools.py fetch`), cache (regenerate: see the modules below and their docstrings), light (in "
             "git under main/DATA_light: `python main/data_tools.py light-restore`). Per-file list: DATA_MANIFEST.tsv.",
             "", "| folder | size | files | archive | light | mentioned by |", "|---|---|---|---|---|---|"]
    for k in sorted(agg):
        tot, n, ck, lt = agg[k]
        if tot < 10e6 and k.count("/") > 0:
            continue
        leaf = k.split("/")[-1]
        who = sorted({os.path.basename(f) for f, t in code if f'"{leaf}"' in t or f"'{leaf}'" in t or f"/{leaf}" in t})
        lines.append(f"| {k} | {tot / 1e9:.2f} GB | {n:,} | {ck / 1e9:.2f} GB | {lt / 1e6:.1f} MB | "
                     f"{', '.join(who[:6]) + (' ...' if len(who) > 6 else '')} |")
    with open(os.path.join(ROOT, "main", "DATA_MANIFEST.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"manifest: {len(rows):,} files; light {len(light)} files")


def light_build():
    files = light_list()
    if os.path.isdir(LIGHT):
        shutil.rmtree(LIGHT)
    tot = 0
    for p in files:
        dst = os.path.join(LIGHT, p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(DATA, p), dst)
        tot += os.path.getsize(dst)
    with open(os.path.join(LIGHT, "LIGHT_FILES.txt"), "w") as f:
        f.write("\n".join(files) + "\n")
    print(f"light: {len(files)} files, {tot / 1e6:.0f} MB -> {LIGHT}")


def light_restore():
    n = 0
    for d, _, fs in os.walk(LIGHT):
        for f in fs:
            src = os.path.join(d, f)
            rel = os.path.relpath(src, LIGHT)
            if rel == "LIGHT_FILES.txt":
                continue
            dst = os.path.join(DATA, rel)
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                n += 1
    print(f"restored {n} files into {DATA}")


def _repo():
    assert DATA_REPO, "set CGW_DATA_REPO=<hf-user>/<dataset-name> (a Hugging Face dataset repo)"
    return DATA_REPO


def upload(pattern=None):
    """checkpoints (default) or the files matching pattern -> the archive, keeping their main/DATA paths"""
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(_repo(), repo_type="dataset", private=True, exist_ok=True)
    allow = [pattern] if pattern else ARCHIVE_PATTERNS
    api.upload_large_folder(repo_id=_repo(), repo_type="dataset", folder_path=DATA, allow_patterns=allow)


def fetch(pattern):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=_repo(), repo_type="dataset", local_dir=DATA, allow_patterns=[pattern])


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["manifest"]
    {"manifest": manifest, "light-build": light_build, "light-restore": light_restore,
     "upload": lambda: upload(*rest), "fetch": lambda: fetch(*rest)}[cmd]()
