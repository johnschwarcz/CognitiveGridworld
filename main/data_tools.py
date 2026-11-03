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
  python main/data_tools.py stage           # link the archive selection (archive_selection) into main/DATA/_hf_stage
  python main/data_tools.py upload [glob]   # the staged selection (or files matching glob) -> the archive (write token)
  python main/data_tools.py fetch <glob>    # archive -> main/DATA, e.g. 'experimental/training_dynamics/trained_s*/*'
  python main/data_tools.py trace <name> <script.py> [args]  # run it, record the DATA files it reads / writes
  python main/data_tools.py plan <name>     # minimal rebuild for a traced command: what to fetch, what to run first
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
DATA_REPO = os.environ.get("CGW_DATA_REPO", "johnschwarcz/cognitive-gridworld-data")               # e.g. "<hf-user>/cognitive-gridworld-data" (dataset)
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


# what goes to the archive (John 11-03): weights through training only where analyses use them, final weights elsewhere
ARCH_TT_RUNS = {"trained_s20261029", "echo_s20261029", "trained_p1_s20261029", "trained_N50_s20261029",
                "trained_N200_s20261029", *(f"trained_N100_s{s}" for s in range(20261029, 20261034))}
ARCH_DROP = ("experimental/ff_deep/widths/", "lazyrich/")    # the very wide networks (N = 10,000 / 20,000)
STAGE = os.path.join(DATA, "_hf_stage")                       # symlink tree of the selection (upload_large_folder)


def archive_selection():
    """the archive files to upload (paths relative to main/DATA): every archive-tier file, except: the dropped
    folders; in training-run folders outside ARCH_TT_RUNS only the last weight checkpoint (ckpt_ep*_net.pth); in every
    run folder only the last optimizer state (adam_ep*.pth); plus the runs' top-level logs (<run>.log)"""
    import csv
    rows = list(csv.DictReader(open(os.path.join(ROOT, "main", "DATA_MANIFEST.tsv")), delimiter="\t"))
    last = {}
    for r in rows:
        m = re.match(r"(.*_s\d{8})/(ckpt_ep|adam_ep)(\d+)", r["path"])
        if m:
            k = (m.group(1), m.group(2))
            last[k] = max(last.get(k, -1), int(m.group(3)))
    out = []
    for r in rows:
        p = r["path"]
        if r["tier"] != "archive" and not re.search(r"_s\d{8}\.log$", p):
            continue
        if any(p.startswith(d) for d in ARCH_DROP):
            continue
        m = re.match(r"(.*_s\d{8})/(ckpt_ep|adam_ep)(\d+)", p)
        if m:
            run = os.path.basename(m.group(1))
            keep_all = m.group(2) == "ckpt_ep" and run in ARCH_TT_RUNS
            if not keep_all and int(m.group(3)) != last[(m.group(1), m.group(2))]:
                continue
        out.append((p, int(r["bytes"])))
    return out


def stage():
    """main/DATA/_hf_stage: symlinks to the selection at their main/DATA paths"""
    sel = archive_selection()
    n = 0
    for p, _ in sel:
        dst = os.path.join(STAGE, p)
        if not os.path.lexists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(os.path.join(DATA, p), dst)
            n += 1
    print(f"selection: {len(sel):,} files, {sum(b for _, b in sel) / 1e9:.1f} GB; {n} new links in {STAGE}")


def _repo():
    assert DATA_REPO, "set CGW_DATA_REPO=<hf-user>/<dataset-name> (a Hugging Face dataset repo)"
    return DATA_REPO


def upload(pattern=None):
    """the archive selection (default) or the archive-tier files matching pattern -> the archive (a private Hugging
    Face dataset repo) at their main/DATA paths, in commits of <= ~3 GB / 300 files with retries; files already in the
    repo are skipped, so a rerun resumes"""
    import time
    import fnmatch
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    api.create_repo(_repo(), repo_type="dataset", private=True, exist_ok=True)
    have = set(api.list_repo_files(_repo(), repo_type="dataset"))
    sel = archive_selection()
    if pattern:
        sel = [(p, b) for p, b in sel if fnmatch.fnmatch(p, pattern)]
    todo = [(p, b) for p, b in sel if p not in have]
    batches, cur, size = [], [], 0
    for p, b in sorted(todo):
        if cur and (size + b > 3e9 or len(cur) >= 300 or os.path.dirname(p) != os.path.dirname(cur[-1][0])):
            batches.append(cur)
            cur, size = [], 0
        cur.append((p, b))
        size += b
    if cur:
        batches.append(cur)
    tot = sum(b for _, b in todo)
    print(f"{len(sel):,} selected, {len(sel) - len(todo):,} already in {_repo()}; uploading {len(todo):,} files "
          f"({tot / 1e9:.1f} GB) in {len(batches)} commits", flush=True)
    t0, done = time.time(), 0
    for i, batch in enumerate(batches):
        gb = sum(b for _, b in batch) / 1e9
        for attempt in range(6):
            try:
                api.create_commit(repo_id=_repo(), repo_type="dataset", commit_message=f"archive: {os.path.dirname(batch[0][0])} "
                                  f"({len(batch)} files, {gb:.2f} GB)",
                                  operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=os.path.join(DATA, p))
                                              for p, _ in batch])
                break
            except Exception as e:                                                      # noqa: BLE001
                print(f"  batch {i + 1} attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            print(f"  batch {i + 1} gave up; rerun upload to resume", flush=True)
            continue
        done += gb * 1e9
        el = time.time() - t0
        print(f"{time.strftime('%H:%M:%S')} {i + 1}/{len(batches)} {os.path.dirname(batch[0][0])}: {gb:.2f} GB; "
              f"total {done / 1e9:.1f}/{tot / 1e9:.1f} GB, {done / 1e6 / max(el, 1):.0f} MB/s, ETA "
              f"{(tot - done) / max(done / max(el, 1), 1) / 60:.0f} min", flush=True)
    print("done", flush=True)


def fetch(pattern):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=_repo(), repo_type="dataset", local_dir=DATA, allow_patterns=[pattern])


DEPS = os.path.join(ROOT, "main", "deps")                      # one JSON per traced command (trace / plan)


def trace(name, script, *args):
    """run `script args` (a python file, as __main__) and record the main/DATA files it reads and writes -> main/deps/
    <name>.json; the exact inputs of that command, the base of `plan`"""
    import json
    import time
    import runpy
    reads, writes = set(), set()

    def hook(event, a):
        if event == "open" and isinstance(a[0], (str, bytes, os.PathLike)):
            p = os.path.abspath(os.fsdecode(a[0]))
            if p.startswith(DATA + os.sep):
                mode = a[1] if len(a) > 1 and isinstance(a[1], str) else "r"
                (writes if any(c in mode for c in "wax+") else reads).add(os.path.relpath(p, DATA))
        elif event == "os.rename" and isinstance(a[1], (str, bytes, os.PathLike)):
            p = os.path.abspath(os.fsdecode(a[1]))
            if p.startswith(DATA + os.sep):
                writes.add(os.path.relpath(p, DATA))
    sys.addaudithook(hook)
    t0 = time.time()
    sys.argv = [script, *args]
    runpy.run_path(script, run_name="__main__")
    reads -= writes
    rec = dict(name=name, command=" ".join([os.path.relpath(script, ROOT), *args]), seconds=round(time.time() - t0, 1),
               cuda=os.environ.get("CUDA_VISIBLE_DEVICES", ""), reads=sorted(reads),
               writes=sorted(w for w in writes if not w.endswith(".tmp.npz")))
    os.makedirs(DEPS, exist_ok=True)
    with open(os.path.join(DEPS, f"{name}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(f"trace {name}: {len(rec['reads'])} files read, {len(rec['writes'])} written, {rec['seconds']} s")


def plan(name):
    """the minimal way to rebuild what the traced command `name` needs: its inputs that are archive files (fetch), light
    files (in git: light-restore), caches written by another trace (run that trace's command first, recursively), and
    caches no trace writes yet (trace their producer, see the module docstrings)"""
    import json
    T = {os.path.splitext(f)[0]: json.load(open(os.path.join(DEPS, f))) for f in os.listdir(DEPS) if f.endswith(".json")}
    light = set(open(os.path.join(LIGHT, "LIGHT_FILES.txt")).read().split()) if os.path.isdir(LIGHT) else set()
    writer = {w: n for n, t in T.items() for w in t["writes"]}
    # per-checkpoint families: a traced writer of .../trained_49995.npz also writes .../trained_00025.npz with the
    # checkpoint number substituted in its command and inputs (digit runs of the written path -> a pattern)
    pats = [(re.compile("^" + re.sub(r"\\d+|\d+", r"(\\d+)", re.escape(w)) + "$"), re.findall(r"\d+", w), n)
            for n, t in T.items() for w in t["writes"]]
    order, fetch, unknown, lt, fam = [], set(), set(), set(), {}

    def classify(r, n):
        if r.endswith(CKPT) or RUN_DIR.search(r) or r in ARCHIVE_FILES:
            fetch.add(r)
        elif r in light:
            lt.add(r)
        elif r in writer and writer[r] != n:
            visit(writer[r])
        else:
            for rx, old, src in pats:
                m = rx.match(r)
                if m and src != n:
                    sub = {a: b for a, b in zip(old, m.groups()) if a != b}
                    fam.setdefault(src, set()).add(tuple(sorted(sub.items())))
                    for x in T[src]["reads"]:
                        for a, b in sub.items():
                            x = x.replace(a, b)
                        if x.endswith(CKPT) or RUN_DIR.search(x) or x in ARCHIVE_FILES:
                            fetch.add(x)
                    if src not in order:
                        visit(src)
                    return
            unknown.add(r)

    def visit(n):
        if n in order:
            return
        for r in T[n]["reads"]:
            classify(r, n)
        order.append(n)
    visit(name)
    size = sum(os.path.getsize(os.path.join(DATA, f)) for f in fetch if os.path.exists(os.path.join(DATA, f)))
    print(f"plan for {name}:")
    print(f"  1. fetch {len(fetch)} archive files ({size / 1e9:.2f} GB):")
    for f in sorted(fetch)[:12]:
        print(f"       {f}")
    if len(fetch) > 12:
        print(f"       ... ({len(fetch) - 12} more; patterns in main/deps/{name}.json)")
    print(f"  2. light-restore ({len(lt)} light files used)")
    print("  3. run, in order:")
    for n in order:
        extra = ""
        if n in fam:
            subs = sorted(fam[n])
            vals = [int(v[0][1]) for v in subs if len(v) == 1]
            extra = (f" -- and for {len(subs)} more checkpoints: replace {subs[0][0][0] if subs[0] else ''} by each of "
                     f"{', '.join(str(v) for v in vals[:4])} ... {', '.join(str(v) for v in vals[-2:])} (or the "
                     "module's batch command)")
        print(f"       {T[n]['command']}   (~{T[n]['seconds']:.0f} s each{', GPU' if T[n]['cuda'] else ''}){extra}")
    if unknown:
        print(f"  ! {len(unknown)} cache files read that no trace writes yet (trace their producers):")
        for f in sorted(unknown)[:10]:
            print(f"       {f}")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["manifest"]
    {"manifest": manifest, "light-build": light_build, "light-restore": light_restore,
     "stage": stage, "upload": lambda: upload(*rest), "fetch": lambda: fetch(*rest), "trace": lambda: trace(*rest),
     "plan": lambda: plan(*rest)}[cmd]()
