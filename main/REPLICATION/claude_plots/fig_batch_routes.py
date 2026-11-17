"""Generator route vs Classifier route under the RCC's own feedback at batch 10,000 vs batch 1,000 (John 11-13: "maybe we
should dramatically reduce the batch size? This could emphasize the instability differences"; "use default RCC feedback
from now on"). Same runs and style as fig13_routes.py panel a; per-seed lines, blue = Generator route, red = Classifier
route, x = training iterations (updates; log axis), y = the Classifier's goal accuracy at T = 30 on novel environments.

    a  batch 10,000: rcc60k_lr4_s0-5 (-> rcc120k_lr4 when it exists) / rccclf60k_lr4 -> rccclf120k_lr4_s0-5 (10,000 test
       episodes per point; DATA/experimental/cluster_runs/jsonl)
    b  batch 1,000: rccb1k_lr4_s0-5 / rccclfb1k_lr4_s0-5 (train_cluster_runs.py tasks 240-251, local GPUs; 1,000 test
       episodes per point = the batch; runs/cluster_runs)
Everything else identical (mode RL, gen_zflow, lr 1e-4, entropy .01, |S| 500, d_E 30, one test per 250 iterations); the
same seed = the same environments and initial network at both batch sizes. Batch 1,000 sees 10x fewer episodes per
iteration. Joint / Naive = mean over the drawn tests of the per-test values.
Status markers at each run's last test (John 11-13: "add a marker at the last episode of the running (or queued) runs"):
filled = still training (local: a train_cluster_runs.py process with its task id is alive; cluster: its log was written in
the last RECENT_S seconds, remote mtime fetched with --pull, and has not reached its last episode); open = paused or queued, will continue (its continuation waits in a local launcher queue
in QUEUE_DIR or in a pending cluster job, it was paused by hand with runs/STOP_<name>, or it paused itself on leaving
chance, train_cluster_runs.PAUSE_AT_ESCAPE); no marker = finished. The status time is printed in the figure.

EFFICIENCY (11-14 02:15; John: "what would be the most efficient? Small or large batch runs?"). Millions of episodes,
median [range] n reached/n runs; chance = trailing 5-test mean >= .2, Naive = 3 consecutive tests above that test's Naive:
                            leave chance        pass Naive          5-test mean .60     .65
    batch 10,000 Generator  75 [68-110] 6/6     124 [120-142] 6/6   192 [188-210] 6/6   298 [278-308] 6/6
    batch 10,000 Classifier 270 [175-338] 6/6   486 [440-632] 6/6   765 [545-798] 6/6   872 [668-945] 5/6
    batch 1,000 Generator   61 [46-132] 5/6     88 [69-152] 5/6     119 [118-120] 2/6   (paused before)
    batch 1,000 Classifier  169 [125-215] 9/9   291 [245-323] 5/9   346 1/9             -
(batch 1,000 medians over the runs that got there: the rest are still training, so they are biased low). Throughput per A100:
batch 10,000 as 2 runs per GPU, 60k iterations in ~8.05 h (66656989, elscn-126, before the speed patch) = ~41k episodes/s;
batch 1,000 as 3 runs per GPU (66685959_0 / 66686727_0, patched) = 39-43k episodes/s. So GPU time ~ episodes, and batch 1,000
needs ~1.4-1.7x fewer episodes to pass Naive / reach .60; the speed patch on batch 10,000 (estimated up to ~2x for a lone run,
never measured on the cluster) could cancel that.

    python fig_batch_routes.py [--pull]
"""
import os, re, sys, json, time, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path

LOGS = [os.path.join(ROOT, "runs", "cluster_runs"), os.path.join(ROOT, "main", "DATA", "experimental", "cluster_runs", "jsonl")]
BLUE, RED = "#2c7bb6", "#d7191c"
QUEUE_DIR = os.environ.get("CGW_QUEUE_DIR", "/tmp/claude-1002/-home-johns-Desktop-CognitiveGridworld/"
                           "79a05095-e9ee-5940-869f-ccd41c6362f6/scratchpad/smallbatch")       # local launcher queues
QUEUE_FILES = ("queue_long.txt", "queue_gen_paused.txt")
RECENT_S = 900
STATUS = os.path.join(LOGS[1], "cluster_status.json")      # written by --pull: remote log mtimes + pending cluster tasks
PANELS = [("a", "batch 10,000", ("rcc60k_lr4", "rcc120k_lr4"), ("rccclf60k_lr4", "rccclf120k_lr4"), 10000, range(6)),
          ("b", "batch 1,000", ("rccb1k_lr4", ("rccb1k1mR2_lr4", "rccb1k1mR_lr4", "rccb1k1m_lr4")),
           ("rccclfb1k_lr4", ("rccclfb1k1mR_lr4", "rccclfb1k1m_lr4")), 1000, range(12))]   # newest continuation first


def load(base, ext, k, with_name=False):
    exts = (ext,) if isinstance(ext, str) else (ext or ())
    for n in [f"{x}_s{k}" for x in exts] + [f"{base}_s{k}"]:
        for d in LOGS:
            f = os.path.join(d, n + ".jsonl")
            if os.path.exists(f):
                load.name = n
                rows, last = [], -10
                for r in sorted((json.loads(l.replace("NaN", "null")) for l in open(f) if l.strip()), key=lambda r: r["ep"]):
                    if r["ep"] > 0 and r["ep"] - last > 1:
                        rows.append(r)
                        last = r["ep"]
                return rows
    return []


def runner():
    """TASKS (index -> name) and RESUME (continuation -> source) of train_cluster_runs.py, without running it"""
    f = os.path.join(ROOT, "main", "REPLICATION", "trainings", "train_cluster_runs.py")
    g = {"__file__": f, "__name__": "tcr"}
    exec(compile(open(f).read().split("if __name__")[0], f, "exec"), g)
    return [t[0] for t in g["TASKS"]], g["RESUME"], g.get("PAUSE_AT_ESCAPE", set()), {t[0]: t[1] for t in g["TASKS"]}


def pull_status():
    """cluster: each run log's remote mtime and the task ids of pending jobs (array index -> group in the job's arguments,
    or the index itself for runs.sbatch arrays)"""
    cmd = r"""cd ~/cgw/runs/cluster_runs; echo "now $(date +%s)"; for f in rcc*b1k*_s*.jsonl rcc*120k_lr4_s*.jsonl rcc*perm100k_lr4_s*.jsonl rcc*perm300k_lr4_s*.jsonl rcc*perm100kR_lr4_s*.jsonl; do
      [ -f "$f" ] && echo "log $f $(stat -c %Y $f)"; done
      for j in $(squeue -u $USER -h -t PD -o "%i:%r" | awk -F: '$2 != "JobHeldUser" {print $1}'); do
        echo "pd $j $(scontrol show job ${j%%_*} | grep -o 'Command=.*')"; done"""
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", "elsc-cluster", cmd], capture_output=True, text=True).stdout
    st = dict(time=time.time(), logs={}, pending=[])
    for line in out.splitlines():
        w = line.split()
        if w[:1] == ["now"]:
            st["remote_now"] = int(w[1])
        elif w[:1] == ["log"] and len(w) == 3:
            st["logs"][w[1][:-6]] = int(w[2])
        elif w[:1] == ["pd"] and len(w) >= 3:
            spec, args = w[1], w[3:]
            # 11-14: only our runner's sbatch files (runs*.sbatch): another session's array (e.g. cgw-zscale_[149-252])
            # has indices that are not task ids; held jobs (JobHeldUser) are skipped above
            if not os.path.basename(w[2].split("=", 1)[-1]).startswith("runs"):
                continue
            idx = spec.split("_", 1)[1].strip("[]").split("%")[0] if "_" in spec else ""
            ids = []
            for part in idx.split(","):
                if "-" in part:
                    a, b = part.split("-"); ids += range(int(a), int(b) + 1)
                elif part:
                    ids.append(int(part))
            for i in ids:
                st["pending"] += args[i].split(":") if args and i < len(args) else [str(i)]   # task ids or names
    json.dump(st, open(STATUS, "w"))


def run_status():
    """{run name: 'running' | 'waiting'} for every run that will still train (see the docstring)"""
    names, resume, pause_at_escape, cfgs = runner()
    out = {}
    for line in subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines():
        m = re.search(r"train_cluster_runs\.py --task (\d+)", line)
        if m and int(m.group(1)) < len(names):
            out[names[int(m.group(1))]] = "running"
    st = json.load(open(STATUS)) if os.path.exists(STATUS) else {}
    if st and time.time() - st["time"] < RECENT_S:
        for n, m in st["logs"].items():
            f = os.path.join(LOGS[1], n + ".jsonl")
            last = json.loads(open(f).read().strip().splitlines()[-1].replace("NaN", "null"))["ep"] if os.path.exists(f) else -1
            if st["remote_now"] - m < RECENT_S and n in cfgs and last < cfgs[n]["episodes"] - 1:
                out[n] = "running"
    wait = [names[int(t)] if t.isdigit() and int(t) < len(names) else t for t in st.get("pending", [])] if st else []
    for q in QUEUE_FILES:
        f = os.path.join(QUEUE_DIR, q)
        if os.path.exists(f):
            wait += [names[int(t)] for t in open(f).read().split() if t.isdigit() and int(t) < len(names)]
    stopped = [f[5:] for f in os.listdir(os.path.join(ROOT, "runs")) if f.startswith("STOP_")]
    wait += stopped
    for n in pause_at_escape:
        for d in LOGS:
            f = os.path.join(d, n + ".jsonl")
            if os.path.exists(f):
                r = [json.loads(l.replace("NaN", "null")) for l in open(f) if l.strip()][-3:]
                if len(r) == 3 and all(q["net"] >= .2 for q in r):
                    wait.append(n)
                break
    for n in wait:
        for m in (n, resume.get(n)):
            if m and out.get(m) != "running":
                out[m] = "waiting"
    for n in stopped:      # 11-14: a stop file beats a recent log write (cluster runs pause at their next test; the file is
        out[n] = "waiting"  # mirrored locally, e.g. runs/STOP_rccclfpermb1k1m_lr4_s0, paused 11-14 03:56 IST)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/{{rccclf120k_lr4,rcc120k_lr4,rccb1k1m_lr4,"
                       f"rccclfb1k1m_lr4,rccb1k1mR_lr4,rccclfb1k1mR_lr4,rccb1k1mR2_lr4}}_s*.jsonl' {LOGS[1]}/", shell=True)
        pull_status()
    status = run_status()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "serif", "font.serif": "cmr10", "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7.5, "axes.linewidth": .6})
    fig, axs = plt.subplots(2, 2, figsize=(5.5, 4.6), sharey=True)
    for col, (tag, title, gen, clf, batch, seeds) in enumerate(PANELS):
        jt, nv, n = [], [], {}
        for arm, colr in ((clf, RED), (gen, BLUE)):
            n[colr] = 0
            for k in seeds:
                rs = load(*arm, k)
                if not rs:
                    continue
                n[colr] += 1
                e, y = np.array([r["ep"] for r in rs]), np.array([r["net"] for r in rs])
                stt = status.get(load.name)
                for row, x in enumerate((e, e * batch)):
                    axs[row, col].plot(x, y, color=colr, lw=.7, alpha=.85)
                    if stt:
                        axs[row, col].plot(x[-1], y[-1], "o", ms=3.6, mew=.8, zorder=5,
                                           mfc=colr if stt == "running" else "white", mec="k" if stt == "running" else colr)
                jt += [r["joint"] for r in rs]
                nv += [r["naive"] for r in rs]
                print(f"{tag} {load.name}: last test {e[-1]}, acc {y[-5:].mean():.3f} (last 5), {stt or 'finished'}")
        for row in range(2):
            ax = axs[row, col]
            if jt:
                ax.axhline(np.mean(jt), color="k", ls="--", lw=.8)
                ax.axhline(np.mean(nv), color="k", ls=":", lw=.9)
            ax.set(xscale="log", ylim=(.05, .8))
            ax.set_xlim(*((250, 1.05e6) if row == 0 else (2.5e5, 1.25e9)))
            ax.set_xlabel("training iterations" if row == 0 else "training episodes (iterations x batch)")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axs[0, col].set_title(r"$\mathbf{%s}$   %s (%d + %d seeds)" % (tag, title, n[BLUE], n[RED]), fontsize=8, loc="left")
    for ax in axs[:, 0]:
        ax.set_ylabel("goal accuracy, final step\n(novel environments)")
    h = [Line2D([], [], color=BLUE, lw=1.2), Line2D([], [], color=RED, lw=1.2),
         Line2D([], [], color="k", ls="--", lw=.8), Line2D([], [], color="k", ls=":", lw=.9),
         Line2D([], [], color=".4", ls="", marker="o", ms=3.6, mec="k"),
         Line2D([], [], color="white", ls="", marker="o", ms=3.6, mec=".3")]
    fig.legend(h, ["Generator route", "Classifier route", "Joint Bayes", "Naive Bayes", "still training",
                   "paused or queued (will continue)"], ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(.5, 1.0))
    fig.text(.5, .872, "the RCC's own feedback, lr $10^{-4}$; each line = one network; 1 test every 250 iterations "
                       "($10^4$ / $10^3$ test episodes)\nmarkers: status on " + time.strftime("%d-%m at %H:%M"),
             ha="center", fontsize=7, linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, .855))
    out = fig_path("fig_batch_routes.png")
    fig.savefig(out, dpi=300)
    print(out)


if __name__ == "__main__":
    main()
