"""Lazyrich runs: test accuracy through training, for finished, running and stopped runs.

    python main/REPLICATION/claude_plots/plot_lazyrich.py [filter ...]

Runs are named N{N}[_skip][_frozenJ]_gamma{g} (train_lazyrich.py; _frozenJ: recurrent J frozen at init, dotted). Finished runs come from
main/DATA/lazyrich/<name>_net.pth, their full test-accuracy history. Unfinished runs come from
log_<name>.txt, where every test checkpoint leaves a tqdm line "n/total [...] acc A | joint J |
naive N": "running" if written in the last LIVE_MIN minutes, "stopped" otherwise; stopped logs
with fewer than MIN_CKPTS checkpoints are aborted starts and are skipped. Filters are substrings
every run name must contain (e.g. `skip`, `N20000`, `gamma0.05`); with filters the figure is
saved as lazyrich_curves_<filters>.png; without filters it shows the no-skip runs only (the gamma
sweep) as lazyrich_gamma_curves.png.
"""
import os, re, sys, glob, time, inspect
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

root = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
while not os.path.exists(os.path.join(root, 'main', 'CognitiveGridworld.py')):
    root, prev = os.path.dirname(root), root
    if root == prev:
        raise RuntimeError("project root not found")
sys.path.insert(0, root)
from main.utils import fig_path, apply_fig_style

LOG_DIR = os.path.join(root, 'main', 'DATA', 'lazyrich')
CKPT = 50           # checkpoint_every of the saved runs; the .pth does not record it
LIVE_MIN = 15       # a log written this recently belongs to a running job
MIN_CKPTS = 100     # shorter unfinished logs are aborted starts, not stopped runs
RAMP = LinearSegmentedColormap.from_list("gamma", ["#86b6ef", "#0d366b"])    # light = lazy, dark = rich
NAME = re.compile(r"N(\d+)(_skip)?(_frozenJ)?_gamma(.+)")
LINE = re.compile(r"(\d+)/(\d+) \[[^\]]*acc ([0-9.]+) \| joint ([0-9.]+) \| naive ([0-9.]+)\]")


def read_log(path):
    """(episodes, acc, joint, naive, total, text). A new postfix appears on the tqdm line
    written at each test episode, so the first line carrying it gives that checkpoint's episode."""
    text = open(path, errors = "ignore").read()
    rows, last = [], None
    for n, total, acc, joint, naive in LINE.findall(text):
        if (acc, joint, naive) != last:
            rows.append((int(n), float(acc), float(joint), float(naive)))
            last = (acc, joint, naive)
    ep, acc, joint, naive = map(np.array, zip(*rows)) if rows else ([],) * 4
    return ep, acc, joint, naive, int(total) if rows else 0, text


def run(name, ep, acc, status):
    N, skip, frozen, gamma = NAME.fullmatch(name).groups()
    return dict(name = name, N = int(N), skip = bool(skip), frozen = bool(frozen), gamma = float(gamma), tag = gamma, ep = ep, acc = acc,
                status = status)


def all_runs():
    saved = {}
    for path in glob.glob(os.path.join(LOG_DIR, "N*_net.pth")):
        name = os.path.basename(path)[:-len("_net.pth")]
        acc = np.asarray(torch.load(path, weights_only = False, map_location = "cpu", mmap = True)["test_acc_through_training"], float)[:, -1]
        saved[name] = run(name, np.arange(len(acc)) * CKPT, acc, "")
    runs = list(saved.values())
    for path in glob.glob(os.path.join(LOG_DIR, "log_N*.txt")):
        name = os.path.basename(path)[len("log_"):-len(".txt")]
        if name in saved:
            continue
        ep, acc, _, _, total, text = read_log(path)
        live = time.time() - os.path.getmtime(path) < LIVE_MIN * 60
        if len(ep) < 2 or (not live and len(ep) < MIN_CKPTS):
            continue
        status = "" if "ENV SAVED" in text else f", {'running' if live else 'stopped at'} {ep[-1] / total:.0%}"
        runs.append(run(name, ep, acc, status))
    return runs


def bayes_baselines():
    """Mean (naive, joint) over every test checkpoint in every log: both depend only on the task."""
    logs = [read_log(p) for p in glob.glob(os.path.join(LOG_DIR, "log_*.txt"))]
    return (np.mean(np.concatenate([l[3] for l in logs if len(l[3])])),
            np.mean(np.concatenate([l[2] for l in logs if len(l[2])])))


def main(filters = ()):
    runs = [r for r in all_runs() if all(f in r['name'] for f in filters) and (filters or not r['skip'])]
    runs.sort(key = lambda r: (-r['gamma'], -r['N'], r['skip'], r['frozen']))
    if not runs:
        raise SystemExit(f"no runs matching {list(filters)} in {LOG_DIR}")

    fs = apply_fig_style()
    fig, ax = plt.subplots(figsize = (4.6, 2.9))
    logg = np.log10([r['gamma'] for r in runs])
    shade = (logg - logg.min()) / (np.ptp(logg) or 1)
    for r, s in zip(runs, shade):
        label = rf"$\gamma$ = {r['tag']}, N = {r['N'] // 1000}k{', skip' if r['skip'] else ''}{', J frozen' if r['frozen'] else ''}{r['status']}"
        ax.plot(r['ep'] / 1000, r['acc'], color = RAMP(s), lw = 1.2, ls = ':' if r['frozen'] else '--' if r['skip'] else '-', label = label)
        print(f"{r['name']:<24} {r['status'].strip(', ') or 'done':<16} ep {r['ep'][-1]:>6}  acc {r['acc'][-1]:.3f}")
    naive, joint = bayes_baselines()
    ax.axhline(naive, ls = '--', c = '#8a8984', lw = 1, label = 'naive Bayes')
    ax.axhline(joint, ls = ':', c = '#1f1f1e', lw = 1.2, label = 'joint Bayes')
    ax.xaxis.set_major_locator(MaxNLocator(nbins = 6, integer = True))
    ax.set_xlabel("training episodes (thousands)")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc = "center left", bbox_to_anchor = (1.01, .5), frameon = False, fontsize = fs['tick'])
    fig.tight_layout()
    out = fig_path(f"lazyrich_curves_{'_'.join(filters)}.png" if filters else "lazyrich_gamma_curves.png")
    fig.savefig(out, bbox_inches = "tight", dpi = 200)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1:])
