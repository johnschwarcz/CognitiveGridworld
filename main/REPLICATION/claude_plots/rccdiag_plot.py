"""Why the RCC stayed at chance under Fig 13's settings, and which change fixes it (John 11-09/10; every Fig 13 arm at
batch 10,000). The full RCC under its original feedback (train_cluster_runs.py, |S| = 500, the updated Generator), one run
per line:
  Fig 13's settings  rcc30k_S500_s0-2: lr 1e-3, entropy bonus .01 (seed 0 left chance at 3.5k; seeds 1, 2 never)
  entropy .1         rccdiag_ent1_s1/s2: lr 1e-3, entropy bonus .1 (cancelled 11-10 02:30 at 18.75k / 12.75k episodes)
  lr 1e-4            rccdiag_lr4_s*: lr 1e-4, entropy bonus .01 (the Fig 13 RCC arm from 11-10; seed 1 is the stuck seed)
Left: accuracy at T = 30 on novel environments (10,000 episodes per test, one test per 250 training episodes). Right: the
Classifier's answer entropy at T = 30 on the same tests (nats; ln 10 = uniform), logged by the rccdiag runs only.
11-10 02:35: lr4_s1 left chance at ~6k (.16 at 6k, .24 at 7k) and reached .60 at 19k, entropy 2.2 -> .8 as it learned;
ent1_s1 entropy hit 0 at 3k (62% of answers on one value at 2k), then oscillated, .26 at 18.75k; ent1_s2 at chance at 12.75k.
The 'entropy bonus' (model_backward.py RL_loss) adds -b log b of the SAMPLED answer's probability b, whose gradient
-(log b + 1) is positive for b < 1/e: near chance it pushes up whatever was sampled, so a larger bonus favours collapse.

    python rccdiag_plot.py [--pull]
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "cluster_runs", "jsonl")
ARMS = (("rcc30k_S500_s{}", (0, 1, 2), "#999999", "Fig 13's settings (lr $10^{-3}$, entropy .01)"),
        ("rccdiag_ent1_s{}", (1, 2), "#e66101", "entropy bonus .1 (lr $10^{-3}$)"),
        ("rccdiag_lr4_s{}", (1, 0, 2), "#2c7bb6", "lr $10^{-4}$ (entropy .01)"))


def pull():
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/{{rcc30k_S500,rccdiag}}_*.jsonl' {LOGS}/",
                   shell=True)


def load(name):
    f = os.path.join(LOGS, name + ".jsonl")
    if not os.path.exists(f):
        return None
    R = [json.loads(l) for l in open(f)]
    get = lambda k: np.array([r.get(k, np.nan) for r in R], float)
    return get("ep"), get("net"), get("pol_entropy"), R[-1]["joint"], R[-1]["naive"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        pull()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    fig, ax = plt.subplots(1, 2, figsize=(2 * 2.9 + 2.4, 2.9), constrained_layout=True)
    refs = []
    for pat, seeds, col, lab in ARMS:
        first = True
        for k in seeds:
            d = load(pat.format(k))
            if d is None or len(d[0]) < 2:
                continue
            ep, net, ent, j, n = d
            refs.append((j, n))
            stuck = k == 1
            kw = dict(color=col, lw=2.2 if stuck else 1.1, alpha=1 if stuck else .7)
            ax[0].plot(ep, net, label=lab if first else None, **kw)
            if np.isfinite(ent).any():
                ax[1].plot(ep, ent, **kw)
            first = False
    j, n = np.mean(refs, 0)
    for y, t in ((j, "Joint"), (n, "Naive"), (.1, "chance")):
        ax[0].axhline(y, color="k", lw=.9, ls="--" if t != "chance" else ":")
        ax[0].text(3e4, y, t, ha="right", va="bottom", fontsize=fs["legend"] - 2)
    ax[0].set(xlabel="training episodes", ylabel="accuracy, novel environments", xlim=(0, 3e4), ylim=(0, .8))
    ax[0].set_title("Accuracy at T = 30 (thick: stuck seed 1)", fontsize=fs["title"] - 3)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False,
               fontsize=fs["legend"] - 2)
    ax[1].axhline(np.log(10), color="k", lw=.9, ls="--")
    ax[1].text(3e4, np.log(10), "uniform", ha="right", va="bottom", fontsize=fs["legend"] - 2)
    ax[1].set(xlabel="training episodes", ylabel="answer entropy (nats)", xlim=(0, 3e4), ylim=(0, 2.5))
    ax[1].set_title(r"Answer entropy $-\Sigma_r\, b_r \log b_r$ (new runs)", fontsize=fs["title"] - 3)
    for a in ax:
        a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.suptitle(r"RCC under its own feedback, batch 10,000, $|\mathcal{S}| = 500$: one run per line; 10,000 novel episodes "
                 "per test", fontsize=fs["legend"] - 1)
    out = fig_path("rccdiag.png")
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
