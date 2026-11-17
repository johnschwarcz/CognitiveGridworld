"""A new sensor, learned by predicting its observations or through classification (trainings/new_sensor.py; John 11-11).
Channel 4 of the trained Generator-route network (gen60k_lr4_s0) is redrawn at random; only its embedding slices learn,
through the network's own Generator (generative) or a channel-wise evidence reader (discriminative), both frozen and both
reading every channel with one shared function; values = true labels (solid) or no labels (dashed: sampled from the Bayes
posterior given channels 0-3). a: how well the learned channel-4 code copies the true one (linear R^2); b: accuracy gained
by adding channel 4 (each learner's own read-out: Bayes with the Generator's likelihood / the reader); 1,000 novel
episodes per test (every 200 steps of 1,000), 2 seeds (thin), mean (thick).

RESULT 11-11 20:00 (proof of concept, cluster 66669441; channel 4 starts as a 50/50 mix of its trained and random codes,
lr 1e-4, 2,000 steps): every learner recovers the new sensor; code R^2 .53 -> .98-.99, accuracy gain +.04-.05 at the end;
predicting the observations ~2x faster (R^2 at 400k episodes .92-.95 vs classification .79-.84). My prediction that
classification without labels would learn to ignore the sensor was WRONG: a frozen reader cannot switch a channel off.

    python fig_new_sensor.py [--pull]
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "new_sensor")
SUFFIX = "_corrupt50"                 # 11-11 19:50 proof of concept: channel 4 starts as a 50/50 mix of trained and random codes
LINES = (("generative", "true", "#08519c", "-", "predict its observations, true values"),
         ("generative", "none", "#08519c", (0, (4, 2)), "predict its observations, no labels"),
         ("discriminative", "true", "#a50f15", "-", "classification, true values"),
         ("discriminative", "none", "#a50f15", (0, (4, 2)), "classification, no labels (soft targets)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        os.makedirs(LOGS, exist_ok=True)
        subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/new_sensor/*.jsonl' {LOGS}/", shell=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for learner, labels, col, ls, lab in LINES:
        runs = []
        for k in (0, 1):
            f = os.path.join(LOGS, f"{learner}_{labels}_s{k}{SUFFIX}.jsonl")
            if os.path.exists(f):
                runs.append([json.loads(l) for l in open(f) if l.strip()])
        if not runs:
            continue
        n = min(len(r) for r in runs)
        x = [q["episodes"] / 1e6 for q in runs[0][:n]]
        for i, key in enumerate(("code_r2", "gain")):
            val = (lambda q: q["acc_all"] - q["acc_known"]) if key == "gain" else (lambda q: q[key])
            for r in runs:
                ax[i].plot([q["episodes"] / 1e6 for q in r], [val(q) for q in r], color=col, ls=ls, lw=.6, alpha=.4)
            m = np.mean([[val(q) for q in r[:n]] for r in runs], 0)
            ax[i].plot(x, m, color=col, ls=ls, lw=2, label=f"{lab} ({len(runs)})")
            print(f"{lab:45s} {key:8s} end {m[-1]:+.3f} (step {runs[0][n - 1]['step']})")
    ax[0].set(ylabel="new channel's code vs the true one (linear $R^2$)", ylim=(-.05, 1.02))
    ax[1].set(ylabel="accuracy gained by adding the new channel")
    ax[1].axhline(0, color="k", lw=.8)
    for a_, t in zip(ax, ("a   Learning what the new sensor means", "b   Its use for inference")):
        a_.set_xlabel("training episodes (millions; batch 1,000)")
        a_.set_title(t, loc="left", fontsize=fs["title"] - 3)
        a_.spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False, fontsize=fs["legend"] - 3, loc="lower right")
    fig.suptitle("A new sensor in a trained network (Generator route, Fig 13, 60k episodes): only its codes learn (start: a "
                 "50/50 mix of the true and random codes; lr $10^{-4}$), both\nlearners share one function across channels; "
                 "no labels = values sampled from the posterior given the other 4 channels; novel environments, 1,000 "
                 "episodes per test", fontsize=fs["legend"] - 2)
    out = fig_path("fig_new_sensor.png")
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
