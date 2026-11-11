"""Generalization vs memorization against |S| and the embedding dimension d_E, for every source of the Generator's answers
(John 11-08/09; heat maps, John 11-09 21:40; all agents over a larger range, John 11-09 22:40): one row per agent, left =
training environments, right = novel environments; x = |S|, y = d_E, colour and number = accuracy at T = 30 at the end of
training. Grey cells have no finished run (not run, or still running: the figure fills in as runs finish).

Rows
  True answers / Joint's answers / Naive's answers / True answer, goal only
      train_bayes_generator.py {truth, joint, naive, truth1}_S{S}_d{d}_s0: the Generator and embeddings trained on that
      source's answers, no LSTM (batch 10,000, lr 1e-3; 8,000 episodes, the first Joint / Naive runs at d_E 30 6,000).
      Accuracy = Bayes inference with the LEARNED likelihood (goal only: Naive-style Bayes with the learned marginal,
      ceiling = Naive Bayes), on the last test: 10,000 episodes of training environments, 10,000 of novel ones.
  RCC, oracle feedback
      train_cluster_runs.py oracle_S{S}_d{d} (oracle mode at Fig 13's settings: batch 10,000, lr 1e-3, 10,000 episodes;
      d_E = KQ_dim, which also sets the network's embedding size). Accuracy = the Classifier's own: novel = the run's last
      test (10,000 episodes), training = the last training batch before it (train_acc_through_training of the saved run,
      pulled as small arrays by ~/cgw/tools/extract_acc.py).
One seed per cell.

RCC with oracle feedback, 11-09 22:15 (all 12 cells; final test, training / novel):
  d_E 10   |S| 100 .65 / .10   250 .30 / .10   500 .14 / .10   1000 .11 / .10
  d_E 30        .71 / .54       .71 / .69       .70 / .69        .70 / .69
  d_E 90        .69 / .26       .67 / .67       .66 / .66        .65 / .64

11-11 11:20, fine grid complete (Generator rows, |S| 25-1000 x d_E 10 / 30 / 90; "both true" row and |S| 150 / 200 not
drawn, John 11-10): novel-environment transfer from |S| 100-125 at d_E 30 (Joint answers .59 / .68), from 250 at d_E 90
(.66; .39 at 125), never at d_E 10 (.08-.11, and training accuracy falls .57 -> .11 with |S|). Naive answers / goal only
follow the same pattern at their plateaus (.48-.52 / .37-.39). Larger d_E needs more variables before it generalizes.

    python oracle_grid_plot.py [--pull]
"""
import os, sys, json, glob, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

EXP = os.path.join(ROOT, "main", "DATA", "experimental")
LOGS = os.path.join(EXP, "cluster_runs", "jsonl")
ACC = os.path.join(EXP, "cluster_runs", "acc_through_training.npz")
BAYES_LOGS = os.path.join(EXP, "bayes_gen", "jsonl")
DIMS = (10, 30, 90)
AGENTS = (("Joint's answers", "joint"), ("Naive's answers", "naive"), ("True answer, goal only", "truth1"))
DROP_S = (150, 200)            # John 11-10: no "both true" row, no |S| 150 / 200 (their runs stay on disk, not drawn)


def pull():
    os.makedirs(BAYES_LOGS, exist_ok=True)
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/bayes_gen/*.jsonl' {BAYES_LOGS}/", shell=True)
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/oracle_S*.jsonl' {LOGS}/", shell=True)
    subprocess.run(f"scp -q -o BatchMode=yes elsc-cluster:cgw/runs/cluster_runs/acc_through_training.npz {ACC}", shell=True)


def bayes_cells(label):
    """{(d_E, |S|): (training, novel)} plug-in accuracy at the last test of every FINISHED run of this label"""
    out, refs = {}, []
    for f in glob.glob(os.path.join(BAYES_LOGS, f"{label}_S*_d*_s0.jsonl")):
        S, d = (int(os.path.basename(f).split(k)[1].split("_")[0]) for k in ("_S", "_d"))
        recs = [json.loads(l) for l in open(f)]
        if not recs or recs[-1]["ep"] < (6000 if d == 30 and label in ("joint", "naive") else 8000):
            continue
        out[d, S] = (recs[-1]["train_plugin"], recs[-1]["novel_plugin"])
        if S >= 100:                                         # below that the novel set is one or a few environments
            refs.append((recs[-1]["novel_joint"], recs[-1]["novel_naive"]))
    return out, refs


def rcc_cells():
    """{(d_E, |S|): (training, novel)} Classifier accuracy of the finished oracle-mode RCC runs"""
    acc = np.load(ACC) if os.path.exists(ACC) else {}
    out = {}
    for f in glob.glob(os.path.join(LOGS, "oracle_S*_d*.jsonl")):
        name = os.path.basename(f)[:-6]
        S, d = (int(name.split(k)[1].split("_")[0]) for k in ("_S", "_d"))
        recs = [json.loads(l) for l in open(f)]
        if name + "__train" not in acc or recs[-1]["ep"] < 9999:
            continue
        out[d, S] = (acc[name + "__train"][len(recs) - 1, -1], recs[-1]["net"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        pull()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    rows, refs = [], []
    for title, label in AGENTS:
        cells, r = bayes_cells(label)
        rows.append((title, cells))
        refs += r
    rows.append(("RCC, oracle feedback (Classifier)", rcc_cells()))
    rows = [(t, c) for t, c in rows if c]
    joint, naive = np.mean(refs, 0)
    SS = sorted({S for _, c in rows for _, S in c} - set(DROP_S))
    n = sum(S in SS for _, c in rows for _, S in c)

    fig, ax = plt.subplots(len(rows), 2, figsize=(2 * (.46 * len(SS) + .5) + 1.3, 1.42 * len(rows) + 1.2), sharex=True,
                           sharey=True, constrained_layout=True, squeeze=False)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#e4e4e4")
    for i, (title, cells) in enumerate(rows):
        for j, env in enumerate(("training environments", "novel environments")):
            g = np.full((len(DIMS), len(SS)), np.nan)
            for (d, S), v in cells.items():
                if S in SS:
                    g[DIMS.index(d), SS.index(S)] = v[j]
            a = ax[i, j]
            im = a.imshow(np.ma.masked_invalid(g), cmap=cmap, vmin=.1, vmax=joint, origin="lower", aspect="auto")
            for (y, x), v in np.ndenumerate(g):
                if np.isfinite(v):
                    a.text(x, y, f"{v:.2f}"[1:], ha="center", va="center", fontsize=9, color="black" if v > .5 else "white")
            a.set_title(env, loc="right", fontsize=fs["legend"])
            a.set_xticks(range(len(SS)))
            a.set_xticklabels([str(S) for S in SS])
            a.set_yticks(range(len(DIMS)))
            a.set_yticklabels([str(d) for d in DIMS])
            a.tick_params(length=0)
        ax[i, 0].set_title(title, loc="left", fontsize=fs["title"] - 3)
        ax[i, 0].set_ylabel(r"$d_E$", rotation=0, ha="right", va="center")
    for a in ax[-1]:
        a.set_xlabel(r"number of variables $|\mathcal{S}|$")
    cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.015, aspect=45)
    cb.set_label("accuracy at T = 30, end of training")
    for y, lab in ((naive, "Naive"), (joint, "Joint")):
        cb.ax.axhline(y, color="white", lw=1.4)
        cb.ax.text(-.25, y, lab, transform=cb.ax.get_yaxis_transform(), ha="right", va="center", fontsize=9)
    fig.suptitle("Generator trained on each source's answers: accuracy of Bayes with its learned likelihood, "
                 r"$\langle[\arg\max_{r_g}\hat P(r_g\,|\,o_{1:30}) = r_g]\rangle$" + "\n" +
                 rf"embedding dimension $d_E$ against $|\mathcal{{S}}|$; {n} finished runs, one seed per cell, 10,000 "
                 "episodes per test; grey = no finished run", fontsize=fs["legend"])
    out = fig_path("oracle_grid.png")
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
