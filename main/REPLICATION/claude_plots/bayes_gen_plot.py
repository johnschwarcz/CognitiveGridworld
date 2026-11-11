"""Figure (John 11-09): the Generator and embeddings trained on an observer's answers, by |S|, in the format of Fig 12's
"Overlaid learning curves" (compare_RL_state_num.py): rows = training / novel environments, x = |S|, one curve per training
stage, one column per source of the answers:
  True answers             both variables' true realizations, every answer right (oracle mode's Generator input)
  Joint's answers          sampled from the Bayes posterior's marginals at T (right .69 of the time)
  Naive's answers          sampled from the factorized observer's marginals at T (right .40 of the time)
  True answer, goal only   (John 11-09: "only provide the generator with a single variable's realization") the goal's true
                           realization alone; the contextual variable's sample embedding and confidence are zeroed, so the
                           Generator can learn only the marginal likelihood P(o | r_g). Its ceiling is Naive Bayes.
  Network's answers (RCC)  the RCC at the main-text protocol (train_cluster_runs.py rcc_rl_S{S}_s0: batch 20,000, lr 1e-4,
                           30,000 episodes, 20,000 episodes per test, novel environments only); drawn once runs exist at
                           MIN_S values of |S| (11-10: only |S| 400 has run)

Training stages are absolute episode counts, log-spaced and the same in every column. Early learning is fast, so the runs
test densely early (train_bayes_generator.py TEST_AT).
y = accuracy at T = 30 of Bayes inference with the LEARNED likelihood (the Generator's predictions at all 100 joint values,
confidence 1; goal only: Naive-style Bayes with the learned marginal over the goal's 10 values), so every column measures
what the learned embeddings + Generator support.
  train_bayes_generator.py: no LSTM, batch 10,000, lr 1e-3; Joint / Naive 6,000 episodes, True / goal only 8,000.
  Convergence (last test against the test 2,000 episodes earlier, novel environments): median |change| .005. Not flat:
  Naive's answers above the threshold still creep (+.009 on average, +.027 at |S| 500), and at |S| 100 (.114 at 1k ->
  .134 at 6k). True answers at |S| 100 creep too (.567 at 1k -> .609 at 8k).
One seed per |S|, d_E = 30; every test is 10,000 episodes (random variable pairs) per environment set. Grey: Joint and
Naive Bayes with the true likelihood (mean over all tests). The first Joint / Naive runs (tests every 500) are in
DATA/experimental/bayes_gen/jsonl_every500/.

Novel environments, last test (11-10 05:00; all / low-FR / high-FR cells):
  threshold: True and Joint's answers transfer from |S| 100 (.61 / .59; 125: .71 / .68), Naive's and goal only from 125
  (.48 / .30); below it every source is at chance (.07-.13) although training environments are learned (True .74 / .67 /
  .58, Joint's .73 / .65 / .51, Naive's .52 / .49 / .42, goal only .41 / .40 / .36 at |S| 25 / 50 / 75).
  |S| 500: True .714 (.754 / .666), Joint's .705 (.752 / .653), Naive's .517 (.627 / .397), goal only .394 (.596 / .162);
  Naive Bayes with the true likelihood .455 (high-FR .14), Joint .77.

    python bayes_gen_plot.py [--pull]
"""
import os, sys, json, glob, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

EXP = os.path.join(ROOT, "main", "DATA", "experimental")
BAYES_LOGS = os.path.join(EXP, "bayes_gen", "jsonl")
RCC_LOGS = os.path.join(EXP, "cluster_runs", "jsonl")
# column: (title, log files by |S|, key per row)
_bayes = lambda lab: (lambda: _runs(BAYES_LOGS, f"{lab}_S*_d30_s0.jsonl"))
COLS = [("True answers", _bayes("truth"), ("train_plugin", "novel_plugin")),
        ("Joint's answers", _bayes("joint"), ("train_plugin", "novel_plugin")),
        ("Naive's answers", _bayes("naive"), ("train_plugin", "novel_plugin")),
        ("True answer, goal only", _bayes("truth1"), ("train_plugin", "novel_plugin")),
        ("Network's answers (RCC)", lambda: _runs(RCC_LOGS, "rcc_rl_S*_s0.jsonl"), ("train_plugin", "plugin"))]
NOTES = {"True answer, goal only": r"learns only $\hat P(o\,|\,r_g)$:" + "\nceiling = Naive Bayes"}
MIN_S = 3                                            # a column is drawn once it has runs at this many |S|
STAGES = (50, 100, 250, 500, 1000, 2500, 6000, 8000, 10000, 30000)


def pull():
    for src, dst, pat in (("bayes_gen", BAYES_LOGS, "*.jsonl"), ("cluster_runs", RCC_LOGS, "rcc*.jsonl")):
        os.makedirs(dst, exist_ok=True)
        subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/{src}/{pat}' {dst}/", shell=True)


def _runs(folder, pattern):
    """{|S|: list of test records}"""
    out = {}
    for f in glob.glob(os.path.join(folder, pattern)):
        S = int(os.path.basename(f).split("_S")[1].split("_")[0])
        out[S] = [json.loads(l) for l in open(f)]
    return out


def _longest(*sets):
    """per |S|, the run that has trained furthest (rcc30k_S{S}_s0 repeats rcc_S{S}_s0's first 10,000 episodes)"""
    out = {}
    for runs in sets:
        for S, recs in runs.items():
            if S not in out or recs[-1]["ep"] > out[S][-1]["ep"]:
                out[S] = recs
    return out


def at_episode(recs, key, n):
    """key at the test taken after n training episodes (an RCC run's last test is at n - 1); NaN if the
    run has no such test (yet)"""
    hit = [r[key] for r in recs if key in r and abs(r["ep"] - n) <= max(1, .02 * n)]
    return hit[0] if hit else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        pull()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = apply_fig_style()
    has = lambda recs, key: any(np.isfinite(at_episode(recs, key, n)) for n in STAGES)
    cols = [(t, load(), keys) for t, load, keys in COLS]
    cols = [c for c in cols if sum(has(recs, c[2][1]) for recs in c[1].values()) >= MIN_S]
    every = [json.loads(l) for f in glob.glob(os.path.join(BAYES_LOGS, "*_S*_d30_s0.jsonl")) for l in open(f)]
    ref = {k: np.mean([r[f"{env}_{k}"] for r in every for env in ("train", "novel")]) for k in ("joint", "naive")}
    Ss = sorted({S for c in cols for S, recs in c[1].items() if has(recs, c[2][1])})
    stages = [n for n in STAGES if any(np.isfinite(at_episode(recs, k, n)) for c in cols for recs in c[1].values() for k in c[2])]
    colors = dict(zip(stages, plt.cm.plasma(np.linspace(.05, .85, len(stages)))))

    fig, ax = plt.subplots(2, len(cols), figsize=(len(cols) * 2.9 + 1.2, 2 * 2.0 + 1.3), sharex=True, sharey=True,
                           constrained_layout=True, squeeze=False)
    for j, (title, runs, keys) in enumerate(cols):
        x = np.array(sorted(runs))
        for i, key in enumerate(keys):
            a = ax[i, j]
            prev = None
            for n in stages:
                col = colors[n]
                y = np.array([at_episode(runs[S], key, n) for S in x])
                ok = np.isfinite(y)                  # stages a run has not reached (or does not test): gaps, never bridged
                a.plot(x, y, ":", marker="o", ms=3, color=col, lw=1.0)
                if prev is not None:
                    a.fill_between(x, np.where(ok, prev, 0), np.where(ok, y, 0), where=ok & np.isfinite(prev),
                                   color=col, alpha=.35, lw=0)
                if ok.any():
                    prev = y
            for k, ls in (("joint", "--"), ("naive", ":")):
                a.axhline(ref[k], color="#8d8d8d", lw=.9, ls=ls, zorder=0)
            a.set_xscale("log")
            a.set_xticks(Ss)
            a.set_xticklabels([str(S) for S in Ss], fontsize=fs["tick"] - 1, rotation=90 if len(Ss) > 6 else 0)
            a.minorticks_off()
            a.set_ylim(.05, .85)
        ax[0, j].set_title(title, fontsize=fs["title"] - 2)
        ax[1, j].set_xlabel(r"number of variables $|\mathcal{S}|$")
        if title in NOTES:
            ax[1, j].text(.04, .82, NOTES[title], transform=ax[1, j].transAxes, ha="left", va="top",
                          fontsize=fs["legend"] - 2)
    ax[0, 0].set_ylabel("training environments\naccuracy")
    ax[1, 0].set_ylabel("novel environments\naccuracy")
    lab = lambda n: f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else (f"{n / 1000:g}k" if n >= 1000 else str(n))
    h = [plt.Line2D([], [], color=colors[n], marker="o", ms=3, lw=1, label=lab(n)) for n in stages]
    h += [plt.Line2D([], [], color="#8d8d8d", ls="--", label="Joint Bayes"),
          plt.Line2D([], [], color="#8d8d8d", ls=":", label="Naive Bayes (true likelihood)")]
    fig.legend(handles=h, title="training episodes", loc="outside lower center", ncol=len(h), fontsize=fs["legend"] - 2,
               title_fontsize=fs["legend"] - 2, frameon=False, columnspacing=1.0, handlelength=1.6)
    rcc = any(t.startswith("Network") for t, _, _ in cols)
    fig.suptitle("Generator trained on an observer's answers (column). Accuracy of Bayes with its learned likelihood, " +
                 r"$\langle[\arg\max_{r_g}\hat P(r_g\,|\,o_{1:30}) = r_g]\rangle$; batch 10,000, one seed per $|\mathcal{S}|$, "
                 "10,000 episodes per test" + (" (RCC: batch 20,000, 20,000 per test)" if rcc else ""),
                 fontsize=fs["legend"] - 1)
    out = fig_path("bayes_gen.png")
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
