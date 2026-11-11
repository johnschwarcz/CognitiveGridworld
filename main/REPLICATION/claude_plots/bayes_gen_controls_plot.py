"""Controls for "what the feedback must carry" (11-10): whose answers does the Generator need?

The Generator and the embeddings are trained on a Bayesian observer's answers in the RCC's feedback format
(train_bayes_generator.py: no Classifier; each variable's answer is sampled from the observer's marginal belief at T; the
goal's confidence is the reward; the loss is weighted towards rewarded episodes, >= 97.8% of its weight). The source of
each of the two answers is varied separately:
    goal's answer   contextual variable's answer   run
    Joint           Joint                          joint_S500_d30_s0   (6,000 episodes)
    Naive           Joint                          nj_...              (8,000)
    Joint           Naive                          jn_...              (8,000)
    Naive           Naive                          naive_...           (6,000)
    Joint's answers, each replaced by a uniform draw with the probability that brings the goal's accuracy down to Naive's
                                                   jointU_...          (8,000; errors unrelated to the state)
    true values, both variables                    truth_...           (8,000)
    true goal value, no contextual answer          truth1_...          (8,000; can learn only the marginal likelihood)
The goal's answer decides which episodes are rewarded; the contextual variable has no reward, so its value reaches the
Generator only through the answer.
y = accuracy at T = 30 of exact Bayes with the LEARNED likelihood (truth1: Naive-style Bayes with the learned marginal) in
novel environments (both variables held out), 10,000 episodes per test. Panels: all states | other states | high-FR states
(the goal's cell is one where the factorized observer is inconsistent, Delta(r) > 0; env_generators.naive_consistent_cells).
Grey: Joint and Naive Bayes with the TRUE likelihood (mean over the plotted runs' tests). A dot ends every line at its
last test, so unfinished runs show where they are.
|S| = 500, d_E = 30, one run per source, batch 10,000, lr 1e-3.

How often the answers are right (exact expectations from the beliefs, training environments; scratchpad label_stats.py of
session f79b4ed6): goal .69 (Joint) / .40 (Naive, jointU); contextual answer in REWARDED episodes: joint .69, nj .64,
jointU .40, jn .37, naive .31 (Naive's two answers are anti-correlated: .40 over all episodes).

Final 11-10 08:15 (every run finished; mean of the last 3 tests up to 6,000 episodes, where all four cells of the 2 x 2
exist), novel, all states | high-FR states:
  joint .701 | .652   nj .677 | .624   jn .577 | .477   naive .509 | .393   jointU .448 | .360   truth .711 | .660
  truth1 .379 | .159   (at 8,000: nj .681, jn .581, jointU .461, truth .714, truth1 .389: the 8,000-episode runs moved
  < .015 after 6,000)
  contextual answer's source (Joint - Naive): +.124 with the goal from Joint, +.168 with it from Naive (high-FR +.176 /
  +.231); goal answer's source: +.024 with the contextual answer from Joint, +.068 with it from Naive (high-FR +.029 /
  +.084). novel_kl_env: joint .031, nj .042, jn .152, naive .227, jointU .291.
  -> the contextual answer's source is the main effect; the goal's source matters mainly when the contextual answers are
     Naive's. jointU (uniformly placed errors at Naive's goal accuracy) ends at Naive Bayes's level (.45), below Naive's
     answers (.51); its wrong answers keep Joint's confidence (.61 vs jn .52), so it is not a clean "random vs structured
     errors" control. One run per line.

    python bayes_gen_controls_plot.py [--pull]
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path, apply_fig_style

LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "bayes_gen", "jsonl")
S, D_E = 500, 30
JOINT, NAIVE, RAND = "#2c7bb6", "#d7191c", "#8d8d8d"
# run, legend label, colour = source of the contextual variable's answer, style = source of the goal's answer, width
# (legend order: 3 columns filled downwards = contextual answer from Joint | from Naive | random errors)
DASH = (0, (3.5, 1.6))
RUNS = (("joint", "goal: Joint, contextual: Joint", JOINT, "-", 2.3),
        ("nj", "goal: Naive, contextual: Joint", JOINT, DASH, 2.3),
        ("truth", "goal: true value, contextual: true value", "k", "-", 1.1),
        ("jn", "goal: Joint, contextual: Naive", NAIVE, "-", 2.3),
        ("naive", "goal: Naive, contextual: Naive", NAIVE, DASH, 2.3),
        ("truth1", "goal: true value, contextual: none", "k", (0, (1, 1.4)), 1.5),
        ("jointU", "both: Joint's, with random errors", RAND, "-", 2.0))
PANELS = (("", "All states"), ("_low", r"Other states, $\Delta(r) \leq 0$"), ("_high", r"High-FR states, $\Delta(r) > 0$"))


def pull():
    os.makedirs(LOGS, exist_ok=True)
    names = ",".join(r[0] for r in RUNS)
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/bayes_gen/{{{names}}}_S{S}_d{D_E}_s0.jsonl' {LOGS}/", shell=True)


def load(name):
    f = os.path.join(LOGS, f"{name}_S{S}_d{D_E}_s0.jsonl")
    return [json.loads(l) for l in open(f) if l.strip()] if os.path.exists(f) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        pull()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    fs = apply_fig_style()
    runs = {name: load(name) for name, *_ in RUNS}
    fig, ax = plt.subplots(1, len(PANELS), figsize=(len(PANELS) * 3.3 + .7, 4.5), sharex=True, sharey=True,
                           constrained_layout=True)
    for a, (suffix, title) in zip(ax, PANELS):
        for name, label, col, ls, lw in RUNS:
            recs = [r for r in runs[name] if r["ep"] > 0]
            if not recs:
                continue
            x, y = np.array([r["ep"] for r in recs]), np.array([r["novel_plugin" + suffix] for r in recs])
            a.plot(x, y, color=col, ls=ls, lw=lw)
            a.plot(x[-1], y[-1], "o", color=col, ms=4.5, mec="w", mew=.6, zorder=5)
        for k, ls, lab in (("joint", "--", "Joint Bayes"), ("naive", ":", "Naive Bayes")):
            ref = np.mean([r[f"novel_{k}{suffix}"] for recs in runs.values() for r in recs])
            a.axhline(ref, color="#555555", lw=1.0, ls=ls, zorder=0)
            a.text(95, ref + .012, lab, color="#555555", ha="left", va="bottom", fontsize=fs["tick"] + .5)
        a.set(xscale="log", xlim=(85, 9500), ylim=(0, .86), xlabel="training episodes")
        a.set_title(title, fontsize=fs["title"] - 3)
        a.xaxis.set_major_locator(mticker.FixedLocator((100, 300, 1000, 3000, 8000)))
        a.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        a.xaxis.set_minor_formatter(mticker.NullFormatter())
        a.grid(alpha=.25, lw=.6)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    ax[0].set_ylabel("accuracy of Bayes with the learned likelihood")
    h = [plt.Line2D([], [], color=col, ls=ls, lw=lw, label=label) for name, label, col, ls, lw in RUNS if runs[name]]
    fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False, fontsize=fs["legend"], handlelength=3.2,
               title="Whose answers the Generator is trained on (the goal's answer decides the reward; "
                     "the contextual variable's answer is never rewarded)", title_fontsize=fs["legend"])
    fig.suptitle(r"Novel environments, $|\mathcal{S}| = 500$, $d_E = 30$; Generator and embeddings trained on a Bayesian "
                 "observer's answers (no Classifier);\n1 run per line, 10,000 episodes per test, dot = last test",
                 fontsize=fs["legend"])
    out = fig_path("bayes_gen_controls.png")
    fig.savefig(out, dpi=200)
    print(out)
    for n in (1000, 2500, 6000, 8000):
        row = [f"{name} {r['novel_plugin']:.3f} ({r['novel_plugin_low']:.3f}/{r['novel_plugin_high']:.3f})"
               for name, *_ in RUNS for r in runs[name] if r["ep"] == n]
        print(f"{n:5d}: " + " | ".join(row))


if __name__ == "__main__":
    main()
