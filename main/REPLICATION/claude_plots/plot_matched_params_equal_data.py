"""plot_matched_params_joint.py with one test per block behind every curve and one smoothing on all of them (John 2026-11-04: "We
need all curves in the same figure to have the same amount of data", then "let's use the log guassian"): one 8,000-episode
test per 500 training episodes, then a Gaussian of constant width on the log axis (SD decades), the same weights for every curve
at a given episode. So curves match each other at each episode, while a point late in training averages more tests than an early
one. Tests sit at their true episode; the untrained test at 0 is left out. Near a curve's ends the Gaussian is one-sided (the
weights renormalized), at 40k for a finished run and at its last test for a running one. The paper's runs test every 499
updates, not 500: under 0.2% off this grid. Only batch means were stored, so tests cannot be cut down to fewer episodes. The
trained N = 100 and N = 200 are single runs here, seed 20261029 (the mean of five seeds holds five times the test data, and the
paper's five N = 200 reps ran 2,000 test episodes each). The Echo N = 10k could not hold the 8,000 batch: it trained and tested
on batches of 2,000, so a quarter of the examples per episode and of the test episodes, stated in the title (John 11-05: not in the legend). Every
gap is taken from one cached bound. A single low test spreads over the kernel: the joint-loss N = 1k crash at 11,480 lasted ~50
episodes. Curves are drawn by width, the largest on top, and listed by width, the largest at the bottom (John 11-04).
-> main/REPLICATION/figure_bin/compare_matched_params_loglog_gaussian.png
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/plot_matched_params_equal_data.py --draw"""
import glob
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
import compare_matched_params as cmp                                              # noqa: E402
import plot_matched_params_new as pm                                              # noqa: E402
import plot_matched_params_joint as pj                                            # noqa: E402

SD = float(os.environ.get("GAUSS_SD", .05))                                       # decades


def weights(x, c):
    k = np.exp(-.5 * (np.log(x / c) / (SD * np.log(10))) ** 2)
    return k / k.sum()


def window(x, acc):
    """x: block ends, each test made at the start of its block: the tests' episodes, the untrained one dropped, and the Gaussian"""
    x, acc = np.asarray(x, float)[1:] - cmp.CKPT_EVERY, np.asarray(acc, float)[1:]
    return x, np.array([(weights(x, c) * acc).sum() for c in x])


def tests_at(c):
    """the effective number of tests behind a point at episode c, inside a full run"""
    return 1 / (weights(np.arange(cmp.CKPT_EVERY, pm.XMAX, cmp.CKPT_EVERY, dtype=float), c) ** 2).sum()


def legend(ax):
    """by width, the largest at the bottom: the order of the curves' heights (John 11-04)"""
    lines = sorted((ln for ln in ax.get_lines() if not ln.get_label().startswith("_")), key=lambda ln: ln.width)
    ax.legend(lines, [ln.get_label() for ln in lines], loc="lower left", frameon=False, fontsize=12)


def figure(step=cmp.STEP, path=None):
    from main.utils import fig_path
    cmp.apply_fig_style()
    b = dict(np.load(cmp.BOUNDS_NPZ))
    J = b[f"{cmp.BOUND_KEY}_joint"][step - 1]
    fig, axes = plt.subplots(2, 3, figsize=(30, 10.4), sharex=True, sharey=True)
    (t1, t2, t3), (e1, e2, e3) = axes

    def curve(ax, x, acc, width, col, label):
        """drawn by width: the wider over the narrower, under the naive bound"""
        ln, = ax.plot(x, pm._pos(J - acc), color=col, lw=2, label=label, zorder=2 + np.log10(width))
        ln.width = width

    def draw(ax, run, name, width, col):
        """a training-dynamics run: one test per block, the Gaussian; the legend says how far a running one has got"""
        if not glob.glob(os.path.join(pm._dir(run), "test_*.npz")):
            print(f"  {run}: no test yet")
            return
        x, acc, reached = pj.pooled(run, n_tests=1)
        if len(x) < 2:
            print(f"  {run}: no trained test yet")
            return
        x, acc = window(x, acc)
        note = "" if pm.finished(run) else f" (training: {min(reached, pm.XMAX) / 1e3:.1f}k of {pm.XMAX / 1e3:.0f}k)"
        curve(ax, x, acc, width, col, f"{name}, {pj.count(pj.params(run))} trainable params{note}")
        print(f"  {run:24s} {acc[-1]:.3f} at training episode {x[-1]:.0f}")

    nets = cmp.resolve_colors([dict(n) for n in cmp.NETWORKS if n["stem"] not in pm.DROP] + [dict(pm.ESN2K)])
    draw(e1, "echo_N200", r"Echo ($N{=}200$)", 200, pj.C_E200)
    draw(e1, "echo_N500", r"Echo ($N{=}500$)", 500, pj.C_E500)
    for net in nets:
        if "reservoir" in net["stem"]:
            a = cmp.step_acc(net, step)[0]
            x = (np.arange(len(a)) + 1) * cmp.CKPT_EVERY
            keep = x <= pm.XMAX
            x, a = window(x[keep], a[keep])
            name = net["label"].split(r" $\to$")[0]
            width = {"sanity/reservoir_ctx_2": 1000, pm.ESN2K["stem"]: 2000}.get(net["stem"], 10000)
            curve(e1, x, a, width, net["color"], f"{name}, {cmp.trainable_params(net) / 1e3:.0f}k trainable params")
    TRAINED = ((r"Trained ($N{=}50$)", 50, pj.C_N50), (r"Trained ($N{=}100$)", 100, pm.C_N100), (r"Trained ($N{=}150$)", 150, pj.C_N150),
               (r"Trained ($N{=}200$)", 200, cmp.C_TRAIN), (r"Trained ($N{=}300$)", 300, pj.C_N300), (r"Trained ($N{=}400$)", 400, pj.C_N400), (r"Trained ($N{=}500$)", 500, pj.C_N500), (r"Trained ($N{=}1k$)", 1000, pj.C_N1K))
    for run, (name, width, col) in zip(("trained_N50", "trained_N100", "trained_N150", "trained_N200", "trained_N300", "trained_N400", "trained_N500", "trained"), TRAINED):
        draw(t1, run, name, width, col)
    c_echo = next(n["color"] for n in nets if n["stem"] == "sanity/reservoir_ctx_2")
    c_2k = next(n["color"] for n in nets if n["stem"] == pm.ESN2K["stem"])
    c_10k = next(n["color"] for n in nets if "10000N" in n["stem"])
    for at, ae, (title, echoes, trained) in zip((t2, t3), (e2, e3), pj.PANELS):
        for run, (name, width, col) in zip(echoes, ((r"Echo ($N{=}200$)", 200, pj.C_E200), (r"Echo ($N{=}500$)", 500, pj.C_E500),
                                                    (r"Echo ($N{=}1k$)", 1000, c_echo), (r"Echo ($N{=}2k$)", 2000, c_2k),
                                                    (r"Echo ($N{=}10k$)", 10000, c_10k))):
            draw(ae, run, name, width, col)
        for run, spec in zip(trained, TRAINED):
            draw(at, run, *spec)
        at.set_title(title, fontsize=15)
    pm._style(t1, J, b, step, label=False)                                        # the trained curves start on the bound at
    pm._style(e1, J, b, step)                                                     # the left: its label goes right, above the line
    t1.text(.99, (J - b[f"{cmp.BOUND_KEY}_naive"][step - 1]) * 1.1, "Naive bound", color="k", fontsize=15, va="bottom", ha="right",
            transform=t1.get_yaxis_transform(), zorder=11)
    for ax in (t1, e1):
        ax.set_ylabel("Optimal $-$ Network Accuracy")
    for ax in (t2, t3, e2, e3):
        pm._style(ax, J, b, step, label=False)
    for ax in axes.flat:
        legend(ax)
    for ax in (t1, t2, t3):
        ax.set_xlabel("")
    t1.set_title("Marginal output, goal supervision", fontsize=15)
    fig.suptitle(f"one 8,000-episode test per 500 training episodes, Gaussian of SD {SD:g} decades on the log axis: the same weights on every curve at each "
                 f"episode (tests per point: about {tests_at(1e3):.0f} at 1k, {tests_at(1e4):.0f} at 10k, {tests_at(2e4):.0f} at 20k); Echo $N{{=}}10k$: batch 2,000; 1 seed per curve", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.canvas.draw()
    for ax in axes.flat:
        for name in pm._collisions(ax):
            print(f"  WARNING: a curve passes within 3 pt of the text {name!r}")
    print(f"  y-axis {t1.get_ylim()[0]:.4f} to {t1.get_ylim()[1]:.3f}")
    path = path or fig_path("compare_matched_params_loglog_gaussian.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


if __name__ == "__main__":
    if "--draw" in sys.argv:                                                      # John 11-05: figure not in use; running queues call this
        figure()
