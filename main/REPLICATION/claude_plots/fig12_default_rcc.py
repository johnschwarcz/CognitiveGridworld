"""The paper's Figure 12 ("Learning and generalization with Compositional Embeddings depend on |S|") redrawn for the new
default RCC (John 11-14: "Yes please redraw figure 12"): the updated Generator (gen_zflow), the RCC's own answers as its
feedback, Fig 12's settings (train_cluster_runs.py rcc_rl_S<S>_s0: d_E 30, batch 20,000, lr 1e-4, entropy .1, 30,000
iterations), one run per |S|. The layout and styling are compare_RL_state_num.py's "Version C" (the paper's figure),
copied, with its data replaced by these runs' logs:
    training-set accuracy = train_net (goal accuracy at T = 30 on the training batch before each test)
    testing-set accuracy  = net (goal accuracy at T = 30 on a batch of novel environments)
at 10%, 20%, ..., 100% of training (iterations 3,000, 6,000, ..., 30,000), as the paper's 10 checkpoints. Each point is
one batch of 20,000 episodes. Sizes = the paper's (100 ... 1,000); --small adds |S| 25 / 50 / 75.
|S| 500 = zflow_s0 (the same settings; Fig 12's RCC column used the zflow runs there, there is no rcc_rl_S500): its log
has no train_net, so both curves come from its saved net's train/test_acc_through_training at T = 30 (checked on
rcc_rl_S250_s0: those arrays equal the log's train_net / net exactly). The net is pulled to
DATA/experimental/fig12_default_rcc/zflow_s0_net.pth.

    python fig12_default_rcc.py [--pull] [--small] [--paper]
--paper (John 11-14: "copy the size and styling of the paper figure (labels (a),(b),etc), let's also remove |S| = 300,
400, to try to make it smoother. And please send this as a pdf"): |S| 25-1000 without 300 / 400 (11-16, John: "can we
use all S that we trained": all 12 again, 25-1000); the look of the
manuscript's embeddings_learning_2.pdf (Version C plus its hand edits): no header (the data counts go in the caption),
|S| labels in both rows of (a) (11-16, John: "put the |S| = ... slightly above the top row so we dont need it twice":
now once, just above the top row), "Training Set" / "Testing Set" inside (a) and (b), the 3-D panels' names at their top
left, panel letters "(a)"-"(d)" centred under the panels; 24 x 18.5 in (the paper's aspect) -> fig12_default_rcc_paper.pdf
(TrueType fonts) + .png preview. 11-14 (John: "the spacing of the 3D panels are still a bit off"): 3-D boxes zoomed 1.1x
and spread over the whole bottom half, titles tilted 15 degrees as in the paper. --top-only (John: "should we remove these
panels and stick to just the top row?"): (a) and (b) alone, 24 x 9.4 in -> fig12_default_rcc_paper_top.pdf.
11-16 (John: "put the 'Training progress' under the relevant ticks"): the label sits under the first column, where the
progress ticks are; the ticks now sit at their true positions, 10% / 50% / 100% (Version C labelled the first point,
taken at 10% of training, "0%" and the 60% point "50%").
"""
import os, sys, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from main.utils import fig_path

LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "cluster_runs", "jsonl")
PAPER_SIZES = [100, 125, 150, 200, 250, 300, 400, 500, 1000]
SMALL = [25, 50, 75]
PAPER_FIG_SIZES = SMALL + PAPER_SIZES          # --paper: every trained |S| (John 11-16: "use all S that we trained")
ITERS = 30000


def pull():
    os.makedirs(LOGS, exist_ok=True)
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/{{rcc_rl_S*_s0,zflow_s0}}.jsonl' {LOGS}/",
                   shell=True, executable="/bin/bash")
    if not os.path.exists(NET500):
        os.makedirs(os.path.dirname(NET500), exist_ok=True)
        subprocess.run(f"scp -q -o BatchMode=yes elsc-cluster:cgw/main/DATA/cluster_runs/zflow_s0_net.pth {NET500}", shell=True)


NET500 = os.path.join(ROOT, "main", "DATA", "experimental", "fig12_default_rcc", "zflow_s0_net.pth")


def run_curves(S):
    """(test iterations, training accuracy, testing accuracy) at T = 30 for every test of the |S| run"""
    if S == 500:
        import torch
        d = torch.load(NET500, map_location="cpu", weights_only=False)
        tr, te = (np.asarray(d[k])[:, -1] for k in ("train_acc_through_training", "test_acc_through_training"))
        r = [json.loads(l) for l in open(os.path.join(LOGS, "zflow_s0.jsonl")) if l.strip()]
        ep = np.array([q["ep"] for q in r])
        assert len(ep) == len(te) and np.allclose([q["net"] for q in r], te), "zflow_s0: net file and log disagree"
        return ep, tr, te
    r = [json.loads(l) for l in open(os.path.join(LOGS, f"rcc_rl_S{S}_s0.jsonl")) if l.strip()]
    return (np.array([q["ep"] for q in r]), np.array([q.get("train_net", np.nan) for q in r]),
            np.array([q["net"] for q in r]))


def curves(state_nums, n_eps=10):
    """(train, test): (|S|, n_eps) accuracies at n_eps equally spaced fractions of training (the last = the final test)"""
    Ztr, Zte = np.full((len(state_nums), n_eps), np.nan), np.full((len(state_nums), n_eps), np.nan)
    for i, S in enumerate(state_nums):
        ep, tr, te = run_curves(S)
        for e in range(n_eps):
            k = int(np.searchsorted(ep, round((e + 1) * ITERS / n_eps) - 1, side="right")) - 1   # the last test at or before
            Ztr[i, e], Zte[i, e] = tr[k], te[k]
        assert abs(ep[-1] - (ITERS - 1)) < 1, f"|S| {S}: the run ended at {ep[-1]}"
    return Ztr, Zte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--top-only", action="store_true", help="--paper without the 3-D panels")
    a = ap.parse_args()
    if a.pull:
        pull()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns

    paper, top_only = a.paper or a.top_only, a.top_only
    state_nums = PAPER_FIG_SIZES if paper else (SMALL if a.small else []) + PAPER_SIZES
    Z_train, Z_test = curves(state_nums)
    n_states, n_eps = Z_train.shape
    for S, tr, te in zip(state_nums, Z_train[:, -1], Z_test[:, -1]):
        print(f"|S| {S}: training {tr:.3f}, testing {te:.3f}")

    # ==========================================
    # compare_RL_state_num.py, VERSION C (the paper's figure), unchanged but for the data (and --paper's edits)
    # ==========================================
    if paper:
        plt.rcParams["pdf.fonttype"] = 42                               # TrueType fonts in the PDF, not Type 3
    fig_h = 9.4 if top_only else 18.5 if paper else 20
    fig = plt.figure(figsize=(24, fig_h))
    gs_main = gridspec.GridSpec(1 if top_only else 2, 1, figure=fig, height_ratios=[1] if top_only else [1, 1], hspace=0.25)
    cmap_eps = plt.get_cmap("plasma")
    colors_eps = cmap_eps(np.linspace(0, 1, n_eps))
    cmap_states = plt.get_cmap("plasma")
    colors_states = cmap_states(np.linspace(0, 1, n_eps + 6))
    cmap_cool = plt.get_cmap("cool")
    colors_sub = [sns.desaturate(cmap_cool(x), 0.4) for x in np.linspace(0, 1, n_states)]
    x_pct = np.linspace(100 / n_eps, 100, n_eps)

    gs_super = gs_main[0].subgridspec(1, 2, width_ratios=[2, 1], wspace=0.08)
    title_y = 1 - 0.185 / fig_h if paper else 0.945                    # 0.185 in from the top edge
    fig.text(0.33, title_y, "Learning curves per $\\mathcal{S}$", ha='center', va='top', fontsize=22, fontweight='bold')
    fig.text(0.83, title_y, "Overlaid learning curves", ha='center', va='top', fontsize=22, fontweight='bold')
    if not paper:                                   # data counts (the paper's figure has them in its caption)
        fig.text(0.5, 0.993, "The new default RCC (updated Generator, its own answers as feedback): $d_E$ = 30, batch "
                 "20,000, lr 1e-4, 30,000 iterations, one run per $\\mathcal{S}$\neach point = one batch of 20,000 episodes "
                 "(training set: the training batch before the test; testing set: novel environments)", ha='center',
                 va='top', fontsize=15)

    gs_sub = gs_super[0].subgridspec(2, n_states, wspace=0.0, hspace=0.05)
    ax_curves_sub = np.empty((2, n_states), dtype=object)
    mid_idx = n_states // 2
    # --paper (John 11-14: "(a) and [the set names] can be centered better"): 10 columns have no middle one, so the progress
    # ticks go under the first column and "Training Progress" / the set names are centred on the whole block, like "(a)"
    tick_idx = 0 if paper else mid_idx
    for s in range(n_states):
        if s == 0:
            ax_curves_sub[0, s] = fig.add_subplot(gs_sub[0, s])
            ax_curves_sub[1, s] = fig.add_subplot(gs_sub[1, s], sharex=ax_curves_sub[0, s])
        else:
            ax_curves_sub[0, s] = fig.add_subplot(gs_sub[0, s], sharey=ax_curves_sub[0, 0])
            ax_curves_sub[1, s] = fig.add_subplot(gs_sub[1, s], sharey=ax_curves_sub[1, 0], sharex=ax_curves_sub[0, s])
        line_color = fill_color = colors_sub[s]
        ax_curves_sub[0, s].plot(x_pct, Z_train[s, :], '-', color=line_color, alpha=1, zorder=10, lw=2)
        ax_curves_sub[0, s].scatter(x_pct, Z_train[s, :], c=[line_color], s=50, zorder=11)
        ax_curves_sub[0, s].fill_between(x_pct, Z_train[s, :], 0.05, color=fill_color, alpha=0.15, zorder=-s)
        ax_curves_sub[1, s].plot(x_pct, Z_test[s, :], '-', color=line_color, alpha=1, zorder=10, lw=2)
        ax_curves_sub[1, s].scatter(x_pct, Z_test[s, :], c=[line_color], s=50, zorder=11)
        ax_curves_sub[1, s].fill_between(x_pct, Z_test[s, :], 0.05, color=fill_color, alpha=0.15, zorder=-s)
        if paper:                                       # John 11-16: once, just above the top row
            ax_curves_sub[0, s].text(0.5, 1.02, f"$\\mathcal{{S}} =$ {state_nums[s]}", transform=ax_curves_sub[0, s].transAxes,
                                     ha='center', va='bottom', fontsize=14, fontweight='bold', color=colors_sub[s], zorder=20)
        for row in range(2):
            if not paper:                               # Version C: inside the bottom row (drawn twice)
                ax_curves_sub[1, s].text(0.5, 0.95, f"$\\mathcal{{S}} =$ {state_nums[s]}",
                                         transform=ax_curves_sub[1, s].transAxes, ha='center', va='top', fontsize=14,
                                         fontweight='bold', color=colors_sub[s],
                                         bbox=dict(facecolor='white', alpha=1, edgecolor='none', pad=2), zorder=20)
            ax_curves_sub[row, s].set_ylim([0.08, 0.8])
            ax_curves_sub[row, s].grid(True, linestyle='--', alpha=0.4, zorder=0)
            ax_curves_sub[row, s].spines['top'].set_visible(False)
            ax_curves_sub[row, s].spines['right'].set_visible(False)
            ax_curves_sub[row, s].tick_params(axis='both', which='major', labelsize=14)
            if s > 0:
                ax_curves_sub[row, s].tick_params(labelleft=False)
            if s == 0:
                phase_str = "Training" if row == 0 else "Testing"
                ax_curves_sub[row, s].set_ylabel(f"{phase_str} Set Accuracy", fontsize=16)
            if s == tick_idx:
                ax_curves_sub[1, s].xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=100, decimals=0))
                if paper:                                       # 11-16: true positions (Version C put "0%" at 10%, "50%" at 60%)
                    ax_curves_sub[1, s].set_xticks([x_pct[0], 50, 100], labels=[f"{x_pct[0]:.0f}%", "50%", "100%"])
                else:
                    ax_curves_sub[1, s].set_xticks([x_pct[0], x_pct[len(x_pct) // 2], x_pct[-1]], labels=["0%", "50%", "100%"])
                if n_states > 9:                                # narrower columns (--small): smaller progress labels
                    ax_curves_sub[1, s].tick_params(axis='x', labelsize=12 if paper else 11)
                if not paper:                                   # --paper: centred on the block below
                    ax_curves_sub[1, s].set_xlabel("Training Progress", fontsize=16)
                ax_curves_sub[1, s].tick_params(labelbottom=True)
            ax_curves_sub[row, s].set_xlim(x_pct[0], x_pct[-1])
        if s != tick_idx:
            ax_curves_sub[0, s].tick_params(labelbottom=False)
            ax_curves_sub[1, s].tick_params(labelbottom=False)
    ax_curves_sub[0, tick_idx].tick_params(labelbottom=False)

    gs_conglom = gs_super[1].subgridspec(2, 1, hspace=0.05)
    ax_conglom_top = fig.add_subplot(gs_conglom[0, 0])
    ax_conglom_bot = fig.add_subplot(gs_conglom[1, 0], sharex=ax_conglom_top)
    ax_curves_conglom = [ax_conglom_top, ax_conglom_bot]
    for i, phase in enumerate(["Training", "Testing"]):
        for e in range(n_eps):
            line_color, fill_color = colors_states[e], colors_eps[e]
            y_data = Z_train[:, e] if i == 0 else Z_test[:, e]
            ax_curves_conglom[i].plot(state_nums, y_data, ':', color=line_color, lw=2, alpha=1, zorder=10)
            ax_curves_conglom[i].scatter(state_nums, y_data, c=[line_color], s=50, zorder=11)
            ax_curves_conglom[i].fill_between(state_nums, y_data, 0.01, color=fill_color, alpha=0.25, zorder=-e)
        ax_curves_conglom[i].set_xscale("log")
        ax_curves_conglom[i].set_xlim(state_nums[0], state_nums[-1])
        ax_curves_conglom[i].minorticks_off()
        ax_curves_conglom[i].set_ylim([0.08, 0.8])
        ax_curves_conglom[i].grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax_curves_conglom[i].spines['top'].set_visible(False)
        ax_curves_conglom[i].spines['right'].set_visible(False)
        ax_curves_conglom[i].tick_params(axis='x', which='major', labelsize=14)
        ax_curves_conglom[i].tick_params(axis='y', left=True, labelleft=True, labelsize=14)
        phase_str = "Train" if i == 0 else "Test"
        ax_curves_conglom[i].set_ylabel(f"{phase_str} Accuracy", fontsize=15)
    ax_curves_conglom[0].tick_params(labelbottom=False)
    ax_curves_conglom[1].set_xticks(state_nums)
    if paper:                                           # 1.6 decades: 100 / 125 / 150 touch when horizontal
        ax_curves_conglom[1].set_xticklabels([str(s) for s in state_nums], fontsize=13, rotation=45, ha='right',
                                             rotation_mode='anchor')
    else:
        ax_curves_conglom[1].set_xticklabels([str(s) for s in state_nums], fontsize=14, rotation=90 if n_states > 9 else 0)
    ax_curves_conglom[1].set_xlabel("$\\mathcal{S}$", fontsize=16)
    if paper:                                           # the paper's set names inside (b)
        for i, name in enumerate(("Training Set", "Testing Set")):
            ax_curves_conglom[i].text(0.5, 0.95, name, transform=ax_curves_conglom[i].transAxes, ha='center', va='top',
                                      fontsize=15, fontweight='bold', zorder=21)
    legend_elements = [mpl.lines.Line2D([0], [0], color=colors_states[e], ls='-', lw=2, label=f'{int(x_pct[e])}%')
                       for e in reversed(range(n_eps))]
    ax_curves_conglom[1].legend(handles=legend_elements, loc='upper left', frameon=True, ncol=1, fontsize=14,
                                title="Training Progress")

    gs_bottom = None if top_only else gs_main[1].subgridspec(1, 2, wspace=0.05)
    X_3d, Y_log_3d = np.meshgrid(x_pct, np.log10(state_nums))
    axes_3d = []
    for k, (Z, title, col) in enumerate(() if top_only else
                                        ((Z_train, "Training Set", '#D9E8D9'), (Z_test, "Testing Set", '#EDD9D9'))):
        ax3 = fig.add_subplot(gs_bottom[0, k], projection='3d')
        axes_3d.append(ax3)
        ax3.set_box_aspect((1 / 2, 1, 1 / 2), zoom=1.1 if paper else 1)
        ax3.plot_surface(X_3d, Y_log_3d, Z, color=col, shade=False, edgecolor='none', alpha=1, zorder=1, antialiased=True)
        ax3.plot_surface(X_3d, Y_log_3d, Z, color=col, shade=True, edgecolor='none', alpha=.25, zorder=2, antialiased=True)
        for s in range(n_states):
            ax3.plot(X_3d[s, :], Y_log_3d[s, :], Z[s, :], color=colors_sub[s], lw=2, ls='-', zorder=3)
        for e in range(n_eps):
            ax3.plot(X_3d[:, e], Y_log_3d[:, e], Z[:, e], color=colors_states[e], lw=2, ls=':', zorder=3)
        if paper:                                       # the paper: the panel's name at its top left, tilted with the box
            ax3.text2D(0.07, 0.8, title, transform=ax3.transAxes, ha='left', va='bottom', fontsize=20, fontweight='bold',
                       rotation=15)
        else:
            ax3.set_title(title, fontsize=20, fontweight='bold', pad=-5)
        ax3.set_xlabel("Training Progress (%)", fontsize=14, labelpad=5)
        ax3.set_ylabel("$\\mathcal{S}$", fontsize=14, labelpad=5)
        ax3.set_zlabel("Accuracy", fontsize=14, labelpad=5)
        ax3.set_zlim(0.08, 0.8)
        for pane in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            pane.set_pane_color((1.0, 1.0, 1.0, 0.0))
            pane._axinfo["grid"].update({"linestyle": ":", "color": "lightgray", "alpha": 0.})
        ax3.view_init(elev=30, azim=-210)
        ax3.set_yticks(np.log10(state_nums))
        shown = (25, 50, 75, 100, 150, 250, 500, 1000) if paper else (25, 50, 100, 200, 500, 1000)
        ax3.set_yticklabels([str(s) if n_states <= 9 or s in shown else "" for s in state_nums])
        ax3.invert_yaxis()
        ax3.tick_params(axis='both', which='major', labelsize=10)

    if not paper:
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.905)
        out = fig_path("fig12_default_rcc" + ("_small" if a.small else "") + ".png")
        fig.savefig(out, dpi=120)
        print(out)
        return
    if top_only:
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.14, top=1 - 0.98 / fig_h)    # 11-16: room for the |S| labels
    else:
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.03, top=1 - 0.98 / fig_h)    # 11-16: room for the |S| labels
        for k, ax3 in enumerate(axes_3d):              # the 3-D boxes fill the bottom half, as in the paper
            ax3.set_position([0.01 + 0.49 * k, 0.05, 0.47, 0.44])
    # the paper's panel letters: "(a)" plain, centred under each panel
    p0, p1, q = ax_curves_sub[1, 0].get_position(), ax_curves_sub[1, -1].get_position(), ax_conglom_bot.get_position()
    x_a = (p0.x0 + p1.x1) / 2                           # the centre of (a)'s block of columns
    for row, name in enumerate(("Training Set", "Testing Set")):        # the paper's set names inside (a)
        r = ax_curves_sub[row, 0].get_position()
        fig.text(x_a, r.y0 + 0.8 * r.height, name, ha='center', va='top', fontsize=15, fontweight='bold', zorder=21)
    fig.text((p0.x0 + p0.x1) / 2, p0.y0 - 0.3 / fig_h, "Training Progress", ha='center', va='top', fontsize=16)  # 11-16: under its ticks
    y_ab = p0.y0 - 0.925 / fig_h                        # 0.925 in below the bottom row of (a)
    fig.text(x_a, y_ab, "(a)", ha='center', va='top', fontsize=20)
    fig.text((q.x0 + q.x1) / 2, y_ab, "(b)", ha='center', va='top', fontsize=20)
    for ax3, letter in zip(axes_3d, "cd"):
        r = ax3.get_position()
        fig.text((r.x0 + r.x1) / 2, 0.045, f"({letter})", ha='center', va='center', fontsize=20)
    out = fig_path("fig12_default_rcc_paper" + ("_top" if top_only else "") + ".pdf")
    fig.savefig(out)
    fig.savefig(out[:-4] + ".png", dpi=100)
    print(out)


if __name__ == "__main__":
    main()
