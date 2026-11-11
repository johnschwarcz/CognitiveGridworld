"""The width-10K ESN at batch 8,000 (echo_N10000_b8k, joint_echo_N10000_b8k; lean, seed 20261029, started 11-06/07) against its
batch-2,000 counterparts (the paper's Echo N = 10k, sanity_default_reps/reservoir_ctx_2_10000N_rep0; joint_echo_N10000), John
2026-11-08. Columns: marginal output, joint output with goal supervision. Top: against training updates (the figures' x-axis);
bottom: against training episodes seen (updates x batch), where batch 8,000 gets four times the data per update. Gap to joint
Bayes on log axes, as plot_matched_params_joint.py. Test data per point: batch 8,000 and the joint batch 2,000, the first 5 tests
of each 500-update block (40,000 and 10,000 episodes); the paper's marginal run, its single test (2,000).
-> main/REPLICATION/figure_bin/compare_10k_batch.png"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
import compare_matched_params as cmp                                              # noqa: E402
import plot_matched_params_new as pm                                              # noqa: E402
import plot_matched_params_joint as pj                                            # noqa: E402


def figure(step=cmp.STEP):
    from main.utils import fig_path
    cmp.apply_fig_style()
    b = dict(np.load(cmp.BOUNDS_NPZ))
    J = b[f"{cmp.BOUND_KEY}_joint"][step - 1]
    fig, axes = plt.subplots(2, 2, figsize=(20, 10.4), sharey=True)
    paper = next(n for n in cmp.NETWORKS if "10000N" in n["stem"])
    a = cmp.step_acc(paper, step)[0]
    x = (np.arange(len(a)) + 1) * cmp.CKPT_EVERY
    keep = x <= pm.XMAX
    small = {"marginal": (x[keep], a[keep], 2000, "paper run, 1 test per point")}
    xs, acs, reached = pj.pooled("joint_echo_N10000")
    small["joint"] = (xs, acs, 2000, "5 tests per point")
    big = {}
    for key, run in (("marginal", "echo_N10000_b8k"), ("joint", "joint_echo_N10000_b8k")):
        xb, ab, rb = pj.pooled(run)
        big[key] = (xb, ab, 8000, f"5 tests per point; training: {min(rb, pm.XMAX) / 1e3:.1f}k of 40k" if not pm.finished(run) else "5 tests per point")
    for col, (key, title) in enumerate((("marginal", "Marginal output, goal supervision"), ("joint", "Joint output, goal supervision"))):
        for row, per in enumerate(("updates", "episodes")):
            ax = axes[row, col]
            for (xv, av, batch, note), color, name in ((small[key], "#f4a582", "batch 2,000"), (big[key], cmp.C_ECHO, "batch 8,000")):
                xx = np.asarray(xv, float) * (batch / 1e6 if per == "episodes" else 1)
                ax.plot(xx, pm._pos(J - np.asarray(av, float)), color=color, lw=2, label=f"Echo ($N{{=}}10k$), {name} ({note})")
            pm._style(ax, J, b, step, label=(col == 0))
            if per == "episodes":
                ax.set_xlim(500 * 2000 / 1e6, pm.XMAX * 8000 / 1e6)
                ax.xaxis.set_major_locator(plt.FixedLocator([1, 2, 5, 10, 20, 50, 100, 200, 320]))
                ax.set_xlabel("Training episodes seen (millions: updates x batch)")
            ax.legend(loc="upper right", frameon=False, fontsize=12)
        axes[0, col].set_title(title, fontsize=15)
        axes[0, col].set_xlabel("Training updates")
    for row in range(2):
        axes[row, 0].set_ylabel("Optimal $-$ Network Accuracy")
    fig.tight_layout()
    path = fig_path("compare_10k_batch.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("saved", path)
    for key in ("marginal", "joint"):
        for lab, d in (("2,000", small[key]), ("8,000", big[key])):
            xv, av = np.asarray(d[0]), np.asarray(d[1])
            m = (xv > xv.max() - 2500)
            print(f"  {key:8s} batch {lab}: last update {xv.max():.0f}, mean acc over its last 2,500 updates {av[m].mean():.3f}")
        xb, ab = np.asarray(big[key][0]), np.asarray(big[key][1]); xs_, as_ = np.asarray(small[key][0]), np.asarray(small[key][1])
        common = xb.max()
        m2 = (xs_ > common - 2500) & (xs_ <= common)
        print(f"  {key:8s} at the same {common:.0f} updates: batch 2,000 {as_[m2].mean():.3f} vs batch 8,000 {ab[xb > common - 2500].mean():.3f}")


if __name__ == "__main__":
    figure()
