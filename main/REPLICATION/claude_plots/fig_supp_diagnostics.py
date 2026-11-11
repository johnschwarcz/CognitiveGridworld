"""The two supplementary figures of supplementary_plotter2.py that involve Factorization Regret, rendered without running the
plotter's whole main block (which also draws the event-dynamics panels and opens windows):

  figure_bin/diagnostics_panel_early.svg   learning dynamics (revised draft Fig. 15, submission Fig. 13): accuracy, the FR-accuracy
                                           correlation and PR(recurrent) through training (forward FR; sp.diagnostics_runs)
  figure_bin/calibration_by_tier.svg/.png  confidence-accuracy calibration (revised draft Fig. 11): (a) calibration lines,
                                           (b) accuracy / confidence by Factorization Regret (forward FR)

The calibration data come from harvests of the trained networks (main/DATA/cache/calibration_ctx<C>.npz; C = 2 is the main
supplementary harvest's networks). To rebuild one (CPU; 11-10: C = 2 with 250 x 1,000, C = 3 with 80 x 500 trajectories per
network, ~1.5 h each on one core):

    python fig_supp_diagnostics.py --harvest C EPISODES BATCH

Run from anywhere with the eureka env; MPLBACKEND=Agg. Seconds once the harvests and the main supplementary harvest exist
(this replaces the scratch driver used on 11-10, which wrote the same files)."""
import os
import sys
import time

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
import supplementary_plotter2 as sp                                               # noqa: E402


def harvest(ctx, episodes, batch):
    t0 = time.time(); out = sp.harvest_calibration(ctx, episodes=episodes, batch_num=batch)
    for tag in ("tr", "ec"):
        print(f"C = {ctx} {tag}: n {out[tag + '_fr'].size:,}; accuracy joint {out[tag + '_joint_acc'].mean():.4f} naive "
              f"{out[tag + '_naive_acc'].mean():.4f} net {out[tag + '_net_acc'].mean():.4f}; FR mean {out[tag + '_fr'].mean():.3f} "
              f"max {out[tag + '_fr'].max():.1f}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


def figures():
    # the plotter's own configuration (its main block): the cached main harvest is keyed by it
    CFG = dict(mode="SANITY", cuda=0, episodes=50, checkpoint_every=1, realization_num=10, hid_dim=1000, obs_num=5, show_plots=False,
               batch_num=10000, step_num=30, state_num=500, learn_embeddings=False, classifier_LR=.001, ctx_num=2, training=False)
    trained, echo = sp.collect_or_load(CFG, sp.DYN_PARAMS)
    diag_tr, diag_ec = sp.diagnostics_runs()
    figA2, axesA2 = plt.subplots(2, 3, figsize=sp.PLOT_CFG["figsize_A_early"], squeeze=False)
    sp.plot_diagnostics_early(axesA2, diag_tr, "Trained"); sp.plot_diagnostics_early(axesA2, diag_ec, "Echo")
    sp.mark_bounds((diag_tr, diag_ec), y_axes=(axesA2[0, 0], axesA2[1, 0]), x_axes=(axesA2[1, 1],))
    sp.finalize_layout_early(axesA2); sp.note_counts(figA2, diag_tr, y=-.05)
    figE, axesE = plt.subplots(1, 3, figsize=sp.PLOT_CFG["figsize_E"], gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
    ctxs = sp.plot_calibration_tier(figE, axesE, (trained, echo))
    figA2.savefig(sp.fig_path("diagnostics_panel_early.svg"), format="svg", bbox_inches="tight")
    figE.savefig(sp.fig_path("calibration_by_tier.svg"), format="svg", bbox_inches="tight")
    figE.savefig(sp.fig_path("calibration_by_tier.png"), dpi=200, bbox_inches="tight")
    print(f"written to figure_bin: diagnostics_panel_early.svg, calibration_by_tier.svg, calibration_by_tier.png (calibration columns {ctxs})")


if __name__ == "__main__":
    if "--harvest" in sys.argv:
        i = sys.argv.index("--harvest"); harvest(*(int(a) for a in sys.argv[i + 1:i + 4]))
    else:
        figures()
