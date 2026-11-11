"""fig_learning_dynamics_forward.png: the paper's learning-dynamics figure (Fig. 13 of the submission, Fig. 15 of the
11-08 draft) with the paper's FORWARD Factorization Regret (John 11-10: "Can you please show me the fixed version of
Fig. 13"). A PNG twin, for looking at, of what supplementary_plotter2.py now writes as diagnostics_panel_early.svg:
the same functions draw it (plot_diagnostics_early, mark_bounds, finalize_layout_early on diagnostics_runs()).

What was wrong: the submission's panels d-f printed each test's LOGGED LOGISTIC SLOPE of correctness at T on env.SII,
labelled "r (Correlation)" (printed curve = the logs to 1e-4: fr_definition_check.py, panel e), and the 11-08 draft the
Pearson r that slope implies. env.SII was the symmetric KL with probabilities clipped at 1e-15, which is all the runs
logged, and a logged fit cannot be converted to the forward KL.

What is drawn now (2 x 3, the draft's layout):
  top, the first 800 testing episodes    accuracy at T | the FR-correctness correlation | PR(RNN) - PR(read-in)
  bottom, all testing episodes           accuracy at T | the correlation against accuracy | against the PR gain
All from the two standard runs whose checkpoints were kept (training_dynamics/trained_s20261029 and echo_s20261029,
the paper's configuration): accuracy and the PR gain from every test of the run (10,001 tests of 8,000 trajectories);
the correlation = Pearson r between FR_T = KL(Joint_T || Naive_T) of the goal belief (exact, log space) and correctness
at T over 10,000 fixed test trajectories (the probe set: 50 test environments x 100 cells x 2), at each of the 114
saved checkpoints (dots; 50 of them in the first 800 tests); dotted: the two observers' own r on those trajectories.
The submission drew the accuracy and PR panels from its original pair of runs (same configuration, other seeds), of
which only the final weights exist, so a forward correlation cannot be replayed on them.

RESULTS (11-10, --numbers)
  r at the end of training (49,995 raw episodes): Trained RNN -.058, Echo -.265; Joint Bayes -.061, Naive Bayes -.322.
  Trained RNN: -.106 at 500 raw episodes (test 100), most negative -.125 at 700 (test 140; accuracy .33, before it
  passes Naive Bayes's .46), back at Joint Bayes's own level by ~3,000 (test 600) and there to the end. Echo: -.086 at
  500, -.215 at 4,000 (test 800), still falling slowly toward Naive Bayes's at the end (minimum -.269 at 49,000).
  The same replay with the symmetric clipped KL: Trained -.028, Echo -.324 (Joint -.030, Naive -.433); with the forward
  KL clipped at 1e-15: -.055, -.298. Same shape under all three.
  Wording: "this correlation dissolves for the Fully Trained network" holds for the symmetric KL only; with the paper's
  own definition the Trained RNN ends at Joint Bayes's level, r = -.06, not at zero.
An every-test correlation needs both trainings rerun with the forward KL logged (env.SII is the forward KL since 11-10):
about 7 h per network on a free GPU with the lean path, 18-21 h with the recorder.
Data: supplementary_plotter2.diagnostics_runs() (the runs' test files + fr_definition_check's fr_correct cache). Seconds.
usage:  python fig_learning_dynamics_forward.py [--numbers]"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from main.utils import fig_path                                                   # noqa: E402
import supplementary_plotter2 as sp                                               # noqa: E402


def figure(path=None):
    trained, echo = sp.diagnostics_runs()
    fig, axes = plt.subplots(2, 3, figsize=sp.PLOT_CFG["figsize_A_early"], squeeze=False)
    sp.plot_diagnostics_early(axes, trained, "Trained")
    sp.plot_diagnostics_early(axes, echo, "Echo")
    sp.mark_bounds((trained, echo), y_axes=(axes[0, 0], axes[1, 0]), x_axes=(axes[1, 1],))
    sp.finalize_layout_early(axes)
    sp.note_counts(fig, trained, y=-.05)
    path = path or fig_path("fig_learning_dynamics_forward.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)
    return path


def numbers():
    import fr_definition_check as fd
    Z = fd.data("fr_correct")
    for run in fd.RUNS:
        raw, r = Z[f"{run}_raw"], Z[f"{run}_fwd"]
        j = int(np.argmin(r))
        print(f"{run}: {len(raw)} checkpoints ({int((raw <= 4000).sum())} in the first 800 tests); r forward exact at raw "
              + ", ".join(f"{int(raw[i])}: {r[i]:+.3f}" for i in np.searchsorted(raw, [0, 500, 1000, 4000, 49995]))
              + f"; minimum {r[j]:+.3f} at raw {int(raw[j])} (accuracy {Z[f'{run}_acc'][j]:.3f})")
        print("   end of training: forward exact %+.3f | forward clipped %+.3f | symmetric clipped %+.3f"
              % (r[-1], Z[f"{run}_fwd_clip"][-1], Z[f"{run}_sym_clip"][-1]))
    print("observers on the same trajectories (forward | symmetric): Joint %+.3f | %+.3f, Naive %+.3f | %+.3f; "
          "accuracy %.3f / %.3f" % (Z["bayes_fwd"], Z["bayes_sym_clip"], Z["naive_fwd"], Z["naive_sym_clip"],
                                    Z["acc_bayes"], Z["acc_naive"]))


if __name__ == "__main__":
    numbers() if "--numbers" in sys.argv else figure()
