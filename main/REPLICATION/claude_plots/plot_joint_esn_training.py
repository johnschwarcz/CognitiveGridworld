"""Accuracy through training: the 2-latent SANITY networks, both ESNs trained again from scratch with an entropy bonus
(and the joint one's control), and the two reservoirs continued under several variants (the joint-output ESN overnight
2026-10-21/22, 14,000 updates; extended to 54,000 on 2026-10-22, with the marginal-output ESN added). One panel. Works on
partial data: each run is drawn up to its latest test (the legend then shows its progress).

    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/plot_joint_esn_training.py

CPU only, a few seconds. Writes figure_bin/joint_esn_training.png.

Networks: mode SANITY, ctx 2, hid 1000, batch 8000, Adam lr 1e-3, 50,000 episodes with a test episode every 500
(main/REPLICATION/trainings/train_sanity_networks.py, train_sanity_update_joint.py):
  Trained RNN        main/DATA/sanity/fully_trained_ctx_2_net.pth    reservoir False
  ESN                main/DATA/sanity/reservoir_ctx_2_net.pth        reservoir True (frozen LSTM, trained read-in / out),
                                                                     marginal output (a softmax per latent)
  ESN, joint output  main/DATA/sanity/joint_reservoir_ctx_2_net.pth  as the ESN, output_joint True (100 cells)
These three share one schedule. Figure 1's two networks (the _e5 runs: a test every 5 episodes, so 40,000 updates) end
at the same levels (harvest: Trained 0.760, ESN 0.431, means of their last 1000 tests).
Continuations, appended where the original training stopped (Adam restarted: the checkpoints hold no optimizer state;
same batch, lr, fp16 autocast):
  ESN, joint output: goal loss (control)   its own loss (SANITY_loss on the goal marginal), trained initial state
  ESN, joint output: joint loss            KL(optimal joint posterior || model joint posterior) over the 100 cells;
                                           not drawn since 2026-11-01: the from-scratch joint-loss run below replaces
                                           it (John: "if so we can remove it from continued")
  ESN: both latents' loss                  the marginal ESN, its own loss on both latents' marginals (held-out test-state
                                           latents masked; run_variant.py --loss both), trained initial state
Not drawn since 2026-11-01 (John: "remove the entropy bonus curves now that we have the joint-loss"): the three
from-scratch entropy runs below and their control (redundant with the joint-output ESN, as on the frozen-widths
figures). Trained from scratch with the native recipe (added 2026-10-27; learn5/fullesn_ent/run_variant_ent.py, a copy of
run_variant.py): a new read-in, read-out AND start state (the fresh model's own randn STM / LTM, trainable, as in the
original trainings), the originals' frozen reservoirs and environments, own loss, batch 8,000 (micro-batches of 4,000
joint / 2,000 marginal), Adam 1e-3, fp16, seed 20261021 (read-in / read-out bit-identical to the start-0 scratch run's):
  ESN, joint output, from scratch: entropy bonus   own loss - 0.03 x mean (over episodes and steps) entropy of the belief
                                                   over the 100 cells (joint_native_ent0.03, 20,000 updates on 10-27,
                                                   continued to 50,000 from 10-28, see below)
  ESN, joint output, from scratch, no bonus (control)   the same without the bonus (joint_native, 10,000 updates);
                                                   red dashed beside the bonus run (in purple it hid under the original)
  ESN, from scratch: entropy bonus                 marginal output; the bonus on the product of its two marginals
                                                   (marginal_native_ent0.03, 20,000 updates, continued as the joint one)
  ESN, joint output, from scratch, joint loss   the joint loss from update 0 (joint_echo_jl_s20261029 in
                                                   DATA/experimental/training_dynamics: train_training_dynamics.py, another
                                                   session, 10-30; seed 20261029's world, joint Bayes .7677 / naive .4471 on
                                                   its tests; a test every 5th episode, which trains nothing, so 50,000
                                                   episodes = 40,000 updates); drawn, as every curve here, by one test
                                                   per 500 updates (td_log: the nearest; 2026-11-01), its legend value the
                                                   last test; its Bayes accuracies stay out of the reference lines
  Data: main/DATA/experimental/esn_learnability/full_esn_entropy/<name>/log.jsonl (a test every 500 updates on 8,000
  test-state episodes; values in the legend: the latest test, as these runs have no fixed-episode evaluations).
  The two entropy runs were continued to 50,000 updates (John 2026-10-28; the original trainings' length; finished
  10-28 16:27 joint, 17:06 marginal) in full_esn_entropy/continued/<name>/, resumed from a copy of their ckpt_20000.pt
  with Adam, the loss scaler and every RNG state, so the same run as an uninterrupted one; their log.jsonl holds every
  test since update 0 and is read first. The control was not continued (10,000 updates).
Not drawn: every start-state-0 run. John 2026-10-28 removed the two that were drawn: the joint ESN trained again from
scratch for 50,000 updates with its start state fixed at 0 (joint_scratch_zero, run_variant.py --init scratch0; 0.620 on
the fixed episodes) and the marginal ESN continued with start state 0 (marginal_zero; 0.446). Never drawn: the joint
ESN's goal-loss start-state-0 continuation (orig_zero, 0.625 at 54,000), its joint-loss start-state-0 run (joint_zero,
14,000 updates: the same as the joint-loss run, 0.642 vs 0.643), and the marginal ESN's control (dropped before it
started).
Data, main/DATA/experimental/interaction_bounds/analyses/:
  overnight_joint_esn/    the 14,000-update runs (run_variant.py, eval_run.py), reference_eval[_zero].json
  joint_esn_extension/    the extension: runs/<name>/log.jsonl (every test since update 0), eval_<update>.json, DONE;
                          reference_eval_marginal[_zero].json
A run is read from joint_esn_extension when it is there, else from overnight_joint_esn.

Accuracy: argmax accuracy of the goal value at the last step (t = 30) on each test episode: 8,000 episodes of the
network's held-out test states, forward only. This is test_acc_through_training[:, -1] in the .pth files and acc_T in
log.jsonl, the same quantity (model_acc.mean(0)[-1] of the native code). The values in the legend: the final accuracy of
the same networks on 20,000 fixed harvest episodes (eval_<update>.json, from another environment). x: gradient updates.
A test episode replaces an update, so test k of the original runs follows 499 k updates and the continuations start at
49,900.
Bounds: the optimal (joint) and naive Bayes observers on the joint-output ESN's runs' test episodes (its environment),
0.769 and 0.448. On the 20,000 harvest episodes: 0.769 and 0.451.
"""
import os, sys, json, glob, time, inspect
import numpy as np


def _repo_root():
    """the folder holding main/CognitiveGridworld.py, above this file; __file__ can be missing or a placeholder that
    does not exist in an interactive window, so the working directory is tried next"""
    here = None
    try:
        here = inspect.getfile(inspect.currentframe())
    except (TypeError, OSError):
        pass
    for start in (globals().get("__file__"), here, globals().get("__vsc_ipynb_file__"), os.getcwd()):
        d = os.path.abspath(start) if start and os.path.exists(start) else None
        while d:
            if os.path.exists(os.path.join(d, "main", "CognitiveGridworld.py")):
                return d
            d = None if os.path.dirname(d) == d else os.path.dirname(d)
    raise RuntimeError("project root (the folder holding main/CognitiveGridworld.py) not found")


ROOT = _repo_root()
sys.path.insert(0, ROOT)
import matplotlib.pyplot as plt                                                  # noqa: E402
from main.utils import fig_path, apply_fig_style                                 # noqa: E402

SANITY = os.path.join(ROOT, "main", "DATA", "sanity")
AN = os.path.join(ROOT, "main", "DATA", "experimental", "interaction_bounds", "analyses")
OV14, EXT = os.path.join(AN, "overnight_joint_esn"), os.path.join(AN, "joint_esn_extension")
ENT = os.path.join(ROOT, "main", "DATA", "experimental", "esn_learnability", "full_esn_entropy")
TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")   # train_training_dynamics.py (another session)
CHECKPOINT_EVERY = 500                                    # the original runs' test cadence, in episodes
C_TR, C_EC, C_JE, C_JL = "#1f77b4", "#ff7f0e", "#7b3294", "#1b9e77"   # Trained, ESN (as Figure 1), joint ESN, joint loss
C_MB = "#a6611a"                                                               # marginal ESN, both latents' loss
C_ENT, C_ENTM = "#d7191c", "#c51b7d"       # from scratch with the entropy bonus: joint (its control dashed), marginal ESN
C_OPT, C_NAIVE, DASH, DOT = "k", "0.5", (0, (3, 1.6)), (0, (1, 1.3))
ORIG = (("Trained RNN", "fully_trained_ctx_2", C_TR), ("ESN", "reservoir_ctx_2", C_EC),
        ("ESN, joint output", "joint_reservoir_ctx_2", C_JE))
RUNS = (("ESN, joint output, from scratch, joint loss", "joint_echo_jl_s20261029", "joint", C_JL, "-", False),
        ("ESN, joint output: goal loss (control)", "orig_trained", "joint", C_JE, "-", True),
        ("ESN: both latents' loss", "marginal_both_trained", "marginal", C_MB, "-", True))
#        label, name, net, colour, line style, continued (appended at the end of the original training)
TOTAL = {"orig_trained": 54000, "joint_trained": 54000, "marginal_both_trained": 54000, "joint_native": 10000,
         "joint_native_ent0.03": 50000, "marginal_native_ent0.03": 50000,
         "joint_echo_jl_s20261029": 40000}                                    # each run's planned updates
X_CONT = 54000                                                                 # the continuations' length


def original(stem):
    """(updates before each test, final-step test accuracy) of one original run"""
    import torch
    d = torch.load(os.path.join(SANITY, stem + "_net.pth"), map_location="cpu", weights_only=False)
    acc = np.asarray(d["test_acc_through_training"], np.float64)[:, -1]
    return np.arange(len(acc)) * (CHECKPOINT_EVERY - 1), acc


def run_dir(name):
    """the extension's copy of a run when there is one, else the 14,000-update run, else the 10-27 from-scratch runs
    (their continuation, ENT/continued/<name>, first, then ENT/<name>); None if none has started"""
    for d in (os.path.join(EXT, "runs", name), os.path.join(OV14, "runs", name), os.path.join(ENT, "continued", name),
              os.path.join(ENT, name)):
        if os.path.exists(os.path.join(d, "log.jsonl")):
            return d
    d = os.path.join(TD, name)                                    # a training_dynamics run (test_*.npz, run.json)
    return d if os.path.exists(os.path.join(d, "run.json")) else None


def td_log(d, every_n=500):
    """a training_dynamics run as continuation() returns a log, tested as every other curve here (John 2026-11-01: the
    same test sampling in one figure): of its tests (every 5th episode, which trains nothing) the one nearest each
    multiple of `every_n` updates (within 4), with the updates before it, its accuracy and Bayes's on the same batch;
    and the run's last write"""
    every = json.load(open(os.path.join(d, "run.json")))["cfg"]["checkpoint_every"]
    z = [np.load(f) for f in sorted(glob.glob(os.path.join(d, "test_*.npz")))]
    ep = np.concatenate([q["episode"] for q in z]).astype(int)
    o = np.argsort(ep)
    trained = (ep - ep // every)[o]                                             # updates before each test
    get = lambda key: np.concatenate([q[key][:, -1] for q in z]).astype(float)[o]
    marks = np.arange(0, trained[-1] + every_n / 2, every_n)
    i = np.clip(np.searchsorted(trained, marks), 1, len(trained) - 1)
    i = np.where(np.abs(trained[i - 1] - marks) <= np.abs(trained[i] - marks), i - 1, i)
    i = i[np.abs(trained[i] - marks) <= 4]
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(os.path.join(d, "progress.log"))))
    return trained[i].astype(np.float64), get("acc")[i], get("joint_acc")[i], get("naive_acc")[i], stamp


def continuation(d):
    """a run's test log: updates, final-step test accuracy, the test batches' Bayes accuracies, last test's time"""
    if d.startswith(TD):
        return td_log(d)
    with open(os.path.join(d, "log.jsonl")) as f:
        L = [json.loads(line) for line in f if line.strip()]
    get = lambda k: np.array([r[k] for r in L], np.float64)
    return get("update"), get("acc_T"), get("opt_acc_T"), get("naive_acc_T"), L[-1]["time"]


def fixed_episodes(name, net, d):
    """(updates, final-step accuracy on the 20,000 fixed harvest episodes): the unmodified network at update 0, then
    the run's evaluated checkpoints"""
    ref = "reference_eval" + ("_marginal" if net == "marginal" else "") + ("_zero" if name.endswith("zero") else "")
    u, acc = [], []
    refs = [] if ("scratch" in name or "native" in name or d.startswith(TD)) else \
        [os.path.join(OV14 if net == "joint" else EXT, ref + ".json")]                      # from scratch: none
    for p in refs + \
             sorted(glob.glob(os.path.join(d, "eval_[0-9]*.json")), key=lambda p: int(os.path.basename(p)[5:-5])):
        if os.path.exists(p):
            with open(p) as f:
                acc.append(json.load(f)["T30"]["acc"])
            u.append(0 if "reference" in os.path.basename(p) else int(os.path.basename(p)[5:-5]))
    return np.array(u, np.float64), np.array(acc)


def main():
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"                       # math in Computer Modern, as the text
    orig = {stem: original(stem) for _, stem, _ in ORIG}
    x0 = orig["joint_reservoir_ctx_2"][0][-1]                    # the continuations start where the training ended
    assert x0 == 49900 and all(len(o[0]) == 101 for o in orig.values())
    runs, dots, done, running, stopped = {}, {}, {}, {}, {}
    for _, name, net, _, _, _ in RUNS:
        d = run_dir(name)
        if d is None:
            continue
        runs[name] = continuation(d)
        dots[name] = fixed_episodes(name, net, d)
        done[name] = runs[name][0][-1] >= TOTAL[name] or (d.startswith(TD) and open(os.path.join(d, "progress.log")).read().rstrip()
                                                             .endswith("DONE"))   # its blocks sit at their mean, < TOTAL
        st = os.path.join(d, "status.json")                     # written every 50 updates while a run trains
        started = not done[name] and (d.startswith(EXT) or d.startswith(ENT)) and os.path.exists(st)
        if started:
            with open(st) as f:
                age = time.time() - time.mktime(time.strptime(json.load(f)["time"], "%Y-%m-%d %H:%M:%S"))
        running[name] = started and age < 25 * 60
        stopped[name] = started and not running[name]                 # died, or given up by the scheduler
    live = any(not done.get(r[1], False) and not stopped.get(r[1], False) for r in RUNS)
    stamp = max(r[4] for r in runs.values())
    R = [runs[r[1]] for r in RUNS if r[2] == "joint" and r[1] in runs and not run_dir(r[1]).startswith(TD)]   # its world
    opt, naive = (float(np.mean(np.concatenate([r[i] for r in R]))) for i in (2, 3))

    W, H = 7.25, 3.5                                             # inches: one 2.9 x 2.0 panel, legends to its right
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([.62 / W, .55 / H, 2.9 / W, 2.0 / H])
    lw, fl, ft = 1.2, fs["tick"] - 1, fs["tick"] - 1.5

    def legend_text(label, name):
        """label: final accuracy on the fixed episodes (else latest test), and the run's state"""
        if name not in runs:
            return f"{label}: queued"
        u, y = runs[name][:2]
        yd = dots[name][1]
        v = f"{yd[-1]:.3f}" if len(yd) else f"{y[-1]:.3f}"                    # no fixed-episode evals: latest test
        if running[name]:
            state = f" (at {u[-1] / 1e3:.1f}k)"
        elif stopped[name]:
            state = f" (stopped at {u[-1] / 1e3:.1f}k)"
        else:
            state = "" if done[name] else f" (queued at {u[-1] / 1e3:.0f}k)"
        return f"{label}: {v}{state}"

    ax.axvspan(x0 / 1e3, (x0 + X_CONT) / 1e3, color="0.93", lw=0, zorder=0)
    ax.text((x0 + X_CONT / 2) / 1e3, .805, "continued", ha="center", va="top", fontsize=ft, color="0.35")
    for y, c in ((opt, C_OPT), (naive, C_NAIVE)):
        ax.axhline(y, color=c, ls=(0, (4, 2)), lw=1, zorder=1)
    first = [plt.Line2D([], [], color=C_OPT, ls=(0, (4, 2)), lw=1, label=f"Optimal Bayes (joint): {opt:.3f}"),
             plt.Line2D([], [], color=C_NAIVE, ls=(0, (4, 2)), lw=1, label=f"Naive Bayes: {naive:.3f}")]
    for label, stem, c in ORIG:
        x, y = orig[stem]
        first += ax.plot(x / 1e3, y, color=c, lw=lw, label=label, zorder=3 if stem.startswith("joint") else 2)
    second = []
    for label, name, net, c, ls, cont in RUNS:
        if name in runs:
            u, y = runs[name][:2]
            ax.plot((x0 * cont + u) / 1e3, y, color=c, ls=ls, lw=lw * (.85 if cont else 1.1),
                    zorder=4.5 if ls != "-" else 4)          # broken lines on top: their gaps show what is under them
            if running[name]:
                ax.plot((x0 * cont + u[-1]) / 1e3, y[-1], ">", ms=4, color=c, zorder=5)    # the run's newest test
        h = plt.Line2D([], [], color=c, ls=ls, lw=lw * (.85 if cont else 1.1), label=legend_text(label, name))
        (second if cont else first).append(h)
    ax.set_xlim(0, (x0 + X_CONT) / 1e3 + 1)
    ax.set_ylim(0, .82)
    ax.set_xlabel("Training updates (thousands; 8,000 episodes each)", fontsize=fl)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy through training\n" + r"$\Pr(\hat{r}_g = r_g)$ at $t = 30$, test episodes",
                 fontsize=fs["label"], loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    kw = dict(loc="upper left", frameon=False, fontsize=fl, handlelength=2.2, borderaxespad=0, title_fontsize=fl,
              alignment="left")
    fig.legend(handles=first, bbox_to_anchor=(3.72 / W, 3.4 / H), ncol=1, title="trained from update 0", **kw)
    fig.legend(handles=second, bbox_to_anchor=(3.72 / W, 1.36 / H), ncol=1,
               title="continued from update 49,900", **kw)
    fig.text(3.72 / W, .5 / H, "values: final accuracy on 20,000 fixed episodes;\njoint loss from scratch: "
             "last test, 8,000 episodes",
             fontsize=ft, color="0.35", va="top", linespacing=1.3)
    if live:
        fig.text(3.72 / W, .12 / H, f"live: runs in progress, newest test {stamp}", ha="left", va="bottom",
                 fontsize=ft, color="0.35")
    out = fig_path("joint_esn_training.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)

    print(f"bounds on the joint network's logged test episodes: optimal {opt:.4f}, naive {naive:.4f}")
    for label, stem, _ in ORIG:
        x, y = orig[stem]
        print(f"{label:20s} first {y[0]:.3f}  last 10 tests {y[-10:].mean():.4f}  max {y.max():.4f}")
    for label, name, net, _, _, _ in RUNS:
        if name not in runs:
            print(f"{label[:40]:40s} queued")
            continue
        u, y = runs[name][:2]
        ud, yd = dots[name]
        print(f"{label[:40]:40s} update {int(u[-1])}/{TOTAL[name]}"
              f"{' (running)' if running[name] else ' (stopped)' if stopped[name] else '' if done[name] else ' (queued)'}"
              f" | test: 0 {y[0]:.3f}, last {y[-1]:.4f}" + (f" | fixed episodes: final {yd[-1]:.4f}" if len(yd) else ""))
    print(("LIVE " if live else "") + "saved", out, time.strftime("%F %T"))


if __name__ == "__main__":
    main()
