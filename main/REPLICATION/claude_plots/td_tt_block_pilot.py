"""Causal pilot (John 11-09: "which of them are causal and which are consequences?"): do the activity changes of the
heat maps still appear when the learning signal from half of the steps is blocked from the start (learnmask *_s0 runs,
gradient blocks only, per step 50%)? Per checkpoint: the probes' hidden states u_t; the state's participation ratio
(u_t's covariance over the 10,000 probes at each step, then the mean over steps 2-30), the activity norm and turn,
their entanglement locks (the locked statistic, steps 17-30), and goal accuracy at T.
-> DATA/experimental/training_dynamics/through_training/block_pilot/<run>.npz
   main/REPLICATION/figure_bin/learn_block_pilot.png (figure())
RESULTS (11-09), at 2,400 training episodes: goal accuracy / norm lock 1-4 steps after (tau 1-4) / state PR / lag-0
norm lock:  control .700 / .097 / 43.7 / .059;  random half .695 / .094 / 44.3 / .047;  highest-FR half .608 / .066 /
42.7 / .073;  lowest-FR half .673 / .082 / 53.2 / .046;  most entangled half .672 / .089 / 46.4 / .120;  least
entangled half .683 / .091 / 51.7 / .050.  Accuracy follows the after-step response (same rank order; .672 / .673 tie),
not the state's dimension (two blocks end ABOVE the control) and not the lag-0 norm lock (highest when the most
entangled half is blocked). The highest-FR block delays the dimension rise (24.7 vs 31.1 at 880) and cuts the
after-step response by a third. One seed per run; the response's range is small (.066-.097); in the full Trained RNN
it peaks at 2,400-4,000 (.099) and falls to .050 by the end while accuracy still rises: a learning-phase signature.
Run from the repository root (GPU 0, ~10 min for the six runs; step by step to share the GPU):
    CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_block_pilot.py
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_block_pilot.py figure"""
import os, sys, time
import numpy as np
import torch
ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, ROOT + "/main/REPLICATION", ROOT + "/main/REPLICATION/claude_plots"]
import td_analysis as td
import plot_training_dynamics as ptd
import td_tt_gap as gap
OUT = os.path.join(gap.TT, "block_pilot")
RUNS = ("learnmask_ctrl_s0", "learnmask_det_rand_f50_s0", "learnmask_det_hi_fr_f50_s0", "learnmask_det_lo_fr_f50_s0",
        "learnmask_det_hi_f50_s0", "learnmask_det_lo_f50_s0")
EPS = (0, 100, 200, 300, 400, 500, 700, 900, 1100, 1300, 1500, 1800, 2100, 2400, 2700, 3000)


def one(m, P, dev):
    """step by step (GPU memory is shared with John's runs): PR of u_t's covariance over probes at steps 2-30, norm and
    turn per probe-step, log beliefs"""
    n = len(P["obs"])
    obs = torch.as_tensor(P["obs"], dtype=torch.float32, device=dev)
    zin = torch.as_tensor(P["Z"][P["env"]], device=dev).reshape(n, -1)
    W_ih, W_hh, b = td.ga.lstm_weights(m)
    h, c = td._start(m, n)
    pr, lg = [], []
    norm, turn = np.zeros((n, td.T)), np.zeros((n, td.T))
    with torch.no_grad():
        for t in range(td.T):
            x = td._readin(m, obs[:, t], zin)
            h_new, c, _ = td._cell(x @ W_ih.T + b + h @ W_hh.T, c)
            lg.append(m.classifier_readout(h_new))
            norm[:, t] = h_new.norm(dim=-1).double().cpu().numpy()
            turn[:, t] = td._turn(h, h_new)
            if t >= 1:
                ev = torch.linalg.eigvalsh(torch.cov(h_new.double().T)).clamp_min(0)
                pr.append(float(ev.sum() ** 2 / (ev ** 2).sum()))
            h = h_new
        cum = torch.stack(lg, 1).double().cumsum(1)
        lq = td.marginals(cum, m.joint_out).float().cpu().numpy()                 # (n, T, 2, R)
    return np.array(pr), norm, turn, lq


def main(runs):
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda:0")
    P = np.load(os.path.join(td.OUT, "probes.npz"))
    rev = P["ent"].astype(np.float64).mean(-1)
    goal, truth = P["goal"], P["r"]
    for run in runs:
        path = os.path.join(OUT, f"{run}.npz")
        if os.path.exists(path):
            continue
        res = {k: [] for k in ("pr", "lock_norm", "lock_turn", "acc", "norm", "turn")}
        m = td.build(run, EPS[0], dev)
        for ep in EPS:
            t0 = time.time()
            td.load_weights(m, run, ep)
            pr, norm, turn, lq = one(m, P, dev)
            res["pr"].append(pr)
            res["lock_norm"].append(ptd._xcorr(norm[:, ptd.WIN], rev[:, ptd.WIN]))
            res["lock_turn"].append(ptd._xcorr(turn[:, ptd.WIN], rev[:, ptd.WIN]))
            ar = np.arange(len(goal))
            res["acc"].append(float((lq[ar, -1, goal].argmax(-1) == truth[ar, goal]).mean()))
            res["norm"].append(norm.mean(0))
            res["turn"].append(turn.mean(0))
            print(f"{run} {ep}: PR {pr.mean():.1f} acc {res['acc'][-1]:.3f} ({time.time() - t0:.1f} s)", flush=True)
        np.savez(path, episode=np.array(EPS), x=gap.train_x(np.array(EPS)), **{k: np.array(v) for k, v in res.items()})


NAME = {"learnmask_ctrl_s0": "control", "learnmask_det_rand_f50_s0": "random half blocked",
        "learnmask_det_hi_fr_f50_s0": "highest-FR half blocked", "learnmask_det_lo_fr_f50_s0": "lowest-FR half blocked",
        "learnmask_det_hi_f50_s0": "most entangled half blocked", "learnmask_det_lo_f50_s0": "least entangled half blocked"}
STYLE = {"learnmask_ctrl_s0": ("k", "-"), "learnmask_det_rand_f50_s0": ("0.55", "-"),
         "learnmask_det_hi_fr_f50_s0": ("#2a9d6e", "-"), "learnmask_det_lo_fr_f50_s0": ("#2a9d6e", (0, (4, 2))),
         "learnmask_det_hi_f50_s0": ("#7b4fa6", "-"), "learnmask_det_lo_f50_s0": ("#7b4fa6", (0, (4, 2)))}


def figure():
    """a goal accuracy at T; b the state's participation ratio (heat-map column a's white line); c the activity norm's
    lock with entanglement 1-4 steps AFTER the entangled step (the lock that forms while the network passes naive); d the
    turn's lag-0 lock (the input's: the untrained network and the ESN have it); per blocking run through training"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from main.utils import fig_path, apply_fig_style
    import td_xcorr_lags as xl
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    i0 = int(np.flatnonzero(xl.TAU == 0)[0])
    post = (xl.TAU >= 1) & (xl.TAU <= 4)
    Z = {r: np.load(os.path.join(OUT, f"{r}.npz")) for r in RUNS}
    fig, axs = plt.subplots(1, 5, figsize=(15.5, 3.3))
    fig.subplots_adjust(left=.045, right=.8, bottom=.2, top=.78, wspace=.5)
    ae, axs = axs[4], axs[:4]
    for r in RUNS:
        z, (col, ls) = Z[r], STYLE[r]
        x = z["x"]
        for a, y in zip(axs, (z["acc"], z["pr"].mean(1), z["lock_norm"][:, post].mean(1), z["lock_turn"][:, i0])):
            a.plot(x, y, color=col, ls=ls, lw=1.8, label=NAME[r])
    for a, (t, f) in zip(axs, (("a   Goal accuracy at T", ""), ("b   State dimension", "PR of $u_t$, mean over steps"),
                               ("c   Norm after entangled steps", r"$r(\Vert u_{t+\tau}\Vert, E_t)$, $\tau$ = 1-4"),
                               ("d   Turn at entangled steps", r"$r(1-\cos(u_t, u_{t-1}), E_t)$, lag 0"))):
        a.set_title(t + ("\n" + f if f else ""), loc="left", fontsize=11, linespacing=1.35)
        a.set_xscale("symlog", linthresh=100)
        a.set_xticks([0, 100, 1000, 2400])
        a.set_xticklabels(["0", "100", "1k", "2.4k"])
        a.minorticks_off()
        a.set_xlabel("training episodes", fontsize=10)
        a.axvline(880, color="0.6", lw=.8, ls=":")
        a.spines[["top", "right"]].set_visible(False)
    for a in axs[2:]:
        a.axhline(0, color="0.7", lw=.8)
    for r in RUNS:                                                    # e: each run at the last checkpoint (2,400)
        z, (col, ls) = Z[r], STYLE[r]
        ae.plot(z["lock_norm"][-1, post].mean(), z["acc"][-1], "o", ms=8, color=col,
                mfc=col if ls == "-" else "w", mew=1.8)
    ae.set_xlabel(r"norm after entangled steps ($\tau$ = 1-4)", fontsize=10)
    ae.set_ylabel("goal accuracy at T", fontsize=10)
    ae.set_title("e   At 2,400 training episodes\neach run: c against a", loc="left", fontsize=11, linespacing=1.35)
    ae.spines[["top", "right"]].set_visible(False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(.81, .5), frameon=False, fontsize=10)
    fig.text(.5, .02, "From-scratch runs of 11-08 (the learnmask runs): the loss gradient of half of each step's episodes "
             "blocked (gradient only); 10,000 probe episodes, 16 checkpoints per run; locks: steps 17-30 (the heat maps' "
             "statistic). Dotted: the Trained RNN passes naive (880).", ha="center", fontsize=8.5, color="0.3")
    out = fig_path("learn_block_pilot.png")
    fig.savefig(out, dpi=200)
    print("saved", out)




EPS_B = (0, 200, 400, 600, 900, 1200, 1500, 2100, 2700, 3000)


def behaviour(runs=RUNS):
    """John 11-09 ("if you found a strong lever can you show me the effect of it"): per checkpoint of the from-scratch
    blocking runs (seed 20261029: the probes' world), on the 10,000 probes at T, the goal variable: accuracy; on the probes
    where naive is wrong and Bayes right, the network's accuracy and how often it gives naive's answer (behaviour a);
    the belief error KL(O_T || q_T) of the goal variable in the probes' top and bottom half of goal FR_T
    -> block_pilot/behav_<run>.npz"""
    import td_tt_order as to
    dev = torch.device("cuda:0")
    P = np.load(os.path.join(td.OUT, "probes.npz"))
    goal, truth = P["goal"], P["r"]
    n = len(goal)
    ar = np.arange(n)
    lB = to.lsm(P["log_jb"].astype(np.float64))[ar, -1, goal]                       # (n, R) Bayes's goal belief at T
    nb_ans = P["log_nb"][ar, -1, goal].argmax(-1)
    jb_ans = lB.argmax(-1)
    tg = truth[ar, goal]
    only = (nb_ans != tg) & (jb_ans == tg)                                           # naive wrong, Bayes right
    frT = to.refs()["FR"][ar, -1, goal]
    hi = frT >= np.median(frT)
    for run in runs:
        path = os.path.join(OUT, f"behav_{run}.npz")
        if os.path.exists(path):
            continue
        res = {k: [] for k in ("acc", "acc_only", "copy_only", "kl_hi", "kl_lo", "kl")}
        m = td.build(run, EPS_B[0], dev)
        for ep in EPS_B:
            t0 = time.time()
            td.load_weights(m, run, ep)
            lq = one(m, P, dev)[3].astype(np.float64)[ar, -1, goal]
            lq = lq - np.log(np.exp(lq - lq.max(-1, keepdims=True)).sum(-1, keepdims=True)) - lq.max(-1, keepdims=True)
            ans = lq.argmax(-1)
            kl = (np.exp(lB) * (lB - lq)).sum(-1)
            res["acc"].append(float((ans == tg).mean()))
            res["acc_only"].append(float((ans == tg)[only].mean()))
            res["copy_only"].append(float((ans == nb_ans)[only].mean()))
            res["kl"].append(float(kl.mean()))
            res["kl_hi"].append(float(kl[hi].mean()))
            res["kl_lo"].append(float(kl[~hi].mean()))
            print(f"{run} {ep}: acc {res['acc'][-1]:.3f} beyond-naive {res['acc_only'][-1]:.3f} copy {res['copy_only'][-1]:.3f}"
                  f" KL hi/lo FR {res['kl_hi'][-1]:.3f}/{res['kl_lo'][-1]:.3f} ({time.time() - t0:.1f} s)", flush=True)
        np.savez(path, episode=np.array(EPS_B), x=gap.train_x(np.array(EPS_B)), n_only=int(only.sum()),
                 **{k: np.array(v) for k, v in res.items()})


if __name__ == "__main__":
    if sys.argv[1:] == ["figure"]:
        figure()
    elif sys.argv[1:] == ["behaviour"]:
        behaviour()
    else:
        main(sys.argv[1:] or RUNS)
