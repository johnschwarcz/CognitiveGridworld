"""Data for the theory session's questions about the training-dynamics runs (memory learning-dynamics-phases; John 11-01:
"please go through everything you wanted to analyze"). Reads td_analysis.py's per-checkpoint files (10,000 probe
episodes: 50 environments x 100 cells x 2 streams, seed-20261029 world) and the runs' own test logs, keeps only what the
questions need, and caches it in main/DATA/experimental/training_dynamics/questions/:
  ck_<run>.npz   per analysed checkpoint: raw episode "ep", training episodes "x" (a test episode trains nothing:
                 ep - ep // 5), goal accuracy at T "acc", memory cut to w = 1 / 2 / 5 and intact "trunc" (4; the cell
                 rerun from its start state over the last w inputs at every step, its own read-out summed: td_analysis
                 an6), on the particles where naive is wrong and Bayes right the share of answers equal to naive's
                 "copy_naive" and to Bayes's "copy_bayes" (goal at T), the per-step effective dimension (PR) of u and of
                 the recurrent drive W_hh u averaged over steps "pr_u" / "pr_hh" (steps 2-30 for W_hh u: step 1's is the
                 constant start state's), joint read-outs: cells more than 10 nats below the best of their row and
                 column at T "excl" (theory_v9's rule on the probe-mean logits), the true pair's gate log q(c*|g*) at T
                 per particle summarised as mean and quartiles "gate" (3) and the share of particles whose true cell
                 is below e^-10 of its goal row "gate_dead", the goal accuracy at T on the probes whose true pair is
                 not / is shut out "acc_split" (2), ridge-decoding R^2 from u "dec" (10 steps, [counts, joint
                 log posterior, marginal log posteriors]; a side statistic), read-out row gradient norms up to 3k
                 episodes "ro" (rows), the exact XL at T "xl" (KL(Bayes || network), goal)
  test_<run>.npz the run's own test log: every test episode's goal accuracy at T "acc_T" and Bayes's "bayes_T" on the
                 same episodes, raw episode "ep" (10,001 test episodes per full run, every 5th episode, each a
                 batch of 8,000 environments)
Run from the repository root (CPU):
    OMP_NUM_THREADS=4 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_questions_data.py"""
import os
import glob
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld/main/DATA/experimental/training_dynamics"
AN = os.path.join(ROOT, "analysis")
OUT = os.path.join(ROOT, "questions")
RUNS = {"trained": "trained_s20261029", "echo": "echo_s20261029", "joint_trained": "joint_trained_s20261029",
        "joint_echo": "joint_echo_s20261029", "joint_echo_jl": "joint_echo_jl_s20261029", "mlp55": "mlp55_s20261029",
        "trained_N50": "trained_N50_s20261029", "trained_N100": "trained_N100_s20261029",
        "trained_N200": "trained_N200_s20261029", "trained_p1": "trained_p1_s20261029"}
REPS = ["trained_N100_s20261030", "trained_N100_s20261031", "trained_N100_s20261032", "trained_N100_s20261033",
        "trained_lastT_s20261029", "trained_lastT_p1_s20261029"]


def gate(cell_lq, goal, r):
    """log q(c* | g*) at T for each particle from the joint read-out's cell log q (n, 100; cell = r0 * 10 + r1)"""
    lq = cell_lq.astype(np.float64).reshape(-1, 10, 10)                       # [particle, r0, r1]
    n = len(goal)
    ar = np.arange(n)
    g_true = r[ar, goal]
    c_true = r[ar, 1 - goal]
    row = np.where(goal[:, None] == 0, lq[ar, g_true, :], lq[ar, :, g_true])   # the goal value's row over c
    m = row.max(1, keepdims=True)
    lse = (m[:, 0] + np.log(np.exp(row - m).sum(1)))
    return row[ar, c_true] - lse


def checkpoints(run, P):
    files = sorted(glob.glob(os.path.join(AN, run, "ck_*.npz")))
    ar = np.arange(len(P["goal"]))
    g = P["goal"]
    truth = P["r"][ar, g]
    nb_ans = P["log_nb"][ar, -1, g].argmax(-1)
    jb_ans = P["log_jb"][ar, -1, g].argmax(-1)
    only = (nb_ans != truth) & (jb_ans == truth)
    rows = {k: [] for k in ("ep", "x", "acc", "trunc", "copy_naive", "copy_bayes", "pr_u", "pr_hh", "excl", "gate",
                            "gate_dead", "acc_split", "dec", "ro", "xl")}
    for f in files:
        z = np.load(f)
        ep = int(z["episode"])
        rows["ep"].append(ep)
        rows["x"].append(ep - ep // 5)
        rows["acc"].append(z["acc"][-1])
        rows["trunc"].append(z["trunc_acc"][:, 0])
        ans = z["logq"][ar, -1, g].astype(np.float32).argmax(-1)
        rows["copy_naive"].append((ans == nb_ans)[only].mean())
        rows["copy_bayes"].append((ans == jb_ans)[only].mean())
        fr = list(z["fac_rows"])
        fac = z["fac"]
        rows["pr_u"].append(np.nanmean(fac[:, fr.index("u"), 0, 1]))
        rows["pr_hh"].append(np.nanmean(fac[1:, fr.index("Whh_u"), 0, 1]))
        rows["excl"].append(int(z["excl_T"]))
        if "cell_lqT" in z.files:
            gl = gate(z["cell_lqT"], g, P["r"])
            rows["gate"].append(np.array([gl.mean(), *np.quantile(gl, [.25, .5])]))
            rows["gate_dead"].append((gl < -10).mean())
            right = z["right"][ar, -1, g]
            rows["acc_split"].append(np.array([right[gl >= -10].mean(), right[gl < -10].mean() if (gl < -10).any()
                                               else np.nan]))
        else:
            rows["gate"].append(np.full(3, np.nan))
            rows["gate_dead"].append(np.nan)
            rows["acc_split"].append(np.full(2, np.nan))
        rows["dec"].append(z["dec_r2"][:, 0, :])
        rows["ro"].append(z["ro_gnorm"] if "ro_gnorm" in z.files else None)
        rows["xl"].append(z["kl_pq"][-1])
    out = {k: v for k, v in rows.items() if k != "ro"}
    out = {k: np.array(v) for k, v in out.items()}
    ro = [r for r in rows["ro"] if r is not None]
    out["ro"] = np.array(ro) if ro else np.zeros((0, 0))
    out["ro_x"] = np.array([x for x, r in zip(rows["x"], rows["ro"]) if r is not None])
    return out


def test_log(folder):
    files = sorted(glob.glob(os.path.join(ROOT, folder, "test_*.npz")))
    ep, acc, bay = [], [], []
    for f in files:
        z = np.load(f)
        ep.append(z["episode"])
        acc.append(z["acc"][:, -1])
        bay.append(z["joint_acc"][:, -1])
    return {"ep": np.concatenate(ep), "acc_T": np.concatenate(acc), "bayes_T": np.concatenate(bay)}


if __name__ == "__main__":
    import sys
    os.makedirs(OUT, exist_ok=True)
    P = dict(np.load(os.path.join(AN, "probes.npz")))
    only = sys.argv[1:]                                     # optional: run keys to (re)do, e.g. joint_trained
    for key, folder in RUNS.items():
        if only and key not in only:
            continue
        np.savez(os.path.join(OUT, f"test_{key}.npz"), **test_log(folder))
        if os.path.isdir(os.path.join(AN, key)):
            d = checkpoints(key, P)
            np.savez(os.path.join(OUT, f"ck_{key}.npz"), **d)
            print(key, len(d["ep"]), "checkpoints; final acc", round(float(d["acc"][-1]), 3), flush=True)
    for folder in ([] if only else REPS):
        np.savez(os.path.join(OUT, f"test_{folder}.npz"), **test_log(folder))
        print(folder, "test log", flush=True)
