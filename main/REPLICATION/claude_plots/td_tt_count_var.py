"""Code panels tried for the behaviour figure (John 11-09: "can we put the decoding of N?" -- he meant the counts n, not
naive's belief N; "add the neural variance explained by count (that should capture the decoding count result,
right?)"; then "for panel e, can we switch to decoding counts?"). The figure now uses only load_counts (td_tt_count_code's
count decoders, with the ESN's missing checkpoints from harvest_counts); count_var and the posterior decoding stay as
records. All on td_tt_count_code's fresh episodes (50 test environments x 100 cells x 10 trajectories, 30 steps) at
the same checkpoints for both networks.
  count variance  the fraction of u_30's within-environment variance (pooled over units; u and the counts n_30 demeaned
                  within each environment) explained by a linear function of the 5 counts: 'shared' one map for every
                  environment (least squares on the 25 fitting environments, R^2 on the 25 test environments; the
                  encoding twin of td_tt_count_code's across-environment count decoder), 'own' each test environment
                  its own map (5-fold over trajectory pairs).
  posteriors      R^2 of naive's (N) and optimal Bayes's (O) log marginals (20 numbers, floored 30 nats below each
                  variable's maximum) decoded from u_30 across environments (td_tt_count_code.lin_x: ridge, grouped
                  5-fold CV over the fitting environments, R^2 'global' on the test environments); reused from
                  td_tt_count_code's post_<run>_<raw>.npz where it exists.
-> main/DATA/experimental/training_dynamics/through_training/count_code/behav_<run>_<raw>.npz
RESULTS (11-09; training episodes: shared / own count variance, decoded naive / optimal):
  Trained RNN  10: .59 / .63, .02 / .02;  40: .75 / .79;  100: .14 / .80, .23 / .16;  640: .017 / .46, .46 / .44;
               880: .012 / .42, .52 / .53;  2,000: .013 / .41, .57 / .71;  end: .055 / .46, .68 / .87
  ESN          end: .42 / .65, .33 / .18 (naive's belief is carried better than optimal's throughout)
  The one-map count variance collapses at ~100 training episodes (the early saturation: the state's variance taken
  over by a few large directions), long before td_tt_count_code's across-environment count DECODING dips (.88 at 40,
  .47 at 880) and it never recovers (.055), while the decoding recovers to .92: after the crossing the counts are still
  linearly readable, from directions that carry ~5% of the state's variance. The per-environment map dips with the
  decoding (.80 at 100 -> .40 at 1,040) and recovers only to .46. So the variance measure does not reproduce the
  decoding result: it tracks what dominates the state, the decoder what can be read out.
Run from the repository root (~15-35 s per checkpoint on a GPU):
    /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_tt_count_var.py 1"""
import os
import sys
import time
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import td_tt_count_code as cc                                                                 # noqa: E402
import td_tt_gap as gap                                                                       # noqa: E402

X = (10, 40, 100, 200, 320, 480, 640, 800, 880, 1040, 1200, 1600, 2000, 3000, 5000, 9600, 20000, 39996)
RUNS = ("trained", "echo")


def count_var(U, N, env, traj, fit_envs, test_envs):
    """{'shared', 'own'}: see the module docstring"""
    U, N = U.double(), N.double()
    envn = env.cpu().numpy()
    Ud, Nd = U.clone(), N.clone()
    for e in np.unique(envn):
        k = torch.as_tensor(envn == e, device=U.device)
        Ud[k] -= U[k].mean(0)
        Nd[k] -= N[k].mean(0)
    fit = torch.as_tensor(np.isin(envn, fit_envs), device=U.device)
    te = torch.as_tensor(np.isin(envn, test_envs), device=U.device)
    A = torch.linalg.lstsq(Nd[fit], Ud[fit]).solution
    sst = float((Ud[te] ** 2).sum())
    shared = 1 - float(((Ud[te] - Nd[te] @ A) ** 2).sum()) / sst
    trj = torch.as_tensor(traj, device=U.device)
    sse = 0.
    for e in test_envs:
        rows = torch.nonzero(env == int(e)).squeeze(1)
        Ue, Ne, tj = U[rows], N[rows], trj[rows]
        for f in range(5):
            va = (tj // 2) == f
            tr = ~va
            mu_u, mu_n = Ue[tr].mean(0), Ne[tr].mean(0)
            Ae = torch.linalg.lstsq(Ne[tr] - mu_n, Ue[tr] - mu_u).solution
            sse += float(((Ue[va] - mu_u - (Ne[va] - mu_n) @ Ae) ** 2).sum())
    return {"shared": shared, "own": 1 - sse / sst}


def harvest(gpu=1):
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_per_process_memory_fraction(min(1., 5 * 2 ** 30 / torch.cuda.get_device_properties(dev).total_memory),
                                               dev)
    E = cc.episodes()
    env = torch.as_tensor(E["env"], device=dev)
    Nc = torch.as_tensor(E["counts"][:, cc.T - 1], device=dev)
    Ys = None
    for run in RUNS:
        for xv in X:
            raw = cc.raw_of(run, xv)
            path = os.path.join(cc.OUT, f"behav_{run}_{raw:05d}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            S = cc.states(run, raw, E, dev)
            U = S[("u", cc.T - 1)]
            res = {"countvar": count_var(U, Nc, env, E["traj"], E["split_fit"], E["split_test"])}
            post = os.path.join(cc.OUT, f"post_{run}_{raw:05d}.npz")
            if os.path.exists(post):
                pr = np.load(post, allow_pickle=True)["res"].item()
                for k in ("naive", "bayes"):
                    res[k] = pr[("u", 30, k)]["lin_x"]
            else:
                if Ys is None:
                    Ys = {k: torch.as_tensor(v, device=dev) for k, v in cc.post_targets(E, cc.T - 1).items()
                          if k in ("naive", "bayes")}
                for k, Y in Ys.items():
                    res[k] = cc.lin_x(U, Y, env, E["split_fit"], E["split_test"])
            np.savez(path, res=np.array(res, dtype=object), x=gap.train_x(raw), raw=raw)
            del S, U
            torch.cuda.empty_cache()
            print(f"{run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.0f} s | count variance shared "
                  f"{res['countvar']['shared']:.3f} own {res['countvar']['own']:.3f} | decoded naive "
                  f"{res['naive']['global']:.3f} bayes {res['bayes']['global']:.3f}", flush=True)


def load(run):
    """x (C,), shared, own, naive, bayes (C,) sorted by x"""
    rows = []
    for xv in X:
        raw = cc.raw_of(run, xv)
        path = os.path.join(cc.OUT, f"behav_{run}_{raw:05d}.npz")
        if os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            r = z["res"].item()
            rows.append((float(z["x"]), r["countvar"]["shared"], r["countvar"]["own"], r["naive"]["global"],
                         r["bayes"]["global"]))
    a = np.array(sorted(set(rows)))
    return {k: a[:, i] for i, k in enumerate(("x", "shared", "own", "naive", "bayes"))}




def harvest_counts(gpu=1):
    """the count decoders of td_tt_count_code (lin_x across environments, lin_w within) on u_30 at the X checkpoints
    that td_tt_count_code did not harvest (most of the ESN's): count_code/countdec_<run>_<raw>.npz (John 11-09: "for
    panel e, can we switch to decoding counts?")"""
    dev = torch.device(f"cuda:{gpu}")
    torch.cuda.set_per_process_memory_fraction(min(1., 5 * 2 ** 30 / torch.cuda.get_device_properties(dev).total_memory),
                                               dev)
    E = cc.episodes()
    env = torch.as_tensor(E["env"], device=dev)
    Y = torch.as_tensor(E["counts"][:, cc.T - 1], device=dev)
    for run in RUNS:
        for xv in X:
            raw = cc.raw_of(run, xv)
            if os.path.exists(os.path.join(cc.OUT, f"{run}_{raw:05d}.npz")):
                continue
            path = os.path.join(cc.OUT, f"countdec_{run}_{raw:05d}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            S = cc.states(run, raw, E, dev)
            U = S[("u", cc.T - 1)]
            res = {"lin_x": cc.lin_x(U, Y, env, E["split_fit"], E["split_test"]),
                   "lin_w": cc.lin_w(U, Y, env, E["traj"], E["split_fit"], E["split_test"])}
            np.savez(path, res=np.array(res, dtype=object), x=gap.train_x(raw), raw=raw)
            del S, U
            torch.cuda.empty_cache()
            print(f"counts {run} {raw} ({gap.train_x(raw):,.0f}): {time.time() - t0:.0f} s | across {res['lin_x']['global']:.3f}"
                  f" within {res['lin_w']['within']:.3f}", flush=True)


def load_counts(run):
    """x (C,), across (lin_x R^2 'global'), within (lin_w R^2) of the counts n_30 decoded from u_30, sorted by x"""
    rows = []
    for xv in X:
        raw = cc.raw_of(run, xv)
        full = os.path.join(cc.OUT, f"{run}_{raw:05d}.npz")
        path = full if os.path.exists(full) else os.path.join(cc.OUT, f"countdec_{run}_{raw:05d}.npz")
        if os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            r = z["res"].item()
            r = r["u@30"] if "u@30" in r else r
            rows.append((float(z["x"]), r["lin_x"]["global"], r["lin_w"]["within"]))
    a = np.array(sorted(set(rows)))
    return {k: a[:, i] for i, k in enumerate(("x", "across", "within"))}


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "counts":
        harvest_counts(int(a[1]) if len(a) > 1 else 1)
    else:
        harvest(int(a[0]) if a else 1)
