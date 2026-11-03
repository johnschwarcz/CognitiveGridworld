"""Scan for "flipping gradients" through training (John 11-01, on the memory-use fifths: "The flipping gradient in the
trained around when it surpasses Naive is exactly the kind of thing we want to see, do you see these kinds of flipping
gradients anywhere else (in other measures, other splits, other analyses, etc)").
-> data main/DATA/experimental/training_dynamics/analysis/flip_scan_paired.npz; printed table of the flips found

For every run (trained, echo), measure (per particle-step, steps 2-30, all 114 checkpoints) and split, the particle-
steps are split into 5 fifths ranked within each step (2,000 particles of every step per fifth;
fig_entanglement_driver.within_step) and the gradient G = mean(top fifth) - mean(bottom fifth) is taken in the
measure's own units at every checkpoint, on all probes and separately on environments 0-24 and 25-49 (halves A, B).
Splits: entanglement (a variable's step_dkl; rev = its mean over the two variables), factorization regret, Bayes's
update size J(pi_{t-1}, pi_t) of a variable's marginal, Bayes's confidence (its max marginal posterior); each
ranked either as is ("across": particles at the same step) or after removing its episode's and its step's means
("within": the same step, more or less than the episode's usual).
Measures: correctness, KL(Bayes || network) of the marginal beliefs, the network's belief entropy, the update error
and the memory cost (td_entanglement_memcut.py var: per variable); |u|, |c|, turn of u and c (1 - cos with the
previous step), u's per-particle effective dimension (td_analysis); and on the update-half probes (5,000, error steps
17-30): credit coherence, |dW|^2, dW's effective dimension, cross-band share (per particle-step: whole network).
Pairing (John 11-01, after the FR-correctness correlation: no averaging over the variables before pairing): a
per-variable measure is split by the SAME variable's entanglement / FR / update size / confidence (both variables'
values ranked together within each step: 4,000 of every step per fifth); a whole-network measure by the split
variable averaged over the two variables (rev for entanglement; rule fr-average-both-variables), labelled so.
Versions: flip_scan.npz (first, 11-01) averaged both sides over the variables (cross terms: one variable's FR with
the other's correctness); flip_scan_own.npz paired FR / update size / confidence but still split correctness and the
(variable-averaged) update error / memory cost by rev; this version pairs everything per variable.
A flip: G is reliably of one sign (|G| > 2 sigma, sigma = the median over checkpoints of |G_A - G_B| / 2) for at least
RUN consecutive checkpoints and later reliably of the other sign for at least RUN consecutive checkpoints; reported
with the checkpoints where each run ends / starts, the median G in each, and the share of those checkpoints where both
halves have G's sign.
Run from the repository root (CPU, ~3 min):
    /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_flip_scan.py"""
import os
import sys
import glob
import time
import numpy as np
from scipy.special import logsumexp

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import fig_entanglement_driver as fd                                              # noqa: E402

AN = fd.AN
S = slice(1, 30)
N_BIN, RUN = 5, 5
RUNS = ("trained", "echo")
MEASURES = ("correct", "kl_bayes_net", "entropy", "update_error", "memory_cost", "norm_u", "norm_c", "turn_u", "turn_c",
            "effdim_u", "cr_coh", "cr_E", "cr_dim", "cr_bandx")
CR = ("cr_coh", "cr_E", "cr_dim", "cr_bandx")


def _jeff(la, lb):
    pa, pb = np.exp(la), np.exp(lb)
    return .5 * ((pa - pb) * (la - lb)).sum(-1)


def split_vars(P):
    """the four split variables per particle-step and variable (n, T, 2)"""
    lj = P["log_jb"].astype(np.float64)
    lj = lj - logsumexp(lj, -1, keepdims=True)
    prior = np.concatenate([np.full_like(lj[:, :1], -np.log(lj.shape[-1])), lj[:, :-1]], 1)
    return {"entanglement": P["ent"].astype(np.float64), "FR": P["fr"].astype(np.float64),
            "update size": _jeff(prior, lj), "confidence": np.exp(lj).max(-1)}


def _per_var_bins(x):
    """(n, T, 2) -> (n, 29, 2): fifths of both variables' values ranked together within each step 2-30"""
    n = x.shape[0]
    b = fd.within_step(np.concatenate([x[..., 0], x[..., 1]], 0), N_BIN)           # (2n, 29)
    return np.stack([b[:n], b[n:]], -1)


def bins(P):
    """{split name: {"step": (n, 29) fifth index of the per-step split (per-variable splits: the mean over the two
    variables), "var": (n, 29, 2) fifth index per variable (entanglement: its per-step fifth for both)}}, splits =
    variable x (across, within)"""
    out = {}
    for name, x in split_vars(P).items():
        for rank in ("across", "within"):
            f = (lambda v: v) if rank == "across" else (lambda v: np.pad(fd_two_way(v), [(0, 0), (1, 0)] + [(0, 0)] * (v.ndim - 2)))
            if x.ndim == 2:
                b = fd.within_step(f(x), N_BIN)
                out[f"{name} | {rank}"] = {"step": b, "var": np.repeat(b[..., None], 2, -1)}
            else:
                out[f"{name} | {rank}"] = {"step": fd.within_step(f(x.mean(-1)), N_BIN), "var": _per_var_bins(f(x))}
    return out


def fd_two_way(x):
    """x (n, T[, 2]) at steps 2-30 minus its episode's and its step's means (per variable)"""
    x = x[:, S].astype(np.float64)
    return x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + x.mean((0, 1), keepdims=True)


def measures(run, k, ck, P, memcut):
    """{measure: (values, rows, steps)}: values (rows, steps) aligned to probes rows and steps (0-based, within 2-30);
    per-variable measures (rows, steps, 2)"""
    n = len(P["goal"])
    allrows, allsteps = np.arange(n), np.arange(1, 30)
    out = {}
    out["correct"] = ck["right"][:, S].astype(np.float64)
    lq = ck["logq"][:, S].astype(np.float64)
    lq = lq - logsumexp(lq, -1, keepdims=True)
    lj = P["log_jb"][:, S].astype(np.float64)
    lj = lj - logsumexp(lj, -1, keepdims=True)
    pj = np.exp(lj)
    out["kl_bayes_net"] = (pj * (lj - lq)).sum(-1)
    out["entropy"] = -(np.exp(lq) * lq).sum(-1)
    out["update_error"] = memcut["err_int"][k][:, S].astype(np.float64)              # (n, 29, 2) per variable
    out["memory_cost"] = memcut["err_cut1"][k][:, S].astype(np.float64) - out["update_error"]
    for key in ("norm_u", "norm_c", "turn_u", "turn_c", "effdim_u"):
        out[key] = ck[key][:, S].astype(np.float64)
    res = {m: (v, allrows, allsteps) for m, v in out.items()}
    if "cr_coh" in ck.files:
        rows, steps = ck["cr_rows"], ck["cr_steps"] - 1                              # 0-based steps 16..29
        for key in CR:
            res[key] = (ck[key].astype(np.float64), rows, steps)
    return res


def bins_for(Bs, v, rows, steps):
    """the fifth index aligned to values v: per variable for a per-variable measure, else per step"""
    return (Bs["var"] if v.ndim == 3 else Bs["step"])[rows][:, steps - 1]


def gradient(v, b, half):
    """top-minus-bottom fifth mean over all and over each half; v, b (rows, steps[, 2]), half (rows,) bool"""
    out = []
    for sel in (np.ones(len(v), bool), half, ~half):
        vv, bb = v[sel], b[sel]
        top, bot = vv[bb == N_BIN - 1].mean(), vv[bb == 0].mean()
        out.append(top - bot)
    return out


def scan():
    P = dict(np.load(os.path.join(AN, "probes.npz")))
    B = bins(P)
    half = P["env"] < 25
    res = {}
    t0 = time.time()
    for run in RUNS:
        files = sorted(glob.glob(os.path.join(AN, run, "ck_?????.npz")))
        memcut = np.load(os.path.join(AN, f"ent_memcut_var_{run}.npz"))
        assert len(memcut["episode"]) == len(files) and memcut["err_int"].ndim == 4
        eps = []
        for k, f in enumerate(files):
            ck = np.load(f)
            eps.append(int(ck["episode"]))
            for m, (v, rows, steps) in measures(run, k, ck, P, memcut).items():
                for sname, Bs in B.items():
                    g = gradient(v, bins_for(Bs, v, rows, steps), half[rows])
                    res.setdefault((run, m, sname), []).append(g)
        res[(run, "episode")] = eps
        print(f"{run}: {len(files)} checkpoints ({time.time() - t0:.0f} s)", flush=True)
    return res


def flips(G, GA, GB, run_len=RUN):
    """sign changes of G that are reliable (|G| > 2 sigma) for run_len checkpoints on both sides"""
    sigma = np.median(np.abs(GA - GB)) / 2
    s = np.where(G > 2 * sigma, 1, np.where(G < -2 * sigma, -1, 0))
    runs, i = [], 0
    while i < len(s):                                                              # maximal runs of one nonzero sign
        if s[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j - i + 1 >= run_len:
            runs.append((i, j, s[i]))
        i = j + 1
    out = []
    for (a0, a1, sa), (b0, b1, sb) in zip(runs[:-1], runs[1:]):
        if sa != sb:
            agree = lambda lo, hi: np.mean((np.sign(GA[lo:hi + 1]) == np.sign(G[lo:hi + 1]))  # noqa: E731
                                           & (np.sign(GB[lo:hi + 1]) == np.sign(G[lo:hi + 1])))
            out.append(dict(end=a1, start=b0, before=np.median(G[a0:a1 + 1]), after=np.median(G[b0:b1 + 1]),
                            rep=min(agree(a0, a1), agree(b0, b1)), sigma=sigma))
    return out


def main():
    res = scan()
    out = {}
    for key, v in res.items():
        out["|".join(key)] = np.array(v)
    np.savez(os.path.join(AN, "flip_scan_paired.npz"), **out)
    print("\nflips (reliable sign changes; x = training episodes):")
    for run in RUNS:
        x = fd.train_x(np.array(res[(run, "episode")]))
        for m in MEASURES:
            for sname in [k[2] for k in res if k[0] == run and k[1] == m]:
                g = np.array(res[(run, m, sname)])
                for fl in flips(g[:, 0], g[:, 1], g[:, 2]):
                    print(f"  {run:7s} {m:13s} {sname:24s} {fl['before']:+.4g} -> {fl['after']:+.4g} between "
                          f"{x[fl['end']]:7,.0f} and {x[fl['start']]:7,.0f}  (halves agree {fl['rep']:.0%}, "
                          f"sigma {fl['sigma']:.2g})")


if __name__ == "__main__":
    main()
