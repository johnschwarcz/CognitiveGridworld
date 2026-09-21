import os
import sys
import inspect
import numpy as np
import contextlib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_config

path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
def find_project_root(start=None):
    d = os.path.abspath(start or path)
    while True:
        if os.path.exists(os.path.join(d, 'main', 'CognitiveGridworld.py')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("project root not found")
        d = parent

project_root = find_project_root()
sys.path.insert(0, project_root)

from main.CognitiveGridworld import CognitiveGridworld

SAVE_DIR = os.path.join(project_root, 'main', 'DATA', 'lazyrich')

BASE = dict(
    mode = "lazyrich", training = True, show_plots = False, learn_embeddings = False, reservoir = False,       
    state_num = 500, ctx_num = 2, obs_num = 5, realization_num = 10, checkpoint_every = 500,
    hid_dim          = 1000,        # N
    batch_num        = 10000,
    episodes         = 100000,
    step_num         = 10, # 30, 

    classifier_LR    = 1.0,         # eta_0; N*gamma^2 applied automatically
    rnn_gain         = 1.5,         # g, at/above the edge of chaos
    rnn_beta         = 1e7,         # temperature; keep noise floor under restructuring
    rnn_tau          = 1.0,
    rnn_dt           = 0.4,         # only dt/tau matters
    rnn_x0           = 0.0,
    rnn_grad_clip    = 10.0,        # must not bind; lazyrich_clip_rate() checks
    rnn_langevin     = True,
    rnn_nonlinearity = "tanh",
    cuda             = 1,
)


def flow_time(cfg):
    """s = lr * episodes, and the weight drift sqrt(2 s / beta) that injected noise alone
    produces over it. Reported for choosing beta, never targeted."""
    s = cfg['classifier_LR'] * cfg['episodes']
    return s, np.sqrt(2 * s / cfg['rnn_beta'])


def run_one(gamma, save = True, **overrides):
    """Train one (g, gamma) network and report the diagnostics that verify the knob works."""
    cfg = dict(BASE, gamma = gamma, **overrides)
    if save:
        os.makedirs(SAVE_DIR, exist_ok = True)
        cfg['save_env'] = f"/lazyrich/g{cfg['rnn_gain']}_gamma{gamma}_N{cfg['hid_dim']}"

    s, diff = flow_time(cfg)
    print("\n" + "=" * 78)
    print(f"lazyrich | g = {cfg['rnn_gain']}  gamma = {gamma}  N = {cfg['hid_dim']}")
    print(f"  flow time s = {s:,.0f}, noise-only drift {diff:.3f}")
    print("=" * 78)

    self = CognitiveGridworld(**cfg)
    m = self.model

    # The final episode is always a test episode, so net and Bayesian numbers below come from the same held-out-state trials.
    net, joint = self.test_accs[self.test_e - 1, -1], self.joint_acc.mean(0)[-1]
    naive = self.naive_acc.mean(0)[-1]
    drift, floor = m.lazyrich_restructuring(), m.lazyrich_noise_floor()
    ev, ev0 = np.abs(m.lazyrich_spectrum()), np.abs(m.lazyrich_spectrum(at_init = True))
    bulk = ev0.max()                                          # circular-law radius, ~g
    outliers = int((ev > bulk).sum())
    _, dlam_half = m.lazyrich_equilibration()
    
    # Store the accuracy curve for plotting
    acc_curve = self.test_accs[:self.test_e, -1]

    print(f"  accuracy    {np.round(acc_curve, 3).tolist()}")
    print(f"  final       net {net:.3f}  |  joint {joint:.3f}  naive {naive:.3f}")
    print(f"  drift       {drift:.3f}  (noise floor {floor:.3f}, excess {max(drift-floor,0):.3f})")
    print(f"  spectrum    max|lambda| {ev.max():.2f} vs bulk {bulk:.2f} at init, {outliers} outside")
    print(f"  equilibr.   2nd half: max|lambda| moved {100*dlam_half:.1f}%  (train until ~0)")
    print(f"  clip rate   {m.lazyrich_clip_rate():.3f}  (must stay ~0)")
    
    return dict(
        gamma = gamma, 
        gain = cfg['rnn_gain'], 
        net = net, 
        joint = joint, 
        naive = naive, 
        drift = drift, 
        excess = max(drift - floor, 0), 
        outliers = outliers, 
        loss = float(self.classifier_loss),
        acc_curve = acc_curve # Pack the curve data to plot later
    )


def transfer_check(gamma = 1.0, widths = (1000, 2000), **overrides):
    """muP's central claim and the sharpest correctness test: at fixed gamma and eta_0 the
    curves should be invariant to N. Systematic drift with N means one of the three scalings is
    wrong -- the 1/(N gamma) readout, g/sqrt(N) coupling, or N gamma^2 energy prefactor."""
    short = dict(episodes = 600, checkpoint_every = 100, batch_num = 2000, state_num = 200)
    out = {}
    for N in widths:
        self = CognitiveGridworld(**dict(BASE, **short, gamma = gamma, hid_dim = N,  save_env = None, **overrides))
        out[N] = np.round(self.test_accs[:self.test_e, -1], 3).tolist()
        print(f"  N = {N:>5}: {out[N]}")
        del self
    return out


def _worker(gamma, gain):
    """One grid cell in its own process. Each run is only ~17% GPU-bound and ~2.6 GiB, so the
    cells overlap well: their numpy halves run on separate cores while the GPU interleaves.
    stdout/stderr go to a per-run log so six tqdm bars do not fight over the terminal."""
    os.makedirs(SAVE_DIR, exist_ok = True)
    log = os.path.join(SAVE_DIR, f"log_g{gain}_gamma{gamma}.txt")
    with open(log, "w", buffering = 1) as f, contextlib.redirect_stdout(f), \
         contextlib.redirect_stderr(f):
        return run_one(gamma, rnn_gain = gain)


if __name__ == "__main__":
    # Chaos onset is g = 1 (Sompolinsky-Crisanti-Sommers); Clark et al. restrict to g > 1.
    # Input drive and gamma both push the effective boundary up (their Fig. 3H).
    gains  = [3]
    gammas = [1, .5, .1]
    jobs = [(gam, g) for gam in gammas for g in gains]

    # inner_max_num_threads caps each worker's BLAS pool so the numpy halves do not
    # oversubscribe the cores; without it every worker grabs all of them.
    print(f"launching {len(jobs)} runs in parallel -> {SAVE_DIR}/log_g*_gamma*.txt")
    with parallel_config(backend = "loky",  inner_max_num_threads = max(1, (os.cpu_count() or 8) // len(jobs))):
        rows = Parallel(n_jobs = len(jobs))(delayed(_worker)(gam, g) for gam, g in jobs)

    print("\n" + "=" * 86)
    print(f"{'gamma':>7}{'gain':>7}{'acc':>8}{'joint':>8}{'naive':>8}"
          f"{'excess drift':>14}{'outliers':>10}{'loss':>9}")
    for r in rows:
        print(f"{r['gamma']:>7.2f}{r['gain']:>7.2f}{r['net']:>8.3f}{r['joint']:>8.3f}"
              f"{r['naive']:>8.3f}{r['excess']:>14.3f}{r['outliers']:>10d}{r['loss']:>9.4f}")

    # One panel per gamma, one curve per gain.
    fig, axes = plt.subplots(1, len(gammas), figsize = (6 * len(gammas), 5), sharey = True)
    axes = np.atleast_1d(axes)
    for ax, gam in zip(axes, gammas):
        for r in [r for r in rows if r['gamma'] == gam]:
            x = np.arange(1, len(r['acc_curve']) + 1) * BASE['checkpoint_every']
            ax.plot(x, r['acc_curve'], label = f"g = {r['gain']}", linewidth = 2)
        ax.axhline(rows[0]['naive'], ls = '--', c = 'gray', lw = 1, label = 'naive Bayes')
        ax.axhline(rows[0]['joint'], ls = '--', c = 'k', lw = 1, label = 'joint Bayes')
        ax.set_title(f"gamma = {gam}")
        ax.set_xlabel("Training Episodes")
        ax.grid(True, linestyle = '--', alpha = 0.7)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    plt.tight_layout()
    plt.show()
