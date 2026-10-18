"""Lazy/rich RNN (mode "lazyrich", Clark, Bordelon, Zavatone-Veth & Pehlevan): one network per gamma.
gamma -> 0+ is lazy (reservoir-like), large gamma is rich. Everything else is held fixed."""
import os
import sys
import inspect
import contextlib
import threading
import queue
from multiprocessing import Manager
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from joblib import Parallel, delayed, parallel_config
root = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
while not os.path.exists(os.path.join(root, 'main', 'CognitiveGridworld.py')):
    root, prev = os.path.dirname(root), root
    if root == prev:
        raise RuntimeError("project root not found")
sys.path.insert(0, root)
from main.CognitiveGridworld import CognitiveGridworld
from main.utils import fig_path, apply_fig_style

SAVE_DIR = os.path.join(root, 'main', 'DATA', 'lazyrich')

BASE = dict(
    mode = "lazyrich", training = True, show_plots = False, learn_embeddings = False, reservoir = False, skip_training_analyses = True, gpu_inference = True,
    cuda = 1, state_num = 500, ctx_num = 2, obs_num = 5, realization_num = 10, step_num = 30, checkpoint_every = 50,
    hid_dim = 5000, 
    batch_num = 5000, 
    episodes = 250000, # 50000,
    classifier_LR = 1,      # eta_0; N*gamma^2 is applied automatically
    rnn_gain = 1.5,         # g
    rnn_dt = 0.4,           # only dt/tau matters (tau = 1)
    rnn_beta = 1e7,         # Langevin temperature; the class default is 1e6
    rnn_skip = False,       # memoryless bypass past the RNN: tanh(U I) with its own readout
)

def train(gamma, q, gpus):
    cuda = gpus.get()       # borrow a free GPU slot for this run, give it back when done
    def report(self):
        test = (self.test_accs[self.test_e - 1, -1], self.naive_acc.mean(0)[-1], self.joint_acc.mean(0)[-1]) if self.test_set else None
        q.put((gamma, cuda, self.e, test))
        return {}

    name = (f"N{BASE['hid_dim']}{'_skip' if BASE['rnn_skip'] else ''}{'_frozenJ' if BASE['reservoir'] else ''}"
            f"_gamma{gamma}")                                                       # e.g. N20000_skip_gamma0.05, N5000_frozenJ_gamma1
    try:
        os.makedirs(SAVE_DIR, exist_ok = True)
        with open(os.path.join(SAVE_DIR, f"log_{name}.txt"), "w", buffering = 1) as f, \
             contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            self = CognitiveGridworld(**dict(BASE, cuda = cuda), gamma = gamma, save_env = f"/lazyrich/{name}", external_logger = report)
    finally:
        gpus.put(cuda)
    return dict(acc = self.test_accs[:self.test_e, -1], joint = self.joint_acc.mean(0)[-1], naive = self.naive_acc.mean(0)[-1])


def train_all(gammas, per_gpu = 1, cudas = None):
    """cudas lists the GPUs to use (default [BASE['cuda']]); each runs up to per_gpu gammas at once.
    Queued gammas take whichever slot frees up first."""
    cudas = cudas or [BASE['cuda']]
    slots = [c for _ in range(per_gpu) for c in cudas]      # per_gpu 2, cudas [0, 1] -> 0, 1, 0, 1
    n_jobs = min(len(slots), len(gammas))
    out = {}
    with Manager() as m:
        q, gpus = m.Queue(), m.Queue()
        for c in slots[:n_jobs]:
            gpus.put(c)
        bars = {g: tqdm(total = BASE['episodes'], desc = f"gamma {g}", position = i) for i, g in enumerate(gammas)}
        def run():
            try:
                # inner_max_num_threads stops each worker's BLAS pool from grabbing every core.
                with parallel_config(backend = "loky", inner_max_num_threads = max(1, (os.cpu_count() or 8) // n_jobs)):
                    out['runs'] = Parallel(n_jobs = n_jobs)(delayed(train)(g, q, gpus) for g in gammas)
            except Exception as err:
                out['err'] = err
        t = threading.Thread(target = run)
        t.start()
        while t.is_alive() or not q.empty():
            try:
                g, cuda, e, test = q.get(timeout = 1)
            except queue.Empty:
                continue
            if e == 0:
                bars[g].reset()     # a queued gamma's clock starts when it does, not when the bar was drawn
                bars[g].set_description(f"gamma {g} (cuda {cuda})")
            bars[g].update(e + 1 - bars[g].n)
            if test is not None:
                bars[g].set_postfix_str("acc {:.3f} | naive {:.3f} | joint {:.3f}".format(*test))
        for b in bars.values():
            b.close()
    if 'err' in out:
        raise out['err']
    return out['runs']


if __name__ == "__main__":
    gammas = [.05] # [1, .1, .01]
    runs = train_all(gammas, per_gpu = 1, cudas = [1])

    apply_fig_style()
    fig, ax = plt.subplots(figsize = (4.5, 3))
    for gamma, r in zip(gammas, runs):
        x = np.minimum(np.arange(len(r['acc'])) * BASE['checkpoint_every'], BASE['episodes'] - 1)
        ax.plot(x, r['acc'], label = rf"$\gamma$ = {gamma}")
        print(f"gamma {gamma:<5} acc {r['acc'][-1]:.3f}  naive {r['naive']:.3f}  joint {r['joint']:.3f}")
    ax.axhline(np.mean([r['naive'] for r in runs]), ls = '--', c = 'gray', lw = 1, label = 'naive Bayes')
    ax.axhline(np.mean([r['joint'] for r in runs]), ls = '--', c = 'k', lw = 1, label = 'joint Bayes')
    ax.set_xlabel("training episodes")
    ax.set_ylabel("test accuracy")
    ax.legend(loc = "lower right")
    fig.tight_layout()
    fig.savefig(fig_path("lazyrich_gamma_sweep.png"), bbox_inches = "tight")
    plt.show()
