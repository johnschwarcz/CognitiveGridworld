"""Trained networks vs echo states, measured against the Bayesian bounds.

Writes compare_matched_params_loglog.svg: excess error above the optimal bound at the
final inference step, where a power-law approach is a straight line.

Checkpoints are read on CPU. The bounds depend only on the environment, not the
network, and are cached in BOUNDS_NPZ; pass --bounds to recompute them (needs a GPU).
"""
import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap


def _repo_root():
    """Walk up from this file (or cwd) until a dir containing main/DATA is found."""
    start = os.path.abspath(globals().get("__file__") or os.getcwd())
    d = start if os.path.isdir(start) else os.path.dirname(start)
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "main", "DATA")):
            return d
        d, prev = os.path.dirname(d), d
        if d == prev:
            break
    raise RuntimeError(f"could not locate the repo root (main/DATA) from {start}")


path = _repo_root()
sys.path.insert(0, path)
from main.utils import fig_path, apply_fig_style

DATA = os.path.join(path, "main", "DATA")
BOUNDS_NPZ = os.path.join(DATA, "matched_params_bounds.npz")
BOUND_TOTAL = 300000   # episodes per environment behind the cached bounds
BOUND_KEY = "30step"   # which cached environment the bounds are read from
STEP = 30              # inference step read out
SMOOTH = 1             # checkpoints, for the log-log view only
CKPT_EVERY = 500       # training episodes per checkpoint (train_sanity_reps.py)

C_TRAIN, C_ECHO = "#2c7bb6", "#d7191c"
ECHO_LIGHT = "#fcb8b4"   # echoes are shaded along this ramp by parameter count

NETWORKS = (
    dict(label=r"Trained ($N{=}200$) $\to$ marginal belief updates",
         stem="sanity_reps_30step/fully_trained_ctx_2_200N_rep",
         reps=range(5), color=C_TRAIN, ls="-"),
    dict(label=r"Echo ($N{=}10k$) $\to$ marginal belief updates",
         stem="sanity_reps_30step/reservoir_ctx_2)19j_30step_rep",
         reps=range(1), color=None, ls="-"),
    dict(label=r"Echo ($N{=}1k$) $\to$ marginal belief updates",
         stem="sanity/reservoir_ctx_2",
         reps=None, color=None, ls="-"),
    dict(label=r"Echo ($N{=}1k$) $\to$ joint belief updates",
         stem="sanity/joint_reservoir_ctx_2",
         reps=None, color=None, ls="-"),
)

# Environments whose bounds --bounds recomputes. Only BOUND_KEY is plotted; the three
# sanity_* entries are the check that separate environment draws agree (they do, to
# within ~2 SE), which licenses one bound pair for networks from different draws.
BOUND_ENVS = {
    "30step":              ("/sanity_reps_30step/fully_trained_ctx_2_200N_rep0", 200, 30, {}),
    "30step_sanity_ft":    ("/sanity/fully_trained_ctx_2",   1000, 30, {}),
    "30step_sanity_res":   ("/sanity/reservoir_ctx_2",       1000, 30, dict(reservoir=True)),
    "30step_sanity_joint": ("/sanity/joint_reservoir_ctx_2", 1000, 30,
                            dict(reservoir=True, output_joint=True)),
}

FIGSIZE = (10, 3.6)   # one panel; wide enough for the long legend labels


# --------------------------------------------------------------------------- data

def resolve_colors(nets):
    """Networks with color=None are echo states: shaded light -> solid red by their
    trainable parameter count (log-spaced, the counts span an order of magnitude) and
    reordered among themselves so the legend runs in the same direction as the ramp."""
    nets = list(nets)
    todo = [n for n in nets if n["color"] is None]
    if not todo:
        return nets
    ramp = LinearSegmentedColormap.from_list("echo", [ECHO_LIGHT, C_ECHO])
    k = np.log([trainable_params(n) for n in todo])
    t = (k - k.min()) / np.ptp(k) if np.ptp(k) else np.zeros_like(k)
    for n, ti in zip(todo, t):
        n["color"] = ramp(ti)
    # echoes first (lightest to darkest), everything else after, so the legend runs
    # along the ramp and the trained curve sits at the bottom of it
    ids = {id(n) for n in todo}
    return [todo[i] for i in np.argsort(k)] + [n for n in nets if id(n) not in ids]


def step_acc(net, step):
    """(n_reps, n_checkpoints) of test accuracy at the 1-indexed inference `step`."""
    stem, reps = net["stem"], net["reps"]
    stems = [f"{stem}{r}" for r in reps] if reps is not None else [stem]
    return np.stack([
        np.asarray(torch.load(f"{DATA}/{s}_net.pth", map_location="cpu",
                              weights_only=False)["test_acc_through_training"],
                   dtype=float)[:, step - 1] for s in stems])


def trainable_params(net):
    """reservoir=True freezes only the LSTM (model_architecture.py:48-52)."""
    stem = net["stem"] + ("0" if net["reps"] is not None else "")
    w = torch.load(f"{DATA}/{stem}_net.pth", map_location="cpu",
                   weights_only=False)["weights"]
    res = "reservoir" in os.path.basename(stem)
    return sum(v.numel() for k, v in w.items() if not (res and k.startswith("LSTM")))


def legend_label(net):
    n = len(net["reps"]) if net["reps"] is not None else 1
    tail = f" ($n{{=}}{n}$)" if n > 1 else ""
    return f"{net['label']}, {trainable_params(net)/1e3:.0f}k trainable params{tail}"


def compute_bounds(total=BOUND_TOTAL, cuda=1):
    """Chunked to a fixed episode total; the batch shrinks when the network is big."""
    import time
    from main.CognitiveGridworld import CognitiveGridworld
    cfg = dict(mode="SANITY", cuda=cuda, episodes=1, checkpoint_every=1,
               realization_num=10, obs_num=5, show_plots=False, state_num=500,
               learn_embeddings=False, classifier_LR=.001, ctx_num=2, training=False)
    res = {}
    for tag, (env, hid, steps, extra) in BOUND_ENVS.items():
        batch = 25000 if hid * steps <= 10000 else 6000
        rounds = max(1, total // batch)
        acc = {"joint": 0., "naive": 0.}
        t0 = time.time()
        for _ in range(rounds):
            g = CognitiveGridworld(**cfg, **extra, batch_num=batch, hid_dim=hid,
                                   step_num=steps, load_env=env)
            for k in acc:
                acc[k] = acc[k] + np.asarray(getattr(g, f"{k}_acc"), float).sum(0)
            del g
        n = batch * rounds
        for k in acc:
            res[f"{tag}_{k}"] = acc[k] / n
        j = res[f"{tag}_joint"][-1]
        print(f"  {tag:<22} N={n:,}  joint {j:.4f}  naive {res[f'{tag}_naive'][-1]:.4f}  "
              f"(SE ~{np.sqrt(j * (1 - j) / n):.4f}, {time.time() - t0:.0f}s)", flush=True)
    np.savez(BOUNDS_NPZ, **res)
    return res


# --------------------------------------------------------------------------- plots

def _despine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=4, width=.9)


def plot_loglog(bounds, step=STEP, w=SMOOTH):
    """Excess error above the optimal bound: a power-law approach is a straight line.

    The optimal bound is y = 0 here, off-scale below, so the curves descend toward it
    rather than flattening onto it. The factorized bound is the constant gap between
    the two bounds, so it draws as a horizontal line.
    """
    fs = apply_fig_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    J = bounds[f"{BOUND_KEY}_joint"][step - 1]
    kern = np.ones(w) / w
    for net in resolve_colors(list(NETWORKS)):
        a = step_acc(net, step)
        m = np.convolve(a.mean(0), kern, mode="valid")
        x = (np.arange(len(m)) + (w - 1) / 2 + 1) * CKPT_EVERY
        gap = J - m
        if a.shape[0] > 1:
            se = np.convolve(a.std(0, ddof=1) / np.sqrt(a.shape[0]), kern, mode="valid")
            ax.fill_between(x, gap - se, gap + se, color=net["color"], alpha=.22, lw=0)
        ax.plot(x, gap, color=net["color"], lw=2, ls=net["ls"],
                label=legend_label(net))

    fact = J - bounds[f"{BOUND_KEY}_naive"][step - 1]
    # +/- 1 SD of the plotted gap across the independent environment draws in the
    # cache: covers Monte-Carlo error and genuine draw-to-draw variation.
    draws = [k[:-6] for k in bounds if k.endswith("_joint") and bounds[k].size == step]
    gaps = np.array([bounds[f"{t}_joint"][step - 1] - bounds[f"{t}_naive"][step - 1]
                     for t in draws])
    if gaps.size > 1:
        sd = gaps.std(ddof=1)
        ax.axhspan(fact - sd, fact + sd, color='k', alpha=.13, lw=0, zorder=9)
        print(f"  factorized bound {fact:.4f} +/- {sd:.4f} (1 SD over {gaps.size} environment draws)")
    ax.axhline(fact, c='k', lw=2, ls='--', zorder=10)
    ax.text(.01, fact, 'Naive bound', color='k', fontsize=15, va='center',
            ha='left', transform=ax.get_yaxis_transform(), zorder=11,
            bbox=dict(fc='white', ec='none', alpha=.85, pad=1.5))

    ax.set(xscale="log", yscale="log", xlim=(CKPT_EVERY, 110 * CKPT_EVERY),
           xlabel="Training Episode", ylabel="Optimal $-$ Network Accuracy")
    ax.set_title("Controlling for potential confounds", fontsize=15)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.LogLocator(subs=(1., 2., 3., 5.)))
        axis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v/1000:g}k" if v >= 1000 else f"{v:g}"))
        axis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(alpha=.25, lw=.6, which="both")
    _despine(ax)
    ax.legend(loc='lower left', frameon=False, fontsize = 12)
    fig.tight_layout()
    out = fig_path("compare_matched_params_loglog.svg")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print("saved", out)
    return out


if __name__ == "__main__":
    if "--bounds" in sys.argv or not os.path.exists(BOUNDS_NPZ):
        b = compute_bounds()
    else:
        b = dict(np.load(BOUNDS_NPZ))
    print(f"  step {STEP}: optimal {b[f'{BOUND_KEY}_joint'][STEP-1]:.4f}  "
          f"factorized {b[f'{BOUND_KEY}_naive'][STEP-1]:.4f}")
    plot_loglog(b)
