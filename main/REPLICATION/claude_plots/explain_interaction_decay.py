"""explain_interaction_decay.png: how the interaction decays with the delay between the two flipped bits, as one curve
per agent (John 11-10, after the distribution versions of explain_interaction: "the trained network falls off at the
tails too strongly. Do you think you have a better way to show this? Maybe we could stick to probability scaling but
turn it into an animation through delay, with xlim +/- .3. But that sounds overcomplicated.").

Measure: |Delta' - Delta| per value (fig_learn_interaction's |mixed difference|, nats) with bit i in o_30 and bit j in
o_30 itself (delay 0: the 10 bit pairs, both orders) or in o_(30 - d), d = 1, ..., 29 (25 bit pairs): every delay at
the same step t = 30, so the curves differ only by the delay. Per agent and delay: the median (line) and the 25-75%
range (band) of |Delta' - Delta|, which the rare large values (Joint Bayes's heavy tails) do not move; the mean (the
measure of fig_learn_interaction) in --numbers. 1,000 probe episodes (md.side: 50 test environments x 20), both
variables, 10 values; networks at the end of training (39,996 training episodes).
Data: computed here (md.observer_md / md.net_md on the 29 pairs (30, 30 - d); the same step from explain_interaction's
cache) into through_training/mixed_diff/explain_decay.npz (per agent and delay: quantiles and mean of |MD|).
RESULTS (11-10; median |Delta' - Delta|, nats; same step / delay 1 / 2 / 5 / 10 / 20 / 29):
  Joint Bayes   .0174 / .0193 / .0193 / .0193 / .0193 / .0193 / .0193   (flat: no decay)
  Trained RNN   .0562 / .0360 / .0291 / .0214 / .0181 / .0154 / .0153   (decays, levels off at ~80% of Bayes's median)
  ESN           .0935 / .0106 / .0056 / .0012 / .0002 / .0000 / .0000   (gone within ~5 steps)
  means (fig_learn_interaction's measure) same step / 1 / 5 / 29: Bayes .0383 / .0431 / .0431 / .0427, Trained RNN
  .0727 / .0469 / .0283 / .0213, ESN .1141 / .0133 / .0016 / .0000. The median is not moved by Bayes's heavy tails,
  so the Trained RNN's long-delay interaction is near Bayes's in the typical value, half of it in the mean.
SIGNED (the drawn version since 11-10 ~16:15; mean amplification / suppression, nats, delay 0 / 1 / 5 / 10 / 29):
  Joint Bayes   .017/.022  .018/.025  .018/.025  .018/.025  .018/.025   (flat; suppression a little larger)
  Trained RNN   .029/.043  .022/.025  .014/.014  .012/.013  .010/.011   (both decay to about half of Bayes's, then level)
  ESN           .051/.064  .006/.007  .0008/.0008  .0001/.0001  0/0  (both gone within ~5 steps)
  share of values strengthened (|s| > .001): 45-53% for every agent and delay.
usage:  python explain_interaction_decay.py [--data] [--signed] [--numbers]"""
import os
import sys
import time
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402
import td_tt_mixed_diff as md                                                     # noqa: E402
import explain_interaction as ei                                                  # noqa: E402

T = md.T
D = tuple(range(1, T))                                                            # delays 1-29
PAIRS = tuple((T, T - d) for d in D)
Q = (.25, .5, .75)
CACHE = os.path.join(md.OUT, "explain_decay.npz")
AGENTS = ei.AGENTS
FS = dict(tick=7, key=7, label=8, title=8)


def stats(a):
    a = np.abs(a.ravel())
    return np.concatenate([np.quantile(a, Q), [a.mean()]])


def build():
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    S = md.side()
    Z = ei.load()
    ti = list(md.TS).index(T)
    out = {}
    for key, _, _ in AGENTS:
        t0 = time.time()
        if key == "bayes":
            M = md.observer_md(key, S, pairs=PAIRS)
        else:
            cell = md.LSTMCell(key, ei.RAW)
            envt, code = md.codes(S["rows"])
            U, C = cell.replay(envt, code)
            M = md.net_md(cell, envt, code, U, C, pairs=PAIRS)
        same = Z[f"{key}_within"][ti]                                             # (10 bit pairs, n, 2, R) at t = 30
        out[key] = np.stack([stats(same)] + [stats(M[p]) for p in range(len(PAIRS))])   # (30 delays, 4)
        print(f"{key}: {time.time() - t0:.0f} s", flush=True)
    np.savez(CACHE[:-4] + ".tmp.npz", delays=np.array((0,) + D), q=np.array(Q), **out)
    os.replace(CACHE[:-4] + ".tmp.npz", CACHE)


CACHE2 = os.path.join(md.OUT, "explain_decay_signed.npz")


def build_signed():
    """per agent, delay (0-29, all at t = 30) and test environment: the mean amplification E[max(s, 0)] and the mean
    suppression E[max(-s, 0)] of s = sign(Delta)(Delta' - Delta) per value (their sum is the mean |Delta' - Delta|,
    fig_learn_interaction's measure), and the share of values with s > 0 among |s| > .001"""
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    S = md.side()
    Z = ei.load()
    env = np.asarray(S["env"])
    envs = np.unique(env)
    ti = list(md.TS).index(T)
    BITS = ei.BITS
    I, J = np.array([i for i, j in BITS]), np.array([j for i, j in BITS])
    out = {}
    for key, _, _ in AGENTS:
        t0 = time.time()
        eff = Z[f"{key}_eff"][ti].astype(np.float64)                              # (5 i, n, 2, R) at t = 30
        if key == "bayes":
            M = md.observer_md(key, S, pairs=PAIRS)
        else:
            cell = md.LSTMCell(key, ei.RAW)
            envt, code = md.codes(S["rows"])
            U, C = cell.replay(envt, code)
            M = md.net_md(cell, envt, code, U, C, pairs=PAIRS)
        w = Z[f"{key}_within"][ti].astype(np.float64)                             # (10 bit pairs, n, 2, R)
        sig = [np.concatenate([w * np.sign(eff[I]), w * np.sign(eff[J])])]        # same step: both orders
        sig += [M[p] * np.sign(eff)[None] for p in range(len(PAIRS))]             # (5 j, 5 i, n, 2, R): i = axis 1
        amp, sup, pos = [], [], []
        for sg in sig:
            a_ = np.moveaxis(sg, -3, 0).reshape(sg.shape[-3], -1)                 # (n, values)
            amp.append([np.maximum(a_[env == e], 0).mean() for e in envs])
            sup.append([np.maximum(-a_[env == e], 0).mean() for e in envs])
            big = np.abs(sg) > 1e-3
            pos.append((sg[big] > 0).mean() if big.any() else np.nan)
        out[f"{key}_amp"], out[f"{key}_sup"], out[f"{key}_pos"] = np.array(amp), np.array(sup), np.array(pos)
        print(f"{key}: {time.time() - t0:.0f} s", flush=True)
    np.savez(CACHE2[:-4] + ".tmp.npz", delays=np.array((0,) + D), envs=envs, **out)
    os.replace(CACHE2[:-4] + ".tmp.npz", CACHE2)


def load():
    if not os.path.exists(CACHE):
        build()
    Z = dict(np.load(CACHE))
    if os.path.exists(CACHE2):
        Z.update({f"s_{k}": v for k, v in np.load(CACHE2).items()})
    return Z


def numbers(Z):
    d = Z["delays"]
    for key, name, _ in AGENTS:
        s = Z[key]
        print(f"{name:12s} median |MD| (nats) same step {s[0, 1]:.4f}; delay 1 / 2 / 5 / 10 / 20 / 29: "
              + " / ".join(f"{s[np.flatnonzero(d == k)[0], 1]:.4f}" for k in (1, 2, 5, 10, 20, 29))
              + "; mean same step " + f"{s[0, 3]:.4f}, delay 1 / 5 / 29: "
              + " / ".join(f"{s[np.flatnonzero(d == k)[0], 3]:.4f}" for k in (1, 5, 29)))


def figure(Z, path=None):
    apply_fig_style()
    plt.rcParams.update({"mathtext.fontset": "cm", "axes.linewidth": .6})
    W, H = 5.5, 2.3
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([.62 / W, .5 / H, (W - .62 - 1.55) / W, (H - .5 - .18) / H])
    d = Z["delays"]
    style = {"bayes": dict(color="k", lw=1.0, ls=(0, (3.5, 2))), "trained": dict(color=AGENTS[1][2], lw=1.2),
             "echo": dict(color=AGENTS[2][2], lw=1.2)}
    for key, name, c in AGENTS:
        s = Z[key]
        ax.fill_between(d, s[:, 0], s[:, 2], color=c if key != "bayes" else "0.5", alpha=.18, lw=0)
        ax.plot(d, s[:, 1], marker="o", ms=2.2, mew=0, **style[key])
    ax.set_xlim(-.6, 29.6)
    ax.set_xticks([0, 1, 5, 10, 15, 20, 25, 29])
    ax.set_xticklabels(["0", "1", "5", "10", "15", "20", "25", "29"])
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=FS["tick"], length=2.5, width=.6, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(r"steps between the two flipped bits ($j$ in $o_{30-d}$, $i$ in $o_{30}$; 0 = the same observation)",
                  fontsize=FS["label"],
                  labelpad=2)
    ax.set_ylabel(r"median $|\Delta' - \Delta|$ (nats)", fontsize=FS["label"], labelpad=2)
    hs = [Line2D([], [], **style[k]) for k, _, _ in AGENTS] + [plt.matplotlib.patches.Patch(fc="0.5", alpha=.3, lw=0)]
    fig.legend(hs, [n for _, n, _ in AGENTS] + ["25-75% of values"], loc="center left",
               bbox_to_anchor=((W - 1.45) / W, .55), fontsize=FS["key"], frameon=False, handlelength=2.0,
               labelspacing=.5)
    path = path or fig_path("explain_interaction_decay.png")
    fig.savefig(path, dpi=400)
    plt.close(fig)
    print("saved", path)


def boot(per_env, n=2000, seed=0):
    """5-95% over resampled test environments of the mean over environments: per_env (delays, envs)"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, per_env.shape[1], (n, per_env.shape[1]))
    m = per_env[:, idx].mean(-1)                                                  # (delays, n)
    return np.quantile(m, .05, 1), np.quantile(m, .95, 1)


def figure_signed(Z, path=None):
    """John 11-10: "I'm not a big fan of medians. Can we also split the positive (amplification) and negative
    (suppression) modulation of the Delta? Or maybe you have a better idea of how to emphasize both effects?": with
    s = sign(Delta)(Delta' - Delta) per value, the mean amplification E[max(s, 0)] (up) and the mean suppression
    E[max(-s, 0)] (drawn down) at every delay, all at t = 30. The two add up to the mean |Delta' - Delta|
    (fig_learn_interaction's measure); means over all values (no medians), bands 5-95% over resampled environments."""
    apply_fig_style()
    plt.rcParams.update({"mathtext.fontset": "cm", "axes.linewidth": .6})
    W, H = 5.5, 2.9
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([.66 / W, .5 / H, (W - .66 - 1.45) / W, (H - .5 - .12) / H])
    d = Z["s_delays"]
    style = {"bayes": dict(color="k", lw=1.0, ls=(0, (3.5, 2))), "trained": dict(color=AGENTS[1][2], lw=1.2),
             "echo": dict(color=AGENTS[2][2], lw=1.2)}
    for key, name, c in AGENTS:
        for part, sgn in (("amp", 1), ("sup", -1)):
            pe = Z[f"s_{key}_{part}"]                                             # (delays, envs)
            lo, hi = boot(pe)
            ax.fill_between(d, sgn * lo, sgn * hi, color=c if key != "bayes" else "0.4", alpha=.2, lw=0)
            ax.plot(d, sgn * pe.mean(1), marker="o", ms=2.0, mew=0, **style[key])
    ax.axhline(0, color="0.5", lw=.6)
    ax.set_xlim(-.6, 29.6)
    ax.set_xticks([0, 1, 5, 10, 15, 20, 25, 29])
    yl = max(abs(v) for v in ax.get_ylim())
    ax.set_ylim(-yl, yl)
    yt = [v for v in ax.get_yticks() if abs(v) <= yl + 1e-9]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{abs(v):g}" for v in yt])
    ax.tick_params(labelsize=FS["tick"], length=2.5, width=.6, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(r"steps between the two flipped bits ($j$ in $o_{30-d}$, $i$ in $o_{30}$; 0 = the same observation)",
                  fontsize=FS["label"], labelpad=2)
    ax.set_ylabel("mean modulation (nats)", fontsize=FS["label"], labelpad=2)
    ax.text(.985, .97, r"amplification  E$[\max(s, 0)]$", transform=ax.transAxes, ha="right", va="top",
            fontsize=FS["key"])
    ax.text(.985, .03, r"suppression  E$[\max(-s, 0)]$", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=FS["key"])
    hs = [Line2D([], [], **style[k]) for k, _, _ in AGENTS]
    fig.legend(hs, [n for _, n, _ in AGENTS], loc="center left", bbox_to_anchor=((W - 1.35) / W, .62),
               fontsize=FS["key"], frameon=False, handlelength=2.0, labelspacing=.5)
    fig.text((W - 1.35) / W, .36, r"$s = \mathrm{sign}(\Delta)\,(\Delta' - \Delta)$" + "\nbands: 5-95% over\nenvironments",
             fontsize=FS["key"], color="0.3", va="top", linespacing=1.3)
    path = path or fig_path("explain_interaction_decay.png")
    fig.savefig(path, dpi=400)
    plt.close(fig)
    print("saved", path)


def signed_numbers(Z):
    d = Z["s_delays"]
    for key, name, _ in AGENTS:
        a, s_ = Z[f"s_{key}_amp"].mean(1), Z[f"s_{key}_sup"].mean(1)
        print(f"{name:12s} amplification / suppression (nats) at delay 0 / 1 / 5 / 10 / 29: "
              + "  ".join(f"{a[np.flatnonzero(d == k)[0]]:.4f}/{s_[np.flatnonzero(d == k)[0]]:.4f}" for k in (0, 1, 5, 10, 29))
              + "; share strengthening (|s| > .001) " + " / ".join(f"{Z[f's_{key}_pos'][np.flatnonzero(d == k)[0]]:.2f}"
                                                                    for k in (0, 1, 5, 10, 29)))


if __name__ == "__main__":
    if "--data" in sys.argv:
        build()
    if "--signed" in sys.argv:
        build_signed()
    Z = load()
    if "--numbers" in sys.argv:
        numbers(Z)
        if "s_delays" in Z:
            signed_numbers(Z)
    if "s_delays" in Z:
        figure(Z, path=fig_path("explain_interaction_decay_median.png"))
        figure_signed(Z)
    else:
        figure(Z)
