"""fig_manifold_fr.png: the belief manifolds of single interaction sets, coloured by Factorization Regret: panels (c) and
(d) of the paper's manifold figure ("Compositionality of a representation is trivial where Factorization Regret is
low"), regenerated with the paper's FORWARD regret (John 11-10: "we need to make sure every plot that uses FR in any
way is using the same forward form ... I have already added this new figure, does it need to be updated?").

  c  five test environments (interaction sets; columns) x Joint Bayes, Trained RNN, Naive Bayes, Echo (rows): each
     of the 100 cells (r_0, r_1) at the agent's posterior means (E[r_0], E[r_1]) at T = 30, averaged over the cell's
     24 episodes, joined to its lattice neighbours; dotted: the true lattice. An edge's colour: the mean of its two
     cells' regret in that environment.
  d  the same for all 50 test environments, overlaid (edges in one fixed random order).
Regret of a cell in an environment, as the paper's panel has it: FR_0 + FR_1, the SUM over the two variables of
FR_v = KL(Joint_T || Naive_T) of variable v's marginal belief (nats; exact, log space), mean over the cell's 24
episodes (2 x fig_learn_fr's grid cache, pair_env_fr); colours on the paper's scale, .5 to 25 nats (log).
--form sym draws the superseded colouring, FR_v = [KL(J||N) + KL(N||J)] / 2 with probabilities clipped at 1e-15,
for comparison -> fig_manifold_fr_sym.png in DATA/experimental/fr_definition/. --svg also writes a vector copy for the
paper's composite.

WHAT THE PAPER'S PANEL WAS (the 11-05 composite, panels c and d; read from its vector graphics, 11-10): the five
environments are 10, 23, 43, 44, 48 of the probe set (identified by their shapes; its grids are from another draw of
episodes of the same environments: its line positions match these to .09 cells in the median, not exactly). Its edge
colours follow the SYMMETRIC clipped regret, summed over the two variables, each edge the arithmetic mean of its two
cells, on LogNorm(.5, 25): mapping that quantity's distribution over the 9,000 edges onto the composite's colours
leaves a residual of .0016 (colour-map units, 7 quantiles) and implies the scale .50 to 25.1; the forward regret leaves
.031 (20 x worse). So only the colours change here, and only by the KL's direction; the convention (sum of both
variables, edge mean, scale) is kept. With the forward KL the Joint and Trained grids lose most of their red border
columns (environment 10's right side, 44's left side); the folds of the Naive and Echo grids stay red.
THE SUM IS DELIBERATE (John 11-10: "keep the sum of both, just note in the code that we are doing this"): a cell's
regret here is FR_0 + FR_1, the SUM over the two variables, so it is twice the per-variable average that
fig_learn_grids shows (scale .4-11) and covers both variables, whereas the paper's FR_t is the goal variable's alone.

RESULTS (11-10, --numbers): per (environment, cell), 5,000 pairs, sum of the two variables: forward median 2.14 nats
(5-95%: .39-26.0, max 82.4), symmetric clipped median 3.10 (.47-24.9, max 50.6); their ratio median 1.33 (5-95%:
.76-2.63); rank correlation .96. Edges above 10 nats (the red end): 14.5% forward against 21.9% symmetric over all
environments; in the paper's five, forward 2 / 7 / 16 / 11 / 3% against symmetric 16 / 9 / 23 / 21 / 3%.
Data: summary_data/grids (td_grids_fwd.py: 50 test environments x 100 cells x 24 episodes; Trained RNN and Echo at the
last checkpoint, 39,996 training episodes) through fig_learn_fr.load("grids"). Seconds once that cache exists.
usage:  python fig_manifold_fr.py [--form sym] [--svg] [--numbers]"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.collections import LineCollection                                 # noqa: E402
from matplotlib.colors import LogNorm                                             # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402

OUT = os.path.join(ROOT, "main", "DATA", "experimental", "fr_definition")
ENVS = (10, 23, 43, 44, 48)                                                       # the paper's five (of the 50 probe environments)
ROWS = (("bayes", "Joint", "k"), ("trained", "Trained", "#1f77b4"), ("naive", "Naive", "0.5"), ("echo", "Echo", "#ff7f0e"))
NORM = (.5, 25.)                                                                  # the colour scale of the paper's panel (read off its bar)
R = 10
FS = dict(name=10, tick=8.5, label=10)


def edges():
    idx = {(a, b): a * R + b for a in range(R) for b in range(R)}
    return np.array([(k, idx[(a + da, b + db)]) for (a, b), k in idx.items() for da, db in ((1, 0), (0, 1))
                     if (a + da, b + db) in idx])


def regret(form="fwd"):
    """each cell's regret in each environment (50, 100), FR_0 + FR_1 (the sum over the two variables, as the paper's
    panel), mean over the cell's 24 episodes: 'fwd' = the paper's FR_v = KL(Joint || Naive); 'sym' = the superseded
    symmetric clipped one"""
    import fig_learn_fr as flf
    G = flf.load("grids")
    if form == "fwd":
        return 2 * G["pair_env_fr"]          # SUM over the two variables (John 11-10): pair_env_fr is their mean, so x 2
    path = os.path.join(OUT, "manifold_fr_sym.npz")
    if not os.path.exists(path):
        P = np.load(os.path.join(flf.AN, "probes.npz"))
        E = np.load(os.path.join(flf.GRID_DIR, "fwd_episodes.npz"))
        env, key = E["env"], E["env"] * R * R + E["cell"]
        order = np.lexsort((np.arange(len(env)), key))
        L, obs = P["L"].astype(np.float64), E["obs"].astype(np.float64)
        clip = lambda x, y: (np.clip(x, 1e-15, 1 - 1e-15) * np.log(np.clip(x, 1e-15, 1 - 1e-15)        # noqa: E731
                                                                   / np.clip(y, 1e-15, 1 - 1e-15))).sum(-1)
        fw, sy = np.zeros((len(env), 2)), np.zeros((len(env), 2))
        for s in range(0, len(env), 20000):
            sl = slice(s, s + 20000)
            lb, ln = flf._observers_T(obs[sl], L[env[sl]])
            fw[sl], sy[sl] = (np.exp(lb) * (lb - ln)).sum(-1), (clip(np.exp(lb), np.exp(ln)) + clip(np.exp(ln), np.exp(lb))) / 2
        pair = lambda v: flf._per_pair(v.mean(1)[:, None], order)[..., 0].mean(-1)                      # noqa: E731
        assert np.allclose(pair(fw), G["pair_env_fr"])                                # the same episodes as the forward cache
        os.makedirs(OUT, exist_ok=True)
        np.savez(path, sym=pair(sy))
    return 2 * np.load(path)["sym"]


def positions():
    """{agent: (50, 100, 2)}: each cell's mean posterior means at T; the networks at their last checkpoint"""
    import fig_learn_fr as flf
    G = flf.load("grids")
    return {"bayes": G["pos_bayes"], "naive": G["pos_naive"], "trained": G["pos_trained"][-1], "echo": G["pos_echo"][-1]}


def _panel(ax, lattice=True):
    if lattice:
        for v in range(R):
            ax.plot([v, v], [0, R - 1], color="0.78", lw=.45, ls=(0, (1, 1.6)), zorder=0)
            ax.plot([0, R - 1], [v, v], color="0.78", lw=.45, ls=(0, (1, 1.6)), zorder=0)
    ax.set_xlim(-.75, R - .25)
    ax.set_ylim(-.75, R - .25)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("0.75")
        sp.set_linewidth(.8)


def figure(form="fwd", svg=False, path=None):
    import plot_fig1c_mesh as pm
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    fr, pos, ed = regret(form), positions(), edges()
    norm = LogNorm(*NORM)
    colour = lambda f: (f[ed[:, 0]] + f[ed[:, 1]]) / 2                              # noqa: E731  (the edge's two cells)
    s1, g1, s2, g2 = .86, .06, 1.52, .12
    left, top, bot = .34, .12, .1
    H = top + 4 * s1 + 3 * g1 + bot
    x2 = left + 5 * s1 + 4 * g1 + .45
    W = x2 + 2 * s2 + g2 + .72
    fig = plt.figure(figsize=(W, H))
    add = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])              # noqa: E731
    for r, (a, name, col) in enumerate(ROWS):                                      # c: five environments
        y = H - top - (r + 1) * s1 - r * g1
        fig.text((left - .12) / W, (y + s1 / 2) / H, name, rotation=90, ha="center", va="center", color=col,
                 fontsize=FS["name"])
        for c, e in enumerate(ENVS):
            ax = add(left + c * (s1 + g1), y, s1, s1)
            _panel(ax)
            lc = LineCollection(pos[a][e][ed], cmap=pm.FR_CMAP, norm=norm, linewidths=1.0, capstyle="round", zorder=3)
            lc.set_array(colour(fr[e]))
            ax.add_collection(lc)
    seg_c = np.concatenate([colour(fr[e]) for e in range(len(fr))])
    perm = np.random.default_rng(0).permutation(len(seg_c))                        # one drawing order for the four panels
    h2 = (H - top - bot - .34 - g2 - .22) / 2
    s2 = min(s2, h2)
    for k, (a, name, col) in enumerate(ROWS):                                      # d: all 50 environments
        r, c = divmod(k, 2)
        y = H - top - .22 - (r + 1) * s2 - r * (g2 + .22)
        ax = add(x2 + c * (s2 + g2), y, s2, s2)
        _panel(ax, lattice=False)
        seg = np.concatenate([pos[a][e][ed] for e in range(len(fr))])
        lc = LineCollection(seg[perm], cmap=pm.FR_CMAP, norm=norm, linewidths=.45, alpha=.75, zorder=3, rasterized=not svg)
        lc.set_array(seg_c[perm])
        ax.add_collection(lc)
        ax.set_title(name, color=col, fontsize=FS["name"], pad=3)
    cax = add(x2 + 2 * s2 + g2 + .14, bot + .12, .11, H - top - bot - .3)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=pm.FR_CMAP, norm=norm), cax=cax, extend="both")
    cb.set_label("factorization regret", fontsize=FS["label"], rotation=270, labelpad=11)
    cb.ax.tick_params(labelsize=FS["tick"], length=2.5)
    cb.outline.set_linewidth(.6)
    if path is None:
        path = fig_path("fig_manifold_fr.png") if form == "fwd" else os.path.join(OUT, "fig_manifold_fr_sym.png")
    fig.savefig(path, dpi=300)
    if svg:
        fig.savefig(path[:-4] + ".svg")
    plt.close(fig)
    print("saved", path)
    return path


def numbers():
    from scipy.stats import spearmanr
    f, s, ed, pos = regret("fwd"), regret("sym"), edges(), positions()
    q = lambda v: "median %.2f (5-95%%: %.2f-%.2f, max %.1f)" % (np.median(v), *np.quantile(v, [.05, .95]), v.max())   # noqa: E731
    print(f"per (environment, cell), {f.size:,} pairs: forward {q(f)}; symmetric clipped {q(s)}")
    print("ratio symmetric / forward: median %.2f (5-95%%: %.2f-%.2f); rank correlation %.3f"
          % (np.median(s / f), *np.quantile(s / f, [.05, .95]), spearmanr(f.ravel(), s.ravel())[0]))
    for name, v in (("forward", f), ("symmetric", s)):
        ec = (v[:, ed[:, 0]] + v[:, ed[:, 1]]) / 2
        print(f"  edges above 10 nats, {name}: {100 * (ec > 10).mean():.1f}% of all; the paper's five environments "
              + ", ".join(f"{e}: {100 * (ec[e] > 10).mean():.0f}%" for e in ENVS))
    print("last checkpoints (training episodes):", {k: v.shape for k, v in pos.items()})


if __name__ == "__main__":
    if "--numbers" in sys.argv:
        numbers()
    else:
        figure("sym" if "sym" in sys.argv else "fwd", svg="--svg" in sys.argv)
