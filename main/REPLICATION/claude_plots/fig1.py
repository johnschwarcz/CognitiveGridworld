"""Figure 1, at its final size (7.5 x 5.6 in). From the repository root, in the eureka
environment (other pythons on this machine lack numpy or cannot parse the repo), on the CPU:

    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \
        main/REPLICATION/claude_plots/fig1.py
    -> main/REPLICATION/figure_bin/fig1_master.pdf (vector; scatters and heatmaps rasterized) and fig1_master.png
       (300 dpi); prints the numbers the caption quotes about the panels (fig1's return value)

Panels: (a) the task: the latent space key over example observation streams; (b) the agents: the two networks and
the two Bayesian observers as graphical models; (c) belief geometry: each agent's beliefs at t = 1 and T = 30, with
the mean trajectories of four cells of one environment; (d-f) learning: accuracy over training, the correlation of
factorization regret with correctness, and the dimensionality added by the recurrence; (g1-g5) lagged
cross-correlations with entanglement; (h, i) the trained network's maps by variance band.

Inputs, in main/DATA/, made with the same python by main/REPLICATION/gradflow_analysis.py (its add-ons on the GPU:
own_quantities runs on cuda:0) and main/REPLICATION/supplementary_plotter2.py. A missing one is remade by
gradflow_analysis' own builders where they can (a figure series also needs gradflow_ctx's ctx_grams_* and turns_*):
    sanity/{fully_trained,reservoir}_ctx_2_e5_net.pth           the two networks (replayed only to build beliefs.npz
                                                                and the alias cache)
    cache/gradflow_{trained,echo}.npz                           gradflow_analysis' harvests: episodes, beliefs,
                                                                entanglement (a warm render reads only the trained's)
    cache/gradflow_basis_trained.npz                            its fixed PCA basis: the bands of (h, i)
    experimental/gradflow_ctx/figure_series_{trained,echo}.pkl  its figure series, entry (None, 1): (g), (i) and the
                                                                band rows of (h, i)
    experimental/gradflow_ctx/band1_both_{trained,echo}.npz     its 1% band add-on (only to build band_energy_r.npz)
    cache/supplementary_harvest.npz                             supplementary_plotter2's harvest: (d-f)

Figure caches, built on first use in main/DATA/experimental/fig1/ (or in the directory $FIG1_CACHE
names). A cold build takes about 9-10 min on the CPU, 7-8 of them in fr_correlation's root-finds, and peaks at
~4.6 GB of RAM; a warm render takes ~5 s and ~1.3 GB:
    beliefs.npz                            the networks' beliefs at t = 1 and T = 30 (collect_beliefs)
    fr_corr_implied.npz                    the FR-correctness correlation through training (fr_correlation)
    band_energy_r.npz                      each band's activity energy against entanglement (band_energy_r)
    alias_traj_1434_2000_02_58_93_97.npz   (c)'s trajectories, 2000 streams per cell (alias_trajectories)
A rebuild reproduces fr_corr_implied.npz and band_energy_r.npz byte for byte, and beliefs.npz too when torch runs on
12 threads as it did (OMP_NUM_THREADS=12; the default 16 here differ by float32 rounding, at most ~1e-6). The alias
cache was replayed on the GPU, so a CPU rebuild differs in its network beliefs by up to ~7e-6 (no answer changes). A
cold rebuild at 16 threads changes 44 of the PNG's 3.78 M pixels, by 1/255. From the shipped caches the PNG and the
PDF are byte-identical run to run; the reference renders used Python 3.12.9, numpy 1.26.4, scipy 1.15.1, matplotlib
3.10.0 (FreeType 2.12.1), Pillow 11.1.0, fontTools 4.55.3, scikit-learn 1.6.1 and torch 2.5.1. The analyses behind
the caption's caveats are in fig1_checks.py."""
import logging
import os
import sys
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


def _repo_root():
    for start in (globals().get("__file__"), globals().get("__vsc_ipynb_file__"), os.getcwd()):
        d = os.path.abspath(start) if start and os.path.exists(start) else None
        while d:
            if os.path.exists(os.path.join(d, "main", "CognitiveGridworld.py")):
                return d
            d = None if os.path.dirname(d) == d else os.path.dirname(d)
    raise RuntimeError("project root not found")


ROOT = _repo_root()
sys.path[:0] = [p for p in (os.path.join(ROOT, "main", "REPLICATION"), ROOT) if p not in sys.path]   # (main.utils)
import gradflow_analysis as ga                                                   # noqa: E402
from main.utils import apply_fig_style                                           # noqa: E402

apply_fig_style()                         # the text font (cmr10) of every figure; supplementary_plotter2 applies it too
CACHE_DIR = os.environ.get("FIG1_CACHE") or os.path.join(ROOT, "main", "DATA", "experimental", "fig1")
BELIEFS = os.path.join(CACHE_DIR, "beliefs.npz")
FR_IMPLIED = os.path.join(CACHE_DIR, "fr_corr_implied.npz")
BAND_ENERGY = os.path.join(CACHE_DIR, "band_energy_r.npz")
N_EP = 20000                                                                     # the harvests' episodes used
STEPS = (0, 29)                                                                  # t = 1 and t = T = 30 (indices)
TS = tuple(s + 1 for s in STEPS)                                                 # the same, as steps t
N_R = ga.CFG["realization_num"]                                                  # values per latent (10)
REPLAY = ("ep_obs_flat", "ep_active_Z", "ep_goal_ind", "ep_joint_goal_belief")   # what a replay feeds the network


def _harvest(net, keys):
    """the members keys of net's gradflow harvest, read alone (the archive holds 8.6 GB; ga.harvest, which collects it
    when it is missing, loads all of it)"""
    path = ga.CACHE.format(net)
    if not os.path.exists(path):
        ga.harvest(net)
    with np.load(path) as f:
        return {k: f[k] for k in keys}


def _beliefs(net, D, batches, check=True, steps=STEPS):
    """forward replay of net on the episodes in D -> beliefs (N, 2 steps, contexts, realizations); with check, the
    harvest's per-step error sizes must match (only when D is net's own harvest)"""
    import gc
    from main.CognitiveGridworld import CognitiveGridworld
    load_env, reservoir = ga.AGENTS[net]
    agent = CognitiveGridworld(**dict(ga.CFG, episodes=1), reservoir=reservoir, load_env=load_env,
                               external_logger=lambda env: {})
    m, (B, T) = agent.model, (ga.CFG["batch_num"], ga.CFG["step_num"])
    dev, out = m.STM.device, []
    for b in range(batches):
        rows = slice(b * B, (b + 1) * B)
        m.obs_flat, m.active_Z, m.goal_ind, m.joint_goal_belief = (torch.as_tensor(D[k][rows], device=dev)
                                                                   for k in REPLAY)
        with torch.no_grad(), torch.autocast(dev.type, enabled=False):
            x = ga.lstm_input(m)
            st, _ = ga.forward_states(m, x)
            r = m.classifier_readout(st["u"])
            if check:
                loss = ga.StepLoss(m, r)
                err = torch.stack([ga.step_error(m, loss, r, t).norm(dim=-1) for t in range(T)], 1).cpu().numpy()
                rel = np.abs(err - D["err"][rows]) / np.maximum(np.abs(D["err"][rows]), 1e-12)
                assert (rel > 1e-3).mean() < .01 and np.median(rel) < 1e-5, f"{net} batch {b}: errors differ"
            q = r.double().cumsum(1).reshape(B, T, m.ctx_num, m.realization_num).softmax(-1)   # as StepLoss
            out.append(q[:, list(steps)].float().cpu().numpy())
        print(f"{net}: belief batch {b + 1} of {batches}", flush=True)
    del agent
    gc.collect()
    return np.concatenate(out)


def collect_beliefs(n_ep=N_EP):
    """the networks' marginal beliefs (softmax over realizations of the summed logits, per context) at t = 1 and
    T = 30 on the first n_ep harvested episodes, by forward replay: each network on its own episodes (the replay
    checks the harvest's errors), and the ESN on the trained network's episodes too (same inputs, so the
    representations can be compared) -> BELIEFS. The figure draws trained and echo_on_trained; echo (and ctx_echo)
    is kept because its replay is the check of the ESN's replay against the ESN's own harvest"""
    if os.path.exists(BELIEFS):
        B = dict(np.load(BELIEFS))
        assert len(B["trained"]) == n_ep, f"{BELIEFS} holds {len(B['trained'])} episodes, not n_ep = {n_ep}"
        return B
    Dt, De = (_harvest(net, REPLAY + ("err", "ep_ctx_vals")) for net in ("trained", "echo"))
    nb = n_ep // ga.CFG["batch_num"]
    res = {"trained": _beliefs("trained", Dt, nb), "echo": _beliefs("echo", De, nb),
           "echo_on_trained": _beliefs("echo", Dt, nb, check=False),
           "ctx_trained": Dt["ep_ctx_vals"][:n_ep], "ctx_echo": De["ep_ctx_vals"][:n_ep], "steps": np.array(STEPS)}
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(BELIEFS, **res)
    print(f"saved {BELIEFS}")
    return res


# ═══════════════════════════════════════════════════════════════════
# Style, at the final size: the full text width of letter paper with 0.5 in margins (7.5 in) x H_IN
# ═══════════════════════════════════════════════════════════════════

sp = ga.sp                                                                       # supplementary_plotter2
W_IN, H_IN = 7.5, 5.6
C_TR, C_EC = sp.AGENT_COLORS["Trained"], sp.AGENT_COLORS["Echo"]
C_JOINT, C_NAIVE = "k", "0.5"
NAME = {"Trained": "Trained RNN", "Echo": "ESN", "Joint": "Optimal Bayes", "Naive": "Naive Bayes"}   # shown names
COL = {"Trained": C_TR, "Echo": C_EC, "Joint": C_JOINT, "Naive": C_NAIVE}
FS_TICK, FS_LAB, FS_TITLE, FS_LETTER = 6, 6.5, 7, 8
RC = {"font.size": FS_LAB, "axes.labelsize": FS_LAB, "axes.titlesize": FS_TITLE, "xtick.labelsize": FS_TICK,
      "ytick.labelsize": FS_TICK, "legend.fontsize": FS_TICK, "axes.linewidth": .6, "lines.linewidth": 1.,
      "xtick.major.width": .6, "ytick.major.width": .6, "xtick.major.size": 2, "ytick.major.size": 2,
      "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "axes.labelpad": 2, "pdf.fonttype": 42, "ps.fonttype": 42,
      "savefig.dpi": 600, "mathtext.fontset": "cm"}                              # math in Computer Modern, as the text


def _ax(fig, x, y, w, h):
    """axes from inches (x, y: lower left)"""
    return fig.add_axes([x / W_IN, y / H_IN, w / W_IN, h / H_IN])


def _letter(fig, x, y, s):
    """a bold panel letter: cmr10 has no bold weight, so the bold Computer Modern of mathtext"""
    fig.text(x / W_IN, y / H_IN, r"$\mathbf{%s}$" % s, fontsize=FS_LETTER, ha="left", va="baseline")


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def _title2(ax, name, formula):
    """a two-line panel title, centred on the axes: the quantity's name over its formula, each on a fixed baseline so
    the titles of a row line up"""
    for txt, dy, fs in ((formula, 4, FS_TICK), (name, 13.5, FS_LAB)):
        ax.annotate(txt, xy=(.5, 1), xycoords="axes fraction", xytext=(0, dy), textcoords="offset points",
                    fontsize=fs, ha="center", va="baseline")


# ═══════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════

def training_series():
    """(d-f)'s data, from supplementary_plotter2's harvest (no smoothing, as there): per network the final-step
    accuracy of every test batch through training, the FR-correctness correlation and the participation-ratio gain;
    the Optimal and Naive bounds; the testing episode where the trained network first passes the Naive bound"""
    Hc, corr = np.load(sp.HARVEST_CACHE), fr_correlation()
    out = {}
    for lab, tag in (("Trained", "tr_"), ("Echo", "ec_")):
        A = sp.Bunch({k[len(tag):]: Hc[k] for k in Hc.files if k.startswith(tag)})
        out[lab] = {"acc": A.test_acc_through_training[:, -1].astype(np.float64), "corr": corr[lab],
                    "pr": sp.pr_gain(A), "joint": float(np.mean(A.joint_acc)), "naive": float(np.mean(A.naive_acc))}
    joint = np.mean([out[l]["joint"] for l in out])
    naive = np.mean([out[l]["naive"] for l in out])
    crossing = int(np.flatnonzero(out["Trained"]["acc"] > naive)[0]) + 1                 # testing episode (x = index + 1)
    return out, joint, naive, crossing


def fr_correlation():
    """the point-biserial r between final-step FR and final-step correctness in each test batch, through training.
    The run logged only the logistic fit of correctness on FR (env_model_data_manager: sklearn's LogisticRegression,
    with intercept, lightly L2-penalised); its fit reproduces cov(FR, correct) (exactly at the maximum-likelihood fit,
    to 5e-5 here), so r follows from the logged slope and accuracy, given FR's distribution (a task property: the
    harvest's final-step FR, pooled). Error against a batch's exact r: about 1% of |r| (SD), below its sampling noise"""
    if os.path.exists(FR_IMPLIED):
        f = np.load(FR_IMPLIED)
        return {"Trained": f["trained"], "Echo": f["echo"]}
    from scipy.optimize import brentq
    Hc = np.load(sp.HARVEST_CACHE)
    sig = lambda z: 1 / (1 + np.exp(-z))
    out = {}
    for lab, tag in (("Trained", "tr_"), ("Echo", "ec_")):
        fr = Hc[tag + "fr"].astype(np.float64).ravel()
        acc = np.clip(Hc[tag + "test_acc_through_training"][:, -1].astype(np.float64), 1e-3, 1 - 1e-3)
        r = []
        for b, p in zip(Hc[tag + "test_SII_coef_through_training"].astype(np.float64), acc):
            a = brentq(lambda a: sig(a + b * fr).mean() - p, -30, 30)               # intercept from the accuracy
            r.append(np.cov(fr, sig(a + b * fr))[0, 1] / (fr.std() * np.sqrt(p * (1 - p))))
        out[lab] = np.array(r)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(FR_IMPLIED, trained=out["Trained"], echo=out["Echo"])
    return out


def learning_numbers(T):
    """(d-f)'s caption numbers from training_series' per-network series, as (e, f) draw them (raw values against the
    smoothed accuracy): the ESN's plateau and final FR-correctness r (means of the last 1000 test batches); the trained
    network's r plateau where (e) dips (batches at smoothed accuracy 0.25-0.43) and at the end; the PR gain's slope
    per 0.1 of accuracy (least squares over all batches), the smoothed accuracies of the last negative and the first
    positive batch above it, and the ESN's largest PR gain"""
    tr, ec = T["Trained"], T["Echo"]
    acc_s = sp._smooth(tr["acc"], 11)
    pr = np.asarray(tr["pr"], np.float64)
    ok = np.isfinite(pr)
    last_neg = acc_s[ok & (pr < 0)].max()
    return {"echo_plateau": float(ec["acc"][-1000:].mean()), "echo_r_end": float(ec["corr"][-1000:].mean()),
            "trained_r_dip": float(tr["corr"][(acc_s >= .25) & (acc_s <= .43)].mean()),
            "trained_r_end": float(tr["corr"][-1000:].mean()),
            "pr_per_0.1_acc": float(np.polyfit(tr["acc"][ok], pr[ok], 1)[0] / 10),
            "pr_sign_change_acc": (float(last_neg), float(acc_s[ok & (pr > 0) & (acc_s > last_neg)].min())),
            "echo_pr_max": float(np.nanmax(ec["pr"]))}


def band_energy_r():
    """cross-correlation with entanglement of each 1% band's absolute activity energy ||P_B (u_t - mu)||^2, each band's
    series z-scored within the episode as figure_series' shares -> {net: (TAU, bands)}, bands as figure_series' ids
    ((h) draws the trained network's; the ESN's is kept for comparison)"""
    if os.path.exists(BAND_ENERGY):
        return dict(np.load(BAND_ENERGY))
    N, s = ga.BAND1_EP, slice(ga.SRC[0], ga.SRC[-1] + 1)
    out = {}
    for net in ("trained", "echo"):
        f = np.load(ga.CACHE.format(net))
        Ea = np.asarray(ga.band_both(net, 1)["E"][:N], dtype=np.float64)[:, s]
        out[net] = ga._lagged(ga._zs(Ea), ga._zs(f["rev"][:N, s].astype(np.float64)))
        assert out[net].shape == ga.figure_series(net, 1)["r"]["energy"].shape
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(BAND_ENERGY, **out)
    return out


def _band_edges(net, ids):
    """each kept band's span of the cumulative variance (in %), from the fixed PCA basis"""
    Bs = ga.basis(net)
    cum = np.concatenate([[0], np.cumsum(Bs["evr"])])
    sid = ga.cut_segments(Bs["evr"], 100)
    starts = [cum[np.flatnonzero(sid == b)[0]] for b in ids]
    return 100 * np.array(starts + [cum[-1]])


ENV = 1434                                                # (a, c)'s environment: the trained harvest's episode ENV
CELLS = ((0, 2), (5, 8), (9, 3), (9, 7))                  # (its tables asymmetric in the two contexts, as 98% are) and
                                                          # its four cells (r_g, r_c), chosen as alias_trajectories says
STREAMS = (None, 3)                                       # (a)'s streams: two of the five observables (None: the two
                                                          # that differ most across the cells), seed
ALIAS_M, ALIAS_SEED = 2000, 1                             # (c)'s trajectories: streams per cell, seed
ALIAS = os.path.join(CACHE_DIR, f"alias_traj_{ENV}_{ALIAS_M}_" + "_".join(f"{g}{c}" for g, c in CELLS)
                     + ("" if ALIAS_SEED == 1 else f"_seed{ALIAS_SEED}") + ".npz")   # seed 1: the name predates it
MARKS = ("o", "s", "^", "*")                              # per cell
MARK_SCALE = {"*": 1.45}                                  # (c): a star of the same nominal size looks smaller
MARK_SCALE_A = {"*": 1.15}                                # (a): enlarged less, to fit the key and the streams
MS_START, MS_END, MS_HOME = 6.5, 7.6, 13.5                # (c) marker sizes (pt): curve start, curve end, target
BRIDGE = 12                                               # _time_curves' blend length in steps


def streams():
    """observation streams of two observables for CELLS, sampled as the task samples them (o_t^i = 1 with probability
    P(o^i = 1 | r_g, r_c), independently per step; env_generators)"""
    obs, seed = STREAMS
    f = np.load(ga.CACHE.format("trained"))
    JL = np.asarray(f["ep_joint_likelihood"][ENV], np.float64)                  # P(o^i = 1 | r_0, r_1)
    P = JL if int(f["ep_goal_ind"][ENV]) == 1 else np.swapaxes(JL, 1, 2)       # [i, r_c, r_g]
    if obs is None:                                                             # the two observables that differ
        obs = sorted(np.argsort(np.ptp([[P[i, rc, rg] for rg, rc in CELLS] for i in range(len(P))], 1))[-2:].tolist())
    P = P[list(obs)]                                                            # most across the cells
    rng = np.random.default_rng(seed)
    return [(rg, rc, rng.random((ga.CFG["step_num"], len(obs))) < P[:, rc, rg]) for rg, rc in CELLS]


def _align(X, w_g, w_c):
    """turn (rotate, and mirror where needed) a 2-PC plane so that the r_g decoder points right and the r_c decoder up,
    as in the key: the decoders' bisector goes to 45 degrees, r_g on its clockwise side. Orthogonal, so distances, R^2
    and the angle between the decoders are unchanged; the ridge coefficients turn with the data (X Q, Q^T w)"""
    u_g, u_c = w_g / np.linalg.norm(w_g), w_c / np.linalg.norm(w_c)
    M = np.eye(2) if u_g[0] * u_c[1] - u_g[1] * u_c[0] >= 0 else np.diag([1., -1.])   # mirror: r_g -> r_c anticlockwise
    b = M @ (u_g + u_c)
    th = np.pi / 4 - np.arctan2(b[1], b[0])
    Q = (np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ M).T     # X' = X Q, w' = Q^T w
    return X @ Q, Q.T @ w_g, Q.T @ w_c


def belief_geometry():
    """each agent's own PCs of the four agents' marginal beliefs at t = 1 and T = 30 on the same 20,000 episodes (the
    trained network's harvest; the ESN replayed on them), each belief ordered [goal context, other context]; per agent
    and step the ridge (alpha 1) decoders of r_g and r_c from the plane: R^2, coefficient vectors and the angle between
    them; each plane turned by _align. The plane is the two of PCs 1-3 that hold the most latent variance (single-PC
    R^2 of r_g plus r_c): PCs 1-2 except where a non-latent direction (belief mass at the range ends, ESN and Naive at
    T = 30) is PC1"""
    B = collect_beliefs()
    N = len(B["trained"])
    f = np.load(ga.CACHE.format("trained"))
    goal, ctx = f["ep_goal_ind"][:N].astype(int), f["ep_ctx_vals"][:N].astype(int)
    assert np.array_equal(ctx, B["ctx_trained"])
    ar = np.arange(N)
    rg, rc = ctx[ar, goal], ctx[ar, 1 - goal]
    agents = {"Trained": B["trained"], "Echo": B["echo_on_trained"],
              "Joint": f["ep_joint_belief"][:N][:, list(STEPS)], "Naive": f["ep_naive_belief"][:N][:, list(STEPS)]}
    out = {}

    def latent_pcs(P3):                                                              # the 2 of PCs 1-3 holding most
        lat = [sum(np.corrcoef(P3[:, q], y)[0, 1] ** 2 for y in (rg, rc)) for q in range(3)]   # latent variance
        return sorted(int(q) for q in np.argsort(lat)[-2:])

    for i, t in enumerate(TS):
        for name, bel in agents.items():
            b = np.asarray(bel[:, i], np.float64)
            V = np.concatenate([b[ar, goal], b[ar, 1 - goal]], -1)
            pca = PCA(n_components=3).fit(V)
            P3 = pca.transform(V)
            pcs = latent_pcs(P3)
            X = P3[:, pcs]
            d = {"X": X, "rg": rg, "rc": rc, "pcs": tuple(q + 1 for q in pcs)}
            for k, y in (("g", rg), ("c", rc)):
                m = Ridge(alpha=1.0).fit(X, y)
                d[f"r2_{k}"], d[f"w_{k}"] = m.score(X, y), m.coef_
            d["angle"] = np.degrees(np.arccos(np.clip(d["w_g"] @ d["w_c"] / np.linalg.norm(d["w_g"]) /
                                                      np.linalg.norm(d["w_c"]), -1, 1)))
            Q = _align(np.eye(2), d["w_g"], d["w_c"])[0]                             # the turn itself: X' = X Q
            d["X"], d["w_g"], d["w_c"] = X @ Q, Q.T @ d["w_g"], Q.T @ d["w_c"]
            d["proj"] = (pca.mean_, pca.components_[pcs], Q)                         # for other beliefs: (v - m) C^T Q
            out[(name, t)] = d
    return out


def targets(cells):
    """where a correct belief about each cell sits: the Optimal observer's mean T = 30 belief over the trained harvest's
    first N_EP episodes in which it is right about both latents of that cell (goal-ordered, 20-dim: the same space for
    every agent) -> (len(cells), 20), the episodes per cell"""
    f = np.load(ga.CACHE.format("trained"))
    ar = np.arange(N_EP)
    goal, ctx = f["ep_goal_ind"][:N_EP].astype(int), f["ep_ctx_vals"][:N_EP].astype(int)
    b = np.asarray(f["ep_joint_belief"][:N_EP, STEPS[1]], np.float64)
    V = np.concatenate([b[ar, goal], b[ar, 1 - goal]], -1)
    rg, rc = ctx[ar, goal], ctx[ar, 1 - goal]
    ok = (V[:, :N_R].argmax(-1) == rg) & (V[:, N_R:].argmax(-1) == rc)
    sel = [ok & (rg == g) & (rc == c) for g, c in cells]
    return np.array([V[s].mean(0) for s in sel]), [int(s.sum()) for s in sel]


def alias_sim(EP, cells, M, seed, nets=("trained", "echo")):
    """beliefs at every step for M streams sampled from each (r_g, r_c) cell of environment EP (the trained harvest's
    episode EP: its likelihood table, goal context and context input Z): Optimal (joint) and Naive Bayes exactly, the
    networks replayed on the same streams -> {agent: (n, T, 20) goal-ordered}, cell_of (n,), and each stream's
    entanglement rev (n, T) as the harvest defines it (mean over both contexts of 0.5 Jeffreys(naive update, joint's
    history-conditioned update)); n = len(cells) * M must be a multiple of the replay batch (1000)"""
    f = np.load(ga.CACHE.format("trained"))
    T, g = ga.CFG["step_num"], int(f["ep_goal_ind"][EP])
    L = np.asarray(f["ep_joint_likelihood"][EP], np.float64)                     # (5, r0, r1)
    cell_of = np.repeat(np.arange(len(cells)), M)
    r01 = [(rc, rg) if g == 1 else (rg, rc) for rg, rc in cells]
    obs = np.random.default_rng(seed).random((len(cell_of), T, len(L))) < np.stack([L[:, a, b] for a, b in r01])[cell_of][:, None]
    o = obs.astype(np.float64)
    cum = np.cumsum(np.einsum("nti,ijk->ntjk", o, np.log(L)) + np.einsum("nti,ijk->ntjk", 1 - o, np.log1p(-L)), 1)
    post = np.exp(cum - cum.max((-1, -2), keepdims=True))
    post /= post.sum((-1, -2), keepdims=True)
    Lb = (L.mean(2), L.mean(1))
    px = np.stack([np.exp(o @ np.log(Lb[k]) + (1 - o) @ np.log1p(-Lb[k])) for k in (0, 1)], 2)   # (n, T, 2, R)
    naive = np.cumprod(px, 1)
    joint = np.stack([post.sum(-1), post.sum(-2)], 2)
    n = len(cell_of)
    rev = ga.sp.step_dkl(px / px.sum(-1, keepdims=True),
                         np.stack([ga.sp.approx_lik(joint[:, :, k]) for k in (0, 1)], 2)).mean(-1)
    D = {"ep_obs_flat": obs.astype(f["ep_obs_flat"].dtype)}
    for k in REPLAY[1:]:                                                         # the environment's own inputs
        D[k] = np.repeat(f[k][EP:EP + 1], n, 0)
    nb = n // ga.CFG["batch_num"]
    B = {"Joint": joint, "Naive": naive / naive.sum(-1, keepdims=True)}
    for net in nets:
        B["Trained" if net == "trained" else "Echo"] = _beliefs(net, D, nb, check=False, steps=range(T))
    out = {a: np.concatenate([b[:, :, g], b[:, :, 1 - g]], -1).astype(np.float32) for a, b in B.items()}
    return out, cell_of, rev.astype(np.float32)


def alias_trajectories():
    """alias_sim for CELLS of ENV, cached -> ({agent: (n, T, 20)}, cell_of, cells, rev (n, T); rev, each stream's
    entanglement, is not drawn but kept for analyses of the trajectories). The cells, by rule, from
    80 random environment orientations scanned at T = 30 (all 100 cells x 200 streams; networks replayed, Bayes
    observers exact): pairs of cells >= 3.5 apart that BOTH the ESN and the naive observer merge (the same modal
    answer for both cells, neither cell itself), trained and optimal >= 60% both latents right; of the 54 two-pair
    choices (ESN and naive give both pairs the same merged answer, answers >= 5 apart), 21 have Trained and Optimal
    mean curves that do not cross as drawn here, and 1434 (0, 2), (5, 8) -> (0, 8) and (9, 3), (9, 7) -> (9, 0) has
    the strongest ESN collapse among them: of the 2000 streams per cell the ESN gives the merged answer for
    1326 / 1513 / 1981 / 1993, the naive for 1206 / 1130 / 1929 / 1839, and neither ever names one of the four cells;
    trained 1530 / 1582 / 1514 / 1451, optimal 1538 / 1530 / 1534 / 1504 get both latents right (alias_answers). (A
    10-19 search of all 2547 orientations for choices whose curves avoid every target found none better overall.) The
    selection scan is not in the repository. A record of the session's scripts is in main/DATA/experimental/
    fig1/scratch_code_2026-10-18_19/ (selectors/: esn_scan*.py, nocross_*.py; nocross2/: the 2547-orientation
    search); they read intermediate files under the session's /tmp scratchpad, so they do not run as they are."""
    if os.path.exists(ALIAS):
        f_ = np.load(ALIAS)
        cells = [tuple(int(v) for v in c) for c in f_["cells"]]
        assert cells == list(CELLS), f"{ALIAS} holds cells {cells}, not CELLS"
        return {a: f_[a] for a in ("Trained", "Echo", "Joint", "Naive")}, f_["cell_of"], cells, f_["rev"]
    out, cell_of, rev = alias_sim(ENV, CELLS, ALIAS_M, ALIAS_SEED)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(ALIAS, **out, cell_of=cell_of, cells=np.array(CELLS), rev=rev)
    return out, cell_of, list(CELLS), rev


def alias_answers(AV, cell_of, cells):
    """(c)'s caption numbers at T = 30, per agent and cell, as counts of the cell's streams (exact; shares can tie at
    a rounding boundary): the most common answer (r_g, r_c), the streams giving it, the streams with both latents
    right, and the streams naming any of the cells"""
    out = {}
    for a, V in AV.items():
        ans = np.stack([V[:, -1, :N_R].argmax(-1), V[:, -1, N_R:].argmax(-1)], 1)   # goal-ordered beliefs
        for j, cell in enumerate(cells):
            A = ans[cell_of == j]
            vals, cnt = np.unique(A, axis=0, return_counts=True)
            named = sum(int((A == np.array(c)).all(1).sum()) for c in cells)
            out[(a, cell)] = (tuple(int(v) for v in vals[cnt.argmax()]), int(cnt.max()),
                              int((A == np.array(cell)).all(1).sum()), named, len(A))
    return out


# ═══════════════════════════════════════════════════════════════════
# The bivariate key: a teal ink for r_g and a magenta ink for r_c, mixed in OKLab (Ottosson 2020)
# ═══════════════════════════════════════════════════════════════════

_LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                 [0.0883024619, 0.2817188376, 0.6299787005]])
_LAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                 [0.0259040371, 0.7827717662, -0.8086757660]])


def _lch(L, C, h):
    return np.array([L, C * np.cos(np.radians(h)), C * np.sin(np.radians(h))])


def _srgb(lab):
    lin = np.clip((lab @ np.linalg.inv(_LAB).T) ** 3 @ np.linalg.inv(_LMS).T, 0, 1)
    return np.where(lin <= .0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - .055)


KEY = (_lch(.80, .06, 55), _lch(.56, .12, 192), _lch(.56, .15, 352), _lch(.28, .09, 285))
                                                         # OKLCh corners at (r_g, r_c) = (0, 0), (9, 0), (0, 9), (9, 9)


def bivariate(rg, rc, R=N_R):
    """colour of each (r_g, r_c), bilinear in OKLab between KEY's four corners: r_g adds teal, r_c magenta (both
    high: indigo, both low: a light peach, so all four corners differ). Each step of either latent moves the colour by
    3.4 / 3.3 OKLab x 100 (median; at least 1.6, also under simulated deuteranopia and protanopia), and neither ink is
    the networks' blue or orange (the peach corner is 14 units from the ESN orange)"""
    c00, c90, c09, c99 = KEY
    g, c = (np.asarray(v, np.float64)[..., None] / (R - 1) for v in (rg, rc))
    return _srgb((1 - g) * (1 - c) * c00 + g * (1 - c) * c90 + (1 - g) * c * c09 + g * c * c99)


# ═══════════════════════════════════════════════════════════════════
# (a) the task and (b) the agents, on one canvas in inches
# ═══════════════════════════════════════════════════════════════════

COL1 = dict(ks=1.20, kx=.12, yt=3.91, first=.10, pitch=.19, drow=.048, ms=7.5, arrow=.07, tlab=.095,
            x0=.29, x1=1.32, mx=.17, dot=4)
                                                         # (a): key size and x (as wide as the streams); the streams'
                                                         # title baseline and first-row offset, stream pitch, row
                                                         # gap, marker size (pt), time arrow and label offsets,
                                                         # first / last step x, marker x, dot size
COLS = dict(cx=1.965, bw=.80, bh=.40, th=.185, ag=.09)  # (b), the networks: centre, split box (about the original
                                                         # sketch's proportions), trapezoid height, arrows
GRAPH = dict(r=.078, ro=.09, xj=1.728, xn=(2.067, 2.301), dx=.10, yl=3.83, yo=3.29, m=.022)
                                                         # (b), the observers' graphs: latent / observed node radii
                                                         # (o_1:T is .155 in wide at 5.5 pt), the optimal observer's
                                                         # x and the naive's two, the optimal's latents either side,
                                                         # latent / observed rows, box margin


def _panel_canvas(fig):
    sa = _ax(fig, 0, 2.92, 2.47, 2.68)                                            # (a, b)'s canvas in inches
    sa.set_xlim(0, 2.47), sa.set_ylim(2.92, 5.6)
    sa.axis("off")
    return sa


def _latent_key(fig, sa, S, top):
    """(a), top: the latent space key, under its title on the letters' baseline top, the cells' markers on it (sized
    with the key: MS_HOME per inch of key)"""
    R, ks, kx = N_R, COL1["ks"], COL1["kx"]
    ax = _ax(fig, kx, top - .05 - ks, ks, ks)
    gg, cc = np.meshgrid(np.arange(R), np.arange(R))
    ax.imshow(bivariate(gg, cc), origin="lower", extent=(-.5, R - .5, -.5, R - .5), interpolation="nearest")
    for (rg, rc, _), m in zip(S, MARKS):
        ax.scatter(rg, rc, marker=m, s=(MS_HOME * ks * MARK_SCALE_A.get(m, 1)) ** 2, facecolor="white",
                   edgecolor="k", lw=.9, clip_on=False, zorder=3)
    ax.set_xticks([0, 9]), ax.set_yticks([0, 9])
    ax.tick_params(length=1.5, pad=1)
    ax.set_xlabel("$r_g$", labelpad=-3)
    ax.set_ylabel("$r_c$", labelpad=-3)
    sa.text(kx + ks / 2, top, "Latent space", ha="center", va="baseline", fontsize=FS_LAB)


def _obs_streams(sa, S):
    """(a), bottom: the cells' observation streams (all 30 steps, a row per observable) under "Observation space",
    with the t = 1 -> T = 30 axis"""
    p = COL1
    xs = np.linspace(p["x0"], p["x1"], S[0][2].shape[0])
    yt, ms, pitch, drow = (p[k] for k in ("yt", "ms", "pitch", "drow"))
    sa.text((p["mx"] - ms / 144 + xs[-1]) / 2, yt, "Observation space", ha="center", va="baseline",  # over the block,
            fontsize=FS_LAB)                                                                        # markers in
    y0 = yt - p["first"]
    for k, ((_, _, o), m) in enumerate(zip(S, MARKS)):
        yk = y0 - k * pitch
        sa.scatter(p["mx"], yk - drow / 2, marker=m, s=(ms * MARK_SCALE_A.get(m, 1)) ** 2, facecolor="white",
                   edgecolor="k", lw=.7, clip_on=False)
        for i in range(o.shape[1]):                                              # a row per observable
            y, on = yk - i * drow, o[:, i]
            sa.scatter(xs[on], np.full(on.sum(), y), s=p["dot"], color="k", lw=0)
            sa.scatter(xs[~on], np.full((~on).sum(), y), s=p["dot"], facecolor="white", edgecolor="0.55", lw=.35)
    ya = y0 - 3 * pitch - drow - p["arrow"]
    sa.annotate("", xy=(xs[-1] + .04, ya), xytext=(xs[0] - .02, ya),
                arrowprops=dict(arrowstyle="-|>", lw=.5, color="0.3", mutation_scale=5, shrinkA=0, shrinkB=0))
    sa.text(xs[0], ya - p["tlab"], "$t = 1$", ha="center", va="baseline", fontsize=FS_TICK)
    sa.text(xs[-1], ya - p["tlab"], "$T = 30$", ha="center", va="baseline", fontsize=FS_TICK)


def _split_box(ax, cx, yc, w, h, fills, labels):
    """a rounded box split at cx (the networks' LSTM box): fills and coloured labels per half -> its bottom and top"""
    by = yc - h / 2
    box = ax.add_patch(matplotlib.patches.FancyBboxPatch((cx - w / 2, by), w, h, boxstyle="round,pad=0,rounding_size=.05",
                                                         fc="none", ec="0.2", lw=.7, zorder=3))
    for x_, fc_ in ((cx - w / 2, fills[0]), (cx, fills[1])):                       # halves clipped to the corners
        ax.add_patch(plt.Rectangle((x_, by), w / 2, h, fc=fc_, ec="none", zorder=1)).set_clip_path(box)
    ax.plot([cx, cx], [by, by + h], c="0.2", lw=.4, ls=":", zorder=2)
    for sx, (lab, c) in zip((-w / 4, w / 4), labels):
        ax.text(cx + sx, yc, lab, ha="center", va="center", fontsize=FS_TICK, color=c, linespacing=1.3, zorder=4)
    return by, by + h


def _networks(ax, top):
    """(b), top: o_t -> read-in -> LSTM (Trained RNN | ESN, frozen at its initialization) -> read-out -> belief
    updates Delta b_t (logits; the belief b_t is the softmax of their running sum, model_forward; the environment
    input is not drawn); green = trained weights; the output on the panel letters' baseline top"""
    cx, bw, bh, th, ag = (COLS[k] for k in ("cx", "bw", "bh", "th", "ag"))
    green, up = "#cde5c9", dict(arrowstyle="-|>", lw=.6, color="0.2", mutation_scale=5, shrinkA=0, shrinkB=0)
    arrow = lambda x, y0, y1: ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=up)

    def trap(yb, wb, wt, label):
        ax.add_patch(plt.Polygon([(cx - wb / 2, yb), (cx + wb / 2, yb), (cx + wt / 2, yb + th), (cx - wt / 2, yb + th)],
                                 closed=True, fc=green, ec="0.2", lw=.6))
        ax.text(cx, yb + th / 2, label, ha="center", va="center", fontsize=FS_TICK)

    hi = top - .035 - th - 2 * ag
    lo, hi = _split_box(ax, cx, hi - bh / 2, bw, bh, (green, "white"), (("Trained\nRNN", C_TR), ("ESN\n(frozen)", C_EC)))
    ri = lo - ag - th
    trap(ri, .532 * bw, .894 * bw, "Read-in")
    arrow(cx, ri + th, lo)
    arrow(cx, ri - ag, ri)
    ax.text(cx, ri - ag - .035, "$o_t$", ha="center", va="top", fontsize=FS_TICK)
    trap(hi + ag, .894 * bw, .532 * bw, "Read-out")
    arrow(cx, hi, hi + ag)
    arrow(cx, hi + ag + th, hi + 2 * ag + th)
    ax.text(cx, hi + 2 * ag + th + .035, r"belief updates $\Delta b_t$", ha="center", va="baseline", fontsize=FS_TICK)


def _bayes_graphs(ax):
    """(b), bottom: the Bayesian observers as graphical models. Optimal: one model, r_g -> o_1:T <- r_c (the task's true
    structure), one belief over both latents, p(r_g, r_c); Naive: two models, r_g -> o_1:T and r_c -> o_1:T, each
    latent on its own (its model of an observation averages the other latent out), each in its own box, one belief
    per latent, p(r_g) and p(r_c) (its own, not the marginals of p(r_g, r_c)); observed nodes shaded"""
    R, RO, XJ, XN, DX, YL, YO, m = (GRAPH[k] for k in ("r", "ro", "xj", "xn", "dx", "yl", "yo", "m"))
    obs = "$o_{1:T}$"

    def node(x, y, lab, observed=False):
        ax.add_patch(plt.Circle((x, y), RO if observed else R, fc="0.85" if observed else "white", ec="k", lw=.6,
                                zorder=5))
        ax.text(x, y, lab, ha="center", va="center", fontsize=5.5 if observed else FS_TICK, zorder=6)

    def edge(p, q):
        ax.annotate("", xy=q, xytext=p, zorder=4, arrowprops=dict(arrowstyle="-|>", lw=.6, color="0.15",
                                                                 mutation_scale=5, shrinkA=R * 72 + .3,
                                                                 shrinkB=RO * 72 + .6))

    def square(x0, x1):                                                          # a model's box around its nodes
        h = max(R, RO)
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x0 - h - m, YO - RO - m), x1 - x0 + 2 * (h + m),
                                                       YL - YO + R + RO + 2 * m, boxstyle="round,pad=0,rounding_size=.03",
                                                       fc="none", ec="0.35", lw=.55, zorder=3))

    node(XJ - DX, YL, "$r_g$")
    node(XJ + DX, YL, "$r_c$")
    node(XJ, YO, obs, observed=True)
    for x_ in (XJ - DX, XJ + DX):
        edge((x_, YL), (XJ, YO))
    for x_, lab in zip(XN, ("$r_g$", "$r_c$")):
        node(x_, YL, lab)
        node(x_, YO, obs, observed=True)
        edge((x_, YL), (x_, YO))
        square(x_, x_)
    yt = YO - RO - m - .04
    ax.text(XJ, yt, "Optimal", ha="center", va="top", fontsize=FS_LAB, color=C_JOINT)
    ax.text(sum(XN) / 2, yt, "Naive", ha="center", va="top", fontsize=FS_LAB, color=C_NAIVE)
    yb = YL + R + m + .03                                                        # one belief over both latents;
    ax.plot([XJ - DX, XJ - DX, XJ + DX, XJ + DX], [yb - .02, yb, yb, yb - .02], c="0.3", lw=.5, zorder=4)  # one per
    ax.text(XJ, yb + .015, "$p(r_g, r_c)$", ha="center", va="bottom", fontsize=FS_TICK, zorder=6)          # latent,
    for x_, lab in zip(XN, ("$p(r_g)$", "$p(r_c)$")):                                                      # over the
        ax.text(x_, yb - .005, lab, ha="center", va="bottom", fontsize=FS_TICK, zorder=6)                 # boxes


def _panels_ab(fig, top):
    """(a) the task: the latent space key over the observation streams (level with the observers' graphs); (b) the
    agents: the networks over the Bayesian observers"""
    S, sa = streams(), _panel_canvas(fig)
    _latent_key(fig, sa, S, top)
    _networks(sa, top)
    _bayes_graphs(sa)
    _obs_streams(sa, S)
    _letter(fig, .02, top, "a")
    _letter(fig, 1.45, top, "b")


# ═══════════════════════════════════════════════════════════════════
# (c) belief geometry
# ═══════════════════════════════════════════════════════════════════

def _geometry_panel(ax, D, s=.12):
    """one belief cloud (its plane, turned by _align), coloured by (r_g, r_c) with opaque points (transparent ones would
    darken where they pile up, and darkness here means both latents high)"""
    X = D["X"]
    ax.scatter(X[:, 0], X[:, 1], c=bivariate(D["rg"], D["rc"]), s=s, lw=0, rasterized=True)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    ax.apply_aspect()


def _curve_paths(fig, p1, p30, paths, bridge=BRIDGE, sub=8):
    """the drawn course (figure fraction) of each mean belief trajectory across a (t = 1, T = 30) pair p = (ax, D), as
    _time_curves describes -> tt (steps, sub-sampled), [(len(tt), 2)]"""
    to_fig, T = fig.transFigure.inverted(), len(paths[0])
    tt = np.linspace(0, T - 1, (T - 1) * sub + 1)
    s_ = np.clip(tt / (bridge - 1), 0, 1)
    w = (s_ * s_ * (3 - 2 * s_))[:, None]
    out = []
    for V in paths:
        V = np.stack([np.interp(tt, np.arange(T), V[:, i]) for i in range(V.shape[1])], 1)
        F = [to_fig.transform(ax.transData.transform((V - D["proj"][0]) @ D["proj"][1].T @ D["proj"][2]))
             for ax, D in (p1, p30)]
        out.append((1 - w) * F[0] + w * F[1])
    return tt, out


def _time_curves(fig, p1, p30, paths, cols, homes, sub=8):
    """mean belief trajectories t = 1 -> T across a (t = 1, T = 30) pair of panels p = (ax, D): each curve starts at
    its exact t = 1 position in the t = 1 plane and is exact in the T = 30 plane from t = BRIDGE on; before, each
    step's mean belief is projected into both planes and the drawn point slides from the first to the second (weight
    smoothstep((t - 1) / (BRIDGE - 1))), so a split seen in both planes shows on the way; the beliefs are interpolated
    linearly between steps (sub points per step). Each curve runs from light and thin (t = 1) to black and thick (T),
    in log t (most of the change in a belief comes early), from its cell's white marker to the marker in the cell's
    key colour; homes (figure fraction, per cell): the large white targets where a correct belief about the cell sits,
    under the curves"""
    T = len(paths[0])
    tt, Ps = _curve_paths(fig, p1, p30, paths, sub=sub)
    for j, (P, c) in enumerate(zip(Ps, cols)):
        mk = MARKS[j]
        sc = MARK_SCALE.get(mk, 1)
        fig.add_artist(Line2D([homes[j][0]], [homes[j][1]], transform=fig.transFigure, marker=mk, ms=MS_HOME * sc,
                              mfc="white", mec="k", mew=.9, ls="none", zorder=6.8,
                              path_effects=[pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()]))
        e = np.log1p(tt) / np.log(T)                                                 # log t, 0 .. 1 along the curve
        e = (e[1:] + e[:-1]) / 2
        seg = np.stack([P[:-1], P[1:]], 1)
        widths = .45 + 1.25 * e
        for col_, wd, z in (("white", widths + 1.0, 6.9), (None, widths, 7)):
            fig.add_artist(LineCollection(seg, transform=fig.transFigure, linewidths=wd, zorder=z, capstyle="round",
                                          colors=col_ if col_ else [str(.6 * (1 - v)) for v in e]))
        for q, ms, fc in ((P[0], MS_START, "white"), (P[-1], MS_END, c)):
            fig.add_artist(Line2D([q[0]], [q[1]], transform=fig.transFigure, marker=mk, ms=ms * sc, mfc=fc, mec="k",
                                  mew=.6, ls="none", zorder=8 if sc > 1 else 8.1))   # the large star under the rest


# ═══════════════════════════════════════════════════════════════════
# (g) lag curves
# ═══════════════════════════════════════════════════════════════════

def _curve(ax, y, c, ms=2.2):
    """a lag curve in the dynamics figure's style: dotted and faded before the event, solid after"""
    past, fut = ga.TAU <= 0, ga.TAU >= 0
    ax.plot(ga.TAU[past], y[past], c=c, ls=":", lw=1., alpha=.55)
    ax.plot(ga.TAU[fut], y[fut], c=c, lw=1.1, marker="o", ms=ms, mec="k", mew=.3)


# ═══════════════════════════════════════════════════════════════════
# The figure
# ═══════════════════════════════════════════════════════════════════

def fig1():
    """Figure 1, 7.5 x H_IN in: row 1 the task (a), the agents (b) and belief geometry (c); row 2 learning (d-f); row 3
    the cross-correlations with entanglement (g1-g5) and the trained network's band maps (h, i) -> the numbers the
    caption quotes"""
    path = ga.fig_path("fig1_master.pdf")
    T, joint, naive, crossing = training_series()
    G = belief_geometry()
    AV, a_of, a_cells, _ = alias_trajectories()
    home_v, n_home = targets(a_cells)
    X = {lab: ga.figure_series(net, 1) for net, lab in (("trained", "Trained"), ("echo", "Echo"))}
    name, col = NAME, COL
    top = H_IN - .17                                                                 # the top row's panel letters
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W_IN, H_IN))
        _panels_ab(fig, top)
        # ══ row 1, right: (c) belief geometry, each agent at t = 1 and T = 30 (planes turned to the key) ══════════
        bx0, bs, bgap, bname, bpair = 2.55, 1.11, .03, .14, .10                     # names to the side of each pair
        rows_y, ey = (4.33, 3.16), 3.06                                              # the rows; the key line
        xc = bx0 - .06                                                               # c's letter (e's under it)
        for r, agents in enumerate((("Trained", "Echo"), ("Joint", "Naive"))):
            for p, agent in enumerate(agents):
                x0 = bx0 + p * (bname + 2 * bs + bgap + bpair) + bname
                pair = []
                for k, t in enumerate(TS):
                    ax = _ax(fig, x0 + k * (bs + bgap), rows_y[r], bs, bs)
                    D_ = G[(agent, t)]
                    _geometry_panel(ax, D_)
                    pair.append((ax, D_))
                    if r == 0:
                        fig.text((x0 + k * (bs + bgap) + bs / 2) / W_IN, (rows_y[r] + bs + .065) / H_IN,
                                 "$t = 1$" if t == 1 else "$T = 30$", ha="center", va="baseline", fontsize=FS_TICK,
                                 color="0.3")
                fig.text((x0 - .07) / W_IN, (rows_y[r] + bs / 2) / H_IN, name[agent], rotation=90, ha="center",
                         va="center", fontsize=FS_TITLE, color=col[agent])
                mean_paths = [AV[agent][a_of == j].mean(0) for j in range(len(a_cells))]
                ax30, D30 = pair[1]
                homes = fig.transFigure.inverted().transform(ax30.transData.transform(
                    (home_v - D30["proj"][0]) @ D30["proj"][1].T @ D30["proj"][2]))
                _time_curves(fig, *pair, mean_paths, [bivariate(rg_, rc_) for rg_, rc_ in a_cells], homes)
        ex = np.linspace(3.66, 4.26, 40)                                            # key: a projected trajectory
        nT = ga.CFG["step_num"]
        ev = np.log1p((np.arange(39) + .5) / 39 * (nT - 1)) / np.log(nT)            # log t, as the curves
        for col_, wd in (("white", .45 + 1.25 * ev + 1.0), ([str(.6 * (1 - v)) for v in ev], .45 + 1.25 * ev)):
            fig.add_artist(LineCollection(np.stack([np.stack([ex[:-1], ex[1:]], 1) / W_IN,
                                                    np.full((39, 2), ey / H_IN)], -1),
                                          transform=fig.transFigure, colors=col_, linewidths=wd, capstyle="round"))
        for x_, fc_, ms_ in ((ex[0], "white", MS_START), (ex[-1], "0.55", MS_END)):
            fig.add_artist(Line2D([x_ / W_IN], [ey / H_IN], transform=fig.transFigure, marker="o", ls="none", mfc=fc_,
                                  mec="k", mew=.6, ms=ms_, zorder=3))
        fig.text((ex[0] - .09) / W_IN, ey / H_IN, "projection:  $t = 1$", ha="right", va="center", fontsize=FS_TICK,
                 color="0.3")
        fig.text((ex[-1] + .09) / W_IN, ey / H_IN, "$T = 30$", ha="left", va="center", fontsize=FS_TICK, color="0.3")
        for x_, ms_, mew_, lab in ((5.62, MS_START, .6, "initial"), (6.38, MS_HOME, .9, "target")):   # key for the
            fig.add_artist(Line2D([x_ / W_IN], [ey / H_IN], transform=fig.transFigure, marker="o", ls="none",   # white
                                  mfc="white", mec="k", mew=mew_, ms=ms_))                                # markers
            fig.text((x_ + ms_ / 144 + .045) / W_IN, ey / H_IN, lab, ha="left", va="center", fontsize=FS_TICK,
                     color="0.3")
        _letter(fig, xc, top, "c")
        # ══ row 2: (d) accuracy over training; (e), (f) against accuracy ═══════════════════════════════════════
        cy, ch, cw, cgap = 1.93, .85, 1.86, .66
        axd, axe, axf = (_ax(fig, .50 + i * (cw + cgap), cy, cw, ch) for i in range(3))
        n = len(T["Trained"]["acc"])
        for lab in ("Echo", "Trained"):
            axd.plot(np.arange(1, n + 1), T[lab]["acc"], c=col[lab], lw=.7, label=name[lab])
            acc_s = sp._smooth(T[lab]["acc"], 11)
            for ax, key in ((axe, "corr"), (axf, "pr")):
                ax.plot(T[lab]["acc"], T[lab][key], c=col[lab], lw=.45, alpha=.35)
                ax.plot(acc_s, T[lab][key], c=col[lab], lw=.7)
        axd.set_xscale("log")
        axd.set_xlim(1, n)
        axd.set_xticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "1000", "10000"])   # 6 pt, not 4 pt exponents
        axd.minorticks_off()
        axd.axhline(joint, c=C_JOINT, ls="--", lw=.8)
        axd.axhline(naive, c=C_NAIVE, ls="--", lw=.8)
        for y, txt, c in ((joint, "Optimal Bayes", C_JOINT), (naive, "Naive Bayes", C_NAIVE)):     # off the dashes
            axd.annotate(txt, (1.3, y), xytext=(0, 2), textcoords="offset points", fontsize=FS_TICK, va="bottom",
                         color=c)
        for lab in ("Trained", "Echo"):                                                  # direct labels, under the ends
            axd.text(n / 1.25, T[lab]["acc"][-200:].mean() - (.09 if lab == "Trained" else .06), name[lab],
                     fontsize=FS_TICK, color=col[lab], ha="right", va="top")
        axd.set_ylim(.05, .88)
        axd.set_yticks([.25, .5, .75])
        axd.set_xlabel("Testing episode", labelpad=1)
        axd.set_ylabel("Final-step\naccuracy", linespacing=.95)
        axe.set_ylabel("Corr. $r$\n(FR, correct)", linespacing=.95)
        axf.set_ylabel("PR(recurrence) $-$\nPR(read-in)", linespacing=.95)
        for ax in (axe, axf):
            ax.axvline(naive, c=C_NAIVE, ls="--", lw=.8)
            ax.axvline(joint, c=C_JOINT, ls="--", lw=.8)
            ax.axhline(0, c="k", lw=.5, alpha=.4)
            ax.set_xlim(.05, .82)
            ax.set_xlabel("Final-step accuracy", labelpad=1)
            for xv, txt, c, ha, dx in ((naive, "Naive", C_NAIVE, "left", .008), (joint, "Optimal", C_JOINT, "right", -.008)):
                ax.text(xv + dx, .04, txt, transform=ax.get_xaxis_transform(), fontsize=FS_TICK, color=c, ha=ha,
                        va="bottom")
        for ax, x_, k in zip((axd, axe, axf), (.02, xc, axf.get_position().x0 * W_IN - .46), "def"):
            _clean(ax)
            _letter(fig, x_, cy + ch - .06, k)
        # ══ row 3: (g) cross-correlations with entanglement; (h), (i) by variance band (trained network) ════════
        hy, hh = .38, .78
        hx0, hw, hg = .46, .80, .09
        tops = (("l2", "Activity norm", r"$\|u_t\|$"),
                ("pr", "Activity eff. dim.", r"$(\Sigma_i\, a_i^2)^2 / \Sigma_i\, a_i^4$"),        # inline sums
                ("coh", "Credit coherence", r"$\|\Sigma_k\, \delta_k\|^2 / \Sigma_k \|\delta_k\|^2$"),
                ("dW", "Update energy", r"$\|\Delta W\|^2$"),
                ("band_x", "Cross-band share", r"$1 - \Sigma_B \|\Delta W_B\|^2 / \|\Delta W\|^2$"))
        G_axes = []
        for i, (key, title, formula) in enumerate(tops):
            ax = _ax(fig, hx0 + i * (hw + hg), hy, hw, hh)
            for lab in ("Echo", "Trained"):
                _curve(ax, X[lab]["r"][key], col[lab])
            ax.axhline(0, c="k", lw=.5, alpha=.4)
            ax.axvline(0, c="k", lw=.5, alpha=.5)
            ax.set_xlim(-8.8, 8.8)
            ax.set_xticks([-8, -4, 0, 4, 8])
            _title2(ax, r"$\mathbf{g}_\mathbf{%d}$ " % (i + 1) + title, formula)
            ax.set_xlabel(r"Lag $\tau$", labelpad=1)
            _clean(ax)
            G_axes.append(ax)
        lo = min(min(X[l]["r"][k].min() for l in X) for k, _, _ in tops)
        hi = max(max(X[l]["r"][k].max() for l in X) for k, _, _ in tops)
        for ax in G_axes:
            ax.set_ylim(lo - .08 * (hi - lo), hi + .08 * (hi - lo))
            ax.tick_params(labelleft=ax is G_axes[0])
        G_axes[0].set_ylabel("$r$ with\nentanglement", linespacing=.95)
        G_axes[0].text(-8.4, .33, "Trained\nRNN", color=C_TR, fontsize=FS_TICK, ha="left", va="center", linespacing=.9)
        G_axes[0].text(-8.4, -.17, "ESN", color=C_EC, fontsize=FS_TICK, ha="left", va="center")   # which curve is which
        row3 = hy + hh + 13.5 / 72                                                    # on the title names' line
        _letter(fig, .02, row3, "g")
        mx0, mw, mg = hx0 + 5 * hw + 4 * hg + .62, .765, .14                    # room between g and h; i's letter
        maps = ((band_energy_r()["trained"], "Activity energy", r"$\|P_B (u_t - \mu)\|^2$"),
                (X["Trained"]["r"]["cross"], "Cross-band share",
                 r"$\langle \Delta W_B, \Delta W - \Delta W_B \rangle / \|\Delta W\|^2$"))
        vmax = max(np.abs(M).max() for M, _, _ in maps)
        edges = _band_edges("trained", list(X["Trained"]["ids"]))
        for j, (M, title, formula) in enumerate(maps):
            x = mx0 + j * (mw + mg)
            ax = _ax(fig, x, hy, mw, hh)
            im = ax.pcolormesh(np.arange(ga.TAU[0] - .5, ga.TAU[-1] + 1), edges, M.T, cmap=ga.HEAT_CMAP, vmin=-vmax,
                               vmax=vmax, shading="flat", rasterized=True)
            ax.set_ylim(0, 100)                                                   # leading PCs at the bottom
            ax.axvline(0, c="k", ls="--", lw=.6, alpha=.6)                        # through the tau = 0 column
            ax.set_xticks([-8, -4, 0, 4, 8])
            ax.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
            ax.tick_params(labelleft=j == 0)
            ax.set_xlabel(r"Lag $\tau$", labelpad=1)
            _title2(ax, title, formula)
            if j == 0:
                ax.set_ylabel("high $\\to$ low var.", labelpad=1)
            _letter(fig, x - (.36 if j == 0 else .10), row3, "hi"[j])
        fig.text((mx0 + mw + mg / 2) / W_IN, (hy + hh + .335) / H_IN, "Trained RNN, by band", ha="center",
                 va="baseline", fontsize=FS_LAB, color=C_TR)
        cax = _ax(fig, mx0 + 2 * mw + mg + .04, hy + .03, .05, hh - .06)
        cb = fig.colorbar(im, cax=cax)
        cb.outline.set_linewidth(.4)
        cb.ax.tick_params(labelsize=FS_TICK, length=1.5, pad=1)
        cb.set_label("$r$", fontsize=FS_LAB, rotation=0, labelpad=1, va="center")
        logging.getLogger("fontTools").setLevel(logging.ERROR)                      # (cmr10's old font timestamps)
        fig.savefig(path, metadata={"CreationDate": None})                         # no date: byte-identical reruns
        fig.savefig(path.replace(".pdf", ".png"), dpi=300)
        plt.close(fig)
    print(f"saved {path} and its .png ({W_IN} x {H_IN} in)")
    t0 = list(ga.TAU).index(0)
    return {"optimal": joint, "naive": naive, "crossing": crossing,                       # (d): bounds; first pass
            "learning": learning_numbers(T),                                                # (d-f)
            "geometry": {k: (v["r2_g"], v["r2_c"], v["angle"]) for k, v in G.items()},  # (c): R^2 of r_g, r_c; angle
            "target_episodes": dict(zip(a_cells, n_home)), "answers_T30": alias_answers(AV, a_of, a_cells),
            "g_tau0": {lab: {k: round(float(X[lab]["r"][k][t0]), 3) for k, _, _ in tops} for lab in X},
            "vmax": vmax, "ranges": [(title, float(M.min()), float(M.max())) for M, title, _ in maps]}   # (h, i)


if __name__ == "__main__":
    if sys.argv[1:]:                                              # (an interactive kernel passes its own arguments)
        print(f"no arguments taken, ignored: {sys.argv[1:]} (the diagnostics are fig1_checks.py's modes)")
    print(fig1())
