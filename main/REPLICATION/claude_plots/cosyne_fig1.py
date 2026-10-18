"""Figure 1 of the Cosyne 2027 abstract, at its final size (7.5 x 5.3 in); every quantity comes from an existing
analysis or cache.

collect_beliefs()  the networks' marginal beliefs (softmax over realizations of the summed logits, per context) at
                   t = 1 and t = T = 30 for the first N_EP harvested episodes, by forward replay (CPU is fine):
                   each network on its own episodes (the replay checks the harvest's errors), and the echo network
                   on the trained network's episodes too (same inputs, so the two representations can be compared)
                   -> DATA/experimental/cosyne_fig1/beliefs.npz
fr_correlation()   the FR-correctness correlation of every test batch through training, from the logged logistic fit
band_energy_r()    each 1% band's absolute activity energy against entanglement
fig1()             -> figure_bin/fig1_master.pdf (vector; scatters and heatmaps rasterized) and .png
"""
import os
import sys
import numpy as np
import torch


def _repo_root():
    for start in (globals().get("__file__"), os.getcwd()):
        d = os.path.abspath(start) if start and os.path.exists(start) else None
        while d:
            if os.path.exists(os.path.join(d, "main", "CognitiveGridworld.py")):
                return d
            d = None if os.path.dirname(d) == d else os.path.dirname(d)
    raise RuntimeError("project root not found")


ROOT = _repo_root()
sys.path.insert(0, os.path.join(ROOT, "main", "REPLICATION"))
sys.path.insert(0, os.path.join(ROOT, "main", "REPLICATION", "claude_plots"))
import gradflow_analysis as ga                                                   # noqa: E402

BELIEFS = os.path.join(ROOT, "main", "DATA", "experimental", "cosyne_fig1", "beliefs.npz")
N_EP = 20000
STEPS = (0, 29)                                                                  # t = 1 and t = T = 30


def _beliefs(net, D, batches, check=True):
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
        m.obs_flat, m.active_Z, m.goal_ind, m.joint_goal_belief = (
            torch.as_tensor(D[f"ep_{k}"][rows], device=dev) for k in ("obs_flat", "active_Z", "goal_ind",
                                                                      "joint_goal_belief"))
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
            out.append(q[:, list(STEPS)].float().cpu().numpy())
        print(f"{net}: belief batch {b + 1} of {batches}", flush=True)
    del agent
    gc.collect()
    return np.concatenate(out)


def collect_beliefs(n_ep=N_EP):
    if os.path.exists(BELIEFS):
        return dict(np.load(BELIEFS))
    Dt, De = ga.harvest("trained"), ga.harvest("echo")
    nb = n_ep // ga.CFG["batch_num"]
    res = {"trained": _beliefs("trained", Dt, nb), "echo": _beliefs("echo", De, nb),
           "echo_on_trained": _beliefs("echo", Dt, nb, check=False),
           "ctx_trained": Dt["ep_ctx_vals"][:n_ep], "ctx_echo": De["ep_ctx_vals"][:n_ep], "steps": np.array(STEPS)}
    os.makedirs(os.path.dirname(BELIEFS), exist_ok=True)
    np.savez(BELIEFS, **res)
    print(f"saved {BELIEFS}")
    return res




# ═══════════════════════════════════════════════════════════════════
# The figure, at its final size: 7.5 x 5.3 in (the full text width of letter paper with 0.5 in margins)
# ═══════════════════════════════════════════════════════════════════

import matplotlib                                                                # noqa: E402
import matplotlib.pyplot as plt                                                  # noqa: E402
import matplotlib.patheffects as pe                                              # noqa: E402
from matplotlib.colors import Normalize                                          # noqa: E402
from sklearn.decomposition import PCA                                            # noqa: E402
from sklearn.linear_model import Ridge                                           # noqa: E402

sp = ga.sp                                                                       # supplementary_plotter2
W_IN, H_IN = 7.5, 5.3
C_TR, C_EC = sp.AGENT_COLORS["Trained"], sp.AGENT_COLORS["Echo"]
C_JOINT, C_NAIVE = "k", "0.55"
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
    fig.text(x / W_IN, y / H_IN, r"$\mathbf{%s}$" % s, fontsize=FS_LETTER, ha="left", va="bottom")


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def training_series():
    """the early-learning panels' data, from supplementary_plotter2's harvest (no smoothing, as there)"""
    Hc = np.load(sp.HARVEST_CACHE)
    out = {}
    for lab, tag in (("Trained", "tr_"), ("Echo", "ec_")):
        A = sp.Bunch({k[len(tag):]: Hc[k] for k in Hc.files if k.startswith(tag)})
        out[lab] = {"acc": A.test_acc_through_training[:, -1].astype(np.float64),
                    "coef": A.test_SII_coef_through_training.astype(np.float64), "corr": fr_correlation()[lab],
                    "pr": sp.pr_gain(A), "joint": float(np.mean(A.joint_acc)), "naive": float(np.mean(A.naive_acc))}
    joint = np.mean([out[l]["joint"] for l in out])
    naive = np.mean([out[l]["naive"] for l in out])
    crossing = int(np.flatnonzero(out["Trained"]["acc"] > naive)[0]) + 1                 # testing episode (x = index + 1)
    return out, joint, naive, crossing


def _curve(ax, y, c, ms=2.2):
    """a lag curve in the dynamics figure's style: dotted and faded before the event, solid after"""
    past, fut = ga.TAU <= 0, ga.TAU >= 0
    ax.plot(ga.TAU[past], y[past], c=c, ls=":", lw=1., alpha=.55)
    ax.plot(ga.TAU[fut], y[fut], c=c, lw=1.1, marker="o", ms=ms, mec="k", mew=.3)


def _band_edges(net, ids):
    """each kept band's span of the cumulative variance (in %), from the fixed PCA basis"""
    Bs = ga.basis(net)
    cum = np.concatenate([[0], np.cumsum(Bs["evr"])])
    sid = ga.cut_segments(Bs["evr"], 100)
    starts = [cum[np.flatnonzero(sid == b)[0]] for b in ids]
    return 100 * np.array(starts + [cum[-1]])


FR_IMPLIED = os.path.join(ROOT, "main", "DATA", "experimental", "cosyne_fig1", "fr_corr_implied.npz")
BAND_ENERGY = os.path.join(ROOT, "main", "DATA", "experimental", "cosyne_fig1", "band_energy_r.npz")
EXAMPLE = (2525, 0)                    # episode, observable: an entangled likelihood table with informative marginals


def fr_correlation():
    """the point-biserial r between final-step FR and final-step correctness in each test batch, through training.
    The run logged only the logistic fit of correctness on FR (env_model_data_manager: sklearn's LogisticRegression,
    with intercept); its maximum-likelihood fit reproduces cov(FR, correct) exactly, so r follows from the logged slope
    and accuracy, given FR's distribution (a task property: the harvest's final-step FR, pooled). On the 50 harvest
    batches this recovers each batch's r to 0.0002 (Trained) and 0.005 (ESN) on average"""
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
    np.savez(FR_IMPLIED, trained=out["Trained"], echo=out["Echo"])
    return out


def band_energy_r():
    """cross-correlation with entanglement of each 1% band's absolute activity energy ||P_B (u_t - mu)||^2, each band's
    series z-scored within the episode as figure_series' shares -> {net: (TAU, bands)}, bands as figure_series' ids"""
    if os.path.exists(BAND_ENERGY):
        return dict(np.load(BAND_ENERGY))
    N, s = ga.BAND1_EP, slice(ga.SRC[0], ga.SRC[-1] + 1)
    out = {}
    for net in ("trained", "echo"):
        f = np.load(ga.CACHE.format(net), mmap_mode="r")
        Ea = np.asarray(ga.band_both(net, 1)["E"][:N], dtype=np.float64)[:, s]
        out[net] = ga._lagged(ga._zs(Ea), ga._zs(f["rev"][:N, s].astype(np.float64)))
        assert out[net].shape == ga.figure_series(net, 1)["r"]["energy"].shape
    np.savez(BAND_ENERGY, **out)
    return out


def bivariate(rg, rc, R=10):
    """colour of each (r_g, r_c): hue from r_g, lightness from r_c (10 levels each)"""
    import colorsys
    h = np.linspace(0, .78, R)[np.asarray(rg)]
    l = np.linspace(.24, .74, R)[np.asarray(rc)]
    return np.array([colorsys.hls_to_rgb(a, b, .85) for a, b in zip(np.ravel(h), np.ravel(l))]).reshape(
        np.shape(rg) + (3,))


def belief_geometry():
    """top-2 PCs of the four agents' marginal beliefs at t = 1 and T = 30 on the same 20,000 episodes (the trained
    network's harvest; the ESN replayed on them), each belief ordered [goal context, other context]; per agent and step
    the ridge (alpha 1) decoders of r_g and r_c from the 2 PCs: R^2, coefficient vectors and the angle between them"""
    B = collect_beliefs()
    N = len(B["trained"])
    f = np.load(ga.CACHE.format("trained"), mmap_mode="r")
    goal, ctx = f["ep_goal_ind"][:N].astype(int), f["ep_ctx_vals"][:N].astype(int)
    assert np.array_equal(ctx, B["ctx_trained"])
    ar = np.arange(N)
    rg, rc = ctx[ar, goal], ctx[ar, 1 - goal]
    agents = {"Trained": B["trained"], "Echo": B["echo_on_trained"],
              "Joint": f["ep_joint_belief"][:N][:, [0, 29]], "Naive": f["ep_naive_belief"][:N][:, [0, 29]]}
    out = {}
    for name, bel in agents.items():
        for i, t in enumerate((1, 30)):
            b = np.asarray(bel[:, i], np.float64)
            X = PCA(n_components=2).fit_transform(np.concatenate([b[ar, goal], b[ar, 1 - goal]], -1))
            d = {"X": X, "rg": rg, "rc": rc}
            for k, y in (("g", rg), ("c", rc)):
                m = Ridge(alpha=1.0).fit(X, y)
                d[f"r2_{k}"], d[f"w_{k}"] = m.score(X, y), m.coef_
            d["angle"] = np.degrees(np.arccos(np.clip(d["w_g"] @ d["w_c"] / np.linalg.norm(d["w_g"]) /
                                                      np.linalg.norm(d["w_c"]), -1, 1)))
            out[(name, t)] = d
    return out


def fig1(energy="absolute"):
    """energy: what (i) shows, "absolute" (||P_B (u_t - mu)||^2) or "share" (over ||u_t - mu||^2, as the main
    figure); the share version is a PNG preview only"""
    path = ga.fig_path("fig1_master.pdf" if energy == "absolute" else "fig1_master_share.pdf")
    T, joint, naive, _ = training_series()
    G = belief_geometry()
    X = {lab: ga.figure_series(net, 1) for net, lab in (("trained", "Trained"), ("echo", "Echo"))}
    Xn = {lab: ga.figure_series(net, 1, ctrl="novelty") for net, lab in (("trained", "Trained"), ("echo", "Echo"))}
    name = {"Trained": "Trained RNN", "Echo": "ESN", "Joint": "Joint Bayes", "Naive": "Naive Bayes"}
    col = {"Trained": C_TR, "Echo": C_EC, "Joint": C_JOINT, "Naive": "0.45"}
    f = np.load(ga.CACHE.format("trained"), mmap_mode="r")
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W_IN, H_IN))
        # ══ band 1, left: (a) the task ══════════════════════════════════════════════════════════════════════════
        ep, oi = EXAMPLE
        g_i = int(f["ep_goal_ind"][ep])
        cv = f["ep_ctx_vals"][ep].astype(int)
        rg0, rc0 = cv[g_i], cv[1 - g_i]
        JL = f["ep_joint_likelihood"][ep, oi].astype(np.float64)           # P(o^i = 1 | r_0, r_1)
        jl = JL.T if g_i == 0 else JL                                       # rows r_c, columns r_g
        NL = f["ep_naive_likelihood"][ep, oi].astype(np.float64)           # (contexts, R)
        nl = np.outer(NL[1 - g_i], NL[g_i])                                 # rows r_c, columns r_g
        nl = np.clip(nl * jl.mean() / nl.mean(), 0, 1)                      # on the joint's probability scale
        obs = f["ep_obs_flat"][ep].astype(int)                              # (T, 5)
        R = 10
        gg, cc = np.meshgrid(np.arange(R), np.arange(R))
        sq = .52
        for (x, y, title), img in zip(((.30, 4.58, "Latents $(r_g, r_c)$"),
                                       (1.18, 4.58, f"Joint $P(o^{{{oi + 1}}}{{=}}1 \\mid r_g, r_c)$"),
                                       (1.18, 3.62, "Naive: $\\propto P(o \\mid r_g)\\,P(o \\mid r_c)$")),
                                      (bivariate(gg, cc), jl, nl)):
            ax = _ax(fig, x, y, sq, sq)
            ax.imshow(img, origin="lower", cmap="Greys", vmin=0, vmax=1, extent=(-.5, R - .5, -.5, R - .5),
                      interpolation="nearest")
            ax.add_patch(plt.Rectangle((rg0 - .5, rc0 - .5), 1, 1, fill=False, ec="k" if img.ndim == 3 else C_TR,
                                       lw=.9))
            ax.set_xticks([0, 9]), ax.set_yticks([0, 9])
            ax.tick_params(length=1.5, pad=1)
            ax.set_xlabel("$r_g$", labelpad=-3)
            ax.set_ylabel("$r_c$", labelpad=-3)
            ax.set_title(title, fontsize=FS_TICK, pad=2)
        # observations o_1 ... o_10 and o_T as rows of dots; the shown observable's column outlined
        ax = _ax(fig, .30, 3.40, sq, .80)
        rows = list(range(10)) + [29]
        for k, t in enumerate(rows):
            yk = len(rows) - k - (1.4 if t == 29 else 1)
            for j in range(5):
                ax.scatter(j, yk, s=7, marker="o", facecolors="k" if obs[t, j] else "white", edgecolors="k",
                           linewidths=.4, zorder=3)
            ax.text(-.9, yk, f"$o_{{{t + 1}}}$", fontsize=FS_TICK, ha="right", va="center")
        ax.scatter([2, 2, 2], len(rows) - 10.7 + np.array([-.18, 0, .18]), s=.6, c="k", lw=0)   # vertical ellipsis
        ax.add_patch(plt.Rectangle((oi - .45, -1.0), .9, len(rows) + .25, fill=False, ec=C_TR, lw=.7))
        ax.set_xlim(-2.1, 4.6), ax.set_ylim(-1.2, len(rows) + .1)
        ax.axis("off")
        ax.set_title("Observations", fontsize=FS_TICK, pad=1)
        _letter(fig, .02, 5.13, "a")
        # ══ band 1, right: (b) belief geometry ══════════════════════════════════════════════════════════════════
        gx0, gw, gh, ggap = 2.55, 1.52, .86, .16
        cols = [((("Trained", 1), ("Joint", 1)), "$t = 1$ (all four agents alike)"),
                ((("Echo", 30), ("Naive", 30)), "$T = 30$"), ((("Trained", 30), ("Joint", 30)), "$T = 30$")]
        for c, (pair, title) in enumerate(cols):
            x = gx0 + c * (gw + ggap)
            for r, (agent, t) in enumerate(pair):
                y = 4.30 - r * (gh + .10)
                ax = _ax(fig, x, y, gw, gh)
                D = G[(agent, t)]
                ax.scatter(D["X"][:, 0], D["X"][:, 1], c=bivariate(D["rg"], D["rc"]), s=.12, alpha=.35, lw=0,
                           rasterized=True)
                ax.set_aspect("equal", adjustable="datalim")
                ax.set_xticks([]), ax.set_yticks([])
                for s_ in ax.spines.values():
                    s_.set_visible(False)
                ctr = D["X"].mean(0)
                scale = .49 * min(np.ptp(D["X"], 0))                             # R^2 = 1 -> 49% of the cloud
                for k, lab, side in (("g", "r_g", 1), ("c", "r_c", -1)):
                    u = D[f"w_{k}"] / np.linalg.norm(D[f"w_{k}"])
                    w = u * scale * D[f"r2_{k}"]
                    a = ax.annotate("", xy=ctr + w, xytext=ctr, arrowprops=dict(arrowstyle="-|>", lw=.9, color="k",
                                                                                 mutation_scale=6, shrinkA=0,
                                                                                 shrinkB=0), zorder=5)   # no flip
                    a.arrow_patch.set_path_effects([pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])
                    # label past the tip and to one side, in points (equal aspect: data and page directions agree),
                    # so near-collinear arrows keep their labels apart
                    ax.annotate(f"${lab}$", xy=ctr + w, xytext=(5 * u[0] - side * 5 * u[1], 5 * u[1] + side * 5 * u[0]),
                                textcoords="offset points", fontsize=FS_TICK, ha="center", va="center", zorder=6,
                                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
                ax.text(.02, .98, name[agent], transform=ax.transAxes, fontsize=FS_TICK, color=col[agent],
                        ha="left", va="top", path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
                ax.text(.98, .02, f"{D['angle']:.0f}$^\\circ$", transform=ax.transAxes, fontsize=FS_TICK,
                        ha="right", va="bottom", path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
            fig.text((x + gw / 2) / W_IN, (4.30 + gh + .04) / H_IN, title, ha="center", va="bottom", fontsize=FS_TITLE)
        for r, lab in enumerate(("Networks", "Bayes observers")):
            fig.text((gx0 - .12) / W_IN, (4.30 - r * (gh + .10) + gh / 2) / H_IN, lab, rotation=90, ha="center",
                     va="center", fontsize=FS_LAB)
        _letter(fig, gx0 - .30, 5.13, "b")
        # ══ band 2: (c) accuracy over training, (d), (e) against accuracy ════════════════════════════════════════
        by, bh = 2.36, .62
        bw = (7.40 - .50 - 2 * .62) / 3
        bx = (.50, .50 + bw + .62, .50 + 2 * bw + 1.24)
        axc, axd, axe = (_ax(fig, x, by, bw, bh) for x in bx)
        n = len(T["Trained"]["acc"])
        epx = np.arange(1, n + 1)
        for lab in ("Echo", "Trained"):
            axc.plot(epx, T[lab]["acc"], c=col[lab], lw=.7, label=name[lab])
            acc_s = sp._smooth(T[lab]["acc"], 11)
            for ax, key in ((axd, "corr"), (axe, "pr")):
                ax.plot(T[lab]["acc"], T[lab][key], c=col[lab], lw=.3, alpha=.18)
                ax.plot(acc_s, T[lab][key], c=col[lab], lw=.7)
        axc.set_xscale("log")
        axc.set_xlim(1, n)
        axc.axhline(joint, c=C_JOINT, ls="--", lw=.8)
        axc.axhline(naive, c=C_NAIVE, ls="--", lw=.8)
        axc.text(1.2, joint + .012, "Optimal joint Bayes", fontsize=FS_TICK, va="bottom", color=C_JOINT)
        axc.text(1.2, naive + .012, "Naive Bayes", fontsize=FS_TICK, va="bottom", color=C_NAIVE)
        axc.set_ylim(.05, .86)
        axc.set_xlabel("Testing episode")
        axc.set_ylabel("Final-step accuracy")
        hd, lb = axc.get_legend_handles_labels()
        axe.legend(hd[::-1], lb[::-1], loc="upper left", frameon=False, handlelength=1.2, labelspacing=.2)
        axd.set_ylabel("Correlation $r$\n(FR, correct)", linespacing=.95)
        axe.set_ylabel("PR(RNN) $-$\nPR(read-in)", linespacing=.95)
        for ax in (axd, axe):
            ax.axvline(naive, c=C_NAIVE, ls="--", lw=.8)
            ax.axvline(joint, c=C_JOINT, ls="--", lw=.8)
            for xv, txt, c, ha, dx in ((naive, "Naive", C_NAIVE, "left", .008), (joint, "Joint", C_JOINT, "right", -.008)):
                ax.text(xv + dx, .03, txt, transform=ax.get_xaxis_transform(), fontsize=FS_TICK, color=c, ha=ha,
                        va="bottom")                                             # the bottom of both panels is free
            ax.axhline(0, c="k", lw=.5, alpha=.4)
            ax.set_xlim(.05, .82)
            ax.set_xlabel("Final-step accuracy")
        for ax in (axc, axd, axe):
            _clean(ax)
        for ax, k in zip((axc, axd, axe), "cde"):
            _letter(fig, ax.get_position().x0 * W_IN - .45, by + bh - .02, k)
        # ══ band 3: (h) cross-correlation with entanglement ═══════════════════════════════════════════════════════
        hy, hh, hgap, hx0 = 1.20, .56, .15, .50
        hw = (7.42 - hx0 - 4 * hgap) / 5
        tops = (("l2", r"$\|u_t\|_2$"), ("pr", "Activity eff. dim."), ("coh", "Credit coherence"),
                ("dW", r"$\|\Delta W\|^2$"), ("band_x", "Cross-band share"))
        H_axes = []
        for i, (key, title) in enumerate(tops):
            ax = _ax(fig, hx0 + i * (hw + hgap), hy, hw, hh)
            for lab in ("Echo", "Trained"):
                if key == "l2":                                                      # beyond novelty: thin
                    ax.plot(ga.TAU, Xn[lab]["r"][key], c=col[lab], lw=.6, alpha=.6, zorder=2)
                _curve(ax, X[lab]["r"][key], col[lab])
            ax.axhline(0, c="k", lw=.5, alpha=.4)
            ax.axvline(0, c="k", lw=.5, alpha=.5)
            ax.set_xlim(-8.8, 8.8)
            ax.set_xticks([-8, -4, 0, 4, 8])
            ax.set_title(r"$\mathbf{h_{%d}}$  " % (i + 1) + title, fontsize=FS_TITLE, pad=2, loc="left")
            ax.set_xlabel(r"Lag $\tau$", labelpad=1)
            _clean(ax)
            H_axes.append(ax)
        H_axes[0].text(8.6, .36, "thin: beyond\nnovelty", fontsize=FS_TICK, ha="right", va="top", color="0.35",
                       linespacing=.9)
        lo = min(min(X[l]["r"][k].min() for l in X) for k, _ in tops)
        hi = max(max(X[l]["r"][k].max() for l in X) for k, _ in tops)
        for ax in H_axes:
            ax.set_ylim(lo - .08 * (hi - lo), hi + .08 * (hi - lo))
            ax.tick_params(labelleft=ax is H_axes[0])
        H_axes[0].set_ylabel("$r$ with\nentanglement", linespacing=.95)
        _letter(fig, .02, hy + hh + .06, "h")
        # ══ band 4: band-resolved heatmaps ════════════════════════════════════════════════════════════════════════
        my, mh, mx0 = .25, .52, .50
        mw = (6.80 - mx0 - .16 - .38) / 3
        xs = (mx0, mx0 + mw + .16, mx0 + 2 * mw + .16 + .38)
        EA = (band_energy_r() if energy == "absolute" else
              {net: X[lab]["r"]["energy"] for net, lab in (("trained", "Trained"), ("echo", "Echo"))})
        etitle = "Band energy" if energy == "absolute" else "Activity energy share"
        maps = (("Trained", EA["trained"], etitle), ("Echo", EA["echo"], etitle),
                ("Trained", X["Trained"]["r"]["cross"], "Cross-band share"))
        vmax = max(np.abs(M).max() for _, M, _ in maps)
        for (lab, M, title), x in zip(maps, xs):
            ax = _ax(fig, x, my, mw, mh)
            net = "trained" if lab == "Trained" else "echo"
            im = ax.pcolormesh(np.arange(ga.TAU[0] - .5, ga.TAU[-1] + 1), _band_edges(net, list(X[lab]["ids"])),
                               M.T, cmap=ga.HEAT_CMAP, vmin=-vmax, vmax=vmax, shading="flat", rasterized=True)
            ax.set_ylim(100, 0)
            ax.axvline(0, c="k", ls="--", lw=.6, alpha=.6)                    # through the tau = 0 column
            ax.set_xticks([-8, -4, 0, 4, 8])
            ax.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
            ax.tick_params(labelleft=x == xs[0])
            ax.set_xlabel(r"Lag $\tau$", labelpad=1)
            ax.set_title(f"{title}, {name[lab]}", fontsize=FS_TICK + .5, pad=2, color=col[lab])
            if x == xs[0]:
                ax.set_ylabel("Variance band\n(top PCs $\\to$ tail)", linespacing=.95)
            if x == xs[2]:
                emax = np.abs(X["Echo"]["r"]["cross"]).max()
                ax.text(.98, .04, f"ESN: max $|r|$ = {emax:.2f}", transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=FS_TICK, color=C_EC, path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
        _letter(fig, .02, my + mh + .06, "i")
        _letter(fig, xs[2] - .32, my + mh + .06, "j")
        cax = _ax(fig, xs[2] + mw + .1, my + .03, .055, mh - .06)
        cb = fig.colorbar(im, cax=cax)
        cb.outline.set_linewidth(.4)
        cb.ax.tick_params(labelsize=FS_TICK, length=1.5, pad=1)
        cb.set_label("$r$ with\nentanglement", fontsize=FS_TICK, labelpad=2, linespacing=.95)
        if energy == "absolute":
            fig.savefig(path)
        fig.savefig(path.replace(".pdf", ".png"), dpi=300)
        plt.close(fig)
    print(f"saved {path} ({W_IN} x {H_IN} in)")
    return {"geometry": {k: (v["r2_g"], v["r2_c"], v["angle"]) for k, v in G.items()}, "joint": joint, "naive": naive,
            "vmax": vmax, "ranges": [(lab, title, float(M.min()), float(M.max())) for lab, M, title in maps]}


if __name__ == "__main__":
    if sys.argv[1:] == ["beliefs"]:
        collect_beliefs()
    else:
        print(fig1(*sys.argv[1:]))
