"""The analyses behind Figure 1's caption caveats (not needed to make the figure; fig1.py does that). From the
repository root, as fig1.py (eureka environment, CPU):

    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \
        main/REPLICATION/claude_plots/fig1_checks.py [planes] [time] [forget] [entcheck]   (default: all four)

Each mode writes main/REPLICATION/figure_bin/<name> and prints its numbers:
    planes    -> fig1_pc_planes.png: at T = 30, each agent's beliefs in its PCs 1-2 and 2-3 with the ridge decoders
                 (why (c) shows PCs 2-3 for the ESN and Naive at T = 30: their PC1 carries no linear latent information)
    time      -> fig1_belief_time.png: the latents' PCs at every step, and that one decoder fit at T = 30 serves every
                 step (the PC re-ranking in (c) is not a moving read-out)
    forget    -> fig1_forgetting.png: the dense ESN / Naive T = 30 centre is low-entanglement episodes, not forgetting
                 (it holds fewer forgotten episodes than the rest: 'forgot_T_near' / 'forgot_T_rest')
    entcheck  -> fig1_entanglement_checks.png: where entanglement sits on the grid, which per-episode selector goes
                 with the ESN's failure, and how beliefs move at entangled steps

Inputs: fig1's caches (beliefs.npz and the alias cache, built by a check run if missing) and the trained
network's harvest. Own caches, built on first use next to fig1's (together ~1:45 on the CPU, ~4.3 GB):
beliefs_alltime.npz (the networks' beliefs at every step on 20,000 episodes, forward replay) and belief_time.npz
(belief_time_course's statistics, from its own 5,000-episode replay)."""
import os
import sys
import numpy as np


def _here():
    """this file's folder, also where VS Code's interactive window gives no usable __file__ (then from the repository
    holding the notebook's file or the working directory)"""
    starts = (globals().get("__file__"), globals().get("__vsc_ipynb_file__"), os.getcwd())
    for f in starts[:2]:
        if f and os.path.exists(f) and os.path.basename(f) == "fig1_checks.py":
            return os.path.dirname(os.path.abspath(f))
    for s in starts:
        d = os.path.abspath(s) if s and os.path.exists(s) else None
        while d:
            if os.path.exists(os.path.join(d, "main", "CognitiveGridworld.py")):
                return os.path.join(d, "main", "REPLICATION", "claude_plots")
            d = None if os.path.dirname(d) == d else os.path.dirname(d)
    raise RuntimeError("project root not found")


sys.path.insert(0, _here())
from fig1 import (ga, plt, matplotlib, pe, PCA, Ridge, RC, FS_TICK, FS_LAB, FS_TITLE, NAME, COL,  # noqa: E402
                         N_EP, N_R, STEPS, CACHE_DIR, REPLAY, _harvest, _beliefs, collect_beliefs, belief_geometry,
                         alias_trajectories, _align, _geometry_panel, _title2, _clean)
from main.utils import factorization_regret                                       # noqa: E402  (the root: fig1's path)

BELIEFS_ALL = os.path.join(CACHE_DIR, "beliefs_alltime.npz")
BELIEF_TIME = os.path.join(CACHE_DIR, "belief_time.npz")


def beliefs_alltime(n_ep=N_EP):
    """the networks' marginal beliefs at every step on the trained harvest's first n_ep episodes (forward replay, CPU;
    the ESN replayed on the same episodes), goal-ordered (n, T, 20), float32 -> beliefs_alltime.npz"""
    if os.path.exists(BELIEFS_ALL):
        f_ = np.load(BELIEFS_ALL)
        assert len(f_["Trained"]) == n_ep, f"{BELIEFS_ALL} holds {len(f_['Trained'])} episodes, not n_ep = {n_ep}"
        return {k: f_[k] for k in f_.files}
    Dt = _harvest("trained", REPLAY + ("err",))
    nb, T = n_ep // ga.CFG["batch_num"], ga.CFG["step_num"]
    n = nb * ga.CFG["batch_num"]
    goal, ar = Dt["ep_goal_ind"][:n].astype(int), np.arange(n)
    out = {}
    for name, net, check in (("Trained", "trained", True), ("Echo", "echo", False)):
        b = _beliefs(net, Dt, nb, check=check, steps=range(T))
        out[name] = np.concatenate([b[ar, :, goal], b[ar, :, 1 - goal]], -1).astype(np.float32)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(BELIEFS_ALL, **out)
    return out


def _decoder_arrows(ax, D):
    """the two ridge decoders as arrows from the cloud's centroid, each as long as its R^2 (R^2 = 1 -> 38% of the
    cloud's smaller extent). All halos are drawn before both arrows, each label sits on its own arrow's outer side
    (r_c is anticlockwise of r_g after _align), and nearly parallel arrows are set 1.4 pt apart so both stay visible"""
    X = D["X"]
    pt = np.ptp(ax.get_xlim()) / (ax.get_position().width * ax.figure.get_figwidth() * 72)    # data units per point
    scale = .38 * min(np.ptp(X, 0))
    u = {k: D[f"w_{k}"] / np.linalg.norm(D[f"w_{k}"]) for k in "gc"}
    bis = u["g"] + u["c"]
    nrm = np.array([-bis[1], bis[0]]) / np.linalg.norm(bis)                       # anticlockwise of the bisector
    near = u["g"] @ u["c"] > np.cos(np.radians(30))
    side = {"g": -1, "c": 1}
    ctr = X.mean(0)
    ends = {k: (ctr + side[k] * 1.4 * pt * nrm * near, ctr + side[k] * 1.4 * pt * nrm * near +
                u[k] * scale * D[f"r2_{k}"]) for k in "gc"}
    props = dict(arrowstyle="-|>", lw=.9, mutation_scale=6, shrinkA=0, shrinkB=0)
    for k in "gc":
        a = ax.annotate("", xy=ends[k][1], xytext=ends[k][0], arrowprops=dict(props, color="white"), zorder=4)
        a.arrow_patch.set_path_effects([pe.Stroke(linewidth=2.2, foreground="white")])
    for k in "gc":
        ax.annotate("", xy=ends[k][1], xytext=ends[k][0], arrowprops=dict(props, color="k"), zorder=5)
    for k in "gc":
        v, sd = u[k], side[k]
        ax.figure.add_artist(matplotlib.text.Annotation(                             # on the figure: above any
            f"$r_{k}$", xy=ends[k][1], xycoords=ax.transData, xytext=(9 * v[0] - sd * 9 * v[1], 9 * v[1] + sd * 9 * v[0]),
            textcoords="offset points", fontsize=12, ha="center", va="center", zorder=9,   # trajectory drawn across
            path_effects=[pe.withStroke(linewidth=2, foreground="white")]))


def fig_pc_planes():
    """the (c) caveat, at T = 30: each agent's beliefs in its own PCs 1-2 and 2-3, each plane turned by _align, with the
    2-D ridge decoders, R^2 and their angle, and which plane (c) shows (figure_bin/fig1_pc_planes.png) ->
    {(agent, PCs): (R^2 of r_g, R^2 of r_c, angle)}"""
    shown = {a: d["pcs"] for (a, t), d in belief_geometry().items() if t == 30}
    B = collect_beliefs()
    N = len(B["trained"])
    f = np.load(ga.CACHE.format("trained"))
    goal, ctx = f["ep_goal_ind"][:N].astype(int), f["ep_ctx_vals"][:N].astype(int)
    ar = np.arange(N)
    rg, rc = ctx[ar, goal], ctx[ar, 1 - goal]
    agents = {"Trained": B["trained"], "Echo": B["echo_on_trained"],
              "Joint": f["ep_joint_belief"][:N][:, list(STEPS)], "Naive": f["ep_naive_belief"][:N][:, list(STEPS)]}
    W, Hh, bs = 7.5, 3.32, 1.15                                                    # (layout from the top; the
    res = {}                                                                         # bottom .17 in: the key line)
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W, Hh))
        for j, (a, bel) in enumerate(agents.items()):
            b = np.asarray(bel[:, 1], np.float64)
            pca = PCA(3).fit(np.concatenate([b[ar, goal], b[ar, 1 - goal]], -1))
            P = pca.transform(np.concatenate([b[ar, goal], b[ar, 1 - goal]], -1))
            ev = 100 * pca.explained_variance_ratio_
            x0 = .75 + j * 1.70
            fig.text((x0 + bs / 2) / W, (Hh - .14) / Hh, NAME[a], ha="center", va="baseline", fontsize=FS_TITLE,
                     color=COL[a])
            fig.text((x0 + bs / 2) / W, (Hh - .255) / Hh, "(c) at $T = 30$: PCs %d-%d" % shown[a], ha="center",
                     va="baseline", fontsize=FS_TICK, color="0.4")
            for i, (p1, p2) in enumerate(((0, 1), (1, 2))):
                X = P[:, [p1, p2]]
                d = {"rg": rg, "rc": rc}
                for k, y in (("g", rg), ("c", rc)):
                    m = Ridge(alpha=1.0).fit(X, y)
                    d[f"r2_{k}"], d[f"w_{k}"] = m.score(X, y), m.coef_
                ang = np.degrees(np.arccos(np.clip(d["w_g"] @ d["w_c"] / np.linalg.norm(d["w_g"]) /
                                                   np.linalg.norm(d["w_c"]), -1, 1)))
                d["X"], d["w_g"], d["w_c"] = _align(X, d["w_g"], d["w_c"])
                y0 = Hh - .30 - (i + 1) * bs - i * .22
                ax = fig.add_axes([x0 / W, y0 / Hh, bs / W, bs / Hh])
                _geometry_panel(ax, d)
                _decoder_arrows(ax, d)
                fig.text((x0 + bs / 2) / W, (y0 - .06) / Hh,
                         f"PC{p1 + 1} {ev[p1]:.0f}%, PC{p2 + 1} {ev[p2]:.0f}%:  $R^2$ {d['r2_g']:.2f} / {d['r2_c']:.2f},"
                         f"  {ang:.0f}$^{{\\circ}}$", ha="center", va="top", fontsize=FS_TICK, color="0.25")
                res[(a, (p1 + 1, p2 + 1))] = (round(d["r2_g"], 3), round(d["r2_c"], 3), round(float(ang), 1))
        for i, lab in enumerate(("PCs 1-2", "PCs 2-3")):
            y0 = Hh - .30 - (i + 1) * bs - i * .22
            fig.text(.30 / W, (y0 + bs / 2) / Hh, lab, rotation=90, ha="center", va="center", fontsize=FS_LAB,
                     linespacing=1.1)
        fig.text(.5, .05 / Hh, f"$T = 30$ beliefs ({N:,} episodes), each agent's own PCs, turned as in (c). Under each "
                 r"panel: variance shares, $R^2$ of the $r_g$ / $r_c$ decoders, their angle", ha="center",
                 va="baseline", fontsize=FS_TICK, color="0.3")
        fig.savefig(ga.fig_path("fig1_pc_planes.png"), dpi=300)
        plt.close(fig)
    return res


def belief_time_course(n_ep=5000):
    """does the latent readout of the beliefs move over the trajectory? For each agent (the networks replayed on the
    trained harvest's first n_ep episodes, every step; the Bayes observers from the harvest) and each step t, on the
    goal-ordered 20-dim beliefs:
      pc_r2  (T, 6, 2)  single-PC R^2 of r_g and r_c for PCs 1-6 of that step (PCs are uncorrelated, so these add up)
      own    (T, 2)     held-out R^2 of the ridge decoder fit at t (all 20 dims; fit on even, scored on odd episodes)
      fixed  (T, 2)     held-out R^2 of the decoder fit at T = 30, applied unchanged at every t
      mean   (T, 2)     squared correlation of the posterior mean sum_r r b_t(r) with r (no fitting at all)"""
    if os.path.exists(BELIEF_TIME):
        f = np.load(BELIEF_TIME)
        return {a: {k[len(a) + 1:]: f[k] for k in f.files if k.startswith(a + "_")} for a in ("Trained", "Echo", "Joint", "Naive")}
    T = ga.CFG["step_num"]
    Dt = _harvest("trained", REPLAY + ("err", "ep_ctx_vals", "ep_joint_belief", "ep_naive_belief"))
    nb = max(1, n_ep // ga.CFG["batch_num"])
    n = nb * ga.CFG["batch_num"]
    bel = {"Trained": _beliefs("trained", Dt, nb, steps=range(T)),
           "Echo": _beliefs("echo", Dt, nb, check=False, steps=range(T)),
           "Joint": np.asarray(Dt["ep_joint_belief"][:n], np.float32), "Naive": np.asarray(Dt["ep_naive_belief"][:n], np.float32)}
    goal, ctx = Dt["ep_goal_ind"][:n].astype(int), Dt["ep_ctx_vals"][:n].astype(int)
    ar = np.arange(n)
    ys = np.stack([ctx[ar, goal], ctx[ar, 1 - goal]], 1).astype(np.float64)
    tr, te = ar % 2 == 0, ar % 2 == 1
    r = np.arange(N_R, dtype=np.float64)
    out = {}
    for a, b in bel.items():
        b = np.asarray(b, np.float64)
        V = np.concatenate([b[ar, :, goal], b[ar, :, 1 - goal]], -1)                   # (n, T, 20)
        res = {"pc_r2": np.zeros((T, 6, 2)), "own": np.zeros((T, 2)), "fixed": np.zeros((T, 2)), "mean": np.zeros((T, 2))}
        for t in range(T):
            X = V[:, t]
            Pc = PCA(6).fit_transform(X)
            for j in range(2):
                y = ys[:, j]
                res["pc_r2"][t, :, j] = [np.corrcoef(Pc[:, q], y)[0, 1] ** 2 for q in range(6)]
                m = Ridge(alpha=1e-3).fit(X[tr], y[tr])
                res["own"][t, j] = m.score(X[te], y[te])
                res["mean"][t, j] = np.corrcoef(b[ar, t, goal if j == 0 else 1 - goal] @ r, y)[0, 1] ** 2
        last = [Ridge(alpha=1e-3).fit(V[tr, T - 1], ys[tr, j]) for j in range(2)]
        for t in range(T):
            for j in range(2):
                res["fixed"][t, j] = last[j].score(V[te, t], ys[te, j])
        out[a] = res
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(BELIEF_TIME, **{f"{a}_{k}": v for a, d in out.items() for k, v in d.items()})
    return out


def fig_belief_time():
    """figure_bin/fig1_belief_time.png: where the latents sit in each step's PCs, and whether one readout serves all steps"""
    D = belief_time_course()
    names, colour = NAME, COL
    W, Hh = 7.5, 3.0
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W, Hh))
        t = np.arange(1, D["Trained"]["own"].shape[0] + 1)
        for j, a in enumerate(("Trained", "Echo", "Joint", "Naive")):
            x0 = .55 + j * 1.75
            ax = fig.add_axes([x0 / W, 1.72 / Hh, 1.40 / W, .95 / Hh])
            im = ax.imshow(D[a]["pc_r2"].sum(-1).T, origin="upper", aspect="auto", cmap="Greys", vmin=0, vmax=1,
                           extent=(.5, t[-1] + .5, 6.5, .5), interpolation="nearest")
            ax.set_yticks(range(1, 7))
            ax.tick_params(labelleft=j == 0)
            ax.set_title(names[a], fontsize=FS_TITLE, color=colour[a], pad=3)
            if j == 0:
                ax.set_ylabel("PC of that step\n($R^2$ of $r_g$ + $r_c$)", linespacing=.95)
            ax.set_xticks([1, 10, 20, 30])
            ax = fig.add_axes([x0 / W, .40 / Hh, 1.40 / W, .95 / Hh])
            for key, ls, lab in (("own", "-", "best decoder at each step"), ("fixed", "--", "decoder fit at $T = 30$"),
                                 ("mean", ":", "posterior mean (no fit)")):
                ax.plot(t, D[a][key].mean(-1), c=colour[a] if colour[a] != "k" else "0.15", ls=ls, lw=1.1, label=lab)
            ax.set_ylim(0, 1)
            ax.set_xticks([1, 10, 20, 30])
            ax.set_xlabel("Step $t$", labelpad=1)
            ax.tick_params(labelleft=j == 0)
            if j == 0:
                ax.set_ylabel("$R^2$ (mean of $r_g$, $r_c$)")
                ax.legend(frameon=False, loc="lower right", handlelength=1.6, fontsize=FS_TICK - .5)
            _clean(ax)
        cax = fig.add_axes([(.55 + 3 * 1.75 + 1.46) / W, 1.76 / Hh, .05 / W, .87 / Hh])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=FS_TICK, length=1.5, pad=1)
        cb.ax.set_title("$R^2$", fontsize=FS_TICK, pad=3)                           # of r_g plus r_c, per PC
        fig.savefig(ga.fig_path("fig1_belief_time.png"), dpi=300)
        plt.close(fig)
    return {a: {"own_T": D[a]["own"][-1].round(3).tolist(), "fixed_t1": D[a]["fixed"][0].round(3).tolist(),
                "own_t1": D[a]["own"][0].round(3).tolist(), "mean_t1": D[a]["mean"][0].round(3).tolist(),
                "mean_T": D[a]["mean"][-1].round(3).tolist(),
                "fixed_min": D[a]["fixed"].mean(-1).min().round(3), "own_minus_fixed_max": (D[a]["own"] - D[a]["fixed"]).mean(-1).max().round(3),
                "top_pc_by_t": [int(np.argmax(D[a]["pc_r2"][tt].sum(-1))) + 1 for tt in range(0, 30, 3)]} for a in D}


def forgetting():
    """per-episode forgetting on the trained harvest's first 20,000 episodes: p_t = b_t(r_g*), each agent's belief in
    the true goal value (networks from beliefs_alltime, Bayes observers from the harvest). An episode has forgotten at
    t when its belief once reached twice the prior (max_{s <= t} p_s >= 0.2) and has since lost half of it (p_t <=
    max_{s <= t} p_s / 2). Also the summed entanglement sum_t rev_t and the factorization regret FR_T = KL(b*_T || b_T)
    of the joint goal belief b*_T from the naive one b_T (the paper's forward KL, exact:
    main.utils.factorization_regret, as the run's SII since 2026-11-10; until then the symmetric [KL(b*_T || b_T) +
    KL(b_T || b*_T)] / 2), per episode"""
    BA = beliefs_alltime()
    f = np.load(ga.CACHE.format("trained"))
    N = len(BA["Trained"])
    ar = np.arange(N)
    goal, ctx = f["ep_goal_ind"][:N].astype(int), f["ep_ctx_vals"][:N].astype(int)
    rg = ctx[ar, goal]
    Bj = np.asarray(f["ep_joint_belief"][:N], np.float64)[ar, :, goal]               # (N, T, 10) goal context
    Bn = np.asarray(f["ep_naive_belief"][:N], np.float64)[ar, :, goal]
    P = {"Trained": BA["Trained"][ar, :, rg].astype(np.float64), "Echo": BA["Echo"][ar, :, rg].astype(np.float64),
         "Joint": Bj[ar, :, rg], "Naive": Bn[ar, :, rg]}
    out = {}
    for a, p in P.items():
        run = np.maximum.accumulate(p, 1)
        out[a] = (run >= .2) & (p <= .5 * run)                                       # (N, T) forgotten at t
    FR = factorization_regret(Bj[:, -1], Bn[:, -1])                                # KL(joint || naive), float64
    E = np.asarray(f["rev"][:N], np.float64).sum(1)
    return P, out, FR, E


def fig_forgetting():
    """figure_bin/fig1_forgetting.png: is the dense centre of the ESN / Naive T = 30 planes forgetting? Top: each agent's
    T = 30 plane (as in c) coloured by the percentile of summed entanglement. Bottom: belief in the true value over t
    in the four aliased cells of c (2000 streams each), and the share of all episodes that have forgotten by t"""
    G = belief_geometry()
    _, forgot, FR, E = forgetting()
    AV, a_of, cells, _ = alias_trajectories()
    names, colour = NAME, COL
    order = ("Trained", "Echo", "Joint", "Naive")
    pct = np.argsort(np.argsort(E)) / (len(E) - 1)
    perm = np.random.default_rng(0).permutation(len(E))                             # no colour drawn last everywhere
    W, Hh = 7.5, 3.9
    res = {}
    axr = lambda fig, x, y, w, h: fig.add_axes([x / W, y / Hh, w / W, h / Hh])     # inches, at this figure's size
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W, Hh))
        for j, a in enumerate(order):
            D = G[(a, 30)]
            ax = fig.add_axes([(.30 + j * 1.62) / W, 2.10 / Hh, 1.40 / W, 1.40 / Hh])
            sc = ax.scatter(*D["X"][perm].T, c=pct[perm], cmap="viridis", s=.12, lw=0, rasterized=True, vmin=0, vmax=1)
            ax.set_aspect("equal", adjustable="datalim")
            ax.axis("off")
            ax.set_title(names[a] + ("" if D["pcs"] == (1, 2) else "  (PCs %d-%d)" % D["pcs"]), fontsize=FS_TITLE,
                         color=colour[a], pad=2)
            rms = np.sqrt((D["X"] ** 2).sum(1).mean())
            near = np.hypot(*D["X"].T) < .35 * rms
            res[a] = {"near_centre": near.mean().round(3), "E_pct_near": np.median(pct[near]).round(2),
                      "E_pct_rest": np.median(pct[~near]).round(2), "forgot_T_near": forgot[a][near, -1].mean().round(3),
                      "forgot_T_rest": forgot[a][~near, -1].mean().round(3)}
        cax = fig.add_axes([(.30 + 3 * 1.62 + 1.58) / W, 2.30 / Hh, .05 / W, 1.0 / Hh])
        cb = fig.colorbar(sc, cax=cax, ticks=[0, .5, 1])
        cb.ax.set_yticklabels(["0", "50", "100"])
        cb.ax.tick_params(labelsize=FS_TICK, length=1.5, pad=1)
        cb.ax.set_title("percentile", fontsize=FS_TICK, pad=3)
        fig.text(.5, 3.82 / Hh, r"$T = 30$ beliefs (planes as in c) coloured by summed entanglement, "
                 r"$\Sigma_t$ entanglement$_t$ (percentile over episodes)",
                 ha="center", va="top", fontsize=FS_LAB)
        t = np.arange(1, 31)
        for k, (g_, c_) in enumerate(cells):
            ax = axr(fig, .50 + k * 1.22, .42, .98, 1.10)
            for a in order:
                ax.plot(t, AV[a][a_of == k][:, :, g_].mean(0), c=colour[a], lw=1.1)
            ax.axhline(.1, c=".6", lw=.6, ls=":")
            ax.set_ylim(0, 1)
            ax.set_xticks([1, 10, 20, 30])
            ax.set_xlabel("Step $t$", labelpad=1)
            ax.tick_params(labelleft=k == 0)
            _title2(ax, f"Cell $r_g = {g_}$, $r_c = {c_}$", r"mean $b_t(r_g^{*})$")
            if k == 0:
                ax.set_ylabel("Belief in the\ntrue value", linespacing=.95)
                ax.text(30, .105, "prior", fontsize=FS_TICK - .5, color=".45", ha="right", va="bottom")
            _clean(ax)
            res[f"cell {g_},{c_}"] = {a: AV[a][a_of == k][:, [0, 9, 29], g_].astype(np.float64).mean(0).round(3).tolist()
                                      for a in order}
        ax = axr(fig, .50 + len(cells) * 1.22 + .30, .42, 1.30, 1.10)
        top = FR > np.quantile(FR, .75)
        for a in order:
            ax.plot(t, forgot[a].mean(0), c=colour[a], lw=1.1)
            ax.plot(t, forgot[a][top].mean(0), c=colour[a], lw=.8, ls="--")
            res[f"forgot_T {a}"] = (forgot[a][:, -1].mean().round(3), forgot[a][top, -1].mean().round(3))
        ax.set_xticks([1, 10, 20, 30])
        ax.set_xlabel("Step $t$", labelpad=1)
        ax.set_ylim(0, None)
        _title2(ax, "Forgotten episodes", r"$p_t \leq \frac{1}{2}\,\mathrm{max}_{s \leq t}\, p_s$, $\mathrm{max} \geq 0.2$")
        ax.set_ylabel("Share of episodes", labelpad=1)
        ax.plot([], [], c=".3", lw=1.1, label="all")
        ax.plot([], [], c=".3", lw=.8, ls="--", label="top 25% FR")
        ax.legend(frameon=False, loc="upper left", handlelength=1.6, fontsize=FS_TICK - .5)
        _clean(ax)
        fig.savefig(ga.fig_path("fig1_forgetting.png"), dpi=300)
        plt.close(fig)
    return res


def entanglement_checks():
    """on the trained harvest's first 20,000 episodes: per episode the summed entanglement E = sum_t rev_t, its peak
    max_t rev_t, the factorization regret FR_T (KL(joint || naive) of the goal beliefs at T: the paper's forward KL,
    exact, main.utils.factorization_regret; (KL + KL) / 2 until 2026-11-10), the joint
    observer's information gain (2 log 10 - H of both final marginals) and the grid distance of the truth to the
    nearest border d = min(r_g, 9 - r_g, r_c, 9 - r_c); each agent's final correctness (goal); and the lag curves (g-row
    method, steps 17-30: within-episode z-scores, mean of a[t + tau] rev[t]) of two per-step changes, the log belief
    in both true values and the beliefs' distance from that step's mean belief (20-dim); and by entanglement decile
    (within each step), the share of steps where the two truth beliefs b_t(r_g*), b_t(r_c*) change in opposite
    directions and the mean change of their sum"""
    BA = beliefs_alltime()
    f = np.load(ga.CACHE.format("trained"))
    N = len(BA["Trained"])
    ar = np.arange(N)
    goal, ctx = f["ep_goal_ind"][:N].astype(int), f["ep_ctx_vals"][:N].astype(int)
    rg, rc = ctx[ar, goal], ctx[ar, 1 - goal]
    rev = np.asarray(f["rev"][:N], np.float64)
    Bj, Bn = (np.asarray(f[k][:N], np.float64) for k in ("ep_joint_belief", "ep_naive_belief"))
    V = {"Trained": BA["Trained"].astype(np.float64), "Echo": BA["Echo"].astype(np.float64),
         "Joint": np.concatenate([Bj[ar, :, goal], Bj[ar, :, 1 - goal]], -1),
         "Naive": np.concatenate([Bn[ar, :, goal], Bn[ar, :, 1 - goal]], -1)}
    lg = lambda x: np.log(np.maximum(x, 1e-300))
    bj, bn = V["Joint"][:, -1, :10], V["Naive"][:, -1, :10]
    H = lambda b: -(b * lg(b)).sum(-1)
    out = {"E": rev.sum(1), "peak": rev.max(1), "FR": factorization_regret(bj, bn), "rg": rg, "rc": rc,
           "info": 2 * np.log(10) - H(V["Joint"][:, -1, :10]) - H(V["Joint"][:, -1, 10:]),
           "d": np.minimum.reduce([rg, 9 - rg, rc, 9 - rc])}
    L = np.clip(np.asarray(f["ep_joint_likelihood"][:N], np.float64), 1e-9, 1 - 1e-9).reshape(N, 5, 100)
    tr = np.where(goal == 0, rg * 10 + rc, rc * 10 + rg)                              # the truth's raw (r0, r1) cell
    p_ = L[ar, :, tr][:, :, None]
    kl = (p_ * np.log(p_ / L) + (1 - p_) * np.log((1 - p_) / (1 - L))).sum(1)       # (N, 100) per-step KL to each cell
    out["ident"] = -np.log(np.exp(-ga.CFG["step_num"] * kl).sum(1))                  # identifiability of the truth: its
                                                                                      # log share at T, expected (no data)
    s_ = slice(16, 30)
    b = ga._zs(rev[:, s_])
    R_ = np.argsort(np.argsort(rev[:, 1:], 0), 0) / (N - 1)                          # entanglement decile within each
    dec = np.minimum((R_ * 10).astype(int), 9)                                        # step (steps 2..30)
    for a, v in V.items():
        out[f"correct_{a}"] = (v[:, -1, :10].argmax(-1) == rg).astype(float)
        lp = np.log(np.maximum(v[ar, :, rg], 1e-12)) + np.log(np.maximum(v[ar, :, 10 + rc], 1e-12))
        spread = np.linalg.norm(v - v.mean(0, keepdims=True), axis=-1)
        for k, y in (("dlogp", lp), ("dspread", spread)):
            dy = np.diff(y, axis=1)[:, s_.start - 1:s_.stop - 1]                     # change into steps 17..30
            out[f"lag_{k}_{a}"] = ga._lagged(ga._zs(dy), b)
        pg, pc = np.diff(v[ar, :, rg], axis=1), np.diff(v[ar, :, 10 + rc], axis=1)   # into steps 2..30
        out[f"opp_{a}"] = [((pg * pc) < 0)[dec == q].mean() for q in range(10)]
        out[f"dp_{a}"] = [(pg + pc)[dec == q].mean() for q in range(10)]
    return out


def fig_entanglement_checks():
    """figure_bin/fig1_entanglement_checks.png: where entanglement sits on the grid (and that it is not informativeness),
    which per-episode selector goes with the ESN's failure, and how beliefs move at entangled steps"""
    from scipy.stats import rankdata
    D = entanglement_checks()
    names, colour = NAME, COL
    order = ("Joint", "Trained", "Naive", "Echo")
    W, Hh = 7.5, 4.1
    axr = lambda fig, x, y, w, h: fig.add_axes([x / W, y / Hh, w / W, h / Hh])
    cell = D["rg"] * 10 + D["rc"]
    cnt = np.bincount(cell, minlength=100)
    res = {}
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W, Hh))
        # top 1: mean summed entanglement per true cell, in the key's layout (r_g right, r_c up)
        ax = axr(fig, .42, 2.35, 1.15, 1.15)
        Gm = (np.bincount(cell, D["E"], 100) / cnt).reshape(10, 10)                   # [r_g, r_c]
        im = ax.imshow(Gm.T, origin="lower", cmap="viridis", interpolation="nearest")
        ax.set_xticks([0, 9]); ax.set_yticks([0, 9])
        ax.set_xlabel("$r_g$", labelpad=-4); ax.set_ylabel("$r_c$", labelpad=-4)
        ax.tick_params(length=1.5, pad=1)
        _title2(ax, "Summed entanglement", r"$\Sigma_t$ entanglement$_t$, mean per cell")
        cax = axr(fig, 1.60, 2.40, .05, 1.05)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=FS_TICK, length=1.5, pad=1)
        res["E_corner_centre"] = (Gm[[0, 0, 9, 9], [0, 9, 0, 9]].mean().round(2), Gm[4:6, 4:6].mean().round(2))
        # top 2: against the distance to the border: entanglement, and what is left after informativeness
        ax = axr(fig, 2.25, 2.35, 1.15, 1.15)
        d = D["d"]
        X = np.stack([rankdata(D["info"]), np.ones(len(d))], 1)
        rE = rankdata(D["E"]) / len(d)
        left = rE - X @ np.linalg.lstsq(X, rE, rcond=None)[0] + rE.mean()
        for y, ls, lab in ((rE, "-", "entanglement"), (left, "--", "beyond Optimal info"),
                           (rankdata(D["info"]) / len(d), ":", "Optimal info")):
            ax.plot(range(5), [y[d == k].mean() for k in range(5)], c="k", ls=ls, lw=1.1, marker="o", ms=2.2, label=lab)
        ax.set_xticks(range(5)); ax.set_ylim(0, 1)
        ax.set_xlabel("Distance to border $d$", labelpad=1)
        ax.set_ylabel("Mean percentile", labelpad=1)
        ax.legend(frameon=False, fontsize=FS_TICK - .5, loc="upper right", handlelength=2.6)   # dash gaps beside the
                                                                                                # marker
        _title2(ax, "Border vs centre", r"$d = \min(r_g, 9 - r_g, r_c, 9 - r_c)$")
        _clean(ax)
        # top 3-5: final accuracy against each selector's percentile within the true cell
        sel = (("E", "Summed entanglement", r"$\Sigma_t$ entanglement$_t$"),
               ("peak", "Peak entanglement", r"$\mathrm{max}_t$ entanglement$_t$"),
               ("FR", "Factorization regret", r"$\mathrm{FR}_T$"))
        for i, (k, nm, fm) in enumerate(sel):
            ax = axr(fig, 4.05 + i * 1.18, 2.35, .98, 1.15)
            x = rankdata(D[k])
            Xi = np.stack([rankdata(D["ident"]), np.ones(len(x))], 1)
            x = x - Xi @ np.linalg.lstsq(Xi, x, rcond=None)[0]                       # beyond the truth's identifiability
            lo = np.zeros(len(x))
            for q in range(100):                                                     # percentile within each cell
                m = cell == q
                lo[m] = (rankdata(x[m]) - .5) / m.sum()
            dec = np.minimum((lo * 10).astype(int), 9)
            for a in order:
                ax.plot(np.arange(10) * 10 + 5, [D[f"correct_{a}"][dec == j].mean() for j in range(10)], c=colour[a], lw=1.1)
                res[f"{k} {a} acc bottom/top decile"] = (D[f"correct_{a}"][dec == 0].mean().round(3), D[f"correct_{a}"][dec == 9].mean().round(3))
            ax.set_ylim(0, 1); ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100])
            ax.set_xlabel("Percentile (within cell,\nbeyond identifiability)", labelpad=1, linespacing=.95)
            ax.tick_params(labelleft=i == 0)
            if i == 0:
                ax.set_ylabel("Final-step accuracy", labelpad=1)
            _title2(ax, nm, fm)
            _clean(ax)
        # bottom 1-2: lag curves, how beliefs move at entangled steps; bottom 3-4: by entanglement decile
        for i, (k, nm, fm) in enumerate((("dlogp", "Belief in the truth", r"$\Delta[\log b_t(r_g^*) + \log b_t(r_c^*)]$"),
                                         ("dspread", "Belief spread", r"$\Delta \|b_t - \bar b_t\|$"))):
            ax = axr(fig, .55 + i * 1.78, .42, 1.30, 1.15)
            for a in order:
                y = D[f"lag_{k}_{a}"]
                ax.plot(ga.TAU, y, c=colour[a], lw=1.1, marker="o", ms=1.8)
                res[f"lag {k} {a} tau0"] = y[list(ga.TAU).index(0)].round(3)
            ax.axhline(0, c=".6", lw=.6); ax.axvline(0, c=".6", lw=.6)
            ax.set_xticks([-8, -4, 0, 4, 8])
            ax.set_xlabel(r"Lag $\tau$", labelpad=1)
            if i == 0:
                ax.set_ylabel("$r$ with\nentanglement", linespacing=.95)
            _title2(ax, nm, fm)
            _clean(ax)
        for i, (k, nm, fm, yl) in enumerate((("opp", "Credit to the wrong latent", r"$\Delta b_t(r_g^*)\,\Delta b_t(r_c^*) < 0$",
                                              "Share of steps"),
                                             ("dp", "Mass on the truth", r"$\Delta[b_t(r_g^*) + b_t(r_c^*)]$", "Mean change"))):
            ax = axr(fig, 4.15 + i * 1.75, .42, 1.30, 1.15)
            for a in order:
                ax.plot(np.arange(10) * 10 + 5, D[f"{k}_{a}"], c=colour[a], lw=1.1, marker="o", ms=1.8)
                res[f"{k} {a} bottom/top decile"] = (np.round(D[f"{k}_{a}"][0], 3), np.round(D[f"{k}_{a}"][-1], 3))
            ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100])
            ax.set_xlabel("Entanglement percentile\n(within step)", labelpad=1, linespacing=.95)
            ax.set_ylabel(yl, labelpad=1)
            if k == "dp":
                ax.set_ylim(0, None)
            _title2(ax, nm, fm)
            _clean(ax)
        for j, a in enumerate(order):
            fig.text(6.00 / W, (1.50 - j * .14) / Hh, names[a], color=colour[a], fontsize=FS_TICK, va="center",
                     ha="left")
        fig.savefig(ga.fig_path("fig1_entanglement_checks.png"), dpi=300)
        plt.close(fig)
    return res


if __name__ == "__main__":
    run = {"planes": fig_pc_planes, "time": fig_belief_time, "forget": fig_forgetting, "entcheck": fig_entanglement_checks}
    flags = [a for a in sys.argv[1:] if a.startswith("-") or a.endswith(".json")]   # an interactive kernel's own
    if flags:
        print(f"ignored: {flags}")
    modes = [a for a in sys.argv[1:] if a not in flags]
    unknown = [a for a in modes if a not in run]
    if unknown:
        sys.exit(f"unknown mode(s) {unknown}; the modes are {list(run)}")
    for arg in modes or run:
        print(arg, run[arg]())
