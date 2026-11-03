"""The representation as a sheet: each cell's mean belief, joined to its grid neighbours, in panel (c)'s T = 30 plane
(25 Oct 2026). Variants (python plot_fig1c_mesh.py <variant> ...):
    particles   one environment's 5,000 streams (dots in their cell's colour) under its mesh
                -> figure_bin/fig1c_movie_mesh_particles.gif
    fr          one environment's mesh only, each edge coloured by the factorization regret at the step (mean over the
                two cells' streams) -> figure_bin/fig1c_movie_mesh_fr.gif
    envs        the meshes of several environments overlaid, one colour each -> figure_bin/fig1c_movie_mesh_envs.gif
    storyboard  the same overlay at steps 1, 2, 4, 8, 16, 30 (static) -> figure_bin/fig1c_mesh_storyboard.png
    fr_step, ent, ent_step   the mesh coloured by factorization regret or entanglement, ranked among the edges at each
                step (_step) or on one scale across time -> figure_bin/fig1c_movie_mesh_<variant>.gif
    envdist     all 39 environments' meshes, each edge coloured by its distance from the environments' average grid
                (black) -> figure_bin/fig1c_movie_mesh_envdist.gif
    avgvar      the environments' average grid in each step's own plane, edges coloured by their variance across
                environments -> figure_bin/fig1c_movie_mesh_avgvar.gif
    formation   all 39 environments: the cells' mean beliefs split into the shared grid (cell effect) and the
                environment-specific distortion (cell x environment interaction: inter-environment interference), with
                the cumulative joint movement -> figure_bin/fig1c_mesh_formation.png
Data: the alias scan (main/DATA/experimental/fig1c_compression; 100 cells x 50 streams per environment, e = 2 left
out). A cell's mean belief: the mean of its 50 streams' 20 numbers. Plane: fig1.belief_geometry's T = 30 plane
per agent (as fig1c_movie_fixed); between steps the fig1_movie._blend spline. Dashed grey: the grid of the cells'
targets (fig1.targets: the Optimal's mean correct T = 30 belief). Joint movement: figdata/round4b/updates_50k.npz.
Run from the repository root:
    CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/plot_fig1c_mesh.py particles fr envs storyboard formation"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.collections import LineCollection                                 # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm                    # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from PIL import Image                                                             # noqa: E402

SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
UPD = os.path.join(SCAN, "figdata", "round4b", "updates_50k.npz")
T, M = 30, 50
AGENTS = (("Trained", "Echo"), ("Joint", "Naive"))                                # the fixed movie's 2 x 2 order
FLAT = [a for row in AGENTS for a in row]
NAME = {"Joint": "Optimal", "Trained": "Trained RNN", "Naive": "Naive", "Echo": "ESN"}
COLOR = {"Joint": "k", "Trained": "#1f77b4", "Naive": "0.5", "Echo": "#ff7f0e"}
ENV1 = 34                                                                         # the env movies' environment
ENVS = (34, 7, 15, 20, 29, 39)
ENV_COL = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#c49a00")
FR_CMAP = LinearSegmentedColormap.from_list("fr", ["#bdbdbd", "#fdbb84", "#e34a33", "#7f0000"])
DIST_MAX = 3.                                                                     # envdist colour scale (cells)
DPI = 80
FRAME_STEP = 2                                                                    # every 2nd frame (same total time)


# ═══════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════

def load(envs, streams=False):
    """per agent: cell means (len(envs), 100, T, 20), a projector into its T = 30 plane, the targets' positions; per
    environment the cells' mean FR (len(envs), 100, T); the grid's edges; optionally the streams (5000, T, 20)"""
    import fig1 as cf
    import fig1_movie as cm
    G = cf.belief_geometry()
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    assert (meta["cell_of"] == np.repeat(np.arange(100), M)).all()
    cells = meta["cells"]
    idx = {(int(g), int(c)): k for k, (g, c) in enumerate(cells)}
    edges = np.array([(k, idx[(g + dg, c + dc)]) for (g, c), k in idx.items() for dg, dc in ((1, 0), (0, 1))
                      if (g + dg, c + dc) in idx])
    home, _ = cf.targets([tuple(int(v) for v in c) for c in cells])
    fr = meta["FR"][list(envs)].astype(np.float64).reshape(len(envs), 100, M, T).mean(2)
    D = {}
    for a in FLAT:
        B = np.load(os.path.join(SCAN, f"scan_{a}.npz"))["beliefs"]
        X = B[list(envs)].astype(np.float32)                                        # (E, 5000, T, 20)
        proj = (lambda V, Dg=G[(a, T)]: cm._proj(Dg, V))
        D[a] = dict(Mn=X.reshape(len(envs), 100, M, T, 20).mean(2), proj=proj, home=proj(home),
                    S=X[0] if streams else None)
    return D, fr, edges, cells, cm


def _box(points):
    lo, hi = points.min(0), points.max(0)
    c, s = (lo + hi) / 2, (hi - lo).max() * 1.06
    return (c[0] - s / 2, c[0] + s / 2), (c[1] - s / 2, c[1] + s / 2)


# ═══════════════════════════════════════════════════════════════════
# Movies
# ═══════════════════════════════════════════════════════════════════

def build_movie(variant, dpi=DPI):
    """the movie's figure and moving artists -> fig, update(tau), the frame schedule [(tau, ms)]"""
    import fig1 as cf
    envs = {"envs": ENVS, "envdist": tuple(e for e in range(40) if e != 2)}.get(variant, (ENV1,))
    D, fr, edges, cells, cm = load(envs, streams=variant == "particles")
    stat_kind = variant.split("_")[0] if variant.split("_")[0] in ("fr", "ent") else None
    per_step = variant.endswith("_step")
    if stat_kind == "ent":
        meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
        stat = meta["rev"][list(envs)].astype(np.float64).reshape(len(envs), 100, M, T).mean(2)
    else:
        stat = fr
    s0 = cm.schedule()
    sched = [(s0[i][0], sum(ms for _, ms in s0[i:i + FRAME_STEP])) for i in range(0, len(s0), FRAME_STEP)]
    pw, gap = 3.0, .4
    cbar = variant != "particles" and variant != "envs"
    W = .3 + 2 * pw + gap + (1.25 if cbar else .3)
    H = .6 + 2 * (pw + .32) + .55
    fig = plt.figure(figsize=(W, H), facecolor="white", dpi=dpi)
    frn = (plt.Normalize(0, 1) if per_step else
           {"fr": LogNorm(.05, 30), "ent": LogNorm(.02, 2)}.get(stat_kind, plt.Normalize(0, DIST_MAX)))
    cmap = FR_CMAP
    art = []
    for r, row in enumerate(AGENTS):
        for c, a in enumerate(row):
            ax = fig.add_axes([(.3 + c * (pw + gap)) / W, (H - .6 - (r + 1) * pw - r * .32) / H, pw / W, pw / H])
            P = np.stack([D[a]["proj"](D[a]["Mn"][:, :, t]) for t in range(T)], 2)  # (E, 100, T, 2)
            ax.set_xlim(*_box(np.vstack([P.reshape(-1, 2), D[a]["home"]]))[0])
            ax.set_ylim(*_box(np.vstack([P.reshape(-1, 2), D[a]["home"]]))[1])
            ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(NAME[a], fontsize=11, pad=3)
            ax.add_collection(LineCollection(D[a]["home"][edges], colors="0.78", linewidths=.55,
                                             linestyles=(0, (3, 2)), zorder=0))
            items = {}
            if variant == "particles":
                col = cf.bivariate(cells[np.repeat(np.arange(100), M), 0], cells[np.repeat(np.arange(100), M), 1])
                perm = np.random.default_rng(0).permutation(100 * M)
                items["dots"] = ax.scatter(*D[a]["proj"](D[a]["S"][:, 0])[perm].T, c=np.asarray(col)[perm], s=1.2,
                                           lw=0, alpha=.55, zorder=1)
                items["perm"] = perm
                items["mesh"] = [ax.add_collection(LineCollection(P[0, :, 0][edges], colors="0.1", linewidths=.75,
                                                                  zorder=3))]
            elif stat_kind:
                lc = LineCollection(P[0, :, 0][edges], cmap=cmap, norm=frn, linewidths=1.4, zorder=3)
                lc.set_array(np.zeros(len(edges)))
                items["mesh"] = [ax.add_collection(lc)]
            elif variant == "envdist":
                hm = D[a]["home"]
                items["unit"] = np.linalg.norm(hm[edges[:, 0]] - hm[edges[:, 1]], axis=1).mean()   # one cell
                items["mesh"] = []
                for k in range(len(envs)):
                    lc = LineCollection(P[k, :, 0][edges], cmap=cmap, norm=frn, linewidths=.45, alpha=.7, zorder=2)
                    lc.set_array(np.zeros(len(edges)))
                    items["mesh"].append(ax.add_collection(lc))
                items["avg"] = ax.add_collection(LineCollection(P[:, :, 0].mean(0)[edges], colors="k", linewidths=1.1,
                                                                zorder=4))
            else:
                items["mesh"] = [ax.add_collection(LineCollection(P[k, :, 0][edges], colors=ENV_COL[k], linewidths=.8,
                                                                  alpha=.85, zorder=3)) for k in range(len(envs))]
            art.append((a, items))
    if cbar:
        cx = fig.add_axes([(.3 + 2 * pw + gap + .25) / W, (H - .6 - 2 * pw - .32) / H, .13 / W, (2 * pw + .32) / H])
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=frn), cax=cx, extend="neither" if per_step else "both")
        cb.set_label({"fr": "factorization regret at the step", "ent": "entanglement of the step"}.get(
            stat_kind, "distance from the environments' average grid (cells)") +
            (" (rank among the edges)" if per_step else (" (nats)" if stat_kind else "")), fontsize=8)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_visible(False)
    clock = fig.text(.3 / W, (H - .27) / H, "", fontsize=13, va="center")
    note = {"particles": "Environment %d: dots, its 5,000 streams (colour: their cell); black grid: the cells' mean beliefs "
                         "joined to their grid neighbours; dashed: the targets' grid." % ENV1,
            "fr": "Environment %d: the cells' mean beliefs joined to their grid neighbours, each edge coloured by the "
                  "factorization regret of its two cells at the step; dashed: the targets' grid." % ENV1,
            "envs": "Six environments (%s), one colour each: the cells' mean beliefs joined to their grid neighbours; "
                    "dashed: the targets' grid." % ", ".join(map(str, ENVS)),
            "envdist": "All 39 environments' grids, each edge coloured by its distance from the environments' average "
                       "grid (black), in cells of the targets' grid; dashed: the targets' grid."}.get(
        variant, "Environment %d: the cells' mean beliefs joined to their grid neighbours, each edge coloured by the %s "
                 "of its two cells at the step%s; dashed: the targets' grid." % (
                     ENV1, {"fr": "factorization regret", "ent": "entanglement"}.get(stat_kind, ""),
                     " (ranked among the edges at each step)" if per_step else ""))
    import textwrap
    fig.text(.3 / W, .32 / H, textwrap.fill(note, 135), fontsize=7, color="0.35", va="top", linespacing=1.3)

    def update(tau):
        s = int(np.floor(tau))
        f = min(tau - s, 1.)
        s1 = min(s + 1, T - 1)
        for a, items in art:
            Mt = cm._blend(D[a]["Mn"], tau, axis=2)                                # (E, 100, 20)
            Pt = np.stack([D[a]["proj"](Mt[k]) for k in range(len(Mt))])              # (E, 100, 2)
            if variant == "envdist":
                avg = Pt.mean(0)
                items["avg"].set_segments(avg[edges])
            for k, lc in enumerate(items["mesh"]):
                pos = Pt[k]
                lc.set_segments(pos[edges])
                if stat_kind:
                    sc = (1 - f) * stat[0, :, s] + f * stat[0, :, s1]
                    v = np.sqrt(np.maximum(sc[edges[:, 0]], 1e-6) * np.maximum(sc[edges[:, 1]], 1e-6))
                    lc.set_array(np.argsort(np.argsort(v)) / (len(v) - 1) if per_step else v)
                elif variant == "envdist":
                    dv = np.linalg.norm(pos - avg, axis=1)
                    lc.set_array((dv[edges[:, 0]] + dv[edges[:, 1]]) / 2 / items["unit"])
            if variant == "particles":
                items["dots"].set_offsets(D[a]["proj"](cm._blend(D[a]["S"], tau, axis=1))[items["perm"]])
        clock.set_text("step $t$ = %d" % (int(np.floor(tau + 1e-9)) + 1))

    return fig, update, sched


def movie(variant):
    fig, update, sched = build_movie(variant)

    def grab():
        fig.canvas.draw()
        return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    protos = []
    for tau in (0., 3., 9., T - 1.):
        update(tau)
        protos.append(np.asarray(grab()))
    pal = Image.fromarray(np.concatenate(protos, 0)).quantize(colors=255, method=Image.Quantize.MEDIANCUT,
                                                              dither=Image.Dither.NONE)
    frames = []
    for tau, _ in sched:
        update(tau)
        frames.append(grab().quantize(palette=pal, dither=Image.Dither.NONE))
    plt.close(fig)
    from main.utils import fig_path
    path = fig_path(f"fig1c_movie_mesh_{variant}.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[ms for _, ms in sched], loop=0)
    print(f"saved {path}: {len(frames)} frames, {frames[0].size[0]} x {frames[0].size[1]} px, "
          f"{os.path.getsize(path) / 1e6:.1f} MB", flush=True)


# ═══════════════════════════════════════════════════════════════════
# Static figures
# ═══════════════════════════════════════════════════════════════════

def storyboard(steps=(1, 2, 4, 8, 16, 30)):
    D, fr, edges, cells, cm = load(ENVS)
    pw, gap = 1.85, .12
    W, H = .9 + len(steps) * (pw + gap), .75 + 4 * (pw + gap) + .55
    fig = plt.figure(figsize=(W, H))
    for r, a in enumerate(("Joint", "Trained", "Naive", "Echo")):
        P = np.stack([D[a]["proj"](D[a]["Mn"][:, :, t - 1]) for t in steps], 2)      # (E, 100, len(steps), 2)
        xl, yl = _box(np.vstack([P.reshape(-1, 2), D[a]["home"]]))
        for c, t in enumerate(steps):
            ax = fig.add_axes([(.9 + c * (pw + gap)) / W, (H - .75 - (r + 1) * pw - r * gap) / H, pw / W, pw / H])
            ax.add_collection(LineCollection(D[a]["home"][edges], colors="0.8", linewidths=.4, linestyles=(0, (3, 2))))
            for k in range(len(ENVS)):
                ax.add_collection(LineCollection(P[k, :, c][edges], colors=ENV_COL[k], linewidths=.55, alpha=.85))
            ax.set_xlim(*xl), ax.set_ylim(*yl), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title("$t$ = %d" % t, fontsize=11, pad=3)
            if c == 0:
                ax.set_ylabel(NAME[a], fontsize=11)
    fig.text(.9 / W, .3 / H, "The cells' mean beliefs joined to their grid neighbours, in each agent's T = 30 plane; six "
             "environments of the alias scan (%s), one colour each; dashed: the targets' grid." % ", ".join(map(str, ENVS)),
             fontsize=8, color="0.35")
    from main.utils import fig_path
    path = fig_path("fig1c_mesh_storyboard.png")
    fig.savefig(path, dpi=250)
    plt.close(fig)
    print("saved", path, flush=True)


def _crossing_share(pos, edges):
    """share of the grid's edges that cross at least one edge they do not share a cell with"""
    p, q = pos[edges[:, 0]], pos[edges[:, 1]]                                     # (n, 2)
    def orient(a, b, c):
        return np.sign((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))
    A, B = p[:, None], q[:, None]
    C, Dd = p[None], q[None]
    cross = (orient(A, B, C) * orient(A, B, Dd) < 0) & (orient(C, Dd, A) * orient(C, Dd, B) < 0)
    share = (edges[:, None, 0] == edges[None, :, 0]) | (edges[:, None, 0] == edges[None, :, 1]) | \
            (edges[:, None, 1] == edges[None, :, 0]) | (edges[:, None, 1] == edges[None, :, 1])
    return (cross & ~share).any(1).mean()


def decompose(X):
    """X (E, K, T, d): cell means of E environments and K cells -> per step the variances (summed over the d belief
    numbers, mean over environments and cells) of the cell effect (the grid shared by all environments), the
    environment effect (a shift of the whole grid) and their interaction (the environment-specific distortion of the
    grid: inter-environment interference); they add up to the total"""
    mu = X.mean((0, 1), keepdims=True)
    ce, ee = X.mean(0, keepdims=True) - mu, X.mean(1, keepdims=True) - mu
    it = X - mu - ce - ee
    V = lambda Y: (Y ** 2).sum(-1).mean((0, 1))
    return V(ce), V(ee), V(it)


def formation():
    envs = [e for e in range(40) if e != 2]
    D, fr, edges, cells, cm = load(envs)
    dec = {a: decompose(D[a]["Mn"].astype(np.float64)) for a in FLAT}
    U = np.load(UPD)
    jm = {}
    for a in FLAT:
        u = np.abs(U["u_" + a].astype(np.float64))
        uc = u - u.mean(0)
        jm[a] = np.cumsum((uc[..., 0] * uc[..., 1]).sum(0) / (len(u) - 1))       # cumulative Cov over steps
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    t = np.arange(1, T + 1)
    pw, ph, gx = 2.9, 2.0, .85
    W, H = .85 + 3 * pw + 2 * gx + .3, .95 + ph + 1.0
    fig = plt.figure(figsize=(W, H))
    sty = {"Trained": dict(lw=2.6, alpha=.45), "Joint": dict(lw=1.3), "Naive": dict(lw=1.3), "Echo": dict(lw=1.3)}
    order = ("Joint", "Trained", "Naive", "Echo")
    panels = (("The shared grid and the interference",
               r"cell effect (solid) and cell $\times$ environment interaction (dashed)", "variance (probability$^2$)"),
              ("Inter-environment interference", r"interaction / (cell effect + interaction)", "share of the grid's variance"),
              ("Joint movement so far", r"$\Sigma_{s \leq t}\,\mathrm{Cov}(|u_g(s)|, |u_c(s)|)$", r"grid units$^2$"))
    for c_, (name, formula, ylab) in enumerate(panels):
        ax = fig.add_axes([(.85 + c_ * (pw + gx)) / W, .95 / H, pw / W, ph / H])
        for a in order:
            vc, ve, vi = dec[a]
            if c_ == 0:
                ax.plot(t, vc, c=COLOR[a], **sty[a])
                ax.plot(t, vi, c=COLOR[a], ls=(0, (3, 2)), **sty[a])
            elif c_ == 1:
                ax.plot(t, vi / (vc + vi), c=COLOR[a], **sty[a])
            else:
                ax.plot(t, jm[a], c=COLOR[a], **sty[a])
        ax.set_xlim(.5, T + .5), ax.set_xticks([1, 10, 20, 30]), ax.set_xlabel("Step $t$", labelpad=1)
        if c_ == 1:
            ax.set_ylim(0, .6)
        ax.set_ylabel(ylab)
        ax.spines[["top", "right"]].set_visible(False)
        ax.annotate(name, (.5, 1), xycoords="axes fraction", xytext=(0, 17), textcoords="offset points", ha="center",
                    va="baseline", fontsize=FS["label"] + .5)
        ax.annotate(formula, (.5, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                    ha="center", va="baseline", fontsize=FS["tick"] - .5, color="0.25")
        x0, yy, w, h = ax.get_position().bounds
        fig.text(x0 - .62 / W, yy + h + 17 / 72 / H, r"$\mathbf{%s}$" % "abc"[c_], fontsize=FS["label"] + 2,
                 va="baseline")
    hand = [Line2D([], [], c=COLOR[a], label=NAME[a], **sty[a]) for a in order]
    fig.legend(handles=hand, loc="upper center", bbox_to_anchor=(.5, 1 - .04 / H), ncol=4, frameon=False,
               fontsize=FS["tick"], handlelength=1.8, columnspacing=1.4)
    fig.text(.85 / W, .45 / H, "a, b: the cells' mean beliefs X[e, k, t] (20 numbers; 50 streams each) of all 39 environments "
             "of the alias scan, split additively: grand mean + cell effect (the grid every environment shares) +\n"
             "environment effect (a shift of the whole grid, near 0 here) + their interaction (the environment-specific "
             "distortion of the grid). c: the harvest's 48,968 trajectories, the covariance of the two update sizes at "
             "each step, summed.", fontsize=FS["tick"] - 1.5, color="0.35", va="top", linespacing=1.4)
    path = fig_path("fig1c_mesh_formation.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print("saved", path)
    for a in order:
        vc, ve, vi = dec[a]
        print("%-8s cell effect t=1/5/10/30 %s | env effect %s | interaction %s | share at T %.2f" % (
            a, np.round(vc[[0, 4, 9, 29]], 3), np.round(ve[[0, 4, 9, 29]], 4), np.round(vi[[0, 4, 9, 29]], 3),
            vi[-1] / (vc[-1] + vi[-1])))


def avg_data():
    """per agent: the environments' average grid Xm (100, T, 20), each cell's variance across the 39 environments
    (100, T), the per-step projector; the cells' FR averaged over environments (100, T); targets, edges, cells"""
    envs = [e for e in range(40) if e != 2]
    D, fr, edges, cells, cm = load(envs)
    BV, rg, rc = cm.beliefs_every_step()
    import fig1 as cf
    home, _ = cf.targets([tuple(int(v) for v in c) for c in cells])
    A = {}
    for a in FLAT:
        X = D[a]["Mn"].astype(np.float64)                                            # (E, 100, T, 20)
        Xm = X.mean(0)
        S = cm.step_planes(BV[a], rg, rc)
        proj = (lambda V, tau, S=S: (np.asarray(V, np.float64) - cm._blend(S["m"], tau)) @ cm._polar(cm._blend(S["U"], tau)))
        mu = X.mean((0, 1), keepdims=True)
        it = X - X.mean(0, keepdims=True) - X.mean(1, keepdims=True) + mu         # cell x environment interaction
        A[a] = dict(Xm=Xm, var=((X - Xm) ** 2).sum(-1).mean(0), inter=(it ** 2).sum(-1).mean(0), proj=proj)
    return A, fr.mean(0), home, edges, cells, cm


def movie_avgvar(kind="var"):
    """the environments' average grid in each step's own plane (fig1_movie.step_planes of the harvest cloud, as
    fig1c_movie_pcs), each edge coloured by the variance of its two cells across the 39 environments (kind "var": full
    belief space, probability^2, the same scale for every agent) or by their factorization regret averaged over the
    environments (kind "fr": geometric mean of the two cells; fixed log scale)
    -> figure_bin/fig1c_movie_mesh_avg<kind>.gif"""
    A, frm, home, edges, cells, cm = avg_data()
    s0 = cm.schedule()
    sched = [(s0[i][0], sum(ms for _, ms in s0[i:i + FRAME_STEP])) for i in range(0, len(s0), FRAME_STEP)]
    if kind == "var":
        vmax = np.quantile(np.concatenate([A[a]["var"].ravel() for a in FLAT]), .99)
        norm = plt.Normalize(0, vmax)
    else:
        norm = LogNorm(.05, 30)
    pw, gap = 3.0, .4
    W, H = .3 + 2 * pw + gap + 1.25, .6 + 2 * (pw + .32) + .45
    fig = plt.figure(figsize=(W, H), facecolor="white", dpi=DPI)
    art = []
    for r, row in enumerate(AGENTS):
        for c, a in enumerate(row):
            ax = fig.add_axes([(.3 + c * (pw + gap)) / W, (H - .6 - (r + 1) * pw - r * .32) / H, pw / W, pw / H])
            P = np.vstack([A[a]["proj"](A[a]["Xm"][:, t], float(t)) for t in range(T)] +
                          [A[a]["proj"](home, float(t)) for t in (0, 15, 29)])
            xl, yl = _box(P)
            ax.set_xlim(*xl), ax.set_ylim(*yl), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(NAME[a], fontsize=11, pad=3)
            hm = ax.add_collection(LineCollection(A[a]["proj"](home, 0.)[edges], colors="0.8", linewidths=.55,
                                                  linestyles=(0, (3, 2)), zorder=0))
            lc = LineCollection(A[a]["proj"](A[a]["Xm"][:, 0], 0.)[edges], cmap=FR_CMAP, norm=norm, linewidths=1.5,
                                zorder=3)
            lc.set_array(np.zeros(len(edges)))
            ax.add_collection(lc)
            art.append((a, lc, hm))
    cx = fig.add_axes([(.3 + 2 * pw + gap + .25) / W, (H - .6 - 2 * pw - .32) / H, .13 / W, (2 * pw + .32) / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=FR_CMAP, norm=norm), cax=cx, extend="max" if kind == "var" else "both")
    cb.set_label("variance across the 39 environments (probability$^2$)" if kind == "var" else
                 "factorization regret at the step (nats; mean over the environments)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    clock = fig.text(.3 / W, (H - .27) / H, "", fontsize=13, va="center")
    import textwrap
    fig.text(.3 / W, .32 / H, textwrap.fill("The environments' average grid (each cell's mean belief averaged over the 39 "
             "environments) in each step's own PC plane (as the pcs movie); edge colour: " + (
                 "how much the grid varies across environments" if kind == "var" else
                 "the factorization regret of the edge's two cells (averaged over the environments)") +
             "; dashed: the targets' grid.", 135), fontsize=7, color="0.35", va="top", linespacing=1.3)

    def update(tau):
        s = int(np.floor(tau))
        f = min(tau - s, 1.)
        s1 = min(s + 1, T - 1)
        for a, lc, hm in art:
            pos = A[a]["proj"](cm._blend(A[a]["Xm"], tau, axis=1), tau)
            lc.set_segments(pos[edges])
            if kind == "var":
                v = (1 - f) * A[a]["var"][:, s] + f * A[a]["var"][:, s1]
                lc.set_array((v[edges[:, 0]] + v[edges[:, 1]]) / 2)
            else:
                v = np.maximum((1 - f) * frm[:, s] + f * frm[:, s1], 1e-6)
                lc.set_array(np.sqrt(v[edges[:, 0]] * v[edges[:, 1]]))
            hm.set_segments(A[a]["proj"](home, tau)[edges])
        clock.set_text("step $t$ = %d" % (int(np.floor(tau + 1e-9)) + 1))

    def grab():
        fig.canvas.draw()
        return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    protos = []
    for tau in (0., 3., 9., T - 1.):
        update(tau)
        protos.append(np.asarray(grab()))
    pal = Image.fromarray(np.concatenate(protos, 0)).quantize(colors=255, method=Image.Quantize.MEDIANCUT,
                                                              dither=Image.Dither.NONE)
    frames = []
    for tau, _ in sched:
        update(tau)
        frames.append(grab().quantize(palette=pal, dither=Image.Dither.NONE))
    plt.close(fig)
    from main.utils import fig_path
    path = fig_path(f"fig1c_movie_mesh_avg{kind}.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[ms for _, ms in sched], loop=0)
    print(f"saved {path}: {len(frames)} frames, {frames[0].size[0]} x {frames[0].size[1]} px, "
          f"{os.path.getsize(path) / 1e6:.1f} MB", flush=True)



def avg_grid_fr_figure():
    """the environments' average grid at T = 30 in the step's own plane, edges coloured by factorization regret, and per
    cell: the average grid's distance from the cell's target and the environment-specific distortion (cell x
    environment interaction), against the cell's FR -> figure_bin/fig1c_avg_grid_fr.png"""
    A, frm, home, edges, cells, cm = avg_data()
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    order = ("Joint", "Trained", "Naive", "Echo")
    t = T - 1
    norm = LogNorm(.05, 30)
    pw, gap = 2.45, .3
    W = .55 + 4 * pw + 3 * gap + .25 + .12 + .9
    H = .7 + pw + 1.05 + 2.1 + .95
    fig = plt.figure(figsize=(W, H))
    fr_t = np.maximum(frm[:, t], 1e-6)
    for c_, a in enumerate(order):
        ax = fig.add_axes([(.55 + c_ * (pw + gap)) / W, (H - .7 - pw) / H, pw / W, pw / H])
        pos, hm = A[a]["proj"](A[a]["Xm"][:, t], float(t)), A[a]["proj"](home, float(t))
        xl, yl = _box(np.vstack([pos, hm]))
        ax.add_collection(LineCollection(hm[edges], colors="0.8", linewidths=.55, linestyles=(0, (3, 2))))
        lc = LineCollection(pos[edges], cmap=FR_CMAP, norm=norm, linewidths=1.6)
        lc.set_array(np.sqrt(fr_t[edges[:, 0]] * fr_t[edges[:, 1]]))
        ax.add_collection(lc)
        ax.set_xlim(*xl), ax.set_ylim(*yl), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(NAME[a], fontsize=FS["label"] + .5, pad=3)
    cx = fig.add_axes([(.55 + 4 * pw + 3 * gap + .25) / W, (H - .7 - pw) / H, .12 / W, pw / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=FR_CMAP, norm=norm), cax=cx, extend="both")
    cb.set_label("factorization regret at $T$\n(nats; mean over environments)", fontsize=FS["tick"])
    cb.outline.set_visible(False)
    fig.text(.55 / W, (H - .35) / H, "The environments' average grid at $T = 30$ (each cell's mean belief averaged over the "
             "39 environments), edges coloured by the factorization regret of their cells; dashed: the targets' grid",
             fontsize=FS["label"] + .5, va="center")
    fig.text(.12 / W, (H - .35) / H, r"$\mathbf{a}$", fontsize=FS["label"] + 2, va="center")
    tgt = home                                                                      # (100, 20)
    rows = (("Where the average grid folds", r"$\|\bar X_k - \mathrm{target}_k\|$ (belief space)",
             lambda a: np.linalg.norm(A[a]["Xm"][:, t] - tgt, axis=1), "b"),
            ("Where the environments disagree", "environment-specific distortion\n" r"(probability$^2$)",
             lambda a: A[a]["inter"][:, t], "c"))
    bw = (W - .55 - .9 - 1.2) / 2
    for r_, (name, ylab, get, letter) in enumerate(rows):
        ax = fig.add_axes([(.55 + .3 + r_ * (bw + 1.2)) / W, .95 / H, bw / W, 2.1 / H])
        for a in order:
            ax.scatter(fr_t, get(a), s=9, c=COLOR[a], alpha=.75 if a != "Trained" else .45, lw=0, label=NAME[a])
            o = np.argsort(fr_t)
            k = 15
            xs_ = np.convolve(np.log(fr_t[o]), np.ones(k) / k, mode="valid")
            ys_ = np.convolve(get(a)[o], np.ones(k) / k, mode="valid")
            ax.plot(np.exp(xs_), ys_, c=COLOR[a], lw=1.3 if a != "Trained" else 2.6, alpha=1 if a != "Trained" else .45)
        ax.set_xscale("log")
        ax.set_xlabel("factorization regret of the cell at $T$ (nats; mean over environments)", labelpad=1)
        ax.set_ylabel(ylab)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(name, fontsize=FS["label"] + .5, pad=4)
        x0, y0, w, h = ax.get_position().bounds
        fig.text(x0 - .55 / W, y0 + h + 6 / 72 / H, r"$\mathbf{%s}$" % letter, fontsize=FS["label"] + 2, va="baseline")
        if r_ == 0:
            fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=FS["tick"], loc="center",
                       bbox_to_anchor=(.5, (.95 + 2.1 + .62) / H), ncol=4, markerscale=1.8, handletextpad=.3,
                       columnspacing=1.6)
    fig.text(.55 / W, .4 / H, "One point per cell (100 cells; lines: running means over 15 cells ordered by FR). Alias "
             "scan, 39 environments x 100 cells x 50 streams; FR is the same for every agent (the Optimal's and the "
             "Naive's marginals differ by it).", fontsize=FS["tick"] - 1.5, color="0.35")
    path = fig_path("fig1c_avg_grid_fr.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print("saved", path)
    for a in order:
        d = np.linalg.norm(A[a]["Xm"][:, t] - tgt, axis=1)
        hi, lo = fr_t >= np.quantile(fr_t, .8), fr_t <= np.quantile(fr_t, .2)
        print("%-8s distance from target: top-FR fifth %.3f vs bottom fifth %.3f | interaction %.4f vs %.4f | "
              "Spearman(FR, distance) %.2f" % (a, d[hi].mean(), d[lo].mean(), A[a]["inter"][hi, t].mean(),
                                                A[a]["inter"][lo, t].mean(),
                                                np.corrcoef(np.argsort(np.argsort(fr_t)), np.argsort(np.argsort(d)))[0, 1]))


if __name__ == "__main__":
    for v in sys.argv[1:]:
        if v in ("particles", "fr", "envs", "fr_step", "ent", "ent_step", "envdist"):
            movie(v)
        elif v == "storyboard":
            storyboard()
        elif v == "formation":
            formation()
        elif v == "avgvar":
            movie_avgvar("var")
        elif v == "avgfr":
            movie_avgvar("fr")
        elif v == "avgfr_static":
            avg_grid_fr_figure()
