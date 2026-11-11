"""Supplementary figures of the learning dynamics, selected from learning_figures.pdf for the supplementary (John 11-08: "make
a figure (or 2) of the panels from the learning_figures.pdf that you would include in the supplementary ..."). Every panel is redrawn from its verified source module's caches with that module's own code; nothing new
is measured here.
-> main/REPLICATION/figure_bin/supp_learning_behaviour.png   behaviour and representation through training (a-l; built at
   its final size, 7.5 in, in Figure 1's style. 11-09, John: accuracy, the update ladder, FR vs correctness, corner
   splits, the fold strip and the shared grid's purity removed; then the notes and the stage shading removed, Pearson
   instead of rank r; then one row with combined linestyles; then the count decoding instead of the posterior decoding
   ("decoding of N" meant the counts n); then the per-environment decoders and the memory-use panel removed; then
   "let's split shared and context curves again, move the count decoder to over the naive errors panel. And another
   row for: PR, activity norm, effective dim, credit coherence, Delta W, Delta W dim. Plus the FR - credit by lag
   correlation (trained on row 1, ESN on row 2)"; then "i meant we should put a,b,c,e,f,g onto 1 row"; 11-10: "For the
   existing learning_behavior figure remove the grids and turn it into a 2x6 (let's remove panel j and k)": the
   effective dimension and the credit coherence removed, the belief grids moved to fig_learn_grids.py; then "let's also
   polish the learning behavior figure. First let's rearrange top panels so they are: top row: [copy naive, count
   code, activity norm, Delta W, PR(Delta W), PR(u)] bottom row: [Credit vs FR trained, credit vs FR ESN, shared grid,
   shared PR change, context grid, context PR change]"). Two rows of six (letters as drawn since 11-10):
   a copying naive, b count code, c-f activity norm, dW energy, PR(dW), PR(u); g, h credit vs FR (Trained RNN, ESN);
   i shared grid code, j its PR change, k context grid code, l its PR change. Then (11-10) "let's be consitent with the
   wording, let's do titles: [Naive errors, Count decoding ( r^2 (u_t [to] [sum] o_t) ), Activity norm, Weight update
   energy, Weight update dimension, Activity Dimension] But make sure we're not writing PR if it's not actually a
   participation ratio ... Also let's switch the order of c and d. For the bottom row let's make sure i-l share a
   y-lim. And for j and l maybe it would be simpler to show just PR(u-g)?": c weight update energy, d activity norm;
   j, l PR(u_30 - g) itself; i-l one y range. Both dimensions ARE participation ratios: the activity dimension of the
   eigenvalues of cov(u_t) (per step, then the mean), the weight update dimension (td_analysis cr_dim = tr(A)^2 /
   tr(A^2), A = DD ZZ) of the eigenvalues of dW^T dW, dW = Sigma_k delta_k z_k^T one episode's step-t update: both
   written (Sigma lambda)^2 / Sigma lambda^2. The count decoding is R^2 = 1 - SSE / SST (not a squared correlation)
   of the 5 counts n_30 = Sigma_t o_t. Panel definitions (old letters):
   a  the shared count code: the counts n_30 decoded from u_30 by one ridge decoder for every environment (R^2 on
      held-out environments; td_tt_count_code, td_tt_count_var.load_counts; the per-environment decoders, R^2 >= .91
      throughout, removed)
   b  copying naive's errors (naive wrong, optimal right)
   c, d  the dimension PR(g) of the shared grid code g_sh (each cell's mean state over the environments) and of the
      context-specific one g_ctx (each cell's mean within its environment minus the cell's and the environment's
      means; td_tt_griddim, u_30)
   e, f  the total's PR with minus without each, PR(u_30) - PR(u_30 - g). PR is not additive: at 8,000 training
      episodes g_sh is ~4 strong directions holding 24% of the variance, which dominate Sigma lambda^2 and hold the
      total's PR at 46; without them the flat rest has PR 108 (-61, although PR(g_sh) = 4.4)
   k, l  credit vs FR, Trained RNN / ESN: r across episodes, at each error step t (17-30) then the mean, between the
      share of the error's credit that lands k steps back, ||delta_k||^2 / Sigma_k ||delta_k||^2 (td_tt_credit_reach;
      not the credit coherence), and FR at the landing step t - k (td_xcorr_across); colour scale +-.25 (values reach
      -.27; fig_learn_maps' shared scale is +-.2)
   g-j  fig_learn_maps' raw-data maps as lines, the mean over the steps of each checkpoint (the state's PR per step
      then the mean, td_summary_data.state_spectrum_steps; activity norm over steps 1-30; dW energy and dW effective
      dimension over the error steps 17-30; fig_learn_maps.raw_data; the ESN's update measures: what its frozen
      recurrent weights would receive). RESULTS (11-09, start -> extreme -> end): Trained RNN state PR 10.7 -> 1.3 (20)
      -> 57; norm 2.3 -> 7.9 (20) -> 4.1; dW energy 3.2 -> 93 (1,040, the crossing) -> 6.5; dW effective dim 1.01 ->
      1.85 (9,600) -> 1.72. ESN: PR 10.7 -> 18, norm 2.3 -> 3.6, dW energy 3.2 -> 4,750 (rising to the end), dW
      effective dim 1.01 throughout. Removed 11-10: effective dimension (Trained 5.7 -> 2.5 (20) -> 37, ESN 5.7 -> 9.7)
      and credit coherence (Trained 3.1 -> 8.5 (80) -> .54, ESN 3.1-3.6).
   The belief grids (panel o until 11-10) went to fig_learn_grids.py, then came back (John 11-10: "let's actually try to
   put the grids back into learning_behavior as extra rows"). LAYOUT SINCE 11-10 ~15:00 (John: "remove d and f and make
   the figure a 2 x 5, with the heatmaps as the first column"; "the cross-corr cbar looks bad and h and j can be
   clearer. Maybe we title them With / Without X code, and put the PR(...) solid and dashed curves as a legend"):
   a / f credit k steps back vs FR (Trained RNN / ESN, a slim colour bar each), b naive errors, c count decoding, d
   weight update energy, e weight update dimension PR(dW^T dW), g shared grid code, h with / without it (dashed
   PR(u_30), solid PR(u_30 - g_sh)), i context grid code, j with / without it, k the belief grids (fig_learn_grids'
   data and style: 100-5,000 training episodes without 200 / 400, Joint / Naive Bayes at the right). Activity norm
   and activity dimension removed (their values above). PRINT VERSION (11-10 ~16:00, John: "polish this figure to get it
   ready for the paper. readable text, etc."; grid codes "[trained context grid (PR(u_t), PR(grid), PR(u_t - grid)), ESN
   context grid, trained shared grid, ESN shared grid]"): 5.5 in, 7 / 8 / 9 pt, one x label per row; g-j = Trained
   / ESN x context / shared grid code, each with PR(u_30) solid, PR(g) dashed, PR(u_30 - g) dotted (one y range).
   Then (John 11-10 ~16:10: "make the figure much wider and move the legend from j to g. Also let's put the shared
   grids before the context specific grids and change their title to "Z-specific grids". Also for the heatmaps maybe
   "Correlation(Factorization Regret, temporal credit assignment)" ... The training episodes xlabel can go below every
   panel"): 8 in wide; g, h shared (Trained, ESN), i, j Z-specific; legend "PR of u_30 / g / u_30 - g" in g; map titles
   as quoted; y labels on b and c. 11-11: the grids moved back to fig_learn_grids.py ("lets move the grid rows back
   into their own figure again"); 8 in wide; the two maps fill their column with a .10 in gap, the top one under a
   title wrapped to the column ("Correlation between the gradient reaching k steps back and FR k steps back") that
   starts at the other titles' height; one horizontal r bar under the maps on the key line; titles "Shared PR" /
   "Context-dependent PR". Then (11-11 ~15:00): x labels on the top row too; 9 in wide with the extra width to the line
   panels; the bottom row = a key column (the PR line styles and the <u>_x definition) and two pairs of panels (shared,
   context-dependent; Trained and Echo each) .10 in apart within a pair, one title, formula and x label per pair
   (John: "push the bottom row (nonheatmap) panels closer together since they share a ylabel and then put the legend as
   the first panel as a vertical column"); then the two pairs as close as their panels, one vertical r bar right of the
   Echo map, and the notation <u_T|x> (the mean of the mean-centred u_T over the episodes with x; John 11-11: "maybe we
   should also change the symbol to be <u_T | x > so the given x does not need to be explained?"); then the centred state
   defined in the key, ubar_T = u_T - <u_T>, and used in the formulas (John: "maybe \\bar{u} = u - <u>"), "Shared" ->
   "Context-invariant", the key column narrowed to .95 in for wider PR panels. Then (John 11-10 ~16:20: "for g - j we dont need Trained and Echo, the colors
   denote that. Also please remove your a,b,c,... labels ... make panels a and f wider so the title can fit on one line
   (and we dont need to repeat it for both networks)"): no panel letters; one map title on one line above the Trained
   map (it sits above the other titles, so the map column is the title's width minus .84 in), the formula under it,
   "Trained" / "Echo" in colour above each map; g-j titled only. The maps (John asked what they show): for each probe
   and error step t (17-30), the step-t error e_t = dL_t/du_t is carried back k = 0..15 steps; delta_k = its gradient on
   the gate pre-activations at step t - k; the share ||delta_k||^2 / Sum_k' ||delta_k'||^2 is correlated (Pearson,
   across 2,000 probe episodes) with FR at the LANDING step t - k (forward KL, both variables), then averaged over t.
   The memory-use panel (r between use_t, the rise of the step-t update error when the state carried from t-1 is
   replaced by the start state, td_entanglement_memcut, and FR_t-1; memory_corr here) was REMOVED (John 11-09: "is
   panel e ... just restating that the network is learning a Joint bayes strategy ... Can we remove this panel if it's
   just another behavioral measure?"): yes, mostly. The cut leaves a memoryless update, and for Bayes the error of the
   memoryless (naive) update correlates with FR_t-1 at r = .34 (memory-free step-wise Bayes's: .12); the Trained RNN
   rises toward it (-.19 at 480, .25 at 10,400, .19 at the end) as its updates become Bayes-like, the ESN moves away
   (-.28): what the network's answers (b) already show.
   (until 11-10 behaviour(by="ent") / behaviour(scale=True) drew entanglement-coloured / self-scaled grids: John chose FR,
   unscaled; the grids are fig_learn_grids' now). Data: b, g, h 10,000 probe episodes (50 test environments x 100
   latent pairs x 2 trajectories) at 114 checkpoints per network, i, j 5,000 of them; c-f the 10,000 probes at 13
   checkpoints; k, l 2,000 of them (40 per test environment) at 32 checkpoints; a 50,000 fresh episodes (the 50
   environments x 100 cells x 10 trajectories) at 18 checkpoints.
   main/REPLICATION/figure_bin/supp_learning_mechanism.png   what changes inside the network (a-e)
Why these panels:
  The link from FR to LEARNING a compositional representation: fig_learn_grids (the belief grid unfolds where
    naive's grid folds, i.e. where FR is high, in the Trained RNN, and folds further in the ESN), behaviour c-f (the
    dimension rise is the context-specific grid code), a (the shared count code dips while the network passes naive and
    recovers; within environments it stays), k, l (before the crossing the credit of high-FR episodes lands 8-15 steps
    back, after it 2-4; the ESN's stays at lag 0), and mechanism a (memory is recruited at entangled observations and from ever
    older steps).
  Whether the ESN's failure is just accuracy: behaviour b (the ESN converges onto naive's answers, copying half of
    naive's errors).
  The mechanism in network terms: behaviour g-j (the state and its updates through training), mechanism b-e (the state's
    dimension; the learning signal locking onto entangled observations; td_xcorr_lags has every entanglement-locked
    measure).
  Left out: the chaos panels (td_tt_chaos_streams: the expansion is not short-term-stream dynamics), the count variance
    and the posterior decoding (td_tt_count_var, 11-09: tried, replaced by d), the update-measure distributions
    (carried by the lag maps), the restart heat map, the relative memory-use panels (duplicates) and memory use vs FR
    (11-09: implied by b, see above).
Sources (definitions, checks and caveats in their docstrings):
  behaviour a         td_tt_count_code's decoders (the ESN's extra checkpoints: td_tt_count_var.harvest_counts)
  behaviour b         fig_td_summary b (plot_training_dynamics)
  behaviour c-f       td_tt_griddim (u at step 30 split into context + shared grid + context-specific grid + trajectory)
  behaviour g-j       td_summary_data.state_spectrum_steps (g), fig_learn_maps.raw_data (h-j; td_analysis ck_*.npz)
  behaviour k, l      td_xcorr_across (credit_reach/full_<run>_*.npz against probes.npz's FR)
  mechanism a         td_tt_memlen c (random-cell memory cuts, 10,000 probes, all 114 checkpoints)
  mechanism b, c      fig_td_summary g (u's PCA spectrum per step and its participation ratio)
  mechanism d, e      fig_td_summary j, k (credit coherence's r with entanglement at every lag; its columns at stages)
Networks: the seed-20261029 Trained RNN (LSTM N = 1,000) and ESN (the same LSTM frozen at initialization), one world,
identical initial weights; x = training episodes (episodes with an update; the initial network at 10 = "0"); dotted:
the Trained RNN's accuracy first exceeds naive's (880 training episodes).
Run from the repository root (CPU, ~1 min):
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig_supp_learning.py"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from matplotlib.cm import ScalarMappable                                          # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm                        # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402
import fig_td_summary as fts                                                      # noqa: E402
import plot_training_dynamics as ptd                                              # noqa: E402
import fig_entanglement_driver as fd                                              # noqa: E402
import td_summary_data as sd                                                      # noqa: E402
import stage_bands as sb                                                          # noqa: E402

RUNS, COL, NAME = fts.RUNS, fts.COL, fts.NAME
TS, FL, FT = 13.5, 12, 11
PW, PH, GX, GY, LM = 3.3, 2.45, 1.0, 1.25, .95               # panel size and gaps, inches


def grid(nrow, ncol, top=1.0, bottom=1.5, right=.35, gx=GX):
    W = LM + ncol * PW + (ncol - 1) * gx + right
    H = top + nrow * PH + (nrow - 1) * GY + bottom
    fig = plt.figure(figsize=(W, H))
    ax = [[fig.add_axes([(LM + j * (PW + gx)) / W, (bottom + (nrow - 1 - i) * (PH + GY)) / H, PW / W, PH / H])
           for j in range(ncol)] for i in range(nrow)]
    return fig, np.array(ax, dtype=object), W, H


def head(ax, letter, name, sub=None):
    ax.set_title(f"{letter}   {name}" + (f"\n{sub}" if sub else ""), loc="left", fontsize=TS, linespacing=1.35)


def tag(ax, run, y=.04):
    ax.text(.98, y, NAME[run], transform=ax.transAxes, ha="right", va="bottom", fontsize=FL + 1, color=COL[run])


def corner_split(ax):
    """td_tt_wormholes panel a, its own code and data"""
    import td_tt_wormholes as tw
    F = {r: tw.fwd_stats(r) for r in tw.RUNS}
    Rf = tw.fwd_refs()
    W = tw.boot_weights()
    x = F["trained"]["x"]
    for r in tw.RUNS:
        for key, ls in (("joint_held", "solid"), ("commit", "dashed")):
            tw.band_line(ax, x, tw.agg(F[r][key], F[r]["n_two"]), tw.agg(F[r][key], F[r]["n_two"], W), COL[r],
                         tw.LS[ls])
    tw.refs_edge(ax, [(tw.ref(Rf, "bayes_joint_held", Rf["n_two"]), "k", tw.LS["solid"]),
                      (tw.ref(Rf, "naive_joint_held", Rf["n_two"]), "0.55", tw.LS["solid"]),
                      (tw.ref(Rf, "naive_commit", Rf["n_two"]), "0.55", tw.LS["dashed"]),
                      (tw.ref(Rf, "bayes_commit", Rf["n_two"]), "k", tw.LS["dashed"])])
    ax.set_ylim(-.02, 1.04)
    ax.set_ylabel("share of Bayes's corner splits", fontsize=FL)


def ladder(ax, S):
    """td_analysis's update ladder (plot_training_dynamics.series 'beta'): coefficients on N, W, M"""
    for r in RUNS:
        b = S[r]["beta"]
        for j, ls in ((0, "-"), (1, (0, (4, 2))), (2, (0, (1, 1.2)))):
            ax.plot(S[r]["x"], b[:, j], color=COL[r], lw=2 if j == 2 else 1.7, ls=ls)
    ax.axhline(1, color="0.6", lw=.8, ls=":")
    ax.text(fts.XMIN * 1.15, 1.01, "Bayes: 1, 1, 1", fontsize=FT, color="0.35", va="bottom")
    ax.axhline(0, color="0.8", lw=.7)
    ax.set_ylim(-.08, 1.12)
    ax.set_ylabel("regression coefficient", fontsize=FL)


GD_COL = {"cell": "#d62728", "envcell": "#9467bd"}            # griddim: the shared grid, the context-specific grid
GD_NAME = {"cell": "shared grid", "envcell": "context-specific grid"}
MESH_X = (100, 200, 300, 400, 560, 800, 1040, 2000, 5000)   # John 11-09: "[100, 200, 300, 400, 600, 800, 1K, 2K, 5K]";
                                                         # no checkpoint at 600 / 1,000: 560 and 1,040 are the nearest


def mesh_grids():
    """{run: {training episodes: (positions (50, 100, 2), modal answers (50, 100))}}: fig_learn_fr's grid cache plus
    td_grids_fwd's extra checkpoints (fwd_<run>_extra.npz: 100, 200, 300, 2,000, 5,000), reduced with fig_learn_fr's
    own code; the extra part cached in analysis/learn_supp_mesh_extra.npz"""
    import fig_learn_fr as flf
    Gz = flf.load("grids")
    cache = os.path.join(fd.AN, "learn_supp_mesh_extra.npz")
    if not os.path.exists(cache):
        E = np.load(os.path.join(flf.GRID_DIR, "fwd_episodes.npz"))
        key = E["env"] * flf.R * flf.R + E["cell"]
        order = np.lexsort((np.arange(len(key)), key))
        out = {}
        for r in RUNS:
            F = np.load(os.path.join(flf.GRID_DIR, f"fwd_{r}_extra.npz"))
            out[f"tr_ep_{r}"] = F["tr_ep"]
            out[f"pos_{r}"] = np.array([flf._per_pair(F["m"][k, :, -1], order).astype(np.float64).mean(2)
                                        for k in range(len(F["tr_ep"]))])
            out[f"modal_{r}"] = np.array([flf._modal(F["a"][k], order)[0] for k in range(len(F["tr_ep"]))], np.int16)
        np.savez(cache, **out)
    X = dict(np.load(cache))
    G = {}
    for r in RUNS:
        G[r] = {int(e): (Gz[f"pos_{r}"][k], Gz[f"modal_{r}"][k]) for k, e in enumerate(Gz[f"tr_ep_{r}"])}
        G[r].update({int(e): (X[f"pos_{r}"][k], X[f"modal_{r}"][k]) for k, e in enumerate(X[f"tr_ep_{r}"])})
    for ref in ("bayes", "naive"):
        G[ref] = (Gz[f"pos_{ref}"], Gz[f"modal_{ref}"])
    return G


def memory_series(run, by):
    """fig_entanglement_driver.series' memory use (cut to the last step - intact update error, per variable, steps
    2-30, fifths ranked within each step with both variables pooled) binned by each variable's own entanglement ('ent',
    the original panels) or by the FR its belief carried into the step, FR_t-1 = KL(B_t-1 || N_t-1) ('fr'); diamonds:
    memory-free Bayes's update error in the same fifths"""
    import td_tt_order as to
    I = fd.inputs()
    x = I["ent"] if by == "ent" else to.prev(to.refs()["FR"])
    B = fd.per_var_bins(x, fd.N_BIN)
    m = np.load(os.path.join(fd.AN, f"ent_memcut_var_{run}.npz"))
    b = [fd._mean_by(m["err_cut1"][k][:, fd.S].astype(np.float64) - m["err_int"][k][:, fd.S].astype(np.float64), B,
                     fd.N_BIN) for k in range(len(m["episode"]))]
    return {"x_ab": fd.train_x(m["episode"]), "b": np.array(b), "stepwise": fd._mean_by(I["stepwise"][:, fd.S], B,
                                                                                          fd.N_BIN)}


def memory_corr(run, by="fr", kind="pearson"):
    """John 11-09 ("merge d and e into correlations instead of gradients"; then "use pearson correlation for panel b
    (instead of rank)"): at every checkpoint, the correlation across probes x variables between memory use (cut to the
    last step - intact update error) and the FR the belief carried into the step, FR_t-1 ('fr'), or the step's own
    entanglement ('ent'), at each step 2-30 separately, then the mean over the steps (statistic per step, average
    after). kind: 'pearson' or 'rank'. Cached in analysis/learn_supp_memcorr_<run>_<by>[_pearson].npz"""
    import td_tt_order as to
    cache = os.path.join(fd.AN, f"learn_supp_memcorr_{run}_{by}" + ("_pearson" if kind == "pearson" else "") + ".npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["x"], z["r"]
    I = fd.inputs()
    v = I["ent"] if by == "ent" else to.prev(to.refs()["FR"])

    def std(a):                                                      # (n, 29, 2) -> (29, n x 2), standardized per step
        b = np.moveaxis(a, 1, 0).reshape(a.shape[1], -1).astype(np.float64)
        if kind == "rank":
            b = np.argsort(np.argsort(b, 1), 1).astype(np.float64)
        return (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)
    rv = std(v[:, fd.S])
    m = np.load(os.path.join(fd.AN, f"ent_memcut_var_{run}.npz"))
    r = []
    for k in range(len(m["episode"])):
        mu = std(m["err_cut1"][k][:, fd.S].astype(np.float64) - m["err_int"][k][:, fd.S].astype(np.float64))
        r.append((mu * rv).mean(1).mean())
    x, r = fd.train_x(m["episode"]), np.array(r)
    np.savez(cache, x=x, r=r)
    return x, r


def memory_panels(axs, Dd, cols):
    """fig_td_summary.e_memory with the variant's colours"""
    lo = min(min(Dd[r]["b"].min(), Dd[r]["stepwise"].min()) for r in RUNS)
    hi = max(max(Dd[r]["b"].max(), Dd[r]["stepwise"].max()) for r in RUNS)
    for ax, r in zip(axs, RUNS):
        ax.set_ylim(lo - .005, hi + .005)
        d = Dd[r]
        for k in range(5):
            ax.plot(d["x_ab"], d["b"][:, k], color=cols[k], lw=1.8)
            ax.plot(fts.XR2, d["stepwise"][k], "D", ms=4, color=cols[k], mec="k", mew=.8)
        ax.axvline(fts.XSEP, color="0.85", lw=.8)
        ax.axhline(0, color="0.6", lw=.8)
        ax.set_ylabel(r"cut $-$ intact update error (nats)", fontsize=FL)


def cell_entanglement():
    """per cell (r0, r1): the mean entanglement of its probe episodes' steps (both variables, both streams, steps
    1-30), per environment, then the mean over the 50 environments (probes.npz)"""
    P = np.load(os.path.join(fd.AN, "probes.npz"))
    E, env, r = P["ent"].astype(np.float64), P["env"], P["r"]
    cell = r[:, 0] * 10 + r[:, 1]
    ce = np.stack([np.bincount(cell[env == e], E[env == e].mean((1, 2)), 100) /
                   np.bincount(cell[env == e], minlength=100) for e in range(50)])
    return ce.mean(0)


def folds_both(ax, G):
    """fig_learn_fr's fold strip: both networks in one panel (where naive's grid folds solid, elsewhere dashed)"""
    for r in RUNS:
        ax.plot(G[f"x_{r}"], G[f"where_naive_{r}"], color=COL[r], lw=2, marker="o", ms=3)
        ax.plot(G[f"x_{r}"], G[f"elsewhere_{r}"], color=COL[r], lw=2, ls=(0, (4, 2)), marker="o", ms=3)
    ax.axhline(G["bayes_where_naive"], color="k", lw=1, ls=":")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of the grid folded at T", fontsize=FL)


def griddim(axs):
    """td_tt_griddim (step 30): f the dimension (PR) of the shared and the context-specific grid code; g what each adds
    to the total's PR (PR(total) - PR(total without it)); h the shared grid's share of all the variance of u_30 (no
    subspace; John 11-09: the top-10-PC measure was arbitrary, and it was dimension-confounded: with the context's
    subspace set by 90% of its variance the grid stays inside it, .96 at the end). Solid: Trained RNN, dashed: ESN"""
    import td_tt_griddim as tg
    Z = {r: np.load(os.path.join(tg.OUT, f"{r}.npz")) for r in RUNS}
    for r, ls in (("trained", "-"), ("echo", (0, (4, 2)))):
        z, x = Z[r], Z[r]["x"]
        for f in ("cell", "envcell"):
            axs[0].plot(x, z[f"t30_pr_{f}"], color=GD_COL[f], ls=ls, lw=2)
            axs[1].plot(x, z["t30_pr_total"] - z[f"t30_loo_{f}"], color=GD_COL[f], ls=ls, lw=2)
        if len(axs) > 2:
            axs[2].plot(x, z["t30_pureG90"], color=GD_COL["cell"], ls=ls, lw=2)
    axs[0].set_yscale("log")
    axs[0].set_ylim(1, 200)
    axs[0].set_ylabel("participation ratio", fontsize=FL)
    axs[1].axhline(0, color="0.6", lw=.8)
    axs[1].set_ylabel("PR added to the total", fontsize=FL)
    if len(axs) > 2:
        axs[2].set_ylim(0, 1.02)
        axs[2].set_ylabel("grid's share of the variance", fontsize=FL)
    for f, y in (("cell", .2), ("envcell", .9)):
        axs[0].text(.03, y, GD_NAME[f], transform=axs[0].transAxes, fontsize=FT + .5, color=GD_COL[f],
                    va="bottom" if y < .5 else "top")
    for a, loc, bb in zip(axs, ("center right", "lower left", "center right"), ((1, .6), (0, 0), (1, .45))):
        a.legend(handles=[Line2D([], [], color="0.3", lw=2, label="Trained RNN"),
                          Line2D([], [], color="0.3", lw=2, ls=(0, (4, 2)), label="ESN")],
                 loc=loc, bbox_to_anchor=bb, frameon=False, fontsize=FT, handlelength=2.2, borderaxespad=.3)


def mesh_rows(fig, W, H, top, pw=1.42, gap=.14, left=None, by="fr", scale=False):
    """fig_learn_fr a's belief grids at T in the mesh style of plot_fig1c_hybrid_unfold: each cell at its posterior
    means averaged over the 50 environments, neighbouring cells joined, every edge coloured by the geometric mean of
    its two cells' FR_T (log scale, plot_fig1c_mesh.FR_CMAP); dashed: the true lattice; number: distinct answers per
    environment (of 100)"""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm
    import plot_fig1c_mesh as pm
    import fig_learn_fr as flf
    Gz = flf.load("grids")
    R = 10
    idx = {(g, c): g * R + c for g in range(R) for c in range(R)}
    edges = np.array([(k, idx[(g + dg, c + dc)]) for (g, c), k in idx.items() for dg, dc in ((1, 0), (0, 1))
                      if (g + dg, c + dc) in idx])
    cells = np.array([(g, c) for g in range(R) for c in range(R)], float)
    if by == "fr":
        fr = Gz["pair_env_fr"].mean(0)
        norm, cmap, clab = LogNorm(.05, 30), pm.FR_CMAP, "FR$_T$ of the edge's cells (nats)"
    else:
        fr = cell_entanglement()
        norm, cmap = LogNorm(.1, .45), plt.matplotlib.colors.LinearSegmentedColormap.from_list(
            "ent", plt.cm.Purples(np.linspace(.25, 1., 256)))
        clab = "entanglement of the edge's cells"
    ecol = np.sqrt(fr[edges[:, 0]] * fr[edges[:, 1]])

    def answers(m):
        return np.mean([len(np.unique(m[e])) for e in range(len(m))])

    def one(x0, y0, pos, num, edge_col="0.75"):
        ax = fig.add_axes([x0 / W, y0 / H, pw / W, pw / H])
        ax.add_collection(LineCollection(cells[edges], colors="0.82", linewidths=.4, linestyles=(0, (3, 2))))
        P = pos.mean(0)
        if scale:                                          # each grid scaled to its own extent (John 11-09: "% max")
            c0 = P.mean(0)
            P = 4.5 + (P - c0) * 4.6 / max(np.abs(P - c0).max(), 1e-9)
        lc = LineCollection(P[edges], cmap=cmap, norm=norm, linewidths=1.25)
        lc.set_array(ecol)
        ax.add_collection(lc)
        ax.set_xlim(-.7, 9.7), ax.set_ylim(-.7, 9.7), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(edge_col)
            sp.set_linewidth(.8)
        ax.text(.5, -.09, num, transform=ax.transAxes, ha="center", va="top", fontsize=FL)
        return ax
    left = LM if left is None else left
    row_h = pw + .55
    MG = mesh_grids()
    for i, r in enumerate(RUNS):
        y0 = H - top - pw - i * row_h
        for j, xv in enumerate(MESH_X):
            pos, mod = MG[r][xv]
            ax = one(left + j * (pw + gap), y0, pos, f"{answers(mod):.1f}")
            if i == 0:
                ax.set_title(f"{xv:,.0f}", fontsize=FL, pad=3)
        ref = "bayes" if r == "trained" else "naive"
        one(left + len(MESH_X) * (pw + gap) + .15, y0, MG[ref][0],
            f"{'Bayes' if ref == 'bayes' else 'naive'} {answers(MG[ref][1]):.1f}")
        fig.text((left - .2) / W, (y0 + pw / 2) / H, NAME[r], rotation=90, ha="right", va="center", fontsize=TS,
                 color=COL[r])
    fig.text((left + (len(MESH_X) * (pw + gap) - gap) / 2) / W, (H - top + .33) / H, "training episodes",
             ha="center", va="bottom", fontsize=FL)
    cx = fig.add_axes([(left + (len(MESH_X) + 1) * (pw + gap) + .3) / W, (H - top - pw - row_h) / H, .12 / W,
                       (pw + row_h) / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cx, extend="both")
    cb.set_label(clab, fontsize=FL)
    cb.ax.tick_params(labelsize=FT)
    cb.outline.set_visible(False)


PRINT = dict(tick=7, key=7, label=8, title=8, letter=9)                         # points at print size (5.5 in)
SHORT = {"trained": "Trained", "echo": "Echo"}                                    # one naming throughout (John 11-10)


def grid_block(fig, W, H, top, fs, letter):
    """fig_learn_grids' belief grids as two extra rows (John 11-10: "let's actually try to put the grids back into
    learning_behavior as extra rows"): Trained RNN / ESN at 100-5,000 training episodes (200 and 400 left out, John
    11-10), Joint Bayes / Naive Bayes at the right, FR colour bar (plot_fig1c_mesh.FR_CMAP, log, the cells' range).
    Returns the block's bottom (inches from the top)"""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm
    import plot_fig1c_mesh as pm
    import fig_learn_grids as flg
    MG, stages, fr = flg.data()
    stages = tuple(x for x in stages if x not in (200, 400))
    lm, gp, gref, cbw = .30, .04, .14, .55
    pw = (W - lm - .04 - cbw - (len(stages) - 1) * gp - gref) / (len(stages) + 1)
    norm = LogNorm(np.floor(fr.min() * 10) / 10, np.ceil(fr.max()))
    R_ = 10
    idx = {(g, c): g * R_ + c for g in range(R_) for c in range(R_)}
    edges = np.array([(k, idx[(g + dg, c + dc)]) for (g, c), k in idx.items() for dg, dc in ((1, 0), (0, 1))
                      if (g + dg, c + dc) in idx])
    cells = np.array([(g, c) for g in range(R_) for c in range(R_)], float)

    def one(x0, tp, pos, title=None, tcol="k"):
        ax = fig.add_axes([x0 / W, (H - tp - pw) / H, pw / W, pw / H])
        ax.add_collection(LineCollection(cells[edges], colors="0.86", linewidths=.2, linestyles=(0, (3, 2))))
        P = pos.mean(0)
        ax.add_collection(LineCollection(P[edges], colors="0.7", linewidths=.25, zorder=2))
        ax.scatter(P[:, 0], P[:, 1], c=fr, cmap=pm.FR_CMAP, norm=norm, s=9, lw=.25, edgecolors="0.3", zorder=3)
        ax.set_xlim(-.7, 9.7), ax.set_ylim(-.7, 9.7), ax.set_aspect("equal"), ax.set_xticks([]), ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_color("0.7")
            sp_.set_linewidth(.4)
        if title:
            ax.text(.5, 1.03, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=fs["key"], color=tcol)
    letter(.02, top, "k")
    rows = (("trained", "Trained", "bayes", "Joint Bayes", "k"), ("echo", "Echo", "naive", "Naive Bayes", "0.45"))
    xs = [lm + j * (pw + gp) for j in range(len(stages))]
    x_ref = xs[-1] + pw + gref
    t0 = top + .12                                                     # no block title (John 11-10)
    for i, (run, nm, ref, ref_nm, rc) in enumerate(rows):
        tp = t0 + i * (pw + .17)
        for j, xv in enumerate(stages):
            one(xs[j], tp, MG[run][xv][0])
        one(x_ref, tp, MG[ref][0], ref_nm, rc)
        fig.text((lm - .06) / W, (H - tp - pw / 2) / H, nm, rotation=90, ha="right", va="center",
                 fontsize=fs["label"], color=COL[run])
    y_num = t0 + 2 * pw + .17 + .03
    for j, xv in enumerate(stages):
        fig.text((xs[j] + pw / 2) / W, (H - y_num) / H, f"{xv:,}", ha="center", va="top", fontsize=fs["tick"])
    fig.text((xs[0] + (xs[-1] + pw - xs[0]) / 2) / W, (H - y_num - .14) / H, "training episodes", ha="center",
             va="top", fontsize=fs["label"])
    cx = fig.add_axes([(x_ref + pw + .1) / W, (H - t0 - 2 * pw - .17) / H, .05 / W, (2 * pw + .17) / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=pm.FR_CMAP, norm=norm), cax=cx, extend="both", extendfrac=.04)
    cb.ax.tick_params(labelsize=fs["tick"], length=1.8, width=.5, pad=1)
    cb.ax.tick_params(which="minor", length=1, width=.4)
    cb.outline.set_visible(False)
    cb.set_label("factorization regret", fontsize=fs["label"], labelpad=2)
    return y_num + .14 + .16


def wrap(text, width_in, fontsize):
    """text broken into lines no wider than width_in inches at fontsize (measured with the current rcParams)"""
    f_ = plt.figure(figsize=(4, 1))
    rend = f_.canvas.get_renderer()
    lines, cur = [], ""
    for w_ in text.split(" "):
        cand = (cur + " " + w_).strip()
        t_ = f_.text(0, 0, cand, fontsize=fontsize)
        wide = t_.get_window_extent(rend).width / f_.dpi > width_in
        t_.remove()
        if wide and cur:
            lines.append(cur)
            cur = w_
        else:
            cur = cand
    plt.close(f_)
    return "\n".join(lines + [cur])


def behaviour():
    """Drawn at print size for the paper (5.5 in, ticks and keys 7 pt, labels and titles 8 pt; John 11-10: "there is a
    bit too much empty space, can you please polish this figure to get it ready for the paper. readable text, etc.";
    and for the grid codes "maybe we should we arrange it into [trained context grid (PR(u_t), PR(grid), PR(u_t -
    grid)), ESN context grid (...), trained shared grid (...), ESN shared grid (...)]"). Row 1: a credit vs FR (Trained
    RNN), b naive errors, c count decoding, d weight update energy, e weight update dimension; row 2: f credit vs FR
    (ESN), g-j one panel per network and grid code with PR(u_30), PR(g) and PR(u_30 - g) (one y range); k the belief
    grids. One x label per row; no notes (data counts in the module docstring)."""
    import td_tt_griddim as tg
    import td_tt_count_var as cv
    import td_xcorr_across as xa
    import td_xcorr_lags as xl
    import fig_learn_maps as flm
    from matplotlib.colors import Normalize
    fs = PRINT
    apply_fig_style()
    plt.rcParams.update({"mathtext.fontset": "cm", "axes.linewidth": .6, "xtick.major.width": .6,
                         "ytick.major.width": .6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    P = ptd.probes()
    S = {r: ptd.series(r, P) for r in RUNS}
    cross = fd.passes_naive(P)
    W = 9.0                                                            # John 11-11: narrower, so the key line holds;
                                                                       # then "widen the figure a bit and give all the
                                                                       # extra space to the nonheatmap pannels"
    X0, PH, GX, GW = .46, .86, .44, .50                                # less empty space; then "a bit more space
                                                                       # between the columns" (gaps .34 -> .44)
    MW = 2.30                                                          # the map column (its title wraps to fit it)
    # John 11-10: "error propogated k steps back or something of the sort", then "would it be fair to replace error
    # with gradient?" (yes: delta_{t,k} is the gradient of the step-t loss on the gates at t - k); g is free since g-j
    # say "grid"; 11-11: "expand the title into a multi-line sentence 'Correlation between...' so the title isnt wider
    # than the actual column"
    MAPT = "Correlation between the gradient reaching $k$ steps back and FR $k$ steps back"
    MAPF = (r"$\langle\, r(g_{t,k},\, \mathrm{FR}_{t-k}) \,\rangle_t$,   "
            r"$g_{t,k} = \Vert\delta_{t,k}\Vert^2 / \Sigma_{k'}\Vert\delta_{t,k'}\Vert^2$")
    PW = (W - X0 - .20 - MW - GW - 3 * GX) / 4                        # .20 in right margin: the last title fits
    CX = [X0, X0 + MW + GW] + [X0 + MW + GW + j * (PW + GX) for j in range(1, 4)]
    # the bottom row (John 11-11: "push the bottom row (nonheatmap) panels closer together since they share a ylabel and
    # then put the legend as the first panel as a vertical column"): the key column, then two pairs of panels (shared,
    # context-dependent; Trained and Echo in each) .10 in apart within a pair, .30 between the pairs, one title per pair
    LX, LW, LG, PG, PPG = X0 + MW + GW, .95, .42, .10, .10             # key column, its width, gap to the first panel;
                                                                       # the pairs as close as their panels (John 11-11:
                                                                       # "push the shared and context-dependent panels
                                                                       # together as well")
    PWB = (W - .20 - LX - LW - LG - 2 * PG - PPG) / 4
    CXB = [X0] + [LX + LW + LG + j * (PWB + PG) + (PPG - PG if j >= 2 else 0) for j in range(4)]
    HEAD, FOOT = .40, .40                                              # title block above, ticks + x label below
    tops = [HEAD, HEAD + PH + FOOT + .44]                              # both rows labelled (John 11-11: "put the
                                                                       # training episodes xlabel in the top row")
                                                                       # 11-10); row 2 heads hold title + u_grid
    MGAP = .10                                                         # between the two maps (John 11-11: the top
                                                                       # map lower, very little space between them)
    H = tops[1] + PH + FOOT + .06                                      # (no key line: the bar is beside the Echo map)
    fig = plt.figure(figsize=(W, H))

    def add(x, top, w, h):
        return fig.add_axes([x / W, (H - top - h) / H, w / W, h / H])

    def title2(ax, name, formula, color="k", fcolor="0.15"):           # name (8 pt) over formula (7 pt)
        ax.annotate(formula, xy=(.5, 1), xycoords="axes fraction", xytext=(0, 3.5), textcoords="offset points",
                    fontsize=fs["key"], ha="center", va="baseline", color=fcolor)
        ax.annotate(name, xy=(.5, 1), xycoords="axes fraction", xytext=(0, 12.5), textcoords="offset points",
                    fontsize=fs["title"], ha="center", va="bottom", color=color, linespacing=1.0)

    def letter(x, top, s_):
        fig.text(x / W, (H - top) / H, r"$\mathbf{%s}$" % s_, fontsize=fs["letter"], ha="left", va="baseline")

    def style(ax):
        ax.tick_params(labelsize=fs["tick"], pad=1.5)
        ax.spines[["top", "right"]].set_visible(False)

    def xax(ax, labels=True):
        fts.xaxis(ax, label=False)
        ax.set_xlim(fts.XMIN, 4.6e4)
        ax.set_xticks([fts.X0, 1e2, 1e3, 1e4])
        ax.set_xticklabels(["0", "100", "1k", "10k"] if labels else [])
        ax.axvline(cross, color="0.6", lw=.6, ls=(0, (1, 1.5)), zorder=0)
        if labels:                                                     # the bottom row only: every column shares its
            ax.set_xlabel("training episodes", fontsize=fs["label"], labelpad=1.5)   # x axis (John 11-10)
        style(ax)

    MAPW = wrap(MAPT, MW + .05, fs["title"])                          # the maps fill their column: the top one
    MT0 = tops[0] - .29 + (MAPW.count("\n") + 1) * .128 + .22           # starts under the title and formula, the
    MH = (tops[1] + PH - MT0 - MGAP) / 2                               # bottom one ends with row 2
    A = [[add(CX[0], MT0, MW, MH) if (i, j) == (0, 0) else add(CX[0], tops[1] + PH - MH, MW, MH) if j == 0
          else add(CX[j], tops[0], PW, PH) if i == 0 else add(CXB[j], tops[1], PWB, PH) for j in range(5)]
         for i in range(2)]                                            # no panel letters
                                                                                                # (John adds them)
    # b: copying naive's errors
    a = A[0][1]
    for r in RUNS:
        a.plot(S[r]["x"], S[r]["agree"][:, 2], color=COL[r], lw=1.0)
    a.set_ylim(0, .55)
    a.set_yticks([0, .25, .5])
    a.set_yticklabels(["0", ".25", ".5"])
    title2(a, "Naive errors", "naive wrong, Bayes right")
    a.set_ylabel("fraction as naive", fontsize=fs["label"], labelpad=1.5)
    a.text(.05, .98, "Trained", color=COL["trained"], fontsize=fs["key"], ha="left", va="top", transform=a.transAxes)
    a.text(.05, .84, "Echo", color=COL["echo"], fontsize=fs["key"], ha="left", va="top", transform=a.transAxes)
    # c: the shared count code: ridge R^2 of the counts n_30 from u_30, one decoder for every environment
    a = A[0][2]
    for r in RUNS:
        c_ = cv.load_counts(r)
        a.plot(c_["x"], c_["across"], color=COL[r], lw=1.0)
    a.set_ylim(.4, 1.)
    a.set_yticks([.4, .7, 1])
    a.set_yticklabels([".4", ".7", "1"])
    title2(a, "Count decoding", r"$R^2(u_T\to\Sigma_t\,o_t)$")              # u_T throughout (John 11-10)
    a.set_ylabel(r"decoded $R^2$", fontsize=fs["label"], labelpad=1.5)
    # d, e: the weight update's energy and dimension (fig_learn_maps' raw data, the mean over the error steps 17-30)
    RD = {r: flm.raw_data(r) for r in RUNS}
    for a, (k, nm, form) in zip(A[0][3:], (("cr_E", "Update energy", r"$\Vert\Delta W\Vert^2$"),
                                           ("cr_dim", "Update dimension", r"PR$(\Delta W^{\top}\Delta W)$"))):
        for r in RUNS:
            a.plot(RD[r]["x"], np.nanmean(RD[r][k], 1), color=COL[r], lw=1.0)
        if k == "cr_E":
            a.set_yscale("log")
            a.set_yticks([1, 1e2, 1e4])
            a.minorticks_off()
            a.set_ylabel(r"$\Vert\Delta W\Vert^2$", fontsize=fs["label"], labelpad=1.5)
        else:
            a.set_yticks([1, 1.4, 1.8])
            a.set_yticklabels(["1", "1.4", "1.8"])
            a.set_ylabel("participation ratio", fontsize=fs["label"], labelpad=1.5)
        title2(a, nm, form)
    # g-j: one panel per network and grid code: PR(u_30), PR(g), PR(u_30 - g) (td_tt_griddim, step 30)
    Z = {r: np.load(os.path.join(tg.OUT, f"{r}.npz")) for r in RUNS}
    LS = {"tot": "-", "g": (0, (3.5, 1.8)), "rest": (0, (1, 1.3))}
    quart = [("cell", "trained"), ("cell", "echo"), ("envcell", "trained"), ("envcell", "echo")]   # shared first
    sub = {"cell": r"{\rm sh}", "envcell": r"{\rm ctx}"}
    for a, (f_, r) in zip(A[1][1:], quart):
        z = Z[r]
        a.plot(z["x"], z["t30_pr_total"], color=COL[r], lw=1.0, ls=LS["tot"])
        a.plot(z["x"], z[f"t30_pr_{f_}"], color=COL[r], lw=1.0, ls=LS["g"])
        a.plot(z["x"], z[f"t30_loo_{f_}"], color=COL[r], lw=1.1, ls=LS["rest"])
        if r == "trained":                                             # one title per pair, centred over both
            for txt, dy, fsz, col in ((("Context-dependent" if f_ == "envcell" else "Context-invariant") + " PR", 12.5,
                                       fs["title"], "k"),
                                      (r"$u_{\rm grid} = \langle \bar u_T|\mathbf{r},Z\rangle - \langle \bar u_T|\mathbf{r}\rangle"
                                       r" - \langle \bar u_T|Z\rangle$" if f_ == "envcell"   # the centred state: the
                                       else r"$u_{\rm grid} = \langle \bar u_T|\mathbf{r}\rangle$", 3.5, fs["key"],   # grand mean drops;
                                       "0.15")):                       # <u_T|x>: John 11-11, so that "given x" needs
                                                                       # no explaining
                a.annotate(txt, xy=(1 + PG / 2 / PWB, 1), xycoords="axes fraction", xytext=(0, dy),
                           textcoords="offset points", fontsize=fsz, ha="center",
                           va="bottom" if dy > 10 else "baseline", color=col)
    top_ = max(ln.get_ydata().max() for a in A[1][1:] for ln in a.get_lines())
    for j, a in enumerate(A[1][1:]):
        a.set_ylim(0, 1.06 * top_)                                    # one tight y range for g-j (John 11-10)
        a.set_yticks([0, 50, 100])
        if j:
            a.set_yticklabels([])
    A[1][1].set_ylabel("participation ratio", fontsize=fs["label"], labelpad=1)
    fig.legend(handles=[Line2D([], [], color="0.3", lw=1.0, ls=LS[k]) for k in ("tot", "g", "rest")],
               labels=[r"PR$(u_T)$", r"PR$(u_{\rm grid})$", r"PR$(u_T - u_{\rm grid})$"], loc="upper left", ncol=1,
               frameon=False, bbox_to_anchor=(LX / W, (H - tops[1] - .05) / H), fontsize=fs["key"],
               handlelength=1.6, handletextpad=.4, labelspacing=.7, borderaxespad=0)   # the key column, row 2
    fig.text(LX / W, (H - tops[1] + 12.5 / 72) / H, "Grid decomposition", fontsize=fs["title"], ha="left",
             va="bottom")                                              # the key's title, level with the pair titles
                                                                       # (John 11-11: "give a title to the grid legend")
    fig.text(LX / W, (H - tops[1] - .64) / H, r"$\bar u_T = u_T - \langle u_T\rangle$", fontsize=fs["key"], ha="left",
             va="top")                                                 # John 11-11: "maybe \bar{u} = u - <u>"
                                                                                                    # (John 11-10)
    for i_, row_ in enumerate(A):
        for a in row_[1:]:
            xax(a, True)                                               # every panel b-j labelled (the maps share theirs)
    for j in (1, 3):                                                   # the bottom pairs: one x label, centred under
        A[1][j].xaxis.label.set_x(1 + PG / 2 / PWB)                    # both panels (like their title)
        A[1][j + 1].set_xlabel("")
    # a / f: credit share by lag vs FR at the landing step, across episodes (td_xcorr_across), Trained RNN / ESN
    CN = Normalize(-.25, .25)
    for i, r in enumerate(RUNS):
        d_ = xa.data(r)
        a = A[i][0]
        a.pcolormesh(xl.log_edges(d_["credit_x_fr"]), np.arange(d_["credit_fr"].shape[1] + 1) - .5, d_["credit_fr"].T,
                     cmap="viridis", norm=CN, shading="flat", rasterized=True)
        a.set_yticks([0, 5, 10, 15])
        a.set_ylabel(r"$k$ (steps back)", fontsize=fs["label"], labelpad=1)
        if i == 0:                                                       # one title for both maps: wrapped to the
            fig.text((CX[0] + MW / 2) / W, 1 - (tops[0] - .29) / H,      # column, starting at the other titles'
                     MAPW, fontsize=fs["title"], ha="center", va="top",                  # height
                     linespacing=1.15)
            a.annotate(MAPF, xy=(.5, 1), xycoords="axes fraction", xytext=(0, 3.5), textcoords="offset points",
                       fontsize=fs["key"], ha="center", va="baseline", color="0.15")
        a.text(.025, .955, SHORT[r], transform=a.transAxes, ha="left", va="top", fontsize=fs["title"], color=COL[r],
               zorder=6, bbox=dict(fc="white", ec="none", alpha=.85, pad=1.2))   # the network, inside the map
        xax(a, i == 1)
        a.axvline(cross, color="w", lw=.6, ls=(0, (1, 1.5)), zorder=5)
    # one vertical bar for both maps, right of the Echo map (John 11-11: "move the cbar to the right of the ESN heatmap,
    # vertically")
    em = A[1][0].get_position()                                        # the Echo map, figure fraction
    cxh = fig.add_axes([em.x0 + em.width + .07 / W, em.y0 + .05 / H, .05 / W, em.height - .22 / H])
    cbh = fig.colorbar(plt.cm.ScalarMappable(norm=CN, cmap="viridis"), cax=cxh, ticks=[-.2, -.1, 0, .1, .2],
                       extend="both", extendfrac=.03)
    cbh.ax.set_yticklabels(["$-$.2", "$-$.1", "0", ".1", ".2"])
    cbh.ax.tick_params(labelsize=fs["tick"], length=1.8, pad=1.2, width=.5)
    cbh.outline.set_visible(False)
    cbh.ax.set_title("$r$", fontsize=fs["label"], pad=4)
    fig.canvas.draw()                                                  # every text inside the figure
    rr, fb = fig.canvas.get_renderer(), fig.bbox
    for t in fig.texts + [t for ax in fig.axes for t in ax.texts]:
        e = t.get_window_extent(rr)
        if t.get_text() and (e.x0 < fb.x0 - .5 or e.x1 > fb.x1 + .5 or e.y0 < fb.y0 - .5 or e.y1 > fb.y1 + .5):
            print(f"  WARNING: text {t.get_text()!r} leaves the figure")
    path = fig_path("supp_learning_behaviour.png")
    fig.savefig(path, dpi=400)
    with plt.rc_context({"pdf.fonttype": 42}):                         # and a PDF (John 11-11: "as pdf please";
        fig.savefig(path[:-4] + ".pdf", dpi=400)                       # TrueType fonts, maps rasterized at 400 dpi)
    fig.savefig(path[:-4] + ".svg", dpi=400)                           # John 11-11: "make SVGs for the learning
                                                                       # behavior and learn grids figures"
    plt.close(fig)
    print("saved", path)
    return path


BAND_COL = ("#6baed6", "#3182bd", "#1f5f9e", "#08306b")          # td_tt_memlen c: light = recent, dark = old


def memory_worth(ax, strip):
    """td_tt_memlen panel c (its own data and drawing), with Bayes's per-band values [5-95%] in the strip beside it"""
    import td_tt_memlen as tm
    bd = tm.bands("trained", "random")
    x = bd["x"]
    for j, (k0, k1) in enumerate(tm.BANDS):
        lo, hi = tm._iv(bd["net_b"][:, j])
        ax.fill_between(x, lo, hi, color=BAND_COL[j], alpha=.25, lw=0, zorder=3)
        ax.plot(x, bd["net"][:, j], color=BAND_COL[j], lw=1.8, zorder=4,
                label="the 2nd step back" if k0 == k1 else f"steps {k0}-{k1} back")
    for j in (0, len(tm.BANDS) - 1):            # the observer's 2nd step and steps 10-15 (its 3-6, 7-9 run with its 2nd)
        ax.plot(x, bd["mat"][:, j], color=BAND_COL[j], lw=1.4, ls=(0, (1., 1.2)), zorder=5)
    blo, bhi = tm._iv(bd["bayes_b"])
    for j in range(len(tm.BANDS)):
        strip.plot([j + .5] * 2, [blo[j], bhi[j]], color=BAND_COL[j], lw=1.3)
        strip.plot(j + .5, bd["bayes"][j], "o", ms=4.5, mew=0, color=BAND_COL[j])
    ax.set_ylim(-.004, .15)
    strip.set_ylim(ax.get_ylim())
    strip.set_xlim(-.1, len(tm.BANDS) + .1)
    strip.set_yticks([])
    strip.set_xticks([len(tm.BANDS) / 2])
    strip.set_xticklabels(["Bayes"], rotation=90, fontsize=FT - .5)
    ax.set_ylabel("accuracy gained per step kept", fontsize=FL)
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color="0.4", lw=1.4, ls=(0, (1., 1.2))))
    l.append("perfect-memory observer\nat the same accuracy (dotted)")
    ax.legend(h, l, fontsize=FT - 1, frameon=False, loc="upper left", handlelength=1.6, labelspacing=.3,
              borderaxespad=.2)


def mechanism():
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({"xtick.labelsize": 10.5, "ytick.labelsize": 10.5})
    P = ptd.probes()
    S = {r: ptd.series(r, P) for r in RUNS}
    cross = fd.passes_naive(P)
    fig, ax, W, H = grid(1, 5, top=1.05, bottom=2.0, gx=1.2)
    ax = ax[0]
    p0 = ax[0].get_position()
    strip = fig.add_axes([p0.x1 + .06 / W, p0.y0, .36 / W, p0.height])
    memory_worth(ax[0], strip)
    head(ax[0], "a", "What each step of own history is worth",
         r"$L(k-1)-L(k)$, random-cell cut; mean per band")
    tag(ax[0], "trained", .86)
    fts.g_spectrum([ax[1], ax[2]])
    head(ax[1], "b", "Network state spectrum and PR",
         r"$\lambda_k/\Sigma\lambda$ of $u_t$ per step; white: PR")
    head(ax[2], "c", "Network state spectrum and PR", "as b")
    for a, r in ((ax[1], "trained"), (ax[2], "echo")):
        a.text(.96, .8, NAME[r], transform=a.transAxes, ha="right", va="top", fontsize=FL + 1, color="w")
    fts.j_lags(ax[3], S)
    head(ax[3], "d", "Credit coherence vs entanglement",
         r"$r$ at every lag $\tau$ (Trained RNN)")
    fts.k_profiles(ax[4], S)
    head(ax[4], "e", "d's columns at training stages", "(training episodes; ESN at the end)")
    for a in ax[:4]:
        fts.xaxis(a, label=True)
        a.set_xlim(fts.XMIN, 4.2e4)                                   # the maps end at the last checkpoint
        a.axvline(cross, color="w" if a in (ax[1], ax[2], ax[3]) else "0.55", lw=1, ls=":", zorder=5)
        (sb.shade if a is ax[0] else sb.strip)(a)
    for a, norm, lab in ((ax[1], fts.NORM["spec"], "b, c: fraction of the variance per PC"),
                         (ax[3], fts.NORM["lag"], r"d: $r$ with entanglement")):
        p = a.get_position()
        x0 = p.x0 if a is ax[3] else p.x0 + .1 * (ax[2].get_position().x1 - p.x0)
        x1 = p.x1 if a is ax[3] else ax[2].get_position().x1 - .1 * (ax[2].get_position().x1 - p.x0)
        cax = fig.add_axes([x0, .98 / H, x1 - x0, .1 / H])
        cb = plt.colorbar(ScalarMappable(norm=norm, cmap="viridis"), cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=FT)
        cb.ax.tick_params(labelsize=FT - 1)
    fig.text(.5, .3 / H, "10,000 probe episodes (50 test environments x 100 latent pairs x 2 trajectories) at 114 "
             "checkpoints per network (d, e: the update measures on 5,000 of them, error steps 17-30). a: the state at t - k "
             "replaced by another cell's (same environment), the cell rerun over the last k inputs; L(k) = accuracy at T "
             "lost; bands 5-95% over 1,000\nenvironment resamples. Dotted vertical: the Trained RNN passes naive "
             f"({cross:,.0f} training episodes). Every entanglement-locked measure at every lag: the companion lag-map figure.",
             ha="center", va="center", fontsize=FT - .5, color="0.3", linespacing=1.5)
    path = fig_path("supp_learning_mechanism.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("saved", path)
    return path


if __name__ == "__main__":
    behaviour()
    mechanism()
