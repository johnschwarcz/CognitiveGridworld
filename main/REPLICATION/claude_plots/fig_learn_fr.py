"""Figure 1 of the learning story (how does FR relate to LEARNING a compositional representation?).
Factorization regret (FR) is naive's excess log loss. Training a recurrence removes it in every FR fifth, and its belief
grid on held-out environments unfolds into Bayes's lattice where naive's grid folds (= where FR is high). The competence
on high-FR trajectories is learned from high-FR steps (d). The ESN (the same LSTM, recurrence frozen at initialization,
read-in and read-out trained) approaches naive's accuracy from below without becoming naive: its excess loss stays above
naive's in every FR fifth. Figure 2: fig_learn_mechanism.py.
-> main/REPLICATION/figure_bin/fig_learn_fr.png (5.5 in wide = the paper's text width, 300 dpi; type 10-11 pt at that size)
Caches (CPU; `--rebuild` recomputes all, `--rebuild=probes,blocks` the named ones) in main/DATA/experimental/
training_dynamics/analysis/:
  learn_main_fig1_grids.npz   per environment x cell at T: positions, within-cell variance, modal answers, FR
  learn_main_fig1_probes.npz  per checkpoint x environment (x FR fifth): accuracy, copying, KL(B_T || q_T), the training
                              objective by the goal's fifth
  learn_main_fig1_blocks.npz  d: per from-scratch block run x checkpoint x environment x FR fifth: right answers at T
  learn_main_fig1_lesion.npz  supplement: per lesion checkpoint x environment x FR third, accuracy-loss sums

NETWORKS AND AXES
  Seed-20261029 LSTM, N = 1,000, one training run per network: the Trained RNN; the ESN. Training episodes = raw
  episodes - raw // 5 (every 5th episode is a test episode and trains nothing); the initial network is drawn at 10 ("0").
  Dotted vertical line in every training-axis panel: the first probe checkpoint at which the Trained RNN's goal accuracy
  at T passes naive's (880; the stays-past crossing, log-interpolated, is 835 [798-876]). Bands: 5-95% over td_tt_gap's
  1,000 resamples of the 50 test environments (ratios of environment sums; d: the same resamples, paired across runs).

DEFINITIONS
  B_t, N_t, q_t: Bayes's exact marginal posterior of one variable after t observations, naive's (the product of each
  variable's own marginal likelihood tables, the other variable marginalized uniformly), the network's read-out.
  FR_T = KL(B_T || N_T), the FORWARD factorization regret at T of that variable (not probes 'fr', the clipped symmetric
  version: mean 3.44 against 2.97). Cell = one value pair (r0, r1) of an environment's two active variables.
  a  Belief grid at T. Grid episodes (summary_data/grids, td_grids_fwd.py): 50 held-out test environments x 100 cells x
     24 fresh trajectories of T = 30 observations, replayed at 19 checkpoints. A cell sits at its posterior means
     (E[r0 | o_1:T], E[r1 | o_1:T]), averaged over its 24 trajectories, then over the 50 environments; neighbouring cells
     are joined into the 10 x 10 lattice; every mini-grid spans -0.5..9.5 on both axes. Colour: the cell's FR_T averaged
     over both variables and its 24 x 50 trajectories (viridis, 0 to the 95th percentile of the 100 cell values; the only
     viridis code in the figure). Number under a grid: distinct answers per environment = the number of distinct modal
     joint answers (argmax of each variable's marginal, modal over the cell's 24 trajectories; ties to the lowest joint
     index) among the environment's 100 cells, mean over the 50 environments.
     Strip: each environment's OWN grid (cell means over 24 trajectories) is cut into 162 triangles of neighbouring
     cells; a triangle is folded when its signed area is negative (td_summary_data.folded). % folded at T among the
     triangles where naive's grid of the same environment folds (solid) and among the rest (dashed).
  b  Goal accuracy at T, P(g_hat_T = g): 10,000 probe episodes (50 test environments x 100 cells x 2 trajectories) at
     114 checkpoints. References: Bayes, memory-free Bayes (the step-wise rule: each step's marginal likelihood with the
     other variable marginalized within the step, summed over steps), naive.
  c  KL(B_T || q_T) = the learner's expected excess log loss over Bayes at T for one variable (naive's is FR_T by
     definition: FR is naive's excess log loss). The 20,000 probe-variables are pooled and cut into fifths of their own FR_T (each
     variable paired with itself); KL is computed per probe-variable, then averaged within a fifth. Dotted lines in each
     fifth's colour, labelled at the right: naive's value (= the fifth's mean FR_T). Purple, light -> dark = low -> high
     FR. This is not the training objective (SANITY_loss, power .5, all steps, goal only); its shares are in RESULTS.
  d  Learning blocked at FR-ranked steps (train_learn_masks.py det_{hi,lo}_fr / det_rand / ctrl, _f50_s0 runs, replayed
     by td_tt_learnmask_fr.py; per-checkpoint files through_training/learnmask_fr/<run>_<raw>.npz): four runs from the
     same initialization and episode stream, trained from episode 0 with no gradient through the goal read-out increment
     at 50% of each step's (episode, step) terms, chosen within each step of every training batch by the goal belief's
     per-step FR_t = KL(Bayes_t || naive_t): the highest-FR half (red), the lowest-FR half (green), a random half (grey),
     or nothing (blue, the unblocked Trained RNN of this stream). y: accuracy at T in the top / bottom fifth of FR_T,
     both variables (each variable's own answer against its own truth, in the fifth of its own FR_T; the fifths of c).
     The read-out ignores which variable is the goal (main run, end: goal .760, context .763), so both variables are
     pooled; goal-only numbers are in RESULTS. 31 checkpoints per run (raw 0, 100, ..., 3,000 = 10 (initial), 80, ...,
     2,400 training episodes).

RESULTS (11-09; every number printed by `--numbers` from the caches)
  a  distinct answers per environment (of 100) at 60 / 240 / 880 / 2,400 / 39,996 training episodes:
       Trained RNN 6.4 / 22.3 / 53.2 / 79.4 / 91.7 (Bayes 94.4); ESN 2.1 / 8.8 / 18.4 / 28.7 / 42.1 (naive 44.3);
       ties to the highest joint index instead: within .6 everywhere.
     % folded where naive's grid folds (naive folds 24.0% of its triangles): Trained RNN 53.0 (0) -> 59.4 (240, max)
       -> 23.4 [19.7-27.5] (880) -> 5.8 (2,400) -> 2.5 [1.5-3.6] (end); Bayes 2.2. ESN 53.0 -> 63.5 (880) -> 76.2
       (8,000, max) -> 72.0 [67.8-76.0] (end).
     % folded elsewhere: both start at 50.2; Trained RNN 6.2 at 880 (against 23.4 where naive folds) -> .4 at the end;
       ESN 31.7 at 880 -> 6.6 at the end; Bayes .3.
     Where naive folds = where FR is high: mean triangle FR_T (its three cells, per environment) 4.93 nats where naive
       folds vs 1.41 elsewhere (AUC .844; 11% of naive-fold triangles lie in the lower half of triangle FR). 67% of
       naive's folds touch the border (40% of all triangles), and the split holds within border and interior: Trained
       RNN at 880 border 27.9% vs 10.3%, interior 14.5% vs 4.4%; ESN at the end border 77.4% vs 11.7%, interior 61.1%
       vs 4.4%.
     Means check: the exact Bayes / naive posterior means recomputed here equal the stored ones to 4.8e-7.
  b  goal accuracy at T: Bayes .771, memory-free Bayes .608, naive .464. The Trained RNN passes naive at 835 [798-876]
     (first checkpoint past: 880), passes memory-free Bayes at 1,346, ends at .760; the ESN ends at .431, below naive.
  c  FR_T fifths: edges .073 / .263 / .740 / 2.70 nats, means .028 / .156 / .465 / 1.44 / 12.78 (labels .03 / .16 /
     .47 / 1.4 / 13 in both figures); the top fifth holds .859 of naive's excess at T (.794 summed over steps 1-30).
     Accuracy by fifth: naive .824 / .732 / .559 / .163 / .025; Bayes .833 / .777 / .758 / .753 / .727.
     Trained RNN: top fifth 16.5 (40) -> 4.33 [3.61-5.07] (880) -> .79 (2,000) -> .020 (end); at 880 by fifth .808 /
       1.06 / 1.36 / 1.81 / 4.33 (28.5x / 6.8x / 2.9x / 1.25x / .34x naive's; accuracy - naive's -.163 / -.187 / -.078 /
       +.261 / +.326); below naive's level for good from 9,600 / 2,600 / 1,520 / 1,040 / 320 (fifths 1 .. 5: crossing
       a level is earliest where the level is high); every fifth ends at .010-.020; accuracy by fifth at the end .827 /
       .768 / .748 / .744 / .721.
     ESN: bottom fifth 7.74 (40) -> .430 (end); top fifth 2.27 (initial) -> 8.05 (20) -> 7.07 (60, its lowest after
       initialization) -> 10.3 (720) -> 8.73 (1,520, the lowest within 640-4,000; 8.86 at 1,280) -> 14.70
       [12.88-16.57] (end). At the end its KL is 15.2x / 5.1x / 2.7x /
       1.6x / 1.15x naive's by fifth, above naive's for good in no fifth; accuracy - naive's -.112 / -.165 / -.102 /
       +.149 / +.070 (top fifth .094; all .431 vs .464). The top-fifth excess over naive is a tail of confident errors:
       per item, mean 14.70 vs 12.78, median 7.98 vs 7.84 (the ESN is above naive on 51.9% of items), 90th percentile
       37.4 vs 29.1, 99th 88.9 vs 60.3.
     Training objective (SANITY_loss: mean over steps and values of sqrt(1/2 (p - q) log(p / q)) on soft-clipped goal
       beliefs), share of its sum by the goal's FR_T fifth: naive .065 / .102 / .148 / .233 / .450 (45% in the top fifth,
       against 86% of KL at T); Trained RNN .159 / .176 / .192 / .213 / .260 (480), .145 / .172 / .191 / .215 / .277
       (880), .156 / .181 / .202 / .215 / .246 (end); ESN .097 / .132 / .163 / .216 / .392 (end). So the objective's
       residual is concentrated by FR for naive but nearly flat for the Trained RNN as it passes naive: no
       'pressure' reading of c.
  d  Control (no block) by fifth at 480 / 880 / 1,280 / 2,400: .442 / .354 / .282 / .226 / .184; .642 / .513 / .451 /
     .396 / .336; .742 / .662 / .604 / .548 / .485; .791 / .708 / .696 / .672 / .630.
     Run - control, bottom fifth | top fifth [5-95%, paired]:
       highest-FR half  480 +.057 | -.066 [-.084, -.047]; 880 +.025 | -.183 [-.209, -.155]; 1,280 +.004 | -.289
                        [-.319, -.258]; 2,400 -.003 [-.015, +.008] | -.301 [-.323, -.280] (top fifth .329, control .630)
       lowest-FR half   480 -.157 [-.182, -.132] | +.013; 880 -.189 [-.216, -.164] | -.006; 1,280 -.155 | +.002;
                        2,400 -.050 [-.065, -.036] | +.000 [-.012, +.012]
       random half      within .05 of the control in every fifth; 2,400 +.001 | -.006
     Against the random half at 2,400: highest-FR half, top fifth -.295 [-.316, -.272], bottom -.004; lowest-FR half,
     bottom fifth -.051 [-.064, -.038], top +.007. Dissociation (highest - lowest)_top - (highest - lowest)_bottom:
     -.145 (80), -.293 (480), -.391 (880), -.450 [-.482, -.418] (1,280, its peak), -.349 [-.377, -.319] (2,400);
     interval below 0 at all 30 checkpoints after initialization.
     Goal only (the variable the loss and the block act on): highest-FR half top fifth -.310 [-.335, -.287] at 2,400;
     lowest-FR half bottom fifth -.201 at 880, -.057 [-.074, -.038] at 2,400; dissociation -.362 [-.393, -.329] at
     2,400 (peak -.447 at 1,200).
     What the per-step ranking selects (probes as a stand-in for a batch): the FR-ranked block covers .119 / .197 /
     .463 / .778 / .942 of a trajectory's steps by its goal FR_T fifth, i.e. it nearly removes whole high-FR
     trajectories; an entanglement-ranked block covers .461 / .413 / .430 / .491 / .705, and the entanglement-ranked
     runs give a dissociation of only -.061 (880), -.090 [-.113, -.067] (1,280), -.056 (2,400) (supplement).
     World 2 (queue4 replicate: seed + 1000 = new initialization, episode stream and world; the runs' own tests,
     goal accuracy at T, mean of the tests in the last 100 raw episodes; no FR split there): highest-FR half .629,
     random half .694, lowest-FR half .672 at 2,400 training episodes; .561 / .624 / .590 at 1,440; .464 / .494 / .432
     at 880 (the lowest-FR half's early cost replicates too). Against the random half at the end (td_tt_learnmask
     scratch_summary, the runs' own tests, mean of the last 20): world 1 highest-FR -.088, lowest-FR -.027, no block
     +.007; world 2 -.065 / -.022.
     Controls (11-09). The blocked half does not carry more of the error signal: in the control run the highest-FR
     half's share of |dL/d increment|^2 (the training loss's gradient at the goal read-out increments, probes, 16
     checkpoints; td_tt_learnmask_grad.py) is .442-.509, a random half's .499-.502, and its read-out weight gradient is
     no larger than a random half's (.33-.56 vs .49-.51 of the batch's). Per step, FR_t is nearly uncorrelated with
     Bayes's surprise -log p(o_t | o_<t) (Pearson .05, Spearman .12 on the 1,000-probe subset; top halves overlap 54%,
     chance 50%), so the FR-ranked block is not a surprise block.

SUPPLEMENT (computed here, no longer drawn)
  Spread of a cell's position across environments (the former strip; sqrt(Sigma_d (Var_Z E[r_d | o_1:T] - mean
  within-cell variance / 24)), lattice spacings, mean over cells): Bayes .39 (raw .42), naive 2.26; Trained RNN .81
  (initial: a collapsed grid of size .06 shifted per environment) -> 3.19 (60) -> 1.47 (800) -> .41 (end); ESN 2.78 (60)
  -> 2.89 (800) -> 2.14 (end). It is .72-.92 (Trained RNN) and .57-.90 (ESN) of the RMS distance to the true cell at
  every checkpoint from 60 on (Bayes .39 / .45, naive 2.26 / 2.54), so it restates accuracy; dropped with the word
  'invariant'.
  Copying naive's errors (fig_td_summary b): on naive-wrong / Bayes-right probes, the Trained RNN gives naive's answer
  at most .279 (640), .026 at the end; the ESN .490 (max .515 at 6,800); the memory-free rule itself .318.
  Interaction-direction lesion (td_tt_lesions_fixed frsplit_*.npz; Trained RNN only), accuracy lost beyond random
  clamps, by the goal's FR_T third (edges .181 / 1.062): high third +4.2 pp (800) -> +11.8 (1,600, max) -> +5.5 (end);
  low <= +3.9, middle <= +4.6. Random clamps alone cost +4.2 / +6.5 / +11.6 pp (low / middle / high third, 2,400) and
  +2.9 / +4.2 / +7.6 (end): part of the FR specificity is general fragility of high-FR probes. Moved out of the main
  figure for that reason (a matched main-effect-subspace clamp and >= 5 random seeds are needed first).

CHECKS
  The grid-cache fold percentages (summary_data/grids.npz) equal this script's recomputation at every checkpoint; the
  series caches' accuracy and copying (analysis/series_<run>.npz) equal the recomputation from the ck files; the
  step-wise rule recomputed here reproduces the stored right_window1 on every probe; probes are environment-major
  (environment = probe // 200) and the grid episodes use the probes' 50 environments (identical Z); every block run's
  checkpoints match the control's (same 31 raw episodes).
CAVEATS
  One training run per network and per block condition in world 1; the intervals resample the 50 test environments
  only. World 2 replicates d's overall costs (no FR split). The environment-averaged grids in a mix folding with
  shrinkage; the fold strip (each environment's own grid) is the measure.
Run from the repository root (CPU, 4 threads; ~1 min with the caches, ~2 min to rebuild probes + blocks):
    MPLBACKEND=Agg OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/fig_learn_fr.py --numbers"""
import os
import re
import sys
import glob
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.colors import Normalize                                           # noqa: E402
from matplotlib.cm import ScalarMappable                                          # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from scipy.special import logsumexp, log_softmax                                  # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402
import td_summary_data as sd                                                      # noqa: E402
import td_tt_gap as gap                                                           # noqa: E402

TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
AN = os.path.join(TD, "analysis")
GRID_DIR = os.path.join(TD, "summary_data", "grids")
LES_DIR = os.path.join(TD, "through_training", "lesions", "trained")
BLK_DIR = os.path.join(TD, "through_training", "learnmask_fr")                   # td_tt_learnmask_fr replays (read only)
CACHE = {k: os.path.join(AN, f"learn_main_fig1_{k}.npz") for k in ("grids", "probes", "lesion", "blocks")}
RUNS = ("trained", "echo")
COL = {"trained": "#1f77b4", "echo": "#ff7f0e", "bayes": "k", "naive": "0.55"}
NAME = {"trained": "Trained RNN", "echo": "ESN", "bayes": "Bayes", "naive": "naive"}
T, R, NE, S_GRID, N_PROBE = 30, 10, 50, 24, 10000
SHOW = (60, 240, 880, 2400, 39996)           # grid columns (training episodes)
LES_CKS = (100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 2000, 2400, 3000, 4000, 5000, 7000, 10000, 15000, 20000,
           30000, 49995)                     # td_tt_lesions.CKS (raw episodes)
X0, XMIN, XMAX = 10, 8, 7e4
TS, FL, FT = 11, 10, 10                      # title, axis label, tick / annotation sizes (points, at print size)
FIG_W, DPI = 5.5, 300                        # print width (the paper's text width), inches
FIFTH_C = plt.cm.Purples(np.linspace(.3, 1., 5))           # FR fifths, light (low FR) to dark (high FR)
# d: from-scratch runs, 50% of each step's goal read-out increments detached, ranked by the goal's per-step FR_t
BLOCKS = (("ctrl_s0", "no block", "#1f77b4"), ("det_rand_f50_s0", "random half", "0.5"),
          ("det_lo_fr_f50_s0", "lowest-FR half", "#2ca02c"), ("det_hi_fr_f50_s0", "highest-FR half", "#d62728"))
BLOCK_RUNS = tuple(b[0] for b in BLOCKS) + ("det_hi_f50_s0", "det_lo_f50_s0")    # + the entanglement-ranked halves
REPLICATE = ("det_hi_fr_f50_s0_r2", "det_rand_f50_s0_r2", "det_lo_fr_f50_s0_r2")   # world 2 (queue4; own tests only)
DXMAX = 3000                                  # d: the from-scratch runs end at 2,400 training episodes


# ── caches ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def _observers_T(obs, L):
    """Bayes's and naive's exact log marginal posteriors at T, (n, 2, R) each, from observations (n, T, 5) and the
    episodes' tables L (n, 5, R, R) = p(o_i = 1 | r0, r1); uniform priors"""
    n1 = obs.sum(1)
    ll = (n1[..., None, None] * np.log(L) + (T - n1)[..., None, None] * np.log1p(-L)).sum(1)
    lj = ll - logsumexp(ll, (1, 2), keepdims=True)
    lb = np.stack([logsumexp(lj, 2), logsumexp(lj, 1)], 1)
    ln = np.stack([(n1[..., None] * np.log(p) + (T - n1)[..., None] * np.log1p(-p)).sum(1)
                   for p in (L.mean(3), L.mean(2))], 1)                   # naive's tables p(o_i | r0), p(o_i | r1)
    return lb, ln - logsumexp(ln, -1, keepdims=True)


def _per_pair(m, order):
    """(n, k) episode values -> (50, 100, 24, k), environment x cell x trajectory"""
    m = np.asarray(m)
    return m[order].reshape(NE, R * R, S_GRID, -1)


def _modal(a, order):
    """(n, 2) answers -> (50, 100) modal joint answer (r0 * 10 + r1) over each cell's 24 trajectories, and the same with
    ties broken to the highest joint index"""
    j = _per_pair(a.astype(np.int64), order)
    j = j[..., 0] * R + j[..., 1]
    cnt = np.zeros((NE, R * R, R * R), np.int64)
    np.add.at(cnt, (np.arange(NE)[:, None, None], np.arange(R * R)[None, :, None], j), 1)
    return cnt.argmax(-1), R * R - 1 - cnt[..., ::-1].argmax(-1)


def build_grids():
    P = np.load(os.path.join(AN, "probes.npz"))
    E = np.load(os.path.join(GRID_DIR, "fwd_episodes.npz"))
    env, cell = E["env"], E["cell"]
    assert np.array_equal(E["Z"], P["Z"]), "grid episodes and probes must share the 50 test environments"
    assert np.array_equal(cell, E["r"][:, 0] * R + E["r"][:, 1])
    key = env * R * R + cell
    assert (np.bincount(key, minlength=NE * R * R) == S_GRID).all()
    order = np.lexsort((np.arange(len(env)), key))
    L, obs = P["L"].astype(np.float64), E["obs"].astype(np.float64)
    fr, chk = np.zeros((len(env), 2)), 0.
    for s in range(0, len(env), 20000):
        sl = slice(s, s + 20000)
        lb, ln = _observers_T(obs[sl], L[env[sl]])
        fr[sl] = (np.exp(lb) * (lb - ln)).sum(-1)
        for lp, k in ((lb, "m_bayes"), (ln, "m_naive")):
            chk = max(chk, float(np.abs((np.exp(lp) * np.arange(R)).sum(-1) - E[k][sl, -1]).max()))
    out = {"pair_env_fr": _per_pair(fr.mean(1)[:, None], order)[..., 0].mean(-1), "means_check": chk}
    for ref in ("bayes", "naive"):
        X = _per_pair(E[f"m_{ref}"][:, -1], order).astype(np.float64)
        out[f"pos_{ref}"], out[f"wvar_{ref}"] = X.mean(2), X.var(2, ddof=1).sum(-1)
        out[f"modal_{ref}"], out[f"modal2_{ref}"] = _modal(E[f"a_{ref}"], order)
    for run in RUNS:
        F = np.load(os.path.join(GRID_DIR, f"fwd_{run}.npz"))
        out[f"tr_ep_{run}"] = F["tr_ep"]
        pos, wv, mo, mo2 = [], [], [], []
        for k in range(len(F["tr_ep"])):
            X = _per_pair(F["m"][k, :, -1], order).astype(np.float64)
            pos.append(X.mean(2))
            wv.append(X.var(2, ddof=1).sum(-1))
            a, b = _modal(F["a"][k], order)
            mo.append(a)
            mo2.append(b)
        out[f"pos_{run}"], out[f"wvar_{run}"] = np.array(pos), np.array(wv)
        out[f"modal_{run}"], out[f"modal2_{run}"] = np.array(mo, np.int16), np.array(mo2, np.int16)
    return out


def _softclip(x, eps=1e-3):
    """model_backward.soft_clip"""
    ciel = 1 - eps
    x = ciel - (1 / (1 + np.exp(-(ciel - x) / eps))) * (ciel - x)
    return eps + (1 / (1 + np.exp(-(x - eps) / eps))) * (x - eps)


def sanity(p, q, power=.5, eps=1e-8):
    """the training objective per item (model_backward.SANITY_loss = DKL_loss(DKL_sym(q, p, PM=False)), power .5): the
    mean over steps and values of ((p - q) log(p / q) / 2 + eps)^power on soft-clipped beliefs p (Bayes) and q (the
    learner), (n, T, R) -> (n,). The loss's subtraction of the batch minimum is dropped (every term is >= 0, the minimum
    ~0)"""
    p, q = _softclip(p), _softclip(q)
    t = .5 * (p * np.log(p / q) + q * np.log(q / p))
    return ((np.clip(t, 0, None) + eps) ** power).mean((1, 2))


def build_probes():
    P = np.load(os.path.join(AN, "probes.npz"))
    ar, goal, env, truth = np.arange(N_PROBE), P["goal"], P["env"], P["r"]
    assert np.array_equal(env, ar // 200)
    lj, ln = (log_softmax(P[k].astype(np.float64), -1) for k in ("log_jb", "log_nb"))
    pj = np.exp(lj)
    fr = (pj[:, -1] * (lj[:, -1] - ln[:, -1])).sum(-1)                              # (n, 2) forward FR_T
    edges = np.quantile(fr.ravel(), np.linspace(0, 1, 6))
    fifth = np.clip(np.searchsorted(edges, fr, side="right") - 1, 0, 4)
    eb = (env[:, None] * 5 + fifth).ravel()                                           # environment x fifth bins
    ebg = env * 5 + fifth[ar, goal]                                                   # the goal variable's fifth

    def by5(v):
        return np.bincount(eb, np.asarray(v, np.float64).ravel(), NE * 5).reshape(NE, 5)

    def by5g(v):
        return np.bincount(ebg, np.asarray(v, np.float64), NE * 5).reshape(NE, 5)
    tg = truth[ar, goal]
    nb_ans, jb_ans = ln[ar, -1, goal].argmax(-1), lj[ar, -1, goal].argmax(-1)
    no = (nb_ans != tg) & (jb_ans == tg)                                               # naive wrong, Bayes right
    # the memory-free (step-wise, window 1) rule's answers at T: the sum over steps of each step's own marginal log
    # likelihood (td_analysis.bayes_side, TRUNC_W = 1)
    sw = np.zeros((N_PROBE, 2, R))
    for s in range(0, N_PROBE, 2000):
        sl = slice(s, s + 2000)
        o, Le = P["obs"][sl].astype(np.float64), P["L"][env[sl]].astype(np.float64)
        ll = np.einsum("nti,nijk->ntjk", o, np.log(Le)) + np.einsum("nti,nijk->ntjk", 1 - o, np.log1p(-Le))
        sw[sl] = np.stack([logsumexp(ll, 3), logsumexp(ll, 2)], 2).sum(1)
    sw_ans = sw.argmax(-1)
    assert np.array_equal(sw_ans == truth, P["right_window1"])
    pg = pj[ar, :, goal]                                                               # (n, T, R) Bayes's goal beliefs
    out = {"edges5": edges, "n5": by5(np.ones_like(fr)), "fr5": by5(fr), "n5g": by5g(np.ones(N_PROBE)),
           "naive_sum5": by5((pj * (lj - ln)).sum(-1).sum(1)),
           "naive_acc5": by5(ln[:, -1].argmax(-1) == truth), "bayes_acc5": by5(lj[:, -1].argmax(-1) == truth),
           "obj_naive5g": by5g(sanity(pg, np.exp(ln[ar, :, goal]))),
           "n_env": np.bincount(env, minlength=NE).astype(np.float64),
           "n_no": np.bincount(env, no.astype(np.float64), NE),
           "ref_acc": np.stack([np.bincount(env, P[f"right_{k}"][ar, goal].astype(np.float64), NE)
                                for k in ("bayes", "window1", "naive")]),
           "sw_copy": np.bincount(env, ((sw_ans[ar, goal] == nb_ans) & no).astype(np.float64), NE)}
    for run in RUNS:
        fs = sorted(glob.glob(os.path.join(AN, run, "ck_*.npz")))
        ep, acc, cp, kl5, kls5, acc5, obj5 = [], [], [], [], [], [], []
        for f in fs:
            z = np.load(f)
            lq = log_softmax(z["logq"].astype(np.float64), -1)
            right = z["right"][:, -1]
            ep.append(int(z["episode"]))
            acc.append(np.bincount(env, right[ar, goal].astype(np.float64), NE))
            cp.append(np.bincount(env, ((lq[ar, -1, goal].argmax(-1) == nb_ans) & no).astype(np.float64), NE))
            kl5.append(by5((pj[:, -1] * (lj[:, -1] - lq[:, -1])).sum(-1)))
            kls5.append(by5((pj * (lj - lq)).sum(-1).sum(1)))
            acc5.append(by5(right))
            obj5.append(by5g(sanity(pg, np.exp(lq[ar, :, goal]))))
        out.update({f"ep_{run}": np.array(ep), f"acc_{run}": np.array(acc), f"copy_{run}": np.array(cp),
                    f"kl5_{run}": np.array(kl5), f"kls5_{run}": np.array(kls5), f"acc5_{run}": np.array(acc5),
                    f"obj5g_{run}": np.array(obj5)})
    return out


def build_lesion():
    """supplement (no longer drawn): the interaction-direction lesion split by thirds of the goal's own FR_T"""
    P = np.load(os.path.join(AN, "probes.npz"))
    ar, goal, env = np.arange(N_PROBE), P["goal"], P["env"]
    lj, ln = (log_softmax(P[k][:, -1].astype(np.float64), -1) for k in ("log_jb", "log_nb"))
    frg = (np.exp(lj) * (lj - ln)).sum(-1)[ar, goal]                                   # the goal's own FR_T
    edges = np.quantile(frg, [0, 1 / 3, 2 / 3, 1])
    third = np.clip(np.searchsorted(edges, frg, side="right") - 1, 0, 2)
    eb = env * 3 + third

    def by3(v):
        return np.bincount(eb, np.asarray(v, np.float64), NE * 3).reshape(NE, 3)
    out = {"edges3": edges, "n3": by3(np.ones(N_PROBE)), "fr_goal_fwd": frg, "fr_mean_sym": P["fr_sym"][:, -1].mean(1)}
    x, loss, intact, rnd = [], [], [], []
    for E in LES_CKS:
        z = np.load(os.path.join(LES_DIR, f"frsplit_{E:05d}.npz"))
        assert int(z["E"]) == E
        x.append(float(z["x"]))
        vm = (z["fix5_vm_s0"].astype(np.float64) + z["fix5_vm_s1"].astype(np.float64)) / 2
        loss.append(by3(vm - z["fix5"].astype(np.float64)))
        intact.append(by3(z["intact"]))
        rnd.append(by3(z["intact"].astype(np.float64) - vm))
    out.update(x=np.array(x), loss3=np.array(loss), intact3=np.array(intact), rand3=np.array(rnd))
    return out


def _blk_files(run):
    """the run's per-checkpoint replay files (exact name: det_hi_* would also match det_hi_f50_*)"""
    pat = re.compile(rf"{re.escape(run)}_\d{{5}}\.npz")
    return sorted(f for f in glob.glob(os.path.join(BLK_DIR, f"{run}_*.npz")) if pat.fullmatch(os.path.basename(f)))


def build_blocks():
    """d: per from-scratch run, checkpoint, environment and FR_T fifth, the number of right answers at T, for both
    variables (each variable in the fifth of its own FR_T, the pooled quintiles of b-c) and for the goal alone"""
    P = np.load(os.path.join(AN, "probes.npz"))
    ar, goal, env = np.arange(N_PROBE), P["goal"], P["env"]
    lj, ln = (log_softmax(P[k][:, -1].astype(np.float64), -1) for k in ("log_jb", "log_nb"))
    fr = (np.exp(lj) * (lj - ln)).sum(-1)
    edges = np.quantile(fr.ravel(), np.linspace(0, 1, 6))
    fifth = np.clip(np.searchsorted(edges, fr, side="right") - 1, 0, 4)
    eb, ebg = (env[:, None] * 5 + fifth).ravel(), env * 5 + fifth[ar, goal]

    def by5(v):
        return np.bincount(eb, np.asarray(v, np.float64).ravel(), NE * 5).reshape(NE, 5)

    def by5g(v):
        return np.bincount(ebg, np.asarray(v, np.float64), NE * 5).reshape(NE, 5)
    out = {"edges5": edges, "n5": by5(np.ones((N_PROBE, 2))), "n5g": by5g(np.ones(N_PROBE))}
    for k in ("bayes", "naive"):
        out[f"{k}5"], out[f"{k}5g"] = by5(P[f"right_{k}"]), by5g(P[f"right_{k}"][ar, goal])
    for run in BLOCK_RUNS:
        x, s, sg = [], [], []
        for f in _blk_files(run):
            z = np.load(f)
            x.append(float(z["x"]))
            s.append(by5(z["right"]))
            sg.append(by5g(z["right"][ar, goal]))
        out[f"x_{run}"], out[f"s5_{run}"], out[f"s5g_{run}"] = np.array(x), np.array(s), np.array(sg)
    return out


def load(name, rebuild=()):
    """the named cache; rebuilt when its name is in `rebuild` (or `rebuild` is True) or when it does not exist"""
    path = CACHE[name]
    if rebuild is True or name in rebuild or not os.path.exists(path):
        out = {"grids": build_grids, "probes": build_probes, "lesion": build_lesion, "blocks": build_blocks}[name]()
        np.savez(path, **out)
        print("cached", path)
    return dict(np.load(path))


# ── statistics ──────────────────────────────────────────────────────────────────────────────────────────────────────

def W():
    """(1 + 1,000, 50) environment weights: the full sample, then td_tt_gap's shared resamples"""
    return np.vstack([np.ones((1, NE)), gap.boot_counts()])


def band(v):
    """value, 5% and 95% of a (1 + B, ...) array"""
    return v[0], np.percentile(v[1:], 5, 0), np.percentile(v[1:], 95, 0)


def train_x(ep):
    return gap.train_x(ep)


TRUTH = np.stack(np.meshgrid(np.arange(R), np.arange(R), indexing="ij"), -1).reshape(R * R, 2).astype(float)


def grid_stats(G):
    """fold %, spread across environments, RMS error and answers per environment, per checkpoint and reference"""
    fN = sd.folded(G["pos_naive"])                                    # (50, 162) where naive's grid folds
    Wm = W()
    S = {"naive_folded": 100 * fN.mean(), "fN": fN}

    def folds(pos):                                                   # (K, 50, 100, 2) -> 2 x (1 + B, K)
        f = sd.folded(pos)                                            # (K, 50, 162)
        num_w, num_e = (f & fN).sum(-1).T, (f & ~fN).sum(-1).T      # (50, K)
        den_w, den_e = fN.sum(-1).astype(float), (~fN).sum(-1).astype(float)
        return 100 * (Wm @ num_w) / (Wm @ den_w)[:, None], 100 * (Wm @ num_e) / (Wm @ den_e)[:, None]

    def spread(pos, wv):                                              # (K, 50, 100, 2), (K, 50, 100) -> (1 + B, K)
        out = []
        for w in Wm:
            mu = np.einsum("e,kepd->kpd", w, pos) / w.sum()
            v = np.einsum("e,kepd->kp", w, (pos - mu[:, None]) ** 2) / (w.sum() - 1)
            noise = np.einsum("e,kep->kp", w, wv) / w.sum() / S_GRID
            out.append(np.sqrt(np.clip(v - noise, 0, None)).mean(-1))
        return np.array(out)

    def rms_err(pos):                                                 # RMS distance to the true cell, mean over cells
        return np.sqrt(((pos - TRUTH) ** 2).sum(-1).mean(-2)).mean(-1)

    def answers(mo):                                                  # (K, 50, 100) -> (K, 50)
        return np.array([[len(np.unique(m)) for m in mk] for mk in mo], np.float64)
    for r in ("bayes", "naive"):
        p, wv = G[f"pos_{r}"][None], G[f"wvar_{r}"][None]
        S[f"where_{r}"], S[f"else_{r}"] = (v[:, 0] for v in folds(p))
        S[f"spread_{r}"] = spread(p, wv)[:, 0]
        S[f"spread_raw_{r}"] = spread(p, np.zeros_like(wv))[0, 0]
        S[f"err_{r}"] = rms_err(p)[0]
        S[f"ans_{r}"] = answers(G[f"modal_{r}"][None])[0]
        S[f"ans2_{r}"] = answers(G[f"modal2_{r}"][None])[0]
        g = G[f"pos_{r}"].mean(0)
        S[f"grid_{r}"] = g                                            # (100, 2) the environment-averaged grid
        S[f"size_{r}"] = np.sqrt(((g - g.mean(0)) ** 2).sum(-1).mean())
    for r in RUNS:
        pos = G[f"pos_{r}"]
        S[f"x_{r}"] = np.where(G[f"tr_ep_{r}"] > 0, G[f"tr_ep_{r}"], X0).astype(float)
        S[f"where_{r}"], S[f"else_{r}"] = folds(pos)
        S[f"spread_{r}"] = spread(pos, G[f"wvar_{r}"])
        S[f"spread_raw_{r}"] = spread(pos, np.zeros_like(G[f"wvar_{r}"]))[0]
        S[f"err_{r}"] = rms_err(pos)
        S[f"ans_{r}"], S[f"ans2_{r}"] = answers(G[f"modal_{r}"]), answers(G[f"modal2_{r}"])
        g = pos.mean(1)
        S[f"grid_{r}"] = g                                            # (K, 100, 2)
        S[f"size_{r}"] = np.sqrt(((g - g.mean(1, keepdims=True)) ** 2).sum(-1).mean(-1))
    S["pair_fr"] = G["pair_env_fr"].mean(0)
    return S


def probe_stats(Q):
    """behaviour and loss-by-fifth curves with their resamples"""
    Wm = W()
    S = {"fr5": (Q["fr5"].sum(0) / Q["n5"].sum(0)), "edges5": Q["edges5"]}
    S["naive_sum5"] = Q["naive_sum5"].sum(0) / Q["n5"].sum(0)
    S["naive_acc5"], S["bayes_acc5"] = (Q[k].sum(0) / Q["n5"].sum(0) for k in ("naive_acc5", "bayes_acc5"))
    S["fr5_b"] = (Wm @ Q["fr5"]) / (Wm @ Q["n5"])                                     # (1 + B, 5)
    ref = (Wm @ Q["ref_acc"].T) / (Wm @ Q["n_env"])[:, None]                           # (1 + B, 3)
    S["bayes"], S["stepwise"], S["naive"] = ref[:, 0], ref[:, 1], ref[:, 2]
    S["sw_copy"] = (Wm @ Q["sw_copy"]) / (Wm @ Q["n_no"])
    S["obj_naive_share"] = Q["obj_naive5g"].sum(0) / Q["obj_naive5g"].sum()
    for r in RUNS:
        S[f"x_{r}"] = train_x(Q[f"ep_{r}"])
        S[f"acc_{r}"] = (Wm @ Q[f"acc_{r}"].T) / (Wm @ Q["n_env"])[:, None]            # (1 + B, K)
        S[f"copy_{r}"] = (Wm @ Q[f"copy_{r}"].T) / (Wm @ Q["n_no"])[:, None]
        n5 = Wm @ Q["n5"]                                                              # (1 + B, 5)
        S[f"kl5_{r}"] = np.einsum("be,kef->bkf", Wm, Q[f"kl5_{r}"]) / n5[:, None]      # (1 + B, K, 5)
        S[f"kls5_{r}"] = Q[f"kls5_{r}"].sum(1) / Q["n5"].sum(0)                        # (K, 5) full sample
        S[f"acc5_{r}"] = Q[f"acc5_{r}"].sum(1) / Q["n5"].sum(0)
        o = Q[f"obj5g_{r}"].sum(1)                                                     # (K, 5) summed objective
        S[f"obj_share_{r}"] = o / o.sum(1, keepdims=True)
        S[f"obj_mean_{r}"] = o / Q["n5g"].sum(0)
    S["obj_naive_mean"] = Q["obj_naive5g"].sum(0) / Q["n5g"].sum(0)
    x, acc = S["x_trained"], S["acc_trained"]
    S["pass_first"] = float(x[np.argmax(acc[0] > S["naive"][0])])                      # fd.passes_naive's rule
    S["pass_cross"] = gap.crossing(x, acc[0] - S["naive"][0], 0.)
    S["pass_iv"] = gap.interval(gap.crossing(x, acc[1:] - S["naive"][1:, None], 0.))
    return S


def lesion_stats(Lz):
    Wm = W()
    n = (Wm @ Lz["n3"])[:, None]
    return {"x": Lz["x"], "loss": 100 * np.einsum("be,keg->bkg", Wm, Lz["loss3"]) / n, "edges3": Lz["edges3"],
            "rand": 100 * np.einsum("be,keg->bkg", Wm, Lz["rand3"]) / n,
            "intact": Lz["intact3"].sum(1) / Lz["n3"].sum(0)}


def block_stats(Bz):
    """d: accuracy by FR_T fifth through training per run, (1 + B, K, 5) with its environment resamples; 'g' = goal"""
    Wm = W()
    S = {"edges5": Bz["edges5"]}
    for sfx in ("", "g"):
        n = Wm @ Bz[f"n5{sfx}"]                                                        # (1 + B, 5)
        for k in ("bayes", "naive"):
            S[f"{k}{sfx}"] = (Wm @ Bz[f"{k}5{sfx}"]) / n
        for run in BLOCK_RUNS:
            S[f"acc{sfx}_{run}"] = np.einsum("be,kef->bkf", Wm, Bz[f"s5{sfx}_{run}"]) / n[:, None]
    for run in BLOCK_RUNS:
        S[f"x_{run}"] = np.where(Bz[f"x_{run}"] > 0, Bz[f"x_{run}"], X0)
    return S


# ── drawing helpers ─────────────────────────────────────────────────────────────────────────────────────────────────

def xaxis(ax, label=True, passing=None, xmax=XMAX, ticks=(X0, 1e2, 1e3, 1e4), labels=("0", "100", "1k", "10k")):
    ax.set_xscale("log")
    ax.set_xticks(list(ticks))
    ax.set_xticklabels(list(labels))
    ax.set_xlim(XMIN, xmax)
    ax.minorticks_off()
    ax.tick_params(labelsize=FT, pad=2)
    if label:
        ax.set_xlabel("training episodes", fontsize=FL, labelpad=1)
    else:
        ax.tick_params(labelbottom=False)
    ax.spines[["top", "right"]].set_visible(False)
    if passing is not None:
        ax.axvline(passing, color="0.4", lw=.9, ls=":", zorder=1)


def title(ax, letter, name, formula=None):
    """panel title above the axes: the name (TS) and, on the line below it, the formula (FL)"""
    head = f"{letter}   {name}" if letter else name
    if formula:
        ax.annotate(formula, xy=(0, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                    ha="left", va="bottom", fontsize=FL, color="0.1", annotation_clip=False)
    ax.annotate(head, xy=(0, 1), xycoords="axes fraction", xytext=(0, 4 + (15 if formula else 0)),
                textcoords="offset points", ha="left", va="bottom", fontsize=TS, annotation_clip=False)


def mini_grid(ax, g, c, norm):
    """one belief grid: cells (100, 2) in r0-major order, coloured by c (100,)"""
    pts = g.reshape(R, R, 2)
    for i in range(R):
        ax.plot(pts[i, :, 0], pts[i, :, 1], color="0.62", lw=.35, zorder=1, solid_capstyle="round")
        ax.plot(pts[:, i, 0], pts[:, i, 1], color="0.62", lw=.35, zorder=1, solid_capstyle="round")
    ax.scatter(pts[..., 0].ravel(), pts[..., 1].ravel(), c=c, cmap="viridis", norm=norm, s=3.2, lw=0, zorder=3)
    ax.set_xlim(-.5, R - .5)
    ax.set_ylim(-.5, R - .5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.75")
        s.set_linewidth(.6)


def _dark(c, f=.8):
    return np.array(plt.matplotlib.colors.to_rgba(c)) * [f, f, f, 1]


def fifth_label(v):
    """the same fifth labels in both figures: .03 / .16 / .47 / 1.4 / 13"""
    return f"{v:.2f}".lstrip("0") if v < 1 else (f"{v:.1f}" if v < 10 else f"{v:.0f}")


# ── figure ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def figure(rebuild=()):
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({"xtick.labelsize": FT, "ytick.labelsize": FT, "axes.labelsize": FL, "legend.fontsize": FT,
                         "axes.linewidth": .7, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    G = grid_stats(load("grids", rebuild))
    Q = probe_stats(load("probes", rebuild))
    Bk = block_stats(load("blocks", rebuild))
    PASS = Q["pass_first"]
    Wf, Hf = FIG_W, 8.36
    fig = plt.figure(figsize=(Wf, Hf))

    def add(x, top, w, h, **k):                       # axes placed by its left edge and its top (inches from the top)
        return fig.add_axes([x / Wf, (Hf - top - h) / Hf, w / Wf, h / Hf], **k)

    def ftext(x, top, s, **k):                        # figure text with its baseline `top` inches below the top
        k.setdefault("va", "baseline")
        return fig.text(x / Wf, (Hf - top) / Hf, s, **k)

    # ── a: belief grids ─────────────────────────────────────────────────────────────────────────────────────────────
    x0, gs, gx, gr = .42, .655, .075, .1               # first grid's left edge, grid size, gap, extra reference gap
    xcol = [x0 + j * (gs + gx) for j in range(5)] + [x0 + 5 * (gs + gx) + gr]
    ftext(.04, .2, "a   Belief grid at T, mean over 50 held-out environments", fontsize=TS)
    ftext(.04, .42, r"$(\mathbb{E}[r_0\,|\,o_{1:T}],\ \mathbb{E}[r_1\,|\,o_{1:T}])$ of each cell $(r_0, r_1)$;  "
          "number: distinct answers (of 100)", fontsize=FL, color="0.1")
    ys, yh, y1, y2 = .615, .78, .84, 1.72               # spanner, column headers, tops of the two grid rows
    xa, xb = xcol[0] + .03, xcol[4] + gs - .03
    fig.add_artist(Line2D([xa / Wf, xb / Wf], [(Hf - ys - .03) / Hf] * 2, color="0.45", lw=.7))
    ftext((xa + xb) / 2, ys, "training episodes", ha="center", fontsize=FT, color="0.2")
    norm = Normalize(0, float(np.quantile(G["pair_fr"], .95)))
    ia = {r: [int(np.argmin(np.abs(G[f"x_{r}"] - s))) for s in SHOW] for r in RUNS}
    for row, (r, refn, top) in enumerate((("trained", "bayes", y1), ("echo", "naive", y2))):
        for j in range(6):
            ax = add(xcol[j], top, gs, gs)
            if j < 5:
                g, nans = G[f"grid_{r}"][ia[r][j]], G[f"ans_{r}"][ia[r][j]].mean()
            else:
                g, nans = G[f"grid_{refn}"], G[f"ans_{refn}"].mean()
            mini_grid(ax, g, G["pair_fr"], norm)
            lab = f"{nans:.1f}" if j < 5 else f"{NAME[refn]} {nans:.1f}"
            ftext(xcol[j] + gs / 2, top + gs + .15, lab, ha="center", fontsize=FT, color="0.2")
            if row == 0 and j < 5:
                ftext(xcol[j] + gs / 2, yh, f"{SHOW[j]:,}", ha="center", fontsize=FT)
        ftext(.2, top + gs / 2, NAME[r], rotation=90, ha="center", va="center", fontsize=FL + .5, color=COL[r],
              weight="bold")
    cx = xcol[5] + gs + .1
    cax = add(cx, y1 + .05, .07, y2 + gs - y1 - .1)
    cb = plt.colorbar(ScalarMappable(norm=norm, cmap="viridis"), cax=cax, extend="max")
    cb.set_ticks([0, 2, 4, 6])
    cb.ax.tick_params(labelsize=FT, length=2, pad=1.5)
    cb.set_label(r"FR$_T$ of the cell (nats)", fontsize=FT, labelpad=2)

    # ── a (strip) and b: grid folds, accuracy ───────────────────────────────────────────────────────────────────────
    t2, h2 = 3.09, .82                                 # axes top, height (inches)
    la, wa = .52, 2.12
    a2 = add(la, t2, wa, h2)
    for r in RUNS:
        x = G[f"x_{r}"]
        for key, ls in (("where", "-"), ("else", (0, (3, 1.6)))):
            v, lo, hi = band(G[f"{key}_{r}"])
            a2.fill_between(x, lo, hi, color=COL[r], alpha=.15, lw=0)
            a2.plot(x, v, color=COL[r], lw=1.5, ls=ls, marker="o", ms=1.8)
    a2.axhline(G["where_bayes"][0], color="k", lw=.8, ls=":", zorder=1)
    a2.set_ylim(0, 100)
    a2.set_yticks([0, 50, 100])
    a2.set_ylabel("% folded", fontsize=FL, labelpad=1)
    xaxis(a2, passing=PASS)
    for (x1, x2), y, lab, ls in (((100, 190), 92, "where naive folds", "-"),
                                  ((3.2e3, 6e3), 33, "elsewhere", (0, (3, 1.6)))):
        a2.plot([x1, x2], [y, y], color="0.3", lw=1.3, ls=ls, clip_on=False, zorder=6)
        a2.text(x2 * 1.12, y, lab, fontsize=FT, color="0.2", ha="left", va="center", zorder=2.5,
                bbox=dict(fc="w", ec="none", pad=.5))
    a2.text(XMIN * 1.1, G["where_bayes"][0] + 3, "Bayes", fontsize=FT, ha="left", va="bottom")
    title(a2, "", "Grid folded at T", r"% of triangles with signed area $<0$")

    lb, wb = 3.27, 2.12
    b = add(lb, t2, wb, h2)
    for r in RUNS:
        x = Q[f"x_{r}"]
        v, lo, hi = band(Q[f"acc_{r}"])
        b.fill_between(x, lo, hi, color=COL[r], alpha=.2, lw=0)
        b.plot(x, v, color=COL[r], lw=1.6)
    for key, lab, c in (("bayes", "Bayes", "k"), ("stepwise", "mem.-free Bayes", "0.4"), ("naive", "naive", "0.4")):
        b.axhline(Q[key][0], color=c, lw=.8, ls=":", zorder=1)
        b.text(XMIN * 1.1, Q[key][0] + .012, lab, fontsize=FT, color=c, ha="left", va="bottom")
    b.set_ylim(0, .9)
    b.set_yticks([0, .4, .8])
    b.set_ylabel("goal accuracy", fontsize=FL, labelpad=1)
    xaxis(b, passing=PASS)
    for r, yy in (("trained", .9), ("echo", .36)):
        b.text(XMAX, yy, NAME[r], fontsize=FT, color=COL[r], ha="right", va="top", weight="bold")
    title(b, "b", "Goal accuracy at T", r"$P(\hat g_T=g)$")

    # ── c: excess log loss by FR fifth ──────────────────────────────────────────────────────────────────────────────
    t3, h3 = 4.80, 1.12
    lc, wc, gc = .52, 2.2, .1
    cax_ = [add(lc, t3, wc, h3), add(lc + wc + gc, t3, wc, h3)]
    for ax, r in zip(cax_, RUNS):
        x = Q[f"x_{r}"]
        for k in range(5):
            ax.axhline(Q["fr5"][k], color=FIFTH_C[k], lw=.9, ls=(0, (1, 1.5)), zorder=1)
            v, lo, hi = band(Q[f"kl5_{r}"][..., k])
            ax.fill_between(x, lo, hi, color=FIFTH_C[k], alpha=.25, lw=0, zorder=2)
            ax.plot(x, v, color=FIFTH_C[k], lw=1.5, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(4e-3, 45)
        xaxis(ax, label=False, passing=PASS)
        ax.tick_params(labelbottom=True)
        ax.text(.04, .04, NAME[r], transform=ax.transAxes, fontsize=FL, color=COL[r], weight="bold", ha="left",
                va="bottom")
    cax_[0].set_ylabel(r"nats  (Bayes $=0$)", fontsize=FL, labelpad=1)
    cax_[0].set_yticks([.01, .1, 1, 10])
    cax_[0].set_yticklabels(["0.01", "0.1", "1", "10"])
    cax_[1].set_yticks([.01, .1, 1, 10])
    cax_[1].tick_params(labelleft=False)
    for k in range(5):
        cax_[1].text(XMAX * 1.12, Q["fr5"][k], fifth_label(Q["fr5"][k]), fontsize=FT, va="center", ha="left",
                     color=_dark(FIFTH_C[k], .7 if k < 2 else 1.))
    cax_[1].text(XMAX * 1.12, 40, r"FR$_T$", fontsize=FT, va="center", ha="left", color="0.2")
    title(cax_[0], "c", r"Excess log loss at T, by FR$_T$ fifth",
          r"$\mathrm{KL}(B_T\,\Vert\,q_T)$;  dotted: naive, $\mathrm{KL}(B_T\,\Vert\,N_T)=\mathrm{FR}_T$")
    ftext(lc + wc + gc / 2, t3 + h3 + .34, "training episodes", ha="center", fontsize=FL)

    # ── d: learning blocked at FR-ranked steps, from episode 0 ──────────────────────────────────────────────────────
    t4, h4 = 6.78, .9
    ld, wd, gd = .52, 1.62, .1
    dax = [add(ld, t4, wd, h4), add(ld + wd + gd, t4, wd, h4)]
    for ax, f, lab, ly in zip(dax, (4, 0), (r"FR$_T$ top fifth", r"FR$_T$ bottom fifth"), (.53, .53)):
        for run, _, c in BLOCKS:
            x = Bk[f"x_{run}"]
            v, lo, hi = band(Bk[f"acc_{run}"][..., f])
            ax.fill_between(x, lo, hi, color=c, alpha=.18, lw=0, zorder=2)
            ax.plot(x, v, color=c, lw=1.5, zorder=3)
        for k, c in (("bayes", "k"), ("naive", "0.45")):
            ax.axhline(Bk[k][0, f], color=c, lw=.8, ls=":", zorder=1)
        ax.set_ylim(0, 1.)
        ax.set_yticks([0, .4, .8])
        xaxis(ax, label=False, passing=PASS, xmax=DXMAX, ticks=(X0, 1e2, 1e3), labels=("0", "100", "1k"))
        ax.tick_params(labelbottom=True)
        ax.text(XMIN * 1.1, ly, lab, fontsize=FT, ha="left", va="bottom", color="0.1")
    dax[0].set_ylabel("accuracy", fontsize=FL, labelpad=1)
    dax[1].tick_params(labelleft=False)
    dax[0].text(XMIN * 1.1, Bk["bayes"][0, 4] + .015, "Bayes", fontsize=FT, ha="left", va="bottom")
    dax[0].text(DXMAX, Bk["naive"][0, 4] + .015, "naive", fontsize=FT, ha="right", va="bottom", color="0.4",
                zorder=2.5, bbox=dict(fc="w", ec="none", pad=.4))           # above the dotted line, below the data
    dax[1].text(XMIN * 1.1, Bk["bayes"][0, 0] + .015, r"Bayes $\approx$ naive", fontsize=FT, ha="left", va="bottom")
    ftext(ld + wd + gd / 2, t4 + h4 + .34, "training episodes", ha="center", fontsize=FL)
    kx = ld + 2 * wd + gd + .14
    ftext(kx, t4 + .02, "50% blocked:", fontsize=FT, ha="left", color="0.2")
    for i, (_, lab, c) in enumerate(BLOCKS[::-1]):
        yy = t4 + .22 + i * .19
        fig.add_artist(Line2D([kx / Wf, (kx + .22) / Wf], [(Hf - yy + .035) / Hf] * 2, color=c, lw=1.6))
        ftext(kx + .27, yy, lab, fontsize=FT, ha="left")
    title(dax[0], "d", "Learning blocked at FR-ranked steps", r"$P(\hat r_{v,T}=r_v)$, both variables")

    note = ftext(Wf / 2, Hf - .05, r"50 environments $\times$ 200 probes $\times$ 114 checkpoints (a: 2,400 $\times$ 19;"
                 r" d: 31); one run each", ha="center", fontsize=FT, color="0.25")
    fig.canvas.draw()
    bb = note.get_window_extent()
    assert bb.x0 >= 0 and bb.x1 <= fig.bbox.width, "the data line does not fit"
    path = fig_path("fig_learn_fr.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("saved", path)
    return G, Q, Bk


# ── numbers ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def _auc(pos, score):
    """area under the ROC curve of `score` for the boolean `pos` (ties averaged)"""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _iv(b, nd=3):
    """'v [lo, hi]' from a (1 + B, ...) array's first axis"""
    v, lo, hi = band(b)
    return f"{v:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def _at(x, xv):
    return int(np.argmin(np.abs(np.log(x) - np.log(xv))))


def replicate_numbers():
    """world 2 (seed + 1000: a new initialization, episode stream and world; queue4): the runs' own tests only (8,000
    episodes each, every 5th raw episode); no FR split exists there"""
    import json
    out = {}
    for run in REPLICATE:
        d = os.path.join(TD, f"learnmask_{run}_s20261029")
        f = os.path.join(d, "tests.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        done = os.path.exists(os.path.join(d, "progress.log")) and "DONE" in open(os.path.join(d, "progress.log")).read()
        raw, acc = z["raw"], z["acc"][:, -1].astype(np.float64)
        keep = raw > 0
        out[run] = dict(done=done, raw=raw[keep], acc=acc[keep])
    return out


def numbers(G=None, Q=None, Bk=None, rebuild=()):
    G = G if G is not None else grid_stats(load("grids"))
    Q = Q if Q is not None else probe_stats(load("probes"))
    Bk = Bk if Bk is not None else block_stats(load("blocks"))
    Ls = lesion_stats(load("lesion", rebuild))
    Gz = load("grids")
    print("== a  belief grids")
    print(f"  means check (exact posterior means against the stored ones): {float(Gz['means_check']):.1e}")
    print(f"  cell FR_T: min {G['pair_fr'].min():.2f}, median {np.median(G['pair_fr']):.2f}, max {G['pair_fr'].max():.2f},"
          f" 95th pct {np.quantile(G['pair_fr'], .95):.2f}")
    print(f"  naive folded {G['naive_folded']:.1f}%; Bayes where naive folds {G['where_bayes'][0]:.2f}%, elsewhere "
          f"{G['else_bayes'][0]:.2f}%")
    print(f"  answers per environment: Bayes {G['ans_bayes'].mean():.1f} (ties high {G['ans2_bayes'].mean():.1f}), naive "
          f"{G['ans_naive'].mean():.1f} ({G['ans2_naive'].mean():.1f})")
    gm = np.load(os.path.join(TD, "summary_data", "grids.npz"))                      # read only
    for r in RUNS:
        same = np.allclose(G[f"where_{r}"][0], gm[f"where_naive_{r}"]) and np.allclose(G[f"else_{r}"][0],
                                                                                      gm[f"elsewhere_{r}"])
        print(f"  {NAME[r]}: fold % equals summary_data/grids.npz: {same}; max |ties low - ties high| answers "
              f"{np.abs(G[f'ans_{r}'].mean(1) - G[f'ans2_{r}'].mean(1)).max():.2f}")
        for k, x in enumerate(G[f"x_{r}"]):
            w, e = band(G[f"where_{r}"]), band(G[f"else_{r}"])
            print(f"    {x:7.0f}: answers {G[f'ans_{r}'][k].mean():5.1f} | folded where naive folds {w[0][k]:5.1f} "
                  f"[{w[1][k]:4.1f}-{w[2][k]:4.1f}], elsewhere {e[0][k]:5.1f} [{e[1][k]:4.1f}-{e[2][k]:4.1f}]")
    print("  -- where naive folds = where FR is high (C14): triangle FR = mean FR_T of its 3 cells per environment")
    fN, tri_fr = G["fN"], G_tri_fr(Gz)
    print(f"    mean triangle FR where naive folds {tri_fr[fN].mean():.2f} nats, elsewhere {tri_fr[~fN].mean():.2f}; AUC of "
          f"triangle FR for naive's folds {_auc(fN.ravel(), tri_fr.ravel()):.3f}; naive-fold triangles in the lower "
          f"half of triangle FR {np.mean(tri_fr[fN] < np.median(tri_fr)):.2f}")
    rc = np.stack([sd.TRI // R, sd.TRI % R], -1)
    border = np.broadcast_to(((rc == 0) | (rc == R - 1)).any((1, 2)), fN.shape)
    print(f"    triangles touching the border: {border[0].mean():.2f} of all, {border[fN].mean():.2f} of naive's folds")
    for r in RUNS:
        f = sd.folded(Gz[f"pos_{r}"])
        for k in (_at(G[f"x_{r}"], 880), len(G[f"x_{r}"]) - 1):
            fk = f[k]
            print(f"    {NAME[r]:11s} {G[f'x_{r}'][k]:6.0f}: border triangles folded {100 * fk[fN & border].mean():4.1f}% "
                  f"where naive folds vs {100 * fk[~fN & border].mean():4.1f}% elsewhere; interior "
                  f"{100 * fk[fN & ~border].mean():4.1f}% vs {100 * fk[~fN & ~border].mean():4.1f}%")
    print("  -- supplement (no longer drawn): spread of a cell's position across environments, noise-corrected, lattice "
          "spacings, against the RMS error to the true cell (C4)")
    print(f"    Bayes spread {G['spread_bayes'][0]:.3f} (raw {G['spread_raw_bayes']:.3f}) / RMS error {G['err_bayes']:.3f};"
          f" naive {G['spread_naive'][0]:.3f} (raw {G['spread_raw_naive']:.3f}) / {G['err_naive']:.3f}")
    for r in RUNS:
        sp, er = G[f"spread_{r}"][0], G[f"err_{r}"]
        rat = sp / er
        print(f"    {NAME[r]:11s} spread / RMS error: " + "  ".join(
            f"{x:,.0f} {a:.2f}/{b:.2f}" for x, a, b in zip(G[f"x_{r}"][[0, 1, 3, 9, 12, 18]], sp[[0, 1, 3, 9, 12, 18]],
                                                           er[[0, 1, 3, 9, 12, 18]])) +
            f";  ratio range from 60 on {rat[1:].min():.2f}-{rat[1:].max():.2f} (initial {rat[0]:.2f}); grid size "
            f"{G[f'size_{r}'][0]:.2f} (initial) -> {G[f'size_{r}'][-1]:.2f}")
    print("== b  goal accuracy at T")
    print(f"  references: Bayes {Q['bayes'][0]:.4f}, memory-free {Q['stepwise'][0]:.4f}, naive {Q['naive'][0]:.4f}")
    print(f"  Trained RNN passes naive: first checkpoint {Q['pass_first']:.0f}; crossing {Q['pass_cross']:.0f} "
          f"[{Q['pass_iv'][0]:.0f}-{Q['pass_iv'][1]:.0f}]; passes memory-free Bayes: crossing "
          f"{gap.crossing(Q['x_trained'], Q['acc_trained'][0] - Q['stepwise'][0], 0.):.0f}")
    for r in RUNS:
        x, acc, cp = Q[f"x_{r}"], Q[f"acc_{r}"][0], Q[f"copy_{r}"][0]
        print(f"  {NAME[r]}: end {acc[-1]:.3f}; supplement: copying naive's errors max {cp.max():.3f} at "
              f"{x[cp.argmax()]:.0f}, end {cp[-1]:.3f} (memory-free rule {Q['sw_copy'][0]:.3f})")
    print("== c  excess log loss at T by FR_T fifth")
    print("  fifth edges " + " / ".join(f"{v:.3f}" for v in Q["edges5"]) + "; means " +
          " / ".join(f"{v:.3f}" for v in Q["fr5"]) + f"; top share of naive's excess at T {Q['fr5'][4] / Q['fr5'].sum():.3f}"
          f" (summed over steps {Q['naive_sum5'][4] / Q['naive_sum5'].sum():.3f})")
    print("  accuracy by fifth: naive " + " / ".join(f"{v:.3f}" for v in Q["naive_acc5"]) + "; Bayes " +
          " / ".join(f"{v:.3f}" for v in Q["bayes_acc5"]))
    print("  training objective (SANITY, power .5, goal only, all steps) by the goal's fifth: share of the summed "
          "objective / mean per item")
    print("    naive " + " / ".join(f"{v:.3f}" for v in Q["obj_naive_share"]) + "  |  " +
          " / ".join(f"{v:.4f}" for v in Q["obj_naive_mean"]))
    for r in RUNS:
        x = Q[f"x_{r}"]
        for xv in (40, 480, 880, 1280, 2000, 39996):
            i = _at(x, xv)
            print(f"    {NAME[r]:11s} {x[i]:6.0f}: " + " / ".join(f"{v:.3f}" for v in Q[f"obj_share_{r}"][i]) + "  |  " +
                  " / ".join(f"{v:.4f}" for v in Q[f"obj_mean_{r}"][i]))
    for r in RUNS:
        x, kl = Q[f"x_{r}"], Q[f"kl5_{r}"]
        print(f"  {NAME[r]}: KL by fifth [5-95%] / accuracy by fifth / accuracy - naive's")
        for xv in (40, 120, 480, 640, 880, 1040, 1280, 1520, 2000, 2400, 5000, 39996):
            i = _at(x, xv)
            print(f"    {x[i]:6.0f}: " + " ".join(f"{kl[0, i, f]:.3f}" for f in range(5)) +
                  f" [top {np.percentile(kl[1:, i, 4], 5):.2f}-{np.percentile(kl[1:, i, 4], 95):.2f}] | " +
                  " ".join(f"{v:.3f}" for v in Q[f"acc5_{r}"][i]) + " | " +
                  " ".join(f"{v:+.3f}" for v in Q[f"acc5_{r}"][i] - Q["naive_acc5"]))
        for f in range(5):
            below = kl[0, :, f] < Q["fr5"][f]
            j = next((j for j in range(len(x)) if below[j:].all()), None)
            print(f"    fifth {f + 1}: KL below naive's (= FR) for good from " +
                  (f"{x[j]:,.0f}" if j is not None else f"never (end / naive {kl[0, -1, f] / Q['fr5'][f]:.2f}x)"))
    x, kl = Q["x_echo"], Q["kl5_echo"][0, :, 4]
    m = x > X0
    i = np.flatnonzero(m)[np.argmin(kl[m])]
    j = np.flatnonzero((x >= 640) & (x <= 4000))[np.argmin(kl[(x >= 640) & (x <= 4000)])]
    k = np.flatnonzero((x >= 200) & (x <= 1500))[np.argmax(kl[(x >= 200) & (x <= 1500)])]
    print(f"  ESN top fifth: initial {kl[0]:.2f}, 20 {kl[1]:.2f}; lowest after initialization {kl[i]:.2f} at {x[i]:.0f};"
          f" highest within 200-1,500 {kl[k]:.2f} at {x[k]:.0f}; lowest within 640-4,000 {kl[j]:.2f} at {x[j]:.0f}; "
          f"1,280 {kl[_at(x, 1280)]:.2f}; end {kl[-1]:.2f}")
    # the ESN's top-fifth excess against naive's per item at the end (N3)
    P = np.load(os.path.join(AN, "probes.npz"))
    lj, ln = (log_softmax(P[k][:, -1].astype(np.float64), -1) for k in ("log_jb", "log_nb"))
    fr = (np.exp(lj) * (lj - ln)).sum(-1)
    top = fr >= Q["edges5"][4]
    lq = log_softmax(np.load(sorted(glob.glob(os.path.join(AN, "echo", "ck_*.npz")))[-1])["logq"][:, -1]
                     .astype(np.float64), -1)
    kle = (np.exp(lj) * (lj - lq)).sum(-1)
    d = kle[top] - fr[top]
    print(f"  ESN end, top fifth per item: mean {kle[top].mean():.2f} vs naive {fr[top].mean():.2f}; median "
          f"{np.median(kle[top]):.2f} vs {np.median(fr[top]):.2f}; ESN above naive on {np.mean(d > 0):.3f} of items; "
          f"90th pct {np.percentile(kle[top], 90):.1f} vs {np.percentile(fr[top], 90):.1f}, 99th "
          f"{np.percentile(kle[top], 99):.1f} vs {np.percentile(fr[top], 99):.1f}")
    print("== d  learning blocked at FR-ranked steps (from scratch, 50% of each step's goal increments detached)")
    for sfx, nm in (("", "both variables"), ("g", "goal only")):
        print(f"  -- {nm}; references by fifth: Bayes " + " / ".join(f"{v:.3f}" for v in Bk[f"bayes{sfx}"][0]) +
              "; naive " + " / ".join(f"{v:.3f}" for v in Bk[f"naive{sfx}"][0]))
        c = Bk[f"acc{sfx}_ctrl_s0"]
        xc = Bk["x_ctrl_s0"]
        for xv in (480, 880, 1280, 2400):
            i = _at(xc, xv)
            print(f"    {xc[i]:5.0f}: control " + " / ".join(f"{v:.3f}" for v in c[0, i]))
            for run, lab, _ in BLOCKS[1:]:
                a = Bk[f"acc{sfx}_{run}"]
                assert np.array_equal(Bk[f"x_{run}"], xc)
                dlt = a[:, i] - c[:, i]
                print(f"           {lab:15s} - control: " + "  ".join(_iv(dlt[:, f]) for f in (0, 4)) +
                      " (bottom, top fifth); all fifths " + " ".join(f"{v:+.3f}" for v in dlt[0]) +
                      f"; top fifth level {a[0, i, 4]:.3f}")
        hi, lo = Bk[f"acc{sfx}_det_hi_fr_f50_s0"], Bk[f"acc{sfx}_det_lo_fr_f50_s0"]
        dd = (hi[..., 4] - lo[..., 4]) - (hi[..., 0] - lo[..., 0])
        v, l_, h_ = band(dd)
        pos = xc > X0
        print(f"    dd_FR = (highest - lowest)_top - (highest - lowest)_bottom: " + ", ".join(
            f"{xc[_at(xc, xv)]:,.0f} {_iv(dd[:, _at(xc, xv)])}" for xv in (80, 480, 880, 1280, 2400)) +
            f"; interval below 0 at {(h_[pos] < 0).sum()} of {pos.sum()}; peak {v.min():+.3f} at {xc[v.argmin()]:,.0f}")
        he, le = Bk[f"acc{sfx}_det_hi_f50_s0"], Bk[f"acc{sfx}_det_lo_f50_s0"]
        dde = (he[..., 4] - le[..., 4]) - (he[..., 0] - le[..., 0])
        print(f"    entanglement-ranked halves, same contrast: " + ", ".join(
            f"{xc[_at(xc, xv)]:,.0f} {_iv(dde[:, _at(xc, xv)])}" for xv in (880, 1280, 2400)))
        r_ = Bk[f"acc{sfx}_det_rand_f50_s0"]
        print("    against the random half at 2,400: highest-FR top fifth "
              f"{_iv(hi[:, -1, 4] - r_[:, -1, 4])}, lowest-FR bottom fifth {_iv(lo[:, -1, 0] - r_[:, -1, 0])}, "
              f"lowest-FR top fifth {_iv(lo[:, -1, 4] - r_[:, -1, 4])}, highest-FR bottom fifth "
              f"{_iv(hi[:, -1, 0] - r_[:, -1, 0])}")
    # what the per-step rankings select, on the probes as a stand-in for a training batch (C3)
    ar, goal = np.arange(N_PROBE), P["goal"]
    lgj = log_softmax(P["log_jb"].astype(np.float64), -1)[ar, :, goal]
    lgn = log_softmax(P["log_nb"].astype(np.float64), -1)[ar, :, goal]
    frt = (np.exp(lgj) * (lgj - lgn)).sum(-1)                                 # per-step FR_t of the goal (n, T)
    ent = P["ent"][ar, :, goal]
    fg = np.clip(np.searchsorted(Q["edges5"], fr[ar, goal], side="right") - 1, 0, 4)
    for nm, sc in (("per-step FR", frt), ("entanglement", ent)):
        hi_ = sc > np.median(sc, 0, keepdims=True)
        sh = [hi_[fg == k].mean() for k in range(5)]
        print(f"  {nm:12s} top-50% block: share of a trajectory's steps blocked by the goal's FR_T fifth " +
              " / ".join(f"{v:.3f}" for v in sh))
    print("  world-2 replicate (own tests, goal accuracy at T; mean of the tests in the last 100 raw episodes):")
    for run, z in replicate_numbers().items():
        last = z["raw"] >= z["raw"].max() - 100
        at = lambda rv: z["acc"][np.abs(z["raw"] - rv) <= 50].mean()       # noqa: E731
        print(f"    {run:22s} {'done' if z['done'] else 'RUNNING'} at raw {z['raw'].max()}: {z['acc'][last].mean():.3f}; "
              f"around raw 1,800 (1,440 training) {at(1800):.3f}; around raw 1,100 (880) {at(1100):.3f}")
    print("== supplement: interaction-direction lesion (no longer drawn; C9: random clamps' own cost by third)")
    print(f"  goal FR_T thirds edges {np.round(Ls['edges3'], 3)}")
    for j, nm in enumerate(("low", "middle", "high")):
        v, lo, hi = band(Ls["loss"][..., j])
        rv = Ls["rand"][0, :, j]
        print(f"  {nm:6s}: " + ", ".join(f"{x:,.0f} {a:+.1f}" for x, a in zip(Ls["x"], v)) + f"; max {v.max():+.1f};"
              f" random clamps' own cost at 2,400 / end {rv[_at(Ls['x'], 2400)]:+.1f} / {rv[-1]:+.1f} pp")


def G_tri_fr(Gz):
    """(50, 162) mean FR_T of each grid triangle's three cells, per environment"""
    return Gz["pair_env_fr"][:, sd.TRI].mean(-1)


def _rebuild_arg(argv):
    """--rebuild: every cache; --rebuild=probes,blocks: the named ones"""
    for a in argv:
        if a == "--rebuild":
            return True
        if a.startswith("--rebuild="):
            return tuple(a.split("=", 1)[1].split(","))
    return ()


if __name__ == "__main__":
    rb = _rebuild_arg(sys.argv)
    out = figure(rb)
    if "--numbers" in sys.argv:
        numbers(*out, rebuild=rb)
