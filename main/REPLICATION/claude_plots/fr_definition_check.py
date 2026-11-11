"""fr_definition_check.png: which KL has "factorization regret" (FR) been, in the paper and in this repository, and does
the choice matter?
John 11-10: "Can you tell whether we have been using a symmetric or asymmetric KL? We want to be consistent everywhere."
Then: "are you sure the figures in the paper use the forward KL? Can you please try to regenerate light figures in the
paper with the code here ... The probability clipping is fine ... we can just make a note that the saturation is due to
probability clipping, if you can confirm that this is true."

The paper (Sec. 4.1, B.1) defines FR_t = KL(Joint_t || Naive_t) of the goal variable's marginal belief: the FORWARD KL.

STATE SINCE 11-10-2026 (John: "correct the codebase to calculate factorization regret consistently everywhere"):
one function, main.utils.factorization_regret(joint, naive, logs=False, clip=None) = KL(Joint || Naive), exact.
  env.SII (bayes_inference.log_outcomes)       forward exact; logs of runs made before 11-10 are of the old quantity
  probes.npz 'fr', both scan_meta.npz 'FR',    forward exact (rewritten 11-10; the old values kept as 'fr_sym' /
  round7_jumps/fr_both.npz                     'FR_sym' / fr_both_sym_superseded.npz); builders: td_analysis.bayes_side,
                                               plot_fig1c_all
  Fig. 2b (Collector)                          the same function with clip=1e-15, the printed figure's probability clip
                                               (numbers identical to before); the caption notes that the saturation at
                                               long times is the clip
  the learning-dynamics figure (Fig. 13 / 15)  supplementary_plotter2.diagnostics_runs: Pearson r with the forward FR
                                               replayed at the standard runs' 114 checkpoints (this module's fr_correct
                                               cache); fig_learning_dynamics_forward.png; fig1.py d-f the same
  calibration by FR (draft Fig. 11b)           supplementary_plotter2.harvest_calibration (calibration_ctx<C>.npz)
  the manifold figure, panels c-d              fig_manifold_fr.py
  already forward before 11-10                 fr_theory.py; fig_learn_profile.forward_fr (fig_learn_profile,
                                               fig_learn_holdout); train_learn_masks.goal_fr; fig_learn_fr;
                                               fig_learn_grids; td_tt_order.refs()["FR"]; bayes_gen_fr_div; rcc_pairs;
                                               train_cluster_runs
  converted by re-reading the rewritten data   td_summary_data.fr_acc (fig_td_summary c, fig_main_1_two_waves b),
  (caches rebuilt, old ones *_sym_superseded)  td_xcorr_across (fig_learn_maps rows 5-6, fig_supp_learning's credit maps,
                                               fig_learn_lever f, g), fig_learn_lever e, the td_* exploration figures,
                                               the fig1c family
  NOT convertible                              anything drawn from the SII score / coefficient a run LOGGED before
                                               11-10 (td_atlas/test_sii.png; supplementary_plotter2.fr_correlation and
                                               plot_fr_panel_e_compare, both kept for the record only)
Symmetric by design, and said so in the paper: the Classifier loss L_FR, the Generator loss, entanglement (Jeffreys).

WHAT IT WAS UNTIL THEN (the audit this figure documents). Three things carried the name:
  forward, exact (log space)         the "already forward" list above
  forward, probabilities clipped to  Collector.DKL -> the paper's Fig. 2b, bottom right (collection_plotters.plot_perf)
  [1e-15, 1 - 1e-15]                 a term p ln(p / q) cannot exceed ln 1e15 = 34.5 nats
  symmetric, [KL(J||N) + KL(N||J)]/2 env.SII (unchanged from 2026-03-09 to 11-10) -> the logged logistic slope / score
  with the same clip                 -> the submission's Fig. 13d-f and the 11-08 draft's implied r; the supplementary
                                     harvest's 'fr'; probes.npz 'fr' (td_analysis.fr_sym) and everything that read it;
                                     plot_fig1c_all.fr_sym / the scan's FR / fr_both -> the fig1c family; the paper's
                                     manifold figure, panels c-d (summed over the two variables)

Panels
  a-c  The paper's Fig. 2b regret curve, C = 2, 3, 4. Dots: the curve AS PRINTED, read from the submission PDF's vector
       graphics (build_published; every 8th step shown). Lines: the paper's own code path at reduced size
       (Sanity_Collector.collect_bayes, 4,000 episodes per C): its joint_naive_DKL = KL(Joint || Naive) (solid, what
       plot_perf draws), naive_joint_DKL (dotted) and their mean, the symmetric form (dashed); all clipped at 1e-15.
  d    The same forward KL on the environment's own episodes (C = 2) with the clip moved from 1e-15 toward 0, and the
       exact value (log space; dashed black); dots: the printed curve.
  e    The submission's Fig. 13d. Dots: as printed (PDF vector graphics, every 40th test). Solid: each test's logged
       logistic slope of correctness at T on env.SII (the paper's logs). Dotted: the Pearson r that slope implies
       (supplementary_plotter2.fr_correlation, recomputed: the revised figure's version).
  f    The Pearson r between FR_T and correctness at T through training with the forward exact KL (solid) and the
       symmetric clipped one (dashed): the standard runs' 114 checkpoints replayed on the 10,000 probes (goal variable);
       dotted: the two observers' own r (forward).
  g    One point per probe episode at T = 30 (goal variable): the symmetric clipped value against the exact forward one.

RESULTS (11-10, --numbers)
  a-c  printed value | the code's KL(Joint || Naive) | symmetric | KL(Naive || Joint), at t = 100:
         C = 2    6.125 |  5.99 |  7.35 |  8.70
         C = 3   13.121 | 12.97 | 11.90 | 10.82
         C = 4   14.696 | 14.63 | 10.83 |  7.02
       (4,000 episodes per C; with 40,000 / 20,000 / 10,000 of the environment's own episodes the clipped forward KL is
       6.02 / 13.14 / 14.75; printed at t = 30: 2.880 / 5.777 / 6.638, code 2.83 / 5.78 / 6.62). The printed panel is
       the forward KL; the symmetric form would even put C = 4 below C = 3.
  d    C = 2, t = 100 (20,000 episodes): clip 1e-15 6.03, 1e-30 7.92, 1e-60 9.11, 1e-120 9.33, none 9.33; C = 3
       (10,000): 13.18, 17.87, 20.87, 21.71, 21.82; C = 4 (3,000): 15.27, 21.63, 26.13, 27.39, 27.43. The saturation of
       Fig. 2b is the clip: with it at 1e-15 the code reproduces the printed curve, and as it goes the curve becomes
       the straight exact line (exact slope over steps 60-100: C = 2 .090 nats per step, the theory's E[Delta] = .089
       at lambda = 2; C = 3 .221; C = 4 .276; clipped: .036, .083, .090). 8.5% / 20.1% / 24.7% of episodes are past 34
       nats at t = 100; at t = 30 the clip costs 5 / 8 / 10% of the mean. Exact means at t = 100 (40,000 / 20,000 /
       10,000 episodes): 9.31 / 21.81 / 26.57. Interaction sets that repeat a variable (drawn with replacement from the
       50 test variables): 2.0% / 5.7% / 11.5% for C = 2 / 3 / 4.
  e    printed vs logged slope over tests 1-800: mean |difference| 7e-5 (Trained), 2e-4 (Echo), max 1.5e-3: the printed
       "r (Correlation)" is the logistic slope (log-odds of a correct answer per nat of SII), not a correlation, and its
       regret is the symmetric KL. The implied Pearson r differs from it by .04-.06 on average (recomputed r equals
       fig1's cache exactly).
  f    r(FR_T, correct_T) at the end of training (49,995 raw episodes), forward exact | forward clipped | symmetric:
         Trained RNN  -.058 | -.055 | -.028        Joint Bayes  -.061 | -.056 | -.030
         ESN          -.265 | -.298 | -.324        Naive Bayes  -.322 | -.367 | -.433
       Same shape under all: the correlation appears within ~100 episodes in both networks, keeps deepening in the ESN
       and recedes in the Trained RNN from ~800 episodes. With the forward KL the Trained RNN settles at Joint Bayes's
       own level (-.06), not at zero: "dissolves" in the Fig. 13 text holds only for the symmetric one.
  g    10,000 probes: Spearman .978, Pearson .82; the top fifth by one definition is 87% the top fifth by the other.
       Means at T: forward 2.97 (2.75 with the clip), reverse 4.43, symmetric clipped 3.43; the symmetric value cannot
       pass 34.5 (maximum 34.4; the forward maximum is 94.9). 60% of the symmetric mean is the reverse term.
  Environment check (--numbers, last block): E[FR_t] of fr_theory.py's simulation (its cache, lambda = 2, 2,000
       interaction sets x 50 episodes) against the environment's own episode loop (C = 2, 40,000 episodes): z = -0.3 to
       +0.7 at t = 1, 5, 10, 30, 50, 100 (e.g. t = 100: 9.26 +- .14 against 9.31 +- .12).
  Entanglement (John: "fine as is ... unless you think we would get better signals ... with the forward KL"): no.
       fig1c_entanglement_definitions.scores(), within-stream r at lag 0 with seven network measures (39 environments x
       5,000 streams, Trained RNN): the Jeffreys average, KL(Naive || Optimal) and KL(Optimal || Naive) agree within
       .015 on every measure, late and early window, raw and beyond update size (late raw, activity norm: +.368 /
       +.363 / +.363; credit coherence +.335 / +.340 / +.321).

  Calibration by FR (draft Fig. 11b, figure_bin/calibration_by_tier.png; accuracy / confidence per FR bin, C = 2 | 3):
       same networks as before (C = 3 accuracy Trained .621 / ESN .257, was .618 / .255), forward FR, bins to 64 nats:
         Joint .99-1.01 and Trained .95-.99 in every bin (unchanged: calibrated at every FR).
         Naive C=2  1.05 .67 .23 .10 .05 .04 .03 .02    C=3  .96 .63 .36 .23 .13 .07 .04 .02
         ESN   C=2   .83 .60 .44 .29 .18 .10 .06 .02    C=3  .77 .61 .53 .44 .30 .18 .09 .05
       symmetric (bins to 40): Naive C=2 1.01 .88 .56 .26 .11 .03 .00 | C=3 .91 .65 .44 .24 .09 .03 .00;
                               ESN   C=2  .83 .67 .54 .39 .25 .16 .08 | C=3 .77 .60 .52 .41 .28 .16 .08.
       The C = 2 and C = 3 rows pair up less well with the forward FR: from 1 nat on, the C = 3 row is the C = 2 row
       moved one bin right (a factor 2 in FR); ESN mean |C=2 - C=3| .071 (symmetric .029), largest .15 at 2-4 nats.
       40,000 trajectories per network at C = 3 (CPU; the symmetric file had 180,000), 250,000 at C = 2.
Data: DATA/experimental/fr_definition/: fig2b_clip (~12 min CPU, 8 threads; C = 4 is most of it), clip_sweep (~5 min),
collector (~2 min), fig13_code (~8 min, CPU), per_episode and fr_correct (seconds), published (--pdf PATH: needs the
submission PDF and poppler's pdftocairo).
usage:  python fr_definition_check.py [--data] [--pdf SUBMISSION.pdf] [--numbers]"""
import os
import sys
import glob
import io
import contextlib
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from scipy.special import logsumexp, log_softmax                                  # noqa: E402
from scipy.stats import spearmanr                                                 # noqa: E402
from main.utils import fig_path                                                   # noqa: E402
import fig_main_common as mc                                                      # noqa: E402

TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
AN = os.path.join(TD, "analysis")
SEED = 20261029                                                                   # the standard runs replayed in f
OUT = os.path.join(ROOT, "main", "DATA", "experimental", "fr_definition")
THEORY = os.path.join(ROOT, "main", "DATA", "experimental", "fr_theory", "fr_theory_cache", "derived.pkl")
T_LONG, R, CLIP = 100, 10, 1e-15
LOOP = {2: (400, 100), 3: (200, 100), 4: (100, 100)}                              # C: (fresh worlds, episodes per world)
SWEEP = {2: 200, 3: 100, 4: 30}                                                   # C: fresh worlds (x 100 episodes) of the clip sweep
SWEEP_EPS = (1e-15, 1e-30, 1e-60, 1e-120, 1e-300)
RUNS = ("trained", "echo")
C_COL = {2: "#3b528b", 3: "#21918c", 4: "#5ec962"}                                # Fig. 2b's viridis order


def kl_clipped(x, y, eps=CLIP):
    """Collector.DKL / env.DKL without the batch mean: probabilities clipped to [eps, 1 - eps]"""
    x, y = np.clip(x, eps, 1 - eps), np.clip(y, eps, 1 - eps)
    return (x * np.log(x / y)).sum(-1)


def kl_exact(lp, lq):
    """KL(p || q) from log beliefs"""
    return (np.exp(lp) * (lp - lq)).sum(-1)


def log_goal_beliefs(L, obs, goal):
    """the two observers' log beliefs over the goal variable at every step, in log space: L (B, O, R, ..., R) with one
    axis per variable, obs (B, T, O), goal (B,) -> Joint (B, T, R), Naive (B, T, R)"""
    B, O = L.shape[:2]
    C = L.ndim - 2
    flat = L.reshape(B, O, -1)
    ll = (np.einsum("bto,bok->btk", obs, np.log(flat)) + np.einsum("bto,bok->btk", 1 - obs, np.log1p(-flat))).cumsum(1)
    ll = ll.reshape((B, obs.shape[1]) + (R,) * C)
    lj, ln = np.zeros((B, obs.shape[1], R)), np.zeros((B, obs.shape[1], R))
    for c in range(C):
        b = np.flatnonzero(goal == c)
        if not len(b):
            continue
        other = tuple(a for a in range(C) if a != c)
        lj[b] = log_softmax(logsumexp(ll[b], axis=tuple(2 + a for a in other)), axis=-1)
        m = L[b].mean(axis=tuple(2 + a for a in other))                           # the goal's marginal likelihood (b, O, R)
        ln[b] = log_softmax((np.einsum("bto,bor->btr", obs[b], np.log(m))
                             + np.einsum("bto,bor->btr", 1 - obs[b], np.log1p(-m))).cumsum(1), axis=-1)
    return lj, ln


def build_fig2b():
    """the environment's own episode loop, per C: the exact forward KL (= env.SII since 11-10), the clipped forward KL
    (Fig. 2b's) and the symmetric clipped KL that env.SII was until 11-10 ("sii")"""
    from main.CognitiveGridworld import CognitiveGridworld
    out = {}
    for C, (W, B) in LOOP.items():
        acc = {k: [] for k in ("exact", "clip", "sii", "capped")}
        chk, same = 0., []
        for w in range(W):
            np.random.seed(50_000 + 1000 * C + w)
            with contextlib.redirect_stdout(io.StringIO()):
                a = CognitiveGridworld(mode=None, cuda=0, ctx_num=C, state_num=500, hid_dim=10, load_env=False,
                                       training=False, batch_num=B, obs_num=5, show_plots=False, step_num=T_LONG,
                                       episodes=1, realization_num=R, learn_embeddings=False)
            J, N = np.asarray(a.joint_goal_belief, np.float64), np.asarray(a.naive_goal_belief, np.float64)
            lj, ln = log_goal_beliefs(np.asarray(a.joint_likelihood, np.float64), np.asarray(a.obs_flat, np.float64),
                                      np.asarray(a.goal_ind).astype(int))
            chk = max(chk, np.abs(np.exp(lj) - J).max(), np.abs(np.exp(ln) - N).max())
            fr = kl_exact(lj, ln)
            acc["exact"].append(fr.mean(0))
            acc["clip"].append(kl_clipped(J, N).mean(0))
            acc["sii"].append(((kl_clipped(J, N) + kl_clipped(N, J)) / 2).mean(0))   # what env.SII was until 11-10 (symmetric, clipped)
            assert np.abs(np.asarray(a.SII, np.float64) - fr).max() < 1e-8          # env.SII is the forward exact KL since then
            acc["capped"].append((fr > 34).mean(0))
            ci = np.asarray(a.ctx_inds)
            same.append((np.sort(ci, 1)[:, 1:] == np.sort(ci, 1)[:, :-1]).any(1).mean())
            del a
        for k, v in acc.items():
            v = np.array(v)
            out[f"C{C}_{k}"], out[f"C{C}_{k}_se"] = v.mean(0), v.std(0) / np.sqrt(W)
        out[f"C{C}_n"], out[f"C{C}_chk"], out[f"C{C}_same"] = W * B, chk, np.mean(same)
        print(f"C = {C}: {W} worlds x {B} episodes; log-space beliefs equal the environment's to {chk:.1e}; "
              f"repeated variable in {100 * np.mean(same):.2f}% of interaction sets", flush=True)
    return out


def build_clip_sweep():
    """the code's forward KL (kl_clipped) with the clip moved from 1e-15 toward 0, on the environment's own episodes: if
    the bend of Fig. 2b is the clip, the curve must straighten onto the exact one as the clip goes"""
    from main.CognitiveGridworld import CognitiveGridworld
    out = {"eps": np.array(SWEEP_EPS)}
    for C, W in SWEEP.items():
        acc = {e: [] for e in SWEEP_EPS}
        ex = []
        for w in range(W):
            np.random.seed(70_000 + 1000 * C + w)
            with contextlib.redirect_stdout(io.StringIO()):
                a = CognitiveGridworld(mode=None, cuda=0, ctx_num=C, state_num=500, hid_dim=10, load_env=False,
                                       training=False, batch_num=100, obs_num=5, show_plots=False, step_num=T_LONG,
                                       episodes=1, realization_num=R, learn_embeddings=False)
            lj, ln = log_goal_beliefs(np.asarray(a.joint_likelihood, np.float64), np.asarray(a.obs_flat, np.float64),
                                      np.asarray(a.goal_ind).astype(int))
            pj, pn = np.exp(lj), np.exp(ln)
            for e in SWEEP_EPS:
                acc[e].append(kl_clipped(pj, pn, e).mean(0))
            ex.append(kl_exact(lj, ln).mean(0))
            del a
        out[f"C{C}"] = np.array([np.mean(acc[e], 0) for e in SWEEP_EPS])              # (clips, T)
        out[f"C{C}_exact"], out[f"C{C}_n"] = np.mean(ex, 0), W * 100
        print(f"clip sweep C = {C}: {W * 100} episodes; at t = {T_LONG}: "
              + ", ".join(f"{e:g}: {out[f'C{C}'][i, -1]:.2f}" for i, e in enumerate(SWEEP_EPS))
              + f", exact {out[f'C{C}_exact'][-1]:.2f}", flush=True)
    return out


def _svg_paths(pdf, page):
    """the vector paths of one page of a PDF (poppler's pdftocairo): [(style string, transform, points (n, 2))]"""
    import re
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "page.svg")
        subprocess.run(["pdftocairo", "-svg", "-f", str(page), "-l", str(page), pdf, f], check=True)
        s = open(f).read()
    out = []
    for p in re.findall(r"<path\b[^>]*>", s):
        dd = re.search(r'\bd="([^"]+)"', p)
        if dd:
            pts = np.array([[float(a), float(b)] for a, b in re.findall(r"[ML]\s*([-\d.]+)\s+([-\d.]+)", dd.group(1))])
            tr = re.search(r'transform="([^"]+)"', p)
            out.append((p, tr.group(1) if tr else "", pts))
    return out


def build_published(pdf=None):
    """the curves the submission PRINTS, read from the PDF's vector graphics (no pixels): Fig. 2b's regret panel
    (page 4; dashed lines, one per C) and Fig. 13d (page 27; Trained blue, Echo orange). Axes are calibrated on the
    tick marks: Fig. 2b y ticks 0 / 4 / 8 / 12 at 515.25 / 462.30 / 409.35 / 356.40 and x ticks 0 .. 100 at 948.28 ..
    1153.91 (the curve's first point, t = 1 observation, sits on the 0 tick: plot_perf draws np.arange(T)); Fig. 13d
    y ticks 0 .. -.25 at 351.57 .. 518.92 and x ticks 0 .. 800 at 64.06 .. 257.70"""
    assert pdf and os.path.exists(pdf), "build_published needs the submission PDF: --pdf PATH"
    out = {}
    C_RGB = {1: "27.45", 2: "16.86", 3: "14.50", 4: "60.78"}                           # viridis(.15 .. .85), red channel %
    for style, _, pts in _svg_paths(pdf, 4):
        if "stroke-dasharray:5.55" in style and "stroke-width:1.5" in style:
            C = [c for c, r in C_RGB.items() if f"stroke:rgb({r}" in style]
            assert len(C) == 1, style[:200]
            out[f"fig2b_C{C[0]}_t"] = (pts[:, 0] - 948.27651) / ((1153.91 - 948.28) / 100) + 1
            out[f"fig2b_C{C[0]}"] = (515.2455 - pts[:, 1]) / ((515.25 - 356.40) / 12)
    assert all(f"fig2b_C{c}" in out for c in C_RGB), sorted(out)
    assert np.abs(out["fig2b_C1"]).max() < 1e-3 and abs(out["fig2b_C2_t"][0] - 1) < 1e-3                     # C = 1 is 0
    k = 0
    for style, _, pts in _svg_paths(pdf, 27):
        box = len(pts) > 100 and pts[:, 0].min() > 54 and pts[:, 0].max() < 268 and pts[:, 1].min() > 329 and pts[:, 1].max() < 542
        if box and "stroke-width:1.4" in style:
            name = "trained" if "stroke:rgb(12.15" in style else "echo" if "stroke:rgb(100%,49.8" in style else None
            assert name, style[:200]
            out[f"fig13d_{name}_test"] = (pts[:, 0] - 64.06) / ((257.70 - 64.06) / 800)
            out[f"fig13d_{name}"] = -(pts[:, 1] - 351.57) / ((518.92 - 351.57) / .25)
            k += 1
    assert k == 2, k
    return out


def build_collector():
    """Fig. 2b's own code path at reduced size: compare_sanity.py's Sanity_Collector.collect_bayes (batch 250 x 16
    repetitions = 4,000 episodes per C; the paper: 10,000 x 50) -> its joint_naive_DKL (what plot_perf draws as
    'Factorization Regret') and naive_joint_DKL, (4, 100)"""
    from main.REPLICATION.Collector import Sanity_Collector
    np.random.seed(2027)
    with contextlib.redirect_stdout(io.StringIO()):
        c = Sanity_Collector(0, 500, 250, R, T_LONG, 4, 5)       # cuda, state_num, batch_num, realization_num, step_num, ctx_num, obs_num
        c.collect_bayes(rep=16)
    return {"joint_naive": c.joint_naive_DKL, "naive_joint": c.naive_joint_DKL, "n": 4000}


def build_fig13_code():
    """Fig. 13d's code paths, from the stored logs (the harvest cache): slope_<run> = each test's logged logistic slope
    of correctness at T on env.SII at T (env_model_data_manager.perform_sanity_training_analyses; what the submission's
    plotter, supplementary_plotter.plot_diagnostics, drew as "CORR"); <run> = the Pearson r it implies
    (supplementary_plotter2.fr_correlation, recomputed here: the revised figures' version)"""
    import supplementary_plotter2 as sp
    tmp = os.path.join(OUT, "_fr_corr_tmp.npz")
    if os.path.exists(tmp):
        os.remove(tmp)
    r = sp.fr_correlation(cache=tmp)
    old = np.load(sp.FR_IMPLIED)
    os.remove(tmp)
    Hc = np.load(sp.HARVEST_CACHE)
    return {"trained": r["Trained"], "echo": r["Echo"],
            "max_diff_from_cache": max(np.abs(r["Trained"] - old["trained"]).max(), np.abs(r["Echo"] - old["echo"]).max()),
            "slope_trained": Hc["tr_test_SII_coef_through_training"].astype(np.float64),   # what the runs logged
            "slope_echo": Hc["ec_test_SII_coef_through_training"].astype(np.float64)}


def build_per_episode():
    """panel b: every definition on the probe set, goal variable, all steps (n, 30)"""
    P = np.load(os.path.join(AN, "probes.npz"))
    g, i = P["goal"], np.arange(len(P["goal"]))
    lj = log_softmax(P["log_jb"].astype(np.float64), -1)[i, :, g]
    ln = log_softmax(P["log_nb"].astype(np.float64), -1)[i, :, g]
    pj, pn = np.exp(lj), np.exp(ln)
    out = dict(fwd=kl_exact(lj, ln), rev=kl_exact(ln, lj), fwd_clip=kl_clipped(pj, pn), rev_clip=kl_clipped(pn, pj),
               right_bayes=P["right_bayes"][i, g], right_naive=P["right_naive"][i, g])
    out["sym_clip"] = (out["fwd_clip"] + out["rev_clip"]) / 2
    assert np.abs(out["sym_clip"] - P["fr_sym"][i, :, g]).max() < 1e-5            # what probes stored as 'fr' until 11-10
    assert np.abs(out["fwd"] - P["fr"][i, :, g]).max() < 1e-4                     # probes 'fr' is the forward KL since then
    return out


def build_fr_correct():
    """panel c: r(FR_T, correct_T) per checkpoint, each definition"""
    E = data("per_episode")
    P = np.load(os.path.join(AN, "probes.npz"))
    g, i = P["goal"], np.arange(len(P["goal"]))
    defs = {k: E[k][:, -1] for k in ("fwd", "fwd_clip", "sym_clip", "rev")}
    out = {}
    for run in RUNS:
        fs = sorted(glob.glob(os.path.join(AN, run, "ck_*.npz")))
        ep, acc, r = [], [], {k: [] for k in defs}
        for f in fs:
            z = np.load(f)
            right = z["right"][i, -1, g].astype(np.float64)
            ep.append(int(z["episode"]))
            acc.append(right.mean())
            for k, v in defs.items():
                r[k].append(np.corrcoef(v, right)[0, 1])
        out[f"{run}_x"] = np.array(ep) - np.array(ep) // 5                         # training episodes (every 5th is a test)
        out[f"{run}_raw"], out[f"{run}_acc"] = np.array(ep), np.array(acc)
        A = [np.load(f) for f in sorted(glob.glob(os.path.join(TD, f"{run}_s{SEED}", "test_*.npz")))]   # the run's own tests
        te = np.concatenate([a["episode"] for a in A])                             # (8,000 episodes each, every 5th episode)
        it = np.searchsorted(te, np.array(ep))
        assert np.array_equal(te[it], np.array(ep)), run
        pr = lambda e: e.sum(-1) ** 2 / ((e * e).sum(-1) + 1e-12)                    # noqa: E731  (supplementary_plotter2.pr_stat)
        ud, idim = (np.concatenate([a[k] for a in A])[it].astype(np.float64) for k in ("update_dim", "input_dim"))
        out[f"{run}_pr_gain"] = pr(ud) - pr(idim)
        out[f"{run}_test_acc"] = np.concatenate([a["acc"] for a in A])[it, -1].astype(np.float64)
        for k in defs:
            out[f"{run}_{k}"] = np.array(r[k])
    for k, v in defs.items():
        out[f"bayes_{k}"] = np.corrcoef(v, E["right_bayes"])[0, 1]
        out[f"naive_{k}"] = np.corrcoef(v, E["right_naive"])[0, 1]
    out["acc_bayes"], out["acc_naive"] = E["right_bayes"].mean(), E["right_naive"].mean()
    return out


BUILD = {"fig2b_clip": build_fig2b, "per_episode": build_per_episode, "fr_correct": build_fr_correct,
         "clip_sweep": build_clip_sweep, "collector": build_collector, "fig13_code": build_fig13_code}


def data(name, fresh=False):
    path = os.path.join(OUT, name + ".npz")
    if os.path.exists(path) and not fresh:
        return dict(np.load(path))
    out = BUILD[name]()
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, **out)
    return out


def a_which(axs, K, Pb):
    """Fig. 2b's regret panel: the printed curve against the three KLs the code could have drawn"""
    t = np.arange(1, T_LONG + 1)
    for ax, C, letter in zip(axs, (2, 3, 4), "abc"):
        f, r = K["joint_naive"][C - 1], K["naive_joint"][C - 1]
        ax.plot(t, r, color="0.55", lw=1.6, ls=(0, (1, 1.4)))
        ax.plot(t, (f + r) / 2, color="0.55", lw=1.6, ls=(0, (3, 1.6)))
        ax.plot(t, f, color=C_COL[C], lw=2.2)
        pt, pv = Pb[f"fig2b_C{C}_t"], Pb[f"fig2b_C{C}"]
        k = np.unique(np.searchsorted(pt, np.arange(4, 101, 8)).clip(0, len(pt) - 1))
        ax.plot(pt[k], pv[k], "o", ms=4.2, mfc="k", mec="w", mew=.6, zorder=6)
        ax.set_xlim(0, T_LONG + 3)
        ax.set_ylim(0, 16.5)
        ax.set_xlabel("inference step $t$", fontsize=mc.FL)
        ax.spines[["top", "right"]].set_visible(False)
        if C == 2:
            ax.set_ylabel("nats", fontsize=mc.FL)
            ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=4.2, mfc="k", mec="w", label="as printed (the PDF)"),
                               Line2D([], [], color="0.2", lw=2.2, label=r"KL(Joint$\,\Vert\,$Naive)"),
                               Line2D([], [], color="0.55", lw=1.6, ls=(0, (3, 1.6)), label="symmetric"),
                               Line2D([], [], color="0.55", lw=1.6, ls=(0, (1, 1.4)), label=r"KL(Naive$\,\Vert\,$Joint)")],
                      fontsize=mc.FT - .5, frameon=False, loc="upper left", handlelength=2.1, borderaxespad=.1,
                      labelspacing=.25)
        else:
            ax.tick_params(labelleft=False)
        ax.set_yticks([0, 5, 10, 15])
        mc.title(ax, letter, f"Fig. 2b's regret, $C={C}$", "dots: printed; lines: the code" if C == 2 else " ")


def d_sweep(ax, S, Pb, C=2):
    """the clip moved toward 0: the printed curve is the 1e-15 one, and the bend goes with the clip"""
    t = np.arange(1, T_LONG + 1)
    ramp = ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    for i, (e, c, dy) in enumerate(zip(S["eps"][:3], ramp[:3], (0, 0, -.45))):
        ax.plot(t, S[f"C{C}"][i], color=c, lw=2)
        ax.text(T_LONG + 2, S[f"C{C}"][i, -1] + dy, "$10^{%d}$" % int(np.log10(e)), color=c, fontsize=mc.FT, va="center")
    ax.plot(t, S[f"C{C}_exact"], color="k", lw=1.2, ls=(0, (4, 2)))
    ax.text(T_LONG + 2, S[f"C{C}_exact"][-1] + .55, "no clip", color="k", fontsize=mc.FT, va="center")
    pt, pv = Pb[f"fig2b_C{C}_t"], Pb[f"fig2b_C{C}"]
    k = np.unique(np.searchsorted(pt, np.arange(4, 101, 8)).clip(0, len(pt) - 1))
    ax.plot(pt[k], pv[k], "o", ms=4.2, mfc="k", mec="w", mew=.6, zorder=6)
    ax.set_xlim(0, T_LONG + 3)
    ax.set_ylim(0, 10.6)
    ax.set_xlabel("inference step $t$", fontsize=mc.FL)
    ax.spines[["top", "right"]].set_visible(False)
    mc.title(ax, "d", "The clip makes the bend", r"KL(Joint$\,\Vert\,$Naive), clipped at $\epsilon$")


def e_fig13(ax, F, Pb):
    """Fig. 13d: the printed curves against what the runs logged (the logistic slope on env.SII, the symmetric KL) and
    against the Pearson r that slope implies"""
    x = np.arange(1, 801)
    for run, col in (("trained", mc.COL["trained"]), ("echo", mc.COL["echo"])):
        ax.plot(x, F[run][:800], color=col, lw=.9, ls=(0, (1, 1.2)), alpha=.9)
        ax.plot(x, F[f"slope_{run}"][:800], color=col, lw=1.1)
        pt, pv = Pb[f"fig13d_{run}_test"], Pb[f"fig13d_{run}"]
        k = np.unique(np.searchsorted(pt, np.arange(20, 801, 40)).clip(0, len(pt) - 1))
        ax.plot(pt[k], pv[k], "o", ms=4.2, mfc="k", mec="w", mew=.6, zorder=6)
    ax.axhline(0, color="0.6", lw=.8, zorder=1)
    ax.set_xlim(0, 820)
    ax.set_ylim(-.3, .03)
    ax.set_xlabel("testing episode", fontsize=mc.FL)
    ax.set_ylabel("printed value", fontsize=mc.FL)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=4.2, mfc="k", mec="w", label="as printed"),
                       Line2D([], [], color="0.2", lw=1.4, label="logged slope"),
                       Line2D([], [], color="0.2", lw=1, ls=(0, (1, 1.2)), label="implied $r$")],
              fontsize=mc.FT - .5, frameon=False, loc="lower left", handlelength=1.8, borderaxespad=.2, labelspacing=.25)
    mc.title(ax, "e", "Fig. 13d: a slope on the symmetric KL", "correctness at $T$ regressed on env.SII")


def f_corr(ax, Z):
    for run in RUNS:
        x = mc.xs(Z[f"{run}_x"])
        ax.plot(x, Z[f"{run}_fwd"], color=mc.COL[run], lw=2)
        ax.plot(x, Z[f"{run}_sym_clip"], color=mc.COL[run], lw=1.6, ls=(0, (3, 1.6)))
    for name, key, va, dy in (("Joint", "bayes_fwd", "top", -.008), ("Naive", "naive_fwd", "bottom", .006)):
        ax.axhline(float(Z[key]), color="0.2", lw=.9, ls=":", zorder=1)                # the observers' own r, forward
        ax.text(9.5, float(Z[key]) + dy, name, fontsize=mc.FT - .5, color="0.2", va=va)
    ax.text(4.5e4, .004, mc.NAME["trained"], color=mc.COL["trained"], fontsize=mc.FT, ha="right", va="bottom")
    ax.text(4.5e4, -.25, mc.NAME["echo"], color=mc.COL["echo"], fontsize=mc.FT, ha="right", va="bottom")
    mc.xaxis(ax, passing=False)
    ax.set_ylim(-.37, .035)
    ax.set_ylabel("Pearson $r$", fontsize=mc.FL)
    ax.legend(handles=[Line2D([], [], color="0.2", lw=2, label="forward, exact"),
                       Line2D([], [], color="0.2", lw=1.6, ls=(0, (3, 1.6)), label="symmetric, clipped\n(as Fig. 13d)")],
              fontsize=mc.FT, frameon=False, loc="lower left", bbox_to_anchor=(.0, .17), handlelength=2.2,
              borderaxespad=.2, labelspacing=.35)
    mc.title(ax, "f", "The same with the forward KL", r"$r(\mathrm{FR}_T,\ \mathrm{correct}_T)$, replayed runs")


def g_scatter(ax, E):
    lo, hi, cap = 1e-3, 150, np.log(1 / CLIP)
    x, y = np.clip(E["fwd"][:, -1], lo, None), np.clip(E["sym_clip"][:, -1], lo, None)
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=1, zorder=1)
    ax.axhline(cap, color="0.35", lw=1, ls=":", zorder=1)
    ax.text(lo * 1.4, cap * 1.25, "cap of the clip, 34.5", fontsize=mc.FT - .5, color="0.35", va="bottom")
    ax.scatter(x, y, s=2.2, color="#5c4bc4", alpha=.3, lw=0, zorder=3, rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100])
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1, 10, 100])
    ax.minorticks_off()
    ax.set_xlabel(r"forward, exact (nats)", fontsize=mc.FL)
    ax.set_ylabel("symmetric, clipped (nats)", fontsize=mc.FL)
    ax.spines[["top", "right"]].set_visible(False)
    rho = spearmanr(E["fwd"][:, -1], E["sym_clip"][:, -1])[0]
    ax.text(.97, .06, f"rank correlation {rho:.2f}".replace("0.", "."), transform=ax.transAxes, ha="right",
            fontsize=mc.FT, color="0.2")
    mc.title(ax, "g", "Each episode at $T=30$", r"symmetric vs forward $\mathrm{FR}_T$")


def figure(path=None):
    mc.setup()
    E, Z, S, K = data("per_episode"), data("fr_correct"), data("clip_sweep"), data("collector")
    Pb, F = dict(np.load(os.path.join(OUT, "published.npz"))), data("fig13_code")
    W, H = 13.4, 7.45
    fig, add = mc.canvas(W, H)
    pw, ph, y1, y2 = 2.42, 2.05, 4.42, .95
    xs1 = [.62, .62 + pw + .22, .62 + 2 * (pw + .22), .62 + 3 * pw + .44 + .95]
    a_which([add(x, y1, pw, ph) for x in xs1[:3]], K, Pb)
    d_sweep(add(xs1[3], y1, pw, ph), S, Pb)
    pw2 = 2.95
    xs2 = [.82, .82 + pw2 + 1.45, .82 + 2 * (pw2 + 1.45)]
    e_fig13(add(xs2[0], y2, pw2, ph), F, Pb)
    f_corr(add(xs2[1], y2, pw2, ph), Z)
    g_scatter(add(xs2[2], y2, pw2, ph), E)
    mc.note(fig, .058, f"a-c: the paper's own collector code, {int(K['n']):,} episodes per $C$ (the paper: 500,000).   "
            f"d: the environment's own episodes, {int(S['C2_n']):,}, $C=2$.   e: the paper's logs, {len(F['trained']):,} tests "
            f"x 8,000 episodes.\nf, g: {len(E['fwd']):,} probe episodes of 50 test environments, goal variable; f: "
            f"{len(Z['trained_x'])} checkpoints of the replayed standard runs.")
    path = path or fig_path("fr_definition_check.png")
    fig.savefig(path, dpi=mc.DPI)
    plt.close(fig)
    print("saved", path)
    return path


def numbers():
    import pickle
    D, E, Z = data("fig2b_clip"), data("per_episode"), data("fr_correct")
    print("a  E[FR_t], exact / clipped (share) / SII; episodes past 34 nats")
    for C in LOOP:
        for t in (10, 30, 50, 100):
            e, c = D[f"C{C}_exact"][t - 1], D[f"C{C}_clip"][t - 1]
            print(f"   C = {C}  t = {t:3d}   {e:6.2f} (s.e. {D[f'C{C}_exact_se'][t - 1]:.2f}) / {c:6.2f} ({c / e:.2f}) / "
                  f"{D[f'C{C}_sii'][t - 1]:6.2f};  {100 * D[f'C{C}_capped'][t - 1]:.1f}%")
        s = np.diff(D[f"C{C}_exact"])
        print(f"   C = {C}  slope of the exact curve, nats per step: steps 10-30 {s[9:29].mean():.3f}, 30-60 "
              f"{s[29:59].mean():.3f}, 60-100 {s[59:].mean():.3f};  clipped: {np.diff(D[f'C{C}_clip'])[59:].mean():.3f}")
    f, s, r = E["fwd"][:, -1], E["sym_clip"][:, -1], E["rev"][:, -1]
    top = lambda v: v >= np.quantile(v, .8)                                        # noqa: E731
    print(f"b  Spearman {spearmanr(f, s)[0]:.3f}, Pearson {np.corrcoef(f, s)[0, 1]:.3f}; top fifth shared "
          f"{(top(f) & top(s)).sum() / top(f).sum():.2f}; means at T: forward {f.mean():.2f} (clipped "
          f"{E['fwd_clip'][:, -1].mean():.2f}), reverse {r.mean():.2f} (clipped {E['rev_clip'][:, -1].mean():.2f}), "
          f"symmetric clipped {s.mean():.2f}; maxima {f.max():.1f} / {s.max():.1f}")
    print("c  r(FR_T, correct_T): forward | forward clipped | symmetric clipped | reverse")
    for run in RUNS:
        for j in (0, np.argmin(np.abs(Z[f"{run}_x"] - 640)), len(Z[f"{run}_x"]) - 1):
            print(f"   {run:8s} {int(Z[f'{run}_x'][j]):6d} training episodes (accuracy {Z[f'{run}_acc'][j]:.3f}):  "
                  + " | ".join(f"{Z[f'{run}_{k}'][j]:+.3f}" for k in ("fwd", "fwd_clip", "sym_clip", "rev")))
    for o in ("bayes", "naive"):
        print(f"   {o:8s}:  " + " | ".join(f"{float(Z[f'{o}_{k}']):+.3f}" for k in ("fwd", "fwd_clip", "sym_clip", "rev")))
    if os.path.exists(THEORY):
        c = pickle.load(open(THEORY, "rb"))[1]["D"]["curve"][2.0]
        print(f"environment check: fr_theory.py's simulation ({c['nZ']} interaction sets) against panel a's C = 2 loop "
              f"({100 * float(D['C2_same']):.2f}% of its interaction sets repeat a variable)")
        for t in (1, 5, 10, 30, 50, 100):
            z = (c["mean"][t - 1] - D["C2_exact"][t - 1]) / np.hypot(c["se"][t - 1], D["C2_exact_se"][t - 1])
            print(f"   t = {t:3d}   script {c['mean'][t - 1]:.3f} +- {c['se'][t - 1]:.3f}   environment "
                  f"{D['C2_exact'][t - 1]:.3f} +- {D['C2_exact_se'][t - 1]:.3f}   z = {z:+.1f}")


if __name__ == "__main__":
    if "--pdf" in sys.argv:
        os.makedirs(OUT, exist_ok=True)
        np.savez(os.path.join(OUT, "published.npz"), **build_published(sys.argv[sys.argv.index("--pdf") + 1]))
    if "--data" in sys.argv:
        for name in BUILD:
            data(name, fresh=True)
    if "--numbers" in sys.argv:
        numbers()
    else:
        figure()
