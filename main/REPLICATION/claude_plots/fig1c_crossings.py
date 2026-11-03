"""Coupled and uncoupled crossings of the marginal beliefs: what they go with (28 Oct 2026; John: "so we can identify
coupled and uncoupled crossings. Are 'splitting' trajectories characterized by a crossing? How do crossings relate to
factorization regret, entanglement, the speed of the marginal belief updates, the network properties? Can we visualize
particles and grids by coupled and uncoupled crossings as well? ... Can wormhole crossings explain the
entanglement-dependent representation unfoldings? (ideally you can answer these questions with gifs or cross
correlations, etc)")
-> main/REPLICATION/figure_bin/fig1c_crossings.png        figure()
   main/REPLICATION/figure_bin/fig1c_movie_crossings.gif  movie()
   (later the same day: fig1c_crossings_coupling.png figure_coupling(), fig1c_crossings_jumps.png figure_jumps(),
   fig1c_wormhole_help_hurt.png figure_help_hurt(), fig1c_split_pairs_lags.png figure_split_lags(), then
   fig1c_split_pairs_lags_coupling.png, fig1c_jumps_before.png figure_before(), fig1c_jumps_amplitude_check.png
   figure_amplitude_check(), fig1c_movie_crossings_combined.gif movie(combined=True); see the end of this docstring)

A crossing (plot_fig1c_figure8_traj.flow): from one step to the next a marginal moves >= CYCLE of its mass between its
two ends (values 0 - 1 and 8 - 9). Coupled: both marginals at once (same direction: along (0, 0) - (9, 9); opposite:
along (0, 9) - (9, 0)); uncoupled: one marginal alone. A crossing is dated to the step it arrives at.
Data: the alias scan (39 environments x 100 cells x 50 streams x 30 steps; all agents on the same streams); the
measures of plot_fig1c_measures_movie (the Trained RNN's); speed = the marginals' total-variation change per step,
(TV(b_g) + TV(b_c)) / 2. Primitives cached in figdata/round7_jumps/crossings.npz.
figure(): rows 1 - 2, every measure around the crossings (lags -6 .. 6 from the arrival step) as the departure from the
same cell's other streams (same environment, same step), the Trained RNN's crossings (its read-out) and the Optimal's;
row 3, crossing rates by step (all four agents), the split pairs' crossings around the split step, the hybrid observer.
movie(): row 1 particles flashing as crossings arrive; rows 2 - 3 the grids of all 39 environments, edges coloured by
the share of their cells' streams with a coupled / an uncoupled crossing so far; columns: the four agents.
Measured 28 Oct (CYCLE 0.25):
- rates per 1,000 stream-steps, coupled / uncoupled: Optimal 0.98 / 2.23, Trained RNN 0.91 / 2.75, Naive 0.04 / 6.59,
  ESN 0.01 / 2.55; coupled crossings come early (median step 5 - 6), the Naive's and ESN's uncoupled ones late (11).
- splits: of the 743 split pairs, 52% have a crossing (either stream) at t_s .. t_s + 2 in the Optimal (coupled 31%,
  uncoupled 27%) vs 3.2% for two random streams at the same steps; the Trained RNN on the same streams 52% (28 / 32);
  the Naive 4%, the ESN 5% (their base rates). The rest split without an end-to-end move (interior aliases).
- around a crossing (the Trained's; the Optimal's nearly identical): entanglement builds over ~4 steps and peaks at the
  update BEFORE the crossing (+0.12 log10 nats; rev[t] is the update into step t, as the crossing), near typical at
  the crossing; factorization regret +0.27 (coupled) / +0.21 (uncoupled) log10 at -1 / 0; error +0.37 - 0.41 log10 at
  0; dW energy +0.39 log10 at -1 / 0; |u| +0.2 at 0 / +1; activity effective dimension: coupled -3.2 at -1 then +1.2,
  uncoupled +1.6 before and +3.9 at 0; belief effective dimension -0.6 at -1 (coupled); cross-band share -0.25 to
  -0.34 around coupled crossings (all lags), ~0 around uncoupled; coherence within +-0.03; speed 0.30 / 0.20 at 0.
- hybrid (plot_fig1c_hybrid_unfold's observer; unfolded % at 0.5 / 1 / 2 / 5 / 10 / 25% of the environment's
  stream-steps given the joint likelihood): most entangled 1.7 / 4.6 / 11.1 / 29.6 / 48.4 / 74.9; the same with every
  step of coupled flow >= 0.1 removed (0.51% of all stream-steps) 1.7 / 4.6 / 11.2 / 29.8 / 48.5 / 75.0; the Optimal's
  coupled crossings first 2.0 / 4.2 / 8.2 / 19.7 / 34.0 / 53.2; random 1.3 / 2.6 / 5.0 / 12.1 / 22.2 / 42.9. Only
  0.88% / 0.68% of the top 1% / 5% most entangled steps carry a coupled flow >= 0.1 (all steps 0.51%): wormhole
  crossings do not explain the entanglement-dependent unfolding; entanglement precedes crossings, but the unfolding it
  drives runs through ordinary (non-crossing) joint updates.
figure_coupling() -> fig1c_crossings_coupling.png (John: "treat it as a continuous variable from uncoupled to fully
coupled ... where is the wormhole acting meaningfully different from the uncoupled crossing?"): a crossing = the larger
of the two flows >= CYCLE; coupling = smaller / larger flow (0 alone, 1 both alike). Of the Optimal's 18,166 crossings
22% have coupling < 0.1, 18% > 0.9, the rest spread evenly (Trained RNN 24 / 10%; Naive and ESN 88 - 89% < 0.1): the
binary split (both >= 0.25) cut through a continuum. By coupling decile (0 -> 9, the Optimal's crossings):
- shortcut: evidence distance moved per grid cell 0.665 -> 0.270 (grid moves 3.4 - 4.5 cells alike; evidence 2.23 ->
  1.20 lattice spacings); large non-crossing moves 0.640;
- correction: P(true cell) after 0.126 -> 0.239 (non-crossing large moves 0.066); accuracy below typical before every
  crossing, above typical after the coupled ones only;
- entanglement: uncoupled crossings at the crossing step (+0.11 log10 at lag 0), coupled ones one step BEFORE (+0.10
  at -1, +0.01 at 0); FR +0.15 -> +0.29; |Cov| +3.6 -> +10.5 at -1;
- network: activity effective dimension +5.7 / +9.6 (lag -1 / 0, uncoupled) vs -3.4 / +0.4 (coupled); belief
  effective dimension +0.1 vs -0.7; cross-band share +0.03 vs -0.6 (coupled streams low at every lag); coherence +0.1
  at uncoupled crossings;
- corners (hybrid observer): the Optimal's coupled-crossing steps unfold the 16 corner cells 2.8 / 5.3 / 9.7% at 0.5 /
  1 / 2% of stream-steps vs most entangled 0.9 / 2.4 / 6.7% and random 0.8 / 1.7 / 3.4%; for the other 84 cells they
  are no better than random (1.1 / 2.9 / 6.6 vs 1.7 / 3.4 / 6.8); from 5% on entanglement wins everywhere.
  50 bins (John: "bin coupling at a much finer resolution"): the correction is concentrated at near-full coupling
  (P(true) after ~0.10 - 0.13 up to coupling ~0.9, then 0.31 in the last bin); evidence per grid cell 0.69 -> 0.22;
  the cross-band drop is confined to the top bins.
- the joint's coupling (John: "what is the joint's split? that's the coupling according to the joint?"): |Cov| is the
  covariance of r_g and r_c inside the Optimal's joint belief, the joint's coupling weighted by distance (a 50 / 50
  split between (0, 0) and (9, 9): 20.25; between (3, 3) and (4, 4): 0.25). Distance-free: the mutual information
  I(r_g; r_c) (joint_mi; ln 2 for any 50 / 50 diagonal split). At the crossings it is 0.60 - 0.68 nats at EVERY
  coupling (all other steps: median 0.03): in the joint nearly every crossing is a coupled jump of both variables; at
  coupling 0 the other variable's mean still moves 1.2 cells (the leading one 3.0; at coupling 1 both ~3.0). So the
  marginal coupling measures the jump's shape: the other variable stays inside its range (oblique jump) or spans it
  too (corner to corner, the wormhole proper: the shortcut and the corrections).
- is the shortcut trivial? Not diagonal motion: diagonal non-crossing moves of >= 3 cells cost 0.61 evidence spacings
  per grid cell (non-diagonal 0.76) vs 0.27 for crossings with coupling > 0.9; it is the wrap of opposite corners in
  the evidence itself (what makes them wormholes), so expected from the geometry; the findings are who uses it.
figure_jumps() -> fig1c_crossings_jumps.png (John: "This coupling measure still feels arbiturary. We may need to have
2 variables, an unthresholded coupling and an amplitude?"). The crossing above has three arbitrary choices (the ends,
the 0.25, the ratio); here every step is described by the jump of the posterior mean, Delta mu = (Delta E[r_g],
Delta E[r_c]) (the movies' trajectories; jumps() caches it per agent in figdata/round7_jumps/jumps.npz), in polar form:
amplitude |Delta mu| (cells) and coupling = its angle from the nearest axis (0 deg: one variable moves alone, 45 deg:
both alike), no threshold. Mean coupling of jumps of 0.1 - 1 / 1 - 2 / 2 - 4 / >= 4 cells: Optimal 20.5 / 25.0 / 31.6 /
39.5 deg (>= 30 deg: 30 / 42 / 65 / 93%), Trained RNN 20.1 / 23.7 / 29.3 / 37.5, Naive 20.2 / 16.1 / 12.8 / 5.9, ESN
19.8 / 18.5 / 15.4 / 4.2 (22.5 = no preferred direction): the joint agents' large jumps move both variables, the
factorized agents' one; the two curves cross near 0.3 cells. The joint's split before the jump, I(r_g; r_c) at t - 1,
grows with amplitude and coupling: at >= 2 cells 0.18 / 0.28 / 0.42 nats for 0 - 15 / 15 - 30 / 30 - 45 deg (small
steps 0.02 - 0.08). Large jumps raise P(true cell) +0.036 / +0.039 / +0.056 (Optimal); entanglement of the jump's own
update is high for every jump above ~0.2 cells whatever its coupling (a size effect), the update BEFORE only for large
diagonal ones (+0.076 log10); FR after +0.01 / +0.08 / +0.17. Row 4, where the wormhole acts differently (the Trained
RNN's network, the Optimal's jumps of 3.5 - 5.5 cells): diagonal corner to corner (both end-to-end flows >= 0.2; 95%
land in a corner 2 x 2 block; n 5,306) cross-band share -0.26 / -0.21 (lag -1 / 0), activity effective dimension -2.6
/ +0.2; diagonal, the rest (n 8,637) +0.01 / +0.07, +0.4 / +3.1; along one axis (n 248) +0.03 / +0.13, +4.7 / +9.1.
The cross-band dip is flat over lags -6 .. 6: a STATE, not the jump (but see fig1c_two_processes.py: with a
fixed 5-step credit window the dip sits at the jump with no lead; the lead grows with the credit window). Every step,
by the Optimal's |Cov(r_g, r_c)|:
cross-band ~0 below |Cov| 8, -0.30 at 14 - 15, -1.14 at 18 - 19, -2.74 at 20 - 21 (a 50 / 50 split between opposite
corners); among split beliefs (MI > 0.4) +0.32 at |Cov| < 1 (a split between neighbouring cells: the same MI, the
opposite sign). Corner truths, steps 8 - 30: split across the wormhole (|Cov| > 12, n 5,913) -1.16 vs on the true
corner +0.08 (n 96,493); on the partner or an adjacent corner too rare (66 / 52) to compare. So while the belief
straddles the wormhole the bands' weight updates cancel (cross-band share is negative when they do) and activity is
lower-dimensional; the old "cross-band -0.6 at coupled crossings" was this state.
figure_help_hurt() -> fig1c_wormhole_help_hurt.png (John: "So 'wormhole need' is basically, the probability that a
neighbor gets you to the truth ... I'm not sure i folllow the math. Also i think you should plot the % in true cell in
probability scale so it doesn't fall off at 0, and maybe a finer resolution of the x axis. It could be that the
wormhole doesn't 'help' optimal / trained, but it hurts the Naive / ESN. But we'd need to see this clearly. It's still
very unclear what the wormholes give us besides unfolding of the grid corners."). Wormhole need (geodesic_need) is not
a probability: each grid step costs the evidence distance between its two cells; G_grid = the shortest route from the
belief's MAP cell to the true cell stepping between grid neighbours, G_jumps = the same when each cell may also jump to
its k = 8 nearest cells in evidence; need = 1 - G_jumps / G_grid, the fraction of the route a jump saves (0.84 between
opposite corners, 0.02 between adjacent ones). k = 16 agrees with k = 8 (r = 0.96 over all pairs, corners 0.84) but k
= 4 does not (corners 0.15): the opposite corner is only the ~8th nearest cell in evidence (median; 3rd - 25th).
The wormhole in numbers (lattice steps = the environment's median distance between grid neighbours; medians): from
(0, 0) the partner (9, 9) is 2.3 lattice steps away in evidence, the adjacent corners 6.2 / 6.4; in the Naive's view
(its averaged tables, plot_fig1c_evidence_geometry) 3.1 vs 4.4 / 4.5. It is a corner-only feature: the point reflection
(9 - g, 9 - c) has 6 - 7 cells closer than it at the four corners, 24 - 30 one cell in along the border, up to 97
elsewhere (the central four trivially near). Corner truths (7,800 streams per agent), MAP at steps 2 / 5 / 10 / 30:
Optimal on the truth 0.19 / 0.34 / 0.50 / 0.78, the partner corner 0.064 / 0.054 / 0.030 / 0.002, the adjacent corners
0.106 / 0.048 / 0.018 / 0.001; Trained RNN alike; Naive on the truth 0.20 throughout, partner 0.040 -> 0.017, adjacent
0.184 -> 0.135; ESN 0.16 -> 0.15, 0.042 -> 0.009, 0.176 -> 0.126. So the wormhole hurts nobody at the end and helps
nobody: for the joint agents it is the slowest corner confusion to clear (the partner outlasts the adjacent corners
from step 5); the factorized agents' corner errors are one variable flipped (adjacent corners, which the evidence
separates but their averaged tables do not), not the partner. MAP 8 - 11 cells from the truth at step 5, ending on
the true cell over need (20 bins): Optimal 0.69 - 0.81 and Trained RNN 0.63 - 0.80 with no trend (pooled, need < 0.45
vs above: 0.730 vs 0.743, 0.705 vs 0.723); Naive 0.12 -> 0.00 per bin (0.090 vs 0.024) and ESN 0.09 -> 0.00 (0.079 vs
0.039). Over time (step-5 MAP 8 - 11 cells away, need < 0.2 vs >
0.6): Optimal on the truth at step 6 0.22 vs 0.075, step 10 0.44 vs 0.30, step 30 0.72 vs 0.74 (Trained alike):
routes through the wormhole cost the joint agents time, not accuracy; Naive 0.095 vs 0.025 and ESN 0.083 vs 0.037 at
step 30. (Step-5 MAPs >= 5 cells from the truth, by quartile of their evidence distance to it, ending on the truth: the
Naive 0.03 - 0.06, the ESN 0.06 - 0.09 in every quartile; the Optimal 0.46 - 0.69, the Trained RNN 0.30 - 0.67: the
factorized agents' far errors are folds, wormhole or not.)
figure_split_lags() (John: "can you show the curves aligned on jump for that FR and entanglement of the 743 split
pairs"): the split movie's 743 pairs (39 environments; 442 distinct jumping streams, 685 distinct partners), the
stream that jumps = the one whose Optimal posterior mean moves more over the split (t_s - 1 -> t_s + 1; median 4.5
cells vs 0.7 for its partner), step 0 = its belief just after its larger one-step move (t_s 57%, t_s + 1 43%; median
3.6 cells); top row medians over the pairs, bottom row the mean of log10 x minus the same cell's typical (the mean of
log10 x over its 50 streams at that step); bands 5 - 95% over environments resampled (500). Factorization regret
does NOT spike: medians -2 / -1 / 0 / +1 / +2: the jumper 1.37 / 1.76 / 2.08 / 2.42 / 2.52 nats, its partner 0.68 /
0.83 / 0.96 / 1.11 / 1.19, all streams at the same steps 0.39 / 0.46 / 0.52 / 0.58 / 0.64; minus the cell's typical
(log10) the jumper +0.12 / +0.14 / +0.19 / +0.19 / +0.18 (a step, then a slow decline to +0.11 at +10), the partner
+0.08 flat, decaying to +0.02. Entanglement peaks at the jump's own update: minus typical (log10) -2 / -1 / 0 / +1 /
+2 the jumper +0.02 / +0.08 / +0.13 / +0.05 / +0.04, the partner -0.00 / +0.03 / +0.04 / +0.03 / +0.02 (medians:
the jumper 0.35 / 0.37 / 0.39 / 0.33 / 0.34, the partner ~0.25, all streams 0.20). Only 10% of the pairs saw the
same observations through t_s - 1 (29% hold identical beliefs at t_s - 1: Bayes ignores the order).
Later (John: "doesnt it seem like there is a meaningful phase transition at amplitude .1? or is that due to the data
processing? ... Maybe we should be looking at measures on the step before the jump? ... a version where we combine both
coupled and uncoupled crossings ... split the jumps by various levels of couplings"):
- figure_amplitude_check(): the white band near 0.1 cells in the maps is processing: 'minus typical' is zero where a
  jump is typical for its cell and step, and the typical jump shrinks from 1.2 cells (step 2) to 0.07 (step 30; 0.15
  or less from step 15 on, so the pooled maps put the band near 0.1). By arrival step 2 - 4 / 5 - 9 / 10 - 16 / 17 -
  30, median jump vs where minus-typical turns positive: entanglement 0.82 / 0.79, 0.33 / 0.36, 0.18 / 0.22, 0.094 /
  0.13; error 0.84 / 1.02, 0.34 / 0.47, 0.18 / 0.28, 0.095 / 0.099 (activity dim., cross-band share, dW energy alike).
  Raw values change smoothly with amplitude: entanglement flat below ~0.1 - 0.2 cells then rising, error and dW energy
  rising throughout, activity dimension rising from ~0.05 to ~1 cell, cross-band share peaking at 0.4 - 1 cell and
  falling for larger late jumps; jump sizes have one peak per step group, no gap: no natural threshold at 0.1.
- figure_before(): the step before the jump (t - 1) beside the jump (t). Jumps >= 2 cells (Optimal 96,461; Trained
  RNN 101,260), minus typical at coupling 0 - 15 / 15 - 30 / 30 - 45 deg: entanglement at t - 1 -0.021 / -0.007 /
  +0.059 (t - 3: +0.023 / +0.030 / +0.049), at t +0.128 / +0.134 / +0.097; factorization regret at t - 1 -0.043 /
  -0.008 / +0.112, at t +0.007 / +0.070 / +0.152; activity dimension at t - 1 +3.1 / +1.5 / -0.3, at t +5.3 / +3.6 /
  +1.9; cross-band share at t - 1 +0.011 / +0.001 / -0.045 (t - 3: -0.089 for diagonal), at t +0.080 / +0.049 / -0.008;
  error and dW energy without a direction effect (+0.15 - 0.34). Diagonal jumps come out of a STATE that lasts >= 3
  steps (entanglement and FR up, activity dimension and cross-band share down); at the jump, entanglement is a size
  effect, largest for one-axis jumps.
- figure_split_lags(by_coupling=True): the jumping stream of the 743 split pairs by its jump's angle (0 - 15 / 15 - 30
  / 30 - 40 / 40 - 45 deg: 56 / 89 / 164 / 434 jumps, median 1.7 / 2.1 / 3.4 / 4.3 cells). FR minus the cell's typical
  (log10) at -1 / 0 / +2: -0.22 / -0.05 / -0.02, +0.00 / +0.04 / +0.11, +0.18 / +0.25 / +0.23, +0.19 / +0.23 / +0.20
  (median raw at 0: 0.49 / 0.67 / 1.55 / 2.62 nats); entanglement at -2 / 0 / +1: -0.21 / +0.33 / +0.10, +0.03 / +0.22
  / +0.10, -0.04 / +0.17 / +0.03, +0.07 / +0.08 / +0.05 (median raw 40 - 45: 0.36 - 0.44 nats at every lag, no
  spike): the diagonal (wormhole) splits leave a sustained high-FR, high-entanglement state; the one-axis splits are a
  transient entanglement spike with FR at or below typical. Caveat: the levels differ in jump size.
- movie(combined=True): row 2 marks every stream (4 per cell) with any crossing so far where it is now; row 3 the grids
  by the share of each cell's streams with any crossing so far. Where crossings end up: crossings that end farther from
  the true cell than they started, Optimal 23% (n 18,166), Trained RNN 24%, Naive 49% (37,541), ESN 46%; streams with
  a crossing end median 0.27 / 0.26 / 3.02 / 3.70 cells from their cell vs 0.40 / 0.41 / 1.22 / 1.19 without; across
  cells (39 x 100), r(crossing share, the cell mean's displacement at step 30) 0.03 / 0.06 / 0.39 / 0.39, but cells
  with a crossing share above 0.2 hold only 25% (Naive) / 16% (ESN) of all displacement: crossings mark part of the
  factorized agents' tangling, most of it happens without an end-to-end crossing.
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig1c_crossings.py \\
        [numbers|hybrid|figure|movie|movie_continuous|movie_combined|coupling|jumps|help_hurt|split_lags|
         split_lags_coupling|before|amplitude_check]"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"),
                os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import matplotlib.pyplot as plt                                                   # noqa: E402
import plot_fig1c_measures_movie as mm                                            # noqa: E402
from plot_fig1c_figure8_traj import flow, CYCLE                                   # noqa: E402

R, T, E, NS = 10, 30, 39, 50
AGENTS = (("Joint", "Optimal"), ("Trained", "Trained RNN"), ("Naive", "Naive"), ("Echo", "ESN"))
CACHE = os.path.join(mm.R7, "crossings.npz")
LAGS = np.arange(-6, 7)


def primitives():
    """{agent: (flows (E, 5000, T - 1, 2) float16 [r_g, r_c], speed (E, 5000, T - 1) float16)}, cached"""
    if not os.path.exists(CACHE):
        out = {}
        for agent, _ in AGENTS:
            B = np.load(os.path.join(mm.SCAN, f"scan_{agent}.npz"))["beliefs"][mm.ENVS].astype(np.float32)
            bg = B[..., :R] / B[..., :R].sum(-1, keepdims=True)
            bc = B[..., R:] / B[..., R:].sum(-1, keepdims=True)
            del B
            out[f"{agent}_flow"] = np.stack([flow(bg), flow(bc)], -1).astype(np.float16)
            out[f"{agent}_speed"] = (.25 * (np.abs(np.diff(bg, axis=2)).sum(-1)
                                            + np.abs(np.diff(bc, axis=2)).sum(-1))).astype(np.float16)
            print(agent, "primitives", flush=True)
        np.savez(CACHE, **out)
    with np.load(CACHE) as f:
        return {a: (f[f"{a}_flow"].astype(np.float32), f[f"{a}_speed"].astype(np.float32)) for a, _ in AGENTS}


def labels(W, level=CYCLE):
    """flows (..., 2) -> crossing kind per transition: 0 none, 1 coupled, 2 uncoupled; and the coupled direction
    (+1 same, -1 opposite, 0)"""
    a = np.abs(W) >= level
    both = a[..., 0] & a[..., 1]
    one = a[..., 0] ^ a[..., 1]
    kind = np.where(both, 1, np.where(one, 2, 0)).astype(np.int8)
    return kind, np.where(both, np.sign(W[..., 0] * W[..., 1]), 0).astype(np.int8)


def at_steps(x):
    """per-transition (E, S, T - 1) -> per arrival step (E, S, T), step 1 empty"""
    out = np.zeros(x.shape[:2] + (T,), x.dtype)
    out[..., 1:] = x
    return out


def departures(V):
    """(E, 5000, T) values -> minus the mean of the same cell's 50 streams (same environment, same step); NaN kept"""
    import warnings
    V = V.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(V.reshape(E, R * R, NS, -1), 2)
    return V - np.repeat(m, NS, axis=1)


def triggered(events, V, lags=LAGS):
    """event-aligned mean of V (E, S, T; NaN = missing) per environment -> sums (E, L), counts (E, L)"""
    pad = len(lags)
    Vp = np.full(V.shape[:2] + (T + 2 * pad,), np.nan)
    Vp[..., pad:pad + T] = V
    s, c = np.zeros((E, len(lags))), np.zeros((E, len(lags)))
    for k, L in enumerate(lags):
        sh = Vp[..., pad + L:pad + L + T]
        ok = events & np.isfinite(sh)
        s[:, k] = np.where(ok, sh, 0).sum((1, 2))
        c[:, k] = ok.sum((1, 2))
    return s, c


def triggered_bins(B, V, nb, lags=LAGS):
    """event-aligned means of V (E, S, T) for events labelled by bin B (E, S, T; -1 = no event, 0 .. nb - 1) -> means
    (nb, L), counts (nb, L); one pass per lag"""
    pad = len(lags)
    Vp = np.full(V.shape[:2] + (T + 2 * pad,), np.nan)
    Vp[..., pad:pad + T] = V
    s, c = np.zeros((nb, len(lags))), np.zeros((nb, len(lags)))
    ev = B >= 0
    for k, L in enumerate(lags):
        sh = Vp[..., pad + L:pad + L + T]
        ok = ev & np.isfinite(sh)
        s[:, k] = np.bincount(B[ok], sh[ok], minlength=nb)[:nb]
        c[:, k] = np.bincount(B[ok], minlength=nb)[:nb]
    return s / np.maximum(c, 1), c


def boot(s, c, n=500, seed=0):
    """mean over all events and the 5 - 95% band over environments resampled"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, E, (n, E))
    bs = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return s.sum(0) / np.maximum(c.sum(0), 1), np.percentile(bs, [5, 95], axis=0)


def measures():
    """{key: (E, 5000, T)} raw (log10 for the movies' heavy-tailed measures), plus the movies' panel names"""
    _, _, _, strm, _ = mm.data(raw=True)
    out = {}
    for key, *_ in mm.PANELS:
        v = strm[key].astype(np.float64)
        if bool((v > 0).all()) and np.percentile(v, 98) > 20 * np.percentile(v, 2):
            v = np.log10(v)
            out[key] = (v, True)
        else:
            out[key] = (v, False)
    return out


def numbers():
    P = primitives()
    meta = np.load(os.path.join(mm.SCAN, "scan_meta.npz"))
    kinds = {a: labels(P[a][0]) for a, _ in AGENTS}
    print("when crossings happen (arrival step quartiles) and how often (per 1,000 stream-steps):")
    for a, name in AGENTS:
        k, d = kinds[a]
        for kk, lab in ((1, "coupled"), (2, "uncoupled")):
            st = np.nonzero(k == kk)[2] + 2
            print(f"  {name:12s} {lab:9s} {1000 * (k == kk).mean():.2f}; steps 25/50/75%: "
                  f"{np.percentile(st, 25):.0f} / {np.median(st):.0f} / {np.percentile(st, 75):.0f}")
    # overlap with the most entangled steps (the Optimal's coupled crossings)
    rev = meta["rev"][mm.ENVS].astype(np.float64)
    kO = at_steps(kinds["Joint"][0])
    pct = np.zeros_like(rev)
    for e in range(E):
        r_ = rev[e].ravel()
        pct[e] = (np.argsort(np.argsort(r_)) / (len(r_) - 1)).reshape(rev[e].shape)
    for kk, lab in ((1, "coupled"), (2, "uncoupled")):
        m = kO == kk
        print(f"  Optimal {lab}: entanglement percentile in its environment at the arrival step median "
              f"{100 * np.median(pct[m]):.0f}, step before {100 * np.median(pct[:, :, :-1][m[:, :, 1:]]):.0f}; "
              f"share in the top 5% {100 * np.mean(pct[m] >= .95):.1f}%")
    return kinds


def splits():
    """the split pairs (plot_fig1c_split_movie: two streams together through step t_s, >= 3 cells further apart two
    steps later; 743, selected in the Optimal): per agent and kind, the share of pairs with a crossing (either
    stream) arriving at t_s + lag, and the base rate (two random streams of the environment at the same step)"""
    import plot_fig1c_split_movie as sm
    Z = np.load(sm.DATA)
    rows = np.array([mm.ENVS.index(int(e)) for e in Z["env"]])
    i, j, ts = Z["i"].astype(int), Z["j"].astype(int), Z["ts"].astype(int)
    P = primitives()
    out = {}
    for a in ("Joint", "Trained"):
        K = at_steps(labels(P[a][0])[0])
        for kk in (1, 2):
            rate = (K == kk).mean(1)                                                  # (E, T) per environment
            hit, base = np.full((len(rows), len(LAGS)), np.nan), np.full((len(rows), len(LAGS)), np.nan)
            for n, L in enumerate(LAGS):
                t = ts + L
                ok = (t >= 1) & (t < T)
                hit[ok, n] = (K[rows[ok], i[ok], t[ok]] == kk) | (K[rows[ok], j[ok], t[ok]] == kk)
                base[ok, n] = 1 - (1 - rate[rows[ok], t[ok]]) ** 2
            out[a, kk] = (np.nanmean(hit, 0), np.nanmean(base, 0), np.isfinite(hit).sum(0))
    return out, len(rows)


def hybrid(levels=(.5, 1, 2, 5, 10, 25), seed=0):
    """the hybrid observer of plot_fig1c_hybrid_unfold (joint likelihood at chosen steps, the Naive's elsewhere) with
    the steps chosen over each environment's 150,000 stream-steps: the most entangled (ent), the Optimal's strongest
    coupled crossings (cross: min of the two flows into the step, ties at random), the most entangled after removing
    every step with a coupled flow >= 0.1 (entx), random (rand) -> cell means at T, cached"""
    import plot_fig1c_hybrid_unfold as hu
    path = os.path.join(mm.R7, "crossings_hybrid.npz")
    if not os.path.exists(path):
        meta = np.load(os.path.join(mm.SCAN, "scan_meta.npz"))
        goal = meta["goal"].astype(int)
        W = primitives()["Joint"][0]
        cstr = np.zeros((E, 5000, T))
        cstr[..., 1:] = np.minimum(np.abs(W[..., 0]), np.abs(W[..., 1]))
        rng = np.random.default_rng(seed)
        conds = ["naive", "optimal"] + [f"{s}{L:g}" for s in ("ent", "cross", "entx", "rand") for L in levels]
        cm = {c: np.zeros((E, R * R, 2 * R), np.float32) for c in conds}
        frac = {c: np.zeros(E) for c in conds}
        for ie, e in enumerate(mm.ENVS):
            lJ, lN = hu._terms(meta["L"][e].astype(np.float64), meta["obs"][e].astype(np.float64))
            n = len(lJ)
            rev = meta["rev"][e].astype(np.float64).ravel()
            cs = cstr[ie].ravel()
            o_ent = np.argsort(-rev)
            o_cross = np.argsort(-(cs + 1e-9 * rng.random(cs.size)))
            o_entx = np.argsort(-np.where(cs >= .1, -np.inf, rev))
            o_rand = rng.permutation(cs.size)
            masks = {"naive": np.zeros((n, T), bool), "optimal": np.ones((n, T), bool)}
            for L in levels:
                nsel = int(round(n * T * L / 100))
                for s_, o in (("ent", o_ent), ("cross", o_cross), ("entx", o_entx), ("rand", o_rand)):
                    m = np.zeros(n * T, bool)
                    m[o[:nsel]] = True
                    masks[f"{s_}{L:g}"] = m.reshape(n, T)
            for c, mask in masks.items():
                b = hu._run(lJ, lN, mask, goal[e])
                cm[c][ie] = b[:, -1].reshape(R * R, NS, 2 * R).mean(1)
                frac[c][ie] = mask.mean()
            print(f"hybrid env {e} ({ie + 1}/{E})", flush=True)
        np.savez(path, levels=np.array(levels), cstr_ge_01=float((cstr >= .1).mean()),
                 cstr_ge_cycle=float((cstr >= CYCLE).mean()), **{f"cm_{c}": v for c, v in cm.items()},
                 **{f"frac_{c}": v for c, v in frac.items()})
    return np.load(path)


def unfolded(Z):
    """unfolded share (d_Naive - d) / (d_Naive - d_Optimal) per condition, d = mean distance of the cells' posterior
    means (averaged over the environments) from their true cells (plot_fig1c_hybrid_unfold)"""
    meta = np.load(os.path.join(mm.SCAN, "scan_meta.npz"))
    cells = meta["cells"].astype(np.float64)
    r = np.arange(R)

    def d(c):
        cm = Z[f"cm_{c}"].astype(np.float64)
        pm = np.stack([cm[..., :R] @ r, cm[..., R:] @ r], -1).mean(0)
        return np.linalg.norm(pm - cells, axis=1).mean()
    dn, do = d("naive"), d("optimal")
    return {k[3:]: (dn - d(k[3:])) / (dn - do) for k in Z.files if k.startswith("cm_")}


PURPLE, ORANGE = "#5e3c99", "#e66101"


def figure():
    from main.utils import apply_fig_style, fig_path
    import fig1 as cf
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    P = primitives()
    M = measures()
    names = {k: (n, f) for k, n, f in mm.PANELS}
    names["speed"] = ("Speed of the marginals", r"$\frac{1}{2}\,(\mathrm{TV}(\Delta b_g) + \mathrm{TV}(\Delta b_c))$")
    keys = [k for k, *_ in mm.PANELS] + ["speed"]
    units = dict(mm.UNITS, speed="total variation per step")
    curves, rates = {}, {}
    for a, _ in AGENTS:
        K = at_steps(labels(P[a][0])[0])
        rates[a] = [1000 * (K == kk).mean((0, 1)) for kk in (1, 2)]
        if a not in ("Trained", "Joint"):
            continue
        sp = np.full(K.shape, np.nan)
        sp[..., 1:] = P[a][1]
        for key in keys:
            v = sp if key == "speed" else M[key][0]
            dv = departures(v)
            for kk in (1, 2):
                curves[a, kk, key] = boot(*triggered(K == kk, dv))
    sp_out, n_pairs = splits()
    U = unfolded(hybrid())
    levels = list(hybrid()["levels"])
    PW, PH, GX, GY, X0 = 2.55, 1.75, .78, 1.05, .75
    W = X0 + 5 * PW + 4 * GX + .3
    H = 1.05 + 2 * (PH + GY) + .25 + 2.3 + .2
    fig = plt.figure(figsize=(W, H))
    fig.text(.5, 1 - .22 / H, "Coupled and uncoupled crossings (alias scan: 39 environments x 5,000 streams x 30 "
             "steps): the measures around them, as departures from the same cell's other streams at the same step",
             ha="center", va="center", fontsize=FS["label"] + 1)
    fig.text(.5, 1 - .5 / H, "solid: the Trained RNN's crossings (its read-out); dashed: the Optimal's; purple: "
             "coupled (both marginals at once); orange: uncoupled (one); band: 5 - 95% over environments resampled; "
             "lag 0: the step the crossing arrives at", ha="center", va="center", fontsize=FS["label"])
    for n, key in enumerate(keys):
        r_, c_ = divmod(n, 5)
        ax = fig.add_axes([(X0 + c_ * (PW + GX)) / W, (H - 1.05 - PH - r_ * (PH + GY)) / H, PW / W, PH / H])
        ax.axhline(0, color="0.6", lw=.7), ax.axvline(0, color="0.6", lw=.7, ls=":")
        for a, ls in (("Trained", "-"), ("Joint", "--")):
            for kk, col in ((1, PURPLE), (2, ORANGE)):
                mean, band = curves[a, kk, key]
                if a == "Trained":
                    ax.fill_between(LAGS, band[0], band[1], color=col, alpha=.18, lw=0)
                ax.plot(LAGS, mean, color=col, lw=1.6 if a == "Trained" else 1.1, ls=ls)
        name, formula = names[key]
        ax.set_title(f"{name}\n{formula}", fontsize=FS["tick"] + 1, pad=3)
        ax.set_xlim(LAGS[0], LAGS[-1]), ax.set_xticks([-6, -3, 0, 3, 6])
        ax.tick_params(labelsize=FS["tick"])
        ax.set_ylabel(("log10 " if key != "speed" and M[key][1] else "") + units[key], fontsize=FS["tick"])
        ax.spines[["top", "right"]].set_visible(False)
        if r_ == 1:
            ax.set_xlabel("steps from the crossing", fontsize=FS["tick"] + 1)
    y3 = .75
    ax = fig.add_axes([X0 / W, y3 / H, 1.6 * PW / W, 2.0 / H])
    t = np.arange(1, T + 1)
    for a, name in AGENTS:
        c = cf.COL[a]
        ax.plot(t[1:], rates[a][0][1:], color=c, lw=1.6, label=f"{name}: coupled")
        ax.plot(t[1:], rates[a][1][1:], color=c, lw=1.1, ls=(0, (3, 2)), label=f"{name}: uncoupled")
    ax.set_xlim(2, T), ax.set_xlabel("step", fontsize=FS["tick"] + 1)
    ax.set_ylabel("crossings per 1,000 streams", fontsize=FS["tick"])
    ax.set_title("when crossings happen", fontsize=FS["label"], pad=4)
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FS["tick"] - 1, frameon=False, ncol=2, loc="upper right")
    ax = fig.add_axes([(X0 + 1.6 * PW + 1.15 * GX) / W, y3 / H, 1.55 * PW / W, 2.0 / H])
    for kk, col in ((1, PURPLE), (2, ORANGE)):
        hit, base, _ = sp_out["Joint", kk]
        ax.plot(LAGS, 100 * hit, color=col, lw=1.8, marker="o", ms=3,
                label=f"{'coupled' if kk == 1 else 'uncoupled'}: the split pairs")
        ax.plot(LAGS, 100 * base, color=col, lw=.9, ls=":", label="two random streams, same step")
    ax.axvline(0, color="0.6", lw=.7, ls=":")
    ax.set_xlim(LAGS[0], LAGS[-1]), ax.set_xlabel("steps from the split step $t_s$", fontsize=FS["tick"] + 1)
    ax.set_ylabel("% of pairs with a crossing\n(either stream; the Optimal's)", fontsize=FS["tick"])
    ax.set_title(f"splitting pairs ({n_pairs}: together through $t_s$, $\\geq$ 3 cells apart 2 steps later)",
                 fontsize=FS["label"], pad=4)
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FS["tick"] - 1, frameon=False, loc="upper left")
    ax = fig.add_axes([(X0 + 3.15 * PW + 2.3 * GX) / W, y3 / H, 1.6 * PW / W, 2.0 / H])
    for s_, lab, col, ls, mk in (("ent", "most entangled steps", "#d7301f", "-", "o"),
                                 ("entx", "most entangled, every coupled crossing removed (the same)", "#67000d",
                                  (0, (2, 2)), "s"),
                                 ("cross", "the Optimal's coupled crossings (strongest first)", PURPLE, "-", "o"),
                                 ("rand", "random steps", "0.45", "-", "o")):
        ax.plot(levels, [100 * U[f"{s_}{L:g}"] for L in levels], color=col, lw=1.4 if s_ == "entx" else 1.7, ls=ls,
                marker=mk, ms=5 if s_ == "entx" else 3, mfc="none" if s_ == "entx" else col, label=lab)
    ax.set_xscale("log"), ax.set_xticks(levels), ax.set_xticklabels([f"{L:g}" for L in levels])
    ax.set_xlabel("% of the environment's stream-steps given the joint likelihood", fontsize=FS["tick"] + 1)
    ax.set_ylabel("unfolded (%)", fontsize=FS["tick"])
    ax.set_title("the hybrid observer: where the unfolding comes from", fontsize=FS["label"], pad=4)
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FS["tick"] - 1, frameon=False, loc="upper left")
    path = fig_path("fig1c_crossings.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    for s_ in ("ent", "cross", "entx", "rand"):
        print(f"unfolded, {s_}: " + ", ".join(f"{L:g}%: {100 * U[f'{s_}{L:g}']:.0f}" for L in levels))
    return path


def joint_mi():
    """the coupling according to the joint: the mutual information between r_g and r_c under each belief of the
    Optimal, MI = H(g) + H(c) - H(g, c) (nats; 0 for a belief on one cell or a product of marginals, ln 2 for any
    50 / 50 split along a diagonal, whatever the distance) -> (E, 5000, T), cached in
    figdata/round7_jumps/joint_mi.npz"""
    import plot_fig1c_figure8 as f8
    path = os.path.join(mm.R7, "joint_mi.npz")
    if not os.path.exists(path):
        out = np.zeros((E, 5000, T), np.float32)
        for ie, e in enumerate(mm.ENVS):
            p, _ = f8.posterior(e)
            P = np.maximum(p.reshape(*p.shape[:2], R, R), 1e-300)
            h = lambda q: -(q * np.log(np.maximum(q, 1e-300))).sum(-1)              # noqa: E731
            out[ie] = h(P.sum(3)) + h(P.sum(2)) - h(P.reshape(*P.shape[:2], -1))
        np.savez(path, mi=out)
    with np.load(path) as f:
        return f["mi"].astype(np.float64)


def coupling(W, level=CYCLE):
    """flows (..., 2) -> crossing mask (the larger flow >= level), coupling kappa = smaller / larger flow in [0, 1]
    (0: one marginal alone, 1: both move as much), direction sign(W_g W_c) (+: along (0, 0) - (9, 9))"""
    a = np.abs(W)
    m = a.max(-1)
    return m >= level, a.min(-1) / np.maximum(m, 1e-12), np.sign(W[..., 0] * W[..., 1])


def where_numbers(nb=10):
    """per coupling bin (nb) of the Optimal's crossings: grid move (cells), evidence move (lattice spacings in the
    common evidence geometry), their ratio, max |Cov| of t - 1, t, the change of P(true cell), P(true cell) after, and
    the joint's correlation max |rho| of t - 1, t (rho = Cov / sqrt(Var_g Var_c); undefined, so unstable, for beliefs on
    one cell: figure_coupling uses joint_mi instead); the same for large non-crossing moves; the corner / other cells'
    unfolding in the hybrid observer"""
    import plot_fig1c_figure8_wrap as fw
    Z = fw.pooled()
    _, nn = fw.lattice_sheet(Z["Y"])
    X, CV = Z["X"].astype(np.float64), Z["CV"].astype(np.float64)
    B = np.load(os.path.join(mm.SCAN, "scan_Joint.npz"))["beliefs"][mm.ENVS].astype(np.float64)
    bg, bc = B[..., :R] / B[..., :R].sum(-1, keepdims=True), B[..., R:] / B[..., R:].sum(-1, keepdims=True)
    del B
    r = np.arange(R)
    meta = np.load(os.path.join(mm.SCAN, "scan_meta.npz"))
    tg, tc = (meta["cells"][meta["cell_of"]][:, k].astype(int) for k in (0, 1))
    ptrue = (np.take_along_axis(bg, tg[None, :, None, None], 3)[..., 0]
             * np.take_along_axis(bc, tc[None, :, None, None], 3)[..., 0])
    cr, kap, _ = coupling(primitives()["Joint"][0])
    mg, mc = bg @ r, bc @ r
    rho = np.abs(CV) / np.maximum(np.sqrt(np.maximum((bg @ r ** 2 - mg ** 2) * (bc @ r ** 2 - mc ** 2), 0)), 1e-9)
    rho = np.minimum(rho, 1.)
    dgrid = np.hypot(np.diff(mg, axis=2), np.diff(mc, axis=2))
    devid = np.linalg.norm(np.diff(X, axis=2), axis=-1) / nn
    covpk = np.maximum(np.abs(CV[..., 1:]), np.abs(CV[..., :-1]))
    rhopk = np.maximum(rho[..., 1:], rho[..., :-1])
    kb = np.minimum((kap * nb).astype(int), nb - 1)
    rows = []
    for b_ in range(nb):
        sel = cr & (kb == b_)
        rows.append((sel.sum(), np.median(dgrid[sel]), np.median(devid[sel]),
                     np.median(devid[sel] / np.maximum(dgrid[sel], 1e-6)), np.median(covpk[sel]),
                     np.mean(np.diff(ptrue, axis=2)[sel]), np.mean(ptrue[..., 1:][sel]), np.median(rhopk[sel])))
    big = (dgrid >= 3) & ~cr
    base = (big.sum(), np.median(dgrid[big]), np.median(devid[big]), np.median(devid[big] / dgrid[big]),
            np.median(covpk[big]), np.mean(np.diff(ptrue, axis=2)[big]), np.mean(ptrue[..., 1:][big]),
            np.median(rhopk[big]))
    H = hybrid()
    cells = meta["cells"].astype(np.float64)
    corner = np.zeros(R * R, bool)
    for g in (0, 1, 8, 9):
        for c in (0, 1, 8, 9):
            corner[R * g + c] = True

    def dist(c):
        cm = H[f"cm_{c}"].astype(np.float64)
        return np.linalg.norm(np.stack([cm[..., :R] @ r, cm[..., R:] @ r], -1).mean(0) - cells, axis=1)
    dn, do = dist("naive"), dist("optimal")
    unf = {}
    for s_ in ("ent", "cross", "rand"):
        for grp, msk in (("corner", corner), ("other", ~corner)):
            unf[s_, grp] = [(dn[msk].mean() - dist(f"{s_}{L:g}")[msk].mean()) / (dn[msk].mean() - do[msk].mean())
                            for L in H["levels"]]
    return np.array(rows), base, unf, list(H["levels"]), ptrue, rho


def figure_coupling(nb=50):
    """crossings with a continuous coupling, in nb bins (John: "treat it as a continuous variable from uncoupled to
    fully coupled ... where is the wormhole acting meaningfully different from the uncoupled crossing?"; "bin coupling
    at a much finer resolution"; "what is the joint's split? that's the coupling according to the joint?") ->
    fig1c_crossings_coupling.png"""
    from main.utils import apply_fig_style, fig_path
    import fig1 as cf
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    P = primitives()
    rows, base, unf, levels, ptrue, _ = where_numbers(nb)
    mi = joint_mi()
    M = measures()
    W = P["Joint"][0]
    cr, kap, _ = coupling(W)
    Bn = np.full(cr.shape[:2] + (T,), -1)
    Bn[..., 1:] = np.where(cr, np.minimum((kap * nb).astype(int), nb - 1), -1)          # bin at the arrival step
    sp = np.full(Bn.shape, np.nan)
    sp[..., 1:] = P["Joint"][1]
    r = np.arange(R, dtype=np.float32)
    Bj = np.load(os.path.join(mm.SCAN, "scan_Joint.npz"))["beliefs"][mm.ENVS].astype(np.float32)
    mg, mc = Bj[..., :R] @ r / Bj[..., :R].sum(-1), Bj[..., R:] @ r / Bj[..., R:].sum(-1)
    del Bj
    lead_g = np.abs(W[..., 0]) >= np.abs(W[..., 1])
    dg, dc = np.abs(np.diff(mg, axis=2)), np.abs(np.diff(mc, axis=2))
    kb = np.minimum((kap * nb).astype(int), nb - 1)
    mipk = np.maximum(mi[..., 1:], mi[..., :-1])
    moves = np.array([(np.median(np.where(lead_g, dg, dc)[cr & (kb == b_)]),
                       np.median(np.where(lead_g, dc, dg)[cr & (kb == b_)]),
                       np.median(mipk[cr & (kb == b_)])) for b_ in range(nb)])
    panels = [(k, n_, f_, M[k][1]) for k, n_, f_ in mm.PANELS]
    panels += [("speed", "Speed of the marginals", r"$\frac{1}{2}(\mathrm{TV}(\Delta b_g) + \mathrm{TV}(\Delta b_c))$",
                False),
               ("mi", "The joint's coupling", r"$I(r_g; r_c) = H_g + H_c - H_{gc}$ (Optimal)", False),
               ("ptrue", "Accuracy", r"$b_g(r_g^{\ast})\, b_c(r_c^{\ast})$ (true values)", False)]
    vals = {k: M[k][0] for k, *_ in mm.PANELS}
    vals.update(speed=sp, mi=mi, ptrue=ptrue)
    heat = {key: triggered_bins(Bn, departures(vals[key]), nb)[0] for key, *_ in panels}
    PW, PH, GX, X0 = 3.0, 2.15, .95, .75
    W_ = X0 + 4 * PW + 3 * GX + .5
    P1, G1 = 2.05, .62
    H = .95 + 1.95 + 1.3 + 3 * (PH + 1.0) + .2
    fig = plt.figure(figsize=(W_, H))
    fig.text(.5, 1 - .22 / H, "Crossings with a continuous coupling: of the Optimal's crossings (one marginal moves "
             r"$\geq$ 0.25 of its mass between its ends), coupling = the other marginal's move / this one's (0: alone, "
             f"1: both alike), in {nb} bins", ha="center", va="center", fontsize=FS["label"] + 1)
    fig.text(.5, 1 - .5 / H, f"alias scan: 39 environments x 5,000 streams x 30 steps; {int(cr.sum()):,} crossings of "
             "the Optimal", ha="center", va="center", fontsize=FS["label"])
    y1 = H - .95 - 1.95
    mid = (np.arange(nb) + .5) / nb

    def top(i, title, ylab):
        ax = fig.add_axes([(X0 + i * (P1 + G1)) / W_, y1 / H, P1 / W_, 1.95 / H])
        ax.set_title(title, fontsize=FS["tick"] + 1, pad=4)
        ax.set_ylabel(ylab, fontsize=FS["tick"])
        ax.tick_params(labelsize=FS["tick"] - 1)
        ax.spines[["top", "right"]].set_visible(False)
        return ax
    ax = top(0, "how coupled the crossings are", "% of the agent's crossings")
    for a, name in AGENTS:
        crA, kA, _ = coupling(P[a][0])
        h = np.histogram(kA[crA], np.linspace(0, 1, nb + 1))[0]
        ax.plot(mid, 100 * h / h.sum(), color=cf.COL[a], lw=1.5, label=name,
                ls=(0, (3, 2)) if a == "Naive" else "-", zorder=3 if a == "Naive" else 2)
    ax.set_yscale("log"), ax.set_xlabel("coupling", fontsize=FS["tick"])
    ax.legend(fontsize=FS["tick"] - 2, frameon=False, loc="upper center")
    ax = top(1, "the joint's coupling at the crossing\n(Optimal, larger of the two steps)", r"$I(r_g; r_c)$ (nats)")
    ax.plot(mid, moves[:, 2], color=PURPLE, lw=1.6)
    ax.axhline(np.log(2), color="0.5", lw=.8, ls=":", label="ln 2: a two-cell diagonal split")
    ax.axhline(np.median(mipk[~cr]), color="0.7", lw=.8, ls="--", label="all other steps")
    ax.set_ylim(0, .8), ax.set_xlabel("coupling", fontsize=FS["tick"])
    ax.legend(fontsize=FS["tick"] - 2, frameon=False, loc="lower right")
    ax = top(2, "how far each variable's mean moves", "cells")
    ax.plot(mid, moves[:, 0], color="k", lw=1.6, label="the one moving between its ends")
    ax.plot(mid, moves[:, 1], color=PURPLE, lw=1.6, label="the other")
    ax.set_ylim(0, 4), ax.set_xlabel("coupling", fontsize=FS["tick"])
    ax.legend(fontsize=FS["tick"] - 2, frameon=False, loc="lower right")
    ax = top(3, "a shortcut: evidence distance\nper grid cell moved (medians)", "lattice spacings / cell")
    ax.plot(mid, rows[:, 3], color=PURPLE, lw=1.6, label="crossings")
    ax.axhline(base[3], color="0.5", lw=.8, ls=":", label=r"other moves $\geq$ 3 cells")
    ax.set_ylim(0, .8), ax.set_xlabel("coupling", fontsize=FS["tick"])
    ax.legend(fontsize=FS["tick"] - 2, frameon=False, loc="lower left")
    ax = top(4, "a correction: P(true cell)\nafter the move", "P(true cell)")
    ax.plot(mid, rows[:, 6], color=PURPLE, lw=1.6, label="after the crossing")
    ax.axhline(base[6], color="0.5", lw=.8, ls=":", label=r"after other moves $\geq$ 3 cells")
    ax.set_ylim(0, .32), ax.set_xlabel("coupling", fontsize=FS["tick"])
    ax.legend(fontsize=FS["tick"] - 2, frameon=False, loc="upper left")
    ax = top(5, "the hybrid observer: 16 corner\ncells (solid), the other 84 (dashed)", "unfolded (%)")
    for s_, lab, col in (("ent", "most entangled", "#d7301f"), ("cross", "coupled crossings", PURPLE),
                         ("rand", "random", "0.45")):
        ax.plot(levels, [100 * v for v in unf[s_, "corner"]], color=col, lw=1.6, marker="o", ms=2.5, label=lab)
        ax.plot(levels, [100 * v for v in unf[s_, "other"]], color=col, lw=.9, ls=(0, (3, 2)))
    ax.set_xscale("log"), ax.set_xticks(levels), ax.set_xticklabels([f"{L:g}" for L in levels])
    ax.set_xlabel("% of stream-steps given the joint", fontsize=FS["tick"])
    ax.legend(fontsize=FS["tick"] - 2, frameon=False, loc="upper left")
    fig.text(.5, (y1 - .85) / H, "the measures around the crossings, by coupling: departure from the same cell's other "
             f"streams at the same step (rows: {nb} coupling bins; columns: steps from the crossing's arrival; red "
             "above, blue below)", ha="center", va="center", fontsize=FS["label"])
    for n, (key, name, formula, lg) in enumerate(panels):
        r_, c_ = divmod(n, 4)
        x, y = X0 + c_ * (PW + GX), y1 - 1.3 - PH - r_ * (PH + 1.0)
        ax = fig.add_axes([x / W_, y / H, PW / W_, PH / H])
        Hm = heat[key]
        lim = np.nanpercentile(np.abs(Hm), 99)
        im = ax.imshow(Hm, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest",
                       extent=(LAGS[0] - .5, LAGS[-1] + .5, 0, 1))
        ax.axvline(0, color="k", lw=.6, ls=":")
        ax.set_xticks([-6, -3, 0, 3, 6]), ax.set_yticks([0, .5, 1])
        ax.tick_params(labelsize=FS["tick"])
        if c_ == 0:
            ax.set_ylabel("coupling", fontsize=FS["tick"] + 1)
        if r_ == 2:
            ax.set_xlabel("steps from the crossing", fontsize=FS["tick"] + 1)
        ax.set_title(f"{name}\n{formula}", fontsize=FS["tick"] + 1, pad=3)
        cb = fig.colorbar(im, ax=ax, fraction=.06, pad=.02)
        unit = {"speed": "TV per step", "mi": "nats", "ptrue": "probability"}.get(key, mm.UNITS.get(key, ""))
        cb.set_label(("log10 " if lg else "") + unit, fontsize=FS["tick"] - 1)
        cb.ax.tick_params(labelsize=FS["tick"] - 2)
    path = fig_path("fig1c_crossings_coupling.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    q = [0, nb // 4, nb // 2, 3 * nb // 4, nb - 1]
    print("bins " + ", ".join(f"{b_ / nb:.2f}" for b_ in q) + ": crossings "
          + ", ".join(f"{int(rows[b_, 0])}" for b_ in q))
    print("  evidence/grid " + ", ".join(f"{rows[b_, 3]:.2f}" for b_ in q) + f" (base {base[3]:.2f}); P(true) after "
          + ", ".join(f"{rows[b_, 6]:.3f}" for b_ in q) + f" (base {base[6]:.3f}); joint MI "
          + ", ".join(f"{moves[b_, 2]:.2f}" for b_ in q) + "; other variable's move "
          + ", ".join(f"{moves[b_, 1]:.2f}" for b_ in q))
    return path


def geodesic_need(k=8):
    """per environment, the wormhole need of the route between every two cells: 1 - (the evidence geodesic over a graph
    joining each cell to its k nearest cells in evidence and to its lattice neighbours) / (the evidence geodesic that
    may only step between lattice neighbours); weights: the environment's sqrt-Jeffreys distances. ~0.85 between
    opposite corners, ~0 between adjacent corners -> (E, 100, 100), cached in figdata/round7_jumps/geodesic_need.npz"""
    from scipy.sparse.csgraph import shortest_path
    import plot_fig1c_evidence_geometry as eg
    path = os.path.join(mm.R7, "geodesic_need.npz")
    if not os.path.exists(path):
        D3, _, _ = eg.geometries()
        cells = np.array([(g, c) for g in range(R) for c in range(R)])
        lat = np.abs(cells[:, None] - cells[None]).sum(-1) == 1
        out = np.zeros((E, R * R, R * R))
        for i in range(E):
            De = D3["evidence"][i].astype(np.float64)
            knn = np.zeros_like(lat)
            o = np.argsort(De, 1)[:, 1:k + 1]
            knn[np.repeat(np.arange(R * R), k), o.ravel()] = True
            knn = knn | knn.T | lat
            out[i] = 1 - shortest_path(np.where(knn, De, 0), directed=False) / np.maximum(
                shortest_path(np.where(lat, De, 0), directed=False), 1e-12)
        np.savez(path, need=out.astype(np.float32))
    with np.load(path) as f:
        return f["need"].astype(np.float64)


JUMPS = os.path.join(mm.R7, "jumps.npz")


def jumps():
    """{agent: (E, 5000, T - 1, 2)}: each step's jump of the belief's posterior mean, (Delta E[r_g], Delta E[r_c]) in
    cells, from the agent's marginals (the trajectories of the movies), cached in figdata/round7_jumps/jumps.npz"""
    if not os.path.exists(JUMPS):
        r = np.arange(R, dtype=np.float32)
        out = {}
        for agent, _ in AGENTS:
            B = np.load(os.path.join(mm.SCAN, f"scan_{agent}.npz"))["beliefs"][mm.ENVS].astype(np.float32)
            mu = np.stack([B[..., :R] @ r / B[..., :R].sum(-1), B[..., R:] @ r / B[..., R:].sum(-1)], -1)
            del B
            out[agent] = np.diff(mu, axis=2).astype(np.float16)
        np.savez(JUMPS, **out)
    with np.load(JUMPS) as f:
        return {a: f[a].astype(np.float32) for a, _ in AGENTS}


def polar(d):
    """jumps (..., 2) -> amplitude (cells: the jump's length) and coupling (its angle from the nearest axis, degrees:
    0 = one variable moves alone, 45 = both move alike, a diagonal jump)"""
    a = np.abs(d)
    return np.hypot(a[..., 0], a[..., 1]), np.degrees(np.arctan2(a.min(-1), a.max(-1)))


def _ptrue(agent, meta):
    """the agent's probability on the true cell from its marginals, b_g(r_g*) b_c(r_c*) -> (E, 5000, T)"""
    B = np.load(os.path.join(mm.SCAN, f"scan_{agent}.npz"))["beliefs"][mm.ENVS].astype(np.float32)
    tg, tc = (meta["cells"][meta["cell_of"]][:, k].astype(int) for k in (0, 1))
    bg, bc = B[..., :R] / B[..., :R].sum(-1, keepdims=True), B[..., R:] / B[..., R:].sum(-1, keepdims=True)
    del B
    return (np.take_along_axis(bg, tg[None, :, None, None], 3)[..., 0]
            * np.take_along_axis(bc, tc[None, :, None, None], 3)[..., 0]).astype(np.float64)


def figure_jumps(na=40, nphi=30, amin=.01, amax=12., min_n=50):
    """every step, no threshold: the jump of the posterior mean in two variables, amplitude x coupling (John: "This
    coupling measure still feels arbiturary. We may need to have 2 variables, an unthresholded coupling and an
    amplitude?") -> fig1c_crossings_jumps.png"""
    from main.utils import apply_fig_style, fig_path
    import fig1 as cf
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    J = jumps()
    meta = np.load(os.path.join(mm.SCAN, "scan_meta.npz"))
    ae, pe = np.geomspace(amin, amax, na + 1), np.linspace(0, 45, nphi + 1)
    nbin = na * nphi

    def bmean(B, V):
        ok = (B >= 0) & np.isfinite(V)
        n_ = np.bincount(B[ok], minlength=nbin)
        m_ = np.bincount(B[ok], V[ok], minlength=nbin) / np.maximum(n_, 1)
        return np.where(n_ >= min_n, m_, np.nan).reshape(na, nphi)

    Bs, counts, curves = {}, {}, {}
    for a_, name in AGENTS:
        amp, phi = polar(J[a_])
        ia = np.searchsorted(ae, amp, side="right") - 1
        ok = (ia >= 0) & (ia < na)
        Bs[a_] = np.where(ok, ia * nphi + np.minimum((phi * nphi / 45).astype(int), nphi - 1), -1)
        counts[a_] = np.bincount(Bs[a_][ok], minlength=nbin).reshape(na, nphi)
        n_ = np.bincount(ia[ok], minlength=na)
        curves[a_] = np.where(n_ >= min_n, np.bincount(ia[ok], phi[ok], minlength=na) / np.maximum(n_, 1), np.nan)
        print(f"{name:12s} jumps: " + "; ".join(
            f"a {lo:g}-{hi:g}: {100 * s.mean():.2f}% of steps, mean angle {phi[s].mean():.1f}, >= 30 deg "
            f"{100 * np.mean(phi[s] >= 30):.0f}%" for lo, hi in ((.1, 1), (1, 2), (2, 4), (4, 13))
            for s in [(amp >= lo) & (amp < hi)]), flush=True)
    del J
    M = measures()
    dep = {k: departures(M[k][0]) for k in ("rev", "fr", "err", "pr", "band_x", "dW")}
    lg = {k: M[k][1] for k in dep}
    del M
    mi = joint_mi()
    pO, pT = _ptrue("Joint", meta), _ptrue("Trained", meta)
    rows = [("Joint", [("mi", "The joint's coupling before the jump", r"$I(r_g; r_c)$ at $t - 1$ (Optimal)",
                        mi[..., :-1], "nats", False),
                       ("dp", "Accuracy change", r"$\Delta\, b_g(r_g^{\ast})\, b_c(r_c^{\ast})$ (true values)",
                        np.diff(pO, axis=2), "probability", True),
                       ("rev", "Entanglement, the update before", r"$\mathrm{rev}_{t-1}$",
                        dep["rev"][..., :-1], None, True),
                       ("rev", "Entanglement, the jump's update", r"$\mathrm{rev}_{t}$", dep["rev"][..., 1:], None,
                        True),
                       ("fr", "Factorization regret after", r"$\mathrm{FR}_t$ (both variables)", dep["fr"][..., 1:],
                        None, True)]),
            ("Trained", [("dp", "Accuracy change", r"$\Delta\, b_g(r_g^{\ast})\, b_c(r_c^{\ast})$ (its read-out)",
                          np.diff(pT, axis=2), "probability", True),
                         ("err", "Error after", r"$\delta_t$ vs the Optimal", dep["err"][..., 1:], None, True),
                         ("pr", "Effective dim., activity", r"$\mathrm{PR}(u_t)$", dep["pr"][..., 1:], None, True),
                         ("band_x", "Cross-band share", r"of $\Delta W_t$", dep["band_x"][..., 1:], None, True),
                         ("dW", r"$\Delta W$ energy", r"$\|\Sigma_k \delta_k z_k^\top\|^2$", dep["dW"][..., 1:], None,
                          True)])]
    heat = {(a_, n): bmean(Bs[a_], v[3]) for a_, panels in rows for n, v in enumerate(panels)}
    # row 4: the wormhole traversal against other jumps of the same size (the Optimal's jumps, the Trained's network)
    amp, phi = polar(jumps()["Joint"])
    Wa = np.abs(primitives()["Joint"][0])
    size = (amp >= 3.5) & (amp < 5.5)
    diag = size & (phi >= 35)
    groups = ((r"diagonal, corner to corner ($\min(|W_g|, |W_c|) \geq 0.2$)", PURPLE, diag & (Wa.min(-1) >= .2)),
              ("diagonal, the rest", "0.45", diag & ~(Wa.min(-1) >= .2)),
              (r"along one axis ($\varphi_t < 10^\circ$)", ORANGE, size & (phi < 10)))
    lagged = {}
    for key in ("band_x", "pr"):
        for gname, _, sel in groups:
            lagged[key, gname] = boot(*triggered(at_steps(sel), dep[key]))
    w_edges = np.linspace(0, .5, 21)
    wb = np.digitize(Wa.min(-1), w_edges) - 1
    corner_mass = []
    for b_ in range(len(w_edges) - 1):
        sel = diag & (wb == b_)
        corner_mass.append((np.nanmean(dep["band_x"][..., :-1][sel]) if sel.sum() >= 30 else np.nan,
                            np.nanmean(dep["band_x"][..., 1:][sel]) if sel.sum() >= 30 else np.nan, int(sel.sum())))
    corner_mass = np.array(corner_mass)
    import plot_fig1c_figure8_wrap as fw                                          # the state: |Cov| of every belief
    CVa = np.abs(fw.pooled()["CV"].astype(np.float64))
    mi_all = joint_mi()
    c_edges = np.arange(0, 22, 1.)
    cb_ = np.digitize(CVa, c_edges) - 1
    state = {}
    for lab, msk in (("all", np.ones(CVa.shape, bool)), ("split", mi_all > .4)):
        rows_ = []
        for b_ in range(len(c_edges) - 1):
            sel = msk & (cb_ == b_)
            rows_.append((np.nanmean(dep["band_x"][sel]) if sel.sum() >= 30 else np.nan, int(sel.sum())))
        state[lab] = np.array(rows_)
    print("  every step, cross-band share minus typical by |Cov| (bins of 1): all " + ", ".join(
        f"{v:+.2f}" for v in state["all"][:, 0]) + "; MI > 0.4: "
        + ", ".join(f"{v:+.2f}" for v in state["split"][:, 0]))
    for gname, _, sel in groups:
        print(f"  size-matched {gname[:40]:40s}: n {int(sel.sum()):,}, amplitude {amp[sel].mean():.2f}; cross-band "
              f"lag -1 / 0 {np.nanmean(dep['band_x'][..., :-1][sel]):+.3f} / "
              f"{np.nanmean(dep['band_x'][..., 1:][sel]):+.3f}; activity PR {np.nanmean(dep['pr'][..., :-1][sel]):+.2f}"
              f" / {np.nanmean(dep['pr'][..., 1:][sel]):+.2f}")
    PW, PH, GX, GY, X0 = 2.45, 1.95, 1.35, 1.12, .75                                # GX: colour bar + 2-line y label
    W_ = X0 + 5 * PW + 4 * GX + .8
    H_ = 1.2 + 3 * (PH + GY) + .55 + PH + .75
    fig = plt.figure(figsize=(W_, H_))
    fig.text(.5, 1 - .24 / H_, r"Every step's jump of the belief's posterior mean, $\Delta\mu_t = (\Delta E[r_g], "
             r"\Delta E[r_c])$, as two variables, no threshold: amplitude $|\Delta\mu_t|$ (cells, x axis) and coupling "
             r"= its angle from the nearest axis, $\varphi_t = \arctan(|\Delta|_{\mathrm{smaller}} / "
             r"|\Delta|_{\mathrm{larger}})$ (y axis; 0$^\circ$: one variable moves alone, 45$^\circ$: both alike)",
             ha="center", va="center", fontsize=FS["label"])
    fig.text(.5, 1 - .55 / H_, "alias scan: 39 environments x 5,000 streams x 29 jumps = 5,655,000 jumps per agent; "
             f"rows 2 - 3: the mean over each bin's jumps (at least {min_n}; grey: fewer); 'minus typical' = minus "
             "the mean of the same cell's 50 streams at the same step; dotted: 2 cells, the movies' dashed jumps",
             ha="center", va="center", fontsize=FS["label"])

    def place(r_, c_):
        x, y = X0 + c_ * (PW + GX), H_ - 1.2 - PH - r_ * (PH + GY)
        ax = fig.add_axes([x / W_, y / H_, PW / W_, PH / H_])
        ax.tick_params(labelsize=FS["tick"])
        return ax

    def hm(ax, Mx, cmap, vmin, vmax, unit):
        ax.set_facecolor("0.85")
        im = ax.pcolormesh(ae, pe, Mx.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", rasterized=True)
        ax.set_xscale("log"), ax.set_xlim(amin, amax), ax.set_ylim(0, 45), ax.set_yticks([0, 15, 30, 45])
        ax.axvline(2, color="k", lw=.6, ls=":")
        ax.set_xlabel(r"amplitude $|\Delta\mu_t|$ (cells)", fontsize=FS["tick"] + 1, labelpad=1)
        ax.set_ylabel(r"coupling $\varphi_t$ (degrees)" + "\n0: one variable moves, 45: both", fontsize=FS["tick"] + 1)
        cb = fig.colorbar(im, ax=ax, fraction=.06, pad=.02)
        cb.set_label(unit, fontsize=FS["tick"] - 1)
        cb.ax.tick_params(labelsize=FS["tick"] - 2)

    vmax_c = max(np.log10(max(c.max(), 1)) for c in counts.values())
    for c_, (a_, name) in enumerate(AGENTS):
        ax = place(0, c_)
        hm(ax, np.where(counts[a_] > 0, np.log10(np.maximum(counts[a_], 1)), np.nan), "viridis", 0, vmax_c,
           r"log$_{10}$ jumps")
        ax.set_title(f"{name}: how its jumps spread\n" + r"log$_{10}$ number of jumps per bin", fontsize=FS["tick"] + 1,
                     pad=3, color=cf.COL[a_])
    ax = place(0, 4)
    mid = np.sqrt(ae[:-1] * ae[1:])
    for a_, name in AGENTS:
        ax.plot(mid, curves[a_], color=cf.COL[a_], lw=1.7, label=name, ls=(0, (3, 2)) if a_ == "Naive" else "-",
                zorder=3 if a_ == "Naive" else 2)
    ax.axhline(22.5, color="0.5", lw=.8, ls=":")
    ax.text(amin * 1.3, 23.5, "22.5$^\\circ$: no preferred direction", fontsize=FS["tick"] - 1, color="0.4")
    ax.axvline(2, color="k", lw=.6, ls=":")
    ax.set_xscale("log"), ax.set_xlim(amin, amax), ax.set_ylim(0, 45), ax.set_yticks([0, 15, 30, 45])
    ax.set_xlabel(r"amplitude $|\Delta\mu_t|$ (cells)", fontsize=FS["tick"] + 1, labelpad=1)
    ax.set_ylabel(r"mean coupling $\varphi_t$ (degrees)", fontsize=FS["tick"] + 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FS["tick"] - 1, frameon=False, loc="lower left")
    ax.set_title("how coupled the jumps are, by size\n" + r"mean $\varphi_t$ per amplitude bin",
                 fontsize=FS["tick"] + 1, pad=3)
    for r_, (a_, panels) in enumerate(rows, start=1):
        for c_, (key, name, formula, _, unit, signed) in enumerate(panels):
            ax = place(r_, c_)
            Hm = heat[a_, c_]
            if signed:
                lim = np.nanpercentile(np.abs(Hm), 99)
                u_ = unit or (("log$_{10}$ " if lg.get(key) else "") + mm.UNITS[key] + ", minus typical")
                hm(ax, Hm, "RdBu_r", -lim, lim, u_)
            else:
                hm(ax, Hm, "viridis", 0, np.log(2), unit)
            who = "Optimal" if a_ == "Joint" else "Trained RNN"
            ax.set_title(f"{who}: {name[0].lower() + name[1:]}\n{formula}", fontsize=FS["tick"] + 1, pad=3)
    fig.text(.5, (H_ - 1.2 - PH - .62) / H_, "row 2: the Optimal's jumps (its exact marginals); row 3: the Trained "
             "RNN's jumps (its read-out) and its network at the jump's step", ha="center", va="center",
             fontsize=FS["label"])
    y4 = H_ - 1.2 - PH - 3 * (PH + GY) - .55
    fig.text(.5, (y4 + PH + .72) / H_, "row 4: where the wormhole acts differently - the Trained RNN's network on "
             "the same streams, minus typical; panels 1 - 3: the Optimal's jumps of 3.5 - 5.5 cells ($W_v$: the mass "
             "marginal v moves between its ends, values 0 - 1 and 8 - 9; bands: 5 - 95% over environments resampled); "
             "panel 4: every step", ha="center", va="center", fontsize=FS["label"])
    G4 = 1.3
    P4 = (5 * PW + 4 * GX - 3 * G4) / 4
    for c_, (key, name, formula) in enumerate((("band_x", "Cross-band share", r"of $\Delta W$"),
                                               ("pr", "Effective dim., activity", r"$\mathrm{PR}(u)$"))):
        ax = fig.add_axes([(X0 + c_ * (P4 + G4)) / W_, y4 / H_, P4 / W_, PH / H_])
        lo_all, hi_all = [], []
        for gname, col, sel in groups:
            m_, (lo_, hi_) = lagged[key, gname]
            ax.fill_between(LAGS, lo_, hi_, color=col, alpha=.2, lw=0)
            ax.plot(LAGS, m_, color=col, lw=1.7, marker="o", ms=2.5, label=f"{gname} (n {int(sel.sum()):,})")
            lo_all.append(np.nanmin(lo_)), hi_all.append(np.nanmax(hi_))
        span = max(hi_all) - min(lo_all)
        ax.set_ylim(min(lo_all) - .55 * span, max(hi_all) + .05 * span)              # room for the key below
        ax.axhline(0, color="0.6", lw=.7, ls=":"), ax.axvline(0, color="k", lw=.6, ls=":")
        ax.set_xticks(LAGS[::2]), ax.tick_params(labelsize=FS["tick"])
        ax.set_xlabel("steps from the jump's arrival", fontsize=FS["tick"] + 1, labelpad=1)
        ax.set_ylabel(("log$_{10}$ " if lg[key] else "") + mm.UNITS[key] + ", minus typical",
                      fontsize=FS["tick"] + 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"Trained RNN: {name[0].lower() + name[1:]} around the jump\n{formula}", fontsize=FS["tick"] + 1,
                     pad=3)
        if c_ == 0:
            ax.legend(fontsize=FS["tick"] - 1, frameon=False, loc="lower left")
    ax = fig.add_axes([(X0 + 2 * (P4 + G4)) / W_, y4 / H_, P4 / W_, PH / H_])
    mid_w = (w_edges[:-1] + w_edges[1:]) / 2
    ax.plot(mid_w, corner_mass[:, 0], color=PURPLE, lw=1.7, marker="o", ms=2.5, label="the step before the jump")
    ax.plot(mid_w, corner_mass[:, 1], color=PURPLE, lw=1.2, ls=(0, (3, 2)), marker="o", ms=2,
            label="the jump's step")
    ax.axhline(0, color="0.6", lw=.7, ls=":")
    ax.set_xlim(0, .5), ax.tick_params(labelsize=FS["tick"])
    ax.set_xlabel(r"mass moved corner to corner, $\min(|W_g|, |W_c|)$", fontsize=FS["tick"] + 1, labelpad=1)
    ax.set_ylabel("share, minus typical", fontsize=FS["tick"] + 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Trained RNN: cross-band share at diagonal jumps of 3.5 - 5.5\ncells, by the mass both "
                 "marginals move between their ends", fontsize=FS["tick"] + 1, pad=3)
    ax.legend(fontsize=FS["tick"] - 1, frameon=False, loc="lower left")
    ax = fig.add_axes([(X0 + 3 * (P4 + G4)) / W_, y4 / H_, P4 / W_, PH / H_])
    mid_c = (c_edges[:-1] + c_edges[1:]) / 2
    for lab, col, ls, txt in (("all", "0.45", "-", "every belief"),
                              ("split", PURPLE, "-", r"split beliefs only, $I(r_g; r_c) > 0.4$ nats")):
        ax.plot(mid_c, state[lab][:, 0], color=col, lw=1.7, ls=ls, marker="o", ms=2.5,
                label=f"{txt} (n {int(state[lab][:, 1].sum()):,})")
    ax.axhline(0, color="0.6", lw=.7, ls=":")
    ax.set_xlim(0, 21), ax.tick_params(labelsize=FS["tick"])
    ax.set_xlabel(r"the Optimal's $|\mathrm{Cov}(r_g, r_c)|$ within the belief (20.25: split between opposite "
                  "corners)", fontsize=FS["tick"], labelpad=1)
    ax.set_ylabel("share, minus typical", fontsize=FS["tick"] + 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Trained RNN: cross-band share at every step, by the\nbelief's state (steps 1 - 30, all streams)",
                 fontsize=FS["tick"] + 1, pad=3)
    ax.legend(fontsize=FS["tick"] - 1, frameon=False, loc="lower left")
    path = fig_path("fig1c_crossings_jumps.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    for a_, panels in rows:
        for c_, v in enumerate(panels):
            Hm = heat[a_, c_]
            big = np.searchsorted(ae, 2, side="right") - 1
            print(f"  {a_:7s} {v[1]:38s}: amplitude >= 2 cells, coupling 0-15 / 15-30 / 30-45 deg: " + " / ".join(
                f"{np.nanmean(Hm[big:, k0:k1]):+.3f}" for k0, k1 in ((0, nphi // 3), (nphi // 3, 2 * nphi // 3),
                                                                     (2 * nphi // 3, nphi))))
    return path


def figure_help_hurt(steps=(5, 10), band=(8, 11), nb=20, nboot=500):
    """does the wormhole help the joint agents or hurt the factorized ones? (John: "It could be that the wormhole
    doesn't 'help' optimal / trained, but it hurts the Naive / ESN. But we'd need to see this clearly"; "plot the % in
    true cell in probability scale so it doesn't fall off at 0, and maybe a finer resolution of the x axis") ->
    fig1c_wormhole_help_hurt.png"""
    from main.utils import apply_fig_style, fig_path
    import fig1 as cf
    import plot_fig1c_evidence_geometry as eg
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    need = geodesic_need()
    D3, _, _ = eg.geometries()
    cells = np.array([(g, c) for g in range(R) for c in range(R)])
    lat = np.abs(cells[:, None] - cells[None]).sum(-1) == 1
    meta = np.load(os.path.join(mm.SCAN, "scan_meta.npz"))
    true_k = (meta["cells"][meta["cell_of"]] @ np.array([R, 1])).astype(int)
    maps = {}
    for a, _ in AGENTS:
        B = np.load(os.path.join(mm.SCAN, f"scan_{a}.npz"))["beliefs"][mm.ENVS].astype(np.float32)
        maps[a] = (B[..., :R].argmax(-1) * R + B[..., R:].argmax(-1)).astype(np.int16)
        del B

    def lattice_steps(D):                         # per environment / its median distance between lattice neighbours
        return D / np.array([np.median(D[i][lat]) for i in range(E)])[:, None, None]
    Ev = lattice_steps(D3["evidence"].astype(np.float64))
    Nv = lattice_steps(D3["Naive"].astype(np.float64))
    refl = (R - 1 - cells[:, 0]) * R + (R - 1 - cells[:, 1])
    Dv = D3["evidence"]
    closer = np.median([[np.sum(Dv[i, k] < Dv[i, k, refl[k]]) - 1 for k in range(R * R)] for i in range(E)], 0)
    partner = {0: 99, 99: 0, 9: 90, 90: 9}
    adjc = {0: (9, 90), 99: (9, 90), 9: (0, 99), 90: (0, 99)}
    cm_ = np.isin(true_k, list(partner))
    tk = true_k[cm_]
    pk = np.array([partner[k] for k in tk])
    a1, a2 = (np.array([adjc[k][j] for k in tk]) for j in (0, 1))
    tcs = {}
    for a, _ in AGENTS:
        m = maps[a][:, cm_].astype(int)
        g, c = m // R, m % R
        at = m == tk[None, :, None]
        nbr = (np.abs(g - (tk // R)[None, :, None]) <= 1) & (np.abs(c - (tk % R)[None, :, None]) <= 1) & ~at
        tcs[a] = {"truth": at.mean((0, 1)), "nbr": nbr.mean((0, 1)), "partner": (m == pk[None, :, None]).mean((0, 1)),
                  "adjacent": ((m == a1[None, :, None]) | (m == a2[None, :, None])).mean((0, 1))}
        tcs[a]["else"] = 1 - sum(tcs[a].values())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, E, (nboot, E))
    curves, rec = {}, {}
    for a, _ in AGENTS:
        mp = maps[a].astype(int)
        correct = mp[..., -1] == true_k[None]
        for t in steps:
            k = mp[..., t - 1]
            gd = np.linalg.norm(cells[k] - cells[true_k][None], axis=-1)
            nd = need[np.arange(E)[:, None], k, true_k[None]]
            inb = (gd >= band[0]) & (gd < band[1])
            ib = np.clip((nd * nb).astype(int), 0, nb - 1)
            s_ = np.stack([np.bincount(ib[i][inb[i]], correct[i][inb[i]], minlength=nb) for i in range(E)])
            c_ = np.stack([np.bincount(ib[i][inb[i]], minlength=nb) for i in range(E)])
            bs = s_[idx].sum(1) / np.maximum(c_[idx].sum(1), 1)
            curves[a, t] = (s_.sum(0) / np.maximum(c_.sum(0), 1), *np.percentile(bs, [5, 95], 0), c_.sum(0))
            if t == steps[0]:
                hit = mp == true_k[None, :, None]
                for lab, sel in (("low", inb & (nd < .2)), ("high", inb & (nd > .6))):
                    rec[a, lab] = (hit[sel].mean(0), int(sel.sum()))
    PM, GXm, X0 = 2.35, 1.05, .75
    W_ = X0 + 4 * PM + 3 * GXm + .9
    PH2, PH3 = 2.0, 2.1
    H_ = 1.0 + PM + 1.05 + PH2 + 1.1 + PH3 + .65
    fig = plt.figure(figsize=(W_, H_))
    fig.text(.5, 1 - .22 / H_, "Does the wormhole help the joint agents or hurt the factorized ones?  alias scan: 39 "
             "environments x 100 cells x 50 streams; corner truths: 4 corners x 50 streams x 39 environments = 7,800 "
             "streams per agent", ha="center", va="center", fontsize=FS["label"] + 1)
    y1 = H_ - 1.0 - PM

    def mapax(i, Mx, title, unit, vmax, marks=True, star=(0, 0)):
        ax = fig.add_axes([(X0 + i * (PM + GXm)) / W_, y1 / H_, PM / W_, PM / H_])
        im = ax.imshow(Mx.reshape(R, R).T, origin="lower", cmap="viridis", vmin=0, vmax=vmax,
                       extent=(-.5, 9.5, -.5, 9.5), interpolation="nearest")
        if marks:
            ax.plot(*star, "*", ms=13, color="w", mec="k", mew=.8)
        ax.set_xticks([0, 9]), ax.set_yticks([0, 9])
        ax.set_xlabel(r"$r_g$", fontsize=FS["tick"] + 1, labelpad=-6)
        ax.set_ylabel(r"$r_c$", fontsize=FS["tick"] + 1, labelpad=-6)
        ax.tick_params(labelsize=FS["tick"])
        ax.set_title(title, fontsize=FS["tick"] + 1, pad=4)
        cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
        cb.set_label(unit, fontsize=FS["tick"] - 1)
        cb.ax.tick_params(labelsize=FS["tick"] - 2)
        return ax
    ev0, nv0 = np.median(Ev[:, 0], 0), np.median(Nv[:, 0], 0)
    vm = max(ev0.max(), nv0.max())
    ax = mapax(0, ev0, "Evidence distance from (0, 0)\n" + r"$\sqrt{\mathrm{Jeffreys}}$ (median over environments)",
               "lattice steps", vm)
    ax.text(9, 9, f"{ev0[99]:.1f}", color="w", ha="center", va="center", fontsize=FS["tick"], fontweight="bold")
    for k_ in (9, 90):
        ax.text(*divmod(k_, R), f"{ev0[k_]:.1f}", color="k", ha="center", va="center", fontsize=FS["tick"])
    ax = mapax(1, nv0, "The Naive's view of the same\n(its averaged tables; median)", "lattice steps", vm)
    ax.text(9, 9, f"{nv0[99]:.1f}", color="w", ha="center", va="center", fontsize=FS["tick"], fontweight="bold")
    for k_ in (9, 90):
        ax.text(*divmod(k_, R), f"{nv0[k_]:.1f}", color="w", ha="center", va="center", fontsize=FS["tick"])
    nd0 = np.median(need[:, :, 0], 0)
    nd0[0] = np.nan                                                                # the truth itself
    mapax(2, nd0, "Wormhole need of the route to (0, 0)\n"
          + r"$1 - G_\mathrm{jumps} / G_\mathrm{grid}$ (median)", "fraction of the route saved", 1)
    mapax(3, closer, "Is the point reflection (9 - g, 9 - c) near in\nevidence? (median; centre: it is a neighbour)",
          "cells closer than it", 99, marks=False)
    y2 = y1 - 1.05 - PH2
    styles = (("truth", "the true corner", "k", "-"), ("nbr", "its 3 neighbours", "0.55", "-"),
              ("partner", "the partner corner (across the wormhole)", PURPLE, "-"),
              ("adjacent", "the 2 adjacent corners", ORANGE, "-"), ("else", "elsewhere", "0.75", (0, (3, 2))))
    tt = np.arange(1, T + 1)
    for i, (a, name) in enumerate(AGENTS):
        ax = fig.add_axes([(X0 + i * (PM + GXm)) / W_, y2 / H_, PM / W_, PH2 / H_])
        for key, lab, col, ls in styles:
            ax.plot(tt, tcs[a][key], color=col, lw=1.6, ls=ls, label=lab)
        ax.set_ylim(0, 1), ax.set_xlim(1, T), ax.set_xticks([1, 10, 20, 30])
        ax.set_xlabel("step $t$", fontsize=FS["tick"] + 1, labelpad=1)
        if i == 0:
            ax.set_ylabel("share of the streams", fontsize=FS["tick"] + 1)
        ax.tick_params(labelsize=FS["tick"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{name}, corner truths: where the MAP is\n" + r"$P(\hat{k}_t \in$ set$)$",
                     fontsize=FS["tick"] + 1, pad=3, color=cf.COL[a])
        if i == 2:
            ax.legend(fontsize=FS["tick"] - 1, frameon=False, loc="upper center", ncol=1)
    y3 = y2 - 1.1 - PH3
    mid = (np.arange(nb) + .5) / nb
    for j, t in enumerate(steps):
        ax = fig.add_axes([(X0 + j * (PM + GXm)) / W_, y3 / H_, PM / W_, PH3 / H_])
        for a, name in AGENTS:
            m_, lo_, hi_, n_ = curves[a, t]
            ok = n_ >= 30
            mm_, ll_, hh_ = (np.where(ok, v, np.nan) for v in (m_, lo_, hi_))
            ax.fill_between(mid, ll_, hh_, color=cf.COL[a], alpha=.2, lw=0)
            ax.plot(mid, mm_, color=cf.COL[a], lw=1.5, marker="o", ms=2.2, label=name,
                    ls=(0, (3, 2)) if a == "Naive" else "-")
        ax.set_ylim(0, 1), ax.set_xlim(0, 1)
        ax.set_xlabel("wormhole need of the route (20 bins)", fontsize=FS["tick"] + 1, labelpad=1)
        if j == 0:
            ax.set_ylabel(r"$P(\hat{k}_{30} = k^\ast)$", fontsize=FS["tick"] + 1)
        ax.tick_params(labelsize=FS["tick"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"MAP {band[0]} - {band[1]} cells from the truth at step {t}:\n"
                     r"how many end on the true cell, $P(\hat{k}_{30} = k^\ast)$", fontsize=FS["tick"] + 1, pad=3)
        if j == 0:
            ax.legend(fontsize=FS["tick"] - 1, frameon=False, loc="center left")
    ax = fig.add_axes([(X0 + 2 * (PM + GXm)) / W_, y3 / H_, (2 * PM + GXm) / W_, PH3 / H_])
    for a, name in AGENTS:
        for lab, ls in (("low", "-"), ("high", (0, (3, 2)))):
            v, n_ = rec[a, lab]
            ax.plot(tt[steps[0] - 1:], v[steps[0] - 1:], color=cf.COL[a], lw=1.5, ls=ls,
                    label=f"{name}: need {'$<$ 0.2' if lab == 'low' else '$>$ 0.6'} (n {n_:,})")
    ax.set_ylim(0, 1), ax.set_xlim(steps[0], T), ax.set_xticks([steps[0], 10, 20, 30])
    ax.set_xlabel("step $t$", fontsize=FS["tick"] + 1, labelpad=1)
    ax.set_ylabel(r"$P(\hat{k}_t = k^\ast)$", fontsize=FS["tick"] + 1)
    ax.tick_params(labelsize=FS["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"the same beliefs (MAP {band[0]} - {band[1]} cells from the truth at step {steps[0]}) over time: "
                 r"on the true cell, $P(\hat{k}_t = k^\ast)$" + "\nsolid: the shortest evidence route walks the grid "
                 "(need $<$ 0.2); dashed: it takes the wormhole (need $>$ 0.6)", fontsize=FS["tick"] + 1, pad=3)
    ax.legend(fontsize=FS["tick"] - 2, frameon=False, loc="upper left", ncol=2)
    path = fig_path("fig1c_wormhole_help_hurt.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"corner (0, 0) in lattice steps, evidence / Naive's view: partner {ev0[99]:.1f} / {nv0[99]:.1f}, adjacent "
          f"{ev0[90]:.1f} / {nv0[90]:.1f}, {ev0[9]:.1f} / {nv0[9]:.1f}; reflection closer-than counts: corners "
          f"{closer[[0, 9, 90, 99]]}, next along the border {closer[[1, 10, 8, 19]]}")
    for a, name in AGENTS:
        print(f"{name:12s} corner truths at steps 2 / 5 / 10 / 30: " + "; ".join(
            f"{key} " + " / ".join(f"{tcs[a][key][t - 1]:.3f}" for t in (2, 5, 10, 30)) for key, *_ in styles))
        for t in steps:
            m_, lo_, hi_, n_ = curves[a, t]
            print(f"    step {t:2d}, P(end on the truth) by need bin (n >= 30): " + ", ".join(
                f"{mid[b]:.3f}: {m_[b]:.3f}" for b in range(nb) if n_[b] >= 30))
        for lab in ("low", "high"):
            v, n_ = rec[a, lab]
            print(f"    recovery ({lab} need, n {n_}): steps 6 / 10 / 15 / 30: "
                  + " / ".join(f"{v[t - 1]:.3f}" for t in (6, 10, 15, 30)))
    return path


LEVELS = ((0, 15, "#b35806"), (15, 30, "#f1a340"), (30, 40, "#998ec3"), (40, 45, "#542788"))   # coupling levels


def figure_split_lags(lags=np.arange(-4, 11), nboot=500, seed=0, floor=1e-3, by_coupling=False):
    """the split pairs aligned on the jump (John 10-28: "can you show the curves aligned on jump for that FR and
    entanglement of the 743 split pairs") -> fig1c_split_pairs_lags.png: the stream that jumps and its partner;
    by_coupling (John: "can you please split the jumps by various levels of couplings?"): the stream that jumps, by the
    coupling angle of its jump (LEVELS, degrees) -> fig1c_split_pairs_lags_coupling.png"""
    import plot_fig1c_split_movie as sm
    from main.utils import apply_fig_style, fig_path
    from matplotlib.lines import Line2D
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    Z = np.load(sm.DATA)
    rows = np.array([mm.ENVS.index(int(e)) for e in Z["env"]])
    i, j, ts = Z["i"].astype(int), Z["j"].astype(int), Z["ts"].astype(int)
    n = len(rows)
    D = jumps()["Joint"]                                                          # (E, 5000, T - 1, 2)
    move = [np.hypot(*(D[rows, s, ts - 1] + D[rows, s, ts]).T) for s in (i, j)]   # over the split, t_s - 1 -> + 1
    jumper, partner = np.where(move[0] >= move[1], i, j), np.where(move[0] >= move[1], j, i)
    k = ts + (np.hypot(*D[rows, jumper, ts].T) > np.hypot(*D[rows, jumper, ts - 1].T))   # lag 0: just after the jump
    amp_j, phi_j = polar(D[rows, jumper, k - 1])                                  # the jump itself (into step k)
    step = np.full((E, R * R * NS, T), np.nan)
    step[..., 1:] = np.hypot(D[..., 0], D[..., 1])                                # the move into step t (cells)
    del D
    meta = np.load(os.path.join(mm.SCAN, "scan_meta.npz"))
    V = {"step": step, "fr": mm.fr_both(), "rev": meta["rev"][mm.ENVS].astype(np.float64)}
    tt = k[:, None] + lags[None, :]                                               # (n, L) absolute steps
    assert tt.min() >= 1 and tt.max() < T
    if by_coupling:
        groups = []
        for lo, hi, col in LEVELS:
            m = (phi_j >= lo) & ((phi_j < hi) if hi < 45 else (phi_j <= hi))
            groups.append((f"{lo} - {hi}" + r"$^\circ$" + f": {m.sum()} jumps ({np.median(amp_j[m]):.1f} cells)",
                           col, jumper, m))
    else:
        groups = [("the stream that jumps", PURPLE, jumper, np.ones(n, bool)),
                  ("its partner (stays)", ORANGE, partner, np.ones(n, bool))]
    rng = np.random.default_rng(seed)
    Wb = np.stack([np.bincount(s_, minlength=E) for s_ in rng.integers(0, E, (nboot, E))])[:, rows]   # (nboot, n)

    def median_band(X, m):
        idx = np.flatnonzero(m)
        boots = [np.median(X[r_], 0) for r_ in (np.repeat(idx, w_[idx]) for w_ in Wb) if len(r_)]
        return np.median(X[idx], 0), np.percentile(boots, [5, 95], 0)

    def mean_band(X, m):
        Wm = Wb[:, m]
        ok = Wm.sum(1) > 0
        return X[m].mean(0), np.percentile(Wm[ok] @ X[m] / Wm[ok].sum(1, keepdims=True), [5, 95], 0)

    res = {}
    for key, v in V.items():
        dep = departures(np.log10(np.maximum(v, floor)))                          # log10 minus the cell's typical
        for g, (_, _, s, m) in enumerate(groups):
            res[key, g, "raw"] = median_band(v[rows[:, None], s[:, None], tt], m)
            res[key, g, "dep"] = mean_band(dep[rows[:, None], s[:, None], tt], m)
        res[key, "all"] = np.array([np.median(v[rows, :, tt[:, l_]]) for l_ in range(len(lags))])
        print(f"{key}: below the floor {floor:g} at the steps used {np.mean(v[rows, :, 1:] < floor):.4f}")
    PW, PH, GX, GY, X0 = 2.75, 1.9, .85, 1.25, .8
    Y0 = 2.1 if by_coupling else 1.55
    W_ = X0 + 3 * PW + 2 * GX + .3
    H_ = Y0 + 2 * PH + GY + .55
    fig = plt.figure(figsize=(W_, H_))
    fig.text(.5, 1 - .27 / H_, f"The {n} split pairs, aligned on the jump" + (
        r", by the jump's coupling angle $\varphi$ (0$^\circ$: one variable moves, 45$^\circ$: both)" if by_coupling
        else ": the Optimal's belief, factorization regret and entanglement"), ha="center", va="center",
        fontsize=FS["label"] + 2)
    fig.text(.5, 1 - .62 / H_, f"alias scan, {n} pairs in {len(set(rows))} environments (the split movie's pairs: "
             r"two streams of different cells together through $t_s$, $\geq$ 3 cells further apart two steps later)"
             "\nthe stream that jumps = the one whose posterior mean moves more over the split; step 0 = its belief "
             r"just after its largest one-step move ($t_s$ or $t_s + 1$); bands: 5 - 95% over environments resampled"
             + ("\nlines: the stream that jumps, grouped by the angle of that move (its partner: the jumper-vs-partner "
                "version of this figure)" if by_coupling else ""),
             ha="center", va="center", fontsize=FS["tick"])
    cols = (("step", "How far the Optimal's belief moves", r"$|\Delta\mu_t|$, its posterior mean", "cells"),
            ("fr", "Factorization regret",
             r"$\mathrm{FR}_t = \frac{1}{2}\Sigma_v\,\mathrm{KL}_\mathrm{sym}(b^\mathrm{Opt}_v, b^\mathrm{Naive}_v)$",
             "nats"),
            ("rev", "Entanglement", r"$\mathrm{rev}_t$, of the update into step $t$", "nats"))
    for r_, kind in enumerate(("raw", "dep")):
        for c_, (key, name, formula, unit) in enumerate(cols):
            x, y = X0 + c_ * (PW + GX), H_ - Y0 - PH - r_ * (PH + GY)
            ax = fig.add_axes([x / W_, y / H_, PW / W_, PH / H_])
            if kind == "raw":
                ax.plot(lags, res[key, "all"], color="0.5", lw=1.3, ls=(0, (3, 2)))
            else:
                ax.axhline(0, color="0.5", lw=1.1, ls=(0, (3, 2)))
            for g, (lab, col, _, _) in enumerate(groups):
                m_, band = res[key, g, kind]
                ax.fill_between(lags, *band, color=col, alpha=.22 if not by_coupling else .16, lw=0)
                ax.plot(lags, m_, color=col, lw=1.7, marker="o", ms=2.6, label=lab)
            ax.axvline(0, color="k", lw=.6, ls=":")
            ax.set_xlim(lags[0], lags[-1]), ax.set_xticks([-4, -2, 0, 2, 4, 6, 8, 10])
            ax.set_xlabel("steps from the jump", fontsize=FS["tick"] + 1, labelpad=1)
            ax.tick_params(labelsize=FS["tick"])
            ax.spines[["top", "right"]].set_visible(False)
            if kind == "raw":
                ax.set_ylim(bottom=0)
                ax.set_ylabel(f"median ({unit})", fontsize=FS["tick"] + 1)
                ax.set_title(f"{name}\n{formula}", fontsize=FS["tick"] + 1, pad=3)
            else:
                ax.set_ylabel(r"log$_{10}$, minus the cell's typical", fontsize=FS["tick"] + 1)
                ax.set_title(f"{name}, minus the cell's typical\n"
                             r"$\log_{10} x - \langle \log_{10} x \rangle_\mathrm{cell}$ (mean over the pairs)",
                             fontsize=FS["tick"] + 1, pad=3)
    keys_ = [Line2D([], [], color=col, lw=1.7, marker="o", ms=2.6, label=lab) for lab, col, _, _ in groups]
    keys_.append(Line2D([], [], color="0.5", lw=1.3, ls=(0, (3, 2)), label="top row: all streams of the same "
                        "environments at the same steps; bottom row: typical (the same cell's 50 streams)"
                        if not by_coupling else "top: all streams, same steps; bottom: the cell's typical"))
    fig.legend(handles=keys_, loc="center", bbox_to_anchor=(.5, 1 - (1.3 if by_coupling else 1.02) / H_),
               ncol=3, frameon=False, fontsize=FS["tick"], handlelength=2.6,
               columnspacing=2.2)
    path = fig_path("fig1c_split_pairs_lags%s.png" % ("_coupling" if by_coupling else ""))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"{n} pairs; distinct jumping streams {len(set(zip(rows, jumper)))}, partners {len(set(zip(rows, partner)))};"
          f" jump at t_s {np.mean(k == ts):.2f}, at t_s + 1 {np.mean(k == ts + 1):.2f}")
    for key in V:
        print(f"{key:4s} all streams median: " + " ".join(f"{v:.2f}" for v in res[key, "all"]))
        for g, (lab, _, _, m) in enumerate(groups):
            for kind in ("raw", "dep"):
                m_, band = res[key, g, kind]
                print(f"     {lab[:22]:22s} (n {m.sum()}) {kind}: "
                      + " ".join(f"{L:+d} {v:+.2f}" for L, v in zip(lags, m_))
                      + f"  (band at 0: {band[0][lags == 0][0]:+.2f} .. {band[1][lags == 0][0]:+.2f})")
    return path


def _heat_bins(amp, phi, ae, nphi):
    """amplitude / coupling of every jump -> its bin (amplitude bin x coupling bin), -1 outside"""
    ia = np.searchsorted(ae, amp, side="right") - 1
    ok = (ia >= 0) & (ia < len(ae) - 1)
    return np.where(ok, ia * nphi + np.minimum((phi * nphi / 45).astype(int), nphi - 1), -1)


def _heat_mean(B, V, shape, min_n):
    """the mean of V over each bin's jumps; NaN where fewer than min_n"""
    ok = (B >= 0) & np.isfinite(V)
    n_ = np.bincount(B[ok], minlength=shape[0] * shape[1])
    m_ = np.bincount(B[ok], V[ok], minlength=shape[0] * shape[1]) / np.maximum(n_, 1)
    return np.where(n_ >= min_n, m_, np.nan).reshape(shape)


def figure_before(na=40, nphi=30, amin=.01, amax=12., min_n=50, big=2., lags=(-3, -2, -1, 0, 1), nboot=500):
    """each measure one step BEFORE the jump beside the jump's own step, by the jump's amplitude x coupling (John
    10-28: "look at how entanglement the update before binds amplitude and coupling more than the entanglement at the
    jump. Maybe we should be looking at measures on the step before the jump?") -> fig1c_jumps_before.png"""
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    J = jumps()
    pol = {a: polar(J[a]) for a in ("Joint", "Trained")}
    del J
    M = measures()
    dep = {k: departures(M[k][0]) for k in ("rev", "fr", "err", "pr", "band_x", "dW")}
    lg = {k: M[k][1] for k in dep}
    del M
    ae, pe = np.geomspace(amin, amax, na + 1), np.linspace(0, 45, nphi + 1)
    Bs = {a: _heat_bins(*pol[a], ae, nphi) for a in pol}
    panels = (("rev", "Joint", "entanglement", lambda s: r"$\mathrm{rev}_{%s}$" % s),
              ("fr", "Joint", "factorization regret", lambda s: r"$\mathrm{FR}_{%s}$ (both variables)" % s),
              ("err", "Trained", "error", lambda s: r"$\delta_{%s}$ vs the Optimal" % s),
              ("pr", "Trained", "effective dim., activity", lambda s: r"$\mathrm{PR}(u_{%s})$" % s),
              ("band_x", "Trained", "cross-band share", lambda s: r"of $\Delta W_{%s}$" % s),
              ("dW", "Trained", r"$\Delta W$ energy", lambda s: r"$\|\Delta W_{%s}\|^2$" % s))
    who = {"Joint": "Optimal", "Trained": "Trained RNN"}
    heat = {}
    for key, a, *_ in panels:
        heat[key, -1] = _heat_mean(Bs[a], dep[key][..., :-1], (na, nphi), min_n)    # at t - 1, jump into t
        heat[key, 0] = _heat_mean(Bs[a], dep[key][..., 1:], (na, nphi), min_n)      # at t
    nbp = 9                                                                         # coupling bins of 5 degrees
    lines = {}
    for key, a, *_ in panels:
        amp, phi = pol[a]
        e_, s_, j_ = np.nonzero(amp >= big)
        ph = phi[e_, s_, j_]
        b = np.minimum((ph / 5).astype(int), nbp - 1)
        for L in lags:
            t_ = j_ + 1 + L
            ok = (t_ >= 0) & (t_ < T)
            v = dep[key][e_[ok], s_[ok], t_[ok]]
            fin = np.isfinite(v)
            ss, cc = np.zeros((E, nbp)), np.zeros((E, nbp))
            np.add.at(ss, (e_[ok][fin], b[ok][fin]), v[fin])
            np.add.at(cc, (e_[ok][fin], b[ok][fin]), 1)
            lines[key, L] = boot(ss, cc, n=nboot)
        lines[key, "n"] = len(e_)
    PW, PH, GX, GY, X0, TOP, GY4, BOT = 2.45, 1.95, 1.35, 1.12, .75, 1.3, 1.35, .7
    W_ = X0 + 4 * PW + 3 * GX + .8
    G6 = .62
    P6 = (4 * PW + 3 * GX - 5 * G6) / 6
    H_ = TOP + 3 * PH + 2 * GY + GY4 + PH + BOT
    fig = plt.figure(figsize=(W_, H_))
    fig.text(.5, 1 - .25 / H_, r"The step before the jump: each measure at $t - 1$ (left of each pair) and at the jump "
             r"$t$ (right), by the jump's amplitude $|\Delta\mu_t|$ (x) and coupling angle $\varphi_t$ (y; 0$^\circ$: "
             r"one variable moves, 45$^\circ$: both)", ha="center", va="center", fontsize=FS["label"] + 1)
    fig.text(.5, 1 - .62 / H_, "alias scan: 39 environments x 5,000 streams x 29 jumps per agent; rows 1 - 3: the "
             f"mean over each bin's jumps (at least {min_n}; grey: fewer), minus typical (the same cell's 50 streams "
             "at the same step)\nthe Optimal's measures on its own jumps, the Trained RNN's on its read-out's jumps; "
             "dotted: 2 cells", ha="center", va="center", fontsize=FS["tick"])

    def hm(ax, Mx, unit):
        ax.set_facecolor("0.85")
        lim = np.nanpercentile(np.abs(Mx), 99)
        im = ax.pcolormesh(ae, pe, Mx.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="flat", rasterized=True)
        ax.set_xscale("log"), ax.set_xlim(amin, amax), ax.set_ylim(0, 45), ax.set_yticks([0, 15, 30, 45])
        ax.axvline(2, color="k", lw=.6, ls=":")
        ax.set_xlabel(r"amplitude $|\Delta\mu_t|$ (cells)", fontsize=FS["tick"] + 1, labelpad=1)
        ax.set_ylabel(r"coupling $\varphi_t$ (degrees)" + "\n0: one variable moves, 45: both", fontsize=FS["tick"] + 1)
        ax.tick_params(labelsize=FS["tick"])
        cb = fig.colorbar(im, ax=ax, fraction=.06, pad=.02)
        cb.set_label(unit, fontsize=FS["tick"] - 1)
        cb.ax.tick_params(labelsize=FS["tick"] - 2)

    for n_, (key, a, name, form) in enumerate(panels):
        r_, c0 = divmod(n_, 2)
        unit = ("log$_{10}$ " if lg[key] else "") + mm.UNITS[key] + ", minus typical"
        for q, (L, lab, sub) in enumerate(((-1, "the step before", "t-1"), (0, "at the jump", "t"))):
            x, y = X0 + (2 * c0 + q) * (PW + GX), H_ - TOP - PH - r_ * (PH + GY)
            ax = fig.add_axes([x / W_, y / H_, PW / W_, PH / H_])
            hm(ax, heat[key, L], unit)
            ax.set_title(f"{who[a]}: {name}, {lab}\n{form(sub)}", fontsize=FS["tick"] + 1, pad=3)
    y4 = H_ - TOP - 3 * PH - 2 * GY - GY4 - PH
    fig.text(.5, (y4 + PH + .82) / H_, rf"row 4: jumps of at least {big:g} cells, by their coupling angle: each "
             r"measure minus typical at $t - 3$ ... $t + 1$ (the jump arrives at $t$); bands: 5 - 95% over "
             "environments resampled", ha="center", va="center", fontsize=FS["label"])
    lag_style = {-3: ("#c6dbef", "-"), -2: ("#6baed6", "-"), -1: ("#2171b5", "-"), 0: ("k", "-"),
                 1: ("0.55", (0, (3, 2)))}
    mid = (np.arange(nbp) + .5) * 5
    for n_, (key, a, name, form) in enumerate(panels):
        ax = fig.add_axes([(X0 + n_ * (P6 + G6)) / W_, y4 / H_, P6 / W_, PH / H_])
        for L in lags:
            m_, band = lines[key, L]
            col, ls = lag_style[L]
            ax.fill_between(mid, *band, color=col, alpha=.18, lw=0)
            ax.plot(mid, m_, color=col, ls=ls, lw=1.6, marker="o", ms=2.2,
                    label={0: r"$t$ (the jump)", 1: r"$t + 1$"}.get(L, rf"$t - {-L}$"))
        ax.axhline(0, color="0.6", lw=.8, ls=":")
        ax.set_xlim(0, 45), ax.set_xticks([0, 15, 30, 45])
        ax.set_xlabel(r"coupling $\varphi_t$ (degrees)", fontsize=FS["tick"] + 1, labelpad=1)
        ax.set_ylabel(("log$_{10}$ " if lg[key] else "") + mm.UNITS[key] + ", minus typical", fontsize=FS["tick"] + 1)
        ax.tick_params(labelsize=FS["tick"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{who[a]}: {name}\n({lines[key, 'n']:,} jumps)", fontsize=FS["tick"] + 1, pad=3)
        if n_ == len(panels) - 1:
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=FS["tick"] - 1, frameon=False, loc="lower right", ncol=2)
    path = fig_path("fig1c_jumps_before.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    for key, a, name, _ in panels:
        print(f"{who[a]:12s} {name[:24]:24s} jumps >= {big:g} cells (n {lines[key, 'n']:,}), minus typical at "
              "coupling 0-15 / 15-30 / 30-45 deg: " + "; ".join(
                  f"lag {L:+d} " + " / ".join(f"{np.nanmean(lines[key, L][0][sl]):+.3f}"
                                              for sl in (slice(0, 3), slice(3, 6), slice(6, 9))) for L in lags))
    return path


def figure_amplitude_check(groups=((1, 3), (4, 8), (9, 15), (16, 29)), amin=.003, amax=12., nb=32, min_n=200):
    """is amplitude ~0.1 a transition? (John 10-28: "doesnt it seem like there is a meaningful phase transition at
    amplitude .1? or is that due to the data processing?") -> fig1c_jumps_amplitude_check.png: the measures against
    the jump's amplitude, raw and minus typical, separately for jumps early and late in the episode"""
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    J = jumps()
    amp = {a: polar(J[a])[0] for a in ("Joint", "Trained")}                     # (E, 5000, T - 1): the move into t
    del J
    M = measures()
    keys = (("rev", "Joint", "Entanglement of the jump's update", r"$\mathrm{rev}_t$ (Optimal)"),
            ("err", "Trained", "Error after the jump", r"$\delta_t$ vs the Optimal (Trained RNN)"),
            ("pr", "Trained", "Effective dim., activity", r"$\mathrm{PR}(u_t)$ (Trained RNN)"),
            ("band_x", "Trained", "Cross-band share", r"of $\Delta W_t$ (Trained RNN)"),
            ("dW", "Trained", r"$\Delta W$ energy", r"$\|\Delta W_t\|^2$ (Trained RNN)"))
    edges = np.geomspace(amin, amax, nb + 1)
    mid = np.sqrt(edges[:-1] * edges[1:])
    gcol = plt.get_cmap("viridis")(np.linspace(.82, .05, len(groups)))
    glab = [f"steps {lo + 1} - {hi + 1}" for lo, hi in groups]                     # arrival steps, 1-based
    curves, meds = {}, {}
    for key, a, *_ in keys:
        V, lg = M[key]
        D = departures(V)
        for g, (lo, hi) in enumerate(groups):
            A = amp[a][..., lo - 1:hi].ravel()                                    # jumps into steps lo .. hi
            b = np.digitize(A, edges) - 1
            ok = (b >= 0) & (b < nb)
            for q, X in (("raw", V[..., lo:hi + 1].ravel()), ("dep", D[..., lo:hi + 1].ravel())):
                okq = ok & np.isfinite(X)
                n_ = np.bincount(b[okq], minlength=nb)
                curves[key, g, q] = np.where(n_ >= min_n, np.bincount(b[okq], X[okq], minlength=nb)
                                             / np.maximum(n_, 1), np.nan)
            meds[key, g] = np.median(A)
            dq = curves[key, g, "dep"]
            pos = np.flatnonzero(np.isfinite(dq) & (dq > 0) & (mid > meds[key, g] / 20))
            curves[key, g, "zero"] = mid[pos[0]] if len(pos) else np.nan
        del D
    hist = []
    for lo, hi in groups:
        h_, _ = np.histogram(amp["Joint"][..., lo - 1:hi], edges)
        hist.append(h_ / h_.sum())
    med_step = np.median(amp["Joint"].reshape(-1, T - 1), 0)
    PW, PH, GX, GY, X0, TOP = 2.45, 1.95, .95, 1.15, .8, 1.25
    W_ = X0 + 6 * PW + 5 * GX + .3
    H_ = TOP + 2 * PH + GY + .7
    fig = plt.figure(figsize=(W_, H_))
    fig.text(.5, 1 - .25 / H_, "Is amplitude 0.1 a transition? 'Minus typical' puts zero at each step's typical jump, "
             "which shrinks from 1.2 cells (step 2) to 0.07 (step 30): the white band moves with the step",
             ha="center", va="center", fontsize=FS["label"] + 1)
    fig.text(.5, 1 - .6 / H_, "alias scan: 39 environments x 5,000 streams; jumps grouped by the step they arrive at; "
             f"lines: the mean per amplitude bin (at least {min_n} jumps); dotted verticals: each group's median jump; "
             "top: raw values, bottom: minus typical (the same cell's 50 streams at the same step)",
             ha="center", va="center", fontsize=FS["tick"])

    def axes(r_, c_):
        x, y = X0 + c_ * (PW + GX), H_ - TOP - PH - r_ * (PH + GY)
        ax = fig.add_axes([x / W_, y / H_, PW / W_, PH / H_])
        ax.tick_params(labelsize=FS["tick"])
        ax.spines[["top", "right"]].set_visible(False)
        return ax

    ax = axes(0, 0)
    for g in range(len(groups)):
        ax.plot(mid, hist[g], color=gcol[g], lw=1.7, label=glab[g])
        ax.axvline(np.median(amp["Joint"][..., groups[g][0] - 1:groups[g][1]]), color=gcol[g], lw=1, ls=":")
    ax.set_xscale("log"), ax.set_xlim(amin, amax), ax.set_ylim(0, 1.55 * max(h_.max() for h_ in hist))
    ax.set_xlabel(r"amplitude $|\Delta\mu_t|$ (cells)", fontsize=FS["tick"] + 1, labelpad=1)
    ax.set_ylabel("share of the jumps per bin", fontsize=FS["tick"] + 1)
    ax.set_title("How big the jumps are\n(the Optimal; one peak per group, no gap)", fontsize=FS["tick"] + 1, pad=3)
    ax.legend(fontsize=FS["tick"] - 1, frameon=False, loc="upper left")
    ax = axes(1, 0)
    ax.plot(np.arange(2, T + 1), med_step, color="k", lw=1.7, marker="o", ms=2.4)
    ax.axhline(.1, color="0.5", lw=.8, ls=":")
    ax.set_yscale("log"), ax.set_xlim(1, T), ax.set_xticks([2, 10, 20, 30])
    ax.set_xlabel("step $t$", fontsize=FS["tick"] + 1, labelpad=1)
    ax.set_ylabel(r"median $|\Delta\mu_t|$ (cells)", fontsize=FS["tick"] + 1)
    ax.set_title("The typical jump by step\n(the Optimal; dotted: 0.1 cells)", fontsize=FS["tick"] + 1, pad=3)
    for c_, (key, a, name, form) in enumerate(keys, start=1):
        lg = M[key][1]
        for r_, q in enumerate(("raw", "dep")):
            ax = axes(r_, c_)
            for g in range(len(groups)):
                ax.plot(mid, curves[key, g, q], color=gcol[g], lw=1.7)
                ax.axvline(meds[key, g], color=gcol[g], lw=1, ls=":")
            if q == "dep":
                ax.axhline(0, color="0.5", lw=.8)
            ax.set_xscale("log"), ax.set_xlim(amin, amax)
            ax.set_xlabel(r"amplitude $|\Delta\mu_t|$ (cells)", fontsize=FS["tick"] + 1, labelpad=1)
            ax.set_ylabel(("log$_{10}$ " if lg else "") + mm.UNITS[key] + (", minus typical" if q == "dep" else ""),
                          fontsize=FS["tick"] + 1)
            ax.set_title(f"{name}{', minus typical' if q == 'dep' else ''}\n{form}", fontsize=FS["tick"] + 1, pad=3)
    path = fig_path("fig1c_jumps_amplitude_check.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print("median jump by step (Optimal): " + " ".join(f"{v:.2f}" for v in med_step))
    for key, *_ in keys:
        print(f"{key:7s} medians / where minus-typical turns positive, by group: " + "; ".join(
            f"{glab[g]}: {meds[key, g]:.3f} / {curves[key, g, 'zero']:.3f}" for g in range(len(groups))))
    return path


def movie(dpi=110, ns_show=4, flash=1.25, continuous=False, combined=False):
    """particles and grids by crossings, the four agents (columns): row 1, the first ns_show streams of every cell in
    all 39 environments at the agent's posterior means, grey, flashing red / blue (coupled, along (0, 0) - (9, 9) /
    (0, 9) - (9, 0)) or orange (uncoupled) as a crossing arrives (fading over +- flash steps around the move); rows 2 -
    3, every environment's grid of cell means (its 50 streams) on top of each other, each edge coloured by the share of
    its two cells' streams that have made a coupled (row 2) or uncoupled (row 3) crossing so far.
    continuous (John: "a version of movie crossing gifs with the continuous coupling measure"): a crossing = the larger
    flow >= CYCLE, its flash colour running from orange (coupling 0) to red / blue (coupling 1, by direction); rows 2 -
    3, the crossings so far per stream weighted by coupling / by 1 - coupling (the cell's mean over its 50 streams)
    -> fig1c_movie_crossings_continuous.gif
    combined (John: "maybe we want a version where we combine both coupled and uncoupled crossings. It looks like where
    crossings end up is a good explainer of the Naive and ESN tangling"): row 2, the same particles, magenta once they
    have made any crossing (coupled or uncoupled) so far: where crossing streams end up; row 3, the grids by the share
    of each cell's streams with any crossing so far -> fig1c_movie_crossings_combined.gif"""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba
    P = primitives()
    r = np.arange(R, dtype=np.float32)
    show = np.arange(5000) % NS < ns_show
    data = {}
    for a, _ in AGENTS:
        B = np.load(os.path.join(mm.SCAN, f"scan_{a}.npz"))["beliefs"][mm.ENVS].astype(np.float32)
        X = np.stack([B[..., :R] @ r / B[..., :R].sum(-1), B[..., R:] @ r / B[..., R:].sum(-1)], -1)
        del B
        if continuous:
            cr, kap, dirn = coupling(P[a][0])
            K = at_steps(cr.astype(np.int8))
            KP, D = at_steps(kap.astype(np.float32)), at_steps(dirn.astype(np.int8))
            cum = [np.cumsum(K * w_, axis=2).reshape(E, R * R, NS, T).mean(2) for w_ in (KP, 1 - KP)]
        else:
            kind, dirn = labels(P[a][0])
            K, D = at_steps(kind), at_steps(dirn)
            KP = np.zeros(K.shape, np.float32)
            cum = [np.maximum.accumulate(K == k, axis=2).reshape(E, R * R, NS, T).mean(2) for k in (1, 2)]
        if combined:                                                              # any crossing so far
            cum = [np.maximum.accumulate(K > 0, axis=2).reshape(E, R * R, NS, T).mean(2)]
        data[a] = dict(Xs=X[:, show], Xc=X.reshape(E, R * R, NS, T, 2).mean(2), K=K[:, show].reshape(-1, T),
                       D=D[:, show].reshape(-1, T), KP=KP[:, show].reshape(-1, T), cum=cum,
                       crossed=np.maximum.accumulate(K[:, show] > 0, axis=2).reshape(-1, T))
        print(a, "ready", flush=True)
    edges, _ = mm._grid()
    vmax = [float(f"{max(np.percentile(d['cum'][k][..., -1], 98) for d in data.values()):.2g}")
            for k in range(len(next(iter(data.values()))["cum"]))]
    grey = np.array(to_rgba("0.8"))
    MAGENTA = np.array(to_rgba("#d01c8b"))
    ev_col = {(1, 1): np.array(to_rgba("#b2182b")), (1, -1): np.array(to_rgba("#2166ac")),
              (2, 0): np.array(to_rgba("#e08214"))}
    cmap = plt.get_cmap("viridis")
    PW, GX, GY, X0, YTOP = 2.4, .3, .62, .75, 1.25
    W, H = X0 + 4 * PW + 3 * GX + 2.35, YTOP + 3 * PW + 2 * GY + .3
    taus = mm._taus()
    with plt.rc_context(mm.RC):
        fig = plt.figure(figsize=(W, H), dpi=dpi, facecolor="white")
        arts = []
        for c_, (a, name) in enumerate(AGENTS):
            x = X0 + c_ * (PW + GX)
            row = []
            for r_ in range(3):
                y = H - YTOP - PW - r_ * (PW + GY)
                ax = fig.add_axes([x / W, y / H, PW / W, PW / H])
                ax.set_xlim(-.6, 9.6), ax.set_ylim(-.6, 9.6), ax.set_aspect("equal")
                ax.set_xticks([0, 9]), ax.set_yticks([0, 9])
                ax.tick_params(labelsize=8, length=2, pad=1)
                if r_ == 0 or (combined and r_ == 1):
                    if r_ == 0:
                        ax.set_title(name, fontsize=12, pad=5)
                    n = len(data[a]["K"])
                    art = ax.scatter(*data[a]["Xs"][:, :, 0].reshape(-1, 2).T, s=(72 / dpi * 2) ** 2, marker="s",
                                     c=np.tile(grey, (n, 1)), lw=0, antialiased=False)
                else:
                    art = ax.add_collection(LineCollection([], linewidths=72 / dpi, antialiased=False))
                if c_ == 0:
                    ax.set_ylabel(r"$E[r_c]$", labelpad=-4)
                if r_ == 2:
                    ax.set_xlabel(r"$E[r_g]$", labelpad=-4)
                row.append(art)
            arts.append((a, row))
        row_labs = (("particles: crossing now", "grids: crossings so far per\nstream, weighted by coupling",
                     "grids: crossings so far per\nstream, weighted by 1 - coupling") if continuous else
                    ("particles: crossing now", "particles: where the streams\nwith a crossing so far are",
                     "grids: share of each cell's streams\nwith any crossing so far") if combined else
                    ("particles: crossing now", "grids: share of each cell's\nstreams with a coupled crossing so far",
                     "grids: share with an\nuncoupled crossing so far"))
        for r_, lab in enumerate(row_labs):
            fig.text(.3 / W, (H - YTOP - PW / 2 - r_ * (PW + GY)) / H, lab, rotation=90, ha="center", va="center",
                     fontsize=10)
        for k, r_ in (((0, 2),) if combined else ((0, 1), (1, 2))):
            y = H - YTOP - PW - r_ * (PW + GY)
            mm._colorbar(fig, cmap, 0, vmax[k], [(X0 + 4 * PW + 3 * GX + .3) / W, y / H, .12 / W, PW / H],
                         "crossings per stream" if continuous else "share of the cells' streams")
        lx = X0 + 4 * PW + 3 * GX + .25
        if continuous:
            for i_, (end, lab) in enumerate(((ev_col[1, 1], "along (0, 0) - (9, 9)"),
                                             (ev_col[1, -1], "along (0, 9) - (9, 0)"))):
                yy = H - YTOP - .3 - i_ * .62
                gax = fig.add_axes([lx / W, (yy - .07) / H, .9 / W, .14 / H])
                u = np.linspace(0, 1, 64)[:, None]
                gax.imshow(((1 - u) * ev_col[2, 0][:3] + u * end[:3])[None], aspect="auto", extent=(0, 1, 0, 1))
                gax.set_yticks([]), gax.set_xticks([0, 1])
                gax.tick_params(labelsize=7, length=2, pad=1)
                fig.text((lx + 1.0) / W, yy / H, lab, fontsize=8, va="center")
                if i_ == 0:
                    fig.text(lx / W, (yy + .3) / H, "a crossing's colour: coupling\n0 (one marginal) to 1 (both)",
                             fontsize=8, va="center")
            yy = H - YTOP - .3 - 2 * .62
            fig.patches.append(plt.Rectangle((lx / W, (yy - .06) / H), .12 / W, .12 / H, color="0.8",
                                             transform=fig.transFigure, figure=fig))
            fig.text((lx + .2) / W, yy / H, "no crossing", fontsize=8, va="center")
        else:
            for i_, (lab, col) in enumerate((("coupled, along (0, 0) - (9, 9)", "#b2182b"),
                                             ("coupled, along (0, 9) - (9, 0)", "#2166ac"), ("uncoupled", "#e08214"),
                                             ("no crossing", "0.8"))
                                            + ((("row 2: a crossing so far", "#d01c8b"),) if combined else ())):
                yy = H - YTOP - .25 - i_ * .32
                fig.patches.append(plt.Rectangle(((lx) / W, (yy - .06) / H), .12 / W, .12 / H, color=col,
                                                 transform=fig.transFigure, figure=fig))
                fig.text((lx + .2) / W, yy / H, lab, fontsize=8, va="center")
        clock = mm._clock(fig, W, H, X0, X0 + 4 * PW + 3 * GX)
        fig.text((X0 + 4 * PW + 3 * GX + .15) / W, (H - .42) / H, f"all {E} environments;\nparticles: {ns_show} of "
                 f"each cell's\n{NS} streams", fontsize=9, ha="left", va="center")
        frames = []
        for i, tau in enumerate(taus):
            for a, row in arts:
                d = data[a]
                sc = row[0]
                pos = mm._blend(d["Xs"], tau, 2).reshape(-1, 2)
                inten = np.zeros(len(pos))
                col = np.tile(grey, (len(pos), 1))
                for ti in range(max(1, int(np.floor(tau)) - 1), min(T, int(np.floor(tau)) + 3)):
                    w_ = max(0., 1 - abs(tau - (ti - .5)) / flash)
                    if w_ <= 0:
                        continue
                    if continuous:
                        m = (d["K"][:, ti] == 1) & (w_ > inten)
                        kp = d["KP"][m, ti][:, None].astype(np.float64)
                        end = np.where(d["D"][m, ti][:, None] > 0, ev_col[1, 1], ev_col[1, -1])
                        inten[m] = w_
                        col[m] = np.clip((1 - w_) * grey + w_ * ((1 - kp) * ev_col[2, 0] + kp * end), 0, 1)
                        continue
                    for (kk, dd), c in ev_col.items():
                        m = (d["K"][:, ti] == kk) & ((d["D"][:, ti] == dd) | (kk == 2)) & (w_ > inten)
                        inten[m] = w_
                        col[m] = (1 - w_) * grey + w_ * c
                o = np.argsort(inten, kind="stable")
                sc.set_offsets(pos[o])
                sc.set_facecolor(col[o])
                if combined:                                                    # crossed by now: magenta, on top
                    cr_ = d["crossed"][:, min(T - 1, int(np.floor(tau + .5)))]
                    o2 = np.argsort(cr_, kind="stable")
                    row[1].set_offsets(pos[o2])
                    row[1].set_facecolor(np.where(cr_[o2, None], MAGENTA, grey))
                Xc = mm._blend(d["Xc"], tau, 2)                                         # (E, 100, 2)
                seg = Xc[:, edges].reshape(-1, 2, 2)
                for k, lc in (((0, row[2]),) if combined else ((0, row[1]), (1, row[2]))):
                    v = mm._lin(d["cum"][k], tau)[:, edges].mean(-1).ravel()
                    o = np.argsort(v, kind="stable")
                    lc.set_segments(seg[o])
                    lc.set_color(cmap(np.clip(v[o] / vmax[k], 0, 1)))
            clock(tau)
            frames.append(mm._grab(fig))
            if i % 100 == 0:
                print(f"frame {i + 1} of {len(taus)}", flush=True)
        plt.close(fig)
    if continuous:
        bases = [(1 - k) * ev_col[2, 0][:3] + k * e[:3] for e in (ev_col[1, 1], ev_col[1, -1])
                 for k in np.linspace(0, 1, 9)]
        blends = [(1 - w) * grey[:3] + w * c for c in bases for w in np.linspace(.14, 1, 7)]
        return _save(frames, cmap, "fig1c_movie_crossings_continuous.gif", levels=48, extra=blends, greys=64)
    blends = [(1 - w) * grey[:3] + w * c[:3] for c in ev_col.values() for w in np.linspace(.08, 1, 13)]
    if combined:
        return _save(frames, cmap, "fig1c_movie_crossings_combined.gif", extra=blends + [MAGENTA[:3]])
    return _save(frames, cmap, "fig1c_movie_crossings.gif", extra=blends)


def _save(frames, cmap, name, levels=64, extra=(), greys=96):
    """as plot_fig1c_measures_movie._save, with extra exact palette colours (the crossings' flash colours and their
    fades to grey: without them the quantizer maps red / blue / orange onto viridis and grey)"""
    from main.utils import fig_path
    keys = [tuple(int(round(255 * v)) for v in cmap(u)[:3]) for u in np.linspace(0, 1, levels)]
    keys += [(int(round(v)),) * 3 for v in np.linspace(0, 255, greys)]
    keys += [tuple(int(round(255 * v)) for v in c) for c in extra]
    pal = np.unique(np.array(keys, np.uint8), axis=0)
    assert len(pal) <= 255, len(pal)
    quant = mm._Exact(pal)
    q = [quant(fr) for fr in frames]
    path = fig_path(name)
    q[0].save(path, save_all=True, append_images=q[1:], duration=1000 // mm.FPS, loop=0)
    print(f"saved {path}: {len(q)} frames, {os.path.getsize(path) / 1e6:.1f} MB", flush=True)
    return path


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "numbers"
    print({"numbers": numbers, "hybrid": hybrid, "figure": figure, "movie": movie,
           "coupling": figure_coupling, "movie_continuous": lambda: movie(continuous=True),
           "jumps": figure_jumps, "help_hurt": figure_help_hurt, "split_lags": figure_split_lags,
           "split_lags_coupling": lambda: figure_split_lags(by_coupling=True), "before": figure_before,
           "amplitude_check": figure_amplitude_check, "movie_combined": lambda: movie(combined=True)}[arg]())
