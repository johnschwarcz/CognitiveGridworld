#!/usr/bin/env python3
"""
fr_theory_figure.py -- the theory figure of the Factorization Regret paper with EVERY proposed panel (fig_fr_theory_all).

John 11-11: "everything for the theory will eventually be in 1 figure.  Next to do is to create an updated single figure with 4 rows
and as many columns as you need for all of the proposed panels, then we will trim it down to what we want.  But all proposed panels
should already be high quality, clear and polished."  It gathers the panels of fig_fr_theory and fig_fr_averages (fr_theory.py) and
of fig_fr_threshold and fig_fr_candidates (fr_theory_additions.py); the panels of those two were reworked for readability (11-11).

    python fr_theory_figure.py      seconds.  Reads the caches of fr_theory.py and fr_theory_additions.py and never rebuilds one:
                                    a missing or stale cache stops it (run `fr_theory.py all` / `fr_theory_additions.py data` first).
    outputs: figure_bin/fig_fr_theory_all.png + .pdf; the same and 300-dpi crops per row in main/DATA/experimental/fr_theory.
    python fr_theory_figure.py numbers   the values behind every statement of the theory text (FR_theory_block_v17.tex, which quotes
                                    none) about a panel; read-only, seconds.

11-11 evening (John: "I still dont understand j, s or t ... they have to go"; f + l -> mean +- SD per lambda; m-o -> two panels, very low
and very high q, theory and simulation for the mean and deviation, red line at 0: "how quickly the bulk converges to being above 0";
r: "show q as in i"; "make the figure a 3 x something"): 3 rows x 6; later 4 x 6.  11-12: 4 x 5 (John: "we cant get q^* curves.  Lets
remove panels i o and q and make the figure a 4x5.  Also lets add R=2 to panel k").
11-13 (John: "adjust the (a), (b), etc style to how i do it in the paper ... put the bold row titles in the center with a bit more
spacing"): the letters left the titles.  As in the paper's figures (his Inkscape letters, e.g. learning_curves_2x3_b), "(a)" stands in
plain sans-serif (DejaVu Sans, the title's size) centred under its panel, below the axis label, and the panel title is centred over it.
Row titles: centred over the block of panels, their baseline 0.23 in above the panel titles' (0.17 before) and their box 0.15-0.18 in
clear of the letters above.  Every x-axis label on one baseline (XLAB_Y), so each letter is as far under its label as the others; the
Theory / Simulation key at the top right, on the first row title's baseline (KEY_ROW).  11.50 x 9.18 in (7.85 in high before).
11-13 later (John: "can we make the figure a bit thinner, without cluttering the text.  If we can make better use of space that is
also great, right now the figure renders a bit small and cluttered.  An alternative strategy is to make the figure a 7 x 3, moving
columns 4 and 5 down to become 3 new rows?"): 5 ROWS x 4 (LAYOUT; the 4 x 5 is kept as LAYOUT_4x5).  At the paper's 5.5-in text width a
figure prints at 5.5 / its width: four columns print at 60% (the 4 x 5 at 48%), and five rows are then 6.67 in tall, which with a
caption of about 200 words fills one page exactly.  A 7 x 3 would print at 79% but 12.7 in tall (the page is 9 in); shrunk to fit a page
with its caption it would be at 42%, smaller than the 4 x 5.  Rows 2-4 keep their three panels by third of q and take one panel of the
old columns 4-5 each (across lambda, growth rate, synergy); the last row gathers the four panels on dips of averages (the toy's critical
step first).  0.02 in came off each vertical gap (TOP, the letter line, GAP; FOOT 0.01): 9.23 x 11.20 in.
LETTERS: 5 x 4 (now) <- 4 x 5 (the letters used by THE PANELS below and by the panel docstrings): a-d <- a-d; q <- e; e-g <- f-h; h <- i;
l <- j; i-k <- k-m; p <- n; r <- o; m-o <- p-r; s <- s; t <- t.  `numbers` and the theory text take their letters from LAYOUT.
11-13 night (John: "what is the difference between the 'toy example' and 'Cognitive gridworld'?  Just the world parameters?  Could we
simulate with the toy example params and have simulation data for everything?  Then just specify the params in the figure?"; then "yes
please rebuild row 1 and panel q this way.  I think it looks best to avoid having to write 'toy' example."): NO TOY ANY MORE.  Panels
a-d and q are the COGNITIVE GRIDWORLD ITSELF with one observable and R = 3 (sw_* panels; fr_theory_additions.build_small): tables from
the paper's generator, where a history is one count and every expectation an exact sum, picked by a stated rule (seed 3131: set 59, the
first interaction set with no trap at lambda = 1, 2 and 3, for a, c; set 1, the first with exactly one trap at lambda = 2, for b; of
2,000 other sets 5.0% have no trap and 6.5% one), with exact lines AND simulation markers (100,000 episodes per table; q: 2,000
averages per N).  c: the trap-free set at lambda = 1, 2, 3 on log-log axes (delta .0012 / .031 / .125, peak at step 7,079 / 224 / 40) in
place of three hand-built tables: the peak time is roughly, not exactly, proportional to 1 / delta (peak step x delta = 8.7 / 6.8 / 5.0).
d, q: mixtures of the two sets, q* = .00324 at t* = 435 (the toy: .0041 at 152).  a, b run to 500 steps.  Row titles name each world by
its parameters; the key is "Theory / Simulation" (mean +- 2 SEM is in the caption: the long form ran into the first row title).  The
hand-built panels and their caches are kept (LAYOUT_TOY, LAYOUT_4x5).
11-13 late night (John: "simplify the figure text ... we dont need 'Cognitive Gridworld' anywhere anymore.  And we can just write the
embedding dim instead of saying 'Five observables' ... The 'Lambda = 2 unless [varied]' can go in the caption, and other 'all except...'
rules can go in the caption.  The caption is the place for these single panel hyperparam specifics.  We also don't need so much
repetition, like 'step of a trapped episode' when it's already in the title."): LESS TEXT.  A row title says what its panels share and
gives the world by its symbols (d_o = number of observables, R = realizations per variable); in rows 2-4 it is centred over the three
panels by third of q that it describes (LAYOUT's third entry, the span), whose own titles are only "q = 0.26" etc., and the fourth
panel of the row keeps its own title.  One y label per triplet; "P(dip)" in the last row; no key in (e) (the row title says mean +-
s.d.); one-entry keys in (m) and (t).  What holds panel by panel is in the CAPTION (theory text v17): lambda = 2 except in (c, h); d_o =
5 and R = 10 in (e-p, r-t) except (l), which varies both; d_o = 1 and R = 3 in (a-d, q).  TITLES_LONG keeps the long panel titles.
11-13, later still (John: "I would like to make the figure a bit larger still.  Do you think we can reduce the width (so it renders
larger) without hurting the readability?"): NARROWER AND SHORTER, SAME TYPE.  With its caption the figure fills the page top to bottom,
so it prints larger only if it loses height in the same proportion as width: 7.92 x 9.74 in (9.23 x 11.20 before), 69% at the paper's
text width instead of 60% (7-9 pt type prints at 4.9-6.2 pt, 4.2-5.4 before), 6.76 in tall there, and every panel is at least as wide in
print as before (1.04 in; 0.72 in high in rows 2-5, 0.74 before, and 0.86 in row 1).  What made room (see the geometry section):
  * panels 1.50 in wide, not 1.7175; a y label only on the first and the fourth panel of a row (b as a; c is the expected curve of the
    row title; r, s as q), so the gaps before columns 2 and 3 hold tick labels alone (0.36 in; 0.53 before column 4);
  * rows 2-4: the bold row title stands on the panel-title line, over the three panels it describes, and each of these says its q inside
    (qtag; CTX set by make()): one line of text less per row;
  * rows 2-5 are 1.04 in high; row 1 keeps 1.245 in for the keys of the anatomy and its inset;
  * shorter keys: c = a small table, each lambda over its delta (the phrase "pull of the weakest distractor" is in the caption and the
    Details); d = "q*/2: dips", "2q*: never dips"; i-k = "decreases" over a line (the law: P(X < 0)) and a marker (simulated share) with
    their percentages, in place of the words theory / simulation; p = E[FR_t], Syn_t; q, s = one row of values (multiples of q*; q, as
    the title says); t = "Chernoff", y label "episodes N".
The 9.23 x 11.20 in version (script, figure and its theory text v17, as sent on 11-13) is kept in main/DATA/experimental/fr_theory/layout_9x11.
11-14 (John: "cut the theory block in half or more ... keep the math and new text as minimal and bootstrapped off existing
literature as possible"; "the theory figure caption would maybe do better to list the non-main-text regime panels"): the SHORT theory text (FR_theory_block_v20.tex: four propositions, one proof for the time course) has
one identity for the expected curve, E[FR_t] = t E[Delta] + N~_t - Lambda_t, and panels a, b now stack exactly these three terms
(_sw_anatomy): Joint term -Lambda_t, Naive term N~_t, linear term t E[Delta], in place of the distraction and trap terms of the old
Prop. 1(b).  Nothing else in the figure changed.  The caption starts with the panels that are NOT in the main text's setting.
11-13 night (John: "in the caption, can you please clarify when a panel is the setting of the paper?  For example, do we see anywhere
that the probability of a dip is vanishingly small in the paper's regime?  Maybe panel t?"): THE MAIN TEXT'S REGIME IS MARKED (MAIN):
dashed vertical lines at its batch of 8,000 trajectories in (s) (x axis to 10^4; every band is at zero there) and at its 30 steps in
(t) (up to the Chernoff guarantee, 519 episodes; 375 give 99%), as at its five observables in (l).  The caption (theory text v19) names
the panels with the main text's setting and the dashed lines.  The dips panels start at step 10,000; from the FIRST step, the pool of
fr_theory_additions (1,002,000 episodes of 250 steps) has no dip in 400 averages for N from 1,000 to 30,000 (`numbers`).

11-14 evening (John rewrote the theory text: "I know I made some changes that need to be reflected in the plot (e.g. Episodes is 'E') ... please add
references to the figure panels in order that they are shown (if you want to change the order that they are referenced, that is fine, just
change the order they are shown, but keep the clean organization of the figure)"): THE FIGURE FOLLOWS HIS TEXT (FR_theory_block_v23.tex).
  * the number of episodes averaged is E in every title and label (N in the code, the caches and the notes above);
  * his text writes the terms of the expected curve out (no Lambda_t, N~_t) and has no k: the keys of the first two panels say "Joint term",
    "Naive term", the titles "No trap", "One trap";
  * the text cites the panels in the order a ... t, and the synergy part comes last, so the synergy panel is the LAST panel.  Row 4 = the
    step of an average (three thirds of q) + the average at the critical step of the mixtures (sw_critical, titled by the panel of those
    mixtures); row 5 = dips within T steps, by q, episodes needed (one row title over the three, each with its own title: "stacked") +
    synergy.  LETTERS now <- before: p <- q; q, r, s <- r, s, t; t <- p (a-o unchanged).  The third panel of row 5 has no room for a y
    label (the gap holds tick labels only): its title names the axis ("Episodes E ...").

TRIM by editing LAYOUT (rows of panel names; the letters follow the reading order).  Every panel is drawn at its print size, the
panel of the 9-in rotated figures (four per row reproduce their scale), so any subset can be laid out again without rescaling.
TITLES holds every panel title.
THE CONVENTION: lines and shaded bands = analytical, markers = simulation (mean +- 2 SEM).
THE PANELS (4 x 5; in brackets the letter in the 4 x 6 figure of 11-11 / 11-12 morning, which the notes of those days use)
  a-e       toy, exact (d at q = 0, q*/2, q*, 2q*; e = Corollary 2 at the critical step)
  f-h       mean +- s.d. of FR_t; k-m [l-n] the step of a trapped episode; p-r [r-t] the step of an N-average: one column per third of q
            (q = .26 / .49 / .71), the SAME 800 interaction sets per third throughout (fr_theory.third_tables; build_thirds_more); one
            y-range per row, red line at 0.  Theory from the third's tables; simulation of those sets.
  i [j]     E[FR_t] across lambda = 1, 2, 3 against t E[Delta] (fig_fr_theory e; 11-11 late, the fourth try: "look back at past plots")
  j [k]     growth rate E[Delta] vs. observables, R = 2, 3, 4, 5 realizations per variable (11-12, John: "add R=2 to panel k", then "can
            we try 2,3,4,5?"; before: 5, 10, 20).  R = 2, 3, 4 from fr_theory_additions.build_opt_more, R = 5 (and 10, 20, not drawn:
            OBS_RS) from fr_theory.build_optimum; see p_observables
  n [p]     Proposition 3, Factorization Regret vs synergy in the population: synergy .19 > E[FR_1] .09, then it vanishes (Lambda_t ->
            0) while E[FR_t] grows like t E[Delta]: neither bounds the other
  o [u]     dips within T steps, T = 1, 10, 10^2, 10^4 (moved up from row 4 for the 4 x 5); s [v] dips within 30 steps by five bands of
            q, .2 / .35 / .5 / .65 / .8 (fr_theory_additions.build_qbands).  The markers of o and t come from fr_theory_dipsim.py (groups
            over 10^4 steps; settled episodes by the exact closed-form step, checked against the full update to 1.3e-7 nats a step).
            John 11-11 night: "if an analytical line does not have enough numerical points across it, then we need to drop the line":
            T = 10^8 went from o; 99.9 / 99.99% and T > 10^4 from t.
  t [w]     N for a chance P of NO dip within T steps, P = 10 / 50 / 90 / 99% over T = 1 ... 10^4; grows like ln T (fr_theory_horizon.py),
            under the Chernoff bound ln(T / (1 - P)) / I (drawn for 99%).  N on a linear axis: evenly spaced straight lines (the square
            log-log version put five levels within 0.4 decades of N)
  MARKERS 11-12 (John: "all the numerical simulation scatters the same size ... some panels have too many numerical scatters ... ideally
            ... equidistant"): every simulation marker is SIM_MS = 4.4 pt, keys included; on log axes the markers are thinned to an even
            log grid (on_log_grid; the theory lines keep every point): j one per doubling (d_o = 1, 2, 4, 8, 15, 30), n two per decade, t
            three (T = 1, 2, 5, 10, 22, 46 ...), o six (N = 10, 15, 22 ... 681: fr_theory_dipsim.py on 24 N, 12 per decade).  Already
            even: f-h and i (every 25 steps), k-m (every third bin), p-r and s (N = 1, 2, 5, 10 ...).  Theory lines are thin in k-m, o, t.
  (the lambda versions of f-h and k-m, p_meansd_* and p_step_*, are kept: put them back in LAYOUT to compare)
  DROPPED 11-12 (their functions and caches are kept: put the names back in LAYOUT)
    grid_mixture [i]  the Gridworld's simulated untrapped and trapped curves U_t, W_t MIXED at made-up trapped fractions q = 0, q*/2, q*,
            2q* (q* = .003).  It is not the Gridworld's own curve (q = .47, ~150 q*: it never dips), and the panel did not say so: John,
            "very misleading".  It can have no theory line: q* = F / (F + E[Delta | trapped]), F the steepest fall per step of U_t, and
            U_t, the transient of the untrapped episodes, has no formula (John: "so we cant get q^* curves").  Scratch simulations of
            11-12 (300 sets x 400 episodes, not cached): the Gridworld's own curve does dip with few realizations and many observables
            (R = 3, d_o = 15: q = .020 < q* = .061), never in the paper's setting.
    accuracy [o]      Proposition 4, the Joint - Naive accuracy gap -> q (gap at 10^9 steps .257 / .492 / .714 vs q .257 / .491 / .711)
    dips_step [q]     P(an average of N episodes dips at step t) against its limiting law: reached by about step 15
  DROPPED earlier (John: not understood): the Gridworld threshold q*_t vs q, dips of Gridworld averages in the first 250 steps, toy dips
  in 3,000 steps (Corollary 2 stays in e).  Their caches (grid, pool, toy_sim, toy_fine) are still on disk.  Tried in [j] and replaced
  (11-11 late): growth rate vs trapped fraction (the three R share one line, E[Delta] / q = .19 / .18 / .16), trapped fraction vs
  observables (John: "not a good plot"; q .76 / .48 / .18 at d_o = 1 / 5 / 30, R = 10), single episodes against their lines t Delta(r)
  (John: "I dont like this j either").
"""
import os, pickle, sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib                                                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                            # noqa: E402
from matplotlib.colors import to_rgb                                                       # noqa: E402
from matplotlib.lines import Line2D                                                        # noqa: E402
from matplotlib.legend_handler import HandlerTuple                                         # noqa: E402
from matplotlib.patches import Patch                                                       # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator       # noqa: E402
import fr_theory as F                                                                      # noqa: E402
import fr_theory_additions as A                                                            # noqa: E402
import fr_theory_additions_plot as AP                                                      # noqa: E402

OUT = os.path.join(ROOT, "main", "DATA", "experimental", "fr_theory")
NAME = "fig_fr_theory_all"

LAYOUT = [    # (row title, panels[, span[, "stacked"]]): 5 rows x 4.  The bold title is centred over its first `span` panels (default: all of them) and says
              # what they share, parameters included; a panel beyond the span stands alone under its own title (John 11-13 night: "simplify the
              # figure text ... we dont need 'Cognitive Gridworld' anywhere ... write the [observation] dim instead of saying 'Five
              # observables' ... 'Lambda = 2 unless varied' can go in the caption ... we also don't need so much repetition, like 'step of a
              # trapped episode' when it's already in the title").  "stacked": the spanned panels differ in what they show, so each keeps
              # its own title under the row title (as in a row whose title spans every panel)
    (r"Exact expected curve ($d_o=1$, $R=3$)", ["sw_anatomy_k0", "sw_anatomy_k1", "sw_time_scale", "sw_mixture"]),
    (r"Mean $\pm$ s.d. of $\mathrm{FR}_t$ ($d_o=5$, $R=10$)", ["qsd_0", "qsd_1", "qsd_2", "linear"], 3),
    (r"Step of a trapped episode ($d_o=5$, $R=10$)", ["qstep_0", "qstep_1", "qstep_2", "observables"], 3),
    (r"Step of an average over $E$ episodes ($d_o=5$, $R=10$)", ["avg_lo", "avg_mid", "avg_hi", "sw_critical"], 3),
    (r"Dips of averages over $E$ episodes ($d_o=5$, $R=10$)", ["dips_T", "dips_q", "horizon", "synergy"], 3, "stacked"),
]
LAYOUT_TOY = [                                       # the 5 x 4 of 11-13 evening, with the hand-built toy tables in row 1 and in the first dips panel (kept)
    ("Toy example (exact)", ["anatomy_k0", "anatomy_k1", "time_scale", "toy_mixture"]),
    (r"Cognitive Gridworld ($\lambda=2$ unless varied): the expected curve, and the spread of single episodes", ["qsd_0", "qsd_1", "qsd_2", "linear"]),
    ("Cognitive Gridworld: one step of a trapped episode, and the growth rate", ["qstep_0", "qstep_1", "qstep_2", "observables"]),
    ("Cognitive Gridworld: one step of an average over $E$ episodes, and synergy", ["avg_lo", "avg_mid", "avg_hi", "synergy"]),
    ("Dips of averages over $E$ episodes: toy example (left) and Cognitive Gridworld", ["toy_critical", "dips_T", "dips_q", "horizon"]),
]
LAYOUT_4x5 = [                                       # the figure of 11-12 / 11-13 morning (4 rows of 5: 11.50 x 9.18 in), kept: make(C, LAYOUT_4x5, NAME + "_4x5")
    ("Toy example (exact)", ["anatomy_k0", "anatomy_k1", "time_scale", "toy_mixture", "toy_critical"]),
    (r"Cognitive Gridworld ($\lambda=2$ unless varied): the expected curve, and the spread of single episodes", ["qsd_0", "qsd_1", "qsd_2", "linear", "observables"]),
    ("Cognitive Gridworld: one step, synergy, and dips of averages", ["qstep_0", "qstep_1", "qstep_2", "synergy", "dips_T"]),
    ("Cognitive Gridworld: averages over $E$ episodes, and their dips", ["avg_lo", "avg_mid", "avg_hi", "dips_q", "horizon"]),
]
LAYOUT_4x4 = [   # John 11-15 19:15 FL: "can we try a version where we remove f, j, n, p. Put panel t after h, panel s before l and then q and r
                 # after o? Making it a 4 x4" (f, j, n = the middle third q = .49 of rows 2-4; p = the dip at the critical step).  Rows 2-4: the bold
                 # title over the two q panels, then two panels under their own titles; `python fr_theory_figure.py 4x4` (the 5 x 4 stays as it is)
    (r"Exact expected curve ($d_o=1$, $R=3$)", ["sw_anatomy_k0", "sw_anatomy_k1", "sw_time_scale", "sw_mixture"]),
    (r"Mean $\pm$ s.d. of $\mathrm{FR}_t$ ($d_o=5$, $R=10$)", ["qsd_0", "qsd_2", "linear", "synergy"], 2),
    (r"Step of a trapped episode ($d_o=5$, $R=10$)", ["qstep_0", "qstep_2", "horizon", "observables"], 2),
    (r"Step of an average over $E$ episodes ($d_o=5$, $R=10$)", ["avg_lo", "avg_hi", "dips_T", "dips_q"], 2),
]


def letters_of(layout=None):
    """Panel name -> its letter (reading order), and g(first[, last]) -> "e" or "e-g", for texts that must follow LAYOUT."""
    names = [n for r in (layout or LAYOUT) for n in r[1]]; lt = dict(zip(names, "abcdefghijklmnopqrstuvwxyz"))
    return lt, (lambda a, b=None: lt[a] if b is None else f"{lt[a]}-{lt[b]}")


# (kept, to put back in LAYOUT: the lambda versions of f-h and k-m, p_meansd_1..3 and p_step_1..3, and the panels dropped 11-12,
#  grid_mixture, accuracy, dips_step)
TITLES = dict(
    anatomy_k0=r"$k=0$: no trap", anatomy_k1=r"$k=1$: one trap", time_scale=r"$k=0$: time scale $\propto 1/\delta$",
    toy_mixture="Monotone vs. eventually monotone", toy_critical=r"Toy example: dip at the critical step $t^*$",
    sw_anatomy_k0="No trap", sw_anatomy_k1="One trap", sw_time_scale=r"No trap: time scale $\sim 1/\delta$",
    sw_mixture="Monotone vs. eventually monotone", sw_critical=r"Dip at the critical step $t^*$",
    meansd_1=r"Mean $\pm$ s.d. of $\mathrm{FR}_t$, $\lambda=1$", meansd_2=r"Mean $\pm$ s.d. of $\mathrm{FR}_t$, $\lambda=2$",
    meansd_3=r"Mean $\pm$ s.d. of $\mathrm{FR}_t$, $\lambda=3$",
    observables="Growth rate vs. observables", linear=r"Expected curve across $\lambda$", grid_mixture=r"Expected curve, varying $q$",
    step_1=r"Step of a trapped episode, $\lambda=1$", step_2=r"Step of a trapped episode, $\lambda=2$", step_3=r"Step of a trapped episode, $\lambda=3$",
    avg_lo=lambda C: rf"Step of an average, $q={C['D']['fan3'][0]['q']:.2f}$", avg_mid=lambda C: rf"Step of an average, $q={C['D']['fan3'][1]['q']:.2f}$",
    avg_hi=lambda C: rf"Step of an average, $q={C['D']['fan3'][2]['q']:.2f}$",
    qsd_0=lambda C: rf"Mean $\pm$ s.d. of $\mathrm{{FR}}_t$, $q={C['D']['by_q'][0]['q']:.2f}$", qsd_1=lambda C: rf"Mean $\pm$ s.d. of $\mathrm{{FR}}_t$, $q={C['D']['by_q'][1]['q']:.2f}$",
    qsd_2=lambda C: rf"Mean $\pm$ s.d. of $\mathrm{{FR}}_t$, $q={C['D']['by_q'][2]['q']:.2f}$",
    qstep_0=lambda C: rf"Step of a trapped episode, $q={C['D']['by_q'][0]['q']:.2f}$", qstep_1=lambda C: rf"Step of a trapped episode, $q={C['D']['by_q'][1]['q']:.2f}$",
    qstep_2=lambda C: rf"Step of a trapped episode, $q={C['D']['by_q'][2]['q']:.2f}$",
    accuracy=r"Joint $-$ Naive accuracy $\to q$", synergy="Factorization Regret vs. synergy", horizon=r"Episodes for a $P$ chance of no dip",
    dips_step=r"Dip at step $t$ vs. its limit", dips_T=r"Dips within $T$ steps", dips_q=r"Dips within 30 steps, by $q$")

TITLES_LONG = dict(TITLES)   # as until 11-13 night, each panel title repeating its row's subject: make(C, LAYOUT_TOY, name, titles=TITLES_LONG) for the kept layouts
_q = lambda key, k: (lambda C: rf"$q={C['D'][key][k]['q']:.2f}$")
TITLES.update(qsd_0=_q("by_q", 0), qsd_1=_q("by_q", 1), qsd_2=_q("by_q", 2), qstep_0=_q("by_q", 0), qstep_1=_q("by_q", 1), qstep_2=_q("by_q", 2), avg_lo=_q("fan3", 0),
              avg_mid=_q("fan3", 1), avg_hi=_q("fan3", 2), observables=r"Growth rate vs. $d_o$", dips_T=r"Within $T$ steps", dips_q=r"Within 30 steps, by $q$",
              sw_critical=lambda C: f"Dip at the critical step of ({CTX['letters'].get('sw_mixture', 'd')})", horizon=r"Episodes $E$ for a $P$ chance of no dip")

# ---------------------------------------------------------------- geometry (inches): the panel of the 9-in two-row theory figures
# 11-13 (John: "make the figure a bit larger still ... reduce the width (so it renders larger) without hurting the readability"): the page is
# full top to bottom, so a narrower figure prints larger only if it is also shorter in the same proportion.  Same type, less white:
# panels 1.50 in wide (1.7175 before, the panel of the 9-in two-row figures); only the first and the fourth panel of a row keep a y label,
# so the gaps before columns 2 and 3 hold tick labels alone (GXS); in a row whose title spans its first panels only (rows 2-4) that
# title stands on the panel-title line and the q of each panel inside it (CTX); rows 2-5 are 1.04 in high, row 1 keeps 1.245 (its
# panels hold the keys of the anatomy and its inset).  7.92 x 9.74 in: 69% at the paper's text width, where the 9.23 x 11.20 in figure
# was at 60%; in print every panel is as wide as before (1.04 in) and rows 2-5 are 3% lower (0.72 in; row 1 0.86 in).
PW, PH = 1.50, 1.04                              # panel width; panel height (rows in PH_ROW: their own)
PH_ROW = {0: 1.245}
LM, GX, RM = 0.52, 0.55, 0.15                    # left margin, gap between panels with a y label (other layouts), right margin
GXS = (0.36, 0.36, 0.53)                         # four columns: the gaps before columns 2, 3 (tick labels only) and 4 (a y label too)
TITLE_PAD, MERGED_PAD = F.RC["axes.titlepad"], 4.5   # points from the axes to the baseline of a panel title; of the titles of a merged row (a bold
                                                 # row title with its brackets needs a little more air over the panels)
CTX = dict(inlabel=False, ylab=True, letters={})  # set by make(): inlabel where the panel has no title and names itself inside; ylab where the gap to its
                                                 # left has room for a y label (four columns: the first and the fourth panel of a row); letters of the layout drawn
# a row, top to bottom (John 11-13: the letters "how i do it in the paper"; "the bold row titles in the center with a bit more spacing"):
# TOP above the axes holds the row title (baseline HEAD_Y up) over the panel titles; BOT below them holds the tick labels, the axis label
# (baseline XLAB_Y down in EVERY panel, so that each letter sits the same distance under its label: left to matplotlib, the labels of the
# log axes sit 0.04 in lower than the others, under the empty depth of the boxes of their 10^k tick labels, whose ink ends where plain
# digits do) and the panel letter (baseline LETTER_Y down); GAP between a row's letters and the next row's title; FOOT under the last row
TOP, HEAD_Y, BOT, XLAB_Y, LETTER_Y, GAP, FOOT = 0.42, 0.275, 0.505, 0.28, 0.48, 0.08, 0.03   # (0.44, ., 0.525, ., 0.50, 0.10, 0.04 until the 5 x 4 layout:
                                                 # 0.02 in off each gap, so that the figure and a caption of about 200 words fill exactly one page of the paper)
LETTER = dict(fontfamily="DejaVu Sans", fontsize=F.FS["title"], ha="center", va="baseline")   # as in the paper's figures: "(a)" in plain sans-serif under its panel
KEY_ROW = 0                                      # the row title line with the Theory / Simulation key at its right end (rows from 0; on row 2's
                                                 # line, where it was, that row's centred title ends 0.3 in before it)

# ---------------------------------------------------------------- colours (fr_theory.py's ramps; ROSES3 new: same OKLab lightness steps)
ROSES3 = ["#e977bb", "#aa3680", "#5c1143"]                                            # N, the number of episodes averaged (dips_step)
GOLD = "#b08a00"                                                                       # synergy (n): a hue no other quantity uses
QCOL = {0.0: F.BLUES5[0], 0.5: F.BLUES5[1], 1.0: F.INK, 1.1: F.BLUES5[2], 2.0: F.BLUES5[3], 27.0: F.BLUES5[4]}   # q as a multiple of q*
DASH = dict(ls=(0, (3, 1.6)), lw=F.LW * 1.1)                                          # the threshold q* itself
RED = "#d62839"                                                                        # zero: below it a step is a decrease (f-h, k-m, p-r)
LAMS = F.CFG["LAMS_FULL"]
LAM_LABELS = [rf"$\lambda={LAMS[0]:g}$"] + [rf"${lam:g}$" for lam in LAMS[1:]]


# ============================================================================================================
# helpers
# ============================================================================================================
THIN, SIM_MS = F.LW * 0.5, F.MS * 1.3      # John 11-12: thin theory lines in k-m, o, t; ONE size for every simulation marker (4.4 pt)
GRID_DEV = []                              # how far the markers chosen by on_log_grid sit from their even grid (decades)


def theory(ax, x, y, color, lw=F.LW, **kw):
    """Analytical curve: always a line."""
    return ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=3, **kw)


def simulated(ax, x, y, yerr, color, filled=True, ms=SIM_MS, clip=False):
    """Simulation: always a marker (+- 2 SEM bar; the white ring keeps it legible on its own theory line)."""
    if yerr is not None: ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.7, capsize=0, zorder=4)
    return ax.plot(x, y, ls="none", marker="o", ms=ms, mfc=color if filled else "white", mec="white" if filled else color,
                   mew=0.5 if filled else 0.8, zorder=5, clip_on=clip)


def on_log_grid(x, step, base=1.0):
    """Markers evenly spaced on a log axis (John 11-12: "ideally equidistant"): of the x, the one nearest in log to each base * 10^(k step)
    that some x rounds to.  A boolean mask: the markers are thinned, the theory lines keep every x."""
    lx = np.log10(np.asarray(x, float) / base); mask = np.zeros(len(lx), bool)
    for k in np.unique(np.round(lx / step)):
        i = int(np.argmin(np.abs(lx - k * step))); mask[i] = True; GRID_DEV.append(abs(lx[i] - k * step))
    return mask


DOUBLING = np.log10(2.0)                                                               # one marker per doubling: integers can sit on it exactly


def h_both(color, label, lw=F.LW, ms=SIM_MS): return Line2D([], [], color=color, lw=lw, marker="o", ms=ms, mfc=color, mec="white", mew=0.5, label=label)
def h_line(color, label, **kw): return Line2D([], [], color=color, lw=kw.pop("lw", F.LW), label=label, **kw)
def h_dot(color, label, filled=True):
    return Line2D([], [], ls="none", marker="o", ms=SIM_MS, mfc=color if filled else "white", mec="white" if filled else color, mew=0.5 if filled else 0.8, label=label)
def pastel(c): return tuple(0.30 * np.array(to_rgb(c)) + 0.70)                        # the anatomy bands, opaque: they meet without a seam
def decimals(ax, d, axis="y"): (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(FuncFormatter(lambda v, _: f"${v:.{d}f}$"))
def se_p(p, n): return np.sqrt(np.maximum(p * (1 - p), 1.0 / n) / n)                 # binomial SEM (floored at one count)


def axsize(ax):
    """Width and height of an axes in inches (rows differ in height: PH_ROW)."""
    f = ax.figure; b = ax.get_position(); return b.width * f.get_figwidth(), b.height * f.get_figheight()


def qtag(ax, q, right=False):
    """The trapped fraction of a panel, written inside it where the row title stands on the panel-title line (CTX, set by make())."""
    if CTX["inlabel"]: ax.text(0.965 if right else 0.035, 0.955, rf"$q={q:.2f}$", transform=ax.transAxes, ha="right" if right else "left", va="top", fontsize=F.FS["title"], color=F.INK)


def qlab(r, first=False):
    """q as a multiple of the threshold q*."""
    s = {0.0: "0", 0.5: r"q^*\!/2", 1.0: r"q^*"}.get(r, f"{r:g}" + r"q^*")
    return f"$q={s}$" if first else f"${s}$"


PROB_TOP = 1.30                                                                        # (q-s): one row of key above a probability


def prob_axis(ax, top=1.0):
    """A probability on y: ticks 0, 0.5, 1 (headroom above 1 holds the key)."""
    ax.set_ylim(0, top); ax.set_yticks([0, 0.5, 1.0] if top > 1 else [0, 0.2, 0.4]); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.1f}$"))


def log_n_axis(ax, lo, hi, every=1):
    """Episodes averaged, on a log axis with powers of ten (every `every`-th one labelled)."""
    ax.set_xscale("log"); ax.set_xlim(lo, hi); e0, e1 = int(np.ceil(np.log10(lo))), int(np.floor(np.log10(hi)))
    ax.set_xticks([10.0 ** e for e in range(e0, e1 + 1)]); ax.set_xticklabels([rf"$10^{{{e}}}$" if (e - e0) % every == 0 else "" for e in range(e0, e1 + 1)])
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xlabel(r"episodes averaged, $E$")


# ============================================================================================================
# data: the caches of fr_theory.py and fr_theory_additions.py, read only
# ============================================================================================================
class _ReadOnly(F.Store):
    """fr_theory.py's cache store, reading only: the figure must not start a build (cf. the --out mishap of 11-11)."""
    def get(self, name, signature, builder, fresh=False):
        path = self.path(name)
        if name == "row1" and os.path.exists(path):
            with open(path, "rb") as fh: sig, data = pickle.load(fh)
            if sig == signature: return data
        raise SystemExit(f"fr_theory cache '{name}' is missing or stale (the derived quantities no longer match the configuration): "
                         "run `python fr_theory.py figure --rederive` (or `all`) in main/DATA/experimental/fr_theory first")


def load():
    cfg = F.CFG; row1, summary, D = F.load_all(cfg, _ReadOnly(OUT, quick=False))
    sig_o = (F.OPT_VERSION, F.env_fingerprint(cfg), tuple(cfg["OPT_RS"]), tuple(cfg["OPT_DOS"]), cfg["OPT_NZ"], cfg["OPT_NEP"], cfg["OPT_T1"],
             cfg["OPT_T2"], tuple(cfg["OPT_TACC"]), cfg["LAM_MAIN"])                  # as in fr_theory.load_all (checked, not trusted)
    with open(os.path.join(OUT, "fr_theory_cache", "optimum.pkl"), "rb") as fh: sig, opt = pickle.load(fh)
    if sig != sig_o: raise SystemExit("fr_theory cache 'optimum' is stale: run fr_theory.py first")
    add = {}
    for name, keys in A.KEYS.items():
        path = os.path.join(A.OUT, f"{name}.pkl"); want = repr([(k, A.CFG[k]) for k in keys])
        if not os.path.exists(path): raise SystemExit(f"fr_theory_additions cache '{name}' is missing: run `python fr_theory_additions.py data`")
        with open(path, "rb") as fh: sig, data = pickle.load(fh)
        if sig != want: raise SystemExit(f"fr_theory_additions cache '{name}' is stale: run `python fr_theory_additions.py data`")
        add[name] = data
    return dict(cfg=cfg, row1=row1, summary=summary, D=D, opt_sets={R_: opt[(R_, 5)] for R_ in cfg["OPT_RS"]}, A=add)


# ============================================================================================================
# panels: one function per panel, drawing into its axes (titles and letters are added by the layout)
# ============================================================================================================
# ---------------------------------------------------------------- row 1: toy example, exact (environment-independent)
def _anatomy(ax, C, k):
    r = C["row1"][k]; an, st = r["an"], r["st"]; t = np.arange(F.T_ANATOMY + 1); base = -an["Lam"]; mid = base + an["D"]
    ax.axhline(0, color=F.AXIS, lw=0.6, zorder=1)
    ax.fill_between(t, base, mid, color=pastel(F.ORANGE), lw=0, zorder=2)
    if st["k"]: ax.fill_between(t, mid, an["FR"], color=pastel(F.BLUE), lw=0, zorder=2)
    theory(ax, t, base, F.AQUA); theory(ax, t, an["FR"], F.INK, lw=F.LW * 1.15)
    ax.set_xlabel("step $t$"); ax.set_ylabel("nats"); ax.set_xlim(0, F.T_ANATOMY); ax.set_xticks([0, 100, 200, 300]); decimals(ax, 1)
    return an


def p_anatomy_k0(ax, C):
    """E[FR_t] = Joint term + distraction term (Proposition 1(b)) for the trap-free table; inset: E[FR_t] magnified."""
    an = _anatomy(ax, C, "a"); ax.set_ylim(-1.15, 0.95); ax.set_yticks([-1.0, -0.5, 0.0, 0.5])
    ax.legend(handles=[h_line(F.INK, r"$\mathbb{E}[\mathrm{FR}_t]$ = sum of"), h_line(F.AQUA, r"Joint term $-\Lambda_t$"),
                       Patch(color=pastel(F.ORANGE), lw=0, label=r"distraction $\mathbb{E}[\mathcal{D}_t]$")], loc="upper right")
    SQ = 0.50; ins = ax.inset_axes([1 - SQ / PW - 0.03, 0.045, SQ / PW, SQ / PH]); ins.grid(False)       # a square in the empty lower right
    theory(ins, np.arange(F.T_ANATOMY + 1), an["FR"], F.INK)
    ins.set_xlim(0, F.T_ANATOMY); ins.set_ylim(0, 0.065); ins.set_xticks([0, 100, 200, 300]); ins.set_xticklabels([]); ins.set_yticks([0, 0.05])
    ins.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:.2f}".lstrip("0")))
    ins.tick_params(length=1.6, pad=1.0, labelsize=F.FS["tick"])
    for sp in ins.spines.values(): sp.set_visible(True); sp.set_linewidth(0.5)


def p_anatomy_k1(ax, C):
    """As (a) for the table with one trapped joint realization: the trap term grows linearly."""
    _anatomy(ax, C, "b"); ax.set_ylim(-1.25, 2.25); ax.set_yticks([-1, 0, 1, 2])
    ax.legend(handles=[Patch(color=pastel(F.BLUE), lw=0, label=r"trap term $\mathbb{E}[\mathcal{T}_t]$")], loc="upper left")


def p_time_scale(ax, C):
    """No trap: E[FR_t] returns to 0 on a time scale set by the pull delta of the weakest distractor."""
    T2 = F.T_DISTRACTORS; t = np.arange(T2 + 1)
    for e, col in zip(C["row1"]["d"], F.ORANGES3): theory(ax, t, e["efr"][:T2 + 1], col)
    ax.legend(handles=[h_line(col, f"{e['st']['delta']:.2f}") for e, col in zip(C["row1"]["d"], F.ORANGES3)], title=r"Pull $\delta$ of the weakest distractor",
              loc="upper right", ncol=3, handlelength=1.1, columnspacing=0.7, alignment="left")
    ax.set_ylim(0, 0.1); ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08]); decimals(ax, 2); ax.set_xlim(0, T2); ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_xlabel("step $t$"); ax.set_ylabel(r"$\mathbb{E}[\mathrm{FR}_t]$ (nats)")


def _toy_qstar(C):
    E0, E1 = C["row1"]["a"]["efr"], C["row1"]["b"]["efr"]; share = C["row1"]["b"]["st"]["k"] / C["row1"]["b"]["ell"].size
    w_star, t_star = F.mixture_threshold(E0, E1); q_star = w_star * share
    assert np.isclose(q_star, C["A"]["toy_theory"]["q_star"]) and t_star == C["A"]["toy_theory"]["t_star"], "the two scripts disagree on q*"
    return E0, E1, share, q_star


def p_toy_mixture(ax, C):
    """Corollary 1: a fraction q of the episodes from table (b), the rest from (a); the expected curve dips iff q < q*."""
    E0, E1, share, q_star = _toy_qstar(C); t = np.arange(len(E0)); hnd = []
    labels = {0.0: r"$q=0$: returns to 0", 0.5: r"$q^*\!/2$: dips, then rises", 1.0: r"$q^*$", 2.0: r"$2q^*$: rises from $t=0$"}
    for r in (0.0, 0.5, 1.0, 2.0):
        w = r * q_star / share; kw = DASH if r == 1.0 else {}
        theory(ax, t[1:], ((1 - w) * E0 + w * E1)[1:], QCOL[r], **kw); hnd.append(h_line(QCOL[r], labels[r], **dict(kw)))
    ax.set_xscale("log"); ax.set_xlim(1, F.T_MIXTURE); ax.set_ylim(-0.006, 0.28); ax.set_yticks([0, 0.1, 0.2]); decimals(ax, 1)
    ax.xaxis.set_major_locator(LogLocator(numticks=5)); ax.xaxis.set_minor_formatter(NullFormatter()); ax.axhline(0, color=F.AXIS, lw=0.6, zorder=1)
    ax.legend(handles=hnd, loc="upper left", alignment="left", handlelength=1.6, borderaxespad=0.15)
    ax.set_xlabel("step $t$"); ax.set_ylabel(r"$\mathbb{E}[\mathrm{FR}_t]$ (nats)")


def p_toy_critical(ax, C):
    """Corollary 2 at the critical step t* (where q*_t is largest): P(an average of N episodes decreases there), exact law of the
    increment (FFT up to 10^6 episodes, saddlepoint beyond): -> 1 below q*, 1/2 at q*, -> 0 above, slower the closer q is to q*."""
    th = C["A"]["toy_theory"]; hnd = []
    for k, r in enumerate((0.5, 1.0, 1.1, 2.0)):
        kw = DASH if r == 1.0 else {}
        theory(ax, th["Ns"], th["p_star"][th["ratios"].index(r)], QCOL[r], **kw); hnd.append(h_line(QCOL[r], qlab(r, k == 0), **dict(kw)))
    log_n_axis(ax, 1e3, 1e8, every=2); prob_axis(ax, 1.36)
    ax.legend(handles=hnd, loc="upper left", ncol=2, columnspacing=0.9, handlelength=1.5, borderaxespad=0.1)
    ax.set_ylabel(r"P(dip at step $t^*$)")


# ---------------------------------------------------------------- row 1 since 11-13 night: the Cognitive Gridworld with ONE observable, exact AND simulated
# (John: "yes please rebuild row 1 and panel q this way.  I think it looks best to avoid having to write 'toy' example."  The tables come
#  from the paper's generator at R = 3 with one observable, picked by a stated rule: fr_theory_additions.build_small.)
def _sw_anatomy(ax, C, key):
    """E[FR_t] as the stack of the three terms of Proposition 1(a) of the SHORT theory text (v20, 11-14), in the one-observable
    Gridworld: E[FR_t] = t E[Delta] + N~_t - Lambda_t, with Lambda_t the Joint observer's log loss on the optimal action (line -Lambda_t),
    N~_t the Naive observer's log loss on the action it converges to (orange band) and t E[Delta] the linear term (blue band, absent
    without a trap).  Until v19 the bands were the distraction and trap terms of the old Prop. 1(b) (the same Naive log loss on the
    optimal action, split differently; cache keys D, T): without a trap N~_t = E[D_t], so (a) is unchanged.  Exact sums; markers:
    100,000 simulated episodes, on the two LINES (the bands are differences of the lines): E[FR_t] every tenth of the axis, the Joint
    term on the log steps of its rise (22, 46, 100, 215, 464: on the flat part it would sit under the markers of E[FR_t])."""
    S = C["A"]["small"]; e = S["an"][key]; an, st = e["an"], e["st"]; T = len(an["FR"]) - 1; t = np.arange(T + 1); base = -an["Lam"]; mid = base + an["N"]
    assert np.abs(an["FR"] - (t * st["Delta"].mean() + an["N"] - an["Lam"])).max() < 1e-9, "E[FR_t] = t E[Delta] + N~_t - Lambda_t fails"
    ax.axhline(0, color=F.AXIS, lw=0.6, zorder=1)
    ax.fill_between(t, base, mid, color=pastel(F.ORANGE), lw=0, zorder=2)
    if st["k"]: ax.fill_between(t, mid, an["FR"], color=pastel(F.BLUE), lw=0, zorder=2)
    theory(ax, t, base, F.AQUA); theory(ax, t, an["FR"], F.INK, lw=F.LW * 1.15)
    sm = S["sims"][("free" if key == "a" else "one", S["lam"])]; ts = S["ts"]; mf = np.isin(ts, S["t_lin"]); mj = np.isin(ts, S["t_log"]) & (ts >= 20) & (ts <= T // 2)
    simulated(ax, ts[mj], -sm[mj, 2], 2 * sm[mj, 3], F.AQUA); simulated(ax, ts[mf], sm[mf, 0], 2 * sm[mf, 1], F.INK)
    ax.set_xlabel("step $t$"); ax.set_ylabel("nats" if key == "a" else ""); ax.set_xlim(0, T); ax.set_xticks(np.linspace(0, T, 5)); decimals(ax, 1)   # (b): as (a)
    return S, an, sm, mf


def p_sw_anatomy_k0(ax, C):
    """The trap-free interaction set (lambda = 2): E[FR_t] = Naive term - Joint term (N~_t - Lambda_t); it peaks at .097 nats at step 220 and is back at
    .072 at step 500, .031 at 1,000, 1e-3 at 3,000 (the return to zero is on the log axes of the third and fourth panels; 500 steps keep
    the two terms, which resolve within 200 steps, legible).  Inset: E[FR_t] magnified, with every second marker."""
    S, an, sm, mf = _sw_anatomy(ax, C, "a"); ax.set_ylim(-1.15, 0.95); ax.set_yticks([-1.0, -0.5, 0.0, 0.5]); T = len(an["FR"]) - 1
    ax.legend(handles=[h_both(F.INK, r"$\mathbb{E}[\mathrm{FR}_t]$ = sum of"), h_both(F.AQUA, "Joint term"),
                       Patch(color=pastel(F.ORANGE), lw=0, label="Naive term")], loc="upper right", handlelength=1.7)
    w, h = axsize(ax); IW, IH = 0.70, 0.50; ins = ax.inset_axes([1 - IW / w - 0.03, 0.045, IW / w, IH / h]); ins.grid(False)       # in the empty lower right
    theory(ins, np.arange(T + 1), an["FR"], F.INK); m2 = mf & (S["ts"] % (2 * S["t_lin"][0]) == 0); simulated(ins, S["ts"][m2], sm[m2, 0], 2 * sm[m2, 1], F.INK)
    tick = float(f"{an['FR'].max():.1g}"); ins.set_xlim(0, T); ins.set_ylim(0, 1.3 * max(an["FR"].max(), tick)); ins.set_xticks(np.linspace(0, T, 5)); ins.set_xticklabels([]); ins.set_yticks([0, tick])
    ins.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:g}".lstrip("0")))
    ins.tick_params(length=1.6, pad=1.0, labelsize=F.FS["tick"])
    for sp in ins.spines.values(): sp.set_visible(True); sp.set_linewidth(0.5)


def p_sw_anatomy_k1(ax, C):
    """The interaction set with one trapped joint realization of nine (lambda = 2): the linear term t E[Delta] (Delta = .0335 at the
    trapped realization: E[Delta] = .0037 nats a step)."""
    S, an, sm, mf = _sw_anatomy(ax, C, "b"); top = an["FR"].max(); ax.set_ylim(-1.25, 1.22 * top); ax.set_yticks(np.arange(-1, top + 0.5, 1.0)); decimals(ax, 1)
    ax.legend(handles=[Patch(color=pastel(F.BLUE), lw=0, label=r"linear term $t\,\mathbb{E}[\Delta]$")], loc="upper left")


def p_sw_time_scale(ax, C):
    """KEY: a table, each lambda over its delta (the pull of the weakest distractor: caption and Details); no y label (as d).
    No trap: the interaction set of the first panel at three likelihood temperatures.  The bump of E[FR_t] comes later the smaller the
    pull delta of the weakest distractor: delta = .0012 / .031 / .125 at lambda = 1 / 2 / 3, peak at step 7,079 / 224 / 40 (peak step x
    delta = 8.7 / 6.8 / 5.0; the hand-built tables it replaces: 4.9-5.3 by design).  Log-log: the bumps are .26 / .097 / .013 nats high.
    Markers: 100,000 simulated episodes per temperature, three steps per decade, where above the axis floor and above 2 SEM.
    The title says "~", not "proportional to" (independent check 11-13): the peak step goes as delta^-1.12 over the 102-fold range of
    delta, and the temperature changes the height of the bump too, so the panel cannot isolate delta as the hand-built tables did."""
    S = C["A"]["small"]; hnd = []; floor = 1e-4; dl = lambda d: f"{d:.1g}" if d < 0.01 else f"{d:.2f}"
    for i, (lam, col) in enumerate(zip(S["lams"], F.VIOLETS3)):
        T_ = S["temp"][lam]; theory(ax, S["tl"], np.maximum(T_["efr"], 1e-300), col); sm = S["sims"][("free", lam)]
        m = np.isin(S["ts"], S["t_log"]) & (sm[:, 0] > 2 * sm[:, 1]) & (sm[:, 0] > floor); simulated(ax, S["ts"][m], sm[m, 0], 2 * sm[m, 1], col, clip=True)
        hnd += [h_both(col, (r"$\lambda=" if i == 0 else "$") + f"{lam:g}$"), Line2D([], [], ls="none", label=(r"$\delta=" if i == 0 else "$") + dl(T_["st"]["delta"]) + "$")]
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.8, 1.3 * S["tl"][-1]); ax.set_ylim(floor, 40); ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    ax.xaxis.set_major_locator(LogLocator(numticks=6)); ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(handles=hnd, loc="upper left", ncol=3, handlelength=1.1, columnspacing=1.0, handletextpad=0.3, labelspacing=0.12)   # a table: each lambda over its
    ax.set_xlabel("step $t$")                                                              # delta, the pull of the weakest distractor (caption, Details); y as (d)


def p_sw_mixture(ax, C):
    """Corollary 1: a fraction q of the episodes is trapped (a fraction 9 q of them in the one-trap interaction set, the rest in the
    trap-free one); the expected curve dips iff q < q* = .00324 (critical step 435; the dip of q*/2 spans steps 264-971).  Markers: the
    two sets' simulated means (100,000 episodes each) mixed in the same proportions, so the four rows of markers share their noise."""
    S = C["A"]["small"]; tl = S["tl"]; E0, E1 = S["temp"][S["lam"]]["efr"], S["E1l"]; sa, sb = S["sims"][("free", S["lam"])], S["sims"][("one", S["lam"])]
    xmax = 10 ** 4; m = np.isin(S["ts"], S["t_log"]) & (S["ts"] >= 2) & (S["ts"] <= xmax); hnd = []
    labels = {0.0: r"$q=0$: returns to 0", 0.5: r"$q^*\!/2$: dips", 1.0: r"$q^*$", 2.0: r"$2q^*$: never dips"}
    for r in (0.0, 0.5, 1.0, 2.0):
        w = r * S["w_star"]; kw = dict(DASH) if r == 1.0 else dict(lw=F.LW)
        theory(ax, tl, (1 - w) * E0 + w * E1, QCOL[r], **kw)
        simulated(ax, S["ts"][m], (1 - w) * sa[m, 0] + w * sb[m, 0], 2 * np.hypot((1 - w) * sa[m, 1], w * sb[m, 1]), QCOL[r], clip=True)
        hnd.append(Line2D([], [], color=QCOL[r], marker="o", ms=SIM_MS, mfc=QCOL[r], mec="white", mew=0.5, label=labels[r], **kw))
    ax.set_xscale("log"); ax.set_xlim(0.8, 1.3 * xmax); ax.set_ylim(-0.012, 0.52); ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4]); decimals(ax, 1)
    ax.xaxis.set_major_locator(LogLocator(numticks=6)); ax.xaxis.set_minor_formatter(NullFormatter()); ax.axhline(0, color=F.AXIS, lw=0.6, zorder=1)
    ax.legend(handles=hnd, loc="upper left", alignment="left", handlelength=1.9, borderaxespad=0.15)
    ax.set_xlabel("step $t$"); ax.set_ylabel(r"$\mathbb{E}[\mathrm{FR}_t]$ (nats)")


def p_sw_critical(ax, C):
    """Corollary 2 at the critical step t* of the mixtures of the previous panel: P(an average of N episodes decreases there).  Exact law
    of the average increment (FFT up to 10^6 episodes, saddlepoint beyond; the two agree to five decimals at 10^6): -> 1 below q*, 1/2
    at q*, -> 0 above, slower the closer q is to q*.  Markers: 2,000 simulated averages per N, the N episodes of an average drawn as the
    numbers that fall on each outcome (one multinomial draw; checked against episode-by-episode simulation at N = 1,000); 52 markers,
    |z| <= 2.2."""
    S = C["A"]["small"]; hnd = []; ms = S["sim_Ns"] >= 1e3
    for k, r in enumerate((0.5, 1.0, 1.1, 2.0)):
        i = S["ratios"].index(r); kw = dict(DASH) if r == 1.0 else dict(lw=F.LW)
        theory(ax, S["Ns"], S["p_star"][i], QCOL[r], **kw); simulated(ax, S["sim_Ns"][ms], S["sim"][i][ms], 2 * S["sim_se"][i][ms], QCOL[r], clip=True)
        hnd.append(Line2D([], [], color=QCOL[r], marker="o", ms=SIM_MS, mfc=QCOL[r], mec="white", mew=0.5, label=qlab(r), **kw))
    log_n_axis(ax, 7e2, 1.5e9, every=2); prob_axis(ax, PROB_TOP)
    ax.legend(handles=hnd, loc="upper center", ncol=4, columnspacing=0.6, handlelength=0.8, handletextpad=0.3, borderaxespad=0.1)   # q, as multiples of q*
    ax.set_ylabel("P(dip)")


# ---------------------------------------------------------------- row 2: Cognitive Gridworld, the expected curve
def _meansd(ax, C, lam):
    """Propositions 1(c) and 2(c): mean and s.d. of FR_t across episodes.  Theory (tables alone, at the simulated episodes' own joint
    realizations): t E[Delta] and sqrt(t^2 Var(Delta) + t E[v]) (the line omits the bounded remainder of eq. fr_linear, a fraction of a
    nat).  Simulation: mean and mean +- s.d. over 2,000 sets x 50 episodes (2 SEM by resampling sets)."""
    SP = C["D"]["spread"][lam]; col = F.VIOLETS3[LAMS.index(lam)]; t = np.r_[0, SP["t"]]; m = np.r_[0, SP["mean_th"]]; s = np.r_[0, SP["th"]]
    ax.axhline(0, color=RED, lw=1.0, zorder=2.6)
    ax.fill_between(t, m - s, m + s, color=col, alpha=0.15, lw=0, zorder=2)
    for e in (m - s, m + s): theory(ax, t, e, col, lw=F.LW * 0.5)
    theory(ax, t, m, col)
    simulated(ax, SP["tm"], SP["mean_m"], 2 * SP["mean_se"], col)
    for v, e in (("lo_m", "lo_se"), ("hi_m", "hi_se")): simulated(ax, SP["tm"], SP[v], 2 * SP[e], col)
    ext = [(S["mean_th"] + S["th"]).max() for S in C["D"]["spread"].values()] + [S["hi_m"].max() for S in C["D"]["spread"].values()]
    low = [(S["mean_th"] - S["th"]).min() for S in C["D"]["spread"].values()] + [S["lo_m"].min() for S in C["D"]["spread"].values()]
    top, bot = max(ext), min(low)                                                      # shared by the three panels (John 11-11)
    ax.set_xlim(0, C["cfg"]["T"]); ax.set_xticks(np.arange(0, C["cfg"]["T"] + 1, 100)); ax.set_ylim(bot - 0.06 * (top - bot), top + 0.06 * (top - bot))
    ax.set_xlabel("step $t$"); ax.set_ylabel(r"$\mathrm{FR}_t$ (nats)")
    if lam == LAMS[0]:
        ax.legend(handles=[h_both(col, "mean"), (Patch(facecolor=col, alpha=0.15, edgecolor=col, lw=0.5), h_dot(col, ""))], labels=["mean", r"mean $\pm$ s.d."],
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, loc="upper left", handlelength=2.2, borderaxespad=0.15)


def p_meansd_1(ax, C): _meansd(ax, C, LAMS[0])
def p_meansd_2(ax, C): _meansd(ax, C, LAMS[1])
def p_meansd_3(ax, C): _meansd(ax, C, LAMS[2])


# THE MAIN TEXT'S REGIME (John 11-13: "in the caption, can you please clarify when a panel is the setting of the paper?  For example, do we
# see anywhere that the probability of a dip is vanishingly small in the paper's regime?  Maybe panel t?").  The manuscript's environment
# (its Environment Hyperparameters): d_o = 5, R = 10, lambda = 2, T = 30 steps; its test batches hold 8,000 trajectories (caption of its
# learning-dynamics figure; the comparison of the two observers, compare_sanity.py, uses 10,000): the "episodes" of this figure.  Marked
# by dashed vertical lines: d_o in (l), the batch in (r), the horizon in (s) (letters of the 11-14 layout).  At N = 8,000 the Chernoff bound on a dip within 30 steps
# is 30 exp(-N I) = 7e-53, and every band of (s) is at zero in theory (q = .2: 2e-10 at N = 10^4) and simulation.
MAIN = dict(d_o=5, N=8000, T=30)
MAIN_LINE = dict(color=F.MUTED, lw=0.9, ls=(0, (2, 1.6)), zorder=1)                    # (0.6 pt until the marks of s and t: too faint at 69%)


OBS_RS = (2, 3, 4, 5)        # (j): realizations per variable drawn (John 11-12: "can we try 2,3,4,5?"; 10 and 20 are in the caches too)


def p_observables(ax, C):
    """E[Delta] (tables) against late simulated slopes, by number of observables d_o, for OBS_RS realizations per variable (10,000
    interaction sets x 100 episodes per point, slope between steps 10,000 and 20,000).  R = 5, 10, 20: fr_theory.build_optimum; R = 2, 3,
    4 (John 11-12: "add R=2", then "can we try 2,3,4,5?" in place of 2, 5, 10, 20): fr_theory_additions.build_opt_more, the same point
    function.  E[Delta] peaks at d_o = 1 / 2 / 2 / 3 for R = 2 / 3 / 4 / 5 (.0061 / .025 / .041 / .054 nats a step) and falls with more
    observables: .0017 / .018 / .036 / .051 at d_o = 5, 0 / .0004 / .0067 / .019 at d_o = 30.  Fewer realizations, fewer traps: the sets
    without any trap are 88 / 36 / 7% at d_o = 5 for R = 2 / 3 / 4, and with R = 2 no set of the 10,000 has one from d_o = 25 on.
    Simulated slopes: within 1.7% of E[Delta] wherever E[Delta] > .001 (43 of the 52 points) and within 0.7 s.e. at every point (largest
    relative deviation 4%, at R = 2, d_o = 20, where E[Delta] = 1.2e-5 +- 0.6e-5).  Dashed vertical line: the Gridworld's five observables."""
    xs = np.array(C["cfg"]["OPT_DOS"]); OP = {**C["A"]["opt_more"], **C["D"]["opt"]}; OP = {k: v for k, v in OP.items() if k[0] in OBS_RS}
    assert sorted({k[0] for k in OP}) == sorted(OBS_RS) and len(OBS_RS) == len(GRAYS4)
    xm = xs[on_log_grid(xs, DOUBLING)]                                                # markers at d_o = 1, 2, 4, 8, 15, 30
    for R_, col in zip(OBS_RS, GRAYS4):
        theory(ax, xs, [OP[(R_, d)]["ed"][0] for d in xs], col)
        simulated(ax, xm, [OP[(R_, d)]["slope"][0] for d in xm], [2 * OP[(R_, d)]["slope"][1] for d in xm], col)
    ax.axvline(MAIN["d_o"], ymax=0.84, **MAIN_LINE)                                   # the main text's five observables (ends under the key)
    ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 30]); ax.set_xticklabels(["1", "2", "5", "10", "30"]); ax.set_xlim(0.9, 33)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 1.32 * max(max(v["ed"][0], v["slope"][0]) for v in OP.values()))  # from the curves drawn
    ax.legend(handles=[h_both(col, (r"$R=" if i == 0 else "$") + f"{R_}$") for i, (R_, col) in enumerate(zip(OBS_RS, GRAYS4))], loc="upper center", ncol=4,
              columnspacing=0.7, handlelength=1.2, handletextpad=0.3, borderaxespad=0.1)
    ax.set_xlabel(r"observables $d_o$"); ax.set_ylabel(r"$\mathbb{E}[\Delta]$ (nats / step)")


def p_linear(ax, C):
    """Proposition 1(c): E[FR_t] follows t E[Delta] (theory: tables alone) at every lambda; simulation: 2,000 sets x 50 episodes per
    lambda (fig_fr_theory e, back as j 11-11 late: John, after two other j's, "look back at past plots for an idea of something good")."""
    D, S, T = C["D"], C["summary"], C["cfg"]["T"]; tt = D["tt"]; tm = np.arange(T // 10, T + 1, T // 10) - 1; pts = []; top = 0.0
    for lam, col in zip(LAMS, F.VIOLETS3):
        cv = D["curve"][lam]; line = np.r_[0, tt] * S[lam]["EDelta"]
        theory(ax, np.r_[0, tt], line, col); simulated(ax, tt[tm], cv["mean"][tm], 2 * cv["se"][tm], col)
        pts += [np.c_[np.r_[0, tt], line], np.c_[tt[tm], cv["mean"][tm] + 2 * cv["se"][tm]]]; top = max(top, cv["mean"][-1] + 2 * cv["se"][-1], T * S[lam]["EDelta"])
    leg = ax.legend(handles=[h_both(col, s) for s, col in zip(LAM_LABELS, F.VIOLETS3)], loc="upper left", ncol=3, columnspacing=0.8, handlelength=1.3)
    ax.set_xlabel("step $t$"); ax.set_ylabel(r"$\mathbb{E}[\mathrm{FR}_t]$ (nats)"); ax.set_xlim(0, T); ax.set_ylim(0, 1.07 * top); ax.set_xticks(np.arange(0, T + 1, 100))
    return dict(headroom=([leg], pts))


def p_grid_mixture(ax, C):
    """DROPPED 11-12 (John: "very misleading"): NOT the Gridworld's own expected curve (its q = .47 never dips) but its simulated untrapped
    (U) and trapped (W) curves mixed at made-up trapped fractions, as the toy of (d): (1 - q) U_t + q W_t, lambda = 2,
    for q relative to the simulated threshold q* (windows of 3 grid steps) and the Gridworld's own q."""
    G = C["A"]["grid"][C["cfg"]["LAM_MAIN"]]; Cm = A.components(G); TS = np.asarray(G["TS"]); qs = AP.window_threshold(G, AP.WINDOW)["q_star"]; hnd = []
    sh = on_log_grid(TS, DOUBLING)                                                     # t = 1, 2, 4 ... 7943 (of 70 simulated steps)
    for k, (r, q) in enumerate([(0.0, 0.0), (0.5, 0.5 * qs), (1.0, qs), (2.0, 2 * qs)]):
        col = QCOL[r]; m = (1 - q) * Cm["U"] + q * Cm["W"]; se = np.hypot((1 - q) * Cm["U_se"], q * Cm["W_se"])
        simulated(ax, TS[sh], m[sh], 2 * se[sh], col, filled=r != 1.0, clip=True)
        hnd.append(h_dot(col, qlab(r, k == 0) if r != 1.0 else rf"$q^*\approx{qs:.1g}$", filled=r != 1.0))
    ax.set_xscale("log"); ax.set_xlim(1, A.CFG["GRID_TMAX"]); ax.xaxis.set_major_locator(LogLocator(numticks=5)); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.5, 1.0]); decimals(ax, 1)
    ax.legend(handles=[hnd[0], hnd[2], hnd[1], hnd[3]], loc="upper left", ncol=2, columnspacing=0.8, handletextpad=0.25, borderaxespad=0.1)   # row-wise order
    ax.set_xlabel("step $t$"); ax.set_ylabel(r"$\mathbb{E}[\mathrm{FR}_t]$ (nats)")


# ---------------------------------------------------------------- row 3: one step of single episodes and of averages
def _step(ax, C, lam):
    """Proposition 2(a): the law of one step X of a trapped episode (tables alone) against the first simulated step after step 10,000
    (2,000 interaction sets x 150 episodes, its trapped episodes), shared bins and axes across lambda; red: zero, where a step is a decrease."""
    ST = C["D"]["step_lam"][lam]; ed = ST["edges"]; mid = 0.5 * (ed[1:] + ed[:-1]); col = F.VIOLETS3[LAMS.index(lam)]
    ax.stairs(ST["dens"], ed, fill=True, color=col, alpha=0.14, lw=0, zorder=2); ax.stairs(ST["dens"], ed, color=col, lw=F.LW, zorder=3)
    simulated(ax, mid, ST["sim"], ST["sim_2se"], col)
    ax.axvline(0, color=RED, lw=1.0, zorder=2.6); ax.set_xlim(-ST["xmax"], ST["xmax"])
    top = max(max(S["dens"].max(), S["sim"].max()) for S in C["D"]["step_lam"].values()); ax.set_ylim(0, 1.12 * top)
    ax.text(0.97, 0.97, f"decreases:\n{100 * ST['p_pred']:.0f}% theory\n{100 * ST['p_sim']:.0f}% simulation", transform=ax.transAxes,
            va="top", ha="right", linespacing=1.15, fontsize=F.FS["key"], color=F.INK)
    ax.set_xlabel(r"$\mathrm{FR}_{t+1}-\mathrm{FR}_t$ (nats)"); ax.set_ylabel("density")


def p_step_1(ax, C): _step(ax, C, LAMS[0])
def p_step_2(ax, C): _step(ax, C, LAMS[1])
def p_step_3(ax, C): _step(ax, C, LAMS[2])


def _avg(ax, C, k):
    """Proposition 2(b): the one-step change X-bar_N of an average of N episodes (from step 10,000), interaction sets in the k-th third
    of the trapped fraction (q = that third's share of trapped joint realizations).  Theory (tables alone): mean +- s.d. of X-bar_N =
    mu +- sigma / sqrt(N), mu and sigma the exact moments of one step X; the bulk is above zero once N > (sigma / mu)^2.  Simulation:
    mean +- s.d. over independent groups of N simulated episodes (2 SEM by bootstrap over groups)."""
    FN = C["D"]["fan3"][k]; col = F.BLUES3[k]; mu, sg = FN["mean"], FN["sigma"]; Nf = np.logspace(0, np.log10(1300), 300)
    ax.axhline(0, color=RED, lw=1.0, zorder=2.6)
    ax.fill_between(Nf, mu - sg / np.sqrt(Nf), mu + sg / np.sqrt(Nf), color=col, alpha=0.16, lw=0, zorder=2)
    for e in (mu - sg / np.sqrt(Nf), mu + sg / np.sqrt(Nf)): theory(ax, Nf, e, col, lw=F.LW * 0.8)
    for j in (0, 1): simulated(ax, FN["Ns"], FN["msd"][:, j], 2 * FN["msd_se"][:, j], col)
    log_n_axis(ax, 0.8, 1300); ax.set_ylim(-0.5, 0.72); ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6]); decimals(ax, 1)
    ax.set_ylabel(r"$\widehat{\mathrm{FR}}_{t+1}-\widehat{\mathrm{FR}}_t$ (nats)" if k == 0 else "")
    if k == 0:                                                                             # band = theory, markers = simulation: the figure's key says so
        ax.legend(handles=[(Patch(facecolor=col, alpha=0.16, edgecolor=col, lw=0.8), h_dot(col, ""))], labels=[r"mean $\pm$ s.d."],
                  handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, loc="lower right", handlelength=2.2, borderaxespad=0.2)
    qtag(ax, FN["q"], right=True)                                                          # (the band is widest at the left)


def p_avg_lo(ax, C): _avg(ax, C, 0)
def p_avg_mid(ax, C): _avg(ax, C, 1)
def p_avg_hi(ax, C): _avg(ax, C, 2)


# ---------------------------------------------------------------- dips of averaged curves
def p_dips_step(ax, C):
    """P(an average of N episodes decreases at step t) against t, simulated (lambda = 2), against its limiting law (tables alone):
    reached by about step 15; before that dips are rarer."""
    S = C["A"]["steps"]; ts = np.asarray(S["TS"]); sh = on_log_grid(ts, 1 / 3)     # t = 1, 2, 5, 10, 22 ... 10^4 (of 24 simulated steps)
    for i, (N, col) in enumerate(zip(S["Ns"], ROSES3)):
        theory(ax, [0.5, 2e4], [0.5 * (S["law_lo"][i] + S["law_hi"][i])] * 2, col); simulated(ax, ts[sh], S["sim"][i][sh], 2 * S["se"][i][sh], col)
    ax.set_xscale("log"); ax.set_xlim(0.8, 1.3e4); ax.xaxis.set_major_locator(LogLocator(numticks=5)); ax.xaxis.set_minor_formatter(NullFormatter())
    prob_axis(ax, 0.5)
    ax.legend(handles=[h_both(col, (r"$E=" if i == 0 else "$") + f"{N}$") for i, (N, col) in enumerate(zip(S["Ns"], ROSES3))], loc="upper left", ncol=3,
              columnspacing=0.8, handlelength=1.3, borderaxespad=0.1)
    ax.set_xlabel("step $t$"); ax.set_ylabel(r"P(dip at step $t$)")


U_TS = (1, 10, 100, 10 ** 4)                                                           # (o): horizons (John 11-11: only lines the simulation covers)
GREENS4U = ["#4fb886", "#2a9566", "#11704a", "#044a30"]                                  # = fr_theory.GREENS4


def _cloglog_curve(Ns, P, Nf):
    """P(N) between the N of the theory grid, linear on the complementary log-log scale (as _n_at)."""
    return 1 - np.exp(-np.exp(np.interp(np.log(Nf), np.log(Ns), _cloglog(P))))


def p_dips_T(ax, C):
    """The probability that an average of N episodes decreases within the next T steps, from step 10,000, all interaction sets
    (lambda = 2).  Theory (tables alone; fr_theory_horizon.py): 1 - E_rho[(1 - p_rho)^T].  Simulation (fr_theory_dipsim.py, the same
    population): 2,000 groups for each of 24 N from 10 to 825 (12 per decade), followed for 10^4 steps, the first dip of each kept; every
    second N drawn (six per decade, evenly spaced: John 11-12), where 0 < P < 1 (elsewhere every group dipped, or none, as the lines say).
    John 11-11 night: "a lot more numerical points ... if an analytical line does not have enough numerical points across it, then we
    need to drop the line": T = 10^8 (no simulation reaches it) is dropped.
    N at P = 1/2: 67, 182, 433 for T = 10, 10^2, 10^4."""
    Z = np.load(os.path.join(A.OUT, "horizon_scaling.npz")); Zt = list(Z["Ts"]); S = np.load(os.path.join(A.OUT, "dip_sim.npz")); St = list(S["Ts"])
    Nf = np.logspace(np.log10(8), np.log10(1500), 400); sh = on_log_grid(S["Ns"], 1 / 6, base=10)   # N = 10, 15, 22 ... 681: every second simulated N
    for T_, col in zip(U_TS, GREENS4U):
        theory(ax, Nf, _cloglog_curve(Z["Ns"], Z["Pd"][:, Zt.index(T_)], Nf), col, lw=THIN)
        k = St.index(T_); m = sh & (S["P"][:, k] > 0) & (S["P"][:, k] < 1)
        simulated(ax, S["Ns"][m], S["P"][m, k], 2 * S["se"][m, k], col)
    log_n_axis(ax, 8, 1500); prob_axis(ax, PROB_TOP)
    lab = lambda T_: "1" if T_ == 1 else ("10" if T_ == 10 else rf"10^{{{int(np.log10(T_))}}}")
    ax.legend(handles=[h_both(col, (r"$T=" if i == 0 else "$") + lab(T_) + "$", lw=THIN) for i, (T_, col) in enumerate(zip(U_TS, GREENS4U))], loc="upper right", ncol=4,
              columnspacing=0.6, handlelength=1.0, handletextpad=0.3, borderaxespad=0.1)
    if CTX["ylab"]: ax.set_ylabel("P(dip)")                                                                   # (the next panel shares it)


def p_dips_q(ax, C):
    """As (o) for T = 30, by band of the trapped fraction q of the interaction set (John 11-11 late: more curves, a broader range, evenly
    spaced, structured values): 200 sets per band |q - c| < .03, c = .2, .35, .5, .65, .8 (fr_theory_additions.build_qbands; the thirds
    of q = .26 / .49 / .71 before).  Theory (the band's tables): 1 - E_R[(1 - p_R)^30]; simulation: 2,000 episodes per set from step
    10,000, in as many independent groups of N as they allow.  Labels: the band centres; pooled q .205 / .345 / .499 / .650 / .796.
    N at P = 1/2: 839, 258, 112, 53, 22 (0.5, 0.36, 0.33, 0.38 decades apart).  Theory and simulation within 2 SEM except at small N in
    the low bands (q = .2: .918 vs .899 at N = 10, .994 vs .990 at N = 20; q = .35: .988 vs .984 at N = 10): there groups made only of
    untrapped episodes are common ((1 - q)^N = .10 at N = 10), and some untrapped episodes have not settled by step 10,000 (their FR still
    falls, by up to ~2 nats a step), so such a group can dip; the asymptotic theory (Proposition 2: X = 0 at an untrapped joint
    realization) counts it as flat.
    DASHED: the main text's batch of 8,000 (MAIN); the theory lines run to N = 10^4, the simulation to 5,000."""
    hnd = []
    for i, (B, col) in enumerate(zip(C["A"]["qbands"]["bands"], F.BLUES5)):
        theory(ax, B["Nth"], B["theory"], col); simulated(ax, B["Ns"], B["sim"], 2 * B["se"], col)
        hnd.append(h_both(col, f"${B['center']:g}$"))                                    # q (the title says so)
    log_n_axis(ax, 8, 1.3e4); prob_axis(ax, PROB_TOP)                                      # to 10^4: the theory lines end there
    ax.axvline(MAIN["N"], ymax=0.80, **MAIN_LINE)                                          # the main text's batch (ends under the key): no dips left
    ax.legend(handles=hnd, loc="upper center", ncol=5, columnspacing=0.4, handlelength=0.7, handletextpad=0.2, borderaxespad=0.1)   # y as (q)


# ---------------------------------------------------------------- 11-11 latest: by third of q (the sets of o-q), accuracy, synergy, horizon
def _qsd(ax, C, k):
    """Propositions 1(c), 2(c) for the k-th third of q (800 interaction sets, lambda = 2): mean and s.d. of FR_t across episodes.  Theory
    (tables alone, at the simulated episodes' own joint realizations): t E[Delta] and sqrt(t^2 Var(Delta) + t E[v]).  Simulation: 50
    episodes per set from step 0 (2 SEM by resampling sets).  One y-range for the three thirds; red: zero.  No key: the row title says
    "Mean +- s.d. of FR_t" (thick line: the mean; thin lines and band: mean +- s.d.)."""
    Q = C["D"]["by_q"][k]; col = F.BLUES3[k]; t = np.r_[0, Q["t"]]; m = np.r_[0, Q["mean_th"]]; s = np.r_[0, Q["sd_th"]]
    ax.axhline(0, color=RED, lw=1.0, zorder=2.6)
    ax.fill_between(t, m - s, m + s, color=col, alpha=0.16, lw=0, zorder=2)
    for e in (m - s, m + s): theory(ax, t, e, col, lw=F.LW * 0.5)
    theory(ax, t, m, col)
    simulated(ax, Q["tm"], Q["mean_m"], 2 * Q["mean_se"], col)
    for v, e in (("lo_m", "lo_se"), ("hi_m", "hi_se")): simulated(ax, Q["tm"], Q[v], 2 * Q[e], col)
    allq = C["D"]["by_q"]; top = max(max((P["mean_th"] + P["sd_th"]).max(), P["hi_m"].max()) for P in allq); bot = min(min((P["mean_th"] - P["sd_th"]).min(), P["lo_m"].min()) for P in allq)
    ax.set_xlim(0, C["cfg"]["T"]); ax.set_xticks(np.arange(0, C["cfg"]["T"] + 1, 100)); ax.set_ylim(bot - 0.06 * (top - bot), top + 0.06 * (top - bot))
    ax.set_xlabel("step $t$"); ax.set_ylabel(r"$\mathrm{FR}_t$ (nats)")
    if k: ax.set_ylabel("")                                                            # one y label per triplet: the three panels share the range
    qtag(ax, C["D"]["by_q"][k]["q"])


def p_qsd_0(ax, C): _qsd(ax, C, 0)
def p_qsd_1(ax, C): _qsd(ax, C, 1)
def p_qsd_2(ax, C): _qsd(ax, C, 2)


def _qstep(ax, C, k):
    """Proposition 2(a) for the k-th third of q: the law of one step X of a trapped episode (the third's tables) against the first
    simulated step after step 10,000 (800 sets x 150 episodes, its trapped episodes); shared bins and axes; red: zero (a decrease).
    NEXT TO ZERO the simulated density lies above the law in the two thirds with the fewer traps (bins [-.1, 0) and [0, .1): +3.7 and
    +6.9 SEM at q = .26, +1.9 and +4.6 at .49, +1.1 and +1.0 at .71; found by the independent check of 11-13).  Checked on the caches
    thirds.pkl / thirds_more.pkl: 1.8 / 1.2 / 0.8% of the trapped episodes have |step| < 1e-3, against 0.16 / 0.23 / 0.36% under the law
    of X, and at step 10,000 the limiting walk (mean t Delta, variance t v) of 2.2 / 1.2 / 0.7% of the trapped joint realizations is
    still below zero: weak traps, at which the Naive belief has not yet left r_g, so FR does not move.  DIRECT TEST (11-13, 900 fresh
    sets x 150 episodes at lambda = 2, the Naive answer read at step 10,000; scratch near_zero_cause.py): the Naive belief is still on
    r_g in 2.1 / 1.2 / 0.6% of the trapped episodes of sets in the three thirds of q, and of the steps with |step| < 1e-3, 93 / 80 / 45%
    are theirs (median Delta of those episodes .0015 against .053 for all trapped ones).  The shares of decreases are unaffected
    (44.3 / 44.0, 38.9 / 38.9, 34.1 / 34.0%).  Said in the Details of the theory text.
    KEY "decreases": the line is the law's P(X < 0), the marker the share of simulated steps below zero (the figure's Theory /
    Simulation key; the words "theory" and "simulation" were as wide as half a 1.5-in panel and ran into the peak)."""
    ST = C["D"]["by_q"][k]["step"]; ed = ST["edges"]; mid = 0.5 * (ed[1:] + ed[:-1]); col = F.BLUES3[k]
    ax.stairs(ST["dens"], ed, fill=True, color=col, alpha=0.16, lw=0, zorder=2); ax.stairs(ST["dens"], ed, color=col, lw=THIN, zorder=3)
    sh = _thin(mid); simulated(ax, mid[sh], ST["sim"][sh], ST["sim_2se"][sh], col)                  # every third bin: the law stays visible
    ax.axvline(0, color=RED, lw=1.0, zorder=2.6); ax.set_xlim(-ST["xmax"], ST["xmax"])
    top = max(max(P["step"]["dens"].max(), P["step"]["sim"].max()) for P in C["D"]["by_q"]); ax.set_ylim(0, 1.12 * top)
    ax.legend(handles=[h_line(col, "", lw=F.LW * 0.8), h_dot(col, "")], labels=[f"{100 * ST['p_pred']:.0f}%", f"{100 * ST['p_sim']:.0f}%"], title="decreases",
              loc="upper right", handlelength=1.1, handletextpad=0.4, borderaxespad=0.2)   # line: the law; marker: simulation (the figure's key)
    ax.set_xlabel(r"$\mathrm{FR}_{t+1}-\mathrm{FR}_t$ (nats)"); ax.set_ylabel("density")
    if k: ax.set_ylabel("")
    qtag(ax, C["D"]["by_q"][k]["q"])


def _thin(mid, every=3):
    """Every `every`-th bin outward from the two bins next to zero (so the markers sit symmetrically and include the peak)."""
    i0 = int(np.searchsorted(mid, 0.0)); k = np.arange(len(mid)); return ((k >= i0) & ((k - i0) % every == 0)) | ((k < i0) & ((i0 - 1 - k) % every == 0))


def p_qstep_0(ax, C): _qstep(ax, C, 0)
def p_qstep_1(ax, C): _qstep(ax, C, 1)
def p_qstep_2(ax, C): _qstep(ax, C, 2)


def p_accuracy(ax, C):
    """Proposition 4: the Joint - Naive accuracy gap tends to the trapped fraction q (lines: q of each third, tables alone; markers: the
    third's 800 sets x 100 episodes, counts drawn binomially up to 10^9 steps, 2 SEM over sets)."""
    hnd = []
    for k, (Q, col) in enumerate(zip(C["D"]["by_q"], F.BLUES3)):
        A = Q["acc"]; sh = on_log_grid(A["t"], 1.0)                                    # one marker per decade (of four simulated)
        theory(ax, [0.7, 2e9], [Q["q"]] * 2, col); simulated(ax, np.asarray(A["t"])[sh], np.asarray(A["gap"])[sh], 2 * np.asarray(A["gap_se"])[sh], col)
        hnd.append(h_both(col, (r"$q=" if k == 0 else "$") + f"{Q['q']:.2f}$"))
    ax.set_xscale("log"); ax.set_xlim(0.7, 2e9); ax.set_xticks([1e0, 1e3, 1e6, 1e9]); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_minor_locator(LogLocator(numticks=12, subs=[1.0])); ax.set_ylim(0, 0.95); ax.set_yticks([0, 0.25, 0.5, 0.75]); decimals(ax, 2)
    ax.legend(handles=hnd, loc="upper left", ncol=3, columnspacing=0.55, handlelength=1.0, handletextpad=0.3, borderaxespad=0.1)
    ax.set_xlabel("step $t$"); ax.set_ylabel(r"Joint $-$ Naive accuracy")


def p_synergy(ax, C):
    """Proposition 3 and 'When does Factorization Regret bound synergy?': in the Gridworld's population (2,000 sets x 50 episodes,
    lambda = 2) the synergy I(r_g; r_c | o_1:t) exceeds E[FR_t] at the first step, then vanishes (it is at most Lambda_t -> 0) while
    E[FR_t] grows like t E[Delta] (line: tables alone).  Simulated synergy is drawn where it is above 2 SEM.  Key: the symbols of the
    theory text (Syn_t = I(r_g; r_c | o_1:t)); the title names them."""
    S = C["D"]["syn"]; t = np.asarray(S["t"]); col_f, col_s = F.VIOLETS3[1], GOLD; sh = on_log_grid(t, 0.5)   # t = 1, 3, 10, 32 ... 10^4
    theory(ax, [0.8, 1.3e4], np.array([0.8, 1.3e4]) * S["EDelta"], col_f); simulated(ax, t[sh], S["FR"][sh], 2 * S["FR_se"][sh], col_f, clip=True)
    ok = sh & (S["Syn"] > 2 * S["Syn_se"]); simulated(ax, t[ok], S["Syn"][ok], 2 * S["Syn_se"][ok], col_s, clip=True)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.8, 1.3e4); ax.set_ylim(1e-4, 3e3); ax.set_yticks([1e-4, 1e-2, 1e0, 1e2])
    ax.xaxis.set_major_locator(LogLocator(numticks=6)); ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(handles=[h_both(col_f, r"$\mathbb{E}[\mathrm{FR}_t]$"), h_dot(col_s, r"$\mathrm{Syn}_t$")], loc="upper left", handlelength=1.3, borderaxespad=0.15)
    ax.set_xlabel("step $t$"); ax.set_ylabel("nats")


def _cloglog(p): return np.log(-np.log1p(-np.clip(p, 1e-12, 1 - 1e-12)))


def _n_at(Ns, P, X):
    """N at which P(N), decreasing, crosses X, interpolated between the two bracketing N on the complementary log-log scale ln(-ln(1 - P)),
    on which P(dip within T) = 1 - exp(-T p_N), p_N ~ e^{-N I}, is linear in N; None unless two N with 0 < P < 1 bracket X."""
    for i in range(len(Ns) - 1):
        a, b = P[i], P[i + 1]
        if a >= X > b and 0 < b and a < 1:
            return float(Ns[i] + (_cloglog(X) - _cloglog(a)) * (Ns[i + 1] - Ns[i]) / (_cloglog(b) - _cloglog(a)))
    return None


HORIZON_P = (0.1, 0.5, 0.9, 0.99)                                                      # (t): chances of NO dip within T steps (John 11-11 night: only levels
GRAYS4 = ["#a8a7a0", "#767570", "#45443f", "#0b0b0b"]                                   # the simulation covers across T); colours: green would read as T


def p_horizon(ax, C):
    """How many episodes must be averaged for a chance P that the average does NOT dip within T steps?  Theory (tables alone;
    fr_theory_horizon.py, the population of o): the N at which P(no dip within T) = P, for P = 10, 50, 90, 99% (lines, evaluated at the
    T of the theory grid), and the Chernoff + union bound N = ln(T / (1 - P)) / I for 99% (dotted: beyond it P(no dip) >= 99%).  Simulation
    (fr_theory_dipsim.py, the groups of o: 2,000 per N, 24 N from 10 to 825): the N at which the simulated chance crosses P, at 13
    horizons, three per decade (T = 1, 2, 5, 10, 22 ... 10^4: evenly spaced, John 11-12), where two simulated N bracket it (bars: the
    crossing of P -+ 2 SEM; an independent run on 41 N gave the same crossings within the bars).  John 11-11 night: every line needs simulated points
    across it, else it goes: T stops at 10^4 (the simulated horizon) and 99.9 / 99.99% went (only 100,000-group runs over a few steps
    reach them; there the simulation lies 5-20 episodes above the asymptotic theory: episodes not settled at step 10,000, see
    fr_theory_dipsim.py).  N on a linear axis: the lines are straight in ln T (N at T = 10^4: 311, 433, 576, 733)."""
    Z = np.load(os.path.join(A.OUT, "horizon_scaling.npz")); Ns, Ia = Z["Ns"], float(Z["Ia"]); S = np.load(os.path.join(A.OUT, "dip_sim.npz"))
    keepT = Z["Ts"] <= 10 ** 4; Ts, Pd = Z["Ts"][keepT].astype(float), Z["Pd"][:, keepT]; Tf = np.logspace(0, 4, 60); hnd = []
    for P, col in zip(HORIZON_P, GRAYS4):
        X = 1 - P                                                                       # the matching chance of a dip
        n = np.array([_n_at(Ns, Pd[:, j], X) or np.nan for j in range(len(Ts))]); ok = np.isfinite(n); theory(ax, Ts[ok], n[ok], col, lw=THIN)
        if P == max(HORIZON_P): theory(ax, Tf, np.log(Tf / X) / Ia, col, lw=THIN * 1.3, ls=(0, (1, 1.4)))          # its guarantee
        xs, ys, lo, hi = [], [], [], []
        for j in np.flatnonzero(on_log_grid(S["Ts"], 1 / 3)):                         # T = 1, 2, 5, 10, 22 ... 10^4
            T_ = S["Ts"][j]; y, e = S["P"][:, j], 2 * S["se"][:, j]; c = _n_at(S["Ns"], y, X)
            if c is None: continue
            a, b = _n_at(S["Ns"], y - e, X), _n_at(S["Ns"], y + e, X); xs.append(T_); ys.append(c); lo.append(c - (a if a is not None else c)); hi.append((b if b is not None else c) - c)
        if xs: simulated(ax, xs, ys, np.abs(np.vstack([lo, hi])), col)
        hnd.append(h_both(col, (r"$P=$" if P == max(HORIZON_P) else "") + f"{100 * P:g}%", lw=THIN))
    ax.set_xscale("log"); ax.set_xlim(0.7, 1.5e4); ax.set_xticks([1e0, 1e1, 1e2, 1e3, 1e4]); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 1150); ax.set_yticks([0, 250, 500, 750, 1000])
    ax.axvline(MAIN["T"], ymax=np.log(MAIN["T"] / (1 - max(HORIZON_P))) / Ia / 1150, **MAIN_LINE)   # the main text's horizon, up to the guarantee
    leg = ax.legend(handles=[hnd[3], hnd[1], hnd[2], hnd[0]], loc="upper left", ncol=2, columnspacing=0.7, handlelength=1.2,
                    handletextpad=0.35, borderaxespad=0.15)                                # row-wise: 99, 90 / 50, 10%
    ax.add_artist(leg); fig = ax.figure; fig.draw_without_rendering()                      # the Chernoff key: a third row right under that key, clear of
    bb = leg.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())   # every line and marker (John 11-12: at the lower
    rc = plt.rcParams; fs = leg.get_texts()[0].get_fontsize()                              # right the 10% markers covered its dotted sample)
    up = (2 * rc["legend.borderpad"] - rc["legend.labelspacing"]) * fs / 72 / (ax.get_position().height * fig.get_figheight())   # same row spacing
    ax.legend(handles=[Line2D([], [], color=GRAYS4[-1], lw=THIN * 1.3, ls=(0, (1, 1.4)), label="Chernoff")], loc="upper left",
              bbox_to_anchor=(bb.x0, bb.y0 + up), handlelength=1.6, handletextpad=0.35, borderaxespad=0.0)
    ax.set_xlabel(r"horizon $T$ (steps)"); ax.set_ylabel(r"episodes $E$" if CTX["ylab"] else "")   # (in a column without room for a y label the title names it)


PANELS = {k[2:]: v for k, v in globals().items() if k.startswith("p_") and callable(v)}


# ============================================================================================================
# layout, checks, output
# ============================================================================================================
def headroom(fig, ax, artists, pts, pad=0.03):
    """Raise (never lower) the upper y-limit just enough for the data to pass `pad` (axes fraction) below each artist above them."""
    R = fig.canvas.get_renderer(); inv = ax.transAxes.inverted()
    P = inv.transform(ax.transData.transform(np.vstack(pts))); f = 1.0
    for a in artists:
        b = a.get_window_extent(R).transformed(inv); m = (P[:, 0] > b.x0 - 0.015) & (P[:, 0] < b.x1 + 0.015)
        if m.any() and b.y0 - pad > 0.1: f = max(f, P[m, 1].max() / (b.y0 - pad))
    ax.set_ylim(0, ax.get_ylim()[1] * f)


def ink_rows(ink, dpi, x0, x1, y0, y1):
    """The runs of rows holding ink in a box of a rendered page (inches from its left and top; ink: boolean pixels) -> (first, last + 1) in inches."""
    dark = np.r_[False, ink[int(y0 * dpi):int(y1 * dpi), int(x0 * dpi):int(x1 * dpi)].any(1), False]
    return int(y0 * dpi) / dpi + np.flatnonzero(dark[1:] != dark[:-1]).reshape(-1, 2) / dpi


def text_checks(fig, rows, heads, merged=None):
    """Print text that overlaps other text or leaves the figure, and panel titles that come too close (as fr_theory.finish_figure); the
    clearances of the panel letters (under the axis labels) and of the row titles (over the panel titles, under the letters above)."""
    from matplotlib.text import Text
    fig.draw_without_rendering(); R = fig.canvas.get_renderer(); boxes = []; pooled = set(); drawn = []; own = set()
    for ax in fig.findobj(lambda o: hasattr(o, "xaxis") and hasattr(o, "yaxis")):
        own |= {(id(ax.xaxis.label), id(tk.label1)) for tk in ax.xaxis.majorTicks + ax.xaxis.minorTicks}   # an x label and its own tick labels: the ink
        for axis in (ax.xaxis, ax.yaxis):                                                                  # check of make() (their boxes are deeper than their ink)
            pooled |= {id(lb) for tk in axis.majorTicks + axis.minorTicks for lb in (tk.label1, tk.label2)}
            lo_, hi_ = sorted(axis.get_view_interval())
            for locs, ticks in ((axis.get_majorticklocs(), axis.get_major_ticks), (axis.get_minorticklocs(), axis.get_minor_ticks)):
                drawn += [tk.label1 for tk, loc in zip(ticks(len(locs)), locs) if lo_ - 1e-9 <= loc <= hi_ + 1e-9 and tk.label1.get_visible()]
    for t in [t for t in fig.findobj(Text) if id(t) not in pooled] + drawn:
        if t.get_visible() and t.get_text().strip() and t.figure is not None:
            bb = t.get_window_extent(R)
            if bb.width > 0: boxes.append((t.get_text(), bb, id(t)))
    n = 0
    for s, bb, _ in boxes:
        if bb.x0 < fig.bbox.x0 - 0.5 or bb.y0 < fig.bbox.y0 - 0.5 or bb.x1 > fig.bbox.x1 + 0.5 or bb.y1 > fig.bbox.y1 + 0.5: print(f"   check: text outside the figure: {s!r}"); n += 1
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (s1, b1, i1), (s2, b2, i2) = boxes[i], boxes[j]
            if (i1, i2) in own or (i2, i1) in own: continue
            if b1.overlaps(b2) and min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > 1 and min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > 1:
                print(f"   check: texts overlap: {s1!r} / {s2!r}"); n += 1
    for row in rows:
        for l, r in zip(row[:-1], row[1:]):
            gap = (r.title.get_window_extent(R).x0 - l.title.get_window_extent(R).x1) / fig.dpi
            if gap < 0.06: print(f"   check: only {gap:.2f} in between the titles {l.get_title()!r} and {r.get_title()!r}"); n += 1
    ext = lambda t: t.get_window_extent(R); inch = lambda px: px / fig.dpi
    under = [inch(ext(a.xaxis.label).y0 - ext(a.letter).y1) for row in rows for a in row]             # axis label -> its panel letter
    merged = merged or [False] * len(rows)                                                            # merged: the row title is on the panel-title line
    over = [inch(ext(h).y0 - max(ext(a.title).y1 for a in row)) for h, row, m in zip(heads, rows, merged) if not m]   # panel titles -> their row title
    for h, row, m in zip(heads, rows, merged):                                                        # a merged row title and the next panel title
        nxt = [a for a in row if a.get_title()]
        if m and nxt and inch(ext(nxt[0].title).x0 - ext(h).x1) < 0.15: print(f"   check: the row title {h.get_text()!r} runs into the title {nxt[0].get_title()!r}"); n += 1
    above = [inch(min(ext(a.letter).y0 for a in up) - ext(h).y1) for h, up in zip(heads[1:], rows[:-1])]   # the letters above -> the row title
    print(f"   clearances (in): panel letters {min(under):.3f}-{max(under):.3f} under the axis labels; row titles {min(over):.3f}-{max(over):.3f} over the panel "
          f"titles" + (f" and {min(above):.3f}-{max(above):.3f} under the letters above" if above else ""))
    if min(under) < 0.03 or (over and min(over) < 0.06): print("   check: a panel letter or a row title is too close to its neighbours"); n += 1
    print(f"   text checks: {n} problem(s)")


def make(C, layout=LAYOUT, name=NAME, titles=None, gaps=None, ylab_cols=None):
    """gaps: the gaps between the columns (default GXS for four columns); ylab_cols: the columns with room for a y label (default: with
    GXS the first and the last, else all)"""
    plt.rcParams.update(F.RC); GRID_DEV.clear()
    ncol = max(len(r[1]) for r in layout); nrow = len(layout); titles = titles or TITLES; CTX["letters"] = letters_of(layout)[0]
    if gaps is None: gaps = GXS if len(GXS) == ncol - 1 else (GX,) * (ncol - 1)
    if ylab_cols is None: ylab_cols = (0, ncol - 1) if gaps is GXS else tuple(range(ncol))
    xcol = [LM + j * PW + sum(gaps[:j]) for j in range(ncol)]; W = xcol[-1] + PW + RM
    spans = [r[2] if len(r) > 2 else len(r[1]) for r in layout]; merged = [sp < len(r[1]) and "stacked" not in r[3:] for sp, r in zip(spans, layout)]   # merged: the row title
    TB = MERGED_PAD / 72; tops = [TB + TOP - HEAD_Y if m else TOP for m in merged]; phs = [PH_ROW.get(i, PH) for i in range(nrow)]  # on the panel-title line
    H = sum(tops) + sum(phs) + nrow * BOT + (nrow - 1) * GAP + FOOT; y0s = []; y = H
    for tp, ph in zip(tops, phs): y -= tp + ph; y0s.append(y); y -= BOT + GAP
    fig = plt.figure(figsize=(W, H)); rows = []; heads = []; extra = []; letters = iter("abcdefghijklmnopqrstuvwxyz")
    for i, r in enumerate(layout):
        header, names = r[0], r[1]; span = spans[i]; xh = 0.5 * (xcol[0] + xcol[span - 1] + PW); y0, ph = y0s[i], phs[i]; row = []   # the title over the panels it names
        heads.append(fig.text(xh / W, (y0 + ph + (TB if merged[i] else HEAD_Y)) / H, header, fontfamily="cmb10", fontsize=F.FS["header"], ha="center", va="baseline"))
        for j, pname in enumerate(names):
            ax = fig.add_axes([xcol[j] / W, y0 / H, PW / W, ph / H]); CTX["inlabel"] = merged[i] and j < span; CTX["ylab"] = j in ylab_cols
            out = PANELS[pname](ax, C) or {}
            tt = titles[pname]; ax.set_title("" if CTX["inlabel"] else (tt(C) if callable(tt) else tt), pad=MERGED_PAD if merged[i] else TITLE_PAD)
            ax.tick_params(which="both", width=0.6); row.append(ax)
            ax.letter = fig.text((xcol[j] + 0.5 * PW) / W, (y0 - LETTER_Y) / H, f"({next(letters)})", **LETTER)
            if "headroom" in out: extra.append((ax, *out["headroom"]))
        rows.append(row)
    CTX.update(inlabel=False, ylab=True)
    key = [Line2D([], [], color=F.INK, lw=F.LW, label="Theory"), h_dot(F.INK, "Simulation")]     # (mean +- 2 SEM: said in the caption; the long form
                                                                                                # ran into the first row title once that named the world)
    if len(layout[0][1]) < ncol:                                                           # an empty slot in row 1 (which has no markers): the key there
        j = len(layout[0][1]); fig.legend(handles=key, loc="center left", bbox_to_anchor=(xcol[j] / W, (y0s[0] + 0.5 * phs[0]) / H), ncol=1,
                                          fontsize=F.FS["label"], handlelength=1.8, labelspacing=0.8, borderpad=0.1, frameon=False)
    else:                                                                                  # else at the right end of a row title's line, on its baseline
        yb = y0s[KEY_ROW] + phs[KEY_ROW] + HEAD_Y; kw = dict(loc="center right", ncol=2, fontsize=F.FS["label"], handlelength=1.8, columnspacing=1.4, borderpad=0.1, frameon=False)
        lg = fig.legend(handles=key, bbox_to_anchor=(1 - RM / W, yb / H), **kw); fig.draw_without_rendering(); t = lg.get_texts()[0]
        off = yb - t.get_transform().transform(t.get_position())[1] / fig.dpi; lg.remove()                # how far its text sits from the title's baseline
        fig.legend(handles=key, bbox_to_anchor=(1 - RM / W, (yb + off) / H), **kw)
    for _ in range(2):                                                                     # second pass: tick labels may have changed the layout
        fig.draw_without_rendering()
        for ax, arts, pts in extra: headroom(fig, ax, arts, pts)
    for row, ph in zip(rows, phs):
        for ax in row: ax.xaxis.label.set_verticalalignment("baseline"); ax.xaxis.set_label_coords(0.5, -XLAB_Y / ph)   # one baseline
    text_checks(fig, rows, heads, merged)
    if GRID_DEV: print(f"   thinned markers: at most {max(GRID_DEV):.3f} decades off their even log grid")
    stem = os.path.join(OUT, name); fig.savefig(stem + ".pdf"); fig.savefig(stem + ".png", dpi=150)
    try:
        from main.utils import fig_path
        fig.savefig(fig_path(f"{name}.png"), dpi=150); fig.savefig(fig_path(f"{name}.pdf")); print("   also saved in figure_bin")
    except Exception as err:
        print(f"   figure_bin copy skipped ({err})")
    from PIL import Image                                                                  # 300-dpi crops per row, for checking by eye
    insp = os.path.join(OUT, "fr_theory_inspect"); os.makedirs(insp, exist_ok=True); big = os.path.join(insp, f"{name}_300dpi.png"); fig.savefig(big, dpi=300)
    im = Image.open(big); Wp, Hp = im.size; ink = np.asarray(im.convert("L")) < 128; gaps = []; bad = []
    for i, row in enumerate(rows):                                                         # the ink under each panel: tick labels, axis label, letter
        for j, ax in enumerate(row):
            x0 = xcol[j]; yb = H - y0s[i]; runs = ink_rows(ink, Wp / W, x0 - 0.05, x0 + PW + 0.05, yb + 0.045, yb + BOT + 0.01)
            if len(runs) == 3: gaps.append((runs[1, 0] - runs[0, 1], runs[2, 0] - runs[1, 1]))
            else: bad.append(ax.letter.get_text())
    gaps = np.array(gaps).reshape(-1, 2)
    if len(gaps): print(f"   ink at 300 dpi (in): axis labels {gaps[:, 0].min():.3f}-{gaps[:, 0].max():.3f} under their tick labels, panel letters {gaps[:, 1].min():.3f}-"
                        f"{gaps[:, 1].max():.3f} under the axis labels")
    if bad or (len(gaps) and gaps.min() < 0.04): print(f"   check: tick labels, axis label and letter do not stand clear of one another (not three lines of ink: {bad})")
    for i in range(nrow):
        top = int((H - y0s[i] - phs[i] - tops[i] - 0.02) / H * Hp); bot = int(min(H, H - y0s[i] + BOT + 0.02) / H * Hp)
        for part, (x0, x1) in (("left", (0, 0.55)), ("right", (0.45, 1.0))):
            im.crop((int(x0 * Wp), max(top, 0), int(x1 * Wp), min(bot, Hp))).save(os.path.join(insp, f"{name}_row{i + 1}_{part}.png"))
    plt.close(fig); print(f"saved {stem}.pdf/.png ({W:.2f} x {H:.2f} in) and crops in {insp}/")


def cfg_pool(cfg): return cfg["T_POOL"]


def numbers(C):
    """The values behind every statement of the theory text (FR_theory_block_v16.tex: "What the figure shows", "Details", and the
    sentence on synergy in the Gridworld) about a panel of this figure; the letters follow LAYOUT.  Read-only: the caches the figure is
    drawn from, plus the population cache and the dip simulation for what the text says about episodes at step 10,000."""
    cfg, D, S = C["cfg"], C["D"], C["summary"]; p = print; pc = lambda v: f"{100 * v:.1f}%"; _, g = letters_of()
    S_ = C["A"]["small"]; lam0 = S_["lam"]; ts = S_["ts"]; ea, eb = S_["an"]["a"], S_["an"]["b"]; kst = S_["k_stat"]
    p(f"ONE OBSERVABLE, R = {S_['R']} (the paper's generator; every line exact)")
    p(f"  selection: of the interaction sets drawn in order, set {S_['i_free']} is the first with no trap at lambda = {', '.join(f'{l:g}' for l in S_['lams'])} and set {S_['i_one']} the first with "
      f"exactly one at lambda = {lam0:g} ({S_['n_drawn']} sets drawn); of {len(kst):,} other sets at lambda = {lam0:g}, {pc(np.mean(kst == 0))} have no trap and {pc(np.mean(kst == 1))} exactly one")
    p(f"  ({g('sw_anatomy_k0')}, {g('sw_anatomy_k1')}) trapped joint realizations: {ea['st']['k']} and {eb['st']['k']} of {eb['ell'].size}; E[FR_t] without a trap peaks at "
      f"{ea['an']['FR'].max():.3f} nats at step {int(ea['an']['FR'].argmax())}; with one, E[FR_t] / (t Delta / 9) = {eb['an']['FR'][-1] / ((len(eb['an']['FR']) - 1) * eb['st']['Delta'][eb['st']['trapped']][0] / 9):.2f} "
      f"at step {len(eb['an']['FR']) - 1} (Delta = {eb['st']['Delta'][eb['st']['trapped']][0]:.4f}); identity E[FR_t] = t E[Delta] + N~_t - Lambda_t (Prop. 1a of the short text) to "
      f"{max(np.abs(e['an']['FR'] - (np.arange(len(e['an']['FR'])) * e['st']['Delta'].mean() + e['an']['N'] - e['an']['Lam'])).max() for e in (ea, eb)):.0e}; E[Delta] = "
      f"{ea['st']['Delta'].mean():.4f} / {eb['st']['Delta'].mean():.4f}; N~_t at steps 100, 500: {ea['an']['N'][100]:.3f}, {ea['an']['N'][500]:.3f} / {eb['an']['N'][100]:.3f}, {eb['an']['N'][500]:.3f}; "
      f"Lambda_t: {ea['an']['Lam'][100]:.3f}, {ea['an']['Lam'][500]:.4f} / {eb['an']['Lam'][100]:.3f}, {eb['an']['Lam'][500]:.3f}")
    syn, frb = eb["an"]["Syn"], eb["an"]["FR"]; later = np.flatnonzero(frb[1:] <= syn[1:]) + 1
    p(f"      ({g('sw_anatomy_k1')}) 'FR_1 = 0 < Syn_1 ... E[FR_t] > Syn_t for all large t': E[FR_1] = {frb[1]:.1e}, Syn_1 = {syn[1]:.4f}; E[FR_t] > Syn_t from step {int(later.max()) + 1 if len(later) else 1} on "
      f"(to step {len(frb) - 1}: Syn {syn[-1]:.4f}, E[FR] {frb[-1]:.2f})")
    for key, name in ((("free", lam0), g("sw_anatomy_k0")), (("one", lam0), g("sw_anatomy_k1"))):
        sm, an = S_["sims"][key], (ea if key[0] == "free" else eb)["an"]; mf = np.isin(ts, S_["t_lin"]); mj = np.isin(ts, S_["t_log"]) & (ts >= 20) & (ts <= (len(an["FR"]) - 1) // 2)
        p(f"      ({name}) simulation ({S_['n_ep']:,} episodes) - exact at the markers: E[FR_t] at most {np.abs((sm[mf, 0] - an['FR'][ts[mf]]) / sm[mf, 1]).max():.1f} SEM, "
          f"Joint term at most {np.abs((sm[mj, 2] - an['Lam'][ts[mj]]) / sm[mj, 3]).max():.1f} SEM")
    tl = list(S_["tl"])
    for lam in S_["lams"]:
        T_ = S_["temp"][lam]; e = T_["efr"]; j = int(e.argmax()); sm = S_["sims"][("free", lam)]; m = np.isin(ts, S_["t_log"]) & (sm[:, 0] > 2 * sm[:, 1]) & (sm[:, 0] > 1e-4)
        z = [(sm[i, 0] - e[tl.index(ts[i])]) / sm[i, 1] for i in np.flatnonzero(m) if ts[i] in tl]
        p(f"  ({g('sw_time_scale')}) lambda = {lam:g}: traps {T_['st']['k']}; pull delta of the weakest distractor {T_['st']['delta']:.4f}; E[FR_t] peaks at {e[j]:.3f} nats at step {tl[j]:,}; "
          f"peak step x delta = {tl[j] * T_['st']['delta']:.1f}; {len(z)} markers, simulation - exact at most {np.abs(z).max():.1f} SEM")
    E0, E1, w = S_["E0"], S_["E1"], S_["w_star"]
    def dec(r):
        d = np.flatnonzero(np.diff((1 - r * w) * E0 + r * w * E1)[1:] < -1e-13) + 1; return f"{d.min()}-{d.max()}" if len(d) else "none"
    p(f"  ({g('sw_mixture')}) q* = {S_['q_star']:.6f} (a fraction {w:.4f} of the episodes in the one-trap set), critical step t* = {S_['t_star']}; steps at which the expected curve decreases (of "
      f"{len(E0) - 1}): " + "; ".join(f"q = {r:g} q*: {dec(r)}" for r in (0.0, 0.5, 1.0, 2.0)))
    iN = [int(np.argmin(np.abs(S_["Ns"] - N))) for N in (1e3, 1e6, 1e9)]; zq = (S_["sim"] - S_["p_star"][:, [list(np.round(S_["Ns"], 6)).index(round(N, 6)) for N in S_["sim_Ns"]]]) / S_["sim_se"]
    p(f"  ({g('sw_critical')}) P(an average of N episodes decreases at t*) at N = 10^3, 10^6, 10^9: "
      + "; ".join(f"q = {r:g} q*: " + ", ".join(f"{v:.4f}" for v in S_["p_star"][S_["ratios"].index(r)][iN]) for r in S_["ratios"]))
    p("      Fourier transform / saddlepoint at N = 1e+06: " + "; ".join(f"{a_:.5f} / {b_:.5f}" for a_, b_ in S_["both"]) + f" (largest difference {max(abs(a_ - b_) for a_, b_ in S_['both']):.0e}); "
      f"simulation ({S_['paths']:,} averages per N) - exact: at most {np.abs(zq[:, S_['sim_Ns'] >= 1e3]).max():.1f} SEM over {int((S_['sim_Ns'] >= 1e3).sum()) * len(S_['ratios'])} markers")
    p("FIVE OBSERVABLES, R = 10 (the Cognitive Gridworld of the main text)")
    p("  'about half of the joint realizations are trapped at every lambda': q " + ", ".join(f"{S[l]['q']:.3f}" for l in cfg["LAMS_FULL"])
      + f" at lambda = {', '.join(f'{l:g}' for l in cfg['LAMS_FULL'])}; over lambda {min(cfg['LAMS_POOL']):g}-{max(cfg['LAMS_POOL']):g}: {D['numbers']['q_min']:.3f}-{D['numbers']['q_max']:.3f}")
    p("  'almost every interaction set has some': smallest trapped fraction of a simulated set " + ", ".join(f"{D['curve'][l]['min_frac']:.2f}" for l in cfg["LAMS_FULL"]))
    for k, (Q, FN) in enumerate(zip(D["by_q"], D["fan3"])):
        i = len(Q["tm"]) - 1; t = Q["tm"][i]; mt, st_ = Q["mean_th"][t - 1], Q["sd_th"][t - 1]; off = Q["mean_m"] - Q["mean_th"][Q["tm"] - 1]; mu, sg = FN["mean"], FN["sigma"]
        st = Q["step"]; i0 = int(np.searchsorted(st["edges"], 0.0)); z0 = [(st["sim"][j] - st["dens"][j]) / (st["sim_2se"][j] / 2) for j in (i0 - 1, i0)]
        p(f"  third {k + 1} (q = {Q['q']:.3f}, {Q['n_sets']} sets)")
        p(f"    ({g('qsd_0', 'qsd_2')}) step {t}: mean {mt:.2f} theory, {Q['mean_m'][i]:.2f} simulation (simulation - line at the markers: {off.min():.2f} to {off.max():.2f} nats); "
          f"s.d. {st_:.2f} theory, {Q['hi_m'][i] - Q['mean_m'][i]:.2f} simulation; s.d. / mean {st_ / mt:.2f}")
        p(f"    ({g('qstep_0', 'qstep_2')}) decreases: P(X < 0) = {pc(st['p_pred'])} theory, {pc(st['p_sim'])} of {st['n']:,} simulated steps; density in the two bins next to "
          f"zero, simulation - law: {st['sim'][i0 - 1] - st['dens'][i0 - 1]:+.3f}, {st['sim'][i0] - st['dens'][i0]:+.3f} ({z0[0]:+.1f}, {z0[1]:+.1f} SEM)")
        p(f"    ({g('avg_lo', 'avg_hi')}) E[Delta] = {mu:.4f}, sigma = {sg:.4f}: mean > s.d. from N = (sigma / E[Delta])^2 = {(sg / mu) ** 2:.0f}; band edges, simulation - theory: at most "
          f"{np.abs(FN['msd'] - np.c_[mu - sg / np.sqrt(FN['Ns']), mu + sg / np.sqrt(FN['Ns'])]).max():.4f} nats")
    for lam in cfg["LAMS_FULL"]:
        cv = D["curve"][lam]; d = np.diff(np.r_[0, cv["mean"]])
        p(f"  ({g('linear')}) lambda = {lam:g}: E[Delta] = {S[lam]['EDelta']:.4f}; step {cfg['T']}: line {cfg['T'] * S[lam]['EDelta']:.2f}, simulation {cv['mean'][-1]:.2f} +- {cv['se'][-1]:.2f}; "
          f"remainder N-tilde - Lambda there {cv['off'][-1]:.2f}; steps at which the simulated mean decreases: {int((d < 0).sum())} of {len(d)}; "
          f"untrapped episodes: mean FR peaks at {cv['U'].max():.2f} (step {cv['U_peak']}), {cv['U'][-1]:.2f} at step {cfg['T']}")
    OP = {**C["A"]["opt_more"], **D["opt"]}; xs = list(cfg["OPT_DOS"])
    for R_ in OBS_RS:
        ed = np.array([OP[(R_, d)]["ed"][0] for d in xs]); q = np.array([OP[(R_, d)]["q"][0] for d in xs])
        z = [(OP[(R_, d)]["slope"][0] - OP[(R_, d)]["ed"][0]) / OP[(R_, d)]["slope"][1] for d in xs if OP[(R_, d)]["slope"][1] > 0]     # (no trap in any set: both 0)
        p(f"  ({g('observables')}) R = {R_}: E[Delta] largest at d_o = {xs[int(ed.argmax())]}, falling from there on: {bool(np.all(np.diff(ed[int(ed.argmax()):]) <= 0))}; at d_o = 1, 5, 30: E[Delta] "
          + ", ".join(f"{ed[xs.index(d)]:.4f}" for d in (1, 5, 30)) + "; q " + ", ".join(f"{q[xs.index(d)]:.3f}" for d in (1, 5, 30)) + "; E[Delta] / q "
          + ", ".join(f"{ed[xs.index(d)] / q[xs.index(d)]:.3f}" for d in (1, 5)) + (f"; sets without a trap at d_o = 5: {pc(OP[(R_, 5)]['no_trap'])}" if "no_trap" in OP[(R_, 5)] else "")
          + f"; simulated slope - E[Delta]: at most {np.abs(z).max():.1f} s.e.")
    p("      a lower growth rate with fewer realizations at every d_o: " + str(all(OP[(a, d)]["ed"][0] < OP[(b, d)]["ed"][0] for a, b in zip(OBS_RS[:-1], OBS_RS[1:]) for d in xs))
      + "; not drawn, in the caches: R = 10 largest at d_o = " + str(xs[int(np.argmax([D["opt"][(10, d)]["ed"][0] for d in xs]))]) + ", R = 20 at d_o = " + str(xs[int(np.argmax([D["opt"][(20, d)]["ed"][0] for d in xs]))]))
    Sy = D["syn"]; t = np.asarray(Sy["t"]); i3 = int(np.argmin(np.abs(t - 1000))); rem = Sy["FR"] / (t * Sy["EDelta"])
    p(f"  ({g('synergy')}) lambda = {cfg['LAM_MAIN']:g}; step 1: synergy {Sy['Syn'][0]:.3f}, E[FR] {Sy['FR'][0]:.3f}; synergy > E[FR] at the steps {list(t[Sy['Syn'] > Sy['FR']])}; step {t[i3]}: synergy "
      f"{Sy['Syn'][i3]:.5f}, E[FR] {Sy['FR'][i3]:.1f}; E[FR] / line: at most {rem.max():.2f} (step {t[int(rem.argmax())]}), {rem[-1]:.3f} at step {t[-1]}")
    Z = np.load(os.path.join(A.OUT, "horizon_scaling.npz")); Zt = list(Z["Ts"]); Ia = float(Z["Ia"]); at = lambda T_, X: _n_at(Z["Ns"], Z["Pd"][:, Zt.index(T_)], X)
    p(f"  ({g('dips_T')}) N at P(dip within T) = 1/2: " + ", ".join(f"T = {T_}: {at(T_, 0.5):.0f}" for T_ in U_TS[1:]) + f" (T = 1: {Z['Pd'][0, 0]:.2f} at N = {Z['Ns'][0]}, below 1/2 from the start)")
    QB = C["A"]["qbands"]["bands"]
    p(f"  ({g('dips_q')}) N at P(dip within 30 steps) = 1/2: " + ", ".join(f"q = {B['center']:g}: {_n_at(B['Nth'], B['theory'], 0.5):.0f}" for B in QB)
      + "; simulation - theory at N = 10: " + ", ".join(f"{B['sim'][0] - B['theory'][list(B['Nth']).index(B['Ns'][0])]:+.3f}" for B in QB)
      + "; trapped fractions of the sets: " + ", ".join(f"{B['q_sets'].min():.2f}-{B['q_sets'].max():.2f}" for B in QB))
    p(f"  ({g('horizon')}) Chernoff rate I = {Ia:.4f}; N for a 99% chance of no dip within T = 10, 10^2, 10^3, 10^4 steps: " + ", ".join(f"{at(T_, 0.01):.0f}" for T_ in (10, 100, 1000, 10 ** 4))
      + "; the guarantee ln(T / 0.01) / I: " + ", ".join(f"{np.log(T_ / 0.01) / Ia:.0f}" for T_ in (10, 100, 1000, 10 ** 4)))
    jT = Zt.index(MAIN["T"]); PL = C["A"]["pool"]; bq = QB[0]
    p(f"  THE MAIN TEXT'S REGIME (T = {MAIN['T']} steps, batches of N = {MAIN['N']:,}; dashed in ({g('dips_q')}, {g('horizon')})): N for a 10 / 50 / 90 / 99% chance of no dip within {MAIN['T']} steps: "
      + ", ".join(f"{_n_at(Z['Ns'], Z['Pd'][:, jT], 1 - P_):.0f}" for P_ in HORIZON_P) + f"; the guarantee ln(T / 0.01) / I: {np.log(MAIN['T'] / 0.01) / Ia:.0f}; Chernoff bound on a dip within "
      f"{MAIN['T']} steps at N = {MAIN['N']:,}: {MAIN['T'] * np.exp(-MAIN['N'] * Ia):.0e}; ({g('dips_q')}) lowest band q = {bq['center']:g}: theory {bq['theory'][-1]:.0e} at N = {bq['Nth'][-1]:,}, "
      f"simulation {bq['sim'][-1]:.3f} at N = {bq['Ns'][-1]:,}")
    p(f"      from the FIRST step (pool of {PL['n_paths']:,} simulated episodes of {cfg_pool(A.CFG)} steps, trapped fraction {PL['q_nat']:.3f}; {PL['groups']} averages per N, drawn with replacement): share of "
      "averages that decrease at some step: " + ", ".join(f"N = {N_:,}: {v:.2f}" for N_, v in zip(PL["Ns"], PL["curves"][None])))
    with open(os.path.join(OUT, "fr_theory_cache", "population.pkl"), "rb") as fh: _, pop = pickle.load(fh)
    un = np.abs(pop["step"][~pop["trapped"]].astype(float)); Sd = np.load(os.path.join(A.OUT, "dip_sim.npz")); cl = Sd["classes"] / Sd["classes"].sum()
    p(f"  at step {cfg['POP_T0']:,}: untrapped episodes whose FR still changes at a step: {pc(np.mean(un > 1e-6))} (by more than 0.1 nat: {pc(np.mean(un > 0.1))}).  The dip simulation of "
      f"({g('dips_T')}, {g('horizon')}): both beliefs settled (ahead by more than {float(Sd['gap']):g} nats) in {pc(cl[0] + cl[1])} of the episodes, only the Joint one in {pc(cl[2])}, neither in "
      f"{pc(cl[3])}; its shortcuts against the exact update: at most {float(Sd['check_worst']):.1e} nats a step")


GXS_4x4 = (0.36, 0.53, 0.53)                     # LAYOUT_4x4: columns 3 and 4 stand alone (each its own quantity), so both have room for a y label
TITLES_4x4 = dict(TITLES, dips_T=r"Dips within $T$ steps", dips_q=r"Dips within 30 steps, by $q$")   # (no "Dips of averages" row title above them)

if __name__ == "__main__":
    if "numbers" in sys.argv[1:]: numbers(load())
    elif "4x4" in sys.argv[1:]: make(load(), LAYOUT_4x4, NAME + "_4x4", titles=TITLES_4x4, gaps=GXS_4x4, ylab_cols=(0, 2, 3))
    else: make(load())
