#!/usr/bin/env python3
"""
fr_theory.py -- one-file pipeline for the two theory figures of the Factorization Regret paper (fig_fr_theory: expected curve and
single episodes, 2 x 4 panels; fig_fr_averages: averages over N episodes, 3 + 2 panels) and for the environment-dependent values
behind the qualitative statements of the theory block (FR_theory_block_v12.tex, which quotes no numbers: `numbers` prints them).
11-11 night (John: f noisy -> more data or R = 5, 10, 20; g and h split into more panels; bring back good lost panels): f with
R = 5, 10, 20 and 10,000 interaction sets per point (the wiggles were the sampling of interaction sets, in theory and simulation
alike); Figure 1 g, h = the step law of a single trapped episode and the spread of FR_t across episodes (Proposition 2(a, c), panels
of the 11-09/11-10 versions that nothing tested since); the N-average panels moved to Figure 2 (one panel per third of q; dips by
horizon T in the whole population and by third of q at T = 30).
11-11 noon: a third figure (fig_fr_threshold: the threshold q* and dips of averaged curves, Corollary 2) and candidate panels are
in fr_theory_additions.py; every identity and inequality of the tex is checked on random tables by fr_theory_verify.py; the figure
key says SEM (the manuscript's word).
11-10 (John): no Monte Carlo dots in the top row; panels c (number of distractors) and i (dips at three temperatures) dropped;
pairs that repeat a variable are excluded (the env's allow_repeated_vars = False; they violate (A2)). Late 11-10: (a) square inset
at the bottom right; (d) q* dashed, no bars; (e) no inset; (f) the slope of every interaction set against its own E[Delta] at
R = 3 realizations (trap-free sets exist there); (g) one-step change of N-episode averages, N = 1, 10, 100 (violins); (h) thirds
of the trapped fraction; "Theory" / "Simulation" key.  The full-size figure is also written to figure_bin (PNG and PDF).
Titles: the title(...) calls in make_figure; sizes: FIGSIZE and FS.
11-11 late: caches steps_lam (step law at lambda 1, 3), thirds_more (the thirds' SAME sets from step 0: mean +- s.d., step law, accuracy
up to 10^9 steps) and synergy (population, I(r_g; r_c | o)) feed fr_theory_figure.py (the single theory figure, now drawn from there).

WHAT IT COMPUTES
  Row 1 (panels a-d)  Small likelihood tables with one binary observable: exact expectations (finite sums over the
                      sufficient statistic); their Monte Carlo is computed and tested (selftest) but not drawn.
                      Independent of the environment.
  Row 2 (panels e-h)  Cognitive Gridworld: closed-form one-step predictions computed from likelihood tables alone,
                      against simulated episodes.  Depends on the environment ONLY through make_env() (section 1).
  Figure 2            Cognitive Gridworld, averages over N episodes: quantiles of the exact law of their one-step change, and
                      the probability that they decrease within the next T steps, against groups of simulated episodes.

COMMANDS   (outputs go to the current directory, or to --out DIR)
  python fr_theory.py selftest    1-2 min   environment-independent regression tests: exact tables, every table number quoted
                                            in the tex, internal consistency of the simulator.  Must pass before anything else.
  python fr_theory.py envcheck    < 1 min   validates make_env() (contract, speed, exchangeable variables) and compares its one-step
                                            statistics with the reference.
  python fr_theory.py fixture     seconds   writes fr_theory_observer_fixture.npz (tables, observations, FR and beliefs computed
                                            here) for checking the project's own observers on identical inputs.
  python fr_theory.py quick       ~ 3 min   selftest + envcheck + the whole pipeline at reduced size (separate *_quick caches and
                                            outputs).  Smoke test.
  python fr_theory.py all         ~ 12 min  (one core, ~1 GB RAM, ~0.5 GB of caches) selftest + envcheck + full pipeline: data ->
                                            checks -> numbers -> figure.
  python fr_theory.py figure      seconds   re-render the figure from the caches.  This is the loop to use while polishing.
  python fr_theory.py check                 theory-versus-simulation checks, from the caches.
  python fr_theory.py numbers               environment-dependent values behind the qualitative statements of the tex (v12 quotes
                                            none): reference run (the values v11 quoted) versus this run.
  options:  --quick     use the reduced-size configuration and caches (with figure / check / numbers)
            --fresh     recompute all caches of this mode          --rederive  recompute only the derived quantities (~30 s)
            --out DIR

OUTPUTS
  fig_fr_theory.pdf, fig_fr_theory.png     Figure 1, drawn at its print size (9 x 3.94 in: set 9 in wide on a rotated page in the tex);
                                           --fmode accuracy: the variant with the accuracy gap in f (fig_fr_theory_faccuracy)
  fig_fr_averages.pdf, fig_fr_averages.png Figure 2, same size
  fr_theory_inspect/*.png                  300-dpi crops of the figure, for checking overlaps and small text
  fr_theory_numbers.json                   everything `numbers` prints (machine-readable)
  fr_theory_cache/                         row1 (exact tables + Monte Carlo), pools (one-step quantities), sims, derived.
                                           Caches are keyed by the configuration and by a fingerprint of make_env(), and are
                                           rebuilt automatically when either changes.

THE ONE CONVENTION OF THE FIGURE:   lines and shaded bands = analytical,   markers = simulation (mean +- 2 s.e.).
No simulated observation may enter anything drawn as a line; nothing analytical may be drawn as a marker.

VOCABULARY (as in the paper)   joint realization r = (r_g, r_c): one realization of every latent variable; interaction set Z;
episode; observable; likelihood temperature lambda.  At a joint realization r a wrong action a is a TRAP if its Naive model
fits the observations better than the true action's, kappa(r, a) < kappa(r, r_g) (never ruled out), and a DISTRACTOR if it
fits worse (eventually ruled out).  r is TRAPPED if it has a trap; k = number of trapped joint realizations, q = their prior
weight.  Delta(r) = kappa(r, r_g) - min_a kappa(r, a) is the asymptotic growth rate of Factorization Regret (FR) at r.

Tested with Python 3.13, numpy 2.5, scipy 1.18, matplotlib 3.11.  Needs matplotlib >= 3.6 and pillow.
"""
import argparse, hashlib, itertools, json, os, pickle, sys, time
import numpy as np
from scipy.special import expit, logit, logsumexp, gammaln, xlogy, softmax
from scipy.stats import norm

# ============================================================================================================
# 0. CONFIGURATION
# ============================================================================================================
CFG = dict(
    LAMS_FULL=[1.0, 2.0, 3.0],                                         # per-episode FR is stored at these (exactly three: panel e)
    LAMS_SIM=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],                 # simulated likelihood temperatures (panel g)
    LAMS_POOL=[round(x, 3) for x in np.arange(0.5, 4.001, 0.125)],     # likelihood temperatures of the one-step predictions
    LAM_MAIN=2.0,                                                      # panels g and h
    T=250,                                                             # steps per simulated episode
    SLOPE_WINDOW=(150, 250),                                           # the simulated slope is measured between these steps (panel f)
    T_EARLY=29,                                                        # second time point of the dip check (the paper's horizon is 30; not drawn since 11-10)
    N_EP=50, NZ_FULL=2000, NZ_LIGHT=500,                               # episodes per interaction set; interaction sets per lambda
    NZ_POOL=2000,                                                      # interaction sets behind every one-step prediction
    N_MC=100_000,                                                      # Monte Carlo episodes per exact table (row 1)
    POOL_BOTH_GOALS=True,                                              # see make_env(): use each interaction set with either variable as goal
    FRAC_BINS=[(0.0, 0.33), (0.33, 0.66), (0.66, 1.01)],               # panels g, h: thirds of the trapped fraction of an interaction set (John 11-10)
    G_TIMES=(3, 30, 250),                                              # panel g: steps t of the one-step changes FR_t - FR_(t-1) of single episodes
                                                                       # (11-11: averages of N episodes match their law by construction once single
                                                                       # episodes do, so g shows single episodes over time; N is panel h)
    R_SMALL=3, NZ_SMALL=300, NEP_SMALL=100, T_SMALL=1500,              # per-set check (11-10; not drawn since 11-11): 3 realizations per variable
    OPT_RS=(5, 10, 20), OPT_DOS=(1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30),  # panel f: realizations x observables (John 11-11: R = 5, 10, 20)
    OPT_NZ=10_000, OPT_NEP=100, OPT_T1=10_000, OPT_T2=20_000, OPT_TACC=(30, 10 ** 9),   # f: interaction sets per point (theory and simulation on the
                                                                       # same tables; 10,000: with 600 the between-set spread of E[Delta], s.d. .03-.07,
                                                                       # left visible wiggles in BOTH curves), episodes per set, slope window (late, as g, h and the
                                                                       # averages: before t ~ 1,500 averages of 10-100 episodes dip slightly more often
                                                                       # than the limiting law.  CORRECTED 11-11: the mean step equals E[Delta] of
                                                                       # the SAME sets within 0.2% from t ~ 200 at lambda = 2; the earlier "3% below
                                                                       # at t ~ 1,000" compared two independent samples of sets), steps of the
                                                                       # accuracy variant: the task's 30 and
                                                                       # one late enough for the asymptote q (with ONE observable the Joint observer
                                                                       # needs ~10^8 steps: at 10^4 it is still at .65 for R = 10; counts are drawn
                                                                       # binomially, so a late step costs nothing)
    F_MODE="rate",                                                     # panel f: "rate" (E[Delta]) or "accuracy" (Joint - Naive accuracy; --fmode)
    TH_T0=10_000, TH_G_EP=500, TH_H_EP=400, TH_TS=(1, 30),             # g, h, per third of the trapped fraction (the NZ_BAND sets of each third):
                                                                       # start step (asymptotic regime), episodes per set (g, h), horizons T of h
    G_NS=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000), G_GROUPS=10 ** 9,  # Fig. 2 a-c: N of the averages; independent groups per N (simulation): as many
                                                                       # as the episodes allow (400,000 at N = 1; 400 groups left the quantiles at
                                                                       # small N visibly noisy, all of them match the theory to ~.003 nats)
    MON_NS=(10, 20, 50, 100, 200, 500, 1000), MON_GROUPS=10 ** 9,      # Fig. 2 d, e: N (from 10: groups that are all untrapped, (1 - q)^N, count as
                                                                       # flat in the theory); groups per N for e: as many as the episodes allow
    POP_GROUPS=3000,                                                   # Fig. 2 d: groups per N (bounded: each keeps a running sum over max(POP_TS) steps)
    MON_TUPLES=3000, N_TH_GRID=30,                                     # Fig. 2 d, e (theory): sets of N joint realizations drawn from the prior per
                                                                       # N; size of the log grid of N (plus every simulated N)
    NZ_BAND=800, BAND_MAX_DRAWS=80_000,                                # panel h: interaction sets per band, found by drawing tables (build_bands); 800:
                                                                       # with 200 the extreme bands drifted from the law by sampling (mean of the
                                                                       # increments 1-1.6 s.e. off, which moves P(dip) most where the law is shallow)
    PAPER_HORIZON=30, ALPHA=0.05,                                      # for the sample-size rule N = 2 SNR^-2 ln(T / alpha)
    POP_NZ=2000, POP_EP=150, POP_T0=10_000, POP_TS=(1, 10, 100, 1000),  # Figure 2 d and Figure 1 g: interaction sets drawn WITHOUT selection (the
                                                                       # Gridworld's population), episodes per set, start step, horizons T
    WORKERS=int(os.environ.get("FR_WORKERS", 12)),                     # worker processes of the heavy builds (results do not depend on it)
    Q3_ACC_EP=100, ACC_TMAX=10 ** 9, ACC_PER_DECADE=4,                 # fr_theory_figure (11-11): Joint and Naive accuracy per third of q, episodes per
                                                                       # set and log grid of steps (counts drawn binomially: a late step costs nothing)
    SYN_EP=50, SYN_TMAX=10 ** 4, SYN_PER_DECADE=8,                     # fr_theory_figure: synergy I(r_g; r_c | o_1:t) vs FR in the population (POP_NZ sets)
)
QUICK = dict(NZ_FULL=150, NZ_LIGHT=60, NZ_POOL=200, N_MC=20_000, NZ_BAND=30, NZ_SMALL=40, OPT_NZ=20, OPT_NEP=30,
             TH_G_EP=60, TH_H_EP=60, G_GROUPS=100, MON_GROUPS=30, MON_TUPLES=150, POP_NZ=60, POP_EP=60, POP_GROUPS=30)   # overrides of `quick` / --quick


# ============================================================================================================
# 1. ENVIRONMENT ADAPTER  --  the only environment-specific code in this file
# ============================================================================================================
def reference_make_env(rng, lam, R=10, d_o=5, dE=30):
    """Re-implementation of the Cognitive Gridworld generative process (App. A.2 of the paper) for C = 2 latent variables,
    written from the paper's description, NOT from the project's code.  It produced the reference numbers at the bottom of
    this file.  Keep it unchanged: `envcheck` compares the real environment with it."""
    L = 2 * R + 1; n = np.arange(L); r = np.arange(R)
    Theta = lam * np.sin(2 * np.pi / L * ((((n[:, None] - r[None, :]) % L) + 1 - L / 2)))

    def v_of_z(z):
        m_ = (n - L * z) % L; d = np.minimum(m_, L - m_)
        return softmax(-d ** 2) @ Theta

    def gram_schmidt_rows(M):
        Q = np.zeros_like(M)
        for i in range(M.shape[0]):
            v = M[i].copy()
            for j in range(i):
                v -= np.dot(Q[j], v) * Q[j]
            Q[i] = v / np.linalg.norm(v)
        return Q

    k1, q1 = gram_schmidt_rows(rng.standard_normal((d_o, dE))), gram_schmidt_rows(rng.standard_normal((d_o, dE)))
    k2, q2 = gram_schmidt_rows(rng.standard_normal((d_o, dE))), gram_schmidt_rows(rng.standard_normal((d_o, dE)))
    ell = np.zeros((d_o, R, R))
    for i in range(d_o):
        ell[i] = expit(np.outer(v_of_z(k1[i] @ q2[i]), v_of_z(k2[i] @ q1[i])))
    return ell


def make_env(rng, lam, R=10, d_o=5):
    """ADAPTER.  Sample one interaction set Z at likelihood temperature `lam` and return its likelihood tables

        ell[i, r_g, r_c] = P(o^i = 1 | goal realization r_g, context realization r_c)        shape (d_o, R, R_c)

    Contract (checked by `envcheck`, except for the generative assumption, which cannot be checked from here):
      * a numpy array with values strictly inside (0, 1): compute it in double precision, because a float32 sigmoid saturates
        to exactly 0 or 1 at high lambda; axis 1 = realizations of the goal variable, axis 2 = joint realizations of all the
        other variables (C = 2: R_c = R; for C > 2 flatten them, R_c = R^(C-1), and set CFG["POOL_BOTH_GOALS"] = False);
      * ALL randomness comes from `rng` (a numpy Generator), so that caches are reproducible.  If the project's code needs a
        seed, draw it:  seed = int(rng.integers(2**31));
      * the random draws do not depend on `lam`: with the same rng state, two values of lam give the SAME interaction set at
        two likelihood temperatures.  This is what makes the predicted curves smooth in lambda (panel g).
    Generative assumption (the one the theory block makes): uniform priors over r_g and r_c; at every step of an episode each
    observable is an independent Bernoulli draw with the probability in this table, the same at every step.  The simulator
    below draws observations exactly so.  If the project's environment generates observations differently (probabilities that
    change within an episode or depend on actions, observables that are not binary), this file does not describe it: report.
    CFG["POOL_BOTH_GOALS"]: with C = 2 and exchangeable variables every interaction set is used twice in the predictions,
    with either variable as the goal (ell and its transpose).  Set it to False if the two variables are not exchangeable
    (`envcheck` tests this), and return the table with the goal on axis 1.

    Everything else in this file needs only these tables.

    THIS REPOSITORY (main/env, the project's own generator, float64 throughout): every call builds a fresh world of 500
    variables (Env_generators.default_state_embeddings: Gram-Schmidt K and Q), draws the pair of variables and the goal as
    in every test episode (Env_generators.default_context with test_set = True: two of the 50 test variables, with
    replacement, goal slot uniform), and computes the pair's tables (Bayes_likelihood.trigger_build_likelihood, inside
    Env_generators.default_likelihoods) at likelihood_temp = lam, with the paper's defaults R = 10, d_o = 5, d_E = 30.  The
    project's code draws from numpy's global generator: it is seeded from `rng` here and its state restored afterwards.
    The table comes back with the goal variable on axis 1 (the project's axes are the two variables in index order).
    TWO DISTINCT VARIABLES (11-10): by default the project's sampler draws the pair with replacement, so 2% of its pairs
    (1 in 50) repeat a variable.  Such a table is symmetric, ell[i, a, b] = ell[i, b, a], so the goal is not identifiable
    ((A2) fails): the Joint belief splits between r_g and r_c, and FR keeps growing linearly even at joint realizations
    where the Naive observer converges to the truth (measured at lambda = 2: 69 such sets, FR of their untrapped episodes
    5.8 nats at t = 30 and 50 at t = 250, against a peak of .45 at t = 64 and decay in 400 sets of distinct variables).
    The theory assumes (A2), so the env's option allow_repeated_vars = False (Env_generators.default_context, ported from
    coggrid's CogGridConfig; John 11-10: "if there is no downside we should take the exclusion functionality from there")
    is set here: every pair is two distinct variables.
    """
    import os, sys
    envs = make_env.__dict__.setdefault("envs", {}); env = envs.get((R, d_o))
    if env is None:                                       # the project's environment object, built once per process (and R)
        here = os.path.dirname(os.path.abspath(__file__)) if os.path.exists(os.path.abspath(__file__)) else os.getcwd()
        starts = [os.environ.get("COGNITIVE_GRIDWORLD_ROOT", ""), here, os.getcwd()]
        roots = [os.path.join(*([s] + [os.pardir] * k)) for s in starts if s for k in range(5)]
        root = next((os.path.abspath(r) for r in roots if os.path.isfile(os.path.join(r, "main", "CognitiveGridworld.py"))), None)
        if root is None: raise SystemExit("make_env: the CognitiveGridworld repository was not found above this file or the working directory; "
                                          "set COGNITIVE_GRIDWORLD_ROOT to the folder that holds main/CognitiveGridworld.py")
        os.environ.setdefault("MPLBACKEND", "Agg")
        if root not in sys.path: sys.path.insert(0, root)
        from main.CognitiveGridworld import CognitiveGridworld
        state = np.random.get_state()
        env = CognitiveGridworld(mode=None, state_num=500, ctx_num=2, realization_num=R, obs_num=max(d_o, 2), KQ_dim=30, likelihood_temp=lam,
                                 batch_num=2, step_num=2, show_plots=False)   # the constructor runs one tiny simulation; 2 x 2 keeps its squeezes valid
        np.random.set_state(state)
        env.batch_num = 1; env.allow_repeated_vars = False; env.obs_num = d_o; envs[(R, d_o)] = env; make_env.env = env   # distinct variables
    env.likelihood_temp = lam
    state = np.random.get_state(); np.random.seed(int(rng.integers(2 ** 31)))
    try:
        env.preprocess_env()                              # sinusoidal base at this lambda; test_set = True
        env.EC_gen_state_embeddings()                     # a fresh world: K, Q of every variable
        env.EC_gen_context()                              # the pair of two DISTINCT variables (ctx_inds, sorted) and the goal slot
        env.joint_likelihood, _ = env.trigger_build_likelihood()      # joint_likelihood[b, i, r_0, r_1] (what EC_gen_likelihoods builds;
                                                                        # its Naive marginal is not needed here and squeezes d_o = 1)
    finally:
        np.random.set_state(state)
    ell = np.asarray(env.joint_likelihood[0], dtype=np.float64).reshape(d_o, R, R)
    return ell if int(env.goal_ind[0]) == 0 else ell.transpose(0, 2, 1)


# ============================================================================================================
# 2. ONE-STEP QUANTITIES AND SIMULATION OF THE TWO OBSERVERS   (any environment; shapes are read from the tables)
# ============================================================================================================
def dkl_bern(p, q):
    return xlogy(p, p / q) + xlogy(1 - p, (1 - p) / (1 - q))


def one_step_table(ell):
    """Closed-form one-step quantities of every joint realization r = (r_g, r_c) of one interaction set:
        m[i, a]            Naive marginal likelihood P(o^i = 1 | r_g = a) (uniform average over the context)
        kap[r_g, r_c, a]   kappa(r, a) = KL( P(o | r) || Naive model of action a )
        Delta, ahat, trapped
        p, theta, const    the law of the asymptotic step  X = sum_i o^i theta_i + const,  o^i ~ Bernoulli(p_i)
                           (all zero at untrapped joint realizations, where X = 0);  v = Var(X | r)."""
    d_o, R, Rc = ell.shape
    m = ell.mean(axis=2)
    kap = dkl_bern(ell[:, :, :, None], m[:, None, None, :]).sum(axis=0)
    rg = np.broadcast_to(np.arange(R)[:, None], (R, Rc)); ahat = kap.argmin(axis=2)
    kg = np.take_along_axis(kap, rg[:, :, None], 2)[:, :, 0]
    Delta = kg - kap.min(axis=2); trapped = ahat != rg
    lm, l1m = logit(m), np.log1p(-m)
    theta = np.where(trapped[None], lm[:, ahat] - lm[:, rg], 0.0)
    const = np.where(trapped, (l1m[:, ahat] - l1m[:, rg]).sum(axis=0), 0.0)
    p = np.where(trapped[None], ell, 0.0)
    v = np.sum(p * (1 - p) * theta ** 2, axis=0)
    return dict(Delta=Delta, trapped=trapped, ahat=ahat, p=np.moveaxis(p, 0, -1), theta=np.moveaxis(theta, 0, -1), const=const, v=v, kap=kap, m=m)


def checked_table(ell, lam):
    """The adapter's output as a float64 array; stops with an explanation if it is not a table of probabilities inside (0, 1)."""
    ell = np.asarray(ell, dtype=float)
    if ell.ndim != 3 or not np.isfinite(ell).all() or ell.min() <= 0.0 or ell.max() >= 1.0:
        raise SystemExit(f"make_env(rng, {lam:g}) must return ell[i, r_g, r_c] with every probability strictly inside (0, 1); got shape {ell.shape}, min {np.nanmin(ell):.3g}, "
                         f"max {np.nanmax(ell):.3g}.  Probabilities of exactly 0 or 1 usually come from a float32 sigmoid: compute the tables in double precision.")
    return ell


def one_step_pool(lam, nZ, rng, both_goals=True, env=None):
    """One-step quantities pooled over nZ interaction sets (x 2 goal assignments if both_goals) x all joint realizations.
    `frac` = fraction of trapped joint realizations of the (interaction set, goal) an entry belongs to; `set` = its interaction set."""
    env = env or make_env
    out = {k: [] for k in ["Delta", "v", "trapped", "p", "theta", "const", "frac", "set"]}
    for z in range(nZ):
        ell = checked_table(env(rng, lam), lam); d_o = ell.shape[0]
        tabs = [ell] + ([ell.transpose(0, 2, 1)] if both_goals and ell.shape[1] == ell.shape[2] else [])
        for e in tabs:
            t = one_step_table(e); n = t["Delta"].size
            out["Delta"].append(t["Delta"].ravel()); out["v"].append(t["v"].ravel()); out["trapped"].append(t["trapped"].ravel())
            out["p"].append(t["p"].reshape(-1, d_o)); out["theta"].append(t["theta"].reshape(-1, d_o)); out["const"].append(t["const"].ravel())
            out["frac"].append(np.full(n, t["trapped"].mean())); out["set"].append(np.full(n, z, np.int32))
    return {k: np.concatenate(v) for k, v in out.items()}


def pool_summary(p):
    """Population one-step moments.  Standard errors are computed across interaction sets."""
    D, V, tr = p["Delta"], p["v"], p["trapped"]; cnt = np.bincount(p["set"]); zmean = np.bincount(p["set"], weights=D) / cnt
    return dict(EDelta=D.mean(), EDelta_se=zmean.std() / np.sqrt(len(zmean)), VarDelta=D.var(), Ev=V.mean(), sigma=np.sqrt(V.mean() + D.var()),
                q=tr.mean(), Dtrap=D[tr].mean() if tr.any() else 0.0, n=len(D), n_sets=len(zmean))


def simulate(lam, nZ, n_ep, T, rng, store_fr=True, keep_last=False, env=None):
    """Simulate n_ep episodes (uniformly drawn joint realization, T steps) in each of nZ interaction sets and run both observers.

    Per episode (arrays of length nZ * n_ep, ordered by interaction set):
        FR[n, t-1]    Factorization Regret KL(B_t || Btilde_t) after t steps, in nats (float32; only if store_fr)
        trapped, Delta, frac   of the episode's joint realization / interaction set
    Per interaction set (arrays [nZ, T], means over its n_ep episodes):
        FR_mean;  N = -ln Btilde_t(ahat(r));  J = -ln B_t(r_g)       (identity: E[FR_t] = t E[Delta] + E[N_t] - E[J_t])
        FR_trap_sum, FR_untrap_sum (sums over its trapped / untrapped episodes);  n_trap [nZ];  Delta_mean [nZ]
    keep_last: also return the tables, joint realizations and observations of the last interaction set (for tests)."""
    env = env or make_env; n = nZ * n_ep
    out = dict(trapped=np.zeros(n, bool), Delta=np.zeros(n), frac=np.zeros(n), n_trap=np.zeros(nZ), Delta_mean=np.zeros(nZ), n_ep=n_ep)
    for k in ["FR_mean", "N", "J", "FR_trap_sum", "FR_untrap_sum"]: out[k] = np.zeros((nZ, T))
    if store_fr: out["FR"] = np.zeros((n, T), np.float32)
    for zi in range(nZ):
        ell = checked_table(env(rng, lam), lam); d_o, R, Rc = ell.shape; tab = one_step_table(ell); m = tab["m"]
        l1, l0, lm1, lm0 = np.log(ell), np.log1p(-ell), np.log(m), np.log1p(-m)
        rs = rng.integers(0, R, size=(n_ep, 2)) if R == Rc else np.stack([rng.integers(0, R, n_ep), rng.integers(0, Rc, n_ep)], axis=1)
        rg, rc = rs[:, 0], rs[:, 1]; idx = np.arange(n_ep); sl = slice(zi * n_ep, (zi + 1) * n_ep)
        ah = tab["ahat"][rg, rc]; tr = tab["trapped"][rg, rc]
        out["trapped"][sl] = tr; out["Delta"][sl] = tab["Delta"][rg, rc]; out["frac"][sl] = tab["trapped"].mean()
        out["n_trap"][zi] = tr.sum(); out["Delta_mean"][zi] = out["Delta"][sl].mean()
        p = ell[:, rg, rc].T; o = (rng.random((T, n_ep, d_o)) < p[None]).astype(float)
        Lj = np.zeros((n_ep, R, Rc)); Ln = np.zeros((n_ep, R))
        for t in range(T):
            ot = o[t]
            Lj += np.einsum('ei,ijk->ejk', ot, l1) + np.einsum('ei,ijk->ejk', 1 - ot, l0); Ln += ot @ lm1 + (1 - ot) @ lm0
            Bj = np.exp(Lj - Lj.max(axis=(1, 2), keepdims=True)); Bj /= Bj.sum(axis=(1, 2), keepdims=True)
            Bg = Bj.sum(axis=2); logBn = Ln - logsumexp(Ln, axis=1, keepdims=True)           # Joint goal marginal; Naive log belief
            FR = np.sum(xlogy(Bg, Bg) - Bg * logBn, axis=1)
            if store_fr: out["FR"][sl, t] = FR
            out["FR_mean"][zi, t] = FR.mean(); out["N"][zi, t] = (-logBn[idx, ah]).mean(); out["J"][zi, t] = (-np.log(Bg[idx, rg])).mean()
            out["FR_trap_sum"][zi, t] = FR[tr].sum(); out["FR_untrap_sum"][zi, t] = FR[~tr].sum()
    if keep_last: out["last"] = dict(ell=ell, rs=rs, obs=o)
    return out


def fr_episode(ell, obs):
    """Reference implementation of the two observers for ONE episode, written independently of simulate() (log space).
    ell[i, r_g, r_c] as returned by make_env; obs[t, i] in {0, 1}.  Returns FR[t] (nats), joint[t, a] = B_t(a), naive[t, a] = Btilde_t(a)
    after t + 1 observations.  Use it to check the project's own observers on identical observations."""
    ell = np.asarray(ell, float); obs = np.asarray(obs, float); m = ell.mean(axis=2)
    Lj = np.cumsum(np.einsum('ti,ijk->tjk', obs, np.log(ell)) + np.einsum('ti,ijk->tjk', 1 - obs, np.log1p(-ell)), axis=0)
    Ln = np.cumsum(obs @ np.log(m) + (1 - obs) @ np.log1p(-m), axis=0)
    logBg = logsumexp(Lj, axis=2) - logsumexp(Lj, axis=(1, 2))[:, None]; logBn = Ln - logsumexp(Ln, axis=1, keepdims=True)
    return dict(FR=np.sum(np.exp(logBg) * (logBg - logBn), axis=1), joint=np.exp(logBg), naive=np.exp(logBn))


# ============================================================================================================
# 3. EXACT ASYMPTOTIC LAW OF A DIP OF THE N-EPISODE AVERAGE   (from likelihood tables alone; no sampling)
# ============================================================================================================
def step_atoms(pool, mask=None):
    """Atoms of the law of the step X over the pooled joint realizations selected by `mask`: every trapped joint realization
    contributes 2^{d_o} atoms, every untrapped one an atom at 0.  Returns x, w (trapped part; the weights sum to q) and q."""
    sel = np.ones(len(pool["Delta"]), bool) if mask is None else mask
    tr = sel & pool["trapped"]; M = sel.sum()
    p, th, co = pool["p"][tr].astype(float), pool["theta"][tr].astype(float), pool["const"][tr]; d_o = p.shape[1]
    if not tr.any(): raise SystemExit("No trapped joint realization in this population: panels h-j need some. Report this instead of working around it.")
    if d_o > 12: raise SystemExit(f"step_atoms enumerates the 2^d_o outcomes of one step; d_o = {d_o} is too large for that.")
    O = np.array(list(itertools.product([0, 1], repeat=d_o)), float)
    x = th @ O.T + co[:, None]
    logw = np.log(p) @ O.T + np.log1p(-p) @ (1 - O).T
    return x.ravel(), (np.exp(logw) / M).ravel(), tr.sum() / M


def step_moments(x, w):
    mu = np.sum(w * x); return mu, np.sqrt(np.sum(w * x ** 2) - mu ** 2)


def dip_law(x, w, q, Ns, n=2 ** 20):
    """Lower and upper asymptotic dip probabilities P(S_N < 0) and P(S_N <= 0) for each N in Ns, S_N = sum of N independent
    draws of X.  Here they differ by (1 - q)^N, the probability that all N episodes are untrapped.  (Exactly, the difference is
    P(S_N = 0) >= (1 - q)^N: steps of two episodes of the SAME interaction set whose optimal actions and traps are exchanged can
    cancel; measured per set at lambda = 2, N = 2: 7e-5 to 5e-4 (math check of 11-11).  Groups that mix interaction sets, as in
    every panel, have no such pairs to speak of, and the figures draw the midpoint of the two.)  N-fold convolution by FFT."""
    mu, sd = step_moments(x, w); Nmax = max(Ns)
    ring = 2.0 * (Nmax * abs(mu) + 14.0 * np.sqrt(Nmax) * sd + np.abs(x).max()); h = ring / n
    j = x / h; j0 = np.floor(j).astype(np.int64); f = j - j0                                 # split each atom between two grid points
    pmf = np.bincount(j0 % n, weights=w * (1 - f), minlength=n) + np.bincount((j0 + 1) % n, weights=w * f, minlength=n)
    pmf[0] += 1.0 - q; F = np.fft.rfft(pmf); lo, hi = [], []
    for N in Ns:
        pN = np.fft.irfft(F ** int(N), n)
        first = int(np.ceil((N * mu - ring / 2) / h)); assert first < 0, "zero must lie inside the window"
        mid = pN[np.arange(first, 0) % n].sum() + 0.5 * pN[0]; atom = (1.0 - q) ** int(N)
        lo.append(mid - 0.5 * atom); hi.append(mid + 0.5 * atom)
    return np.clip(np.array(lo), 0, 1), np.clip(np.array(hi), 0, 1)


def gaussian_limit(x, w, Ns):
    mu, sd = step_moments(x, w); return norm.cdf(-np.sqrt(np.asarray(Ns, float)) * mu / sd)


def boot_dip(d, N, rng, B=20000, chunk=4_000_000):
    """Simulated probability that the mean of N episodes' increments is negative (bootstrap over the simulated episodes)."""
    hits = 0; done = 0
    while done < B:
        b = min(B - done, max(1, chunk // N)); hits += np.count_nonzero(d[rng.integers(0, len(d), size=(b, N))].sum(axis=1) < 0); done += b
    return hits / B


# ============================================================================================================
# 4. EXACTLY SOLVABLE TABLES   (two latent variables, one binary observable, uniform priors)
# ============================================================================================================
def additive_table(R, alpha, beta, Rc=3, perturb=None, gamma=0.0):
    """ell(r_g, r_c) = sigmoid(alpha (r_g - (R+1)/2) + beta w(r_c) / 2), w equally spaced in [-1, 1].
    perturb = [(r_g, r_c, sign), ...] adds sign * gamma to the logit of those entries (turning them into trapped joint realizations)."""
    u = np.arange(1, R + 1) - (R + 1) / 2
    w = np.linspace(-1, 1, Rc) if Rc > 1 else np.zeros(1)
    lg = alpha * u[:, None] + beta * w[None, :] / 2
    for (rg, rc, sign) in (perturb or []):
        lg[rg, rc] += sign * gamma
    return expit(lg)


def equispaced_table(R, alpha, beta, Rc=3):
    """R goal realizations that are equally distinguishable: baseline probabilities equally spaced in 2 arcsin(sqrt(p)), with
    the spacing of additive_table(3, alpha, beta), which it reproduces for R = 3.  Adding goal realizations then adds
    distractors without weakening the weakest one."""
    dphi = 2 * np.arcsin(np.sqrt(expit(alpha))) - np.pi / 2
    phi = np.pi / 2 + dphi * (np.arange(R) - (R - 1) / 2); base = logit(np.sin(phi / 2) ** 2)
    return expit(base[:, None] + beta * np.linspace(-1, 1, Rc)[None, :] / 2)


def table_stats(ell):
    """kappa[rg, rc, a], Delta[rg, rc], ahat, trapped mask, k = number of trapped joint realizations,
    delta = smallest drift at which a distractor is ruled out."""
    ell = np.asarray(ell, float); R, Rc = ell.shape; m = ell.mean(1)
    kap = dkl_bern(ell[:, :, None], m[None, None, :])
    kg = np.take_along_axis(kap, np.broadcast_to(np.arange(R)[:, None, None], (R, Rc, 1)), 2)[:, :, 0]
    ahat = kap.argmin(axis=2); trapped = ahat != np.arange(R)[:, None]
    gaps = kap - kg[:, :, None]; delta = gaps[gaps > 0].min() if (gaps > 0).any() else np.inf
    return dict(kap=kap, Delta=kg - kap.min(axis=2), ahat=ahat, trapped=trapped, k=int(trapped.sum()), delta=delta, m=m)


def _beliefs(ell, m, t):
    """For t observations and every count ks of ones: log Joint goal marginal logBg[ks, a], Naive log belief logBn[ks, a],
    Naive log-likelihoods logLn[ks, a], the probability P[ks, rg, rc] of each count at each joint realization, and logL."""
    ks = np.arange(t + 1)
    logL = ks[:, None, None] * np.log(ell) + (t - ks)[:, None, None] * np.log1p(-ell)
    logBg = logsumexp(logL, axis=2) - logsumexp(logL, axis=(1, 2))[:, None]
    logLn = ks[:, None] * np.log(m) + (t - ks)[:, None] * np.log1p(-m); logBn = logLn - logsumexp(logLn, axis=1, keepdims=True)
    logC = gammaln(t + 1) - gammaln(ks + 1) - gammaln(t - ks + 1)
    return logBg, logBn, logLn, np.exp(logC[:, None, None] + logL), logL


def _fr_by_count(logBg, logBn):
    return np.sum(np.exp(logBg) * (logBg - logBn), axis=1)


def exact_anatomy(ell, T):
    """Exact expectations for t = 0..T (uniform prior over joint realizations):
        FR  = E[FR_t]                     T = trap term E[T_t]          D = distraction term E[D_t]
        Lam = Joint log loss Lambda_t     N = Naive log loss on the action it converges to, N_t
        Syn = I(r_g; r_c | o_{1:t})
    Identities (Proposition 1):  FR = T + D - Lam  =  t E[Delta] + N - Lam."""
    ell = np.asarray(ell, float); R, Rc = ell.shape; st = table_stats(ell); kap, m, ahat = st["kap"], st["m"], st["ahat"]
    kg = np.take_along_axis(kap, np.broadcast_to(np.arange(R)[:, None, None], (R, Rc, 1)), 2)
    is_trap = kap < kg
    out = {k: np.zeros(T + 1) for k in ["FR", "T", "D", "Lam", "N", "Syn"]}
    for t in range(T + 1):
        logBg, logBn, logLn, P, logL = _beliefs(ell, m, t); W = P / (R * Rc)
        out["FR"][t] = (P.mean(axis=(1, 2)) * _fr_by_count(logBg, logBn)).sum()
        out["Lam"][t] = np.sum(W * (-logBg[:, :, None]))
        out["Syn"][t] = out["Lam"][t] - np.sum(W * (logsumexp(logL, axis=1, keepdims=True) - logL))      # minus the Oracle log loss
        for rg in range(R):
            S = logLn - logLn[:, [rg]]                                                                # Naive log-odds against the truth
            for rc in range(Rc):
                wgt = W[:, rg, rc]; mask = is_trap[rg, rc]
                Tt = np.logaddexp(0.0, logsumexp(np.where(mask[None, :], S, -np.inf), axis=1)) if mask.any() else np.zeros(t + 1)
                out["T"][t] += wgt @ Tt; out["D"][t] += wgt @ (-logBn[:, rg] - Tt); out["N"][t] += wgt @ (-logBn[:, ahat[rg, rc]])
    return out, st


def exact_efr(ell, T):
    """Exact E[FR_t] only, t = 0..T (faster than exact_anatomy)."""
    ell = np.asarray(ell, float); m = ell.mean(1); out = np.zeros(T + 1)
    for t in range(1, T + 1):
        logBg, logBn, _, P, _ = _beliefs(ell, m, t)
        out[t] = (P.mean(axis=(1, 2)) * _fr_by_count(logBg, logBn)).sum()
    return out


def simulate_table(ell, T, n_ep, rng):
    """Monte Carlo: mean and standard error of FR_t over n_ep episodes (uniformly drawn joint realization), t = 0..T."""
    ell = np.asarray(ell, float); R, Rc = ell.shape; m = ell.mean(1)
    idx = rng.integers(0, R * Rc, size=n_ep); p = ell.ravel()[idx]
    k = np.zeros(n_ep, int); mean = np.zeros(T + 1); se = np.zeros(T + 1)
    for t in range(1, T + 1):
        k += rng.random(n_ep) < p
        logBg, logBn, _, _, _ = _beliefs(ell, m, t); f = _fr_by_count(logBg, logBn)[k]
        mean[t] = f.mean(); se[t] = f.std() / np.sqrt(n_ep)
    return mean, se


def mixture_threshold(E0, E1):
    """Corollary 1 for a population made of a fraction w of curve E1 (increasing) and 1 - w of curve E0: the mixture is
    non-decreasing at every step t >= 1 exactly when w >= w* = max_t w*_t.  Returns w*, argmax step."""
    d0, d1 = np.diff(E0)[1:], np.diff(E1)[1:]; assert (d1 > 0).all(), "the trapped component must be increasing"
    wt = np.maximum(-d0, 0) / (np.maximum(-d0, 0) + d1); return wt.max(), int(wt.argmax()) + 1


def decreasing_steps(m):
    """First and last step of the decrease of a curve m[t] (None if it never decreases after t = 1): m[t0] > ... > m[t1]."""
    dec = np.flatnonzero(np.diff(m)[1:] < -1e-13) + 1
    return (int(dec.min()), int(dec.max()) + 1) if len(dec) else None


# ============================================================================================================
# 5. DATA: BUILD AND CACHE
# ============================================================================================================
ROW1_VERSION, POOL_VERSION, SIM_VERSION, DERIVED_VERSION, BAND_VERSION, SMALL_VERSION, OPT_VERSION, MON_VERSION = 3, 5, 5, 21, 2, 1, 2, 3
POP_VERSION = 1   # DERIVED 18: spread theory from the simulated episodes' own joint realizations; 19: mean +- s.d. (fr_theory_figure); 20: step law per lambda; 21: by third of q, synergy
T_ANATOMY, T_DISTRACTORS, T_MIXTURE = 300, 400, 3000


class Store:
    """Caches under <out>/fr_theory_cache, keyed by a signature (configuration + code version + environment fingerprint)."""
    def __init__(self, out_dir, quick):
        self.out = out_dir; self.dir = os.path.join(out_dir, "fr_theory_cache"); self.tag = "_quick" if quick else ""
        os.makedirs(self.dir, exist_ok=True)

    def path(self, name): return os.path.join(self.dir, f"{name}{self.tag}.pkl")

    def get(self, name, signature, builder, fresh=False):
        path = self.path(name)
        if os.path.exists(path) and not fresh:
            with open(path, "rb") as fh: sig, data = pickle.load(fh)
            if sig == signature: return data
            print(f"[cache] {os.path.basename(path)} is stale (configuration, code or environment changed): rebuilding", flush=True)
        t0 = time.time(); data = builder()
        with open(path, "wb") as fh: pickle.dump((signature, data), fh, protocol=4)
        print(f"[cache] built {os.path.basename(path)} in {time.time() - t0:.0f} s", flush=True)
        return data


def env_fingerprint(cfg):
    """Changes whenever make_env() returns different tables for the same random state."""
    ell = np.asarray(make_env(np.random.default_rng(12345), cfg["LAM_MAIN"]), float)
    return hashlib.sha1(np.round(ell, 10).tobytes() + str(ell.shape).encode()).hexdigest()[:16]


def build_row1(cfg):
    """Exact tables of panels a-e and their Monte Carlo."""
    rng = np.random.default_rng(7); d = {}; n_ep = cfg["N_MC"]
    base = dict(R=3, alpha=1.2, beta=0.72)
    tabs = {"a": additive_table(**base), "b": additive_table(**base, perturb=[(0, 2, +1)], gamma=0.6)}
    for key, ell in tabs.items():                      # a, b: anatomy to T_ANATOMY; e: E[FR_t] to T_MIXTURE
        an, st = exact_anatomy(ell, T_ANATOMY)
        d[key] = dict(ell=ell, an=an, st=st, efr=exact_efr(ell, T_MIXTURE), mc=simulate_table(ell, T_MIXTURE, n_ep, rng))
    d["c"] = []                                        # number of distractors, strength of the weakest one fixed
    for R in [2, 3, 4, 5]:
        ell = equispaced_table(R, 1.2, 0.72); st = table_stats(ell); assert st["k"] == 0
        d["c"].append(dict(R=R, st=st, efr=exact_efr(ell, T_DISTRACTORS), mc=simulate_table(ell, T_DISTRACTORS, n_ep, rng)))
    d["d"] = []                                        # strength of the distractors
    for alpha in [0.8, 1.2, 1.8]:
        ell = additive_table(3, alpha, 0.6 * alpha); st = table_stats(ell); assert st["k"] == 0
        d["d"].append(dict(alpha=alpha, st=st, efr=exact_efr(ell, T_DISTRACTORS), mc=simulate_table(ell, T_DISTRACTORS, n_ep, rng)))
    return d


def build_pools(cfg):
    """One-step quantities of NZ_POOL sampled interaction sets at every lambda of LAMS_POOL.  The generator is re-seeded for each
    lambda, so the SAME interaction sets are evaluated at every likelihood temperature (see the contract of make_env)."""
    out = dict(pool={}, summary={})
    for lam in cfg["LAMS_POOL"]:
        p = one_step_pool(lam, cfg["NZ_POOL"], np.random.default_rng(2026), cfg["POOL_BOTH_GOALS"]); out["summary"][lam] = pool_summary(p)
        if lam in cfg["LAMS_FULL"]: out["pool"][lam] = {k: (v.astype(np.float32) if k in ("p", "theta") else v) for k, v in p.items()}
    return out


def build_sims(cfg):
    """Simulated episodes: interaction sets x N_EP episodes x T steps at each simulated lambda (independent of the pools)."""
    sims = {}; t0 = time.time()
    for i, lam in enumerate(cfg["LAMS_SIM"]):
        full = lam in cfg["LAMS_FULL"]
        sims[lam] = simulate(lam, cfg["NZ_FULL"] if full else cfg["NZ_LIGHT"], cfg["N_EP"], cfg["T"], np.random.default_rng(3000 + i), store_fr=full)
        print(f"   simulated lambda = {lam} ({time.time() - t0:.0f} s)", flush=True)
    return sims


def build_small(cfg):
    """Panel f: NZ_SMALL interaction sets of the same generator with R_SMALL realizations per variable (trap-free sets exist
    there), NEP_SMALL episodes of T_SMALL steps each: per set the trapped count k, the mean Delta of its simulated joint
    realizations (the expected slope of its simulated mean curve) and its simulated slope over the second half."""
    lam, R = cfg["LAM_MAIN"], cfg["R_SMALL"]; rng = np.random.default_rng(5151)
    tabs = [checked_table(make_env(rng, lam, R=R), lam) for _ in range(cfg["NZ_SMALL"])]
    it = iter(tabs); T = cfg["T_SMALL"]; h = T // 2
    S = simulate(lam, len(tabs), cfg["NEP_SMALL"], T, np.random.default_rng(5252), store_fr=False, env=lambda r_, l_: next(it))
    k = np.array([int(one_step_table(e)["trapped"].sum()) for e in tabs])
    return dict(k=k, pred=S["Delta_mean"].copy(), slope=(S["FR_mean"][:, -1] - S["FR_mean"][:, h - 1]) / (T - h), frT=S["FR_mean"][:, -1].copy())


def fr_counts(n, t, L1, L0, M1, M0):
    """FR_t of episodes from their counts of ones n (n_ep, d_o) after t steps: the exact Joint and Naive observers need nothing
    else.  L1, L0 = log ell, log(1 - ell) (d_o, R, Rc); M1, M0 = log m, log(1 - m) (d_o, R) of the Naive marginals m."""
    LJ = np.einsum("ei,ijk->ejk", n, L1) + np.einsum("ei,ijk->ejk", t - n, L0)
    lB = logsumexp(LJ, axis=2); lB -= logsumexp(lB, axis=1, keepdims=True)
    LN = n @ M1 + (t - n) @ M0; LN -= logsumexp(LN, axis=1, keepdims=True)
    return np.sum(np.exp(lB) * (lB - LN), axis=1)


def table_logs(ell):
    m = ell.mean(axis=2); return np.log(ell), np.log1p(-ell), np.log(m), np.log1p(-m)


def goal_answers(n, t, L1, L0, M1, M0):
    """The goal answers (argmax) of the exact Joint and Naive observers after t steps with counts n (n_ep, d_o)."""
    LJ = np.einsum("ei,ijk->ejk", n, L1) + np.einsum("ei,ijk->ejk", t - n, L0)
    return logsumexp(LJ, axis=2).argmax(1), (n @ M1 + (t - n) @ M0).argmax(1)


def run_parallel(fn, tasks, workers, label):
    """fn over tasks in `workers` forked processes (1: in this process), results in task order.  Every task carries its own seeds,
    so the results do not depend on the number of workers."""
    if workers <= 1 or len(tasks) <= 1: return [fn(t) for t in tasks]
    import multiprocessing as mp
    out = [None] * len(tasks); t0 = time.time(); step = max(1, len(tasks) // 8)
    with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
        for k, (i, r) in enumerate(pool.imap_unordered(_indexed, [(fn, i, t) for i, t in enumerate(tasks)])):
            out[i] = r
            if (k + 1) % step == 0 or k + 1 == len(tasks): print(f"   {label}: {k + 1}/{len(tasks)} ({time.time() - t0:.0f} s)", flush=True)
    return out


def _indexed(a):
    fn, i, t = a; return i, fn(t)


def _opt_point(task):
    """Panel f, ONE point (R, d_o) of build_optimum (module level, so that worker processes can run it)."""
    cfg, R, d_o = task; lam = cfg["LAM_MAIN"]; T1, T2, TA = cfg["OPT_T1"], cfg["OPT_T2"], cfg["OPT_TACC"]
    rng, erng = np.random.default_rng(70_000 + 100 * R + d_o), np.random.default_rng(90_000 + 100 * R + d_o)
    rec = dict(ed=[], q=[], slope=[], accJ=[], accN=[])
    for _ in range(cfg["OPT_NZ"]):
        ell = checked_table(make_env(rng, lam, R=R, d_o=d_o), lam); tab = one_step_table(ell); lg = table_logs(ell); n_ep = cfg["OPT_NEP"]
        rec["ed"].append(tab["Delta"].mean()); rec["q"].append(tab["trapped"].mean())
        rg, rc = erng.integers(0, R, n_ep), erng.integers(0, R, n_ep); pr = ell[:, rg, rc].T
        n1 = erng.binomial(T1, pr).astype(float); n2 = n1 + erng.binomial(T2 - T1, pr)
        rec["slope"].append((fr_counts(n2, T2, *lg).mean() - fr_counts(n1, T1, *lg).mean()) / (T2 - T1))
        n = np.zeros_like(pr); t_prev = 0; aJ, aN = [], []
        for t in TA:
            n = n + erng.binomial(t - t_prev, pr); t_prev = t; bj, bn = goal_answers(n, t, *lg); aJ.append(np.mean(bj == rg)); aN.append(np.mean(bn == rg))
        rec["accJ"].append(aJ); rec["accN"].append(aN)
    return {k: np.array(v) for k, v in rec.items()}


def build_optimum(cfg):
    """Panel f, per number of realizations R and observables d_o at LAM_MAIN: OPT_NZ interaction sets per point (tables as returned,
    random goal slot), the SAME tables for theory and simulation.  THEORY (tables alone): E[Delta] and the trapped fraction q of each
    set (q is also the asymptotic Joint - Naive accuracy gap: the Joint observer becomes exact, the Naive one is wrong exactly at
    trapped r).  SIMULATION (OPT_NEP episodes per set, counts drawn binomially, exact observers): the slope of the mean FR between
    OPT_T1 and OPT_T2, and the accuracy of both observers at the steps OPT_TACC.  One point per worker process (WORKERS)."""
    tasks = [(cfg, R, d) for R in sorted(cfg["OPT_RS"], reverse=True) for d in sorted(cfg["OPT_DOS"], reverse=True)]   # the slow points first
    return {(R, d): r for (_, R, d), r in zip(tasks, run_parallel(_opt_point, tasks, cfg["WORKERS"], "panel f"))}


def third_tables(cfg):
    """The NZ_BAND interaction sets of each third of the trapped fraction (FRAC_BINS) at LAM_MAIN, found by drawing tables (goal as
    returned): the same seed and order for build_thirds and build_thirds_more, so both use the SAME sets.  Returns them and the draws."""
    lam, bins, need = cfg["LAM_MAIN"], [tuple(b) for b in cfg["FRAC_BINS"]], cfg["NZ_BAND"]
    rng = np.random.default_rng(4141); keep = [[] for _ in bins]; draws = 0
    while any(len(k) < need for k in keep):
        ell = checked_table(make_env(rng, lam), lam); draws += 1; f = one_step_table(ell)["trapped"].mean()
        for k, (lo, hi) in zip(keep, bins):
            if lo <= f < hi and len(k) < need: k.append(ell)
        if draws >= cfg["BAND_MAX_DRAWS"]: raise SystemExit(f"third_tables: after {draws} tables the thirds hold {[len(k) for k in keep]} of {need} sets")
    return keep, draws


def build_thirds(cfg):
    """Panels g, h: NZ_BAND interaction sets in each third of the trapped fraction (CFG FRAC_BINS) at LAM_MAIN, found by drawing tables
    (goal as returned).  THEORY (tables alone): the one-step pool of each third.  SIMULATION (counts drawn binomially, exact observers,
    from step TH_T0 on, every episode at its own joint realization): g, the change at the next step of TH_G_EP episodes per set; h, the
    changes over the next max(TH_TS) steps of TH_H_EP episodes per set."""
    lam, bins = cfg["LAM_MAIN"], [tuple(b) for b in cfg["FRAC_BINS"]]; keep, draws = third_tables(cfg)
    erng = np.random.default_rng(4242); t0, Tm = cfg["TH_T0"], max(cfg["TH_TS"]); out = dict(bins=bins, draws=draws, thirds=[]); tq = time.time()
    for tabs in keep:
        it = iter(tabs); pool = one_step_pool(lam, len(tabs), None, both_goals=False, env=lambda r_, l_: next(it)); g, h = [], []
        for ell in tabs:
            lg = table_logs(ell); d_o, R, Rc = ell.shape
            for n_ep, steps, dst in ((cfg["TH_G_EP"], 1, g), (cfg["TH_H_EP"], Tm, h)):
                rg, rc = erng.integers(0, R, n_ep), erng.integers(0, Rc, n_ep); pr = ell[:, rg, rc].T
                n = erng.binomial(t0, pr).astype(float); fr = [fr_counts(n, t0, *lg)]
                for st in range(steps):
                    n = n + (erng.random(pr.shape) < pr); fr.append(fr_counts(n, t0 + st + 1, *lg))
                dst.append(np.diff(np.array(fr), axis=0).T.astype(np.float32))
        out["thirds"].append(dict(pool={k: (v.astype(np.float32) if k in ("p", "theta") else v) for k, v in pool.items()},
                                  g=np.concatenate(g)[:, 0], h=np.concatenate(h)))
        print(f"   panels g, h: a third simulated ({time.time() - tq:.0f} s)", flush=True)
    return out


def outcome_atoms(P, TH, C):
    """The 2^d_o outcomes of the step X of each trapped joint realization (rows): values x and probabilities w (k, 2^d_o)."""
    d_o = P.shape[1]; O = np.array(list(itertools.product([0, 1], repeat=d_o)), float)
    return TH @ O.T + C[:, None], np.exp(np.log(P) @ O.T + np.log1p(-P) @ (1 - O).T)


def dip_exact(P, TH, C, n=2 ** 14):
    """P(sum of the steps of these trapped joint realizations < 0), by FFT convolution of their exact laws (a zero sum counts half)."""
    x, w = outcome_atoms(P, TH, C); ring = 2.0 * np.abs(x).max(axis=1).sum() + 1e-9; h = ring / n; F = np.ones(n // 2 + 1, complex)
    for xk, wk in zip(x, w):
        j = xk / h; j0 = np.floor(j).astype(np.int64); f = j - j0
        F *= np.fft.rfft(np.bincount(j0 % n, weights=wk * (1 - f), minlength=n) + np.bincount((j0 + 1) % n, weights=wk * f, minlength=n))
    pN = np.fft.irfft(F, n); return float(np.clip(pN[n // 2:].sum() + 0.5 * pN[0], 0, 1))


def dip_saddle(P, TH, C):
    """The same probability by the Lugannani-Rice saddlepoint formula, from the exact cumulant generating function
    K(s) = sum_k [s C_k + sum_i log(1 - p_ki + p_ki exp(s theta_ki))] (accurate for many terms; used above 25)."""
    if (C + np.minimum(TH, 0).sum(axis=1)).sum() >= 0: return 0.0                  # the sum cannot be negative
    def K(s): return s * C.sum() + np.log1p(P * np.expm1(s * TH)).sum()
    def K1(s): e = np.exp(s * TH); return C.sum() + (P * TH * e / (1 + P * (e - 1))).sum()
    def K2(s): e = np.exp(s * TH); d = 1 + P * (e - 1); return (P * (1 - P) * TH ** 2 * e / d ** 2).sum()
    lo, hi = -1.0, 0.0
    while K1(lo) > 0: lo *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi); (lo, hi) = (lo, mid) if K1(mid) > 0 else (mid, hi)
    sh = 0.5 * (lo + hi); wv = -np.sqrt(max(-2.0 * K(sh), 0.0)); u = sh * np.sqrt(K2(sh))
    return float(np.clip(norm.cdf(wv) + norm.pdf(wv) * (1.0 / wv - 1.0 / u), 0, 1)) if wv < -1e-8 else 0.5


def _mono_one(task):
    """monotone_theory for ONE N (module level, so that worker processes can run it)."""
    Pp, Tp, Cp, tr, N, Ts, n_tuples, seed, exact_max = task
    rng = np.random.default_rng(seed); T_ = np.array(Ts, float); v = np.zeros((n_tuples, len(Ts)))
    p0 = (1.0 - tr.mean()) ** N                           # all N episodes untrapped: the limiting average is flat (p = 0); weighted EXACTLY
    for j in range(n_tuples):                             # (drawing it left a Bernoulli noise of ~.004 at N = 10, 2.8 s.e. off in one panel, 11-11)
        while True:                                       # the N joint realizations, given that at least one is trapped
            idx = rng.integers(0, len(tr), N); idx = idx[tr[idx]]
            if len(idx): break
        v[j] = (1.0 - (dip_exact if len(idx) <= exact_max else dip_saddle)(Pp[idx], Tp[idx], Cp[idx])) ** T_
    return p0 + (1.0 - p0) * v.mean(0), (1.0 - p0) * v.std(0) / np.sqrt(n_tuples)


def monotone_theory(pool, Ns, Ts, n_tuples, seed, exact_max=25, workers=1):
    """THEORY of the dips (tables alone, no simulated observation): an average of N episodes, each at its own joint realization drawn
    from the prior, stays monotone over the next T steps with probability E[(1 - p_R)^T], p_R the per-step dip probability given the
    N joint realizations R (steps i.i.d. given R: Proposition 2(a)); p_R exact (FFT) or by saddlepoint (the two agree to < .1% where
    both apply), averaged over n_tuples draws of R.  One N per worker process.  Returns the mean and its s.e. over draws,
    (len(Ns), len(Ts))."""
    tr = pool["trapped"]; Pp, Tp, Cp = pool["p"].astype(float), pool["theta"].astype(float), pool["const"]
    res = run_parallel(_mono_one, [(Pp, Tp, Cp, tr, int(N), tuple(Ts), n_tuples, seed + 1000 * i, exact_max) for i, N in enumerate(Ns)], workers,
                       "theory of the dips")
    return np.array([r[0] for r in res]), np.array([r[1] for r in res])


def theory_ns(cfg):
    """N at which the dip theory is evaluated: a log grid of N_TH_GRID values from min(MON_NS) to 2,000 AND every simulated N (the curves
    are steep in log N at long horizons: interpolating a coarse grid moved them by up to .02 at T = 1,000)."""
    grid = np.round(np.logspace(np.log10(min(cfg["MON_NS"])), np.log10(2000), cfg["N_TH_GRID"])).astype(int)
    return np.unique(np.r_[grid, np.array(cfg["MON_NS"], int)])


def build_third_theory(cfg, th3):
    """Panels g, h, THEORY per third (tables alone): the quantiles of the exact law of the N-average (g) and E_R[(1 - p_R)^T] (h)."""
    Nfine = np.unique(np.r_[np.round(np.logspace(0, 3, 40)).astype(int), np.array(cfg["G_NS"], int)]); qs = (0.05, 0.5, 0.95)   # + every simulated N
    Nth = theory_ns(cfg); out = []
    for i, T3 in enumerate(th3["thirds"]):
        x, w, q = step_atoms(T3["pool"])
        m, se = monotone_theory(T3["pool"], Nth, cfg["TH_TS"], cfg["MON_TUPLES"], 7171 + i, workers=cfg["WORKERS"])
        out.append(dict(Nfine=Nfine, gq=np.array([law_quantiles(x, w, q, N, qs) for N in Nfine]), Nth=Nth, mono=m, mono_se=se, q=float(q)))
    return dict(thirds=out, qs=qs)


def _pop_chunk(task):
    """build_population, one chunk of interaction sets (module level, so that worker processes can run it)."""
    cfg, tabs, asg, seed = task; t0, Tm, n_ep = cfg["POP_T0"], max(cfg["POP_TS"]), cfg["POP_EP"]; erng = np.random.default_rng(seed)
    sums = {N: np.zeros((ng, Tm)) for N, (ng, a) in asg.items()}; first, trapped = [], []
    for s_, ell in enumerate(tabs):
        lg = table_logs(ell); d_o, R, Rc = ell.shape; tr_tab = one_step_table(ell)["trapped"]
        rg, rc = erng.integers(0, R, n_ep), erng.integers(0, Rc, n_ep); pr = ell[:, rg, rc].T
        n = erng.binomial(t0, pr).astype(float); prev = fr_counts(n, t0, *lg); inc = np.empty((n_ep, Tm))
        for st in range(Tm):
            n += erng.random(pr.shape) < pr; cur = fr_counts(n, t0 + st + 1, *lg); inc[:, st] = cur - prev; prev = cur
        loc = s_ * n_ep + np.arange(n_ep)
        for N, (ng, a) in asg.items():
            g = a[loc]; ok = g >= 0
            if ok.any(): np.add.at(sums[N], g[ok], inc[ok])
        first.append(inc[:, 0].astype(np.float32)); trapped.append(tr_tab[rg, rc])
    return sums, np.concatenate(first), np.concatenate(trapped)


def build_population(cfg):
    """Figure 2 d (dips by horizon T) and Figure 1 g (the step of a single episode): POP_NZ interaction sets drawn at LAM_MAIN
    WITHOUT selection (the Gridworld's population; goal as returned).  THEORY (tables alone): their one-step pool.  SIMULATION
    (counts drawn binomially at POP_T0, exact observers): POP_EP episodes per set, each at its own joint realization, followed for
    max(POP_TS) steps.  For every N of MON_NS the episodes are dealt at random into POP_GROUPS independent groups of N (groups mix
    interaction sets), and the first step at which each group's average decreases is kept; also every episode's first step and
    whether its joint realization is trapped."""
    lam, Tm, n_ep = cfg["LAM_MAIN"], max(cfg["POP_TS"]), cfg["POP_EP"]; rng = np.random.default_rng(5151)
    tabs = [checked_table(make_env(rng, lam), lam) for _ in range(cfg["POP_NZ"])]
    it = iter(tabs); pool = one_step_pool(lam, len(tabs), None, both_goals=False, env=lambda r_, l_: next(it))
    E = len(tabs) * n_ep; grng = np.random.default_rng(5353); assign = {}
    for N in cfg["MON_NS"]:
        ng = min(cfg["POP_GROUPS"], E // N); a = np.full(E, -1, np.int32); a[grng.permutation(E)[:ng * N]] = np.arange(ng * N) // N; assign[N] = (ng, a)
    per = max(1, int(np.ceil(len(tabs) / (4 * cfg["WORKERS"])))); tasks = []
    for c0 in range(0, len(tabs), per):
        e0, e1 = c0 * n_ep, min(len(tabs), c0 + per) * n_ep
        tasks.append((cfg, tabs[c0:c0 + per], {N: (ng, a[e0:e1]) for N, (ng, a) in assign.items()}, 6000 + c0))
    res = run_parallel(_pop_chunk, tasks, cfg["WORKERS"], "population")
    first_dip = {}
    for N, (ng, _) in assign.items():
        neg = sum(r[0][N] for r in res) < 0                                               # the group's average decreases at that step
        first_dip[N] = np.where(neg.any(axis=1), neg.argmax(axis=1), Tm)                  # index of its first decrease (Tm: none)
    return dict(pool={k: (v.astype(np.float32) if k in ("p", "theta") else v) for k, v in pool.items()}, first_dip=first_dip,
                step=np.concatenate([r[1] for r in res]), trapped=np.concatenate([r[2] for r in res]), n_sets=len(tabs), n_ep=n_ep)


def _steps_lam_chunk(task):
    """build_steps_lam, one chunk: its own tables (own seed) and the first step after POP_T0 of POP_EP episodes per set."""
    cfg, lam, n_sets, seed = task; rng = np.random.default_rng(seed); erng = np.random.default_rng(seed + 10 ** 6); t0, n_ep = cfg["POP_T0"], cfg["POP_EP"]
    tabs = [checked_table(make_env(rng, lam), lam) for _ in range(n_sets)]; step, trapped = [], []
    for ell in tabs:
        lg = table_logs(ell); d_o, R, Rc = ell.shape; tr_tab = one_step_table(ell)["trapped"]
        rg, rc = erng.integers(0, R, n_ep), erng.integers(0, Rc, n_ep); pr = ell[:, rg, rc].T
        n = erng.binomial(t0, pr).astype(float); prev = fr_counts(n, t0, *lg); n += erng.random(pr.shape) < pr
        step.append((fr_counts(n, t0 + 1, *lg) - prev).astype(np.float32)); trapped.append(tr_tab[rg, rc])
    it = iter(tabs); pool = one_step_pool(lam, len(tabs), None, both_goals=False, env=lambda r_, l_: next(it))
    return {k: pool[k] for k in ("p", "theta", "const", "trapped", "Delta")}, np.concatenate(step), np.concatenate(trapped)


def build_steps_lam(cfg):
    """fr_theory_figure l at the likelihood temperatures other than LAM_MAIN (whose panel uses build_population): POP_NZ interaction sets
    drawn without selection, POP_EP episodes each, the first step after POP_T0 (counts drawn binomially, exact observers), and the
    sets' one-step pool (theory).  The same procedure as build_population, with its own seeds."""
    out = {}
    for lam in [l for l in cfg["LAMS_FULL"] if l != cfg["LAM_MAIN"]]:
        k = 4 * cfg["WORKERS"]; sizes = [len(c) for c in np.array_split(np.arange(cfg["POP_NZ"]), k)]
        res = run_parallel(_steps_lam_chunk, [(cfg, lam, m, 8100 + int(100 * lam) * 1000 + i) for i, m in enumerate(sizes) if m], cfg["WORKERS"],
                           f"steps at lambda = {lam:g}")
        out[lam] = dict(pool={f: np.concatenate([r[0][f] for r in res]) for f in ("p", "theta", "const", "trapped", "Delta")},
                        step=np.concatenate([r[1] for r in res]), trapped=np.concatenate([r[2] for r in res]))
    return out


def acc_times(cfg): return np.unique(np.round(10.0 ** (np.arange(0, np.log10(cfg["ACC_TMAX"]) * cfg["ACC_PER_DECADE"] + 0.5) / cfg["ACC_PER_DECADE"])).astype(np.int64))
def syn_times(cfg): return np.unique(np.round(10.0 ** (np.arange(0, np.log10(cfg["SYN_TMAX"]) * cfg["SYN_PER_DECADE"] + 0.5) / cfg["SYN_PER_DECADE"])).astype(np.int64))


def _q3_chunk(task):
    """build_thirds_more, one chunk of one third's interaction sets (own seed).  (i) FR_t for t = 1..T of N_EP episodes per set: per set
    the sums of FR_t and FR_t^2, and each episode's Delta; (ii) the first step after POP_T0 of POP_EP episodes per set, with whether the
    episode is trapped; (iii) the goal accuracy of the Joint and the Naive observer, Q3_ACC_EP episodes per set, on the grid acc_times."""
    cfg, tabs, seed = task; T, t0 = cfg["T"], cfg["POP_T0"]; erng = np.random.default_rng(seed); TA = acc_times(cfg); k = len(tabs)
    S1, S2, aJ, aN = np.zeros((k, T)), np.zeros((k, T)), np.zeros((k, len(TA))), np.zeros((k, len(TA))); dl, step, trp = [], [], []
    for s_, ell in enumerate(tabs):
        lg = table_logs(ell); d_o, R, Rc = ell.shape; tab = one_step_table(ell)
        n_ep = cfg["N_EP"]; rg, rc = erng.integers(0, R, n_ep), erng.integers(0, Rc, n_ep); pr = ell[:, rg, rc].T; n = np.zeros_like(pr)
        for t in range(1, T + 1):
            n += erng.random(pr.shape) < pr; f = fr_counts(n, t, *lg); S1[s_, t - 1] = f.sum(); S2[s_, t - 1] = (f ** 2).sum()
        dl.append(tab["Delta"][rg, rc])
        n_ep = cfg["POP_EP"]; rg, rc = erng.integers(0, R, n_ep), erng.integers(0, Rc, n_ep); pr = ell[:, rg, rc].T
        n = erng.binomial(t0, pr).astype(float); prev = fr_counts(n, t0, *lg); n += erng.random(pr.shape) < pr
        step.append((fr_counts(n, t0 + 1, *lg) - prev).astype(np.float32)); trp.append(tab["trapped"][rg, rc])
        n_ep = cfg["Q3_ACC_EP"]; rg, rc = erng.integers(0, R, n_ep), erng.integers(0, Rc, n_ep); pr = ell[:, rg, rc].T; n = np.zeros_like(pr); tp = 0
        for i, t in enumerate(TA):
            n = n + erng.binomial(int(t - tp), pr); tp = t; bj, bn = goal_answers(n, float(t), *lg); aJ[s_, i] = np.mean(bj == rg); aN[s_, i] = np.mean(bn == rg)
    return S1, S2, np.concatenate(dl), np.concatenate(step), np.concatenate(trp), aJ, aN


def build_thirds_more(cfg):
    """fr_theory_figure (11-11, John: f-h and l-n by the same q as o-q): for the SAME interaction sets as build_thirds (third_tables),
    simulations from step 0 (mean and s.d. of FR_t), the first step after POP_T0 (step law of a trapped episode) and the accuracy of both
    observers up to ACC_TMAX (Proposition 4).  Counts drawn binomially, exact observers."""
    keep, draws = third_tables(cfg); out = []
    for i, tabs in enumerate(keep):
        parts = [c for c in np.array_split(np.arange(len(tabs)), 2 * cfg["WORKERS"]) if len(c)]
        res = run_parallel(_q3_chunk, [(cfg, [tabs[j] for j in c], 33_000 + 100 * i + k) for k, c in enumerate(parts)], cfg["WORKERS"], f"third {i + 1}: simulations")
        out.append(dict(S1=np.concatenate([r[0] for r in res]), S2=np.concatenate([r[1] for r in res]), Delta=np.concatenate([r[2] for r in res]),
                        step=np.concatenate([r[3] for r in res]), trapped=np.concatenate([r[4] for r in res]), aJ=np.concatenate([r[5] for r in res]),
                        aN=np.concatenate([r[6] for r in res]), n_ep=cfg["N_EP"], acc_ep=cfg["Q3_ACC_EP"]))
    return dict(thirds=out, draws=draws, acc_t=acc_times(cfg))


def _syn_chunk(task):
    """build_synergy, one chunk of interaction sets (own seeds): per set and step of syn_times, the means over SYN_EP episodes of FR_t, of
    H(r_g | o_1:t) - H(r_g | o_1:t, r_c) (whose mean over episodes estimates the synergy I(r_g; r_c | o_1:t)) and of H(r_g | o_1:t) (Lambda_t)."""
    cfg, n_sets, seed = task; lam = cfg["LAM_MAIN"]; rng, erng = np.random.default_rng(seed), np.random.default_rng(seed + 10 ** 6); TS = syn_times(cfg)
    Fm, Sy, La = np.zeros((n_sets, len(TS))), np.zeros((n_sets, len(TS))), np.zeros((n_sets, len(TS))); dl = []
    for s_ in range(n_sets):
        ell = checked_table(make_env(rng, lam), lam); lg = table_logs(ell); L1, L0 = lg[0], lg[1]; d_o, R, Rc = ell.shape; tab = one_step_table(ell)
        n_ep = cfg["SYN_EP"]; rg, rc = erng.integers(0, R, n_ep), erng.integers(0, Rc, n_ep); pr = ell[:, rg, rc].T; n = np.zeros_like(pr); tp = 0; e = np.arange(n_ep)
        dl.append(tab["Delta"][rg, rc])
        for i, t in enumerate(TS):
            n = n + erng.binomial(int(t - tp), pr); tp = t
            LJ = np.einsum("ei,ijk->ejk", n, L1) + np.einsum("ei,ijk->ejk", t - n, L0)              # log P(o | r_g, r_c), up to a constant
            lm = logsumexp(LJ, axis=2); lm -= logsumexp(lm, axis=1, keepdims=True)                  # log P(r_g | o): the Joint goal belief
            lc = LJ[e, :, rc]; lc -= logsumexp(lc, axis=1, keepdims=True)                            # log P(r_g | o, r_c): told the contextual variable
            Hm = -np.sum(np.exp(lm) * lm, axis=1); Hc = -np.sum(np.exp(lc) * lc, axis=1)
            Fm[s_, i] = fr_counts(n, float(t), *lg).mean(); Sy[s_, i] = np.mean(Hm - Hc); La[s_, i] = Hm.mean()
    return Fm, Sy, La, np.concatenate(dl)


def build_synergy(cfg):
    """fr_theory_figure: Factorization Regret and synergy (Proposition 3) through time in the Gridworld's population at LAM_MAIN (POP_NZ
    interaction sets drawn without selection, SYN_EP episodes each, counts drawn binomially on the grid syn_times, exact observers)."""
    parts = [len(c) for c in np.array_split(np.arange(cfg["POP_NZ"]), 4 * cfg["WORKERS"]) if len(c)]
    res = run_parallel(_syn_chunk, [(cfg, m, 44_000 + k) for k, m in enumerate(parts)], cfg["WORKERS"], "synergy")
    return dict(t=syn_times(cfg), FR=np.concatenate([r[0] for r in res]), Syn=np.concatenate([r[1] for r in res]), Lam=np.concatenate([r[2] for r in res]),
                Delta=np.concatenate([r[3] for r in res]), n_ep=cfg["SYN_EP"])


def build_population_theory(cfg, pop):
    """THEORY for the population (tables alone): E_R[(1 - p_R)^T] for every N and every T of POP_TS (Figure 2 d)."""
    Nth = theory_ns(cfg); m, se = monotone_theory(pop["pool"], Nth, cfg["POP_TS"], cfg["MON_TUPLES"], 7272, workers=cfg["WORKERS"])
    return dict(Nth=Nth, mono=m, mono_se=se)


def build_bands(cfg):
    """Panel h: NZ_BAND interaction sets in each band of CFG['FRAC_BINS'] (the fraction of trapped joint realizations of the
    table as returned, goal on axis 1) at LAM_MAIN, found by drawing tables and keeping those that fall in a band that is not
    full yet.  The law of the step (line) and the simulated episodes (markers) use the SAME tables."""
    lam, bins, need = cfg["LAM_MAIN"], [tuple(b) for b in cfg["FRAC_BINS"]], cfg["NZ_BAND"]
    rng = np.random.default_rng(4040); keep = [[] for _ in bins]; draws = 0
    while any(len(k) < need for k in keep):
        ell = checked_table(make_env(rng, lam), lam); draws += 1; f = one_step_table(ell)["trapped"].mean()
        for k, (lo, hi) in zip(keep, bins):
            if lo <= f < hi and len(k) < need: k.append(ell)
        if draws >= cfg["BAND_MAX_DRAWS"]:
            raise SystemExit(f"build_bands: after {draws} tables the bands hold {[len(k) for k in keep]} of {need} interaction sets: "
                             "widen CFG['FRAC_BINS'] or lower NZ_BAND (and report it)")
    out = dict(bins=bins, draws=draws, bands=[])
    for i, tabs in enumerate(keep):
        it = iter(tabs); pool = one_step_pool(lam, len(tabs), None, both_goals=False, env=lambda r_, l_: next(it))
        it = iter(tabs); S = simulate(lam, len(tabs), cfg["N_EP"], cfg["T"], np.random.default_rng(5000 + i), env=lambda r_, l_: next(it))
        FR = S["FR"].astype(float); tr = S["trapped"]
        out["bands"].append(dict(pool={k: (v.astype(np.float32) if k in ("p", "theta") else v) for k, v in pool.items()},
                                 inc=FR[:, -1] - FR[:, -2], n_sets=len(tabs), trapped=tr,
                                 snap={t: (FR[:, t - 1] - FR[:, t - 2])[tr].astype(np.float32) for t in cfg["G_TIMES"]}))
        print(f"   panel h: band {bins[i][0]:.2f}-{bins[i][1]:.2f}: {len(tabs)} interaction sets", flush=True)
    print(f"   panel h: {draws} tables drawn", flush=True)
    return out


G_BOOT = 20000                                      # panel g: bootstrap N-episode averages per violin
G_ASINH, G_RANGE, G_BW = 0.3, (-2.0, 2.5), 0.12      # panel g: violins on z = asinh(change / 0.3) (linear near 0, logarithmic in the
                                                    # heavy tails), over this range of the change (nats), smoothed with this bandwidth in z


def z_density(v, w, z):
    """Density of z = asinh(v / G_ASINH) on the grid z (weights w, or equal): histogram + Gaussian smoothing (G_BW), the same
    for simulated values and for the atoms of a law; and the 2nd and 98th percentiles of z, where the violin is cut."""
    v = np.asarray(v, float); w = np.full(len(v), 1.0 / len(v)) if w is None else np.asarray(w, float) / np.sum(w)
    zv = np.arcsinh(v / G_ASINH); h = z[1] - z[0]; edges = np.r_[z - h / 2, z[-1] + h / 2]
    d, _ = np.histogram(zv, bins=edges, weights=w); d = d / h
    n = int(np.ceil(4 * G_BW / h)); k = np.exp(-0.5 * (np.arange(-n, n + 1) * h / G_BW) ** 2)
    o = np.argsort(zv); c = np.cumsum(w[o]); cut = np.interp([0.02, 0.98], c, zv[o])
    return np.convolve(d, k / k.sum(), mode="same"), cut


def mean_law(x, w, q, N, n=2 ** 18):
    """Law of the mean of N independent draws of X (atoms x with weights w summing to q, and 1 - q at 0): values and their
    probabilities on a grid (the FFT convolution of dip_law)."""
    mu, sd = step_moments(x, w); ring = 2.0 * (N * abs(mu) + 14.0 * np.sqrt(N) * sd + np.abs(x).max()); h = ring / n
    j = x / h; j0 = np.floor(j).astype(np.int64); f = j - j0
    pmf = np.bincount(j0 % n, weights=w * (1 - f), minlength=n) + np.bincount((j0 + 1) % n, weights=w * f, minlength=n)
    pmf[0] += 1.0 - q; pN = np.clip(np.fft.irfft(np.fft.rfft(pmf) ** int(N), n), 0, None)
    idx = np.arange(n); return np.where(idx < n // 2, idx, idx - n) * h / N, pN / pN.sum()


def law_quantiles(x, w, q, N, ps):
    """Quantiles ps of the mean of N independent draws of X (exact law, mean_law)."""
    v, pr = mean_law(x, w, q, N); o = np.argsort(v); return np.interp(ps, np.cumsum(pr[o]), v[o])


def derive(pools, sims, cfg, bands, small, opt, mon, mth, pop, pth, sl, qmore, syn):
    """Everything the figure, the checks and the numbers need, computed once from the pools and the simulations."""
    rng = np.random.default_rng(11); T = cfg["T"]; n_ep = cfg["N_EP"]; lam0 = cfg["LAM_MAIN"]; tt = np.arange(1, T + 1)
    S0, P0 = sims[lam0], pools["pool"][lam0]; D = dict(tt=tt)
    inc = lambda FR, t: FR[:, t].astype(float) - FR[:, t - 1].astype(float)          # FR_{t+1} - FR_t  (column c = after c + 1 steps)

    # ---- expected curve (f), slope (g), components and threshold (Corollary 1)
    D["curve"] = {}
    for lam in cfg["LAMS_FULL"]:
        S = sims[lam]; nZ = S["FR_mean"].shape[0]; off = S["N"] - S["J"]; ntr = S["n_trap"].sum()
        U = S["FR_untrap_sum"].sum(0) / max(nZ * n_ep - ntr, 1); W = S["FR_trap_sum"].sum(0) / max(ntr, 1)
        ker = np.ones(31) / 31; su, sw = np.convolve(np.diff(U), ker, "valid"), np.convolve(np.diff(W), ker, "valid")
        qst = np.maximum(-su, 0) / np.where(np.maximum(-su, 0) + sw > 0, np.maximum(-su, 0) + sw, np.inf)
        mean = S["FR_mean"].mean(0); se = S["FR_mean"].std(0) / np.sqrt(nZ); dz = np.diff(S["FR_mean"], axis=1)      # per-set increments
        P = pools["pool"][lam]; zq = np.bincount(P["set"], weights=P["trapped"]) / np.bincount(P["set"])                  # per-set share of trapped r in the pool
        q_z = (ntr / (nZ * n_ep) - P["trapped"].mean()) / max(np.hypot((S["n_trap"] / n_ep).std() / np.sqrt(nZ), zq.std() / np.sqrt(len(zq))), 1e-12)
        D["curve"][lam] = dict(mean=mean, se=se, off=off.mean(0), off_se=off.std(0) / np.sqrt(nZ), N=S["N"].mean(0), J=S["J"].mean(0),
                               ident_z=np.abs(mean - (tt * S["Delta_mean"].mean() + off.mean(0))).max() / se.max(), nZ=nZ, q_sim=ntr / (nZ * n_ep),
                               U=U, W=W, U_peak=int(U.argmax()) + 1, qstar_sim=float(qst.max()), min_frac=float(S["frac"].min()),
                               neg_steps=int(np.sum(dz.mean(0) < -2 * dz.std(0) / np.sqrt(nZ))), q_z=float(q_z))
    w0, w1 = cfg["SLOPE_WINDOW"]; D["slope"] = {}
    for lam in cfg["LAMS_SIM"]:
        s = (sims[lam]["FR_mean"][:, w1 - 1] - sims[lam]["FR_mean"][:, w0 - 1]) / (w1 - w0)
        D["slope"][lam] = (float(s.mean()), float(s.std() / np.sqrt(len(s))))

    # ---- law of the one-step change at trapped joint realizations (h)
    x, w, q = step_atoms(P0); cw = np.argsort(np.abs(x)); cum = np.cumsum(w[cw]) / w.sum()
    xmax = float(max(0.5, np.ceil(np.abs(x[cw])[np.searchsorted(cum, 0.96)] / 0.5) * 0.5)); edges = np.linspace(-xmax, xmax, 41)     # axis range of panel h: holds 96% of the law
    hgt, _ = np.histogram(x, bins=edges, weights=w / w.sum()); dX = inc(S0["FR"], T - 1)[S0["trapped"]]
    cnt, _ = np.histogram(dX, bins=edges); ph = cnt / len(dX); wd = np.diff(edges)
    D["step"] = dict(edges=edges, dens=hgt / wd, sim=ph / wd, sim_2se=2 * np.sqrt(ph * (1 - ph) / len(dX)) / wd, xmax=xmax, n=len(dX),
                     p_pred=float(w[x < 0].sum() / w.sum()), p_sim=float(np.mean(dX < 0)),
                     tied_share=float(np.mean(P0["Delta"][P0["trapped"]] < 0.01)))

    # ---- probability that an N-episode average decreases (i, j)
    Nline = np.unique(np.round(np.logspace(np.log10(5), np.log10(2000), 60)).astype(int)); Nmark = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    D["Nline"], D["Nmark"], D["dip"] = Nline, Nmark, {}
    for lam in cfg["LAMS_FULL"]:
        x, w, q = step_atoms(pools["pool"][lam]); lo, hi = dip_law(x, w, q, Nline); mu, sd = step_moments(x, w); FR = sims[lam]["FR"]
        late = np.array([boot_dip(inc(FR, T - 1), N, rng) for N in Nmark]); early = np.array([boot_dip(inc(FR, cfg["T_EARLY"]), N, rng) for N in Nmark])
        mid = (lo + hi) / 2; at = np.searchsorted(Nline, Nmark)
        D["dip"][lam] = dict(mid=mid, gauss=gaussian_limit(x, w, Nline), late=late, early=early, q=float(q), mu=float(mu), sd=float(sd),
                             gauss_maxdiff=float(np.abs(gaussian_limit(x, w, Nline) - mid).max()), sim_maxdiff=float(np.abs(late - mid[at]).max()))
    bins = bands["bins"]; D["bins"], D["bins_auto"], D["dip_frac"], D["band_draws"] = bins, False, [], bands["draws"]     # panel h
    w0, w1 = cfg["SLOPE_WINDOW"]                                                                                             # panel f
    D["perset"] = dict(pred10=S0["Delta_mean"].copy(), slope10=(S0["FR_mean"][:, w1 - 1] - S0["FR_mean"][:, w0 - 1]) / (w1 - w0),
                       pred3=small["pred"], slope3=small["slope"], k3=small["k"], frT3=small["frT"])
    for b in bands["bands"]:
        x, w, q = step_atoms(b["pool"]); lo, hi = dip_law(x, w, q, Nline)
        D["dip_frac"].append(dict(mid=(lo + hi) / 2, late=np.array([boot_dip(b["inc"], N, rng) for N in Nmark]), n_episodes=len(b["inc"]), q=float(q)))

    # ---- panel f (theory and simulation on the same interaction sets): per point, mean and s.e. over sets
    ms = lambda v: (float(np.mean(v)), float(np.std(v) / np.sqrt(len(v))))
    D["opt"] = {k: dict(ed=ms(v["ed"]), slope=ms(v["slope"]), q=ms(v["q"]), gap=[ms(v["accJ"][:, i] - v["accN"][:, i]) for i in range(len(cfg["OPT_TACC"]))],
                        accJ=[ms(v["accJ"][:, i]) for i in range(len(cfg["OPT_TACC"]))]) for k, v in opt.items()}
    # ---- panels g, h per third of the trapped fraction.  g: quantiles of the change of an N-average (theory: exact law of X-bar_N from
    #      the third's tables; simulation: independent groups of N simulated episodes, bootstrap 95% intervals over groups).  h: the
    #      probability of at least one decrease in the next T steps (theory: 1 - E_R[(1 - p_R)^T]; simulation: fraction of groups)
    grng = np.random.default_rng(13); qs = mth["qs"]; D["fan3"], D["mono3"], D["third_bins"] = [], [], mon["bins"]
    mrng = np.random.default_rng(19)                     # 11-11 (fr_theory_figure): mean +- s.d. of the group means; own generator, so the streams above are unchanged
    for T3, TT in zip(mon["thirds"], mth["thirds"]):
        gs, sq, slo, shi, msd, msd_se = T3["g"].astype(float), [], [], [], [], []
        for N in cfg["G_NS"]:
            ng = min(cfg["G_GROUPS"], len(gs) // N); means = gs[grng.permutation(len(gs))[:ng * N]].reshape(ng, N).mean(1)
            sq.append(np.quantile(means, qs)); bt = np.array([np.quantile(means[grng.integers(0, ng, ng)], qs) for _ in range(200)])
            slo.append(np.quantile(bt, 0.025, axis=0)); shi.append(np.quantile(bt, 0.975, axis=0))
            msd.append([means.mean() - means.std(ddof=1), means.mean() + means.std(ddof=1)])
            bm = np.array([[b.mean() - b.std(ddof=1), b.mean() + b.std(ddof=1)] for b in (means[mrng.integers(0, ng, ng)] for _ in range(200))]); msd_se.append(bm.std(axis=0))
        D["fan3"].append(dict(Nfine=TT["Nfine"], theory=TT["gq"], Ns=np.array(cfg["G_NS"]), sim=np.array(sq), lo=np.array(slo), hi=np.array(shi),
                              msd=np.array(msd), msd_se=np.array(msd_se)))
        hs, msim, mse = T3["h"].astype(float), [], []
        for N in cfg["MON_NS"]:
            ng = min(cfg["MON_GROUPS"], len(hs) // N); avg = hs[grng.permutation(len(hs))[:ng * N]].reshape(ng, N, -1).mean(1)
            dip = np.array([np.any(avg[:, :T] < 0, axis=1) for T in cfg["TH_TS"]]).T.astype(float)
            msim.append(dip.mean(0)); mse.append(np.sqrt(np.maximum(dip.mean(0) * (1 - dip.mean(0)), 1.0 / ng) / ng))
        D["mono3"].append(dict(Ns=np.array(cfg["MON_NS"]), sim=np.array(msim), se=np.array(mse), Nth=TT["Nth"], theory=1 - TT["mono"], theory_se=TT["mono_se"], q=TT["q"]))

    for T3, FN in zip(mon["thirds"], D["fan3"]):                                           # the mean step of each third (Figure 2 a-c)
        x3, w3, q3 = step_atoms(T3["pool"]); FN["mean"], FN["sigma"] = (float(v) for v in step_moments(x3, w3)); FN["q"] = float(q3)
    # ---- Figure 2 d: dips of N-averages by horizon T, the Gridworld's population (theory: 1 - E_R[(1 - p_R)^T]; simulation: groups)
    fd = pop["first_dip"]; Ns_ = np.array(sorted(fd)); TT_ = np.array(cfg["POP_TS"])
    dsim = np.array([[np.mean(fd[N] < T_) for T_ in TT_] for N in Ns_]); ng_ = np.array([len(fd[N]) for N in Ns_])[:, None]
    D["pop_h"] = dict(Ns=Ns_, Ts=TT_, sim=dsim, se=np.sqrt(np.maximum(dsim * (1 - dsim), 1.0 / ng_) / ng_), Nth=pth["Nth"], theory=1 - pth["mono"],
                      theory_se=pth["mono_se"], n_sets=pop["n_sets"], n_ep=pop["n_ep"])
    # ---- Figure 1 g: one step of a single episode at a trapped joint realization (theory: law of X; simulation: first step at POP_T0)
    x, w, q = step_atoms(pop["pool"]); wt = w / w.sum(); stp = pop["step"][pop["trapped"]].astype(float)
    cw = np.argsort(np.abs(x)); cum = np.cumsum(wt[cw]); xmax = float(max(0.5, np.ceil(np.abs(x[cw])[np.searchsorted(cum, 0.96)] / 0.5) * 0.5))
    edges = np.linspace(-xmax, xmax, 41); wd = np.diff(edges); hgt, _ = np.histogram(x, bins=edges, weights=wt); cnt, _ = np.histogram(stp, bins=edges)
    ph = cnt / len(stp)
    D["pop_step"] = dict(edges=edges, dens=hgt / wd, sim=ph / wd, sim_2se=2 * np.sqrt(ph * (1 - ph) / len(stp)) / wd, xmax=xmax, n=len(stp),
                         p_pred=float(wt[x < 0].sum()), p_sim=float(np.mean(stp < 0)), q=float(q), q_sim=float(np.mean(pop["trapped"])))
    # ---- fr_theory_figure l (11-11): the same, at every lambda of LAMS_FULL, on SHARED bins (0.1 nats) over the 96% range of the widest law
    src = {lam: (pop["pool"], pop["step"], pop["trapped"]) if lam == lam0 else (sl[lam]["pool"], sl[lam]["step"], sl[lam]["trapped"]) for lam in cfg["LAMS_FULL"]}
    xs = 0.0
    for lam, (pl, _, _) in src.items():
        x, w, q = step_atoms(pl); wt = w / w.sum(); cw = np.argsort(np.abs(x)); xs = max(xs, np.abs(x[cw])[np.searchsorted(np.cumsum(wt[cw]), 0.96)])
    xs = float(np.ceil(xs / 0.5) * 0.5); edges = np.linspace(-xs, xs, int(round(2 * xs / 0.1)) + 1); wd = np.diff(edges); D["step_lam"] = {}
    for lam, (pl, st, trf) in src.items():
        x, w, q = step_atoms(pl); wt = w / w.sum(); stp = st[trf].astype(float); hgt, _ = np.histogram(x, bins=edges, weights=wt)
        cnt, _ = np.histogram(stp, bins=edges); ph = cnt / len(stp)
        D["step_lam"][lam] = dict(edges=edges, dens=hgt / wd, sim=ph / wd, sim_2se=2 * np.sqrt(ph * (1 - ph) / len(stp)) / wd, xmax=xs, n=len(stp),
                                  p_pred=float(wt[x < 0].sum()), p_sim=float(np.mean(stp < 0)), q=float(q))
    # ---- fr_theory_figure (11-11): per third of q (the sets of Figure 2 a-c): mean +- s.d. of FR_t, the step of a trapped episode, accuracy
    qrng = np.random.default_rng(23); D["by_q"] = []; tmk3 = np.arange(T // 10, T + 1, T // 10) - 1; xs3 = 0.0
    for T3 in mon["thirds"]:
        x, w, q = step_atoms(T3["pool"]); wt = w / w.sum(); cw = np.argsort(np.abs(x)); xs3 = max(xs3, np.abs(x[cw])[np.searchsorted(np.cumsum(wt[cw]), 0.96)])
    xs3 = float(np.ceil(xs3 / 0.5) * 0.5); e3 = np.linspace(-xs3, xs3, int(round(2 * xs3 / 0.1)) + 1); wd3 = np.diff(e3)
    for T3, Q in zip(mon["thirds"], qmore["thirds"]):
        x, w, q = step_atoms(T3["pool"]); wt = w / w.sum(); ne = Q["n_ep"]; S1, S2 = Q["S1"][:, tmk3], Q["S2"][:, tmk3]; nz = len(S1)
        def msd(idx): m_ = S1[idx].sum(0) / (len(idx) * ne); s_ = np.sqrt(np.maximum(S2[idx].sum(0) / (len(idx) * ne) - m_ ** 2, 0)); return m_, m_ - s_, m_ + s_
        m0, lo0, hi0 = msd(np.arange(nz)); bs = np.array([np.stack(msd(qrng.integers(0, nz, nz))) for _ in range(200)])     # (200, 3, steps)
        stp = Q["step"][Q["trapped"]].astype(float); hgt, _ = np.histogram(x, bins=e3, weights=wt); cnt, _ = np.histogram(stp, bins=e3); ph = cnt / len(stp)
        gap = Q["aJ"] - Q["aN"]
        D["by_q"].append(dict(q=float(q), t=tt, mean_th=tt * Q["Delta"].mean(), sd_th=np.sqrt(tt ** 2 * Q["Delta"].var() + tt * T3["pool"]["v"].mean()),
                              tm=tt[tmk3], mean_m=m0, lo_m=lo0, hi_m=hi0, mean_se=bs[:, 0].std(0), lo_se=bs[:, 1].std(0), hi_se=bs[:, 2].std(0), n_sets=nz,
                              step=dict(edges=e3, dens=hgt / wd3, sim=ph / wd3, sim_2se=2 * np.sqrt(ph * (1 - ph) / len(stp)) / wd3, xmax=xs3, n=len(stp),
                                        p_pred=float(wt[x < 0].sum()), p_sim=float(np.mean(stp < 0))),
                              acc=dict(t=qmore["acc_t"], gap=gap.mean(0), gap_se=gap.std(0) / np.sqrt(nz), J=Q["aJ"].mean(0), N=Q["aN"].mean(0))))
    # ---- fr_theory_figure: Factorization Regret and synergy through time, the population (Proposition 3)
    nzs = len(syn["FR"]); D["syn"] = dict(t=syn["t"], EDelta=float(syn["Delta"].mean()), n_sets=nzs, n_ep=syn["n_ep"],
                                         **{k: syn[k].mean(0) for k in ("FR", "Syn", "Lam")}, **{k + "_se": syn[k].std(0) / np.sqrt(nzs) for k in ("FR", "Syn", "Lam")})
    # ---- Figure 1 h: spread of FR_t across episodes (Proposition 2(c); theory: sqrt(t^2 Var(Delta) + t E[v]), Var(Delta) over the joint
    #      realizations of the simulated episodes themselves (tables alone; with the moments of the pool's OTHER 2,000 sets the curves
    #      sat 2.5% above the simulation at lambda = 2, 3, a sampling difference between the two sets of tables), E[v] (a ~1% term)
    #      from the pool; simulation: s.d. over all episodes, s.e. by resampling interaction sets)
    D["spread"] = {}; brng = np.random.default_rng(17); tmk = np.arange(T // 10, T + 1, T // 10) - 1
    for lam in cfg["LAMS_FULL"]:
        P = pools["pool"][lam]; FR = sims[lam]["FR"].astype(float); nZ = FR.shape[0] // n_ep
        sd = FR.std(axis=0); th = np.sqrt(tt ** 2 * sims[lam]["Delta"].var() + tt * P["v"].mean())
        blk = FR[:, tmk].reshape(nZ, n_ep, -1); bs = []; bms = []
        for _ in range(200):
            rs = blk[brng.integers(0, nZ, nZ)].reshape(nZ * n_ep, -1); bs.append(rs.std(axis=0)); bms.append(np.r_[rs.mean(0) - rs.std(0), rs.mean(0) + rs.std(0)])
        mn = FR.mean(axis=0); bms = np.array(bms); k_ = len(tmk)
        D["spread"][lam] = dict(t=tt, sd=sd, th=th, tm=tt[tmk], sd_m=sd[tmk], se_m=np.std(bs, axis=0),
                                mean_th=tt * sims[lam]["Delta"].mean(), mean_m=mn[tmk], mean_se=FR[:, tmk].reshape(nZ, n_ep, -1).mean(1).std(0) / np.sqrt(nZ),
                                lo_m=mn[tmk] - sd[tmk], hi_m=mn[tmk] + sd[tmk], lo_se=bms[:, :k_].std(0), hi_se=bms[:, k_:].std(0))

    # ---- values behind the qualitative statements of the tex (v12 quotes no numbers; `numbers` prints these)
    sm = pools["summary"]; snr = {lam: sm[lam]["EDelta"] / sm[lam]["sigma"] for lam in cfg["LAMS_FULL"]}
    qs = [sm[lam]["q"] for lam in cfg["LAMS_POOL"]]
    D["numbers"] = dict(
        snr={str(l): float(v) for l, v in snr.items()},
        episodes_needed={str(l): float(2 * np.log(cfg["PAPER_HORIZON"] / cfg["ALPHA"]) / v ** 2) for l, v in snr.items()},
        p_decrease_pred=D["step"]["p_pred"], p_decrease_sim=D["step"]["p_sim"],
        gauss_maxdiff=max(D["dip"][l]["gauss_maxdiff"] for l in cfg["LAMS_FULL"]),
        q_min=float(min(qs)), q_max=float(max(qs)),
        min_frac=min(D["curve"][l]["min_frac"] for l in cfg["LAMS_FULL"]), n_sets_full=int(sum(D["curve"][l]["nZ"] for l in cfg["LAMS_FULL"])),
        qstar_sim={str(l): D["curve"][l]["qstar_sim"] for l in cfg["LAMS_FULL"]},
        offset_T={str(l): float(D["curve"][l]["off"][-1]) for l in cfg["LAMS_FULL"]},
        Lambda_T={str(l): float(D["curve"][l]["J"][-1]) for l in cfg["LAMS_FULL"]},
        atom_bound=float(max((1 - D["dip"][l]["q"]) ** 5 for l in cfg["LAMS_FULL"])),
        frac_bins=[[float(a), float(b)] for a, b in bins], tied_share=D["step"]["tied_share"],
        slope_weakest=[float(min(cfg["LAMS_SIM"])), D["slope"][min(cfg["LAMS_SIM"])][0], float(sm[min(cfg["LAMS_SIM"])]["EDelta"])],
        EDelta={str(l): float(sm[l]["EDelta"]) for l in cfg["LAMS_FULL"]}, sigma={str(l): float(sm[l]["sigma"]) for l in cfg["LAMS_FULL"]},
        q={str(l): float(sm[l]["q"]) for l in cfg["LAMS_FULL"]})
    return D


def load_all(cfg, store, fresh=False, need_sims=True, rederive=False):
    row1 = store.get("row1", (ROW1_VERSION, cfg["N_MC"]), lambda: build_row1(cfg), fresh)
    if not need_sims: return row1, None, None                        # selftest: nothing below, so it does not depend on make_env()
    fp = env_fingerprint(cfg)
    sig_p = (POOL_VERSION, fp, tuple(cfg["LAMS_POOL"]), tuple(cfg["LAMS_FULL"]), cfg["NZ_POOL"], cfg["POOL_BOTH_GOALS"])
    sig_s = (SIM_VERSION, fp, tuple(cfg["LAMS_SIM"]), tuple(cfg["LAMS_FULL"]), cfg["NZ_FULL"], cfg["NZ_LIGHT"], cfg["N_EP"], cfg["T"])
    sig_b = (BAND_VERSION, fp, str(cfg["FRAC_BINS"]), cfg["NZ_BAND"], cfg["N_EP"], cfg["T"], cfg["LAM_MAIN"], tuple(cfg["G_TIMES"]))
    sig_m = (SMALL_VERSION, fp, cfg["R_SMALL"], cfg["NZ_SMALL"], cfg["NEP_SMALL"], cfg["T_SMALL"], cfg["LAM_MAIN"])
    sig_o = (OPT_VERSION, fp, tuple(cfg["OPT_RS"]), tuple(cfg["OPT_DOS"]), cfg["OPT_NZ"], cfg["OPT_NEP"], cfg["OPT_T1"], cfg["OPT_T2"], tuple(cfg["OPT_TACC"]), cfg["LAM_MAIN"])
    sig_n = (MON_VERSION, fp, str(cfg["FRAC_BINS"]), cfg["NZ_BAND"], cfg["TH_T0"], cfg["TH_G_EP"], cfg["TH_H_EP"], max(cfg["TH_TS"]), cfg["LAM_MAIN"])
    sig_t = (MON_VERSION, sig_n, tuple(cfg["TH_TS"]), cfg["MON_TUPLES"], tuple(cfg["MON_NS"]), cfg["N_TH_GRID"], "stratified")
    sig_pop = (POP_VERSION, fp, cfg["POP_NZ"], cfg["POP_EP"], cfg["POP_T0"], max(cfg["POP_TS"]), tuple(cfg["MON_NS"]), cfg["POP_GROUPS"], cfg["LAM_MAIN"])
    sig_popt = (POP_VERSION, sig_pop, tuple(cfg["POP_TS"]), cfg["MON_TUPLES"], cfg["N_TH_GRID"], "stratified")
    sig_sl = (POP_VERSION, fp, cfg["POP_NZ"], cfg["POP_EP"], cfg["POP_T0"], tuple(cfg["LAMS_FULL"]), cfg["LAM_MAIN"], "steps_lam 2")
    sig_q3 = ("thirds_more 1", fp, str(cfg["FRAC_BINS"]), cfg["NZ_BAND"], cfg["BAND_MAX_DRAWS"], cfg["LAM_MAIN"], cfg["T"], cfg["N_EP"], cfg["POP_T0"], cfg["POP_EP"],
              cfg["Q3_ACC_EP"], cfg["ACC_TMAX"], cfg["ACC_PER_DECADE"])
    sig_sy = ("synergy 1", fp, cfg["POP_NZ"], cfg["SYN_EP"], cfg["SYN_TMAX"], cfg["SYN_PER_DECADE"], cfg["LAM_MAIN"])
    sig_d = (DERIVED_VERSION, sig_p, sig_s, sig_b, sig_m, sig_o, sig_n, sig_t, sig_pop, sig_popt, sig_sl, sig_q3, sig_sy, cfg["LAM_MAIN"], tuple(cfg["SLOPE_WINDOW"]), cfg["T_EARLY"],
             cfg["PAPER_HORIZON"], cfg["ALPHA"], tuple(cfg["G_TIMES"]), tuple(cfg["G_NS"]), cfg["G_GROUPS"], tuple(cfg["MON_NS"]), cfg["MON_GROUPS"])
    path_d = store.path("derived")
    if os.path.exists(path_d) and not fresh and not rederive:        # the usual case while polishing: nothing heavy is loaded
        with open(path_d, "rb") as fh: sig, data = pickle.load(fh)
        if sig == sig_d: return row1, data["summary"], data["D"]
    pools = store.get("pools", sig_p, lambda: build_pools(cfg), fresh)
    sims = store.get("sims", sig_s, lambda: build_sims(cfg), fresh)
    bands = store.get("bands", sig_b, lambda: build_bands(cfg), fresh)
    small = store.get("small", sig_m, lambda: build_small(cfg), fresh)
    opt = store.get("optimum", sig_o, lambda: build_optimum(cfg), fresh)
    mon = store.get("thirds", sig_n, lambda: build_thirds(cfg), fresh)
    mth = store.get("thirds_theory", sig_t, lambda: build_third_theory(cfg, mon), fresh)
    pop = store.get("population", sig_pop, lambda: build_population(cfg), fresh)
    pth = store.get("population_theory", sig_popt, lambda: build_population_theory(cfg, pop), fresh)
    sl = store.get("steps_lam", sig_sl, lambda: build_steps_lam(cfg), fresh)
    qmore = store.get("thirds_more", sig_q3, lambda: build_thirds_more(cfg), fresh)
    syn = store.get("synergy", sig_sy, lambda: build_synergy(cfg), fresh)
    data = store.get("derived", sig_d, lambda: dict(summary=pools["summary"], D=derive(pools, sims, cfg, bands, small, opt, mon, mth, pop, pth, sl, qmore, syn)), fresh=True)
    return row1, data["summary"], data["D"]


# ============================================================================================================
# 6. SELF-TEST, ENVIRONMENT CHECK, THEORY-VERSUS-SIMULATION CHECKS, NUMBERS
# ============================================================================================================
class Report:
    def __init__(self, name): self.name = name; self.rows = []; print(f"\n=== {name} ===")
    def add(self, ok, what, detail="", warn_only=False):
        tag = "PASS" if ok else ("WARN" if warn_only else "FAIL"); self.rows.append(tag); print(f"[{tag}] {what}" + (f"   {detail}" if detail else ""), flush=True)
    def close(self):
        n = {k: self.rows.count(k) for k in ("PASS", "WARN", "FAIL")}; print(f"--- {self.name}: {n['PASS']} passed, {n['WARN']} warnings, {n['FAIL']} failed"); return n["FAIL"] == 0


def near(x, ref, tol): return abs(x - ref) <= tol


def selftest(cfg, store):
    """Environment-independent: must pass on any machine.  Reference values are the ones quoted in the tex."""
    R = Report("selftest (exact tables; environment-independent)"); row1, _, _ = load_all(cfg, store, need_sims=False); a, b = row1["a"], row1["b"]
    t = np.arange(T_ANATOMY + 1)
    for key, e in (("a", a), ("b", b)):
        an, st = e["an"], e["st"]
        R.add(np.abs(an["FR"] - (an["T"] + an["D"] - an["Lam"])).max() < 1e-9, f"table {key}: E[FR_t] = trap term + distraction term - Lambda_t (Prop. 1b)")
        R.add(np.abs(an["FR"] - (t * st["Delta"].mean() + an["N"] - an["Lam"])).max() < 1e-9, f"table {key}: E[FR_t] = t E[Delta] + N_t - Lambda_t (Prop. 1c)")
        R.add(np.diff(-an["Lam"]).min() > -1e-12 and np.diff(an["T"]).min() > -1e-12, f"table {key}: Joint term and trap term are non-decreasing")
        tm = np.arange(30, T_ANATOMY + 1, 30); z = np.abs(e["mc"][0][tm] - an["FR"][tm]) / e["mc"][1][tm]
        R.add(z.max() < 4.5, f"table {key}: Monte Carlo agrees with the exact expectation", f"max |z| = {z.max():.2f} over {len(tm)} steps")
    R.add(a["st"]["k"] == 0 and near(a["an"]["FR"].max(), 0.055006, 2e-5) and a["an"]["FR"].argmax() == 82, "table a: no trap, peak 0.0550 at t = 82",
          f"k = {a['st']['k']}, peak {a['an']['FR'].max():.6f} at t = {a['an']['FR'].argmax()}")
    R.add(a["efr"][T_MIXTURE] < 1e-6, "table a: E[FR_t] returns to zero (Prop. 1d)", f"E[FR_{T_MIXTURE}] = {a['efr'][T_MIXTURE]:.1e}")
    R.add(b["st"]["k"] == 1 and near(b["st"]["Delta"].mean(), 0.0055298, 2e-6) and near(b["an"]["FR"][300], 1.609061, 2e-4), "table b: one trapped joint realization, E[Delta] = 0.0055",
          f"k = {b['st']['k']}, E[Delta] = {b['st']['Delta'].mean():.7f}, E[FR_300] = {b['an']['FR'][300]:.6f}")
    w_star, step = mixture_threshold(a["efr"], b["efr"]); q_star = w_star / 9
    R.add(near(q_star, 0.004114, 3e-6), "panel d / Corollary 1: threshold q* = 0.0041", f"w* = {w_star:.6f} at step {step}, q* = {q_star:.6f}")
    for q, ref in ((0.001, (88, 444)), (0.002, (97, 318))):
        got = decreasing_steps((1 - 9 * q) * a["efr"] + 9 * q * b["efr"]); R.add(got == ref, f"panel d: at q = {q} the curve decreases over t in [{ref[0]}, {ref[1]}]", f"got {got}")
    R.add(decreasing_steps((1 - w_star * 1.0001) * a["efr"] + w_star * 1.0001 * b["efr"]) is None, "panel d: no decrease just above q*")
    for e in row1["c"]:
        R.add(e["st"]["k"] == 0 and 4.5 < e["efr"].argmax() * e["st"]["delta"] < 5.5, f"distractor-count tables (not drawn since 11-10): {e['R'] - 1} distractor(s), no trap, peak at t = 5 / delta",
              f"delta = {e['st']['delta']:.4f}, peak {e['efr'].max():.4f} at t = {e['efr'].argmax()}")
    for e in row1["d"]:
        R.add(e["st"]["k"] == 0 and 4.5 < e["efr"].argmax() * e["st"]["delta"] < 5.5, f"panel c: alpha = {e['alpha']}, no trap, peak at t = 5 / delta",
              f"delta = {e['st']['delta']:.4f}, peak {e['efr'].max():.4f} at t = {e['efr'].argmax()}")
    # --- the 2 x 2 examples of the tex
    an, st = exact_anatomy([[0.9, 0.6], [0.3, 0.2]], 400); F = an["FR"]; nl = an["T"] + an["D"]; dN, dJ = -np.diff(nl), -np.diff(an["Lam"]); first = int(np.flatnonzero((dN > dJ + 1e-15)[1:])[0]) + 1
    R.add(st["k"] == 0 and near(F.max(), 0.0655, 6e-4) and F.argmax() == 31 and F[400] < 1e-4, "Example 1 (no trap): maximum 0.066 at t = 31, below 1e-4 by t = 400",
          f"max {F.max():.4f} at t = {F.argmax()}, E[FR_400] = {F[400]:.1e}")
    R.add(first == 31 and near(dN[31], 0.0032, 6e-5) and near(dJ[31], 0.0031, 6e-5), "Example 1: first decrease at t = 31, Naive loss reduction 0.0032 > Joint information gain 0.0031",
          f"step {first}: {dN[first]:.5f} vs {dJ[first]:.5f}")
    an, st = exact_anatomy([[0.9, 0.1], [0.7, 0.6]], 400); F = an["FR"]
    R.add(st["k"] == 1 and bool(st["trapped"][0, 0]) and st["ahat"][0, 0] == 1 and near(st["Delta"].max(), 0.2005, 6e-5) and near(st["Delta"].mean(), 0.050, 6e-4),
          "Example 2: joint realization (1, 1) trapped by a = 2, Delta = 0.2005, E[Delta] = 0.050", f"Delta = {st['Delta'].max():.5f}, E[Delta] = {st['Delta'].mean():.5f}")
    R.add(bool((np.diff(F)[1:] > 0).all()) and near(np.diff(F)[-1], 0.050, 6e-4) and near(st["delta"], 0.015, 6e-4), "Example 2: increasing at every step, increments -> 0.050, weakest distractor drift 0.015",
          f"last increment {np.diff(F)[-1]:.5f}, delta = {st['delta']:.5f}")
    R.add(F[1] < 1e-12 and near(an["Syn"][1], 0.079, 6e-4) and near(F[10], 0.54, 6e-3) and near(an["Syn"][10], 0.046, 6e-4), "Example 2 (Prop. 5 discussion): E[FR_1] = 0 < Syn_1 = 0.079, E[FR_10] = 0.54 > Syn_10 = 0.046",
          f"E[FR_1] = {F[1]:.1e}, Syn_1 = {an['Syn'][1]:.4f}, E[FR_10] = {F[10]:.4f}, Syn_10 = {an['Syn'][10]:.4f}")
    F = exact_efr([[0.9, 0.1], [0.5, 0.5]], 400); R.add(near(F[400], np.log(2), 1e-4), "Example 3 (tie, violates A3): E[FR_t] -> ln 2", f"E[FR_400] = {F[400]:.5f}")
    # --- a single interaction set with one trapped joint realization whose expected curve dips (tex: 'What is and is not guaranteed')
    ell = additive_table(6, 0.8, 0.48, Rc=24, perturb=[(0, 23, +1)], gamma=0.2); st = table_stats(ell); E = exact_efr(ell, 700); got = decreasing_steps(E)
    R.add(st["k"] == 1 and ell.size == 144 and near(st["Delta"].max(), 0.0013, 6e-5) and got == (256, 581) and near(E[256] - E[581], 0.0018, 6e-5),
          "6 x 24 table with one barely trapped joint realization: decreases from t = 256 to t = 581 by 0.0018 nats", f"k = {st['k']}, Delta = {st['Delta'].max():.5f}, decrease {got}, depth {E[256] - E[581]:.5f}")
    # --- the Gridworld simulator against the independent single-episode implementation, on the reference environment
    S = simulate(2.0, 2, 8, 60, np.random.default_rng(5), keep_last=True, env=reference_make_env); L = S["last"]
    err = max(np.abs(fr_episode(L["ell"], L["obs"][:, e]) ["FR"] - S["FR"][8 + e]).max() for e in range(8))
    R.add(err < 2e-4, "simulate() and fr_episode() give the same Factorization Regret on identical observations", f"max difference {err:.1e} nats")
    return R.close()


def envcheck(cfg):
    """Validate the adapter and compare its one-step statistics with the reference re-implementation."""
    R = Report("envcheck (the adapter make_env)"); lam0 = cfg["LAM_MAIN"]; lam_hi = max(cfg["LAMS_POOL"]); env = lambda seed, lam: np.asarray(make_env(np.random.default_rng(seed), lam))
    t0 = time.time(); tabs = [env(s, lam0) for s in range(20)]; ms = (time.time() - t0) / 20 * 1e3
    ell = tabs[0]; both = tabs + [env(s, lam_hi) for s in range(20)]
    R.add(ell.ndim == 3 and np.issubdtype(ell.dtype, np.floating), "returns a float array ell[i, r_g, r_c]", f"shape {ell.shape} = (d_o, R, R_c), {ell.dtype}")
    R.add(all(t.ndim == 3 and np.isfinite(t).all() and t.min() > 0 and t.max() < 1 for t in both), "values strictly inside (0, 1)",
          f"min {min(float(np.nanmin(t)) for t in both):.1e}, 1 - max {1 - max(float(np.nanmax(t)) for t in both):.1e}  (20 interaction sets each at lambda = {lam0:g} and {lam_hi:g}; "
          "exact 0 or 1 usually means a float32 sigmoid: use double precision)")
    R.add(np.array_equal(env(3, lam0), env(3, lam0)), "deterministic given the generator (all randomness comes from rng)")
    R.add(not np.array_equal(tabs[0], tabs[1]), "different generators give different interaction sets")
    d = np.abs(env(3, lam0).astype(float) - env(3, lam0 * 1.001).astype(float)).max() if env(3, lam0).shape == env(3, lam0 * 1.001).shape else np.inf
    R.add(d < 0.02, "same generator, nearby lambda -> the same interaction set at a nearby temperature", f"max |change of ell| for a 0.1% change of lambda: {d:.1e}")
    R.add(cfg["POOL_BOTH_GOALS"] is False or (ell.ndim == 3 and ell.shape[1] == ell.shape[2]), "POOL_BOTH_GOALS requires square tables (C = 2)")
    if "FAIL" in R.rows: print("Fix the adapter (see the contract in the docstring of make_env) before going on."); return R.close()
    if cfg["POOL_BOTH_GOALS"]:                                       # predictions pool both goal assignments; simulations use the table as returned
        rng = np.random.default_rng(1); two = []
        for _ in range(400):
            e = checked_table(make_env(rng, lam0), lam0); a, b = one_step_table(e), one_step_table(e.transpose(0, 2, 1))
            two.append([a["Delta"].mean(), b["Delta"].mean(), a["trapped"].mean(), b["trapped"].mean()])
        two = np.array(two); dif = two[:, 0] - two[:, 1]; z = dif.mean() / max(dif.std() / np.sqrt(len(dif)), 1e-12)
        R.add(abs(z) < 4, "POOL_BOTH_GOALS: the two latent variables are exchangeable (same statistics with either one as the goal)",
              f"E[Delta] {two[:, 0].mean():.4f} vs {two[:, 1].mean():.4f} (z = {z:+.1f}), q {two[:, 2].mean():.3f} vs {two[:, 3].mean():.3f}; if they differ, set CFG['POOL_BOTH_GOALS'] = False")
        if "FAIL" in R.rows: return R.close()
    calls = cfg["NZ_POOL"] * len(cfg["LAMS_POOL"]) + cfg["NZ_FULL"] * len(cfg["LAMS_FULL"]) + cfg["NZ_LIGHT"] * (len(cfg["LAMS_SIM"]) - len(cfg["LAMS_FULL"]))
    R.add(ms * calls / 6e4 < 30, "make_env is fast enough", f"{ms:.2f} ms per call; `all` makes {calls:,} calls = {ms * calls / 6e4:.1f} min in make_env alone", warn_only=True)
    print("\none-step statistics over 300 interaction sets (same random states for both columns)\n"
          f"{'lambda':>7} | {'adapter: q':>10} {'E[Delta]':>9} {'sigma':>7} {'SNR':>6} | {'reference: q':>12} {'E[Delta]':>9} {'sigma':>7} {'SNR':>6}")
    same = True
    for lam in cfg["LAMS_FULL"]:
        s = [pool_summary(one_step_pool(lam, 300, np.random.default_rng(99), cfg["POOL_BOTH_GOALS"], env=e)) for e in (make_env, reference_make_env)]
        print(f"{lam:7.2f} | {s[0]['q']:10.3f} {s[0]['EDelta']:9.4f} {s[0]['sigma']:7.3f} {s[0]['EDelta'] / s[0]['sigma']:6.3f} | {s[1]['q']:12.3f} {s[1]['EDelta']:9.4f} {s[1]['sigma']:7.3f} {s[1]['EDelta'] / s[1]['sigma']:6.3f}"
              f"     (s.e. of E[Delta]: {s[0]['EDelta_se']:.4f})")
        same &= abs(s[0]["EDelta"] - s[1]["EDelta"]) < 4 * np.hypot(s[0]["EDelta_se"], s[1]["EDelta_se"]) + 1e-12
    R.add(same, "one-step statistics agree with the reference re-implementation within sampling error",
          "identical tables" if np.array_equal(ell, reference_make_env(np.random.default_rng(0), lam0)) else "if this is a WARN the numbers in the tex will change: expected, report by how much", warn_only=True)
    return R.close()


def fixture(cfg, out_dir, n_ep=6, T=30):
    """Write a small file for checking the project's own observers against fr_episode(): one interaction set from make_env(),
    n_ep joint realizations, T observations each, and the Factorization Regret and beliefs computed here."""
    rng = np.random.default_rng(2024); ell = np.asarray(make_env(rng, cfg["LAM_MAIN"]), float); d_o, R, Rc = ell.shape
    rg, rc = rng.integers(0, R, n_ep), rng.integers(0, Rc, n_ep)
    obs = (rng.random((n_ep, T, d_o)) < ell[:, rg, rc].T[:, None, :]).astype(np.int8)
    ref = [fr_episode(ell, obs[e]) for e in range(n_ep)]
    path = os.path.join(out_dir, "fr_theory_observer_fixture.npz")
    np.savez(path, ell=ell, r_g=rg, r_c=rc, obs=obs, FR=np.array([r["FR"] for r in ref]), joint=np.array([r["joint"] for r in ref]), naive=np.array([r["naive"] for r in ref]))
    print(f"written {path}\n  ell[i, r_g, r_c] {ell.shape}: P(o^i = 1 | r_g, r_c), lambda = {cfg['LAM_MAIN']:g}   r_g, r_c [{n_ep}]: true joint realizations (0-based)\n"
          f"  obs[e, t, i] {obs.shape}: observations   FR[e, t]: KL(B_t || Btilde_t) in nats after t + 1 observations\n"
          f"  joint[e, t, a] = B_t(a), naive[e, t, a] = Btilde_t(a): beliefs over the goal realization\n"
          "Feed ell and obs to the project's own observers and compare FR, joint and naive (agreement should be ~1e-6).")
    for e in range(n_ep): print(f"  episode {e}: r = ({rg[e]}, {rc[e]}),  FR after 1, 10, {T} observations: {ref[e]['FR'][0]:.6f}, {ref[e]['FR'][9]:.6f}, {ref[e]['FR'][-1]:.6f}")
    return True


def check(cfg, summary, D):
    """Theory versus simulation.  WARN means: look at it and report it; it is not a licence to tune anything."""
    R = Report("check (theory versus simulation in the Cognitive Gridworld)"); T = cfg["T"]
    for lam in cfg["LAMS_FULL"]:
        c = D["curve"][lam]
        R.add(c["ident_z"] < 3, f"lambda = {lam:g}: E[FR_t] = t E[Delta] + N_t - Lambda_t holds in the simulation (Prop. 1c)", f"max deviation = {c['ident_z']:.2f} s.e. of E[FR_T]", warn_only=True)
        R.add(c["neg_steps"] == 0, f"lambda = {lam:g}: the simulated expected curve never decreases by more than 2 s.e.", f"{c['neg_steps']} such steps", warn_only=True)
        R.add(abs(c["off"][-1]) < 1.0, f"lambda = {lam:g}: the offset E[FR_t] - t E[Delta] is a fraction of a nat", f"{c['off'][-1]:.3f} +- {2 * c['off_se'][-1]:.3f} at t = {T}", warn_only=True)
        R.add(c["qstar_sim"] < c["q_sim"] / 20, f"lambda = {lam:g}: far inside the monotone regime (q >> q*)", f"q = {c['q_sim']:.3f}, q* from simulated components = {c['qstar_sim']:.4f}", warn_only=True)
        R.add(c["min_frac"] > 0, f"lambda = {lam:g}: every simulated interaction set has trapped joint realizations", f"smallest fraction {100 * c['min_frac']:.0f}%", warn_only=True)
        R.add(abs(c["q_z"]) < 4, f"lambda = {lam:g}: predictions and simulations are about the same population (share of trapped joint realizations)",
              f"simulated {c['q_sim']:.3f}, pooled {summary[lam]['q']:.3f} (z = {c['q_z']:+.1f}); if they differ, see CFG['POOL_BOTH_GOALS']", warn_only=True)
    for lam in cfg["LAMS_SIM"]:
        m, se = D["slope"][lam]; s = summary[lam]; z = (m - s["EDelta"]) / np.hypot(se, s["EDelta_se"])
        R.add(abs(z) < 3, f"lambda = {lam:g}: simulated slope = E[Delta] (Prop. 1c)", f"{m:.4f} +- {2 * se:.4f} vs {s['EDelta']:.4f} +- {2 * s['EDelta_se']:.4f}  (z = {z:+.1f}; at weak lambda the transient may not be over)", warn_only=True)
    st = D["step"]
    R.add(abs(st["p_pred"] - st["p_sim"]) < 0.02, f"lambda = {cfg['LAM_MAIN']:g}: probability that an episode at a trapped joint realization decreases (Prop. 2)", f"predicted {st['p_pred']:.3f}, simulated {st['p_sim']:.3f}", warn_only=True)
    tv = 0.5 * np.sum(np.abs(st["dens"] - st["sim"]) * np.diff(st["edges"]))
    R.add(tv < 0.08, f"lambda = {cfg['LAM_MAIN']:g}: binned law of the one-step change, predicted vs simulated", f"total variation {tv:.3f} (excess of small simulated steps is expected: nearly tied traps, {100 * st['tied_share']:.0f}% of trapped r have Delta < 0.01)", warn_only=True)
    for lam in cfg["LAMS_FULL"]:
        d = D["dip"][lam]
        R.add(d["sim_maxdiff"] < 0.05, f"lambda = {lam:g}: dip probability of N-episode averages at t = {T - 1}, N >= 5 (Prop. 2b)", f"max |simulated - law| = {d['sim_maxdiff']:.3f}", warn_only=True)
    R.add(not D["bins_auto"] and min(d["n_episodes"] for d in D["dip_frac"]) >= 100 * cfg["N_EP"], "panel h: every band of trapped fraction holds at least 100 simulated interaction sets",
          "; ".join(f"{100 * lo:.0f}-{100 * hi:.0f}%: {d['n_episodes'] // cfg['N_EP']}" for (lo, hi), d in zip(D["bins"], D["dip_frac"])) + ("  (automatic bands: update the tex)" if D["bins_auto"] else ""), warn_only=True)
    qs = [summary[lam]["q"] for lam in cfg["LAMS_POOL"]]; snr = [summary[lam]["EDelta"] / summary[lam]["sigma"] for lam in cfg["LAMS_FULL"]]
    R.add(0.3 < min(qs) and max(qs) < 0.7, "about half of the joint realizations are trapped at every lambda", f"q between {min(qs):.3f} and {max(qs):.3f}", warn_only=True)
    R.add(all(np.diff(snr) > 0), "the signal-to-noise ratio E[Delta] / sigma grows with lambda", ", ".join(f"{x:.3f}" for x in snr), warn_only=True)
    return R.close()


# Values of the run that produced the numbers quoted in FR_theory_block_v11.tex (reference re-implementation of the environment;
# `python fr_theory.py all` with make_env = reference_make_env reproduces them exactly).  That run gives a single WARN in `check`:
# the simulated slope at lambda = 0.5 (0.0009 nats/step) exceeds E[Delta] (0.0007) because the transient is not over at t = 250.
REFERENCE = dict(snr={"1.0": 0.061135, "2.0": 0.170268, "3.0": 0.242297}, episodes_needed={"1.0": 3423.16, "2.0": 441.30, "3.0": 217.92},
                 p_decrease_pred=0.385592, p_decrease_sim=0.396149, gauss_maxdiff=0.028875, q_min=0.474583, q_max=0.553858, min_frac=0.05, n_sets_full=6000,
                 qstar_sim={"1.0": 0.0, "2.0": 0.0043815, "3.0": 0.0037269}, offset_T={"1.0": 0.314885, "2.0": 0.433615, "3.0": 0.333775},
                 Lambda_T={"1.0": 0.603689, "2.0": 0.042808, "3.0": 0.050844}, atom_bound=0.040043,
                 frac_bins=[[0.25, 0.35], [0.45, 0.55], [0.6, 0.7]], slope_weakest=[0.5, 0.00090741, 0.00069164])


def numbers(cfg, summary, D, out_dir, tag=""):
    """The environment-dependent values behind the qualitative Gridworld statements of FR_theory_block_v12.tex (which quotes no
    numbers), each with the statement it supports, against the reference run (the values v11 quoted)."""
    N, Rf = D["numbers"], REFERENCE; L = [str(l) for l in cfg["LAMS_FULL"]]; fm = lambda d, f: ", ".join(format(d[k], f) for k in L if k in d)
    rows = [
        ("SNR at lambda = %s" % ", ".join(f"{l:g}" for l in cfg["LAMS_FULL"]), "not in v12 (v11 Significance; panel i dropped)", fm(Rf["snr"], ".3f"), fm(N["snr"], ".3f")),
        ("episodes for T = %d, alpha = %g" % (cfg["PAPER_HORIZON"], cfg["ALPHA"]), "not in v12 (v11 Significance)", fm(Rf["episodes_needed"], ".0f"), fm(N["episodes_needed"], ".0f")),
        ("P(X < 0 | trapped), lambda = %g" % cfg["LAM_MAIN"], "Significance: 'an episode decreases at a step with probability P(X < 0)' (panel g)", f"{Rf['p_decrease_pred']:.3f} pred / {Rf['p_decrease_sim']:.3f} sim", f"{N['p_decrease_pred']:.3f} pred / {N['p_decrease_sim']:.3f} sim"),
        ("max |law - Gaussian limit|, N >= 5", "Significance: 'whose large-N limit is Phi(-sqrt(N) SNR)' (the law stays this close to it)", f"{Rf['gauss_maxdiff']:.3f}", f"{N['gauss_maxdiff']:.3f}"),
        ("q over lambda in [%g, %g]" % (cfg["LAMS_POOL"][0], cfg["LAMS_POOL"][-1]), "What is and is not guaranteed: 'about half of the joint realizations are trapped at every likelihood temperature'", f"{Rf['q_min']:.3f} - {Rf['q_max']:.3f}", f"{N['q_min']:.3f} - {N['q_max']:.3f}"),
        ("smallest trapped fraction of a set", "same sentence: 'almost every simulated interaction set has some' (11-10: 1 trap-free set of 2,000 at lambda = 1)", f"{100 * Rf['min_frac']:.0f}% of {Rf['n_sets_full']}", f"{100 * N['min_frac']:.0f}% of {N['n_sets_full']}"),
        ("q* from simulated components", "same paragraph: 'a threshold q* far below that'", fm(Rf["qstar_sim"], ".4f"), fm(N["qstar_sim"], ".4f")),
        ("slope at the weakest simulated lambda", "not in v12 (v11 Details)",
         f"lambda = {Rf['slope_weakest'][0]:g}: {Rf['slope_weakest'][1]:.4f} simulated / {Rf['slope_weakest'][2]:.4f} predicted", f"lambda = {N['slope_weakest'][0]:g}: {N['slope_weakest'][1]:.4f} simulated / {N['slope_weakest'][2]:.4f} predicted"),
        ("offset E[FR_T] - T E[Delta]", "Expectation over interaction sets: 'stays a fraction of a nat (inset of e)'", fm(Rf["offset_T"], ".2f"), fm(N["offset_T"], ".2f")),
        ("Lambda_T (Joint log loss at t = T)", "not in v12 (v11 Details)", fm(Rf["Lambda_T"], ".2f"), fm(N["Lambda_T"], ".2f")),
        ("(1 - q)^5", "Details: 'the two bounds differ by (1 - q)^N'", f"{Rf['atom_bound']:.3f}", f"{N['atom_bound']:.3f}"),
        ("bands of trapped fraction (panel h)", "Details / legend of h (construction)", "; ".join(f"{100 * a:.0f}-{100 * b:.0f}%" for a, b in Rf["frac_bins"]), "; ".join(f"{100 * a:.0f}-{100 * b:.0f}%" for a, b in N["frac_bins"])),
        ("simulation sizes", "Details (construction)", "2000 x 50 and 500 x 50 episodes, T = 250, pool 2000",
         f"{cfg['NZ_FULL']} x {cfg['N_EP']} and {cfg['NZ_LIGHT']} x {cfg['N_EP']} episodes, T = {cfg['T']}, pool {cfg['NZ_POOL']}, {cfg['NZ_BAND']} sets per band (h)"),
    ]
    print("\n=== values behind the panels as drawn (11-11) ===")
    O = D["opt"]; Rs = sorted({k[0] for k in O}); ds = sorted({k[1] for k in O})
    for R_ in Rs:
        ed = np.array([O[(R_, d)]["ed"][0] for d in ds]); sl = np.array([O[(R_, d)]["slope"][0] for d in ds]); se = np.array([np.hypot(O[(R_, d)]["ed"][1], O[(R_, d)]["slope"][1]) for d in ds])
        qq = np.array([O[(R_, d)]["q"][0] for d in ds])
        print(f"panel f, R = {R_}: E[Delta] largest at d_o = {ds[int(ed.argmax())]} ({ed.max():.4f}); at d_o = {ds[0]} / {ds[-1]}: {ed[0]:.4f} / {ed[-1]:.4f}; q {qq[0]:.2f} -> {qq[-1]:.2f}; "
              f"mean Delta of the trapped {ed[0] / qq[0]:.3f} -> {ed[-1] / qq[-1]:.3f}; largest |simulated slope - E[Delta]| {np.abs(sl - ed).max():.4f}")
    ST = D["pop_step"]; print(f"panel g: a trapped episode decreases at a step with probability {ST['p_pred']:.3f} (law of X) / {ST['p_sim']:.3f} (simulation, {ST['n']:,} trapped episodes)")
    for lam in cfg["LAMS_FULL"]:
        SP = D["spread"][lam]; th = np.interp(SP["tm"], SP["t"], SP["th"]); print(f"panel h, lambda = {lam:g}: simulated s.d. / theory at t = {SP['tm'][0]}: {SP['sd_m'][0] / th[0]:.3f}; at t = {SP['tm'][-1]}: {SP['sd_m'][-1] / th[-1]:.3f}")
    for FN, MO, b in zip(D["fan3"], D["mono3"], D["third_bins"]):
        th = np.array([np.interp(np.log(FN["Ns"]), np.log(FN["Nfine"]), FN["theory"][:, k]) for k in range(3)]).T; k30 = list(cfg["TH_TS"]).index(max(cfg["TH_TS"]))
        thh = np.interp(np.log(MO["Ns"]), np.log(MO["Nth"]), MO["theory"][:, k30]); n0 = FN["Nfine"][np.argmax(FN["theory"][:, 0] > 0)] if (FN["theory"][:, 0] > 0).any() else None
        print(f"Figure 2, third {100 * b[0]:.0f}-{min(100, 100 * b[1]):.0f}%: q = {MO['q']:.3f}, E[Delta] = {FN['mean']:.4f}; largest |simulated - theory| quantile of the N-average {np.abs(FN['sim'] - th).max():.4f} nats; "
              f"5% quantile above zero from N = {n0}; P(dip within {max(cfg['TH_TS'])} steps) at N = {list(MO['Ns'])}: theory {np.round(thh, 3)}, simulation {np.round(MO['sim'][:, k30], 3)}")
    PH = D["pop_h"]
    for j, T_ in enumerate(PH["Ts"]):
        thh = np.interp(np.log(PH["Ns"]), np.log(PH["Nth"]), PH["theory"][:, j]); print(f"Figure 2 d, T = {T_}: theory {np.round(thh, 3)}, simulation {np.round(PH['sim'][:, j], 3)} at N = {list(PH['Ns'])}")
    print("\n=== environment-dependent values behind the qualitative statements of the tex ===")
    for q, where, ref, new in rows: print(f"{q}\n      reference run : {ref}\n      this run      : {new}{'' if new == ref else '      <-- differs'}\n      supports      : {where}")
    n_diff = sum(new != ref for _, _, ref, new in rows)
    print(f"{n_diff} of {len(rows)} rows differ from the reference run (the values v11 quoted) at the precision shown; v12 quotes none of them.")
    print("Values of panels a-d and of the 6 x 24 table do not depend on the environment (selftest checks them).")
    path = os.path.join(out_dir, f"fr_theory_numbers{tag}.json")
    with open(path, "w") as fh: json.dump(dict(this_run=N, reference=Rf, rows=[dict(quantity=q, where=w, reference=r, this_run=n) for q, w, r, n in rows]), fh, indent=1)
    print(f"written {path}")


# ============================================================================================================
# 7. FIGURE
# ============================================================================================================
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#3b3a37", "#8a8983", "#e4e3dc", "#b5b4aa"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"                # trap term, distraction term, Joint term
# One-hue ordinal ramps, light -> dark, on the hues above, evenly stepped in OKLab lightness (0.72 -> 0.33): the light end keeps
# a 2.4:1 contrast on white paper, neighbouring steps are equally far apart.
BLUES5 = ["#63a6ff", "#4487df", "#2469be", "#014c9c", "#00336d"]    # prior weight q of trapped r (e)
BLUES3 = ["#63a6ff", "#2469be", "#00336d"]                          # trapped fraction of an interaction set (g, h)
GRAYS3 = ["#9a9992", "#5f5e58", "#0b0b0b"]                          # realizations per variable R (f): neutral, since blue means q (d; Figure 2)
GREENS4 = ["#4fb886", "#2a9566", "#11704a", "#044a30"]             # horizon T (Figure 2 d), on the Joint term's hue
ORANGES4 = ["#ff8350", "#d55c27", "#a63d01", "#722700"]             # number of distractors (c)
ORANGES3 = ["#ff8350", "#c04909", "#722700"]                        # drift of the weakest distractor (d)
VIOLETS3 = ["#8e85ff", "#5c4bc4", "#310888"]; VIOLET = VIOLETS3[1]  # likelihood temperature lambda (f-i)
LW, MS = 1.2, 3.4                                                   # points: analytical lines, simulation markers
FIGSIZE = (9.0, 3.94)       # inches = the print size: the tex sets the figure 9 in wide on a rotated page, so nothing is scaled; with its
                            # caption it must stay within 5.5 in of height (3.94 in is the height the 13.6 x 5.95 in version printed at)
ROW1_MARKERS = False                                   # John 11-10: no Monte Carlo dots in the top row (exact curves alone); True draws them
FS = dict(header=9, title=8, label=8, tick=7, key=7)                # points, at print size; nothing smaller than 7 pt
RC = {
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,   # the paper's figures
    "font.size": FS["label"], "axes.titlesize": FS["title"], "axes.labelsize": FS["label"], "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"],
    "legend.fontsize": FS["key"], "legend.title_fontsize": FS["key"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "axes.labelpad": 2.0, "axes.titlepad": 3.0, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.minor.size": 1.3, "ytick.minor.size": 1.3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45, "ytick.minor.width": 0.45, "xtick.major.pad": 1.8, "ytick.major.pad": 1.8,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.45, "axes.axisbelow": True,
    "legend.frameon": False, "legend.handlelength": 1.5, "legend.handletextpad": 0.45, "legend.borderaxespad": 0.25, "legend.borderpad": 0.2,
    "legend.labelspacing": 0.22, "legend.columnspacing": 0.9, "pdf.fonttype": 42, "savefig.facecolor": "white"}


def make_figure(cfg, row1, summary, D, out_dir, tag=""):
    import matplotlib
    if tuple(int(v) for v in matplotlib.__version__.split(".")[:2]) < (3, 6): raise SystemExit("make_figure needs matplotlib >= 3.6")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    from matplotlib.text import Text
    from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter
    import logging; logging.getLogger("fontTools").setLevel(logging.ERROR)                 # its notes on the fonts embedded in the PDF
    plt.rcParams.update(RC)
    assert len(cfg["LAMS_FULL"]) == 3, "panel e is designed for exactly three likelihood temperatures"
    T = cfg["T"]; lam0 = cfg["LAM_MAIN"]; LF = cfg["LAMS_FULL"]

    def theory(ax, x, y, color, lw=LW, **kw):
        """Analytical curve: always a line."""
        return ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=3, **kw)

    def simulated(ax, x, y, yerr, color, filled=True, ms=MS, clip=False, **kw):
        """Simulation: always a marker (with a +-2 s.e. bar; the white ring keeps it legible on its own theory line).  Markers are not
        clipped by the axes, so one that sits on an axis or at the last step is drawn whole; pass clip=True where data leave the panel."""
        if yerr is not None: ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.7, capsize=0, zorder=4)
        return ax.plot(x, y, ls="none", marker="o", ms=ms, mfc=color if filled else "white", mec="white" if filled else color,
                       mew=0.5 if filled else 0.8, zorder=5, clip_on=clip, **kw)

    def h_both(color, label): return Line2D([], [], color=color, lw=LW, marker="o", ms=MS, mfc=color, mec="white", mew=0.5, label=label)
    def h_line(color, label, **kw): return Line2D([], [], color=color, lw=kw.pop("lw", LW), label=label, **kw)
    def h_dot(color, label, filled=True):
        return Line2D([], [], ls="none", marker="o", ms=MS, mfc=color if filled else "white", mec="white" if filled else color, mew=0.5 if filled else 0.8, label=label)
    def title(ax, letter, text): ax.set_title(f"({letter}) {text}", loc="left")
    def note(ax, x, y, s, **kw): kw.setdefault("color", INK); kw.setdefault("fontsize", FS["key"]); return ax.text(x, y, s, **kw)
    def pow10(n): e = int(np.floor(np.log10(n))); m = n / 10 ** e; return (rf"$10^{e}$" if abs(m - 1) < 1e-9 else rf"${m:g}\times10^{e}$")
    def decimals(ax, d, axis="y"):                     # tick labels with d decimals, in the math font (true minus signs)
        (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(FuncFormatter(lambda v, _: f"${v:.{d}f}$"))

    # ---------------------------------------------------------------- layout, in inches: the figure is drawn at its print size
    W, H = FIGSIZE; NP = 4                   # panels per row
    LM, GX, RM = 0.43, 0.55, 0.05            # left margin, gap between panels (holds the next panel's y labels), right margin
    TOP, BOT1, BOT2, GAP = 0.35, 0.31, 0.39, 0.05      # above a row's axes (row header, panel title), below them (tick labels, axis label), between rows
    pw, ph = (W - LM - (NP - 1) * GX - RM) / NP, (H - 2 * TOP - BOT1 - BOT2 - GAP) / 2
    y_row = (BOT2 + ph + TOP + GAP + BOT1, BOT2)       # bottom edge of the axes of each row
    fig = plt.figure(figsize=FIGSIZE)
    ax1, ax2 = ([fig.add_axes([(LM + j * (pw + GX)) / W, y / H, pw / W, ph / H]) for j in range(NP)] for y in y_row)
    R = fig.canvas.get_renderer()
    sim1, h1 = (simulated, h_both) if ROW1_MARKERS else ((lambda *a_, **k_: None), h_line)      # the top row's markers and its legend handles
    for y, name in ((y_row[0] + ph + TOP - 0.135, "Toy example"), (y_row[1] + ph + TOP - 0.135, "Cognitive Gridworld")):
        fig.text(0.02 / W, y / H, name, fontfamily="cmb10", fontsize=FS["header"], va="baseline")
    key = fig.legend(handles=[Line2D([], [], color=INK, lw=LW, label="Theory"), h_dot(INK, r"Simulation (mean $\pm$ 2 SEM)")], loc="upper right",
                     bbox_to_anchor=(1 - RM / W, (y_row[1] + ph + TOP + 0.02) / H), ncol=2, fontsize=FS["label"], handlelength=1.8, columnspacing=1.4,
                     borderpad=0.1, frameon=False)

    # ---------------------------------------------------------------- a, b: anatomy of E[FR_t]   (environment-independent)
    T1 = T_ANATOMY; t1 = np.arange(T1 + 1); tm1 = np.arange(30, T1 + 1, 30)
    pastel = lambda c: tuple(0.30 * np.array(to_rgb(c)) + 0.70)            # the bands, opaque: they meet without a seam
    for ax, k in ((ax1[0], "a"), (ax1[1], "b")):
        an, st = row1[k]["an"], row1[k]["st"]; mean, se = row1[k]["mc"]
        base = -an["Lam"]; mid = base + an["D"]
        ax.axhline(0, color=AXIS, lw=0.6, zorder=1)
        ax.fill_between(t1, base, mid, color=pastel(ORANGE), lw=0, zorder=2)
        if st["k"]: ax.fill_between(t1, mid, an["FR"], color=pastel(BLUE), lw=0, zorder=2)
        theory(ax, t1, base, AQUA); theory(ax, t1, an["FR"], INK, lw=LW * 1.15)
        sim1(ax, tm1, mean[tm1], 2 * se[tm1], INK)
        ax.set_xlabel("step $t$"); ax.set_ylabel("nats"); ax.set_xlim(0, T1); ax.set_xticks([0, 100, 200, 300]); decimals(ax, 1)
    a = ax1[0]; an = row1["a"]["an"]; mean, se = row1["a"]["mc"]
    title(a, "a", "$k=0:$ no trap"); a.set_ylim(-1.15, 0.95); a.set_yticks([-1.0, -0.5, 0.0, 0.5])
    a.legend(handles=[h1(INK, r"$\mathbb{E}[\mathrm{FR}_t]$ = sum of"), h_line(AQUA, r"Joint term $-\Lambda_t$"),
                      Patch(color=pastel(ORANGE), lw=0, label=r"distraction $\mathbb{E}[\mathcal{D}_t]$")], loc="upper right")
    # inset: E[FR_t] magnified, a square in the empty lower right, under the Joint term (side SQ inches)
    SQ = 0.50; ins_a = a.inset_axes([1 - SQ / pw - 0.03, 0.045, SQ / pw, SQ / ph]); ins_a.grid(False)
    theory(ins_a, t1, an["FR"], INK); sim1(ins_a, tm1, mean[tm1], 2 * se[tm1], INK)
    ins_a.set_xlim(0, T1); ins_a.set_ylim(0, 0.065); ins_a.set_xticks([0, 100, 200, 300]); ins_a.set_xticklabels([]); ins_a.set_yticks([0, 0.05])
    ins_a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:.2f}".lstrip("0")))
    ins_a.tick_params(length=1.6, pad=1.0, labelsize=FS["tick"])
    for sp in ins_a.spines.values(): sp.set_visible(True); sp.set_linewidth(0.5)
    b = ax1[1]
    title(b, "b", r"$k=1:$ one trap"); b.set_ylim(-1.25, 2.25); b.set_yticks([-1, 0, 1, 2])
    b.legend(handles=[Patch(color=pastel(BLUE), lw=0, label=r"trap term $\mathbb{E}[\mathcal{T}_t]$")], loc="upper left")

    # ---------------------------------------------------------------- c: no trap, the drift of the weakest distractor sets the time scale   (environment-independent)
    T2 = T_DISTRACTORS; t2 = np.arange(T2 + 1); c = ax1[2]
    for i, (e, col) in enumerate(zip(row1["d"], ORANGES3)):
        tm = np.arange(30 + 20 * i, T2 + 1, 70); theory(c, t2, e["efr"][:T2 + 1], col); sim1(c, tm, e["mc"][0][tm], 2 * e["mc"][1][tm], col)
    c.legend(handles=[h1(col, f"{e['st']['delta']:.2f}") for e, col in zip(row1["d"], ORANGES3)], title=r"Pull $\delta$ of the weakest distractor",
             loc="upper right", ncol=3, handlelength=1.1, columnspacing=0.7, alignment="left")
    title(c, "c", r"k = 0: time scale $\propto 1/\delta$")
    c.set_ylim(0, 0.1); c.set_yticks([0, 0.02, 0.04, 0.06, 0.08]); decimals(c, 2); c.set_xlim(0, T2); c.set_xticks([0, 100, 200, 300, 400])
    c.set_xlabel("step $t$"); c.set_ylabel(r"$\mathbb{E}[\mathrm{FR}_t]$ (nats)")

    # ---------------------------------------------------------------- d: rare traps -> transient dip, threshold q*   (environment-independent)
    d_ = ax1[3]; E0, E1 = row1["a"]["efr"], row1["b"]["efr"]; (m0, s0), (m1, s1) = row1["a"]["mc"], row1["b"]["mc"]
    share = row1["b"]["st"]["k"] / row1["b"]["ell"].size                       # trapped share of table b: an episode is trapped with probability q = w * share
    w_star, _ = mixture_threshold(E0, E1); q_star = w_star * share
    te = np.arange(len(E0)); tme = np.array([10, 18, 32, 56, 100, 180, 320, 560, 1000, 1800])
    QS = [0.0, 0.001, 0.002, q_star, 0.01]; DCOL = [BLUES5[0], BLUES5[1], BLUES5[2], INK, BLUES5[4]]
    for q, col in zip(QS, DCOL):
        w = q / share; m = (1 - w) * E0 + w * E1
        theory(d_, te[1:], m[1:], col, **(dict(ls=(0, (3, 1.6)), lw=LW * 1.1) if q == q_star else {}))       # q*: the threshold, dashed
        sim1(d_, tme, (1 - w) * m0[tme] + w * m1[tme], 2 * np.hypot((1 - w) * s0[tme], w * s1[tme]), col, clip=True)   # these curves leave the panel
    d_.set_xscale("log"); d_.set_xlim(1, T_MIXTURE); d_.set_ylim(-0.006, 0.28); d_.set_yticks([0, 0.1, 0.2]); decimals(d_, 1)
    d_.xaxis.set_major_locator(LogLocator(numticks=5)); d_.xaxis.set_minor_formatter(NullFormatter())
    d_.axhline(0, color=AXIS, lw=0.6, zorder=1)
    L5 = [h1(col, "") for col in DCOL]; L5[3] = Line2D([], [], color=INK, lw=LW * 1.1, ls=(0, (3, 1.6)))
    d_.legend(handles=[L5[0], (L5[1], L5[2]), L5[3], L5[4]],
              labels=[r"$q=0$: returns to 0", r"$q<q^*$: dips, then rises", r"$q=q^*$", r"$q>q^*$: rises from $t=0$"],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)}, loc="upper left",
              alignment="left", handlelength=1.6, borderaxespad=0.15)
    title(d_, "d", "Monotonically vs. eventually monotonic"); d_.set_xlabel("step $t$"); d_.set_ylabel(r"$\mathbb{E}[\mathrm{FR}_t]$ (nats)")

    # ---------------------------------------------------------------- e: expected curve and its asymptote
    # Every curve lies on or below the steepest line, so the legend and the inset sit in the triangle above it, whatever the slopes.
    tt = D["tt"]; tmf = np.arange(T // 10, T + 1, T // 10) - 1; pts = {k: [] for k in "efgh"}       # pts: plotted points, for the last block
    e2, f2, g2, h2 = ax2; top = 0.0
    for lam, col in zip(LF, VIOLETS3):
        cv = D["curve"][lam]; line = np.r_[0, tt] * summary[lam]["EDelta"]
        theory(e2, np.r_[0, tt], line, col); simulated(e2, tt[tmf], cv["mean"][tmf], 2 * cv["se"][tmf], col)
        pts["e"] += [np.c_[np.r_[0, tt], line], np.c_[tt[tmf], cv["mean"][tmf] + 2 * cv["se"][tmf]]]
        top = max(top, cv["mean"][-1] + 2 * cv["se"][-1], T * summary[lam]["EDelta"])
    lam_labels = [rf"$\lambda={LF[0]:g}$"] + [rf"${lam:g}$" for lam in LF[1:]]
    leg_e = e2.legend(handles=[h_both(col, s) for s, col in zip(lam_labels, VIOLETS3)], loc="upper left", ncol=3, columnspacing=0.8, handlelength=1.3)
    title(e2, "e", r"$\mathbb{E}[\mathrm{FR}_t]$ across $\lambda$"); e2.set_xlabel("step $t$"); e2.set_ylabel(r"$\mathbb{E}[\mathrm{FR}_t]$ (nats)")
    e2.set_xlim(0, T); e2.set_ylim(0, 1.07 * top); e2.set_xticks(np.arange(0, T + 1, 100))

    # ---------------------------------------------------------------- f: against the number of observables, for R = 3, 5, 10 realizations
    xs = np.array(cfg["OPT_DOS"]); OP = D["opt"]; mode = cfg["F_MODE"]
    for R_, col in zip(cfg["OPT_RS"], GRAYS3):
        if mode == "rate":                                                          # growth rate: E[Delta] (theory) vs late slopes (simulation)
            theory(f2, xs, [OP[(R_, d)]["ed"][0] for d in xs], col)
            simulated(f2, xs, [OP[(R_, d)]["slope"][0] for d in xs], [2 * OP[(R_, d)]["slope"][1] for d in xs], col)
        else:                                                                       # accuracy gap: q (theory, t -> infinity) vs simulated gaps
            theory(f2, xs, [OP[(R_, d)]["q"][0] for d in xs], col)
            for i_, filled in ((len(cfg["OPT_TACC"]) - 1, True), (0, False)):
                simulated(f2, xs, [OP[(R_, d)]["gap"][i_][0] for d in xs], [2 * OP[(R_, d)]["gap"][i_][1] for d in xs], col, filled=filled)
    f2.axvline(5, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)                       # the Gridworld's five observables
    f2.set_xscale("log"); f2.set_xticks([1, 2, 5, 10, 30]); f2.set_xticklabels(["1", "2", "5", "10", "30"]); f2.set_xlim(0.9, 33)
    f2.xaxis.set_minor_formatter(NullFormatter())
    top_ = max(max(OP[k]["ed"][0], OP[k]["slope"][0]) for k in OP) if mode == "rate" else max(OP[k]["q"][0] for k in OP)
    f2.set_ylim(0, 1.32 * top_)                                                       # headroom: the key on top
    hnd = [h_both(col, rf"$R={R_}$") for R_, col in zip(cfg["OPT_RS"], GRAYS3)]
    tlab = lambda t: rf"$t=10^{{{int(round(np.log10(t)))}}}$" if t >= 10 ** 5 else rf"$t={t:,}$".replace(",", "{,}")
    if mode != "rate":                                                                # column-major: row 1 the R, row 2 the steps
        hnd = [hnd[0], h_dot(INK2, tlab(cfg["OPT_TACC"][-1])), hnd[1], h_dot(INK2, tlab(cfg["OPT_TACC"][0]), filled=False), hnd[2]]
    leg_f = f2.legend(handles=hnd, loc="upper center", ncol=3, columnspacing=0.8, handlelength=1.3, borderaxespad=0.1)
    title(f2, "f", "Growth rate vs. observables" if mode == "rate" else "Accuracy gap vs. observables"); f2.set_xlabel(r"observables $d_o$")
    f2.set_ylabel(r"$\mathbb{E}[\Delta]$ (nats / step)" if mode == "rate" else "Joint $-$ Naive accuracy")

    # ---------------------------------------------------------------- g: one step of a single episode at a trapped joint realization (Prop. 2(a, c))
    ST = D["pop_step"]; ed_ = ST["edges"]; mid_ = 0.5 * (ed_[1:] + ed_[:-1]); col_ = VIOLETS3[LF.index(lam0)]
    g2.stairs(ST["dens"], ed_, fill=True, color=col_, alpha=0.14, lw=0, zorder=2); g2.stairs(ST["dens"], ed_, color=col_, lw=LW, zorder=3)
    simulated(g2, mid_, ST["sim"], ST["sim_2se"], col_, ms=MS * 0.85)
    g2.axvline(0, color=AXIS, lw=0.6, zorder=1); g2.set_xlim(-ST["xmax"], ST["xmax"]); g2.set_ylim(0, 1.25 * max(ST["dens"].max(), ST["sim"].max()))
    note(g2, 0.03, 0.97, f"decrease at a step:\n{100 * ST['p_pred']:.0f}% theory\n{100 * ST['p_sim']:.0f}% simulation", transform=g2.transAxes,
         va="top", ha="left", linespacing=1.15)
    title(g2, "g", r"One step of a trapped episode"); g2.set_xlabel(r"$\mathrm{FR}_{t+1}-\mathrm{FR}_t$ (nats)"); g2.set_ylabel("density")

    # ---------------------------------------------------------------- h: spread of FR_t across episodes (Prop. 2(c)): sqrt(t^2 Var(Delta) + t E[v])
    top_h = 0.0
    for lam, col in zip(LF, VIOLETS3):
        SP = D["spread"][lam]; theory(h2, np.r_[0, SP["t"]], np.r_[0, SP["th"]], col); simulated(h2, SP["tm"], SP["sd_m"], 2 * SP["se_m"], col)
        pts["h"] += [np.c_[np.r_[0, SP["t"]], np.r_[0, SP["th"]]], np.c_[SP["tm"], SP["sd_m"] + 2 * SP["se_m"]]]; top_h = max(top_h, SP["th"].max(), SP["sd_m"].max())
    leg_h = h2.legend(handles=[h_both(col, s_) for s_, col in zip(lam_labels, VIOLETS3)], loc="upper left", ncol=3, columnspacing=0.8, handlelength=1.3)
    h2.set_xlim(0, T); h2.set_ylim(0, 1.07 * top_h); h2.set_xticks(np.arange(0, T + 1, 100))
    title(h2, "h", r"Spread across episodes"); h2.set_xlabel("step $t$"); h2.set_ylabel(r"s.d. of $\mathrm{FR}_t$ (nats)")

    for ax in list(ax1) + list(ax2): ax.tick_params(which="both", width=0.6)

    # ---------------------------------------------------------------- e-h: keep the data clear of legends and the inset, whatever the numbers
    # Those artists are anchored in axes coordinates, whereas the curves depend on the environment.  Once the layout is known, the upper
    # y-limit of a panel is raised (never lowered) just enough for its data to pass `pad` (in axes fractions) below each artist above them.
    def box(ax, art): return art.get_window_extent(R).transformed(ax.transAxes.inverted())                    # extent of an artist, in axes fractions
    def frac(ax, p): return ax.transAxes.inverted().transform(ax.transData.transform(np.vstack(p)))           # plotted points, in axes fractions
    def under(b, P): m = (P[:, 0] > b.x0 - 0.015) & (P[:, 0] < b.x1 + 0.015); return P[m, 1].max() if m.any() else 0.0   # highest point below a box
    def headroom(ax, artists, p, pad=0.03):
        P = frac(ax, p); return ax.get_ylim()[1] * max([1.0] + [under(b, P) / (b.y0 - pad) for b in (box(ax, a) for a in artists) if b.y0 - pad > 0.1])
    for _ in range(2):                                                                                         # second pass: tick labels may have changed the layout
        fig.draw_without_rendering(); R = fig.canvas.get_renderer()
        e2.set_ylim(0, headroom(e2, [leg_e], pts["e"])); h2.set_ylim(0, headroom(h2, [leg_h], pts["h"]))

    finish_figure(fig, (ax1, ax2), out_dir, "fig_fr_theory", tag)


def finish_figure(fig, rows, out_dir, name, tag):
    """Report text that overlaps other text or leaves the figure; save PDF + PNG (and a copy in figure_bin unless quick) and
    300-dpi crops for visual inspection."""
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    fig.draw_without_rendering(); R = fig.canvas.get_renderer(); boxes = []; pooled = set(); drawn = []
    for ax in fig.findobj(lambda o: hasattr(o, "xaxis") and hasattr(o, "yaxis")):
        for axis in (ax.xaxis, ax.yaxis):
            pooled |= {id(lb) for tk in axis.majorTicks + axis.minorTicks for lb in (tk.label1, tk.label2)}
            lo_, hi_ = sorted(axis.get_view_interval())
            for locs, ticks in ((axis.get_majorticklocs(), axis.get_major_ticks), (axis.get_minorticklocs(), axis.get_minor_ticks)):
                drawn += [tk.label1 for tk, loc in zip(ticks(len(locs)), locs) if lo_ - 1e-9 <= loc <= hi_ + 1e-9 and tk.label1.get_visible()]
    for t in [t for t in fig.findobj(Text) if id(t) not in pooled] + drawn:
        if not t.get_visible() or not t.get_text().strip() or t.figure is None: continue
        bb = t.get_window_extent(R)
        if bb.width > 0: boxes.append((t.get_text(), bb))
    figbb = fig.bbox
    for s, bb in boxes:
        if bb.x0 < figbb.x0 - 0.5 or bb.y0 < figbb.y0 - 0.5 or bb.x1 > figbb.x1 + 0.5 or bb.y1 > figbb.y1 + 0.5: print(f"   figure check: text outside the figure: {s!r}")
    for i1 in range(len(boxes)):
        for i2 in range(i1 + 1, len(boxes)):
            (s1, b1), (s2, b2) = boxes[i1], boxes[i2]
            if b1.overlaps(b2) and min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > 1 and min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > 1:
                print(f"   figure check: texts overlap: {s1!r} / {s2!r}")
    for row in rows:                                                                                           # panel titles side by side
        for l, r in zip(row[:-1], row[1:]):
            gap = (r.title.get_window_extent(R).x0 - l.title.get_window_extent(R).x1) / fig.dpi
            if gap < 0.06: print(f"   figure check: only {gap:.2f} in between the titles {l.get_title(loc='left')!r} and {r.get_title(loc='left')!r}")
    stem = os.path.join(out_dir, f"{name}{tag}")
    fig.savefig(stem + ".pdf"); fig.savefig(stem + ".png", dpi=150)
    if "_quick" not in tag:                           # the full-size figures (and their variants) also go to the repository's figure_bin
        try:
            from main.utils import fig_path
            fig.savefig(fig_path(f"{name}{tag}.png"), dpi=150); fig.savefig(fig_path(f"{name}{tag}.pdf")); print("   also saved in figure_bin")
        except Exception as err:
            print(f"   figure_bin copy skipped ({err})")
    from PIL import Image                             # 300-dpi crops for visual inspection (overlaps, clipped or tiny text)
    insp = os.path.join(out_dir, "fr_theory_inspect"); os.makedirs(insp, exist_ok=True); big = os.path.join(insp, f"full_300dpi_{name}{tag}.png"); fig.savefig(big, dpi=300)
    im = Image.open(big); W_, H_ = im.size; short = "" if name == "fig_fr_theory" else name.replace("fig_fr_", "") + "_"
    for part, bx in (("row1_left", (0, 0, int(0.62 * W_), H_ // 2 + 15)), ("row1_right", (int(0.38 * W_), 0, W_, H_ // 2 + 15)),
                     ("row2_left", (0, H_ // 2 - 15, int(0.62 * W_), H_)), ("row2_right", (int(0.38 * W_), H_ // 2 - 15, W_, H_))):
        im.crop(bx).save(os.path.join(insp, f"{short}{part}{tag}.png"))
    plt.close(fig); print(f"saved {stem}.pdf, {stem}.png and crops in {insp}/")


def make_figure_averages(cfg, D, out_dir, tag=""):
    """Figure 2 (fig_fr_averages): averages over N episodes in the Cognitive Gridworld at LAM_MAIN, from step TH_T0 = POP_T0 on.
    Row 1 (a-c): the one-step change of an N-episode average, one panel per third of the trapped fraction q of the interaction set
    (theory: quantiles of the exact law of X-bar_N; simulation: quantiles over independent groups, bootstrap 95% intervals).
    Row 2: the probability of at least one decrease in the next T steps, (d) for several T in the whole population, (e) by third of
    q at T = 30 (theory: 1 - E_R[(1 - p_R)^T]; simulation: fraction of independent groups, +- 2 s.e.)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator
    import logging; logging.getLogger("fontTools").setLevel(logging.ERROR)
    plt.rcParams.update(RC)

    def theory(ax, x, y, color, lw=LW, **kw): return ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=3, **kw)
    def simulated(ax, x, y, yerr, color, ms=MS, clip=False):
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.7, capsize=0, zorder=4)
        return ax.plot(x, y, ls="none", marker="o", ms=ms, mfc=color, mec="white", mew=0.5, zorder=5, clip_on=clip)
    def h_both(color, label): return Line2D([], [], color=color, lw=LW, marker="o", ms=MS, mfc=color, mec="white", mew=0.5, label=label)
    def h_dot(color, label): return Line2D([], [], ls="none", marker="o", ms=MS, mfc=color, mec="white", mew=0.5, label=label)
    def title(ax, letter, text): ax.set_title(f"({letter}) {text}", loc="left")
    def pct(lo, hi, first): return (r"$q=" if first else "$") + rf"{100 * lo:.0f}\endash{min(100, 100 * hi):.0f}\%$"

    W, H = FIGSIZE; LM, GX, G3, RM = 0.52, 0.55, 0.16, 0.05     # left margin, gap between panels (row 2), between the panels of row 1 (shared y), right
    TOP, BOT1, BOT2, GAP = 0.35, 0.31, 0.39, 0.05
    ph = (H - 2 * TOP - BOT1 - BOT2 - GAP) / 2; y_row = (BOT2 + ph + TOP + GAP + BOT1, BOT2)
    w3, w2 = (W - LM - 2 * G3 - RM) / 3, (W - LM - GX - RM) / 2
    fig = plt.figure(figsize=FIGSIZE)
    top = [fig.add_axes([(LM + j * (w3 + G3)) / W, y_row[0] / H, w3 / W, ph / H]) for j in range(3)]
    bot = [fig.add_axes([(LM + j * (w2 + GX)) / W, y_row[1] / H, w2 / W, ph / H]) for j in range(2)]
    for y, name in ((y_row[0] + ph + TOP - 0.135, r"Change of an average over $N$ episodes, by the trapped fraction $q$ of the interaction set"),
                    (y_row[1] + ph + TOP - 0.135, r"Probability that the average decreases within the next $T$ steps")):
        fig.text(0.02 / W, y / H, name, fontfamily="cmb10", fontsize=FS["header"], va="baseline")
    fig.legend(handles=[Line2D([], [], color=INK, lw=LW, label="Theory"), h_dot(INK, "Simulation")], loc="upper right",
               bbox_to_anchor=(1 - RM / W, (y_row[0] + ph + TOP + 0.02) / H), ncol=2, fontsize=FS["label"], handlelength=1.8, columnspacing=1.4,
               borderpad=0.1, frameon=False)

    # ---------------------------------------------------------------- a-c: one panel per third of q
    zt = lambda v: np.arcsinh(np.asarray(v) / G_ASINH); tv = np.array([-1, -0.3, 0, 0.3, 1])
    for k, (ax, FN, (lo, hi), col) in enumerate(zip(top, D["fan3"], D["third_bins"], BLUES3)):
        ax.fill_between(FN["Nfine"], zt(FN["theory"][:, 0]), zt(FN["theory"][:, 2]), color=col, alpha=0.16, lw=0, zorder=2)
        for j in (0, 2): theory(ax, FN["Nfine"], zt(FN["theory"][:, j]), col, lw=LW * 0.5)
        theory(ax, FN["Nfine"], zt(FN["theory"][:, 1]), col)
        ax.axhline(zt(FN["mean"]), color=INK2, lw=0.8, ls=(0, (1, 1.6)), zorder=2.5)
        for j in range(3):
            yv = zt(FN["sim"][:, j]); simulated(ax, FN["Ns"], yv, np.vstack([yv - zt(FN["lo"][:, j]), zt(FN["hi"][:, j]) - yv]), col, ms=MS * 0.9, clip=True)
        ax.axhline(0, color=AXIS, lw=0.6, zorder=1); ax.set_xscale("log"); ax.set_xlim(0.8, 1300); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks(zt(tv)); ax.set_yticklabels([f"${t:g}$" for t in tv] if k == 0 else []); ax.set_ylim(zt(-1.6), zt(3.5))   # headroom: key of (a)
        ax.yaxis.set_minor_locator(NullLocator()); ax.set_xlabel(r"episodes averaged, $N$"); title(ax, "abc"[k], pct(lo, hi, True))
    top[0].set_ylabel(r"$\widehat{\mathrm{FR}}_{t+1}-\widehat{\mathrm{FR}}_t$ (nats)")
    top[0].legend(handles=[Line2D([], [], color=INK2, lw=LW, label="median"), Patch(facecolor=INK2, alpha=0.2, edgecolor=INK2, lw=0.6, label=r"$5\endash95\%$"),
                           Line2D([], [], color=INK2, lw=0.8, ls=(0, (1, 1.6)), label=r"mean $\mathbb{E}[\Delta]$"), h_dot(INK2, "simulated quantiles")],
                  loc="upper right", ncol=2, handlelength=1.4, columnspacing=1.0, borderaxespad=0.15)

    # ---------------------------------------------------------------- d: by horizon T, the whole population
    PH = D["pop_h"]; d_, e_ = bot
    for j, (T_, col) in enumerate(zip(PH["Ts"], GREENS4)):
        theory(d_, PH["Nth"], PH["theory"][:, j], col); simulated(d_, PH["Ns"], PH["sim"][:, j], 2 * PH["se"][:, j], col)
    d_.legend(handles=[h_both(col, (r"$T=" if j == 0 else "$") + f"{T_:,}$".replace(",", "{,}")) for j, (T_, col) in enumerate(zip(PH["Ts"], GREENS4))],
              loc="upper right", ncol=4, columnspacing=0.8, handlelength=1.3)
    title(d_, "d", r"By horizon $T$, all interaction sets")

    # ---------------------------------------------------------------- e: by third of q, T = 30
    k30 = list(cfg["TH_TS"]).index(max(cfg["TH_TS"]))
    for MO, col in zip(D["mono3"], BLUES3):
        theory(e_, MO["Nth"], MO["theory"][:, k30], col); simulated(e_, MO["Ns"], MO["sim"][:, k30], 2 * MO["se"][:, k30], col)
    e_.legend(handles=[h_both(col, pct(lo, hi, i == 0)) for i, ((lo, hi), col) in enumerate(zip(D["third_bins"], BLUES3))], loc="upper right", ncol=3,
              columnspacing=0.8, handlelength=1.3)
    title(e_, "e", rf"By trapped fraction $q$, $T={max(cfg['TH_TS'])}$")
    for ax in (d_, e_):
        ax.set_xscale("log"); ax.set_xlim(8, 2600); ax.set_ylim(0, 1.22); ax.set_yticks([0, 0.5, 1.0]); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.1f}$")); ax.set_xlabel(r"episodes averaged, $N$")
    d_.set_ylabel(r"P(a decrease within $T$ steps)")
    for ax in top + bot: ax.tick_params(which="both", width=0.6)
    finish_figure(fig, (top, bot), out_dir, "fig_fr_averages", tag)


# ============================================================================================================
# 8. COMMAND LINE
# ============================================================================================================
def main():
    ap = argparse.ArgumentParser(description="Factorization Regret theory figure: data, checks, numbers, figure.")
    ap.add_argument("command", choices=["selftest", "envcheck", "fixture", "quick", "all", "figure", "check", "numbers"])
    ap.add_argument("--quick", action="store_true", help="use the reduced-size configuration and its own caches")
    ap.add_argument("--fresh", action="store_true", help="recompute all caches of this mode")
    ap.add_argument("--rederive", action="store_true", help="recompute only the derived quantities from the cached pools and simulations (after editing derive())")
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--fmode", default=None, choices=["rate", "accuracy"], help="panel f: growth rate (default) or accuracy gap; a variant is saved with _f<mode>")
    a = ap.parse_args(); quick = a.quick or a.command == "quick"; cfg = dict(CFG, **(QUICK if quick else {})); tag = "_quick" if quick else ""
    if a.fmode and a.fmode != cfg["F_MODE"]: cfg["F_MODE"] = a.fmode; tag += f"_f{a.fmode}"
    assert set(cfg["LAMS_FULL"]) <= set(cfg["LAMS_SIM"]) <= set(cfg["LAMS_POOL"]) and cfg["LAM_MAIN"] in cfg["LAMS_FULL"], "LAMS_FULL within LAMS_SIM within LAMS_POOL; LAM_MAIN in LAMS_FULL"
    assert cfg["SLOPE_WINDOW"][1] <= cfg["T"] and cfg["T_EARLY"] < cfg["T"]
    os.makedirs(a.out, exist_ok=True); store = Store(a.out, quick); ok = True
    if a.command == "selftest": ok = selftest(cfg, store)
    elif a.command == "envcheck": ok = envcheck(cfg)
    elif a.command == "fixture": ok = fixture(cfg, a.out)
    else:
        if a.command in ("quick", "all") and not (selftest(cfg, store) and envcheck(cfg)): sys.exit("selftest or envcheck failed: fix that first")
        row1, summary, D = load_all(cfg, store, fresh=a.fresh, rederive=a.rederive)
        if a.command in ("quick", "all", "check"): check(cfg, summary, D)
        if a.command in ("quick", "all", "numbers"): numbers(cfg, summary, D, a.out, tag)
        if a.command in ("quick", "all", "figure"):
            make_figure(cfg, row1, summary, D, a.out, tag)
            if not a.fmode or a.fmode == CFG["F_MODE"]: make_figure_averages(cfg, D, a.out, "_quick" if quick else "")   # Figure 2 has no f variant
        if quick: print("\nNOTE: reduced-size run (smoke test).  Its numbers and figure are not the ones to report, and its extra WARNs\n"
                        "(dip probabilities, binned law, bands of panel h) only reflect the small sample.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
