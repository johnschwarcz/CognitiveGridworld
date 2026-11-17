"""Generator route vs Classifier route for the RCC's shared embeddings, with the channel order shuffled at the
Classifier's input: the supplementary figure that follows the Generator-ablation figure (print size, 5.5 in wide).

WHAT IT SHOWS (layout and encodings as John set them on 11-15, 07:50-08:30 FL)
  Size  5.5 x 1.88 in (the print width; "wider" = shorter). (a) and (b) are equally wide (same training episodes per
        inch, John: "panel b has the same training episode width as panel a").
  (a)   Goal accuracy at the final step (t = T = 30) of every seed on each test, against training episodes (linear,
        raw tests, no smoothing), from 0 to the cut. Blue = Generator route (the embeddings K, Q receive the
        Generator's gradient), red = Classifier route (they receive the Classifier's gradient); Generator drawn first
        and Classifier on top, both with alpha .55, so every red curve stays visible (CURVE_ORDER, CURVE_ALPHA).
        Dashed = Joint Bayes (the paper-wide cached bound, main/DATA/cache/bayes_bounds_ctx2_T30.npy), grey = chance
        1/R (R from the run configs; the manuscript defines chance = 1/R). No Naive Bayes line (removed by John).
        Untrained networks score .08-.11 and every network settles at ~.13 within the first tests, above the 1/R line;
        the caption says so.
  End   Each seed's mean accuracy over its last END_N tests at or before the cut, one dot per seed, one column per route,
        packed by pack(): no overlap (centre spacing END_SPACING dots), each column at most 2 x END_HALF_W pt wide, a dot
        moved off its true height only where it does not fit beside the others (the largest move is printed and goes in
        the caption). The strip is as wide as the columns need (0.30 in at 478k). Open circles (only with a --cut past
        the common length) = runs that end before the cut.
  (b)   "Seeds stuck": the number of seeds that have not left chance (a running mean of WIN tests below LEAVE) by each
        training episode, counted down from 13, one step per seed, no markers; the count at the cut at each curve's end.
        With a --cut past the common length, a route's step ends at its longest run's last test and turns dashed at the
        end of its first run that is still at chance, with a tick at each such run's end.
  Key   Table "Gradient to E-hat from" in (b)'s upper right (empty once most seeds have left chance); the script warns
        if a curve passes under it.

DATA (read only): 13 seeds per route (0-12), fig_batch_routes.load, newest continuation first (its log starts with
  the source run's log, so each run's full history from episode 0 is read; the ep-0 test is taken from the same file).
    Generator route   rccpermb1k1m_lr4_s{k}    -> rccpermb1k450_lr4 -> rccpermb1k643_lr4 -> rccpermb1k700_lr4
    Classifier route  rccclfpermb1k1m_lr4_s{k} -> rccclfpermb1k450_lr4 -> rccclfpermb1k643_lr4 -> rccclfpermb1k700_lr4
    (Classifier seeds 7 and 8 skip some steps: 1M run -> 700k for seed 7, 1M run -> 643k -> 700k for seed 8)
  Settings read from train_cluster_runs.py (fb.runner) and checked identical across the 26 runs except embedding_grad
  and the run length: mode RL (the RCC's own binary feedback), readin_perm (the order of the d_o = 5 observation
  channels shuffled independently for every trajectory at the Classifier's input), batch 1,000 trajectories per training
  episode, lr 1e-4 (both modules), R = 10, C = 2. One test every 250 training episodes = one testing episode of 1,000
  trajectories whose variables are all testing variables (novel environments); "net" = the RCC's goal accuracy at T;
  "train_net" = the Classifier's accuracy at T on the last training batch before the test (training variables).
  Episode axis: the logged episode index, which counts the test episodes (no update) and one extra test per resume, as
  the submitted figure's "1 test per 250 training episodes" does; "training episode 450,000" is ~0.4% fewer updates.

DEFINITIONS (agreed with the author)
  passed Naive  first of RUN = 3 consecutive tests with net > that test's own Naive accuracy; time = that test's episode
  left chance   first test at which the trailing WIN = 5-test mean of net is >= LEAVE = .2; time = that test's episode.
                "At chance" is about TESTING episodes only: the seeds stuck there do learn on training episodes
                (train_net .13 -> ~.20), they do not generalize to testing variables; the text says so.
  failing       never left chance up to min(cut, the run's end), the run reaches 3x the slowest Generator-route Naive
                crossing, and the cut is past it (FAIL_X = 3). Not conservative: Classifier seed 12 left chance at
                404,250 and passed Naive at 432,750 = 2.92x; the text names the latest crossing.
  End           mean of the last END_N = 5 tests at or before the cut
  gain          mean of the last GAIN_N = 20 tests minus the mean of the 20 tests ending GAIN_GAP = 50,000 episodes
                earlier; only runs that reach the cut and passed Naive before that earlier window; median per route
THE CUT  --cut default = the common length = min over the 26 runs of their last test, snapped to the 250-episode test
  grid (the 450k runs end at 449,999 -> 450,000; as the submitted figure); the run that sets it is printed. --cut N
  overrides (runs that end earlier are drawn to their end, open, and treated as censored; a cut past the longest run is
  clamped to it). A run "reaches" the cut if its last test is within one test (250 episodes) of it. The caption says
  "the length of the shortest run" only when the cut is within one test of the common length. The --text file starts
  with "% PREVIEW, DO NOT PASTE" (listing the short runs) only when some run ends more than one test before the cut,
  and with "% INTERIM" while the cut is short of the failing rule; otherwise it is pasteable. Milestones: --cut 450000,
  --cut 643000 (at 643k Classifier seed 8's run ends at 612,750 unless it is continued, so that one is a PREVIEW).
STATISTICS (one function, tests(); every p one-sided, direction fixed in advance: the Classifier route is worse)
  rank test   exact permutation test of the rank sum of the time to pass Naive (Mann-Whitney), seeds that have not
              passed tied at the top rank (censored beyond their end; valid while every such run ends after every
              observed crossing, checked and printed). Complete separation of 13 vs 13 gives p = 1/C(26,13) = 9.6e-8.
              It treats the 26 runs as independent although the routes share seeds; the within-seed association of
              the two routes' times (Kendall tau-b, censored tied last) is printed (.05-.09, p .7-.8 so far) and stated
              in the text.
  sign test   paired by seed number (same seed = same environments and test episodes: Naive/Joint agree on every shared
              test before the first resume, checked; the untrained networks score the same to within one trajectory
              in 1,000; equal initial weights are inferred, not checked): Classifier later than Generator; pairs where
              both are censored are ties (dropped), pairs a censored run cannot order are dropped. Exact, paired;
              smallest possible p with 13 pairs = 2^-13 = 1.2e-4.
  Fisher      exact test on the failing counts (one-sided; the two-sided p is printed too), only once the rule applies.

NUMBERS  (printed by the script; this copy = the default cut,
  rendered 2026-11-15 06:04 Florida time = 13:04 Israel time; runs still training)
  cut = 335,500 training episodes = the common length, set by Generator seed 12; 1,342-1,343 tests per run.
  Settings shared by all 26 runs: mode RL, batch 1,000, lr 1e-4 / 1e-4, readin_perm, R 10, d_o 5, C 2, T 30, |S| 500.
  Bounds Joint .7685, Naive .4455; chance 1/R = .100; test before training .081-.114; every run sits at .130 (mean of
    4,186 tests before episode 40,500, before any run leaves chance).
  Generator route: passed Naive 13/13 at 68,000-148,250. Per seed s0-s12: 111,750 86,000 148,250 108,250 79,000
    123,000 96,500 107,000 116,251 68,000 80,750 113,750 68,250. End .685-.704; gain median .007 (.002-.016).
  Classifier route: passed Naive 9/13 at 153,500-302,750: s3 153,500, s9 167,751, s2 173,000, s4 186,000, s1 188,000,
    s6 189,250, s11 233,250, s5 248,250, s0 302,750. End of those .570-.671; gain median .021 (.016-.066, 8 seeds).
    Not passed: s7, s8, s10, s12, all at chance on testing episodes (End .115-.132); their accuracy on training
    episodes .137 before episode 40,500, End .190-.214.
  Every Classifier-route crossing is later than every Generator-route crossing (148,250 < 153,500): True.
  Rank test U = 0, exact one-sided p = 9.6e-08 = 1/C(26,13) (valid: True); within-seed Kendall tau-b = 0.05 (p =
    0.80); sign test: Classifier later for 13/13 seeds, one-sided p = 0.00012.
  Pairing: the untrained networks differ by at most 1 trajectory in 1,000; Naive/Joint differ on 0 of 7,031 shared
    tests before the first resume.
  Latest Classifier crossing: s0 at 302,750 = 2.04 x 148,250. Failing rule: 3 x 148,250 = 444,750 > cut, so no seed
    can be called failing at this cut.
  Resumes (fresh Adam, seed + 1000), before passing / only after / never: Generator [(8, [72750])] / [0, 1, 2, 3, 4,
    5, 6, 7, 9] / [10, 11, 12]; Classifier [(9, [73000]), (10, [78500])] / [1, 2, 3, 4, 5, 6] / [0, 7, 8, 11, 12].
  End strip widened to 38.9 pt (default 37.4 pt).
  Preview --cut 450000 (same time; 6 runs short of it, censored): Classifier 10/13 passed (153,500-432,750); failing
    Classifier [7, 8] vs Generator [], Fisher one-sided p = 0.24 (two-sided 0.48); not passed and not failing: s10
    (data to 434,251, End .124); rank and sign tests unchanged.

    python fig13_shuffled_routes.py [--cut N] [--paper] [--out DIR] [--text PATH] [--json PATH]
      --paper  also save PDF + SVG (pdf.fonttype 42) beside the PNG
      --out    write the PNG/PDF/SVG to DIR instead of figure_bin (previews at other cuts)
      --text   write the LaTeX caption, appendix paragraph and main-text sentence, filled with this cut's numbers
      --json   write the per-seed summary and the tests (read by the hand-over notes; nothing is retyped)
"""
import os, sys, json, argparse
from math import comb
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
if not os.path.exists(os.path.join(ROOT, "main", "utils.py")):        # VS Code interactive may set __file__ = '/undefined'
    ROOT = "/home/johns/Desktop/CognitiveGridworld"
    HERE = os.path.join(ROOT, "main", "REPLICATION", "claude_plots")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import fig_batch_routes as fb
from main.utils import fig_path

NAME = "fig13_shuffled_routes"
ROUTES = (("Generator", "rccpermb1k1m_lr4", ("rccpermb1kfin_lr4", "rccpermb1k700_lr4", "rccpermb1k643_lr4",
                                             "rccpermb1k450_lr4"), "#2c7bb6"),
          ("Classifier", "rccclfpermb1k1m_lr4", ("rccclfpermb1kfin_lr4", "rccclfpermb1k700_lr4", "rccclfpermb1k643_lr4",
                                                 "rccclfpermb1k450_lr4"), "#d7191c"))
# John 11-15 08:55 FL: the final length X (the episode at which Classifier seed 12 joined the other learned Classifier seeds;
# set by the scratchpad watcher smallbatch/final_watch.py) caps the default cut once it exists; seeds 13-21
# (rcc{,clf}permb1kfin_lr4, trained to X) are drawn with --seeds 22 once every one of them has reached X
FINAL_CUT = os.path.join(ROOT, "main", "DATA", "experimental", "cluster_runs", "final_cut.json")
COL = {r[0]: r[3] for r in ROUTES}
SEEDS = range(13)                 # the original seeds; --seeds N draws seeds 0..N-1
NEW_SEEDS_STOP = 41               # new seeds 13-40 (rcc{,clf}permb1kfin_lr4) join once both routes reach X
EVERY = 250                       # training episodes per test (checked against the logs)
RUN = 3                           # passed Naive: RUN consecutive tests above that test's Naive
LEAVE, WIN = .2, 5                # left chance: trailing WIN-test mean >= LEAVE
CURVE_ORDER = ("Generator", "Classifier")   # (a): drawing order, the last on top
CURVE_ALPHA = {"Generator": .55, "Classifier": .55}  # John 11-15 08:35 FL: "have the red curves use the same alpha"
END_N = 5                         # End strip: mean of the last END_N tests at or before the cut
END_DOT = 2.0                     # End strip: dot diameter (pt); John 11-15 17:10 FL: "make the end circles larger"
END_SPACING = 1.1                 # End strip: centre spacing of neighbouring dots, in dot diameters
END_HALF_W = 9.0                  # End strip: half-width of each route's column of dots (pt); moves < .01 at PH 1.25
FAIL_X = 3                        # failing: at chance past FAIL_X x the slowest Generator-route crossing
GAIN_GAP, GAIN_N = 50_000, 20     # gain: mean of the last GAIN_N tests minus the GAIN_N tests ending GAIN_GAP earlier
BOUNDS = os.path.join(ROOT, "main", "DATA", "cache", "bayes_bounds_ctx2_T30.npy")
SAME = ("mode", "batch_num", "classifier_LR", "generator_LR", "readin_perm", "realization_num", "obs_num", "ctx_num",
        "step_num", "state_num", "gen_zflow", "classifier_ent_bonus", "hid_dim")


# ----------------------------------------------------------------------------------------------------------- data
def load_runs(seeds=None):
    """{route: [run]} with the full history (ep-0 test included) and the episodes after which a run was resumed"""
    seeds = SEEDS if seeds is None else seeds
    _, _, _, cfgs = fb.runner()
    out = {}
    for route, base, ext, _ in ROUTES:
        runs = []
        for k in seeds:
            rows = fb.load(base, ext, k)
            name = fb.load.name
            f = next(os.path.join(d, name + ".jsonl") for d in fb.LOGS if os.path.exists(os.path.join(d, name + ".jsonl")))
            raw = [json.loads(l.replace("NaN", "null")) for l in open(f) if l.strip()]
            r0 = [r for r in raw if r["ep"] == 0]
            rows = r0[:1] + rows                                          # fb.load drops the test before training
            eps = [r["ep"] for r in raw]
            # a continuation logs a test at the source's last episode + 1 before its first update: a resume point
            resumes = [eps[i - 1] for i in range(1, len(eps)) if eps[i] > 0 and 0 <= eps[i] - eps[i - 1] <= 1]
            runs.append(dict(seed=k, name=name, cfg=cfgs.get(name), resumes=resumes,
                             ep=np.array([r["ep"] for r in rows], int),
                             net=np.array([r["net"] for r in rows], float),
                             naive=np.array([r["naive"] for r in rows], float),
                             joint=np.array([r["joint"] for r in rows], float),
                             train=np.array([np.nan if r.get("train_net") is None else r["train_net"] for r in rows],
                                            float)))
        out[route] = runs
    return out


def check_config(runs):
    """the settings every run shares (raises if they differ), plus the route switch"""
    allr = [r for rs in runs.values() for r in rs]
    missing = [r["name"] for r in allr if r["cfg"] is None]
    if missing:
        raise ValueError(f"no config for {missing}")
    cfg = {k: allr[0]["cfg"].get(k) for k in SAME}
    for r in allr:
        diff = {k: r["cfg"].get(k) for k in SAME if r["cfg"].get(k) != cfg[k]}
        if diff:
            raise ValueError(f"{r['name']} differs: {diff}")
    for route, rs in runs.items():
        grads = {r["cfg"].get("embedding_grad", "generator") for r in rs}
        if grads != {route.lower()}:
            raise ValueError(f"{route} route runs have embedding_grad {grads}")
    steps = np.concatenate([np.diff(r["ep"][1:]) for r in allr])       # test spacing after the ep-0 test
    if np.median(steps) != EVERY:
        raise ValueError(f"tests are {np.median(steps)} episodes apart, not {EVERY}")
    return cfg


def snap(x):
    """an episode count on the test grid (the 450k runs end at 449,999, resumed runs log tests at x,001 / x,751)"""
    return int(round(x / EVERY) * EVERY)


def summarize(run, cut):
    m = run["ep"] <= cut
    ep, net, nv, tr = run["ep"][m], run["net"][m], run["naive"][m], run["train"][m]
    ok = np.convolve((net > nv).astype(int), np.ones(RUN, int), "valid") == RUN
    i = int(np.argmax(ok)) if ok.any() else None
    rm = np.convolve(net, np.ones(WIN) / WIN, "valid")
    j = int(np.argmax(rm >= LEAVE)) + WIN - 1 if (rm >= LEAVE).any() else None
    full = bool(ep[-1] >= cut - EVERY)
    t_pass = None if i is None else int(ep[i])
    # gain: seeds past Naive well before the window, runs that reach the cut
    early = net[ep <= cut - GAIN_GAP][-GAIN_N:]
    t_l = None if j is None else int(ep[j])
    gain = (float(net[-GAIN_N:].mean() - early.mean()) if full and t_l is not None and len(early) == GAIN_N
            and t_l <= cut - GAIN_GAP - GAIN_N * EVERY else None)
    return dict(seed=run["seed"], name=run["name"], ep=ep, net=net, naive=nv, train=tr,
                t_pass=t_pass, t_left=None if j is None else int(ep[j]),
                end=int(ep[-1]), n_tests=len(ep), end_acc=float(net[-END_N:].mean()),
                train_end=float(np.nanmean(tr[-END_N:])), gain=gain,
                full=full, resumes=[x for x in run["resumes"] if x + 1 < cut])   # a resume at x changes only tests after x + 1


# ----------------------------------------------------------------------------------------------------------- tests
def perm_rank_p(x, y):
    """exact one-sided permutation p that x's rank sum is as small as observed (Mann-Whitney U, H1: x earlier), with
    midranks for ties (seeds that have not passed share the top rank); returns (p, U)"""
    from scipy.stats import rankdata
    r2 = np.rint(2 * rankdata(np.r_[x, y])).astype(int)          # doubled midranks are integers
    n1, tot = len(x), int(r2.sum())
    ways = np.zeros((n1 + 1, tot + 1))                            # ways[k, s]: k-subsets with doubled rank sum s
    ways[0, 0] = 1
    for v in r2:
        ways[1:, v:] = ways[1:, v:] + ways[:-1, :tot + 1 - v]
    assert ways[n1].sum() == comb(len(r2), n1)
    obs = int(r2[:n1].sum())
    return float(ways[n1, :obs + 1].sum() / ways[n1].sum()), obs / 2 - n1 * (n1 + 1) / 2


def tests(S, cut):
    from scipy.stats import fisher_exact, binomtest
    G, C = S["Generator"], S["Classifier"]
    T = {}
    tg = np.array([np.inf if s["t_pass"] is None else s["t_pass"] for s in G], float)
    tc = np.array([np.inf if s["t_pass"] is None else s["t_pass"] for s in C], float)
    T["slowest"] = int(tg.max()) if np.isfinite(tg).all() else None
    T["thr"] = FAIL_X * T["slowest"] if T["slowest"] else None
    T["rule"] = T["thr"] is not None and cut >= T["thr"]
    for s in G + C:
        s["failing"] = bool(T["rule"] and s["t_left"] is None and s["end"] >= T["thr"])
    T["fail"] = {r: [s["seed"] for s in S[r] if s["failing"]] for r in S}
    fG, fC = len(T["fail"]["Generator"]), len(T["fail"]["Classifier"])
    tab = [[fC, len(C) - fC], [fG, len(G) - fG]]
    T["fisher_one"] = fisher_exact(tab, alternative="greater")[1] if T["rule"] else None
    T["fisher_two"] = fisher_exact(tab)[1] if T["rule"] else None
    fin = np.r_[tg, tc][np.isfinite(np.r_[tg, tc])]
    cens_end = [s["end"] for s in G + C if s["t_pass"] is None]
    T["rank_valid"] = bool(not cens_end or min(cens_end) >= fin.max())
    T["rank_p"], T["U"] = perm_rank_p(tg, tc)
    T["rank_min_p"] = 1 / comb(len(G) + len(C), len(G))
    # the rank test treats the 26 runs as independent; the routes share seeds, so report the within-seed association
    # of the two routes' times (not passed = tied last), Kendall's tau-b
    from scipy.stats import kendalltau, rankdata
    tau = kendalltau(rankdata(tg), rankdata(tc))
    T["tau"] = (float(tau.statistic), float(tau.pvalue))
    later = earlier = undecided = ties = 0
    for g, c, a, b in zip(G, C, tg, tc):
        if np.isinf(a) and np.isinf(b):
            ties += 1
        elif (np.isinf(b) and c["end"] < a) or (np.isinf(a) and g["end"] < b):
            undecided += 1
        elif b > a:
            later += 1
        else:
            earlier += 1
    T["sign"] = (later, later + earlier, binomtest(later, later + earlier, .5, alternative="greater").pvalue
                 if later + earlier else None, ties, undecided)
    T["separated"] = bool(np.isfinite(tg).all() and (not np.isfinite(tc).any() or tc[np.isfinite(tc)].min() > tg.max()))
    # the same tests on the time to leave chance, the event panel (b) counts (still at chance = tied last)
    lg = np.array([np.inf if s["t_left"] is None else s["t_left"] for s in G], float)
    lc = np.array([np.inf if s["t_left"] is None else s["t_left"] for s in C], float)
    L = {}
    L["rank_p"], L["U"] = perm_rank_p(lg, lc)
    fin = np.r_[lg, lc][np.isfinite(np.r_[lg, lc])]
    cens_end = [s["end"] for s in G + C if s["t_left"] is None]
    L["rank_valid"] = bool(not cens_end or min(cens_end) >= fin.max())
    tau = kendalltau(rankdata(lg), rankdata(lc))
    L["tau"] = (float(tau.statistic), float(tau.pvalue))
    later = earlier = 0
    for g, c, a, b in zip(G, C, lg, lc):
        if np.isinf(a) and np.isinf(b) or (np.isinf(b) and c["end"] < a) or (np.isinf(a) and g["end"] < b):
            continue
        later, earlier = (later + 1, earlier) if b > a else (later, earlier + 1)
    L["sign"] = (later, later + earlier,
                 binomtest(later, later + earlier, .5, alternative="greater").pvalue if later + earlier else None)
    L["all_gen_left"] = bool(np.isfinite(lg).all())
    L["overlap"] = int(np.sum(lc[np.isfinite(lc)] < lg.max())) if np.isfinite(lg).all() else None
    T["left"] = L
    return T


# ----------------------------------------------------------------------------------------------------------- numbers
def numbers(runs, S, T, cut, common, setter, cfg, joint, naive, chance):
    """everything the caption and text use, as one dict; printed by report()"""
    # the level every run sits at before any run leaves chance (tests after the first, before the earliest window that
    # leads out of chance), and the test before training
    t0 = min(s["t_left"] for r in S for s in S[r] if s["t_left"] is not None) - (WIN - 1) * EVERY
    pre = np.concatenate([s["net"][(s["ep"] > 0) & (s["ep"] < t0)] for r in S for s in S[r]])
    first = np.array([s["net"][0] for r in S for s in S[r] if s["ep"][0] == 0])
    settle = 0
    for r in S:
        for s in S[r]:
            m20 = np.convolve(s["net"], np.ones(20) / 20, "valid")
            k = np.argmax(m20 >= float(pre.mean()) - .005) if (m20 >= float(pre.mean()) - .005).any() else len(m20) - 1
            settle = max(settle, int(s["ep"][k + 19]))
    settle = int(np.ceil(settle / 5000) * 5000)
    N = dict(cut=cut, common=common, setter=setter, joint=joint, naive=naive, chance=chance, cfg=cfg, T=T, settle=settle,
             test_states=cfg.get("test_states") or cfg["state_num"] // 10,
             plateau=(float(pre.mean()), t0, len(pre)), first=(float(first.min()), float(first.max())),
             n_tests=(min(s["n_tests"] for r in S for s in S[r]), max(s["n_tests"] for r in S for s in S[r])),
             short=[(r, s["seed"], s["end"]) for r in S for s in S[r] if not s["full"]],
             at_common=bool(abs(cut - common) <= EVERY))
    # pairing by seed: the test before training (same seed -> same test episodes) and the Bayes observers on every shared
    # test before the first resume of either run (a resume reseeds training and testing)
    pair_n0, pair_bayes = [], []
    for g, c in zip(runs["Generator"], runs["Classifier"]):
        if g["ep"][0] == 0 and c["ep"][0] == 0:
            pair_n0.append(int(round(abs(g["net"][0] - c["net"][0]) * cfg["batch_num"])))
        stop = min(g["resumes"] + c["resumes"] + [cut])
        sh, gi, ci = np.intersect1d(g["ep"], c["ep"], return_indices=True)
        k = sh <= stop
        pair_bayes.append((int(k.sum()), int(np.sum((g["naive"][gi][k] != c["naive"][ci][k])
                                                   | (g["joint"][gi][k] != c["joint"][ci][k])))))
    N["pair"] = dict(n0_max=max(pair_n0), n0_seeds=len(pair_n0), bayes_tests=sum(a for a, _ in pair_bayes),
                     bayes_diff=sum(b for _, b in pair_bayes))
    for r, X in S.items():
        tp = sorted(s["t_pass"] for s in X if s["t_pass"] is not None)
        allt = sorted(np.inf if s["t_pass"] is None else s["t_pass"] for s in X)
        med = allt[len(allt) // 2]                                 # median over all seeds; censored ones are last
        N[r] = dict(n=len(X), passed=len(tp), t=tp, med_all=med, med_passed=float(np.median(tp)) if tp else None,
                    left=sorted(s["t_left"] for s in X if s["t_left"] is not None),
                    at_chance=[s["seed"] for s in X if s["t_left"] is None],
                    left_not_passed=[s["seed"] for s in X if s["t_left"] is not None and s["t_pass"] is None],
                    end_pass=[s["end_acc"] for s in X if s["t_pass"] is not None],
                    end_chance=[s["end_acc"] for s in X if s["t_left"] is None],
                    end_all=[s["end_acc"] for s in X],
                    res_before=[(s["seed"], [x for x in s["resumes"] if s["t_pass"] is None or x < s["t_pass"]])
                                for s in X if any(s["t_pass"] is None or x < s["t_pass"] for x in s["resumes"])],
                    res_after=[s["seed"] for s in X if s["resumes"] and s["t_pass"] is not None
                               and all(x >= s["t_pass"] for x in s["resumes"])],
                    never_resumed=[s["seed"] for s in X if not s["resumes"]],
                    gain=[s["gain"] for s in X if s["gain"] is not None],
                    last=max(((s["t_pass"], s["seed"]) for s in X if s["t_pass"] is not None), default=None))
        N[r]["gain_med"] = float(np.median(N[r]["gain"])) if N[r]["gain"] else None
    return N


def train_rise(S, N, route, seeds):
    """accuracy on training episodes (the last training batch before each test) of these seeds: pooled mean over the
    early level's window (0 < episode < t0), and each seed's mean over its last END_N tests (min, max)"""
    t0 = N["plateau"][1]
    X = [s for s in S[route] if s["seed"] in seeds]
    if not X:
        return None
    early = np.concatenate([s["train"][(s["ep"] > 0) & (s["ep"] < t0)] for s in X])
    late = [s["train_end"] for s in X]
    return float(np.nanmean(early)), float(min(late)), float(max(late))


def report(S, N):
    T = N["T"]
    c = N["cfg"]
    print(f"cut = {N['cut']:,} training episodes (common length {N['common']:,} on the {EVERY}-episode test grid, set by "
          f"{N['setter']}; the cut {'IS' if N['at_common'] else 'is NOT'} the common length); "
          f"{N['n_tests'][0]}-{N['n_tests'][1]} tests per run up to the cut")
    print(f"settings shared by all 26 runs: mode {c['mode']}, batch {c['batch_num']:,}, lr {c['classifier_LR']:g} / "
          f"{c['generator_LR']:g}, readin_perm {c['readin_perm']}, R {c['realization_num']}, d_o {c['obs_num']}, "
          f"C {c['ctx_num']}, T {c['step_num']}, |S| {c['state_num']}")
    print(f"bounds (cache): Joint {N['joint']:.4f}, Naive {N['naive']:.4f}; chance 1/R = {N['chance']:.3f}; test before "
          f"training {N['first'][0]:.3f}-{N['first'][1]:.3f}; early level (all runs, 0 < episode < {N['plateau'][1]:,}, "
          f"{N['plateau'][2]:,} tests) {N['plateau'][0]:.3f}")
    if N["short"]:
        print("WARNING runs that end before the cut (drawn to their end, censored there): "
              + ", ".join(f"{r} s{k} at {e:,}" for r, k, e in N["short"]))
    for r, X in S.items():
        n = N[r]
        print(f"\n{r} route: passed Naive {n['passed']}/{n['n']}"
              + (f" at {n['t'][0]:,}-{n['t'][-1]:,} (median of those {n['med_passed']:,.0f}; over all {n['n']} seeds "
                 f"{'> cut' if np.isinf(n['med_all']) else format(int(n['med_all']), ',')})" if n["t"] else "")
              + f"; left chance {len(n['left'])}/{n['n']}" + (f" at {n['left'][0]:,}-{n['left'][-1]:,}" if n["left"] else ""))
        for s in X:
            print(f"   s{s['seed']:<2} {s['name']:<26} end {s['end']:>8,}  left chance "
                  f"{'-' if s['t_left'] is None else format(s['t_left'], ','):>8}  passed Naive "
                  f"{'-' if s['t_pass'] is None else format(s['t_pass'], ','):>8}  End {s['end_acc']:.3f}"
                  f"{'  FAILING' if s['failing'] else ''}"
                  + (f"  resumed after {', '.join(format(x, ',') for x in s['resumes'])}" if s["resumes"] else ""))
        if n["end_pass"]:
            print(f"   End accuracy, seeds that passed: {min(n['end_pass']):.3f}-{max(n['end_pass']):.3f}")
        if n["gain"]:
            print(f"   gain over the last {GAIN_GAP:,} episodes (last {GAIN_N} tests minus the {GAIN_N} ending {GAIN_GAP:,} "
                  f"earlier; {len(n['gain'])} seeds past Naive before that window): median {n['gain_med']:.3f}, "
                  f"range {min(n['gain']):.3f}-{max(n['gain']):.3f}")
        if n["end_chance"]:
            tr = train_rise(S, N, r, n["at_chance"])
            print(f"   End accuracy, seeds that never left chance ({n['at_chance']}): "
                  f"{min(n['end_chance']):.3f}-{max(n['end_chance']):.3f}; their accuracy on TRAINING episodes: "
                  f"{tr[0]:.3f} before episode {N['plateau'][1]:,}, End {tr[1]:.3f}-{tr[2]:.3f}")
        if n["left_not_passed"]:
            print(f"   left chance but not past Naive: {n['left_not_passed']}")
        print(f"   resumed before passing: {n['res_before'] or 'none'}; resumed only after passing: {n['res_after']}; "
              f"never resumed: {n['never_resumed']}")
    pr = N["pair"]
    print(f"\npairing by seed: test before training differs by at most {pr['n0_max']} trajectory(ies) in "
          f"{N['cfg']['batch_num']:,} ({pr['n0_seeds']} seeds); Naive/Joint differ on {pr['bayes_diff']} of "
          f"{pr['bayes_tests']:,} shared tests before the first resume")
    print(f"every Classifier-route crossing later than every Generator-route crossing: {T['separated']}")
    print(f"rank test (time to pass Naive, not passed = top rank): U = {T['U']:.1f}, exact one-sided p = {T['rank_p']:.2g} "
          f"(smallest possible {T['rank_min_p']:.2g}); valid (every censored run ends after every crossing): {T['rank_valid']}")
    print(f"   within-seed association of the two routes' times (the rank test treats them as independent): Kendall tau-b "
          f"= {T['tau'][0]:.2f} (two-sided p = {T['tau'][1]:.2f})")
    later, n, p, ties, und = T["sign"]
    print(f"sign test by seed: Classifier later in {later}/{n}" + (f", one-sided p = {p:.2g}" if p is not None else "")
          + f" (both censored: {ties}, undecided: {und})")
    if N["Classifier"]["last"] and T["slowest"]:
        t, k = N["Classifier"]["last"]
        print(f"latest Classifier-route crossing: seed {k} at {t:,} = {t / T['slowest']:.2f} x the slowest Generator crossing")
    if T["slowest"]:
        print(f"failing rule: {FAIL_X} x slowest Generator crossing = {T['thr']:,}; "
              + (f"applies: failing Generator {T['fail']['Generator']}, Classifier {T['fail']['Classifier']}; Fisher exact "
                 f"one-sided p = {T['fisher_one']:.3g}, two-sided p = {T['fisher_two']:.3g}" if T["rule"]
                 else "does not apply at this cut (no seed can be called failing)"))


# ----------------------------------------------------------------------------------------------------------- figure
def beeswarm(v, d):
    """sideways offsets (units of d) so that no two points closer than d (same units as v) overlap; runs centred"""
    off = np.zeros(len(v))
    placed = []
    for i in np.argsort(v):
        for c in [0] + [s * m / 2 for m in range(1, 40) for s in (1, -1)]:      # half-dot steps: hexagonal packing
            if all((v[i] - v[p]) ** 2 + ((c - off[p]) * d) ** 2 >= (.98 * d) ** 2 for p in placed):
                off[i] = c
                break
        placed.append(i)
    o = np.argsort(v)
    start = 0
    for e in range(1, len(o) + 1):
        if e == len(o) or v[o[e]] - v[o[e - 1]] >= d:
            idx = o[start:e]
            off[idx] -= (off[idx].max() + off[idx].min()) / 2
            start = e
    return off


def pack(y, d, half_w):
    """End strip (John 11-15: "add more jitter ... so it's easier to tell how many there are"): positions (x, y) in pt for
    dots of centre spacing d whose true heights are y (pt), each as close to its true height as possible, |x| <= half_w,
    no two closer than d. Candidates are tried by vertical shift first (0, then +-d/4, +-d/2, ...), then by distance from
    the column's centre (half-dot steps, so rows interlock); dots are placed from the lowest value up. Each group of dots
    that touch is centred sideways. Returns x, y (pt) and the largest vertical shift (pt)."""
    n = len(y)
    xs, ys = np.zeros(n), np.zeros(n)
    xc = np.arange(-np.floor(half_w / (d / 2)), np.floor(half_w / (d / 2)) + 1) * d / 2
    xc = xc[np.argsort(np.abs(xc), kind="stable")]
    dys = [0.0] + [sgn * k * d / 4 for k in range(1, 400) for sgn in (1, -1)]
    placed = []
    for i in np.argsort(y, kind="stable"):
        done = False
        for dy in dys:
            for x in xc:
                yy = y[i] + dy
                if all((x - px) ** 2 + (yy - py) ** 2 >= (.999 * d) ** 2 for px, py in placed):
                    xs[i], ys[i] = x, yy
                    placed.append((x, yy))
                    done = True
                    break
            if done:
                break
    # centre each group of touching dots (by height) sideways
    o = np.argsort(ys)
    start = 0
    for e in range(1, n + 1):
        if e == n or ys[o[e]] - ys[o[e - 1]] >= d:
            g = o[start:e]
            xs[g] -= (xs[g].max() + xs[g].min()) / 2
            start = e
    return xs, ys, float(np.abs(ys - y).max()) if n else 0.0


def step_xy(t, x0, x1):
    """vertices of the count of event times t (sorted) that are <= x, on [x0, x1]"""
    c = int(np.sum(np.asarray(t) <= x0))
    xs, ys = [x0], [c]
    for ti in t:
        if x0 < ti <= x1:
            xs += [ti, ti]
            ys += [c, c + 1]
            c += 1
    return np.r_[xs, x1], np.r_[ys, c]


def draw(S, N, out_dir, paper):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    import matplotlib.transforms as mtr
    from matplotlib.lines import Line2D
    from matplotlib.legend import Legend
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": .6, "axes.edgecolor": "k", "xtick.major.width": .6,
        "ytick.major.width": .6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2,
        "ytick.major.pad": 2, "axes.labelpad": 2, "axes.titlepad": 4, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "savefig.facecolor": "white"})
    cut, T = N["cut"], N["T"]
    joint, naive, chance = N["joint"], N["naive"], N["chance"]
    BLUE, RED = COL["Generator"], COL["Classifier"]

    # layout in inches, left to right: margin, (a), gap, End strip, line names, (b)'s tick labels, (b), counts
    # John 11-15 17:10 FL: "shrink the width of the figure": at the fixed 5.5 in print width this means taller panels
    # (PH 1.00 -> 1.25 in; the type stays at print size)
    # John 11-15 18:55 FL (seeing it in the manuscript): "remove some of the vertical white space, and tight the figure up a
    # bit (horizontally) so it renders larger": drawn 4.6 in wide with tighter margins, so at \columnwidth (5.5 in) the
    # whole figure prints 5.5 / 4.6 = 1.2x larger (7 / 8 pt type -> 8.4 / 9.6 pt)
    W, BOT, PH, TOP = 4.6, .50, 1.20, .21
    # John 11-15 17:20 FL: "try putting 'Joint' and 'Chance' above the lines so we can get a bit more space": the names
    # sit above their lines at the left of the End strip, so the name column (LB .42 in) is gone
    L, PA, G1, SW, LB, GB, PB, R = .37, 2.26, .05, .20, .05, .16, 1.65, .27
    XLAB_Y, LETTER_Y = .27, .45                       # baselines below the axes: x labels, then the panel letters
    H = BOT + PH + TOP

    # y range of (a) and the End strip: 3 pt beyond the extremes of the data and the bounds (data only, so the strip's
    # dodged columns can be measured before the axes exist)
    vals = np.r_[np.concatenate([s["net"] for r in S for s in S[r]]), joint, chance]
    f = 3 / 72 / PH
    lo = vals.min() - f * (vals.max() - vals.min()) / (1 - 2 * f)
    hi = vals.max() + f * (vals.max() - vals.min()) / (1 - 2 * f)
    # End strip: each seed's mean over its last END_N tests, one column per route (x in pt across the strip); each
    # column is dodged (beeswarm), then the two columns are set side by side from their measured widths. The strip
    # widens (and (a) narrows by as much) when the columns need more than its default width.
    # John 11-15 08:10 FL: "add more jitter to the end circles so it's easier to tell how many there are": no overlap,
    # a visible gap between neighbouring dots (centre spacing END_SPACING x the dot), and the strip widens (taking the
    # width from (a)) only as far as that needs.
    MS, GAP_PT = END_DOT, 2.5                          # filled dot diameter, gap between the columns (pt)
    MO, EW = MS * 1.15, .5                             # open circle diameter and edge (runs that end before the cut)
    any_open = any(not s["full"] for r in S for s in S[r])
    D = MO + EW if any_open else MS                    # largest marker extent
    d = D * END_SPACING
    cols, shift = [], 0.0
    for route in ("Generator", "Classifier"):
        v = np.array([s["end_acc"] for s in S[route]])
        x_, y_, sh = pack((v - lo) / (hi - lo) * PH * 72, d, END_HALF_W)
        shift = max(shift, sh)
        cols.append((route, lo + y_ / (PH * 72) * (hi - lo), x_, np.array([s["full"] for s in S[route]])))
    N["end_shift"] = shift / (PH * 72) * (hi - lo)      # largest vertical shift, in accuracy units (caption)
    print(f"End strip: largest vertical shift {shift:.2f} pt = {N['end_shift']:.4f} accuracy")
    (_, _, oG, _), (_, _, oC, _) = cols
    need = (oG.max() - oG.min()) + GAP_PT + (oC.max() - oC.min()) + D
    SW = max(SW, (need + 2) / 72)
    # John 11-15 08:30 FL: "make sure panel b has the same training episode width as panel a": (a) and (b) share the
    # width left after the margins, the End strip and the name column, so one training episode spans the same length
    PA = PB = (W - L - G1 - SW - LB - GB - R) / 2
    if PA < 1.5:
        raise ValueError(f"End strip needs {need:.1f} pt: (a) and (b) would shrink to {PA:.2f} in")
    print(f"End strip {SW * 72:.1f} pt wide = {SW:.2f} in (columns need {need:.1f} pt); (a) and (b) are {PA:.3f} in")
    x_s = L + PA + G1
    x_b = x_s + SW + LB + GB
    assert abs(x_b + PB + R - W) < 1e-9
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([L / W, BOT / H, PA / W, PH / H])
    axs = fig.add_axes([x_s / W, BOT / H, SW / W, PH / H], sharey=ax)
    bx = fig.add_axes([x_b / W, BOT / H, PB / W, PH / H])

    # (a) every seed, raw tests. John 11-15 08:30 FL: "make the red curves always visible? Maybe have a slight alpha for
    # the blue curves?": drawn in CURVE_ORDER (last on top) with CURVE_ALPHA per route
    for route in CURVE_ORDER:
        for s in S[route]:
            ax.plot(s["ep"], s["net"], color=COL[route], lw=.5, alpha=CURVE_ALPHA[route], solid_joinstyle="round",
                    zorder=2)
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(mt.MultipleLocator(.2))
    ax.set_ylabel("Accuracy")
    ax.set_title("RCC Ablation experiment")           # John 11-15 19:05 FL (was "Goal accuracy at $t = T$")
    for a_ in (ax, axs):
        a_.axhline(joint, color="k", ls=(0, (4, 2)), lw=.8, zorder=5)
        a_.axhline(chance, color=".55", lw=.7, zorder=1)
    x0 = (SW * 72 - need) / 2 + D / 2                  # the arrangement centred in the strip
    centres = (x0 - oG.min(), x0 + (oG.max() - oG.min()) + GAP_PT - oC.min())
    assert need <= SW * 72 + 1e-6
    for (route, v, off, full), c in zip(cols, centres):
        if full.any():                                 # (an empty scatter with clip_on=False draws a dot at the origin)
            axs.scatter(c + off[full], v[full], s=MS ** 2, color=COL[route], edgecolors="none", zorder=6, clip_on=False)
        if (~full).any():
            axs.scatter(c + off[~full], v[~full], s=MO ** 2, facecolors="white", edgecolors=COL[route],
                        linewidths=EW, zorder=6, clip_on=False)
    axs.set_xlim(0, SW * 72)
    axs.set_xticks([])
    axs.tick_params(left=False, labelleft=False)
    axs.spines["left"].set_visible(False)
    axs.set_xlabel("End")
    # the reference lines' names, just above each line at the left of the strip (the Generator column has no dots at
    # either line: its dots sit below Joint and none is at chance); a dot under a name is reported
    name_tr = (mtr.blended_transform_factory(axs.transAxes, axs.transData)
               + mtr.ScaledTranslation(1 / 72, 1.0 / 72, fig.dpi_scale_trans))   # 1 pt right of the strip edge, 1 pt up
    names = [axs.text(0.0, y, t, transform=name_tr, fontsize=7, ha="left", va="bottom", clip_on=False,
                      color="k" if t != "Chance" else ".35") for y, t in ((joint, "Joint"), (chance, "Chance"))]

    # (b) seeds stuck at chance: seeds that have not left chance by each training episode (a running mean of WIN tests
    # below LEAVE), counted down from the number of seeds. A route's step ends at its longest run's last test (never past
    # the data); it turns dashed at the end of its first run that is still stuck there (censored), with a tick at each.
    n_seeds = max(len(S[r]) for r in S)
    x_end = {}
    for route in ("Generator", "Classifier"):
        X = S[route]
        n = len(X)
        t = sorted(s["t_left"] for s in X if s["t_left"] is not None)
        x_end[route] = min(cut, max(s["end"] for s in X))
        cens = sorted(s["end"] for s in X if s["t_left"] is None and not s["full"])
        t_dash = cens[0] if cens else x_end[route]
        xs, ys = step_xy(t, 0, t_dash)
        bx.plot(xs, n - ys, color=COL[route], lw=1.2, solid_joinstyle="miter", zorder=3)
        if t_dash < x_end[route]:
            xs, ys = step_xy(t, t_dash, x_end[route])
            bx.plot(xs, n - ys, color=COL[route], lw=1.0, ls=(0, (2.5, 1.5)), zorder=3)
        for e in cens:                                  # above the step, so the tick does not read as a cross
            bx.plot([e], [n - np.sum(np.asarray(t) <= e) + .5], "|", ms=4, mew=.8, color=COL[route], zorder=4)
    counts = {r: len(S[r]) - sum(s["t_left"] is not None for s in S[r]) for r in S}
    yG, yC = counts["Generator"], counts["Classifier"]
    if abs(yG - yC) < 1.6 and x_end["Generator"] == x_end["Classifier"]:       # keep the two counts apart
        mid = (yG + yC) / 2
        yG, yC = (mid + .8, mid - .8) if yG >= yC else (mid - .8, mid + .8)
    for route, y in (("Generator", yG), ("Classifier", yC)):                   # the count at the end of each step
        bx.text(x_end[route], y, f"{counts[route]}/{len(S[route])}", color=COL[route], fontsize=7, ha="left",
                va="center", transform=bx.transData + mtr.ScaledTranslation(.04, 0, fig.dpi_scale_trans), clip_on=False)
    bx.set_yticks(sorted({0, 5, 10, n_seeds}))
    bx.spines["left"].set_bounds(0, n_seeds)
    bx.set_title("Seeds stuck")                        # the title names the quantity: no y label

    # x axes: linear, 0 to the cut, one baseline for every x label
    step = 50_000 if cut <= 250_000 else 100_000 if cut <= 500_000 else 200_000
    for a_ in (ax, bx):
        a_.set_xlim(0, cut)
        a_.xaxis.set_major_locator(mt.MultipleLocator(step))
        a_.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1e3:g}k"))
        a_.set_xlabel("Training episodes")
    for a_ in (ax, axs, bx):
        a_.xaxis.set_label_coords(.5, -XLAB_Y / PH)
        a_.xaxis.label.set_va("baseline")
    for a_, x0, w, let in ((ax, L, PA, "a"), (bx, x_b, PB, "b")):
        fig.text((x0 + w / 2) / W, (BOT - LETTER_Y) / H, f"({let})", fontfamily="DejaVu Sans", fontsize=8,
                 ha="center", va="baseline")

    # key: a small table in (b)'s upper right, which is empty once most seeds have left chance; the script checks that
    # no curve or dot passes under it
    h = [Line2D([], [], color=COL[r], lw=1.2) for r in ("Generator", "Classifier")]
    lg = Legend(bx, h, ["Generator", "Classifier"], loc="upper right", bbox_to_anchor=(1, 1),
                bbox_transform=bx.transAxes, frameon=False, title=r"Gradient to $\hat{\mathcal{E}}$ from",
                title_fontsize=7, handlelength=1.5, handletextpad=.45, labelspacing=.22, borderaxespad=0, borderpad=0)
    lg._legend_box.align = "left"
    bx.add_artist(lg)
    bx.set_ylim(-.6, n_seeds + .6)
    fig.canvas.draw()
    rdr = fig.canvas.get_renderer()
    for nm in names:
        nb = nm.get_window_extent(rdr)
        for col in axs.collections:
            xy = col.get_offset_transform().transform(col.get_offsets())
            r = np.sqrt(col.get_sizes().max()) / 2 * fig.dpi / 72 if len(col.get_sizes()) else 0
            if ((xy[:, 0] > nb.x0 - r) & (xy[:, 0] < nb.x1 + r) & (xy[:, 1] > nb.y0 - r) & (xy[:, 1] < nb.y1 + r)).any():
                print(f"WARNING: an End dot lies under the name '{nm.get_text()}'")
    kb = lg.get_window_extent(rdr).expanded(1.0, 1.0)
    pad = 2 * fig.dpi / 72
    for ln in bx.lines:
        xy = bx.transData.transform(np.c_[ln.get_xdata(), ln.get_ydata()])
        # sample along each segment, so a long horizontal step under the key is caught too
        pts = np.concatenate([np.linspace(xy[i], xy[i + 1], 50) for i in range(len(xy) - 1)] or [xy])
        hit = ((pts[:, 0] > kb.x0 - pad) & (pts[:, 0] < kb.x1 + pad) & (pts[:, 1] > kb.y0 - pad)
               & (pts[:, 1] < kb.y1 + pad))
        if hit.any():
            print("WARNING: a curve passes under the key in (b)")
            break

    os.makedirs(out_dir, exist_ok=True) if out_dir else None
    path = (lambda ext: os.path.join(out_dir, f"{NAME}.{ext}")) if out_dir else (lambda ext: fig_path(f"{NAME}.{ext}"))
    fig.savefig(path("png"), dpi=300)
    saved = [path("png")]
    if paper:
        for ext in ("pdf", "svg"):
            fig.savefig(path(ext))
            saved.append(path(ext))
    plt.close(fig)
    print("\nsaved " + ", ".join(saved) + f"  ({W} x {H:.2f} in)")
    return saved


# ----------------------------------------------------------------------------------------------------------- text
def tn(x):
    """LaTeX number with thousands separators"""
    return f"{int(round(x)):,}".replace(",", "{,}")


def tp(p):
    """LaTeX p-value"""
    if p >= .01:
        return f"{p:.2f}"
    m, e = f"{p:.1e}".split("e")
    return rf"{m}\times 10^{{{int(e)}}}"


def seeds_txt(ks):
    ks = [str(k) for k in ks]
    return ks[0] if len(ks) == 1 else ", ".join(ks[:-1]) + " and " + ks[-1]


WORD = dict(enumerate("zero one two three four five six seven eight nine ten".split()))
w = lambda n: WORD.get(n, str(n))                       # small counts as words


def latex(S, N, drawn):
    """caption, appendix paragraph and main-text sentence for the cut shown, as LaTeX; drawn = what the figure shows"""
    T, cfg, cut = N["T"], N["cfg"], N["cut"]
    g, c = N["Generator"], N["Classifier"]
    nG, nC = g["n"], c["n"]
    fC, fG = T["fail"]["Classifier"], T["fail"]["Generator"]
    fails = T["rule"] and len(fC) > len(fG)
    k_not = nC - c["passed"]                            # Classifier-route seeds that have not passed by the cut
    lab = "fig:ablation_shuffled"
    lr = rf"$10^{{{int(round(np.log10(cfg['classifier_LR'])))}}}$"
    n_stuck = len(c["at_chance"])
    title = ("Under naturalistic feedback, Compositional Embeddings are learned later through the Classifier than "
             "through the Generator" + (f", and in {w(n_stuck)} of {nC} seeds not within the training shown" if n_stuck
                                        else "") + " ($C=2$).")
    cut_txt = (f"up to training episode {tn(cut)}, the length every run was trained to" if N.get("final") == cut
               else f"up to training episode {tn(cut)}, the length of the shortest run" if N["at_common"]
               else f"up to training episode {tn(cut)}")
    f0, f1 = N["first"]
    chance_txt = (rf"gray: chance, $1/R = {N['chance']:g}$ (untrained networks score {f0:.2f}--{f1:.2f}, and every "
                  rf"network rises to about {N['plateau'][0]:.2f} within its first {tn(N['settle'])} training episodes; "
                  rf"we count a seed as at chance until a running mean of {w(WIN)} tests first reaches {LEAVE:g})")
    # End: say whether the seeds past Naive Bayes were still rising, with the medians behind it
    gm = {r: N[r]["gain_med"] for r in ("Generator", "Classifier")}
    gain_txt = ""
    if all(v is not None for v in gm.values()):
        pos = {r: (sum(x > 0 for x in N[r]["gain"]), len(N[r]["gain"])) for r in gm}
        rising = all(2 * a > b for a, b in pos.values())       # most seeds of both routes still gaining
        gain_txt = (rf"; accuracies {'were still rising slowly' if rising else 'had levelled off'} (median gain over the "
                    rf"last {tn(GAIN_GAP)} training episodes, seeds that left chance: Generator route "
                    rf"{gm['Generator']:.3f}, Classifier route {gm['Classifier']:.3f}; positive in "
                    rf"{pos['Generator'][0]} of {pos['Generator'][1]} and {pos['Classifier'][0]} of {pos['Classifier'][1]} "
                    rf"seeds)")
    ec = [v for r in ("Generator", "Classifier") for v in N[r]["end_chance"]]
    notes = ([rf"the seeds still at chance end at {min(ec):.2f}--{max(ec):.2f}" if len(ec) > 1 else
              rf"the seed still at chance ends at {ec[0]:.2f}"] if ec else [])
    if N.get("end_shift", 0) >= .005:
        notes.append(rf"dots that would overlap are stacked, which moves them by at most "
                     rf"{np.ceil(N['end_shift'] * 100) / 100:.2f}")
    chance_end = f" ({'; '.join(notes)})" if notes else ""
    test_frac = int(round(100 * N["test_states"] / cfg["state_num"]))
    cap = (rf"\caption{{\textbf{{{title}}} RCCs learn from naturalistic feedback: binary feedback on the Classifier's "
           rf"sampled actions, which also condition the Generator (in \autoref{{fig:ablation}} both were replaced by the "
           rf"true realizations $\mathbf{{r}}$). In both routes the order of the $d_o={cfg['obs_num']}$ observables, each "
           rf"with its estimated interactions $\hat{{\mathbf{{z}}}}^i$, is shuffled in every trajectory at the "
           rf"Classifier's input, so that the Classifier cannot tell them apart by their position. The embeddings "
           rf"$\hat{{\mathcal{{E}}}}$ receive gradients from either the Generator (blue, Generator route) or the "
           rf"Classifier (red, Classifier route); {nG} seeds per route; $|\mathcal{{S}}| = {cfg['state_num']}$ latent "
           rf"variables ({test_frac}\% testing variables); $T = {cfg['step_num']}$; each training episode is a batch "
           rf"of {tn(cfg['batch_num'])} trajectories; learning rate {lr}; "
           rf"{cut_txt}. \textbf{{(a)}} Goal accuracy at the final step of inference on one testing episode "
           rf"({tn(cfg['batch_num'])} trajectories of testing variables) every {EVERY} training episodes, one line per "
           rf"seed (Classifier route drawn on top). Dashed: Joint Bayes; {chance_txt}."
           + rf" End: mean over each seed's last {w(END_N)} tests, one dot per seed{chance_end}" + gain_txt
           + ("; open circles: runs that end before the training episode shown." if drawn["open"] else ".")
           + rf" \textbf{{(b)}} Number of seeds still stuck at chance (each step: the test at which one seed's "
           rf"running mean first reached {LEAVE:g}; right: the count at training episode {tn(cut)})."
           + (" Dashed: after the end of a run that was still at chance (ticks: those ends)." if drawn["dash"] else "")
           + "}")

    # appendix paragraph (answer first; every claim can be checked in the figure)
    t_g = rf"{tn(g['t'][0])}--{tn(g['t'][-1])}"
    med = lambda n: "beyond the end of the runs" if np.isinf(n["med_all"]) else tn(n["med_all"])
    later, npair, psign, _, _ = T["sign"]
    sig = fails and T["fisher_one"] < .05
    para = (rf"Under naturalistic feedback, learning through the Classifier is slower"
            + (" and less reliable" if sig else ", and some seeds did not leave chance on testing episodes within the "
               "training shown" if fails else "") + r". "
            rf"We trained {nG} RCCs per route from naturalistic feedback (batch of {tn(cfg['batch_num'])} "
            rf"trajectories, learning rate {lr}). Because the Classifier could otherwise tell the observables apart by "
            rf"their position, their order (each observable with its $\hat{{\mathbf{{z}}}}^i$) was shuffled in every "
            rf"trajectory at the Classifier's input, in both routes. ")
    Lf = T["left"]
    lg_, lc_ = g["left"], c["left"]
    k_stuck = nC - len(lc_)
    para += (rf"Every Generator-route seed left chance within {tn(lg_[-1])} training episodes "
             rf"({len(lg_)} of {nG}, at {tn(lg_[0])}--{tn(lg_[-1])}; \autoref{{{lab}}}b)" if Lf["all_gen_left"] else
             rf"{len(lg_)} of {nG} Generator-route seeds left chance (at {tn(lg_[0])}--{tn(lg_[-1])}; "
             rf"\autoref{{{lab}}}b)")
    para += ((rf"; {len(lc_)} of {nC} Classifier-route seeds left it at {tn(lc_[0])}--{tn(lc_[-1])}" if lc_ else
              rf"; no Classifier-route seed left it") + (rf", and {w(k_stuck)} had not by training episode {tn(cut)}"
                                                         if k_stuck and lc_ else "") + ". ")
    tau, ptau = Lf["tau"]
    indep = ("" if not np.isfinite(tau) else
             rf" (within a seed the two routes' times are not detectably correlated, Kendall's $\tau = {tau:.2f}$, "
             rf"$p = {ptau:.2f}$)" if ptau > .05
             else rf" (although within a seed the two routes' times are correlated, Kendall's $\tau = {tau:.2f}$; the "
                  rf"paired sign test below does not need this)")
    pair = N["pair"]
    n0 = pair["n0_max"]
    same_test = ("the same seed sets the environments and test episodes, and " if pair["bayes_diff"] == 0 else "")
    later, npair, psign = Lf["sign"]
    para += (rf"An exact one-sided permutation test on the time to leave chance, with seeds still at chance ranked "
             rf"last and seeds treated as independent across routes{indep}, gives $p = {tp(Lf['rank_p'])}$; paired by "
             rf"seed ({same_test}the untrained networks score the same"
             + (rf" to within {w(n0)} trajector{'y' if n0 == 1 else 'ies'} in {tn(cfg['batch_num'])}" if n0 else "")
             + rf"), the Classifier route left chance later for {later} of {npair} seeds (one-sided sign test, "
             rf"$p = {tp(psign)}$). ")
    # the same order for passing Naive Bayes (not drawn): RUN consecutive tests above that test's Naive Bayes accuracy
    t_g = rf"{tn(g['t'][0])}--{tn(g['t'][-1])}"
    if T["separated"]:
        para += (rf"Passing Naive Bayes ({w(RUN)} consecutive tests above its accuracy) separates the routes "
                 rf"completely: every Generator-route seed passed it (at {t_g}) before any Classifier-route seed did"
                 + (rf" ({c['passed']} of {nC}, at {tn(c['t'][0])}--{tn(c['t'][-1])})" if c["t"] else "")
                 + rf"; exact permutation test $p = {tp(T['rank_p'])}$"
                 + (r", the smallest value possible with 13 seeds per route" if abs(T["rank_p"] - T["rank_min_p"]) < 1e-12
                    and nG == nC == 13 else "") + ". ")
    else:
        para += (rf"Generator-route seeds passed Naive Bayes ({w(RUN)} consecutive tests above its accuracy) at {t_g}"
                 + (rf", Classifier-route seeds ({c['passed']} of {nC}) at {tn(c['t'][0])}--{tn(c['t'][-1])}" if c["t"]
                    else "") + rf" (exact permutation test $p = {tp(T['rank_p'])}$). ")
    if T["rule"]:
        nfc = len(fC)
        fail_runs = [s for r in S for s in S[r] if s["failing"]]
        if all(s["full"] for s in fail_runs):
            upto = rf"up to training episode {tn(cut)}, {cut / T['slowest']:.1f} times"
        else:
            e = sorted(snap(s["end"]) for s in fail_runs)
            upto = (rf"up to the end of their runs (training episode{'s' if e[-1] != e[0] else ''} {tn(e[0])}"
                    + (rf"--{tn(e[-1])}" if e[-1] != e[0] else "") + rf"), at least {e[0] / T['slowest']:.1f} times")
        tr = train_rise(S, N, "Classifier", fC)
        tr_txt = ("" if tr is None else rf", although their accuracy on training episodes rose from {tr[0]:.2f} to "
                  + (f"{tr[1]:.2f}" if f"{tr[1]:.2f}" == f"{tr[2]:.2f}" else f"{tr[1]:.2f}--{tr[2]:.2f}"))
        late = N["Classifier"]["last"]
        late_txt = (rf": Classifier-route seed {late[1]} passed Naive Bayes at training episode {tn(late[0])}, "
                    rf"{late[0] / T['slowest']:.1f} times the slowest Generator-route time to pass Naive Bayes"
                    if late and late[0] / T["slowest"] >= 2 else "")
        para += (rf"{w(nfc).capitalize()} of {nC} Classifier-route seeds"
                 + (rf" (seed{'s' if nfc > 1 else ''} {seeds_txt(fC)})" if nfc else "")
                 + (rf", and none of the {nG} Generator-route seeds," if not fG else
                    rf", and {w(len(fG))} of {nG} Generator-route seeds,")
                 + rf" stayed at chance on testing episodes {upto} the slowest Generator-route time to pass Naive "
                 rf"Bayes ({tn(T['slowest'])}; \autoref{{{lab}}}a, End){tr_txt}; with {nG} seeds per route this difference "
                 rf"alone is "
                 + (r"not significant" if T["fisher_one"] > .05 else "significant")
                 + rf" (one-sided Fisher exact test, $p = {tp(T['fisher_one'])}$). These seeds meet our failure rule (a "
                 rf"running mean of {w(WIN)} tests that never reaches {LEAVE:g} up to {w(FAIL_X)} times the slowest "
                 rf"Generator-route time to pass Naive Bayes); a seed at chance at the end of the runs may still learn later"
                 rf"{late_txt}. ")
    elif c["at_chance"]:
        na = len(c["at_chance"])
        tr = train_rise(S, N, "Classifier", c["at_chance"])
        para += (rf"At this length {w(na)} Classifier-route seed{'s' if na > 1 else ''} had not left chance on testing "
                 rf"episodes (a running mean of {w(WIN)} tests below {LEAVE:g})"
                 + rf", although {'their' if na > 1 else 'its'} accuracy on training episodes rose from {tr[0]:.2f} to "
                 + (f"{tr[1]:.2f}" if f"{tr[1]:.2f}" == f"{tr[2]:.2f}" else f"{tr[1]:.2f}--{tr[2]:.2f}")
                 + rf"; we would call such a seed failing only if it stayed there past "
                 rf"{w(FAIL_X)} times the slowest Generator-route time to pass Naive Bayes (training episode "
                 rf"{tn(T['thr'])}), which the "
                 rf"runs shown do not yet reach. ")
    before = [(r, k, x) for r in ("Generator", "Classifier") for k, xs in N[r]["res_before"] for x in xs]
    para += (r"Paused runs were resumed with a fresh optimizer state and the random seed of the episode stream "
             r"(training and testing episodes) reset to the run's seed plus 1{,}000, so a run resumed twice replayed, after "
             r"its second resume, the episodes that had followed its first")
    if before:
        parts = []
        for r in ("Generator", "Classifier"):
            ks = sorted({k for rr, k, _ in before if rr == r})
            if ks:
                each = [rf"{k} (at {seeds_txt([tn(x) for rr, kk, x in before if rr == r and kk == k])})" for k in ks]
                parts.append(rf"{r}-route seed{'s' if len(ks) > 1 else ''} {seeds_txt(each)}")
        para += (r"; " + " and ".join(parts) + r" were resumed before passing Naive Bayes, every other resumed run after "
                 r"passing it.")
    else:
        para += r"; every resumed run was resumed after passing Naive Bayes."

    # main text: the claim panel (b) shows (leaving chance), with the seeds still stuck
    Lf = T["left"]
    lg_, lc_ = g["left"], c["left"]
    k_stuck = nC - len(lc_)
    later, npair, _ = Lf["sign"]
    claim = (rf"every Generator-route seed left chance within {tn(lg_[-1])} training episodes" if Lf["all_gen_left"]
             else rf"{len(lg_)} of {nG} Generator-route seeds left chance")
    claim += (rf", each Classifier-route seed later than the Generator-route run with the same seed" if later == npair == nC
              else rf", Classifier-route seeds later in {later} of {npair} seed pairs")
    if k_stuck:
        claim += rf", and {w(k_stuck)} of {nC} Classifier-route seeds were still at chance at training episode {tn(cut)}"
    main = (r"In a simplified, fully supervised setting, ablating the Generator and using gradient flow from the "
            r"Classifier significantly impairs learning (Supplementary \autoref{fig:ablation}). "
            r"Under naturalistic feedback, with the embeddings trained either through the Generator (Generator "
            r"route) or through the Classifier (Classifier route) and the order of the observables shuffled at the "
            rf"Classifier's input, {claim} (Supplementary \autoref{{{lab}}}).")
    fig_env = ("\\begin{figure}[h!]\n  \\centering\n  \\includegraphics[width=1\\columnwidth]{figures/PDFs/"
               f"{NAME}.pdf}}\n  {cap}\n  \\label{{{lab}}}\n\\end{{figure}}")
    return dict(caption=cap, paragraph=para, main=main, figure=fig_env)


# ----------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cut", type=int, default=None, help="last training episode shown (default: the common length)")
    ap.add_argument("--paper", action="store_true", help="also save PDF + SVG (pdf.fonttype 42)")
    ap.add_argument("--out", default=None, help="directory for the files (default: figure_bin via fig_path)")
    ap.add_argument("--text", default=None, help="write the LaTeX caption / paragraph / main-text sentence here")
    ap.add_argument("--json", default=None, help="write the per-seed summary and the tests here (for hand-over notes)")
    ap.add_argument("--seeds", type=int, default=None, help="draw seeds 0..N-1 (default 13; 22 with the new seeds)")
    a = ap.parse_args()
    global SEEDS
    if a.seeds:
        SEEDS = range(a.seeds)

    runs = load_runs()
    cfg = check_config(runs)
    joint, naive = (float(x) for x in np.load(BOUNDS))
    chance = 1 / cfg["realization_num"]
    ends = {(r, x["seed"]): int(x["ep"][-1]) for r in runs for x in runs[r]}
    common_raw = min(ends.values())
    common = snap(common_raw)                          # on the test grid: 449,999 -> 450,000
    setter = ", ".join(f"{r} seed {k}" for (r, k), e in ends.items() if e == common_raw)
    longest = snap(max(ends.values()))
    final = json.load(open(FINAL_CUT))["X"] if os.path.exists(FINAL_CUT) else None
    cut = (common if final is None else min(common, final)) if a.cut is None else min(a.cut, longest)
    if a.cut is None and final is not None:
        print(f"final length X = {final:,} ({FINAL_CUT}); common length {common:,}; cut {cut:,}")
    # John 11-15 09:55 FL ("capped at 600K ... getting more seeds to 600K"): once seeds 0-12 have all reached X, every new
    # seed (13-40) whose BOTH routes have reached X joins the figure (complete pairs only, so the routes stay balanced)
    if a.cut is None and final is not None and cut == final and not a.seeds:
        extra = []
        for k in range(SEEDS.stop, NEW_SEEDS_STOP):
            got = [fb.load(base, ext, k) for _, base, ext, _ in ROUTES]
            if all(g and snap(g[-1]["ep"]) >= final for g in got):
                extra.append(k)
        if extra:
            more = load_runs(extra)
            for r in runs:
                runs[r] += more[r]
            cfg = check_config(runs)
            ends = {(r, x["seed"]): int(x["ep"][-1]) for r in runs for x in runs[r]}
            print(f"new seeds with both routes at {final:,}: {extra} -> {len(runs['Generator'])} seeds per route")
    if a.cut is not None and a.cut > cut:
        print(f"--cut {a.cut:,} is past the longest run; clamped to {cut:,}")
    S = {r: [summarize(x, cut) for x in runs[r]] for r in runs}
    T = tests(S, cut)
    N = numbers(runs, S, T, cut, common, setter, cfg, joint, naive, chance)
    N["final"] = final if a.cut is None else None
    report(S, N)
    draw(S, N, a.out, a.paper)
    drawn = dict(open=any(not s["full"] for r in S for s in S[r]),
                 dash=any(s["t_pass"] is None and not s["full"] for r in S for s in S[r]))
    short = ", ".join(f"{r} s{s['seed']} at {s['end']:,}" for r in S for s in S[r] if not s["full"])
    if a.text:
        tx = latex(S, N, drawn)
        with open(a.text, "w") as fh:
            if drawn["open"]:                          # some run ends more than one test before the cut
                fh.write("% PREVIEW, DO NOT PASTE: runs that end before the cut are drawn open and censored: "
                         + short + "\n")
            elif not T["rule"]:
                fh.write("% INTERIM: the runs are still shorter than the failing rule needs; regenerate at the final cut\n")
            fh.write(f"% generated by {NAME}.py at cut {cut:,} (common length {common:,}, set by {setter}"
                     f"{'' if N['at_common'] else '; the cut is NOT the common length'})\n\n")
            for k in ("figure", "paragraph", "main"):
                fh.write(f"%% ---- {k}\n{tx[k]}\n\n")
        print(f"text: {a.text}")
    if a.json:
        import datetime
        seeds = {r: [dict(seed=s["seed"], name=s["name"], end=s["end"], full=s["full"], t_pass=s["t_pass"],
                          t_left=s["t_left"], end_acc=s["end_acc"], train_end=s["train_end"], failing=s["failing"])
                     for s in S[r]] for r in S}
        out = dict(rendered=datetime.datetime.now().isoformat(timespec="minutes"), cut=cut, common=common,
                   common_raw=common_raw, setter=setter, at_common=N["at_common"], open=drawn["open"],
                   seeds=seeds, plateau=N["plateau"], pair=N["pair"],
                   routes={r: {k: N[r][k] for k in ("n", "passed", "t", "at_chance", "left_not_passed", "gain_med", "last")}
                           for r in S},
                   T={k: T[k] for k in ("slowest", "thr", "rule", "fail", "fisher_one", "fisher_two", "rank_p", "U",
                                         "rank_valid", "sign", "tau", "separated")})
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1, default=lambda x: None if isinstance(x, float) and np.isinf(x) else str(x))
        print(f"json: {a.json}")


if __name__ == "__main__":
    main()
