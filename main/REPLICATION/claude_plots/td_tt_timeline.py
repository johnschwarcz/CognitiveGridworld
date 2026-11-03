"""When does each end-of-training signature form? A synthesis of the five through-training analyses (John 11-01,
away: "go over our existing plots ... further 'through training' analyses ... wormholes, jumps, couplings, splitting,
unfolding, learning on entangled steps, exploration of latent space"). Every topic module (td_tt_jumps, td_tt_wormholes,
td_tt_splitting, td_tt_latent, td_tt_entangled_learning) exports its timing measures with the shared conventions of
td_tt_gap (command `synthesis`, which also stores every resampled curve); this script only reads those exports.
-> main/REPLICATION/figure_bin/td_tt_timeline.png   figure()
   numbers()   every number below

Definitions (td_tt_gap):
  gap fraction f = (y - y_naive) / (y_Bayes - y_naive) of a measure y at a checkpoint, with naive's and Bayes's levels
    on the same episodes: 0 = naive's level, 1 = Bayes's (rises toward Bayes whatever the sign of the measure).
    Additive: one fixed affine map per measure (naive and Bayes are references, not the measure's own spread).
  c_L = the training episode after which f stays above L to the end of training, log-interpolated between checkpoints:
    c0 = passes naive's level for good, c10 = 10% of the gap for good (onset), c50 = half-way for good. A level that the
    curve passes only for a while (early transients) does not count.
  [5-95%] over 1,000 resamples of the 50 test environments (one matrix, td_tt_gap.boot_counts(), for every probe-based
    measure, so those crossings are paired across topics); naive and Bayes resampled on the same environments. A
    resample that never crosses counts as after the end (open bound). The learning rows resample their own units (the
    training episodes of td_tt_entangled_learning) and are not paired with the rest.
  Families (by what a row measures; the topics' 'movement' rows split by topic): answers (what it answers at T, where
    its end beliefs lie); splitting (near beliefs of different cells moving apart, td_tt_splitting); holding (holding a
    split at both ends of a variable: the held shares and the end-mass slope over every belief; and the distance moved
    when leaving a split, td_tt_wormholes); coupling (the two variables' beliefs moving together: angles of jumps and
    of moves out of splits, the long-path correlation G);
    learning (training loss per entanglement fifth). Shapes: rise; return (first moves away from Bayes past naive's
    level, then comes back; its c50 is a different event, so returns are not in the family test).
  Family test (tested rows: rise-shaped, paired, answers / splitting / holding / coupling): in each paired resample,
    is every answers / splitting c50 before every holding / coupling c50? Two single crossings are ordered only when
    their 5-95% intervals do not overlap.
  Left out (in the topics): measures whose naive and Bayes levels are close, or that the network approaches from outside
    the naive-Bayes interval (update top-direction share, back-and-forth), answer diversity (answer cells used, e^H:
    noise raises them), grid folds (naive's level is 100% by definition), the learning excess (no Bayes level). The
    topics' unexported measures (move sizes and TV at splits: late; the early-step split gradient and the overall split
    level: early) fall on the same sides (checked in the review).

Results (Trained RNN; [5-95%]; every number printed by numbers()):
  Order on the additive scale. The answers at T and the splitting of near beliefs are half-way to Bayes (c50) at
  219-1,575 training episodes (774-1,575 without the six-step move toward own cells, which its topic flags as no clean
  timing marker): fold index 774 [706-846] (the end of an early dip: the untrained network is at f .53), split gradient
  793 [615-1,070], far errors 863 [811-950], distant pairs merged 867 [807-931], distance-matched split gradient 927
  [770-1,305], opposite corners 946 [838-1,127], goal accuracy 1,394 [1,285-1,489], both right 1,403 [1,317-1,475],
  corner answer = truth 1,575 [1,477-1,895]. Holding a split and the two variables moving together follow at
  2,747-11,103: same-diagonal share 2,747 [2,111-3,447], distance out 3,021 [2,815-3,511], angle of jumps >= 4 cells
  3,855 [3,363-4,830], end-mass slope 3,936 [3,635-4,854], held per variable 4,983 [4,677-5,623] / 5,082 [3,943-7,571],
  G 5,656 [4,901-6,481], excess angle 6,248 [5,366-7,529], joint held 9,245 [7,693-10,112], direction slope 10,113
  [8,535-11,982], coupling angle out 11,103 [8,969-12,555]. Every early c50 precedes every late c50 in 1,000 of 1,000
  paired resamples (994 at f = .75; 417 at f = .25, 935 without the same-diagonal share, whose c25 is early; 2 at
  f = .1). It is an ordered sequence, not two clusters: the step at the boundary (x1.74) is about as large as steps
  within the families (up to x1.48; x3.53 at the six-step row).
  Partition. Only the two boundary rows can switch family without breaking the test (corner answer to late .996,
  same-diagonal share to early .897); moving any other row gives at most .102 (the distance out of a split, a move
  size, counted as splitting); counting the returns gives .111.
  Scale. The order of the point estimates holds on every re-expression tried; the size of the separation does not. On a
  log scale of every late row with positive references (a sensitivity check: logs magnify values near naive's floor; G
  and the split gradients have a negative naive level, so no log scale), the late rows are half-way at 1,762-6,363: all
  after the latest answer (1,575), but 8 of 10 at 1,762-2,272, overlapping the corner answer's interval; only the
  direction slope (5,236 [4,238-6,160]) and the angle out (6,363 [5,212-8,578]) stay clearly late (the jump angles:
  >= 4 cells 1,865 [1,354-2,460], excess 2,104 [1,710-2,938]). The paired test then gives .473 (late rows logged), .850
  (early and late rows logged where defined), .569 (held shares only), .739 (the jump rows only, both angles logged).
  Accuracy has closed .80-.94 of its gap at the additive late c50s, .68-.74 at the log ones (.88 / .92 at the
  direction slope / angle out).
  Onset (c10) against accuracy's 885 [847-952], additive scale: the jump-coupling rows start with accuracy (>= 4-cell
  angle 740 [505-1,080], excess angle 1,032 [735-1,684], G 1,046 [622-1,361]), the same-diagonal share before it (213
  [97-328]); the holding rows start after it (end-mass slope 1,166 [1,020-1,556], held per variable 1,547 [1,218-1,682]
  / 1,500 [1,115-1,903], distance out 1,584 [1,349-1,715], joint held 2,127 [1,930-2,509]), as do the direction slope
  3,255 [2,822-3,955] and the angle out 2,609 [2,250-3,036]. On a log scale the held shares and the end-mass slope start
  with accuracy (804-961); distance out (1,318 [1,150-1,559]), direction slope and angle out stay after. c0 is between
  340 and 960 in 17 of 26 rows, but for the holding rows it only marks leaving naive's near-zero floor.
  Width c75 / c25: answers x1.7-3.7; splitting x5.9-6.2 (six-step x39.5: its c25 is the first checkpoint); holding
  x2.5-6.6; coupling x5.0-8.6.
  Returns (not in the test): answer cells lost later back half-way at 1,998 [1,711-28,000], over-splitting at 2,935
  [1,697-4,454], committing to a corner at 18,673 [14,050-26,137].
  Loss per fifth (own resamples, not in the test): most entangled fifth 1,220 [1,167-1,285], least 2,202 [2,085-2,368].
  The fifth with 2.1x more naive loss crosses first; the network's own loss is higher on it at 113 of 114 checkpoints,
  so this is not faster learning there.
  ESN: no row is past half-way for good (26 rows; at most 0.1% of resamples cross), and no onset is established either.
  At the end it is at most .12 of the way on a rise row (same-diagonal share) and below naive's level on 13 of 26 (10
  with the 5-95% band below 0). The 6 rows past naive's level from the start are the untrained network's position (the
  splitting and fold rows; 2 returns that start at Bayes's level, of which .24 is what is left on over-splitting). It
  passes naive's level for good later on 7: excess angle 984 [377-4,171], same-diagonal share 3,207 [53-16,998], loss
  on the most entangled fifth 8,984 [7,213-10,811], G 18,253 [4,829-never] (83%), far errors 20,197 [11,611-37,779]
  (98%), distant pairs merged 20,250 [11,233-never] (71%), opposite corners 25,034 [2,244-never] (84%).
Caveats: one seed, one probe world; intervals resample environments (the learning rows: episodes), not seeds. Crossings
are limited by checkpoint spacing (the wormholes rows: 58 checkpoints). 8 of the 11 late rows are measured at Bayes's
events or regressed on Bayes's values; the three jump-coupling rows are not. Distant pairs merged: local answer jitter
can lower it, so its crossings are noise-proof only together with the far errors (truth-referenced; noise raises them),
which cross at the same time. Interpretation (not tested here): on the additive scale the coupled updates and held
splits are what remains between ~80% and 100% of Bayes's accuracy (on a log scale, from ~70%).

Run from the repository root (CPU, about a minute):
    MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/td_tt_timeline.py [numbers figure]"""
import os
import sys
import warnings
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402

TOPICS = ("jumps", "wormholes", "splitting", "latent", "entangled_learning")
FAM_COL = {"answers": "#6a3d9a", "splitting": "#01665e", "holding": "#3a7a19", "coupling": "#b2182b",
           "learning": "#8c510a"}
FAM_NAME = {"answers": "answers at T, end beliefs", "splitting": "splitting of near beliefs",
            "holding": "holding a split (and leaving it)", "coupling": "the two variables moving together",
            "learning": "training loss per entanglement fifth"}
REFINE = {"splitting": {"movement": "splitting"}, "wormholes": {"movement": "holding"}}    # by topic, not by timing
EARLY, LATE = ("answers", "splitting"), ("holding", "coupling")
COL = {"trained": "#1f77b4", "echo": "#ff7f0e"}
NAIVE_GREY = "0.72"                                       # heat maps: at or below naive's level (f <= 0)
JUMP_ROWS = ("jump excess angle over independent", "long-path coupling G", "angle of jumps >= 4 cells")
HELD_ROWS = ("corner split held (joint)", "held per variable, corner-split ends",
             "held per variable, one-variable end splits")
LOG_LATE = HELD_ROWS + ("distance out of a corner split", "same-diagonal share of moves out of corner splits")
XMIN = 8.
LEVELS = (.1, .25, .5, .75)
SHOWN = .95                                               # a crossing is drawn if >= 95% of the resamples cross


# ═══════════════════════════════════════════════════════════════════
# Rows, crossings, tests
# ═══════════════════════════════════════════════════════════════════

def load_rows():
    """every exported row: dict(topic, name, run, family, shape, sample, paired, note, x, value, frac, naive, bayes,
    c0..., c50..., boot (N_BOOT, K), naive_boot, bayes_boot)"""
    rows = []
    for t in TOPICS:
        z = gap.load(t)
        for i in range(len(z["names"])):
            r = {k: z[k][i] for k in z if k not in ("x", "n_boot", "seed", "names")}
            r.update(topic=t, name=str(z["names"][i]), x=z["x"])
            for k in ("run", "family", "shape", "sample", "note"):
                r[k] = str(r[k])
            r["family"] = REFINE.get(t, {}).get(r["family"], r["family"])
            r["paired"] = bool(r["paired"])
            rows.append(r)
    return rows


def fracs(r, scale="linear"):
    """(full-sample f (K,), resampled f (N_BOOT, K), valid resamples (N_BOOT,)) on the linear scale, or on the log scale
    (f of log y; values <= 0 give NaN = not past; a resample whose naive or Bayes level is <= 0 is invalid; None if a
    full-sample reference is <= 0)"""
    y, yb = np.asarray(r["value"], np.float64), np.asarray(r["boot"], np.float64)
    nv, bv = float(r["naive"]), float(r["bayes"])
    nb, bb = np.asarray(r["naive_boot"], np.float64), np.asarray(r["bayes_boot"], np.float64)
    valid = np.ones(len(nb), bool)
    if scale == "log":
        if min(nv, bv) <= 0:
            return None
        valid = (nb > 0) & (bb > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = lambda v: np.where(v > 0, np.log(np.where(v > 0, v, 1.)), np.nan)              # noqa: E731
            y, yb, nv, bv, nb, bb = lg(y), lg(yb), np.log(nv), np.log(bv), lg(nb), lg(bb)
    return gap.gap_fraction(y, nv, bv), gap.gap_fraction(yb, nb, bb), valid


def cross(r, level, scale="linear"):
    """dict(p, lo, hi, share, boot, valid) of the crossing of f = level (None if the scale does not apply); interval
    and share over the valid resamples"""
    f = fracs(r, scale)
    if f is None:
        return None
    cb = gap.crossing(r["x"], f[1], level)
    lo, hi, share = gap.interval(cb[f[2]])
    return dict(p=gap.crossing(r["x"], f[0], level), lo=lo, hi=hi, share=share, boot=cb, valid=f[2])


def annotate(rows):
    """adds r['L'] = {level: cross(r, level)} for LEVELS and 0 (linear scale)"""
    for r in rows:
        r["L"] = {lev: cross(r, lev) for lev in (0.,) + LEVELS}
    return rows


def blocks(rows):
    """[(key, [(Trained RNN row, its ESN row)])]: 'tested' (rise-shaped, paired, answers / splitting / holding /
    coupling; by c50), 'returns' (by c50), 'learning' (own resamples)"""
    es = {(r["topic"], r["name"]): r for r in rows if r["run"] == "echo"}
    tr = [r for r in rows if r["run"] == "trained"]
    key = lambda r: (np.nan_to_num(r["L"][.5]["p"], nan=1e9), np.nan_to_num(r["L"][.1]["p"], nan=1e9))  # noqa: E731
    out = []
    for name, sel in (("tested", lambda r: r["shape"] == "rise" and r["paired"] and r["family"] in EARLY + LATE),
                      ("returns", lambda r: r["shape"] == "return"),
                      ("learning", lambda r: r["family"] == "learning")):
        b = sorted([r for r in tr if sel(r)], key=key)
        out.append((name, [(r, es.get((r["topic"], r["name"]))) for r in b]))
    assert sum(len(b) for _, b in out) == len(tr), "every Trained RNN row in exactly one block"
    return out


def _inf(cb):
    return np.where(np.isnan(cb), np.inf, cb)


def separation(tested, fam=None, level=.5, scale_of=None):
    """share of paired resamples in which every early-family crossing precedes every late-family crossing.
    fam: {name: family} overrides; scale_of: {name: scale} re-expressions"""
    fam, scale_of = fam or {}, scale_of or {}
    A, B, ok = [], [], np.ones(gap.N_BOOT, bool)
    for r in tested:
        f = fam.get(r["name"], r["family"])
        c = cross(r, level, scale_of.get(r["name"], "linear")) if (level != .5 or r["name"] in scale_of) \
            else r["L"][.5]
        (A if f in EARLY else B).append(_inf(c["boot"]))
        ok &= c.get("valid", ok)
    return float((np.max(A, 0) < np.min(B, 0))[ok].mean())


def acc_frac_at(acc, xv):
    """the Trained RNN's accuracy gap fraction at training episode xv (log-interpolated, full sample)"""
    return float(np.interp(np.log(xv), np.log(acc["x"]), fracs(acc)[0]))


def _fmt(c):
    if c is None:
        return "n/a"
    if not np.isfinite(c["p"]):
        return f"never ({c['share']:.1%} of resamples cross)"
    hi = f"{c['hi']:,.0f}" if np.isfinite(c["hi"]) else "never"
    return f"{c['p']:,.0f} [{c['lo']:,.0f}-{hi}]" + ("" if c["share"] >= .995 else f" ({c['share']:.0%} cross)")


def _rel(c, ref):
    """'before' / 'after' / 'with' a reference crossing by the interval rule"""
    if c is None or not np.isfinite(c["p"]):
        return "-"
    lo, hi = c["lo"], c["hi"] if np.isfinite(c["hi"]) else np.inf
    rhi = ref["hi"] if np.isfinite(ref["hi"]) else np.inf
    return "after" if lo > rhi else "before" if hi < ref["lo"] else "with"


def summary(rows=None):
    """every number of the docstring and the figure text: dict"""
    rows = annotate(rows if rows is not None else load_rows())
    B = dict(blocks(rows))
    tested = [r for r, _ in B["tested"]]
    acc = next(r for r in tested if r["name"] == "goal accuracy at T")
    early = [r for r in tested if r["family"] in EARLY]
    late = [r for r in tested if r["family"] in LATE]
    S = dict(rows=rows, blocks=B, tested=tested, acc=acc, early=early, late=late)
    S["c50_check"] = max(float(np.nanmax(np.abs(np.nan_to_num(r["L"][.5]["boot"], nan=-1)
                                                - np.nan_to_num(r["c50_boot"], nan=-1)))) for r in rows)
    S["sep"] = {lev: separation(tested, level=lev) for lev in LEVELS}
    S["sep_variants"] = {
        "distance out counted as splitting": separation(tested, fam={"distance out of a corner split": "splitting"}),
        "end-mass slope counted as answers": separation(tested, fam={"end-mass slope within steps (maps)": "answers"}),
        "held shares on a log scale": separation(tested, scale_of={n: "log" for n in HELD_ROWS}),
        "held shares, distance out, same diagonal on a log scale":
            separation(tested, scale_of={n: "log" for n in LOG_LATE}),
        "late = the three jump-coupling rows only":
            separation([r for r in tested if r["family"] in EARLY or r["name"] in JUMP_ROWS])}
    ret = [r for r, _ in B["returns"]]
    A = [_inf(r["L"][.5]["boot"]) for r in early + [q for q in ret if q["family"] in EARLY]]
    Z = [_inf(r["L"][.5]["boot"]) for r in late + [q for q in ret if q["family"] in LATE]]
    S["sep_variants"]["returns included (their own family)"] = float((np.max(A, 0) < np.min(Z, 0)).mean())
    logable = lambda r: min(float(r["naive"]), float(r["bayes"])) > 0                              # noqa: E731
    S["log50"] = {r["name"]: cross(r, .5, "log") for r in late if logable(r)}
    S["log10"] = {r["name"]: (cross(r, .1, "log"), _rel(cross(r, .1, "log"), acc["L"][.1])) for r in late
                  if logable(r)}
    S["no_log"] = [r["name"] for r in tested if not logable(r)]
    S["sep_variants"]["every late row with positive references on a log scale"] = separation(
        tested, scale_of={r["name"]: "log" for r in late if logable(r)})
    S["sep_variants"]["early and late rows on a log scale where defined"] = separation(
        tested, scale_of={r["name"]: "log" for r in tested if logable(r)})
    S["sep_variants"]["the three jump-coupling rows only, both angles on a log scale"] = separation(
        [r for r in tested if r["family"] in EARLY or r["name"] in JUMP_ROWS],
        scale_of={n: "log" for n in JUMP_ROWS if n != "long-path coupling G"})
    S["moves"] = {r["name"]: separation(tested, fam={r["name"]: (LATE[0] if r["family"] in EARLY else EARLY[0])})
                  for r in tested}
    S["sep25_wo_diag"] = separation([r for r in tested if not r["name"].startswith("same-diagonal")], level=.25)
    S["acc_at_log"] = {n: acc_frac_at(acc, c["p"]) for n, c in S["log50"].items() if np.isfinite(c["p"])}
    S["late_lo_log"] = min(c["p"] for c in S["log50"].values())
    fold = next(r for r in tested if r["name"].startswith("fold index"))
    S["fold_f0"] = float(fracs(fold)[0][0])
    S["onset"] = {r["name"]: (r["L"][.1], _rel(r["L"][.1], acc["L"][.1])) for r in tested}
    S["width"] = {r["name"]: (r["L"][.75]["p"] / r["L"][.25]["p"],
                              np.nanpercentile(r["L"][.75]["boot"] / r["L"][.25]["boot"], [5, 95]))
                  for r in tested}
    S["acc_at"] = {r["name"]: acc_frac_at(acc, r["L"][.5]["p"]) for r in late}
    e50 = [r["L"][.5]["p"] for r in early]
    l50 = [r["L"][.5]["p"] for r in late]
    S["ranges"] = dict(early=(min(e50), max(e50)), late=(min(l50), max(l50)),
                       early_wo_six=(min(r["L"][.5]["p"] for r in early if not r["name"].startswith("six-step")),
                                     max(e50)))
    srt = sorted(r["L"][.5]["p"] for r in tested)
    gaps = np.array(srt[1:]) / np.array(srt[:-1])
    S["boundary_gap"] = min(l50) / max(e50)
    S["gaps"] = gaps
    ESN = [(r, e) for b in B.values() for r, e in b if e is not None]
    S["esn"] = []
    for r, e in ESN:
        f, fb, _ = fracs(e)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            band = np.nanpercentile(fb[:, -1], [5, 95])
        S["esn"].append(dict(name=r["name"], shape=r["shape"], end=float(f[-1]), band=band,
                             c0=cross(e, 0.), c10=cross(e, .1), c50=cross(e, .5), start=bool(f[0] > 0 and
                             np.isfinite(e["L"][0.]["p"]) and e["L"][0.]["p"] <= e["x"][0])))
    return S


def numbers(rows=None):
    S = summary(rows)
    B, acc = S["blocks"], S["acc"]
    print(f"check: c50 from the stored resampled curves = the exported c50 per resample (max |diff| "
          f"{S['c50_check']:.2g} training episodes)")
    print("== crossings of the gap fraction, Trained RNN (training episodes; [5-95%] over 1,000 resamples)")
    for name, b in B.items():
        print(f"  -- block {name}")
        for r, e in b:
            L = r["L"]
            print(f"  {r['family']:9s} {r['topic'][:10]:10s} {r['name'][:50]:50s} naive {r['naive']:+.4g} "
                  f"Bayes {r['bayes']:+.4g}{'' if r['paired'] else '  (own units)'}")
            print(f"      c0 {_fmt(L[0.])}   c10 {_fmt(L[.1])}   c50 {_fmt(L[.5])}")
            print(f"      c25 {_fmt(L[.25])}   c75 {_fmt(L[.75])}")
    print("== the learning rows' export notes (td_tt_entangled_learning)")
    for r, e in B["learning"]:
        for q in (r, e):
            if q is not None:
                print(f"  {q['run']:8s} {q['name']}: {';'.join(q['note'].split(';')[:2])}")
    print(f"== accuracy: c10 {_fmt(acc['L'][.1])}, c50 {_fmt(acc['L'][.5])}")
    c0s = [r["L"][0.]["p"] for b in B.values() for r, _ in b]
    print(f"== c0 between 340 and 960 in {sum(340 <= c <= 960 for c in c0s)} of {len(c0s)} rows; at the first "
          f"checkpoint in {sum(c <= 10 for c in c0s)}")
    print("== onset (c10) of the tested rows relative to accuracy's (interval rule)")
    for n, (c, rel) in S["onset"].items():
        print(f"  {n[:50]:50s} c10 {_fmt(c):32s} {rel}")
    lo, hi = S["ranges"]["early"], S["ranges"]["late"]
    print(f"== c50 ranges: early (answers, splitting) {lo[0]:,.0f}-{lo[1]:,.0f} (without the six-step move "
          f"{S['ranges']['early_wo_six'][0]:,.0f}); late (holding, coupling) {hi[0]:,.0f}-{hi[1]:,.0f}")
    print(f"== boundary step x{S['boundary_gap']:.2f}; consecutive c50 steps of the tested rows: "
          + " ".join(f"x{g:.2f}" for g in S["gaps"]))
    print("== family test (tested rows): share of paired resamples in which every early crossing precedes every "
          "late one")
    for lev, v in S["sep"].items():
        print(f"  f = {lev}: {v:.3f}")
    for k, v in S["sep_variants"].items():
        print(f"  c50, {k}: {v:.3f}")
    print("== moving one row to the other family (c50 test): share")
    for n, v in S["moves"].items():
        print(f"  {n[:50]:50s} {v:.3f}")
    print(f"== f = .25 without the same-diagonal share: {S['sep25_wo_diag']:.3f}")
    print(f"== log-scale sensitivity, late rows (no log scale, a reference <= 0: {', '.join(S['no_log'])})")
    for n, c in S["log50"].items():
        lin = next(r for r in S["late"] if r["name"] == n)["L"][.5]
        c10, rel = S["log10"][n]
        print(f"  {n[:50]:50s} c50 linear {_fmt(lin):26s} log {_fmt(c):26s} | log c10 {_fmt(c10)} ({rel} accuracy)")
    print(f"  latest early c50 {S['ranges']['early'][1]:,.0f}; earliest late log c50 {S['late_lo_log']:,.0f}")
    print("  accuracy gap fraction at the log c50s: " + ", ".join(f"{v:.3f}" for v in S["acc_at_log"].values()))
    print(f"== fold index: the untrained network's f = {S['fold_f0']:.2f}")
    print("== width c75 / c25 (full sample [paired 5-95%])")
    for r in S["tested"]:
        w, (a, b) = S["width"][r["name"]]
        print(f"  {r['family']:9s} {r['name'][:50]:50s} x{w:.1f} [{a:.1f}-{b:.1f}]")
    for fam in EARLY + LATE:
        w = [S["width"][r["name"]][0] for r in S["tested"] if r["family"] == fam]
        print(f"  {fam}: x{min(w):.1f}-x{max(w):.1f}")
    print("== accuracy gap fraction at each late c50 (linear scale)")
    for n, v in S["acc_at"].items():
        print(f"  {n[:50]:50s} {v:.3f}")
    print("== ESN (end f [5-95%]; crossings)")
    for d in S["esn"]:
        print(f"  {d['shape']:6s} {d['name'][:50]:50s} end {d['end']:+.3f} [{d['band'][0]:+.3f}, {d['band'][1]:+.3f}]"
              f"  c0 {_fmt(d['c0'])}{' (past from the start)' if d['start'] else ''}  c10 {_fmt(d['c10'])}  "
              f"c50 {_fmt(d['c50'])}")
    rise = [d for d in S["esn"] if d["shape"] == "rise"]
    below = [d for d in S["esn"] if d["end"] < 0]
    print(f"  c50 never in {sum(not np.isfinite(d['c50']['p']) for d in S['esn'])} of {len(S['esn'])} rows (largest "
          f"share of resamples crossing {max(d['c50']['share'] for d in S['esn']):.1%}); end f max over rise rows "
          f"{max(d['end'] for d in rise):+.3f}, over all {max(d['end'] for d in S['esn']):+.3f}; below naive at the "
          f"end in {len(below)} rows ({sum(d['band'][1] < 0 for d in below)} with the band below 0); past naive from "
          f"the start in {sum(d['start'] for d in S['esn'])}")
    return S


# ═══════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════

def _tex(name):
    """>= / <= as mathtext (cmr10 has no glyph for them)"""
    return name.replace(">=", r"$\geq$").replace("<=", r"$\leq$")


def figure(rows=None):
    """-> figure_bin/td_tt_timeline.png"""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import matplotlib.transforms as mtr
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    FL, FT, TS = fs["label"], fs["label"], 12.
    S = summary(rows)
    B, acc = S["blocks"], S["acc"]
    GAP = .9
    ypos, y = {}, 0.
    for bi, (name, b) in enumerate(B.items()):
        y += GAP if bi else 0.
        for r, _ in b:
            ypos[(r["topic"], r["name"])] = y
            y += 1.
    ytop, ybot = -.5, y - .5
    xmax = max(float(np.exp(np.log(r["x"][-1]) + (np.log(r["x"][-1]) - np.log(r["x"][-2])) / 2)) for r in S["rows"])
    RH = .27
    PW, LAB, GX, CB, RM = 3.6, 3.15, .45, .55, 2.45
    TOP, BOT = 1.6, 3.05
    PH = (ybot - ytop) * RH
    W = LAB + 3 * PW + 2 * GX + CB + RM
    H = TOP + PH + BOT
    fig = plt.figure(figsize=(W, H))
    xs = [LAB, LAB + PW + GX, LAB + 2 * PW + 2 * GX + CB]
    axes = [fig.add_axes([x0 / W, BOT / H, PW / W, PH / H]) for x0 in xs]
    cmap = plt.cm.viridis.copy()
    cmap.set_under(NAIVE_GREY)
    cmap.set_bad("w")
    halo = [pe.withStroke(linewidth=2.6, foreground="k")]

    def xaxis(ax):
        ax.set_xscale("log")
        ax.set_xlim(XMIN, xmax)
        ax.set_xticks([10, 1e2, 1e3, 1e4])
        ax.set_xticklabels(["0", "100", "1k", "10k"])
        ax.minorticks_off()
        ax.set_xlabel("training episodes", fontsize=FL)
        ax.set_ylim(ybot, ytop)
        ax.set_yticks([])

    def mark(ax, yy, c, color, filled, marker, heat):
        if c is None or not np.isfinite(c["p"]) or c["p"] <= XMIN * 1.3 or c["share"] < SHOWN:
            return                                    # never, past from the start, or not established in the resamples
        lo = c["lo"] if np.isfinite(c["lo"]) else c["p"]
        hi = c["hi"] if np.isfinite(c["hi"]) else xmax
        ax.plot([lo, hi], [yy] * 2, color=("w" if heat else color), lw=2.2 if filled else 1.2, zorder=6,
                solid_capstyle="butt", path_effects=halo if heat else None)
        ax.plot([c["p"]], [yy], marker=marker, ms=6.5 if filled else 5.5, mfc=(color if filled else "w"),
                mec=("k" if marker == "o" else color), mew=1.1, zorder=7, ls="none")

    im = None
    for j, run in enumerate(("trained", "echo")):
        ax = axes[j]
        for b in B.values():
            for r, e in b:
                q = r if run == "trained" else e
                if q is None:
                    continue
                yy = ypos[(r["topic"], r["name"])]
                im = ax.pcolormesh(_edges(q["x"]), [yy - .5, yy + .5], fracs(q)[0][None], cmap=cmap, vmin=0.,
                                   vmax=1., shading="flat", rasterized=True)
                Lq = q["L"] if "L" in q else {lev: cross(q, lev) for lev in (.1, .5)}
                mark(ax, yy, Lq[.1], COL[run], False, "o" if run == "trained" else "s", True)
                mark(ax, yy, Lq[.5], COL[run], True, "o" if run == "trained" else "s", True)
        xaxis(ax)
    n50 = sum(np.isfinite(d["c50"]["p"]) for d in S["esn"])
    n_es = sum((np.isfinite(d[k]["p"]) and d[k]["p"] > XMIN * 1.3 and d[k]["share"] >= SHOWN)
               for d in S["esn"] for k in ("c10", "c50"))
    axes[0].set_title("a   Trained RNN: gap closed, $f$\n"
                      r"$f = (y - y_{\mathrm{naive}})\,/\,(y_{\mathrm{Bayes}} - y_{\mathrm{naive}})$",
                      loc="left", fontsize=TS, linespacing=1.5)
    axes[1].set_title(f"b   ESN: gap closed, $f$\npast $f = 0.5$ for good in {n50} of {len(S['esn'])} rows",
                      loc="left", fontsize=TS, linespacing=1.5)
    for b in B.values():
        for r, _ in b:
            lab = _tex(r["name"]) + ("" if r["shape"] == "rise" else "  (returns)")
            axes[0].text(-.02, ypos[(r["topic"], r["name"])], lab, transform=axes[0].get_yaxis_transform(),
                         ha="right", va="center", fontsize=FT - .3, color=FAM_COL[r["family"]])
    cax = fig.add_axes([(xs[1] + PW + .17) / W, BOT / H + .55 * PH / H, .14 / W, .4 * PH / H])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_ticks([0, .5, 1])
    cb.set_ticklabels(["0 naive", ".5", "1 Bayes"])
    cb.ax.tick_params(labelsize=FT - .5)
    cb.ax.set_title("$f$", fontsize=FT)
    cax.text(.5, -.09, r"$\leq$ naive", transform=cax.transAxes, ha="center", va="top", fontsize=FT - 1.5)
    ax = axes[2]
    for b in B.values():
        for i, (r, e) in enumerate(b):
            yy = ypos[(r["topic"], r["name"])]
            if i % 2 == 0:
                ax.axhspan(yy - .5, yy + .5, color="#f7f3e8", lw=0, zorder=0)
            mark(ax, yy - .14, r["L"][.1], FAM_COL[r["family"]], False, "o", False)
            mark(ax, yy - .14, r["L"][.5], FAM_COL[r["family"]], True, "o", False)
            if e is not None:
                mark(ax, yy + .2, cross(e, .1), COL["echo"], False, "s", False)
                mark(ax, yy + .2, cross(e, .5), COL["echo"], True, "s", False)
    xaxis(ax)
    for v, ls in ((acc["L"][.1]["p"], ":"), (acc["L"][.5]["p"], (0, (5, 2, 1, 2)))):
        ax.axvline(v, color="k", lw=1.1, ls=ls, zorder=4)
    ax.set_title("c   Crossings: Trained RNN" + (" (circles), ESN (squares)" if n_es else " (ESN: none established)")
                 + "\n"
                 r"open: $f > 0.1$ for good (onset);  filled: $f > 0.5$ for good",
                 loc="left", fontsize=TS, linespacing=1.5)
    # the family boundary inside the tested block, and the block labels
    tested = B["tested"]
    yl = max(ypos[(r["topic"], r["name"])] for r, _ in tested if r["family"] in EARLY)
    yf = min(ypos[(r["topic"], r["name"])] for r, _ in tested if r["family"] in LATE)
    tr_ = mtr.blended_transform_factory(ax.transAxes, ax.transData)
    if yl < yf:
        for a_ in axes:
            a_.axhline((yl + yf) / 2, color="k", lw=1.4, ls=(0, (6, 3)), zorder=8,
                       path_effects=[pe.withStroke(linewidth=3.4, foreground="w")])
        e0, e1 = S["ranges"]["early"]
        l0, l1 = S["ranges"]["late"]
        ax.text(1.03, (yl + yf) / 2 - .35, f"answers, splitting:\nhalf-way at {e0:,.0f}-{e1:,.0f}", transform=tr_,
                ha="left", va="bottom", fontsize=FT - .5, linespacing=1.35)
        ax.text(1.03, (yl + yf) / 2 + .35, f"holding, coupling:\nhalf-way at {l0:,.0f}-{l1:,.0f}", transform=tr_,
                ha="left", va="top", fontsize=FT - .5, linespacing=1.35)
    for name, lab in (("returns", "returns: first away from\nBayes past naive, then back;\nnot in the family test"),
                      ("learning", "loss per fifth (own resamples):\nnot in the family test; the\nfifth with more "
                                   "naive loss\ncrosses first")):
        if not B[name]:
            continue
        ys = [ypos[(r["topic"], r["name"])] for r, _ in B[name]]
        for a_ in axes:
            a_.axhline(min(ys) - .5 - GAP / 2, color="k", lw=.8, zorder=8)
        ax.text(1.03, min(ys) - .4, lab, transform=tr_, ha="left", va="top", fontsize=FT - 1, linespacing=1.3)
    h1 = [Line2D([], [], color="k", marker="o", ms=5.5, mfc="w", mec="k", lw=1.2, label=r"onset: $f > 0.1$ for good"),
          Line2D([], [], color="k", marker="o", ms=6.5, mfc="k", mec="k", lw=2.2,
                 label=r"half-way: $f > 0.5$ for good"),
          Line2D([], [], color="k", lw=1.1, ls=":", label=f"Trained RNN accuracy: onset ({acc['L'][.1]['p']:,.0f})"),
          Line2D([], [], color="k", lw=1.1, ls=(0, (5, 2, 1, 2)),
                 label=f"... and half-way ({acc['L'][.5]['p']:,.0f})"),
          Line2D([], [], color=COL["echo"], marker="s", ms=5.5, mfc="w", mec=COL["echo"], lw=1.2,
                 label=("ESN: no onset or half-way established in any row" if n_es == 0 else "ESN")),
          Line2D([], [], color="k", lw=1.4, ls=(0, (6, 3)),
                 label=f"every tested row above the dashed line is half-way before every\ntested row below it "
                       f"({S['sep'][.5] * 1000:,.0f} of 1,000 paired resamples; additive scale)")]
    h2 = [Patch(color=FAM_COL[f], label=FAM_NAME[f]) for f in FAM_COL]
    l1 = fig.legend(handles=h1, loc="center", bbox_to_anchor=(.42, 2.3 / H), ncol=3, frameon=False, fontsize=FT,
                    columnspacing=1.8, handlelength=2.4)
    fig.add_artist(l1)
    fig.legend(handles=h2, loc="center", bbox_to_anchor=(.42, 1.62 / H), ncol=5, frameon=False, fontsize=FT,
               columnspacing=1.6, handlelength=1.4, title="label colour = family", title_fontsize=FT)
    lg = [c["p"] for c in S["log50"].values()]
    sv = S["sep_variants"]
    note = (
        "Data (seed 20261029): probes = 10,000 episodes (50 test environments x 100 latent pairs x 2 trajectories, "
        "30 steps), 114 checkpoints (jumps, splitting, latent; distant pairs merged: 18,432 probe pairs per "
        "environment).\n"
        "Wormholes, 58 checkpoints: forward set = 5,028 replayed episodes from the same 50 environments (corner "
        "answers 2,400; splits 3,859 stream-steps, 7,718 / 4,763 var-steps; 2,204 moves out of splits); maps on the "
        "probes (580,000 variable-steps; 4,823 moves).\n"
        "Entangled learning: 3,000 training-distribution episodes, 114 checkpoints, resampled by episode.   Bars: "
        "5-95% over 1,000 resamples of the 50 environments, the same for every probe-based row (paired); within a "
        "block, rows are sorted by point estimate.\n"
        "Heat maps: grey = at or below naive's level; yellow = Bayes's or past it; white = no value (fewer than 50 "
        "jumps of 4+ cells). Early bright patches fall back: crossings count only levels kept to the end, drawn if at "
        f"least 95% of the resamples cross. Fold index: the untrained network is at f = {S['fold_f0']:.2f}.\n"
        f"Scale: f is additive. On a log scale (sensitivity check) the late rows are half-way at {min(lg):,.0f}-"
        f"{max(lg):,.0f}: after the answers in point estimates, but mostly overlapping them (paired test "
        f"{sv['every late row with positive references on a log scale']:.2f}-"
        f"{sv['early and late rows on a log scale where defined']:.2f}). At 25% of the gap the families interleave "
        f"({S['sep'][.25]:.2f}).")
    fig.text(.5, .58 / H, note, ha="center", va="center", fontsize=FT - 1, color="k", linespacing=1.55)
    e0, e1 = S["ranges"]["early"]
    l0, l1_ = S["ranges"]["late"]
    fig.suptitle(f"The answers at T and the splitting of near beliefs are half-way to Bayes by {e1:,.0f} training "
                 f"episodes; holding a split and moving the two variables together follow\n({l0:,.0f}-{l1_:,.0f} on "
                 f"the additive gap scale, {min(lg):,.0f}-{max(lg):,.0f} on a log scale)."
                 + (" The ESN is never past half-way for good." if n50 == 0 else ""), y=1 - .45 / H,
                 fontsize=TS + 1.5, linespacing=1.4)
    path = fig_path("td_tt_timeline.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


def _edges(x):
    lx = np.log(x)
    mid = (lx[1:] + lx[:-1]) / 2
    return np.exp(np.concatenate([[lx[0] - (mid[0] - lx[0])], mid, [lx[-1] + (lx[-1] - mid[-1])]]))


if __name__ == "__main__":
    what = sys.argv[1:] or ["numbers", "figure"]
    R = load_rows()
    if "numbers" in what:
        numbers(R)
    if "figure" in what:
        figure(R)
