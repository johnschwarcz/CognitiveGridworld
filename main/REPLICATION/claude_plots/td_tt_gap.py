"""Shared conventions of the through-training synthesis (td_tt_timeline.py): one set of environment resamples for every
probe-based measure (so crossings of different topics are paired), the stays-past crossing, and the export that each
topic module (td_tt_<topic>.py, command `synthesis`) writes to through_training/<topic>/synthesis.npz.

Gap fraction of a measure y at a checkpoint: f = (y - y_naive) / (y_Bayes - y_naive), with naive's and Bayes's levels
on the same episodes: 0 = naive's level, 1 = Bayes's, rising toward Bayes whatever the sign of the measure. In every
resample, f uses that resample's naive and Bayes levels. Crossings (crossing()): the training episode after which f
stays above a level to the end of training, log-interpolated between the last checkpoint not past it and the next one
(NaN if f ends below it): c0 = passes naive's level for good, c50 = past the naive-Bayes midpoint for good.
Intervals: 5-95% over the N_BOOT resamples; a resample that never crosses counts as after the end of training (an
open upper bound is NaN).

Resamples: boot_counts() = (N_BOOT, 50) counts of the 50 test environments drawn with replacement (seed 20261103; the
same matrix as td_tt_jumps.boot_counts). Probe-based measures use exactly these counts, with environment e = probe
episode // 200 (probes are environment-major). Episode sets from the same 50 environments in the same order use them
too (paired=True). Other sets resample their own units (paired=False) and cannot be ordered pairwise against the rest.

export(topic, x, rows) row fields:
  name, run ('trained' | 'echo'), family ('answers' | 'movement' | 'coupling' | 'learning'), shape ('rise' = moves
  from naive's side to Bayes's; 'return' = first moves away from Bayes past naive's level, then comes back),
  sample (text: episodes, environments), paired (bool), note (text, '' if none),
  value (K,) the full-sample curve at the topic's checkpoints x (K,), boot (N_BOOT, K) the same curve in every
  resample, naive / bayes (floats, full sample), naive_boot / bayes_boot (N_BOOT,) (pass a constant array if a
  reference cannot be resampled with the same units, and say so in the note).
Written arrays: names, run, family, shape, sample, paired, note (M,); x (K,); value, lo, hi (M, K) with lo/hi the 5-95%
pointwise band of the curve; naive, bayes (M,); frac, frac_lo, frac_hi (M, K); c0, c0_lo, c0_hi, c0_p and c50, c50_lo,
c50_hi, c50_p (M,) (point, interval, share of resamples that cross); c0_boot, c50_boot (M, N_BOOT); boot (M, N_BOOT, K)
float64 (float32 rounding moves late crossings of curves that hover at a level), naive_boot, bayes_boot (M, N_BOOT): the resampled curves and references themselves, so that any other level or
re-expression can be timed on the same resamples (td_tt_timeline); n_boot, seed."""
import os
import warnings
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
TT = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics", "through_training")
N_ENV, PER_ENV = 50, 200
N_BOOT = 1000
BOOT_SEED = 20261103
X0 = 10                                                # the initial network is drawn at 10 training episodes
FIELDS = ("name", "run", "family", "shape", "sample", "paired", "note", "value", "boot", "naive", "bayes",
          "naive_boot", "bayes_boot")
FAMILIES = ("answers", "movement", "coupling", "learning")
SHAPES = ("rise", "return")


def train_x(ep):
    """training episodes of raw checkpoint episodes (every 5th episode is a test episode), the initial network at X0"""
    ep = np.asarray(ep)
    return np.maximum(ep - ep // 5, X0).astype(float)


def boot_counts():
    """(N_BOOT, 50): how often each environment is drawn in each resample (50 draws with replacement)"""
    rng = np.random.default_rng(BOOT_SEED)
    return rng.multinomial(N_ENV, np.full(N_ENV, 1 / N_ENV), size=N_BOOT).astype(np.float64)


def env_mean(v, C=None):
    """an environment-mean statistic: v (50, ...) per-environment values -> (full-sample mean (...), resampled
    (N_BOOT, ...)); environments without a value (NaN) are left out of that mean"""
    C = boot_counts() if C is None else C
    v = np.asarray(v, np.float64)
    ok = np.isfinite(v)
    vz = np.where(ok, v, 0.)
    full = vz.sum(0) / ok.sum(0)
    flat = vz.reshape(N_ENV, -1)
    okf = ok.reshape(N_ENV, -1).astype(np.float64)
    return full, ((C @ flat) / (C @ okf)).reshape((len(C),) + v.shape[1:])


def env_ratio(num, den, C=None):
    """a ratio of sums: num, den (50, ...) per-environment sums -> (full-sample ratio, resampled (N_BOOT, ...))"""
    C = boot_counts() if C is None else C
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    nf, df = num.reshape(N_ENV, -1), den.reshape(N_ENV, -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num.sum(0) / den.sum(0), ((C @ nf) / (C @ df)).reshape((len(C),) + num.shape[1:])


def crossing(x, y, level, sign=1.):
    """the training episode after which y stays past `level` (sign +1: above it, -1: below it) to the end of training,
    log-interpolated between the last checkpoint not past it and the next one; NaN if y ends not past it, x[0] if it is
    past it at every checkpoint. y (..., K) with level a scalar or of shape (...); a checkpoint without a value (NaN)
    counts as not past (same as td_tt_jumps.crossing)"""
    y = np.asarray(y, np.float64)
    shp = y.shape[:-1]
    y2 = y.reshape(-1, y.shape[-1])
    lev = np.broadcast_to(np.asarray(level, np.float64), shp).reshape(-1)
    with np.errstate(invalid="ignore"):
        past = sign * (y2 - lev[:, None]) > 0
    notp = ~past
    last = y2.shape[1] - 1 - np.argmax(notp[:, ::-1], 1)
    out = np.full(len(y2), np.nan)
    out[past.all(1)] = x[0]
    sel = np.where(past[:, -1] & notp.any(1))[0]
    k = last[sel]
    y0, y1 = y2[sel, k], y2[sel, k + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(np.isfinite(y0), (lev[sel] - y0) / (y1 - y0), 1.)
    f = np.clip(np.nan_to_num(f, nan=1.), 0, 1)
    lx = np.log(np.asarray(x, np.float64))
    out[sel] = np.exp(lx[k] + f * (lx[k + 1] - lx[k]))
    return out.reshape(shp) if shp else float(out[0])


def interval(cb):
    """(5%, 95%, share crossing) of resampled crossings; a resample that never crosses counts as after the end of
    training, so an open bound is NaN"""
    v = np.where(np.isnan(cb), np.inf, cb)
    lo, hi = np.percentile(v, 5, method="lower"), np.percentile(v, 95, method="higher")
    return (float(lo) if np.isfinite(lo) else np.nan), (float(hi) if np.isfinite(hi) else np.nan), \
        float(np.isfinite(v).mean())


def gap_fraction(y, naive, bayes):
    """(y - naive) / (bayes - naive); y (..., K) with naive / bayes scalars or of shape (...)"""
    naive, bayes = np.asarray(naive, np.float64), np.asarray(bayes, np.float64)
    return (np.asarray(y, np.float64) - naive[..., None]) / (bayes - naive)[..., None]


def _row(x, r):
    value, boot = np.asarray(r["value"], np.float64), np.asarray(r["boot"], np.float64)
    nb, bb = np.asarray(r["naive_boot"], np.float64), np.asarray(r["bayes_boot"], np.float64)
    assert value.shape == (len(x),) and boot.shape == (N_BOOT, len(x)), (r["name"], value.shape, boot.shape)
    assert nb.shape == (N_BOOT,) and bb.shape == (N_BOOT,), r["name"]
    assert r["family"] in FAMILIES and r["shape"] in SHAPES and r["run"] in ("trained", "echo"), r["name"]
    f = gap_fraction(value, r["naive"], r["bayes"])
    fb = gap_fraction(boot, nb, bb)
    out = {k: r[k] for k in ("name", "run", "family", "shape", "sample", "note")}
    out.update(paired=bool(r["paired"]), value=value, naive=float(r["naive"]), bayes=float(r["bayes"]), frac=f,
               boot=boot, naive_boot=nb, bayes_boot=bb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                               # all-NaN columns
        out["lo"], out["hi"] = np.nanpercentile(boot, [5, 95], 0)
        out["frac_lo"], out["frac_hi"] = np.nanpercentile(fb, [5, 95], 0)
    for name, lev in (("c0", 0.), ("c50", .5)):
        cb = crossing(x, fb, lev)
        lo, hi, p = interval(cb)
        out.update({name: crossing(x, f, lev), f"{name}_lo": lo, f"{name}_hi": hi, f"{name}_p": p, f"{name}_boot": cb})
    return out


def export(topic, x, rows):
    """-> through_training/<topic>/synthesis.npz (see the module docstring); prints one line per row"""
    x = np.asarray(x, np.float64)
    assert x[0] >= X0 and np.all(np.diff(x) > 0), "x: training episodes, increasing, the initial network at X0"
    for r in rows:
        miss = [k for k in FIELDS if k not in r]
        assert not miss, (r.get("name"), miss)
    R = [_row(x, r) for r in rows]
    out = {k: np.array([r[k] for r in R]) for k in R[0]}
    out["names"] = out.pop("name")
    out.update(x=x, n_boot=N_BOOT, seed=BOOT_SEED)
    path = os.path.join(TT, topic, "synthesis.npz")
    np.savez(path, **out)
    print("saved", path)
    for r in R:
        print(f"  {r['run']:8s} {r['name'][:44]:44s} {r['family']:9s} naive {r['naive']:+.4g} Bayes {r['bayes']:+.4g}"
              f"  c0 {r['c0']:,.0f} [{r['c0_lo']:,.0f}-{r['c0_hi']:,.0f}] ({r['c0_p']:.0%})"
              f"  c50 {r['c50']:,.0f} [{r['c50_lo']:,.0f}-{r['c50_hi']:,.0f}] ({r['c50_p']:.0%})")
    return path


def load(topic):
    """the export of one topic as a dict of arrays"""
    with np.load(os.path.join(TT, topic, "synthesis.npz"), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
