"""fr_theory_optimum.png: is there a number of observables at which factorizing costs the most, and what sets it?
John 11-11: "more observables quickly become intractable, but easy to simulate, meanwhile many observables are trivial (easy for
bayes). I wonder if we could show the intractability (combinatorial explosion) vs triviality (reduction in trapping) to show
that there is an optimum, and what it depends on."

For the project's generator (two distinct variables, test-set sampler) with R realizations per variable and d_o observables, at
likelihood temperature lambda, per interaction set:
  from the tables alone (any d_o, no simulation)   q = trapped fraction; E[Delta] = q * Delta_trap, the long-run growth rate of
                                                   E[FR_t] (Prop. 1c); Delta_trap = mean excess mismatch of the trapped r
  simulated, from the counts at t (exact observers) accuracy of the Joint and the Naive observer and FR_t at t = 10, 30, 100
The combinatorial explosion in d_o is a property of exact EXPECTATIONS over observation sequences ((t + 1)^d_o count vectors),
an analysis tool; the observers themselves cost R^C d_o per step, linear in d_o.

usage:  python fr_theory_optimum.py --data R   (one realization count per process; caches in DATA/experimental/fr_theory/optimum)
        python fr_theory_optimum.py [--numbers]  (figure from the caches)"""
import os
import sys
import time
import numpy as np
from scipy.special import logsumexp, xlogy

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import fr_theory as F                                                             # noqa: E402
import fr_theory_observables as FO                                                # noqa: E402

OUT = os.path.join(ROOT, "main", "DATA", "experimental", "fr_theory", "optimum")
RS, DOS, LAMS, TS = (3, 5, 10), (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30), (1.0, 2.0, 3.0), (10, 30, 100)
NZ, NEP = 150, 100                                # interaction sets per point, episodes per set


def episodes(ell, rng, n_ep, ts):
    """Exact Joint and Naive observers from the counts at each t in ts: accuracy (goal answer = argmax) and FR_t, per episode."""
    d_o, R, Rc = ell.shape; m = ell.mean(axis=2); T = max(ts)
    rg, rc = rng.integers(0, R, n_ep), rng.integers(0, Rc, n_ep)
    o = rng.random((T, n_ep, d_o)) < ell[:, rg, rc].T[None]
    cum = np.cumsum(o, axis=0, dtype=np.float64)
    l1, l0, n1, n0 = np.log(ell), np.log1p(-ell), np.log(m), np.log1p(-m); out = {}
    for t in ts:
        n = cum[t - 1]                                                             # (n_ep, d_o) counts of ones after t steps
        LJ = np.einsum("ei,ijk->ejk", n, l1) + np.einsum("ei,ijk->ejk", t - n, l0)
        lB = logsumexp(LJ, axis=2); lB -= logsumexp(lB, axis=1, keepdims=True)
        LN = n @ n1 + (t - n) @ n0; lN = LN - logsumexp(LN, axis=1, keepdims=True)
        out[t] = (lB.argmax(1) == rg, lN.argmax(1) == rg, np.sum(np.exp(lB) * (lB - lN), axis=1))
    return out


def build(R):
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); shape = (len(LAMS), len(DOS), NZ)
    Z = dict(q=np.zeros(shape), ed=np.zeros(shape), accJ=np.zeros(shape + (len(TS),)), accN=np.zeros(shape + (len(TS),)), fr=np.zeros(shape + (len(TS),)))
    for j, d_o in enumerate(DOS):
        for li, lam in enumerate(LAMS):
            rng = np.random.default_rng(10_000 * R + 100 * d_o)                    # the same interaction sets at every lambda
            erng = np.random.default_rng(7 + 1000 * R + d_o + 100_000 * li)
            for z in range(NZ):
                ell = F.checked_table(FO.table(rng, lam, d_o, R), lam); tab = F.one_step_table(ell)
                Z["q"][li, j, z] = tab["trapped"].mean(); Z["ed"][li, j, z] = tab["Delta"].mean()
                ep = episodes(ell, erng, NEP, TS)
                for ti, t in enumerate(TS):
                    aJ, aN, fr = ep[t]; Z["accJ"][li, j, z, ti] = aJ.mean(); Z["accN"][li, j, z, ti] = aN.mean(); Z["fr"][li, j, z, ti] = fr.mean()
        print(f"R = {R}, d_o = {d_o} done ({time.time() - t0:.0f} s)", flush=True)
    np.savez(os.path.join(OUT, f"optimum_R{R}.npz"), **Z)


def load():
    return {R: dict(np.load(os.path.join(OUT, f"optimum_R{R}.npz"))) for R in RS if os.path.exists(os.path.join(OUT, f"optimum_R{R}.npz"))}


def numbers(Zs):
    for R, Z in Zs.items():
        for li, lam in enumerate(LAMS):
            ed = Z["ed"][li].mean(1); q = Z["q"][li].mean(1); gap = (Z["accJ"][li] - Z["accN"][li]).mean(1); fr = Z["fr"][li].mean(1)
            print(f"R = {R:2d} lambda {lam:g}: E[Delta] peak at d_o = {DOS[ed.argmax()]} ({ed.max():.4f}); q {q[0]:.2f} -> {q[-1]:.2f}; "
                  f"accuracy gap at t = 30 peak at d_o = {DOS[gap[:, 1].argmax()]} ({gap[:, 1].max():.3f}); FR_30 peak at d_o = {DOS[fr[:, 1].argmax()]} ({fr[:, 1].max():.2f}); "
                  f"Joint accuracy at t = 30: d_o = 1 {Z['accJ'][li, 0, :, 1].mean():.2f}, 30 {Z['accJ'][li, -1, :, 1].mean():.2f}")


def peak(x, y):
    """location of the maximum of y over x (log scale): a parabola in log x through the grid maximum and its neighbours"""
    i = int(np.argmax(y)); lx = np.log(np.asarray(x, float))
    if i == 0 or i == len(y) - 1: return float(x[i])
    a, b, _ = np.polyfit(lx[i - 1:i + 2], y[i - 1:i + 2], 2); return float(np.exp(-b / (2 * a))) if a < 0 else float(x[i])


def figure(Zs):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from main.utils import fig_path
    plt.rcParams.update(F.RC)
    cols = {3: "#63a6ff", 5: "#2469be", 10: "#00336d"}; x = np.array(DOS); li = LAMS.index(2.0); ti = TS.index(30)
    fig, axs = plt.subplots(1, 3, figsize=(9.0, 2.3)); a, b, c = axs
    for R, Z in Zs.items():
        g = Z["accJ"][li, :, :, ti] - Z["accN"][li, :, :, ti]; m, se = g.mean(1), g.std(1) / np.sqrt(g.shape[1])
        a.fill_between(x, m - 2 * se, m + 2 * se, color=cols[R], alpha=0.2, lw=0); a.plot(x, m, color=cols[R], lw=1.2, label=f"$R={R}$")
        pk = peak(x, m); a.plot([pk], [np.interp(np.log(pk), np.log(x), m)], ls="none", marker="v", ms=4.5, mfc=cols[R], mec="white", mew=0.5, zorder=5)
    G = Zs[10]; a.plot([5], [(G["accJ"][li, DOS.index(5), :, ti] - G["accN"][li, DOS.index(5), :, ti]).mean()], ls="none", marker="o", ms=4.5,
                       mfc="white", mec=cols[10], mew=1.0, zorder=6)
    a.set_title(r"(a) Joint $-$ Naive accuracy at $t=30$", loc="left"); a.set_ylabel("accuracy gap"); a.legend(loc="upper left", title="realizations")
    Z = Zs[10]; aj, an_, q = Z["accJ"][li, :, :, ti].mean(1), Z["accN"][li, :, :, ti].mean(1), Z["q"][li].mean(1)
    b.plot(x, aj, color=F.INK, lw=1.2, label="Joint, $t=30$"); b.plot(x, an_, color=F.ORANGE, lw=1.2, label="Naive, $t=30$")
    b.plot(x, 1 - q, color=F.ORANGE, lw=1.0, ls=(0, (3, 2)), label=r"Naive, $t\to\infty$: $1-q$")
    b.set_ylim(0, 1.02); b.set_ylabel("accuracy"); b.set_title(r"(b) Information vs. traps ($R=10$)", loc="left"); b.legend(loc="lower right")
    lam = np.array(LAMS)
    for R, Z in Zs.items():
        pk = [peak(x, (Z["accJ"][k, :, :, ti] - Z["accN"][k, :, :, ti]).mean(1)) for k in range(len(LAMS))]
        c.plot(lam, pk, color=cols[R], lw=1.2, marker="v", ms=4.5, mfc=cols[R], mec="white", mew=0.5, label=f"$R={R}$")
    c.set_yscale("log"); c.set_yticks([1, 2, 5, 10, 30]); c.set_yticklabels(["1", "2", "5", "10", "30"]); c.set_ylim(1, 35)
    c.set_xticks(lam); c.set_xlabel(r"likelihood temperature $\lambda$"); c.set_ylabel(r"observables at the peak of (a)")
    c.set_title("(c) Where the gap peaks", loc="left"); c.legend(loc="upper right", title="realizations")
    for ax in (a, b):
        ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 30]); ax.set_xticklabels(["1", "2", "5", "10", "30"]); ax.set_xlim(1, 30)
        ax.set_xlabel(r"observables $d_o$"); ax.axvline(5, color=F.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    fig.text(0.995, 0.005, "The project's generator, two distinct variables; exact Joint and Naive observers on simulated episodes; "
             rf"{NZ} interaction sets x {NEP} episodes per point." + "\n" + r"$\lambda=2$ in (a, b); band: $\pm$2 s.e. over interaction sets; "
             r"triangles: peaks; circle and dotted line: the Gridworld ($R=10$, $d_o=5$).", ha="right", va="bottom", fontsize=7, color=F.INK2,
             linespacing=1.3)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.5)
    p = fig_path("fr_theory_optimum.png"); fig.savefig(p, dpi=200); print("saved", p)


if __name__ == "__main__":
    if "--data" in sys.argv:
        build(int(sys.argv[sys.argv.index("--data") + 1]))
    else:
        Zs = load()
        if "--numbers" in sys.argv: numbers(Zs)
        figure(Zs)
