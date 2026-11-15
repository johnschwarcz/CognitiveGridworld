"""The embedding-transfer theory in one inequality and one picture (John 11-14: "simplify and compress the theory as much as
possible"). Transfer starts when a variable has enough training partners for its embedding dimension and for the spread of z:

    M  >=  d_E sinh(x) / x,      x = 8 pi^2 z0^2      (nothing fitted)

M = |S| - |S| // 10 training partners, z0^2 = Var z (1 / d_E in the original world, 1/30 or 1/20 in the scaled runs), so x
is twice the variance of the phase 2 pi z. sinh(x) / x = 1 + rho_mf: the variance of a pair's pooled likelihood read-out over
the part of it that is a straight line in z (zscale_proxy.rho_mf). Reading: each partner gives one equation for the d_E
unknown coordinates of a variable's embedding, of which only the fraction x / sinh(x) is usable; the usable equations must
cover the unknowns. It is the point where least squares on M noisy equations predicts a new pair's z better than ignoring z
(error variance rho d_E / (M - d_E) below the signal's). Why the network obeys an out-of-sample criterion is not derived.
In the original world the need is d_E^2 sinh(8 pi^2 / d_E) / (8 pi^2): least at d_E ~ 41 (72 partners, |S| ~ 80), exploding
below (the fold noise) and growing as d_E above (the unknowns).

The figure: every run's latest n against lambda = M / (d_E sinh(x) / x), partners available over partners needed. The law
says n ~ 0 left of lambda = 1 and transfer right of it. Prints the runs on the wrong side.
Earlier, longer forms of the theory (the proxy r with a measured level, zscale_proxy.py; the bracket, zscale_bootstrap.py)
give the same thresholds at z0^2 = 1/30 and are not needed for this picture.

    python zscale_collapse.py [--pull] [--logs DIR] [--out PATH] [--running NAME,NAME,...] [--stamp TEXT]
"""
import os, sys, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zscale_progress as zp
import zscale_simple as zs

CUT, SETTLED = .2, 3000
SURFACE, INK, INK2, HAIR = zs.SURFACE, zs.INK, zs.INK2, zs.HAIR


def need(d, ref):
    """training partners needed: d_E sinh(x) / x with x = 8 pi^2 z0^2, z0^2 = 1 / ref"""
    x = 8 * np.pi ** 2 / ref
    return d * np.sinh(x) / x


def score(runs, running=()):
    """(right, wrong, [(lambda, key, n, episodes)] of the wrong ones) over runs past SETTLED episodes or far above the line"""
    right, odd = 0, []
    for (d, ref, S, seed), (ep, n, name) in sorted(runs.items()):
        lam = (S - S // 10) / need(d, ref)
        if ep[-1] >= SETTLED or n[-1] >= .5:
            if (n[-1] >= CUT) == (lam >= 1):
                right += 1
            else:
                odd.append((lam, (d, ref, S, seed), n[-1], ep[-1]))
    return right, len(odd), odd


def plot(runs, out, running, stamp=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from main.utils import apply_fig_style
    fs = apply_fig_style()
    right, wrong, odd = score(runs, running)
    fig, a = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True, facecolor=SURFACE)
    a.set_facecolor(SURFACE)
    for y in (0, CUT, 1):
        a.axhline(y, color=INK2 if y == CUT else HAIR, lw=.9 if y == CUT else .6, zorder=0)
    a.axvline(1, color=INK, lw=1.4, zorder=1)
    a.text(1.06, 1.12, "enough partners", ha="left", va="center", color=INK, fontsize=fs["tick"] + 1)
    a.text(.94, 1.12, "too few", ha="right", va="center", color=INK, fontsize=fs["tick"] + 1)
    for (d, ref, S, seed), (ep, n, name) in sorted(runs.items()):
        lam, col = (S - S // 10) / need(d, ref), zs.colour(d, ref)
        live = name in running or (ep[-1] < SETTLED and n[-1] < .5)
        a.plot(lam, n[-1], "o", ms=5.5, mfc=SURFACE if live else col, mec=col, mew=1.2, alpha=.85, zorder=3)
    cells = {}                                               # name the cells with runs on the wrong side, once each
    for lam, (d, ref, S, seed), n, ep in odd:
        cells.setdefault((lam, d, ref, S), []).append(n)
    spots = [((-14, 30), "right"), ((14, -26), "left"), ((0, 24), "center"), ((14, 30), "left")]   # clear of the n = .2 line
    for k, ((lam, d, ref, S), ns) in enumerate(sorted(cells.items())):
        world = "unscaled " if ref == d and d != 30 else "scaled " if d != 30 else ""
        total = sum(key[:3] == (d, ref, S) for key in runs)
        (dx, dy), ha = spots[k % len(spots)]
        a.annotate(rf"{world}$d_E$ {d}, $|\mathcal{{S}}|$ {S}: {len(ns)} of {total} run{'s' if total > 1 else ''}",
                   (lam, float(np.mean(ns))), xytext=(dx, dy), textcoords="offset points", ha=ha, color=INK2,
                   fontsize=fs["tick"], arrowprops=dict(arrowstyle="-", color=INK2, lw=.6))
    a.set_xscale("log")
    a.set_xlim(.02, 45)
    a.set_ylim(-.34, 1.2)                                    # room under the floor for the labels
    a.set_xticks([.03, .1, .3, 1, 3, 10, 30], ["0.03", "0.1", "0.3", "1", "3", "10", "30"])
    a.set_xticks([], minor=True)
    a.set_yticks([0, CUT, .5, 1], ["0", f"{CUT:g}", ".5", "1"])
    a.tick_params(colors=INK2, length=2)
    for s in a.spines.values():
        s.set_color(HAIR)
    a.set_xlabel(r"partners available / partners needed:   $M \,/\, (d_E \sinh x / x)$,   $x = 8\pi^2 z_0^2$", color=INK)
    a.set_ylabel("transfer $n$ at the latest test\n(0 = ignores z, 1 = Joint Bayes)", color=INK)
    handles = [Line2D([], [], color=zs.BLUE, marker="o", ls="none", label=r"z scaled, $z_0^2 = 1/30$"),
               Line2D([], [], color=zs.GREEN, marker="o", ls="none", label=r"unscaled, $z_0^2 = 1/d_E$"),
               Line2D([], [], color=INK, marker="o", ls="none", label=r"$d_E = 30$ (both)"),
               Line2D([], [], color=zs.ORANGE, marker="o", ls="none", label=r"z scaled, $z_0^2 = 1/20$"),
               Line2D([], [], color=INK2, marker="o", mfc=SURFACE, ls="none", label="still training")]
    if not any(k[1] == 20 and k[0] != 20 for k in runs):
        handles.pop(3)
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=fs["legend"])
    fig.suptitle("One inequality, nothing fitted: transfer when a variable has enough training partners"
                 f"\n{len(runs)} runs ({stamp or time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}); "
                 f"{right} of {right + wrong} on the law's side", fontsize=fs["label"] + 1, color=INK)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"figure -> {out}")
    return right, wrong, odd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--logs", default=zs.LOGS)
    ap.add_argument("--out", default=os.path.join(ROOT, "main", "REPLICATION", "figure_bin", "zscale_collapse.png"))
    ap.add_argument("--running", default="", help="names of the runs still training, comma-separated")
    ap.add_argument("--stamp", default="", help="the time of the data, for the title (default: now, UTC)")
    a = ap.parse_args()
    if a.pull:
        zp.pull(a.logs)
    runs = zs.load(a.logs)
    right, wrong, odd = plot(runs, a.out, set(filter(None, a.running.split(","))), a.stamp)
    print(f"the law M >= d_E sinh(x)/x: {right} of {right + wrong} runs on its side")
    for lam, (d, ref, S, seed), n, ep in sorted(odd):
        print(f"  off: d_E {d}, z0^2 = 1/{ref}, |S| {S}, seed {seed}: lambda {lam:.2f}, n {n:.2f} at episode {ep}")
    worlds = sorted({k[:2] for k in runs})
    print("need (partners) by world: " + "; ".join(f"d_E {d}, z0^2 1/{ref}: {need(d, ref):.0f}" for d, ref in worlds))


if __name__ == "__main__":
    main()
