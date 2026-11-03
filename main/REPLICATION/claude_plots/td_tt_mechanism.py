"""How the network gets the answers before Bayes-like dynamics: the neural changes through training against the
behavioural timeline (John 11-02: "link the neural properties to the clear behavioral developments through training").
A synthesis of verified exports only; nothing is recomputed except the coherence-half-way event across runs (the same
rule as td_tt_neural.cross_runs, reproduced: 767 / 2,527 / 5,975 / 11,608).
-> main/REPLICATION/figure_bin/td_tt_mechanism.png   figure()
   numbers()   every number below

Sources: td_tt_neural (synthesis.npz: the Trained RNN's and ESN's neural rows, events with 5-95% over the shared
resamples), td_tt_replication (synthesis.npz: every trained run's behaviour and neural rows), td_tt_jumps (synthesis.npz:
goal accuracy), td_tt_timeline (the behavioural families' half-way ranges).
Definitions: as in td_tt_neural; a crossing is the training episode after which the curve stays past the level to the
end (td_tt_gap.crossing), or the first time for rise-and-fall curves (labelled); intervals over 1,000 resamples of the
50 test environments. Weight rows (W_hh block radius) have no probe noise: point times with checkpoint brackets.
Accuracy gap fraction f = (accuracy - naive) / (Bayes - naive).

Results: see numbers().

Run from the repository root (CPU, seconds):
    MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 /home/johns/anaconda3/envs/eureka/bin/python \\
        main/REPLICATION/claude_plots/td_tt_mechanism.py [numbers figure]"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402

NEU = os.path.join(gap.TT, "neural", "synthesis.npz")
REP = os.path.join(gap.TT, "replication", "synthesis.npz")
GCOL = {"memory": "#6a3d9a", "activity": "#01665e", "credit": "#b2182b", "weights": "#8c510a"}
RUNS = (("trained", "N = 1000"), ("trained_p1", "N = 1000, power 1"), ("trained_N200", "N = 200"),
        ("trained_N100", "N = 100"), ("trained_N50", "N = 50"))
RMARK = {"trained": "o", "trained_p1": "D", "trained_N200": "s", "trained_N100": "^", "trained_N50": "v"}
EARLY_SPAN, LATE_SPAN = (774, 1575), (2747, 11103)        # td_tt_timeline: half-way ranges of the two groups of families
# a: (row key, event label, short text); one event per line, in time order after sorting
EVENTS = (("r_dim", "half-way to 0", "the update decouples from input novelty"),
          ("turn_u", "half its change", "the state starts to turn between steps"),
          ("r_coh14", "half its peak, first", "errors 1-4 steps after entangled steps get coherent credit"),
          ("pr_env", "10% of its change", "the state's dimension starts to grow"),
          ("rho_o", "radius > 1 for good", r"$W_{hh}$ o-block becomes expansive"),
          ("rho_f", "radius > 1 for good", r"$W_{hh}$ f-block becomes expansive"),
          ("coh", "half-way to 1", "credit coherence half-way to 1"),
          ("pr_env", "half its change", "the state's dimension half-way"),
          ("ben5x", "half its peak, first", "reliance on memory older than 5 steps (beyond Bayes's) rises"),
          ("mu_top", "memory helps (> 0) for good", "memory helps the step update (most entangled fifth)"),
          ("mu_all", "memory helps (> 0) for good", "memory helps the step update (all steps)"),
          ("tau10", "10% of its change", "second wave of long time constants starts"),
          ("coh", "below 1 for good (pieces cancel)", "credit pieces cancel (coherence below 1)"),
          ("r_coh0", "10% of its end value", "the learning signal locks onto entangled steps"),
          ("ben5x", "back below half its peak for good", "reliance on long memory falls back"),
          ("r_coh0", "half its end value", "... the lock half-way"),
          ("tau10", "half its change", "long time constants half-way"))
PACED = {("coh", "half-way to 1"): "keeps pace with passing naive across widths (x.92-1.22)",
         ("coh", "below 1 for good (pieces cancel)"): "at the same accuracy across widths (f .74 / .77)",
         ("r_dim", "half-way to 0"): "same time in every width (97-262)",
         ("rho_o", "radius > 1 for good"): "near-fixed time across widths"}


def neural():
    z = np.load(NEU, allow_pickle=True)
    return {k: z[k] for k in z.files}


def rep():
    z = np.load(REP, allow_pickle=True)
    return {k: z[k] for k in z.files}


def event(N, key, lab, run="trained"):
    """(p, lo, hi, share, group, row index) of one event of the neural export"""
    i = [j for j in range(len(N["key"])) if N["key"][j] == key and N["run"][j] == run][0]
    js = [j for j in np.where(N["ev_row"] == i)[0] if str(N["ev_lab"][j]).startswith(lab)]
    if not js:
        raise KeyError((key, lab))
    j = js[0]
    return dict(p=float(N["ev_p"][j]), lo=float(N["ev_lo"][j]), hi=float(N["ev_hi"][j]), share=float(N["ev_share"][j]),
                group=str(N["group"][i]), row=i, rule=str(N["ev_rule"][j]))


def row(N, key, run="trained"):
    i = [j for j in range(len(N["key"])) if N["key"][j] == key and N["run"][j] == run][0]
    return dict(x=N["x"], value=N["value"][i], lo=N["lo"][i], hi=N["hi"][i], b=float(N["b"][i]), name=str(N["name"][i]))


def acc_frac(R, run):
    """(x, f, f resampled) of a run's goal accuracy at T, f = (acc - naive) / (Bayes - naive), per resample"""
    x = R[f"{run}|x"]
    v, b = R[f"{run}|acc|value"], R[f"{run}|acc|boot"]
    nv, bv = float(R[f"{run}|acc|naive"]), float(R[f"{run}|acc|bayes"])
    nb, bb = R[f"{run}|acc|naive_boot"], R[f"{run}|acc|bayes_boot"]
    return x, (v - nv) / (bv - nv), (b - nb[:, None]) / (bb - nb)[:, None]


def rep_event(R, run, key, lab):
    if f"{run}|{key}|events" not in R:                                             # e.g. no credit keys for p1
        return None
    evs = [str(e) for e in R[f"{run}|{key}|events"]]
    hit = [i for i, e in enumerate(evs) if e.startswith(lab)]
    if not hit:
        return None
    i = hit[0]
    g = lambda f: R[f"{run}|{key}|ev{i}|{f}"]                                       # noqa: E731
    return dict(p=float(g("p")), lo=float(g("lo")), hi=float(g("hi")), share=float(g("share")), boot=g("boot"))


def coh_half(R, run):
    """credit coherence half-way from its initial value to 1 (stays below), as td_tt_neural.cross_runs"""
    if f"{run}|coh|value" not in R:
        return None
    x, v, b = R[f"{run}|x"], R[f"{run}|coh|value"], R[f"{run}|coh|boot"]
    cb = gap.crossing(x, b, (b[:, 0] + 1) / 2, -1.)
    lo, hi, share = gap.interval(cb)
    return dict(p=gap.crossing(x, v, (v[0] + 1) / 2, -1.), lo=lo, hi=hi, share=share, boot=cb)


def f_at(R, run, ev):
    """the run's accuracy gap fraction at an event (log-interpolated), full sample and per resample (paired)"""
    x, f, fb = acc_frac(R, run)
    lx = np.log(x)
    full = float(np.interp(np.log(ev["p"]), lx, f))
    if ev.get("boot") is None:
        return full, np.nan, np.nan
    vals = np.array([np.interp(np.log(t), lx, fb[k]) for k, t in enumerate(ev["boot"]) if np.isfinite(t)])
    lo, hi = np.percentile(vals, [5, 95]) if len(vals) else (np.nan, np.nan)
    return full, float(lo), float(hi)


CROSS = (("coh_half", "credit coherence half-way to 1", "#b2182b", "o"),
         ("mem1", "memory benefit at T reaches Bayes's", "#6a3d9a", "s"),
         ("mem5", "benefit of memory older than 5 steps rises", "#9e7cc1", "D"),
         ("r_dim", "update decouples from novelty", "#e08a8f", "v"),
         ("rho", r"$W_{hh}$ expansive (radius above 1)", "#8c510a", "^"),
         ("coh_lt1", "credit pieces cancel (coherence below 1)", "#67000d", "P"))


def cross_events(R):
    """{marker: {run: event}} for the cross-run panels"""
    out = {}
    for key, *_ in CROSS:
        out[key] = {}
        for run, _ in RUNS:
            if key == "coh_half":
                e = coh_half(R, run)
            elif key == "coh_lt1":
                e = rep_event(R, run, "coh", "below 1")
            elif key == "mem1":
                e = rep_event(R, run, "mem1", "Bayes's level")
            elif key == "mem5":
                e = rep_event(R, run, "mem5", "rise")
            elif key == "r_dim":
                e = rep_event(R, run, "r_dim", "half-way")
            else:
                e = rep_event(R, run, "rho", "> 1")
            if e is not None and np.isfinite(e["p"]):
                out[key][run] = e
    return out


def numbers():
    N, R = neural(), rep()
    print("== a: neural events, Trained RNN (training episodes, [5-95%])")
    for key, lab, text in EVENTS:
        e = event(N, key, lab)
        hi = f"{e['hi']:,.0f}" if np.isfinite(e["hi"]) else "never"
        print(f"  {e['group']:8s} {text[:58]:58s} {e['p']:7,.0f} [{e['lo']:,.0f}-{hi}]  ({key}: {lab})")
    print("== c, d: across runs (passing naive = accuracy f past 0 for good)")
    CE = cross_events(R)
    for run, name in RUNS:
        pn = rep_event(R, run, "acc", "0.0")
        line = f"  {name:18s} passes naive {pn['p']:7,.0f}"
        for key, lab, *_ in CROSS:
            e = CE[key].get(run)
            if e is None:
                line += f" | {key} -"
                continue
            fv = f_at(R, run, e)[0]
            line += f" | {key} {e['p']:,.0f} (x{e['p'] / pn['p']:.2f}; f {fv:+.2f})"
        print(line)


def figure():
    """-> figure_bin/td_tt_mechanism.png"""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    FL, FT, TS = fs["label"], fs["label"], 12.
    N, R = neural(), rep()
    acc = gap.load("jumps")
    ia = list(acc["names"]).index("goal accuracy at T")
    acc0, acc50 = float(acc["c0"][ia]), float(acc["c50"][ia])
    W, H = 19.5, 12.6
    fig = plt.figure(figsize=(W, H))
    xmin, xmax = 8., 4.04e4

    def xaxis(ax, label=True):
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_xticks([10, 1e2, 1e3, 1e4])
        ax.set_xticklabels(["0", "100", "1k", "10k"])
        ax.minorticks_off()
        if label:
            ax.set_xlabel("training episodes", fontsize=FL)

    def spans(ax, labels=False):
        ax.axvspan(*EARLY_SPAN, color="#ece4f5", lw=0, zorder=0)
        ax.axvspan(*LATE_SPAN, color="#e3f0dc", lw=0, zorder=0)
        ax.axvline(acc0, color="k", lw=1.1, ls=":", zorder=1)
        if labels:
            tr = ax.get_xaxis_transform()
            ax.text(np.sqrt(EARLY_SPAN[0] * EARLY_SPAN[1]), 1.005, "answers, splitting\nhalf-way", transform=tr,
                    ha="center", va="bottom", fontsize=FT - 1, color="#6a3d9a", linespacing=1.2)
            ax.text(np.sqrt(LATE_SPAN[0] * LATE_SPAN[1]), 1.005, "holding, coupling\nhalf-way", transform=tr,
                    ha="center", va="bottom", fontsize=FT - 1, color="#3a7a19", linespacing=1.2)
    # a: the events
    ax = fig.add_axes([5.0 / W, 5.75 / H, 7.0 / W, 5.2 / H])
    ev = sorted(((event(N, k, lab), k, lab, text) for k, lab, text in EVENTS), key=lambda t: t[0]["p"])
    for i, (e, k, lab, text) in enumerate(ev):
        c = GCOL[e["group"]]
        hi = e["hi"] if np.isfinite(e["hi"]) else xmax
        ax.plot([e["lo"], hi], [i, i], color=c, lw=2.4, solid_capstyle="butt", zorder=3)
        ax.plot([e["p"]], [i], marker="o", ms=6.5, color=c, mec="k", mew=.8, zorder=4)
        ax.text(xmin * .9, i, text, transform=ax.transData, ha="right", va="center", fontsize=FT - .5, color=c)
        note = PACED.get((k, lab))
        if note:
            ax.text(min(hi, xmax) * 1.12, i, note, ha="left", va="center", fontsize=FT - 1.5, color="k",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="w")], zorder=5)
    spans(ax, labels=True)
    xaxis(ax)
    ax.set_ylim(len(ev) - .4, -.6)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(r"a   Neural changes in the Trained RNN, in time order (5-95%; $W_{hh}$: checkpoint bracket)",
                 loc="left", fontsize=TS, pad=30)
    # b: three curves on a shared x
    xs = N["x"]
    b_axes = [fig.add_axes([13.3 / W, (8.95 - 2.45 * j) / H, 5.6 / W, 2.05 / H]) for j in range(3)]
    ia_ = list(acc["names"]).index("goal accuracy at T")
    ie_ = [i for i in range(len(acc["names"])) if acc["names"][i] == "goal accuracy at T" and acc["run"][i] == "echo"][0]
    for a_, i_, col in ((b_axes[0], ia_, "#1f77b4"), (b_axes[0], ie_, "#ff7f0e")):
        a_.plot(acc["x"], acc["value"][i_], color=col, lw=2, zorder=3)
        a_.fill_between(acc["x"], acc["lo"][i_], acc["hi"][i_], color=col, alpha=.2, lw=0)
    b_axes[0].axhline(float(acc["bayes"][ia_]), color="k", lw=1.1)
    b_axes[0].axhline(float(acc["naive"][ia_]), color="0.5", lw=1.1)
    b_axes[0].text(11, float(acc["bayes"][ia_]) + .01, "Bayes", fontsize=FT - 1.5, va="bottom")
    b_axes[0].text(11, float(acc["naive"][ia_]) + .01, "naive", fontsize=FT - 1.5, va="bottom", color="0.4")
    b_axes[0].set_ylabel("goal accuracy\nat T", fontsize=FL)
    b_axes[0].set_title("b   The answers lean on long memory before the step update is Bayes-like", loc="left",
                        fontsize=TS, pad=8)
    for run, col in (("trained", "#1f77b4"), ("echo", "#ff7f0e")):
        r_ = row(N, "ben5x", run)
        b_axes[1].plot(xs, r_["value"], color=col, lw=2, zorder=3)
        b_axes[1].fill_between(xs, r_["lo"], r_["hi"], color=col, alpha=.2, lw=0)
    b_axes[1].axhline(0, color="k", lw=.8)
    b_axes[1].set_ylabel("memory older than 5\nsteps: benefit\nminus Bayes's", fontsize=FL)
    for key, ls in (("mu_top", "-"), ("mu_all", (0, (4, 2)))):
        for run, col in (("trained", "#1f77b4"), ("echo", "#ff7f0e")):
            r_ = row(N, key, run)
            b_axes[2].plot(xs, r_["value"], color=col, lw=2 if run == "trained" else 1.3, ls=ls, zorder=3)
            if run == "trained":
                b_axes[2].fill_between(xs, r_["lo"], r_["hi"], color=col, alpha=.15, lw=0)
    b_axes[2].axhline(0, color="k", lw=.8)
    lo_ = min(float(np.nanmin(row(N, k, r)["lo"])) for k in ("mu_top", "mu_all") for r in ("trained", "echo"))
    hi_ = max(float(np.nanmax(row(N, k, r)["hi"])) for k in ("mu_top", "mu_all") for r in ("trained", "echo"))
    b_axes[2].set_ylim(lo_ - .1 * (hi_ - lo_), hi_ + .1 * (hi_ - lo_))
    b_axes[2].set_ylabel("memory use in the\nstep update (nats)", fontsize=FL)
    b_axes[2].legend(handles=[Line2D([], [], color="k", lw=2, label="most entangled fifth"),
                              Line2D([], [], color="k", lw=2, ls=(0, (4, 2)), label="all steps")],
                     loc="upper left", frameon=False, fontsize=FT - 1.5)
    for j, a_ in enumerate(b_axes):
        spans(a_)
        xaxis(a_, label=j == 2)
        a_.spines[["top", "right"]].set_visible(False)
        if j < 2:
            a_.set_xticklabels([])
    # c: pacing across runs
    CE = cross_events(R)
    ax = fig.add_axes([1.0 / W, 1.15 / H, 5.6 / W, 3.25 / H])
    pn = {run: rep_event(R, run, "acc", "0.0")["p"] for run, _ in RUNS}
    for key, lab, col, mk in CROSS:
        pts = [(pn[run], CE[key][run]["p"]) for run, _ in RUNS if run in CE[key]]
        if pts:
            xx, yy = zip(*sorted(pts))
            ax.plot(xx, yy, color=col, lw=1.2, marker=mk, ms=6, mec="k", mew=.6, label=lab)
    lim = (8., 6e4)
    ax.plot(lim, lim, color="k", lw=.9, ls=(0, (4, 3)))
    ax.text(3.5e4, 2.2e4, "same time as\npassing naive", fontsize=FT - 2, ha="right", va="top", rotation=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(6e2, 3e4)
    ax.set_ylim(*lim)
    ax.set_xlabel("passing naive (training episodes)", fontsize=FL)
    ax.set_ylabel("neural event (training episodes)", fontsize=FL)
    ax.spines[["top", "right"]].set_visible(False)
    for run, name in RUNS:
        ax.annotate(name, (pn[run], lim[0] * 1.05), fontsize=FT - 2.5, ha="center", va="bottom", rotation=90,
                    color="0.25")
    ax.set_title("c   Which changes keep pace with the answers? (5 trained runs)", loc="left", fontsize=TS)
    # d: accuracy at the event across runs
    ax = fig.add_axes([8.4 / W, 1.15 / H, 4.4 / W, 3.25 / H])
    ax.patch.set_visible(False)
    yrun = {run: j for j, (run, _) in enumerate(RUNS)}
    off = {k: (i - (len(CROSS) - 1) / 2) * .12 for i, (k, *_) in enumerate(CROSS)}
    for key, lab, col, mk in CROSS:
        for run, name in RUNS:
            e = CE[key].get(run)
            if e is None:
                continue
            fv, lo, hi = f_at(R, run, e)
            y = yrun[run] + off[key]
            if np.isfinite(lo):
                ax.plot([lo, hi], [y, y], color=col, lw=1.8, zorder=2)
            ax.plot([fv], [y], marker=mk, ms=6, color=col, mec="k", mew=.6, ls="none", zorder=3)
    ax.axvline(0, color="0.5", lw=1, ls=":")
    ax.text(0.01, -.75, "naive", fontsize=FT - 2, color="0.4")
    ax.set_yticks(range(len(RUNS)))
    ax.set_yticklabels([n for _, n in RUNS], fontsize=FT - 1)
    ax.set_ylim(len(RUNS) - .4, -.9)
    ax.set_xlabel(r"accuracy gap fraction $f$ at the event", fontsize=FL)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("d   At what accuracy? (5-95%)", loc="left", fontsize=TS)
    # legend for a and footnote
    h = [Patch(color=GCOL[g], label=g) for g in GCOL]
    h += [Patch(color="#ece4f5", label="answers, splitting half-way (774-1,575)"),
          Patch(color="#e3f0dc", label="holding, coupling half-way (2,747-11,103)"),
          Line2D([], [], color="k", lw=1.1, ls=":", label=f"Trained RNN passes naive ({acc0:,.0f})"),
          Line2D([], [], color="#1f77b4", lw=2, label="Trained RNN (band 5-95%)"),
          Line2D([], [], color="#ff7f0e", lw=2, label="ESN")]
    fig.legend(handles=h, loc="center", bbox_to_anchor=(16.4 / W, 2.75 / H), ncol=2, frameon=False, fontsize=FT - .5,
               title="a, b", title_fontsize=FT, columnspacing=1.2)
    h = [Line2D([], [], color=col, lw=1.2, marker=mk, ms=6, mec="k", mew=.6, label=lab) for _, lab, col, mk in CROSS]
    fig.legend(handles=h, loc="center", bbox_to_anchor=(16.4 / W, 1.25 / H), ncol=2, frameon=False, fontsize=FT - .5,
               title="c, d", title_fontsize=FT, columnspacing=1.2, handlelength=1.6)
    fig.text(.5, .35 / H, "Data: 10,000 probe episodes (50 test environments x 200) per checkpoint, 114 checkpoints per "
             "network (seed 20261029); credit rows on 5,000 probes x 14 error steps; c, d: the five trained runs on the "
             "same probes (power 1 run: no credit record, the credit harvest assumes loss power .5; coherence never "
             "drops below 1 in N = 100, 50).\nIntervals: 1,000 resamples of the 50 environments (paired across rows). "
             f"Accuracy half-way to Bayes at {acc50:,.0f}. Sources: the neural-timeline, replication, memory-at-events "
             "and behavioural-timeline analyses (each independently verified).", ha="center", va="center",
             fontsize=FT - 1.5,
             linespacing=1.5)
    fig.suptitle("Before the answers: the learning signal and the state reorganise. With the answers: credit coherence "
                 "and a reliance on long memory.\nAfter them, toward Bayes-like dynamics: memory helps the step update, "
                 "credit pieces cancel, long time constants, entanglement-locked learning", y=1 - .32 / H,
                 fontsize=TS + 1, linespacing=1.4)
    path = fig_path("td_tt_mechanism.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)
    return path


if __name__ == "__main__":
    what = sys.argv[1:] or ["numbers", "figure"]
    if "numbers" in what:
        numbers()
    if "figure" in what:
        figure()
