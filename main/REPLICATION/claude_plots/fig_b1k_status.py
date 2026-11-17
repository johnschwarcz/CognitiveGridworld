"""Where the batch-1,000 route runs stand (John 11-13: "show me where the 1K batch runs currently stand"; the batch-1,000
version is the first candidate for the new Fig 13, the batch-10,000 one only if batch 1,000 is not usable); since 11-14 00:35
the batch-10,000 runs are drawn in the left column for comparison; since 11-14 02:10 the shuffled-channel runs
(batch 1,000, readin_perm, both routes) in a third column; since 11-14 06:00 the same without the K.Q bottleneck (z_raw) in a
fourth.

One panel = fig_batch_routes.py panel b, top row: the RCC's own feedback, lr 1e-4, batch 1,000, the Classifier's goal
accuracy at T on 1,000 novel-environment episodes per test (one test per 250 iterations), one line per network, x =
training iterations. Generator route (blue) runs pause once 3 consecutive tests are above that test's Naive accuracy
(train_cluster_runs.PAUSE_AT_ESCAPE) until the Classifier routes finish; Classifier route (red) runs go to 1,000,000
iterations. Seeds 0-5 local (tasks 240-251 -> 290-301 -> R / R2 resumes), seeds 6-11 on the cluster (tasks 320-331, packs).
Markers as fig_batch_routes.py: filled = still training, open = paused or queued (will continue), none = finished.
Panel b (John 11-13 23:55: "the paper form of the plot as a second panel"; 11-14 00:25: "instead of stopping the curves
early let's show each individual"): the submitted Fig 13's axes (plot_ablation.py / fig13_paper_form.py): Joint minus network
accuracy (the cached bounds bayes_bounds_ctx2_T30.npy, as the paper), log-log against the testing episode (test 1 = before
training, 1 test per 250 iterations), one line per network with the same status markers as panel a (no mean +- SEM).

    python fig_b1k_status.py [--pull]
"""
import os, sys, time, argparse, subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_batch_routes as fb
from main.utils import fig_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/{{rccb1k1m_lr4,rccclfb1k1m_lr4,rccb1k1mR_lr4,"
                       f"rccclfb1k1mR_lr4,rccb1k1mR2_lr4,rcc120k_lr4,rccclf120k_lr4,rccclfpermb1k1m_lr4,rccpermb1k1m_lr4,rccrawpermb1k1m_lr4,rccclfrawpermb1k1m_lr4,rccfullpermb1k1m_lr4,rccclffullpermb1k1m_lr4,rccattnpermb1k1m_lr4,rccclfattnpermb1k1m_lr4,rccperm100k_lr4,rccclfperm100k_lr4,rccpermb1k450_lr4,rccclfpermb1k450_lr4,rccpermb1k643_lr4,rccclfpermb1k643_lr4,rccpermb1k700_lr4,rccclfpermb1k700_lr4,rccpermb1kfin_lr4,rccclfpermb1kfin_lr4,rccclfperm300k_lr4,rccclfperm100kR_lr4,rccpermb1k1m_lrc1e3,rccclfpermb1k1m_lrc1e3,rccpermb1k1m_lr1e3,rccclfpermb1k1m_lr1e3}}_s*.jsonl' {fb.LOGS[1]}/", shell=True,
                       stderr=subprocess.DEVNULL)
        fb.pull_status()
    status = fb.run_status()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "serif", "font.serif": "cmr10", "mathtext.fontset": "cm",
                         "axes.formatter.use_mathtext": True, "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 9.5,
                         "ytick.labelsize": 9.5, "legend.fontsize": 11, "axes.linewidth": .7})   # 11-14: 5 columns, phone
    import matplotlib.ticker as mticker
    joint, naive = np.load(os.path.join(fb.ROOT, "main", "DATA", "cache", "bayes_bounds_ctx2_T30.npy"))
    # John 11-14 00:35: "put on the left what we have for the 10K equivalents": column 1 = batch 10,000 (fb.PANELS[0]:
    # rcc60k_lr4 -> rcc120k_lr4 / rccclf60k_lr4 -> rccclf120k_lr4, 6 + 6 seeds), column 2 = batch 1,000 (fb.PANELS[1])
    # John 11-14 02:10: "please include the shuffled channel runs as a 3rd panel": column 3 = batch 1,000 with the channel
    # order shuffled in every episode (readin_perm): rccpermb1k1m_lr4_s0-2 (Generator route, tasks 386-388) and
    # rccclfpermb1k1m_lr4_s0-3 (Classifier route, tasks 377/380/383/385); runs not started yet are counted as queued.
    # 11-15 01:25 IST (John: "for the 3rd column, let's train all seeds to the point"): the continuations to 450k
    # (rcc{,clf}permb1k450_lr4_s*, RESUME: their logs start with the source's) are drawn when they exist
    # 11-15 04:35 FL (John: "continue training the seeds ... to the length of the longest run"): the 643k continuations
    # (rcc{,clf}permb1k643_lr4_s*) are drawn when they exist, else the 450k ones; since 11-15 08:10 FL the 700k ones (rcc{,clf}permb1k700_lr4_s*) first
    # 11-15 08:55 FL (John: "start training new seeds"): seeds 13-21 of both routes (rcc{,clf}permb1kfin_lr4), trained to the final
    # length X; counted once their logs exist
    panels = list(fb.PANELS) + [("perm", "batch 1,000, shuffled channels",
                                 ("rccpermb1k1m_lr4", ("rccpermb1kfin_lr4", "rccpermb1k700_lr4", "rccpermb1k643_lr4",
                                                       "rccpermb1k450_lr4")),
                                 ("rccclfpermb1k1m_lr4", ("rccclfpermb1kfin_lr4", "rccclfpermb1k700_lr4", "rccclfpermb1k643_lr4",
                                                          "rccclfpermb1k450_lr4")), 1000, range(17))]   # 11-15 14:20 FL: seeds 17-21 stopped (John: "kill the runs that will finish only after 7pm"); 0-16 = finished + ongoing   # Gen s0-9, Clf s0-12
    # John 11-14 05:40: "remove the bottleneck from both the classifier and the generator ... check 1 seed of each route";
    # 06:00: "the no bottleneck condition can have its own panel": column 4 = z_raw runs (shuffled channels, no K.Q dot
    # products), rccrawpermb1k1m_lr4_s0 (Generator route) / rccclfrawpermb1k1m_lr4_s0 (Classifier route), tasks 389-390
    # 09:45: "remove the downscale and the normalization loss in this 'no world structure' setting": column 4 now draws
    # the z_full pair (full embeddings, no down-scaling, no norm loss; tasks 401-402, solid) and, thin dashed, the z_raw
    # pair it replaced (stopped at ~260k, both routes at chance)
    # 14:40 IST (John: "make the first layer from embeddings a transformer attention layer (in the no world structure
    # version)"): solid = z_attn pair (tasks 403-404); dashed = z_full pair (stopped ~300k); dotted = z_raw pair (282k)
    # 11-14 18:25 IST (John: "move the lr 1e-3 to a new panel so we can see the classifier route"): its own column after
    # the shuffled one, seed 3 of both routes at Classifier lr 1e-3 (solid) beside the same seed at 1e-4 (dashed)
    # 20:00 IST (John: "switch the classifier lr 1e-3 runs to both classifier and generator having lr 1e-3"): solid = both
    # modules (and so the embeddings) at 1e-3 (tasks 407-408); dashed = lr 1e-4; dotted = Classifier-only 1e-3 (stopped)
    # 20:30 IST (John: "remove panels e and k, and put the generator lr 1e-3 in a new column"): the attention column is
    # gone; seed 3 at Classifier lr 1e-3 only (tasks 405-406, stopped) and at Classifier + Generator lr 1e-3 (tasks 407-408)
    # each get a column, with the same seed at lr 1e-4 dashed in both
    # 11-14 ~23:00 IST (John: "kill the lr = 1e-3 runs and remove them from the figure"): both lr-1e-3 columns removed; the
    # local runs (tasks 407-408, 423-424) stopped by STOP files, the held cluster pack 66690228 (tasks 417-422, 425-426)
    # cancelled. The columns were:
    #   ("lrc", seed 3, Classifier lr 1e-3, rcc{,clf}permb1k1m_lrc1e3, (3,)) and
    #   ("lrboth", Classifier + Generator lr 1e-3, rcc{,clf}permb1k1m_lr1e3, seeds 0-5)
    # 11-14 14:40 IST (John: pause the batch-10,000 continuations for "10K shuffled routes"): last column = batch 10,000 with
    # shuffled channels, rcc{,clf}perm100k_lr4_s0-7 (tasks 332-339 + 409-416; one seed's pair per GPU), paused at .60 since
    # 20:30 IST (John: "pause all runs that surpass .6 accuracy")
    panels += [("perm10k", "batch 10,000, shuffled channels", ("rccperm100k_lr4", None),
                ("rccclfperm100k_lr4", ("rccclfperm300k_lr4", "rccclfperm100kR_lr4")), 10000,   # 11-15: s2 to 300k, s7 resumed
                range(12))]   # 11-14 23:00: seeds 8-11 added
    TWIN = [(("rccclfpermb1k1m_lr4", None), fb.RED, "--"), (("rccpermb1k1m_lr4", None), fb.BLUE, "--")]
    EXTRA = {"lrc": TWIN}
    NOTE = {"lrc": "\ndashed: the same seed at lr $10^{-4}$", "lrboth": "\n(the same seeds at lr $10^{-4}$: column c)"}
    LETTERS = "abcdefghijklmnop"
    # John 11-15 00:50 IST: "add a row showing how the paper form would look as a mean +/- fill (continue to the last run,
    # use the network's that exist at a given point to compute the +/-)": row 3 = per route, at every test episode, the mean
    # +- SEM (ddof 1, as plot_ablation.py) of Optimal - accuracy over the runs that have a test there (n shrinks as runs
    # pause or stop; no band where n = 1); n at the last point printed beside each curve
    fig, axs = plt.subplots(3, len(panels), figsize=(3.7 * len(panels), 11.6), sharey="row")
    for col, (tag, title, gen, clf, batch, seeds) in enumerate(panels):
        ax, bx, cx = axs[0, col], axs[1, col], axs[2, col]
        pooled = {fb.BLUE: {}, fb.RED: {}}                          # route -> {test index: [Optimal - accuracy per run]}
        jt, nv = [], []
        n = {(c, l): 0 for c in (fb.BLUE, fb.RED) for l in ("-", "--", ":")}
        nq, ns = dict(n), dict(n)                                   # queued / started without a test yet
        runs = []                                                   # (colour, last test, ever left chance) per main-arm run
        gen_cross = []                                              # Generator-route first tests 3x above Naive
        arms = [(clf, fb.RED, "-"), (gen, fb.BLUE, "-")] + EXTRA.get(tag, [])
        for arm, colr, ls in arms:
            for k in seeds:
                rs = fb.load(*arm, k)
                if not rs:                                          # not started: base or continuation name queued
                    ext = arm[1] if isinstance(arm[1], str) else (arm[1] or ("",))[0]
                    s_ = status.get(f"{arm[0]}_s{k}") or (status.get(f"{ext}_s{k}") if ext else None)
                    if s_ == "running":
                        ns[colr, ls] += 1
                    elif s_:
                        nq[colr, ls] += 1
                    continue
                n[colr, ls] += 1
                e, y = np.array([r["ep"] for r in rs]), np.array([r["net"] for r in rs])
                stt = status.get(fb.load.name)
                x2, g = e / 250 + 1, np.clip(joint - y, 1e-3, None)          # paper form: test 1 = before training
                for a_, xx, yy in ((ax, e, y), (bx, x2, g)):
                    a_.plot(xx, yy, color=colr, lw=.9, ls=ls, alpha=.85 if ls == "-" else .5)
                    if stt:
                        a_.plot(xx[-1], yy[-1], "o", ms=5.5, mew=.9, zorder=5, mfc=colr if stt == "running" else "white",
                                mec="k" if stt == "running" else colr)
                jt += [r["joint"] for r in rs]
                nv += [r["naive"] for r in rs]
                if ls == "-":                                       # row 3: one value per run and test index
                    per = {}
                    for b_, g_ in zip(np.round(e / 250).astype(int), joint - y):
                        per.setdefault(b_, []).append(g_)
                    for b_, v_ in per.items():
                        pooled[colr].setdefault(b_, []).append(float(np.mean(v_)))
                if ls == "-":                                       # failing-seed bookkeeping (John 11-14 23:05)
                    m5 = np.convolve(y, np.ones(5) / 5, mode="valid") if len(y) >= 5 else y
                    runs.append((colr, e[-1], bool((m5 >= .2).any())))
                    if colr == fb.BLUE:
                        above = np.array([r["net"] > r["naive"] for r in rs])
                        hit = next((i for i in range(2, len(above)) if above[i - 2:i + 1].all()), None)
                        if hit is not None:
                            gen_cross.append(e[hit])
                print(f"batch {batch:5d} {fb.load.name:22s} seed {k:2d}: last test {e[-1]:7d}, acc {y[-5:].mean():.3f} "
                      f"(last 5), {stt or 'finished'}")
        ax.axhline(np.mean(jt) if jt else joint, color="k", ls="--", lw=.8)
        ax.axhline(np.mean(nv) if nv else naive, color="k", ls=":", lw=.9)
        ax.set(xscale="log", xlim=(250, 1.05e6), ylim=(0, .8), xlabel=f"training iterations (batch {batch:,})")
        # John 11-14 23:05: "put a marker (with some jitter) below the failing seeds so i can count how many are failing".
        # Failing = never left chance (running mean of 5 tests always < .2) and trained past 3x the slowest Generator-route
        # seed's Naive crossing in the same panel. One triangle per failing run, at its last test, stacked downward so
        # overlapping ones stay countable.
        fail = {fb.BLUE: 0, fb.RED: 0}
        if gen_cross:
            limit = 3 * max(gen_cross)
            ax.axvline(limit, color=".55", lw=.7, ls=(0, (4, 3)), zorder=1)
            for i, (colr, last, left) in enumerate(sorted((r for r in runs if not r[2] and r[1] >= limit),
                                                          key=lambda r: r[1])):
                ax.plot(last, .075 - .016 * i, "^", ms=7, mfc=colr, mec="k", mew=.7, zorder=6, clip_on=False)
                fail[colr] += 1
        extra = [f"{w} {d[fb.BLUE, '-']} + {d[fb.RED, '-']}" for w, d in (("starting", ns), ("queued", nq))
                 if d[fb.BLUE, "-"] + d[fb.RED, "-"]]
        sub = f"{n[fb.BLUE, '-']} Generator + {n[fb.RED, '-']} Classifier route" + (f"\n{', '.join(extra)}" if extra else "")
        if gen_cross:
            sub += f"\nfailing past {3 * max(gen_cross) / 1000:.0f}k: {fail[fb.BLUE]} + {fail[fb.RED]}"
        sub += NOTE.get(tag, "")
        ax.set_title(r"$\mathbf{%s}$   %s" % (LETTERS[col], title) + "\n" + sub, fontsize=11, loc="left")
        bx.axhline(joint - naive, color="k", lw=.8, ls="--")
        bx.text(1.15, joint - naive, "Naive bound", fontsize=9.5, va="center", ha="left",
                bbox=dict(fc="white", ec="none", alpha=.85, pad=1))
        bx.set(xscale="log", yscale="log", xlim=(1, 4001), ylim=(.03, .75), xlabel="testing episode (1 per 250 iterations)")
        for axis, subs in ((bx.xaxis, (1.,)), (bx.yaxis, (1., 2., 5.))):   # x: decades only (1000 / 2000 collided)
            axis.set_major_locator(mticker.LogLocator(subs=subs))
            axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(mticker.NullFormatter())
        bx.grid(alpha=.25, lw=.5, which="both")
        bx.set_title(r"$\mathbf{%s}$   paper form (Fig 13)" % LETTERS[len(panels) + col], fontsize=11, loc="left")
        for colr in (fb.RED, fb.BLUE):
            if not pooled[colr]:
                continue
            b_ = np.array(sorted(pooled[colr]))
            vals = [np.array(pooled[colr][i]) for i in b_]
            cnt = np.array([len(v) for v in vals])
            mu = np.array([v.mean() for v in vals])
            se = np.array([v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan for v in vals])
            xx = b_ + 1
            cx.fill_between(xx, np.clip(mu - se, 1e-3, None), mu + se, color=colr, alpha=.22, lw=0)
            cx.plot(xx, np.clip(mu, 1e-3, None), color=colr, lw=1.3)
            cx.text(xx[-1] * 1.08, max(mu[-1], 1e-3), f"n={cnt[-1]}", color=colr, fontsize=9, va="center", ha="left")
        cx.axhline(joint - naive, color="k", lw=.8, ls="--")
        cx.text(1.15, joint - naive, "Naive bound", fontsize=9.5, va="center", ha="left",
                bbox=dict(fc="white", ec="none", alpha=.85, pad=1))
        cx.set(xscale="log", yscale="log", xlim=(1, 4001), ylim=(.03, .75), xlabel="testing episode (1 per 250 iterations)")
        for axis, subs in ((cx.xaxis, (1.,)), (cx.yaxis, (1., 2., 5.))):
            axis.set_major_locator(mticker.LogLocator(subs=subs))
            axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(mticker.NullFormatter())
        cx.grid(alpha=.25, lw=.5, which="both")
        cx.set_title(r"$\mathbf{%s}$   paper form, mean $\pm$ SEM" % LETTERS[2 * len(panels) + col], fontsize=11,
                     loc="left")
        for a_ in (ax, bx, cx):
            for sp in ("top", "right"):
                a_.spines[sp].set_visible(False)
    axs[0, 0].set_ylabel("goal accuracy, final step\n(novel environments)")
    axs[1, 0].set_ylabel(r"Optimal $-$ Network Accuracy")
    axs[2, 0].set_ylabel("Optimal $-$ Network Accuracy\n(mean $\\pm$ SEM over runs with a test)")
    h = [Line2D([], [], color=fb.BLUE, lw=1.2), Line2D([], [], color=fb.RED, lw=1.2),
         Line2D([], [], color="k", ls="--", lw=.8), Line2D([], [], color="k", ls=":", lw=.9),
         Line2D([], [], color=".4", ls="", marker="o", ms=5.5, mec="k"),
         Line2D([], [], color="white", ls="", marker="o", ms=5.5, mec=".3"),
         Line2D([], [], color=".5", ls="", marker="^", ms=7, mec="k"), Line2D([], [], color=".55", lw=.7, ls=(0, (4, 3)))]
    fig.legend(h, ["Generator route", "Classifier route", "Joint Bayes", "Naive Bayes", "still training",
                   "paused or queued (will continue)", "failing: never left chance", "3x slowest Generator Naive crossing"],
               ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(.5, 1.0), columnspacing=1.5)
    fig.text(.5, .941, "the RCC's own feedback, lr $10^{-4}$; 1 test every 250 iterations on novel environments ($10^4$ episodes "
                       "at batch 10,000, $10^3$ at batch 1,000); status on "   # John is in Florida (11-13): ET
                       + datetime.now(ZoneInfo("America/New_York")).strftime("%a %H:%M ET"),
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .932))
    out = fig_path("fig_b1k_status.png")
    fig.savefig(out, dpi=300)
    print(out)


if __name__ == "__main__":
    main()
