"""Generator- vs Classifier-routed embeddings: one summary figure (John 11-11: "reduce everything to 1 figure summarizing
all meaningful comparisons between the generator and classifier routed embeddings, with as many panels as we need (but
try to avoid showing the same thing in multiple ways)"). Supersedes fig13_rcc_arm (new arms), fig13_memorization,
fig13_embeddings / fig13_z_vs_truth, fig13_fresh_generator, fig13_transfer, fig13_transfer_ch4, fig13_plasticity,
fig13_correction (their plotters keep the numbers in their docstrings).

Every network: |S| = 500, d_E = 30, batch 10,000, lr 1e-4, 60,000 episodes (train_cluster_runs.py gen60k / clf60k:
oracle feedback; rcc60k / rccclf60k: the RCC's own feedback), the embedding gradient routed through the Generator or the
Classifier. Tests on novel environments (both variables held out).
    a  learning: Joint minus the network's accuracy, mean +- SEM over seeds (as the paper's Fig 13)
    b  memorization at equal competence: 2 novel / (novel + training accuracy), bins of .01 novel accuracy
    c  each channel's learned code against the truth (seed 0 networks, oracle feedback): colour = the true Z of the channel
       (hue = angle, saturation = size); titles: the model slot holding the code, and how it copies the truth
    d  a fresh Generator trained on the FROZEN embeddings (only the Generator learns; true values; lr 1e-3, batch 2,000):
       plug-in accuracy (exact Bayes with its learned likelihood)
    e-h  a fresh reader on the embeddings (CPU, batch 512, lr 1e-3, target = the true joint value): e an LSTM like the
       RCC's Classifier; f a channel-wise reader (one evidence function applied to every channel, summed: the form Joint
       Bayes has here); g the channel-wise reader trained on channels 0-3, the change when channel 4 is added at test;
       h the LSTM with the embeddings plastic (lr 1e-3), with or without the network's own Generator still training on
       them
Corrections of the Classifier-route code (d, f, g): by hand (each channel's code moved to its slot and its sign / swap
undone, from fig13_embeddings' decoding), or LEARNED by a map of the 10 code numbers placed before the reader and trained
with it: linear (10 x 10) or MLP (identity + 10 -> 128 -> 128 -> 10), both the identity at the start.

RESULT 11-11 18:30 (end values; the plotters of the superseded figures hold the details):
    a  Joint - network at 60k: oracle Generator route .044 (4 seeds there), oracle Classifier route .092 (6), the RCC .071
       (6), original Classifier route .19 (4); Generator route faster at every stage (2-4x to each accuracy level)
    b  2 novel / (novel + training) below .5 accuracy: Generator route .99 / .98 (oracle / original), Classifier route .94 /
       .92
    c  both routes copy the true Z per channel (R^2 .99); Generator route: -swap in every channel, in place; Classifier route
       (oracle s0): -id, +id, -swap, -id, +id, slots [1, 0, 4, 2, 3]
    d  fresh Generator, novel plug-in at 3,000 steps: Generator route .73 (by 250 steps); Classifier route as learned .14,
       by hand .72, learned linear map .72 (at ~1,000 / ~1,750 steps for the two networks), MLP map .71 (~500 / ~750)
    e  LSTM: .48 / .49 (Generator / Classifier route), from scratch .14
    f  channel-wise reader: Generator route .53; Classifier route .15, by hand .53, linear map .22, MLP map .43
    g  channel 4 added: Generator route +.067, Classifier by hand +.062, as learned +.011, linear map +.001, MLP map -.12,
       LSTM -.016
    h  plastic embeddings (lr 1e-3) under an LSTM: Generator route + its own Generator .38; Classifier route + its Generator
       .15; either alone .15

    python fig13_summary.py [--pull]
"""
import os, sys, json, glob, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
from main.utils import fig_path, apply_fig_style

DATA = os.path.join(ROOT, "main", "DATA")
EXP = os.path.join(DATA, "experimental")
ARM_LOGS = os.path.join(EXP, "cluster_runs", "jsonl")
EMB = os.path.join(EXP, "fig13_embeddings")
FG = os.path.join(EMB, "fresh_generator")
TR = os.path.join(EMB, "transfer_reader")
ARMS = (("gen60k_lr4", "#08519c", "oracle, Generator route"), ("clf60k_lr4", "#a50f15", "oracle, Classifier route"),
        ("rcc60k_lr4", "#4fa3e0", "original, Generator route (RCC)"), ("rccclf60k_lr4", "#f4845f", "original, Classifier route"))
C_GEN, C_CLF, C_HAND, C_LIN, C_MLP = "#08519c", "#a50f15", "#1b9e77", "#e6550d", "#7b3294"
JOINT_T, NAIVE_T = .767, .451                          # the test bounds of the reader experiments (|S| 500 dataset)


def pull():
    for d, src in ((FG, "fresh_generator"), (TR, "transfer_reader")):
        os.makedirs(d, exist_ok=True)
        subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/{src}/*.jsonl' {d}/", shell=True)
    subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/cluster_runs/{{gen60k,clf60k,rcc60k,rccclf60k}}*.jsonl' "
                   f"{ARM_LOGS}/", shell=True)


def jl(path):
    if not os.path.exists(path):
        return []
    out = []
    for l in open(path, "rb").read().replace(b"\x00", b"").decode().splitlines():
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    key = "step" if out and "step" in out[0] else "ep"
    return sorted({q[key]: q for q in out}.values(), key=lambda q: q[key])


def mean_sem(curves):
    n = min(len(c) for c in curves)
    A = np.array([c[:n] for c in curves])
    return A.mean(0), (A.std(0, ddof=1) / np.sqrt(len(A)) if len(A) > 1 else np.zeros(n)), n


USED_LONG = []


def best_log(d, tag):
    """the 10x longer rerun (<tag>_long, 11-11 18:50) once it has passed the original run, else the original"""
    short, long_ = jl(os.path.join(d, f"{tag}.jsonl")), jl(os.path.join(d, f"{tag}_long.jsonl"))
    key = "step"
    if long_ and (not short or long_[-1][key] > short[-1][key]):
        USED_LONG.append(tag)
        return long_
    return short


def lines(a, specs, d, xkey, ykey, x_floor=256, diff=None):
    """specs: (tags, colour, linestyle, label); thin = runs, thick = mean"""
    for tags, col, ls, lab in specs:
        runs = [r for r in (best_log(d, t) for t in tags) if r]
        if not runs:
            continue
        val = (lambda q: q[ykey] - q[diff]) if diff else (lambda q: q[ykey])
        for r in runs:
            a.plot([max(q[xkey], x_floor) for q in r], [val(q) for q in r], color=col, ls=ls, lw=.6, alpha=.35)
        m, _, n = mean_sem([[val(q) for q in r] for r in runs])
        a.plot([max(q[xkey], x_floor) for q in min(runs, key=len)[:n]], m, color=col, ls=ls, lw=2, label=f"{lab} ({len(runs)})")
        print(f"   {lab:45s} n {len(runs)} end {m[-1]:+.3f}" if diff else f"   {lab:45s} n {len(runs)} end {m[-1]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        pull()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.colors import hsv_to_rgb
    fs = apply_fig_style()
    fig = plt.figure(figsize=(19, 10.8))
    G = fig.add_gridspec(2, 5, hspace=.42, wspace=.3, left=.045, right=.99, top=.86, bottom=.12)
    small, TS = 9, 11.5                                           # legend / title points at this size

    # a: learning
    a = fig.add_subplot(G[0, 0])
    joint, naive = np.load(os.path.join(DATA, "cache", "bayes_bounds_ctx2_T30.npy"))
    print("a learning")
    for arm, col, lab in ARMS:
        runs = [r for r in (jl(os.path.join(ARM_LOGS, f"{arm}_s{k}.jsonl")) for k in range(6)) if r]
        if len(runs) < 2:
            continue
        L = max(len(r) for r in runs)                             # per test: the seeds that have reached it
        A = np.full((len(runs), L), np.nan)
        for i_, r in enumerate(runs):
            A[i_, :len(r)] = [joint - q["net"] for q in r]
        cnt = np.isfinite(A).sum(0)
        ok = cnt >= 2
        m = np.nanmean(A[:, ok], 0)
        se = np.nanstd(A[:, ok], 0, ddof=1) / np.sqrt(cnt[ok])
        x = np.array([q["ep"] for q in max(runs, key=len)])[ok]
        a.fill_between(x[1:], (m - se)[1:], (m + se)[1:], color=col, alpha=.25, lw=0)
        a.plot(x[1:], m[1:], color=col, lw=1.8, label=f"{lab} ({len(runs)})")
        print(f"   {lab:40s} n {len(runs)} end gap {m[-1]:.3f} at {x[-1]} ({cnt[ok][-1]} seeds there)")
    a.axhline(joint - naive, color="k", lw=1, ls=":")
    a.set(xscale="log", yscale="log", xlabel="training episodes", ylabel="Joint $-$ network accuracy")
    for ax_ in (a.xaxis, a.yaxis):
        ax_.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax_.set_minor_formatter(mticker.NullFormatter())
    a.set_title("a   Learning (mean $\\pm$ SEM over seeds)", loc="left", fontsize=TS)
    a.legend(frameon=False, fontsize=small, loc="lower left")
    a.text(260, (joint - naive) * 1.07, "Naive Bayes", fontsize=small)

    # b: memorization
    b = fig.add_subplot(G[0, 1])
    import fig13_memorization as M
    edges = np.arange(.10, .7501, .01)
    mid = (edges[:-1] + edges[1:]) / 2
    print("b memorization")
    for arm, col, lab in ARMS:
        R = M.runs(arm)
        Y = np.array([M.binned(nov, 2 * nov / (nov + tr), edges) for nov, tr, _ in R])
        n = np.isfinite(Y).sum(0)
        mu = np.where(n >= 1, np.nanmean(Y, 0), np.nan)
        se = np.where(n >= 2, np.nanstd(Y, 0, ddof=1) / np.sqrt(np.maximum(n, 1)), np.nan)
        ok = np.isfinite(se)
        b.fill_between(mid[ok], (mu - se)[ok], (mu + se)[ok], color=col, alpha=.25, lw=0)
        ok = np.isfinite(mu)
        b.plot(mid[ok], mu[ok], color=col, lw=1.6)
        print(f"   {lab:40s} below .5 accuracy: mean ratio {np.nanmean(mu[(mid < .5) & ok]):.3f}")
    b.axhline(1, color="k", lw=.8, ls="--")
    b.set(xlabel="stage: novel-environment accuracy (bins of .01)", ylabel="2 novel / (novel + training)")
    b.set_title("b   Memorization at equal competence\n     (1 = all generalization; colours as in a)", loc="left",
                fontsize=TS)

    # c: learned codes against the truth
    R = json.load(open(os.path.join(EMB, "measures.json")))
    Gc = G[0, 2:].subgridspec(3, 5, hspace=.32, wspace=.06)
    rows = (("true $Z$", None), ("Generator\nroute", "gen60k_lr4_s0"), ("Classifier\nroute", "clf60k_lr4_s0"))
    rng = np.random.default_rng(0)
    import fig13_embeddings as E
    for i, (lab, name) in enumerate(rows):
        src = name or "gen60k_lr4_s0"
        D = np.load(os.path.join(EMB, f"z_{src}.npz"))
        Tz = D["novel_ztrue"]
        perm = np.arange(5) if name is None else np.array(R[name]["cross"]).argmax(1)
        Lz = Tz if name is None else D["novel_z"][:, perm]
        idx = rng.choice(len(Tz), 1500, replace=False)
        for o in range(5):
            c = fig.add_subplot(Gc[i, o])
            t, l = Tz[idx, o], Lz[idx, o]
            col = hsv_to_rgb(np.c_[(np.arctan2(t[:, 1], t[:, 0]) / (2 * np.pi)) % 1,
                                   np.clip(np.hypot(t[:, 0], t[:, 1]) / .35, 0, 1), np.full(len(t), .92)])
            c.scatter(l[:, 0], l[:, 1], c=col, s=3, lw=0)
            c.set(xlim=(-.6, .6), ylim=(-.6, .6), xticks=[], yticks=[], aspect="equal")
            if name is None:
                c.set_title(f"channel {o}", fontsize=small + 1, pad=2)
            else:
                A = np.linalg.lstsq(Tz[:, o], Lz[:, o], rcond=None)[0]
                c.set_title(f"slot {perm[o]}, {E.frame_name(A)}", fontsize=small, pad=2)
            if o == 0:
                c.set_ylabel(lab, fontsize=small + 2)
            if i == 0 and o == 0:
                c.text(0, 1.28, "c   Each channel's learned code against the truth (colour = true $Z$): both routes "
                       "copy it exactly; the Generator route with one sign / swap\n     for every channel, in place; the "
                       "Classifier route with a different one per channel, in shuffled slots (titles)",
                       transform=c.transAxes, fontsize=TS, va="bottom")

    # d-h
    d = fig.add_subplot(G[1, 0])
    print("d fresh Generator")
    lines(d, ((["gen_s0_own"], C_GEN, "-", "Generator route"),
              ([f"clf_s{k}_as_learned" for k in (0, 1)], C_CLF, "-", "Classifier route, as learned"),
              ([f"clf_s{k}_aligned" for k in (0, 1)], C_HAND, "-", "  corrected by hand"),
              ([f"clf_s{k}_map_linear" for k in (0, 1)], C_LIN, (0, (4, 2)), "  + learned linear map"),
              ([f"clf_s{k}_map_mlp" for k in (0, 1)], C_MLP, (0, (4, 2)), "  + learned MLP map")),
          FG, "step", "novel_plugin", x_floor=10)
    d.set(xscale="log", xlabel="training steps of the fresh Generator", ylabel="accuracy, novel environments")
    d.set_title("d   A fresh Generator on the\n     frozen embeddings", loc="left", fontsize=TS)
    e = fig.add_subplot(G[1, 1], sharey=d)
    print("e LSTM")
    lines(e, (([f"lstm_gen_frozen_s{k}" for k in (0, 1)], C_GEN, "-", "Generator route"),
              ([f"lstm_clf_frozen_s{k}" for k in (0, 1)], C_CLF, "-", "Classifier route"),
              ([f"lstm_random_updated_s{k}" for k in (0, 1)], "0.5", (0, (1, 1.5)), "random embeddings (from scratch)")),
          TR, "episodes", "novel")
    e.set_title("e   A fresh LSTM reader\n     (frozen embeddings)", loc="left", fontsize=TS)
    f = fig.add_subplot(G[1, 2], sharey=d)
    print("f channel-wise reader")
    lines(f, (([f"shared_gen_frozen_s{k}" for k in (0, 1)], C_GEN, "-", "Generator route"),
              ([f"shared_clf_frozen_s{k}" for k in (0, 1)], C_CLF, "-", "Classifier route, as learned"),
              ([f"shared_clf_aligned_frozen_s{k}" for k in (0, 1)], C_HAND, "-", "  corrected by hand"),
              ([f"shared_adapt_clf_frozen_s{k}" for k in (0, 1)], C_LIN, (0, (4, 2)), "  + learned linear map"),
              ([f"shared_mlpmap_clf_frozen_s{k}" for k in (0, 1)], C_MLP, (0, (4, 2)), "  + learned MLP map")),
          TR, "episodes", "novel")
    f.set_title("f   A fresh channel-wise reader (one\n     evidence function for every channel)", loc="left",
                fontsize=TS)
    g = fig.add_subplot(G[1, 3])
    print("g new channel")
    lines(g, (([f"shared_gen_frozen_ch4_s{k}" for k in (0, 1)], C_GEN, "-", "Generator route"),
              ([f"shared_clf_frozen_ch4_s{k}" for k in (0, 1)], C_CLF, "-", "Classifier route, as learned"),
              ([f"shared_clf_aligned_frozen_ch4_s{k}" for k in (0, 1)], C_HAND, "-", "  corrected by hand"),
              ([f"shared_adapt_clf_frozen_ch4_s{k}" for k in (0, 1)], C_LIN, (0, (4, 2)), "  + learned linear map"),
              ([f"shared_mlpmap_clf_frozen_ch4_s{k}" for k in (0, 1)], C_MLP, (0, (4, 2)), "  + learned MLP map"),
              ([f"lstm_gen_frozen_ch4_s{k}" for k in (0, 1)], "0.45", (0, (5, 2, 1, 2)), "LSTM reader, Generator route")),
          TR, "episodes", "novel_all5", diff="novel")
    g.axhline(0, color="k", lw=.8)
    g.set_ylabel("change in accuracy when channel 4 is added", fontsize=small + 1)
    g.set_title("g   A channel never seen: the reader of f\n     trained on channels 0-3, channel 4 added", loc="left",
                fontsize=TS)
    h = fig.add_subplot(G[1, 4], sharey=d)
    print("h plasticity")
    lines(h, (([f"lstm_gen_updated_gen_s{k}" for k in (0, 1)], C_GEN, (0, (6, 2, 1, 2)), "Generator route + its Generator"),
              ([f"lstm_clf_updated_gen_s{k}" for k in (0, 1)], C_CLF, (0, (6, 2, 1, 2)), "Classifier route + its Generator"),
              ([f"lstm_gen_updated_s{k}" for k in (0, 1)], C_GEN, (0, (1, 1.5)), "Generator route alone"),
              ([f"lstm_clf_updated_s{k}" for k in (0, 1)], C_CLF, (0, (1, 1.5)), "Classifier route alone")),
          TR, "episodes", "novel")
    h.set_title("h   The LSTM of e with plastic\n     embeddings (lr $10^{-3}$)", loc="left", fontsize=TS)
    for a_ in (e, f, g, h):
        a_.set(xscale="log", xlabel="episodes seen by the fresh reader")
    for a_ in (d, e, f, h):
        a_.axhline(JOINT_T, color="k", lw=.8, ls="--"); a_.axhline(NAIVE_T, color="k", lw=.8, ls=":")
        a_.set_ylim(.05, .8)
    for a_ in (a, b, d, e, f, g, h):
        a_.spines[["top", "right"]].set_visible(False)
        a_.tick_params(labelsize=small + 1)
        a_.xaxis.label.set_size(small + 1); a_.yaxis.label.set_size(small + 1)
    for a_ in (e, h):
        a_.legend(frameon=False, fontsize=small, loc="upper left", bbox_to_anchor=(0, .94))
    hs, ls_ = g.get_legend_handles_labels()                 # d, f, g share the conditions: one legend under them
    fig.legend(hs, [l.strip().rsplit(" (", 1)[0] for l in ls_], loc="lower center", bbox_to_anchor=(.42, .005), ncol=6, frameon=False,
               fontsize=small + 1, handlelength=2.6, columnspacing=1.6)
    fig.suptitle("Embeddings learned through the Generator or through the Classifier (Fig 13's networks: $|\\mathcal{S}| = "
                 "500$, 60,000 episodes, lr $10^{-4}$; novel environments; thin = seeds, thick = mean, (n) = runs)",
                 fontsize=TS + 1, y=.985)
    fig.text(.42, .045, "d, f, g: the Classifier-route code as learned, corrected by hand (slots and signs / swaps put back), or "
             "through a learned map of its 10 numbers (linear or MLP, identity at the start); 2 runs per line (d, "
             "Generator route: 1); dashed / dotted lines = Joint / Naive Bayes" + (f"; d-h: {len(set(USED_LONG))} runs "
             "extended 10x" if USED_LONG else ""), ha="center", fontsize=small + 1)
    out = fig_path("fig13_summary.png")
    fig.savefig(out, dpi=170)
    print(out)


if __name__ == "__main__":
    main()
