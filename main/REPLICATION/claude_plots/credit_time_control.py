"""credit_time_control.png: is the credit-vs-FR map (supp_learning_behaviour a, f) only the credit moving from far-back
to recent steps during training, with FR standing in for time? (John 11-10: "can you confirm that what we see is not
just the delta_k gradient being put on further back timesteps in the beginning and more recent timesteps later? i.e. if
we do the same cross-corr replacing FR with time?")
Per network, lag k x checkpoint (td_tt_credit_reach full_<run>_*.npz: E_k = ||delta_k||^2 of the step-t error carried
back k steps, 2,000 probes, error steps t = 17-30; share s_k = E_k / Sum_k' E_k'):
  1  the figure's map: r over episodes between s_k and FR at the landing step t - k, at each t, then the mean over t
     (every episode in one r has the same landing step, so time is constant inside each correlation)
  2  the same with FR shuffled across episodes within each t (the null)
  3  pooled over the 14 error steps: r over (episode, t) pairs between s_k and FR_(t-k)
  4  pooled the same way with TIME in place of FR: r between s_k and the landing step t - k
  5  the mean share s_k (where the credit lands on average)
FR: probes.npz 'fr' (forward KL, both variables averaged, as td_xcorr_across).
RESULTS (11-10): figure's map max |r| .26 (Trained) / .26 (Echo), mean |r| .11 / .15; FR shuffled across episodes max
|r| .02 / .02 (the null); pooled over t with FR .24 / .25 (the same pattern as 1); pooled over t with TIME max .22 / .14,
mean .06 / .05, and a different pattern (cell-by-cell r with map 1: -.30 Trained, +.30 Echo). The mean share (5) does
not move from far back to recent either: the Trained RNN's credit moves from lag 0 (start) to lags 1-4; shares beyond
lag 5 stay small throughout. Inside each correlation of map 1 every episode has the same landing step, so time cannot
enter it; the FR pattern is not the time pattern.
usage:  python credit_time_control.py"""
import os
import sys
import glob
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib.pyplot as plt                                                   # noqa: E402
from matplotlib.colors import Normalize                                           # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402
import td_tt_credit_reach as cr                                                   # noqa: E402
import td_xcorr_lags as xl                                                        # noqa: E402
import fig_td_summary as fts                                                      # noqa: E402

AN = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics", "analysis")
RUNS = (("trained", "Trained", "#1f77b4"), ("echo", "Echo", "#ff7f0e"))


def r_cols(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def r_flat(a, b):
    return float(r_cols(a.ravel()[:, None], b.ravel()[:, None])[0])


def maps(run, seed=0):
    P = np.load(os.path.join(AN, "probes.npz"))
    fr = P["fr"].astype(np.float64).mean(-1)                                      # (n, 30)
    rng = np.random.default_rng(seed)
    t_idx = np.asarray(cr.SRC)
    out = {k: [] for k in ("x", "within", "shuffled", "pooled_fr", "pooled_time", "mean_share")}
    for f in sorted(glob.glob(os.path.join(cr.OUT, f"full_{run}_*.npz"))):
        z = np.load(f)
        E, rows = z["E"].astype(np.float64), z["rows"]
        sh = E / E.sum(-1, keepdims=True)                                         # (n, 14 t, 16 k)
        K = sh.shape[-1]
        w, s_, pf, pt = [], [], [], []
        for k in range(K):
            F = fr[rows][:, t_idx - k]                                            # FR at the landing step (n, 14)
            w.append(np.nanmean(r_cols(sh[:, :, k], F)))
            Fs = np.stack([F[rng.permutation(len(F)), j] for j in range(F.shape[1])], 1)
            s_.append(np.nanmean(r_cols(sh[:, :, k], Fs)))
            pf.append(r_flat(sh[:, :, k], F))
            pt.append(r_flat(sh[:, :, k], np.broadcast_to((t_idx - k)[None].astype(float), F.shape)))
        out["x"].append(float(z["x"]))
        for key, v in (("within", w), ("shuffled", s_), ("pooled_fr", pf), ("pooled_time", pt)):
            out[key].append(v)
        out["mean_share"].append(sh.mean((0, 1)))
    o = np.argsort(out["x"])
    return {k: np.asarray(v)[o] for k, v in out.items()}


def figure():
    apply_fig_style()
    plt.rcParams.update({"mathtext.fontset": "cm"})
    M = {r: maps(r) for r, _, _ in RUNS}
    cols = (("within", "1  figure's map:\nr(share$_k$, FR$_{t-k}$) within each $t$"),
            ("shuffled", "2  FR shuffled\nacross episodes"),
            ("pooled_fr", "3  pooled over $t$:\nr(share$_k$, FR$_{t-k}$)"),
            ("pooled_time", "4  pooled over $t$:\nr(share$_k$, time $t-k$)"),
            ("mean_share", "5  mean share$_k$\n(where credit lands)"))
    W, H = 10.5, 4.6
    fig, axs = plt.subplots(2, 5, figsize=(W, H), gridspec_kw=dict(left=.07, right=.95, top=.84, bottom=.12,
                                                                   wspace=.38, hspace=.5))
    CN, SN = Normalize(-.25, .25), Normalize(0, .5)
    for i, (run, nm, c) in enumerate(RUNS):
        d = M[run]
        for j, (key, title) in enumerate(cols):
            ax = axs[i, j]
            v = d[key]
            im = ax.pcolormesh(xl.log_edges(d["x"]), np.arange(v.shape[1] + 1) - .5, v.T, cmap="viridis",
                               norm=SN if key == "mean_share" else CN, shading="flat", rasterized=True)
            fts.xaxis(ax, label=i == 1)
            ax.set_xlim(fts.XMIN, 4.6e4)
            ax.set_yticks([0, 5, 10, 15])
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{nm}\n$k$ (steps back)", fontsize=8, color=c)
            if key != "mean_share":
                ax.text(.97, .04, f"max $|r|$ {np.nanmax(np.abs(v)):.2f}", transform=ax.transAxes, ha="right",
                        va="bottom", fontsize=6.5, color="w")
        for key, norm, x0 in (("within", CN, .955), ("mean_share", SN, .955)):
            pass
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=CN, cmap="viridis"), ax=axs[:, :4], fraction=.015, pad=.01)
    cb.set_label("$r$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb2 = fig.colorbar(plt.cm.ScalarMappable(norm=SN, cmap="viridis"), ax=axs[:, 4], fraction=.06, pad=.04)
    cb2.set_label("mean share", fontsize=8)
    cb2.ax.tick_params(labelsize=7)
    path = fig_path("credit_time_control.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)
    for run, nm, _ in RUNS:
        d = M[run]
        for key in ("within", "shuffled", "pooled_fr", "pooled_time"):
            v = d[key]
            print(f"{nm:8s} {key:12s} max |r| {np.nanmax(np.abs(v)):.3f}; mean |r| {np.nanmean(np.abs(v)):.3f}")
        # agreement of the figure's map with the time map and the mean-share map, cell by cell
        w = d["within"].ravel()
        for key in ("pooled_time", "mean_share"):
            o = d[key].ravel()
            ok = np.isfinite(w) & np.isfinite(o)
            print(f"{nm:8s} cell-by-cell r(figure's map, {key}) = {np.corrcoef(w[ok], o[ok])[0, 1]:+.2f}")


if __name__ == "__main__":
    figure()
