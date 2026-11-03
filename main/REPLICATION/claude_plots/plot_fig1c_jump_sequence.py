"""What happens around a jump, inside the networks (27 Oct 2026; John's hypothesis: splits are passages through extra
dimensions, entanglement-induced) -> main/REPLICATION/figure_bin/fig1c_jump_sequence.png

Data: figdata/round7_jumps/scan_jumps.npz and decoding_rf_controls.npz (claude_plots/fig1c_scan_jumps.py, from the
replays fig1c_scan_hidden.py and fig1c_scan_credit.py of the alias scan: 39 environments x 100 cells x 50 streams).
    a  decoding the Optimal's exact joint covariance Cov(r_g, r_c) (steps 2-16; 5-fold cross-validated ridge, folds by
       stream; one dot per environment, line: median): from the network's own two marginals (20 numbers and their
       outer product), the same plus 1000 random ReLU features of them (a generic nonlinear code of the marginals,
       matched in size), the output-null part of u (1000 units), and the marginals plus the output-null part. (In this
       task the exact posterior is a function of the 5 observation counts, so its covariance is a function of its
       marginals: the question is how linearly readable each code makes it, not whether it holds it)
    b-k  jump-triggered averages: the Trained RNN's isolated jumps (a step moving its posterior mean by more than 2
       cells; no other jump within lags -4 .. +10) at steps 6-20, each minus a control stream of the same environment at
       the same step, as far from its true cell before the step (within 0.5 cells), without a jump (b-g: all 3382 such
       events; h-k, and the novelty in d: the 611 among the first 10 streams of each cell, where the update measures were
       computed). The ESN is read on the SAME streams (same observations); the Optimal's belief move on the same streams
       in b. Shading: +-2 SEM over events. Raw units
       b  the belief move |x_t - x_{t-1}| (cells)       c  the step's entanglement (percentile among the environment's
       d  the observation's novelty (nats)                   5000 streams at that step)
       e  the hidden update's energy outside the read-out |du_t|^2 - |P_Q du_t|^2
       f  the effective dimension of u_t on the network's fixed basis   g  the activity turn 1 - cos(u_t, u_{t-1})
       h  credit coherence   i  the update's energy |dW|^2 (Trained RNN only: the ESN's is on another scale)
       j  the cross-band share (10% bands)   k  the update's effective dimension
    l  the 743 split pairs (the split movie's), aligned on the split step t_s: the distance between the two streams'
       posterior means (cells, left axis) and between their hidden states' output-null parts (right axis), median over
       pairs, Trained RNN
Run from the repository root:
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/plot_fig1c_jump_sequence.py"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION")]
import matplotlib.pyplot as plt                                                   # noqa: E402

R7 = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression", "figdata", "round7_jumps")
C_TR, C_EC, C_OPT = "#1f77b4", "#ff7f0e", "k"


def _band(ax, x, d, col, lab=None, lw=1.6):
    m, s = np.nanmean(d, 0), 2 * np.nanstd(d, 0, ddof=1) / np.sqrt(np.isfinite(d).sum(0))
    ax.fill_between(x, m - s, m + s, color=col, alpha=.18, lw=0)
    ax.plot(x, m, color=col, lw=lw, label=lab)


def figure():
    from main.utils import apply_fig_style, fig_path
    FS = apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    Z = np.load(os.path.join(R7, "scan_jumps.npz"))
    RF = np.load(os.path.join(R7, "decoding_rf_controls.npz"))
    L = Z["jump_lags"]
    diff = lambda k: Z[f"jump_ev_{k}"] - Z[f"jump_ct_{k}"]
    PW, PH, GX, GY, X0, Y0 = 2.55, 1.75, .75, .95, .75, .55
    W, H = X0 + 4 * PW + 3 * GX + .65, Y0 + 3 * PH + 2 * GY + .75                  # (room for l's right axis)
    fig = plt.figure(figsize=(W, H))
    axs = {}
    letters = "abcdefghijkl"
    for n, s in enumerate(letters):
        r_, c_ = divmod(n, 4)
        x, y = X0 + c_ * (PW + GX), Y0 + (2 - r_) * (PH + GY)
        axs[s] = fig.add_axes([x / W, y / H, PW / W, PH / H])
        fig.text((x - .55) / W, (y + PH + .3) / H, r"$\mathbf{%s}$" % s, fontsize=FS["label"] + 2, va="bottom")
    title = lambda ax, name, formula: ax.set_title(name + "\n" + formula, fontsize=FS["tick"] + .5, pad=3)
    # a: decoding
    ax = axs["a"]
    rows = [("own marginals", Z["dec_trained_marg_env"], Z["dec_echo_marg_env"]),
            ("+ random code", RF["trained_lin+rf"], RF["echo_lin+rf"]),
            ("output-null $u$", Z["dec_trained_null_env"], Z["dec_echo_null_env"]),
            ("marginals + null", Z["dec_trained_marg_null_env"], Z["dec_echo_marg_null_env"])]
    rng = np.random.default_rng(0)
    for i, (lab, tr, ec) in enumerate(rows):
        for v, col, dx in ((tr, C_TR, -.17), (ec, C_EC, .17)):
            ax.scatter(i + dx + rng.uniform(-.06, .06, len(v)), v, s=6, color=col, alpha=.6, lw=0)
            ax.plot([i + dx - .13, i + dx + .13], [np.median(v)] * 2, color=col, lw=2)
    ax.axhline(np.median(RF["opt_lin+rf"]), color="0.5", lw=.8, ls=":")
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], rotation=25, ha="right", fontsize=FS["tick"] - 1)
    ax.set_ylim(0, 1), ax.set_ylabel(r"CV $R^2$")
    title(ax, "Decoding the joint covariance", r"$\mathrm{Cov}(r_g, r_c)$ of the Optimal; dotted: exact marginals + code")
    # b-k: jump-triggered averages
    def lagax(ax, name, formula, ylabel=None):
        ax.axvline(0, color="0.8", lw=.8, zorder=0), ax.axhline(0, color="0.8", lw=.8, zorder=0)
        ax.set_xlim(L[0], L[-1]), ax.set_xticks([-4, 0, 4, 8])
        title(ax, name, formula)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=FS["tick"])
    _band(axs["b"], L, diff("move_Trained"), C_TR, "Trained RNN")
    _band(axs["b"], L, diff("move_Joint"), C_OPT, "Optimal")
    _band(axs["b"], L, diff("move_Echo"), C_EC, "ESN")
    lagax(axs["b"], "Belief move", r"$\|x_t - x_{t-1}\|$, event $-$ control", "cells")
    axs["b"].legend(frameon=False, fontsize=FS["tick"] - 1, loc="upper right")
    _band(axs["c"], L, diff("entpct"), "0.3")
    lagax(axs["c"], "Entanglement", r"percentile of $\mathrm{rev}_t$ in its step")
    _band(axs["d"], L, Z["mjump_trained_nov"], "0.3")
    lagax(axs["d"], "Novelty", r"$-\Sigma_i \log p(o_{ti} \mid o_{<t})$", "nats")
    for s, key, name, formula in (("e", "dE_NULL", "Output-null update", r"$\|du_t\|^2 - \|P_Q\, du_t\|^2$"),
                                  ("f", "pr", "Effective dim. of $u_t$", r"$(\Sigma_k E_k)^2 / \Sigma_k E_k^2$"),
                                  ("g", "turn", "Activity turn", r"$1 - \cos(u_t, u_{t-1})$")):
        _band(axs[s], L, diff(f"trained_{key}"), C_TR)
        _band(axs[s], L, diff(f"echo_{key}"), C_EC)
        lagax(axs[s], name, formula)
    for s, key, name, formula, both in (("h", "coh", "Credit coherence", r"$\|\Sigma_k \delta_k\|^2 / \Sigma_k \|\delta_k\|^2$", True),
                                        ("i", "dW", r"$\Delta W$ energy", r"$\|\Sigma_k \delta_k z_k^\top\|^2$ (Trained RNN)", False),
                                        ("j", "band_x", "Cross-band share", r"$\Sigma_B (\langle \Delta W_B, \Delta W\rangle - \|\Delta W_B\|^2) / \|\Delta W\|^2$", True),
                                        ("k", "dim", r"$\Delta W$ effective dim.", r"$(\mathrm{tr}\,DZ)^2 / \mathrm{tr}((DZ)^2)$", True)):
        _band(axs[s], L, Z[f"mjump_trained_{key}"], C_TR)
        if both:
            _band(axs[s], L, Z[f"mjump_echo_{key}"], C_EC)
        lagax(axs[s], name, formula)
    for s in "bcdefghijk":
        axs[s].tick_params(labelsize=FS["tick"] - 1)
        if s in "hijk":
            axs[s].set_xlabel("steps from the jump", fontsize=FS["tick"])
    # l: split pairs
    ax = axs["l"]
    Ls = Z["split_lags"]
    ax.plot(Ls, np.nanmedian(Z["split_trained_db"], 0), color="k", lw=1.6, label="posterior means")
    ax.set_ylabel("cells", fontsize=FS["tick"])
    ax2 = ax.twinx()
    ax2.plot(Ls, np.nanmedian(Z["split_trained_dn"], 0), color=C_TR, lw=1.6, ls="--")
    ax2.set_ylabel(r"output-null $u$", color=C_TR, fontsize=FS["tick"])
    ax2.tick_params(labelsize=FS["tick"] - 1, colors=C_TR)
    ax.axvline(0, color="0.8", lw=.8, zorder=0)
    ax.set_ylim(0, None), ax2.set_ylim(0, None)                                    # both from 0: honest shapes
    ax.set_xlim(Ls[0], Ls[-1]), ax.set_xticks([-6, -3, 0, 3, 6])
    ax.set_xlabel("steps from the split $t_s$", fontsize=FS["tick"])
    ax.tick_params(labelsize=FS["tick"] - 1)
    title(ax, "Split pairs: the two streams' distance", r"solid: beliefs; dashed: output-null $u$ (Trained)")
    fig.legend(handles=[plt.Line2D([], [], color=C_TR, lw=2, label="Trained RNN"),
                        plt.Line2D([], [], color=C_EC, lw=2, label="ESN (same streams)")],
               loc="upper center", ncol=2, frameon=False, fontsize=FS["tick"], bbox_to_anchor=(.5, 1 - .12 / H))
    path = fig_path("fig1c_jump_sequence.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", figure())
