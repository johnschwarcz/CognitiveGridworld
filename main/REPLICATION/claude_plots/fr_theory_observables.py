"""fr_theory_observables.png: how trapping depends on the number of observables, and whether the interaction sets whose
Factorization Regret returns to zero are exactly the trap-free ones (k = 0).
John 11-10: "does the theory extrapolate to multiple observables? Or is that not yet done / simple? Ideally we could find a
Z where FR = 0 and check if k = 0 there, but also seeing how the theory depends on the observation dimensionality would be
interesting as well."

The theory block's propositions already hold for any number d_o of binary observables: kappa, Delta and the step X are sums
over observables. The theory figure's top row uses d_o = 1 only because every expectation is then an exact finite sum over
the count of ones; with d_o observables the sum runs over (t + 1)^d_o count vectors (feasible for d_o = 2, not for 5), so
here the curves are simulated.

  a  trapped fraction q of an interaction set against d_o for R = 3, 5, 10 realizations per variable (the Cognitive
     Gridworld: R = 10, d_o = 5): mean and 5-95% over interaction sets, from the likelihood tables alone
  b  share of interaction sets with no trapped joint realization, P(k = 0)
  c  E[FR_t] of single interaction sets at R = 3, d_o = 5 (simulated, mean +- 2 s.e. over episodes): the trap-free ones
     (k = 0) return to zero, the trapped ones (k >= 1) grow along t E[Delta](Z) (dashed: predicted from the table alone)
  d  every interaction set at R = 3, d_o = 5: simulated slope of E[FR_t] over the second half of the episode against the
     predicted E[Delta](Z); trap-free sets have E[Delta](Z) = 0
All interaction sets come from the project's generator (two distinct variables: allow_repeated_vars = False; test-set
sampler), lambda = 2. Uses fr_theory.py's one_step_table and simulate.

RESULTS (11-10): see the --numbers printout; caches in DATA/experimental/fr_theory/observables/.
usage:  python fr_theory_observables.py [--data] [--numbers]"""
import os
import sys
import time
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import fr_theory as F                                                             # noqa: E402
from main.utils import fig_path                                                   # noqa: E402

OUT = os.path.join(ROOT, "main", "DATA", "experimental", "fr_theory", "observables")
LAM, RS, DOS = 2.0, (3, 5, 10), (1, 2, 3, 5, 8, 12, 20, 30)
NZ_SCAN = 300                                     # interaction sets per (R, d_o) in a, b
R_SIM, DO_SIM = 3, 5                              # c, d
N_SINGLE, T_SINGLE, NEP_SINGLE = 3, 1500, 2000    # c: interaction sets per kind (k = 0 / k >= 1), steps, episodes each
NZ_SETS, NEP_SETS = 300, 100                      # d: interaction sets, episodes each (T_SINGLE steps)
_ENVS = {}


def table(rng, lam, d_o, R):
    """One interaction set's tables ell[i, r_g, r_c] from the project's generator (as fr_theory.make_env, any d_o and R)."""
    from main.CognitiveGridworld import CognitiveGridworld
    env = _ENVS.get((d_o, R))
    if env is None:
        st = np.random.get_state()
        env = CognitiveGridworld(mode=None, state_num=500, ctx_num=2, realization_num=R, obs_num=d_o, KQ_dim=30, likelihood_temp=lam,
                                 batch_num=2, step_num=2, show_plots=False)
        np.random.set_state(st); env.batch_num = 1; env.allow_repeated_vars = False; _ENVS[(d_o, R)] = env
    env.likelihood_temp = lam
    st = np.random.get_state(); np.random.seed(int(rng.integers(2 ** 31)))
    try:
        env.preprocess_env(); env.EC_gen_state_embeddings(); env.EC_gen_context()
        env.joint_likelihood, _ = env.trigger_build_likelihood()          # the joint tables only (the env squeezes d_o = 1 elsewhere)
    finally:
        np.random.set_state(st)
    ell = np.asarray(env.joint_likelihood[0], np.float64).reshape(d_o, R, R)
    return ell if int(env.goal_ind[0]) == 0 else ell.transpose(0, 2, 1)


def build():
    os.makedirs(OUT, exist_ok=True); t0 = time.time(); out = {}
    q = np.zeros((len(RS), len(DOS), NZ_SCAN)); ed = np.zeros_like(q)                      # a, b
    for i, R in enumerate(RS):
        for j, d_o in enumerate(DOS):
            rng = np.random.default_rng(1000 * R + d_o)
            for z in range(NZ_SCAN):
                tab = F.one_step_table(F.checked_table(table(rng, LAM, d_o, R), LAM)); q[i, j, z] = tab["trapped"].mean(); ed[i, j, z] = tab["Delta"].mean()
        print(f"scan R = {R} done ({time.time() - t0:.0f} s)", flush=True)
    out.update(q=q, edelta=ed)
    rng = np.random.default_rng(77); free, trapped = [], []                                # c: the first interaction sets of each kind
    while len(free) < N_SINGLE or len(trapped) < N_SINGLE:
        ell = F.checked_table(table(rng, LAM, DO_SIM, R_SIM), LAM); k = int(F.one_step_table(ell)["trapped"].sum())
        (free if k == 0 else trapped).append(ell) if len(free if k == 0 else trapped) < N_SINGLE else None
    curves, ses, dl, ks = [], [], [], []
    for n, ell in enumerate(free + trapped):
        S = F.simulate(LAM, 1, NEP_SINGLE, T_SINGLE, np.random.default_rng(500 + n), env=lambda r_, l_: ell)
        curves.append(S["FR"].mean(0)); ses.append(S["FR"].std(0) / np.sqrt(NEP_SINGLE))
        tab = F.one_step_table(ell); dl.append(tab["Delta"].mean()); ks.append(int(tab["trapped"].sum()))
    out.update(single_mean=np.array(curves), single_se=np.array(ses), single_edelta=np.array(dl), single_k=np.array(ks))
    print(f"single sets done ({time.time() - t0:.0f} s)", flush=True)
    rng = np.random.default_rng(88); sl, pdl, kk, frT = [], [], [], []                      # d: every interaction set
    tabs = [F.checked_table(table(rng, LAM, DO_SIM, R_SIM), LAM) for _ in range(NZ_SETS)]
    it = iter(tabs)
    S = F.simulate(LAM, NZ_SETS, NEP_SETS, T_SINGLE, np.random.default_rng(99), store_fr=False, env=lambda r_, l_: next(it))
    h = T_SINGLE // 2
    for z, ell in enumerate(tabs):
        tab = F.one_step_table(ell); pdl.append(tab["Delta"].mean()); kk.append(int(tab["trapped"].sum()))
        sl.append((S["FR_mean"][z, -1] - S["FR_mean"][z, h - 1]) / (T_SINGLE - h)); frT.append(S["FR_mean"][z, -1])
    out.update(sets_slope=np.array(sl), sets_edelta=np.array(pdl), sets_k=np.array(kk), sets_frT=np.array(frT))
    np.savez(os.path.join(OUT, "observables.npz"), **out); print(f"built in {time.time() - t0:.0f} s", flush=True)
    return out


def load(fresh=False):
    p = os.path.join(OUT, "observables.npz")
    return build() if fresh or not os.path.exists(p) else dict(np.load(p))


def numbers(Z):
    q, ed = Z["q"], Z["edelta"]
    for i, R in enumerate(RS):
        print(f"R = {R:2d}: q (mean) " + " ".join(f"{d}:{q[i, j].mean():.3f}" for j, d in enumerate(DOS)))
        print(f"        P(k = 0) " + " ".join(f"{d}:{np.mean(q[i, j] == 0):.3f}" for j, d in enumerate(DOS)))
        print(f"        E[Delta] " + " ".join(f"{d}:{ed[i, j].mean():.4f}" for j, d in enumerate(DOS)))
    k, sl, pd_, fT = Z["sets_k"], Z["sets_slope"], Z["sets_edelta"], Z["sets_frT"]
    free = k == 0
    print(f"d (R = {R_SIM}, d_o = {DO_SIM}, {len(k)} sets x {NEP_SETS} episodes, T = {T_SINGLE}): trap-free {free.sum()}; "
          f"late slope trap-free mean {sl[free].mean():+.5f} (max {sl[free].max():+.5f}), trapped mean {sl[~free].mean():+.4f}; "
          f"r(slope, E[Delta]) over trapped {np.corrcoef(sl[~free], pd_[~free])[0, 1]:.3f}; regression slope {np.polyfit(pd_[~free], sl[~free], 1)[0]:.3f}; "
          f"FR_T trap-free median {np.median(fT[free]):.3f} vs trapped {np.median(fT[~free]):.2f}")
    for n in range(len(Z["single_k"])):
        m = Z["single_mean"][n]
        print(f"c set {n}: k = {Z['single_k'][n]}, E[Delta] {Z['single_edelta'][n]:.4f}; E[FR] at t = 30, 300, {T_SINGLE}: "
              f"{m[29]:.3f}, {m[299]:.3f}, {m[-1]:.3f}; peak {m.max():.3f} at t = {m.argmax() + 1}")


def figure(Z):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update(F.RC)
    cols = {3: "#63a6ff", 5: "#2469be", 10: "#00336d"}
    fig, axs = plt.subplots(1, 4, figsize=(9.0, 2.15))
    a, b, c, d = axs
    q, x = Z["q"], np.array(DOS)
    for i, R in enumerate(RS):
        lo, hi = np.quantile(q[i], [0.05, 0.95], axis=1)
        a.fill_between(x, lo, hi, color=cols[R], alpha=0.18, lw=0); a.plot(x, q[i].mean(1), color=cols[R], lw=1.2, label=f"$R={R}$")
        b.plot(x, np.mean(q[i] == 0, axis=1), color=cols[R], lw=1.2, label=f"$R={R}$")
    for ax in (a, b):
        ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 30]); ax.set_xticklabels(["1", "2", "5", "10", "30"]); ax.set_xlim(1, 30)
        ax.set_xlabel(r"observables $d_o$"); ax.axvline(5, color=F.MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    a.plot([5], [q[RS.index(10), DOS.index(5)].mean()], ls="none", marker="o", ms=4, mfc="white", mec=cols[10], mew=1.0, zorder=5)
    a.set_ylim(0, 1); a.set_ylabel("trapped fraction $q$"); a.set_title("(a) Trapped fraction $q=k/R^2$", loc="left")
    a.legend(loc="upper right", title="realizations")
    b.set_ylim(0, 1); b.set_ylabel(r"share of sets with $k=0$"); b.set_title("(b) Trap-free interaction sets", loc="left")
    tt = np.arange(1, T_SINGLE + 1)
    for n in range(len(Z["single_k"])):
        k = Z["single_k"][n]; col = F.ORANGE if k == 0 else F.BLUE; m, se = Z["single_mean"][n], Z["single_se"][n]
        c.fill_between(tt, m - 2 * se, m + 2 * se, color=col, alpha=0.25, lw=0); c.plot(tt, m, color=col, lw=1.0)
        if k: c.plot(tt, tt * Z["single_edelta"][n], color=col, lw=0.8, ls=(0, (3, 2)))
    c.set_xscale("log"); c.set_yscale("symlog", linthresh=0.1); c.set_xlim(1, T_SINGLE); c.set_ylim(0, None)
    c.set_xlabel("step $t$"); c.set_ylabel(r"$\mathbb{E}[\mathrm{FR}_t]$ (nats)")
    c.set_title(r"(c) Single sets, $R=3$, $d_o=5$", loc="left")
    c.legend(handles=[Line2D([], [], color=F.ORANGE, lw=1.0, label="$k=0$ (simulated)"), Line2D([], [], color=F.BLUE, lw=1.0, label=r"$k\geq1$ (simulated)"),
                      Line2D([], [], color=F.INK2, lw=0.8, ls=(0, (3, 2)), label=r"$t\,\mathbb{E}[\Delta]$ (predicted)")], loc="upper left")
    k, sl, pd_ = Z["sets_k"], Z["sets_slope"], Z["sets_edelta"]
    mx = max(pd_.max(), sl.max()) * 1.05
    d.plot([0, mx], [0, mx], color=F.INK, lw=0.8, zorder=1)
    d.plot(pd_[k > 0], sl[k > 0], ls="none", marker="o", ms=2.6, mfc=F.BLUE, mec="white", mew=0.3, alpha=0.8, label=rf"$k\geq1$ ({(k > 0).sum()} sets)")
    d.plot(pd_[k == 0], sl[k == 0], ls="none", marker="o", ms=2.6, mfc=F.ORANGE, mec="white", mew=0.3, alpha=0.9, label=rf"$k=0$ ({(k == 0).sum()} sets, at 0)")
    d.set_xlim(-0.02 * mx, mx); d.set_ylim(min(sl.min(), 0) - 0.02 * mx, mx)
    d.set_xlabel(r"predicted slope $\mathbb{E}[\Delta](Z)$"); d.set_ylabel("simulated slope (nats / step)")
    d.set_title(r"(d) Every set, $R=3$, $d_o=5$", loc="left"); d.legend(loc="upper left")
    fig.text(0.995, 0.005, f"The project's generator, $\\lambda={LAM:g}$, two distinct variables.  (a, b) {NZ_SCAN} interaction sets per point "
             f"(dotted: the Gridworld's $d_o=5$; circle: its $R=10$).\n(c) {NEP_SINGLE:,} episodes per set.  "
             f"(d) {NZ_SETS} sets x {NEP_SETS} episodes; slope over steps {T_SINGLE // 2}-{T_SINGLE}.",
             ha="right", va="bottom", fontsize=7, color=F.INK2, linespacing=1.3)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.2)
    p = fig_path("fr_theory_observables.png"); fig.savefig(p, dpi=200); print("saved", p)


if __name__ == "__main__":
    Z = load(fresh="--data" in sys.argv)
    if "--numbers" in sys.argv: numbers(Z)
    figure(Z)
