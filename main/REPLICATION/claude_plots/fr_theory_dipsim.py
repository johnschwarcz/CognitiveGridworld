#!/usr/bin/env python3
"""fr_theory_dipsim.py -- simulated dips of N-episode averages over long horizons (fig_fr_theory_all u and w markers).

John 11-11: "add more numerical data to u and w", then "we still need a lot more numerical points in u and w.  If an analytical line
does not have enough numerical points across it, then we need to drop the line."  fr_theory.build_population's simulation (exact
observers from step 10,000) stops at 1,000 steps on 7 values of N.  This one follows groups of N episodes of the SAME population (the
2,000 interaction sets of build_population, seed 5151; checked) for 10^4 steps, on an even log grid of N (24 N from 10 to 825, 12 per
decade: John 11-12, "ideally equidistant" markers), and keeps each group's first dip, so P(dip within T) comes out for every T <= 10^4
at once.

THE SHORTCUT.  At step 10,000 most episodes have SETTLED: the Joint goal belief and the Naive belief each favour one value by more
than GAP nats over every other.  Then FR_t = -ln Btilde_t(r_g) up to ~e^-GAP, and its change at a step is the Naive log-likelihood
ratio of the new observation between the Naive answer a-hat and r_g, X = o . theta + const (fr_theory.one_step_table), or 0 when
a-hat = r_g (untrapped).  Episodes that have not settled are updated exactly at every step: the Naive belief alone when only it is
undecided (FR = -ln Btilde(r_g) once the Joint has settled), both beliefs otherwise.  `check` compares every class with the full update
(fr_theory.fr_counts on the counts) on the same observations, and the removal of dipped groups with keeping them.  A dip = the group
average decreases at a step (by any amount).

    python fr_theory_dipsim.py [check]     ~8 min on 15 workers (FR_WORKERS); `check` runs only the comparisons.
    output: main/DATA/experimental/fr_theory/additions/dip_sim.npz: Ns, first[N, group] (step of the first dip, TMAX + 1 = none), and
      Ts, P[N, T], se (fraction of groups whose average decreases within T steps, binomial SEM)

RESULTS, run 1 (11-11 night: 18 N, 400 groups; 100,000 groups over 10 steps).  Classes at step 10,000: 45% settled trapped, 53%
settled untrapped, 6% Naive undecided, 0.2% Joint undecided.  Check against the full update: <= 1.6e-7 nats a step (settled trapped),
<= 3.4e-9 (others).  Against the theory (fr_theory_horizon.py): 2 of 90 (N, T) beyond 2 SEM; against build_population's full-update
simulation at the common N and T: |z| < 1.5.  Rare dips over T = 1 step match the EXACT law of the mean step (fr_theory.dip_law, no
sampling of group compositions) within noise; over 10 steps the simulation exceeds the theory, by 6% at P = 3e-2 up to ~8x at N = 650
(3e-4 vs 4e-5).  Cause, verified: with every episode put on its asymptotic step law (GAP = -2: all "settled") the same simulation gives
2.2e-4 / 6e-5 / 1e-5 at N = 550 / 600 / 650 against the theory's 2.0e-4 / 8.5e-5 / 3.8e-5: the excess is the ~6% of episodes not yet
settled at step 10,000, whose FR still drifts -- a finite-time effect the asymptotic theory leaves out, visible only in rare dips of
large groups.  (So the 99.9 and 99.99% levels of w, which only 100,000-group runs over a few steps can reach, were dropped.)

RESULTS, run 3 (11-12, this configuration; run 2 = 2,000 groups x 41 N from 10 to 850, kept as additions/dip_sim_run2.npz): 562 s on
15 workers.  Classes at step 10,000: 43.2% settled trapped, 50.7% settled untrapped, 6.0% Naive undecided, 0.19% Joint undecided.
Check: <= 1.3e-7 nats a step.  Against the theory (fr_theory_horizon.py, its 1,500-draw Monte Carlo error included; every (N, T <= 10^4)
with the theory between 1e-3 and 1 - 1e-3): 269 points, |z| > 2 in 5.6% (chance 4.6%), mean z +0.06 (run 2: 465 points, 3.2%, +0.02;
the points of one N share its groups, so they are not independent).  The crossings of w (N at P(no dip) = 10 / 50 / 90 / 99%, 13
horizons) against run 2's: every difference within the two runs' combined half-bars (largest 0.93 of it).
"""
import os, sys, time
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import fr_theory as F                                                             # noqa: E402

OUT = os.path.join(ROOT, "main", "DATA", "experimental", "fr_theory", "additions", "dip_sim.npz")
CFG = dict(WORKERS=int(os.environ.get("FR_WORKERS", 12)), T0=F.CFG["POP_T0"], GAP=25.0, GROUPS=2000, TMAX=10_000,
           NS=tuple(int(round(10 * 10 ** (k / 12))) for k in range(24)),       # 10 ... 825, 12 per decade (u draws every second one)
           TS=tuple(int(round(10 ** (k / 3))) for k in range(13)),              # 1, 2, 5, 10, 22, 46 ... 10^4: 3 per decade (w's horizons)
           BATCH=600_000)                                                         # episodes per task (memory, balance)
_P = {}                                                                           # the population's tables, inherited by forked workers


def population():
    """The 2,000 interaction sets of fr_theory.build_population (same seed and draws), as per-set arrays."""
    lam = F.CFG["LAM_MAIN"]; rng = np.random.default_rng(5151); tabs = [F.checked_table(F.make_env(rng, lam), lam) for _ in range(F.CFG["POP_NZ"])]
    T = [F.one_step_table(e) for e in tabs]; lg = [F.table_logs(e) for e in tabs]
    _P.update(p=np.stack([np.moveaxis(e, 0, -1) for e in tabs]), theta=np.stack([t["theta"] for t in T]), const=np.stack([t["const"] for t in T]),
              trapped=np.stack([t["trapped"] for t in T]), ahat=np.stack([t["ahat"] for t in T]), Delta=np.stack([t["Delta"] for t in T]),
              L1=np.stack([g[0] for g in lg]), L0=np.stack([g[1] for g in lg]), M1=np.stack([g[2] for g in lg]), M0=np.stack([g[3] for g in lg]))


def _lse(a, axis):
    m = a.max(axis=axis, keepdims=True); return np.squeeze(m, axis) + np.log(np.exp(a - m).sum(axis=axis))


def _fr_full(LJ, LN):
    """FR from unnormalized Joint (m, R, Rc) and Naive (m, R) log-likelihoods (as fr_theory.fr_counts)."""
    lB = _lse(LJ, 2); lB -= _lse(lB, 1)[:, None]; ln = LN - _lse(LN, 1)[:, None]
    return np.sum(np.exp(lB) * (lB - ln), axis=1)


def _fr_naive(LN, rg):
    """FR = -ln Btilde(r_g), exact once the Joint goal belief has settled on r_g."""
    return _lse(LN, 1) - LN[np.arange(len(rg)), rg]


class Members:
    """The episodes of G groups of N, each at its own (set, joint realization), drawn uniformly; counts at step t0; sorted into the
    classes: settled (shortcut), Naive undecided (exact Naive update), Joint undecided (exact update of both)."""
    def __init__(self, G, N, rng, t0, gap):
        P = _P; nZ, R, Rc, d_o = P["p"].shape; M = G * N
        s, rg, rc = rng.integers(0, nZ, M), rng.integers(0, R, M), rng.integers(0, Rc, M); grp = np.repeat(np.arange(G), N)
        p = P["p"][s, rg, rc]; n = rng.binomial(t0, p).astype(float); ah = P["ahat"][s, rg, rc]; tr = P["trapped"][s, rg, rc]
        gJ, gN = np.empty(M), np.empty(M); order = np.argsort(s, kind="stable"); bounds = np.searchsorted(s[order], np.arange(nZ + 1))
        for z in range(nZ):                                                       # the two beliefs at t0, one interaction set at a time
            idx = order[bounds[z]:bounds[z + 1]]
            if not len(idx): continue
            LB = _lse(np.einsum("mi,ijk->mjk", n[idx], P["L1"][z]) + np.einsum("mi,ijk->mjk", t0 - n[idx], P["L0"][z]), 2)
            LN = n[idx] @ P["M1"][z] + (t0 - n[idx]) @ P["M0"][z]
            sb = np.sort(LB, axis=1); gJ[idx] = np.where(LB.argmax(1) == rg[idx], sb[:, -1] - sb[:, -2], -1.0)
            sn = np.sort(LN, axis=1); gN[idx] = np.where(LN.argmax(1) == ah[idx], sn[:, -1] - sn[:, -2], -1.0)
        settled, joint_ok = (gJ > gap) & (gN > gap), gJ > gap
        self.need = np.flatnonzero(~(settled & ~tr))                              # episodes whose FR can move (settled untrapped: 0)
        self.p, self.g, self.G = p[self.need].astype(np.float32), grp[self.need], G
        cls = np.where(settled, 0, np.where(joint_ok, 1, 2))[self.need]; self.ix = [np.flatnonzero(cls == c) for c in range(3)]
        m = self.need[self.ix[0]]; self.th, self.c = P["theta"][s[m], rg[m], rc[m]], P["const"][s[m], rg[m], rc[m]]
        m = self.need[self.ix[1]]; self.nM1, self.nM0, self.nrg = P["M1"][s[m]], P["M0"][s[m]], rg[m]
        self.nLN = np.einsum("mi,mia->ma", n[m], self.nM1) + np.einsum("mi,mia->ma", t0 - n[m], self.nM0); self.nfr = _fr_naive(self.nLN, self.nrg)
        m = self.need[self.ix[2]]; self.fL1, self.fL0, self.fM1, self.fM0 = P["L1"][s[m]], P["L0"][s[m]], P["M1"][s[m]], P["M0"][s[m]]
        self.fLJ = np.einsum("mi,mijk->mjk", n[m], self.fL1) + np.einsum("mi,mijk->mjk", t0 - n[m], self.fL0)
        self.fLN = np.einsum("mi,mia->ma", n[m], self.fM1) + np.einsum("mi,mia->ma", t0 - n[m], self.fM0); self.ffr = _fr_full(self.fLJ, self.fLN)
        self.counts = np.array([(settled & tr).sum(), (settled & ~tr).sum(), (~settled & joint_ok).sum(), (~joint_ok).sum()])
        self.members = dict(s=s, rg=rg, rc=rc, n=n, tr=tr, cls=np.where(settled, 0, np.where(joint_ok, 1, 2)))   # for `check`

    def increments(self, o):
        """The change of FR of the episodes in `need` given their observations o (len(need), d_o; bool) at this step."""
        d = np.zeros(len(self.p))
        i = self.ix[0]; d[i] = (self.th * o[i]).sum(1) + self.c
        i = self.ix[1]
        if len(i):
            of = o[i].astype(float); self.nLN += np.einsum("mi,mia->ma", of, self.nM1) + np.einsum("mi,mia->ma", 1 - of, self.nM0)
            fr = _fr_naive(self.nLN, self.nrg); d[i] = fr - self.nfr; self.nfr = fr
        i = self.ix[2]
        if len(i):
            of = o[i].astype(float)
            self.fLJ += np.einsum("mi,mijk->mjk", of, self.fL1) + np.einsum("mi,mijk->mjk", 1 - of, self.fL0)
            self.fLN += np.einsum("mi,mia->ma", of, self.fM1) + np.einsum("mi,mia->ma", 1 - of, self.fM0)
            fr = _fr_full(self.fLJ, self.fLN); d[i] = fr - self.ffr; self.ffr = fr
        return d

    def step(self, rng):
        """One step of every episode: the change of each group's sum of FR."""
        return np.bincount(self.g, self.increments(rng.random(self.p.shape, dtype=np.float32) < self.p), self.G)

    def compact(self, alive):
        """Drop the episodes of the groups that are not alive and renumber the rest (their order is kept)."""
        keep = alive[self.g]; pos = np.cumsum(keep) - 1; gid = np.cumsum(alive) - 1
        for c, names in ((0, ("th", "c")), (1, ("nM1", "nM0", "nrg", "nLN", "nfr")), (2, ("fL1", "fL0", "fM1", "fM0", "fLJ", "fLN", "ffr"))):
            k = keep[self.ix[c]]; self.ix[c] = pos[self.ix[c][k]]
            for a in names: setattr(self, a, getattr(self, a)[k])
        self.p, self.g, self.G = self.p[keep], gid[self.g[keep]], int(alive.sum())


def _run(task):
    """First dip (1-based step; TMAX + 1 = none) of G groups of N episodes over TMAX steps, own seed; class counts at t0.  Groups that
    have dipped are dropped (all of a group's episodes, so the others' draws are unaffected in law)."""
    N, G, tmax, seed, t0, gap = task; rng = np.random.default_rng(seed); M = Members(G, N, rng, t0, gap); first = np.full(G, tmax + 1, np.int32)
    ids = np.arange(G)                                                            # original id of every live group
    for t in range(1, tmax + 1):
        dip = M.step(rng) < 0
        if dip.any():
            first[ids[dip]] = t; ids = ids[~dip]
            if not len(ids): break
            M.compact(~dip)
    return first, M.counts


def simulate(Ns, G, tmax, seed0, cfg, label):
    tasks = []
    for i, N in enumerate(Ns):
        per = max(1, cfg["BATCH"] // N)
        for k, g0 in enumerate(range(0, G, per)): tasks.append((int(N), int(min(per, G - g0)), tmax, seed0 + 1000 * i + k, cfg["T0"], cfg["GAP"]))
    order = sorted(range(len(tasks)), key=lambda j: -tasks[j][0] * tasks[j][1])    # the longest first
    res = F.run_parallel(_run, [tasks[j] for j in order], cfg["WORKERS"], label); out, cnt = {}, np.zeros(4, int)
    for j, (first, c) in zip(order, res): out.setdefault(tasks[j][0], []).append((tasks[j][3], first)); cnt += c
    return {N: np.concatenate([f for _, f in sorted(out[N])]) for N in Ns}, cnt


def probs(first, Ns, Ts):
    P = np.array([[np.mean(first[N] <= T) for T in Ts] for N in Ns]); g = np.array([len(first[N]) for N in Ns])[:, None]
    return P, np.sqrt(np.maximum(P * (1 - P), 1.0 / g) / g), g[:, 0]


def check(cfg, G=40, N=60, steps=300, seed=77):
    """(1) Every class against the full update (fr_theory.fr_counts on the counts) on the same observations: largest |difference| of an
    episode's change of FR per step.  (2) Dropping groups: the surviving groups' sums with and without compact(), same observations."""
    P = _P; rng = np.random.default_rng(seed); M = Members(G, N, rng, cfg["T0"], cfg["GAP"]); mb = M.members; s, n = mb["s"], mb["n"].copy()
    p = P["p"][s, mb["rg"], mb["rc"]]; zs = np.unique(s); obs = np.random.default_rng(seed + 1)
    def fr_all(n, t):
        out = np.empty(len(s))
        for z in zs: idx = np.flatnonzero(s == z); out[idx] = F.fr_counts(n[idx], t, P["L1"][z], P["L0"][z], P["M1"][z], P["M0"][z])
        return out
    prev = fr_all(n, cfg["T0"]); worst = np.zeros(4); names = ["settled, trapped", "settled, untrapped", "Naive undecided", "Joint undecided"]
    lab = np.where(mb["cls"] == 0, np.where(mb["tr"], 0, 1), mb["cls"] + 1); t0 = time.time()
    M2 = Members(G, N, np.random.default_rng(seed), cfg["T0"], cfg["GAP"]); alive = np.arange(G) % 3 != 0; dmax = 0.0
    for st in range(steps):
        o = obs.random(p.shape) < p; n = n + o; cur = fr_all(n, cfg["T0"] + st + 1); full = cur - prev; prev = cur
        short = np.zeros(len(s)); short[M.need] = M.increments(o[M.need])
        for c in range(4):
            m = lab == c
            if m.any(): worst[c] = max(worst[c], float(np.abs(full[m] - short[m]).max()))
        oM = o[M.need]
        if st == 10: M2.compact(alive); keep2 = alive[M.g]                                       # from step 10 on, 1 group in 3 dropped
        if st < 10: M2.increments(oM); continue
        a = np.bincount(M.g, short[M.need], G)[alive]; b = np.bincount(M2.g, M2.increments(oM[keep2]), M2.G); dmax = max(dmax, float(np.abs(a - b).max()))
    print(f"   check: {len(s)} episodes x {steps} steps ({time.time() - t0:.0f} s); classes {np.bincount(lab, minlength=4)}; largest |full update - this| "
          f"per step: " + ", ".join(f"{a} {b:.1e}" for a, b in zip(names, worst)) + f"; dropping groups changes the others' sums by {dmax:.1e}", flush=True)
    return max(worst.max(), dmax)


def main(argv):
    cfg = dict(CFG); t0 = time.time(); population(); print(f"population: {_P['p'].shape[0]} sets ({time.time() - t0:.0f} s)", flush=True)
    import pickle
    with open(os.path.join(ROOT, "main", "DATA", "experimental", "fr_theory", "fr_theory_cache", "population.pkl"), "rb") as fh: _, pop = pickle.load(fh)
    assert np.array_equal(pop["pool"]["Delta"], _P["Delta"].reshape(-1)), "not the population of build_population"
    print("   the population of fr_theory.build_population (Delta of every joint realization identical)", flush=True)
    worst = check(cfg)
    if "check" in argv: return
    assert worst < 1e-6, "the shortcut departs from the full update"
    first, cnt = simulate(cfg["NS"], cfg["GROUPS"], cfg["TMAX"], 900_000, cfg, "groups over 10^4 steps")
    P, se, g = probs(first, cfg["NS"], cfg["TS"]); print("   classes at step 10,000:", cnt, flush=True)
    np.savez(OUT, Ns=np.array(cfg["NS"]), first=np.stack([first[N] for N in cfg["NS"]]), Ts=np.array(cfg["TS"]), P=P, se=se, groups=g, gap=cfg["GAP"],
             t0=cfg["T0"], tmax=cfg["TMAX"], classes=cnt, check_worst=worst)
    np.set_printoptions(precision=3, suppress=True, linewidth=220)
    print(f"saved {OUT} ({time.time() - t0:.0f} s)\nP(dip within T), T = {cfg['TS']}"); print(np.c_[cfg["NS"], P])


if __name__ == "__main__":
    main(sys.argv[1:])
