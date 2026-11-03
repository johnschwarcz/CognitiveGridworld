"""Jumps, splits and the joint correlation in the networks' hidden states, within the alias scan's environments
(27 Oct 2026; analyses of fig1c_scan_hidden.py's replay) -> main/DATA/experimental/fig1c_compression/figdata/
round7_jumps/scan_jumps.npz (and the printed summary)

Inputs: scan_hidden_<net>.npz (fig1c_scan_hidden.py), the scan (scan_meta.npz: obs, rev = entanglement per stream-step,
FR, cells, cell_of; scan_<Agent>.npz beliefs), figdata/round4b/split_pairs.npz (743 pairs).
1  decoding: CV R^2 of the Optimal's exact joint covariance Cov(r_g, r_c) from each feature set (fig1c_scan_hidden's
   docstring), per environment and per step (steps 2-16), and on the stream-steps where |Cov| is in its top 10% within
   the environment (the two-moded posteriors)
1b decoding controls (offline, from the stored coordinates): is the covariance just a NONLINEAR function of the
   marginals (the output-null part of u can hold the beliefs themselves: the read-out reads only the step's increment)?
   Per environment, the same folds and ridge grid: the network's top 64 output-null coordinates (Z[..., 18:]) against
   random ReLU expansions (64 and 1000 features, seed 0) of the Optimal's exact marginals (20 numbers, standardized)
2  jumps: a step that moves an agent's posterior mean (E[r_g], E[r_c]) by more than 2 cells (the prior mean (4.5, 4.5)
   before step 1); events: the Trained RNN's isolated jumps (no other jump within lags -4 .. +10) at steps 6-20; controls:
   per event, one stream of the SAME environment at the same step, as far from its true cell before the step (within
   0.5 cells), without a jump in the window (seed 0). The same stream-steps are read in both networks (same
   observations), so the ESN is aligned on the Trained RNN's events too. Measures per lag: the belief moves (Trained,
   ESN, Optimal), the hidden updates' energy per subspace (dE_Q, dE_G, dE_IT, dE_ALL; dE_NULL = dE_ALL - dE_Q), the
   activity turn, |u|, the effective dimension, and the step's entanglement as its percentile among the environment's
   5000 streams at that step
3  splits: for each split pair (both streams together through t_s, apart after), the distance between the two streams'
   hidden states u (full 1000 units, the replay's) split into the read-out part (Q) and the output-null part, and the
   distance between their posterior means, per step aligned on t_s (lags -6 .. +8): which separates first
4  the gradflow figure's measures (entanglement rev, novelty, |u|, effective dim., turn, credit coherence, dW energy,
   cross-band share, dW effective dim.; fig1c_scan_credit.py, 10 streams per cell): (a) aligned on the Trained RNN's
   jumps among those streams, event minus a same-environment, same-step, distance-matched control (raw units); (b)
   aligned on the split step t_s of each split pair (scan_credit_<net>_pairs.npz): the stream that moves more over
   t_s .. t_s + 2 ('mover') and its partner ('stayer'), each minus the environment's mean of the 1000 streams at that step
Run from the repository root (a few minutes):
    MPLBACKEND=Agg /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/fig1c_scan_jumps.py"""
import os
import sys
import numpy as np

ROOT = "/home/johns/Desktop/CognitiveGridworld"
SCAN = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c_compression")
OUT = os.path.join(SCAN, "figdata", "round7_jumps")
ENVS = [e for e in range(40) if e != 2]
R, T, M = 10, 30, 50
JUMP, LAGS, K0, K1 = 2., np.arange(-4, 11), 5, 19
NETS = (("trained", "Trained"), ("echo", "Echo"))


def posterior_means(B):
    """(..., T, 20) beliefs -> (..., T + 1, 2) posterior means with the prior mean first"""
    r = np.arange(R, dtype=np.float64)
    X = np.stack([B[..., :R] @ r, B[..., R:] @ r], -1)
    return np.concatenate([np.full(X.shape[:-2] + (1, 2), 4.5), X], -2)


def decoding(H):
    out = {}
    for net, Z in H.items():
        sets = [k[4:-4] for k in Z.files if k.startswith("dec_") and k.endswith("_e00") and not k.startswith("dec_y")]
        r2_env = {s: [] for s in sets}
        r2_step = {s: [] for s in sets}
        r2_top = {s: [] for s in sets}
        for e in ENVS:
            y = Z[f"dec_y_e{e:02d}"].astype(np.float64)
            top = np.abs(y) >= np.quantile(np.abs(y), .9)
            for s in sets:
                p = Z[f"dec_{s}_e{e:02d}"].astype(np.float64)
                r2_env[s].append(1 - ((y - p) ** 2).sum() / (y ** 2).sum())
                r2_step[s].append(1 - ((y - p) ** 2).sum(0) / np.maximum((y ** 2).sum(0), 1e-300))
                r2_top[s].append(1 - ((y[top] - p[top]) ** 2).sum() / (y[top] ** 2).sum())
        out[net] = {s: dict(env=np.array(r2_env[s]), step=np.array(r2_step[s]), top=np.array(r2_top[s])) for s in sets}
    return out


def _ridge_cv(X, y, fold, cs=(1e-4, 1e-3, 1e-2, 1e-1, 1.)):
    """5-fold ridge (X, y centred per step beforehand) -> the best CV R^2 over the ridge strengths"""
    G = X.T @ X
    Gf = [X[fold == k].T @ X[fold == k] for k in range(5)]
    b = X.T @ y
    bf = [X[fold == k].T @ y[fold == k] for k in range(5)]
    best = -np.inf
    for c in cs:
        pred = np.zeros_like(y)
        for k in range(5):
            Gt = G - Gf[k]
            w = np.linalg.solve(Gt + c * np.trace(Gt) / len(Gt) * np.eye(len(Gt)), b - bf[k])
            pred[fold == k] = X[fold == k] @ w
        best = max(best, 1 - ((y - pred) ** 2).sum() / (y ** 2).sum())
    return best


def decoding_controls(H, beliefs):
    steps = np.arange(1, 16)
    n_all = M * R * R
    fold = np.repeat((np.arange(n_all) % 5)[:, None], len(steps), 1).ravel()
    cen = lambda A: (A - A.mean(0, keepdims=True)).reshape(-1, A.shape[-1]) if A.ndim == 3 else (A - A.mean(0)).ravel()
    rng = np.random.default_rng(0)
    Wr = {p: (rng.standard_normal((20, p)), rng.standard_normal(p)) for p in (64, 1000)}
    out = {f"{net}_null64": [] for net in H}
    out.update({f"rf{p}": [] for p in Wr})
    for e in ENVS:
        y = H["trained"][f"dec_y_e{e:02d}"].astype(np.float64).ravel()
        q = beliefs["Joint"][e][:, steps].astype(np.float64)                                  # (5000, 15, 20)
        qs = (q - q.mean((0, 1))) / (q.std((0, 1)) + 1e-9)
        for p, (Wm, bm) in Wr.items():
            F = np.maximum(qs @ Wm / np.sqrt(20) + bm, 0)
            out[f"rf{p}"].append(_ridge_cv(cen(F), y, fold))
        for net, Z in H.items():
            out[f"{net}_null64"].append(_ridge_cv(cen(Z[f"Z_e{e:02d}"][:, steps, 18:].astype(np.float64)), y, fold))
    return {k: np.array(v) for k, v in out.items()}


def jumps(H, meta, beliefs):
    rng = np.random.default_rng(0)
    X = {a: posterior_means(beliefs[a].astype(np.float64)) for a in beliefs}           # (40, 5000, T + 1, 2)
    mv = {a: np.linalg.norm(np.diff(X[a], axis=2), axis=-1) for a in X}                # (40, 5000, T): step k's move
    truth = meta["cells"][meta["cell_of"]].astype(float)                                # (5000, 2) goal-ordered
    dtrue = np.linalg.norm(X["Trained"][:, :, 1:] - truth[None, :, None], axis=-1)     # (40, 5000, T)
    rev = meta["rev"].astype(np.float64)
    pct = rev.argsort(1).argsort(1) / (rev.shape[1] - 1)                               # within (env, step) percentile
    J = mv["Trained"] > JUMP
    win = lambda k: slice(k + LAGS[0], k + LAGS[-1] + 1)
    E, C = [], []
    for e in ENVS:
        calm = {k: np.flatnonzero(~J[e][:, win(k)].any(1)) for k in range(K0, K1 + 1)}
        for i, k in zip(*np.nonzero(J[e])):
            if not (K0 <= k <= K1) or J[e, i, win(k)].sum() != 1:
                continue
            cand = calm[k][np.abs(dtrue[e, calm[k], k - 1] - dtrue[e, i, k - 1]) < .5]
            if len(cand):
                E.append((e, i, k)), C.append((e, int(rng.choice(cand)), k))
    E, C = np.array(E), np.array(C)
    res = {"lags": LAGS, "n_events": len(E)}
    for tag, P in (("ev", E), ("ct", C)):
        st = P[:, 2:3] + LAGS[None]                                                    # (n, L) absolute steps
        ee, ii = P[:, :1], P[:, 1:2]
        for a in X:
            res[f"{tag}_move_{a}"] = mv[a][ee, ii, st]
        res[f"{tag}_dtrue"] = dtrue[ee, ii, st]
        res[f"{tag}_entpct"] = pct[ee, ii, st]
        for net, Z in H.items():
            for key in ("dE_Q", "dE_G", "dE_IT", "dE_ALL", "sE_Q", "sE_ALL", "turn", "l2", "pr"):
                A = np.stack([Z[f"{key}_e{e:02d}"] for e in ENVS]).astype(np.float64)    # (39, 5000, T)
                eidx = np.searchsorted(ENVS, ee)
                res[f"{tag}_{net}_{key}"] = A[eidx, ii, st]
            res[f"{tag}_{net}_dE_NULL"] = res[f"{tag}_{net}_dE_ALL"] - res[f"{tag}_{net}_dE_Q"]
    return res, E


def splits(H, beliefs):
    sp = np.load(os.path.join(SCAN, "figdata", "round4b", "split_pairs.npz"))
    X = posterior_means(beliefs["Trained"].astype(np.float64))[:, :, 1:]                # (40, 5000, T, 2)
    lags = np.arange(-6, 9)
    res = {"lags": lags}
    for net, Z in H.items():
        Q = Z["Q"].astype(np.float64)
        dq, dn, db = [], [], []
        for e in ENVS:
            pu = Z[f"pair_u_e{e:02d}"].astype(np.float64)                                 # (n, 2, T, H)
            rows = Z[f"pair_rows_e{e:02d}"]
            if not len(rows):
                continue
            pin = np.flatnonzero(sp["env"] == e)
            assert (np.stack([sp["i"][pin], sp["j"][pin]], 1) == rows).all()
            d = pu[:, 0] - pu[:, 1]                                                       # (n, T, H)
            q = d @ Q
            nq = np.sqrt(np.maximum((d ** 2).sum(-1) - (q ** 2).sum(-1), 0))
            for r_, p_ in enumerate(pin):
                ts = int(sp["ts"][p_]) - 1                                                # index of step t_s
                idx = ts + lags
                ok = (idx >= 0) & (idx < T)
                rowq, rown, rowb = (np.full(len(lags), np.nan) for _ in range(3))
                rowq[ok], rown[ok] = np.linalg.norm(q[r_, idx[ok]], axis=-1), nq[r_, idx[ok]]
                rowb[ok] = np.linalg.norm(X[e, rows[r_, 0], idx[ok]] - X[e, rows[r_, 1], idx[ok]], axis=-1)
                dq.append(rowq), dn.append(rown), db.append(rowb)
        res[f"{net}_dq"], res[f"{net}_dn"], res[f"{net}_db"] = np.array(dq), np.array(dn), np.array(db)
    return res


MEASURES = ("rev", "nov", "l2", "pr", "turn", "coh", "dW", "band_x", "dim")


def measure_arrays(net, meta, H, C):
    """name -> (39, 1000, T) on the credit streams (the first 10 per cell)"""
    rows = C["rows"]
    out = {"rev": meta["rev"][ENVS][:, rows].astype(np.float64)}
    for k in ("turn", "l2", "pr"):
        out[k] = np.stack([H[f"{k}_e{e:02d}"][rows] for e in ENVS]).astype(np.float64)
    for k in ("nov", "coh", "dW", "band_x", "dim"):
        out[k] = np.stack([C[f"{k}_e{e:02d}"] for e in ENVS]).astype(np.float64)
    return out, rows


def measures_jumps(meta, beliefs, H, Cr):
    """4a: the Trained RNN's isolated jumps (steps 6-20) among the credit streams, distance-matched controls"""
    rng = np.random.default_rng(0)
    rows = Cr["trained"]["rows"]
    X = posterior_means(beliefs["Trained"][ENVS][:, rows].astype(np.float64))
    mv = np.linalg.norm(np.diff(X, axis=2), axis=-1)                                      # (39, n, T)
    truth = meta["cells"][meta["cell_of"][rows]].astype(float)
    dtrue = np.linalg.norm(X[:, :, 1:] - truth[None, :, None], axis=-1)
    J = mv > JUMP
    win = lambda k: slice(k + LAGS[0], k + LAGS[-1] + 1)
    E, C_ = [], []
    for p in range(len(ENVS)):
        calm = {k: np.flatnonzero(~J[p][:, win(k)].any(1)) for k in range(K0, K1 + 1)}
        for i, k in zip(*np.nonzero(J[p])):
            if not (K0 <= k <= K1) or J[p, i, win(k)].sum() != 1:
                continue
            cand = calm[k][np.abs(dtrue[p, calm[k], k - 1] - dtrue[p, i, k - 1]) < .5]
            if len(cand):
                E.append((p, i, k)), C_.append((p, int(rng.choice(cand)), k))
    E, C_ = np.array(E), np.array(C_)
    out = {"n": len(E)}
    for net in ("trained", "echo"):
        A, _ = measure_arrays(net, meta, H[net], Cr[net])
        for k in MEASURES:
            st = E[:, 2:3] + LAGS[None]
            out[f"{net}_{k}"] = A[k][E[:, :1], E[:, 1:2], st] - A[k][C_[:, :1], C_[:, 1:2], st]   # (n, L)
    return out


def measures_splits(meta, beliefs, H, Cr, Cp):
    """4b: split pairs aligned on t_s; mover and stayer, each minus its environment's mean at that step"""
    sp = np.load(os.path.join(SCAN, "figdata", "round4b", "split_pairs.npz"))
    lags = np.arange(-6, 9)
    out = {"lags": lags}
    Xt = posterior_means(beliefs["Trained"].astype(np.float64))[:, :, 1:]
    for net in ("trained", "echo"):
        A, _ = measure_arrays(net, meta, H[net], Cr[net])
        mean_es = {k: A[k].mean(1) for k in MEASURES}                                       # (39, T)
        rows_all = {}
        mov, sta = ({k: [] for k in MEASURES} for _ in range(2))
        for p, e in enumerate(ENVS):
            pin = np.flatnonzero(sp["env"] == e)
            if not len(pin):
                continue
            rws = Cp[net][f"rows_e{e:02d}"]
            pos = {int(r): n for n, r in enumerate(rws)}
            vals = {}
            for k in MEASURES:
                if k == "rev":
                    vals[k] = meta["rev"][e][rws].astype(np.float64)
                elif k in ("turn", "l2", "pr"):
                    vals[k] = H[net][f"{k}_e{e:02d}"][rws].astype(np.float64)
                else:
                    vals[k] = Cp[net][f"{k}_e{e:02d}"].astype(np.float64)
            for q in pin:
                i, j, ts = int(sp["i"][q]), int(sp["j"][q]), int(sp["ts"][q]) - 1
                k2 = min(ts + 2, T - 1)
                mi, mj = (np.linalg.norm(Xt[e, s_, k2] - Xt[e, s_, ts]) for s_ in (i, j))
                a_, b_ = (i, j) if mi >= mj else (j, i)
                idx = ts + lags
                ok = (idx >= 0) & (idx < T)
                for k in MEASURES:
                    for lst, s_ in ((mov, a_), (sta, b_)):
                        row = np.full(len(lags), np.nan)
                        row[ok] = vals[k][pos[s_], idx[ok]] - mean_es[k][p, idx[ok]]
                        lst[k].append(row)
        for k in MEASURES:
            out[f"{net}_{k}_mover"], out[f"{net}_{k}_stayer"] = np.array(mov[k]), np.array(sta[k])
    return out


def main():
    H = {net: np.load(os.path.join(OUT, f"scan_hidden_{net}.npz")) for net, _ in NETS}
    meta = np.load(os.path.join(SCAN, "scan_meta.npz"))
    beliefs = {a: np.load(os.path.join(SCAN, f"scan_{a}.npz"))["beliefs"] for a in ("Joint", "Naive", "Trained", "Echo")}
    Cr = {net: np.load(os.path.join(OUT, f"scan_credit_{net}.npz")) for net, _ in NETS}
    Cp = {net: np.load(os.path.join(OUT, f"scan_credit_{net}_pairs.npz")) for net, _ in NETS}
    D = decoding(H)
    DC = decoding_controls(H, beliefs)
    MJ = measures_jumps(meta, beliefs, H, Cr)
    MS = measures_splits(meta, beliefs, H, Cr, Cp)
    Jr, E = jumps(H, meta, beliefs)
    S = splits(H, beliefs)
    out = {}
    for net in D:
        for s, v in D[net].items():
            for k, a in v.items():
                out[f"dec_{net}_{s}_{k}"] = a
    out.update({f"jump_{k}": v for k, v in Jr.items()})
    out.update({f"decctl_{k}": v for k, v in DC.items()})
    out.update({f"split_{k}": v for k, v in S.items()})
    out.update({f"mjump_{k}": v for k, v in MJ.items()})
    out.update({f"msplit_{k}": v for k, v in MS.items()})
    np.savez(os.path.join(OUT, "scan_jumps.npz"), **out)
    q = lambda a: "%.3f [%.3f, %.3f]" % (np.median(a), *np.percentile(a, [25, 75]))
    print("1  decoding the Optimal's joint covariance (CV R^2, median [IQR] over 39 environments)")
    for net in D:
        for s in D[net]:
            print(f"   {net:8s} {s:10s} all {q(D[net][s]['env'])}   top-10% |cov| {q(D[net][s]['top'])}")
    print("1b decoding controls (CV R^2, median [IQR]): the networks' top-64 output-null coordinates vs random ReLU "
          "expansions of the Optimal's marginals")
    for k, v in DC.items():
        print(f"   {k:14s} {q(v)}")
    L = list(LAGS)
    show = [L.index(l) for l in (-2, -1, 0, 1, 2, 4, 6, 10)]
    print(f"2  {Jr['n_events']} Trained jumps (isolated, steps 6-20) with same-environment distance-matched controls")
    print("   lag " + " ".join(f"{l:>7d}" for l in LAGS[show]))
    for key in ("move_Trained", "move_Echo", "move_Joint", "dtrue", "entpct", "trained_dE_NULL", "trained_dE_Q",
                "echo_dE_NULL", "echo_dE_Q", "trained_turn", "echo_turn", "trained_pr", "echo_pr", "trained_l2"):
        d = Jr[f"ev_{key}"] - Jr[f"ct_{key}"]
        m, s2 = d.mean(0), 2 * d.std(0, ddof=1) / np.sqrt(len(d))
        print(f"   {key:16s}" + " ".join(f"{m[i]:+7.3f}" for i in show) + "   (event - control; 2 SEM ~ %.3f)" % np.median(s2))
    print("3  split pairs: distance between the two streams, aligned on t_s (median over pairs)")
    for net in ("trained", "echo"):
        for part in ("db", "dq", "dn"):
            a = S[f"{net}_{part}"]
            print(f"   {net:8s} {part}: " + " ".join(f"{np.nanmedian(a[:, i]):6.3f}" for i in range(len(S['lags']))))
    print("   lags:", list(S["lags"]))
    print(f"4a measures at the Trained RNN's jumps among the credit streams (n = {MJ['n']}): event - control, mean")
    for net in ("trained", "echo"):
        for k in MEASURES:
            d = MJ[f"{net}_{k}"]
            m_, s_ = d.mean(0), 2 * d.std(0, ddof=1) / np.sqrt(len(d))
            print(f"   {net:8s} {k:7s}" + " ".join(f"{m_[i]:+8.3g}" for i in show) + f"   (2 SEM ~ {np.median(s_):.2g})")
    print("4b measures around splits: mover and stayer minus the environment's step mean (mean over pairs); lags",
          list(MS["lags"]))
    for net in ("trained", "echo"):
        for k in MEASURES:
            for role in ("mover", "stayer"):
                a = MS[f"{net}_{k}_{role}"]
                print(f"   {net:8s} {k:7s} {role:6s}" + " ".join(f"{np.nanmean(a[:, i]):+8.3g}" for i in range(len(MS['lags']))))


if __name__ == "__main__":
    main()
