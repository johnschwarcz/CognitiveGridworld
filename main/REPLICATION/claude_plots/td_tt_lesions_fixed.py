"""The latent-interaction lesion with FIXED directions (the same at every step) and more than the top dimensions (John
11-03), next to the previous per-step top-5 lesion (td_tt_lesions.py (c)), which it recomputes as a reproduction check.

John 11-03 (verbatim): "I dont think we need to constrain it to only top dimensions of interaction, but it is probably
important to use the same directions at every step of inference." ... "Yes please do these lesion so we can recreate the
previous one."

-> through_training/lesions/trained/fixed_EEEEE.npz (run_fixed; the 20 lesion checkpoints; GPU, ~2 min each)
-> main/REPLICATION/figure_bin/td_tt_lesions_fixed.png (figure()); numbers()

DEFINITIONS (per checkpoint; u = h, the short-term memory; clamp after every step u <- u - (u - mu_s) V V^T, mu_s the
clean per-step probe mean, V orthogonal to the 20 read-out rows, so every effect goes through the dynamics; scoring,
families and gap units as td_tt_lesions.py)
  joint5: the previous lesion, recomputed (per step s, the top 5 eigenvectors of Pc C_joint(s) Pc, C_joint from the
    cell means of all probes).
  Fixed interaction directions: the eigenvectors of Pc C_x Pc, C_x = (1/30) Sum_s (E_A(s)^T E_B(s) + E_B^T E_A) / 200,
    the noise-free interaction covariance pooled over the steps (E_A, E_B the r0 x r1 interaction effects of the cell
    means of environments 0-24 and 25-49). fix5 = its top 5; fixA / fixB = the leading k50 / k90 directions holding 50%
    / 90% of its positive eigenvalue mass. main90 = the top k90 of the same estimator for the r0 + r1 main effects.
  Controls (_vm_s0, _vm_s1): fixed random directions orthogonal to the read-out rows (td_tt_lesions's rotations, seeds
    SEED_ROT + 0, 1), as many leading columns as give the same clean variance pooled over the steps (tr V^T C_w V, C_w
    the within-step covariance averaged over steps).
RESULTS (numbers(); specific effect = target - the mean of its two variance-matched random controls, gap units,
  [5-95%]; the 20 lesion checkpoints)
  Reproduction: the recomputed per-step joint5 equals the stored first pass (rows max |d| 4.6e-8; intact 0).
  The noise-free interaction pooled over the steps is concentrated: k50 = 1 fixed direction (2 at the end), k90 = 2
  (80 episodes) to 9 (end); 5 fixed interaction directions carry the clean variance of ~77 random ones at the end.
  Fixed top 5 (the same at every step) recreates the per-step lesion and is slightly stronger late: answers -.167
  [-.188, -.145] at the end (per-step -.159), -.20 to -.23 at 1,600-8,000; holding -.456 [-.486, -.424] (per-step
  -.374), from ~1,600 on; coupling -.391 [-.416, -.367] (-.365). fixB (90%, 2-9 dims) = fix5 within the intervals.
  A single fixed direction (fixA, k50 = 1) already gives answers -.13 to -.21 from 1,120 on, holding -.32, coupling
  -.25 at the end. The main-effect subspace of the same size is less specific: answers -.04 to -.10, not resolved
  from 5,600 to 16,000; holding +.0 to +.11 (less than random dims); coupling -.06 until 8,000, -.24 at the end.
  simple_figure (td_tt_lesions_simple.png; fix5, share of the naive-to-Bayes gap lost beyond the random controls):
  corner answers lose the most, from ~1,600 episodes (.14) to .55-.58 from 8,000 on (per-step lesion .46-.52); the
  other answers .10-.25 up to ~2,000, then down to .06; holding .46 and coupling .39 at the end. By FR at the last
  step (thirds of the probes): goal accuracy lost is largest in the high-FR third (+5 pp at 800, +10-11 at 1,600-5,600,
  +6 at the end), the middle third peaks at +5 (1,600), the low third <= +4 and +.4 at the end.
"""
import os
import sys
import json
import time
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_tt_gap as gap                                                                       # noqa: E402

T, R, H = 30, 10, 1000
FIXED = ("fix5", "fixA", "fixB", "main90")
NAMES = ("fix5 (top 5)", "fixA (50% of the interaction)", "fixB (90%)", "main effects (90% size)")


def fixed_file(E, run="trained"):
    import td_tt_lesions as tl
    return os.path.join(tl.OUT, run, f"fixed_{E:05d}.npz")


def define_fixed(w, ref, dev):
    import torch
    import td_tt_intervene as ti
    import td_tt_lesions as tl
    f64 = torch.float64
    P = ti.probes()
    cell, env = np.asarray(P["cell"], np.int64), np.asarray(P["env"], np.int64)
    cell_t = torch.as_tensor(cell, device=dev)
    half_t = torch.as_tensor(env >= gap.N_ENV // 2, device=dev)
    S = torch.zeros(T + 1, 3, R * R, H, dtype=f64, device=dev)                        # all, half A, half B

    def rec(s, rows, st):
        rr = torch.as_tensor(rows, device=dev)
        u, hb = st["u"].double(), half_t[rr]
        S[s, 0].index_add_(0, cell_t[rr], u)
        S[s, 1].index_add_(0, cell_t[rr][~hb], u[~hb])
        S[s, 2].index_add_(0, cell_t[rr][hb], u[hb])
    tl.lforward(w, None, None, dev=dev, record=rec, budget=5000)
    Q = tl.readout_basis(w)
    Qt = torch.as_tensor(Q, device=dev)
    Pc = torch.eye(H, dtype=f64, device=dev) - Qt @ Qt.T

    def effects(M):
        Mg = M.reshape(R, R, H)
        g = Mg.mean((0, 1))
        a0, a1 = Mg.mean(1) - g, Mg.mean(0) - g
        return a0, a1, (Mg - a0[:, None] - a1[None] - g).reshape(R * R, H)
    Cw = torch.zeros(H, H, dtype=f64, device=dev)
    Cx, Cm = torch.zeros_like(Cw), torch.zeros_like(Cw)
    V5, mu_chk = [], 0.
    for s in range(1, T + 1):
        a0, a1, ej = effects(S[s, 0] / 100.)
        V5.append(torch.linalg.svd(ej @ Pc, full_matrices=False)[2][:5].T.cpu().numpy())   # td_tt_lesions.top_f
        A, B = effects(S[s, 1] / 50.), effects(S[s, 2] / 50.)
        Cx += (A[2].T @ B[2] + B[2].T @ A[2]) / (2 * R * R) / T
        Cm += sum((A[i].T @ B[i] + B[i].T @ A[i]) / (2 * R) for i in (0, 1)) / T
        Cw += torch.as_tensor(ref["cov_u"][s], device=dev) / T
        mu_chk = max(mu_chk, float(np.abs((S[s, 0].sum(0) / len(cell)).cpu().numpy() - ref["mu_u"][s]).max()))
    out = dict(V5=V5, mu_chk=mu_chk)
    for key, C in (("int", Cx), ("main", Cm)):
        M = Pc @ C @ Pc
        e, V = torch.linalg.eigh((M + M.T) / 2)
        out[f"ev_{key}"], out[f"V_{key}"] = e.flip(0).cpu().numpy(), V.flip(1).cpu().numpy()
    pos = np.clip(out["ev_int"], 0, None)
    cum = np.cumsum(pos) / pos.sum()
    out["k50"], out["k90"] = int(np.searchsorted(cum, .5) + 1), int(np.searchsorted(cum, .9) + 1)
    Vi, Vm = out["V_int"], out["V_main"]
    out["sub"] = {"fix5": Vi[:, :5], "fixA": Vi[:, :out["k50"]], "fixB": Vi[:, :out["k90"]],
                  "main90": Vm[:, :out["k90"]]}
    Cw_np = Cw.cpu().numpy()
    out["var"] = {k: float(np.trace(V.T @ Cw_np @ V)) for k, V in out["sub"].items()}
    out["tot"] = float(np.trace(Cw_np))
    out["int_held"] = {k: float(np.trace(V.T @ Cx.cpu().numpy() @ V) / np.trace(Pc.cpu().numpy() @ Cx.cpu().numpy()))
                       for k, V in out["sub"].items()}
    rot = {}
    for sd in range(tl.N_SUB):                                                          # td_tt_lesions.define's rotations
        G = np.random.default_rng(tl.SEED_ROT + sd).standard_normal((H, H - Q.shape[1]))
        G = G - Q @ (Q.T @ G)
        rot[sd] = np.linalg.qr(G)[0]
        c = np.cumsum(((Cw_np @ rot[sd]) * rot[sd]).sum(0))
        for k in FIXED:
            m = int(np.argmin(np.abs(c - out["var"][k]))) + 1
            out["sub"][f"{k}_vm_s{sd}"] = rot[sd][:, :m]
            out["var"][f"{k}_vm_s{sd}"] = float(c[m - 1])
    return out


def run_fixed(E, dev=None, run="trained"):
    import torch
    import td_tt_intervene as ti
    import td_tt_lesions as tl
    dev = ti._device(dev)
    t0 = time.time()
    w = ti.load(run, E)
    ref = ti.reference(w, dev, cov=True)
    D = define_fixed(w, ref, dev)
    ref_nc = {k: v for k, v in ref.items() if k != "cov_u"}
    conds = [("intact", "intact", {}), ("joint5", "sub", {"V": D["V5"], "mode": "clamp"})]
    for k in FIXED:
        conds.append((k, "fixed", {"V": D["sub"][k], "mode": "clamp"}))
        conds += [(f"{k}_vm_s{sd}", "fixed", {"V": D["sub"][f"{k}_vm_s{sd}"], "mode": "clamp"}) for sd in range(tl.N_SUB)]
    t_def = time.time() - t0
    names, cls, val, boot, info = [], [], [], [], []
    lq_int = None
    for nm, cl, spec in conds:
        t1 = time.time()
        fo = tl.lforward(w, spec, ref_nc, dev=dev)
        ff = tl.lforward(w, spec, ref_nc, fwd=True, dev=dev)
        if nm == "intact":
            lq_int = fo["logq"]
        v, b = tl.score_all(fo, ff, lq_int)
        names.append(nm)
        cls.append(cl)
        val.append(v)
        boot.append(b)
        V = spec.get("V")
        info.append(dict(dims=(int(V.shape[1]) if isinstance(V, np.ndarray) else [int(a.shape[1]) for a in V]) if V is
                         not None else 0, var=D["var"].get(nm, np.nan), int_held=D["int_held"].get(nm, np.nan)))
        print(f"{run} {E} {nm:12s} dims {info[-1]['dims'] if isinstance(info[-1]['dims'], int) else 5:4d} acc "
              f"{v[tl.ROWS.index('acc_goal_T')]:.4f} held2 {v[tl.ROWS.index('held_two_fwd')]:.4f} ({time.time() - t1:.1f} s)",
              flush=True)
    fn = fixed_file(E, run)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    tmp = fn + ".tmp.npz"
    np.savez(tmp, run=run, E=E, x=float(ti.train_x(E)), names=np.array(names), cls=np.array(cls), rows=np.array(tl.ROWS),
             value=np.array(val), boot=np.array(boot), info=json.dumps(info), k50=D["k50"], k90=D["k90"],
             ev_int=D["ev_int"], ev_main=D["ev_main"], tot=D["tot"], mu_chk=D["mu_chk"], t_def=t_def)
    os.replace(tmp, fn)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    ti._REF.clear()
    print(f"saved {fn} ({time.time() - t0:.0f} s; k50 {D['k50']}, k90 {D['k90']})", flush=True)


def fr_file(E, run="trained"):
    import td_tt_lesions as tl
    return os.path.join(tl.OUT, run, f"frsplit_{E:05d}.npz")


def run_fr(E, dev=None, run="trained"):
    """per probe: the goal right at T under intact, fix5 and its two random controls (probes only) -> frsplit_EEEEE.npz"""
    import torch
    import td_tt_intervene as ti
    import td_tt_lesions as tl
    dev = ti._device(dev)
    t0 = time.time()
    w = ti.load(run, E)
    ref = ti.reference(w, dev, cov=True)
    D = define_fixed(w, ref, dev)
    ref_nc = {k: v for k, v in ref.items() if k != "cov_u"}
    P = ti.probes()
    ar, goal = np.arange(len(P["env"])), np.asarray(P["goal"], np.int64)
    out = {}
    for nm in ("intact", "fix5", "fix5_vm_s0", "fix5_vm_s1"):
        spec = {} if nm == "intact" else {"V": D["sub"][nm], "mode": "clamp"}
        out[nm] = tl.lforward(w, spec, ref_nc, dev=dev)["right"][ar, -1, goal]
    os.makedirs(os.path.dirname(fr_file(E, run)), exist_ok=True)
    np.savez(fr_file(E, run), E=E, x=float(ti.train_x(E)), **out)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    ti._REF.clear()
    print(f"{E}: {time.time() - t0:.0f} s; acc intact {out['intact'].mean():.4f} fix5 {out['fix5'].mean():.4f}", flush=True)


def worker_fr(part, nparts):
    import td_tt_lesions as tl
    for E in tl.CKS[part::nparts]:
        if not os.path.exists(fr_file(E)):
            run_fr(E)
    print("done", flush=True)


def worker(part, nparts):
    import td_tt_lesions as tl
    for E in tl.CKS[part::nparts]:
        if not os.path.exists(fixed_file(E)):
            run_fixed(E)
    print("done", flush=True)


def load():
    import td_tt_lesions as tl
    Z = [np.load(fixed_file(E)) for E in tl.CKS if os.path.exists(fixed_file(E))]
    return dict(x=np.array([float(z["x"]) for z in Z]), E=np.array([int(z["E"]) for z in Z]),
                names=[str(v) for v in Z[0]["names"]], rows=list(tl.ROWS), value=np.stack([z["value"] for z in Z]),
                boot=np.stack([z["boot"] for z in Z]), info=[json.loads(str(z["info"])) for z in Z],
                k50=np.array([int(z["k50"]) for z in Z]), k90=np.array([int(z["k90"]) for z in Z]))


def specific(L, FE, name):
    """target - the mean of its two variance-matched random controls, per family: (K, fam) and resamples (K, fam, B)"""
    i = L["names"].index(name)
    c = [L["names"].index(f"{name}_vm_s{sd}") for sd in range(2)]
    return (FE["dF"][:, i] - FE["dF"][:, c].mean(1), FE["dF_b"][:, i] - FE["dF_b"][:, c].mean(1))


def numbers():
    import td_tt_lesions as tl
    L = load()
    FE = tl.family_effects(L)
    Lo = tl.load_run("trained")
    FEo = tl.family_effects(Lo)
    fams = list(tl.FAMILY_ROWS)
    io = Lo["names"].index("joint5")
    ko = [list(Lo["E"]).index(E) for E in L["E"]]
    print(f"{len(L['x'])} checkpoints; k50 {L['k50'].tolist()}; k90 {L['k90'].tolist()}")
    d = np.nanmax(np.abs(L["value"][:, L["names"].index("joint5")] - Lo["value"][ko, io]))
    di = np.nanmax(np.abs(L["value"][:, L["names"].index("intact")] - Lo["value"][ko, Lo["names"].index("intact")]))
    print(f"reproduction: joint5 rows vs the stored first pass max |d| {d:.2e}; intact {di:.2e}")
    so = FEo["dF"][ko, io] - FEo["dF"][ko][:, [Lo["names"].index(f"joint5_vm_s{sd}") for sd in range(2)]].mean(1)
    for j, fam in enumerate(fams):
        print(f"== {fam}: specific effect (target - variance-matched random), gap units")
        print("  old joint5 (stored): " + ", ".join(f"{x:,.0f} {v:+.3f}" for x, v in zip(L["x"], so[:, j])))
        for nm in FIXED:
            v, b = specific(L, FE, nm)
            lo, hi = np.percentile(b[:, j], 5, -1), np.percentile(b[:, j], 95, -1)
            print(f"  {nm:7s}: " + ", ".join(f"{x:,.0f} {a:+.3f} [{c:+.3f}, {e:+.3f}]" for x, a, c, e in
                                           zip(L["x"], v[:, j], lo, hi)))
        for nm in ("joint5",) + FIXED:
            i = L["names"].index(nm)
            print(f"  raw {nm:7s}: " + ", ".join(f"{x:,.0f} {v:+.3f}" for x, v in zip(L["x"], FE["dF"][:, i, j])))


def figure(path=None):
    import matplotlib.pyplot as plt
    import td_tt_lesions as tl
    from main.utils import fig_path, apply_fig_style
    apply_fig_style()
    L = load()
    FE = tl.family_effects(L)
    Lo = tl.load_run("trained")
    FEo = tl.family_effects(Lo)
    io = Lo["names"].index("joint5")
    ko = [list(Lo["E"]).index(E) for E in L["E"]]
    so = FEo["dF"][ko, io] - FEo["dF"][ko][:, [Lo["names"].index(f"joint5_vm_s{sd}") for sd in range(2)]].mean(1)
    sob = FEo["dF_b"][ko, io] - FEo["dF_b"][ko][:, [Lo["names"].index(f"joint5_vm_s{sd}") for sd in range(2)]].mean(1)
    fams = list(tl.FAMILY_ROWS)
    lx = np.log10(L["x"])
    cols = ("#fdae61", "#d7191c", "#7b0000", "#2c7bb6")
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.5), gridspec_kw=dict(hspace=.45, wspace=.25, left=.06, right=.98,
                                                                       top=.86, bottom=.1))
    for j, fam in enumerate(fams):
        ax = axs[0, j]
        ax.fill_between(lx, np.percentile(sob[:, j], 5, -1), np.percentile(sob[:, j], 95, -1), color="k", alpha=.12, lw=0)
        ax.plot(lx, so[:, j], "k--", lw=1.8, label="per-step top 5 (previous)")
        for nm, nmf, c in zip(FIXED, NAMES, cols):
            v, b = specific(L, FE, nm)
            ax.fill_between(lx, np.percentile(b[:, j], 5, -1), np.percentile(b[:, j], 95, -1), color=c, alpha=.15, lw=0)
            ax.plot(lx, v[:, j], color=c, lw=2, label=f"fixed: {nmf}")
        ax.axhline(0, color="k", lw=.6)
        ax.set_title(f"{fam}: removed beyond variance-matched random dims", loc="left", fontsize=11)
        ax.set_ylabel("specific effect (gap units)")
        ax = axs[1, j]
        for nm, nmf, c in zip(("joint5",) + FIXED, ("per-step top 5",) + NAMES, ("k",) + cols):
            ax.plot(lx, FE["dF"][:, L["names"].index(nm), j], color=c, lw=2, ls="--" if nm == "joint5" else "-")
            if nm != "joint5":
                ax.plot(lx, FE["dF"][:, [L["names"].index(f"{nm}_vm_s{sd}") for sd in range(2)], j].mean(1), color=c,
                        lw=1, ls=":")
        ax.axhline(0, color="k", lw=.6)
        ax.set_title(f"{fam}: raw change (dotted: the random controls)", loc="left", fontsize=11)
        ax.set_ylabel("change of the gap fraction")
        for a in axs[:, j]:
            a.set_xticks([2, 3, 4])
            a.set_xticklabels(["100", "1k", "10k"])
            a.set_xlabel("training episodes")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .935), ncol=5,
               frameon=False, fontsize=9.5)
    fig.suptitle("Interaction lesion with FIXED directions (the same at every step) vs the previous per-step top-5 "
                 f"lesion; k50 = {int(np.median(L['k50']))}, k90 = {int(np.median(L['k90']))} directions (median over "
                 "checkpoints)", fontsize=12.5)
    fig.text(.5, .01, f"Trained RNN, {len(L['x'])} checkpoints; 10,000 probes + the 5,028-episode forward set per "
             "condition; families and gap units as in the lesion module; bands: 5-95% over 1,000 environment resamples; "
             "controls: 2 seeds of fixed random directions with the same pooled clean variance.", ha="center", fontsize=9)
    path = fig_path("td_tt_lesions_fixed.png") if path is None else path
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path)


GROUPS = (("corner answers", ("corner_truth_fwd",)), ("other answers", ("acc_goal_T", "acc_both_T", "far_T", "merged_T")),
          ("holding", None), ("coupling", None))
GCOLS = ("#d7191c", "#fdae61", "#2c7bb6", "#1a9641")


def row_loss(L, name="fix5"):
    """per row group: the share of the naive-to-Bayes gap lost beyond the two random controls, (K,) and resamples (K, B)"""
    import td_tt_lesions as tl
    rr, f, fb = tl.gapfrac(L)
    i = L["names"].index(name)
    c = [L["names"].index(f"{name}_vm_s{sd}") for sd in range(2)]
    out = {}
    for g, rows in GROUPS:
        ii = [rr.index(r) for r in (rows or tl.FAMILY_ROWS[g])]
        with np.errstate(invalid="ignore"):
            out[g] = (-np.nanmean(f[:, i, ii] - f[:, c][:, :, ii].mean(1), -1),
                      -np.nanmean(fb[:, i, ii] - fb[:, c][:, :, ii].mean(1), 1))
    return out


def fr_loss():
    """goal accuracy at T lost beyond the random controls, per third of FR (at T, both variables): x, (3, K, 1 + B)"""
    import td_tt_lesions as tl
    import td_tt_intervene as ti
    import td_tt_chaos_combined as cc
    P = ti.probes()
    with np.load(os.path.join(ti.AN, "probes.npz")) as Zp:
        fr = Zp["fr"].astype(np.float64)[:, -1].mean(1)
    edges = np.quantile(fr, [0, 1 / 3, 2 / 3, 1])
    g = np.clip(np.searchsorted(edges, fr, side="right") - 1, 0, 2)
    env = np.asarray(P["env"], np.int64)
    xs, out, acc = [], [], []
    for E in tl.CKS:
        with np.load(fr_file(E)) as z:
            xs.append(float(z["x"]))
            loss = (z["fix5_vm_s0"].astype(float) + z["fix5_vm_s1"].astype(float)) / 2 - z["fix5"].astype(float)
            out.append([cc._envr(np.bincount(env[g == j], loss[g == j], gap.N_ENV),
                                 np.bincount(env[g == j], minlength=gap.N_ENV)) for j in range(3)])
            acc.append([z["intact"][g == j].mean() for j in range(3)])
    return np.array(xs), np.array(out).transpose(1, 0, 2), np.array(acc), edges


def simple_figure(path=None):
    import matplotlib.pyplot as plt
    from main.utils import fig_path, apply_fig_style
    apply_fig_style()
    L = load()
    lx = np.log10(L["x"])
    RL = row_loss(L)
    xf, FL, acc, edges = fr_loss()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw=dict(wspace=.25, left=.07, right=.98, top=.8, bottom=.14))
    for (g, _), c in zip(GROUPS, GCOLS):
        v, b = RL[g]
        ax[0].fill_between(lx, np.nanpercentile(b, 5, -1), np.nanpercentile(b, 95, -1), color=c, alpha=.18, lw=0)
        ax[0].plot(lx, v, color=c, lw=2.4, label=g)
    ax[0].set_ylabel("share of the naive-to-Bayes gap lost")
    ax[0].set_title("a  Which behaviours need the 5 interaction directions?", loc="left", fontsize=12)
    ax[0].legend(frameon=False, fontsize=10, loc="upper left")
    tc = ("#9ecae1", "#4292c6", "#08306b")
    lab = (f"low FR ($<${edges[1]:.2f} nats)", "middle FR", f"high FR ($>${edges[2]:.2f} nats)")
    for j in range(3):
        v, lo, hi = FL[j][:, 0], np.percentile(FL[j][:, 1:], 5, 1), np.percentile(FL[j][:, 1:], 95, 1)
        ax[1].fill_between(np.log10(xf), 100 * lo, 100 * hi, color=tc[j], alpha=.2, lw=0)
        ax[1].plot(np.log10(xf), 100 * v, color=tc[j], lw=2.4, label=lab[j])
    ax[1].set_ylabel("goal accuracy at T lost (percentage points)")
    ax[1].set_title("b  Does the need depend on factorization regret (FR)?", loc="left", fontsize=12)
    ax[1].legend(frameon=False, fontsize=10, loc="upper left", title="episodes split by FR at the last step")
    for a in ax:
        a.axhline(0, color="k", lw=.7)
        a.set_xticks([2, 3, 4])
        a.set_xticklabels(["100", "1k", "10k"])
        a.set_xlabel("training episodes")
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Clamping the 5 fixed interaction directions of the short-term memory (the same at every step), minus "
                 "clamping random directions with the same variance", fontsize=12.5)
    fig.text(.5, .01, "Trained RNN, 20 checkpoints; 10,000 probes (b) and the 5,028-episode forward set (corner and "
             "holding rows); random controls: 2 seeds; bands: 5-95% over 1,000 environment resamples.", ha="center",
             fontsize=9)
    path = fig_path("td_tt_lesions_simple.png") if path is None else path
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("saved", path)
    for g, _ in GROUPS:
        print(f"  {g:15s} " + ", ".join(f"{x:,.0f} {v:+.3f}" for x, v in zip(L["x"], RL[g][0])))
    print("  FR edges", np.round(edges, 3))
    for j in range(3):
        print(f"  FR third {j}: loss (pp) " + ", ".join(f"{x:,.0f} {100 * v:+.1f}" for x, v in zip(xf, FL[j][:, 0])) +
              "; intact acc " + ", ".join(f"{a:.2f}" for a in acc[:, j]))


def corner_check():
    """the previous per-step lesion (stored) by row group, for the corner claim"""
    import td_tt_lesions as tl
    Lo = tl.load_run("trained")
    RL = row_loss(Lo, "joint5")
    for g, _ in GROUPS:
        print(f"  old per-step {g:15s} " + ", ".join(f"{x:,.0f} {v:+.3f}" for x, v in zip(Lo["x"], RL[g][0])))


if __name__ == "__main__":
    what = sys.argv[1:] or ["figure", "numbers"]
    if "worker" in what:
        i = what.index("worker")
        worker(int(what[i + 1]), int(what[i + 2]))
    if "worker_fr" in what:
        i = what.index("worker_fr")
        worker_fr(int(what[i + 1]), int(what[i + 2]))
    if "simple" in what:
        simple_figure()
        corner_check()
    if "one" in what:
        run_fixed(int(what[what.index("one") + 1]))
    if "figure" in what:
        figure()
    if "numbers" in what:
        numbers()
