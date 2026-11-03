"""Entanglement as a driver of learning and of memory use, through training (John 11-01: "Im still not convinced that
entanglement is just a consequence, rather than a driver of the need for memory").
-> data main/DATA/experimental/training_dynamics/analysis/ent_driver_<run>.npz (every checkpoint; GPU 0, all 10,000
   probe episodes, fp32 hand-written LSTM = td_analysis's replay)

Bins: as td_entanglement_split.py, quintiles within every step (steps 2-30) of the probe entanglement rev (the mean
over the two variables; the locked definition); group 0 = step 1 (its entanglement is the within-step part only),
groups 1-5 = the rev quintiles. Per checkpoint:
  memory cut (an6's rule, per step): the network's per-step increment when its memory is cut to its last w steps (the
    cell rerun from its start state over inputs t-w+1..t), its disagreement with Bayes's per-step increment
    (Jeffreys/2 between the softmaxes, both variables averaged) summed per (step, bin): "dis_cut" (W, T, N_BIN), and the
    intact network's "dis_int" (T, N_BIN), counts "cnt" (T, N_BIN); for t < w the cut equals the intact run; the same
    weighted by Bayes's belief ("_w": the increment applied to Bayes's belief at t-1 against Bayes's belief at t,
    Jeffreys/2 between the two beliefs; errors on values Bayes already rules out do not count);
  gradient by use step: the run's own training loss (SANITY: goal marginal at every step, symmetric KL with the probe
    set's minimum subtracted, power 0.5, mean over items) has a gradient that is a sum over the steps at which each
    weight is USED (W_hh at step t: delta_t u_{t-1}^T; W_ih: delta_t x_t^T; read-in at step t; read-out at step t:
    e_t u_t^T with e_t the loss's error on the increment of step t, i.e. on every later belief). Each (particle, step)
    use is assigned to its group; one copy of every weight per group in one forward pass gives each group's part g_k
    exactly (sum over groups = the whole gradient g). Per parameter group (read-in, LSTM input weights, LSTM recurrent
    weights, LSTM biases, read-out, start state): "gg" <g_k, g> (additive shares of |g|^2), "gdn" <g_k, -dW_next> with
    the realized change to the next checkpoint, "gdf" <g_k, -dW_final> with the change still to come, "gnorm2"
    |g_k|^2; "dn2", "df2" |dW_next|^2, |dW_final|^2.
Run from the repository root (GPU 0):
    CUDA_VISIBLE_DEVICES=0 /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/claude_plots/td_entanglement_driver.py \\
        trained echo"""
import os
import sys
import time
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION")]
os.environ.setdefault("MPLBACKEND", "Agg")
import td_analysis as td                                                          # noqa: E402

AN = os.path.join(td.TD, "analysis")
N_BIN = 5
N_GRP = 1 + N_BIN
WINDOWS = (1, 2, 5)
CHUNK = 1000
MC_CHUNK = 400
GROUPS = {"readin": ("classifier_readin.weight", "classifier_readin.bias"),
          "lstm_ih": ("LSTM.weight_ih_l0",), "lstm_hh": ("LSTM.weight_hh_l0",),
          "lstm_b": ("LSTM.bias_ih_l0", "LSTM.bias_hh_l0"),
          "readout": ("classifier_readout.weight", "classifier_readout.bias"), "start": ("STM", "LTM")}
PER_STEP = [x for g in ("readin", "lstm_ih", "lstm_hh", "lstm_b", "readout") for x in GROUPS[g]]


def item_groups(P):
    """(n, T) group of every (particle, step): 0 at step 1, 1-5 the within-step quintile of rev"""
    rev = P["ent"].mean(-1)
    g = np.zeros(rev.shape, int)
    for t in range(1, td.T):
        q = np.quantile(rev[:, t], np.linspace(0, 1, N_BIN + 1)[1:-1])
        g[:, t] = np.searchsorted(q, rev[:, t]) + 1
    return g


def _jeff(a, b):
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
    return .5 * ((la.exp() - lb.exp()) * (la - lb)).sum(-1)


def _soft_clip(x, eps=1e-3):
    ciel = 1. - eps
    x = ciel - torch.sigmoid((ciel - x) / eps) * (ciel - x)
    return eps + torch.sigmoid((x - eps) / eps) * (x - eps)


def DKL_sym(x, y):
    """model_backward.DKL_sym with PM = False (soft-clipped, per value)"""
    x, y = _soft_clip(x), _soft_clip(y)
    return (x * torch.log(x / y) + y * torch.log(y / x)) / 2


def _start(p, n):
    H = p["STM"].shape[-1]
    return p["STM"].reshape(1, H).expand(n, H), p["LTM"].reshape(1, H).expand(n, H)


# ── memory cut (no gradients) ─────────────────────────────────────────────────────────────────────────────────────

def _pre(p, obs, zin):
    T = obs.shape[1]
    x = torch.relu(torch.cat([obs, zin[:, None].expand(-1, T, -1)], -1) @ p["classifier_readin.weight"].T
                   + p["classifier_readin.bias"])
    return x @ p["LSTM.weight_ih_l0"].T + p["LSTM.bias_ih_l0"] + p["LSTM.bias_hh_l0"]


def _run(p, x_pre, h, c):
    hs = []
    for t in range(x_pre.shape[1]):
        h, c, _ = td._cell(x_pre[:, t] + h @ p["LSTM.weight_hh_l0"].T, c)
        hs.append(h)
    return torch.stack(hs, 1)


def _readout(p, u):
    return u @ p["classifier_readout.weight"].T + p["classifier_readout.bias"]


@torch.no_grad()
def memory_cut(p, P, G, dev):
    n, T = len(P["goal"]), td.T
    dis_cut, dis_cut_w = np.zeros((len(WINDOWS), T, N_BIN)), np.zeros((len(WINDOWS), T, N_BIN))
    dis_int, dis_int_w, cnt = np.zeros((T, N_BIN)), np.zeros((T, N_BIN)), np.zeros((T, N_BIN))
    for c0 in range(0, n, MC_CHUNK):
        sl = slice(c0, c0 + MC_CHUNK)
        obs = torch.as_tensor(P["obs"][sl], dtype=torch.float32, device=dev)
        zin = torch.as_tensor(P["Z"][P["env"][sl]], device=dev).reshape(len(obs), -1)
        D = torch.as_tensor(P["inc_D"][sl], device=dev).double()
        lj = torch.as_tensor(P["log_jb"][sl], device=dev).double()                   # Bayes's log beliefs (m, T, 2, R)
        prior = torch.cat([torch.full_like(lj[:, :1], -np.log(td.R)), lj[:, :-1]], 1)
        b = G[sl] - 1                                                                # -1 at step 1
        xp = _pre(p, obs, zin)
        m = len(obs)
        u = _run(p, xp, *_start(p, m))
        d_int = _readout(p, u).double().reshape(m, T, td.C, td.R)
        e_int = _jeff(d_int, D).mean(-1).cpu().numpy()                             # (m, T)
        e_int_w = _jeff(prior + d_int, lj).mean(-1).cpu().numpy()
        cuts, cuts_w = [], []
        for w in WINDOWS:
            ts = np.arange(w - 1, T)                     # steps whose memory reaches back further than w inputs
            idx = torch.as_tensor(ts[:, None] - w + 1 + np.arange(w)[None], device=dev)   # (T', w)
            hw = _run(p, xp[:, idx].reshape(m * len(ts), w, -1), *_start(p, m * len(ts)))[:, -1]
            uw = u.clone()
            uw[:, ts] = hw.reshape(m, len(ts), -1)
            d_cut = _readout(p, uw).double().reshape(m, T, td.C, td.R)
            cuts.append(_jeff(d_cut, D).mean(-1).cpu().numpy())
            cuts_w.append(_jeff(prior + d_cut, lj).mean(-1).cpu().numpy())
        for t in range(1, T):
            for k in range(N_BIN):
                s = b[:, t] == k
                cnt[t, k] += s.sum()
                dis_int[t, k] += e_int[s, t].sum()
                dis_int_w[t, k] += e_int_w[s, t].sum()
                for j in range(len(WINDOWS)):
                    dis_cut[j, t, k] += cuts[j][s, t].sum()
                    dis_cut_w[j, t, k] += cuts_w[j][s, t].sum()
    return dis_cut, dis_int, dis_cut_w, dis_int_w, cnt


# ── gradient by use step ──────────────────────────────────────────────────────────────────────────────────────────

def attribution(w, P, G, dev):
    """each group's part of the training-loss gradient, by the step at which each weight is used: one forward and one
    backward per chunk, keeping the error at every step's LSTM pre-activation (delta_t), read-out output (e_t) and
    read-in pre-activation (alpha_t); the part of group k is the sum of the outer products over its (particle, step)
    uses (the start state is used at step 1 only)"""
    n, T = len(P["goal"]), td.T
    p = {x: v.to(dev).float().clone().requires_grad_(True) for x, v in w.items()}
    shapes = {x: p[x].shape for x in PER_STEP}
    gk = {x: [torch.zeros(shapes[x], dtype=torch.float64, device=dev) for _ in range(N_GRP)] for x in PER_STEP}

    def batch(sl):
        obs = torch.as_tensor(P["obs"][sl], dtype=torch.float32, device=dev)
        zin = torch.as_tensor(P["Z"][P["env"][sl]], device=dev).reshape(len(obs), -1)
        goal = torch.as_tensor(P["goal"][sl], device=dev)
        Pg = torch.as_tensor(P["jb"][sl][np.arange(len(obs)), :, P["goal"][sl]], device=dev)
        return obs, zin, goal, Pg

    def forward(obs, zin, goal, Pg, keep):
        m = len(obs)
        inp = torch.cat([obs, zin[:, None].expand(-1, T, -1)], -1)                  # (m, T, 15)
        a = inp @ p["classifier_readin.weight"].T + p["classifier_readin.bias"]
        x = torch.relu(a)
        h, c = _start(p, m)
        hs, pres, lgs = [h], [], []
        for t in range(T):
            pre = x[:, t] @ p["LSTM.weight_ih_l0"].T + p["LSTM.bias_ih_l0"] + p["LSTM.bias_hh_l0"] \
                + h @ p["LSTM.weight_hh_l0"].T
            h, c, _ = td._cell(pre, c)
            lg = h @ p["classifier_readout.weight"].T + p["classifier_readout.bias"]
            if keep:
                pre.retain_grad()
                lg.retain_grad()
            hs.append(h)
            pres.append(pre)
            lgs.append(lg)
        if keep:
            a.retain_grad()
        cum = torch.stack(lgs, 1).cumsum(1).reshape(m, T, td.C, td.R)
        q = torch.softmax(cum, -1)[torch.arange(m, device=dev), :, goal]
        return DKL_sym(q, Pg), (inp, a, x, hs, pres, lgs)

    with torch.no_grad():
        D_min = min(float(forward(*batch(slice(c0, c0 + CHUNK)), False)[0].min()) for c0 in range(0, n, CHUNK))
    for c0 in range(0, n, CHUNK):
        sl = slice(c0, c0 + CHUNK)
        Dk, (inp, a, x, hs, pres, lgs) = forward(*batch(sl), True)
        L = ((Dk - D_min + 1e-8) ** .5).sum() / (n * T * td.R)
        L.backward()
        with torch.no_grad():
            g = torch.as_tensor(G[sl], device=dev).T.reshape(-1)                     # (T * m,) step-major
            o = torch.argsort(g, stable=True)
            bounds = torch.searchsorted(g[o], torch.arange(N_GRP + 1, device=dev)).tolist()
            flat = lambda L_: torch.cat([v for v in L_], 0)[o]                      # noqa: E731  (T * m, d)
            delta, u_prev, u = flat([q.grad for q in pres]), flat(hs[:-1]), flat(hs[1:])
            e = flat([q.grad for q in lgs])
            xs = flat(x.transpose(0, 1))
            al, ins = flat(a.grad.transpose(0, 1)), flat(inp.transpose(0, 1))
            for k in range(N_GRP):
                s_ = slice(bounds[k], bounds[k + 1])
                gk["LSTM.weight_hh_l0"][k] += (delta[s_].T @ u_prev[s_]).double()
                gk["LSTM.weight_ih_l0"][k] += (delta[s_].T @ xs[s_]).double()
                gk["LSTM.bias_ih_l0"][k] += delta[s_].double().sum(0)
                gk["LSTM.bias_hh_l0"][k] += delta[s_].double().sum(0)
                gk["classifier_readout.weight"][k] += (e[s_].T @ u[s_]).double()
                gk["classifier_readout.bias"][k] += e[s_].double().sum(0)
                gk["classifier_readin.weight"][k] += (al[s_].T @ ins[s_]).double()
                gk["classifier_readin.bias"][k] += al[s_].double().sum(0)
        del Dk, inp, a, x, hs, pres, lgs, L, delta, u_prev, u, e, xs, al, ins
    for x in ("STM", "LTM"):                                                         # used at step 1 only
        gk[x] = [p[x].grad.double()] + [torch.zeros_like(p[x], dtype=torch.float64) for _ in range(N_BIN)]
    full = {x: p[x].grad.double() for x in p}
    return gk, D_min, full


def summarize(gk, w, w_next, w_final):
    gnames = list(GROUPS)
    out = {k: np.zeros((N_GRP, len(gnames))) for k in ("gg", "gdn", "gdf", "gnorm2")}
    out["dn2"], out["df2"] = np.zeros(len(gnames)), np.zeros(len(gnames))
    for j, gname in enumerate(gnames):
        for x in GROUPS[gname]:
            g = sum(gk[x]).double()
            dn = (w_next[x] - w[x]).to(g.device).double() if w_next is not None else torch.zeros_like(g)
            df = (w_final[x] - w[x]).to(g.device).double()
            out["dn2"][j] += float((dn * dn).sum())
            out["df2"][j] += float((df * df).sum())
            for k in range(N_GRP):
                a = gk[x][k].double()
                out["gg"][k, j] += float((a * g).sum())
                out["gdn"][k, j] += float((-a * dn).sum())
                out["gdf"][k, j] += float((-a * df).sum())
                out["gnorm2"][k, j] += float((a * a).sum())
    return out


def run(name, dev=torch.device("cuda:0")):
    path = os.path.join(AN, f"ent_driver_{name}.npz")
    P = td.probes()
    G = item_groups(P)
    eps = td.checkpoints(name)
    done = dict(np.load(path)) if os.path.exists(path) else {}
    have = set(done.get("episode", np.array([], int)).tolist())
    rows = {k: list(done[k]) for k in done if k not in ("windows", "param_groups")} if done else {}
    load = lambda e: torch.load(td.ckpt_path(name, e), map_location="cpu", weights_only=False)["weights"]  # noqa: E731
    w_final = load(eps[-1])
    t0 = time.time()
    for i, ep in enumerate(eps):
        if ep in have:
            continue
        w = load(ep)
        w_next = load(eps[i + 1]) if i + 1 < len(eps) else None
        p = {x: v.to(dev).float() for x, v in w.items()}
        dis_cut, dis_int, dis_cut_w, dis_int_w, cnt = memory_cut(p, P, G, dev)
        gk, D_min, full = attribution(w, P, G, dev)
        err = max(float((sum(gk[x]) - full[x]).norm() / full[x].norm().clamp_min(1e-300)) for x in full)
        assert err < 1e-3, f"group parts do not sum to the gradient ({err:.1e})"
        s = summarize(gk, w, w_next, w_final)
        s.update(dis_cut=dis_cut, dis_int=dis_int, dis_cut_w=dis_cut_w, dis_int_w=dis_int_w, cnt=cnt, D_min=D_min,
                 episode=ep)
        for k, v in s.items():
            rows.setdefault(k, []).append(v)
        del p, gk, full
        torch.cuda.empty_cache()
        hh = list(GROUPS).index("lstm_hh")
        sh = s["gdn"][1:, hh] / max(abs(s["gdn"][:, hh].sum()), 1e-30)
        mc = (dis_cut_w[0, 1:].sum(0) - dis_int_w[1:].sum(0)) / cnt[1:].sum(0)
        print(f"{time.strftime('%H:%M:%S')} {name} ep {ep}: W_hh push along the realized change by bin "
              f"{np.round(sh, 3)}; weighted memory-cut(1) cost by bin {np.round(mc, 3)} ({time.time() - t0:.0f} s)", flush=True)
        if len(rows["episode"]) % 10 == 0:
            _save(path, rows)
    _save(path, rows)
    print(f"saved {path} ({time.time() - t0:.0f} s)")


def _save(path, rows):
    order = np.argsort(rows["episode"])
    out = {k: np.array(v)[order] for k, v in rows.items()}
    out["windows"] = np.array(WINDOWS)
    out["param_groups"] = np.array(list(GROUPS))
    np.savez(path + ".tmp.npz", **out)
    os.replace(path + ".tmp.npz", path)


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        run(nm)
