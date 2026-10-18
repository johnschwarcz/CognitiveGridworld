"""Why the dW panels of dynamics_interference_<pct>pct_xcorr.png look the way they do (explanatory, not a paper figure).

collect(net)  one replay of the first N_EP harvested episodes (CPU is fine: run with CUDA_VISIBLE_DEVICES=""), 10% bands,
              dW from both contexts' losses as in gradflow_analysis -> DATA/experimental/gradflow_explain/<net>.npz:
  cross     (N, errors, 3, bands)  each band's cross-band term <dW_B, dW - dW_B>: actual; with the input lag Gram held at
                                   its episode mean (error side varies); with the band's error lag Gram held at its
                                   episode mean (inputs vary)
  E         (N, errors, 3)         ||dW||^2 under the same three versions
  same_lag  (N, errors, B, B)      band x band terms sum_k <dW_{B,k}, dW_{B',k}> over ||dW||^2
  cross_lag (N, errors, B, B)      the same for all pairs of different lags
  du        (N, errors, K, B)      energy of the error carried back k steps (du_k, in u-space) in each forward band
  Ebf       (N, errors, Bb, B + 1) ||dW||^2 split by backward-pass band (rows of dW: PCs of the gate gradients delta)
                                   x forward band of its input (columns: u_{t-k-1} by u's PC band; the last column is
                                   the observation input x)
  probe     (N, errors, 2, K, K)   lag Gram delta_k . delta_l of two FIXED unit directions sent back at every step: the
                                   episode's mean error direction and the error's top principal direction
collect_directions(net)  the error's own direction and two random directions sent back instead (<net>_directions.npz)

`python gradflow_explain.py plot` (after both collects, for trained and echo) writes into figure_bin:
  explain_error.png      the error at entangled steps: its size, its direction, and the swap test (fixed / random
                         directions sent back in its place)
  explain_flip.png       dW's dimension flip = backward-pass part + forward-pass part, and why the rise comes one step late
  explain_crossband.png  why the cross-band share does not flip
  explain_bands.png      the forward bands in the backward pass (mixing across lags), and dW split by backward bands
"""
import os
import sys
import numpy as np
import torch


def _repo_root():
    starts = [globals().get("__file__"), os.getcwd()]
    for start in starts:
        d = os.path.abspath(start) if start and os.path.exists(start) else None
        while d:
            if os.path.exists(os.path.join(d, "main", "CognitiveGridworld.py")):
                return d
            d = None if os.path.dirname(d) == d else os.path.dirname(d)
    raise RuntimeError(f"project root not found above {starts}")


ROOT = _repo_root()
sys.path.insert(0, os.path.join(ROOT, "main", "REPLICATION"))
import gradflow_analysis as ga                                                   # noqa: E402

OUT = os.path.join(ROOT, "main", "DATA", "experimental", "gradflow_explain", "{}.npz")
N_EP = 10000                                                                     # first 10 test batches
PCT = 10
CHUNK = 50


def replay(net, fn, D, batches):
    """gradflow_analysis.replay with a tolerant check, for runs on CPU: there a few error entries (about 0.2%, steps at
    the loss's square-root kink next to the batch minimum, where the gradient is very sensitive) differ from the GPU
    harvest by more than 0.1%; a batch passes if at most 1% do and the median difference is below 1e-5"""
    import gc
    from main.CognitiveGridworld import CognitiveGridworld
    load_env, reservoir = ga.AGENTS[net]
    agent = CognitiveGridworld(**dict(ga.CFG, episodes=1), reservoir=reservoir, load_env=load_env,
                               external_logger=lambda env: {})
    m, (B, T) = agent.model, (ga.CFG["batch_num"], ga.CFG["step_num"])
    dev, out = m.STM.device, {}
    for b in range(batches):
        rows = slice(b * B, (b + 1) * B)
        m.obs_flat, m.active_Z, m.goal_ind, m.joint_goal_belief = (
            torch.as_tensor(D[f"ep_{k}"][rows], device=dev) for k in ("obs_flat", "active_Z", "goal_ind",
                                                                      "joint_goal_belief"))
        with torch.no_grad(), torch.autocast(dev.type, enabled=False):
            x = ga.lstm_input(m)
            st, W_hh = ga.forward_states(m, x)
            r = m.classifier_readout(st["u"])
            loss = ga.StepLoss(m, r)
            err = torch.stack([ga.step_error(m, loss, r, t).norm(dim=-1) for t in range(T)], 1).cpu().numpy()
            assert np.isclose(float(loss.dkl_min), D["dkl_min"][b], rtol=1e-5), f"batch {b}: loss minimum differs"
            rel = np.abs(err - D["err"][rows]) / np.maximum(np.abs(D["err"][rows]), 1e-12)
            assert (rel > 1e-3).mean() < .01 and np.median(rel) < 1e-5, f"batch {b}: errors differ ({(rel > 1e-3).mean():.2%})"
            for k, v in fn(m, x, st, W_hh, r, loss).items():
                out.setdefault(k, []).append(v)
    del agent
    gc.collect()
    return {k: np.concatenate(v) for k, v in out.items()}


def _close(a, ref, scale):
    """at least 99% of the entries within 0.5% of the reference (relative to scale)"""
    return (np.abs(a - ref) > 5e-3 * np.maximum(np.abs(ref), scale)).mean() < .01


def collect(net, n_ep=N_EP, basis_batches=2, path=None):
    path = path or OUT.format(net)
    if os.path.exists(path):
        return dict(np.load(path))
    D = ga.harvest(net)
    sid = ga.cut_segments(D["basis_evr"], 100 // PCT)
    ids = np.unique(sid)
    nb, K, H = len(ids), ga.KMAX + 1, D["basis_V"].shape[0]
    C = np.load(ga.CTX_GRAMS.format(net))["DD"][:n_ep].astype(np.float64)
    DDref = C[:, :, 0] + C[:, :, 1] + C[:, :, 2] + np.swapaxes(C[:, :, 2], -1, -2)
    del C
    Eref = (DDref * D["lag_zz"][:n_ep].astype(np.float64)).sum((-1, -2))
    Bc = ga.band_both(net, PCT)
    cref = (Bc["share"][:n_ep] - Bc["own"][:n_ep]).astype(np.float64)
    batch = [0]
    acc = {}

    def errors(m, r, rows, dev):
        loss_o = ga.other_context_loss(m, r, torch.as_tensor(D["ep_joint_belief"][rows], device=dev))
        loss = ga.StepLoss(m, r)
        return torch.stack([ga.step_error(m, loss, r, t) + ga.step_error(m, loss_o, r, t) for t in ga.SRC], 1)

    # pass 1: the backward-pass basis (PCs of the gate gradients over all lags) and the error's top direction
    def basis_fn(m, x, st, W_hh, r, loss):
        B = x.shape[0]
        rows = slice(batch[0] * B, (batch[0] + 1) * B)
        E_all = errors(m, r, rows, x.device)
        Cd = torch.zeros(4 * H, 4 * H, dtype=torch.float64, device=x.device)
        Ce = torch.zeros(H, H, dtype=torch.float64, device=x.device)
        for j, t in enumerate(ga.SRC):
            e = E_all[:, j]
            Ce += (e.T @ e).double()
            for c0 in range(0, B, CHUNK):
                sl = slice(c0, min(B, c0 + CHUNK))
                d = torch.stack([dk for _, dk in ga.backward({k: v[sl] for k, v in st.items()}, W_hh, e[sl], t, ga.KMAX)], 1)
                d = d.reshape(-1, 4 * H).double()
                Cd += d.T @ d
        acc["Cd"] = acc.get("Cd", 0) + Cd.cpu().numpy()
        acc["Ce"] = acc.get("Ce", 0) + Ce.cpu().numpy()
        batch[0] += 1
        print(f"{net}: basis batch {batch[0]}", flush=True)
        return {}

    replay(net, basis_fn, D, batches=basis_batches)
    evd, Ud = np.linalg.eigh(acc["Cd"])
    evd, Ud = evd[::-1].clip(0), Ud[:, ::-1]
    sidd = ga.cut_segments(evd / evd.sum(), 100 // PCT)
    idsd = np.unique(sidd)
    p2 = np.linalg.eigh(acc["Ce"])[1][:, -1]                                     # the error's top direction

    # pass 2: everything else
    batch[0] = 0

    def fn(m, x, st, W_hh, r, loss):
        B, dev = x.shape[0], x.device
        rows = slice(batch[0] * B, (batch[0] + 1) * B)
        V = torch.as_tensor(D["basis_V"], device=dev)
        OH = torch.as_tensor((sid[:, None] == ids[None]).astype(np.float32), device=dev)
        U = torch.as_tensor(Ud.astype(np.float32), device=dev)
        OHd = torch.as_tensor((sidd[:, None] == idsd[None]).astype(np.float32), device=dev)
        P2 = torch.as_tensor(p2.astype(np.float32), device=dev)
        E_all = errors(m, r, rows, dev)                                          # (B, errors, H)
        p1 = E_all.mean(1)
        p1 = p1 / p1.norm(dim=-1, keepdim=True)
        Zk = torch.cat([x, st["u_prev"]], -1)
        Xd = x.shape[-1]
        nt = len(ga.SRC)
        Xb = torch.zeros(B, nt, nb, K, K, device=dev)
        DD = torch.zeros(B, nt, K, K, device=dev)
        ZZ = torch.zeros(B, nt, K, K, device=dev)
        out = {k: torch.zeros((B, nt) + s, device=dev) for k, s in
               (("same_lag", (nb, nb)), ("cross_lag", (nb, nb)), ("du", (K, nb)), ("Ebf", (len(idsd), nb + 1)),
                ("probe", (2, K, K)))}
        for j, t in enumerate(ga.SRC):
            e = E_all[:, j]
            for c0 in range(0, B, CHUNK):
                sl = slice(c0, min(B, c0 + CHUNK))
                stc = {k: v[sl] for k, v in st.items()}
                probes = ((e[sl] @ V)[:, None, :] * OH.T[None]) @ V.T              # (c, bands, H)
                Dk = torch.stack([d for _, d in ga.backward(stc, W_hh, probes, t, ga.KMAX)], 1)   # (c, K, nb, 4H)
                M = torch.einsum("ckbg,clag->ckbla", Dk, Dk)                      # <delta_{b,k}, delta_{a,l}>
                zk = Zk[sl][:, t - torch.arange(K, device=dev)]                  # (c, K, X + H)
                zz = zk @ zk.transpose(1, 2)
                DD[sl, j], ZZ[sl, j] = M.sum((2, 4)), zz
                Xb[sl, j] = (M.sum(4) - torch.diagonal(M, dim1=2, dim2=4).permute(0, 1, 3, 2)).permute(0, 2, 1, 3)
                Et = (M.sum((2, 4)) * zz).sum((-1, -2))
                same = torch.einsum("ckbka,ck->cba", M, torch.diagonal(zz, dim1=1, dim2=2))
                allp = torch.einsum("ckbla,ckl->cba", M, zz)
                out["same_lag"][sl, j] = same / Et[:, None, None]
                out["cross_lag"][sl, j] = (allp - same) / Et[:, None, None]
                Dw = Dk.sum(2)                                                   # the whole error's deltas
                du = torch.cat([e[sl][:, None], Dw[:, :-1] @ W_hh], 1)           # du_0 = e_t, du_k = delta_{k-1} W_hh
                out["du"][sl, j] = ((du @ V) ** 2) @ OH
                Gd = torch.einsum("ckp,clp,pb->cbkl", Dw @ U, Dw @ U, OHd)       # backward-band lag Grams
                zu = zk[..., Xd:] @ V
                Gz = torch.cat([torch.einsum("ckp,clp,pf->cfkl", zu, zu, OH),
                                (zk[..., :Xd] @ zk[..., :Xd].transpose(1, 2))[:, None]], 1)
                out["Ebf"][sl, j] = torch.einsum("cbkl,cfkl->cbf", Gd, Gz)
                for q, p in enumerate((p1[sl], P2.expand(len(range(*sl.indices(B))), -1))):
                    dp = torch.stack([d for _, d in ga.backward(stc, W_hh, p, t, ga.KMAX)], 1)
                    out["probe"][sl, j, q] = dp @ dp.transpose(1, 2)
        mZZ, mDD, mXb = ZZ.mean(1, keepdim=True), DD.mean(1, keepdim=True), Xb.mean(1, keepdim=True)
        cross = torch.stack([(Xb * ZZ[:, :, None]).sum((-1, -2)), (Xb * mZZ[:, :, None]).sum((-1, -2)),
                             (mXb * ZZ[:, :, None]).sum((-1, -2))], 2)
        E3 = torch.stack([(DD * ZZ).sum((-1, -2)), (DD * mZZ).sum((-1, -2)), (mDD * ZZ).sum((-1, -2))], 2)
        res = {"cross": cross, "E": E3, **out}
        res = {k: v.cpu().numpy() for k, v in res.items()}
        sc = 1e-3 * Eref[rows].mean()
        ok = (_close(res["E"][:, :, 0], Eref[rows], sc) and _close(res["cross"][:, :, 0], cref[rows], sc) and
              np.allclose(res["Ebf"].sum((-1, -2)), res["E"][:, :, 0], rtol=1e-3, atol=sc))
        assert ok, f"batch {batch[0]}: replay does not match the cached ||dW||^2 / cross-band terms"
        batch[0] += 1
        print(f"{net}: batch {batch[0]} of {n_ep // ga.CFG['batch_num']}", flush=True)
        return res

    res = replay(net, fn, D, batches=n_ep // ga.CFG["batch_num"])
    res.update({"bands": ids, "bands_bwd": idsd, "evr_bwd": evd / evd.sum(), "n_ep": np.array(n_ep)})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **res)
    print(f"saved {path}")
    return res


def collect_directions(net, n_ep=N_EP, path=None):
    """the error's direction, from the same replay -> DATA/experimental/gradflow_explain/<net>_directions.npz:
      err    (N, errors, 2, 20)    each context's error (goal, other) in an orthonormal basis of the readout's row space,
                                   where the error lies entirely (so its norm is the error's norm)
      act    (N, 30, 2, 20)        the activity u - mu at every step in its top 20 PCs and in the readout's row space
      probe  (N, errors, 2, K, K)  lag Grams delta_k . delta_l of two RANDOM unit directions sent back in place of the
                                   error (drawn per episode, fixed over its steps): one anywhere in u-space, one inside
                                   the readout's row space"""
    path = path or OUT.format(f"{net}_directions")
    if os.path.exists(path):
        return dict(np.load(path))
    D = ga.harvest(net)
    eref = np.load(ga.BANDP_BOTH.format(PCT, net))["err_E"][:n_ep].sum(-1).astype(np.float64)
    batch = [0]

    def fn(m, x, st, W_hh, r, loss):
        B, dev = x.shape[0], x.device
        rows = slice(batch[0] * B, (batch[0] + 1) * B)
        Q = torch.linalg.qr(m.classifier_readout.weight.detach().float().T)[0]   # (H, 20)
        V = torch.as_tensor(D["basis_V"][:, :20], device=dev)
        mu = torch.as_tensor(D["basis_mu"], device=dev)
        loss_o = ga.other_context_loss(m, r, torch.as_tensor(D["ep_joint_belief"][rows], device=dev))
        e = torch.stack([torch.stack([ga.step_error(m, l, r, t) for t in ga.SRC], 1) for l in (loss, loss_o)], 2)
        g = torch.Generator().manual_seed(int(batch[0]))
        P = [torch.randn(B, Q.shape[0], generator=g).to(dev), torch.randn(B, Q.shape[1], generator=g).to(dev) @ Q.T]
        P = torch.stack([p / p.norm(dim=-1, keepdim=True) for p in P], 1)                # (B, 2, H)
        probe = []
        for t in ga.SRC:
            d = torch.stack([d for _, d in ga.backward(st, W_hh, P, t, ga.KMAX)], 2)      # (B, 2, K, 4H)
            probe.append(d @ d.transpose(-1, -2))
        u = st["u"] - mu
        res = {"err": (e @ Q).cpu().numpy(), "act": torch.stack([u @ V, u @ Q], 2).cpu().numpy(),
               "probe": torch.stack(probe, 1).cpu().numpy()}
        assert _close((res["err"].sum(2) ** 2).sum(-1), eref[rows], 1e-3 * eref[rows].mean()), \
            f"batch {batch[0]}: errors do not match the cached error energy"
        batch[0] += 1
        print(f"{net}: direction batch {batch[0]} of {n_ep // ga.CFG['batch_num']}", flush=True)
        return res

    res = replay(net, fn, D, batches=n_ep // ga.CFG["batch_num"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **res)
    print(f"saved {path}")
    return res


def collect_bandgrams(net, n_ep=5000, path=None):
    """the band self-terms of the backward lag Gram, for the first n_ep episodes ->
    DATA/experimental/gradflow_explain/<net>_bandgrams.npz:
      S   (N, errors, bands, K, K)  delta_{B,k} . delta_{B,l} of each 10% forward band B's piece of the error (both
                                    contexts) sent back k and l steps; summed over bands it is the self-term part of the
                                    lag Gram DD, and DD minus that sum is the cross-band part"""
    path = path or OUT.format(f"{net}_bandgrams")
    if os.path.exists(path):
        return dict(np.load(path))
    D = ga.harvest(net)
    sid = ga.cut_segments(D["basis_evr"], 100 // PCT)
    ids = np.unique(sid)
    nb, K = len(ids), ga.KMAX + 1
    C = np.load(ga.CTX_GRAMS.format(net))["DD"][:n_ep].astype(np.float64)
    DDref = C[:, :, 0] + C[:, :, 1] + C[:, :, 2] + np.swapaxes(C[:, :, 2], -1, -2)
    del C
    batch = [0]

    def fn(m, x, st, W_hh, r, loss):
        B, dev = x.shape[0], x.device
        rows = slice(batch[0] * B, (batch[0] + 1) * B)
        V = torch.as_tensor(D["basis_V"], device=dev)
        OH = torch.as_tensor((sid[:, None] == ids[None]).astype(np.float32), device=dev)
        loss_o = ga.other_context_loss(m, r, torch.as_tensor(D["ep_joint_belief"][rows], device=dev))
        E_all = torch.stack([ga.step_error(m, loss, r, t) + ga.step_error(m, loss_o, r, t) for t in ga.SRC], 1)
        S = torch.zeros(B, len(ga.SRC), nb, K, K)
        DD = torch.zeros(B, len(ga.SRC), K, K)
        for j, t in enumerate(ga.SRC):
            for c0 in range(0, B, CHUNK):
                sl = slice(c0, min(B, c0 + CHUNK))
                probes = ((E_all[sl, j] @ V)[:, None, :] * OH.T[None]) @ V.T                 # (c, bands, H)
                Dk = torch.stack([d for _, d in ga.backward({k: v[sl] for k, v in st.items()}, W_hh, probes, t,
                                                              ga.KMAX)], 1)                    # (c, K, bands, 4H)
                S[sl, j] = torch.einsum("ckbg,clbg->cbkl", Dk, Dk)
                Dw = Dk.sum(2)
                DD[sl, j] = Dw @ Dw.transpose(1, 2)
        res = {"S": S.numpy()}
        assert _close(DD.numpy().astype(np.float64), DDref[rows], 1e-3 * np.abs(DDref[rows]).mean()), \
            f"batch {batch[0]}: the band pieces do not add up to the cached lag Gram"
        batch[0] += 1
        print(f"{net}: band-Gram batch {batch[0]} of {n_ep // ga.CFG['batch_num']}", flush=True)
        return res

    res = replay(net, fn, D, batches=n_ep // ga.CFG["batch_num"])
    res.update(bands=ids, n_ep=np.array(n_ep))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **res)
    print(f"saved {path}")
    return res


def collect_turns(net, n_ep=20000, path=None):
    """the activity's turn at every step, cos(u_s, u_{s-1}) (s = 0 uses the initial state), for the first n_ep
    episodes (forward pass only) -> DATA/experimental/gradflow_explain/<net>_turns.npz: cos_u (N, 30). The cached input
    Grams give it only up to step 28 (the update at s + 1 holds u_s . u_{s-1}); this also covers step 29"""
    path = path or OUT.format(f"{net}_turns")
    if os.path.exists(path):
        return dict(np.load(path))
    D = ga.harvest(net)
    batch = [0]

    def fn(m, x, st, W_hh, r, loss):
        u, up = st["u"].double(), st["u_prev"].double()
        c = (u * up).sum(-1) / (u.norm(dim=-1) * up.norm(dim=-1))
        batch[0] += 1
        print(f"{net}: turn batch {batch[0]} of {n_ep // ga.CFG['batch_num']}", flush=True)
        return {"cos_u": c.cpu().numpy()}

    res = replay(net, fn, D, batches=n_ep // ga.CFG["batch_num"])
    ZU = np.load(ga.CACHE.format(net), mmap_mode="r")["lag_zz_hh"][:n_ep].astype(np.float64)
    ref = ZU[:, 1:, 0, 1] / np.sqrt(ZU[:, 1:, 0, 0] * ZU[:, 1:, 1, 1])            # cos(u_s, u_{s-1}), s = 16..28
    assert np.abs(res["cos_u"][:, 16:29] - ref).max() < 1e-4, "the replayed turns do not match the cached Gram"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **res)
    print(f"saved {path}")
    return res


GATES = ("i", "f", "g", "o", "tc", "cp")
ABLATE = [(0, n) for n in GATES + ("all",)] + [(1, n) for n in GATES + ("all",)] + [(2, "all")]


def backward_split(st, W_hh, du_t, t, kmax, over=None):
    """ga.backward with the gate credit split by the path it arrived on: yields (k, a_k, b_k, du_k, dc_k), delta_k =
    a_k + b_k, a_k from the cell credit dc_k (carried back by the forget gates), b_k from the hidden-state credit du_k
    (carried back through W_hh); over = {k: {name: (B, H)}} replaces the backward factors at step t - k"""
    du, dc = du_t, torch.zeros_like(du_t)
    for k in range(min(kmax, t) + 1):
        fac = {n: st[n][:, t - k] for n in GATES}
        if over and k in over:
            fac.update(over[k])
        i, f, g, o, cp, tc = (fac[n].reshape((-1,) + (1,) * (du.dim() - 2) + (du.shape[-1],))
                              for n in ("i", "f", "g", "o", "cp", "tc"))
        h = du * o * (1 - tc * tc)
        gi, gf, gg = g * i * (1 - i), cp * f * (1 - f), i * (1 - g * g)
        a = torch.cat([dc * gi, dc * gf, dc * gg, torch.zeros_like(du)], -1)
        b = torch.cat([h * gi, h * gf, h * gg, du * tc * o * (1 - o)], -1)
        yield k, a, b, du, dc
        dc, du = (dc + h) * f, (a + b) @ W_hh


def collect_backward(net, n_ep=3000, path=None, ablate=True, sub=None):
    """the backward pass taken apart, for the first n_ep episodes -> DATA/experimental/gradflow_explain/<net>_backward.npz.
    Four directions are sent back from every error step: 0 the real error (both contexts), 1 the error's main axis (one
    fixed direction, the top principal direction of the first 2000 episodes' errors, as in collect), 2 a random unit
    direction, 3 a random unit direction in the readout's row space (2 and 3 drawn per episode as in collect_directions)
      path   (N, errors, 4, 2K, 2K)  Gram of the credit pieces [a_0..a_15, b_0..b_15]: delta_k = a_k + b_k, a_k arrived
                                     through the cell (forget gates), b_k through the hidden state (W_hh); a_0 = 0
      block  (N, errors, 2, 4, K, K) lag Grams per gate block (i, f, g, o) for directions 0 and 1 (they sum to the Gram)
      proj   (N, errors, 2, K, 2, 20) the hidden and cell credit du_k, dc_k on step t - k in the readout basis
                                     (du_0 = the direction sent back), for directions 0 and 1
      wgate  (N, errors, 2, 6)       the gates (i, f, g, o, tanh c, previous c) at the error step, averaged over units
                                     with weights = the squared entries of directions 0 and 1
      abl    (N, errors, len(ABLATE), K, K)  lag Grams of direction 1 with the backward factors at step t - k replaced
                                     by each unit's episode mean (over steps 1-29), for each (k, factor) in ABLATE
      Q      (H, 20)                 the readout basis; v2 (20,) direction 1 in it"""
    path = path or OUT.format(f"{net}_backward")
    if os.path.exists(path):
        return dict(np.load(path))
    D = ga.harvest(net)
    E2 = np.load(OUT.format(f"{net}_directions"))["err"][:2000].astype(np.float64).sum(2)
    v2 = np.linalg.eigh(np.einsum("nti,ntj->ij", E2, E2))[1][:, -1]
    C = np.load(ga.CTX_GRAMS.format(net))["DD"][:n_ep].astype(np.float64)
    DDref = C[:, :, 0] + C[:, :, 1] + C[:, :, 2] + np.swapaxes(C[:, :, 2], -1, -2)
    del C
    Pref = np.load(OUT.format(net))["probe"][:n_ep, :, 1].astype(np.float64)
    K, abl_list = ga.KMAX + 1, ABLATE if ablate else []
    batch, keep = [0], {}

    def fn(m, x, st, W_hh, r, loss):
        B, dev = x.shape[0], x.device
        rows = slice(batch[0] * B, (batch[0] + 1) * B)
        Q = torch.linalg.qr(m.classifier_readout.weight.detach().float().T)[0]
        p2 = Q @ torch.as_tensor(v2, dtype=torch.float32, device=dev)
        p2 = p2 / p2.norm()
        loss_o = ga.other_context_loss(m, r, torch.as_tensor(D["ep_joint_belief"][rows], device=dev))
        E_all = torch.stack([ga.step_error(m, loss, r, t) + ga.step_error(m, loss_o, r, t) for t in ga.SRC], 1)
        g = torch.Generator().manual_seed(int(batch[0]))
        Pr = [torch.randn(B, Q.shape[0], generator=g).to(dev), torch.randn(B, Q.shape[1], generator=g).to(dev) @ Q.T]
        Pr = [p / p.norm(dim=-1, keepdim=True) for p in Pr]
        n = sub or B
        st = {k: v[:n] for k, v in st.items()}
        means = {nm: st[nm][:, 1:].mean(1) for nm in GATES}
        out = {"path": torch.zeros(n, len(ga.SRC), 4, 2 * K, 2 * K), "block": torch.zeros(n, len(ga.SRC), 2, 4, K, K),
               "proj": torch.zeros(n, len(ga.SRC), 2, K, 2, Q.shape[1]), "wgate": torch.zeros(n, len(ga.SRC), 2, 6),
               "abl": torch.zeros(n, len(ga.SRC), len(abl_list), K, K)}
        for j, t in enumerate(ga.SRC):
            P = torch.stack([E_all[:n, j], p2.expand(n, -1), Pr[0][:n], Pr[1][:n]], 1)          # (n, 4, H)
            pcs = list(backward_split(st, W_hh, P, t, ga.KMAX))
            A, Bh = (torch.stack([pc[i] for pc in pcs], 2) for i in (1, 2))                    # (n, 4, K, 4H)
            AB = torch.cat([A, Bh], 2)
            out["path"][:, j] = AB @ AB.transpose(-1, -2)
            Dg = (A + Bh)[:, :2].reshape(n, 2, K, 4, -1)
            out["block"][:, j] = torch.einsum("npkgh,nplgh->npgkl", Dg, Dg)
            du, dc = (torch.stack([pc[i][:, :2] for pc in pcs], 2) for i in (3, 4))            # (n, 2, K, H)
            out["proj"][:, j] = torch.stack([du @ Q, dc @ Q], 3)
            w = P[:, :2] ** 2 / (P[:, :2] ** 2).sum(-1, keepdim=True)
            out["wgate"][:, j] = torch.stack([(w * st[nm][:, t, None]).sum(-1) for nm in GATES], -1)
            for q, (k, nm) in enumerate(abl_list):
                over = {k: {g_: means[g_] for g_ in (GATES if nm == "all" else (nm,))}}
                d = torch.stack([a + b for _, a, b, _, _ in backward_split(st, W_hh, p2.expand(n, -1), t, ga.KMAX,
                                                                           over)], 1)
                out["abl"][:, j, q] = d @ d.transpose(-1, -2)
        res = {k: v.cpu().numpy() for k, v in out.items()}
        G = res["path"][:, :, :2].astype(np.float64)
        DDs = G[..., :K, :K] + G[..., K:, K:] + G[..., :K, K:] + G[..., K:, :K]                 # (n, errors, 2, K, K)
        r0 = slice(rows.start, rows.start + n)
        sc = 1e-3 * np.abs(DDref[r0]).mean()
        assert _close(DDs[:, :, 0], DDref[r0], sc), f"batch {batch[0]}: real-error Gram differs from the cache"
        assert _close(DDs[:, :, 1], Pref[r0], 1e-3 * np.abs(Pref[r0]).mean()), f"batch {batch[0]}: main axis differs"
        assert np.allclose(res["block"].sum(3), DDs, rtol=1e-3, atol=1e-4 * np.abs(DDs).max()), "gate blocks do not sum"
        keep["Q"], keep["v2"] = Q.cpu().numpy(), v2
        batch[0] += 1
        print(f"{net}: backward batch {batch[0]} of {n_ep // ga.CFG['batch_num']}", flush=True)
        return res

    res = replay(net, fn, D, batches=n_ep // ga.CFG["batch_num"])
    res.update(keep, ablate=np.array([f"{k}:{nm}" for k, nm in abl_list]), n_ep=np.array(n_ep))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **res)
    print(f"saved {path}")
    return res


# ═══════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt                                                  # noqa: E402
from matplotlib.patches import FancyBboxPatch                                    # noqa: E402

N_FIG = 20000                                                                    # episodes for the cached panels
S = slice(ga.SRC[0], ga.SRC[-1] + 1)
COL = {"Trained": ga.sp.AGENT_COLORS["Trained"], "Echo": ga.sp.AGENT_COLORS["Echo"], "total": "0.1",
       "error": "#c0392b", "input": "#7f8c8d", "activity": "#2c3e50", "observation": "#a6acaf",
       "swap_q": "#e6a09a", "swap_u": "#b3b6b7"}
FS = ga.sp.FS


def _cen(a):
    return a - a.mean(1, keepdims=True)


def _ref(net, n):
    return ga._zs(np.load(ga.CACHE.format(net))["rev"][:n, S].astype(np.float64))


def _r(y, b):
    """the panel unit: cross-correlation of a series (z-scored within the episode) with entanglement"""
    return ga._lagged(ga._zs(y), b)


def _split(total, pieces, b):
    """additive split of total's cross-correlation: each piece centred within the episode and divided by the TOTAL's
    within-episode SD; the last returned curve is the rest (interaction), so the parts add up to total's curve"""
    sd = total.std(1, keepdims=True) + 1e-12
    parts = [ga._lagged(_cen(p) / sd, b) for p in pieces]
    return parts + [ga._lagged(_cen(total - sum(pieces)) / sd, b)]


def _pr(A, B):
    M = A @ B
    return np.einsum("...ii->...", M) ** 2 / np.einsum("...ij,...ji->...", M, M)


def _grams(net, n):
    f = np.load(ga.CACHE.format(net))
    C = np.load(ga.CTX_GRAMS.format(net))["DD"][:n].astype(np.float64)
    DD = C[:, :, 0] + C[:, :, 1] + C[:, :, 2] + np.swapaxes(C[:, :, 2], -1, -2)
    ZU = f["lag_zz_hh"][:n].astype(np.float64)
    ZX = f["lag_zz"][:n].astype(np.float64) - ZU
    return DD, ZU, ZX


def _lines(ax, items, title, ylabel=None, legend="upper right"):
    """lag curves in the style of the dynamics figure: dotted before the event, solid after; items: (curve, color,
    label[, linestyle])"""
    past, fut = ga.TAU <= 0, ga.TAU >= 0
    for c, col, lab, *ls in items:
        ls = ls[0] if ls else "-"
        ax.plot(ga.TAU[past], c[past], c=col, ls=":", lw=2, alpha=.55)
        ax.plot(ga.TAU[fut], c[fut], c=col, ls=ls, lw=2.4, marker="o", ms=4.5, mec="k", mew=.8, label=lab)
    ax.axhline(0, color="k", alpha=.3, lw=1.1)
    ax.axvline(0, color="k", ls="--", alpha=.4, lw=1.3)
    ax.set_xticks(range(-8, 9, 4))
    ax.set_xlabel(r"Lag ($\tau$, steps after the entangled step)")
    ax.set_title(title, fontsize=FS["title"] * .85)
    ax.grid(True, ls="--", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend(frameon=False, fontsize=FS["legend"] * .9, loc=legend)


def _save(fig, name):
    fig.savefig(ga.fig_path(name), dpi=150, bbox_inches="tight", pad_inches=.3)
    plt.close(fig)
    print(f"saved {ga.fig_path(name)}")


def _quad(F):
    """features and their pairwise products (any direction's energy share is a combination of these)"""
    i, j = np.triu_indices(F.shape[-1])
    return np.concatenate([F, F[..., i] * F[..., j]], -1)


def _decode(F, b, lam=1e-3):
    """the panel unit for a readout: per lag tau, a ridge readout of entanglement b[t] from the features F[t + tau]
    (both centred within the episode), fit on one half of the episodes and tested on the other, both ways; returns the
    mean within-episode r of readout vs entanglement on the held-out episodes"""
    half = np.arange(len(b)) < len(b) // 2
    out = []
    for L in ga.TAU:
        y, Fx = (b[:, :b.shape[1] - L], F[:, L:]) if L >= 0 else (b[:, -L:], F[:, :L])
        y, Fx = _cen(y), _cen(Fx)
        rs = []
        for tr in (half, ~half):
            A = Fx[tr].reshape(-1, Fx.shape[-1])
            G = A.T @ A
            w = np.linalg.solve(G + lam * np.trace(G) / len(G) * np.eye(len(G)), A.T @ y[tr].reshape(-1))
            p, yt = _cen(Fx[~tr] @ w), y[~tr]
            rs.append(((p * yt).sum(1) / np.sqrt((p ** 2).sum(1) * (yt ** 2).sum(1) + 1e-30)).mean())
        out.append(np.mean(rs))
    return np.array(out)


SWAPS = (("the real error", "total", "-"), ("the error's main axis, fixed", "error", "--"),
         ("random, in the readout's space", "swap_q", "-."), ("random", "swap_u", "-"))


def fig_error(R, Rd):
    """the error at entangled steps is ordinary in size and direction; what changes is how the network carries the
    error's usual direction back (its lag pieces cancel less), and that alone makes dW's dimension dip"""
    fig, axs = plt.subplots(1, 4, figsize=(20.5, 4.8), gridspec_kw=dict(wspace=.3))
    for net, lab in (("trained", "Trained"), ("echo", "Echo")):
        b = _ref(net, N_FIG)
        eE = np.load(ga.BANDP_BOTH.format(PCT, net))["err_E"][:N_FIG].astype(np.float64)
        _lines(axs[0], [(_r(eE.sum(-1), b), COL[lab], lab)], "(a) The error is not bigger\n" + r"$\|e_t\|^2$",
               "r (Cross-correlation)", legend=False)
    axs[0].legend(frameon=False, fontsize=FS["legend"] * .9, loc="upper right")
    X = Rd["trained"]
    n = X["err"].shape[0]
    b = _ref("trained", n)
    unit = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
    e = unit(X["err"].astype(np.float64).sum(2))                                 # (n, errors, 20), both contexts
    a = unit(X["act"][:, S, 0].astype(np.float64))                               # top 20 PCs
    turn = 1 - (e[:, 1:] * e[:, :-1]).sum(-1)                                    # between steps s - 1 and s
    _lines(axs[1], [(_decode(_quad(a), b), COL["activity"], "the activity's direction"),
                    (_decode(_quad(e), b), COL["error"], "the error's direction"),
                    (_r(turn, b[:, 1:]), COL["error"], "the error's turn\n" + r"$1 - \cos(e_{t-1}, e_t)$", "--")],
           "(b) Nor does it point anywhere special (trained)\nentanglement read out from each direction", legend=False)
    axs[1].legend(frameon=False, fontsize=FS["legend"] * .8, loc="upper right")
    # the error's main axis is one direction for all episodes and steps (the episode-mean direction is nearly the
    # error itself: within an episode the error keeps its direction, median |cos| 0.95)
    G = [_grams("trained", n)[0], R["trained"]["probe"][:n, :, 1], X["probe"][:, :, 1], X["probe"][:, :, 0]]
    G = [g.astype(np.float64) for g in G]
    ZZ = sum(_grams("trained", n)[1:])
    mZZ = np.broadcast_to(ZZ.mean(1, keepdims=True), ZZ.shape)
    coh = lambda g: g.sum((-1, -2)) / np.einsum("...kk->...", g)
    ce = coh(_grams("echo", n)[0])
    _lines(axs[2], [(_r(coh(g), b), COL[c], lab, ls) for g, (lab, c, ls) in zip(G, SWAPS)]
           + [(_r(ce, _ref("echo", n)), COL["Echo"], "Echo, the real error")],
           "(c) But its lag pieces cancel less\n" + r"$\|\sum_k \delta_k\|^2 / \sum_k \|\delta_k\|^2$", legend=False)
    h, l = axs[2].get_legend_handles_labels()
    axs[2].legend(h[-1:], l[-1:], frameon=False, fontsize=FS["legend"] * .8, loc="lower right")
    lev = [np.median(coh(g)) for g in G]
    axs[2].text(.97, .5, f"typical value (median):\nreal error {lev[0]:.2f}\nrandom {lev[3]:.2f}, Echo "
                f"{np.median(ce):.2f}\n(below 1: the pieces cancel)", transform=axs[2].transAxes, ha="right",
                va="center", fontsize=FS["annot"] * .75, color="0.3")
    _lines(axs[3], [(_split(_pr(g, ZZ), [_pr(g, mZZ)], b)[0], COL[c], lab, ls) for g, (lab, c, ls) in zip(G, SWAPS)],
           r"(d) Backward-pass part of $\Delta W$'s eff. dim." + "\nwhen a fixed direction is sent back instead",
           legend=False)
    axs[3].legend(frameon=False, fontsize=FS["legend"] * .8, loc="upper right", title="sent back (c, d):",
                  title_fontsize=FS["legend"] * .8)
    lo = min(ax.get_ylim()[0] for ax in axs)
    hi = max(ax.get_ylim()[1] for ax in axs)
    for ax in axs:
        ax.set_ylim(lo, hi)
    fig.suptitle("At entangled steps the error has its usual size and direction; it only turns a little more from "
                 "the step before (a, b). What changes is how the network\ncarries the error's usual direction back "
                 "in time: its pieces from different lags cancel less (c). Held fixed, the error's main axis gives "
                 r"$\Delta W$" + "'s whole dip;\na random direction in the readout's space gives part of it, a "
                 "random direction none (d)", y=1.17, fontsize=FS["title"] * .95)
    _save(fig, "explain_error.png")


def fig_flip(R):
    """dW's dimension flip = a backward-pass dip at tau 0 + a forward-pass rise at tau +1/+2, the rise carried mostly
    by the activity's turn at the entangled step, which the update first sees one step later"""
    fig = plt.figure(figsize=(19, 4.8))
    gs = fig.add_gridspec(1, 4, wspace=.3, width_ratios=[1, 1, 1, 1.05])
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    b = _ref("trained", N_FIG)
    DD, ZU, ZX = _grams("trained", N_FIG)
    ZZ = ZU + ZX
    m = lambda A: np.broadcast_to(A.mean(1, keepdims=True), A.shape)
    PR = _pr(DD, ZZ)
    err, inp, rest = _split(PR, [_pr(DD, m(ZZ)), _pr(m(DD), ZZ)], b)
    act, obs, _ = _split(PR, [_pr(m(DD), m(ZX) + ZU), _pr(m(DD), ZX + m(ZU))], b)
    # the same activity part with its latest turn frozen: cos(u_{s-1}, u_{s-2}) at its episode mean, norms kept
    nrm = np.sqrt(ZU[..., 0, 0] * ZU[..., 1, 1])
    ZUf = ZU.copy()
    ZUf[..., 0, 1] = ZUf[..., 1, 0] = (ZU[..., 0, 1] / nrm).mean(1, keepdims=True) * nrm
    actf = _split(PR, [_pr(m(DD), m(ZX) + ZUf)], b)[0]
    _lines(axs[0], [(_r(PR, b), COL["total"], r"$\Delta W$ eff. dim."),
                    (err, COL["error"], "backward-pass part", "--"), (inp, COL["input"], "forward-pass part", "--"),
                    (rest, COL["swap_u"], "rest")],
           r"(a) $\Delta W$ effective dim. = backward + forward pass", "r (Cross-correlation)", legend=False)
    axs[0].legend(frameon=False, fontsize=FS["legend"] * .8, loc="lower left")
    _lines(axs[1], [(inp, COL["input"], "forward-pass part"), (act, COL["activity"], "  from the activity", "--"),
                    (actf, COL["swap_u"], "    ... with its turn in (c) frozen", "--"),
                    (obs, COL["observation"], "  from the observation", "-.")],
           "(b) The forward-pass part, by input", legend="lower left")
    # the activity's turn between consecutive steps: 1 - cos(u_s, u_{s-1}) at step s = t' - 1, from the input Gram
    turn = 1 - ZU[..., 0, 1] / np.sqrt(ZU[..., 0, 0] * ZU[..., 1, 1])           # at steps 15..28
    ent_s = np.load(ga.CACHE.format("trained"))["rev"][:N_FIG, ga.SRC[0]:ga.SRC[-1]].astype(np.float64)  # 16..28
    _lines(axs[2], [(ga._lagged(ga._zs(turn[:, 1:]), ga._zs(ent_s)), COL["Trained"],
                     r"$1 - \cos(u_{t-1}, u_t)$")],
           "(c) The activity turns at the entangled step", legend="lower left")
    lo = min(ax.get_ylim()[0] for ax in axs[:3])
    hi = max(ax.get_ylim()[1] for ax in axs[:3])
    for ax in axs[:3]:
        ax.set_ylim(lo, hi)
    # timeline: which activity each update sees
    ax = axs[3]
    ax.set(xlim=(0, 10), ylim=(0, 7)), ax.axis("off")
    xs = (1.9, 5.0, 8.1)
    for i, (lab, x) in enumerate(zip(("t-1", "t\n(entangled)", "t+1"), xs)):
        ax.add_patch(FancyBboxPatch((x - 1.05, 4.6), 2.1, 1.3, boxstyle="round,pad=.1",
                                    fc="#fdebd0" if i == 1 else "white", ec="0.3"))
        ax.text(x, 5.25, f"activity\n$u$ at {lab}", ha="center", va="center", fontsize=FS["annot"] * .85)
        ax.add_patch(FancyBboxPatch((x - 1.2, 1.4), 2.4, 1.1, boxstyle="round,pad=.1", fc="white", ec=COL["error"]))
        ax.text(x, 1.95, f"update\nat {lab.split()[0]}", ha="center", va="center", fontsize=FS["annot"] * .8,
                color=COL["error"])
    for i in (0, 1):                                                                 # u at s feeds the update at s+1
        ax.annotate("", xy=(xs[i + 1] - .3, 2.6), xytext=(xs[i] + .3, 4.5),
                    arrowprops=dict(arrowstyle="-|>", lw=2.4 if i == 1 else 1.4,
                                    color=COL["activity"] if i == 1 else "0.6"))
    ax.text(5.0, .35, "each update only sees the activity of earlier steps,\nso the turn at t first changes the update "
            "at t+1", ha="center", fontsize=FS["annot"] * .8, color="0.25")
    ax.set_title("(d) Why the rise comes one step late", fontsize=FS["title"] * .85)
    fig.suptitle(r"$\Delta W$'s dimension flip has two sources: the backward pass dips at the entangled step, and the "
                 "forward pass rises one step later (a). The inputs change direction,\nmostly the activity (b), which "
                 "turns at the entangled step (c, freezing that turn halves its part); each update only sees earlier "
                 "activity (d)", y=1.1, fontsize=FS["title"] * .95)
    _save(fig, "explain_flip.png")


def fig_crossband(R):
    """the cross-band share is set by the backward pass; the forward pass's change at tau +1 moves the band pieces and
    their total together, so it barely reaches the share"""
    fig = plt.figure(figsize=(19, 4.8))
    gs = fig.add_gridspec(1, 4, wspace=.3, width_ratios=[1.15, 1, 1, .06])
    a0 = fig.add_subplot(gs[0, 0])
    X = R["trained"]
    n = X["E"].shape[0]
    b = _ref("trained", n)
    tot, E = X["cross"].sum(-1), X["E"]
    P = [tot[:, :, i] / E[:, :, i] for i in range(3)]
    err, inp, rest = _split(P[0], [P[1], P[2]], b)
    _lines(a0, [(_r(P[0], b), COL["total"], "cross-band share (the panel)"),
                (err, COL["error"], "backward-pass part", "--"), (inp, COL["input"], "forward-pass part", "--"),
                (rest, COL["swap_u"], "rest (both together)")],
           "(a) Cross-band share = backward + forward pass", "r (Cross-correlation)", legend=False)
    a0.legend(frameon=False, fontsize=FS["legend"] * .85, loc="upper right")
    Pb = [X["cross"][:, :, i] / E[:, :, i, None] for i in range(3)]
    nb = Pb[0].shape[-1]
    maps = []
    for i in (1, 2):
        sd = Pb[0].std(1, keepdims=True) + 1e-12
        maps.append(ga._lagged(_cen(Pb[i]) / sd, b).T)                            # (bands, TAU)
    vmax = max(np.abs(m_).max() for m_ in maps)
    for k, (m_, title) in enumerate(zip(maps, ("(b) By band: backward-pass part", "(c) By band: forward-pass part"))):
        ax = fig.add_subplot(gs[0, 1 + k])
        im = ax.imshow(m_, aspect="auto", cmap=ga.HEAT_CMAP, vmin=-vmax, vmax=vmax, interpolation="nearest",
                       extent=(ga.TAU[0] - .5, ga.TAU[-1] + .5, nb, 0))
        ax.axvline(-.5, color="k", ls="--", alpha=.6, lw=1.3)
        ax.set_yticks([0, nb / 2, nb], ["0%", "50%", "100%"])
        ax.set_xticks(range(-8, 9, 4))
        ax.set_xlabel(r"Lag ($\tau$)")
        ax.set_title(title, fontsize=FS["title"] * .85)
        if k == 0:
            ax.set_ylabel(f"Band ({PCT}% of variance)")
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 3]))
    cb.set_label("part of r")
    fig.suptitle("The cross-band share does not flip: it is set by the backward pass (a, b). It weighs every pair of "
                 "lags by the same inputs in its numerator and\nits denominator, so the inputs' change one step later "
                 "barely moves it (a, c), while it does move " + r"$\Delta W$" + "'s dimension",
                 y=1.1, fontsize=FS["title"] * .95)
    _save(fig, "explain_crossband.png")


def fig_bands(R):
    """top: the forward bands in the backward pass: where ||dW||^2 comes from (same / other band x same / other lag) and
    the error's band make-up as it travels back; bottom: dW split by the backward pass's own bands instead"""
    fig = plt.figure(figsize=(19, 10.6))
    gs = fig.add_gridspec(2, 5, wspace=.35, hspace=.75, width_ratios=[1, 1, .07, .45, 1.5])
    for k, (net, lab) in enumerate((("trained", "Trained"), ("echo", "Echo"))):
        X = R[net]
        sl, cl = X["same_lag"].mean((0, 1)), X["cross_lag"].mean((0, 1))
        g = 100 * np.array([[np.trace(sl), np.trace(cl)], [sl.sum() - np.trace(sl), cl.sum() - np.trace(cl)]])
        ax = fig.add_subplot(gs[0, k])
        v = 200
        im = ax.imshow(g, cmap=ga.HEAT_CMAP, vmin=-v, vmax=v)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{g[i, j]:+.0f}%" if abs(g[i, j]) >= .5 else "0%", ha="center", va="center",
                        fontsize=FS["annot"] * 1.3,
                        color="white" if abs(g[i, j]) > 110 else "k")
        ax.set_xticks([0, 1], ["same lag", "different lags"])
        ax.set_yticks([0, 1], ["same band", "different\nbands"] if k == 0 else ["", ""])
        ax.set_title(("(a) " if k == 0 else "(b) ") + f"{lab}: where " + r"$\|\Delta W\|^2$" + " comes from",
                     fontsize=FS["title"] * .8, color=COL[lab])
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 2]))
    cb.set_label(r"% of $\|\Delta W\|^2$ (the four add up to 100%)")
    ax = fig.add_subplot(gs[0, 4])
    du = R["trained"]["du"].mean((0, 1))
    du = 100 * du / du.sum(-1, keepdims=True)                                    # (K, bands)
    nb = du.shape[1]
    shades = plt.get_cmap("Greys")(np.linspace(.45, 1, 4))
    for c, k in zip(shades, (0, 1, 4, 15)):
        ax.plot(np.arange(nb), du[k], c=c, lw=2.4, marker="o", ms=4.5, mec="k", mew=.8,
                label="the error at the step itself" if k == 0 else f"carried {k} step{'s' if k > 1 else ''} back")
    ax.set_xticks([0, nb - 1], ["top band", "bottom band"])
    ax.set_xlabel(f"forward band ({PCT}% of the activity variance each)")
    ax.set_ylabel("% of the error's energy")
    ax.set_title("(c) Trained: the error's band make-up as it travels back in time", fontsize=FS["title"] * .8)
    ax.grid(True, ls="--", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=FS["legend"] * .85, loc="upper left")
    # bottom row: the backward pass's own bands (PCs of the gate gradients delta, 10% of their variance each)
    X = R["trained"]
    Eb = X["Ebf"].astype(np.float64)
    sh = Eb / Eb.sum((-1, -2))[..., None, None]                                  # (N, errors, bwd bands, inputs)
    nbb = sh.shape[2]
    ax = fig.add_subplot(gs[1, 0:3])
    M = 100 * sh.mean((0, 1))                                                    # (bwd bands, forward bands + x)
    x = np.arange(nbb)
    for y, c, lab in ((M.sum(1), COL["total"], r"all of $\Delta W$"), (M[:, :-1].sum(1), COL["activity"],
                      "  with the activity as input"), (M[:, -1], COL["observation"], "  with the observation as input")):
        ax.plot(x, y, c=c, lw=2.4, marker="o", ms=4.5, mec="k", mew=.8, label=lab)
    ax.axhline(100 / nbb, color="k", ls="--", alpha=.5, lw=1.3)
    ax.text(0, 100 / nbb + .6, r"each band holds 10% of $\delta$'s variance", ha="left", va="bottom",
            fontsize=FS["annot"] * .8, color="0.3")
    ax.set_xticks([0, nbb - 1], ["top band", "bottom band"])
    ax.set_xlabel(f"backward band ({PCT}% of the variance of the gate gradients " + r"$\delta$" + " each)")
    ax.set_ylabel(r"% of $\|\Delta W\|^2$")
    ax.set_title(r"(d) Trained: where $\|\Delta W\|^2$ sits among the backward bands (an exact split)",
                 fontsize=FS["title"] * .8)
    ax.grid(True, ls="--", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=FS["legend"] * .85, loc="upper left")
    ax = fig.add_subplot(gs[1, 4])
    b = _ref("trained", sh.shape[0])
    rb = np.array([_r(sh[..., i, :].sum(-1), b) for i in range(nbb)])            # (bwd bands, TAU)
    vmax = .3
    im = ax.imshow(rb, aspect="auto", cmap=ga.HEAT_CMAP, vmin=-vmax, vmax=vmax, interpolation="nearest",
                   extent=(ga.TAU[0] - .5, ga.TAU[-1] + .5, nbb, 0))
    ax.axvline(-.5, color="k", ls="--", alpha=.6, lw=1.3)
    ax.set_yticks([0, nbb / 2, nbb], ["0%", "50%", "100%"])
    ax.set_xticks(range(-8, 9, 4))
    ax.set_xlabel(r"Lag ($\tau$, steps after the entangled step)")
    ax.set_ylabel("backward band")
    ax.set_title("(e) Trained: each backward band's share of " + r"$\|\Delta W\|^2$" + "\n(same scale as the main "
                 "figure's heatmaps)", fontsize=FS["title"] * .8)
    cb = fig.colorbar(im, ax=ax, fraction=.05, pad=.03)
    cb.set_label("r (Cross-correlation)")
    fig.text(.5, .99, "Forward bands in the backward pass: within one lag they stay apart; across lags the recurrence "
             "mixes them, and that is where they cancel", ha="center", va="bottom", fontsize=FS["title"] * .95)
    fig.text(.5, .47, "Split by the backward pass's own bands instead, " + r"$\Delta W$" + " has no cross terms. Its "
             "energy sits in the low-variance backward bands (d), and their shares move less (e)\nthan the "
             "forward-band cross terms do", ha="center", va="bottom", fontsize=FS["title"] * .95)
    _save(fig, "explain_bands.png")


def _band_split(net, Rb):
    """dW's effective dimension split as in fig_flip (backward part, forward part, rest), with the backward part split
    further by forward band: DD = S + C, S the band self-terms (sum over bands B of delta_{B,k} . delta_{B,l}) and C the
    cross-band terms. PR(DD . ZZ) depends on DD only through S + C, so the backward version PR(DD . mean ZZ) is
    linearised around the episode mean of the trace-normalised Gram (PR does not depend on DD's scale): its change =
    <G, dS> + <G, dC> + a nonlinear rest, G the gradient of PR at the mean"""
    S = Rb["S"].astype(np.float64).sum(2)                                        # (N, errors, K, K)
    n = len(S)
    b = _ref(net, n)
    DD, ZU, ZX = _grams(net, n)
    ZZ = ZU + ZX
    m = lambda A: np.broadcast_to(A.mean(1, keepdims=True), A.shape)
    tr = np.einsum("...kk->...", DD)[..., None, None]
    Sn, Cn = S / tr, (DD - S) / tr
    A, Z = m(Sn + Cn), m(ZZ)
    P = A @ Z
    T, Q = np.einsum("...ii->...", P), np.einsum("...ij,...ji->...", P, P)
    G = (2 * T / Q)[..., None, None] * Z - (2 * T ** 2 / Q ** 2)[..., None, None] * (Z @ A @ Z)
    lin = [(G * (X - m(X))).sum((-1, -2)) for X in (Sn, Cn)]
    PRb = _pr(DD, Z)
    rest_b = PRb - _pr(A, Z) - lin[0] - lin[1]
    PR = _pr(DD, ZZ)
    parts = _split(PR, [lin[0], lin[1], rest_b, _pr(m(DD), ZZ)], b)
    return {"total": _r(PR, b), "self": parts[0], "cross": parts[1], "nonlin": parts[2], "forward": parts[3],
            "backward": parts[0] + parts[1] + parts[2], "rest": parts[4], "n": n}


def fig_bandsplit(Rb):
    """the panel of fig_flip (a), and its backward part split into the band self-terms and the band cross-terms"""
    c = _band_split("trained", Rb)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw=dict(wspace=.3))
    _lines(axs[0], [(c["total"], COL["total"], r"$\Delta W$ eff. dim."),
                    (c["backward"], COL["error"], "backward-pass part", "--"),
                    (c["forward"], COL["input"], "forward-pass part", "--"), (c["rest"], COL["swap_u"], "rest")],
           r"(a) $\Delta W$ effective dim. = backward + forward pass", "r (Cross-correlation)", legend=False)
    axs[0].legend(frameon=False, fontsize=FS["legend"] * .8, loc="lower left")
    _lines(axs[1], [(c["backward"], COL["error"], "backward-pass part", "--"),
                    (c["cross"], "#7b241c", "  band cross-terms"),
                    (c["self"], "#f1948a", "  band self-terms"),
                    (c["nonlin"], COL["swap_u"], "  rest (nonlinear)"),
                    (c["forward"], COL["input"], "forward-pass part", "--")],
           "(b) The backward-pass part, split by forward band", legend=False)
    axs[1].legend(frameon=False, fontsize=FS["legend"] * .8, loc="upper left")
    lo = min(ax.get_ylim()[0] for ax in axs)
    hi = max(ax.get_ylim()[1] for ax in axs)
    for ax in axs:
        ax.set_ylim(lo, hi)
    fig.suptitle("The backward-pass part is the band cross-terms at the entangled step and the band self-terms one "
                 "step later;\nthe self-terms do not follow the forward-pass part" + f" (trained, first {c['n']} "
                 f"episodes, {PCT}% bands)", y=1.08, fontsize=FS["title"] * .9)
    _save(fig, "explain_bandsplit.png")
    return c


def fig_effcross():
    """John's test: is dW's effective dimension (effective dim of the activity) - (cross-band share)?  Each measure
    against the exact part of dW's effective dimension it stands for: the cross-band share against the backward part,
    the activity's effective dim (one step earlier: the update at step s takes u_{s-1}, u_{s-2}, ... as input) against
    the forward part"""
    N = N_FIG
    b = _ref("trained", N)
    DD, ZU, ZX = _grams("trained", N)
    ZZ = ZU + ZX
    m = lambda A: np.broadcast_to(A.mean(1, keepdims=True), A.shape)
    dW = _pr(DD, ZZ)
    bk, fw, _ = _split(dW, [_pr(DD, m(ZZ)), _pr(m(DD), ZZ)], b)
    pr = np.load(ga.CACHE.format("trained"), mmap_mode="r")["pr"][:N].astype(np.float64)   # all 30 steps
    e0, e1 = pr[:, S], pr[:, S.start - 1:S.stop - 1]
    B10 = np.load(ga.BANDP_BOTH.format(PCT, "trained"))
    share, own = (B10[k][:N].astype(np.float64) for k in ("share", "own"))
    cross = (share - own).sum(-1) / share.sum(-1)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(wspace=.3))
    _lines(axs[0], [(_r(dW, b), COL["total"], r"$\Delta W$ effective dim."),
                    (_r(e0, b) - _r(cross, b), COL["swap_u"], "eff. dim - cross-band share", "--"),
                    (_r(e1, b) - _r(cross, b), COL["Trained"], "  ... eff. dim one step earlier", "--")],
           "(a) Your measure (difference of the two curves)", "r (Cross-correlation)", legend=False)
    axs[0].legend(frameon=False, fontsize=FS["legend"] * .7, loc="lower left")
    _lines(axs[1], [(bk, COL["error"], r"backward-pass part of $\Delta W$'s eff. dim.", "--"),
                    (-_r(cross, b), "#7b241c", "minus the cross-band share")],
           "(b) The cross-band share is the backward part", legend=False)
    axs[1].legend(frameon=False, fontsize=FS["legend"] * .7, loc="lower right")
    _lines(axs[2], [(fw, COL["input"], r"forward-pass part of $\Delta W$'s eff. dim.", "--"),
                    (_r(e1, b), COL["Trained"], "effective dim one step earlier")],
           "(c) The effective dim, one step later, is the forward part", legend=False)
    axs[2].legend(frameon=False, fontsize=FS["legend"] * .7, loc="lower right")
    lo = min(ax.get_ylim()[0] for ax in axs)
    hi = max(ax.get_ylim()[1] for ax in axs)
    for ax in axs:
        ax.set_ylim(lo, hi)
    fig.suptitle("Each measure follows the part of " + r"$\Delta W$" + "'s effective dimension it stands for: the "
                 "cross-band share the backward part (b), the effective dim\nthe forward part one step later (c), except "
                 "at tau 0 (the observation at the entangled step) and part of tau +2 (trained, 10% bands)", y=1.08,
                 fontsize=FS["title"] * .9)
    _save(fig, "explain_effcross.png")


def _fit_parts(y, groups, b, lam=1e-6):
    """within-episode pooled ridge fit of y on all the groups' features at once; each group's fitted combination
    (N, errors), its additive part of y's r curve, what is left, and the share of y's curve explained"""
    F = np.concatenate([g if g.ndim == 3 else g[..., None] for g in groups], -1)
    A = _cen(F).reshape(-1, F.shape[-1])
    G = A.T @ A
    w = np.linalg.solve(G + lam * np.trace(G) / len(G) * np.eye(len(G)), A.T @ _cen(y).ravel())
    cuts = np.cumsum([0] + [1 if g.ndim == 2 else g.shape[-1] for g in groups])
    comb = [F[..., a:z] @ w[a:z] for a, z in zip(cuts[:-1], cuts[1:])]
    parts = _split(y, comb, b)
    tot = _r(y, b)
    return comb, parts, 100 * (1 - (parts[-1] ** 2).sum() / (tot ** 2).sum()), [w[a:z] for a, z in zip(cuts[:-1], cuts[1:])]


def _fit_variables():
    """the candidate variables for the explain_two / _three / _four fits (trained, first N_FIG episodes, the update at
    each error step s = 16..29), each (name, formula, values); band shares are (N, errors, bands)"""
    N = N_FIG
    DD, ZU, ZX = _grams("trained", N)
    turn = lambda Z, k, l: 1 - Z[..., k, l] / np.sqrt(Z[..., k, k] * Z[..., l, l])
    B1 = ga.band_both("trained", 1)
    share, own, Ea = (np.asarray(B1[k][:N], dtype=np.float64) for k in ("share", "own", "E"))
    keep = Ea.sum((0, 1)) > 0
    act = Ea[..., keep] / Ea[..., keep].sum(-1, keepdims=True)                   # (N, 30, bands)
    pr = np.load(ga.CACHE.format("trained"), mmap_mode="r")["pr"][:N].astype(np.float64)
    same = 1 - collect_turns("trained", N)["cos_u"][:, S]                       # 1 - cos(u_s, u_{s-1})
    f_act = r"$\sum_{pc \in B} \mathrm{proj}_{pc}^2 \, / \sum_{pc} \mathrm{proj}_{pc}^2$ (bands weighted by the fit)"
    at = lambda k: act[:, S.start - k:S.stop - k]
    pos = np.asarray(B1["band"])[keep] / 100                                     # place in the variance ranking
    com = (act * pos).sum(-1)                                                    # (N, 30)
    chg = np.concatenate([np.full((N, 1), np.nan), .5 * np.abs(np.diff(act, axis=1)).sum(-1)], 1)
    f_s = r"s_B = \sum_{pc \in B} \mathrm{proj}_{pc}^2 / \sum_{pc} \mathrm{proj}_{pc}^2"
    return {
        "com_1": ("Centre of mass of the activity energy, one step earlier",
                  r"$\sum_B \mathrm{pos}_B\, s_B$ at $t-1$,  $" + f_s + "$", com[:, S.start - 1:S.stop - 1]),
        "chg_2": ("Band-profile change, two steps earlier",
                  r"$\frac{1}{2} \sum_B |s_B(t-2) - s_B(t-3)|$", chg[:, S.start - 2:S.stop - 2]),
        "coh": ("Credit coherence", r"$\|\sum_k \delta_k\|^2 \, / \, \sum_k \|\delta_k\|^2$",
                DD.sum((-1, -2)) / np.einsum("...kk->...", DD)),
        "cross": ("Cross-band share", r"$(\|\Delta W\|^2 - \Sigma_B \|\Delta W_B\|^2) \, / \, \|\Delta W\|^2$",
                  (share - own).sum(-1) / share.sum(-1)),
        "u_new": ("Activity turn, newest", r"$1 - \cos(u_{t-1}, u_{t-2})$", turn(ZU, 0, 1)),
        "u_next": ("Activity turn, next", r"$1 - \cos(u_{t-2}, u_{t-3})$", turn(ZU, 1, 2)),
        "u_3": ("Activity turn, three steps", r"$\sum_{j=0}^{2} [1 - \cos(u_{t-j}, u_{t-j-1})]$",
                same + turn(ZU, 0, 1) + turn(ZU, 1, 2)),
        "x_new": ("Observation turn, newest", r"$1 - \cos(x_t, x_{t-1})$", turn(ZX, 0, 1)),
        "x_next": ("Observation turn, next", r"$1 - \cos(x_{t-1}, x_{t-2})$", turn(ZX, 1, 2)),
        "eff_1": ("Effective dim., one step earlier", r"$(\sum_{pc} \mathrm{proj}_{pc}^2)^2 / \sum_{pc} "
                  r"\mathrm{proj}_{pc}^4$ at $t-1$", pr[:, S.start - 1:S.stop - 1]),
        "act_0": ("Activity energy share by band\nsame step", f_act, at(0)),
        "act_1": ("Activity energy share by band\none step earlier", f_act, at(1)),
        "act_2": ("Activity energy share by band\ntwo steps earlier", f_act, at(2)),
        "act_02": ("Activity energy share by band\nsame step and two steps earlier", f_act,
                   np.concatenate([at(0), at(2)], -1))}


ROLE_COL = {"com_1": "#8e44ad", "chg_2": "#c39bd3", "coh": "#c0392b", "cross": "#c0392b", "u_new": "#1f77b4", "u_next": "#85c1e9", "u_3": "#1f77b4",
            "x_new": "#1e8449", "x_next": "#82e0aa", "eff_1": "#2c3e50", "act_0": "#5b2c6f", "act_1": "#8e44ad",
            "act_2": "#c39bd3", "act_02": "#5b2c6f"}


def _heldout(y, groups, b, lam=1e-6):
    """% of y's r curve kept when the fit is made on one half of the episodes and scored on the other (both ways)"""
    F = np.concatenate([g if g.ndim == 3 else g[..., None] for g in groups], -1)
    half = np.arange(len(y)) < len(y) // 2
    out = []
    for tr in (half, ~half):
        A = _cen(F[tr]).reshape(-1, F.shape[-1])
        G = A.T @ A
        w = np.linalg.solve(G + lam * np.trace(G) / len(G) * np.eye(len(G)), A.T @ _cen(y[tr]).ravel())
        parts = _split(y[~tr], [F[~tr] @ w], b[~tr])
        tot = _r(y[~tr], b[~tr])
        out.append(100 * (1 - (parts[1] ** 2).sum() / (tot ** 2).sum()))
    return float(np.mean(out))


def _rows_figure(name, rows, note):
    """one figure: each row a set of variables (keys of _fit_variables) and a title; a within-episode pooled fit of
    dW's effective dimension on the set, then the additive split of its r curve into the fit and what is left"""
    N = N_FIG
    b = _ref("trained", N)
    DD, ZU, ZX = _grams("trained", N)
    dW = _pr(DD, ZU + ZX)
    V = _fit_variables()
    nv = max(len(keys) for _, keys in rows)
    nr = len(rows)
    fig = plt.figure(figsize=(4.7 * nv + 6.1, 5.6 * nr))
    gs = fig.add_gridspec(nr, nv + 1, wspace=.3, hspace=1.0, width_ratios=[1] * nv + [1.3])
    tot = _r(dW, b)
    axs = []
    for i, (_, keys) in enumerate(rows):
        comb, parts, _, _ = _fit_parts(dW, [V[k][2] for k in keys], b)
        expl = _heldout(dW, [V[k][2] for k in keys], b)
        for col, (k, cv) in enumerate(zip(keys, comb)):
            label, formula, v = V[k]
            ax = fig.add_subplot(gs[i, col])                 # a single measure: its own curve; bands: the fitted sum
            _lines(ax, [(_r(v if v.ndim == 2 else cv, b), ROLE_COL[k], label)], f"{label}\n{formula}", legend=False)
            ax.title.set_fontsize(FS["title"] * (.85 if v.ndim == 2 and len(formula) < 60 else .7 if v.ndim == 2
                                                 else .65))
            axs.append(ax)
            if col == 0:
                ax.set_ylabel("r (Cross-correlation)")
        ax = fig.add_subplot(gs[i, nv])
        _lines(ax, [(tot, COL["total"], "actual"),
                    (sum(parts[:-1]), "0.45", f"fit ({expl:.1f}% of the curve, held out)", "--"),
                    (parts[-1], COL["swap_u"], "left over", "-")],
               r"$\Delta W$ effective dim." + "\n" + r"$(\sum_i \sigma_i^2)^2 \, / \, \sum_i \sigma_i^4$", legend=False)
        ax.legend(frameon=False, fontsize=FS["legend"] * .8, loc="upper left")
        axs.append(ax)
    lo = min(a.get_ylim()[0] for a in axs)
    hi = max(a.get_ylim()[1] for a in axs)
    for a in axs:
        a.set_ylim(lo, hi)
        a.set_xlabel(r"Lag ($\tau$)")
    tops = gs.get_grid_positions(fig)[1]
    for (title, _), top in zip(rows, tops):
        fig.text(.5, top + 1.25 / fig.get_figheight(), title, ha="center", va="bottom", fontsize=FS["title"])
    bottom = gs.get_grid_positions(fig)[0][-1]
    fig.text(.5, bottom - 1.4 / fig.get_figheight(), r"The update at step $t$ is $\Delta W = \sum_k \delta_k\, [x_{t-k};\ "
             r"u_{t-k-1}]^T$ (as in the dW schematic).  $\delta_k$: the credit (gradient on the gate inputs) $k$ steps "
             r"back;  $x_t$: the LSTM's input (observation and context);  $u_t$: the activity;  $\mathrm{proj}_{pc}$: "
             r"its projection on PC $pc$, $B$: a band of 1% of its variance;  lag $\tau$: steps after the entangled "
             "step.\n" + note + "  Trained network.", ha="center", va="top", fontsize=FS["annot"] * .85, color="0.3")
    _save(fig, name)


def fig_four():
    _rows_figure("explain_four.png", [
        ("Credit coherence, the activity turn over three steps, the centre of mass one step earlier, the band-profile "
         "change two steps earlier", ["coh", "u_3", "com_1", "chg_2"]),
        ("Credit coherence, the activity turn over three steps, the centre of mass one step earlier, the observation "
         "turn one step back", ["coh", "u_3", "com_1", "x_next"])],
        "Four single-number variables: the best set (98.5% of the curve on held-out episodes) and the best with an "
        "observation variable (98.2%); 173 sets of four keep more than 97%.")


def fig_three():
    _rows_figure("explain_three.png", [
        ("Credit coherence, the activity turn over three steps, the centre of mass one step earlier",
         ["coh", "u_3", "com_1"]),
        ("Credit coherence, the activity turn over three steps, the effective dim. one step earlier",
         ["coh", "u_3", "eff_1"]),
        ("Credit coherence, the activity turn over three steps, the newest activity turn", ["coh", "u_3", "u_new"])],
        "Three single-number variables that keep more than 95% of the curve on held-out episodes (18 sets do); the "
        "top-band share, the effective number of bands or the band-profile change one step earlier\ncan replace the "
        "centre of mass (97.4-97.7%).  " + r"$\mathrm{pos}_B$" + ": band B's place in the variance ranking (0 = top band).")


def fig_two():
    _rows_figure("explain_two.png", [
        ("Credit coherence and the activity turn over three steps", ["coh", "u_3"])],
        "The only pair of single-number variables that keeps more than 90% of the curve on held-out episodes; the "
        "next best, credit coherence and the newest activity turn, keeps 88%.")


def plot_all(R=None, Rd=None):
    R = R or {net: collect(net) for net in ("trained", "echo")}
    Rd = Rd or {net: collect_directions(net) for net in ("trained", "echo")}
    with plt.rc_context(ga.RC):
        fig_error(R, Rd)
        fig_flip(R)
        fig_crossband(R)
        fig_bands(R)


if __name__ == "__main__":
    if sys.argv[1:] == ["plot"]:
        plot_all()
    elif sys.argv[1:2] == ["directions"]:
        for net in sys.argv[2:] or ("trained", "echo"):
            collect_directions(net)
    else:
        for net in sys.argv[1:] or ("trained", "echo"):
            collect(net)
