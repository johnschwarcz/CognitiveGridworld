"""The channel-local read-in's read-outs on the trained read-in networks (John 11-14: "Do we have that fancy readout for
the trained readin versions?"). esn_convex.py scores its fixed reservoirs with soft-target logistic read-outs of
    sum       summed [x; u]           (x = reservoir state, u = the counts and T x Z, esn_convex.feats2)
    sum_sq    summed [x; x^2; u]      (the read-out of the channel-local points in esn_convex.png)
while esn_trained_readin.py's exact read-out uses the summed states x alone. This refits both feature sets on the saved
trained networks (runs/esn_trained_readin/<name>_net.pt; saved by the runs from 11-13 on).
Test sets = the run's own (the first draws of its generator, seed 1000 + seed); fit set = 200,000 fresh training-
environment episodes from the same generator (the run's fit set came later in its stream, same distribution). The
logistic path is esn_convex.logistic_path (same lambdas, standardized features, L-BFGS, Joint's posterior at T as the
target), run on the GPU.
    python esn_trained_sqreadout.py <name> [<name> ...]   -> runs/esn_trained_readin/<name>_sq.json

RESULT 11-14 03:43 IST (the 4 saved N 1,000 nets; test accuracy, summed x (the run's exact read-out) / summed [x; u] /
summed [x; x^2; u]): rho 1.2 .420 / .422 / .427; alpha .03 .406 / .402 / .423; p .006 .400 / .403 / .398; rho 1.5
.392 / .397 / .396. Squares add <= .02 here, against +.08 for the channel-local fixed read-in (best setting at N 1,000:
.377 -> .458). So with the SAME squared read-out the channel-local still beats the trained read-in by ~.03; with plain
summed read-outs the trained read-in wins by ~.03. Not "all the read-out" (my 11-14 claim, corrected the same hour).
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import esn_trained_readin as TR
import esn_convex as E


def logistic_path(Ftr, vals, tests, soft, iters=200):
    """esn_convex.logistic_path on Ftr's device (identical steps)"""
    import torch.nn.functional as Fn
    dev = Ftr.device
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    X = (Ftr - mu) / sd
    W = torch.zeros(X.shape[1], E.C * E.R, device=dev, requires_grad=True)
    b = torch.zeros(E.C * E.R, device=dev, requires_grad=True)
    out = {k: [None] * len(E.LOG_LAMS) for k in tests}
    for j in reversed(range(len(E.LOG_LAMS))):
        lam = E.LOG_LAMS[j]
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, history_size=20, tolerance_grad=1e-6,
                                tolerance_change=1e-9, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            Zl = (X @ W + b).reshape(len(X), E.C, E.R)
            loss = -(soft * Fn.log_softmax(Zl, -1)).sum(-1).mean(0).sum() + lam * (W ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            for k, (Ft, vt, goal) in tests.items():
                P = (((Ft - mu) / sd) @ W + b).reshape(len(Ft), E.C, E.R)
                i = torch.arange(len(Ft), device=dev)
                out[k][j] = float((P[i, goal].argmax(-1) == vt[i, goal]).float().mean())
    return out


def feats(net, d, chunk=5000):
    """summed [x; u] and summed [x; x^2; u] of each episode"""
    s1, s2, u = [], [], []
    with torch.no_grad():
        for j in range(0, len(d["obs"]), chunk):
            o, Z = d["obs"][j:j + chunk], d["Z"][j:j + chunk]
            X = net.states(o, Z)
            s1.append(X.sum(1))
            s2.append((X * X).sum(1))
            u.append(torch.cat([o.float().sum(1), TR.T * Z.float()], 1))
    s1, s2, u = torch.cat(s1), torch.cat(s2), torch.cat(u)
    return torch.cat([s1, u], 1), torch.cat([s1, s2, u], 1)


def main(name, n_test=20_000, n_fit=200_000):
    ck = torch.load(os.path.join(TR.RUNS, name + "_net.pt"))
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = bool(ck["tf32"])
    torch.manual_seed(ck["seed"])
    genv = TR.GPUEnv(TR.make_env(), dev)
    net = TR.ESN(ck["rho"], ck["alpha"], ck["density"], ck["N"], ck["seed"], dev)
    with torch.no_grad():
        for p, q in zip(net.params, ck["params"]):
            p.copy_(q.to(dev))
    gen = torch.Generator(device=dev).manual_seed(1000 + ck["seed"])
    tests = dict(val=genv.sample(n_test, True, gen), test=genv.sample(n_test, True, gen),
                 test_train=genv.sample(n_test, False, gen))
    t0 = time.time()
    fit = {k: [] for k in ("sum", "sum_sq")}
    post, vals = [], []
    for j in range(0, n_fit, 5000):
        d = genv.sample(min(5000, n_fit - j), False, gen)
        f1, f2 = feats(net, d)
        fit["sum"].append(f1)
        fit["sum_sq"].append(f2)
        post.append(TR.posterior(d["obs"], d["Z"])[:, -1])
        vals.append(d["vals"])
    post, vals = torch.cat(post), torch.cat(vals)
    tf = {s: feats(net, d) for s, d in tests.items()}
    out = dict(name=name, rho=ck["rho"], alpha=ck["alpha"], density=ck["density"], N=ck["N"], seed=ck["seed"])
    for i, k in enumerate(("sum", "sum_sq")):
        Ftr = torch.cat(fit[k])
        fit[k] = None
        path = logistic_path(Ftr, vals, {s: (tf[s][i], d["vals"], d["goal"]) for s, d in tests.items()}, soft=post)
        del Ftr
        rep = E.report_log(path)
        out[k + "_soft"] = {kk: rep[kk] for kk in ("lam", "val", "test", "test_train")}
        out[k + "_soft"]["curves"] = path
        torch.cuda.empty_cache()
    out["secs"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(TR.RUNS, name + "_sq.json"), "w"))
    print(f"{name}: summed [x; u] {out['sum_soft']['test']:.3f}, summed [x; x^2; u] {out['sum_sq_soft']['test']:.3f} "
          f"({out['secs']:.0f} s)", flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        main(n)
