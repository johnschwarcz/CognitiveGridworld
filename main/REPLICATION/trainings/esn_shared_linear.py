"""Best SHARED read-out linear in the counts (John 11-10 16:20: "the best linear readout of the counts needs to be fit over
all environments, no? Maybe it can have Z as a regressor, but it shouldnt be separate per environment, that's cheating").
One multinomial logistic read-out for all environments (esn_convex.logistic_path: convex, L-BFGS, soft targets = the
Joint posterior at T, lambda chosen on the novel-environment val split, accuracy on test), same dataset and protocol as
the ESN read-outs in esn_convex.png. Feature sets, n = the 5 channel counts at T, Z = the 10 context phases (5 channels x
2 variables):
    counts_Z        [n, Z]                                    linear in counts, Z as extra regressors
    counts_xZ       [n, Z, n_k Z_kv]                          + each count's weight linear in its channel's two phases
    counts_xfourier [n, f(Z), n_k f(Z_kv)], f = sin/cos 2 pi m Z, m = 1, 2: each count's weight a smooth periodic
                                                              function of its channel's phases
-> main/DATA/experimental/esn_convex/shared_linear/result.json
    python esn_shared_linear.py
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import esn_convex as E

OUT = os.path.join(E.ROOT, "main", "DATA", "experimental", "esn_convex", "shared_linear")


def fourier(Z):
    return np.concatenate([f(2 * np.pi * m * Z) for m in (1, 2) for f in (np.sin, np.cos)], 1)     # (B, 40)


def feats(split, kind):
    n = split["obs"].sum(1).astype(np.float32)                                    # (B, 5)
    Z = split["Z"].astype(np.float32)                                             # (B, 10): channel k -> Z[2k], Z[2k+1]
    nk = np.repeat(n, 2, 1)                                                       # n_k aligned with Z_kv
    if kind == "counts_Z":
        F = [n, Z]
    elif kind == "counts_xZ":
        F = [n, Z, nk * Z]
    else:
        fz = [np.concatenate([f(2 * np.pi * m * Z) for f in (np.sin, np.cos)], 1) for m in (1, 2)]   # (B, 20) each
        F = [n, fourier(Z)] + [np.tile(nk, 2) * f for f in fz]
    return np.concatenate(F, 1).astype(np.float32)


def main():
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    os.makedirs(OUT, exist_ok=True)
    data = E.load_data()
    post = E.joint_post(data["train"])
    res = {}
    for kind in ("counts_Z", "counts_xZ", "counts_xfourier"):
        t0 = time.time()
        Ftr = torch.as_tensor(feats(data["train"], kind))
        tests = {s: (torch.as_tensor(feats(data[s], kind)), data[s]["vals"], data[s]["goal"]) for s in ("val", "test", "test_train")}
        r = E.report_log(E.logistic_path(Ftr, data["train"]["vals"], tests, soft=post))
        res[kind] = dict(dim=int(Ftr.shape[1]), **{k: r[k] for k in ("lam", "val", "test", "test_train")})
        print(kind, res[kind], f"{time.time() - t0:.0f} s", flush=True)
    json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
