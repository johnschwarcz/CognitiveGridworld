"""Interaction across observation dimensions of the ESN + MLPs network (run mlp55: the ESN's frozen LSTM between a
trained 5-layer read-in and a trained 5-layer read-out), on the grid of td_tt_mixed_diff / td_tt_mixed_diff0:
    through time   MD  = mixed second difference of the centred read-out increment at step t under a flipped bit of
                   o_t and a flipped bit of an earlier observation o_s (32 (t, s) pairs, 25 bit pairs)
    within a step  MD0 = the same with both flipped bits in o_t (6 steps, 10 bit pairs)
Same 1,000 probes (50 test environments x 20), same 34 checkpoints, same statistics and environment resamples as the
Trained RNN and the ESN. Only the cell differs: `DeepCell` gives the network's drive table, recurrent product, LSTM
step and centred read-out increment through the two MLPs, and md.net_md / md0.net_md0 are used unchanged.
Question (11-10): the ESN + MLPs reaches Bayes's accuracy at high factorization regret without recurrent training
(fig_learn_profile c). Does it also combine the present observation with past ones, as the Trained RNN does and the
ESN does not?

RESULTS (mean |MD| per value, nats; Bayes .0748 through time, .0754 within a step; naive 0 in both)
  through time, training episodes 240 / 880 / 2,400 / 39,996:
      ESN + MLPs   .0319 / .0947 / .1112 / .0704  = 43% / 127% / 149% / 94% of Bayes's
      Trained RNN  .0065 / .0199 / .0356 / .0542  (end 72%);   ESN .0002 / .0004 / .0009 / .0049  (end 6.6%)
      pooled kappa = Sigma MD MD_Bayes / Sigma MD_Bayes^2 at the end: ESN + MLPs .222, Trained RNN .193, ESN .002
      (cosine with Bayes's MD = kappa / sqrt(energy): .38, .40, .03)
  range at the end (t = 15-30), lag 1 / 3 / 12: ESN + MLPs .083 / .064 / .033; Trained RNN .053 / .040 / .033;
      ESN .013 / .004 / .0001. The ESN + MLPs's range grows through training (lag 12: .0002, .0012, .0025, .0325).
  within a step, the same episodes: ESN + MLPs .233 / .644 / .467 / .205 = 309% / 855% / 619% / 272% of Bayes's;
      Trained RNN .048 / .073 / .074 / .088 (end 117%); ESN .016 / .026 / .042 / .115 (end 152%); Bayes without
      memory (td_analysis' window 1) .47.
  Validation: the cell's beliefs equal the stored test beliefs (analysis/mlp55/ck_<raw>.npz) to float16 storage
      precision at every checkpoint and give the same answers at T (`validation.json`).
READING: the two networks that end on Bayes's FR profile (Trained RNN, ESN + MLPs) have a through-time interaction of
Bayes's size; the ESN, which ends at chance where FR is high, has 6.6% of it. A signature, neither a sufficient nor a
necessary cause: the Trained RNN with the gradient blocked at the highest-FR half keeps 71% of the unblocked
network's interaction at 2,400 episodes (.0250 vs .0352) while its accuracy in the highest FR tenth is .22 vs .62;
and a rule that is additive through time (linear in the 32 pattern counts, fitted per environment and variable)
reaches .60 there (fr_profile/checks/additive_fit.txt). The ESN + MLPs's read-out is non-linear, so its interaction
can be made after the reservoir; no claim about where it is made.

-> main/DATA/experimental/training_dynamics/through_training/mixed_diff/
   ck_mlp55_<raw>.npz, ck0_mlp55_<raw>.npz   as the other runs' (td_tt_mixed_diff.harvest_one, td_tt_mixed_diff0.harvest)
   summary_deep.npz   x, raw, md / kappa / energy (+ _lo, _hi: 5-95% over 2,000 environment resamples), md_lag
                      (checkpoint, 6 lags; t = 15-30), md0 (all six steps, + _lo, _hi), kappaB0 (t = 15-30)
   validation.json    per checkpoint: max and mean |log q - stored|, share of probes with the same answers at T
Run from the repository root (CPU; ~100 s per checkpoint at 5 threads with three parts in parallel, ~22 min for the
34 checkpoints; the parts split the checkpoints):
    CUDA_VISIBLE_DEVICES="" MD_THREADS=5 /home/johns/anaconda3/envs/eureka/bin/python \
        main/REPLICATION/claude_plots/td_tt_mixed_diff_deep.py harvest [part n_parts] | summary | numbers"""
import os
import sys
import json
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
import td_tt_gap as gap                                                                       # noqa: E402
import td_tt_intervene as ti                                                                  # noqa: E402
import td_tt_wormholes as tw                                                                  # noqa: E402
import td_tt_mixed_diff as md                                                                 # noqa: E402
import td_tt_mixed_diff0 as md0                                                               # noqa: E402

RUN = "mlp55"
T, R, NV, NO, OUT = md.T, md.R, md.NV, md.NO, md.OUT
SUMMARY = os.path.join(OUT, "summary_deep.npz")


def seq(w, prefix, x):
    """an mlp_stack (model_architecture): Linear layers at the even indices, ReLU between them"""
    ks = sorted({int(k.split(".")[1]) for k in w if k.startswith(prefix + ".") and k.endswith(".weight")})
    for n, k in enumerate(ks):
        x = torch.nn.functional.linear(x, w[f"{prefix}.{k}.weight"], w[f"{prefix}.{k}.bias"])
        if n < len(ks) - 1:
            x = torch.relu(x)
    return x


class DeepCell:
    """a checkpoint of the ESN + MLPs with md.LSTMCell's interface: drive table (NE, 32, 4H) incl. both biases (the
    5-layer read-in applied to every observation pattern of every probe environment), recurrent product, LSTM step,
    centred read-out increment (the 5-layer read-out of the state)"""

    def __init__(self, run, raw, dtype=md.DT):
        w = torch.load(ti.ckpt_path(run, raw), map_location="cpu", weights_only=False)["weights"]
        self.w = w = {k: v.detach().to(dtype) for k, v in w.items() if torch.is_tensor(v) and v.is_floating_point()}
        Z = ti.probes()["Z"]
        Zt = torch.as_tensor(np.asarray(Z, np.float32)).reshape(len(Z), -1).to(dtype)
        pat = torch.as_tensor(tw._PAT).to(dtype)
        inp = torch.cat([pat[None].expand(len(Z), 32, NO), Zt[:, None].expand(len(Z), 32, Zt.shape[-1])], -1)
        x = torch.relu(seq(w, "classifier_readin", inp))
        self.table = x @ w["LSTM.weight_ih_l0"].T + (w["LSTM.bias_ih_l0"] + w["LSTM.bias_hh_l0"])
        self.Whh = w["LSTM.weight_hh_l0"]

    def rec(self, h):
        return h @ self.Whh.T

    def act(self, pre, c):
        return md._act(pre, c)

    def raw_inc(self, h):
        return seq(self.w, "classifier_readout", h)

    def inc(self, h):
        x = self.raw_inc(h).reshape(-1, NV, R)
        return x - x.mean(-1, keepdim=True)

    def replay(self, envt, code):
        n, H = len(envt), self.Whh.shape[1]
        U = torch.empty(n, T + 1, H, dtype=self.Whh.dtype)
        C = torch.empty_like(U)
        U[:, 0], C[:, 0] = self.w["STM"].reshape(1, H), self.w["LTM"].reshape(1, H)
        for t in range(T):
            U[:, t + 1], C[:, t + 1] = self.act(self.table[envt, code[:, t]] + self.rec(U[:, t]), C[:, t])
        return U, C


def val_file(raw):
    return os.path.join(OUT, f"val_{RUN}_{int(raw):05d}.json")


@torch.no_grad()
def validate(raw, S, envt, code):
    """the cell's beliefs on the probes against the stored test beliefs of the same checkpoint"""
    cell = DeepCell(RUN, raw)
    U, _ = cell.replay(envt, code)
    lg = torch.stack([cell.raw_inc(U[:, t + 1]) for t in range(T)], 1).double().cumsum(1).reshape(len(envt), T, NV, R)
    mine = torch.log_softmax(lg, -1).numpy()
    ref = np.load(os.path.join(ti.AN, RUN, f"ck_{int(raw):05d}.npz"))["logq"][S["rows"]].astype(np.float64)
    d = np.abs(mine - ref)
    out = dict(raw=int(raw), max=float(d.max()), mean=float(d.mean()),
               same_answer=float((mine[:, -1].argmax(-1) == ref[:, -1].argmax(-1)).mean()))
    json.dump(out, open(val_file(raw), "w"))
    return out


def harvest(part=0, n_parts=1):
    torch.set_num_threads(md.THREADS)
    raws = md.RAWS[int(part)::int(n_parts)]
    S = md.side()
    envt, code = md.codes(S["rows"])
    for raw in raws:
        if not os.path.exists(val_file(raw)):
            v = validate(raw, S, envt, code)
            print(f"validation raw {raw:5d}: max |log q - stored| {v['max']:.3g}, mean {v['mean']:.3g}, same answer at T "
                  f"{v['same_answer']:.4f}", flush=True)
    md.LSTMCell = DeepCell                                    # the harvests build their cell by this name
    md.harvest((RUN,), raws)
    md0.harvest((RUN,), raws)


def summary():
    """-> summary_deep.npz, the statistics of td_tt_mixed_diff.summary / td_tt_mixed_diff0.summary0 for the run"""
    B, O = np.load(md.bayes_file()), np.load(md0.obs_file())
    env, C = B["env"], gap.boot_counts()
    pairs = np.array(md.PAIRS)
    lag = pairs[:, 0] - pairs[:, 1]
    sel_lag = [(lag == d) & np.isin(pairs[:, 0], md.T_BAL) for d in md.LAGS]
    bal = np.isin(np.array(md0.TS), (15, 20, 25, 30))
    n0 = len(md0.BITS) * R
    raws = [r for r in md.RAWS if os.path.exists(md.ck_file(RUN, r)) and os.path.exists(md0.ck0_file(RUN, r))]
    cols = {k: [] for k in ("md", "md_lo", "md_hi", "kappa", "kappa_lo", "kappa_hi", "energy", "md_lag", "md0",
                            "md0_lo", "md0_hi", "kappaB0")}
    for raw in raws:
        z, z0 = np.load(md.ck_file(RUN, raw)), np.load(md0.ck0_file(RUN, raw))
        assert np.array_equal(z["rows"], B["rows"]) and np.array_equal(z0["rows"], B["rows"])
        st = md._stats(env, z["s_abs"], z["s_nb"], z["s_nn"], B["s_bb"], C=C)
        for k in ("md", "kappa"):
            cols[k].append(st[k][0])
            lo, hi = md._band(st[k][1])
            cols[k + "_lo"].append(lo)
            cols[k + "_hi"].append(hi)
        cols["energy"].append(st["energy"][0])
        cols["md_lag"].append([md._stats(env, z["s_abs"], z["s_nb"], z["s_nn"], B["s_bb"], sel=sl, C=C)["md"][0]
                               for sl in sel_lag])
        a = np.bincount(env, z0["s_abs"].sum((1, 2)), gap.N_ENV)                    # all six steps, both variables
        b = np.bincount(env, np.full(len(env), float(len(md0.TS) * NV * n0)), gap.N_ENV)
        boot = (C @ a) / (C @ b)
        cols["md0"].append(a.sum() / b.sum())
        lo, hi = np.percentile(boot, [5, 95])
        cols["md0_lo"].append(lo)
        cols["md0_hi"].append(hi)
        cols["kappaB0"].append(z0["s_nb"][:, bal].sum() / O["bayes_s_bb"][:, bal].sum())
    val = [json.load(open(val_file(r))) for r in raws if os.path.exists(val_file(r))]
    json.dump(val, open(os.path.join(OUT, "validation.json"), "w"), indent=1)
    np.savez(SUMMARY, run=RUN, raw=np.array(raws), x=gap.train_x(np.array(raws)), lags=np.array(md.LAGS),
             md_bayes=md._stats(env, B["s_abs"], None, None, None, C=C)["md"][0],
             md0_bayes=O["bayes_s_abs"].sum() / (O["bayes_s_abs"].size * n0),
             val_max=max(v["max"] for v in val), val_same_answer=min(v["same_answer"] for v in val),
             **{k: np.array(v) for k, v in cols.items()})
    print("saved", SUMMARY, f"({len(raws)} checkpoints; validation: max |log q - stored| "
          f"{max(v['max'] for v in val):.3g}, same answer at T >= {min(v['same_answer'] for v in val):.4f})")


def numbers():
    z = np.load(SUMMARY)
    Z, Z0 = np.load(os.path.join(OUT, "summary.npz")), np.load(os.path.join(OUT, "summary0.npz"))
    print(f"Bayes: through time {float(z['md_bayes']):.4f} (summary.npz {float(Z['md_bayes']):.4f}), within a step "
          f"{float(z['md0_bayes']):.4f} (summary0.npz {float(Z0['md0_bayes_allt'][0]):.4f})")
    print("training episodes: through time [5-95%] (% of Bayes's; kappa; cosine) | by lag 1, 2, 3, 5, 8, 12 | within a "
          "step [5-95%] (kappa_B) || Trained RNN, ESN through time | within a step")
    for i, x in enumerate(z["x"]):
        j = {r: int(np.argmin(np.abs(Z[f"x_{r}"] - x))) for r in ("trained", "echo")}
        print(f"  {x:6.0f}: {z['md'][i]:.4f} [{z['md_lo'][i]:.4f}, {z['md_hi'][i]:.4f}] ({100 * z['md'][i] / z['md_bayes']:3.0f}%; "
              f"{z['kappa'][i]:+.3f}; {z['kappa'][i] / max(np.sqrt(z['energy'][i]), 1e-12):+.2f}) | "
              + " ".join(f"{v:.4f}" for v in z["md_lag"][i])
              + f" | {z['md0'][i]:.4f} [{z['md0_lo'][i]:.4f}, {z['md0_hi'][i]:.4f}] ({z['kappaB0'][i]:+.3f}) || "
              + ", ".join(f"{Z[f'md_{r}'][j[r]]:.4f}" for r in j) + " | "
              + ", ".join(f"{Z0[f'md0_allt_{r}'][j[r]]:.4f}" for r in j))


if __name__ == "__main__":
    {"harvest": harvest, "summary": summary, "numbers": numbers}[sys.argv[1]](*sys.argv[2:])
