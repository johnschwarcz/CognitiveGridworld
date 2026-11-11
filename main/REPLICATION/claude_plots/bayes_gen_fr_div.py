"""How well does the Generator predict observations at the end of training, as a function of FR? (John 11-10 08:50)

Generators of train_bayes_generator.py (no Classifier; Generator + embeddings trained on an observer's answers in the RCC's
feedback format), d_E 30, one run per |S| (25 ... 1,000), at the end of training (joint / naive 6,000 episodes, truth1
8,000). Columns (John 11-10 09:30: 2 x 3 heat maps, axes |S| and FR):
    truth1   the goal's true value only (the other variable's slot empty): can learn only the goal's marginal P(o | r_g)
    naive    both answers sampled from the factorized (Naive) observer's marginals
    joint    both answers sampled from the Joint observer's marginals
Per episode (T = 30), at the episode's own state r = (r_g, r_c):
    colour  divergence of the Generator's predicted observations from the true ones (log scale, shared by all panels),
        sum over the 5 channels of KL(Bern(P(o | r)) || Bern(P^(o | r))), nats; P^ = the Generator's prediction given the
        true values with confidence 1 (truth1: given the goal's true value only, as in its training)
    y   FR of the episode, (FR_g + FR_c) / 2 with FR_v = KL(B_T(v) || B~_T(v)) (Joint vs Naive marginal belief at T, nats),
        in tenths (deciles of all novel test episodes of the goal-only runs; John 11-10 14:20: "a larger range of FR?
        Maybe 10 percentiles"; top tenth > 8.9 nats). Before: tenths, both environment sets (09:30); log quarter-decade
        bins (10:50, numbers below); fifths (11:40)
    x   |S|
Novel environments only (both variables held out). Colour map (John 11-10 10:00): green = low,
red = high divergence (RdYlGn_r, log scale). The goal-only Generator's floor (the TRUE
marginal P(o | r_g), the best any one-variable model can do) is .47 -> 1.23 nats from the lowest to the highest tenth.

11-10 09:20 (all 30 runs; each run's kl_own check matches its training log within 2%): novel environments, mean over the
runs that transfer (|S| 125-1000), FR tenths with mean FR .08 / .21 / .37 / .57 / .85 / 1.29 / 2.06 / 3.55 / 6.54 / 12.8:
    joint    .027 .027 .027 .028 .028 .028 .028 .029 .030 .032   (flat: x1.2 from the lowest to the highest tenth)
    naive    .111 .113 .122 .133 .149 .175 .210 .269 .356 .668   (x6.0)
    truth1   .531 .554 .586 .625 .668 .718 .814 .924 1.08 1.47   (x2.8), true-marginal floor .47 ... 1.23
11-10 10:50, log bins (novel, mean over |S| 125-1000; FR bins from .0056 to 28, quarter decades; the top bin is FR
17.8-28, 0.4% of episodes):
    joint    .023 .027 .027 .027 .027 .027 .027 .028 .028 .028 .028 .030 .030 .031 .031   (flat)
    naive    .163 .116 .119 .107 .111 .109 .115 .126 .145 .175 .229 .284 .392 .605 2.33   (x21 from FR ~.1 to the top bin)
    truth1   .537 .478 .505 .507 .521 .539 .560 .599 .655 .725 .852 .956 1.14 1.49 1.83
11-10 13:50 (novel only, plug-in added; RCC column): the RCC's own Generator (rcc_rl_S400_s0, zflow_s0-2 at |S| 500) is as
close to the true likelihood as the Joint-answer Generator: div .035 / .032-.036 vs joint .031 at 500, plug-in .70-.71 vs
.70; top FR fifth / bottom fifth x1.2-1.3 (Joint answers x1.2, Naive answers x5.0 at |S| 500). Plug-ins match the logs.
11-10 14:20 (John: "KL from Naive p(obs) - KL from Joint p(obs)"; bayes_gen_fr_div_vsnaive.png, DROPPED 15:10 by John
("let's stick to bayes_gen_fr_div_lines without the vsnaive"; plotter removed); pred / lik / lik_m added
to the stored files by --lik-only, same episodes checked): y = (1/2) sum_v KL(P(o | r_v) || P^) - KL(P(o | r) || P^),
FR tenths, mean over |S| 125/250/500/1000 (RCC: 400/500, 4 nets), lowest -> highest tenth:
    truth1  -.23 ... -.78  (closer to Naive's marginal, as it must be: it is given the goal only)
    naive   +.60 flat through the 9th tenth, +.28 in the top tenth
    joint   +.84 -> +1.75;  rcc  +.92 -> +1.86
    the true likelihood itself (P^ = P): .68 -> 1.53
  NOT a closeness score: a difference of KLs is not maximal at P^ = P. The Joint-answer and RCC Generators score ABOVE
  the true likelihood because they are more extreme than it (|P^ - .5| > |P - .5| on 65% of bits, joint |S| 500) while
  only .03 nats from it; the reverse direction KL(P^ || .) has the same overshoot (joint 1.29 vs truth 1.21 at the top).
  Read it for its sign and against the truth's bars; the -KL from the truth (bayes_gen_fr_div_lines.png) is the clean
  'how good' measure.
11-10 15:20 (John: "another gradient emerging at |S| = 25? real? larger at lower |S|?"): no. Held-out variables = |S| // 10,
so |S| 25 has 2 (3 novel environments, two of them a variable paired with itself; 210 distinct environment x state
cells), 50 has 5, 75 has 7, 100 has 10; no transfer there (1.7-3.4 nats). Top minus bottom FR tenth flips sign across
|S| 25 / 50 / 75 / 100 (Joint answers -.61 / +.07 / -.28 / +.02, Naive answers -.30 / +.26 / -.52 / -.05): the tenths sort
a few environments' states. Below |S| 20 there is one held-out variable (one novel environment), below 10 none.
11-10 20:10, 20,000 vs 6,000 episodes (Joint / Naive answers, mean over |S| 125-1000, FR tenths low -> high; measured
locally from pulled checkpoints): joint .027 -> .021 in every tenth (flat, x1.2), plug-in .69-.71 -> .71-.73; naive low
tenths improve (.111 -> .095) while the top two get WORSE (.356 -> .370, .668 -> .708): x6.0 -> x7.5, plug-in .52 -> .53.
Longer training on Naive's answers fits its high-FR errors more closely. |S| <= 100 unchanged (no transfer).
Goal only, 20,000 vs 8,000 (11-11 02:30, cluster job 66662653): .531 -> .511 in the lowest tenth, top unchanged (1.47),
x2.9, on its floor (the true marginal .47 -> 1.23); plug-in .30-.39 -> .32-.41 (its ceiling is Naive, .45).
Training environments: the same picture (joint x1.2, naive x7.6, truth1 x3.1 on its floor). |S| <= 100: no transfer
(novel 1-3.4 nats at every FR; joint at |S| 100 .19).

--tag _e20k: the same runs resumed to 20,000 episodes (train_bayes_generator.py --resume-to 20000; John 11-10: "train
more episodes, I think things haven't converged") -> <label>_S<S>_e20k.npz, bayes_gen_fr_div_e20k.png.

    cluster:  python bayes_gen_fr_div.py --compute <label> [--batches 4] [--tag _e20k]   (bayes_gen_fr_div.sbatch)
    local:    python bayes_gen_fr_div.py [--pull] [--tag _e20k]
-> main/DATA/experimental/bayes_gen_fr/<label>_S<S><tag>.npz (per episode: fr, div, div_floor, novel), figure bayes_gen_fr_div<tag>.png
"""
import os, sys, glob, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "main", "DATA", "experimental", "bayes_gen_fr")
SIZES = (25, 50, 75, 100, 125, 150, 200, 250, 500, 1000)
ROWS = (("truth1", "goal's true value only"), ("naive", "both answers from Naive Bayes"),
        ("joint", "both answers from Joint Bayes"))
B, R, C = 5000, 10, 2
NQ = 10                                      # FR tenths (John 11-10 14:20: "a larger range of FR? Maybe 10 percentiles")
NETS = os.environ.get("BG_NETS", "")         # local copies: BG_NETS=/experimental/bayes_gen_fr/nets (under main/DATA)


def bern_kl(p, q):
    p, q = p.clamp(1e-6, 1 - 1e-6), q.clamp(1e-6, 1 - 1e-6)
    return (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).sum(-1)


RCC_SIZES = (25, 50, 75, 100, 125, 250, 400, 500, 1000)
RCC_RUNS = lambda S: [f"rcc_rl_S{S}_s0"] + ([f"zflow_s{k}" for k in range(3)] if S == 500 else [])


def single_predict(m, sample, keep):
    """train_bayes_generator.predict with only the goal slot (the goal-only Generator's input)"""
    import torch
    B = sample.shape[0]
    emb = m.sample_to_emb(sample * keep.long()) * keep[..., None]
    s = m.sample_to_hid(emb.reshape(B, -1)).unsqueeze(1)
    c = m.conf_to_hid(keep).unsqueeze(1)
    z = m.Z_to_pobs(m.generator_Z())
    s, c, z = m.get_layer_norm(m.gen_stream_norm, (s, c, z))
    pre = s + c + z
    return torch.sigmoid(m.hid_to_pobs(torch.relu(pre + m.gen_hid2hid(torch.relu(pre))))).squeeze().float()


def plugin_hit(m, env, single):
    """per episode: does exact Bayes with the Generator's LEARNED likelihood get the goal right at T? (all R**C joint values
    at confidence 1 -> posterior -> goal marginal; goal-only Generator: its learned marginal, Naive-style, as in its log)"""
    import torch
    dev = env.device
    BR, g = torch.arange(B, device=dev), m.goal_ind.long()
    o = torch.as_tensor(env.obs_flat, device=dev, dtype=torch.float32)                       # (B, T, O)
    out = []
    if single:
        keep = torch.nn.functional.one_hot(g, C).float()
        for r in range(R):
            sample = torch.zeros(B, C, dtype=torch.long, device=dev)
            sample[BR, g] = r
            out.append(single_predict(m, sample, keep))
    else:
        ones = torch.ones(B, C, device=dev)
        for r in range(R ** C):
            m.get_prediction(torch.tensor(np.unravel_index(r, (R,) * C), device=dev).expand(B, C).contiguous(), ones)
            out.append(m.pred_pobs.float())
    P = torch.stack(out, -1).clamp(1e-6, 1 - 1e-6)
    ll = torch.einsum('bto,bok->bk', o, P.log()) + torch.einsum('bto,bok->bk', 1 - o, (1 - P).log())
    if not single:
        post = torch.softmax(ll, -1).reshape(B, R, R)
        ll = torch.where(g[:, None] == 0, post.sum(2), post.sum(1))
    return (ll.argmax(-1) == m.goal_value.long()).float()


def measure(label, S, batches, tag="", run=None, plugin=True):
    """novel environments only (11-10 12:40): per episode fr, div (at the own state), div_floor, plugin / joint / naive hit"""
    import torch
    from main.CognitiveGridworld import CognitiveGridworld
    seed = int(run.rsplit("_s", 1)[1]) if run else 0                 # the run's seed rebuilds its environments' split
    np.random.seed(seed); torch.manual_seed(seed)
    env = CognitiveGridworld(mode="RL", ctx_num=C, obs_num=5, realization_num=R, state_num=S, KQ_dim=30, hid_dim=1000,
                             batch_num=B, step_num=30, learn_embeddings=True, gen_zflow=True, episodes=1, training=False,
                             checkpoint_every=1, gpu_inference=True, show_plots=False, cuda=0,
                             load_env=f"{NETS}/cluster_runs/{run}" if run else f"{NETS}/bayes_gen/{label}_S{S}_d30_s0{tag}")
    m, dev = env.model, env.device
    BR = torch.arange(B, device=dev)
    single = label.endswith("1")
    out = {k: [] for k in ("fr", "div", "div_floor", "novel", "plugin", "joint", "naive", "pred", "lik", "lik_m", "goal")}
    for _ in range(batches):
        env.test_set = True
        env.run_generators()
        with torch.no_grad():
            jb = env.gpu_belief(env.joint_likelihood)[:, -1].double().clamp(min=1e-12)
            nb = env.gpu_belief(env.naive_likelihood, naive=True)[:, -1].double().clamp(min=1e-12)
        env.joint_goal_belief = jb[BR, torch.as_tensor(env.goal_ind, device=dev)].float().cpu().numpy()[:, None]
        env.prep_model()
        fr = (jb * (jb.log() - nb.log())).sum(-1).mean(1)                                    # (B,) mean over both variables
        v, g = m.ctx_vals.long(), m.goal_ind.long()
        L = torch.as_tensor(env.joint_likelihood, device=dev, dtype=torch.float32)[BR, :, v[:, 0], v[:, 1]]       # (B, O)
        Lm = torch.as_tensor(env.naive_likelihood, device=dev, dtype=torch.float32)[BR, :, g, v[BR, g]]        # (B, O)
        with torch.no_grad():
            m.MC_to_interactions()
            if single:
                P = single_predict(m, v, torch.nn.functional.one_hot(g, C).float())
            else:
                m.get_prediction(v.contiguous(), torch.ones(B, C, device=dev))
                P = m.pred_pobs.float()
            out["plugin"].append(plugin_hit(m, env, single).cpu().numpy() if plugin else np.full(B, np.nan))
        tv = m.goal_value.long()
        out["joint"].append((jb[BR, g].argmax(-1) == tv).float().cpu().numpy())
        out["naive"].append((nb[BR, g].argmax(-1) == tv).float().cpu().numpy())
        out["fr"].append(fr.float().cpu().numpy())
        out["div"].append(bern_kl(L, P).cpu().numpy())
        out["div_floor"].append(bern_kl(L, Lm).cpu().numpy())
        out["novel"].append(np.ones(B, bool))
        # 11-10 14:30 (John: "KL from Naive p(obs) - KL from Joint p(obs)"): keep the predicted and both observers'
        # likelihoods at the episode's state: pred P^(o | r), lik P(o | r_g, r_c), lik_m (B, 2, O) P(o | r_v) per variable
        out["pred"].append(P.cpu().numpy().astype(np.float32))
        out["lik"].append(L.cpu().numpy().astype(np.float32))
        nl = torch.as_tensor(env.naive_likelihood, device=dev, dtype=torch.float32)
        out["lik_m"].append(torch.stack([nl[BR, :, k, v[:, k]] for k in range(C)], 1).cpu().numpy())
        out["goal"].append(g.cpu().numpy().astype(np.int8))
    res = {k: np.concatenate(x) for k, x in out.items()}
    print(f"{run or label} S {S}: novel plug-in {res['plugin'].mean():.3f} joint {res['joint'].mean():.3f} "
          f"naive {res['naive'].mean():.3f} div {res['div'].mean():.3f}", flush=True)
    return res


LIK = ("pred", "lik", "lik_m", "goal")


def add_lik(path, res):
    """merge the likelihood fields into an existing file after checking the episodes are the same ones"""
    old = dict(np.load(path))
    assert np.allclose(old["fr"], res["fr"], rtol=1e-3, atol=1e-4) and np.allclose(old["div"], res["div"], rtol=2e-3, atol=1e-4), path
    np.savez(path, **old, **{k: res[k] for k in LIK})


def compute(label, batches, tag="", lik_only=False):
    """lik_only (11-10 14:30): recompute without the plug-in and add LIK to the existing files (same episodes, checked)"""
    os.makedirs(OUT, exist_ok=True)
    if label == "rcc":                                   # the RCC's own Generator, main-text protocol, every run per |S|
        for S in RCC_SIZES:
            runs = [r for r in RCC_RUNS(S)
                    if os.path.exists(os.path.join(ROOT, "main", "DATA", NETS.strip("/"), "cluster_runs", r + "_net.pth"))]
            if runs and lik_only:
                parts = [measure(label, S, batches, run=r, plugin=False) for r in runs]
                add_lik(os.path.join(OUT, f"rcc_S{S}.npz"), {k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
            elif runs:
                parts = [measure(label, S, batches, run=r) for r in runs]
                np.savez(os.path.join(OUT, f"rcc_S{S}.npz"), runs=np.array(runs),
                         **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
        return
    for S in SIZES:
        if lik_only:
            add_lik(os.path.join(OUT, f"{label}_S{S}{tag}.npz"), measure(label, S, batches, tag, plugin=False))
        else:
            np.savez(os.path.join(OUT, f"{label}_S{S}{tag}.npz"), **measure(label, S, batches, tag))


def plot(tag=""):
    """novel environments only, FR in NQ quantile bins of all novel test episodes pooled over the goal-only runs (fifths
    11:40, tenths 14:20): bayes_gen_fr_div.png = heat maps (y = FR bin, x = |S|), bayes_gen_fr_div_lines.png = the same
    numbers as one line per bin (x = |S|, log y)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    data = {(lab, S): dict(np.load(f)) for lab, _ in ROWS for S in SIZES
            if os.path.exists(f := os.path.join(OUT, f"{lab}_S{S}{tag}.npz"))}
    # FR bins from the untagged goal-only runs: FR is the observers' (seeded episodes), so every tag has the same bins
    pooled = np.concatenate([np.load(f)["fr"] for S in SIZES if os.path.exists(f := os.path.join(OUT, f"truth1_S{S}.npz"))])
    edges = np.quantile(pooled, np.linspace(0, 1, NQ + 1))
    names = ([f"$<$ {edges[1]:.2g}"] + [f"{a:.2g}-{b:.2g}" for a, b in zip(edges[1:-2], edges[2:-1])] +
             [f"$>$ {edges[-2]:.2g}"])
    maps, n_ep = {}, 0
    for lab, _ in ROWS:
        M = np.full((NQ, len(SIZES)), np.nan)
        for c, S in enumerate(SIZES):
            if (lab, S) in data:
                d = data[lab, S]
                sel = d["novel"] == 1
                n_ep = sel.sum()
                b = np.clip(np.searchsorted(edges, d["fr"][sel], side="right") - 1, 0, NQ - 1)
                M[:, c] = [d["div"][sel][b == k].mean() for k in range(NQ)]
        maps[lab] = M
    vals = np.concatenate([m.ravel() for m in maps.values()])
    norm = LogNorm(np.nanmin(vals), np.nanmax(vals))
    sub = (r"$d_E = 30$, 1 run per $|\mathcal{S}|$ (" + ("20,000 training episodes each); " if tag else
           "Naive / Joint answers: 6,000 training episodes, goal only: 8,000); ") +
           f"{n_ep:,} novel test episodes per run, 10% in each tenth of FR")
    cb_label = r"$\Sigma_o\,\mathrm{KL}(P(o\,|\,r) \,\|\, \hat P(o\,|\,r))$ at the episode's state (nats)"
    fr_label = r"FR of the episode, $\frac{1}{2}\Sigma_v \mathrm{KL}(B_T(v) \,\|\, \tilde B_T(v))$ (nats), tenths"

    fig, ax = plt.subplots(1, len(ROWS), figsize=(len(ROWS) * 3.3 + 1.3, 3.6), sharey=True, constrained_layout=True)
    for a, (lab, title) in zip(ax, ROWS):
        im = a.imshow(maps[lab], origin="lower", aspect="auto", cmap="RdYlGn_r", norm=norm, interpolation="nearest")
        for k in range(NQ):
            for c in range(len(SIZES)):
                v = maps[lab][k, c]
                a.text(c, k, f"{v:.2f}" if v < 9.95 else f"{v:.0f}", ha="center", va="center", fontsize=fs["legend"] - 4)
        a.set_title(title[0].upper() + title[1:], fontsize=fs["title"] - 3)
        a.set_xticks(range(len(SIZES)))
        a.set_xticklabels([str(S) for S in SIZES], rotation=90)
        a.set_xlabel(r"number of variables $|\mathcal{S}|$")
    ax[0].set_yticks(range(NQ))
    ax[0].set_yticklabels(names)
    ax[0].set_ylabel(fr_label)
    fig.colorbar(im, ax=ax, shrink=.85, pad=.02).set_label(cb_label)
    fig.suptitle("Divergence of the Generator's predicted observations from the true ones, novel environments, end of "
                 "training\n" + sub, fontsize=fs["legend"])
    out = fig_path(f"bayes_gen_fr_div{tag}.png")
    fig.savefig(out, dpi=200)
    print(out)

    # 11-10 13:00 (John: a measure where up is better, "just show the -KL"; the RCC as a 4th column as its data arrive):
    # -div per FR bin, drawn as KL on an inverted log axis with negative tick labels (0 = a perfect likelihood is up)
    cols = plt.cm.viridis(np.linspace(0, .9, NQ))
    COLS = ROWS + (("rcc", "the RCC, its own feedback"),)
    rcc = {S: dict(np.load(f)) for S in RCC_SIZES if os.path.exists(f := os.path.join(OUT, f"rcc_S{S}.npz"))}
    fig, ax = plt.subplots(1, len(COLS), figsize=(len(COLS) * 3.0 + 1.6, 3.5), sharey=True, constrained_layout=True)
    for a, (lab, title) in zip(ax, COLS):
        runs = rcc if lab == "rcc" else {S: data[lab, S] for S in SIZES if (lab, S) in data}
        xs = sorted(runs)
        for k in range(NQ):
            y = []
            for S in xs:
                d = runs[S]
                sel = d["novel"] == 1
                b = np.clip(np.searchsorted(edges, d["fr"][sel], side="right") - 1, 0, NQ - 1) == k
                y.append(np.mean(d["div"][sel][b]))
            a.plot(xs, y, color=cols[k], lw=1.8, marker="o", ms=3, label=names[k])
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xticks(SIZES if lab != "rcc" else RCC_SIZES)
        a.set_xticklabels([str(S) for S in (SIZES if lab != "rcc" else RCC_SIZES)], rotation=90)
        a.minorticks_off()
        a.set_xlim(20, 1250)
        n_runs = sum(len(runs[S].get("runs", [0])) for S in xs) if lab == "rcc" else len(xs)
        a.set_title(title[0].upper() + title[1:] + ("" if lab != "rcc" else f"\n({n_runs} runs so far)"),
                    fontsize=fs["title"] - 4)
        a.set_xlabel(r"number of variables $|\mathcal{S}|$")
        a.grid(alpha=.25, lw=.6)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    ax[0].invert_yaxis()                                 # shared y: up = smaller divergence = better
    ax[0].yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1., 3.)))
    ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"$-${v:g}"))
    ax[0].set_ylabel(r"$-\Sigma_o\,\mathrm{KL}(P(o\,|\,r) \,\|\, \hat P(o\,|\,r))$" + "\nat the episode's state (nats, log scale)")
    ax[-1].legend(handles=[plt.Line2D([], [], color=c, lw=1.8, marker="o", ms=3, label=n) for c, n in zip(cols, names)],
                  title="FR tenth (nats)", frameon=False, fontsize=fs["legend"] - 2, title_fontsize=fs["legend"] - 2,
                  loc="center left", bbox_to_anchor=(1.02, .5))
    fig.suptitle("How far the learned likelihood is from the true one, by FR tenth, novel environments (up = better, "
                 "0 = exact)\n" + sub.replace(", 10% in each tenth of FR", "") +
                 "; RCC: its own Generator, main-text protocol, 30,000 episodes",
                 fontsize=fs["legend"] - 1)
    out = fig_path(f"bayes_gen_fr_div_lines{tag}.png")
    fig.savefig(out, dpi=200)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--lik-only", action="store_true", help="add pred / lik / lik_m / goal to the existing files")
    a = ap.parse_args()
    if a.compute:
        import torch
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
        compute(a.compute, a.batches, a.tag, a.lik_only)
        return
    if a.pull:
        os.makedirs(OUT, exist_ok=True)
        subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/main/DATA/experimental/bayes_gen_fr/*{a.tag}.npz' {OUT}/", shell=True)
    plot(a.tag)


if __name__ == "__main__":
    main()
