"""What do embeddings learned through the Classifier lack? (John 11-11: "Can you see any differences in the properties of
the learned embeddings or Z? Can the generator use the embeddings learned by the classifier or are they not useful for
generation?")

Fig 13's 60k batch (lr 1e-4): oracle Generator route gen60k, oracle Classifier route clf60k, the RCC rcc60k (original
feedback, Generator route), original feedback Classifier route rccclf60k; seeds 0-1 at 60,000 episodes (checkpoints
copied to main/DATA/experimental/fig13_embeddings/nets/cluster_runs).

The model computes, per observation channel o, the code Z_o = (K_g . Q_c, K_c . Q_g) from the learned embeddings, the
same form as the environment's true Z_o, which alone sets channel o's likelihood table P(o_o = 1 | r_g, r_c) (10 x 10).
The Generator reads Z one channel at a time with ONE map shared by the channels (Z_to_pobs per channel); the Classifier
reads all channels' Z and bits together, so it may mix channels.
Measures (per run; 5,000 training-environment and 5,000 novel-environment episodes; a fresh k-nearest-neighbour decoder,
k = 10, fit on the training environments, scored on the novel ones):
    own Z_o -> table_o     channel o's learned Z (2 numbers) -> channel o's true table, one decoder for all channels:
                           what a channel-wise Generator could read
    all Z -> table_o       all 10 learned numbers -> channel o's table (a decoder per channel): is the information there
                           at all, in any arrangement
    true_all               the same 10-number decoder on the TRUE Z: the decoder's own ceiling at this sample size
    own_specific           channel o's learned Z -> channel o's table with a decoder PER channel (not shared)
    best_other             the best channel o' != o: Z_o' -> table_o (swapped channels)
    PR(Z)                  participation ratio of the learned Z across environments (10 numbers)
R^2 = 1 - SSE / SST over the 100 table entries and the channels (SST around the training mean).

RESULT 11-11 13:10 (novel environments):
    own Z_o -> table_o, shared decoder   Generator route .99 / .99 (oracle), .98 / .98 (RCC); Classifier route -.11 / -.05
                                         (oracle), -.11 / -.05 (original)
    best single channel per table        Classifier route .98 (oracle, both seeds), .96 (original): the SAME quality of
                                         per-channel code, under permuted channel labels: oracle s0 table o <- Z of
                                         [1, 0, 4, 2, 3], s1 [3, 1, 0, 4, 2]; original s1 [2, 1, 0, 4, 3] (one at .74), s0
                                         four channels permuted, channel 0's table in no single channel (still forming:
                                         its Classifier is at .57). Generator route: identity in 4 of 4.
    all 10 numbers -> table_o            .63-.64 every run (true Z: .64-.65 = the decoder's ceiling at n = 5,000), one
                                         .47 (original Classifier route s0): the information is there in every route
    PR(Z) ~ 10 in every run
Reading: the Classifier reads all channels at once, so relabeling the model's channels is a symmetry for it; the
Generator predicts channel o from Z_o with one map, which pins each code to its own channel. Without the Generator's
gradient the embeddings encode each channel's likelihood as well as with it, but in the wrong slots: an inference code,
not a generative model; the Generator trained on them stays at its starting plateau (gen_loss .356, plug-in .13) for
all 60,000 episodes.

11-11 13:45, learned Z against the true Z (fig13_z_vs_truth.png; novel environments, codes matched to their channel):
    distribution   SD .182 / .191 / .184 / .189 (gen / clf / rcc / rccclf) vs true .181; tails |Z| > .5 .004-.006 vs .004
    Generator route   learned Z = the true Z with its two numbers swapped and sign-flipped: r = -.999 (oracle), -.998
                      (RCC); swapping the two numbers (with the table transposed) leaves the likelihood exactly unchanged
    Classifier route  (13:45 reading "NOT a copy, a strongly nonlinear re-encoding" was WRONG: an artifact of pooling the
                      channels.) Per channel it IS an exact linear copy (R^2 .996), learned = true x A_o with A_o a sign
                      flip and / or swap, but a DIFFERENT A_o per channel (oracle s0: -id, +id, -swap, -id, +id; original
                      s1: -swap, -id, +swap, -id, -swap), while the Generator route uses -swap in every channel. Pooled over
                      channels the frames cancel (linear R^2 .04). A decoder per channel reads the true Z (k-NN .994) and any
                      new table built from it (sin -> cos, reversal 1 - p, offset sin + 1, sin 2x: .94-.98); one decoder
                      shared by the channels does not (.43-.47). The table is unchanged by (Z1, Z2) -> (Z1 +- 1/2, Z2 +- 1/2)
                      (.0002) but not by a sign flip (.96).

    python fig13_embeddings.py [--figure | --z]
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION", "trainings")]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
OUT = os.path.join(ROOT, "main", "DATA", "experimental", "fig13_embeddings")
NETS = "/experimental/fig13_embeddings/nets/cluster_runs"
RUNS = [f"{arm}_lr4_s{k}" for arm in ("gen60k", "clf60k", "rcc60k", "rccclf60k") for k in (0, 1)]
B = 5000


def harvest(name):
    import torch
    from main.CognitiveGridworld import CognitiveGridworld
    import train_cluster_runs as tcr
    cfg = dict(next(c for n, c, s in tcr.TASKS if n == name))
    seed = int(name.rsplit("_s", 1)[1])
    np.random.seed(seed); torch.manual_seed(seed)                       # the run's seed rebuilds its variable split
    cfg.update(batch_num=B, episodes=1, training=False, checkpoint_every=1, show_plots=False, cuda=0, load_env=f"{NETS}/{name}")
    env = CognitiveGridworld(**cfg)
    m = env.model
    out = {}
    for split, test in (("train", False), ("novel", True)):
        env.test_set = test
        env.run_generators()
        env.prep_model()
        with torch.no_grad():
            m.MC_to_interactions()
        out[split] = dict(z=m.active_Z.detach().float().cpu().numpy(),             # (B, O, 2) learned
                          ztrue=np.asarray(env.joint_Z, np.float32),                # (B, O, 2) true
                          table=np.asarray(env.joint_likelihood, np.float32))      # (B, O, R, R)
    return out


def knn_r2(Xtr, Ytr, Xte, Yte, k=10):
    from sklearn.neighbors import KNeighborsRegressor
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    f = KNeighborsRegressor(k, n_jobs=8).fit((Xtr - mu) / sd, Ytr)
    P = f.predict((Xte - mu) / sd)
    return 1 - ((P - Yte) ** 2).sum() / ((Yte - Ytr.mean(0)) ** 2).sum()


def measures(H):
    tr, te = H["train"], H["novel"]
    O = tr["z"].shape[1]
    tab = lambda d: d["table"].reshape(len(d["table"]), O, -1)                    # (B, O, 100)
    own = knn_r2(tr["z"].reshape(-1, tr["z"].shape[-1]), tab(tr).reshape(-1, tab(tr).shape[-1]),
                 te["z"].reshape(-1, te["z"].shape[-1]), tab(te).reshape(-1, tab(te).shape[-1]))
    allz = np.mean([knn_r2(tr["z"].reshape(B, -1), tab(tr)[:, o], te["z"].reshape(B, -1), tab(te)[:, o]) for o in range(O)])
    true_own = knn_r2(tr["ztrue"].reshape(-1, 2), tab(tr).reshape(-1, tab(tr).shape[-1]),
                      te["ztrue"].reshape(-1, 2), tab(te).reshape(-1, tab(te).shape[-1]))
    true_all = np.mean([knn_r2(tr["ztrue"].reshape(B, -1), tab(tr)[:, o], te["ztrue"].reshape(B, -1), tab(te)[:, o])
                        for o in range(O)])
    # channel o' -> table o with a decoder per (o, o') pair (not shared): swapped channels show off the diagonal,
    # a per-channel but channel-specific code on it, a code spread over channels nowhere
    cross = np.array([[knn_r2(tr["z"][:, q], tab(tr)[:, o], te["z"][:, q], tab(te)[:, o]) for q in range(O)]
                      for o in range(O)])
    ev = np.linalg.eigvalsh(np.cov(tr["z"].reshape(B, -1).T))
    return dict(own=float(own), all=float(allz), true_own=float(true_own), true_all=float(true_all),
                own_specific=float(np.mean(np.diag(cross))), best_other=float(np.max(cross - np.diag(np.full(O, np.inf)))),
                pr=float(ev.sum() ** 2 / (ev ** 2).sum()), z_sd=float(tr["z"].std()), cross=cross.round(3).tolist())


def main():
    import torch
    torch.set_num_threads(8)
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for name in RUNS:
        res[name] = measures(harvest(name))
        print(name, {k: (round(v, 3) if not isinstance(v, list) else None) for k, v in res[name].items()}, flush=True)
    json.dump(res, open(os.path.join(OUT, "measures.json"), "w"), indent=1)




def figure():
    """fig13_embeddings.png: per run, the 5 x 5 matrix R^2(Z of channel o' -> true table of channel o), novel
    environments; a channel-aligned code is bright on the diagonal"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    R = json.load(open(os.path.join(OUT, "measures.json")))
    rows = (("Oracle feedback", ("gen60k", "Generator route"), ("clf60k", "Classifier route")),
            ("Original feedback", ("rcc60k", "Generator route (the RCC)"), ("rccclf60k", "Classifier route")))
    fig, ax = plt.subplots(2, 4, figsize=(10.4, 5.6), constrained_layout=True)
    for i, (feedback, *arms) in enumerate(rows):
        for j, (arm, name) in enumerate(arms):
            for k in (0, 1):
                a = ax[i, 2 * j + k]
                C = np.clip(np.array(R[f"{arm}_lr4_s{k}"]["cross"]), 0, 1)
                im = a.imshow(C, cmap="viridis", vmin=0, vmax=1)
                for (y, x), v in np.ndenumerate(C):
                    if v > .5:
                        a.text(x, y, f"{v:.2f}"[1:], ha="center", va="center", fontsize=fs["legend"] - 3, color="k")
                a.set_xticks(range(5))
                a.set_yticks(range(5))
                a.tick_params(length=0, labelsize=fs["legend"] - 3)
                a.set_title(f"{name}, seed {k}", fontsize=fs["legend"] - 2)
                if j == 0 and k == 0:
                    a.set_ylabel(f"{feedback}\ntrue table of channel", fontsize=fs["legend"] - 1)
                if i == 1:
                    a.set_xlabel("learned Z of channel", fontsize=fs["legend"] - 2)
    cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.02)
    cb.set_label(r"$R^2$, novel environments (bright = the channel's Z determines that table)", fontsize=fs["legend"] - 2)
    fig.suptitle("Which learned channel code carries which channel's likelihood? k-NN decoder (k = 10) from one channel's "
                 "learned $Z$ (2 numbers)\nto one channel's true table $P(o = 1\\,|\\,r_g, r_c)$ (100 numbers), fit on "
                 "5,000 training-environment episodes, scored on 5,000 novel ones; 60,000 episodes, lr $10^{-4}$",
                 fontsize=fs["legend"] - 2)
    out = fig_path("fig13_embeddings.png")
    fig.savefig(out, dpi=200)
    print(out)




def frame_name(A):
    """name of the 2 x 2 signed permutation nearest to A (learned = true @ A)"""
    S = np.array([[0., 1.], [1., 0.]])
    names = {"+identity": np.eye(2), "-identity": -np.eye(2), "+swap": S, "-swap": -S}
    return min(names, key=lambda k: np.abs(A / np.abs(A).max() - names[k]).sum())


def z_figure():
    """fig13_z_vs_truth.png (John 11-11: "a colorcoded Z_1 (x-axis) Z_2 (y-axis) for the True, Generator learned and
    classifier learned"): rows true Z / Generator-route code / Classifier-route code (oracle feedback, seed 0), one column
    per observation channel; each point = one novel environment, coloured by its TRUE Z of that channel (hue = angle,
    saturation = distance from 0), the same colour in all three rows. A learned row that repeats the true row's colour
    wheel, rotated or mirrored, is a linear copy of the truth; the title names the copy (learned = true x A) and the
    model slot the code sits in"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    from main.utils import fig_path, apply_fig_style
    fs = apply_fig_style()
    R = json.load(open(os.path.join(OUT, "measures.json")))
    rows = (("true $Z$", None), ("Generator route", "gen60k_lr4_s0"), ("Classifier route", "clf60k_lr4_s0"))
    D = {n: np.load(os.path.join(OUT, f"z_{n}.npz")) for _, n in rows if n}
    T = D["gen60k_lr4_s0"]["novel_ztrue"]                                    # same environments in every file? no: per run
    fig, ax = plt.subplots(3, 5, figsize=(12.5, 7.6), constrained_layout=True)
    rng = np.random.default_rng(0)
    for i, (lab, name) in enumerate(rows):
        src = name or "gen60k_lr4_s0"
        Tz = D[src]["novel_ztrue"]                                           # (B, O, 2) true, this run's episodes
        perm = np.arange(5) if name is None else np.array(R[name]["cross"]).argmax(1)
        Lz = Tz if name is None else D[name]["novel_z"][:, perm]
        idx = rng.choice(len(Tz), 3000, replace=False)
        for o in range(5):
            a = ax[i, o]
            t, l = Tz[idx, o], Lz[idx, o]
            hue = (np.arctan2(t[:, 1], t[:, 0]) / (2 * np.pi)) % 1
            sat = np.clip(np.hypot(t[:, 0], t[:, 1]) / .35, 0, 1)
            col = hsv_to_rgb(np.c_[hue, sat, np.full(len(t), .92)])
            a.scatter(l[:, 0], l[:, 1], c=col, s=4, lw=0)
            a.set_xlim(-.7, .7); a.set_ylim(-.7, .7); a.set_aspect("equal")
            a.axhline(0, color="0.8", lw=.5, zorder=0); a.axvline(0, color="0.8", lw=.5, zorder=0)
            a.tick_params(labelsize=fs["legend"] - 4)
            if name is None:
                a.set_title(f"channel {o}", fontsize=fs["legend"] - 1)
            else:
                A = np.linalg.lstsq(Tz[:, o], Lz[:, o], rcond=None)[0]
                a.set_title(f"slot {perm[o]}, copy: {frame_name(A)}", fontsize=fs["legend"] - 2)
            if o == 0:
                a.set_ylabel(f"{lab}\nnumber 2", fontsize=fs["legend"] - 1)
            if i == 2:
                a.set_xlabel("number 1", fontsize=fs["legend"] - 2)
    fig.suptitle("Each channel's code against the truth (oracle feedback, seed 0; 3,000 novel environments): colour = the "
                 "environment's TRUE $Z$ of that channel (hue = angle, saturation = size)\nboth routes copy the truth "
                 "exactly (up to a sign and a swap of the two numbers); the Generator route uses the SAME copy in every "
                 "channel and slot, the Classifier route a different one per channel, in permuted slots",
                 fontsize=fs["legend"] - 2)
    out = fig_path("fig13_z_vs_truth.png")
    fig.savefig(out, dpi=170)
    print(out)


if __name__ == "__main__":
    if "--figure" not in sys.argv and "--z" not in sys.argv:
        main()
    figure()
    if os.path.exists(os.path.join(OUT, "z_gen60k_lr4_s0.npz")):
        z_figure()
