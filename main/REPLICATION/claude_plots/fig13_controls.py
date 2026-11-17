"""Is the Generator route's code (one sign / swap for every channel, each channel's code in its own slot) made by the
Generator's weight sharing over channels, or by learning to generate? (John 11-11 19:00: "If there is a real meaningful
difference there it should not depend on the specific architecture"; "please run the cheap controls quickly")

Embeddings learned FROM SCRATCH (random at the start) in the |S| 500 world of Fig 13's networks:
    Generator, shared        trainings/fresh_generator.py scratch_shared: the fixed Generator as built (one Z input
                             layer for every channel), trained on the true values (batch 2,000, lr 1e-3, 8,000 steps)
    Generator, per channel   scratch_unshared: the same with one Z input layer per channel
    classifier, shared       trainings/transfer_reader.py shared + random + updated (_xl): the channel-wise evidence reader
                             (one function for every channel), classification loss, 80,000 steps of 512
    classifier, per channel  the same reader with a separate function per channel
For each, on a novel batch: which learned slot carries which observation channel's code (linear fit of the slot's 2
numbers from the channel's true 2 numbers, R^2), the frame of that copy (nearest sign / swap), and whether all channels
share one frame and sit in their own slots.

11-11 20:15 RESULT: all 8 runs stayed at chance (novel accuracy .09-.13, chance .10; best-slot code R^2 <= .02): no code
was learned from scratch by these trainers, so they do NOT test weight sharing. The from-scratch trainer that works is
trainings/train_bayes_generator.py (batch 10,000, with reg_loss).

11-11 21:50 rerun with that trainer (cluster 66670828; John: "go ahead"): the Generator alone on the true values, from
scratch, |S| 500, 3,000 episodes, seeds 0-1, its Z input layer
    shared                      as built (one layer for every channel)
    per channel, copied start   one layer per channel, each starting as the shared layer's start (Z-flow alignment)
    per channel, random start   one layer per channel, independent default starts
    shared, random start        (cluster 66670946, 2,000 episodes) the shared layer with a default start: a common start
                                without the Z-flow alignment
EARLY READ 11-11 22:30 (episode 1,000; final at 3,000 pending):
    shared                 one frame in place in both seeds (s0 -swap, s1 -id; R^2 .99), novel .65 / .66
    per channel, copied    the SAME frame as shared in each seed (s0 -swap, s1 -id), in place, R^2 .99, novel .65 / .69:
                           sharing during training is not needed; a common start is enough
    per channel, random    channels learned one at a time (2 of 5 per seed by 1,000), novel .23-.24 (left chance ~500 vs
                           ~350, then stuck near .24); learned channels' frames differ (s1: -swap vs +swap; s0: +id, +id)
    -> one frame across channels comes from reading every channel with one function (shared, or identical at the
       start), not from generating; the frame itself differs between seeds (symmetry broken per seed).
UPDATE 11-11 23:20 (2,250-2,750 iterations): shared and copied-start runs .69-.71, one frame, R^2 .99; per channel,
random start STUCK at .23-.24 from 750 to 2,250 with the same 2 of 5 channels learned (R^2 .99) and 3 unlearned
(R^2 < .03): a trap, not slowness. Shared layer, default (unaligned) start (66670946, 1,000): seed 1 as fast as the aligned
start (.67, one frame -swap), seed 0 at chance throughout (.10, no code). -> the aligned common start makes every channel
learn, together and in one frame; without it a channel's code can be left out for good.

    python fig13_controls.py [--pull]
"""
import os, sys, json, glob, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.dirname(os.path.abspath(__file__))]
OUT = os.path.join(ROOT, "main", "DATA", "experimental", "fig13_embeddings", "controls")
RUNS = (("Generator, shared", "bayes_gen", "truth_S500_d30_s{k}_shared"),
        ("Generator, per channel, copied start", "bayes_gen", "truth_S500_d30_s{k}_per_channel"),
        ("Generator, per channel, random start", "bayes_gen", "truth_S500_d30_s{k}_per_channel_random"),
        ("Generator, shared, random start", "bayes_gen", "truth_S500_d30_s{k}_shared_random"),
        # the first controls (11-11 19:00; inconclusive: nothing learned)
        ("old trainer: Generator, shared", "fresh_generator", "scratch_shared_s{k}"),
        ("old trainer: Generator, per channel", "fresh_generator", "scratch_unshared_s{k}"),
        ("old trainer: classifier, shared", "transfer_reader", "shared_random_updated_s{k}_xl"),
        ("old trainer: classifier, per channel", "transfer_reader", "unshared_random_updated_s{k}_xl"))


def pull():
    os.makedirs(OUT, exist_ok=True)
    for _, d, pat in RUNS:
        for k in (0, 1):
            for ext in ("_codes.npz", ".jsonl"):
                subprocess.run(f"scp -q -o BatchMode=yes 'elsc-cluster:cgw/runs/{d}/{pat.format(k=k)}{ext}' {OUT}/ 2>/dev/null",
                               shell=True)


def frame_name(A):
    S = np.array([[0., 1.], [1., 0.]])
    names = {"+id": np.eye(2), "-id": -np.eye(2), "+swap": S, "-swap": -S, "+-": np.diag([1., -1.]), "-+": np.diag([-1., 1.]),
             "+-swap": S @ np.diag([1., -1.]), "-+swap": S @ np.diag([-1., 1.])}
    return min(names, key=lambda k: np.abs(A / np.abs(A).max() - names[k]).sum())


def analyse(z, zt):
    """z, zt (B, O, 2): learned and true codes -> per channel: slot carrying it, R^2 of the linear copy, frame"""
    B, O, _ = z.shape
    R2 = np.zeros((O, O))                                # [true channel o, learned slot q]
    fits = {}
    for o in range(O):
        X = np.c_[zt[:, o], np.ones(B)]
        for q in range(O):
            W, *_ = np.linalg.lstsq(X, z[:, q], rcond=None)
            res = z[:, q] - X @ W
            R2[o, q] = 1 - (res ** 2).sum() / ((z[:, q] - z[:, q].mean(0)) ** 2).sum()
            fits[o, q] = W[:2]
    slot = R2.argmax(1)
    frames = [frame_name(fits[o, slot[o]]) for o in range(O)]
    return slot, R2[np.arange(O), slot], frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    if ap.parse_args().pull:
        pull()
    for lab, d, pat in RUNS:
        for k in (0, 1):
            f = os.path.join(OUT, pat.format(k=k) + "_codes.npz")
            log = os.path.join(OUT, pat.format(k=k) + ".jsonl")
            acc = ""
            if os.path.exists(log):
                r = [json.loads(l) for l in open(log) if l.strip()]
                key = "novel_plugin" if "novel_plugin" in r[-1] else "novel"
                acc = f"novel acc {r[-1][key]:.3f} at step {r[-1].get('step', r[-1].get('ep'))}"
            if not os.path.exists(f):
                print(f"{lab:38s} s{k}: no codes yet  {acc}")
                continue
            D = np.load(f)
            slot, r2, frames = analyse(D["z"], D["ztrue"])
            in_place = bool((slot == np.arange(len(slot))).all())
            one_frame = len(set(frames)) == 1
            print(f"{lab:38s} s{k}: {acc} | slots {slot.tolist()} (in place: {in_place}) | R^2 {np.round(r2, 3).tolist()} | "
                  f"frames {frames} (one frame: {one_frame})")


if __name__ == "__main__":
    main()
