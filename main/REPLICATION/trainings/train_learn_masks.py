"""Causal tests of learnability: does learning to go beyond naive need the learning signal at entangled observations?
(John 11-08: "We really want causal account of learnability.")
A trunk (cond 'trunk') trains the standard Trained RNN from scratch (seed 20261029, the original run's settings except
batch 8000 in micro-batches of 2000, model.forward_pass_micro: near-exact at loss power .5, and gpu_inference) to raw
episode 600 (480 training episodes: naive's rule learned, the memory correction not yet, td_summary 'beta') and stores its
weights and its Adam / AMP-scaler state there. Every condition branches from the trunk at 600 with that optimizer state
(a fresh Adam restart dropped the accuracy from .31 to .15 within 3 updates in a smoke test, so it is not used), and
continues for EPISODES episodes (a test episode every 5), all with the same branch seed (the same episode stream: the
selection draws from its own generator).
Selection, per training batch: the goal variable's entanglement at each step t, J/2 between the normalised naive and
Bayes goal increments at t (log B_t - log B_t-1, B_0 uniform; td_analysis' 'ent' for the goal variable), ranked over the
batch within each step; 'hi' = the top FRAC of each step, 'lo' = the bottom FRAC, 'rand' = a random FRAC of each step.
  ctrl     no selection
  mask_*   the selected (episode, step) terms dropped from SANITY_loss (the mean over the rest); later steps' terms
           still reach the update made at a selected step (beliefs are cumulative)
  det_*    the read-out increments at the selected (episode, step) detached before the cumulative sum: no loss term's
           gradient reaches the network through the update made at a selected step (the state still carries history
           through it, and shared weights still change through the other steps)
Outputs: main/DATA/experimental/training_dynamics/learnmask_<cond>_s20261029/: ckpt_ep<EEEEE>_net.pth (save() format,
raw episodes continuing from 600, so td_tt_intervene.load reads them; every CK_EVERY), tests.npz (every test episode:
accuracy per step of the network, Bayes and naive), sel.npz (per training episode: the selected share of the loss's
weight and the mean entanglement of the selected and unselected terms), progress.log.
Run from the repository root:
    /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/trainings/train_learn_masks.py <cond> <gpu> [episodes] [frac]
frac other than FRAC (John 11-08: "more extreme most / least % entangled steps") goes into the run name:
learnmask_<cond>_f<percent>_s20261029 (e.g. det_hi at .5 -> learnmask_det_hi_f50_s20261029). Every step keeps the same
selected share (k = round(frac B) episodes at each t), so the selection never favours late, more entangled steps.
sel 'fr' (John 11-09: "Maybe selecting by Factorization Regret would be more impactful?") ranks by the goal belief's
factorization regret at each step, KL(Bayes_t || naive_t), instead of the step's entanglement; scratch = 1 (John 11-09:
"maybe starting the blocking from episode 0") trains from initialization (seed 20261029, the trunk's episode stream) with
the selection active from the first episode: run name learnmask_<cond>[_fr][_f<percent>]_s0_s20261029.
    ... train_learn_masks.py <cond> <gpu> [episodes] [frac] [sel: ent | fr] [scratch: 0 | 1]
Every run stores its final Adam / scaler state (adam_end.pth) so that it can be extended.
rep > 1 (John 11-09: replicate seeds) shifts the seed by 1000 (rep - 1): a new initialization and episode stream for
scratch runs, a new episode stream for branches; run name suffix _r<rep>.
    ... train_learn_masks.py <cond> <gpu> [episodes] [frac] [sel] [scratch] [rep]
thr (John 11-10: "blocking learning when FR is above .1 ... below .1 ... With balanced number of blocked steps"): an
absolute threshold on the goal belief's factorization regret (nats, sel 'fr') in place of the within-step ranking. At
each step t, with a(t) of the batch's B updates above thr, every condition blocks m(t) = max(a(t), B - a(t)) updates:
'hi' = every update above thr (plus random ones below where those outnumber the ones above: the first step), 'lo' =
every update below thr plus random ones above, 'rand' = m(t) random ones. So every condition learns from the same number
of updates at every step, min(a(t), B - a(t)), all of them on its own side of the threshold (hi: below, lo: above).
  excl_*   det_* and mask_* together: the selected states are held out of training (their loss terms dropped and their
           increments detached: no gradient comes from, or passes through, the update of a held-out state; the
           recurrent state still runs through it)
Run name learnmask_<cond>_thr<100 thr>_s0_s20261029 (also for rand: its count follows the threshold); sel.npz then
holds per training episode [blocked share, mean FR blocked, mean FR trained, share above thr, share above thr among
the trained updates] and 'steps' (episodes, 2, T) = the blocked share and the share above thr at every step.
    ... train_learn_masks.py <cond> <gpu> [episodes] [frac] [sel] [scratch] [rep] [thr] [cont]
cont (11-11): continue the run from its own ckpt_ep<cont> and adam_end.pth (which must be at cont) for [episodes] more
raw episodes, with a new seed (a fresh episode stream); tests.npz, sel.npz, run.json, adam_end.pth and the checkpoint
at cont are first copied to *_to<cont>.*; checkpoints and progress.log continue in the same folder.
Equal FR chunks (John 11-10 13:58: "Does learn_holdout control for the amount of training data and weight updates?
Everything should be balanced ... order the panels so the columns have an equal chunk of FR (i.e. equal size of green
chunk). Also adding theta = 10 could be good"): excl_bandlo / excl_bandhi / excl_bandrand with thr "lo:hi" learn ONLY
n(t) updates at step t, n(t) = min(#(FR < .01), #(FR > 10)) of the batch (CNT: one count for every pair, so every run
learns the same number of updates at every step of every batch); bandlo picks them at random among the updates with
FR < lo, bandhi among FR > hi, bandrand among all. lo * hi ~ .1: the update FR distribution is nearly symmetric in log
around .3 nats (below .01 / above 10: 4.4% / 3.9%; below .1 / above 1: 29% / 24%), so the chunks have equal log width
and nearly equal mass. Run name learnmask_<cond>_b<100 lo>_<100 hi>_s0_s20261029.
Own counts (John 11-10 16:20, after the (.01 | 10) count left both networks at chance for 4,000 episodes: nothing is
learned before step ~6, so the recurrent weights get almost no gradient): thr "lo:hi:clo:chi" counts n(t) = min(#FR <
clo, #FR > chi) instead of CNT; with clo:chi = lo:hi every column learns its own per-step count (.03 | 3: 8.1%, from
step ~3; .1 | 1: 21%, from step 2, the theta-1 runs' profile). Run name ..._b<lo>_<hi>_c<clo>_<chi>_s0_s20261029."""
import os
import sys
import json
import time
import types
import random
from contextlib import contextmanager
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main")]
from main.CognitiveGridworld import CognitiveGridworld                           # noqa: E402
from main.utils import tnp                                                       # noqa: E402

SEED, START, FRAC, CK_EVERY, MICRO, BRANCH_SEED = 20261029, 600, .2, 100, 2000, 20261108
TD = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
CONDS = ("trunk", "ctrl", "mask_hi", "mask_lo", "mask_rand", "det_hi", "det_lo", "det_rand",
         "excl_hi", "excl_lo", "excl_rand", "excl_bandlo", "excl_bandhi", "excl_bandrand")
CNT = (.01, 10.)                                     # band modes: n(t) = min(#(FR < .01), #(FR > 10)) at every step


def goal_fr(jb, nb):
    """factorization regret of the goal belief at every step, KL(Bayes_t || naive_t): (B, T) from goal beliefs (B, T, R)"""
    lj, ln = (torch.log_softmax(torch.log(x.clamp_min(1e-30)), -1) for x in (jb, nb))
    return (lj.exp() * (lj - ln)).sum(-1)


def goal_entanglement(jb, nb):
    """J/2 between the normalised naive and Bayes goal increments at every step: (B, T) from goal beliefs (B, T, R)"""
    lj, ln = (torch.log(x.clamp_min(1e-30)) for x in (jb, nb))
    R = jb.shape[-1]
    start = torch.full_like(lj[:, :1], -np.log(R))
    ij = torch.log_softmax(lj - torch.cat([start, lj[:, :-1]], 1), -1)
    i_n = torch.log_softmax(ln - torch.cat([start, ln[:, :-1]], 1), -1)
    return .5 * ((ij.exp() - i_n.exp()) * (ij - i_n)).sum(-1)


def threshold_selection(score, thr, kind, gen):
    """the balanced blocks at an absolute threshold: score (B, T) -> blocked (B, T) bool, above (B, T) bool. Every kind
    blocks m(t) = max(a(t), B - a(t)) updates at step t (a(t) = the number above thr): 'hi' every update above thr, 'lo'
    every update below, each filled up to m(t) with random updates of the other side; 'rand' m(t) random updates"""
    B, T = score.shape
    above = score > thr
    n_a = above.sum(0)
    m_t = torch.maximum(n_a, B - n_a)
    prio = torch.rand(B, T, generator=gen).to(score.device)                      # a random order within each side
    if kind == "hi":
        prio = prio + above
    elif kind == "lo":
        prio = prio + ~above
    rank = prio.argsort(0, descending=True).argsort(0)                           # 0 = blocked first
    return rank < m_t[None], above


def band_selection(score, lo, hi, kind, gen, cnt=CNT):
    """equal FR chunks: score (B, T) -> blocked (B, T) bool, above (B, T) bool (FR > hi). Every kind learns exactly
    n(t) = min(#(score < CNT[0]), #(score > CNT[1])) updates at step t: 'bandlo' random ones with score < lo, 'bandhi'
    random ones with score > hi, 'bandrand' random ones; everything else is blocked"""
    B, T = score.shape
    below, above = score < lo, score > hi
    n_t = torch.minimum((score < cnt[0]).sum(0), (score > cnt[1]).sum(0))
    pool = {"bandlo": below, "bandhi": above, "bandrand": torch.ones_like(below)}[kind]
    assert bool((pool.sum(0) >= n_t).all()), "the chunk holds fewer updates than the count"
    prio = torch.rand(B, T, generator=gen).to(score.device) + pool                # pool first, random within it
    rank = prio.argsort(0, descending=True).argsort(0)                            # 0 = learned first
    return ~(rank < n_t[None]), above


class MaskAgent(CognitiveGridworld):
    cond, frac, adam_path, sel, thr = "ctrl", FRAC, None, "ent", None

    def run_inference(self):
        super().run_inference()
        if not self.test_set and self.cond not in ("ctrl", "trunk"):             # naive is computed on test episodes only
            out = self.forward_inference(self.naive_likelihood, naive=True)
            self._naive_goal_train = out[2]

    def prep_model(self):
        super().prep_model()
        m = self.model
        if not getattr(m, "_mask_patched", False):
            _patch(m, self.cond)
            if self.adam_path is not None:                                       # the trunk's optimizer state at START
                st = torch.load(self.adam_path, map_location=m.device if hasattr(m, "device") else self.device,
                                weights_only=False)
                m.classifier_optim.load_state_dict(st["optim"])
                m.scaler.load_state_dict(st["scaler"])
        m._sel_full = m._sel = None
        if self.test_set or self.cond in ("ctrl", "trunk"):
            return
        dev = m.device if hasattr(m, "device") else self.device
        jb = torch.as_tensor(np.asarray(self.joint_goal_belief), device=dev, dtype=torch.float32)
        nb = torch.as_tensor(np.asarray(self._naive_goal_train), device=dev, dtype=torch.float32)
        ent = goal_fr(jb, nb) if self.sel == "fr" else goal_entanglement(jb, nb)  # (B, T): the ranking score
        B, T = ent.shape
        kind = self.cond.split("_")[1]
        if self.thr is not None and kind.startswith("band"):
            lo, hi = self.thr[:2]
            cnt = tuple(self.thr[2:]) if len(self.thr) == 4 else CNT
            sel, above = band_selection(ent, lo, hi, kind, self._gen, cnt)
            m._sel_full = m._sel = sel
            if kind != "bandrand":                                               # every learned update in its chunk
                assert not bool((~sel & ~((ent < lo) if kind == "bandlo" else (ent > hi))).any())
            if not self._sel_log:
                print(f"selection {self.cond} at FR < {lo} / > {hi}: learned {1 - float(sel.float().mean()):.4f} of "
                      f"{B} x {T} updates; per step learned {np.round(tnp((~sel).float().mean(0), 'np'), 3).tolist()}",
                      flush=True)
            self._sel_log.append([float(sel.float().mean()), float(ent[sel].mean()), float(ent[~sel].mean()),
                                  float(above.float().mean()), float(above[~sel].float().mean())])
            self._step_log.append(np.stack([tnp(sel.float().mean(0), "np"), tnp(above.float().mean(0), "np")]))
            return
        if self.thr is not None:
            sel, above = threshold_selection(ent, self.thr, kind, self._gen)
            m._sel_full = m._sel = sel
            if kind != "rand":                                                   # every trained update on its own side
                assert not bool(((above if kind == "hi" else ~above) & ~sel).any())
            if not self._sel_log:
                print(f"selection {self.cond} at FR {self.thr}: blocked {float(sel.float().mean()):.4f} of {B} x {T} "
                      f"updates, above {float(above.float().mean()):.4f}; per step blocked "
                      f"{np.round(tnp(sel.float().mean(0), 'np'), 3).tolist()}", flush=True)
            self._sel_log.append([float(sel.float().mean()), float(ent[sel].mean()), float(ent[~sel].mean()),
                                  float(above.float().mean()), float(above[~sel].float().mean())])
            self._step_log.append(np.stack([tnp(sel.float().mean(0), "np"), tnp(above.float().mean(0), "np")]))
            return
        k = int(round(self.frac * B))
        sel = torch.zeros(B, T, dtype=torch.bool, device=dev)
        for t in range(T):
            if kind == "hi":
                idx = torch.topk(ent[:, t], k).indices
            elif kind == "lo":
                idx = torch.topk(-ent[:, t], k).indices
            else:
                idx = torch.randperm(B, generator=self._gen, device="cpu")[:k].to(dev)
            sel[idx, t] = True
        m._sel_full = m._sel = sel
        if not self._sel_log:
            print(f"selection {self.cond} by {self.sel} frac {self.frac}: {k} of {B} episodes at each of {T} steps; "
                  f"score mean {float(ent.mean()):.4f}", flush=True)
        self._sel_log.append([float(sel.float().mean()), float(ent[sel].mean()), float(ent[~sel].mean())])


def _patch(m, cond):
    """the model's batch slicing carries the selection; mask_*: SANITY_loss drops the selected terms; det_*: the
    selected increments are detached before the cumulative sum; excl_*: both"""
    orig_slice = m.batch_slice

    @contextmanager
    def batch_slice(self_m, sl):
        full = self_m._sel_full
        with orig_slice(sl) as share:
            self_m._sel = None if full is None else full[sl]
            try:
                yield share
            finally:
                self_m._sel = full
    m.batch_slice = types.MethodType(batch_slice, m)
    if cond.startswith(("mask_", "excl_")):
        def sanity_loss(self_m, eps=1e-8, power=.5):
            dkl = self_m.DKL_sym(self_m.classifier_goal_belief, self_m.joint_goal_belief, PM=False)
            term = (dkl - dkl.detach().min() + eps) ** power
            self_m.classifier_loss = term.mean() if self_m._sel is None else term[~self_m._sel].mean()
        m.SANITY_loss = types.MethodType(sanity_loss, m)
    if cond.startswith(("det_", "excl_")):
        orig_post = m.postprocess_belief

        def postprocess_belief(self_m, logits, accumulate=True):
            if self_m._sel is not None and accumulate:
                logits = torch.where(self_m._sel[..., None], logits.detach(), logits)
            return orig_post(logits, accumulate)
        m.postprocess_belief = types.MethodType(postprocess_belief, m)
    m._mask_patched = True


class Recorder:
    def __init__(self, out, offset):
        self.out, self.offset, self.tests, self.t0 = out, offset, [], time.time()

    def save_ck(self, env, raw):
        m = env.model
        save = {"K": env.all_K, "Q": env.all_Q, "model_K": tnp(m.all_K, "np"), "model_Q": tnp(m.all_Q, "np"),
                "weights": {k: v.detach().cpu() for k, v in m.state_dict().items()}, "episode": raw}
        path = os.path.join(self.out, f"ckpt_ep{raw:05d}_net.pth")
        torch.save(save, path + ".tmp")
        os.replace(path + ".tmp", path)

    def __call__(self, env):
        e = env.e
        raw = self.offset + e
        if self.offset == 0 and raw == START:                                    # the trunk's branch point
            m = env.model
            torch.save({"optim": m.classifier_optim.state_dict(), "scaler": m.scaler.state_dict(), "episode": raw},
                       os.path.join(self.out, f"adam_ep{raw:05d}.pth"))
        if env.test_set:
            self.tests.append({"raw": raw, "acc": np.asarray(env.test_accs[env.test_e - 1], np.float32),
                               "joint": np.asarray(env.joint_acc, np.float64).mean(0).astype(np.float32),
                               "naive": np.asarray(env.naive_acc, np.float64).mean(0).astype(np.float32)})
            if raw % CK_EVERY == 0 or (self.offset == 0 and raw <= 500 and raw % 25 == 0):
                self.save_ck(env, raw)
                self.flush(env)
                with open(os.path.join(self.out, "progress.log"), "a") as f:
                    f.write(f"{time.strftime('%H:%M:%S')} raw {raw} test acc T {float(self.tests[-1]['acc'][-1]):.4f} "
                            f"elapsed {(time.time() - self.t0) / 60:.1f} min\n")
        return {}

    def flush(self, env):
        np.savez(os.path.join(self.out, "tests.npz"), raw=np.array([t["raw"] for t in self.tests]),
                 acc=np.array([t["acc"] for t in self.tests]), joint=np.array([t["joint"] for t in self.tests]),
                 naive=np.array([t["naive"] for t in self.tests]))
        if getattr(env, "_sel_log", None):
            steps = {"steps": np.array(env._step_log, np.float32)} if getattr(env, "_step_log", None) else {}
            np.savez(os.path.join(self.out, "sel.npz"), sel=np.array(env._sel_log), **steps)


def main(cond, gpu, episodes=1001, frac=FRAC, sel="ent", scratch=False, rep=1, thr=None, cont=None):
    assert cond in CONDS and sel in ("ent", "fr"), (cond, sel)
    blocked = cond.startswith(("det_", "mask_", "excl_"))
    assert thr is None or (blocked and sel == "fr"), "thr: a factorization-regret threshold of a blocked condition"
    tag = cond
    if cond.startswith("excl_band"):
        assert isinstance(thr, tuple) and sel == "fr", "band modes take thr = (lo, hi)"
        tag += f"_b{round(thr[0] * 100):02d}_{round(thr[1] * 100):02d}"
        if len(thr) == 4:
            tag += f"_c{round(thr[2] * 100):02d}_{round(thr[3] * 100):02d}"
    elif blocked and thr is not None:
        tag += f"_thr{round(thr * 100):02d}"
    elif blocked:
        if sel != "ent" and not cond.endswith("_rand"):
            tag += f"_{sel}"
        if frac != FRAC:
            tag += f"_f{round(frac * 100):02d}"
    if scratch:
        tag += "_s0"
    if rep > 1:
        tag += f"_r{rep}"
    run = f"learnmask_{tag}_s{SEED}"
    out = os.path.join(TD, run)
    os.makedirs(out, exist_ok=True)
    trunk = os.path.join(TD, f"learnmask_trunk_s{SEED}")
    cfg = {"mode": "SANITY", "cuda": gpu, "episodes": episodes, "checkpoint_every": 5, "realization_num": 10,
           "hid_dim": 1000, "obs_num": 5, "show_plots": False, "training": True, "batch_num": 8000, "step_num": 30,
           "state_num": 500, "learn_embeddings": False, "classifier_LR": .001, "ctx_num": 2,
           "skip_training_analyses": True, "reservoir": False, "gpu_inference": True, "micro_batch": MICRO}
    if cond == "trunk":
        cfg["episodes"], offset, seed, adam = START + 1, 0, SEED, None
    elif cont is not None:                                   # continue this run from its own checkpoint at raw cont
        import shutil                                        # (John 11-11: "train the hold_out movie agents for more
        ck, adam = os.path.join(out, f"ckpt_ep{cont:05d}"), os.path.join(out, "adam_end.pth")   # episodes to confirm
        assert os.path.exists(ck + "_net.pth"), ck           # that the learning has saturated"): its weights, Adam
        adam_ep = torch.load(adam, weights_only=False)["episode"]
        if adam_ep != cont:                                  # a run stopped by hand saves no Adam state: with
            assert os.environ.get("LEARNMASK_STALE_ADAM") == "1", f"adam_end.pth is at {adam_ep}, not {cont}"
            print(f"continuing from {cont} with the Adam state of episode {adam_ep} (LEARNMASK_STALE_ADAM)", flush=True)
        d_ = torch.load(ck + "_net.pth", weights_only=False)
        if any(k in d_["weights"] for k in ("all_K", "all_Q")):   # a checkpoint saved by a continued run: load()
            d_["weights"] = {k: v for k, v in d_["weights"].items() if k not in ("all_K", "all_Q")}   # registered the
            ck = ck + "_loadable"                            # embeddings as parameters (equal to model_K / model_Q,
            torch.save(d_, ck + "_net.pth")                  # which load() sets anyway); a fresh model has no slot
        cfg["load_env"] = os.path.relpath(ck, os.path.join(ROOT, "main", "DATA"))   # and scaler state, a fresh
        offset, seed = cont, SEED + 1000 * (rep - 1) + cont  # episode stream (new seed); the first part's logs kept
        for f in ("tests.npz", "sel.npz", "run.json", "adam_end.pth", f"ckpt_ep{cont:05d}_net.pth"):
            src = os.path.join(out, f)
            dst = os.path.join(out, f"{os.path.splitext(f)[0]}_to{cont:05d}{os.path.splitext(f)[1]}")
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
        adam = adam[:-4] + f"_to{cont:05d}.pth"
    elif scratch:
        offset, seed, adam = 0, SEED + 1000 * (rep - 1), None
    else:
        cfg["load_env"] = os.path.relpath(os.path.join(trunk, f"ckpt_ep{START:05d}"), os.path.join(ROOT, "main", "DATA"))
        offset, seed, adam = START, BRANCH_SEED + 1000 * (rep - 1), os.path.join(trunk, f"adam_ep{START:05d}.pth")
        assert os.path.exists(adam), "run the trunk first"
    json.dump({"run": run, "cond": cond, "frac": frac, "sel": sel, "thr": thr, "scratch": bool(scratch), "rep": rep,
               "start": cont if cont is not None else 0 if scratch else START, "seed": seed, "cfg": cfg,
               "continued_from": cont, "adam_state_episode": (adam_ep if cont is not None else None),
               "started": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(out, "run.json"), "w"), indent=1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    Agent = type("Agent", (MaskAgent,), {"cond": cond, "adam_path": adam, "frac": frac, "sel": sel, "thr": thr})
    Agent._gen = torch.Generator().manual_seed(seed + 7)
    Agent._sel_log, Agent._step_log = [], []
    rec = Recorder(out, offset)
    agent = Agent(**cfg, external_logger=rec)
    rec.flush(agent)
    m = agent.model
    torch.save({"optim": m.classifier_optim.state_dict(), "scaler": m.scaler.state_dict(),
                "episode": offset + cfg["episodes"] - 1}, os.path.join(out, "adam_end.pth"))
    with open(os.path.join(out, "progress.log"), "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} DONE\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), int(a[2]) if len(a) > 2 else 1001, float(a[3]) if len(a) > 3 else FRAC,
         a[4] if len(a) > 4 else "ent", bool(int(a[5])) if len(a) > 5 else False, int(a[6]) if len(a) > 6 else 1,
         (tuple(map(float, a[7].split(":"))) if ":" in a[7] else float(a[7])) if len(a) > 7 else None,
         int(a[8]) if len(a) > 8 else None)
