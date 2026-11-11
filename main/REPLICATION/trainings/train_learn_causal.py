"""Causal pilots of the heat-map contenders during learning (John 11-09: "We still dont have good candidates for what makes
an effect. Can you please explore causal manipulations for every contender in learn_maps."). Every run branches from the
learnmask trunk at raw episode 600 (480 training episodes, before the Trained RNN passes naive at 880) with the trunk's
Adam / scaler state and train_learn_masks' branch seed, so every condition sees the same episode stream; it continues
for EPISODES raw episodes (a test episode every 5, checkpoints every 100) like the learnmask branches (ctrl:
learnmask_ctrl_s20261029).
Selection, per batch and per step: the goal variable's entanglement ('ent') or factorization regret ('fr') at each step
(train_learn_masks.goal_entanglement / goal_fr), ranked over the batch within the step; 'hi' = the top FRAC, 'rand' = a
random FRAC (train_learn_masks' generator), 'all' = every item.
Manipulations at the selected (episode, step) items:
  forward: the state is changed in training AND test episodes (the network is tested under the constraint it learned
  with; the test episodes' selection uses their own Bayes / naive beliefs), and the changed state continues the
  recurrence (the cell state is not changed):
    norm   u_s rescaled to the median ||u|| of its step over the micro-batch: no norm signal at s          [map c]
    turn   u_s turned so that 1 - cos(u_s, u_s-1) = the median turn of its step over the micro-batch, norm kept  [e]
    dim    u_s's deviation from its step's micro-batch mean: the part outside the step's top 10 PCs (randomized PCA
           of the episode's first micro-batch) halved, then the deviation rescaled to its old norm (energy moved into
           the top PCs: lower effective dimension, larger top-band share)                                  [a, d]
  learning signal only (training episodes; the forward pass is unchanged):
    cut    the state handed on from s (h_s, c_s) detached: no credit from later errors flows back through the
           state at s (the read-out path of the cumulative belief is untouched)                            [b]
    boost  the selected loss terms weighted x 3 (x 0 is train_learn_masks' mask_*)                         [f-i]
Outputs: main/DATA/experimental/training_dynamics/learncause_<manip>_<kind>[_<sel>]_f<pct>_s20261029/ as the learnmask
runs (ckpt_ep<EEEEE>_net.pth, tests.npz, sel.npz, progress.log, run.json).
Run from the repository root:
    /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/trainings/train_learn_causal.py <manip> <kind> <sel> \
        <frac> <gpu> [episodes]"""
import os
import sys
import json
import time
import types
import random
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main"), os.path.join(ROOT, "main", "REPLICATION", "trainings")]
import train_learn_masks as lm                                                                # noqa: E402

MANIPS = ("norm", "turn", "dim", "cut", "boost")
FORWARD = ("norm", "turn", "dim")
KINDS = ("hi", "rand", "all")
BOOST, DIM_K, DIM_REST = 3., 10, .5


class CausalAgent(lm.MaskAgent):
    manip, kind = "cut", "hi"

    def run_inference(self):
        super(lm.MaskAgent, self).run_inference()
        if not self.test_set and self.kind == "hi":                                           # naive for the ranking
            self._naive_goal_train = self.forward_inference(self.naive_likelihood, naive=True)[2]

    def prep_model(self):
        super(lm.MaskAgent, self).prep_model()
        m = self.model
        if not getattr(m, "_cause_patched", False):
            lm._patch(m, "ctrl")                                                              # batch slices carry _sel
            _patch(m, self.manip)
            st = torch.load(self.adam_path, map_location=self.device, weights_only=False)
            m.classifier_optim.load_state_dict(st["optim"])
            m.scaler.load_state_dict(st["scaler"])
        m._sel_full = m._sel = None
        m._dim_basis = {}
        if self.test_set and self.manip not in FORWARD:
            return
        dev = self.device
        B, T = self.batch_num, self.step_num
        if self.kind == "all":
            sel = torch.ones(B, T, dtype=torch.bool, device=dev)
        else:
            k = int(round(self.frac * B))
            sel = torch.zeros(B, T, dtype=torch.bool, device=dev)
            if self.kind == "hi":
                nb_ = self.naive_goal_belief if self.test_set else self._naive_goal_train
                jb = torch.as_tensor(np.asarray(self.joint_goal_belief), device=dev, dtype=torch.float32)
                nb = torch.as_tensor(np.asarray(nb_), device=dev, dtype=torch.float32)
                score = lm.goal_fr(jb, nb) if self.sel == "fr" else lm.goal_entanglement(jb, nb)
            for t in range(T):
                if self.kind == "hi":
                    idx = torch.topk(score[:, t], k).indices
                else:
                    idx = torch.randperm(B, generator=self._gen, device="cpu")[:k].to(dev)
                sel[idx, t] = True
        m._sel_full = m._sel = sel
        if not self.test_set:
            self._sel_log.append([float(sel.float().mean())])


@torch.autocast("cuda", enabled=False)
def top_pcs(Y, q, seed, it=2):
    """the top q principal axes (H, q) of the centred rows Y (n, H): randomized range finder with its own generator (the
    global random stream, and so the episode stream, is left untouched)"""
    Y = Y.float()
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    Q = torch.linalg.qr(Y @ torch.randn(Y.shape[1], q + 6, generator=g).to(Y.device)).Q
    for _ in range(it):
        Q = torch.linalg.qr(Y.T @ Q).Q
        Q = torch.linalg.qr(Y @ Q).Q
    return torch.linalg.svd(Q.T @ Y, full_matrices=False)[2][:q].T


def _turn_to(u, up, target):
    """u turned in the plane of (u_prev, u) so that 1 - cos(u, u_prev) = target, norm kept"""
    n = u.norm(dim=-1, keepdim=True)
    a = up / up.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    w = u - (u * a).sum(-1, keepdim=True) * a
    w = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cs = (1 - target).clamp(-1, 1)
    return n * (cs * a + (1 - cs ** 2).clamp_min(0).sqrt() * w)


def _patch(m, manip):
    """step-by-step LSTM (the cuDNN weights, one step at a time) with the forward manipulations / the credit cut at the
    selected items; boost: SANITY_loss weights the selected terms"""
    def rnn_classification(self_m):
        x = torch.relu(self_m.RNN_readin())                                                  # (B, T, H)
        B, T = x.shape[0], x.shape[1]
        h = self_m.STM.expand(-1, B, -1).contiguous()
        c = self_m.LTM.expand(-1, B, -1).contiguous()
        sel = self_m._sel
        outs, up = [], h[0]
        for t in range(T):
            o, (h, c) = self_m.LSTM(x[:, t:t + 1], (h, c))
            u = o[:, 0]
            if sel is not None and manip in FORWARD and t >= 1:
                u = _manipulate(self_m, manip, u, up, sel[:, t][:, None], t)
                h = u[None]
            if sel is not None and manip == "cut":
                s = sel[:, t][None, :, None]
                h, c = torch.where(s, h.detach(), h), torch.where(s, c.detach(), c)
            outs.append(u)
            up = u
        update = torch.stack(outs, 1)
        self_m.postprocess_belief(self_m.classifier_readout(update))
        if self_m.testing and (self_m.mode == "SANITY") and self_m.keep_test_activations():
            self_m.update_flat, self_m.input_flat = update.detach(), x.detach()
        else:
            self_m.update_flat = torch.zeros(1, device=self_m.device)
            self_m.input_flat = torch.zeros(1, device=self_m.device)

    @torch.autocast("cuda", enabled=False)
    def _manipulate(self_m, manip, u, up, s, t):
        uf = u.float()
        if manip == "norm":
            nr = uf.norm(dim=-1, keepdim=True)
            un = uf * (nr.detach().median() / nr.clamp_min(1e-6))
        elif manip == "turn":
            upf = up.float()
            tr = 1 - (uf * upf).sum(-1) / (uf.norm(dim=-1) * upf.norm(dim=-1)).clamp_min(1e-6)
            un = _turn_to(uf, upf, tr.detach().median())
        else:
            mu = uf.detach().mean(0)
            if t not in self_m._dim_basis:                                           # the episode's first
                self_m._dim_basis[t] = top_pcs(uf.detach() - mu, DIM_K, t)           # micro-batch's PCs
            V = self_m._dim_basis[t]
            y = uf - mu
            yt = (y @ V) @ V.T
            yn = yt + DIM_REST * (y - yt)
            un = mu + yn * (y.norm(dim=-1, keepdim=True) / yn.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        return torch.where(s, un.to(u.dtype), u)
    m.RNN_classification = types.MethodType(rnn_classification, m)
    if manip == "boost":
        def sanity_loss(self_m, eps=1e-8, power=.5):
            dkl = self_m.DKL_sym(self_m.classifier_goal_belief, self_m.joint_goal_belief, PM=False)
            term = (dkl - dkl.detach().min() + eps) ** power
            if self_m._sel is not None:
                w = 1 + (BOOST - 1) * self_m._sel.to(term.dtype)
                term = term * (w[..., None] if term.dim() == 3 else w)
            self_m.classifier_loss = term.mean()
        m.SANITY_loss = types.MethodType(sanity_loss, m)
    m._cause_patched = True


def run_name(manip, kind, sel, frac):
    tag = f"{manip}_{kind}" + (f"_{sel}" if kind == "hi" else "") + ("" if kind == "all" else f"_f{round(frac * 100):02d}")
    return f"learncause_{tag}_s{lm.SEED}"


def main(manip, kind, sel, frac, gpu, episodes=1001):
    assert manip in MANIPS and kind in KINDS and sel in ("ent", "fr"), (manip, kind, sel)
    run = run_name(manip, kind, sel, frac)
    out = os.path.join(lm.TD, run)
    os.makedirs(out, exist_ok=True)
    trunk = os.path.join(lm.TD, f"learnmask_trunk_s{lm.SEED}")
    cfg = {"mode": "SANITY", "cuda": gpu, "episodes": episodes, "checkpoint_every": 5, "realization_num": 10,
           "hid_dim": 1000, "obs_num": 5, "show_plots": False, "training": True, "batch_num": 8000, "step_num": 30,
           "state_num": 500, "learn_embeddings": False, "classifier_LR": .001, "ctx_num": 2,
           "skip_training_analyses": True, "reservoir": False, "gpu_inference": True, "micro_batch": lm.MICRO,
           "load_env": os.path.relpath(os.path.join(trunk, f"ckpt_ep{lm.START:05d}"), os.path.join(ROOT, "main", "DATA"))}
    adam = os.path.join(trunk, f"adam_ep{lm.START:05d}.pth")
    seed = lm.BRANCH_SEED
    json.dump({"run": run, "manip": manip, "kind": kind, "sel": sel, "frac": frac, "start": lm.START, "seed": seed,
               "cfg": cfg, "started": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(out, "run.json"), "w"),
              indent=1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    Agent = type("Agent", (CausalAgent,), {"manip": manip, "kind": kind, "sel": sel, "frac": frac, "adam_path": adam,
                                           "cond": f"cause_{manip}"})
    Agent._gen = torch.Generator().manual_seed(seed + 7)
    Agent._sel_log = []
    rec = lm.Recorder(out, lm.START)
    agent = Agent(**cfg, external_logger=rec)
    rec.flush(agent)
    with open(os.path.join(out, "progress.log"), "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} DONE\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], float(a[3]), int(a[4]), int(a[5]) if len(a) > 5 else 1001)
