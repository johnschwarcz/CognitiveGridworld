"""Training-dynamics runs (29 Oct 2026; John: "queue up a training run of a standard trained and ESN lstm to record
measures throughout training ... it is better to record more rather than less so that we dont need to redo the
training"). The project's two standard networks (fully_trained_ctx_2_e5 / reservoir_ctx_2_e5, retrain_other_models.py
train_e5_variants: SANITY mode, 50,000 episodes, a test episode every 5, batch 8000, 30 steps, LSTM width 1000,
read-out lr 1e-3, two variables, learn_embeddings False), retrained here from ONE seed for both, so they share the
environment, the initial weights and the episode sequence (the ESN's LSTM frozen at that same init). Recorded:
  checkpoints   the full save() format (K, Q, model_K, model_Q, weights), loadable with load_env, at test episodes on a
                dense-early schedule (every 25 episodes to 500, every 100 to 3,000, every 250 to 10,000, every 1,000 to
                50,000; 116 per run): every measure (activity, beliefs, decoding, entanglement responses, update Grams,
                band interference, ...) can be computed afterwards on fixed probe episodes
  Adam          the optimizer and AMP scaler state every 5,000 episodes and at the end (to resume or extend a run)
  every update  the loss, the per-step training accuracy, the AMP loss scale, and per trainable tensor: the gradient
                norm, the update norm |W_t - W_t-1|, the cosine between consecutive updates, |W_t|, and an 8 x 8
                two-sided random sketch of the update (U^T dW V; vectors U^T dW) -> online_<NNN>.npz, one chunk per
                checkpoint
  every test    (every 5 episodes, 10,000 per run) accuracy per step of the network, the joint and the naive; DKL of the
  episode       network's goal belief to the joint's and to the naive's per step; the logistic fit of final-step
                correctness on final-step SII (score, coefficient); the full PCA spectra (1,000 variances) of the
                LSTM's input and of its update, as eigenvalues of their covariance on the GPU (= sklearn PCA's
                explained_variance_, which the agent computes on the CPU in 4 s per test episode; the agent's own
                analyses are switched off with skip_training_analyses, which changes no training: the same seed gives
                identical accuracies) -> test_<NNN>.npz, one chunk per checkpoint
Networks (argument 1): trained (the standard LSTM), echo (the standard ESN: the same LSTM frozen), and, added the same
day for the weekend queue: joint_echo (the ESN with the joint read-out over the 100 cells, train_sanity_update_joint.py:
output_joint, gpu_inference; the native goal-only loss), joint_echo_jl (the same trained on the joint loss KL(p || q)
over the 100 cells at every step, ported from interaction_bounds/analyses/joint_esn_extension/run_variant.py), mlp55
(the MLP-rescued ESN: 5 trained layers before and 5 after the frozen LSTM, readin_depth / readout_depth 5), mlp91 (9
before, 1 after) and mlp19 (1 before, 9 after). Queued later the same day for the theory session (its questions: is
the trained LSTM's fast phase after naive Bayes the onset of memory use; does trained recurrence avoid the joint-output
trap): joint_trained (the trained LSTM with the joint read-out and the goal-only loss), trained_lastT (the trained LSTM
with the loss at T = 30 only: the same symmetric KL, batch-minimum shift and power on the last step), trained_N50 /
trained_N100 / trained_N200 (the trained LSTM at widths 50, 100, 200). Added 11-01 (John: "yea let's start that
now"): joint_trained_jl (the trained LSTM with the joint read-out on joint_echo_jl's joint loss: the empty cell of
plot_matched_params_new.py's right panel, trained network x joint output x full supervision).
Added for the queued runs (John 10-29: "make sure that you know what you want to record from the others' training"):
  every update  train_DKL: the network's goal belief vs the joint's on the training batch, per step (30); and for the
                joint read-outs (100 cells, r_0-major), the exclusion trap's quantities, which form within ~10 updates
                (before the first regular checkpoint): the batch mean of log q per cell at step 1 and at T (2 x 100),
                the mean log q of the true cell at T, and per cell the read-out bias, |weight row|, |gradient row|,
                |bias gradient| and Adam's second moments (row mean; bias)
  checkpoints   joint read-outs also at every test episode to episode 200
Added after reading theory_v9 (John: "decide whether you still want to record anything else during training"), for
the theory's trap-permanence mechanism ("Adam's epsilon nearly stops those read-out rows, and fp16 underflow stops them
exactly"), its gate (Proposition 3) and its regret measure: per read-out row the actual step |dW_row|, Adam's first
moment (row mean |m|) and the share of exactly-zero gradient entries; the across-episode SD of log q per cell (step 1,
T: the shared first-step component); the gate log q(c*|g*) of the true pair (step 1, T); and XL = KL(Bayes || network)
per step (every update: train_XL; every test episode: XL), the direction of the theory's cumulative regret (train_DKL
and the test DKLs are KL(network || Bayes)).
Outputs: main/DATA/experimental/training_dynamics/<run>/ (ckpt_ep<EEEEE>_net.pth, adam_ep<EEEEE>.pth, online_*.npz,
test_*.npz, agent_logs.npz, final_net.pth, run.json, progress.log). Speed (GPU 1 alone): ~1.0 s per training episode,
~2.3 s per test episode -> ~17 h for 50,000 episodes. TD_GPU_BAYES=1 sets gpu_inference (the same exact posterior in
log space on the GPU; the paper's train_sanity_reps.py setting): at width 100, 0.36 s per episode vs 0.98 on the CPU,
identical test accuracies for the same seed over 55 episodes (10-31); used for the trained_N100 reps 20261030-33 (John
10-31: "start the four now"). Run from the repository root:
    /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/trainings/train_training_dynamics.py \\
        <trained|echo> <gpu> [episodes] [seed]"""
import os
import sys
import json
import time
import types
import random
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main")]
from main.CognitiveGridworld import CognitiveGridworld                            # noqa: E402
from main.utils import tnp                                                        # noqa: E402

OUT = os.path.join(ROOT, "main", "DATA", "experimental", "training_dynamics")
SEED = 20261029
SKETCH_K = 8


def schedule(episodes, dense_until=0):
    """test episodes (multiples of 5) at which a checkpoint is saved; every test episode up to dense_until"""
    s = set(range(0, 501, 25)) | set(range(500, 3001, 100)) | set(range(3000, 10001, 250)) \
        | set(range(10000, episodes + 1, 1000)) | set(range(0, dense_until + 1, 5))
    last = 5 * ((episodes - 1) // 5)
    return sorted(e for e in s | {last} if e < episodes)


class Recorder:
    """external_logger: called after every episode (after the update on training episodes)"""

    def __init__(self, out, sched, opt_every=5000, extra=False):
        self.out, self.sched, self.opt_every, self.extra = out, set(sched), opt_every, extra
        self.prev = self.prev_d = self.U = self.V = None
        self.buf, self.test, self.chunk, self.t0 = [], [], 0, time.time()

    def _params(self, m):
        ids = {id(p) for g in m.classifier_optim.param_groups for p in g["params"]}
        return [(n, p) for n, p in m.named_parameters() if id(p) in ids and p.requires_grad]

    def _init(self, m):
        g = torch.Generator(device="cpu").manual_seed(SEED + 1)
        self.names, self.U, self.V = [], {}, {}
        for n, p in self._params(m):
            self.names.append(n)
            x = p.detach().reshape(p.shape[0], -1) if p.ndim >= 2 else p.detach().reshape(-1, 1)
            self.U[n] = torch.randn(x.shape[0], SKETCH_K, generator=g).to(p.device) / np.sqrt(x.shape[0])
            self.V[n] = torch.randn(x.shape[1], SKETCH_K, generator=g).to(p.device) / np.sqrt(x.shape[1])
        self.prev = {n: p.detach().float().clone() for n, p in self._params(m)}
        self.prev_d = {n: None for n in self.names}

    def _online(self, env):
        m = env.model
        ro = _last_linear(m.classifier_readout)
        w_ro = ro.weight
        drow = None
        rec = {"episode": env.e, "loss": float(np.asarray(env.classifier_loss)), "scale": float(m.scaler.get_scale()),
               "train_acc": np.asarray(env.model_acc, np.float64).mean(0).astype(np.float32)}
        g_n, d_n, cos, w_n, sk = [], [], [], [], []
        with torch.no_grad():
            for n, p in self._params(m):
                gr = p.grad
                g_n.append(float(gr.float().norm()) if gr is not None else np.nan)
                w = p.detach().float()
                d = w - self.prev[n]
                dn = d.norm()
                d_n.append(float(dn))
                pd = self.prev_d[n]
                cos.append(float((d * pd).sum() / (dn * pd.norm() + 1e-30)) if pd is not None else np.nan)
                w_n.append(float(w.norm()))
                x = d.reshape(d.shape[0], -1) if d.ndim >= 2 else d.reshape(-1, 1)
                sk.append((self.U[n].T @ x @ self.V[n]).cpu().numpy().astype(np.float32))
                if p is w_ro:
                    drow = d.norm(dim=1)                                           # the read-out's step per row
                self.prev[n], self.prev_d[n] = w.clone(), d
        rec.update(grad_norm=np.array(g_n, np.float32), update_norm=np.array(d_n, np.float32),
                   update_cos=np.array(cos, np.float32), weight_norm=np.array(w_n, np.float32),
                   sketch=np.stack(sk))
        if self.extra:
            rec["train_DKL"] = np.asarray(env.DKL(env.model_goal_belief, env.joint_goal_belief), np.float32)
            rec["train_XL"] = np.asarray(env.DKL(env.joint_goal_belief, env.model_goal_belief), np.float32)
        if self.extra and not getattr(m, "output_joint", False):                  # marginal read-outs: per row / value
            with torch.no_grad():
                W, st = ro.weight, m.classifier_optim.state
                lb = torch.log(m.classifier_belief.float().clamp_min(1e-30))[:, [0, -1]]   # (B, 2 steps, 2 vars, R)
                nan = torch.full((W.shape[0],), float("nan"), device=W.device)
                rec.update(val_gnorm=W.grad.float().norm(dim=1).cpu().numpy(), val_dnorm=drow.cpu().numpy(),
                           val_gzero=(W.grad == 0).float().mean(1).cpu().numpy(),
                           val_m=(st[W]["exp_avg"].float().abs().mean(1) if W in st else nan).cpu().numpy(),
                           val_v=(st[W]["exp_avg_sq"].float().mean(1) if W in st else nan).cpu().numpy(),
                           val_bias=ro.bias.detach().float().cpu().numpy(),
                           val_logq=lb.mean(0).cpu().numpy(), val_logq_sd=lb.std(0).cpu().numpy())
        if self.extra and getattr(m, "output_joint", False):
            with torch.no_grad():
                lq = m._jl_logq.float()                                              # (B, T, 100), r_0-major
                cv = torch.as_tensor(np.asarray(env.ctx_vals), device=lq.device).long()
                true = cv[:, 0] * env.realization_num + cv[:, 1]
                W, b = m.classifier_readout.weight, m.classifier_readout.bias
                st = m.classifier_optim.state
                vW = st[W]["exp_avg_sq"].float().mean(1) if W in st else torch.full((W.shape[0],), float("nan"))
                vb = st[b]["exp_avg_sq"].float() if b in st else torch.full((b.shape[0],), float("nan"))
                R_ = env.realization_num
                gi = torch.as_tensor(np.asarray(env.goal_ind), device=lq.device).long()
                Q = lq[:, [0, -1]].exp().reshape(len(true), 2, R_, R_)              # (B, 2, r_0, r_1)
                qg = torch.where((gi == 0)[:, None], Q[torch.arange(len(true)), :, cv[:, 0], :].sum(-1),
                                 Q[torch.arange(len(true)), :, :, cv[:, 1]].sum(-1))   # the goal's marginal (B, 2)
                gate = lq[torch.arange(len(true))[:, None], torch.tensor([0, lq.shape[1] - 1], device=lq.device),
                          true[:, None]] - torch.log(qg.clamp_min(1e-30))
                mW = st[W]["exp_avg"].float().abs().mean(1) if W in st else torch.full((W.shape[0],), float("nan"))
                rec.update(cell_logq_sd=torch.stack([lq[:, 0].std(0), lq[:, -1].std(0)]).cpu().numpy(),
                           true_gate=gate.mean(0).cpu().numpy().astype(np.float32),
                           cell_dnorm=drow.cpu().numpy(),
                           cell_m=mW.cpu().numpy(), cell_gzero=(W.grad == 0).float().mean(1).cpu().numpy())
                rec.update(cell_logq=torch.stack([lq[:, 0].mean(0), lq[:, -1].mean(0)]).cpu().numpy(),
                           true_logq=np.float32(lq[torch.arange(len(true)), -1, true].mean().item()),
                           cell_bias=b.detach().float().cpu().numpy(),
                           cell_wnorm=W.detach().float().norm(dim=1).cpu().numpy(),
                           cell_gnorm=W.grad.float().norm(dim=1).cpu().numpy(),
                           cell_bgrad=b.grad.float().abs().cpu().numpy(),
                           cell_v=vW.cpu().numpy(), cell_bv=vb.cpu().numpy())
        self.buf.append(rec)

    def _test(self, env):
        from sklearn.linear_model import LogisticRegression
        m = env.model
        rec = {"episode": env.e, "acc": np.asarray(env.test_accs[env.test_e - 1], np.float32),
               "joint_acc": np.asarray(env.joint_acc, np.float64).mean(0).astype(np.float32),
               "naive_acc": np.asarray(env.naive_acc, np.float64).mean(0).astype(np.float32),
               "joint_DKL": np.asarray(env.DKL(env.model_goal_belief, env.joint_goal_belief), np.float32),
               "naive_DKL": np.asarray(env.DKL(env.model_goal_belief, env.naive_goal_belief), np.float32),
               "XL": np.asarray(env.DKL(env.joint_goal_belief, env.model_goal_belief), np.float32)}
        X, Y = np.asarray(env.SII)[:, -1].reshape(-1, 1), np.asarray(env.model_acc)[:, -1]
        try:
            reg = LogisticRegression().fit(X, Y)
            rec["SII_score"], rec["SII_coef"] = np.float32(reg.score(X, Y)), np.float32(reg.coef_[0][0])
        except ValueError:                                                         # one class only (early)
            rec["SII_score"] = rec["SII_coef"] = np.float32(np.nan)
        with torch.no_grad():
            for k, T in (("update_dim", m.update_flat), ("input_dim", m.input_flat)):
                X_ = T.reshape(-1, m.hid_dim).float()
                X_ = X_ - X_.mean(0, keepdim=True)
                C = (X_.T @ X_ / (X_.shape[0] - 1)).double().cpu().numpy()       # eigenvalues on the CPU: cusolver
                del X_                                                           # crashed on a deep read-in's input
                ok = bool(np.isfinite(C).all())
                rec[k] = (np.linalg.eigvalsh(C)[::-1] if ok else np.full(len(C), np.nan)).astype(np.float32)
                rec[k + "_finite"] = np.bool_(ok)
        self.test.append(rec)

    def _flush(self):
        for buf, name in ((self.buf, "online"), (self.test, "test")):
            if buf:
                extra = {"names": np.array(self.names)} if name == "online" else {}
                np.savez(os.path.join(self.out, f"{name}_{self.chunk:03d}.npz"), **extra,
                         **{k: np.stack([np.asarray(r[k]) for r in buf]) for k in buf[0]})
        self.buf, self.test, self.chunk = [], [], self.chunk + 1

    def _agent_logs(self, env):
        n = env.test_e
        logs = {k: np.asarray(getattr(env, k))[:n] for k in (
            "test_accs", "train_accs", "test_net_joint_DKL", "test_net_naive_DKL", "test_SII_score", "test_SII_coef",
            "test_model_update_dim", "test_model_input_dim", "classifier_loss_log", "readin_grad_log",
            "readout_grad_log", "test_TPs", "test_mses")}
        np.savez(os.path.join(self.out, "agent_logs.npz"), test_episodes=np.arange(n) * env.checkpoint_every, **logs)

    def __call__(self, env):
        m = env.model
        if self.prev is None:
            self._init(m)
        if not env.test_set:
            self._online(env)
            return {}
        self._test(env)
        e = env.e
        if e in self.sched:
            self._flush()
            save = {"K": env.all_K, "Q": env.all_Q, "model_K": tnp(m.all_K, "np"), "model_Q": tnp(m.all_Q, "np"),
                    "weights": {k: v.detach().cpu() for k, v in m.state_dict().items()}, "episode": e}
            path = os.path.join(self.out, f"ckpt_ep{e:05d}_net.pth")
            torch.save(save, path + ".tmp")
            os.replace(path + ".tmp", path)
            self._agent_logs(env)
            acc = float(np.asarray(env.model_acc).mean(0)[-1])
            with open(os.path.join(self.out, "progress.log"), "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} ep {e} test acc T {acc:.4f} "
                        f"elapsed {(time.time() - self.t0) / 60:.1f} min\n")
        if e % self.opt_every == 0:
            torch.save({"optim": m.classifier_optim.state_dict(), "scaler": m.scaler.state_dict(), "episode": e},
                       os.path.join(self.out, f"adam_ep{e:05d}.pth"))
        return {}


NETS = {"trained": dict(reservoir=False), "echo": dict(reservoir=True),
        "joint_echo": dict(reservoir=True, output_joint=True, gpu_inference=True),
        "joint_echo_jl": dict(reservoir=True, output_joint=True, gpu_inference=True),
        "mlp55": dict(reservoir=True, readin_depth=5, readout_depth=5),
        "mlp91": dict(reservoir=True, readin_depth=9, readout_depth=1),
        "mlp19": dict(reservoir=True, readin_depth=1, readout_depth=9),
        "joint_trained": dict(reservoir=False, output_joint=True, gpu_inference=True),
        "joint_trained_jl": dict(reservoir=False, output_joint=True, gpu_inference=True),
        "trained_lastT": dict(reservoir=False),
        "trained_N50": dict(reservoir=False, hid_dim=50), "trained_N100": dict(reservoir=False, hid_dim=100),
        "trained_N200": dict(reservoir=False, hid_dim=200),
        "trained_lastT_p1": dict(reservoir=False), "trained_p1": dict(reservoir=False)}


def joint_log_posterior(env):
    """log P(r_0, r_1 | o_1..o_t) for every step, (B, T, 100) float32, cells r_0-major as the joint read-out's (from
    joint_esn_extension/run_variant.py)"""
    B, n_obs = env.batch_num, env.obs_num
    L = torch.as_tensor(env.joint_likelihood, device=env.device)
    o = torch.as_tensor(env.obs_flat, device=env.device).to(L.dtype)
    flat = L.reshape(B, n_obs, -1)
    cum = (torch.einsum("bto,bok->btk", o, torch.log(flat))
           + torch.einsum("bto,bok->btk", 1 - o, torch.log1p(-flat))).cumsum(1)
    return (cum - torch.logsumexp(cum, -1, keepdim=True)).float()


def _logits_to_belief(self, logits):
    """the native marginals; keeps the joint read-out's log softmax for the joint loss (run_variant.py)"""
    from main.model.model_forward import Model_forward
    self._jl_logq = torch.log_softmax(logits, -1)
    return Model_forward.logits_to_belief(self, logits)


def _joint_loss(self, eps=1e-8, power=0.5):
    """replaces SANITY_loss: KL(P_opt || Q_model) over the 100 cells, mean over episodes and steps (run_variant.py)"""
    logp = self._jl_logp
    self.classifier_loss = (logp.exp() * (logp - self._jl_logq)).sum(-1).mean()


class JointAgent(CognitiveGridworld):
    """the joint ESNs: the model keeps its read-out's log softmax (for the recorder; the native marginals unchanged);
    joint_loss: the loss replaced by the joint loss and the exact joint posterior handed to it every episode"""
    joint_loss = False

    def prep_model(self):
        super().prep_model()
        m = self.model
        if not getattr(m, "_jl_patched", False):
            assert m.output_joint and m.classifier_readout.out_features == 100
            m.logits_to_belief = types.MethodType(_logits_to_belief, m)
            if self.joint_loss:
                m.SANITY_loss = types.MethodType(_joint_loss, m)
            m._jl_patched = True
        if self.joint_loss:
            m._jl_logp = joint_log_posterior(self)


class JointLossAgent(JointAgent):
    joint_loss = True


def _last_step_loss(self, eps=1e-8, power=0.5):
    """SANITY_loss on the last step only (the same symmetric KL, batch-minimum shift and power)"""
    self.DKL_loss(self.classifier_goal_belief[:, -1:], self.joint_goal_belief[:, -1:], PM=False, eps=eps, power=power)


class LastStepAgent(CognitiveGridworld):
    """the trained LSTM supervised at T only: SANITY_loss replaced before the first episode. trained_lastT (power
    0.5) collapsed at its first update (all mass on one goal value, gradients exactly 0 from update 1 on; the known
    collapse of the power-0.5 loss at one step, as in the feedforward series), so the power-1 pair was added on 10-31:
    trained_lastT_p1 (loss at T only, power 1) and its control trained_p1 (every step, power 1)"""
    last_only, power = True, .5

    def prep_model(self):
        super().prep_model()
        m = self.model
        if not getattr(m, "_last_patched", False):
            last_only, power = self.last_only, self.power

            def loss(self_m, eps=1e-8, power_=power):
                sl = slice(-1, None) if last_only else slice(None)
                self_m.DKL_loss(self_m.classifier_goal_belief[:, sl], self_m.joint_goal_belief[:, sl], PM=False,
                                eps=eps, power=power_)
            m.SANITY_loss = types.MethodType(loss, m)
            m._last_patched = True


class LastStepP1Agent(LastStepAgent):
    power = 1.


class AllStepP1Agent(LastStepAgent):
    last_only, power = False, 1.


def _last_linear(module):
    """the read-out's last linear layer (its rows are the output values)"""
    if isinstance(module, torch.nn.Linear):
        return module
    return [x for x in module.modules() if isinstance(x, torch.nn.Linear)][-1]


def main(net, gpu, episodes=50000, seed=SEED):
    run = f"{net}_s{seed}" + ("" if episodes == 50000 else f"_ep{episodes}")
    out = os.path.join(OUT, run)
    os.makedirs(out, exist_ok=True)
    joint = net.startswith("joint")
    sched = schedule(episodes, dense_until=200 if joint else 0)
    cfg = {"mode": "SANITY", "cuda": gpu, "episodes": episodes, "checkpoint_every": 5, "realization_num": 10,
           "hid_dim": 1000, "obs_num": 5, "show_plots": False, "training": True, "batch_num": 8000, "step_num": 30,
           "state_num": 500, "learn_embeddings": False, "classifier_LR": .001, "ctx_num": 2,
           "skip_training_analyses": True, **NETS[net]}
    if os.environ.get("TD_BATCH"):                                                # smoke tests only
        cfg["batch_num"] = int(os.environ["TD_BATCH"])
    if os.environ.get("TD_GPU_BAYES"):                                            # recorded in run.json's cfg
        cfg["gpu_inference"] = True
    json.dump({"run": run, "net": net, "seed": seed, "cfg": cfg, "schedule": sched,
               "source": "retrain_other_models.train_e5_variants, + checkpoints and per-update logs",
               "started": time.strftime("%Y-%m-%d %H:%M:%S")}, open(os.path.join(out, "run.json"), "w"), indent=1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rec = Recorder(out, sched, extra=net not in ("trained", "echo"))
    Agent = {"joint_echo": JointAgent, "joint_trained": JointAgent, "joint_echo_jl": JointLossAgent,
             "joint_trained_jl": JointLossAgent,
             "trained_lastT": LastStepAgent, "trained_lastT_p1": LastStepP1Agent,
             "trained_p1": AllStepP1Agent}.get(net, CognitiveGridworld)
    agent = Agent(**cfg, save_env=os.path.relpath(os.path.join(out, "final"), os.path.join(ROOT, "main", "DATA")),
                  external_logger=rec)
    rec._flush()
    rec._agent_logs(agent)
    torch.save({"optim": agent.model.classifier_optim.state_dict(), "scaler": agent.model.scaler.state_dict(),
                "episode": episodes}, os.path.join(out, f"adam_ep{episodes:05d}.pth"))
    with open(os.path.join(out, "progress.log"), "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} DONE\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), int(a[2]) if len(a) > 2 else 50000, int(a[3]) if len(a) > 3 else SEED)
