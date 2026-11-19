"""John 11-18 ("if we update the weights on each inference step, instead of the BPTT, does that speed up learning from a
single batch? ... an ESN can carry evidence forward, so maybe we dont need recurrent gradients ... much cheaper gradient
and maybe even overfit less?"): the training-dynamics pair (seed 20261029, lr 1e-3, the model's own Adam) on ONE batch of
20,000 with an update at EVERY step and no BPTT: at step t the network runs from the detached state (h, c) and the
detached running sum of read-out logits, its goal belief at t = softmax(sum of read-outs to t), the loss = the model's
SANITY loss restricted to step t ((sym-KL - its detached minimum + 1e-8) ** .5, mean), then one optimizer step (float32,
no AMP). 30 updates per pass over the batch; N_UPD updates in all. Checked first: the step-by-step forward reproduces the
model's own forward. Weights saved after cumulative updates 1, 2, 4, ..., 64 and every 100 -> <out>/ckpt_u<U>_net.pth.
Usage: run_perstep.py <trained|echo> <gpu> <subfolder> [n_updates=2010]"""
import os, sys
ROOT = "/home/johns/Desktop/CognitiveGridworld"
os.chdir(ROOT)
net, gpu, sub = sys.argv[1], int(sys.argv[2]), sys.argv[3]
N_UPD = int(sys.argv[4]) if len(sys.argv) > 4 else 2010
os.environ.update(TD_BATCH=os.environ.get("PS_BATCH", "20000"), TD_MICRO="10000", TD_GPU_BAYES="1")   # PS_BATCH: smoke tests
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION", "trainings")]
import torch
import train_training_dynamics as T
T.OUT = os.path.join(T.OUT, sub)
Env = T.CognitiveGridworld
_fb = Env.forward_backward
SAVE_U = {1, 2, 4, 8, 16, 32, 64} | set(range(100, N_UPD + 1, 100))
DONE = {"x": False}


def save(env, u):
    out = os.path.join(T.OUT, f"{net}_s20261029_ep3")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"ckpt_u{u:05d}_net.pth")
    torch.save({"weights": {k: v.detach().cpu() for k, v in env.model.state_dict().items()}, "episode": env.e,
                "updates": u}, path + ".tmp")
    os.replace(path + ".tmp", path)


def fb(self):
    if self.test_set or DONE["x"]:
        if self.test_set and not DONE["x"]:
            save(self, 0)
        return _fb(self)
    m = self.model
    m.MC_to_interactions()
    B, Tn, R = m.batch_num, m.step_num, m.realization_num
    O = m.obs_flat.float()
    Zc = m.classifier_Z().reshape(B, -1).float()
    lstm, rd, ro = m.LSTM, m.classifier_readin, m.classifier_readout
    W_ih, W_hh, b = lstm.weight_ih_l0, lstm.weight_hh_l0, None
    ar = torch.arange(B, device=O.device)
    g = m.goal_ind.long()
    P = m.joint_goal_belief.float()

    def step(t, h, c, S):
        x = torch.relu(rd(torch.cat((O[:, t], Zc), -1)))
        gates = x @ W_ih.T + (lstm.bias_ih_l0 + lstm.bias_hh_l0) + h @ W_hh.T
        i, f, gg, o = gates.chunk(4, -1)
        c2 = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(gg)
        h2 = torch.sigmoid(o) * torch.tanh(c2)
        L = S + ro(h2)
        q = torch.softmax(L.reshape(B, 2, R), -1)[ar, g]                      # (B, R) goal belief at t
        return h2, c2, L, q

    h0 = m.STM.expand(-1, B, -1)[0].float()
    c0 = m.LTM.expand(-1, B, -1)[0].float()
    with torch.no_grad():                                                     # the check: = the model's own forward
        h, c, S = h0, c0, torch.zeros(B, 2 * R, device=O.device)
        qs = []
        for t in range(Tn):
            h, c, S, q = step(t, h, c, S)
            qs.append(q)
        mine = torch.stack(qs, 1)
        n_chk = min(B, 2000)                                                  # the model's forward on a slice (memory)
        with torch.amp.autocast("cuda", enabled=False), m.batch_slice(slice(0, n_chk)):
            m.MC_to_classification()
            ref = m.classifier_goal_belief.float()
        d = (mine[:n_chk] - ref).abs().max().item()
    print(f"per-step forward vs the model's: max abs {d:.2e}", flush=True)
    assert d < 1e-3, d
    opt, u = m.classifier_optim, 0
    while u < N_UPD:
        h, c, S = m.STM.expand(-1, B, -1)[0].float(), m.LTM.expand(-1, B, -1)[0].float(), torch.zeros(B, 2 * R, device=O.device)
        for t in range(Tn):
            h2, c2, L, q = step(t, h.detach(), c.detach(), S.detach())
            D = m.DKL_sym(q, P[:, t], PM=False)
            loss = ((D - D.detach().min() + 1e-8) ** .5).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            u += 1
            if u in SAVE_U:
                save(self, u)
            h, c, S = h2.detach(), c2.detach(), L.detach()
            if u >= N_UPD:
                break
    print(f"per-step training done: {u} updates, last loss {loss.item():.4f}", flush=True)
    DONE["x"] = True
    self.test_set = True                                                       # outputs for the logs, no update
    try:
        out = _fb(self)
    finally:
        self.test_set = False
    self.classifier_loss, self.generator_loss, self.readin_grad, self.readout_grad = float(loss.item()), 0., 0., 0.
    return out


Env.forward_backward = fb
T.main(net, gpu, 3, 20261029)
