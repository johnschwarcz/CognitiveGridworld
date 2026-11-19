"""CHANNEL-SHUFFLED variant (John 11-18: "If it overfits the R^2 doesnt grow? What if we shuffle the channels?"): before
every update on the fixed batch, each trajectory's 5 observation channels get a fresh random order, applied to its
observations and to the matching rows of its context input active_Z (so it is the same environment, relabelled; the Bayes
targets over the variables' values do not change). Test episodes see the original order.
Base: John 11-18: "What about running the normal lr update multiple times per batch? ... Perhaps we could do 100 updates per
episode and only 10 episodes?" The training-dynamics pair (seed 20261029, batch 8,000, lr 1e-3) with K optimizer
updates on each training episode's batch (forward_backward repeated: the same batch and Bayes targets, the gradient
recomputed after every step); 13 episodes = 10 training episodes (tests at 0, 5, 10) = 10 K updates. Weights saved after
cumulative updates in SAVE_U and after every episode: <out>/ckpt_u<UUUUU>_net.pth ({"weights", "episode", "updates"}).
Usage: run_multistep.py <trained|echo> <gpu> <K> <subfolder> [episodes=13] [batch] [micro]
(John 11-18: "Lets do 100 updates with the 100K batch" -> episodes 3 = one training episode, batch 100,000, slices of
20,000: the extra updates then go through forward_pass_micro(train=True, outputs=False), the same slices and step.)"""
import os, sys
ROOT = "/home/johns/Desktop/CognitiveGridworld"
os.chdir(ROOT)
net, gpu, K, sub = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
EPISODES = int(sys.argv[5]) if len(sys.argv) > 5 else 13
if len(sys.argv) > 7:
    os.environ.update(TD_BATCH=sys.argv[6], TD_MICRO=sys.argv[7], TD_GPU_BAYES="1")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION", "trainings")]
import torch
import train_training_dynamics as T
T.OUT = os.path.join(T.OUT, sub)
SAVE_U = {1, 2, 4, 8, 16, 32, 64}
SAVE_EVERY = int(os.environ.get("MS_SAVE_EVERY", 0))   # also every N updates (John 11-18: one 20K batch, 2K updates)
state = {"u": 0}
Env = T.CognitiveGridworld
_fb = Env.forward_backward


def save(env):
    out = os.path.join(T.OUT, f"{net}_s20261029_ep{EPISODES}")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"ckpt_u{state['u']:05d}_net.pth")
    torch.save({"weights": {k: v.detach().cpu() for k, v in env.model.state_dict().items()}, "episode": env.e,
                "updates": state["u"]}, path + ".tmp")
    os.replace(path + ".tmp", path)


def fb(self):
    if self.test_set:
        if state["u"] == 0:
            save(self)                                   # the initial weights, u = 0
        return _fb(self)
    for _ in range(K - 1):                               # K - 1 extra updates on this batch, then the logged one
        with torch.amp.autocast('cuda', enabled=self.device_type != "cpu"):
            if self.uses_micro_batches():
                self.model.forward_pass_micro(train=True, outputs=False)
            else:
                self.model.forward_pass()
                self.model.backward_pass()
        state["u"] += 1
        if state["u"] in SAVE_U or (SAVE_EVERY and state["u"] % SAVE_EVERY == 0):
            save(self)
    out = _fb(self)
    state["u"] += 1
    save(self)                                           # every episode's end (and u in SAVE_U at K = 1)
    return out


PERM = {"p": None}


def install(model):
    """active_Z follows the current channel order (MC_to_interactions recomputes it at every forward pass)"""
    if getattr(model, "_perm_hook", False):
        return
    orig = model.MC_to_interactions

    def hooked():
        orig()
        p = PERM["p"]
        if p is not None:
            model.active_Z = torch.gather(model.active_Z, 1, p[:, :, None].expand(-1, -1, model.active_Z.shape[2]))
    model.MC_to_interactions = hooked
    model._perm_hook = True


def shuffled(fn):
    def wrapper(self):
        if self.test_set:
            return fn(self)
        install(self.model)
        obs0 = self.model.obs_flat
        B, Tn = obs0.shape[0], obs0.shape[1]
        global_draw = lambda: torch.argsort(torch.rand(B, obs0.shape[2], device=obs0.device), dim=1)

        orig_fp, orig_fpm = self.model.forward_pass, self.model.forward_pass_micro

        def set_perm():
            PERM["p"] = global_draw()
            self.model.obs_flat = torch.gather(obs0, 2, PERM["p"][:, None, :].expand(-1, Tn, -1))

        def fp(*a, **k):                                 # a new order before every forward pass of an update
            set_perm()
            return orig_fp(*a, **k)

        def fpm(*a, **k):
            set_perm()
            return orig_fpm(*a, **k)
        self.model.forward_pass, self.model.forward_pass_micro = fp, fpm
        try:
            return fn(self)
        finally:
            self.model.forward_pass, self.model.forward_pass_micro = orig_fp, orig_fpm
            PERM["p"] = None
            self.model.obs_flat = obs0
    return wrapper


Env.forward_backward = shuffled(fb)
T.main(net, gpu, EPISODES, 20261029)
