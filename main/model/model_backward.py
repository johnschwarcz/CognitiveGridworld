import torch; import torch.nn as nn; import math; import torch.nn.functional as F; from torch.distributions import Categorical;
from main.model.Model_Customization import Model_Customization; from main.utils import tnp; from contextlib import contextmanager

class Model_backward(Model_Customization):
 
    def backward_pass(self):
        if self.mode == "RL":
            self.RL_loss()
        if self.mode == "oracle":
            self.oracle_loss()
        if self.mode == "SANITY":        
            self.SANITY_loss()
        if self.mode == "FF":
            self.FF_loss()
        if self.mode == "lazyrich":
            self.SANITY_loss(power = 1)
            self.lazyrich_update()
            return self.backward_return()
        if getattr(self, "freeze_generator", False):       # teaching runs (model_architecture.teach_setup): Classifier only
            return self.frozen_backward()

        dream = None
        if self.learn_embeddings:
            self.SSL_loss()
            if getattr(self, "z_full", False):     # z_full (no world structure, 2026-11-14): no norm loss on the embeddings
                pass
            elif self.embedding_grad == "classifier":
                self.classifier_loss = self.classifier_loss + self.reg_loss()
            else:
                self.generator_loss = self.generator_loss + self.reg_loss()
            # training iterations only (test episodes never reach backward_pass): the dream batch is drawn from the
            # pre-update Generator here; its Classifier forward and backward run inside the Classifier's update
            if getattr(self, "dream_weight", 0.0) > 0 and self.dream_sample():
                dream = self.dream_backward
            if self.embedding_grad == "both":
                self.update(self.classifier_loss + self.generator_loss, self.combined_optim, extra = dream)
                return self.backward_return()

            self.update(self.generator_loss, self.generator_optim)
        self.update(self.classifier_loss, self.classifier_optim, self.mode == "SANITY", extra = dream)
        return self.backward_return()

    def backward_return(self):
        out = [self.classifier_loss, self.generator_loss, self.readin_grad, self.readout_grad]
        return out if getattr(self, "raw_return", False) else tnp(out, 'np')   # raw: defer_train_outputs (env)
    
    def update(self, loss, optim, collect_grad = False, extra = None):
        optim.zero_grad()
        # torch.cuda.empty_cache()
        self.scaler.scale(loss).backward()  
        if extra is not None:       # dream_backward: its gradients add to these under the same scale, before the one step
            extra()
        if collect_grad:
            self.readin_grad = self.get_gradient_norm(self.classifier_readin)
            self.readout_grad = self.get_gradient_norm(self.classifier_readout)
        
        self.scaler.step(optim)
        self.scaler.update()  

    ########################################################################################################
    """ default loss functions """ 
    ########################################################################################################
    
    def FF_loss(self, eps = 1e-8, power = 0.5):
        P = self.joint_goal_belief[:, -1]
        Q = self.classifier_goal_belief[:, -1]
        self.DKL_loss(Q, P, PM=False, eps=eps, power=power)

    def SANITY_loss(self, eps = 1e-8, power = 0.5):
        P = self.joint_goal_belief
        Q = self.classifier_goal_belief
        self.DKL_loss(Q, P, PM=False, eps=eps, power=power)

    def DKL_loss(self, Q, P, PM, eps, power):
        DKL = self.DKL_sym(Q, P, PM=PM)
        DKL = (DKL - DKL.detach().min() + eps)**power
        self.classifier_loss = DKL.mean()

    def SSL_loss(self, training_controller = False):
        chance = 1/self.realization_num
        OPE = self.DKL_sym(self.pred_pobs, self.obs_flat.mean(1))       # batch x obs_num

        if training_controller or self.oracle:
            OPE = OPE.mean()
        else:
            last_ACC = self.ACC[:, -1]
            OPE__ACC = (last_ACC * OPE.mean(-1)).sum() / last_ACC.sum()
            OPE = OPE.mean() * chance + (1 - chance) * OPE__ACC * self.obs_num 

        self.generator_loss = OPE

    def reg_loss(self):
        K_norm = (torch.norm(self.active_K, dim=-1)-1) ** 2
        Q_norm = (torch.norm(self.active_Q, dim=-1)-1) ** 2
        return (K_norm + Q_norm).mean()

    def oracle_loss(self):
        BR = self.batch_range[:, None, None]
        SR = self.step_range[None, :, None]
        CR = self.ctx_range[None, None, :]
        tgt = self.ctx_vals[:, None, :].expand(-1, self.step_num, -1)
        belief = self.soft_clip(self.classifier_belief[BR, SR, CR, tgt])
        ent = -belief * belief.log() * self.classifier_ent_bonus
        self.classifier_loss = (-belief.log() - ent).mean()

    def dream_sample(self):
        """Offline imagination (2026-11-13), part 1 of 2: draw this training iteration's dream batch, or return False
        before dream_start. The batch is dreamed on the SAME environments (this iteration's active_Z, so the same variable
        pairs), with realization values drawn uniformly and independently per variable and observations sampled from the
        pre-update Generator at confidence 1 (plugin()'s and Experiment 3's call) by the env's rule
        (default_gen_observations: iid Bernoulli over steps and channels, float 0/1 as prep_model hands real ones over).
        dream_shuffle: the training targets are another dream episode's values (rows of vals permuted): same compute and
        loss form, no information about the observations (control). All draws use a private torch RNG stream
        (dream_rng): the global stream, hence every real-batch draw, is the same as in the same run without dreams.
        dream_iter counts training iterations of this process (restarts at 0 on RESUME)."""
        self.dream_loss = self.dream_acc = None
        self.dream_iter = getattr(self, "dream_iter", 0) + 1
        if self.dream_iter <= getattr(self, "dream_start", 0):
            return False
        B, T, O, C, R = self.batch_num, self.step_num, self.obs_num, self.ctx_num, self.realization_num
        pred_pobs = self.pred_pobs
        try:
            with torch.no_grad(), self.dream_rng(0):
                vals = torch.randint(R, (B, C), device = self.device)
                self.get_prediction(vals, torch.ones(B, C, device = self.device))
                p = self.pred_pobs.float().reshape(B, O)                   # fp16 under autocast; squeeze drops B or O = 1
                obs = (torch.rand(B, T, O, device = self.device) < p[:, None, :]).float()
                tgt = vals[torch.randperm(B, device = self.device)] if getattr(self, "dream_shuffle", False) else vals
        finally:
            self.pred_pobs = pred_pobs
        self.dream_batch = (vals, tgt, obs)
        return True

    def dream_backward(self):
        """Part 2 of 2, called by update() after the real Classifier loss's backward (which freed its graph) and before
        the Classifier's step, with no scaler.update in between: the Classifier reads the dream batch through
        classifier_Z() (route unchanged: detached in the Generator route, live in the Classifier route, where the real
        backward freed the Z graph, so MC_to_interactions() rebuilds it from the not-yet-stepped embeddings) and gets
        oracle_loss's terms on both variables at every step; dream_weight * that loss is backpropagated under the same
        scale, so the accumulated gradient equals that of (real loss + dream_weight * dream loss) and the step is the same
        as a single backward of the sum, while only one Classifier graph is alive at a time. dream_mask_test (default:
        mode RL) drops the (episode, variable) entries that are held-out test states (never a training goal, so RL
        feedback never trains them; oracle_loss does supervise them). Every attribute the dream forward overwrites is
        restored. dream_loss: the dream cross-entropy; dream_acc: the Classifier's accuracy at T on the dreamed values
        (the generating ones, also under dream_shuffle), same entries. classifier_loss (logged) includes
        dream_weight * dream_loss, as a number."""
        vals, tgt, obs = self.dream_batch
        self.dream_batch = None
        T = self.step_num
        keys = ("obs_flat", "pred_pobs", "classifier_belief", "classifier_belief_flat", "classifier_goal_belief",
                "classifier_goal_selection", "ACC", "update_flat", "input_flat", "active_K", "active_Q", "active_Z",
                "Z_count", "k", "q")
        saved = {k: self.__dict__[k] for k in keys if k in self.__dict__}
        try:
            with self.dream_rng(1):                                        # Categorical / readin_perm draws
                if self.embedding_grad in ("classifier", "both"):
                    self.MC_to_interactions()
                self.obs_flat = obs
                self.MC_to_classification()                                # with gradients: the Classifier learns from it
            BR = self.batch_range[:, None, None]
            SR = self.step_range[None, :, None]
            CR = self.ctx_range[None, None, :]
            belief = self.soft_clip(self.classifier_belief[BR, SR, CR, tgt[:, None, :].expand(-1, T, -1)])
            ent = -belief * belief.log() * self.classifier_ent_bonus
            terms = -belief.log() - ent                                    # (B, T, C): oracle_loss's terms
            hit = (self.classifier_belief_flat[:, -1].argmax(-1) == vals).float()      # (B, C)
            mask = getattr(self, "dream_mask_test", None)
            if (self.mode == "RL") if mask is None else mask:
                keep = (self.ctx_inds >= self.test_states).float()        # (B, C): 0 on held-out variables
                loss = (terms * keep[:, None, :]).sum() / (keep.sum() * T).clamp_min(1)
                acc = (hit * keep).sum() / keep.sum().clamp_min(1)
            else:
                loss, acc = terms.mean(), hit.mean()
            self.scaler.scale(self.dream_weight * loss).backward()
        finally:
            self.__dict__.update(saved)
        self.dream_loss, self.dream_acc = loss.detach(), acc
        self.classifier_loss = self.classifier_loss + self.dream_weight * self.dream_loss

    @contextmanager
    def dream_rng(self, part):
        """The dream's private torch RNG stream: the global CPU and this device's CUDA generator are forked (restored on
        exit) and seeded from (dream_seed, dream_iter, part), so dream draws neither consume nor depend on the global
        stream"""
        cuda = self.device.type == "cuda"
        with torch.random.fork_rng(devices = [self.device] if cuda else []):
            s = (int(getattr(self, "dream_seed", 0)) * 1000003 + self.dream_iter) * 2 + part
            torch.random.default_generator.manual_seed(s)
            if cuda:
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(s)
            yield

    def frozen_backward(self):
        """Teaching runs (freeze_generator, 2026-11-13): only the Classifier steps. Its loss is the mode's (RL_loss /
        oracle_loss, computed above), or oracle_teach_loss() (oracle_teach), or nothing (dream_only), plus dream_weight *
        the dream loss when dream_weight > 0 (dream_sample / dream_backward, unchanged: dreams by the frozen Generator).
        generator_loss = SSL_loss() without gradients, for the log only (no reg_loss: the embeddings are frozen). With
        dream_only, classifier_loss (logged) = dream_weight * dream_loss; before dream_start nothing is backpropagated and
        nothing steps."""
        with torch.no_grad():
            self.SSL_loss()
        self.classifier_optim.zero_grad()
        if getattr(self, "oracle_teach", False):
            self.classifier_loss = self.oracle_teach_loss()
        grads = not getattr(self, "dream_only", False)
        if grads:
            self.scaler.scale(self.classifier_loss).backward()
        else:
            self.classifier_loss = torch.zeros((), device = self.device)
        if getattr(self, "dream_weight", 0.0) > 0 and self.dream_sample():
            self.dream_backward()
            grads = True
        if grads:
            self.scaler.step(self.classifier_optim)
            self.scaler.update()
        return self.backward_return()

    def oracle_teach_loss(self):
        """The real-label teacher (oracle_teach): oracle_loss's terms on the real batch's true values of every variable at
        every step, with dream_backward's mask rule (dream_mask_test; None = mode RL: (episode, variable) entries whose
        variable is a held-out test state are left out). Unmasked it is oracle_loss exactly. Sets teach_loss (detached)
        and teach_acc (the Classifier's accuracy at T on the same entries)."""
        T = self.step_num
        BR = self.batch_range[:, None, None]
        SR = self.step_range[None, :, None]
        CR = self.ctx_range[None, None, :]
        tgt = self.ctx_vals[:, None, :].expand(-1, T, -1)
        belief = self.soft_clip(self.classifier_belief[BR, SR, CR, tgt])
        ent = -belief * belief.log() * self.classifier_ent_bonus
        terms = -belief.log() - ent
        hit = (self.classifier_belief_flat[:, -1].argmax(-1) == self.ctx_vals).float()         # (B, C)
        mask = getattr(self, "dream_mask_test", None)
        if (self.mode == "RL") if mask is None else mask:
            keep = (self.ctx_inds >= self.test_states).float()
            loss = (terms * keep[:, None, :]).sum() / (keep.sum() * T).clamp_min(1)
            acc = (hit * keep).sum() / keep.sum().clamp_min(1)
        else:
            loss, acc = terms.mean(), hit.mean()
        self.teach_loss, self.teach_acc = loss.detach(), acc
        return loss

    def RL_loss(self):
        CGS = self.classifier_goal_selection
        BR, SR = self.batch_range_, self.step_range_
        CGS = CGS[:, -1, None].repeat(1, self.step_num)
        CGB = self.classifier_goal_belief[BR, SR, CGS]
        belief = self.soft_clip(CGB)

        acc = self.ACC[:, -1, None]
        rew =  acc * -belief.log()
        pun = (1 - acc) * -(1 - belief).log() 
        ent = -belief * belief.log() * self.classifier_ent_bonus 
        self.classifier_loss = (rew + pun - ent).mean()       

    def DKL_sym(self, x, y, PM = True, eps = 1e-3):
        forward  = self.DKL(x, y, eps) + PM * self.DKL(1-x, 1-y, eps)
        backward = self.DKL(y, x, eps) + PM * self.DKL(1-y, 1-x, eps)
        DKL = (forward + backward) / 2
        return DKL
    
    def DKL(self, x, y, eps = 1e-3):
        x = self.soft_clip(x)
        y = self.soft_clip(y)
        return x * torch.log(x/y)

    def soft_clip(self, x, eps = 1e-3):
        ciel = float(1) - eps
        x = ciel - (1/(1+torch.exp(-(ciel-x)/eps)))*(ciel-x)
        x = eps + (1/(1+torch.exp(-(x-eps)/eps)))*(x-eps)
        return x


    def lazyrich_update(self):
        """
        One step of the Langevin gradient flow of Eq. (4) on the energy of Eq. (3):

            Theta <- Theta - lr [ grad(N gamma^2 L) + (1/beta) Theta ] + sqrt(2 lr / beta) xi

        beta is a temperature, not a regularization strength: the ridge's 1/beta cancels inside
        exp(-beta E) = exp(-beta N gamma^2 L) exp(-||Theta||^2 / 2), so the prior stays N(0,1)
        at any beta and beta multiplies only the data term.

        Plain gradient flow, not Adam: Adam normalizes each update by its own gradient RMS and
        would divide out exactly the gamma-dependence separating lazy from rich.
        `self.classifier_loss` keeps the unscaled task loss so logged curves compare across gamma.
        """
        self.classifier_optim.zero_grad()
        self.scaler.scale(self.lazyrich_energy_scale * self.classifier_loss).backward()
        self.scaler.unscale_(self.classifier_optim)

        self.readin_grad = 0. if self.rnn_U.grad is None else self.rnn_U.grad.detach().norm().item()
        self.readout_grad = 0. if self.rnn_V.grad is None else self.rnn_V.grad.detach().norm().item()
        if self.rnn_grad_clip is not None:
            # A binding clip would flatten the gamma-dependence this mode exists to measure;
            # lazyrich_clip_rate() is the check.
            total = nn.utils.clip_grad_norm_(self.lazyrich_params, self.rnn_grad_clip)
            self.lazyrich_clipped += int(float(total) > self.rnn_grad_clip)

        self.scaler.step(self.classifier_optim)
        self.scaler.update()
        self.lazyrich_steps += 1
        if self.lazyrich_steps == self.lazyrich_snapshot_at:
            with torch.no_grad():
                self.rnn_J_half = self.rnn_J.detach().cpu()     # for lazyrich_equilibration()

        if self.rnn_langevin and self.lazyrich_noise_std > 0:
            with torch.no_grad():
                for p in self.lazyrich_params:
                    p.add_(torch.randn_like(p), alpha = self.lazyrich_noise_std)