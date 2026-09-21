import torch; import torch.nn as nn; import math; import torch.nn.functional as F; from torch.distributions import Categorical;
from main.model.Model_Customization import Model_Customization; from main.utils import tnp;

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

        if self.learn_embeddings:
            self.SSL_loss()
            if self.embedding_reg:
                if self.embedding_grad == "classifier":
                    self.classifier_loss = self.classifier_loss + self.reg_loss()
                else:
                    self.generator_loss = self.generator_loss + self.reg_loss()
            if self.embedding_grad == "both":
                self.update(self.classifier_loss + self.generator_loss, self.combined_optim)
                return self.backward_return()

            self.update(self.generator_loss, self.generator_optim)
        self.update(self.classifier_loss, self.classifier_optim, self.mode == "SANITY")
        return self.backward_return()

    def backward_return(self):
        return tnp([self.classifier_loss, self.generator_loss, self.readin_grad, self.readout_grad], 'np')
    
    def update(self, loss, optim, collect_grad = False):
        optim.zero_grad()
        # torch.cuda.empty_cache()
        self.scaler.scale(loss).backward()  
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
        last_ACC = self.ACC[:, -1, None]
        OPE = self.DKL_sym(self.pred_pobs, self.obs_flat.mean(1))
        if training_controller or self.mode == "oracle":
            OPE = OPE.mean() 
        else:
            OPE__ACC = (last_ACC * OPE).sum() / last_ACC.sum()
            OPE = OPE.mean() * chance + (1 - chance) * OPE__ACC

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

    def DKL_sym(self, x, y, PM = True):
        forward  = self.DKL(x, y) + PM * self.DKL(1-x, 1-y)
        backward = self.DKL(y, x) + PM * self.DKL(1-y, 1-x)
        DKL = (forward + backward) / 2
        return DKL
    
    def DKL(self, x, y):
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
                self.rnn_J_half.copy_(self.rnn_J)      # for lazyrich_equilibration()

        if self.rnn_langevin and self.lazyrich_noise_std > 0:
            with torch.no_grad():
                for p in self.lazyrich_params:
                    p.add_(torch.randn_like(p), alpha = self.lazyrich_noise_std)