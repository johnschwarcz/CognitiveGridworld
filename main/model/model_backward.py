import torch; import torch.nn as nn; import math; import torch.nn.functional as F; from torch.distributions import Categorical;
from main.model.Model_Customization import Model_Customization; from main.utils import tnp;

class Model_backward(Model_Customization):
 
    def backward_pass(self):
        if self.mode == "RL":
            self.RL_loss()  
        if self.mode == "SANITY":        
            self.SANITY_loss()
        if self.learn_embeddings:
            self.SSL_loss()
            self.update(self.generator_loss, self.generator_optim)  

        self.update(self.classifier_loss, self.classifier_optim, self.classifier_gradients)
        return tnp([self.classifier_loss, self.generator_loss, self.readin_grad, self.readout_grad], 'np')

    def update(self, loss, optim, collect_grad = None):
        optim.zero_grad()
        torch.cuda.empty_cache()
        loss.backward()
        if collect_grad is not None:
            collect_grad()
        optim.step()

    def classifier_gradients(self):
        if self.mode == "SANITY" :
            self.readin_grad = self.get_gradient_norm(self.classifier_readin)
            self.readout_grad = self.get_gradient_norm(self.classifier_readout)

    ########################################################################################################
    """ default loss functions """ 
    ########################################################################################################

    def SANITY_loss(self, eps = 1e-8):
        P = self.joint_goal_belief
        Q = self.classifier_goal_belief
        DKL = self.DKL_sym(Q, P, PM=False)
        DKL = (DKL - DKL.detach().min() + eps)**0.5
        self.classifier_loss = DKL.mean()
        
    def SSL_loss(self):
        chance = 1/self.realization_num
        last_ACC = self.ACC[:, -1, None]
        OPE = self.DKL_sym(self.pred_pobs, self.obs_flat.mean(1))
        OPE__ACC = (last_ACC * OPE).sum() / last_ACC.sum()
        OPE = OPE.mean() * chance + (1 - chance) * OPE__ACC

        K_norm = (torch.norm(self.active_K, dim=-1)-1) ** 2
        Q_norm = (torch.norm(self.active_Q, dim=-1)-1) ** 2
        self.generator_loss = OPE + (K_norm + Q_norm).mean()

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