import numpy as np; import torch; import torch.nn as nn; import torch.optim as optim
from .model_forward import Model_forward; from main.utils import tnp

class Model_controller(Model_forward):
    def __init__(self, **kwargs):
        super(Model_controller, self).__init__()

    def init_controller(self, preferences, offline_teacher, JL, NL):        
        self.actor_readin, self.critic_readin = [nn.Linear(self.Z_num * self.obs_num, self.hid_dim, device = self.device) for _ in range(2)]
        self.actor_hid2hid, self.critic_hid2hid = [nn.Linear(self.hid_dim, self.hid_dim, device = self.device) for _ in range(2)]
        self.actor_readout = nn.Linear(self.hid_dim, self.R_to_the_ctx, device = self.device)
        self.critic_readout = nn.Linear(self.hid_dim, 1, device = self.device) 
        self.preferences  = tnp(preferences, 'torch', self.device)[None, :]
        self.offline = offline_teacher is not None
        self.offline_teacher = offline_teacher
        self.controller_actions = self.ctx_vals
        self.O = self.obs_flat.mean(1)
        self.control_ent_bonus = 1
        self.joint_likelihood = JL
        self.naive_likelihood = NL

        params = [{'params': 
            list(self.actor_readin.parameters()) +
            list(self.actor_readout.parameters()) +  
            list(self.critic_readin.parameters()) +
            list(self.critic_readout.parameters()), 'lr': self.controller_LR}]            
        self.controller_optim = optim.Adam(params)

        self.MC_to_interactions()
        self.update_environment()
        return tnp(self.controller_actions, 'np')

    def forward_controller(self, O = None):
        if O is None:
            self.get_pred_pobs()        
        else:
            self.O = tnp(O, 'torch', self.device)
        self.evaluate_control()
        self.update_controller()
        self.update_environment()
        return tnp([self.controller_actions, self.argmax_vals, self.controller_policy[0]],'np')

    def evaluate_control(self, eps = 1e-6):       
        self.O = self.O.clip(eps, 1 - eps)
        O = self.O.log() * self.preferences
        O_ = (1 - self.O).log() * (1 - self.preferences)
        self.intrinsic_value = ((O + O_).sum(1) / self.obs_num).exp() 

    def get_pred_pobs(self):
        if self.offline_teacher == "generator":
            with torch.no_grad():
                self.default_pobs(train_controller = True)
                self.O = self.pred_pobs

        if self.offline_teacher == "joint":
            O=self.obs_num
            B=self.batch_num
            OR=self.obs_range[None,:].expand(B,O)
            BR=self.batch_range[:,None].expand(B,O)
            ix=tuple(self.controller_actions[:,i][:,None].expand(B,O) for i in range(self.ctx_num))
            self.O = self.joint_likelihood[BR, OR, *ix].squeeze() 

        if self.offline_teacher == "naive":
            O = self.naive_likelihood
            r = self.controller_actions[:,None,:,None]
            norm = torch.logit(O.mean((-1, -2), keepdims = True)).squeeze()
            OLLR = torch.logit(torch.take_along_dim(O, r, dim=-1)).squeeze()
            self.O = torch.sigmoid(OLLR.sum(2) - (self.ctx_num -1) * norm)

    def update_controller(self):
        self.control_ent_bonus = self.control_ent_bonus * self.control_ent_bonus_decay 
        CPE = self.intrinsic_value - self.predicted_intrinsic_value  
        ent_loss = self.control_ent_bonus * self.control_ent.mean()
        PG_loss = (self.control_NLL * CPE.detach()).mean()
        CPE_loss = (CPE**2).mean()

        loss = PG_loss + CPE_loss - ent_loss
        self.controller_optim.zero_grad()
        loss.backward()
        self.controller_optim.step()

    def update_environment(self):
        self.controller_forward()
        dist = self.controller_postprocess()
        self.controller_sample_actions(dist)

    def controller_forward(self):
        z = self.active_Z.reshape(self.batch_num,-1).detach()

        v = torch.relu(self.critic_hid2hid(torch.relu(self.critic_readin(z))))
        self.predicted_intrinsic_value = torch.sigmoid(self.critic_readout(v)).squeeze()

        a = self.actor_readout(torch.relu(self.actor_hid2hid(torch.relu(self.actor_readin(z)))))
        self.controller_policy = torch.softmax(a, dim = -1).reshape(self.batch_num, *self.ctx_dims)

    def controller_postprocess(self):
        dist = self.controller_policy.reshape(self.batch_num, -1)
        amax = torch.unravel_index(dist.argmax(-1), self.ctx_dims)
        self.argmax_vals = torch.stack(amax,1).long()        
        return torch.distributions.Categorical(probs = dist)

    def controller_sample_actions(self, dist):
        actions = dist.sample()
        self.control_ent = dist.entropy() 
        self.control_NLL = -dist.log_prob(actions)
        a = torch.unravel_index(actions, self.ctx_dims)
        self.controller_actions = torch.stack(a,1).long()
