import torch; from torch.distributions import Categorical; from main.utils import tnp
import torch.nn.functional as F
from .model_backward import Model_backward

class Model_forward(Model_backward):
    def forward_pass(self):
        self.MC_to_interactions()
        self.MC_to_classification()
        self.MC_to_pobs()
        return tnp([self.classifier_belief_flat, self.classifier_goal_belief, self.input_flat, self.update_flat], 'np')

    def default_interactions(self):
        self.Z_count = 0
        self.active_K = self.all_K[self.ctx_inds, :]
        self.active_Q = self.all_Q[self.ctx_inds, :]
        if self.learn_embeddings:
            self.active_K = self.K_downscale(self.active_K)
            self.active_Q = self.Q_downscale(self.active_Q)

        self.active_Z = torch.zeros(self.Z_dims, device = self.device)
        if self.ctx_num == 1:
            self.KQ_to_Z(0,0)
        else:
            for self.k in self.ctx_range:
                for self.q in range(self.k+1, self.ctx_num):     
                    self.KQ_to_Z(self.k, self.q)
                    self.KQ_to_Z(self.q, self.k)      

    def KQ_to_Z(self, ctx_1, ctx_2):
        K = self.active_K[:,ctx_1]
        Q = self.active_Q[:,ctx_2]
        Z = (K * Q).sum(-1)
        self.active_Z[:,:, self.Z_count] = Z
        self.Z_count += 1 

    def default_classification(self):
        inp = self.forward_readin()
        if self.testing:
            self.input_flat = inp.detach()        

        stm = self.STM.expand(-1, self.batch_num, -1).contiguous()
        ltm = self.LTM.expand(-1, self.batch_num, -1).contiguous()
        update, _ = self.LSTM(inp, (stm, ltm))     
        if self.testing:
            self.update_flat = update.detach()        

        belief = self.forward_readout(update)
        self.postprocess_belief(belief)

    def forward_readin(self):
        Z = self.active_Z.detach()
        Z = Z.reshape(self.batch_num, 1, -1)
        Z = Z.expand(-1, self.step_num, -1)
        inp = torch.cat((self.obs_flat, Z), dim = -1)
        inp = torch.relu(self.classifier_readin(inp))
        return inp

    def forward_readout(self, update):
        update = self.classifier_readout(update)
        update = update.reshape(self.BSCR_dims)
        belief = update.cumsum(1)
        belief = torch.softmax(belief, -1) 
        return belief

    def postprocess_belief(self, belief):
        self.classifier_belief_flat = belief.detach()
        self.classifier_goal_belief = belief[self.batch_range, :, self.goal_ind]
        self.classifier_goal_selection = Categorical(self.classifier_goal_belief).sample()
        self.ACC = (self.classifier_goal_selection == self.goal_value[:,None]).float() 
    
    def default_pobs(self, train_controller = False):
        if self.learn_embeddings or train_controller: 
            CBF = self.classifier_belief_flat[:, -1]
            sample = CBF.reshape(self.batch_num, -1)
            sample = torch.distributions.Categorical(probs = sample).sample()
            sample = torch.stack(torch.unravel_index(sample, self.ctx_dims),1)
            sample[self.batch_range, self.goal_ind] = self.classifier_goal_selection[:, -1]

            if train_controller:
                conf = torch.ones(*self.batch_ctx_dims, device = self.device)
                sample = self.controller_actions
            else:
                conf = CBF[self.BR, self.CR, sample]
                conf[self.batch_range, self.goal_ind] = self.ACC[:,-1]
                
            sample_emb = self.sample_to_emb(sample).reshape(self.batch_num, -1)
            s = self.sample_to_hid(sample_emb).unsqueeze(1)
            c = self.conf_to_hid(conf).unsqueeze(1)
            z = self.Z_to_pobs(self.active_Z)
            x = self.hid_to_pobs(torch.relu(s + c + z))
            self.pred_pobs = torch.sigmoid(x).squeeze()
