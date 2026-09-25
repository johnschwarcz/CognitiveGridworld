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

    def classifier_Z(self):
        return self.active_Z if self.embedding_grad in ("classifier", "both") else self.active_Z.detach()

    def generator_Z(self):
        return self.active_Z if self.embedding_grad in ("generator", "both") else self.active_Z.detach()

    def KQ_to_Z(self, ctx_1, ctx_2):
        K = self.active_K[:,ctx_1]
        Q = self.active_Q[:,ctx_2]
        Z = (K * Q).sum(-1)
        self.active_Z[:,:, self.Z_count] = Z
        self.Z_count += 1 

    def default_classification(self):
        if self.mode == "FF":
            self.FF_classification()
        elif self.mode == "lazyrich":
            self.lazyrich_classification()
        else:
            self.RNN_classification()

    def FF_classification(self):
        Z = self.classifier_Z()
        Z = Z.reshape(self.batch_num, 1, -1)
        O = self.obs_flat.mean(1, keepdims = True)
        inp = torch.cat((O, Z), dim = -1)
        inp = torch.relu(self.classifier_readin(inp))
        update = self.classifier_readout(inp)
        update = update.expand(-1, self.step_num, -1)
        self.postprocess_belief(update, accumulate = False)
    
    def RNN_classification(self):
        inp = self.RNN_readin()
        stm = self.STM.expand(-1, self.batch_num, -1).contiguous()
        ltm = self.LTM.expand(-1, self.batch_num, -1).contiguous()
        update, _ = self.LSTM(inp, (stm, ltm))   
        self.postprocess_belief(self.classifier_readout(update))

        if self.testing and (self.mode == "SANITY"):
            self.update_flat = update.detach()        
            self.input_flat = inp.detach()    
        else:
            self.update_flat = torch.zeros(1, device = self.device)    
            self.input_flat = torch.zeros(1, device = self.device)

    def RNN_readin(self):
        Z = self.classifier_Z()
        Z = Z.reshape(self.batch_num, 1, -1)
        Z = Z.expand(-1, self.step_num, -1)
        inp = torch.cat((self.obs_flat, Z), dim = -1)
        inp = torch.relu(self.classifier_readin(inp))
        return inp

    def postprocess_belief(self, logits, accumulate = True):
        if accumulate:
            logits = logits.cumsum(1)
        belief = self.logits_to_belief(logits)
        self.classifier_belief = belief
        self.classifier_belief_flat = belief.detach()
        self.classifier_goal_belief = belief[self.batch_range, :, self.goal_ind]
        self.classifier_goal_selection = Categorical(self.classifier_goal_belief).sample() # SAMPLES FROM MARGINAL BELIEF
        self.ACC = (self.classifier_goal_selection == self.goal_value[:,None]).float() 

    def logits_to_belief(self, logits):
        if self.output_joint:
            belief = torch.softmax(logits, -1)
            B, S, R, C = self.batch_num, self.step_num, self.realization_num, self.ctx_num
            return torch.stack([belief.reshape(B, S, R**c, R, R**(C - 1 - c)).sum((2, 4)) for c in range(C)], dim = 2)            

        logits = logits.reshape(self.BSCR_dims)
        return torch.softmax(logits, -1)
        
    def default_pobs(self, training_controller = False):
        if self.learn_embeddings or training_controller: 
            if training_controller or self.mode == "oracle":
                conf = torch.ones(*self.batch_ctx_dims, device = self.device)
                sample = self.controller_actions if training_controller else self.ctx_vals
            else:
                CBF = self.classifier_belief_flat[:, -1]
                sample = torch.distributions.Categorical(probs=CBF).sample()                 # SAMPLES FROM JOINT BELIEF                 
                sample[self.batch_range, self.goal_ind] = self.classifier_goal_selection[:, -1]
                conf = CBF[self.BR, self.CR, sample]
                conf[self.batch_range, self.goal_ind] = self.ACC[:,-1]
 
            self.get_prediction(sample, conf)
                
    def get_prediction(self, sample, conf):
        sample_emb = self.sample_to_emb(sample).reshape(self.batch_num, -1)
        s = self.sample_to_hid(sample_emb).unsqueeze(1)
        c = self.conf_to_hid(conf).unsqueeze(1)
        z = self.Z_to_pobs(self.generator_Z())

        if self.gen_fix:
            s, c, z = (n(v) for n, v in zip(self.gen_stream_norm, (s, c, z)))

        pre = s + c + z
        x = self.gen_hid2hid(torch.relu(pre))
        
        if self.gen_fix:
            x = self.hid_to_pobs(torch.relu(pre + x))
        else:
            x = self.hid_to_pobs(torch.relu(x))
            
        self.pred_pobs = torch.sigmoid(x).squeeze()

    ########################################################################################################
    """ lazy/rich RNN — Clark, Bordelon, Zavatone-Veth & Pehlevan, bioRxiv 2026.03.02.708943 """
    ########################################################################################################

    def lazyrich_phi(self, x):
        if self.rnn_nonlinearity == "tanh":     return torch.tanh(x)
        if self.rnn_nonlinearity == "relu":     return torch.relu(x)
        if self.rnn_nonlinearity == "linear":   return x
        raise ValueError(f"unknown rnn_nonlinearity: {self.rnn_nonlinearity}")

    def lazyrich_classification(self):
        """
        Euler integration of Eqs. (SI.1)-(SI.3) with alpha = dt/tau:

            x(t_1)  = x^0
            x(t_n)  = (1-alpha) x(t_{n-1}) + alpha [ (g/sqrt(N)) J phi(x(t_{n-1})) + U I(t_{n-1}) ]
            y(t_n)  = (1/(N gamma)) V^T phi(x(t_n))

        Step t consumes observation t and reads out the state it produces, so y[t] sees
        observations 0..t -- the same causal alignment as the LSTM path, keeping accuracy, TP
        and MSE comparable across modes. The readout emits belief *increments*, cumsum-ed
        before the softmax exactly as in postprocess_belief.

        The state stays fp32 under autocast: gamma's whole point is a small but coherent
        perturbation to J, and the recurrence is where that would be lost.
        """
        I = self.lazyrich_readin()                                              # (B, T, D_in)
        drive = self.rnn_input_scale * torch.einsum('bti,ni->btn', I, self.rnn_U)

        testing = bool(self.testing)
        x = torch.full((self.batch_num, self.hid_dim), float(self.rnn_x0), device = self.device)
        phi = self.lazyrich_phi(x)                                              # phi(x^0), not read out
        phis, preacts = [], []
        for t in range(self.step_num):
            x = (1 - self.rnn_alpha) * x + self.rnn_alpha * (
                self.rnn_J_scale * (phi @ self.rnn_J.T) + drive[:, t])
            phi = self.lazyrich_phi(x)
            phis.append(phi)
            if testing:
                preacts.append(x)                                               # only kept for the diagnostics below

        phi_flat = torch.stack(phis, 1)                                         # (B, T, N)
        update = self.rnn_V_scale * (phi_flat @ self.rnn_V)                     # Eq. (2)
        self.postprocess_belief(update)

        if testing:
            # phi is what the DMFT's C(t,t') = (1/N) sum_i phi_i(t) phi_i(t') is built from, so
            # it is the right analogue of the LSTM hidden state for the existing PR/PCA code.
            # .float(): a (B,T,N) fp16 array overflows any downstream variance/PCA reduction.
            self.update_flat = phi_flat.detach().float()
            self.input_flat = drive.detach().float()
            self.lazyrich_preact = torch.stack(preacts, 1).detach().float()     # x(t), for Figs. 3J / 4G
        else:
            self.update_flat = torch.zeros(1, device = self.device)
            self.input_flat = torch.zeros(1, device = self.device)

    def lazyrich_readin(self):
        """Raw I_a(t): observation bits plus the (time-constant) interaction scalars Z. No
        learned projection or nonlinearity -- U is the only thing between task and preactivation."""
        Z = self.classifier_Z().reshape(self.batch_num, 1, -1).expand(-1, self.step_num, -1)
        return torch.cat((self.obs_flat, Z), dim = -1)
