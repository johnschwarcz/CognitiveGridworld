import numpy as np; from main.utils import print_time; 
from main.env_plotting.plotters_EP import Plotters

class Bayes_inference(Plotters):

    @print_time()
    def run_inference(self):
        self.joint_belief, self.joint_goal_belief,\
        self.joint_est, self.joint_acc, self.joint_TP, self.joint_mse\
            = self.forward_inference(self.joint_likelihood)

        self.naive_belief,  self.naive_goal_belief,\
        self.naive_est, self.naive_acc, self.naive_TP, self.naive_mse\
            = self.forward_inference(self.naive_likelihood, naive = True)

        self.log_outcomes()
            
    def forward_inference(self, L, naive = False):
        dist_flat = np.zeros(L.shape).sum(1, keepdims = True).repeat(self.step_num, 1)
        p_x = np.einsum('bto, bo...->bt...', self.obs_flat, np.log(L)) + \
              np.einsum('bto, bo...->bt...', 1-self.obs_flat, np.log1p(-L))
        p_x = np.exp(p_x)

        for t in self.step_range:
            p = p_x[:, t] 
            
            if t > 0:
                p = p * dist_flat[:, t-1]

            dist_flat[:, t] = p / self.avg_until(p, override = "sum", stop_shape = 2 if naive else 1)    

        return self.postprocess_belief(dist_flat, naive)

    def postprocess_belief(self, dist_flat, naive):                            # Functions in inference_helpers
        belief = dist_flat if naive else self.marginalize(dist_flat)
        est, goal_belief, acc, TP, mse = self.get_goal_performance(belief)      
        return belief, goal_belief, est, acc, TP, mse
     
    def marginalize(self, p_x__s):
        marginalized = np.zeros(self.BSCR_dims)                         
        for c in self.ctx_range:
            marginalized[:,:,c,:] = self.avg_over_ctx(p_x__s, keep=c, override='sum')  
        return marginalized                                                    # Dims: batch, time, ctx, realizations        

    def get_goal_performance(self, belief):
        R = self.realization_range[None, None, None,:]
        I = self.goal_ind[:,None,None]
        V = self.goal_value[:,None]
        est = (belief * R).sum(-1)

        GB = np.take_along_axis(belief, I[:,None], 2).squeeze()
        G_est_max = np.argmax(GB, axis = -1) 
        acc = (G_est_max == V).astype(float)
        
        TP = np.take_along_axis(GB, V[...,None], -1).squeeze()        
        E = np.take_along_axis(est, I, -1)
        mse = ((V - E.squeeze())**2)  
        return est, GB, acc, TP, mse

    def log_outcomes(self):
        self.agent_beliefs = [self.joint_belief, self.naive_belief]
        self.mse_diff = self.joint_mse - self.naive_mse
        self.agent_ests = [self.joint_est, self.naive_est]
        self.agent_accs = [self.joint_acc, self.naive_acc]
        self.agent_TPs = [self.joint_TP, self.naive_TP]
        self.agent_mses = [self.joint_mse, self.naive_mse]