from .bayes_inference import Bayes_inference
import numpy as np

class Bayes_likelihood(Bayes_inference):

    def trigger_build_likelihood(self):
        L = self.preprocess_likelihood()
        if self.ctx_num == 1:
            L = L + self.get_V(0,0)
        else:
            for self.k in self.ctx_range:
                for self.q in range(self.k+1, self.ctx_num):     
                    L = L + self.add_pairwise_interaction()
        return self.sigmoid(L), self.curr_Z

    def preprocess_likelihood(self):
        self.Z_count = 0
        self.active_K = self.all_K[self.ctx_inds, :]
        self.active_Q = self.all_Q[self.ctx_inds, :]
        self.curr_Z = np.zeros(self.Z_dims)
        return np.zeros(self.L_dims)
        
    def add_pairwise_interaction(self):
        V1 = self.get_V(self.k, self.q)
        V2 = self.get_V(self.q, self.k)                  
        V = np.einsum('...a,...b->...ab', V1, V2)                            # Outer product

        for i in self.ctx_range:
            if i < self.k:
                V = V[..., None, :, :]
            if i > self.q:
                V = V[..., None]
            if i > self.k and i < self.q:
                V = V[..., None, :]
        return V
    
    def get_V(self, ctx_1, ctx_2):
        K = self.active_K[:,ctx_1]
        Q = self.active_Q[:,ctx_2]
        Z = np.einsum('boc,boc->bo', K, Q)                                 # Inner product
        V = self.soft_roll(Z)
        self.curr_Z[:,:, self.Z_count] = Z 
        self.Z_count += 1 
        return V                                                               # Dims: batch, obs, realizations
    
    def soft_roll(self, Z):
        Z = (self.roll_realization_range - Z[..., None] * self.V_num) % self.V_num
        Z = np.tile(Z[..., None, :], (1, 1,  self.realization_num, 1))
        distance = np.minimum(Z,  self.V_num - Z)
        exp_neg_distance = np.exp(-(distance**2))
        P = exp_neg_distance / exp_neg_distance.sum(axis=-1, keepdims=True)
        return np.sum(P * self.roll_V_range, axis=-1)