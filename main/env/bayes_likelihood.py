from .bayes_inference import Bayes_inference
import numpy as np

class Bayes_likelihood(Bayes_inference):

    def trigger_build_likelihood(self):
        if self.ctx_num >= 3 and getattr(self, "gpu_inference", False) and self.device.type == "cuda":
            return self.build_likelihood_gpu()
        L = self.preprocess_likelihood()
        if self.ctx_num == 1:
            L = L + self.get_V(0,0)
        else:
            for self.k in self.ctx_range:
                for self.q in range(self.k+1, self.ctx_num):     
                    L = L + self.add_pairwise_interaction()
        return self.sigmoid(L), self.curr_Z

    def build_likelihood_gpu(self):
        # C >= 3 with gpu_inference (2026-11-12): the same table, summed and squashed on the GPU in float64. The R**C table
        # per episode is 10x C = 2's, and building it in numpy took ~2/3 of a C = 3 training iteration. Same pair order,
        # same products and sums (exact); only exp may differ from numpy's in the last bit. C <= 2 never comes here.
        import torch
        B, O, R, C = self.batch_num, self.obs_num, self.realization_num, self.ctx_num
        self.Z_count = 0
        self.active_K = self.all_K[self.ctx_inds, :]
        self.active_Q = self.all_Q[self.ctx_inds, :]
        self.curr_Z = np.zeros(self.Z_dims)
        L = torch.zeros(self.L_dims, dtype=torch.float64, device=self.device)
        for self.k in self.ctx_range:
            for self.q in range(self.k+1, self.ctx_num):
                V1, V2 = self.get_V_gpu(self.k, self.q), self.get_V_gpu(self.q, self.k)
                L += (V1[..., :, None] * V2[..., None, :]).reshape(B, O, *[R if i in (self.k, self.q) else 1 for i in range(C)])
        L = 1 / (1 + torch.exp(-L))
        out = L.cpu().numpy()
        self.gpu_table = (out, L)           # default_likelihoods / gpu_belief reuse the GPU copy of exactly this table
        return out, self.curr_Z

    def get_V_gpu(self, ctx_1, ctx_2):
        # get_V with soft_roll on the GPU (float64; Z and its bookkeeping in numpy as in get_V)
        import torch
        K = self.active_K[:,ctx_1]
        Q = self.active_Q[:,ctx_2]
        Z = np.einsum('boc,boc->bo', K, Q)
        self.curr_Z[:,:, self.Z_count] = Z
        self.Z_count += 1
        Z = (torch.as_tensor(self.roll_realization_range, device=self.device)
             - torch.as_tensor(Z, device=self.device)[..., None] * self.V_num) % self.V_num
        Z = Z[..., None, :]
        distance = torch.minimum(Z, self.V_num - Z)
        exp_neg_distance = torch.exp(-(distance**2))
        P = exp_neg_distance / exp_neg_distance.sum(-1, keepdim=True)
        return (P * torch.as_tensor(self.roll_V_range, device=self.device)).sum(-1)

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

    def get_Z(self, ctx_1, ctx_2):
        """get_V's interaction scalar Z (batch, obs), stored in curr_Z the same way, without the soft roll"""
        K = self.active_K[:,ctx_1]
        Q = self.active_Q[:,ctx_2]
        Z = np.einsum('boc,boc->bo', K, Q)
        self.curr_Z[:,:, self.Z_count] = Z
        self.Z_count += 1
        return Z

    def soft_roll_at(self, Z, r):
        """soft_roll(Z)[b, o, r[b]]: one realization per episode, by soft_roll's element-wise operations and the same sum of
        2V+1 terms per entry (Env_generators.realized_observations; bitwise equal to the table's entry)"""
        Z = (self.roll_realization_range - Z[..., None] * self.V_num) % self.V_num             # (B, O, V)
        distance = np.minimum(Z,  self.V_num - Z)
        exp_neg_distance = np.exp(-(distance**2))
        P = exp_neg_distance / exp_neg_distance.sum(axis=-1, keepdims=True)
        return np.sum(P * self.roll_V_range[r][:, None, :], axis=-1)                           # (B, O)
    
    def soft_roll(self, Z):
        Z = (self.roll_realization_range - Z[..., None] * self.V_num) % self.V_num
        Z = Z[..., None, :]
        distance = np.minimum(Z,  self.V_num - Z)
        exp_neg_distance = np.exp(-(distance**2))
        P = exp_neg_distance / exp_neg_distance.sum(axis=-1, keepdims=True)
        return np.sum(P * self.roll_V_range, axis=-1)