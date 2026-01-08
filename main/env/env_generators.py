import numpy as np; import pylab as plt; 
from main.utils import print_time; from .env_helpers import Env_helpers; 

class Env_generators(Env_helpers):
    @print_time()
    def run_generators(self):
        self.EC_gen_context()
        self.EC_gen_likelihoods()
        self.EC_gen_realizations()
        self.EC_gen_observations()

    def default_state_embeddings(self):         
        self.all_K = self.gram_shmidt_KQ()
        self.all_Q = self.gram_shmidt_KQ()   

    def gram_shmidt_KQ(self):
        v = np.random.randn(*self.state_obs_dims, self.KQ_dim)
        for o in self.obs_range:
            # Orthogonalize across observations
            for prev in range(o):
                curr_v = v[:, o]
                prev_v = v[:, prev]
                proj_on_prev = np.einsum('sk,sk->s', curr_v, prev_v)
                v[:, o] -= proj_on_prev[..., None] * prev_v

            v[:,o] /= np.linalg.norm(v[:, o], axis = -1, keepdims = True)
        return v

    def default_context(self):
        B_C = (self.batch_num, self.ctx_num)
        S = self.state_range
        T = self.test_states
        train = S[T:]
        test = S[:T]

        if self.test_set:
            ctx = np.random.choice(test, size=B_C)             # repeats allowed
        else:
            ctx = np.random.choice(S, size=B_C)
            tmask = np.isin(ctx, test)                                       # locations that are in `test`
            keep = np.zeros_like(tmask, dtype=bool)
            first_t = tmask.argmax(axis=1)                                   # 0 if none; gated by `has_t`
            has_t = tmask.any(axis=1)
            keep[self.batch_range, first_t] = has_t                          # keep ≤1 test-state per row
            replace_mask = tmask & ~keep
            nrep = int(replace_mask.sum())
            ctx[replace_mask] = np.random.choice(train, size=nrep)           # remove extra test-states
        self.ctx_inds = np.sort(ctx, axis=1)

        rng = np.random.rand(*B_C)
        if not self.test_set:
            invalid = np.isin(self.ctx_inds, test)                           # train goal must be from `train`
            rng[invalid] = 0.0                          
        self.goal_ind = rng.argmax(axis=1)

    def default_likelihoods(self):
        self.naive_likelihood = np.zeros((*self.batch_obs_dims, *self.ctx_realization_dims))
        self.joint_likelihood, self.joint_Z = self.trigger_build_likelihood()     

        for c in self.ctx_range:                                               # Get naive by marginalizing joint likelihood
            self.naive_likelihood[:,:, c, :] = self.avg_over_ctx(
            self.joint_likelihood, keep = c, includes_obs = True)              # Dims: batch, obs pairs, ctx, realizations                

    def default_realizations(self):
        self.ctx_vals = np.random.choice(self.realization_range, size=self.batch_ctx_dims)
        self.goal_value = self.ctx_vals[self.batch_range, self.goal_ind]      

    def default_gen_observations(self):
        ix = tuple(self.ctx_vals[:, i, None] for i in range(self.ctx_num)) 
        p_x = self.joint_likelihood[self.batch_range[:, None], self.obs_range[None, :], *ix] 
        self.pobs__joint = np.squeeze(p_x)   
        if self.obs_num == 1:
            self.pobs__joint = self.pobs__joint[:, None]
        rand = np.random.rand(*self.batch_step_dims, self.obs_num) 
        self.obs_flat = (self.pobs__joint[:, None, :] > rand).astype(bool)
        # Generates observations based on the joint realizations and random values.