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
        if self.subsample_states is not None:
            train = np.random.choice(train, size=self.subsample_states, replace=False)
            test = np.random.choice(test, size=self.subsample_states, replace=False)

        # allow_repeated_vars = False (option, as in coggrid's CogGridConfig; default True = the behaviour of every run so
        # far, drawn exactly as before): no episode activates the same variable twice. A repeated pair has a symmetric
        # likelihood table, so its goal is not identifiable (11-10-2026).
        distinct = not getattr(self, "allow_repeated_vars", True)
        if self.test_set:
            ctx = self._draw_vars(test, B_C, distinct)                       # repeats allowed unless distinct
        else:
            ctx = self._draw_vars(S, B_C, distinct)
            tmask = np.isin(ctx, test)                                       # locations that are in `test`
            keep = np.zeros_like(tmask, dtype=bool)
            first_t = tmask.argmax(axis=1)                                   # 0 if none; gated by `has_t`
            has_t = tmask.any(axis=1)
            keep[self.batch_range, first_t] = has_t                          # keep ≤1 test-state per row
            replace_mask = tmask & ~keep
            nrep = int(replace_mask.sum())
            if not distinct:
                ctx[replace_mask] = np.random.choice(train, size=nrep)       # remove extra test-states
            else:                                                            # ... without repeating a row's other variables
                for b in np.flatnonzero(replace_mask.any(axis=1)):
                    pool = np.setdiff1d(train, ctx[b, ~replace_mask[b]])
                    ctx[b, replace_mask[b]] = np.random.choice(pool, size=int(replace_mask[b].sum()), replace=False)
        self.ctx_inds = np.sort(ctx, axis=1)

        rng = np.random.rand(*B_C)
        if not self.test_set:
            invalid = np.isin(self.ctx_inds, test)                           # train goal must be from `train`
            rng[invalid] = 0.0                          
        self.goal_ind = rng.argmax(axis=1)

    @staticmethod
    def _draw_vars(pool, B_C, distinct):
        """(batch, ctx) variables from pool: with replacement (np.random.choice, as always), or, if distinct, without
        replacement within each row (coggrid's generative._draw: argsort of uniform draws)"""
        if not distinct:
            return np.random.choice(pool, size=B_C)
        pool = np.asarray(pool)
        if B_C[1] > pool.size:
            raise ValueError(f"cannot draw {B_C[1]} distinct variables from a pool of {pool.size}")
        order = np.argsort(np.random.random((B_C[0], pool.size)), axis=1)
        return pool[order[:, :B_C[1]]]

    def default_likelihoods(self):
        self.naive_likelihood = np.zeros((*self.batch_obs_dims, *self.ctx_realization_dims))
        self.joint_likelihood, self.joint_Z = self.trigger_build_likelihood()     

        for c in self.ctx_range:                                               # Get naive by marginalizing joint likelihood
            self.naive_likelihood[:,:, c, :] = self.avg_over_ctx(
            self.joint_likelihood, keep = c, includes_obs = True)              # Dims: batch, obs pairs, ctx, realizations                

    def default_realizations(self):
        self.ctx_vals = np.random.choice(self.realization_range, size=self.batch_ctx_dims)
        if getattr(self, "fr_select", None) and not self.test_set:
            self.select_fr_realizations()
        self.goal_value = self.ctx_vals[self.batch_range, self.goal_ind]      

    def naive_consistent_cells(self):
        """(B, R**C) bool on self.device: at each joint realization, the Naive observer's asymptotic goal estimate (the goal
        value whose per-channel marginal likelihood is KL-closest to the true observation distribution, uniquely) is the
        true goal value, so its FR vanishes as t grows; elsewhere FR grows linearly. Also returns the (C, R**C) coordinates."""
        import torch
        B, R, C, O, dev = self.batch_num, self.realization_num, self.ctx_num, self.obs_num, self.device
        L = torch.as_tensor(self.joint_likelihood, device=dev, dtype=torch.float32).reshape(B, O, -1).clamp(1e-6, 1 - 1e-6)
        g = torch.as_tensor(self.goal_ind, device=dev).long()
        m = torch.as_tensor(self.naive_likelihood, device=dev, dtype=torch.float32)[torch.arange(B, device=dev), :, g]
        m = m.clamp(1e-6, 1 - 1e-6)                                                     # (B, O, R): goal slot's marginals
        kappa = -(torch.einsum('boc,bog->bcg', L, m.log()) + torch.einsum('boc,bog->bcg', 1 - L, (1 - m).log()))
        best = kappa.topk(2, dim=-1, largest=False)                                     # KL up to a per-cell constant
        coords = torch.stack(torch.unravel_index(torch.arange(R ** C, device=dev), (R,) * C), 0)
        ok = (best.indices[..., 0] == coords[g]) & (best.values[..., 1] - best.values[..., 0] > 1e-7)
        return ok, coords

    def select_fr_realizations(self):
        """redraw each training episode's joint realization uniformly from the cells of its fr_select class; an environment
        with no cell of the class keeps an unrestricted draw (counted in self.fr_select_fallback)"""
        import torch
        ok, coords = self.naive_consistent_cells()
        if self.fr_select == "low":
            allowed = ok
        elif self.fr_select == "high":
            allowed = ~ok
        elif self.fr_select == "half":                                                  # fixed per (pair, goal slot, cell)
            ctx = torch.as_tensor(self.ctx_inds, device=ok.device).long()
            g = torch.as_tensor(self.goal_ind, device=ok.device).long()
            cell = torch.arange(ok.shape[1], device=ok.device)
            h = (ctx[:, :1] * 73856093) ^ (ctx[:, 1:2] * 19349663) ^ (g[:, None] * 83492791) ^ (cell[None] * 2654435761)
            allowed = (h % 1000003) % 2 == 0
        else:
            raise ValueError(f"fr_select {self.fr_select!r}")
        none = ~allowed.any(1)
        allowed[none] = True
        self.fr_select_fallback = int(none.sum())
        u = torch.rand(allowed.shape, device=allowed.device).masked_fill(~allowed, -1.0)
        self.ctx_vals = coords[:, u.argmax(1)].T.cpu().numpy()

    def default_gen_observations(self):
        ix = tuple(self.ctx_vals[:, i, None] for i in range(self.ctx_num)) 
        p_x = self.joint_likelihood[self.batch_range[:, None], self.obs_range[None, :], *ix] 
        self.pobs__joint = np.squeeze(p_x)   
        if self.obs_num == 1:
            self.pobs__joint = self.pobs__joint[:, None]
        rand = np.random.rand(*self.batch_step_dims, self.obs_num) 
        self.obs_flat = (self.pobs__joint[:, None, :] > rand).astype(bool)
        # Generates observations based on the joint realizations and random values.