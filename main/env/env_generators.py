import numpy as np; import pylab as plt; 
from main.utils import print_time; from .env_helpers import Env_helpers; 

class Env_generators(Env_helpers):
    @print_time()
    def run_generators(self):
        self.EC_gen_context()
        if self.realized_likelihood_ok():
            self.realized_likelihoods()     # interaction scalars now; the full tables only if read (lazy)
            self.EC_gen_realizations()
            self.realized_observations()    # p(o | r) at each episode's realized r only
            return
        self.EC_gen_likelihoods()
        self.drop_lazy([n for n in ("joint_likelihood", "naive_likelihood", "active_K", "active_Q") if n in self.__dict__])
        self.EC_gen_realizations()                                             # (^ dead entries of an earlier realized episode)
        self.EC_gen_observations()

    # Realized-cell likelihood (2026-11-13, speed; bitwise identical). A training iteration that runs no Bayes inference
    # (skip_inference: modes RL / oracle) needs the likelihood only at each episode's realized joint value, to draw the
    # observations; the full (B, O, R**C) joint table, its Naive marginals, were built and then left unread on most
    # iterations (~14 ms of ~15 ms of run_generators at batch 1,000, 168 of 178 ms at 10,000). Here the interaction scalars
    # curr_Z / joint_Z are the same numbers as before, and the realized cell's p(o | r) = sigmoid(0 + V1 V2) is formed from
    # the same V entries: looked up in pair_tables() (every ordered variable pair's Z and V, built once by get_V's own
    # einsum and soft_roll), or, if those would be too large, rolled per episode at the realized value only (soft_roll_at:
    # the same element-wise operations and the same sum of 2V+1 terms per entry). joint_likelihood / naive_likelihood (and,
    # with the pair tables, active_K / active_Q) are lazy (Env_helpers.set_lazy): a reader (plugin(), naive_consistent_cells,
    # a custom logger) gets what default_likelihoods() builds, bit for bit. The likelihood draws no random numbers, so the
    # numpy stream (context -> realizations -> observations) is unchanged. Used only where the default methods are in force.
    # Measured (CPU, 1 thread): run_generators 14.3 -> 1.5 ms per training iteration at batch 1,000, 191 -> 12 ms at 10,000.
    _REALIZED_DEFAULTS = ("EC_gen_likelihoods", "EC_gen_observations", "custom_likelihoods", "custom_gen_observations",
                          "default_likelihoods", "default_gen_observations", "trigger_build_likelihood",
                          "preprocess_likelihood", "add_pairwise_interaction", "get_V", "soft_roll", "sigmoid")

    def realized_likelihood_ok(self):
        if self.test_set or not getattr(self, "skip_inference", False) or self.ctx_num < 2 or getattr(self, "fr_select", None):
            return False
        if self.ctx_num >= 3 and getattr(self, "gpu_inference", False) and self.device.type == "cuda":
            return False                                                       # build_likelihood_gpu's table (torch exp)
        if not (isinstance(self.__dict__.get("all_K"), np.ndarray) and isinstance(self.__dict__.get("all_Q"), np.ndarray)):
            return False
        return all(getattr(type(self), m) is _DEFAULT_METHODS[m] and m not in self.__dict__ for m in self._REALIZED_DEFAULTS)

    # what default_likelihoods() reads that preprocess_env() / load() replace (new objects, never edited in place): the lazy
    # tables are built from the values in force when the episode was generated, even if they changed before the read
    _TABLE_INPUTS = ("all_K", "all_Q", "roll_V_range", "roll_realization_range", "V_num", "ctx_inds", "ctx_range", "ctx_num",
                     "Z_dims", "L_dims", "batch_obs_dims", "ctx_realization_dims", "pre_ctx_dims", "obs_num", "gpu_inference")

    def realized_likelihoods(self):
        ctx_inds = self.ctx_inds
        tab = self.pair_tables()
        pairs = []
        if tab is None:
            self.preprocess_likelihood()                                       # Z_count, active_K / active_Q, curr_Z
            for self.k in self.ctx_range:                                      # get_V's order and arithmetic, without soft_roll
                for self.q in range(self.k+1, self.ctx_num):
                    pairs.append((self.k, self.q, self.get_Z(self.k, self.q), self.get_Z(self.q, self.k)))
        else:                                                                  # the same Z and V, looked up per variable pair
            Ztab = tab["Z"]
            self.Z_count = 0
            self.curr_Z = np.zeros(self.Z_dims)
            for self.k in self.ctx_range:
                for self.q in range(self.k+1, self.ctx_num):
                    i, j = ctx_inds[:, self.k], ctx_inds[:, self.q]
                    for a, b in ((i, j), (j, i)):
                        self.curr_Z[:,:, self.Z_count] = Ztab[a, b]
                        self.Z_count += 1
                    pairs.append((self.k, self.q, (i, j), (j, i)))
            K, Q = self.all_K, self.all_Q
            self.set_lazy(("active_K", "active_Q"), lambda env: dict(active_K = K[ctx_inds, :], active_Q = Q[ctx_inds, :]))
        self.joint_Z = self.curr_Z
        self._realized_pairs = (pairs, tab)
        inputs = {n: self.__dict__[n] for n in self._TABLE_INPUTS if n in self.__dict__}
        self.set_lazy(("joint_likelihood", "naive_likelihood"), lambda env: env.realized_tables(inputs))

    def realized_tables(self, inputs):
        """default_likelihoods() on an episode's variables, under the inputs in force when it was generated; the env's
        eager state (curr_Z, Z_count, k, q, joint_Z, active_K / active_Q, the inputs) is left as it was"""
        names = set(inputs) | {"active_K", "active_Q", "curr_Z", "Z_count", "k", "q", "joint_Z", "gpu_table"}
        keep = {n: self.__dict__[n] for n in names if n in self.__dict__}
        self.__dict__.update(inputs)
        self.__dict__.pop("gpu_table", None)
        try:
            self.default_likelihoods()
            return dict(joint_likelihood=self.__dict__.pop("joint_likelihood"),
                        naive_likelihood=self.__dict__.pop("naive_likelihood"))
        finally:
            for n in names:
                self.__dict__.pop(n, None)
            self.__dict__.update(keep)

    def realized_observations(self):
        pairs, tab = self._realized_pairs
        self._realized_pairs = None
        L = np.zeros(self.batch_obs_dims)                                      # L = 0 + V1 V2 (+ ...), as the table
        o = self.obs_range[None, :]
        for k, q, Z1, Z2 in pairs:
            rk, rq = self.ctx_vals[:, k, None], self.ctx_vals[:, q, None]
            if tab is None:
                V1, V2 = self.soft_roll_at(Z1, rk[:, 0]), self.soft_roll_at(Z2, rq[:, 0])
            else:
                V1, V2 = tab["V"][Z1[0][:, None], Z1[1][:, None], o, rk], tab["V"][Z2[0][:, None], Z2[1][:, None], o, rq]
            L = L + V1 * V2
        self.observations_from(self.sigmoid(L))

    PAIR_TABLE_MB = 256                     # pair_tables() only below this size (|S| 500, O 5, R 10: 95 MB + 9.5 MB)

    def pair_tables(self):
        """Z (|S|, |S|, O) and V (|S|, |S|, O, R) of every ordered variable pair under the env's own embeddings (all_K, all_Q),
        by get_V's arithmetic (the same einsum over the KQ dimension, the same soft_roll, the same dtypes: Z in the einsum's
        dtype, e.g. float32 for float32 embeddings, V in soft_roll's), so a lookup equals get_V's entry bit for bit (checked
        on CPU against the full tables, float64 and float32 embeddings). Built once (~1.6 s at |S| 500); rebuilt whenever
        anything get_V reads changes: all_K / all_Q (values, dtype, shape: load() replaces them), roll_V_range
        (likelihood_temp, likelihood_freq, realization_num), roll_realization_range, V_num. None if larger than
        PAIR_TABLE_MB (realized_observations then rolls per episode)."""
        K, Q = self.all_K, self.all_Q
        S, O = K.shape[:2]
        R = self.roll_V_range.shape[0]
        if S * S * O * (R + 1) * 8 > self.PAIR_TABLE_MB * 2**20:
            return None
        rr, rV, Vn = self.roll_realization_range, self.roll_V_range, self.V_num
        key = (K.dtype.str, K.shape, Q.dtype.str, Q.shape, type(Vn).__name__, Vn, rr.dtype.str, rr.shape, rr.tobytes(),
               rV.dtype.str, rV.shape, rV.tobytes())
        tab = self.__dict__.get("_pair_tables")
        if tab is not None and tab["key"] == key and np.array_equal(tab["K"], K) and np.array_equal(tab["Q"], Q):
            return tab                                                         # (~0.05 ms: values, not identity, compared)
        zdt = np.einsum('boc,boc->bo', K[:1], Q[:1]).dtype                     # get_V's Z, then soft_roll's V
        vdt = self.soft_roll(np.zeros((1, O), zdt)).dtype
        Ztab, Vtab = np.zeros((S * S, O), zdt), np.zeros((S * S, O, R), vdt)
        n = max(1, 4000 // S) * S
        for s in range(0, S * S, n):
            I, J = np.divmod(np.arange(s, min(s + n, S * S)), S)
            Ztab[s:s + n] = np.einsum('boc,boc->bo', K[I], Q[J])
            Vtab[s:s + n] = self.soft_roll(Ztab[s:s + n])
        tab = dict(key=key, K=K.copy(), Q=Q.copy(), Z=Ztab.reshape(S, S, O), V=Vtab.reshape(S, S, O, R))
        self._pair_tables = tab
        return tab

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

        gpu = getattr(self, "gpu_table", None)                                 # C >= 3 (bayes_likelihood.build_likelihood_gpu)
        if gpu is not None and gpu[0] is self.joint_likelihood:
            for c in self.ctx_range:
                self.naive_likelihood[:,:, c, :] = gpu[1].mean(dim = tuple(self.pre_ctx_dims + i for i in self.ctx_range
                                                                             if i != c)).cpu().numpy()
            return
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
        self.observations_from(p_x)

    def observations_from(self, p_x):
        """p_x (B, O): each episode's observation probabilities at its realized joint value"""
        self.pobs__joint = np.squeeze(p_x)   
        if self.obs_num == 1:
            self.pobs__joint = self.pobs__joint[:, None]
        rand = np.random.rand(*self.batch_step_dims, self.obs_num) 
        self.obs_flat = (self.pobs__joint[:, None, :] > rand).astype(bool)
        # Generates observations based on the joint realizations and random values.

_DEFAULT_METHODS = {m: getattr(Env_generators, m) for m in Env_generators._REALIZED_DEFAULTS}
