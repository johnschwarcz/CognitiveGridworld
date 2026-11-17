import numpy as np; import torch; import time, functools; 
from main.utils import tnp; from .Env_Customization import Env_Customization

class Env_helpers(Env_Customization):

    # Lazily computed attributes (2026-11-13, speed). A training iteration of modes RL / oracle reads few of the values the
    # episode produces (the full likelihood tables, the network's numpy outputs and losses); those are computed on their
    # first read, by exactly the code that computed them eagerly before, so every reader gets the same values (bitwise).
    def __getattr__(self, name):                       # only called for names missing from __dict__
        lazy = self.__dict__.get("_lazy")
        if lazy is not None and name in lazy:
            lazy[name](self)
            if name in self.__dict__:
                return self.__dict__[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def set_lazy(self, names, compute):
        """names leave __dict__; the first read of any of them runs compute(env) once, which returns {name: value} for all
        of them. compute gets the env as its argument and must hold no reference to it (no closure over self), so a pending
        value never keeps the env alive (no reference cycle). A name assigned in the meantime keeps its new value; a later
        set_lazy of the same name replaces this one. An error inside compute is raised as RuntimeError (never as an
        AttributeError, which getattr(env, name, default) / hasattr would take for a missing attribute) and leaves the names
        pending."""
        lazy = self.__dict__.setdefault("_lazy", {})
        group = _LazyGroup(tuple(names), compute)
        for n in group.names:
            self.__dict__.pop(n, None)
            lazy[n] = group

    def drop_lazy(self, names):
        """forget the pending values of names (their compute and what it holds, e.g. GPU tensors); the names stay unset until
        assigned or set_lazy again"""
        lazy = self.__dict__.get("_lazy")
        if lazy:
            for n in names:
                lazy.pop(n, None)

    def for_each_ctx(self, function, L):
        # Move context self.c to end, apply function and move back.
        for self.c in self.ctx_range:
            ind = -(self.c+1)
            L = np.moveaxis(L,  ind, -1)
            L = function(L)
            L = np.moveaxis(L, -1, ind)
        return L

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def match_dims(self, x, match_to, where):    
        # Expand x from front, back or back-1 to match shape of match_to.
        for m in range(match_to.ndim - x.ndim):
            if where == 0:
                x = x[None, ...]
            if where == -1:
                x = x[..., None]
            if where == -2:
                x = x[..., None, :]
        return x

    def avg_over_ctx(self, L, keep = None, override = 'avg', squeeze = True, includes_obs = False):
        # Average (or override) likelihoods over all dimensions except keep.
        if isinstance(keep, (int, np.int_)):     keep = [keep]
        if keep is None:                         keep = [self.k, self.q] 
        for i in self.ctx_range:
            if i not in keep:
                ind = self.pre_ctx_dims + i
                if override == 'avg':            L = L.mean(ind, keepdims=True)
                if override == 'sum':            L = L.sum(ind, keepdims=True)
        if squeeze:
            L = L.squeeze()
            if includes_obs and (self.obs_num == 1): 
                L = L[:,None,:]
        return L
   
    def avg_until(self, x, override = 'avg', stop_shape = 1, squeeze = False):
        # Average (or override) x from back until x is stope_shape dimensions.
        OG_dim = x.ndim 
        while x.ndim > stop_shape:
            if override == 'avg':                x = x.mean(-1)
            if override == 'sum':                x = x.sum(-1)
            if override == 'max':                x = x.max(-1)
            if override == 'min':                x = x.min(-1)
        if not squeeze:
            while x.ndim < OG_dim:
                x = x[..., None]
        return x

    def sample_batch(self, batch = None):
        self.b = np.random.randint(self.batch_num) if batch is None else batch
        
    def sample_obs(self, obs = None):
        self.o = np.random.randint(self.obs_num) if obs is None else obs

    def DKL(self, x, y, eps = 1e-15, avg_over_batch = True, sym = False):
        x = np.clip(x, a_min = eps, a_max = float(1) - eps)
        y = np.clip(y, a_min = eps, a_max = float(1) - eps)
        DKL = self.DKL_helper(x, y)
        if sym:
            DKL = (DKL + self.DKL_helper(y, x)) / 2
        if avg_over_batch:
            return DKL.mean(0)
        return DKL

    def DKL_helper(self, x, y):
        return (x * np.log(x/y)).sum(-1)


class _LazyGroup:
    """one set_lazy call: names computed together by compute(env) (Env_helpers.set_lazy)"""
    __slots__ = ("names", "compute", "busy")

    def __init__(self, names, compute):
        self.names, self.compute, self.busy = names, compute, False

    def __call__(self, env):
        d, lazy = env.__dict__, env.__dict__["_lazy"]
        pending = [n for n in self.names if lazy.get(n) is self and n not in d]
        if pending:
            if self.busy:
                raise RuntimeError(f"lazy attributes {pending} read while being computed")
            self.busy = True
            try:
                values = self.compute(env)                 # first compute; the entries go only once it succeeded
            except Exception as err:
                raise RuntimeError(f"computing the lazy attributes {pending} failed: {err!r}") from err
            finally:
                self.busy = False
            for n in pending:
                d[n] = values[n]
        for n in self.names:
            if lazy.get(n) is self:
                del lazy[n]
