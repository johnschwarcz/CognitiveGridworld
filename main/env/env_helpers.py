import numpy as np; import torch; import time, functools; 
from main.utils import tnp; from .Env_Customization import Env_Customization

class Env_helpers(Env_Customization):

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
