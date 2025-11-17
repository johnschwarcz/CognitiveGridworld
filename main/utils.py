import numpy as np; import torch; import time, functools

def print_time(show_if_above=None):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(self, *a, **k):
            t = time.time()
            out = fn(self, *a, **k) 
            duration = (time.time() - t) / 60
            title = getattr(self, 'print_time_title', fn.__name__)
            time_thresh = show_if_above or getattr(self, 'showtime', 0.2)
            if duration > time_thresh:       print(f"{title}: {duration:.1f} min")
            return out
        return wrap
    return deco

def tnp(x, to, device = None):
    if type(x) is list:
        return [tnp(x_, to, device = device) for x_ in x]
    
    if x is None:
        return x
    if to == "np":
        return x.detach().cpu().numpy()   
    if (to == "torch") and (type(x) != torch.Tensor):
        return torch.from_numpy(x).float().to(device)
    
    return x
    