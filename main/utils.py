import numpy as np; import torch; import os, time, functools

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
    
    if x is None or type(x) is float:
        return x

    if to == "np":
        return x.detach().cpu().numpy()   
    if (to == "torch") and (type(x) != torch.Tensor):
        return torch.from_numpy(x).float().to(device)
    
    return x
    

def fig_path(name):
    """Resolve a figure filename into main/REPLICATION/figure_bin, independent of cwd."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "REPLICATION", "figure_bin")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


# ─────────────────────────────────────────────────────────────────────────────
# Shared figure style
# ─────────────────────────────────────────────────────────────────────────────

# Type is FIXED, not scaled. Keep panels near FIG_PANEL and every figure matches;
# scaling type to figure or panel width was tried and mis-sizes wide single panels.
FIG_PANEL = (2.9, 2.0)         # target axes size, inches -- the unit figures are built from
FIG_FONTS = dict(title=15, label=9.5, tick=8.5, legend=10, annot=10)


def apply_fig_style():
    """One look for every figure. Sizes assume panels of roughly FIG_PANEL.

    Returns the sizes so callers can use them for per-artist `fontsize=`.
    """
    import matplotlib.pyplot as plt
    fs = dict(FIG_FONTS)
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': 'cmr10',
        'font.sans-serif': 'cmss10', 'font.monospace': 'cmtt10',
        'axes.formatter.use_mathtext': True,
        'font.size': fs["label"], 'axes.labelsize': fs["label"],
        'axes.titlesize': fs["title"], 'axes.titleweight': 'normal',
        'legend.fontsize': fs["legend"],
        'xtick.labelsize': fs["tick"], 'ytick.labelsize': fs["tick"],
        'axes.linewidth': .8, 'lines.linewidth': 1.4,
        'xtick.direction': 'out', 'ytick.direction': 'out', 'figure.dpi': 300,
    })
    return fs
