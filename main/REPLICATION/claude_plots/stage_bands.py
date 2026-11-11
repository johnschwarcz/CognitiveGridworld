"""The two behavioural stages drawn on every through-training panel (John 11-08: the shared stage bands proposed for the
final figures). The spans are td_tt_timeline's half-way ranges of the two groups of behavioural families (the same
constants as td_tt_mechanism.EARLY_SPAN / LATE_SPAN and fig_main_1): answers and the splitting of near beliefs reach
half-way to Bayes at 774-1,575 training episodes, holding a split and the two variables moving together at
2,747-11,103 (additive gap scale; td_tt_timeline RESULTS). Line plots get light shading behind the data; heat maps get a
thin strip above the axes so that the map's colours are untouched."""
from matplotlib.patches import Rectangle

SPANS = ((774, 1575), (2747, 11103))
COLORS = ("#e6dcf2", "#dcebd3")
NAMES = ("answers, splitting half-way", "holding, coupling half-way")


def shade(ax):
    """light bands behind the data (line plots)"""
    for (a, b), c in zip(SPANS, COLORS):
        ax.axvspan(a, b, color=c, lw=0, zorder=0)


def strip(ax, h=.035):
    """a thin strip just above the axes (heat maps): the spans in data x, axes y"""
    tr = ax.get_xaxis_transform()
    for (a, b), c in zip(SPANS, COLORS):
        ax.add_patch(Rectangle((a, 1.005), b - a, h, transform=tr, color=c, clip_on=False, lw=0,
                               zorder=6))


def legend_handles():
    from matplotlib.patches import Patch
    return [Patch(color=c, label=n) for c, n in zip(COLORS, NAMES)]
