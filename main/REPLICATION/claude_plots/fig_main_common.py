"""Shared conventions of the consolidated publication suite (fig_main_1..6, fig_supp_1..5): one x-axis, one palette, one
panel-title form, one output folder. Every figure redraws from the caches of the td_* modules; nothing here computes.
-> main/REPLICATION/figure_bin/publication/"""
import os
import sys
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION"), os.path.join(ROOT, "main", "REPLICATION", "claude_plots")]
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt                                                   # noqa: E402
from main.utils import fig_path, apply_fig_style                                  # noqa: E402

COL = {"trained": "#1f77b4", "echo": "#ff7f0e", "bayes": "k", "naive": "0.55"}
NAME = {"trained": "Trained RNN", "echo": "ESN", "bayes": "Bayes", "naive": "naive"}
WIDTH_COL = {"trained_N50": "#c6dbef", "trained_N100": "#9ecae1", "trained_N200": "#4292c6", "trained": "#1f77b4"}
PASS = 880.                            # first probe checkpoint at which the Trained RNN's accuracy at T passes naive's
X0, XMIN, XMAX = 10, 8, 7e4            # the initial network is drawn at X0, labelled 0
TS, FL, FT = 12.5, 10.5, 9.5           # panel title, axis label, legend / annotation sizes
DPI = 200


def setup():
    apply_fig_style()
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({"xtick.labelsize": FT, "ytick.labelsize": FT, "axes.labelsize": FL})


def canvas(w, h):
    """figure of w x h inches; returns (fig, add) with add(x, y, pw, ph) = axes placed in inches from the bottom left"""
    fig = plt.figure(figsize=(w, h))
    return fig, lambda x, y, pw, ph, **k: fig.add_axes([x / w, y / h, pw / w, ph / h], **k)


def xs(x):
    """training episodes for a log axis: the initial network (0) at X0"""
    return np.where(np.asarray(x, float) > 0, x, X0).astype(float)


def edges(x):
    lx = np.log(x)
    mid = (lx[1:] + lx[:-1]) / 2
    return np.exp(np.concatenate([[lx[0] - (mid[0] - lx[0])], mid, [lx[-1] + (lx[-1] - mid[-1])]]))


def xaxis(ax, label=True, passing=True, xmax=XMAX):
    """the suite's training-episode axis; dotted line: the Trained RNN passes naive"""
    ax.set_xscale("log")
    ax.set_xticks([X0, 1e2, 1e3, 1e4])
    ax.set_xticklabels(["0", "100", "1k", "10k"])
    ax.set_xlim(XMIN, xmax)
    ax.minorticks_off()
    if label:
        ax.set_xlabel("training episodes", fontsize=FL)
    else:
        ax.tick_params(labelbottom=False)
    ax.spines[["top", "right"]].set_visible(False)
    if passing:
        ax.axvline(PASS, color="0.55", lw=1, ls=":", zorder=1)


def title(ax, letter, name, formula=None):
    """panel title: letter, text name and, on a second line, the formula"""
    ax.set_title(f"{letter}   {name}" + (f"\n{formula}" if formula else ""), loc="left", fontsize=TS, linespacing=1.35)


def ref(ax, v, name, color=None, x=None, dy=.012, ls=":"):
    """horizontal reference line with its name"""
    c = color or COL.get(name, "0.45")
    ax.axhline(v, color=c, lw=.9, ls=ls, zorder=1)
    ax.text(x or XMIN * 1.15, v + dy, name, fontsize=FT - .5, color=c, va="bottom")


def note(fig, y, text):
    """data counts and caveats under the panels; y in figure fraction"""
    fig.text(.5, y, text, ha="center", va="top", fontsize=FT, color="0.2", linespacing=1.5)


def save(fig, name):
    path = fig_path(os.path.join("publication", name))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("saved", path)
    return path
