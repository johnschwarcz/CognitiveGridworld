# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 04 · Samplers: SDEs, the probability-flow ODE, DDIM and fast sampling
#
# Notebooks 02–03 sampled with DDPM's 1000 small ancestral steps. Here we take the continuous-time view: a trained
# denoiser defines a **vector field**, and sampling means numerically solving a differential equation, either stochastic
# (an SDE) or deterministic (the probability-flow ODE). This view explains why DDIM is deterministic and fast, what the
# choice of time steps does, and how to trade compute (**NFE**, the number of network evaluations) against sample quality.
# Every sampler first gets the **exact** score of a Gaussian mixture, so that the only error left is the discretisation
# error. At the end we repeat the key comparison with a trained network.
#
# **You will learn:**
# - the VP-SDE as the continuous-time limit of the DDPM chain, Anderson's reverse-time SDE, and the probability-flow ODE (via Fokker–Planck), all in $\epsilon$-form
# - five samplers written out by hand (DDPM ancestral, Euler–Maruyama, Euler and Heun on the ODE, DDIM), how their quality scales with NFE, and how to measure a solver's order of convergence
# - DDIM from its non-Markovian forward process, and why its error comes from replacing a posterior *sample* of $x_0$ by the posterior *mean*; why DDIM *is* Euler's method in the variables $(x/\sqrt{\bar\alpha},\ \sigma)$, and why that makes it robust (each step is a convex combination that cannot overshoot)
# - that the reverse SDE is the ODE plus Langevin MCMC, which is why its noise **corrects** some errors (and which errors it cannot correct), and that the ODE is a bijection between noise and data (DDIM inversion)
# - where to put the steps: uniform-in-$t$ vs Karras/EDM spacing, how to diagnose *where* a solver makes its errors, and when a second-order solver pays off
# - (advanced) exact log-likelihoods from the ODE with the instantaneous change-of-variables formula
# - why model error sets a floor for deterministic samplers (and how stochasticity partially corrects it), why one metric can hide it, and a practical cheat-sheet for choosing a sampler
#
# **Prerequisites:** 01 (forward process, $\bar\alpha_t$, VP vs VE), 02 (DDPM, ancestral sampling, MMD and sliced Wasserstein),
# 03 (scores, Tweedie's formula $\hat\epsilon = -\sqrt{1-\bar\alpha_t}\,\nabla\log q_t$).
#
# **Runtime:** about 3–5 min on a laptop CPU (exact Gaussian-mixture maths in float64, plus one small MLP of about 20–30 s);
# about 2–4 min on a GPU (the exact computations stay on the CPU).

# %% [markdown]
# ## Setup
#
# The exact scores of the Gaussian mixture are computed in float64 on the CPU (Apple's MPS backend has no float64), and
# all samplers keep their state there. Only the trained network in §8 runs on `device`.

# %%
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

import dmtools as dm

device = dm.get_device(verbose=True)
FAST = dm.fast_mode()  # automated smoke-test switch, leave as is
dm.seed_everything(0)
if device.type == "cpu":
    torch.set_num_threads(1)   # our batches are tiny 2-D point clouds: threading overhead outweighs any speed-up

plt.rcParams.update({"figure.dpi": 100, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
# one colour per sampler, used in every figure
COL = {"DDPM ancestral": "tab:orange", "Euler–Maruyama SDE": "tab:red", "Euler ODE": "tab:blue",
       "Heun ODE": "tab:purple", "DDIM": "tab:green", "data": "0.45", "floor": "0.8"}
CMAP_DENSITY = ListedColormap(plt.get_cmap("Greys")(np.linspace(0.0, 0.6, 64)))

# Every expensive quantity is scaled by FAST (FAST only checks that the code runs).
N_EVAL = 500 if FAST else 2000          # samples per sampler for the distribution metrics
N_PATH = 200 if FAST else 500           # particles for pathwise (trajectory) errors
REF_STEPS = 400 if FAST else 4000       # Heun steps of the reference ODE solution
TRAIN_STEPS = 200 if FAST else 4000     # training steps of the MLP in §8
NFES = [5, 10, 20, 50, 100, 250, 1000]  # network-evaluation budgets we compare
MMD_N = 1000                            # dm.mmd_rbf is O(n^2) in time and memory: use the first 1000 samples
print(f"FAST={FAST}: {N_EVAL} samples per metric, reference ODE with {REF_STEPS} Heun steps, {TRAIN_STEPS} training steps")

gmm = dm.default_gmm()   # the ground-truth data distribution
LIM = 3.0                # plotting window [-LIM, LIM]^2

# %% [markdown]
# Sample quality is measured as in notebook 02: MMD$^2$ and sliced Wasserstein-1 distance to a fixed set of exact samples.
# Two independent sets of *exact* samples are not at distance zero, because of finite-sample noise. That **floor**
# (estimated from eight fresh sets) is the best any sampler can score. Because we know the true density, we also count
# **stranded** samples, which land where the true log-density is below $-10$ (in the gaps between modes or far in the
# tails). Exact samples essentially never land there. This becomes important in §8.

# %%
x_true = gmm.sample(N_EVAL, generator=torch.Generator().manual_seed(11)).double()   # reference set for every metric


def quality(x):
    """(MMD^2, sliced W1) between the samples x and the exact reference set x_true. Lower is better."""
    x = x.detach().cpu().double()
    return dm.mmd_rbf(x, x_true, max_n=MMD_N), dm.sliced_wasserstein(x, x_true)


def stranded_fraction(x, threshold=-10.0):
    """Fraction of samples in regions the data essentially never visits: true log-density below `threshold`."""
    return (gmm.log_prob(x.detach().cpu().double()) < threshold).double().mean().item()


floor = np.array([quality(gmm.sample(N_EVAL, generator=torch.Generator().manual_seed(100 + i))) for i in range(8)])
FLOOR_HI = floor.mean(0) + 2 * floor.std(0)       # upper edge of the floor band (MMD^2, sliced W1)
print(f"floor (fresh exact samples): MMD^2 {floor[:, 0].mean():.4f} ± {floor[:, 0].std():.4f}, "
      f"sliced W1 {floor[:, 1].mean():.4f} ± {floor[:, 1].std():.4f}")
n_big = 5_000 if FAST else 50_000
print(f"stranded fraction of {n_big} exact samples: "
      f"{stranded_fraction(gmm.sample(n_big, generator=torch.Generator().manual_seed(12))):.1e}")

# %% [markdown]
# ## 1 · From the DDPM chain to stochastic and ordinary differential equations
#
# ### 1.1 The forward chain is a discretised SDE
#
# In the DDPM paper's notation (1-based: step $k\in\{1,\dots,T\}$ produces latent $x_k$), one forward step is
# $x_k=\sqrt{1-\beta_k}\,x_{k-1}+\sqrt{\beta_k}\,z$ with $z\sim\mathcal N(0,I)$. Now read the
# integer step $k$ as continuous time $t=k/T\in[0,1]$ (notebook 01 §2 called it $\tau$; in this notebook $t$ is continuous and
# $k$ is the integer step) with step $\Delta t=1/T$, and define a rate
# $\beta(t)$ by $\beta_k=\beta(t)\,\Delta t$, that is $\beta(t)\approx T\beta_k$. Since $\sqrt{1-\beta(t)\Delta t}=1-\tfrac12\beta(t)\Delta t+O(\Delta t^2)$:
#
# $$x_{t+\Delta t}-x_t \;\approx\; -\tfrac12\beta(t)\,x_t\,\Delta t+\sqrt{\beta(t)\Delta t}\;z .$$
#
# This is exactly the **Euler–Maruyama** discretisation (drift $\times\,\Delta t$, plus Gaussian noise with variance
# $\propto\Delta t$) of the **variance-preserving SDE**
#
# $$\boxed{\;dx=-\tfrac12\beta(t)\,x\,dt+\sqrt{\beta(t)}\,dW_t\;}\qquad\text{i.e. } f(x,t)=-\tfrac12\beta(t)x,\quad g(t)=\sqrt{\beta(t)} .$$
#
# Its marginals follow from two moment ODEs. The mean obeys $\dot m=-\tfrac12\beta m$ and the variance obeys
# $\dot v=-\beta v+\beta$, so
#
# $$x_t\mid x_0\sim\mathcal N\!\big(\sqrt{\bar\alpha(t)}\,x_0,\ (1-\bar\alpha(t))I\big),\qquad
# \bar\alpha(t)=\exp\!\Big(-\!\int_0^t\beta(s)\,ds\Big).$$
#
# This is the closed form from notebook 01, now in continuous time. `dm.NoiseSchedule`'s linear schedule ($\beta_k$ from
# $10^{-4}$ to $0.02$, $T=1000$) corresponds to $\beta(t)=0.1+19.9\,t$, for which
# $\bar\alpha(t)=\exp(-0.1t-9.95t^2)$.
#
# > **Gotcha: which $t$ is index `k`?** `dmtools` stores everything in 0-based arrays: `betas[k]` and `alpha_bars[k]`,
# > `k` $\in\{0,\dots,T-1\}$, belong to the paper's step $k+1$, i.e. to latent $x_{k+1}$ (notebook 02). So the network's
# > index `k` sits at continuous time $t=(k+1)/T$. The smallest noise level the network ever saw is at $t_{\min}=1/T$. The
# > last step of every sampler therefore jumps from $t_{\min}$ to the clean data at $t=0$.

# %%
T = 1000                      # steps of the discrete chain (dm.NoiseSchedule default)
BETA_MIN, BETA_MAX = 0.1, 20.0  # beta(t) = T * beta_k for dm's linear schedule (1e-4 ... 0.02)
T_MIN = 1.0 / T               # continuous time of the network's index 0


def beta_fn(t):
    """beta(t) of the continuous VP-SDE: linear from BETA_MIN at t=0 to BETA_MAX at t=1."""
    return BETA_MIN + t * (BETA_MAX - BETA_MIN)


def abar_fn(t):
    """alpha_bar(t) = exp(-int_0^t beta(s) ds), in closed form for the linear beta(t)."""
    return math.exp(-(BETA_MIN * t + 0.5 * (BETA_MAX - BETA_MIN) * t * t))


# CHECK: the continuous schedule reproduces dm.NoiseSchedule's discrete alpha_bar_k at t = (k+1)/T
sched = dm.NoiseSchedule(T, "linear")             # a CPU copy for the exact-score experiments
AB_DISC = sched.alpha_bars.double().numpy()       # float32 values, exactly as the dmtools samplers use them
ab_cont = np.array([abar_fn((k + 1) / T) for k in range(T)])
print(f"max_k |abar_k - abar((k+1)/T)| = {np.abs(AB_DISC - ab_cont).max():.1e}   (alpha_bar(1) = {abar_fn(1.0):.1e})")
assert np.abs(AB_DISC - ab_cont).max() < 2e-3

# %% [markdown]
# The two agree to $O(1/T)$. Two approximations contribute errors of similar size, which partly cancel: the
# $O(\beta_k^2)$ terms dropped in $\log(1-\beta_k)\approx-\beta_k$, and replacing the integral $\int\beta\,dt$ by a sum.
# We now simulate the SDE
# itself with Euler–Maruyama from data samples, and check that its marginals match the exact noised mixtures $q_t$ (which
# `dm.GaussianMixture.noised` gives in closed form).

# %%
def forward_em(x0, n_steps, gen, keep=0):
    """Euler–Maruyama on the forward VP-SDE from t=0 to t=1.

    Returns {t: x_t} at a few times, plus the full paths of the first `keep` particles (for plotting)."""
    h = 1.0 / n_steps
    x = x0.clone()
    snaps, paths = {}, [x[:keep].clone()]
    for i in range(n_steps):
        t = i * h
        b = beta_fn(t)
        x = x - 0.5 * b * x * h + math.sqrt(b * h) * torch.randn(x.shape, generator=gen, dtype=x.dtype)
        paths.append(x[:keep].clone())
        if (i + 1) in (n_steps // 10, n_steps * 3 // 10, n_steps):
            snaps[round((i + 1) * h, 3)] = x.clone()
    return snaps, torch.stack(paths)


x0_fwd = gmm.sample(N_EVAL, generator=torch.Generator().manual_seed(1)).double()
snaps, fwd_paths = forward_em(x0_fwd, T, torch.Generator().manual_seed(2), keep=12)
print("CHECK: forward Euler–Maruyama vs exact marginals q_t (sliced W1; 'exact vs exact' = pure sampling noise)")
for t, xt in snaps.items():
    q_t = gmm.noised(abar_fn(t))
    sw = dm.sliced_wasserstein(xt, q_t.sample(N_EVAL, generator=torch.Generator().manual_seed(3)))
    sw0 = dm.sliced_wasserstein(q_t.sample(N_EVAL, generator=torch.Generator().manual_seed(4)),
                                q_t.sample(N_EVAL, generator=torch.Generator().manual_seed(3)))
    print(f"  t = {t:.1f}:  SDE vs exact {sw:.4f}   exact vs exact {sw0:.4f}")

# %% [markdown]
# ### 1.2 Reversing time: Anderson's SDE and the probability-flow ODE
#
# **Reverse-time SDE.** Anderson (1982) showed that the forward SDE $dx=f\,dt+g\,dW$ with marginals $p_t$ has a
# time-reversal: run backwards from $t=1$ to $t=0$ (so $dt<0$, and $d\bar W$ is a Brownian motion in reverse time),
#
# $$dx=\big[f(x,t)-g(t)^2\nabla_x\log p_t(x)\big]\,dt+g(t)\,d\bar W_t
# \quad\xrightarrow{\ \text{VP}\ }\quad dx=\big[-\tfrac12\beta(t)\,x-\beta(t)\nabla\log p_t(x)\big]dt+\sqrt{\beta(t)}\,d\bar W_t .$$
#
# The only unknown is the score, which notebook 03 taught us to estimate by denoising.
#
# **Probability-flow ODE.** The density of the forward SDE obeys the Fokker–Planck equation. Using
# $\Delta p=\nabla\!\cdot(p\,\nabla\log p)$, we can rewrite the diffusion term as a transport term:
#
# $$\partial_t p_t=-\nabla\!\cdot(f\,p_t)+\tfrac12 g^2\Delta p_t=-\nabla\!\cdot\!\Big[\underbrace{\big(f-\tfrac12 g^2\nabla\log p_t\big)}_{v(x,t)}\,p_t\Big].$$
#
# This is a continuity equation, the one satisfied by particles that move **deterministically** with velocity
# $v(x,t)$. The same PDE with the same initial condition has the same solution. So the ODE
#
# $$\boxed{\;\frac{dx}{dt}=-\tfrac12\beta(t)\big[x+\nabla\log p_t(x)\big]\;}$$
#
# has **exactly the same marginals $p_t$** as the SDE, at every $t$. It is *not* the same process: individual paths differ,
# and only the distribution at each time agrees. Integrating it backwards from noise is a deterministic sampler.
#
# **The whole family.** Combining the two, for any $\lambda\ge0$ (here just a knob for the amount of noise, not the log-SNR of
# notebook 01),
# $dx=\big[f-\tfrac{1+\lambda^2}{2}g^2\nabla\log p_t\big]dt+\lambda g\,d\bar W$ also has marginals $p_t$. With $\lambda=0$ it
# is the ODE and with $\lambda=1$ the reverse SDE. Written in reverse time $\tau=-t$, the difference between them,
# $\tfrac{\lambda^2}{2}g^2\nabla\log p_t\,d\tau+\lambda g\,dW_\tau$, is precisely **Langevin dynamics** targeting $p_t$
# (notebook 03). So **reverse SDE = probability-flow ODE + a Langevin MCMC corrector** running at rate $\beta(t)/2$.
# Keep this in mind for §4.
#
# **In $\epsilon$-form.** With a noise predictor, $\nabla\log p_t(x)\approx-\hat\epsilon(x,t)/\sqrt{1-\bar\alpha(t)}$ (Tweedie, notebook 03), so
#
# $$\text{reverse SDE: } dx=\Big[-\tfrac12\beta x+\beta\,\frac{\hat\epsilon}{\sqrt{1-\bar\alpha}}\Big]dt+\sqrt\beta\,d\bar W,
# \qquad
# \text{PF-ODE: } \frac{dx}{dt}=-\tfrac12\beta(t)\Big[x-\frac{\hat\epsilon(x,t)}{\sqrt{1-\bar\alpha(t)}}\Big].$$
#
# **DDPM is Euler–Maruyama of the reverse SDE.** The ancestral step of notebook 02 is
# $x_{k-1}=\frac{1}{\sqrt{1-\beta_k}}\big(x_k-\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\hat\epsilon\big)+\sqrt{\beta_k}\,z$.
# Expand $1/\sqrt{1-\beta_k}\approx1+\tfrac12\beta_k$ and use $\beta_k=\beta(t)\Delta t$: you get
# $x+\big[\tfrac12\beta x-\beta\hat\epsilon/\sqrt{1-\bar\alpha}\big]\Delta t+\sqrt{\beta\Delta t}\,z$, one Euler–Maruyama
# step of the reverse SDE (remember $dt=-\Delta t$).
#
# The figure below shows the schedule, and forward-SDE paths of the first coordinate drawn on top of the exact marginal
# density $p_t(x_1)$, for $t$ running from data (left) to noise (right).

# %%
def x1_marginal_density(ts, xs, dist=gmm):
    """Density of the first coordinate of q_t on a (len(xs), len(ts)) grid: a 1-D Gaussian mixture at every t."""
    w, mu, var = dist.weights.numpy(), dist.means[:, 0].numpy(), dist.covs[:, 0, 0].numpy()
    cols = []
    for t in ts:
        ab = abar_fn(t)
        m, v = math.sqrt(ab) * mu, ab * var + (1 - ab)
        cols.append((w * np.exp(-(xs[:, None] - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)).sum(1))
    return np.stack(cols, 1)


TS_G, XS_G = np.linspace(0, 1, 201), np.linspace(-LIM, LIM, 201)
DENS_TX = x1_marginal_density(TS_G, XS_G)
DENS_TX = (DENS_TX / DENS_TX.max(0, keepdims=True)) ** 0.5   # normalise each time slice; gamma < 1 shows the tails


def show_density_tx(ax):
    """Grey background: p_t(x_1) over (t, x_1), each time slice normalised to its maximum."""
    ax.contourf(TS_G, XS_G, DENS_TX, levels=np.linspace(0, 1, 11), cmap=CMAP_DENSITY)
    ax.set(xlabel="time $t$  (0 = data, 1 = noise)", ylabel="first coordinate $x_1$", xlim=(0, 1), ylim=(-LIM, LIM))


fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.4]})
tt = np.linspace(0, 1, 200)
k_show = np.arange(0, T, 50)
axes[0].plot(tt, [beta_fn(t) for t in tt], color="k", label=r"$\beta(t)=0.1+19.9t$")
axes[0].plot((k_show + 1) / T, T * sched.betas.numpy()[k_show], "o", ms=3, color="tab:orange",
             label=r"$T\cdot$betas[k] (dm), at $t=(k+1)/T$")
axes[0].set(xlabel="$t$", ylabel="rate", title="noise rate")
axes[0].legend()
axes[1].semilogy(tt, [abar_fn(t) for t in tt], color="k", label=r"$\bar\alpha(t)$, closed form")
axes[1].semilogy((k_show + 1) / T, AB_DISC[k_show], "o", ms=3, color="tab:orange", label="alpha_bars[k] (dm)")
axes[1].set(xlabel="$t$", ylabel="signal fraction", title=r"$\bar\alpha$: continuous vs discrete")
axes[1].legend()
show_density_tx(axes[2])
t_paths = np.linspace(0, 1, fwd_paths.shape[0])
for i in range(fwd_paths.shape[1]):
    axes[2].plot(t_paths, fwd_paths[:, i, 0].numpy(), lw=0.6, color="tab:red", alpha=0.8)
axes[2].set_title(r"forward-SDE paths on top of the exact marginal $p_t(x_1)$")
plt.tight_layout()
plt.show()

# %% [markdown]
# The discrete schedule is the continuous one sampled at $t=(k+1)/T$. On the right, each path is jagged, but the cloud of
# paths has the marginals $p_t$ (the printed check): the modes of the mixture at $t=0$ melt into a single Gaussian by
# $t\approx0.5$. Sampling must undo this, either with the reverse SDE (jagged paths) or with the ODE (smooth paths).
#
# ## 2 · Five samplers, one exact score
#
# All samplers below walk down a time grid $1=t_0>t_1>\dots>t_{n-1}=t_{\min}>t_n=0$ and call the noise predictor once per
# step. Heun calls it twice, except on the last step. Writing $a_t=\bar\alpha(t)$ ($\bar\alpha$ itself, whereas the $a$ of notebook
# 02 §9a was $\sqrt{\bar\alpha}$) and
# $\hat x_0=(x_t-\sqrt{1-a_t}\,\hat\epsilon)/\sqrt{a_t}$ (Tweedie), one step from $t$ to $s<t$ is:
#
# | sampler | update $x_t\to x_s$ | NFE / step |
# |---|---|---|
# | (a) DDPM ancestral | sample $q(x_s\mid x_t,\hat x_0)$: mean $\frac{\sqrt{a_s}(1-a_t/a_s)\hat x_0+\sqrt{a_t/a_s}(1-a_s)x_t}{1-a_t}$, variance $\frac{1-a_s}{1-a_t}(1-\frac{a_t}{a_s})$ | 1 |
# | (b) Euler–Maruyama, reverse SDE | $x_t+\big[\tfrac12\beta x_t+\beta\nabla\log p_t\big]h+\sqrt{\beta h}\,z$, with $h=t-s$ | 1 |
# | (c) Euler, PF-ODE | $x_t+(s-t)\,v(x_t,t)$, with $v=-\tfrac12\beta(x+\nabla\log p_t)$ | 1 |
# | (d) Heun, PF-ODE | predictor $x'_s=x_t+(s-t)v(x_t,t)$, then $x_t+(s-t)\tfrac12[v(x_t,t)+v(x'_s,s)]$ | 2 |
# | (e) DDIM ($\eta=0$) | $\sqrt{a_s}\,\hat x_0+\sqrt{1-a_s}\,\hat\epsilon$ (derived in §3) | 1 |
#
# (a) is DDPM's posterior step, with $\alpha=a_t/a_s$ for a step that may skip many of the original $T$ steps ("respacing",
# Nichol & Dhariwal 2021).
#
# **The last step is special.** The network was never trained at $t=0$, and near $t=0$ the ODE in $(x,t)$ is very steep.
# So every sampler ends the same way: from $t_{\min}$ it returns the denoised estimate $\hat x_0(x_{t_{\min}},t_{\min})$,
# with no noise added (1 NFE). DDPM's and DDIM's own formulas reduce to exactly this at $s=0$ (set $a_s=1$). For the other
# three samplers we special-case it. A plain Euler step from $t_{\min}$ to 0 leaves residual noise behind; §3.2 shows how much.
#
# The "network" is the exact noise predictor $\epsilon^*(x,t)=-\sqrt{1-\bar\alpha(t)}\,\nabla\log q_t(x)$ of the mixture,
# so that **all differences between samplers are discretisation error**.

# %%
def exact_eps_fn(dist, abar=abar_fn):
    """The ideal noise predictor eps*(x, t) = -sqrt(1 - abar(t)) * grad log q_t(x) of a dm.GaussianMixture.

    q_t is the VP-noised mixture (dist.noised), so this is exact at every continuous t > 0."""
    def eps_fn(x, t):
        ab = abar(t)
        return -math.sqrt(1.0 - ab) * dist.noised(ab).score(x)
    return eps_fn


eps_exact = exact_eps_fn(gmm)


def uniform_grid(n_steps):
    """Time grid for n_steps steps: n_steps evaluation times uniform from 1 down to T_MIN, then the final time 0."""
    return np.concatenate([np.linspace(1.0, T_MIN, n_steps), [0.0]])


def _noise(x, gen):
    return torch.randn(x.shape, generator=gen, dtype=x.dtype)


def denoise(eps_fn, x, t, abar=abar_fn):
    """Tweedie's estimate x0_hat = (x - sqrt(1 - a_t) eps_hat) / sqrt(a_t): every sampler's last step, t_min -> 0."""
    a_t = abar(t)
    return (x - math.sqrt(1 - a_t) * eps_fn(x, t)) / math.sqrt(a_t)


@torch.no_grad()
def ddpm_sampler(eps_fn, x, ts, *, abar=abar_fn, gen=None, return_traj=False):
    """(a) Ancestral sampling on the grid ts: x_s ~ q(x_s | x_t, x0_hat), DDPM's posterior with x0 replaced by x0_hat."""
    traj = [x]
    for t, s in zip(ts[:-1], ts[1:]):
        a_t, a_s = abar(t), abar(s)
        eps = eps_fn(x, t)
        x0 = (x - math.sqrt(1 - a_t) * eps) / math.sqrt(a_t)
        alpha = a_t / a_s                                   # the "alpha" of this possibly large step
        x = (math.sqrt(a_s) * (1 - alpha) * x0 + math.sqrt(alpha) * (1 - a_s) * x) / (1 - a_t)
        if s > 0:                                           # no noise on the last step: return the posterior mean
            x = x + math.sqrt((1 - a_s) / (1 - a_t) * (1 - alpha)) * _noise(x, gen)
        traj.append(x)
    return traj if return_traj else x


@torch.no_grad()
def euler_maruyama_sampler(eps_fn, x, ts, *, gen=None, return_traj=False):
    """(b) Euler–Maruyama on the reverse-time VP-SDE dx = [-beta x / 2 - beta * score] dt + sqrt(beta) dW, dt < 0."""
    traj = [x]
    for t, s in zip(ts[:-1], ts[1:]):
        if s == 0:                                          # last step: return the denoised estimate, no noise
            x = denoise(eps_fn, x, t)
        else:
            h, b = t - s, beta_fn(t)                        # h > 0: we step backwards in time
            score = -eps_fn(x, t) / math.sqrt(1 - abar_fn(t))
            x = x + (0.5 * b * x + b * score) * h + math.sqrt(b * h) * _noise(x, gen)
        traj.append(x)
    return traj if return_traj else x


def pf_ode_velocity(eps_fn, x, t):
    """dx/dt of the probability-flow ODE: -beta(t)/2 * (x + score), with score = -eps / sqrt(1 - abar)."""
    return -0.5 * beta_fn(t) * (x - eps_fn(x, t) / math.sqrt(1 - abar_fn(t)))


@torch.no_grad()
def euler_ode_sampler(eps_fn, x, ts, *, gen=None, return_traj=False, denoise_last=True):
    """(c) Euler's method on the probability-flow ODE, in the original variables (x, t).

    denoise_last=False replaces the final denoising step by a plain Euler step to t = 0 (a common mistake, see §3.2)."""
    traj = [x]
    for t, s in zip(ts[:-1], ts[1:]):
        if s == 0 and denoise_last:
            x = denoise(eps_fn, x, t)
        else:
            x = x + (s - t) * pf_ode_velocity(eps_fn, x, t)
        traj.append(x)
    return traj if return_traj else x


@torch.no_grad()
def heun_ode_sampler(eps_fn, x, ts, *, gen=None, return_traj=False, denoise_last=True):
    """(d) Heun's method (explicit trapezoidal rule, 2nd order) on the probability-flow ODE.

    2 NFE per step, except the last step (to t = 0), which costs 1: n steps cost 2n - 1 NFE."""
    traj = [x]
    for t, s in zip(ts[:-1], ts[1:]):
        if s == 0:
            x = denoise(eps_fn, x, t) if denoise_last else x + (s - t) * pf_ode_velocity(eps_fn, x, t)
        else:
            v = pf_ode_velocity(eps_fn, x, t)
            x_pred = x + (s - t) * v                        # predictor x'_s (an Euler step) ...
            x = x + (s - t) * 0.5 * (v + pf_ode_velocity(eps_fn, x_pred, s))   # ... then average the two slopes
        traj.append(x)
    return traj if return_traj else x


@torch.no_grad()
def ddim_sampler(eps_fn, x, ts, *, eta=0.0, abar=abar_fn, gen=None, return_traj=False):
    """(e) DDIM (Song, Meng & Ermon 2021). eta = 0: deterministic; eta = 1: DDPM's ancestral sampler."""
    traj = [x]
    for t, s in zip(ts[:-1], ts[1:]):
        a_t, a_s = abar(t), abar(s)
        eps = eps_fn(x, t)
        x0 = (x - math.sqrt(1 - a_t) * eps) / math.sqrt(a_t)
        gamma = eta * math.sqrt((1 - a_s) / (1 - a_t) * (1 - a_t / a_s))     # std of the fresh noise
        x = math.sqrt(a_s) * x0 + math.sqrt(max(1 - a_s - gamma ** 2, 0.0)) * eps   # = x0 when s = 0 (a_s = 1)
        if gamma > 0 and s > 0:
            x = x + gamma * _noise(x, gen)
        traj.append(x)
    return traj if return_traj else x


SAMPLERS = {  # name: (function, uses Heun's 2 NFE per step?, stochastic?)
    "DDPM ancestral": (ddpm_sampler, False, True),
    "Euler–Maruyama SDE": (euler_maruyama_sampler, False, True),
    "Euler ODE": (euler_ode_sampler, False, False),
    "Heun ODE": (heun_ode_sampler, True, False),
    "DDIM": (ddim_sampler, False, False),
}

# %% [markdown]
# *Side note — safe to skim: checks that our `ddpm_sampler` reproduces `dm.ddpm_sample` exactly; rule to remember: dmtools' array index `k` sits at continuous time $t=(k+1)/T$ (the gotcha of §1.1).*
#
# **CHECK against `dmtools`.** `dm.ddpm_sample` (notebook 02) runs the full 1000-step chain with the discrete schedule.
# Our sampler must reproduce it exactly when given the same grid, the same $\bar\alpha_k$ and the same random numbers. To
# hand the exact score to `dmtools` we wrap it as a model with the usual signature `model(x, t, y=None)`.

# %%
class ExactEpsModel:
    """dmtools-style model(x, t, y=None) returning the exact eps at integer timesteps t of a dm.NoiseSchedule."""

    def __init__(self, dist, schedule):
        self.dist, self.ab = dist, schedule.alpha_bars.cpu().double()

    def __call__(self, x, t, y=None):
        assert (t == t[0]).all(), "the dmtools samplers use one timestep per batch"
        ab = float(self.ab[int(t[0])])
        return (-math.sqrt(1 - ab) * self.dist.noised(ab).score(x.cpu().double())).to(x.device, x.dtype)


def abar_disc(t):
    """dm.NoiseSchedule's alpha_bar_k at the continuous time t = (k+1)/T, and 1 at t = 0."""
    return 1.0 if t == 0 else float(AB_DISC[int(round(t * T)) - 1])


def disc_grid(n_steps):
    """The times (k+1)/T of the timesteps dm.ddim_sample picks for n_steps (descending), followed by 0."""
    ks = np.unique(np.linspace(0, T - 1, n_steps).round().astype(int))[::-1]
    return np.concatenate([(ks + 1) / T, [0.0]])


exact_model = ExactEpsModel(gmm, sched)
eps_disc = exact_eps_fn(gmm, abar=abar_disc)
x_chk = torch.randn(300, 2, generator=torch.Generator().manual_seed(5))           # float32, like dmtools
ours = ddpm_sampler(eps_disc, x_chk, disc_grid(T), abar=abar_disc, gen=torch.Generator().manual_seed(6))
theirs = dm.ddpm_sample(exact_model, sched, x_chk.shape, x_T=x_chk, generator=torch.Generator().manual_seed(6))
print(f"CHECK ddpm_sampler == dm.ddpm_sample (1000 steps): max |diff| = {(ours - theirs).abs().max():.1e}")
assert torch.allclose(ours, theirs, atol=1e-3)

# %% [markdown]
# ### 2.1 Quality vs NFE
#
# Every sampler starts from the **same** noise $x_T\sim\mathcal N(0,I)$. $\mathcal N(0,I)$ is not exactly $q_1$, but here
# $\bar\alpha(1)\approx4\cdot10^{-5}$, so the mismatch is negligible. A stochastic sampler produces a fresh random sample
# set on every run, so its metrics fluctuate by about the width of the floor band. We therefore average them over three
# seeds. Heun's NFE is $2n-1$ for $n$ steps, so its even budgets land one below the nominal value (9, 19, 49, ...).
#
# For the deterministic samplers we also record the state at $t_{\min}$ for the first `N_PATH` particles. We compare it with
# a **reference solution** of the same ODE from the same $x_T$ (Heun with thousands of steps). The resulting **pathwise
# error** measures the discretisation error directly, without the finite-sample floor of a distribution metric.

# %%
def steps_for_nfe(name, nfe):
    """Number of steps a sampler can afford with `nfe` network evaluations, and the NFE it actually uses."""
    if SAMPLERS[name][1]:              # Heun: n steps cost 2n - 1 NFE
        n = (nfe + 1) // 2
        return n, 2 * n - 1
    return nfe, nfe


def nfe_sweep(eps_fn, x_T, names, nfes, x_ref=None, n_seeds=3):
    """Run each sampler at each NFE budget. Returns {name: list of dicts(nfe, mmd, sw, stranded, var, path)}.

    Stochastic samplers: metrics averaged over n_seeds noise seeds. Deterministic ones: also the pathwise error at
    t_min (before the final jump to t = 0) against the reference solution x_ref."""
    results = {name: [] for name in names}
    for name in names:
        fn, _, stochastic = SAMPLERS[name]
        for nfe in nfes:
            n, used = steps_for_nfe(name, nfe)
            metrics, path = [], None
            for seed in range(n_seeds if stochastic else 1):
                traj = fn(eps_fn, x_T, uniform_grid(n), gen=torch.Generator().manual_seed(seed), return_traj=True)
                x0 = traj[-1]
                metrics.append(quality(x0) + (stranded_fraction(x0), x0.var(0).sum().item()))
            if x_ref is not None and not stochastic:
                path = (traj[-2][:N_PATH] - x_ref).norm(dim=1).mean().item()
            mmd, sw, stranded, var = np.mean(metrics, axis=0)
            results[name].append(dict(nfe=used, mmd=mmd, sw=sw, stranded=stranded, var=var, path=path))
    return results


def first_nfe_in_floor(rows, key="sw"):
    """Smallest budget at which a sampler's metric is inside the floor band (None if never)."""
    hi = FLOOR_HI[0] if key == "mmd" else FLOOR_HI[1]
    return next((r["nfe"] for r in rows if r[key] <= hi), None)


x_T = torch.randn(N_EVAL, 2, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
t0 = time.time()
x_ref_tmin = heun_ode_sampler(eps_exact, x_T[:N_PATH], uniform_grid(REF_STEPS), return_traj=True)[-2]
print(f"reference ODE solution: {2 * REF_STEPS - 1} NFE on {N_PATH} particles in {time.time() - t0:.0f} s")
t0 = time.time()
sweep = nfe_sweep(eps_exact, x_T, list(SAMPLERS), NFES, x_ref=x_ref_tmin)
print(f"sweep: {len(SAMPLERS)} samplers x {len(NFES)} budgets in {time.time() - t0:.0f} s\n")
print(f"{'sampler':20s} " + " ".join(f"{n:>7d}" for n in NFES) + "   <- nominal NFE; MMD^2 x 1e3")
for name, rows in sweep.items():
    print(f"{name:20s} " + " ".join(f"{1e3 * r['mmd']:7.2f}" for r in rows))
print(f"{'floor':20s} {1e3 * floor[:, 0].mean():7.2f}   (band up to {1e3 * FLOOR_HI[0]:.2f})\n")
print("first NFE inside the floor band (MMD^2 / sliced W1):  "
      + ",  ".join(f"{name} {first_nfe_in_floor(rows, 'mmd')} / {first_nfe_in_floor(rows, 'sw')}"
                   for name, rows in sweep.items()))
print(f"total variance of the samples at NFE {NFES[0]} (exact samples: {x_true.var(0).sum():.2f}):  "
      + ",  ".join(f"{name} {rows[0]['var']:.2f}" for name, rows in sweep.items()))

# %% [markdown]
# Before reading the curves below, look at the samples behind their left half, 5–20 NFE (grey: the true density; the
# stochastic samplers show the first of their three seeds). At 5 NFE, DDPM and DDIM squeeze the samples into the middle (total variance about
# 0.9, against 2.11 for exact samples), Euler–Maruyama sprays them far past the modes (3.3), Euler keeps the right
# variance without finding the modes, and only Heun already shows them. By 20 NFE the ODE solvers in $(x,t)$ look like
# the data, while Euler–Maruyama's modes are still too wide and DDPM's too narrow.

# %%
viz_grid = np.linspace(-LIM, LIM, 161)          # plotting grid for the density backgrounds of the new figures
viz_xy = torch.tensor(np.stack(np.meshgrid(viz_grid, viz_grid), -1).reshape(-1, 2))


def viz_density(dist):
    """dist's density on the plotting grid, normalised to its maximum and square-rooted (to show the tails)."""
    p = dist.log_prob(viz_xy).double().exp().reshape(len(viz_grid), len(viz_grid)).numpy()
    return (p / p.max()) ** 0.5


viz_p0 = viz_density(gmm)
viz_budgets = [5, 10, 20]
viz_fig, viz_axes = plt.subplots(len(viz_budgets), len(SAMPLERS), figsize=(11.5, 7.3), layout="constrained")
for viz_r, viz_nfe in enumerate(viz_budgets):
    for viz_c, viz_name in enumerate(SAMPLERS):
        viz_n, viz_used = steps_for_nfe(viz_name, viz_nfe)
        viz_x = SAMPLERS[viz_name][0](eps_exact, x_T, uniform_grid(viz_n),
                                      gen=torch.Generator().manual_seed(0))     # = the sweep's first seed
        viz_ax = viz_axes[viz_r, viz_c]
        viz_ax.contourf(viz_grid, viz_grid, viz_p0, levels=np.linspace(0, 1, 11), cmap=CMAP_DENSITY)
        viz_ax.scatter(viz_x[:800, 0], viz_x[:800, 1], s=1.5, marker="s", color=COL[viz_name], linewidths=0,
                       antialiased=False)
        viz_ax.text(0.03, 0.97, f"NFE {viz_used}\nvariance {viz_x.var(0).sum():.2f}", transform=viz_ax.transAxes,
                    va="top", fontsize=7.5, bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.0))
        viz_ax.set(xlim=(-LIM, LIM), ylim=(-LIM, LIM), aspect="equal", xticks=[], yticks=[])
        if viz_r == 0:
            viz_ax.set_title(viz_name)
    viz_axes[viz_r, 0].set_ylabel(f"NFE ≈ {viz_nfe}")
viz_fig.suptitle("Few steps, exact score: DDPM and DDIM pull the samples inwards, Euler–Maruyama sprays them outwards "
                 f"(grey: true density; exact samples have total variance {x_true.var(0).sum():.2f})")
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
for ax, (j, key, label) in zip(axes, [(0, "mmd", "MMD$^2$ to exact samples"), (1, "sw", "sliced $W_1$ to exact samples")]):
    for name, rows in sweep.items():
        ls = "--" if SAMPLERS[name][2] else "-"   # dashed = stochastic
        ax.loglog([r["nfe"] for r in rows], [r[key] for r in rows], "o" + ls, color=COL[name], ms=4, label=name)
    lo, hi = floor[:, j].mean() - 2 * floor[:, j].std(), floor[:, j].mean() + 2 * floor[:, j].std()
    ax.axhspan(max(lo, 1e-6), hi, color=COL["floor"], alpha=0.6, label="floor (exact vs exact, ±2 sd)")
    ax.set(xlabel="NFE (network evaluations per sample)", ylabel=label, title=f"exact score: {label.split(' to')[0]} vs NFE")
    ax.grid(alpha=0.3, which="both")
axes[0].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Reading the figure (the solid lines are deterministic, the dashed ones stochastic; the printout above lists where each
# sampler enters the floor band):
#
# * **At 5 NFE everything is far off, and the ranking is mixed.** **From about 10–20 NFE the first-order ODE solvers
#   (Euler, DDIM) are clearly ahead of both stochastic samplers**, by a factor of roughly 1.5 to 6 in MMD$^2$ (Euler
#   by more than DDIM; see the table). The stochastic samplers catch up only at around 50–100 NFE.
# * **Why the stochastic samplers need small steps.** There are two concrete reasons. First, the reverse SDE's drift
#   carries the score with coefficient $\beta$, twice the ODE's $\beta/2$, so an explicit step is stiffer. Euler–Maruyama's
#   drift step overshoots, and its samples come out *over*-dispersed (see the total variance printed above). Second,
#   DDPM's ancestral step plugs the point estimate $\hat x_0=\mathbb E[x_0\mid x_t]$ into $q(x_s\mid x_t,x_0)$. The true
#   reverse kernel is the posterior predictive $p(x_s\mid x_t)=\int q(x_s\mid x_t,x_0)\,p(x_0\mid x_t)\,dx_0$, which is
#   wider (and, for a mixture, multimodal). With large steps the lost posterior variance of $x_0$ matters, and DDPM's
#   samples collapse towards the mean, just like DDIM's. §3.1 makes this precise.
# * **Given enough steps, everything converges** to the floor, because all five samplers discretise equations with the same
#   marginals. (The deterministic curves converge to one fixed sample set, the image of $x_T$ under the exact ODE map.
#   That set is itself one random draw, which is why they settle at one particular point inside the floor band.)
# * **Higher order is not automatically better at very low NFE.** With a handful of steps, Heun's corrector evaluates the
#   field at a poor predicted point, and its ranking at the smallest budgets jumps around: it is the best of all five at
#   5 NFE, but worse than Euler at 9 and 19 NFE.
#   Its second-order accuracy only pays off once the steps are small. The next figure makes this precise.
#
# The distribution metrics saturate at the floor, so they cannot show the *rate* at which each solver converges. The
# pathwise error can:

# %%
fig, ax = plt.subplots(figsize=(5.6, 4))
slopes = {}
for name in ["Euler ODE", "Heun ODE", "DDIM"]:
    nfe = np.array([r["nfe"] for r in sweep[name]])
    err = np.array([r["path"] for r in sweep[name]])
    ax.loglog(nfe, err, "o-", color=COL[name], ms=4, label=name)
    slopes[name] = -np.polyfit(np.log(nfe[-3:]), np.log(err[-3:]), 1)[0]   # fitted order over the 3 largest budgets
anchor = sweep["Euler ODE"][len(NFES) // 2]      # guide lines start at a mid-sized budget of the Euler curve
n_guide = np.array([anchor["nfe"], max(NFES)])
ax.loglog(n_guide, anchor["path"] * (n_guide / anchor["nfe"]) ** -1.0, "k--", lw=0.8, label=r"$\propto$ NFE$^{-1}$")
ax.loglog(n_guide, anchor["path"] * (n_guide / anchor["nfe"]) ** -2.0, "k:", lw=1.0, label=r"$\propto$ NFE$^{-2}$")
ax.set(xlabel="NFE", ylabel=r"mean $\|x_{t_{\min}}-x^{\rm ref}_{t_{\min}}\|$",
       title="pathwise error of the ODE solvers (same $x_T$)")
ax.grid(alpha=0.3, which="both")
ax.legend()
plt.tight_layout()
plt.show()
print("fitted order of convergence (slope over the three largest budgets): "
      + ", ".join(f"{k} {v:.2f}" for k, v in slopes.items()))
print(f"pathwise error at the largest budget: Euler / Heun = "
      f"{sweep['Euler ODE'][-1]['path'] / sweep['Heun ODE'][-1]['path']:.0f}x")
if not FAST:
    assert 0.7 < slopes["Euler ODE"] < 1.3 and 0.7 < slopes["DDIM"] < 1.3 and 1.6 < slopes["Heun ODE"] < 2.4

# %% [markdown]
# This is the textbook picture. Euler and DDIM are **first-order** methods (halving the step halves the error) and Heun is
# **second-order** (halving the step quarters it). That is why it is more than an order of magnitude more accurate at
# 1000 NFE, a gap that keeps growing with NFE. At small NFE all solvers are outside this asymptotic regime, which is where
# *where you put the steps* (§6) and *which variables you integrate in* (§3) matter most.
#
# Notice also that on this toy, DDIM is **not** more accurate than plain Euler in $(x,t)$. That seems at odds with DDIM's
# reputation as "the" fast sampler. §3 explains both the reputation and the exception.
#
# ## 3 · DDIM: a non-Markovian forward process, and Euler in disguise
#
# ### 3.1 Derivation
#
# The DDPM loss only ever looks at the marginals $q(x_t\mid x_0)=\mathcal N(\sqrt{a_t}\,x_0,(1-a_t)I)$, not at how the
# chain gets from one $t$ to the next. Song, Meng & Ermon (2021) exploited this: *any* process with these marginals is
# served by the same trained $\hat\epsilon$. On a time grid $1=t_0>t_1>\dots$ they define a process by running it
# *backwards* from the noisiest time,
#
# $$q_\gamma(x_{t_0},x_{t_1},\dots\mid x_0)=q(x_{t_0}\mid x_0)\prod_i q_\gamma(x_{t_{i+1}}\mid x_{t_i},x_0),$$
#
# with the family of reverse kernels (for $s<t$)
#
# $$q_\gamma(x_s\mid x_t,x_0)=\mathcal N\!\Big(\sqrt{a_s}\,x_0+\sqrt{1-a_s-\gamma_t^2}\;\frac{x_t-\sqrt{a_t}\,x_0}{\sqrt{1-a_t}},\ \gamma_t^2I\Big).$$
#
# **The marginals are preserved, by induction.** If $x_t=\sqrt{a_t}x_0+\sqrt{1-a_t}\,\epsilon$, then the fraction above
# is just $\epsilon$, so $x_s=\sqrt{a_s}x_0+\sqrt{1-a_s-\gamma_t^2}\,\epsilon+\gamma_t z$. This is Gaussian with mean
# $\sqrt{a_s}x_0$ and variance $(1-a_s-\gamma_t^2)+\gamma_t^2=1-a_s$, which is exactly $q(x_s\mid x_0)$, for **every**
# $\gamma_t$.
#
# **Why "non-Markovian".** Read forwards in time, the process has the kernel given by Bayes' rule,
# $q_\gamma(x_t\mid x_s,x_0)\propto q_\gamma(x_s\mid x_t,x_0)\,q(x_t\mid x_0)$. This kernel depends on $x_0$: to take the
# next noising step the process has to remember where it started. There is one exception ($\eta=1$ below), where it
# reduces to DDPM's Markov kernel $\mathcal N(\sqrt{a_t/a_s}\,x_s,(1-a_t/a_s)I)$.
#
# Plugging in the network's guesses $\hat x_0,\hat\epsilon$ gives the **DDIM update**
#
# $$\boxed{\;x_s=\sqrt{a_s}\,\hat x_0+\sqrt{1-a_s-\gamma_t^2}\;\hat\epsilon+\gamma_t z,\qquad
# \gamma_t=\eta\sqrt{\tfrac{1-a_s}{1-a_t}}\sqrt{1-\tfrac{a_t}{a_s}}\;}$$
#
# (Song et al. call the noise standard deviation $\sigma_t$; we write $\gamma_t$, because $\sigma$ will denote the noise *level* in §3.2.)
#
# * $\eta=1$ makes $\gamma_t^2$ the DDPM posterior variance, and the kernel becomes DDPM's $q(x_s\mid x_t,x_0)$: ancestral sampling.
# * $\eta=0$ is **deterministic**: "re-noise the clean guess with the *predicted* noise, to the next noise level".
#   Nothing in the derivation required $s=t-1$, so the steps can be large.
#
# **Where DDIM's error comes from.** The kernel is exact if it is fed a *sample* $x_0\sim p(x_0\mid x_t)$ from the
# posterior: then $(x_0,x_t)$ has the right joint distribution, and $x_s\sim q_s$ exactly, however large the step. In
# other words, the true reverse transition is the posterior predictive
# $p(x_s\mid x_t)=\int q_\gamma(x_s\mid x_t,x_0)\,p(x_0\mid x_t)\,dx_0$. DDIM (and DDPM) replace the posterior by a point
# mass at its mean $\hat x_0=\mathbb E[x_0\mid x_t]$. For a Gaussian mixture the posterior is available in closed form: given
# component $k$ it is Gaussian, with precision $\Sigma_k^{-1}+\frac{a_t}{1-a_t}I$, and the component is picked with the
# responsibilities of $q_t$. So we can test this claim directly.
#
# Four numerical checks: (1) one huge step $t=0.5\to0.05$ fed with posterior samples is exact for every $\eta$, while the
# posterior mean is not; (2) the forward process depends on $x_0$ unless $\eta=1$; (3) our sampler matches
# `dm.ddim_sample`; (4) $\eta=1$ matches our DDPM sampler.

# %%
def posterior_x0_sample(dist, x_t, a_t, gen):
    """One exact draw x0 ~ p(x0 | x_t) per row, for x_t = sqrt(a_t) x0 + sqrt(1 - a_t) eps and x0 ~ dist (a GMM).

    Bayes per component k: a Gaussian prior N(mu_k, Sigma_k) times a Gaussian likelihood in x0 gives a Gaussian posterior
    with precision Sigma_k^-1 + a_t / (1 - a_t) I. The component itself is drawn with the responsibilities of q_t at x_t."""
    k = torch.multinomial(dist.noised(a_t).responsibilities(x_t), 1, generator=gen).squeeze(1)
    prec_prior = torch.linalg.inv(dist.covs)                                              # (K, d, d)
    C = torch.linalg.inv(prec_prior + a_t / (1 - a_t) * torch.eye(dist.d, dtype=torch.float64))  # posterior covariances
    rhs = torch.einsum("nij,nj->ni", prec_prior[k], dist.means[k]) + math.sqrt(a_t) / (1 - a_t) * x_t
    mean = torch.einsum("nij,nj->ni", C[k], rhs)
    return mean + torch.einsum("nij,nj->ni", torch.linalg.cholesky(C)[k], _noise(x_t, gen))


def oracle_eps(x0_given):
    """An 'oracle network' that knows x0: eps = (x_t - sqrt(a_t) x0) / sqrt(1 - a_t), so that its x0_hat is x0_given."""
    def eps_fn(x, t):
        a_t = abar_fn(t)
        return (x - math.sqrt(a_t) * x0_given) / math.sqrt(1 - a_t)
    return eps_fn


# CHECK 1: one big DDIM step fed with a posterior SAMPLE of x0 lands exactly on q_s, for every eta;
#          fed with the posterior MEAN (what DDIM and DDPM actually do) it does not
N_MC = 20_000 if FAST else 100_000
g = torch.Generator().manual_seed(7)
T_BIG, S_BIG = 0.5, 0.05
a_t, a_s = abar_fn(T_BIG), abar_fn(S_BIG)
x0_mc = gmm.sample(N_MC, generator=g).double()
x_t_mc = math.sqrt(a_t) * x0_mc + math.sqrt(1 - a_t) * _noise(x0_mc, g)          # x_t ~ q_t
x0_post = posterior_x0_sample(gmm, x_t_mc, a_t, g)                                # x0 ~ p(x0 | x_t)
q_s = gmm.noised(a_s)
ref_s = q_s.sample(N_MC, generator=torch.Generator().manual_seed(8)).double()
sw_floor_s = dm.sliced_wasserstein(q_s.sample(N_MC, generator=torch.Generator().manual_seed(9)), ref_s)
print(f"CHECK 1: one DDIM step t = {T_BIG} -> s = {S_BIG}, sliced W1 to exact samples of q_s "
      f"(exact vs exact: {sw_floor_s:.4f}); total variance of q_s = {ref_s.var(0).sum():.2f}")
for eta in (0.0, 0.5, 1.0):
    step = np.array([T_BIG, S_BIG])
    x_post = ddim_sampler(oracle_eps(x0_post), x_t_mc, step, eta=eta, gen=torch.Generator().manual_seed(10))
    x_mean = ddim_sampler(eps_exact, x_t_mc, step, eta=eta, gen=torch.Generator().manual_seed(10))
    sw_post, sw_mean = dm.sliced_wasserstein(x_post, ref_s), dm.sliced_wasserstein(x_mean, ref_s)
    print(f"  eta = {eta}:  x0 ~ posterior: W1 {sw_post:.4f}, variance {x_post.var(0).sum():.2f}   |   "
          f"x0 = posterior mean: W1 {sw_mean:.4f}, variance {x_mean.var(0).sum():.2f}")
    assert sw_post < 2 * sw_floor_s + 0.005 and sw_mean > 5 * sw_post

# CHECK 2: the implied forward process is non-Markovian. Regress x_t on (x_s, x0, 1), with x_s drawn from the kernel:
# x0 still helps to predict x_t given x_s unless eta = 1. In closed form the x0 coefficient of E[x_t | x_s, x0] is
# sqrt(a_t) - c sqrt(a_s), with c = sqrt(1 - a_t) sqrt(1 - a_s - gamma^2) / (1 - a_s).
print("CHECK 2: coefficient of x0 in the regression of x_t on (x_s, x0); zero means Markovian")
for eta in (0.0, 0.5, 1.0):
    x_s = ddim_sampler(oracle_eps(x0_mc), x_t_mc, np.array([T_BIG, S_BIG]), eta=eta,
                       gen=torch.Generator().manual_seed(11))
    design = torch.cat([x_s, x0_mc, torch.ones(N_MC, 1, dtype=torch.float64)], 1)
    coef_x0 = torch.linalg.lstsq(design, x_t_mc).solution[2:4].diagonal()
    gamma = eta * math.sqrt((1 - a_s) / (1 - a_t) * (1 - a_t / a_s))
    c = math.sqrt(1 - a_t) * math.sqrt(max(1 - a_s - gamma ** 2, 0.0)) / (1 - a_s)
    expected = math.sqrt(a_t) - c * math.sqrt(a_s)
    print(f"  eta = {eta}: measured {coef_x0.numpy().round(3)}, closed form {expected:.3f}")
    assert torch.allclose(coef_x0, torch.full((2,), expected, dtype=torch.float64), atol=0.1)

# CHECK 3: our DDIM reproduces dm.ddim_sample on the discrete schedule
for n in (10, 50):
    ours = ddim_sampler(eps_disc, x_chk, disc_grid(n), abar=abar_disc)
    theirs = dm.ddim_sample(exact_model, sched, x_chk.shape, n_steps=n, x_T=x_chk)
    print(f"CHECK 3: ddim_sampler == dm.ddim_sample ({n} steps): max |diff| = {(ours - theirs).abs().max():.1e}")
    assert torch.allclose(ours, theirs, atol=1e-4)

# CHECK 4: DDIM with eta = 1 is exactly DDPM ancestral sampling (same grid, same random numbers)
ours_ddim = ddim_sampler(eps_exact, x_T[:300], uniform_grid(20), eta=1.0, gen=torch.Generator().manual_seed(8))
ours_ddpm = ddpm_sampler(eps_exact, x_T[:300], uniform_grid(20), gen=torch.Generator().manual_seed(8))
print(f"CHECK 4: DDIM(eta=1) == DDPM ancestral (20 steps): max |diff| = {(ours_ddim - ours_ddpm).abs().max():.1e}")
assert torch.allclose(ours_ddim, ours_ddpm, atol=1e-9)

# %% [markdown]
# CHECK 1 in pictures: the same $x_t$ (left) takes one DDIM step straight from $t=0.5$ to $s=0.05$, with the exact $q_s$
# in grey. Fed a posterior *sample* of $x_0$, the step lands on all eight modes with the right variance. Fed the
# posterior *mean*, which is what DDIM and DDPM use, it collapses the samples into one blob around the empty centre, and
# DDPM's fresh noise ($\eta=1$) leaves an even tighter blob.

# %%
viz_step = np.array([T_BIG, S_BIG])
viz_panels = [   # (samples, background density, title, colour)
    (x_t_mc, gmm.noised(abar_fn(T_BIG)), f"start: $x_t\\sim q_t$ at $t={T_BIG}$", "k"),
    (ddim_sampler(oracle_eps(x0_post), x_t_mc, viz_step), q_s, "DDIM step fed a posterior sample of $x_0$", COL["DDIM"]),
    (ddim_sampler(eps_exact, x_t_mc, viz_step), q_s, "DDIM step fed the posterior mean", COL["DDIM"]),
    (ddim_sampler(eps_exact, x_t_mc, viz_step, eta=1.0, gen=torch.Generator().manual_seed(10)), q_s,
     r"DDPM step ($\eta=1$) fed the posterior mean", COL["DDPM ancestral"])]   # seed 10, as in CHECK 1
viz_fig, viz_axes = plt.subplots(1, 4, figsize=(13, 3.9), layout="constrained")
for viz_ax, (viz_x, viz_dist, viz_title, viz_col) in zip(viz_axes, viz_panels):
    viz_ax.contourf(viz_grid, viz_grid, viz_density(viz_dist), levels=np.linspace(0, 1, 11), cmap=CMAP_DENSITY)
    viz_ax.scatter(viz_x[:2000, 0], viz_x[:2000, 1], s=1.5, marker="s", color=viz_col, linewidths=0, antialiased=False)
    viz_ax.set(xlim=(-LIM, LIM), ylim=(-LIM, LIM), aspect="equal", xticks=[], yticks=[],
               title=f"{viz_title}\ntotal variance {viz_x.var(0).sum():.2f}")
viz_fig.suptitle(f"One big step $t={T_BIG}\\to s={S_BIG}$ (grey: exact $q_s$, total variance {ref_s.var(0).sum():.2f}): "
                 "a posterior sample of $x_0$ lands on $q_s$; the posterior mean collapses it")
plt.show()

# %% [markdown]
# CHECK 1 sums up the discretisation error of DDIM and DDPM in one line. A single huge step is exact if $x_0$ is drawn
# from the posterior. Plugging in the posterior mean instead collapses the variance (about 0.4 instead of 2.13, printed
# above), and DDPM's added noise ($\eta=1$) does not restore it: the variance drops further, to about 0.24. With small steps the kernel depends on $x_0$ only weakly (for $\eta=0$ the
# coefficient of $x_0$ in its mean, $\sqrt{a_s}-\sqrt{a_t}\sqrt{(1-a_s)/(1-a_t)}$, vanishes as $s\to t$), so the mean is
# almost as good as a sample. So both samplers become accurate once the steps are small enough. §3.2 and §6 ask how small,
# and where.

# %% [markdown]
# ### 3.2 DDIM is Euler's method in the variables $\tilde x=x/\sqrt{\bar\alpha}$ and $\sigma=\sqrt{(1-\bar\alpha)/\bar\alpha}$
#
# Divide the forward process by $\sqrt{a_t}$: $\ \tilde x_t:=x_t/\sqrt{a_t}=x_0+\sigma_t\,\epsilon$. In these coordinates
# the VP process becomes the **variance-exploding** process of notebook 01, with noise level $\sigma$ (Karras et al. 2022).
# Its probability-flow ODE (the same derivation as §1.2 with $f=0$, $g^2=d\sigma^2/dt$) is
#
# $$\frac{d\tilde x}{d\sigma}=-\sigma\,\nabla_{\tilde x}\log p_\sigma(\tilde x)=\hat\epsilon=\frac{\tilde x-D(\tilde x;\sigma)}{\sigma},
# \qquad D(\tilde x;\sigma)=\hat x_0 \text{ (the posterior-mean denoiser).}$$
#
# Now divide the DDIM update ($\eta=0$) by $\sqrt{a_s}$, and use $\hat x_0=\tilde x_t-\sigma_t\hat\epsilon$:
#
# $$\tilde x_s=\hat x_0+\sigma_s\hat\epsilon=\tilde x_t+(\sigma_s-\sigma_t)\,\hat\epsilon(\tilde x_t,\sigma_t).$$
#
# That is **one Euler step of $d\tilde x/d\sigma=\hat\epsilon$**. The same equation, rearranged, gives the most useful way to
# picture a DDIM step:
#
# $$\boxed{\;\tilde x_s=\frac{\sigma_s}{\sigma_t}\,\tilde x_t+\Big(1-\frac{\sigma_s}{\sigma_t}\Big)\,\hat x_0\;}$$
#
# *Move the fraction $1-\sigma_s/\sigma_t$ of the way from where you are to where the denoiser thinks the data is.*
#
# **The exact step, for comparison.** Substitute $y=\tilde x/\sigma$ and $u=1/\sigma$ (the square root of the
# signal-to-noise ratio). Then
# $\frac{dy}{du}=\frac{dy/d\sigma}{du/d\sigma}=\frac{(d\tilde x/d\sigma-\tilde x/\sigma)/\sigma}{-1/\sigma^2}=\tilde x-\sigma\frac{d\tilde x}{d\sigma}=D(\tilde x;\sigma)$:
# the right-hand side is the denoiser output itself, and DDIM is Euler's method in $(u,y)$. Integrating exactly instead
# of with one Euler step gives
#
# $$\tilde x_s=\frac{\sigma_s}{\sigma_t}\,\tilde x_t+\sigma_s\!\int_{\sigma_s}^{\sigma_t}\frac{D(\tilde x(\sigma);\sigma)}{\sigma^2}\,d\sigma,
# \qquad\text{so}\qquad
# \text{DDIM error}=\sigma_s\!\int_{\sigma_s}^{\sigma_t}\frac{D(\tilde x(\sigma);\sigma)-D(\tilde x_t;\sigma_t)}{\sigma^2}\,d\sigma .$$
#
# (Since $\sigma_s\int_{\sigma_s}^{\sigma_t}\sigma^{-2}d\sigma=1-\sigma_s/\sigma_t$, the DDIM step is this formula with $D$ frozen
# at its starting value.) Two consequences follow, and we check each below.
#
# 1. **It is exact whenever $\hat x_0=D$ is constant along the step**, and changes in $D$ count with weight
#    $\sigma_s/\sigma^2$. Changes of $\hat x_0$ at noise levels far above the target $\sigma_s$ therefore cost little.
#    Equivalently, the exact trajectory is a straight line $\tilde x(\sigma)=\hat x_0+\sigma\hat\epsilon$ wherever $D$ is constant.
# 2. **It can never overshoot.** For any step size it is a convex combination of the current point and the denoised guess.
#    Euler or Heun in $(x,t)$ have no such guarantee.

# %%
def sigma_fn(t):
    """EDM noise level sigma(t) = sqrt((1 - abar) / abar) of the VP process at time t."""
    return math.sqrt(1.0 / abar_fn(t) - 1.0)


def t_of_sigma(sig):
    """Inverse of sigma_fn: solve BETA_MIN t + (BETA_MAX - BETA_MIN) t^2 / 2 = log(1 + sigma^2) for t."""
    L = math.log1p(sig * sig)
    return (-BETA_MIN + math.sqrt(BETA_MIN ** 2 + 2 * (BETA_MAX - BETA_MIN) * L)) / (BETA_MAX - BETA_MIN)


SIGMA_MAX, SIGMA_MIN = sigma_fn(1.0), sigma_fn(T_MIN)
print(f"sigma ranges from {SIGMA_MAX:.1f} (t = 1) down to {SIGMA_MIN:.4f} (t = t_min);  "
      f"round trip t -> sigma -> t error {abs(t_of_sigma(sigma_fn(0.37)) - 0.37):.1e}")


def sigmas_of_times(ts):
    return np.array([sigma_fn(t) if t > 0 else 0.0 for t in ts])


@torch.no_grad()
def euler_sigma_sampler(eps_fn, x, sigmas, *, return_traj=False):
    """Euler on d x~/d sigma = eps_hat in the variables x~ = x / sqrt(abar) (= x * sqrt(1 + sigma^2)). Input/output in VP units."""
    xt = x * math.sqrt(1 + sigmas[0] ** 2)
    traj = [x]
    for s_t, s_s in zip(sigmas[:-1], sigmas[1:]):
        eps = eps_fn(xt / math.sqrt(1 + s_t ** 2), t_of_sigma(s_t))   # the eps-network sees the VP-scaled input
        xt = xt + (s_s - s_t) * eps
        traj.append(xt / math.sqrt(1 + s_s ** 2))
    return traj if return_traj else traj[-1]


# CHECK: DDIM (eta = 0) and Euler in (x~, sigma) are the same algorithm
for n in (5, 50):
    a = ddim_sampler(eps_exact, x_T, uniform_grid(n))
    b = euler_sigma_sampler(eps_exact, x_T, sigmas_of_times(uniform_grid(n)))
    print(f"CHECK: DDIM == Euler in (x~, sigma), {n} steps: max |diff| = {(a - b).abs().max():.1e}")
    assert torch.allclose(a, b, atol=1e-9)

# %% [markdown]
# **Consequence 1: straight lines.** We follow a dozen exact ODE trajectories (DDIM with 400 log-spaced steps) and record
# $\tilde x$ and the denoiser output $\hat x_0$ along the way.

# %%
N_LINES = 12
sig_fine = np.exp(np.linspace(math.log(SIGMA_MAX), math.log(SIGMA_MIN), 400))
xt = x_T[:N_LINES] * math.sqrt(1 + SIGMA_MAX ** 2)
rec_xt, rec_x0 = [], []
for s_t, s_s in zip(sig_fine[:-1], sig_fine[1:]):
    eps = eps_exact(xt / math.sqrt(1 + s_t ** 2), t_of_sigma(s_t))
    rec_xt.append(xt.clone())
    rec_x0.append(xt - s_t * eps)                    # D(x~; sigma) = x0_hat
    xt = xt + (s_s - s_t) * eps
rec_xt, rec_x0, sig_rec = torch.stack(rec_xt), torch.stack(rec_x0), sig_fine[:-1]

# CHECK the exact-step formula on the recorded (fine) path, for one big step sigma: 2 -> 0.3
i0, i1 = np.abs(sig_rec - 2.0).argmin(), np.abs(sig_rec - 0.3).argmin()
s0, s1, s_seg = sig_rec[i0], sig_rec[i1], torch.tensor(sig_rec[i0:i1 + 1])
integral = -torch.trapezoid(rec_x0[i0:i1 + 1] / s_seg[:, None, None] ** 2, s_seg, dim=0)   # sigma decreases along the path
x_formula = s1 / s0 * rec_xt[i0] + s1 * integral
x_one_ddim = s1 / s0 * rec_xt[i0] + (1 - s1 / s0) * rec_x0[i0]
print(f"CHECK sigma {s0:.2f} -> {s1:.2f}: exact-step formula vs fine path: mean error "
      f"{(x_formula - rec_xt[i1]).norm(dim=1).mean():.1e};  one DDIM step: {(x_one_ddim - rec_xt[i1]).norm(dim=1).mean():.2f}")
assert (x_formula - rec_xt[i1]).norm(dim=1).mean() < 0.05 * (x_one_ddim - rec_xt[i1]).norm(dim=1).mean()

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
cols = plt.get_cmap("tab20")(np.arange(N_LINES))
for i in range(N_LINES):
    axes[0].plot(sig_rec, rec_xt[:, i, 0], color=cols[i], lw=0.9)
    axes[1].plot(sig_rec, rec_xt[:, i, 0], color=cols[i], lw=0.9)
    axes[2].semilogx(sig_rec, rec_x0[:, i, 0], color=cols[i], lw=0.9)
axes[0].set(xlabel=r"$\sigma$", ylabel=r"$\tilde x_1$", title=r"ODE paths in $(\sigma,\tilde x)$: straight lines", xlim=(0, SIGMA_MAX))
axes[1].set(xlabel=r"$\sigma$", ylabel=r"$\tilde x_1$", title=r"zoom: $\sigma\leq 4$, where the paths bend", xlim=(0, 4), ylim=(-6, 6))
axes[2].set(xlabel=r"$\sigma$ (log scale)", ylabel=r"$\hat x_{0,1}=D_1(\tilde x;\sigma)$",
            title=r"denoiser output along each path", xlim=(SIGMA_MIN, SIGMA_MAX))
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# Over almost the whole range of $\sigma$ the paths are straight lines (left). When $\sigma$ is much larger than the data,
# the noisy point carries little information about which mode it came from. Then $\hat x_0$ is a heavily shrunk guess,
# close to the data mean, which moves away from the mean roughly like $1/\sigma$ as $\sigma$ falls (right). Those changes
# are cheap, because the exact-step formula weights them by $\sigma_s/\sigma^2$ (the printed check confirms the formula).
# The bending that matters happens where $\sigma$ is comparable to the structure of the data (roughly
# $0.1\lesssim\sigma\lesssim5$, measured in §6). There $\hat x_0$ commits to a mode and then to a point within it, and that
# window is where DDIM's error comes from. §6 asks where the steps should go.

# %% [markdown]
# The printed check, drawn: one big DDIM step through that window, from $\sigma_t\approx2$ to $\sigma_s\approx0.3$. On the
# left, the step follows the straight line from the start towards the denoiser's current guess $\hat x_0$ (the cross at
# $\sigma=0$), tangent to the exact path, and stops $1-\sigma_s/\sigma_t\approx85\%$ of the way along, while the exact path
# bends away as the guess sharpens. On the right, the same step in the plane: on every one of the twelve paths DDIM lands
# nearer the centre than the exact path (printed below), which is how few-step DDIM pulls samples inwards in the §2.1
# gallery.

# %%
viz_frac = 1 - s1 / s0                               # the fraction of the way to x0_hat that one DDIM step covers
viz_show = [1, 2, 6, 7, 9, 11]                       # six of the N_LINES paths, spread around the plane
viz_i = 6                                            # the path drawn on the left
viz_sel = sig_rec <= 3.0
viz_fig, viz_axes = plt.subplots(1, 2, figsize=(12.5, 4.9), layout="constrained", gridspec_kw={"width_ratios": [1.25, 1]})
viz_ax = viz_axes[0]
viz_start, viz_exact = float(rec_xt[i0, viz_i, 0]), float(rec_xt[i1, viz_i, 0])
viz_land, viz_guess = float(x_one_ddim[viz_i, 0]), float(rec_x0[i0, viz_i, 0])
viz_ax.plot(sig_rec[viz_sel], rec_xt[:, viz_i, 0].numpy()[viz_sel], color="k", lw=1.3, label="exact ODE path")
viz_ax.plot([s0, 0.0], [viz_start, viz_guess], "--", color=COL["DDIM"], lw=1.3,
            label=r"DDIM step: the straight line to $\hat x_0$")
viz_ax.plot(s0, viz_start, "o", mfc="white", color="k", ms=7, zorder=3)
viz_ax.plot(s1, viz_exact, "o", color="k", ms=6, zorder=3)
viz_ax.plot(s1, viz_land, "s", color=COL["DDIM"], ms=7, zorder=3)
viz_ax.plot(0.0, viz_guess, "X", color=COL["DDIM"], ms=9, zorder=3)
viz_ax.annotate("", xy=(s1 - 0.04, viz_exact), xytext=(s1 - 0.04, viz_land), arrowprops=dict(arrowstyle="<->", lw=1.0))
viz_ax.text(s1 - 0.07, 0.5 * (viz_exact + viz_land), "error", ha="right", va="center")
viz_ax.annotate(r"start $\tilde x_t$", (s0, viz_start), xytext=(0, 10), textcoords="offset points", ha="center")
viz_ax.annotate(r"exact path at $\sigma_s$", (s1, viz_exact), xytext=(4, 10), textcoords="offset points")
viz_ax.annotate(f"DDIM lands {viz_frac:.0%} of the way to the guess", (s1, viz_land), xytext=(12, -4),
                textcoords="offset points", color=COL["DDIM"], va="center")
viz_ax.annotate(r"$\hat x_0=D(\tilde x_t;\sigma_t)$: the denoiser's guess at the start of the step (still blurry)",
                (0.0, viz_guess), xytext=(12, -4), textcoords="offset points", color=COL["DDIM"], va="center")
for viz_s in (s0, s1):
    viz_ax.axvline(viz_s, color="0.7", lw=0.8, ls=":", zorder=0)
viz_ax.set(xlim=(-0.08, 3.0), ylim=(viz_guess - 0.45, None), xlabel=r"noise level $\sigma$",
           ylabel=r"first coordinate $\tilde x_1$",
           title=f"one path, one DDIM step from $\\sigma_t={s0:.1f}$ to $\\sigma_s={s1:.1f}$")
viz_ax.legend(loc="upper left", fontsize=7.5)
viz_ax.grid(alpha=0.3)
viz_ax = viz_axes[1]
viz_ax.contourf(viz_grid, viz_grid, viz_p0, levels=np.linspace(0, 1, 11), cmap=CMAP_DENSITY)
for viz_k in viz_show:
    viz_lw = 1.8 if viz_k == viz_i else 1.0
    viz_ax.plot(rec_xt[i0:i1 + 1, viz_k, 0].numpy(), rec_xt[i0:i1 + 1, viz_k, 1].numpy(), color="k", lw=viz_lw)
    viz_ax.plot(*zip(rec_xt[i0, viz_k].numpy(), rec_x0[i0, viz_k].numpy()), "--", color=COL["DDIM"], lw=viz_lw)
    viz_ax.plot(*rec_x0[i0, viz_k].numpy(), "X", color=COL["DDIM"], ms=6.5)
    viz_ax.plot(*x_one_ddim[viz_k].numpy(), "s", color=COL["DDIM"], ms=5)
    viz_ax.plot(*rec_xt[i1, viz_k].numpy(), "o", color="k", ms=5)
    viz_ax.plot(*rec_xt[i0, viz_k].numpy(), "o", mfc="white", color="k", ms=5.5)
viz_ax.set(xlim=(-LIM, LIM), ylim=(-LIM, LIM), aspect="equal", xlabel=r"$\tilde x_1$", ylabel=r"$\tilde x_2$",
           title=f"the same step on {len(viz_show)} of the {N_LINES} paths, in the plane\n"
                 "(thick: the path on the left; grey: the data)")
viz_fig.suptitle(r"A DDIM step heads straight for the denoiser's current guess $\hat x_0$ and covers the fraction "
                 r"$1-\sigma_s/\sigma_t$ of the way." "\n"
                 r"At $\sigma\approx2$ that guess is still a blurry average near the centre; the exact path bends out to a "
                 "mode as the guess sharpens, so DDIM lands short")
plt.show()
viz_r_ddim, viz_r_exact = x_one_ddim.norm(dim=1), rec_xt[i1].norm(dim=1)
print(f"distance from the centre at sigma_s = {s1:.2f}: DDIM's landing point is nearer than the exact path on "
      f"{int((viz_r_ddim < viz_r_exact).sum())} of {N_LINES} paths (mean {viz_r_ddim.mean():.2f} vs {viz_r_exact.mean():.2f})")

# %% [markdown]
# **Consequence 2: stability.** The cleanest test is one-dimensional Gaussian data $x_0\sim\mathcal N(0,\lambda)$ (in this
# subsection $\lambda$ is the data variance, not a log-SNR). Every
# quantity is then available in closed form: $\hat\epsilon(x,t)=\sqrt{1-a_t}\,x/(\lambda a_t+1-a_t)$, and the exact ODE maps
# $x_T=1$ to $\sqrt\lambda/\sqrt{\lambda a_1+1-a_1}$. Since the ODE is linear, the sampler's output from $x_T=1$ *is* the
# standard deviation of the distribution it generates. We vary the data variance $\lambda$ over seven orders of magnitude.
#
# All three samplers end with the same denoising step at $t_{\min}$, and that step sets a limit of its own. Structure finer
# than $\sigma_{\min}\approx0.01$ cannot be resolved: even the *exact* ODE down to $t_{\min}$, followed by
# $\hat x_0=\mathbb E[x_0\mid x_{t_{\min}}]$, returns a std of about $\sqrt\lambda\cdot\sqrt{\lambda/(\lambda+\sigma_{\min}^2)}$
# (dotted line). We also run Euler with a plain Euler step to $t=0$ (thin dashed line), to see what the special last
# step buys.

# %%
def gaussian_eps_fn(lam):
    """Exact noise predictor for 1-D data x0 ~ N(0, lam): q_t = N(0, lam * abar + 1 - abar)."""
    def eps_fn(x, t):
        ab = abar_fn(t)
        return math.sqrt(1 - ab) * x / (lam * ab + 1 - ab)
    return eps_fn


lams = np.logspace(-5, 2, 29)
one = torch.ones(1, 1, dtype=torch.float64)
a_1, a_m = abar_fn(1.0), abar_fn(T_MIN)
true_std = np.sqrt(lams) / np.sqrt(lams * a_1 + 1 - a_1)
# best possible with a final denoising step: x_{t_min} from the exact ODE, then x0_hat = sqrt(a_m) lam / (lam a_m + 1 - a_m) x
best_std = math.sqrt(a_m) * lams / np.sqrt(lams * a_m + 1 - a_m) / np.sqrt(lams * a_1 + 1 - a_1)
PLAIN = "Euler ODE, plain last step"
gen_std = {}
for nfe in (10, 50):
    for name in ["Euler ODE", "Heun ODE", "DDIM"]:
        n, _ = steps_for_nfe(name, nfe)
        out = [SAMPLERS[name][0](gaussian_eps_fn(lam), one, uniform_grid(n)).abs().item() for lam in lams]
        gen_std[(name, nfe)] = np.array(out)
    gen_std[(PLAIN, nfe)] = np.array([euler_ode_sampler(gaussian_eps_fn(lam), one, uniform_grid(nfe),
                                                        denoise_last=False).abs().item() for lam in lams])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
eig_std = torch.linalg.eigvalsh(gmm.covs).sqrt()             # std of each component along its principal axes
toy_band = (eig_std.min().item(), math.sqrt(x_true.var(0).max().item()))
for ax, nfe in zip(axes, (10, 50)):
    ax.axvspan(*toy_band, color="0.9", label="range of scales in our 2-D mixture")
    ax.loglog(true_std, true_std, "k--", lw=0.8, label="exact (generated std = true std)")
    ax.loglog(true_std, best_std, "k:", lw=1.2, label=r"exact ODE to $t_{\min}$, then $\hat x_0$ (best possible)")
    ax.loglog(true_std, gen_std[(PLAIN, nfe)], "--", lw=1.0, color=COL["Euler ODE"], alpha=0.6, label=PLAIN)
    for name in ["Euler ODE", "Heun ODE", "DDIM"]:
        ax.loglog(true_std, gen_std[(name, nfe)], "o-", ms=3, color=COL[name], label=name)
    ax.set(xlabel=r"true data std $\sqrt{\lambda}$", title=f"1-D Gaussian data, NFE ≈ {nfe} (uniform $t$ grid)",
           ylim=(3e-4, 1e2))
    ax.grid(alpha=0.3, which="both")
axes[0].set_ylabel("std of the generated samples")
axes[1].legend(loc="upper left", fontsize=7)        # the right panel's upper left is empty
plt.tight_layout()
plt.show()
i_01 = np.abs(np.sqrt(lams) - eig_std.min().item()).argmin()      # the narrowest scale in our mixture (std 0.1)
for nfe in (10, 50):
    for i in (0, i_01):
        print(f"NFE {nfe}, true std {math.sqrt(lams[i]):.3f}: generated/true std  "
              + ", ".join(f"{name} {gen_std[(name, nfe)][i] / true_std[i]:.2g}"
                          for name in [PLAIN, "Euler ODE", "Heun ODE", "DDIM"])
              + f"  (best possible {best_std[i] / true_std[i]:.2g})")

# %% [markdown]
# * **Euler and Heun in $(x,t)$ are exact for $\lambda=1$**: unit-variance Gaussian data is a fixed point of the VP process,
#   so the ODE does nothing. Away from $\lambda=1$ they degrade in both directions. For small $\lambda$ the ODE becomes
#   **stiff** near $t\to0$: its right-hand side contains $1/(\lambda a_t+1-a_t)$, which explodes. Heun's corrector then
#   overshoots, its output oscillates in sign (the V-shaped dips are where it happens to cross zero), and it can blow up
#   completely. Euler's errors are smaller and can go either way; at low NFE it gives too much spread for small
#   $\lambda$. Adding steps helps only slowly, because the step must be small
#   compared with the fastest time scale of the ODE, which is set by the smallest data variance. A higher order does not
#   cure stiffness.
# * **The last step matters.** With a plain Euler step from $t_{\min}$ to 0 (thin dashed line), Euler in $(x,t)$ leaves
#   residual noise of order $\sigma_{\min}$ in every low-variance direction, several times the true std for the smallest
#   $\lambda$ (printed above). In $(\tilde x,\sigma)$
#   this cannot happen: the Euler step to $\sigma=0$ *is* the denoiser $\hat x_0$. DDIM gets the right last step for free;
#   in $(x,t)$ it has to be special-cased, as all our samplers do.
# * **DDIM never overshoots.** Its generated std is never larger than the truth, because each step is a convex combination
#   of $\tilde x$ and $\hat x_0$. (For Gaussian data one can show that every step contracts slightly *more* than the exact
#   flow.) Its failure mode is the opposite: **over-smoothing**. With few steps, the long last step of the uniform-$t$ grid
#   (at NFE 10, from $\sigma\approx0.4$ straight down to $\sigma_{\min}$) collapses a narrow component towards its
#   posterior mean.
#   For tiny $\lambda$ the final denoising at $t_{\min}$ sets the limit (dotted line): at the smallest $\lambda$ shown,
#   even the exact ODE returns only about 0.3 of the true std, and the fraction keeps falling as $\lambda$ shrinks. DDIM
#   sits on that limit; Euler and Heun land above it only through their own overshoot.
#
# This explains the §2 result. Our 2-D mixture lives in the shaded band. There nothing blows up, but at NFE 10 DDIM
# under-disperses the narrowest components about three-fold and Heun-in-$t$ over-disperses them about 2.5-fold, while
# Euler-in-$t$ stays within about 20% of the truth (printed above). So on this toy Euler-in-$t$ wins. Real data is different. Images have a few directions of very large variance
# (the coarse layout, $\lambda\gg1$, where Euler-in-$t$ also degrades), and a vast number of directions with tiny variance
# (fine textures, pixel-level detail), where Heun-in-$t$ blows up and Euler-in-$t$ overshoots. DDIM's errors are milder
# and always on the safe side: a little blur, never noise or blow-up. Its trajectories are close to straight lines
# (Karras et al. 2022), which is why it became the robust default (try it in Exercise 2). The convex-combination
# structure is also the starting point of the "exponential integrator" family of fast solvers. These integrate the linear
# part exactly. DPM-Solver (Lu et al. 2022) approximates how $\hat\epsilon$ changes along the path; DPM-Solver++
# approximates how $\hat x_0$ changes, i.e. the integral in the exact-step formula above, by a polynomial in $\log\sigma$.
# Both reduce to DDIM at first order.
#
# ## 4 · Trajectories: smooth ODE paths, jagged SDE paths
#
# Same starting noise, two samplers: the ODE (DDIM, 200 steps) and the reverse SDE (DDPM ancestral, 1000 steps). We keep
# the full trajectories and check two claims from §1. First, the marginals agree at intermediate times. Second, the ODE
# is a deterministic map while the SDE is not.

# %%
N_TRAJ = 1000 if FAST else 2000
x_T_traj = torch.randn(N_TRAJ, 2, generator=torch.Generator().manual_seed(21), dtype=torch.float64)
grid_ode, grid_sde = uniform_grid(200), uniform_grid(1000)
ode_traj = torch.stack(ddim_sampler(eps_exact, x_T_traj, grid_ode, return_traj=True))
sde_traj = torch.stack(ddpm_sampler(eps_exact, x_T_traj, grid_sde, gen=torch.Generator().manual_seed(22),
                                    return_traj=True))

# CHECK: determinism. Re-running the ODE reproduces it bit for bit; a new seed sends SDE particles elsewhere.
ode_again = ddim_sampler(eps_exact, x_T_traj, grid_ode)
sde_again = ddpm_sampler(eps_exact, x_T_traj, grid_sde, gen=torch.Generator().manual_seed(23))
print(f"ODE re-run:          max |diff| = {(ode_again - ode_traj[-1]).abs().max():.1e}")
print(f"SDE with a new seed: mean |diff| = {(sde_again - sde_traj[-1]).norm(dim=1).mean():.2f}  (same x_T!)")

# CHECK: both have the exact marginals q_t at intermediate times (sliced W1 to fresh exact samples of q_t)
print("sliced W1 to exact samples of q_t:")
for t_chk in (0.5, 0.2, 0.05):
    i_o, i_s = np.abs(grid_ode - t_chk).argmin(), np.abs(grid_sde - t_chk).argmin()
    q_t = gmm.noised(abar_fn(grid_ode[i_o]))
    ref_t = q_t.sample(N_TRAJ, generator=torch.Generator().manual_seed(24))
    fl_t = dm.sliced_wasserstein(q_t.sample(N_TRAJ, generator=torch.Generator().manual_seed(25)), ref_t)
    print(f"  t = {t_chk}:  ODE {dm.sliced_wasserstein(ode_traj[i_o], ref_t):.4f}   "
          f"SDE {dm.sliced_wasserstein(sde_traj[i_s], ref_t):.4f}   exact vs exact {fl_t:.4f}")

# %% [markdown]
# The marginal check, drawn: histograms of the first coordinate of all particles at the three checked times and at the
# end, against the exact marginal $p_t(x_1)$. ODE and SDE particles follow the same curve at every time, up to sampling
# noise; the next figure shows how differently they get there.

# %%
viz_bins = np.linspace(-LIM, LIM, 41)
viz_fig, viz_axes = plt.subplots(1, 4, figsize=(13, 3.1), layout="constrained")
for viz_ax, viz_t in zip(viz_axes, (0.5, 0.2, 0.05, 0.0)):
    viz_io, viz_is = np.abs(grid_ode - viz_t).argmin(), np.abs(grid_sde - viz_t).argmin()
    viz_ax.hist(ode_traj[viz_io, :, 0].numpy(), bins=viz_bins, density=True, histtype="step", lw=1.3,
                color=COL["DDIM"], label="ODE (DDIM) particles")
    viz_ax.hist(sde_traj[viz_is, :, 0].numpy(), bins=viz_bins, density=True, histtype="step", lw=1.3, ls="--",
                color=COL["DDPM ancestral"], label="SDE (DDPM) particles")
    viz_ax.plot(XS_G, x1_marginal_density([grid_ode[viz_io]], XS_G)[:, 0], color="k", lw=1, label="exact $p_t(x_1)$")
    viz_ax.set(xlabel="first coordinate $x_1$", title=f"$t={grid_ode[viz_io]:.2f}$", xlim=(-LIM, LIM))
viz_axes[0].set_ylabel("density")
viz_fig.legend(*viz_axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
viz_fig.suptitle(f"Same marginals: at every $t$ the {N_TRAJ} ODE and SDE particles both follow the exact $p_t(x_1)$, "
                 "although their paths differ")
plt.show()

# %%
N_SHOW = 15
angle = torch.atan2(x_T_traj[:, 1], x_T_traj[:, 0]).numpy()   # colour code: direction of the starting noise
fig, axes = plt.subplots(2, 2, figsize=(10, 8.4))
for col, (name, traj, grid) in enumerate([("ODE (DDIM, 200 steps)", ode_traj, grid_ode),
                                          ("reverse SDE (DDPM, 1000 steps)", sde_traj, grid_sde)]):
    ax = axes[0, col]
    show_density_tx(ax)
    for i in range(N_SHOW):
        ax.plot(grid, traj[:, i, 0].numpy(), lw=0.8, color=plt.get_cmap("hsv")((angle[i] + np.pi) / (2 * np.pi)))
    ax.set_title(f"{name}: paths of $x_1$ from the same $x_T$")
    ax = axes[1, col]
    ax.scatter(traj[-1, :, 0], traj[-1, :, 1], c=angle, cmap="hsv", s=3, alpha=0.8, linewidths=0)
    ax.set(xlim=(-LIM, LIM), ylim=(-LIM, LIM), aspect="equal", xticks=[], yticks=[],
           title=f"{name}: samples, coloured by the angle of $x_T$")
plt.tight_layout()
plt.show()

# %% [markdown]
# The ODE paths (top left) are smooth. They barely move while $t\gtrsim0.5$ and then glide into a mode. The SDE paths (top
# right) jitter all the way down, yet the printed check shows that both clouds match the marginal $q_t$ at every checked
# time, up to roughly the sampling noise (compare with the "exact vs exact" column).
# The bottom row shows the difference that matters. The ODE is a **smooth, deterministic map** from noise to data:
# neighbouring noise vectors land next to each other, so the colours form coherent regions, and each mode receives a
# contiguous wedge of noise space. That map is invertible (run the ODE backwards), so it is a **continuous normalising
# flow** (Chen et al. 2018). Notebook 08 trains such flows directly. The SDE forgets where it started: the colours are mixed.
#
# ### 4.1 Why bother with noise? The SDE forgets errors
#
# In §1.2 we saw that reverse SDE = ODE + Langevin dynamics targeting $p_t$. Langevin dynamics pulls *any* distribution
# towards $p_t$, so the SDE should **repair** errors in the state that the ODE would faithfully transport to the output.
# To isolate this, we take *exact* samples of $q_{t_p}$, corrupt them (scale by 1.3, an over-dispersed state like a bad
# step or an imperfect denoiser would produce), and continue with exact scores from $t_p$ down to 0. Besides the overall
# error we measure two things separately: the **mode weights** (the fraction of samples in the four light outer
# components, which hold 25% of the mass) and the **shape within each mode** (the mean squared Mahalanobis distance of a
# sample to its most likely component, about 2 for exact samples, as for any 2-D Gaussian).

# %%
INV_COVS = torch.linalg.inv(gmm.covs)


def mode_stats(x):
    """(fraction of samples assigned to the light outer components, mean squared Mahalanobis distance to the assigned one)."""
    lab = gmm.responsibilities(x).argmax(1)              # most likely component of each sample
    diff = x - gmm.means[lab]
    maha2 = torch.einsum("ni,nij,nj->n", diff, INV_COVS[lab], diff)
    return (lab % 2 == 1).double().mean().item(), maha2.mean().item()   # odd components = outer ring (weight 1/16 each)


PERT_TIMES = [0.9, 0.7, 0.5, 0.35, 0.2, 0.1, 0.05]
x0_p = gmm.sample(N_EVAL, generator=torch.Generator().manual_seed(31)).double()
z_p = torch.randn(N_EVAL, 2, generator=torch.Generator().manual_seed(32), dtype=torch.float64)
PERT_NAMES = ["ODE (DDIM)", "SDE (DDPM)", "no corruption"]
pert = {name: {"sw": [], "outer": [], "spread": []} for name in PERT_NAMES}
for tp in PERT_TIMES:
    ab = abar_fn(tp)
    x_tp = math.sqrt(ab) * x0_p + math.sqrt(1 - ab) * z_p           # exact samples of q_tp ...
    grid = np.concatenate([np.linspace(tp, T_MIN, 200), [0.0]])
    outs = {"ODE (DDIM)": ddim_sampler(eps_exact, 1.3 * x_tp, grid),   # ... corrupted, then denoised
            "SDE (DDPM)": ddpm_sampler(eps_exact, 1.3 * x_tp, grid, gen=torch.Generator().manual_seed(33)),
            "no corruption": ddim_sampler(eps_exact, x_tp, grid)}
    for name, x in outs.items():
        outer, spread = mode_stats(x)
        pert[name]["sw"].append(quality(x)[1])
        pert[name]["outer"].append(outer)
        pert[name]["spread"].append(spread)
exact_outer, exact_spread = mode_stats(x_true)
print(f"exact samples: outer fraction {exact_outer:.3f}, within-mode spread {exact_spread:.2f}")
for name in PERT_NAMES:
    print(f"{name:14s} " + "  ".join(f"t_p={tp}: W1 {sw:.3f} outer {o:.2f} spread {s:.2f}" for tp, sw, o, s in
                                     zip(PERT_TIMES, pert[name]["sw"], pert[name]["outer"], pert[name]["spread"])))

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
style = {"ODE (DDIM)": ("o-", COL["DDIM"], 6), "SDE (DDPM)": ("o--", COL["DDPM ancestral"], 6),
         "no corruption": ("o:", "k", 3)}
panels = [("sw", "sliced $W_1$ of the final samples", "overall error"),
          ("outer", "fraction in the outer components", "mode weights"),
          ("spread", "mean squared Mahalanobis distance", "shape within each mode")]
for ax, (key, ylabel, title) in zip(axes, panels):
    for name in PERT_NAMES:
        fmt, c, ms = style[name]
        ax.plot(PERT_TIMES, pert[name][key], fmt, color=c, ms=ms, label=name)
    ax.set(xlabel=r"time $t_p$ of the corruption ($\times1.3$)", ylabel=ylabel, title=title, xlim=(1, 0))
    ax.grid(alpha=0.3)
axes[0].set_yscale("log")
axes[0].axhspan(floor[:, 1].mean() - 2 * floor[:, 1].std(), FLOOR_HI[1], color=COL["floor"], alpha=0.6, label="floor")
axes[1].axhline(exact_outer, color=COL["data"], lw=1, label="exact samples")
axes[2].axhline(exact_spread, color=COL["data"], lw=1, label="exact samples")
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# The ODE carries the corruption into the output, whenever it happens: the modes come out too wide (right), and too much
# mass ends up in the outer components (middle; except at $t_p=0.05$, when the modes are already so far apart that the
# scaling moves samples within their modes rather than between them). The SDE **erases** corruption made at high noise.
# There the Langevin part runs at rate $\beta(t)/2$, which is large, and $p_t$ is nearly Gaussian, so the chain mixes
# fast. Corruption made at low noise is only partly repaired. The SDE still repairs much of the shape *within* each mode
# (right; fully down to $t_p\approx0.2$), but not the mode weights (middle): at $t_p=0.35$ it removes only part of the
# excess, and from $t_p\approx0.2$ on its outer fraction is nearly as wrong as the ODE's. Once the modes have separated, Langevin dynamics cannot move mass
# between them in any reasonable time: this is the metastability of notebook 03, §3a. Whatever split of mass between modes
# exists by then is final.
#
# This is the real reason to keep some stochasticity. With an imperfect network and a generous budget, a stochastic
# sampler can beat a deterministic one (Karras et al. 2022 add a tunable amount of "churn" for this). We test it on a
# trained network in §8.
#
# ## 5 · DDIM inversion: from data back to noise
#
# A deterministic ODE can be integrated **forwards** too: from data $x_0$ to its latent $x_T$. Swap the roles of $t$ and
# $s$ in the DDIM update and step to *higher* noise:
#
# $$x_s=\sqrt{a_s}\,\hat x_0(x_t,t)+\sqrt{1-a_s}\,\hat\epsilon(x_t,t),\qquad s>t .$$
#
# **Why it is only approximate.** The sampler steps from $s$ down to $t$ using $\hat\epsilon(x_s,s)$, the prediction at the
# *noisier* end. Inversion has not yet computed $x_s$, so it uses $\hat\epsilon(x_t,t)$ instead. The forward and backward
# Euler steps use slopes from opposite ends of the interval, so they do not cancel. The mismatch per step is
# $O(h^2)$, and the round trip (invert, then sample) has an $O(h)$ error. It vanishes as the steps shrink, or wherever
# $\hat\epsilon$ is constant along the path.
#
# **Why it matters.** Inversion gives every real data point a latent code that the sampler maps back to it. Editing
# methods (notebook 07) invert an image, change something (the prompt, the guidance, part of the latent) and sample again,
# so that everything not edited is preserved.

# %%
@torch.no_grad()
def ddim_invert(eps_fn, x0, ts_up, *, abar=abar_fn):
    """Deterministic DDIM run forwards in time (data -> noise) on the ascending grid ts_up = [0, t_1, ..., 1].

    eps is evaluated at the current, less noisy point: the approximation discussed in the text."""
    x = x0
    for t, s in zip(ts_up[:-1], ts_up[1:]):
        a_t, a_s = abar(t), abar(s)
        eps = eps_fn(x, max(t, ts_up[1]))     # the network never saw t = 0: use the first grid time instead
        x0_hat = (x - math.sqrt(1 - a_t) * eps) / math.sqrt(a_t)
        x = math.sqrt(a_s) * x0_hat + math.sqrt(1 - a_s) * eps
    return x


# CHECK: our inversion reproduces dm.ddim_invert on the discrete schedule
x0_inv32 = gmm.sample(300, generator=torch.Generator().manual_seed(41))
for n in (10, 50):
    ours = ddim_invert(eps_disc, x0_inv32, disc_grid(n)[::-1], abar=abar_disc)
    theirs = dm.ddim_invert(exact_model, sched, x0_inv32, n_steps=n)
    print(f"CHECK ddim_invert == dm.ddim_invert ({n} steps): max |diff| = {(ours - theirs).abs().max():.1e}")
    assert torch.allclose(ours, theirs, atol=1e-4)

# Round trip: invert with n steps, then sample back with the same n steps
x0_inv, lab_inv = gmm.sample(N_EVAL, return_labels=True, generator=torch.Generator().manual_seed(42))
x0_inv = x0_inv.double()
INV_STEPS = [5, 10, 20, 50, 100, 250, 1000]
roundtrip = []
for n in INV_STEPS:
    z_inv = ddim_invert(eps_exact, x0_inv, uniform_grid(n)[::-1])
    x_back = ddim_sampler(eps_exact, z_inv, uniform_grid(n))
    roundtrip.append((x_back - x0_inv).norm(dim=1).mean().item())
    if n == max(INV_STEPS):
        z_accurate = z_inv                     # the most accurate latents, for the plots below
print("round-trip error (mean ||x0 - sample(invert(x0))||): "
      + ", ".join(f"{n} steps {e:.1e}" for n, e in zip(INV_STEPS, roundtrip)))
# CHECK: the ODE pushes the data distribution onto N(0, I)
cov = torch.cov(z_accurate.T)
print(f"latents of {N_EVAL} data points: mean {z_accurate.mean(0).numpy().round(3)}, covariance {cov.numpy().round(3).tolist()}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
ax = axes[0]
ax.loglog(INV_STEPS, roundtrip, "o-", color=COL["DDIM"], label="DDIM invert + DDIM sample")
ax.loglog(INV_STEPS, roundtrip[2] * (np.array(INV_STEPS) / INV_STEPS[2]) ** -1.0, "k--", lw=0.8, label=r"$\propto n^{-1}$")
ax.set(xlabel="steps $n$ (each way)", ylabel=r"mean $\|x_0-x_0^{\rm rec}\|$",
       title=r"round trip: $x_0^{\rm rec}$ = sample(invert($x_0$))")
ax.grid(alpha=0.3, which="both")
ax.legend()
comp_cmap = plt.get_cmap("tab10")
dm.scatter2d(x0_inv, ax=axes[1], s=3, alpha=0.7, c=comp_cmap(lab_inv.numpy()), lim=LIM,
             title="data, coloured by mixture component")
dm.scatter2d(z_accurate, ax=axes[2], s=3, alpha=0.7, c=comp_cmap(lab_inv.numpy()), lim=LIM,
             title=f"their latents $x_T$ ({max(INV_STEPS)}-step inversion)")
axes[2].add_patch(plt.Circle((0, 0), 2.0, fill=False, ls="--", color="k", lw=0.8))
plt.tight_layout()
plt.show()

# %% [markdown]
# The round-trip error falls like $1/n$, as the Euler-type argument predicts. With exact scores it is tiny at a few
# hundred steps. In practice, with a network, editing pipelines use about 50 steps and accept a small reconstruction
# error. Some methods remove it with fixed-point iterations (Exercise 4) or by optimising the guidance's null embedding.
#
# On the right, the ODE turns the mixture into a standard Gaussian (the printed mean and covariance are close to $0$ and
# $I$; the dashed circle has radius 2). Each component owns a **region** of latent space whose Gaussian mass equals its
# mixture weight. This is the inverse of the bottom-left panel of §4: the heavy inner components get wedges near the
# middle and the light outer ones get the outskirts.
#
# ## 6 · Where to put the steps: uniform in $t$ vs Karras (EDM) spacing
#
# The ODE is fixed, but the grid $t_0>\dots>t_n$ is ours to choose. Karras et al. (2022) work directly in the
# $(\tilde x,\sigma)$ variables of §3.2. They use Heun's method on $d\tilde x/d\sigma=(\tilde x-D(\tilde x;\sigma))/\sigma$,
# with $D(\tilde x;\sigma)=\tilde x+\sigma^2\nabla\log p_\sigma(\tilde x)$ (Tweedie in VE form), and a polynomial spacing in
# $\sigma$:
#
# $$\sigma_i=\Big(\sigma_{\max}^{1/\rho}+\tfrac{i}{N-1}\big(\sigma_{\min}^{1/\rho}-\sigma_{\max}^{1/\rho}\big)\Big)^{\rho},\quad i=0,\dots,N-1,\qquad\sigma_N=0 .$$
#
# $\rho=1$ is uniform in $\sigma$, $\rho\to\infty$ is uniform in $\log\sigma$, and EDM's default $\rho=7$ sits in between,
# with more steps at low noise. We keep our process's own $\sigma_{\max}=\sigma(1)$ and $\sigma_{\min}=\sigma(t_{\min})$, so
# every sampler starts from the same $x_T$ and only the spacing differs. First we check that the VE denoiser built from our
# VP noise predictor equals Tweedie's formula applied to the VE-noised mixture (`gmm.noised_ve`).

# %%
def denoiser(eps_fn, xt, sig):
    """D(x~; sigma) = x~ - sigma * eps_hat: the posterior-mean denoiser in the VE variables x~ = x / sqrt(abar)."""
    return xt - sig * eps_fn(xt / math.sqrt(1 + sig ** 2), t_of_sigma(sig))


# CHECK: D from the VP eps-predictor == VE Tweedie x~ + sigma^2 grad log p_sigma(x~), with p_sigma = gmm.noised_ve(sigma)
for sig in (0.05, 0.5, 5.0):
    xt_chk = gmm.noised_ve(sig).sample(500, generator=torch.Generator().manual_seed(51)).double()
    d_vp = denoiser(eps_exact, xt_chk, sig)
    d_ve = xt_chk + sig ** 2 * gmm.noised_ve(sig).score(xt_chk)
    print(f"CHECK sigma = {sig}: max |D_VP - D_VE| = {(d_vp - d_ve).abs().max():.1e}")
    assert torch.allclose(d_vp, d_ve, atol=1e-8)


def karras_sigmas(n, rho=7.0, s_max=SIGMA_MAX, s_min=SIGMA_MIN):
    """EDM's n noise levels (descending, s_max ... s_min), followed by 0."""
    i = np.arange(n)
    s = (s_max ** (1 / rho) + i / (n - 1) * (s_min ** (1 / rho) - s_max ** (1 / rho))) ** rho
    return np.concatenate([s, [0.0]])


@torch.no_grad()
def heun_sigma_sampler(eps_fn, x, sigmas, *, return_traj=False):
    """EDM's deterministic sampler (Karras et al. 2022, Alg. 1): Heun on dx~/dsigma = (x~ - D) / sigma.

    Input/output in VP units; Euler on the final step to sigma = 0, so n steps cost 2n - 1 NFE."""
    xt = x * math.sqrt(1 + sigmas[0] ** 2)
    traj = [x]
    for s_t, s_s in zip(sigmas[:-1], sigmas[1:]):
        d = (xt - denoiser(eps_fn, xt, s_t)) / s_t
        x_pred = xt + (s_s - s_t) * d
        if s_s > 0:
            d2 = (x_pred - denoiser(eps_fn, x_pred, s_s)) / s_s
            xt = xt + (s_s - s_t) * 0.5 * (d + d2)
        else:
            xt = x_pred
        traj.append(xt / math.sqrt(1 + s_s ** 2))
    return traj if return_traj else traj[-1]


GRIDS = {"uniform in $t$": lambda n: sigmas_of_times(uniform_grid(n)),
         r"Karras $\rho=7$": lambda n: karras_sigmas(n, 7.0),
         r"uniform in $\sigma$ ($\rho=1$)": lambda n: karras_sigmas(n, 1.0)}
GRID_COL = dict(zip(GRIDS, ["tab:blue", "tab:green", "tab:brown"]))


def plain(label):
    """A LaTeX legend label as plain text, for printing."""
    return label.replace("$", "").replace("\\rho", "rho").replace("\\sigma", "sigma")


NFES_6 = [5, 10, 20, 50, 100, 250]
spacing = {}
t0 = time.time()
for gname, gfun in GRIDS.items():
    for solver in ("Euler = DDIM", "Heun"):
        rows = []
        for nfe in NFES_6:
            if solver == "Heun":
                n = (nfe + 1) // 2
                traj = heun_sigma_sampler(eps_exact, x_T, gfun(n), return_traj=True)
                used = 2 * n - 1
            else:
                traj = euler_sigma_sampler(eps_exact, x_T, gfun(nfe), return_traj=True)
                used = nfe
            rows.append(dict(nfe=used, sw=quality(traj[-1])[1], path=(traj[-2][:N_PATH] - x_ref_tmin).norm(dim=1).mean().item()))
        spacing[(gname, solver)] = rows
print(f"spacing sweep in {time.time() - t0:.0f} s")
for (gname, solver), rows in spacing.items():
    print(f"{plain(gname):24s} {solver:13s} sliced W1: " + " ".join(f"{r['sw']:.3f}" for r in rows))
print("\nsliced W1, Heun / Euler at (almost) the same NFE (> 1: Euler is better):")
for gname in GRIDS:
    pairs = zip(spacing[(gname, "Heun")], spacing[(gname, "Euler = DDIM")])
    print(f"  {plain(gname):24s} " + "  ".join(f"NFE {h['nfe']}/{e['nfe']}: {h['sw'] / e['sw']:.2f}" for h, e in pairs))

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
for (gname, solver), rows in spacing.items():
    for ax, key in zip(axes, ("sw", "path")):
        ax.loglog([r["nfe"] for r in rows], [r[key] for r in rows], "o-" if solver == "Heun" else "s--", ms=3.5,
                  color=GRID_COL[gname], label=f"{gname}, {solver}")
axes[0].axhspan(floor[:, 1].mean() - 2 * floor[:, 1].std(), floor[:, 1].mean() + 2 * floor[:, 1].std(),
                color=COL["floor"], alpha=0.6)
axes[0].set(xlabel="NFE", ylabel="sliced $W_1$ to exact samples", title="sample quality vs NFE, by time grid")
axes[1].set(xlabel="NFE", ylabel=r"pathwise error at $t_{\min}$", title="pathwise error vs NFE, by time grid")
for ax in axes:
    ax.grid(alpha=0.3, which="both")
axes[1].legend(fontsize=7)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Spacing matters a lot.** Uniform-in-$\sigma$ is a disaster at every budget we tried: it spends almost all of its steps
# at enormous noise levels where nothing happens. Between the two sensible grids, though, the outcome is less clear-cut than
# EDM's reputation suggests. On this toy, uniform-in-$t$ (for the linear VP schedule) is at least as good as Karras
# $\rho=7$ at every budget except the smallest (5 NFE): better for Euler at 10–20 NFE (by roughly a third in sliced $W_1$),
# and much better for Heun at 9–19 NFE (printed above). From about 50 NFE on, the two grids agree.
#
# **Second order pays off only with small steps.** The most visible feature of the left panel is that on both sensible
# grids EDM's Heun (solid) is *worse* than Euler = DDIM (dashed) at the same NFE, until the two tie at a few tens of NFE
# (printed above). Heun's corrector evaluates the denoiser at the predicted point $x'_s$ and divides by the *next* noise
# level $\sigma_s$. When one step spans a large ratio $\sigma_t/\sigma_s$, that predicted point is poor and the
# correction is large. Second order buys accuracy only once the steps are small. That is EDM's regime: 18 Heun steps, or
# 35 NFE, for CIFAR-10. Below that, spend the budget on more Euler steps, or use a *multistep* solver such as
# DPM-Solver++(2M), which reaches second order at 1 NFE per step by reusing the previous step's evaluation.
#
# To understand the grids, measure **where the solver makes its errors**. Starting from exact states along the reference trajectories, take one step of fixed size in $\log\sigma$ and
# compare with the exact result. Measured in VP units, this local error density shows how much each region of noise
# levels deserves steps.

# %%
N_PROF, SUB = 40, (10 if FAST else 50)
prof_sig = np.exp(np.linspace(math.log(SIGMA_MAX), math.log(SIGMA_MIN), N_PROF + 1))
x_state = x_T[:N_PATH].clone()            # exact states (VP units) at prof_sig, via fine Heun sub-steps
err_euler, err_heun = [], []
for s_t, s_s in zip(prof_sig[:-1], prof_sig[1:]):
    fine = np.exp(np.linspace(math.log(s_t), math.log(s_s), SUB + 1))
    x_next = heun_sigma_sampler(eps_exact, x_state, fine)
    one_euler = euler_sigma_sampler(eps_exact, x_state, np.array([s_t, s_s]))
    one_heun = heun_sigma_sampler(eps_exact, x_state, np.array([s_t, s_s]))
    err_euler.append((one_euler - x_next).norm(dim=1).mean().item())
    err_heun.append((one_heun - x_next).norm(dim=1).mean().item())
    x_state = x_next
err_euler, sig_mid = np.array(err_euler), np.sqrt(prof_sig[:-1] * prof_sig[1:])
hard = err_euler > 0.1 * err_euler.max()
hard_lo, hard_hi = sig_mid[hard].min(), sig_mid[hard].max()
print(f"'hard' region (local Euler error > 10% of its peak): sigma in [{hard_lo:.2f}, {hard_hi:.2f}]")
N_RUG = 15
for gname, gfun in GRIDS.items():
    s = gfun(N_RUG)[:-1]
    print(f"  {plain(gname):24s}: of its {N_RUG} noise levels {np.mean(s > hard_hi):4.0%} lie above, "
          f"{np.mean((s >= hard_lo) & (s <= hard_hi)):4.0%} inside, {np.mean(s < hard_lo):4.0%} below the hard region")

# %%
fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
ax = axes[0]
ax.loglog(sig_mid, err_euler, "s--", ms=3.5, color=COL["DDIM"], label="one Euler (= DDIM) step")
ax.loglog(sig_mid, err_heun, "o-", ms=3.5, color=COL["Heun ODE"], label="one Heun step")
ax.axvspan(hard_lo, hard_hi, color="0.9", zorder=0, label="hard region")
ax.set(ylabel="local error (VP units)", title=r"where the ODE is hard: error of one step of fixed size in $\log\sigma$")
ax.grid(alpha=0.3, which="both")
ax.legend()
ax = axes[1]
for j, (gname, gfun) in enumerate(GRIDS.items()):
    s = gfun(N_RUG)[:-1]
    ax.plot(s, np.full_like(s, j), "|", ms=14, mew=2, color=GRID_COL[gname])
ax.axvspan(hard_lo, hard_hi, color="0.9", zorder=0)
ax.set(yticks=range(len(GRIDS)), yticklabels=[g for g in GRIDS], ylim=(-0.6, len(GRIDS) - 0.4),
       xlabel=r"noise level $\sigma$ (log scale)", title=f"the {N_RUG} noise levels of each grid")
ax.set_xlim(SIGMA_MIN / 1.3, SIGMA_MAX * 1.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# The solver's errors come from a band of noise levels around the scale of the data's structure (the same window in which
# $\hat x_0$ moved in §3). Outside it, a step costs little accuracy. A good grid concentrates its steps in this band. The
# printed fractions show that both sensible grids spend a large share of their steps *above* the band, where they are
# nearly free. For our 2-D mixture, uniform-in-$t$ puts more of its steps inside the band, while $\rho=7$ puts more of
# them above and below it. That is why uniform-in-$t$ does better here, especially for Heun.
#
# The lesson is not "uniform-in-$t$ is better". **The right spacing depends on the range of scales present in the data.**
# Natural images have structure at *every* scale, from the global layout down to pixel noise (their power spectrum is a
# power law; see Dieleman's "Diffusion is spectral autoregression"). Their hard region therefore spans many decades of
# $\sigma$, which is the regime $\rho=7$ was tuned for. EDM also lowers $\sigma_{\max}$ to 80, so that fewer steps are
# wasted at the top (Exercise 3). When you move a diffusion model to a new data type, plotting this error profile is a
# cheap way to choose a schedule.
#
# > **Gotcha: start where your noise is.** The first grid point must be the noise level that $x_T$ actually has. Lin et al.
# > (2024) found two related flaws in widely used setups. First, many schedules do not reach zero terminal SNR
# > ($\bar\alpha_T$ is not tiny), so the model never sees pure noise during training, yet sampling starts from pure noise.
# > Samples then have a restricted range of brightness. Second, the popular "leading" timestep spacing does not even start
# > at the last index $k=T-1$, which makes the mismatch worse. Their fixes: rescale the schedule to zero terminal SNR (and switch to
# > $v$-prediction, since $\epsilon$-prediction is degenerate at zero SNR), always start at $k=T-1$ ("trailing" spacing),
# > and rescale classifier-free guidance. Our schedule has $\bar\alpha(1)\approx4\cdot10^{-5}$, and our grids start exactly
# > at $t=1$.
#
# ## 7 · Advanced (optional): exact log-likelihood from the probability-flow ODE
#
# Because the ODE is a smooth bijection, it defines a density through the change-of-variables formula, as a normalising
# flow does. Along a trajectory $x(t)$ of $\dot x=f(x,t)$, the continuity equation $\partial_tp+\nabla\!\cdot(fp)=0$ gives
#
# $$\frac{d}{dt}\log p_t(x(t))=\partial_t\log p_t+f\cdot\nabla\log p_t=-\nabla\!\cdot f(x(t),t)
# \quad\Longrightarrow\quad
# \boxed{\;\log p_0(x_0)=\log p_1(x_1)+\int_0^1\nabla\!\cdot f(x(t),t)\,dt\;}$$
#
# This is the **instantaneous change of variables** (Chen et al. 2018). To evaluate $\log p_0$ at a point $x_0$, integrate the
# ODE *forwards* to $t=1$ while accumulating the divergence of the velocity, then add the log-density of the (Gaussian)
# prior. In 2-D the divergence is the trace of a $2\times2$ Jacobian, which we compute exactly with autograd. In high
# dimensions one uses Hutchinson's estimator $\nabla\!\cdot f\approx v^\top(\partial f/\partial x)v$ with random $v$
# (FFJORD; Song et al. 2021). Near $t=0$ the velocity is stiff (the score of the narrow components is steep), so we space
# the steps quadratically, $t_i=(i/n)^2$, and compare with a uniform grid.

# %%
def score_exact(x, t):
    """Exact score of q_t (works at t = 0 too: q_0 is the data mixture)."""
    return gmm.noised(abar_fn(t)).score(x)


def velocity_and_divergence(x, t, score_fn=score_exact):
    """PF-ODE velocity f(x, t) and its divergence (exact trace of the 2x2 Jacobian, by autograd)."""
    with torch.enable_grad():
        x = x.detach().requires_grad_(True)
        f = -0.5 * beta_fn(t) * (x + score_fn(x, t))
        div = sum(torch.autograd.grad(f[:, i].sum(), x, retain_graph=True)[0][:, i] for i in range(x.shape[1]))
    return f.detach(), div.detach()


def ode_log_likelihood(x0, ts, score_fn=score_exact):
    """log p_{ts[0]}(x0): integrate x and the divergence forwards to t = ts[-1] = 1 with Heun's method on the grid ts."""
    x, div_int = x0.clone(), torch.zeros(x0.shape[0], dtype=x0.dtype)
    for t, s in zip(ts[:-1], ts[1:]):
        f1, d1 = velocity_and_divergence(x, t, score_fn)
        f2, d2 = velocity_and_divergence(x + (s - t) * f1, s, score_fn)
        x = x + (s - t) * 0.5 * (f1 + f2)
        div_int = div_int + (s - t) * 0.5 * (d1 + d2)
    log_prior_exact = gmm.noised(abar_fn(ts[-1])).log_prob(x).double()           # the exact q_1
    log_prior_gauss = -0.5 * (x ** 2).sum(1) - x.shape[1] / 2 * math.log(2 * math.pi)   # what you'd use in practice
    return log_prior_exact + div_int, log_prior_gauss + div_int


pts = torch.cat([gmm.sample(6, generator=torch.Generator().manual_seed(61)).double(),
                 torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)])   # 6 data points + 2 between the modes
true_lp = gmm.log_prob(pts).double()
LL_STEPS = [25, 100] if FAST else [25, 100, 400, 1600]
ll_err = {"uniform $t_i=i/n$": [], "quadratic $t_i=(i/n)^2$": []}
t0 = time.time()
for n in LL_STEPS:
    u = np.linspace(0, 1, n + 1)
    for name, ts in zip(ll_err, (u, u ** 2)):
        est, est_gauss = ode_log_likelihood(pts, ts)
        ll_err[name].append((est - true_lp).abs().max().item())
print(f"likelihood runs in {time.time() - t0:.0f} s")
print(f"{'point':>16s} {'true log p':>11s} {'ODE estimate':>13s} {'(N(0,I) prior)':>15s}")
for p, tl, e, eg in zip(pts, true_lp, est, est_gauss):
    print(f"({p[0]:6.2f},{p[1]:6.2f}) {tl:11.4f} {e:13.4f} {eg:15.4f}")
print(f"max |error| with {LL_STEPS[-1]} quadratic steps: {(est - true_lp).abs().max():.1e}")
assert (est - true_lp).abs().max() < (0.05 if FAST else 1e-3)

# %%
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for (name, errs), c in zip(ll_err.items(), ("tab:blue", "tab:red")):
    ax.loglog(LL_STEPS, errs, "o-", color=c, label=name)
ax.set(xlabel="Heun steps $n$ (2 velocity + divergence evaluations each)", ylabel="max |log-likelihood error|",
       title="PF-ODE log-likelihood vs exact log_prob")
ax.grid(alpha=0.3, which="both")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# The ODE likelihood matches `dm.GaussianMixture.log_prob` to about $10^{-5}$ nats with 1600 quadratic steps (printed
# above), including the two points in the low-density gaps between modes. Putting the steps where the ODE is stiff
# (quadratic spacing near $t=0$) is worth about an order of magnitude in accuracy at the same cost (roughly 8× at 25–100
# steps, 30× at 400–1600): the lesson of §6 again. Replacing the exact $q_1$ by
# $\mathcal N(0,I)$ changes nothing visible, since $\bar\alpha(1)$ is tiny.
#
# With a trained network the same procedure gives the exact likelihood of the **ODE model**: the distribution obtained
# by pushing $\mathcal N(0,I)$ through the network's ODE (Exercise 5). If the learned score is imperfect, this is *not*
# the distribution that DDPM or SDE sampling with the same network produces (§8 shows that the two differ), and it is
# not bounded by the ELBO either. With an exact trace it is deterministic. In high dimensions Hutchinson's estimator makes
# it random (but unbiased). Song et al. (2021) report bits-per-dimension this way, with dequantised data. DDPM (Ho et al.
# 2020) and VDM (Kingma et al. 2021) report the variational bound of notebook 02 instead.
#
# ## 8 · With a trained network: model error sets a floor
#
# So far the vector field was perfect. Now we train an $\epsilon$-network exactly as in notebook 03 (`dm.MLPDenoiser`,
# `dm.train_diffusion`, linear schedule, $T=1000$) and repeat a reduced NFE sweep. Using exact scores, we can split the
# error into two parts: **discretisation error** (the exact-score curves of §2) and **model error** (the extra error of
# the trained network).
#
# > **Gotcha: feed the network the noise level it was trained on.** The network was trained on the *discrete*
# > $\bar\alpha_k$, while our samplers use the continuous $\bar\alpha(t)$ at arbitrary times (the two differ slightly, §1.1).
# > The correct conditioning input is the (fractional) index $k^*$ whose training noise level equals the sampler's,
# > $\bar\alpha_{k^*}=\bar\alpha(t)$, not the clock time $tT-1$. The MLP's sinusoidal time embedding accepts fractional
# > inputs and interpolates smoothly. Libraries such as `k-diffusion` and `diffusers`' Karras schedulers convert $\sigma$
# > to a timestep in exactly this way. Here the correction is tiny (printed below), but it is essential whenever the
# > sampler's grid has no natural "clock" that matches the network's, for example a Karras $\sigma$ grid, or a network
# > trained with a different $T$ or schedule.

# %%
dm.seed_everything(0)
train_data = gmm.sample(20_000, generator=torch.Generator().manual_seed(71))
sched_dev = dm.NoiseSchedule(T, "linear", device=device)
net = dm.MLPDenoiser(data_dim=2, hidden=128)
out = dm.train_diffusion(net, train_data, sched_dev, steps=TRAIN_STEPS, batch_size=256, lr=1e-3, device=device,
                         progress=False, log_every=max(TRAIN_STEPS // 4, 1))
ema_net = out["ema"].model.eval()                   # sample with the EMA weights (notebook 02)
print(f"{dm.count_params(net) / 1e3:.0f}k parameters, trained in {out['seconds']:.0f} s")

NEG_LOG_AB = -np.log(AB_DISC)                       # increasing in k


def noise_matched_index(t):
    """Fractional network index k* whose training noise level alpha_bar_{k*} equals abar_fn(t)."""
    return float(np.interp(-math.log(abar_fn(t)), NEG_LOG_AB, np.arange(T)))


@torch.no_grad()
def net_eps(x, t, index_fn=noise_matched_index):
    """The trained network as an eps_fn(x, t): float64 CPU state in, float64 CPU prediction out.

    The network runs in float32 on `device`; we move its output to the CPU *before* casting to float64,
    because Apple's MPS backend has no float64."""
    k = torch.full((x.shape[0],), index_fn(t), device=device)
    return ema_net(x.float().to(device), k).cpu().double()


ks = np.array([0, 10, 250, 500, 999])
print("clock-time index k  ->  noise-matched index k* at t = (k+1)/T:  "
      + ", ".join(f"{k} -> {noise_matched_index((k + 1) / T):.1f}" for k in ks))

# How good is the network? Relative error of its eps-prediction on exact samples of q_t, with the noise-matched index
# and with the naive clock index t*T - 1. Notebook 02 §5: denoising is hardest at low noise.
print("relative eps error  ||eps_net - eps*|| / ||eps*||  on samples of q_t:")
for t_chk in (0.9, 0.5, 0.2, 0.05, 0.01):
    xq = gmm.noised(abar_fn(t_chk)).sample(5000, generator=torch.Generator().manual_seed(72)).double()
    e_star = eps_exact(xq, t_chk)
    rel = [((net_eps(xq, t_chk, fn) - e_star).norm() / e_star.norm()).item()
           for fn in (noise_matched_index, lambda t: t * T - 1)]
    print(f"  t = {t_chk:4}:  noise-matched {rel[0]:.3f}   clock index {rel[1]:.3f}")

# %%
NFES_8 = [5, 10, 20, 50, 100, 250]
NAMES_8 = ["DDPM ancestral", "Heun ODE", "DDIM"]
t0 = time.time()
sweep_net = nfe_sweep(net_eps, x_T, NAMES_8, NFES_8)
# the exact-score reference at the same budgets, matched by NFE (not by position) so that NFES_8 can be edited freely
missing = sorted({nfe for nfe in NFES_8 if nfe not in NFES})
extra = nfe_sweep(eps_exact, x_T, NAMES_8, missing) if missing else {name: [] for name in NAMES_8}
sweep_ref = {}
for name in NAMES_8:
    by_nominal = dict(zip(NFES, sweep[name])) | dict(zip(missing, extra[name]))
    sweep_ref[name] = [by_nominal[nfe] for nfe in NFES_8]
print(f"trained-network sweep in {time.time() - t0:.0f} s")
for key, fmt in (("sw", "{:.3f}"), ("stranded", "{:.2%}")):
    print(f"{key}:")
    for name in NAMES_8:
        print(f"  {name:16s} trained net: " + " ".join(fmt.format(r[key]) for r in sweep_net[name])
              + "   (exact score: " + " ".join(fmt.format(r[key]) for r in sweep_ref[name]) + ")")

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
panels = [(0, "mmd", "MMD$^2$"), (1, "sw", "sliced $W_1$"), (None, "stranded", "% stranded (log $p<-10$)")]
for ax, (j, key, label) in zip(axes, panels):
    scale = 100.0 if key == "stranded" else 1.0
    for name in NAMES_8:
        ax.plot([r["nfe"] for r in sweep_net[name]], [scale * r[key] for r in sweep_net[name]], "o-", ms=4,
                color=COL[name], label=f"{name}, trained network")
        ax.plot([r["nfe"] for r in sweep_ref[name]], [scale * r[key] for r in sweep_ref[name]], "o:",
                ms=3, color=COL[name], alpha=0.6, label=f"{name}, exact score")
    if j is not None:
        ax.set_yscale("log")
        ax.axhspan(max(floor[:, j].mean() - 2 * floor[:, j].std(), 1e-6), FLOOR_HI[j], color=COL["floor"], alpha=0.6,
                   label="floor")
        ax.set(ylabel=f"{label} to exact samples", title=f"trained network vs exact score: {label}")
    else:   # a percentage that can be exactly 0: linear near 0, logarithmic above 0.1%
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_ylim(0, 100)
        ax.set(ylabel="% of samples where log $p(x)<-10$", title="samples stranded in low-density regions")
    ax.set_xscale("log")
    ax.set_xlabel("NFE")
    ax.grid(alpha=0.3, which="both")
axes[0].legend(fontsize=7)
plt.tight_layout()
plt.show()

# %% [markdown]
# At small NFE discretisation error dominates. The trained-network curves (solid) mostly track the exact-score curves
# (dotted), and the choice of sampler matters as much as before. Heun at the smallest budgets is the exception: its
# corrector evaluates the network at a poor predicted point, where the network has seen little training data. As the
# budget grows, the exact-score curves fall to the floor, but in sliced $W_1$ the ODE samplers driven by the network
# **level off** above it. More steps solve the network's ODE more accurately, but that ODE belongs to the wrong vector
# field. The remaining gap is **model error**, and no ODE solver can remove it.
#
# **Where is the model error?** The MMD panel hardly separates the samplers at large budgets: at this sample size the
# model-error floor lies inside MMD's finite-sample band. The right panel shows why. The extra error is concentrated in a
# small fraction of samples that end up **stranded** in regions the data never visits. The network is least accurate in the
# gaps between modes, where it saw almost no training data (notebook 03, §3b and §7; see also how its error grows at low
# noise in the printout above). The ODE transports the latents that lead into those regions faithfully. MMD's bounded
# kernels barely notice a percent of misplaced samples, while a transport distance like $W_1$ does. Look at more than one
# metric, and look at the tails.

# %% [markdown]
# Here are those samples at the largest budget, from the same $x_T$ (DDPM: the sweep's first seed). With the exact score
# no DDIM sample is stranded. With the trained network some twenty are, mostly in the gaps between neighbouring modes,
# with a few in the empty centre and far out in the tails; DDPM strands fewer (the counts in the titles depend on the
# trained network).

# %%
viz_logp0 = gmm.log_prob(viz_xy).double().reshape(len(viz_grid), len(viz_grid)).numpy()
viz_n = NFES_8[-1]                                  # the largest budget of the sweep
viz_runs = [("DDIM, exact score", ddim_sampler(eps_exact, x_T, uniform_grid(viz_n)), COL["DDIM"]),
            ("DDIM, trained network", ddim_sampler(net_eps, x_T, uniform_grid(viz_n)), COL["DDIM"]),
            ("DDPM ancestral, trained network",      # = the sweep's first seed
             ddpm_sampler(net_eps, x_T, uniform_grid(viz_n), gen=torch.Generator().manual_seed(0)),
             COL["DDPM ancestral"])]
viz_fig, viz_axes = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained")
for viz_ax, (viz_title, viz_x, viz_col) in zip(viz_axes, viz_runs):
    viz_lost = gmm.log_prob(viz_x) < -10.0            # stranded, as in stranded_fraction
    viz_ax.contourf(viz_grid, viz_grid, viz_p0, levels=np.linspace(0, 1, 11), cmap=CMAP_DENSITY)
    viz_ax.contour(viz_grid, viz_grid, viz_logp0, levels=[-10.0], colors="k", linewidths=0.6, linestyles="--")
    viz_ax.scatter(viz_x[~viz_lost, 0], viz_x[~viz_lost, 1], s=1.5, marker="s", color=viz_col, linewidths=0,
                   antialiased=False)
    viz_ax.scatter(viz_x[viz_lost, 0], viz_x[viz_lost, 1], s=20, marker="x", color="k", linewidths=1.0,
                   label="stranded sample")
    viz_ax.set(xlim=(-LIM, LIM), ylim=(-LIM, LIM), aspect="equal", xticks=[], yticks=[],
               title=f"{viz_title}, NFE {viz_n}\n{int(viz_lost.sum())} of {len(viz_x)} samples stranded")
viz_axes[1].legend(loc="lower left", fontsize=7)
viz_fig.suptitle("Where the model error hides: a few samples stranded outside the dashed contour ($\\log p=-10$), "
                 "mostly in the gaps between modes")
plt.show()

# %% [markdown]
# **DDPM behaves differently.** At large budgets it typically strands fewer samples, and its $W_1$ comes down into the
# floor band, close to its exact-score counterpart, instead of levelling off above it (see the printout; the exact
# numbers depend on the trained network). This is §4.1 at work. The Langevin part
# of the reverse SDE pulls stranded samples back towards high-density regions. These are regions of the *model's*
# $\hat p_t$, not of the true $p_t$, but the model is accurate exactly where most of the data is. There is a clean theory
# behind this. For the SDE sampler,
# $\mathrm{KL}(p_{\rm data}\,\|\,p^{\rm SDE}_\theta)\le\mathrm{KL}(q_1\,\|\,\mathcal N(0,I))+\tfrac12\int_0^1\beta(t)\,\mathbb E_{q_t}\|\nabla\log q_t-s_\theta\|^2dt$
# (Song, Durkan, Murray & Ermon 2021; with discretisation, Chen et al. 2023). The error is controlled by the same $L^2$
# score error that training minimises. No such bound holds for the ODE, whose error also depends on errors in the
# *derivatives* of the learned score.
#
# **In practice:** at small budgets deterministic samplers win. At large budgets with an imperfect model, some
# stochasticity pays (Exercise 1), especially in the tails. For ODE samplers, **you only need enough steps to get below
# the model-error floor**; beyond that, extra NFE is wasted. This is why well-trained image models are sampled with
# 20–50 DDIM-like steps, not 1000.
#
# ## 9 · Practical guidance
#
# | goal | typical choice | NFE |
# |---|---|---|
# | reference quality, no time pressure | DDPM ancestral / Euler–Maruyama, or Heun with a little churn | 250–1000 |
# | the robust default | DDIM ($\eta=0$) / Euler in $\sigma$, uniform-in-$t$ (trailing) or Karras spacing | 25–50 |
# | best quality per NFE | multistep exponential-integrator solvers: DPM-Solver++(2M), UniPC (2nd order at 1 NFE per step) | 10–20 |
# | high quality, moderate budget | EDM's Heun (2nd order, 2 NFE per step; only pays off once the steps are small, §6) | ≳35 |
# | real-time | distilled or consistency models (progressive distillation, LCM, consistency models, adversarial distillation such as SDXL-Turbo) | 1–4 |
# | editing / inversion / interpolation | deterministic ($\eta=0$) sampler and its inversion, so that noise ↔ data is a fixed map | 25–100 |
#
# * **Count NFE, not steps or seconds per step.** The network evaluation dominates the cost. Heun costs 2 NFE per step.
#   **Classifier-free guidance costs 2 NFE per step** (conditional + unconditional; notebook 06), so 50 guided DDIM steps
#   cost 100 network passes (usually run as one doubled batch).
# * **$\eta$ (stochasticity):** $\eta=0$ for few steps, for reproducibility, and for anything that needs inversion. Some
#   noise ($\eta>0$, or EDM's churn) helps at large budgets when the model is imperfect (§4.1), but it needs more steps to
#   pay off.
# * **The last step** should return the denoised estimate $\hat x_0$ without adding noise (§3.2: a plain Euler step to
#   $t=0$ leaves residual noise). Never evaluate the network at $t=0$: it was never trained there.
# * **$\hat x_0$ clipping** (to $[-1,1]$ for images, `clip_x0` in `dmtools`) and dynamic thresholding stop guided samplers
#   from saturating. They change the update, though, so the sampler no longer solves the ODE and exact inversion breaks.
# * **Precision:** keep schedule coefficients in float64. $1/\sqrt{\bar\alpha}$ reaches about 150 at $t=1$ here, and
#   $1-\bar\alpha$ near $t=0$ loses digits in float32. bf16 is fine for the network itself.
# * **Sample with EMA weights**, and pick the step count by checking where quality stops improving (§8). That plateau is
#   the model's floor. Judge it with more than one metric, and look at the tails.
#
# ## Exercises
#
# **Exercise 1 (the $\eta$ knob).** Run `ddim_sampler` with $\eta\in\{0,0.25,0.5,1\}$ at 20 and at 250 steps (exact score,
# same `x_T`) and print `quality(...)`. Predict first: which $\eta$ wins at 20 steps, and which at 250? Then repeat with
# `net_eps` instead of `eps_exact`.
#
# <details><summary>Hint / solution</summary>
#
# ```python
# for n in (20, 250):
#     for eta in (0, 0.25, 0.5, 1):
#         x = ddim_sampler(eps_exact, x_T, uniform_grid(n), eta=eta, gen=torch.Generator().manual_seed(0))
#         print(n, eta, quality(x))
# ```
# With 10–50 steps the DDPM-like $\eta=1$ is clearly the worst (as in §2). Small values ($\eta\lesssim0.5$) cost little and
# are roughly as good as $\eta=0$. The differences between them are comparable to the seed-to-seed noise, so average over
# a few seeds before concluding anything. At 250 steps every $\eta$ reaches the floor with exact scores. With the trained
# network, compare with §8: at large budgets the injected noise can also correct model error (§4.1).
# </details>
#
# **Exercise 2 (make the toy image-like).** Images have many directions of tiny variance (fine detail). Embed the mixture
# in 12-D with 10 extra coordinates of variance $10^{-6}$: `means = F.pad(gmm.means, (0, 10))`, and block-diagonal
# covariances with `gmm.covs` in the top-left corner and `1e-6 * I` elsewhere. Build
# `dm.GaussianMixture(gmm.weights, means, covs)` and its `exact_eps_fn`. Predict how Euler-in-$t$ (with and without the
# denoising last step), Heun-in-$t$ and DDIM behave in the extra coordinates, then measure the std of the generated
# samples there (true value 0.001) at 20 and 50 NFE.
#
# <details><summary>Hint / solution</summary>
#
# ```python
# import torch.nn.functional as F
# d = 12
# covs = torch.zeros(gmm.K, d, d, dtype=torch.float64); covs[:, :2, :2] = gmm.covs
# covs[:, 2:, 2:] = 1e-6 * torch.eye(d - 2, dtype=torch.float64)
# g12 = dm.GaussianMixture(gmm.weights, F.pad(gmm.means, (0, d - 2)), covs)
# xT12 = torch.randn(2000, d, dtype=torch.float64)
# for nfe in (20, 50):
#     for name in ["Euler ODE", "Heun ODE", "DDIM"]:
#         n, _ = steps_for_nfe(name, nfe)
#         x = SAMPLERS[name][0](exact_eps_fn(g12), xT12, uniform_grid(n))
#         print(nfe, name, x[:, 2:].std().item())   # true: 0.001
#     x = euler_ode_sampler(exact_eps_fn(g12), xT12, uniform_grid(nfe), denoise_last=False)
#     print(nfe, "Euler ODE, plain last step", x[:, 2:].std().item())
# ```
# This is §3.2 in action, and three separate effects show up.
# (i) **Heun-in-$t$ is unstable** in the stiff directions. At 20 NFE its samples carry noise several times the true std
# (grain); at 50 NFE it has calmed down.
# (ii) **Euler-in-$t$ with a plain last step** leaves residual noise of order $\sigma_{\min}\approx0.01$, ten times the
# true std, in every extra direction. With the denoising last step this residual noise disappears.
# (iii) **DDIM (and the denoised Euler) return almost no spread there.** The last step returns
# $\mathbb E[x_0\mid x_{t_{\min}}]$, which shrinks directions whose std is below $\sigma_{\min}$ by
# $\sqrt{\lambda/(\lambda+\sigma_{\min}^2)}\approx0.1$. More steps do not help, because this is the $t_{\min}$ truncation,
# not discretisation error, and DDIM sits exactly at that limit. The relative error is large, but the absolute error is
# below 0.001 and it adds no grain.
# In the two "interesting" coordinates all of them do about as well as before. Real images have thousands of such
# directions. Noise of order $\sigma_{\min}$ in each of them adds up to visible grain, while smoothing below $\sigma_{\min}$
# is invisible.
# </details>
#
# **Exercise 3 (design a schedule).** The error profile in §6 says that the steps at very large $\sigma$ are nearly wasted.
# (a) Start EDM's Heun sampler at a smaller $\sigma_{\max}$ (EDM uses 80; try 80 and 20):
# `heun_sigma_sampler(eps_exact, x_T, karras_sigmas(n, 7.0, s_max=20.0))`. The same unit-variance `x_T` works, because
# the sampler takes its input in VP units and VP noise always has unit variance. Predict, then measure, the change in
# sample quality at 10–20 NFE, and compare with `euler_sigma_sampler` at the same NFE. How much error comes from treating
# $\sigma=20$ as "pure noise"? (b) Go further and build the grid from the measured error profile (`err_heun`,
# `err_euler`, `prof_sig`). How should the step density depend on the local error?
#
# <details><summary>Hint / solution</summary>
#
# (a) At $\sigma=20$ the data (per-coordinate std about 1) changes the noisy distribution's std by only about 0.1%, so
# treating it as pure noise costs almost nothing. Every step saved at the top of the schedule goes into the hard region,
# so Heun's quality at 10–20 NFE improves noticeably. Euler at the same NFE is still at least as good: below a few tens
# of NFE the extra evaluation is better spent on another Euler step (§6).
#
# (b) **Equidistribute the local error.** A solver of order $p$ makes a local error of about $C(u)\,h^{p+1}$ per step of
# size $h$ in $u=\log\sigma$. With $1/h(u)$ steps per unit of $u$, the total error is about $\int C\,h^{p}\,du$.
# Minimising it at a fixed number of steps $\int du/h=N$ (a Lagrange multiplier) gives $h\propto C^{-1/(p+1)}$: every step
# makes the same local error. The profile was measured with one fixed step, so $C\propto$ `err`, and the step density must
# be $\propto$ `err`$^{1/3}$ for Heun ($\propto$ `err`$^{1/2}$ for Euler). A density $\propto$ `err` would pile almost all
# steps into the hard band and leave giant steps at both ends.
#
# ```python
# def equidistributed_sigmas(n, err, p):
#     """n noise levels (then 0) with step density proportional to err**(1/(p+1)) in log sigma."""
#     cdf = np.concatenate([[0.0], np.cumsum(np.asarray(err) ** (1.0 / (p + 1)))])
#     log_s = np.interp(np.linspace(0, 1, n), cdf / cdf[-1], np.log(prof_sig))
#     return np.concatenate([np.exp(log_s), [0.0]])
#
# for nfe in (9, 19):
#     n = (nfe + 1) // 2
#     print(nfe, quality(heun_sigma_sampler(eps_exact, x_T, equidistributed_sigmas(n, err_heun, 2)))[1],
#           quality(euler_sigma_sampler(eps_exact, x_T, equidistributed_sigmas(nfe, err_euler, 1)))[1])
# ```
# Both beat Karras $\rho=7$ at the same NFE, and the Euler version also beats uniform-in-$t$, making it the best
# $\sigma$-space (DDIM-type) sampler in this notebook at these budgets. (Plain Euler in $(x,t)$ from §2 is still a little
# better on this toy, for the reasons of §3.2.) This is a hand-made adaptive schedule.
# </details>
#
# **Exercise 4 (better inversion).** Improve `ddim_invert` with a fixed-point iteration. In each step, first compute
# $x_s$ as now, then recompute $\hat\epsilon$ at $(x_s,s)$ and redo the step with it. Repeat 2–3 times. Plot the round-trip
# error against $n$ as in §5. Predict: what happens to the $O(h)$ error?
#
# <details><summary>Hint / solution</summary>
#
# Inside the loop: `x_s = step(x, eps_fn(x, t))`, then `for _ in range(3): x_s = step(x, eps_fn(x_s, s))`. The update now
# uses (an approximation of) the slope at the noisier end, which is what the sampler uses on the way back. Each
# iteration shrinks the mismatch by a factor of roughly (Lipschitz constant of $\hat\epsilon$) $\times$ (step size), which
# gets smaller as the steps shrink. Expect a few times lower error at $n=5$ and well over an order of magnitude at
# $n=100$. Each iteration costs 1 NFE, and for very large steps the iteration need not converge.
#
# The conceptual answer: at convergence the iteration solves $\tilde x_s=\tilde x_t+(\sigma_s-\sigma_t)\,\hat\epsilon(x_s,s)$,
# which is *exactly* the inverse of the sampler's DDIM step from $s$ to $t$. So the round trip becomes exact, up to the
# fixed-point tolerance. The latent it finds, however, is the exact preimage under the *discrete* DDIM map, which is still
# only an $O(h)$ approximation of the true PF-ODE latent. Compare it with `z_accurate` from §5. Round-trip exactness
# and a correct latent are different things. Editing needs the first; anything that relies on the continuous ODE, such
# as the likelihood of §7, needs the second.
# </details>
#
# **Exercise 5 (the model's likelihood).** Pass the trained network's score
# $-\hat\epsilon_\theta(x,k^*)/\sqrt{1-\bar\alpha(t)}$ as `score_fn` to `ode_log_likelihood` in §7. The divergence needs
# gradients through the network, so do not use `net_eps`, which runs under `torch.no_grad`. Start the grid at $t_{\min}$
# rather than 0, since the network was never trained at $t=0$. Compare the model's log-likelihood with the truth,
# `gmm.noised(abar_fn(T_MIN)).log_prob(pts)`, on the same 8 points. Where is the model most wrong?
#
# <details><summary>Hint / solution</summary>
#
# ```python
# def score_net(x, t):
#     k = torch.full((x.shape[0],), noise_matched_index(t), device=device)
#     return -ema_net(x.float().to(device), k).cpu().double() / math.sqrt(1 - abar_fn(t))   # .cpu() first: MPS has no float64
# u = np.linspace(0, 1, 401)
# est, _ = ode_log_likelihood(pts, T_MIN + (1 - T_MIN) * u ** 2, score_fn=score_net)
# print(est - gmm.noised(abar_fn(T_MIN)).log_prob(pts).double())
# ```
# On the data points the model's log-likelihood is typically within a few tenths of a nat of the truth. The two points
# in the gaps between the modes are far worse. The network had almost no training signal there (notebook 03, §7), and
# because the true density there is tiny, even a small error in the score translates into a large error in
# $\log p$. Starting at $t_{\min}$ gives the likelihood of very slightly noised data, $q_{t_{\min}}$, which is standard.
# </details>
#
# ## Key takeaways
#
# * Sampling a diffusion model means **numerically solving a differential equation**. The DDPM chain is Euler–Maruyama
#   for the reverse-time SDE $dx=[-\tfrac12\beta x-\beta\nabla\log p_t]dt+\sqrt\beta\,d\bar W$. The **probability-flow ODE**
#   $\dot x=-\tfrac12\beta(x+\nabla\log p_t)$ has the same marginals, by Fokker–Planck.
# * The **reverse SDE = ODE + Langevin corrector**. Its noise erases errors made at high noise levels, but it cannot move
#   mass between modes once they have separated, and it needs many steps. Deterministic solvers are clearly better per NFE
#   from about 10–20 NFE up to the point where everything converges.
# * **DDIM** follows from a non-Markovian forward process with DDPM's marginals, so it needs no retraining. Each step is
#   exact if fed a posterior *sample* of $x_0$; using the posterior *mean* is its (and DDPM's) discretisation error. With
#   $\eta=0$ it is **Euler's method in $(\tilde x=x/\sqrt{\bar\alpha},\ \sigma)$**: a convex step towards $\hat x_0$. It
#   is exact when $\hat x_0$ is constant, it never overshoots, and its last step is automatically the denoiser. Its
#   failure mode at low NFE is over-smoothing, never noise or blow-up, which makes it the robust default.
# * **Finish with a denoising step**: a plain Euler step to $t=0$ in $(x,t)$ leaves residual noise.
# * Orders of convergence are real (Euler/DDIM $\propto$ NFE$^{-1}$, Heun $\propto$ NFE$^{-2}$) but only once steps are
#   small. Below a few tens of NFE, a 2-NFE Heun step usually loses to two Euler steps, and *where* the steps go and *which
#   variables* you integrate in matter more.
# * The ODE is a **bijection** between noise and data, a continuous normalising flow. It gives DDIM **inversion** (an
#   $O(h)$ approximation) and **exact likelihoods** via $\log p_0=\log p_1+\int\nabla\!\cdot f\,dt$.
# * **Step spacing** should concentrate on the noise levels where the denoiser's output changes. That depends on the
#   data's range of scales: Karras $\rho=7$ suits images, but on our toy uniform-in-$t$ does at least as well from 10 NFE up. Uniform-in-$\sigma$ is
#   always bad. Equidistributing the measured local error (density $\propto$ error$^{1/(p+1)}$) beats both with the Euler
#   (= DDIM) solver (Exercise 3).
# * With a trained network, **model error sets a floor** for deterministic samplers. Use just enough NFE to reach it, and
#   count NFE, including CFG's 2×. The floor hides in a few samples stranded where the network is poorly trained; $W_1$
#   sees them, MMD barely does. At large budgets, a stochastic sampler can partially correct it.
#
# ## Further reading
#
# * Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2021). *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR. arXiv:2011.13456. The SDE/ODE view, predictor–corrector samplers, and likelihoods.
# * Anderson (1982). *Reverse-time diffusion equation models.* Stochastic Processes and their Applications 12(3).
# * Song, Meng, Ermon (2021). *Denoising Diffusion Implicit Models.* ICLR. arXiv:2010.02502.
# * Karras, Aittala, Aila, Laine (2022). *Elucidating the Design Space of Diffusion-Based Generative Models* (EDM). NeurIPS. arXiv:2206.00364. The $(\tilde x,\sigma)$ view, Heun, $\rho$-spacing and churn.
# * Lu, Zhou, Bao, Chen, Li, Zhu (2022). *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps.* NeurIPS. arXiv:2206.00927; and *DPM-Solver++* (2022), arXiv:2211.01095.
# * Nichol & Dhariwal (2021). *Improved Denoising Diffusion Probabilistic Models.* ICML. arXiv:2102.09672. Respacing the timesteps of a trained model.
# * Lin, Liu, Li, Yang (2024). *Common Diffusion Noise Schedules and Sample Steps are Flawed.* WACV. arXiv:2305.08891. Zero terminal SNR and the "trailing" timestep gotcha.
# * Song, Durkan, Murray, Ermon (2021). *Maximum Likelihood Training of Score-Based Diffusion Models.* NeurIPS. arXiv:2101.09258; Chen, Chewi, Li, Li, Salim, Zhang (2023). *Sampling is as easy as learning the score.* ICLR. arXiv:2209.11215. Why the SDE sampler's error is controlled by the $L^2$ score error.
# * Ho, Jain, Abbeel (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS. arXiv:2006.11239; Kingma, Salimans, Poole, Ho (2021). *Variational Diffusion Models.* NeurIPS. arXiv:2107.00630. Likelihoods from the variational bound, for comparison with §7.
# * Chen, Rubanova, Bettencourt, Duvenaud (2018). *Neural Ordinary Differential Equations.* NeurIPS. arXiv:1806.07366; Grathwohl et al. (2019). *FFJORD.* ICLR. arXiv:1810.01367. The instantaneous change of variables and Hutchinson's trace estimator.
# * Salimans & Ho (2022). *Progressive Distillation for Fast Sampling of Diffusion Models.* ICLR. arXiv:2202.00512; Song, Dhariwal, Chen, Sutskever (2023). *Consistency Models.* ICML. arXiv:2303.01469; Sauer, Lorenz, Blattmann, Rombach (2023). *Adversarial Diffusion Distillation.* arXiv:2311.17042. 1–4-step samplers.
# * Sander Dieleman, *Diffusion is spectral autoregression* (sander.ai, 2024). Why images have structure at every noise level.
# * Yang Song, *Generative Modeling by Estimating Gradients of the Data Distribution* (blog post, 2021). An accessible tour of the SDE view.
