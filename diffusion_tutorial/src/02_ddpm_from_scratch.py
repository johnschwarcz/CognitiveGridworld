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
# # 02 · DDPM from scratch: learning to reverse the noise (2-D)
#
# Notebook 01 built the *forward* process, which destroys data by adding Gaussian noise one small step at a time.
# Here we learn to run it **backwards**. We derive the DDPM training objective from the variational bound,
# implement every piece (time embedding, denoiser, training loop, ancestral sampler) in a few dozen lines, and
# take it apart with ablations. Everything is in 2-D and on a Gaussian-mixture "lab" whose noised densities we
# know exactly. That lets us separate *what the network gets wrong* from *what the sampler gets wrong*.
#
# **You will learn:**
# - why the reverse step $p_\theta(x_{t-1}\mid x_t)$ can be Gaussian, and when that assumption breaks
# - how the ELBO turns into "predict the noise": the loss $L_\text{simple}$, and what it drops
# - the sinusoidal time embedding, a residual-MLP denoiser and a 20-line training loop
# - why the loss plateaus far above zero (the Bayes-optimal floor), and how it depends on $t$
# - ancestral sampling, what the model "believes" along the way ($\hat x_0$ as a posterior mean), and how to score samples (MMD, sliced Wasserstein)
# - practical knobs: EMA weights, $\epsilon$- vs $x_0$- vs $v$-prediction, the number of steps $T$, the sampler variance,
#   and Fourier input features for data with sharp edges
#
# **Prerequisites:** 01 · The forward process (noise schedules, $q(x_t\mid x_0)$, the posterior $q(x_{t-1}\mid x_t, x_0)$).
#
# **Runtime:** about 7–10 min on a laptop CPU (ten small MLPs, 20–35 s each, plus sampling); roughly 2–4 min on a GPU.

# %%
import copy
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dmtools as dm

device = dm.get_device(verbose=True)
FAST = dm.fast_mode()  # automated smoke-test switch, leave as is
dm.seed_everything(0)
if device.type == "cpu":
    # Our 2-D MLPs are tiny: on a CPU the per-op threading overhead outweighs any parallel speed-up
    # (and is disastrous on a busy machine), so one thread is as fast or faster.
    torch.set_num_threads(1)

plt.rcParams.update({"figure.dpi": 100, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
C = {"data": "0.45", "eps": "tab:blue", "x0": "tab:orange", "v": "tab:green", "oracle": "tab:red",
     "raw": "tab:purple", "gauss": "tab:brown"}

# Every expensive quantity is scaled by FAST (FAST only checks that the code runs).
STEPS = 200 if FAST else 4000        # training steps for every 2-D model
N_SAMPLE = 256 if FAST else 2000     # points per generated sample set (plots and metrics)
T = 1000                             # number of diffusion steps for the main model
print(f"FAST={FAST}: {STEPS} training steps per model, {N_SAMPLE} samples per evaluation")

# %% [markdown]
# ## 1 · The reverse process
#
# **Recap of notebook 01.** The forward process adds a little noise at each step,
# $$q(x_t\mid x_{t-1}) = \mathcal N\big(\sqrt{\alpha_t}\,x_{t-1},\ \beta_t I\big),\qquad \alpha_t = 1-\beta_t,$$
# and composes to a closed form, $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ with
# $\bar\alpha_t=\prod_{s\le t}\alpha_s$ and $\epsilon\sim\mathcal N(0,I)$. If we also condition on the clean point $x_0$, the
# step backwards is a Gaussian we can write down:
# $$q(x_{t-1}\mid x_t, x_0) = \mathcal N\big(\tilde\mu_t(x_t,x_0),\ \tilde\beta_t I\big),\quad
# \tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t,\quad
# \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t .$$
#
# **The generative model.** At sampling time we do not know $x_0$. DDPM defines a Markov chain that runs backwards from pure noise:
# $$p(x_T)=\mathcal N(0,I),\qquad p_\theta(x_{t-1}\mid x_t) = \mathcal N\big(\mu_\theta(x_t,t),\ \sigma_t^2 I\big).$$
# A neural network supplies the mean. The variance $\sigma_t^2$ is fixed (we compare $\beta_t$ and $\tilde\beta_t$ in §9c).
#
# **Why is a Gaussian a reasonable model for the *true* reverse step?** Notebook 01 (§8) answered this in 1-D. By Bayes,
# $$q(x_{t-1}\mid x_t) \;\propto\; \underbrace{q(x_t\mid x_{t-1})}_{\text{Gaussian, width }\sqrt{\beta_t}}\;\underbrace{q_{t-1}(x_{t-1})}_{\text{marginal (prior)}} .$$
# For a small step the likelihood is a narrow window over which $\log q_{t-1}$ is almost linear, so the product is nearly
# Gaussian. For a large step the window is wide, the posterior inherits the modes of $q_{t-1}$, and no Gaussian can
# represent it. One more fact matters for what follows. The **mean** of the true reverse step is always available.
# Applying Tweedie's formula (notebook 03) to the single step $x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{\beta_t}\,\epsilon$ gives,
# exactly and for any step size,
# $$\mathbb E[x_{t-1}\mid x_t] = \frac{1}{\sqrt{\alpha_t}}\Big(x_t + \beta_t\,\nabla_{x_t}\log q_t(x_t)\Big).$$
# §4 shows that DDPM's reverse mean, fed the Bayes-optimal noise prediction, is exactly this expression. So, given a
# perfect network, the only approximation in DDPM's reverse step is its *shape*: a Gaussian with a fixed
# variance. Below is the 2-D version of notebook 01's experiment. It uses the data of this whole notebook,
# `dm.default_gmm()` (an 8-component mixture whose noised marginals $q_t$ are again mixtures with known densities), and the
# three step sizes of our ablation in §9b.
#
# > **Gotcha: changing $T$.** To compare step sizes we need the *same* noising process cut into $T$ steps. The usual trick of
# > scaling the linear schedule ($\beta_t \to \beta_t\cdot 1000/T$) is only a first-order approximation. It gives $\beta_t>1$
# > (an invalid schedule, with NaNs) once $T<20$. `make_schedule(T)` below instead takes the $T{=}1000$ process and observes it at $T$
# > evenly spaced times. This keeps $\bar\alpha$ correct at every $T$.
# >
# > **Gotcha: off-by-one.** Maths (Ho et al.) numbers the latents $x_1,\dots,x_T$, with $x_0$ the data. Python arrays start at 0,
# > so in code (here and in `dmtools`) index `t` $\in\{0,\dots,T-1\}$ holds the coefficients of latent $x_{t+1}$:
# > `alpha_bars[t]` $=\bar\alpha_{t+1}$, and `alpha_bars_prev[0]` $=1$ belongs to the data. The network is fed the integer
# > index `t`. What matters is using the same convention in training and sampling.

# %%
gmm = dm.default_gmm()   # the ground-truth data distribution for most of this notebook


def make_schedule(T_steps, device=device):
    """The linear (T=1000) VP process observed at T_steps evenly spaced times, as a dm.NoiseSchedule.

    Index t covers continuous time (t+1)/T_steps. For T_steps=1000 this reproduces dm.NoiseSchedule(1000).
    """
    abar_1000 = torch.cumprod(1 - torch.linspace(1e-4, 0.02, 1000, dtype=torch.float64), 0)
    abar = abar_1000[torch.arange(1, T_steps + 1) * 1000 // T_steps - 1]
    abar_prev = torch.cat([torch.ones(1, dtype=torch.float64), abar[:-1]])
    betas = 1 - abar / abar_prev
    s = dm.NoiseSchedule(T_steps, "cosine")          # a valid instance whose fields we overwrite
    s.kind = f"linear, {T_steps} steps"
    s.betas, s.alphas = betas, 1 - betas
    s.alpha_bars, s.alpha_bars_prev = abar, abar_prev
    s.sqrt_alpha_bars, s.sqrt_one_minus_alpha_bars = abar.sqrt(), (1 - abar).sqrt()
    s.posterior_variance = betas * (1 - abar_prev) / (1 - abar)
    s.posterior_coef_x0 = betas * abar_prev.sqrt() / (1 - abar)
    s.posterior_coef_xt = (1 - abar_prev) * (1 - betas).sqrt() / (1 - abar)
    s.log_snr = torch.log(abar) - torch.log1p(-abar)
    # Guard: if dm.NoiseSchedule ever gains another coefficient, it would silently keep its cosine values here.
    assert {k for k, v in vars(s).items() if torch.is_tensor(v)} == set(OVERWRITTEN), \
        "dm.NoiseSchedule gained a field: update make_schedule"
    return s.to(device)                                 # .to() also casts to float32


OVERWRITTEN = ("betas", "alphas", "alpha_bars", "alpha_bars_prev", "sqrt_alpha_bars", "sqrt_one_minus_alpha_bars",
               "posterior_variance", "posterior_coef_x0", "posterior_coef_xt", "log_snr")


sched = make_schedule(T)
ref_sched = dm.NoiseSchedule(T, "linear", device=device)
for name in ("alpha_bars", "posterior_variance", "posterior_coef_x0", "posterior_coef_xt"):
    assert torch.allclose(getattr(sched, name), getattr(ref_sched, name), rtol=1e-5, atol=1e-7), name
print("make_schedule(1000) == dm.NoiseSchedule(1000)  ✓")
for T_ in (10, 50, 1000):
    print(f"T={T_:4d}: largest beta = {make_schedule(T_).betas.max().item():.3f}  "
          f"(naive scaling would give {0.02 * 1000 / T_:.3f})")


# %%
def true_reverse_kernel(sched_T, t, x_t, n_grid=241, half_width=5.0):
    """Exact q(x_{t-1} | x_t) for GMM data on a grid around x_t/sqrt(alpha_t), plus summaries."""
    a, b = sched_T.alphas[t].cpu().double(), sched_T.betas[t].cpu().double()    # float64 on CPU (MPS has none)
    prior = gmm.noised(float(sched_T.alpha_bars_prev[t]))           # q_{t-1}, again a GMM
    centre, w = x_t / a.sqrt(), half_width * (b / a).sqrt()          # likelihood centre and +-5 std window
    g1 = [torch.linspace(c - w, c + w, n_grid, dtype=torch.float64) for c in centre]
    X, Y = torch.meshgrid(g1[0], g1[1], indexing="xy")
    pts = torch.stack([X.flatten(), Y.flatten()], 1)
    log_k = prior.log_prob(pts).double() - ((x_t - a.sqrt() * pts) ** 2).sum(1) / (2 * b)
    p = torch.softmax(log_k, 0)                                     # normalised probabilities on the grid
    mean = p @ pts
    cov = ((pts - mean).T * p) @ (pts - mean)
    # KL(true kernel || moment-matched Gaussian), a numerical measure of non-Gaussianity
    cell = (g1[0][1] - g1[0][0]) * (g1[1][1] - g1[1][0])
    gauss = torch.distributions.MultivariateNormal(mean, cov)
    kl = (p * (torch.log(p / cell + 1e-300) - gauss.log_prob(pts))).sum().item()
    tweedie = (x_t + b * gmm.noised(float(sched_T.alpha_bars[t])).score(x_t[None].double())[0]) / a.sqrt()
    return dict(X=X, Y=Y, p=p.view(n_grid, n_grid), mean=mean, cov=cov, kl=kl, tweedie=tweedie, beta=b.item(),
                beta_tilde=float(sched_T.posterior_variance[t]), var=cov.trace().item() / 2)   # var per coordinate


def draw_cov_ellipse(ax, mean, cov, n_std=2.0, **kw):
    """Draw the n_std contour of a 2-D Gaussian."""
    evals, evecs = torch.linalg.eigh(cov)
    ang = torch.linspace(0, 2 * math.pi, 100, dtype=torch.float64)
    circle = torch.stack([ang.cos(), ang.sin()], 1)
    pts = mean + n_std * circle @ (evecs * evals.sqrt()).T
    ax.plot(pts[:, 0], pts[:, 1], **kw)


x_query, tau = torch.tensor([0.0, 0.0], dtype=torch.float64), 0.1   # a point between the four inner blobs
# (0, 0) is a symmetry point of the mixture (it is invariant under 90-degree rotations), so there the kernel mean and
# Tweedie's formula are both exactly 0 whatever the formula. The check therefore uses asymmetric points.
x_checks = torch.tensor([[0.4, -0.7], [1.0, 0.3], [-1.2, 0.9]], dtype=torch.float64)
n_typical = 10 if FAST else 40                                       # typical x_t ~ q_t for an average KL
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
dm.scatter2d(gmm.sample(3000, generator=torch.Generator().manual_seed(0)), ax=axes[0], c=C["data"], lim=2.8,
             title="data $q(x_0)$ and the query point")
axes[0].plot(*x_query, "r+", ms=12, mew=2)
for ax, T_ in zip(axes[1:], (1000, 50, 10)):
    s_T = make_schedule(T_)
    t_idx = round(tau * T_) - 1                                      # the step that ends at time tau
    k = true_reverse_kernel(s_T, t_idx, x_query)
    for xq in x_checks:                                              # the mean of the true step is Tweedie's
        kq = true_reverse_kernel(s_T, t_idx, xq)
        assert torch.allclose(kq["mean"], kq["tweedie"], atol=1e-4), f"Tweedie one-step mean mismatch at {xq}"
    x_typ = gmm.noised(float(s_T.alpha_bars[t_idx])).sample(n_typical, generator=torch.Generator().manual_seed(1))
    kl_typ = np.array([true_reverse_kernel(s_T, t_idx, x.double(), n_grid=161)["kl"] for x in x_typ])
    ax.contourf(k["X"], k["Y"], k["p"], levels=15, cmap="Greys")
    draw_cov_ellipse(ax, k["mean"], k["cov"], 1.0, color=C["oracle"], ls="--", lw=1)
    draw_cov_ellipse(ax, k["mean"], k["cov"], 2.0, color=C["oracle"], ls="--", lw=1)
    ax.plot(*k["tweedie"], "x", color=C["oracle"], ms=8)
    ax.set_aspect("equal")
    ax.set_title(f"T={T_}: one step, $\\beta$={k['beta']:.3f}\nKL(true ‖ best Gaussian) = {k['kl']:.3f} nats")
    print(f"T={T_:4d}: at (0,0): KL={k['kl']:.4f}, true Var[x_t-1|x_t] per coord={k['var']:.4f} "
          f"(beta={k['beta']:.4f}, beta_tilde={k['beta_tilde']:.4f})\n"
          f"        typical x_t ~ q_t: KL median {np.median(kl_typ):.4f}, mean {kl_typ.mean():.4f}\n"
          f"        at x_t={xq.numpy()}: grid mean {kq['mean'].numpy().round(4)} == Tweedie {kq['tweedie'].numpy().round(4)}")
print("numerical mean of the true reverse step == Tweedie's formula at every asymmetric test point  ✓")
axes[1].set_ylabel("$x_{t-1}$ (each panel zoomed to its step size)")
plt.suptitle(f"True reverse step $q(x_{{t-1}}\\mid x_t)$ at $x_t$ = (0, 0), continuous time {tau} "
             "(grey), with the moment-matched Gaussian (red dashed: 1 and 2 sd; x: mean)", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# With 1000 steps the true reverse step from this point is a tiny bump that a Gaussian fits essentially perfectly (KL ≈ 0).
# With 50 steps it already has four lobes, one per nearby inner blob. With 10 steps it is four separate modes, and the
# best Gaussian puts its mean (red ×) in the empty space between them. The printout adds three facts.
# - **The mean is always right.** At asymmetric test points, the numerical mean of the true kernel matches Tweedie's
#   formula for every step size (asserted above). (At $(0,0)$ the match would prove nothing: both sides are 0 by symmetry.)
#   Small steps are what make the *Gaussian shape* right.
# - **The variance is not $\beta_t$.** At this ambiguous point, the true reverse step for $T=50$ and $T=10$ is markedly
#   wider than either standard choice of $\sigma_t^2$ ($\beta_t$ or $\tilde\beta_t$). §9c explains why.
# - **This point is a worst case.** Averaged over typical $x_t\sim q_t$, the kernel is much closer to Gaussian.
#   Non-Gaussianity concentrates in the low-density gaps between modes, which few particles visit. That is why $T=50$
#   does far better than $T=10$ in §9b, even though both kernels look bad at this point.
#
# ## 2 · The training objective: from the ELBO to "predict the noise"
#
# We fit $p_\theta$ by maximum likelihood. As in any latent-variable model (VAEs, or the Bayesian observers you know), the
# likelihood is intractable and we maximise the evidence lower bound instead, with the forward process as a *fixed* encoder:
# $$-\log p_\theta(x_0) \;\le\; \mathbb E_q\Big[-\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\Big]
# = \underbrace{D_\text{KL}\big(q(x_T|x_0)\,\|\,p(x_T)\big)}_{L_T}
# + \sum_{t=2}^{T}\underbrace{\mathbb E_q\, D_\text{KL}\big(q(x_{t-1}|x_t,x_0)\,\|\,p_\theta(x_{t-1}|x_t)\big)}_{L_{t-1}}
# \;\underbrace{-\ \mathbb E_q \log p_\theta(x_0|x_1)}_{L_0}.$$
# The step to this form rewrites each forward factor with Bayes, $q(x_t|x_{t-1}) = q(x_{t-1}|x_t,x_0)\,q(x_t|x_0)/q(x_{t-1}|x_0)$
# (allowed because the chain is Markov). The ratios $q(x_t|x_0)/q(x_{t-1}|x_0)$ then telescope away. The result is a sum of
# KL divergences **between two Gaussians**, which have closed forms, so there is no high-variance Monte Carlo over whole chains.
#
# **Step 1: KL between Gaussians with the same fixed variance.** With $\sigma_t^2$ fixed (DDPM does not learn it),
# $$L_{t-1} = \mathbb E_q\Big[\frac{1}{2\sigma_t^2}\big\|\tilde\mu_t(x_t,x_0)-\mu_\theta(x_t,t)\big\|^2\Big] + \text{const}.$$
#
# **Step 2: write $\tilde\mu_t$ in terms of the noise.** Substitute $x_0 = (x_t-\sqrt{1-\bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$.
# Using $\bar\alpha_t = \alpha_t\bar\alpha_{t-1}$, the coefficient of $x_t$ becomes
# $\frac{\alpha_t(1-\bar\alpha_{t-1}) + \beta_t}{\sqrt{\alpha_t}(1-\bar\alpha_t)} = \frac{1}{\sqrt{\alpha_t}}$, and that of $\epsilon$
# becomes $-\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t\sqrt{1-\bar\alpha_t}}{(1-\bar\alpha_t)\sqrt{\bar\alpha_t}} = -\frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\bar\alpha_t}}$:
# $$\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\Big).$$
#
# **Step 3: give the model the same form.** $x_t$ is an input, so the network only has to supply the unknown noise:
# $$\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big)
# \quad\Longrightarrow\quad
# L_{t-1} = \underbrace{\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}}_{\lambda_t}\;
# \mathbb E_{x_0,\epsilon}\big\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\ t)\big\|^2 + \text{const}.$$
#
# **Step 4: drop the weights.** Sampling $t$ uniformly and setting $\lambda_t\equiv 1$ gives Ho et al.'s
# $$L_\text{simple} = \mathbb E_{t\sim\mathcal U,\;x_0,\;\epsilon}\,\big\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\ t)\big\|^2 .$$
# What did we drop, and why is that fine?
# - $L_T$ has no parameters. $L_0$ is treated like the other terms: on continuous data our last step simply outputs the mean.
# - $\lambda_t$ is largest at small $t$ (plot below). The true bound puts a disproportionate share of its weight on nearly
#   imperceptible noise levels: roughly a third on the first 10% of steps. Uniform weighting is a *reweighted* bound that
#   shifts this emphasis to intermediate noise levels, where the global structure (which blob, which digit) is decided.
#   Ho et al. call these the "more difficult denoising tasks at larger $t$". They are difficult in $x_0$-space (recovering
#   the structure), even though, as §5 shows, the $\epsilon$-loss itself is small there. Ho et al. found that $L_\text{simple}$
#   gives better samples, while the true bound gives better likelihoods.
# - Crucially, the weighting does **not** change the optimum. For each $t$ separately, the minimiser of the squared error is the
#   posterior mean $\epsilon^*(x_t,t)=\mathbb E[\epsilon\mid x_t]$. The network is an amortised Bayesian observer that infers the
#   noise from a noisy observation. A weighting only decides where an imperfect network spends its capacity.

# %%
# Check steps 2-3 numerically (float64 on CPU) with random data, timesteps and a deliberately wrong noise prediction.
g = torch.Generator().manual_seed(0)
t_chk = torch.randint(1, T, (6,), generator=g)
x0_chk = gmm.sample(6, generator=g).double()
eps_chk = torch.randn(6, 2, generator=g, dtype=torch.float64)
eps_theta = eps_chk + 0.3 * torch.randn(6, 2, generator=g, dtype=torch.float64)
col = lambda v: v.cpu().double()[t_chk][:, None]                      # coefficient at t, broadcast over the 2 dims
x_t_chk = col(sched.sqrt_alpha_bars) * x0_chk + col(sched.sqrt_one_minus_alpha_bars) * eps_chk
mu_post = col(sched.posterior_coef_x0) * x0_chk + col(sched.posterior_coef_xt) * x_t_chk       # from notebook 01
mu_eps = (x_t_chk - col(sched.betas) / col(sched.sqrt_one_minus_alpha_bars) * eps_chk) / col(sched.alphas).sqrt()
assert torch.allclose(mu_post, mu_eps, atol=1e-5)
mu_theta = (x_t_chk - col(sched.betas) / col(sched.sqrt_one_minus_alpha_bars) * eps_theta) / col(sched.alphas).sqrt()
sigma = col(sched.betas).sqrt()                                         # sigma_t^2 = beta_t
kl_direct = torch.distributions.kl_divergence(torch.distributions.Normal(mu_post, sigma),
                                              torch.distributions.Normal(mu_theta, sigma)).sum(1)
lam = (col(sched.betas) ** 2 / (2 * sigma ** 2 * col(sched.alphas) * (1 - col(sched.alpha_bars))))[:, 0]
kl_formula = lam * ((eps_chk - eps_theta) ** 2).sum(1)
assert torch.allclose(kl_direct, kl_formula, rtol=1e-4)
print("mu_tilde(x_t, x0) == mu_tilde(x_t, eps)  ✓     KL == lambda_t * ||eps - eps_theta||^2  ✓")

# %%
ts = np.arange(T)
b, a, ab = (sched.betas.cpu().double().numpy(), sched.alphas.cpu().double().numpy(),
            sched.alpha_bars.cpu().double().numpy())
ab_prev = sched.alpha_bars_prev.cpu().double().numpy()
lam_beta = b / (2 * a * (1 - ab))                                   # sigma^2 = beta
lam_tilde = b / (2 * a * np.maximum(1 - ab_prev, 1e-12))            # sigma^2 = beta_tilde (infinite at t=0)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
axes[0].loglog(ts + 1, b, label=r"$\beta_t$", color=C["eps"])
axes[0].loglog(ts[1:] + 1, sched.posterior_variance.cpu()[1:], "--", label=r"$\tilde\beta_t$", color=C["x0"])
axes[0].set(xlabel="t + 1 (log scale)", ylabel="variance", title=r"Reverse-step variance choices $\sigma_t^2$")
axes[0].legend()
for lam_, lab, c in ((lam_beta, r"ELBO weight, $\sigma_t^2=\beta_t$", C["eps"]),
                     (lam_tilde, r"ELBO weight, $\sigma_t^2=\tilde\beta_t$", C["x0"])):
    axes[1].semilogy(ts[1:], lam_[1:] / lam_[1:].mean(), label=lab, color=c)
axes[1].axhline(1.0, color="k", lw=1, label=r"$L_\mathrm{simple}$ (uniform)")
axes[1].set(xlabel="t (code index)", ylabel="relative weight (mean = 1)", title=r"Per-step weight $\lambda_t$ of the true ELBO")
axes[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Left (log axes): the two natural choices for the reverse variance differ only for the first few dozen steps, where
# $\tilde\beta_t\to 0$. Right: relative to the uniform weighting of $L_\text{simple}$, the true ELBO weights the
# lowest-noise steps tens of times more than average, and intermediate steps ($t\approx 200$–$600$) only half to
# two-thirds as much.
# At the noisiest steps the two weightings roughly agree. So $L_\text{simple}$ moves weight from the nearly noise-free
# steps to intermediate noise levels, the "structural" denoising tasks.
#
# ## 3 · The denoiser network
#
# ### 3a · Telling the network what $t$ is: sinusoidal embeddings
# One network serves all $T$ noise levels, so it must know $t$. Feeding the raw integer $t\in\{0,\dots,999\}$ is a poor
# idea. It is one badly scaled scalar, and an MLP would have to build every $t$-dependent feature out of it through
# its nonlinearities (MLPs are biased towards smooth functions of their inputs). Instead we use transformer-style
# **sinusoidal features** at geometrically spaced frequencies,
# $$\text{emb}(t) = \big[\cos(\omega_1 t),\dots,\cos(\omega_{d/2} t),\ \sin(\omega_1 t),\dots,\sin(\omega_{d/2} t)\big],\qquad
# \omega_k = 10000^{-(k-1)/(d/2)} .$$
# These form a ready-made multi-scale basis. Low frequencies vary slowly and say roughly "how noisy". High frequencies
# separate neighbouring steps. A small learned MLP on top turns them into the features the network needs.
#
# > **Gotcha:** the frequency range must match how you scale $t$. With $t$ in $[0, 999]$ the fastest feature has
# > a period of about 6 steps and the slowest barely changes over the whole range. If you feed $t/T\in[0,1)$ into the same
# > embedding, all but a few of the fastest features are almost constant. The multi-scale code collapses to a handful of
# > smooth functions of $t$, which is essentially raw $t/T$ again. The network can still read the noise level from them,
# > so it is not blind, but it loses fine resolution in $t$. How much that costs depends on how sharply the ideal
# > denoiser changes with $t$ (Exercise 6 tests it here). Codebases therefore rescale continuous time (e.g. feed $1000\,t$)
# > or embed $\log\sigma$ with a matched frequency range.

# %%
def sinusoidal_embedding(t, dim, max_period=10000.0):
    """Map integer or float times t (shape (B,)) to (B, dim) sinusoidal features."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


t_all = torch.arange(T)
emb = sinusoidal_embedding(t_all, 64)
assert torch.allclose(emb, dm.SinusoidalEmbedding(64)(t_all), atol=1e-6)
print("matches dm.SinusoidalEmbedding  ✓")

emb_n = F.normalize(emb, dim=1)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1.1, 1, 1.1]})
# rows of every 4th step, no interpolation: keeps the image small (the fastest columns alias either way)
im = axes[0].imshow(emb[::4].numpy(), aspect="auto", cmap="RdBu_r", interpolation="nearest", extent=(0, 64, T, 0))
axes[0].set(xlabel="feature (cos | sin, fast → slow)", ylabel="t", title="Embedding matrix emb(t)")
plt.colorbar(im, ax=axes[0], fraction=0.05)
im = axes[1].imshow((emb_n[::4] @ emb_n[::4].T).numpy(), cmap="viridis", vmin=0, vmax=1,
                    extent=(0, T, T, 0))                              # every 4th step keeps the image small
axes[1].set(xlabel="t", ylabel="t'", title="Cosine similarity of emb(t), emb(t')")
plt.colorbar(im, ax=axes[1], fraction=0.05)
emb_unit = sinusoidal_embedding(t_all / T, 64)                       # the gotcha: t rescaled to [0, 1)
for e, ls, lab in ((emb, "-", "integer t in [0, 999]"), (emb_unit, "--", "t/T in [0, 1)")):
    feat_range = (e.max(0).values - e.min(0).values).numpy()        # how much each feature changes over all t
    axes[2].plot(np.arange(32), feat_range[:32], ls, color=C["eps"], label=f"cos features, {lab}")
    axes[2].plot(np.arange(32), feat_range[32:], ls, color=C["x0"], label=f"sin features, {lab}")
axes[2].set(xlabel="frequency index k (fast → slow)", ylabel="max − min over all t",
            title="How much each feature varies with t")
axes[2].legend(loc="center left", fontsize=7)
plt.tight_layout()
plt.show()

# %% [markdown]
# Left: each column is one feature. Fast oscillations on the left, slow drifts on the right, so every $t$ gets a unique,
# smoothly varying code. Middle: nearby steps have similar embeddings and distant steps dissimilar ones; the similarity
# band is narrow, so the code resolves $t$ finely. Right: with integer $t$ (solid) the fast and middle frequencies sweep
# their full range $[-1, 1]$, and only the slowest third change partially. With $t/T$ (dashed) only the first few
# frequencies change appreciably, and the rest of the embedding is dead weight: the gotcha above.
#
# ### 3b · A residual-MLP denoiser
# For 2-D data a small MLP suffices. Two design choices matter more than size:
# - **residual blocks with LayerNorm**, so depth does not hurt optimisation;
# - **the time embedding is added inside every block**, so every layer can modulate its computation by noise level.
#   Image U-Nets do the same thing with scale-and-shift ("FiLM") in each ResBlock (notebook 05).
#
# `dm.MLPDenoiser` is exactly this network (plus optional class labels for notebook 06). We check that below by
# copying our weights into it. The `fourier_octaves` option (Fourier features of the *input*, the same trick as §3a
# applied to $x$) is off by default; §10 explains when it is needed.

# %%
class SinusoidalTime(nn.Module):
    """nn.Module wrapper around sinusoidal_embedding (so it can sit inside nn.Sequential)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        return sinusoidal_embedding(t, self.dim)


class ResMLPDenoiser(nn.Module):
    """eps_theta(x, t) for low-dimensional data: residual MLP with the time embedding added in each block."""

    def __init__(self, data_dim=2, hidden=128, n_blocks=2, emb_dim=64, fourier_octaves=0):
        super().__init__()
        self.fourier_octaves = fourier_octaves
        self.time_emb = nn.Sequential(SinusoidalTime(emb_dim), nn.Linear(emb_dim, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden))
        self.inp = nn.Linear(data_dim * (1 + 2 * fourier_octaves), hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, data_dim))

    def input_features(self, x):
        """x itself, plus sin/cos(2^k pi x) for k < fourier_octaves (used in §10; none by default)."""
        if not self.fourier_octaves:
            return x
        freqs = math.pi * 2.0 ** torch.arange(self.fourier_octaves, device=x.device, dtype=x.dtype)
        args = (x[..., None] * freqs).flatten(1)                    # (B, data_dim * octaves)
        return torch.cat([x, torch.sin(args), torch.cos(args)], dim=1)

    def forward(self, x, t, y=None):           # y unused: keeps the tutorial-wide signature model(x, t, y=None)
        emb = self.time_emb(t)
        h = self.inp(self.input_features(x))
        for block in self.blocks:
            h = h + block(h + emb)             # residual update, conditioned on t
        return self.out(h)


net = ResMLPDenoiser().to(device)
twin = dm.MLPDenoiser(data_dim=2, hidden=128, n_blocks=2, emb_dim=64).to(device)
twin.load_state_dict(net.state_dict())          # same parameter names, so the weights transfer
x_tmp, t_tmp = torch.randn(5, 2, device=device), torch.randint(0, T, (5,), device=device)
assert torch.allclose(net(x_tmp, t_tmp), twin(x_tmp, t_tmp), atol=1e-6)
print(f"ResMLPDenoiser == dm.MLPDenoiser  ✓   ({dm.count_params(net):,} parameters)")

# %% [markdown]
# ## 4 · Training
#
# The training loop is Algorithm 1 of Ho et al.: draw data, an **independent random $t$ for every example**, and noise;
# form $x_t$ in closed form; regress the noise. Four practical details:
# - **Coefficient broadcasting.** `alpha_bars[t]` has shape `(B,)` and `x0` has shape `(B, 2)`. Index with `[:, None]`
#   (images need `(B, 1, 1, 1)`). Without it you get a shape error or, worse, a silent wrong broadcast.
# - **Gradient clipping** (norm 1) guards against the rare huge gradient from an unlucky batch.
# - **EMA weights.** We keep an exponential moving average of the weights, $\bar\theta \leftarrow d\,\bar\theta + (1-d)\,\theta$,
#   and *sample with $\bar\theta$*. §8 shows why. The decay is warmed up as $d=\min(0.999, \frac{1+n}{10+n})$ at step $n$,
#   so the average quickly forgets the random initialisation. ($d$ only reaches 0.999 after about 9000 steps.)
# - **Standardise the data.** The forward process ends at $\mathcal N(0, I)$, so data should have roughly unit scale. Our
#   mixture has standard deviation about 1 per coordinate.
#
# The `pred` argument (which target to regress: $\epsilon$, $x_0$ or $v$) is used in §9a; ignore it for now.

# %%
def diffuse(x0, t, eps, schedule, pred="eps"):
    """Return x_t = sqrt(abar) x0 + sqrt(1-abar) eps and the regression target for parameterisation `pred`."""
    sa = schedule.sqrt_alpha_bars[t][:, None]               # (B, 1): broadcasts over the data dimension
    s1a = schedule.sqrt_one_minus_alpha_bars[t][:, None]
    x_t = sa * x0 + s1a * eps
    target = {"eps": eps, "x0": x0, "v": sa * eps - s1a * x0}[pred]
    return x_t, target


@torch.no_grad()
def ema_update(ema_model, model, step, decay=0.999, warmup=True):
    """theta_ema <- d * theta_ema + (1 - d) * theta, with warm-up d = min(decay, (1+step)/(10+step))."""
    d = min(decay, (1 + step) / (10 + step)) if warmup else decay
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.lerp_(p, 1 - d)


@torch.no_grad()
def heldout_loss(model, x0, t, eps, schedule, pred="eps"):
    """The training loss on a fixed held-out batch: a low-noise progress measure."""
    x_t, target = diffuse(x0, t, eps, schedule, pred)
    return F.mse_loss(model(x_t, t), target).item()


def train_ddpm(model, data, schedule, pred="eps", steps=STEPS, batch_size=256, lr=1e-3, ema_decay=0.999,
               ema_warmup=True, grad_clip=1.0, eval_batch=None, eval_every=None, seed=0, log_every=None):
    """DDPM training (Ho et al. 2020, Alg. 1) with Adam, gradient clipping and an EMA copy of the weights."""
    g = torch.Generator(device=device).manual_seed(seed)
    model.to(device).train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    eval_every = eval_every or max(steps // 40, 1)
    losses, evals, t_start = [], [], time.time()
    for step in range(1, steps + 1):
        x0 = data[torch.randint(0, len(data), (batch_size,), generator=g, device=device)]
        t = torch.randint(0, schedule.T, (batch_size,), generator=g, device=device)   # one t per example
        eps = torch.randn(x0.shape, generator=g, device=device)
        x_t, target = diffuse(x0, t, eps, schedule, pred)
        loss = F.mse_loss(model(x_t, t), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        ema_update(ema, model, step, ema_decay, ema_warmup)
        losses.append(loss.item())          # .item() syncs a GPU every step: fine for tiny models
        if eval_batch is not None and step % eval_every == 0:
            evals.append((step, heldout_loss(model, *eval_batch, schedule, pred),
                          heldout_loss(ema, *eval_batch, schedule, pred)))
        if log_every and step % log_every == 0:
            print(f"  step {step:5d} | loss (last {log_every}) {np.mean(losses[-log_every:]):.4f} "
                  f"| {time.time() - t_start:5.1f} s")
    model.eval()
    return dict(model=model, ema=ema, losses=np.array(losses), evals=np.array(evals),
                seconds=time.time() - t_start)


# Check 1: our loss equals dm.diffusion_loss given the same t and the same noise.
x0_tmp = gmm.sample(64, generator=torch.Generator().manual_seed(1)).to(device)
t_tmp = torch.randint(0, T, (64,), device=device)
torch.manual_seed(123)
x_t_tmp, target_tmp = diffuse(x0_tmp, t_tmp, torch.randn_like(x0_tmp), sched)
ours = F.mse_loss(net(x_t_tmp, t_tmp), target_tmp)
torch.manual_seed(123)                       # dm.diffusion_loss draws its noise with randn_like, like us
assert torch.allclose(ours, dm.diffusion_loss(net, x0_tmp, sched, t=t_tmp), atol=1e-6)
# Check 2: our EMA update equals dm.EMA (same warm-up rule).
ema_ours, ema_dm = copy.deepcopy(net), dm.EMA(net, decay=0.999)
for step in range(1, 6):
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.01 * torch.randn_like(p))
    ema_update(ema_ours, net, step)
    ema_dm.update(net)
assert all(torch.allclose(p, q, atol=1e-6) for p, q in zip(ema_ours.parameters(), ema_dm.model.parameters()))
print("loss == dm.diffusion_loss  ✓    ema_update == dm.EMA  ✓")

# %% [markdown]
# **How low can the loss go?** For each $t$ the best possible predictor is the posterior mean $\epsilon^*(x_t,t)=\mathbb E[\epsilon\mid x_t]$.
# Its loss is not zero: many $(x_0,\epsilon)$ pairs produce the same $x_t$. The loss is a mean over the $d=2$
# coordinates, so its floor is the average posterior variance per coordinate, $\mathbb E\,\mathrm{tr}\,\mathrm{Cov}[\epsilon\mid x_t]/d$.
#
# Our mixture lets us compute $\epsilon^*$ exactly, via Tweedie's formula (notebook 03 develops it fully). The derivation
# takes two lines. The score of a marginal is the posterior average of the likelihood's score (Fisher's identity):
# $$\nabla\log q_t(x_t) = \mathbb E\big[\nabla_{x_t}\log q(x_t\mid x_0)\,\big|\,x_t\big]
# = -\,\mathbb E\Big[\frac{x_t-\sqrt{\bar\alpha_t}\,x_0}{1-\bar\alpha_t}\,\Big|\,x_t\Big]
# = -\frac{\mathbb E[\epsilon\mid x_t]}{\sqrt{1-\bar\alpha_t}}
# \quad\Longrightarrow\quad \epsilon^*(x_t,t) = -\sqrt{1-\bar\alpha_t}\;\nabla\log q_t(x_t).$$
# `gmm.noised(abar).score` is that gradient in closed form. Substituting $\epsilon^*$ into DDPM's mean of §2 gives
# $\mu_\theta = \big(x_t + \beta_t\nabla\log q_t(x_t)\big)/\sqrt{\alpha_t}$, exactly the true reverse mean of §1. With a
# perfect network, DDPM's reverse step has the right mean. This **oracle** is the yardstick for everything that follows:
# it is what a perfect network would output.

# %%
class OracleDenoiser:
    """Bayes-optimal eps-prediction E[eps | x_t] = -sqrt(1-abar_t) * score of q_t, exact for GMM data."""

    def __init__(self, gmm, schedule):
        self.s1a = schedule.sqrt_one_minus_alpha_bars.cpu().double()
        self.q_t = [gmm.noised(a) for a in schedule.alpha_bars.cpu().double().tolist()]   # one GMM per step

    def __call__(self, x, t, y=None):
        out = torch.empty_like(x)
        for tv in t.unique().tolist():             # group the batch by timestep
            m = t == tv
            score = self.q_t[tv].score(x[m].cpu().double())       # float64 on CPU (MPS has no float64)
            out[m] = (-self.s1a[tv] * score).float().to(x.device)
        return out


oracle = OracleDenoiser(gmm, sched)

# The dataset: a fixed, finite training set (as in real life), plus a held-out batch for monitoring.
g = torch.Generator().manual_seed(0)
train_data = gmm.sample(20_000, generator=g).to(device)
n_eval = 1024 if FAST else 4096
eval_batch = (gmm.sample(n_eval, generator=g).to(device), torch.randint(0, T, (n_eval,), generator=g).to(device),
              torch.randn(n_eval, 2, generator=g).to(device))
bayes_floor = heldout_loss(oracle, *eval_batch, sched)
print(f"training data: {tuple(train_data.shape)}, mean {train_data.mean(0).cpu().numpy().round(2)}, "
      f"std {train_data.std(0).cpu().numpy().round(2)}")
print(f"Bayes-optimal loss on the held-out batch (the floor): {bayes_floor:.4f}")

# %%
dm.seed_everything(0)
main = train_ddpm(ResMLPDenoiser(), train_data, sched, eval_batch=eval_batch, log_every=STEPS // 4)
print(f"trained in {main['seconds']:.0f} s; final held-out loss: raw weights {main['evals'][-1, 1]:.4f}, "
      f"EMA weights {main['evals'][-1, 2]:.4f}, floor {bayes_floor:.4f}")


# %%
def running_mean(x, w=100):
    """Trailing moving average (for smoothing loss curves)."""
    w = max(1, min(w, len(x) // 5))
    return np.convolve(x, np.ones(w) / w, mode="valid")


fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(np.arange(1, STEPS + 1), main["losses"], color=C["eps"], alpha=0.25, lw=0.5, label="minibatch loss")
rm = running_mean(main["losses"])
ax.plot(np.arange(STEPS - len(rm) + 1, STEPS + 1), rm, color=C["eps"], lw=1.5, label="running mean")
ax.axhline(bayes_floor, color=C["oracle"], ls="--", label="Bayes-optimal floor (oracle)")
ax.set(xlabel="training step", ylabel=r"$\|\epsilon-\epsilon_\theta\|^2$ (mean per coordinate)", ylim=(0, 1.05),
       title="Training loss of the main model")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# The loss drops quickly and then settles close to the dashed line, the loss of the *perfect* denoiser. So a plateau well
# above zero is not a sign of failure: it is irreducible posterior uncertainty about the noise. The minibatch loss stays
# very noisy because every batch mixes random timesteps whose losses differ by orders of magnitude (next section), plus
# fresh random noise. Use a running mean or a fixed held-out batch to judge progress, never a single minibatch.
#
# ## 5 · Where is denoising hard? The loss as a function of $t$
#
# To compare predictions we convert any network output into an $\epsilon$-prediction and an $x_0$-prediction. For
# $\epsilon$-prediction, $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$. The $x_0$ and $v$ cases are
# explained in §9a. We evaluate the trained (EMA) model and the oracle on the same $x_0,\epsilon$ at 50 timesteps.

# %%
def eps_x0_from_output(out, x_t, t, schedule, pred="eps"):
    """Convert a network output under parameterisation `pred` into (eps_hat, x0_hat)."""
    sa = schedule.sqrt_alpha_bars[t][:, None]
    s1a = schedule.sqrt_one_minus_alpha_bars[t][:, None]
    if pred == "eps":
        eps = out
        x0 = (x_t - s1a * eps) / sa
    elif pred == "x0":
        x0 = out
        eps = (x_t - sa * x0) / s1a
    elif pred == "v":                      # v = sa * eps - s1a * x0  (see §9a)
        eps = s1a * x_t + sa * out
        x0 = sa * x_t - s1a * out
    else:
        raise ValueError(pred)
    return eps, x0


@torch.no_grad()
def per_t_errors(model, schedule, pred="eps", n=None, n_t=50, seed=0):
    """MSE of the implied eps- and x0-predictions at n_t evenly spaced timesteps (same x0, eps for every t)."""
    n = n or (512 if FAST else 2000)
    ts = torch.linspace(0, schedule.T - 1, n_t).round().long()
    g = torch.Generator().manual_seed(seed)
    x0, eps = gmm.sample(n, generator=g).to(device), torch.randn(n, 2, generator=g).to(device)
    err_eps, err_x0 = [], []
    for t in ts.tolist():
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        x_t, _ = diffuse(x0, tt, eps, schedule)
        eps_hat, x0_hat = eps_x0_from_output(model(x_t, tt), x_t, tt, schedule, pred)
        err_eps.append(F.mse_loss(eps_hat, eps).item())
        err_x0.append(F.mse_loss(x0_hat, x0).item())
    return ts.numpy(), np.array(err_eps), np.array(err_x0)


ts_eval, err_main, _ = per_t_errors(main["ema"], sched)
_, err_oracle, _ = per_t_errors(oracle, sched)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
axes[0].plot(ts_eval, err_main, "o-", ms=3, color=C["eps"], label="trained model (EMA)")
axes[0].plot(ts_eval, err_oracle, "--", color=C["oracle"], label="oracle = Bayes floor")
axes[0].set(xlabel="t", ylabel=r"$\epsilon$-MSE", title=r"Noise-prediction error vs $t$")
axes[0].legend()
axes[1].plot(ts_eval, err_main - err_oracle, "o-", ms=3, color=C["eps"])
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set(xlabel="t", ylabel="model MSE − oracle MSE", title="Excess error of the network")
plt.tight_layout()
plt.show()
print(f"average over t: model {err_main.mean():.4f}, oracle {err_oracle.mean():.4f}")

# %% [markdown]
# **Shape of the curve.** Treat the data locally as Gaussian with variance $s^2$. Then
# $\mathrm{Var}[\epsilon\mid x_t] = \frac{\bar\alpha_t s^2}{\bar\alpha_t s^2 + 1-\bar\alpha_t}$, which is set by the
# ratio of signal variance to noise variance:
# - **small $t$** (noise much smaller than the data's local spread): a tiny perturbation cannot be told apart from the data's
#   own variability, so $\epsilon$ is nearly unidentifiable and the error approaches 1, its prior variance. This is the
#   *hard* end for $\epsilon$-prediction. Fortunately, a wrong $\hat\epsilon$ barely changes the sampler update or $\hat x_0$
#   here, because it enters with a factor of about $\sqrt{1-\bar\alpha_t}\approx 0$ or smaller. This is how §2 and Ho et al.
#   can call *large* $t$ the difficult tasks: small $t$ is hard in $\epsilon$-MSE but unimportant, while large $t$ is
#   easy in $\epsilon$-MSE but decides the structure that $x_0$ will have.
# - **large $t$** ($x_t\approx\epsilon$): the noise is essentially the observation itself, so it is easy to predict and the error goes to 0.
# - **in between**, the curve is not monotone. It first drops: once the noise is wider than a blob but narrower than the
#   gaps between blobs, $x_t$ pins down its blob and the noise inside it becomes identifiable. It then rises to a bump where
#   the noise is comparable to the gaps, and it becomes ambiguous *which* blob $x_t$ came from. Uncertainty about $x_0$ is
#   uncertainty about $\epsilon$.
#
# Right panel: the network's excess error over the oracle is concentrated at small $t$, where the optimal
# $\epsilon^*(x_t,t)$ has the finest structure (the edges of the blobs). Elsewhere it is essentially zero. Small negative
# values are Monte-Carlo noise: both are scored on the same finite sample at each $t$.
#
# ## 6 · Sampling: ancestral sampling, written out
#
# Algorithm 2 of Ho et al. starts from $x_T\sim\mathcal N(0,I)$ and applies the learned reverse step $T$ times:
# $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t)\Big) + \sigma_t z,\qquad z\sim\mathcal N(0,I),$$
# with no noise on the very last step. Along the way we record $x_t$ and the model's current guess of the final answer,
# $\hat x_0(x_t,t)$. That guess is the model's estimate of the posterior mean $\mathbb E[x_0\mid x_t]$ (exactly that, for the
# oracle): what a Bayesian observer of $x_t$ would report.
#
# > **Gotcha:** `dm.ddpm_sample` computes the same mean through $\hat x_0$ and the posterior coefficients of §1. That form allows
# > clipping $\hat x_0$ to the data range (`clip_x0`, standard for images in $[-1,1]$). The $\epsilon$-form above cannot clip.
# > The two forms are algebraically identical, which we check below by running both with the same random numbers.

# %%
@torch.no_grad()
def ancestral_sample(model, schedule, n, pred="eps", var="beta_tilde", generator=None, record_every=0,
                     x_start=None, t_start=None):
    """DDPM ancestral sampling (Ho et al. 2020, Alg. 2). var: 'beta_tilde' (posterior variance) or 'beta'.

    By default starts from x_T ~ N(0, I); x_start (shape (2,) or (n, 2)) at index t_start starts mid-chain instead.
    """
    if x_start is None:
        x, t_start = torch.randn((n, 2), generator=generator, device=device), schedule.T - 1   # x_T ~ N(0, I)
    else:
        x = x_start.to(device).expand(n, 2).clone()
    traj = []
    for t in range(t_start, -1, -1):
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        eps_hat, x0_hat = eps_x0_from_output(model(x, tt), x, tt, schedule, pred)
        if record_every and (t % record_every == 0 or t == schedule.T - 1):
            traj.append((t, x.cpu(), x0_hat.cpu()))
        mean = (x - schedule.betas[t] / schedule.sqrt_one_minus_alpha_bars[t] * eps_hat) / schedule.alphas[t].sqrt()
        if t > 0:
            var_t = schedule.posterior_variance[t] if var == "beta_tilde" else schedule.betas[t]
            x = mean + var_t.sqrt() * torch.randn(x.shape, generator=generator, device=device)
        else:
            x = mean                                                             # last step: no noise
    return (x, traj) if record_every else x


def gen(seed):
    """A fresh random-number generator on the compute device (to give samplers identical noise)."""
    return torch.Generator(device=device).manual_seed(seed)


n_chk = 64 if FAST else 256
ours = ancestral_sample(main["ema"], sched, n_chk, generator=gen(1))
theirs = dm.ddpm_sample(main["ema"], sched, (n_chk, 2), generator=gen(1))
print(f"max |ours - dm.ddpm_sample| = {(ours - theirs).abs().max().item():.2e}")
assert torch.allclose(ours, theirs, atol=1e-3)
print("ancestral_sample == dm.ddpm_sample  ✓")

# %%
samples_main, traj = ancestral_sample(main["ema"], sched, N_SAMPLE, generator=gen(5), record_every=10)
final_mode = gmm.responsibilities(samples_main.cpu()).argmax(1)   # colour every particle by where it ends up
colors = plt.cm.tab10(final_mode.numpy() % 10)
snap_ts = [999, 500, 300, 150, 50, 0]
snaps = {t: (x, x0) for t, x, x0 in traj}
n_show = min(700, N_SAMPLE)                                        # fewer points keep the figure light
fig, axes = plt.subplots(2, len(snap_ts), figsize=(2.2 * len(snap_ts), 5.0))
for j, t in enumerate(snap_ts):
    x_t, x0_hat = snaps[t]
    dm.scatter2d(x_t[:n_show], ax=axes[0, j], c=colors[:n_show], lim=3.0, s=2, title=f"$x_t$, t={t}")
    dm.scatter2d(x0_hat[:n_show], ax=axes[1, j], c=colors[:n_show], lim=3.0, s=2, title=rf"$\hat x_0(x_t)$, t={t}")
axes[0, 0].set_ylabel("current state")
axes[1, 0].set_ylabel("model's guess of $x_0$")
plt.suptitle("Reverse trajectory (colour = mixture component the particle finally lands in)", y=1.0)
plt.tight_layout(h_pad=1.5)
plt.show()

# %% [markdown]
# Top row: the cloud contracts from an isotropic Gaussian onto the star, with the colours (the eventual destinations)
# sorting themselves out gradually. Bottom row: at $t=500$ every guess $\hat x_0$ sits near the data mean. A nearly
# pure-noise observation carries little information, so the posterior mean is close to the prior mean. As $t$ falls,
# the guesses first spread out *between* neighbouring blobs (an ambiguous $x_t$ has a posterior mean between the blobs it
# could have come from), then commit to one blob, and only at the end sharpen *within* it. Coarse decisions are made
# early and fine detail late, much like a Bayesian observer accumulating evidence.
#
# > **Gotcha.** The guesses at $t=999$ are scattered for a numerical reason. $\hat x_0 = (x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$
# > divides by $\sqrt{\bar\alpha_{999}}\approx 0.006$, which magnifies tiny errors in $\hat\epsilon$ about 160-fold. The sampler
# > itself is unaffected (its update uses $\hat\epsilon$ directly). Anything that *uses* $\hat x_0$ at high noise, though, needs
# > care. This is one reason image samplers clip $\hat x_0$, and it reappears in §9a.

# %%
path_t = np.array([t for t, _, _ in traj])
first_in_blob = {}                                       # follow 5 particles that end in 5 different blobs
for i, k in enumerate(final_mode.tolist()):
    first_in_blob.setdefault(k, i)
idx = torch.tensor(sorted(first_in_blob.values())[:5])
path_x = torch.stack([x for _, x, _ in traj])[:, idx]    # (snapshots, 5, 2), from t = T-1 down to 0
path_x0 = torch.stack([x0 for _, _, x0 in traj])[:, idx]
g1 = torch.linspace(-3, 3, 200)
X, Y = torch.meshgrid(g1, g1, indexing="xy")
dens = gmm.log_prob(torch.stack([X.flatten(), Y.flatten()], 1)).exp().view(200, 200)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1, 1.25, 1.25]})
axes[0].contour(X, Y, dens, levels=6, colors="0.75", linewidths=0.7)
for i in range(len(idx)):
    axes[0].plot(path_x[:, i, 0], path_x[:, i, 1], color=colors[idx[i]], lw=0.8, alpha=0.8)
for i in range(len(idx)):                                # markers last, so the paths do not hide them
    axes[0].plot(path_x[0, i, 0], path_x[0, i, 1], "s", color=colors[idx[i]], ms=5, mec="k", mew=0.8, zorder=5)
    axes[0].plot(path_x[-1, i, 0], path_x[-1, i, 1], "o", color=colors[idx[i]], ms=7, mec="w", mew=1.2, zorder=5)
axes[0].set(xlim=(-3, 3), ylim=(-3, 3), aspect="equal", title="2-D paths of $x_t$ (■ start, ● final sample)")
for ax, d in zip(axes[1:], (0, 1)):
    for i in range(len(idx)):
        ax.plot(path_t, path_x[:, i, d], color=colors[idx[i]], lw=0.7, alpha=0.5)
        ax.plot(path_t, path_x0[:, i, d], color=colors[idx[i]], lw=1.8)
    ax.set(xlim=(T, 0), ylim=(-3, 3), xlabel="t (sampling runs left to right)", ylabel=f"coordinate {d + 1}",
           title=f"coordinate {d + 1}: state $x_t$ (thin) and guess $\\hat x_0$ (thick)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Individual particles (left) take noisy, wandering routes. Each reverse step injects fresh noise $\sigma_t z$, so
# DDPM sampling is a *stochastic* process, a discretised reverse-time SDE (notebook 04 derives it). The time courses (middle,
# right) separate state from belief. The state $x_t$ (thin) wanders with the injected noise almost to the end. The guess
# $\hat x_0$ (thick) fluctuates much less. Apart from the high-noise artefact at the very start, it hovers near the data
# mean while $t$ is large, drifts towards one blob over a stretch of intermediate $t$, and then changes little.
#
# **The belief is a martingale.** There is an exact statement behind "a Bayesian observer accumulating evidence". Under
# the true reverse process, the posterior mean $\hat x_0(x_t)=\mathbb E[x_0\mid x_t]$ satisfies, for any $s<t$,
# $$\mathbb E\big[\hat x_0(x_s)\,\big|\,x_t\big] = \mathbb E\big[\mathbb E[x_0\mid x_s, x_t]\,\big|\,x_t\big] = \mathbb E[x_0\mid x_t] = \hat x_0(x_t).$$
# The first step holds because $x_t$ depends on $x_0$ only through $x_s$ (the forward chain is Markov), and the second is the
# tower property. Later beliefs can move, but *on average* they equal today's belief. With $s=0$: the average final sample
# reached from $x_t$ is the current guess $\hat x_0(x_t)$. More than that, continuing the true reverse process from $x_t$
# draws $x_0$ from the posterior $q(x_0\mid x_t)$. We check both with the oracle: take one ambiguous state from the run
# above, restart the sampler from it many times, and compare where the copies end up with what the observer believes.

# %%
t_c = 300                                                    # restart from the state at this step
x_t_all = snaps[t_c][0]
post_blob = gmm.noised(float(sched.alpha_bars[t_c])).responsibilities(x_t_all)   # exact p(blob | x_t)
torn = post_blob.sort(1, descending=True).values[:, 1]       # probability of the runner-up blob
torn[x_t_all.norm(dim=1) > 2.0] = 0                          # skip outliers (keep the plot readable)
i_c = torn.argmax().item()                                   # the particle most torn between two blobs
x_c, tt_c = x_t_all[i_c].to(device), torch.full((1,), t_c, device=device)
belief = eps_x0_from_output(oracle(x_c[None], tt_c), x_c[None], tt_c, sched)[1][0].cpu()   # oracle E[x0 | x_t]
n_c = 500 if FAST else 4000
completions = ancestral_sample(oracle, sched, n_c, generator=gen(7), x_start=x_c, t_start=t_c).cpu()
comp_blob = gmm.responsibilities(completions).argmax(1)
frac = torch.bincount(comp_blob, minlength=gmm.K).float() / n_c
se = completions.std(0) / math.sqrt(n_c)                     # Monte-Carlo standard error of the mean
print(f"x_t = {x_c.cpu().numpy().round(3)} at t={t_c}")
print(f"belief x0_hat(x_t): oracle {belief.numpy().round(3)}, trained net {snaps[t_c][1][i_c].numpy().round(3)}")
print(f"mean of {n_c} completions: {completions.mean(0).numpy().round(3)} (± {se.max().item():.3f} s.e.)")
print(f"TV(fraction of completions per blob, exact p(blob | x_t)) = {0.5 * (frac - post_blob[i_c]).abs().sum().item():.3f}")
assert (completions.mean(0) - belief).abs().max() < 4 * se.max() + 0.02, "martingale check failed"

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1, 1.3]})
dm.scatter2d(gmm.sample(2000, generator=torch.Generator().manual_seed(3)), ax=axes[0], c="0.85", lim=3)  # data
dm.scatter2d(completions[:2000], ax=axes[0], c=plt.cm.tab10(comp_blob[:2000].numpy() % 10), s=3, alpha=0.6,
             title=f"Completions of one $x_t$ (t={t_c})")
axes[0].plot(*x_c.cpu(), "k*", ms=12, label="restart state $x_t$")
axes[0].plot(*belief, "o", mfc="none", mec="k", ms=11, mew=2, label=r"belief $\hat x_0(x_t)$")
axes[0].plot(*completions.mean(0), "x", color=C["oracle"], ms=10, mew=2, label="mean of completions")
axes[0].legend(loc="lower right", fontsize=7)
kk = np.arange(gmm.K)
axes[1].bar(kk - 0.2, post_blob[i_c].numpy(), 0.4, color=plt.cm.tab10(kk), alpha=0.45, label="exact $p$(blob | $x_t$)")
axes[1].bar(kk + 0.2, frac.numpy(), 0.4, color=plt.cm.tab10(kk), label="fraction of completions")
axes[1].set(xlabel="blob (mixture component)", ylabel="probability", title="Where the copies end up", xticks=kk)
axes[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# The copies (left) split between the blobs this $x_t$ could have come from, in the proportions of the exact posterior
# $p(\text{blob}\mid x_t)$ (right). Their average lands on the observer's current belief (circle), which can sit in a gap
# between blobs where no sample ever ends up. So $\hat x_0$ summarises where the sampler *might* go; it is not itself a
# plausible sample. The trained network's belief at this state (printed) is close to the oracle's. "Noise partway, then
# run the sampler again" is the move behind SDEdit-style editing (Meng et al. 2022).
#
# ## 7 · How good are the samples? MMD and sliced Wasserstein
#
# Looking is not enough: we need numbers to compare variants. For low-dimensional samples two simple two-sample
# distances work well.
# - **MMD** (maximum mean discrepancy) compares the two sets through a kernel:
#   $\text{MMD}^2 = \mathbb E\,k(x,x') + \mathbb E\,k(y,y') - 2\,\mathbb E\,k(x,y)$. With a sum of RBF kernels at several
#   bandwidths it is sensitive to differences at several length scales. `dm.mmd_rbf` uses the simple *biased* estimator,
#   which is positive even for two samples from the same distribution (a floor of order $1/n$).
# - **Sliced Wasserstein** projects both sets onto random directions, where Wasserstein-1 has a closed form (sort and
#   compare), and averages. It is in data units: roughly how far mass must move.
# - Because our data are a known mixture, we add a ground-truth diagnostic: the **average true log-density** of the
#   samples, $\frac1n\sum\log q(x_i)$. Compared with its value on real data, *lower* means mass in the wrong places (too
#   diffuse) and *higher* means mass piled onto the modes (too concentrated, missing the spread).
#
# Both distances are noisy at $n\approx 2000$, so we always report the **floor**: the same metric between independent data
# samples, as mean ± sd over several draws. A useful bad baseline is the best-fitting single Gaussian. A useful good
# reference is the **oracle** plugged into the same sampler with the same noise: it has no network error, so any gap
# between it and the trained model is the network's fault, and any gap between it and the floor is the sampler's.

# %%
ref = gmm.sample(N_SAMPLE, generator=torch.Generator().manual_seed(1000))       # the fixed reference sample


def evaluate(x, reference=ref, density=gmm):
    """MMD, sliced Wasserstein to the reference sample, and (if known) the mean true log-density."""
    x = x.detach().float().cpu()
    out = dict(mmd=dm.mmd_rbf(x, reference), sw=dm.sliced_wasserstein(x, reference))
    if density is not None:
        out["logq"] = density.log_prob(x).mean().item()
    return out


def fmt(m):
    """One-line summary of an `evaluate` result."""
    s = f"MMD {m['mmd']:.4f}  SW {m['sw']:.3f}"
    return s + (f"  log q {m['logq']:.2f}" if "logq" in m else "")


floor_runs = [evaluate(gmm.sample(N_SAMPLE, generator=torch.Generator().manual_seed(2000 + i))) for i in range(5)]
floor = {k: (np.mean([r[k] for r in floor_runs]), np.std([r[k] for r in floor_runs])) for k in floor_runs[0]}
mu_d, cov_d = train_data.mean(0).cpu().double(), torch.cov(train_data.T).cpu().double()
gauss_fit = torch.distributions.MultivariateNormal(mu_d, cov_d).sample((N_SAMPLE,)).float()
samples_oracle = ancestral_sample(oracle, sched, N_SAMPLE, generator=gen(5))     # same sampler noise as samples_main
results = {"fitted Gaussian": evaluate(gauss_fit), "oracle denoiser": evaluate(samples_oracle),
           "DDPM (EMA weights)": evaluate(samples_main)}
print(f"{'':24s} {'MMD':>17s} {'sliced W1':>15s} {'mean log q(x)':>17s}")
print(f"{'data vs data (floor)':24s} {floor['mmd'][0]:9.4f} ± {floor['mmd'][1]:.4f} {floor['sw'][0]:8.3f} ± {floor['sw'][1]:.3f}"
      f" {floor['logq'][0]:9.2f} ± {floor['logq'][1]:.2f}")
for name, m in results.items():
    print(f"{name:24s} {m['mmd']:9.4f}         {m['sw']:8.3f}         {m['logq']:9.2f}")

fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
dm.scatter2d(gmm.sample(N_SAMPLE, generator=torch.Generator().manual_seed(3)), ax=axes[0], c=C["data"],
             title="fresh data")
dm.scatter2d(gauss_fit, ax=axes[1], c=C["gauss"], title="fitted Gaussian\n" + fmt(results["fitted Gaussian"]))
dm.scatter2d(samples_oracle, ax=axes[2], c=C["oracle"], title="oracle denoiser, T=1000\n" + fmt(results["oracle denoiser"]))
dm.scatter2d(samples_main, ax=axes[3], c=C["eps"], title="DDPM (EMA)\n" + fmt(results["DDPM (EMA weights)"]))
plt.tight_layout()
plt.show()

# %% [markdown]
# The single Gaussian scores far worse than the floor on every metric. The oracle row sits at, or very near, the floor:
# with 1000 small steps, the sampler itself costs little. By MMD and SW, the DDPM samples also come close to the
# floor. The mean log-density sees more, because it uses the true density. Compare the model's value with the data's and
# the oracle's. A value below both means the samples are slightly too diffuse, and since the oracle shares the sampler,
# that is network error. The excess error at small $t$ (§5), where the last steps sharpen each blob, is the natural suspect.
#
# Use the mean log-density only as a *dispersion* diagnostic. It is not a distance between distributions: it cannot see
# misallocated mode mass or displaced blobs (§8), and it rewards piling samples onto the modes (§9a). A sampler that
# put every point exactly on a blob centre would beat real data.
#
# ## 8 · EMA: the weights you should sample with
#
# With a constant learning rate, Adam's iterates never settle. Driven by minibatch noise, they jitter around a good
# solution with a per-step amplitude of order the learning rate (Adam normalises away the scale of the gradient). Averaging the iterates (Polyak–Ruppert averaging) cancels much of that
# jitter. An exponential moving average with decay $d$ averages over roughly the last $1/(1-d)$ steps. With the warm-up rule
# of §4, $d\approx 0.998$ at the end of our 4000-step run, so the window is a few hundred steps.
# For diffusion models this matters more than for a classifier: the sampler calls the network $T$ times and the errors
# **compound**. A small systematic wobble in $\epsilon_\theta$ becomes a visible distortion of the samples.
#
# We trained with an EMA copy all along, and logged the held-out loss of both the raw and the averaged weights. We now
# also sample from the raw weights, with *the same random numbers* as the EMA samples above. The ground truth gives us
# two more diagnostics: whether each blob receives the right share of samples (total-variation error of the blob masses),
# and whether each blob sits in the right place (distance of its sample mean from the true centre).

# %%
def blob_report(x):
    """Ground-truth diagnostics: TV error of the blob masses, mean distance of blob sample means from true centres,
    and the per-blob displacement vectors (sample mean - true centre; zero for an empty blob)."""
    x = x.detach().float().cpu()
    k = gmm.responsibilities(x).argmax(1)                 # assign every point to its most likely blob
    freq = torch.bincount(k, minlength=gmm.K).double() / len(k)
    tv = 0.5 * (freq - gmm.weights).abs().sum().item()
    disp = torch.stack([x[k == c].double().mean(0) - gmm.means[c] if (k == c).any() else torch.zeros(2, dtype=torch.float64)
                        for c in range(gmm.K)])
    shift = disp[freq > 0].norm(dim=1).mean().item()
    return tv, shift, disp


samples_raw = ancestral_sample(main["model"], sched, N_SAMPLE, generator=gen(5))
results["DDPM (raw weights)"] = evaluate(samples_raw)
print(f"{'':12s} {'MMD':>7s} {'SW':>6s} {'log q':>6s} {'blob-mass TV':>13s} {'blob shift':>11s}")
displacements = {}
for name, x in (("fresh data", gmm.sample(N_SAMPLE, generator=torch.Generator().manual_seed(3))),
                ("raw weights", samples_raw), ("EMA weights", samples_main)):
    m, (tv, shift, displacements[name]) = evaluate(x), blob_report(x)
    print(f"{name:12s} {m['mmd']:7.4f} {m['sw']:6.3f} {m['logq']:6.2f} {tv:13.3f} {shift:11.3f}")

ev = main["evals"]
MAG = 10                                                  # blob shifts are a few pixels: magnify the arrows
fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), gridspec_kw={"width_ratios": [1.3, 1, 1]})
axes[0].plot(ev[:, 0], ev[:, 1], color=C["raw"], label="raw weights")
axes[0].plot(ev[:, 0], ev[:, 2], color=C["eps"], label="EMA weights")
axes[0].axhline(bayes_floor, color=C["oracle"], ls="--", label="Bayes floor")
lo = bayes_floor - 0.01
axes[0].set(xlabel="training step", ylabel="held-out loss", title="Held-out loss during training",
            ylim=(lo, lo + max(0.08, 3 * (ev[-1, 1] - lo))))
axes[0].legend()
for ax, (name, x, c) in zip(axes[1:], (("raw weights", samples_raw, C["raw"]), ("EMA weights", samples_main, C["eps"]))):
    dm.scatter2d(x, ax=ax, c=c, alpha=0.25, title=f"{name}\n" + fmt(results[f"DDPM ({name})"]))
    d = displacements[name]
    ax.quiver(gmm.means[:, 0], gmm.means[:, 1], MAG * d[:, 0], MAG * d[:, 1], angles="xy", scale_units="xy",
              scale=1, color="k", width=0.012, label=f"blob-mean shift ×{MAG}")
    ax.plot(gmm.means[:, 0], gmm.means[:, 1], "k.", ms=4)
axes[2].legend(loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# The held-out loss of the raw weights keeps fluctuating, while the EMA curve is smooth and sits lower, close to the floor.
# The gap in loss looks small, yet the samples differ clearly (see the table). The raw-weight samples are *not* more
# diffuse (their log-density is about the same). Instead, whole blobs are displaced from their true centres (the arrows,
# magnified ten-fold, point from each true centre to the mean of the samples assigned to it) and the blob masses are off.
# The last few hundred minibatches leave a small *systematic* bias in $\epsilon_\theta$, and the
# sampler integrates it over $T=1000$ steps. (The raw weights are a single snapshot, at the last step; another step would
# displace the blobs differently.) **Always sample (and evaluate) with the EMA weights.** The
# averaging window must also be short compared with the training run. Large-scale runs use decays like 0.9999, which only
# make sense over hundreds of thousands of steps.
#
# ## 9 · Ablations
#
# Each ablation changes one knob of the pipeline above and keeps everything else fixed (same data, network, optimiser,
# number of steps and sampler noise). The helper below trains and evaluates one variant, so trying more is one line.

# %%
def run_variant(label, schedule=sched, pred="eps", data=train_data, density=gmm, reference=ref, seed=0,
                model_cls=ResMLPDenoiser, model_kw=None, **train_kw):
    """Train a model_cls(**model_kw) variant with train_ddpm(**train_kw); return its EMA model, samples and metrics."""
    dm.seed_everything(seed)
    out = train_ddpm(model_cls(**(model_kw or {})), data, schedule, pred=pred, seed=seed, **train_kw)
    x = ancestral_sample(out["ema"], schedule, N_SAMPLE, pred=pred, generator=gen(5))
    m = evaluate(x, reference, density)
    print(f"{label:28s} trained in {out['seconds']:4.0f} s | {fmt(m)}")
    return dict(model=out["ema"], samples=x, metrics=m, pred=pred, schedule=schedule)


# %% [markdown]
# ### 9a · What should the network predict? $\epsilon$, $x_0$ or $v$
#
# Given $x_t$ and $t$, knowing any one of $\epsilon$, $x_0$ or $v = \sqrt{\bar\alpha_t}\,\epsilon-\sqrt{1-\bar\alpha_t}\,x_0$ determines
# the others ($x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ is the second equation). With $a=\sqrt{\bar\alpha_t}$ and $b=\sqrt{1-\bar\alpha_t}$:
# $$\hat x_0 = \frac{x_t - b\,\hat\epsilon}{a},\qquad \hat\epsilon = \frac{x_t - a\,\hat x_0}{b},\qquad
# \hat\epsilon = b\,x_t + a\,\hat v,\qquad \hat x_0 = a\,x_t - b\,\hat v .$$
# So the three choices differ in only two ways.
# 1. **Implicit loss weighting.** Converting errors gives
#    $\|x_0-\hat x_0\|^2 = \frac{1}{\text{SNR}_t}\|\epsilon-\hat\epsilon\|^2$ and $\|v-\hat v\|^2 = \big(1+\frac{1}{\text{SNR}_t}\big)\|\epsilon-\hat\epsilon\|^2$,
#    with $\text{SNR}_t = \bar\alpha_t/(1-\bar\alpha_t)$. Relative to $\epsilon$-prediction, $x_0$-prediction down-weights the low-noise
#    steps by the SNR (by a factor of ~$10^4$ at $t=0$), and $v$-prediction is a compromise.
# 2. **Error amplification.** An $\epsilon$-error becomes an $x_0$-error multiplied by $b/a$, which explodes as $t\to T$. An
#    $x_0$-error becomes an $\epsilon$-error multiplied by $a/b$, which explodes as $t\to 0$. For $v$ the factors are $a\le 1$ and
#    $b\le 1$, so it never amplifies. This is why $v$ (Salimans & Ho 2022) is popular for distillation, few-step sampling and
#    high-resolution or zero-terminal-SNR schedules.
#    Every parameterisation of $\mu_\theta$ falls out of the ELBO, each with its own weight. $\epsilon$-prediction became the
#    DDPM default because its weight is simple and dropping it gives an unweighted MSE with a unit-scale target at every
#    $t$, because it links directly to denoising score matching ($\epsilon^*\propto$ the score, §4), and because it gave the
#    best samples in Ho et al.
#
# First a check: our conversions agree with `dm.model_predictions` for every parameterisation.

# %%
x_tmp = torch.randn(8, 2, device=device)
t_tmp = torch.randint(0, T, (8,), device=device)
for pred in ("eps", "x0", "v"):
    eps_ours, x0_ours = eps_x0_from_output(net(x_tmp, t_tmp), x_tmp, t_tmp, sched, pred)
    eps_dm, x0_dm = dm.model_predictions(net, x_tmp, t_tmp, sched, pred=pred)
    assert torch.allclose(eps_ours, eps_dm, atol=1e-4) and torch.allclose(x0_ours, x0_dm, atol=1e-4), pred
print("eps/x0/v conversions == dm.model_predictions  ✓")

print("Training the x0- and v-prediction variants (eps = the main model) ...")
variants = {"eps": dict(model=main["ema"], samples=samples_main, metrics=results["DDPM (EMA weights)"], pred="eps")}
for pred in ("x0", "v"):
    variants[pred] = run_variant(f"pred = {pred}", pred=pred)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for pred, var in variants.items():
    ts_eval, e_eps, e_x0 = per_t_errors(var["model"], sched, pred=pred)
    axes[0].semilogy(ts_eval, e_eps, "o-", ms=2.5, color=C[pred], label=f"{pred}-prediction")
    axes[1].semilogy(ts_eval, e_x0, "o-", ms=2.5, color=C[pred], label=f"{pred}-prediction")
_, o_eps, o_x0 = per_t_errors(oracle, sched)
for ax, o, space in ((axes[0], o_eps, r"$\epsilon$"), (axes[1], o_x0, r"$x_0$")):
    ax.semilogy(ts_eval, o, "k--", lw=1, label="oracle (Bayes floor)")
    ax.set(xlabel="t", ylabel=f"MSE in {space}-space (log)", title=f"Error of the implied {space}-prediction")
    ax.legend()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
for ax, (pred, var) in zip(axes, variants.items()):
    dm.scatter2d(var["samples"], ax=ax, c=C[pred], title=f"{pred}-prediction\n" + fmt(var["metrics"]))
plt.tight_layout()
plt.show()

# %% [markdown]
# Converted to a common currency, the three models follow the oracle over most of the range. They depart from it exactly
# where the amplification factors above predict.
# - **Small $t$, $x_0$-prediction.** Its implied $\epsilon$ (left) is far off the floor at the lowest noise levels, and even its
#   $x_0$-error (right) is clearly above the floor there. Its loss weights these steps by $1/\text{SNR}_t$, about $10^{-4}$ at
#   $t=0$, so it never learns that at very low noise the answer is "almost exactly $x_t$". In the last sampling steps it
#   keeps pulling points towards an over-smoothed guess, and the blobs collapse into thin arcs. The result has a
#   log-density far *above* the data's: too concentrated.
# - **Large $t$, $\epsilon$-prediction.** Its $x_0$-error climbs above the floor as $t\to T$, because an $\epsilon$-error is
#   multiplied by $\sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$ (the same effect as the scattered $t=999$ guesses in §6). This is
#   harmless for ancestral sampling, whose update uses $\hat\epsilon$ directly. It hurts whenever $\hat x_0$ itself is used:
#   clipping, guidance, few-step samplers.
#
# $v$-prediction tracks the floor in both spaces at every $t$, and its samples are as good as $\epsilon$-prediction's.
#
# ### 9b · How many steps? $T = 10, 50, 1000$
#
# §1 predicted that big reverse steps break the Gaussian assumption. Our oracle lets us test this *without any
# learning error*: we plug the exact $\mathbb E[\epsilon\mid x_t]$ into the same ancestral sampler with $T=10$, $50$ and $1000$
# steps of the same continuous process. Then we compare with networks trained for each $T$.

# %%
step_counts = (10, 50, 1000)
oracle_runs, trained_runs = {}, {}
for T_ in step_counts[:-1]:
    s_T = make_schedule(T_)
    oracle_T = OracleDenoiser(gmm, s_T)
    x = ancestral_sample(oracle_T, s_T, N_SAMPLE, generator=gen(5))
    oracle_runs[T_] = dict(model=oracle_T, samples=x, metrics=evaluate(x), schedule=s_T)
    print(f"oracle, T={T_:4d}: {fmt(oracle_runs[T_]['metrics'])}")
oracle_runs[T] = dict(model=oracle, samples=samples_oracle, metrics=results["oracle denoiser"], schedule=sched)  # §7
print(f"oracle, T={T:4d}: {fmt(oracle_runs[T]['metrics'])}   (sampled in §7)")
for T_ in step_counts[:-1]:
    trained_runs[T_] = run_variant(f"trained, T = {T_}", schedule=make_schedule(T_))
trained_runs[T] = dict(model=main["ema"], samples=samples_main, metrics=results["DDPM (EMA weights)"], schedule=sched)

fig, axes = plt.subplots(2, 3, figsize=(10, 6.8))
for j, T_ in enumerate(step_counts):
    dm.scatter2d(oracle_runs[T_]["samples"], ax=axes[0, j], c=C["oracle"],
                 title=f"oracle denoiser, T={T_}\n" + fmt(oracle_runs[T_]["metrics"]))
    dm.scatter2d(trained_runs[T_]["samples"], ax=axes[1, j], c=C["eps"],
                 title=f"trained network, T={T_}\n" + fmt(trained_runs[T_]["metrics"]))
plt.tight_layout()
plt.show()

# %% [markdown]
# Even the *perfect* denoiser produces poor samples with $T=10$: mass is smeared between and around the blobs, and the
# log-density drops far below the data's. The network trained for $T=10$ fails in the same way. The culprit is the
# sampler's Gaussian reverse step (the four-mode kernel in §1), not the learning. At $T=50$ the damage is much milder,
# but the oracle's blobs come out somewhat too *tight* (log-density above the data's), which points at the reverse
# variance (§9c). At $T=1000$ the oracle sits at the floor and the trained network close to it. Reducing the number of network evaluations without this failure is the subject of
# notebook 04 (DDIM and ODE solvers take larger *deterministic* steps) and of distillation / consistency methods.
#
# ### 9c · The sampler variance: $\sigma_t^2=\beta_t$ or $\tilde\beta_t$?
#
# Ho et al. used both and found similar results. To see what each choice assumes, split the variance of the true reverse
# step by conditioning on $x_0$ (law of total variance). Given $x_0$, the step is $q(x_{t-1}\mid x_t,x_0)$ of §1, with
# variance $\tilde\beta_t$ and a mean that depends on $x_0$ through the coefficient $c_t=\sqrt{\bar\alpha_{t-1}}\,\beta_t/(1-\bar\alpha_t)$:
# $$\mathrm{Cov}[x_{t-1}\mid x_t] = \underbrace{\tilde\beta_t\, I}_{\text{noise of the step, given }x_0}
# \;+\; c_t^2\,\underbrace{\mathrm{Cov}[x_0\mid x_t]}_{\text{uncertainty about }x_0} .$$
# - $\tilde\beta_t$ is exact when $x_0$ is *known* (a point mass): the second term vanishes.
# - $\beta_t$ is exact when $x_0\sim\mathcal N(0,I)$. For data with unit variance per coordinate, Ho et al. note that the two
#   are the extremes, lower and upper bounds on the reverse-process entropy. The best *single* variance, the same for every
#   $x_t$ and averaged over $x_t$, lies (approximately) between them. Bao et al. (2022, Analytic-DPM) give it in closed form, and Nichol &
#   Dhariwal (2021) *learn* an $x_t$-dependent interpolation.
# - **Pointwise**, though, the second term can be far larger. At an ambiguous $x_t$ between modes, $\mathrm{Cov}[x_0\mid x_t]$
#   is big, and the true reverse step is much wider than $\beta_t$: this is the $T=50$ and $T=10$ panels of §1 (variances
#   printed there). With small steps $c_t\to 0$ and the term is negligible. With large steps, no fixed isotropic variance
#   can be right everywhere.
#
# Swapping the variance is free: no retraining, just a sampler argument.

# %%
var_table = {}
for T_ in step_counts:
    for who, runs in (("oracle", oracle_runs), ("trained", trained_runs)):
        r = runs[T_]
        for var in ("beta_tilde", "beta"):
            if var == "beta_tilde":
                m = r["metrics"]                        # already computed with the default beta_tilde
            else:
                m = evaluate(ancestral_sample(r["model"], r["schedule"], N_SAMPLE, var=var, generator=gen(5)))
            var_table[(who, T_, var)] = m
            print(f"{who:8s} T={T_:4d} sigma^2={var:10s}: {fmt(m)}")

fig, ax = plt.subplots(figsize=(7, 3.3))
width, xpos = 0.2, np.arange(len(step_counts))
bars = [("oracle", "beta_tilde", C["oracle"], 0.45), ("oracle", "beta", C["oracle"], 1.0),
        ("trained", "beta_tilde", C["eps"], 0.45), ("trained", "beta", C["eps"], 1.0)]
for k, (who, var, c, alpha) in enumerate(bars):
    vals = [var_table[(who, T_, var)]["mmd"] for T_ in step_counts]
    lab = rf"{who}, $\sigma^2=$" + (r"$\tilde\beta_t$" if var == "beta_tilde" else r"$\beta_t$")
    ax.bar(xpos + (k - 1.5) * width, vals, width, color=c, alpha=alpha, label=lab)
ax.axhspan(0, floor["mmd"][0] + floor["mmd"][1], color="0.85", zorder=0, label="data-vs-data floor")
ax.set_yscale("log")
ax.set(xticks=xpos, xticklabels=[f"T = {T_}" for T_ in step_counts], ylabel="MMD (log)",
       title="Sampler variance choice vs number of steps")
ax.legend(ncol=2)
plt.tight_layout()
plt.show()

# %% [markdown]
# At $T=1000$ the two variances are indistinguishable, which reproduces Ho et al.'s observation. With few, large steps the
# choice matters, and by MMD the larger $\beta_t$ wins: $\tilde\beta_t$ ignores the uncertainty about $x_0$. The printed
# log-densities tell a more nuanced story. $\beta_t$ always gives the lower log-density: it spreads mass more, which helps
# coverage (MMD) but also puts more samples in low-density regions. For the oracle at $T=50$ the two choices bracket the
# data: $\tilde\beta_t$ gives samples that are too concentrated (log-density above the data's) and $\beta_t$ samples that
# are slightly too diffuse (below it), as the averaged theory above predicts. For the trained network the metrics can
# disagree about which choice is better. And no variance rescues $T=10$: a Gaussian with the right mean and a better
# variance is still a unimodal Gaussian, while the true step is multimodal.
#
# ## 10 · The same pipeline on other toy datasets
#
# Nothing above was specific to Gaussian mixtures. Here is the unchanged recipe (same network, steps, optimiser and
# sampler) on four classic 2-D datasets. We have not coded up their densities (eight_gaussians *is* a mixture; turning it
# into a `dm.GaussianMixture` is a nice extension), so we report MMD and sliced Wasserstein with a data-vs-data floor.

# %%
toy_names = ["moons", "eight_gaussians", "checkerboard", "swiss_roll"]
toy_runs = {}
for name in toy_names:
    data_toy = dm.make_toy(name, 20_000, seed=0).to(device)
    ref_toy = dm.make_toy(name, N_SAMPLE, seed=1)
    floor_toy = evaluate(dm.make_toy(name, N_SAMPLE, seed=2), ref_toy, None)
    toy_runs[name] = run_variant(name, data=data_toy, density=None, reference=ref_toy)
    toy_runs[name].update(data=ref_toy, floor=floor_toy)
    print(f"{'':28s} floor (data vs data)  | {fmt(floor_toy)}")

fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
for j, name in enumerate(toy_names):
    r = toy_runs[name]
    dm.scatter2d(r["data"], ax=axes[0, j], c=C["data"], lim=3.2, title=f"{name}: data")
    dm.scatter2d(r["samples"], ax=axes[1, j], c=C["eps"], lim=3.2,
                 title=f"DDPM samples\nMMD {r['metrics']['mmd']:.4f} (floor {r['floor']['mmd']:.4f})")
plt.tight_layout()
plt.show()

# %% [markdown]
# The same ~20 lines of training code learn disconnected modes (eight Gaussians) and thin curved manifolds (moons, swiss
# roll), up to a faint haze of points between modes. The **checkerboard** is the exception. Its sharp-edged squares come
# out as blurred blobs, with mass leaking into the empty squares. Note that its MMD is clearly above the floor, but by much
# less than the picture suggests: a summary metric can understate a defect that is obvious to the eye.
#
# ### Fixing the checkerboard: Fourier features of the input
# Why does the checkerboard fail? At small $t$ the ideal denoiser must change abruptly across every edge of every square:
# $\epsilon^*(x_t,t)$ varies over distances of the order of the noise level. An MLP fed raw coordinates is biased towards smooth,
# low-frequency functions of its input (the "spectral bias" of neural networks), so it learns such edges slowly. More
# steps, bigger batches or a learning-rate schedule help only a little. The remedy is the one §3a used for $t$: hand the
# network a ready-made multi-scale basis of its input, $\big[x,\ \sin(2^k\pi x),\ \cos(2^k\pi x)\big]_{k=0}^{K-1}$ (Tancik et al. 2020).
# With $K=4$ octaves the fastest feature has a period of a quarter of a square. Everything else (loss, optimiser, number
# of steps, sampler) stays the same.

# %%
data_cb = dm.make_toy("checkerboard", 20_000, seed=0).to(device)
cb_plain = toy_runs["checkerboard"]
cb_fourier = run_variant("checkerboard + Fourier(x)", data=data_cb, density=None, reference=cb_plain["data"],
                         model_kw=dict(fourier_octaves=4))
print(f"{'':28s} floor (data vs data)  | {fmt(cb_plain['floor'])}")


def draw_checkerboard(ax):
    """Outline the 8 occupied unit squares of dm.make_toy('checkerboard')."""
    for col in range(-2, 2):                       # column [col, col+1); squares start at height off and off - 2
        off = col % 2
        for low in (off, off - 2):
            ax.add_patch(plt.Rectangle((col, low), 1, 1, fill=False, ec="k", lw=0.8))


fig, axes = plt.subplots(1, 2, figsize=(8, 4.1))
for ax, (r, title) in zip(axes, ((cb_plain, "plain residual MLP"), (cb_fourier, "+ Fourier features of $x$ (4 octaves)"))):
    dm.scatter2d(r["samples"], ax=ax, c=C["eps"], lim=2.6, s=3, title=f"{title}\n{fmt(r['metrics'])}")
    draw_checkerboard(ax)
plt.suptitle(f"Checkerboard samples, same budget (black: the true squares; floor: {fmt(cb_plain['floor'])})", y=1.0)
plt.tight_layout()
plt.show()

# %% [markdown]
# With Fourier features the squares come back with sharp edges and the empty squares stay nearly empty, at the same budget
# and with about two thousand extra parameters. Both metrics move towards the floor. The lesson generalises:
# when a network struggles with fine structure in a low-dimensional input, look at the input *representation* before
# buying more compute. Fourier/positional features are standard wherever networks read coordinates, from neural fields to
# the time embedding that every diffusion model uses.
#
# ## Exercises
#
# **Exercise 1 (schedule).** Replace `sched` with `dm.NoiseSchedule(1000, "cosine", device=device)` (an `OracleDenoiser`
# for it is one line) and retrain. Predict first: where along $t$ does the per-$t$ loss curve of §5 change, and do the samples improve?
# <details><summary>Hint / solution</summary>
# The linear schedule reaches near-pure noise long before $t=T$ (its last few hundred steps carry almost no signal), while
# the cosine schedule spends more of its steps at intermediate noise levels. Plot `per_t_errors` for both models and both
# oracles against $\log\text{SNR}_t$ (`sched.log_snr`) instead of $t$. The oracle curves then coincide exactly, because the
# optimal error depends only on the noise level, and the trained models' curves nearly coincide too. A schedule mainly
# decides *which* noise levels are visited, and how often (Kingma et al. 2021). Nichol & Dhariwal found the cosine schedule
# helps most for low-resolution images, where the linear one wastes many steps on near-pure noise.
# </details>
#
# **Exercise 2 (is it the model or the sampler?).** Sample the main $T=1000$ model with only 10 of its steps: use
# `s10 = make_schedule(10)` and wrap the model so that index `t` of the 10-step chain is fed to the network as
# time `(t + 1) * 100 - 1`. Predict: better or worse than the network trained with $T=10$?
# <details><summary>Hint / solution</summary>
#
# ```python
# class Respaced(nn.Module):
#     def __init__(self, m, stride): super().__init__(); self.m, self.stride = m, stride
#     def forward(self, x, t, y=None): return self.m(x, (t + 1) * self.stride - 1)
# x = ancestral_sample(Respaced(main["ema"], 100), make_schedule(10), N_SAMPLE, generator=gen(5))
# print(fmt(evaluate(x)))
# ```
# About as bad as the $T{=}10$-trained network. The denoiser is fine (it is accurate at those 10 noise levels); the
# Gaussian step is the bottleneck. Deterministic samplers (notebook 04) use exactly this respacing trick far more successfully.
# </details>
#
# **Exercise 3 (the weighting).** Modify `train_ddpm` to weight each example's loss by the true-ELBO weight $\lambda_t$ of §2
# (normalised to mean 1). Predict how the per-$t$ error curve and the samples change.
# <details><summary>Hint / solution</summary>
# Build the weights once, on the compute device (indexing a NumPy array with the device tensor `t` fails on a GPU):
# `lam = torch.tensor(lam_beta / lam_beta.mean(), dtype=torch.float32, device=device)`. Then replace the loss with
# `(lam[t] * ((model(x_t, t) - target) ** 2).mean(1)).mean()`. `lam_beta[0]` is finite (it is exactly the $L_0$ weight
# of a Gaussian decoder with $\sigma_1^2=\beta_1$); only the $\tilde\beta_t$ version is infinite at $t=0$ and needs that entry
# patched. The error at the lowest-noise steps goes down slightly, and the error at most other $t$ goes up. The
# gradient noise also grows, because a few examples dominate each batch. The samples get somewhat worse (more diffuse). This is the
# likelihood vs sample-quality trade-off of Ho et al., Table 2.
# </details>
#
# **Exercise 4 (EMA horizon).** Retrain the main model (`run_variant("ema", ema_decay=...)`) with `ema_decay=0.99`, with
# `ema_decay=0.9999`, and with `ema_decay=0.9999, ema_warmup=False`. Predict each result before you run it: which is
# closest to the raw weights, which is identical to the main model, and which is ruined?
# <details><summary>Hint / solution</summary>
# 0.99 averages over ~100 steps and lands between the raw and the default EMA weights. 0.9999 gives *exactly* the main
# model's EMA: the warm-up rule $d=\min(\text{decay}, \frac{1+n}{10+n})$ stays below 0.999 for the first ~9000 steps, so
# the `decay` argument has no effect in a 4000-step run (the effective decay reaches only about 0.998, a window of a
# few hundred steps). Without warm-up, 0.9999 is a disaster: after 4000 steps the average still carries
# $0.9999^{4000}\approx e^{-0.4}\approx 67\%$ of the *random initial* weights, and the samples show it. Large-scale runs can
# use 0.9999 only because they train for hundreds of thousands of steps.
# </details>
#
# **Exercise 5 (memorisation).** Train on only `train_data[:32]` for 4000 steps and sample. Measure the distance from each
# sample to its nearest training point (`torch.cdist(...).min(1)`) and compare with fresh data. What has the model learned?
# <details><summary>Hint / solution</summary>
# With 32 points the empirical distribution *is* the optimum of the loss. The ideal denoiser of a finite training set is
# the posterior mean over those 32 points, and sampling with it reproduces the training points exactly. After 4000 steps
# the samples sit clearly closer to the training points than fresh data do (compare the median nearest-neighbour
# distances), and they move closer still the longer you train. Real diffusion models generalise because the network
# *fails* to reach this optimum in a useful way (Kadkhodaie et al. 2024; notebook 03, §9 studies the exact score of a
# training set).
# </details>
#
# **Exercise 6 (architecture knobs).** Two predictions to test with `run_variant`. (a) The gotcha of §3a: feed the
# embedding $t/T$ instead of $t$. Does it hurt *here*? (b) Shrink the network to `hidden=32, n_blocks=1`. At which $t$
# does a small network lose most, relative to the oracle?
# <details><summary>Hint / solution</summary>
#
# ```python
# class TimeRescaled(ResMLPDenoiser):
#     def forward(self, x, t, y=None): return super().forward(x, t / T)     # float t/T in [0, 1)
# rescaled = run_variant("t/T embedding", model_cls=TimeRescaled)
# small = run_variant("small net", model_kw=dict(hidden=32, n_blocks=1))
# plt.plot(ts_eval, per_t_errors(small["model"], sched)[1] - err_oracle)
# ```
# (a) Indistinguishable from the main model, within run-to-run noise. Here $\epsilon^*$ varies smoothly with $t$, so the
# few smooth features that survive (§3a, right panel) carry all the $t$-information this problem needs. The gotcha bites
# when the ideal denoiser must change sharply with $t$.
# (b) Expect the gap to grow where the optimal $\epsilon^*(x_t,t)$ varies most sharply with
# $x_t$, at small and intermediate $t$ (the edges of the blobs, and the boundaries between blobs). At large $t$ the
# target is almost linear ($\epsilon^*\approx x_t$), which any network can represent.
# </details>

# %% [markdown]
# ## Key takeaways
#
# - DDPM's reverse step $p_\theta(x_{t-1}\mid x_t)$ is Gaussian. That is accurate only for **small steps**. With a perfect
#   noise predictor, the *mean* of DDPM's step equals the true reverse mean for any step size (Tweedie); its *shape* and
#   *variance* are only right for small steps.
# - The ELBO is a sum of KLs between Gaussians. With $\mu_\theta$ written in terms of a noise predictor, each term becomes a
#   weighted $\|\epsilon-\epsilon_\theta\|^2$. Dropping the weights gives $L_\text{simple}$, a reweighting that changes how capacity is
#   spent across $t$ but not the optimum, $\mathbb E[\epsilon\mid x_t]$.
# - The loss cannot reach 0: its floor is the posterior variance of the noise. Judge training against that floor and with a
#   held-out batch, not against zero and not from single minibatches.
# - The whole method is small: sinusoidal $t$-embedding + residual MLP + a 20-line loop + a 15-line sampler. Implementation
#   bugs hide in indexing conventions and broadcasting, so check against a reference.
# - The model's $\hat x_0$ is a posterior *mean*, a belief, not a sample. It is a martingale along the reverse process, and
#   continuing the sampler from $x_t$ draws from the posterior over $x_0$.
# - Sample with **EMA weights**. Score samples with MMD / sliced Wasserstein *relative to a data-vs-data floor*, and, where
#   you can, against an oracle that shares the sampler, to separate network error from sampler error.
# - $\epsilon$-, $x_0$- and $v$-prediction are the same model with different implicit weightings and error amplification. $x_0$
#   neglects low noise levels, while $v$ never amplifies errors.
# - Few steps fail even with a perfect denoiser. The variance choice $\beta_t$ vs $\tilde\beta_t$ only matters when steps are
#   large, and then no fixed variance is right everywhere.
# - When an MLP misses sharp structure in a low-dimensional input, change its input representation (Fourier features)
#   before buying more compute.
#
# ## Further reading
#
# - J. Ho, A. Jain, P. Abbeel (2020). *Denoising Diffusion Probabilistic Models.* arXiv:2006.11239. The algorithms implemented here.
# - J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, S. Ganguli (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics.* arXiv:1503.03585. The original diffusion model and the small-step Gaussian argument.
# - A. Nichol, P. Dhariwal (2021). *Improved Denoising Diffusion Probabilistic Models.* arXiv:2102.09672. Cosine schedule, learned variances, fewer steps.
# - T. Salimans, J. Ho (2022). *Progressive Distillation for Fast Sampling of Diffusion Models.* arXiv:2202.00512. $v$-prediction.
# - D. Kingma, T. Salimans, B. Poole, J. Ho (2021). *Variational Diffusion Models.* arXiv:2107.00630. The SNR view of the ELBO and its weightings.
# - T. Karras, M. Aittala, T. Aila, S. Laine (2022). *Elucidating the Design Space of Diffusion-Based Generative Models.* arXiv:2206.00364. Parameterisation, preconditioning and loss weighting as explicit design choices.
# - F. Bao, C. Li, J. Zhu, B. Zhang (2022). *Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models.* arXiv:2201.06503. The best fixed reverse variance of §9c in closed form.
# - B. Efron (2011). *Tweedie's Formula and Selection Bias.* Journal of the American Statistical Association. The posterior-mean identity behind the oracle.
# - M. Tancik, P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ramamoorthi, J. Barron, R. Ng (2020). *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains.* arXiv:2006.10739. Spectral bias and the input features of §10.
# - C. Meng, Y. He, Y. Song, J. Song, J. Wu, J.-Y. Zhu, S. Ermon (2022). *SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations.* arXiv:2108.01073. Noise partway, then resample (§6).
# - Z. Kadkhodaie, F. Guth, E. P. Simoncelli, S. Mallat (2024). *Generalization in diffusion models arises from geometry-adaptive harmonic representations.* arXiv:2310.02557. Memorisation vs generalisation (Exercise 5).
# - L. Weng (2021). *What are Diffusion Models?* lilianweng.github.io. A careful walk-through of the same derivation.
