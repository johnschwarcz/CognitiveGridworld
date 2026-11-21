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
# # 07 · Using a trained diffusion model: inpainting, editing and inverse problems
#
# A trained diffusion model is much more than a sampler. It is a learned **prior** over images that can be queried
# and steered *at sampling time*, with no retraining. In this notebook we take the class-conditional MNIST model of
# notebooks 05–06 and use it to fill in missing pixels, turn crude sketches into digits, super-resolve, denoise and
# edit images. Each time we combine the same prior with a different kind of measurement. A 2-D Gaussian mixture,
# whose posteriors we know exactly, tells us which methods are exact and which are approximations.
#
# **You will learn:**
# - what a trained model gives you (samples, the denoiser $\mathbb E[x_0\mid x_t]$, scores), and the Bayesian recipe
#   *prior × likelihood → posterior* behind every application
# - **inpainting**: the simple *replacement* trick, why it is biased even with a perfect model, and how
#   **RePaint**'s resampling reduces (but does not remove) that bias, at the cost of many more network evaluations
# - **SDEdit**: image-to-image editing by "noise, then denoise", and its realism–faithfulness knob
# - **Diffusion Posterior Sampling (DPS)** for linear inverse problems (super-resolution, sparse pixels): its key
#   approximation $p(y\mid x_t)\approx p(y\mid\hat x_0(x_t))$ and its step size
# - **denoising** with Tweedie's formula or with the reverse process, and the perception–distortion trade-off
# - **DDIM inversion** for reconstruction, interpolation (and why *slerp*) and class swaps that keep the style
#
# **Prerequisites:** 01–04 (forward process, DDPM, scores and Tweedie's formula, DDIM), 05 (U-Net) and 06
# (class-conditional model, classifier-free guidance).
#
# **Runtime:** CPU ≈ 4–10 min (sampling only, no training; measured on a 4-core CPU, depending on its load); CUDA GPU
# ≈ 1–2 min. If no conditional MNIST checkpoint is found, `dmtools` first trains one (a few minutes on a GPU, much
# longer on a CPU).

# %%
import copy
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import dmtools as dm

device = dm.get_device(verbose=True)
FAST = dm.fast_mode()  # automated smoke-test switch, leave as is
dm.seed_everything(0)
plt.rcParams.update({"figure.dpi": 100, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})


def make_gen(seed, dev=None):
    """A seeded random generator (on `device` unless told otherwise), so every experiment is reproducible."""
    return torch.Generator(device=dev or device).manual_seed(seed)


# Every expensive quantity is scaled by FAST (FAST only checks that the code runs, not that it works well).
N_TOY = 400 if FAST else 4000     # particles in the 2-D experiments
TOY_STEPS = 20 if FAST else 100   # reverse steps in the 2-D experiments
STEPS = 10 if FAST else 50        # reverse steps for MNIST (inpainting, SDEdit, DDIM inversion)
DPS_STEPS = 10 if FAST else 100   # DPS needs more, smaller steps (section 4)
N_IMG = 2 if FAST else 6          # test digits per experiment (and per figure)
N_EVAL = 2 if FAST else 10        # test digits per mask for the inpainting scores (section 2.4)
N_POST = 2 if FAST else 4         # posterior samples per measurement

# one colour per method, used in every figure
COL = {"exact": "black", "replacement": "tab:red", "RePaint": "tab:blue", "DPS": "tab:green",
       "baseline": "tab:gray", "Tweedie": "tab:purple", "sample": "tab:orange"}

# %% [markdown]
# ## 1 · A trained diffusion model is a prior you can query
#
# Training gave us one network, $\epsilon_\theta(x_t,t,y)$. Three useful objects come out of it:
#
# 1. **Samples** $x_0\sim p_\theta(x_0)$, or $p_\theta(x_0\mid c)$ for a class $c$, by running a reverse process
#    (notebooks 02 and 04).
# 2. **A denoiser.** By Tweedie's formula (notebook 03) the clean-image estimate
#    $$\hat x_0(x_t)=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}\;\approx\;\mathbb E[x_0\mid x_t]$$
#    is the posterior mean (the MMSE estimate) of the clean image given its noisy version.
# 3. **Scores** of every noised marginal: $\nabla_{x_t}\log p_t(x_t)\approx-\epsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}$.
#
# These pieces can be recombined *at inference time*. Every application in this notebook has the same structure. We
# observe a measurement $y$ of an unknown image $x_0$ and want the posterior
# $$p(x_0\mid y)\;\propto\;\underbrace{p(y\mid x_0)}_{\text{likelihood: what we measured}}\;\underbrace{p(x_0)}_{\text{prior: the diffusion model}} .$$
# A diffusion sampler for the posterior needs the posterior score at every noise level. Bayes' rule for scores
# (notebook 06) splits it into two terms:
# $$\nabla_{x_t}\log p_t(x_t\mid y)=\underbrace{\nabla_{x_t}\log p_t(x_t)}_{\text{the network}}+\nabla_{x_t}\log p_t(y\mid x_t),
# \qquad p_t(y\mid x_t)=\int p(y\mid x_0)\,p(x_0\mid x_t)\,dx_0 .$$
# The second term is the hard part. It averages the likelihood over the *denoising posterior* $p(x_0\mid x_t)$, which
# we never have in closed form. The methods below differ in how they deal with it:
#
# | task | measurement $y$ | method | how it handles $p_t(y\mid x_t)$ |
# |---|---|---|---|
# | inpainting | the known pixels, noise-free | replacement, RePaint | side-steps it: clamp the known pixels at the current noise level (+ resampling) |
# | image-to-image editing | a rough guide image | SDEdit | no explicit likelihood: start the reverse process from the *noised* guide |
# | linear inverse problems | $y=Ax_0+n$ | DPS | replaces $p(x_0\mid x_t)$ by a point mass at $\hat x_0(x_t)$ |
# | denoising | $y=x_0+\sigma n$ | Tweedie, or reverse from $t^\star$ | exact: a rescaled $y$ *is* a sample of $x_{t^\star}$ |
# | latent editing | none | DDIM inversion | uses the deterministic, invertible noise ↔ image map |
#
# In the language of Bayesian observers: the diffusion model is a very rich prior learned from data, each task
# supplies its own likelihood, and the prior is never retrained.
#
# We use the class-conditional MNIST U-Net of notebook 06 (recipe `dm.MNIST_RECIPE`, linear schedule with $T=1000$,
# $\epsilon$-prediction, 10% label dropout). With `y=None` it is the *unconditional* model. The small MNIST classifier
# of notebook 05 serves only as an **evaluator**: it never influences sampling. All inputs come from the test set.

# %%
REF_CKPT = dm.CKPT_DIR / "mnist_unet_cond.pt"   # reference model (12k steps), made by pretrain_checkpoints.py
ALT_CKPT = dm.CKPT_DIR / "06_mnist_cond.pt"     # same recipe, fewer steps, saved while writing notebook 06
if REF_CKPT.exists() or device.type == "cuda" or not ALT_CKPT.exists():
    # loads the reference checkpoint, or trains and caches it if missing (minutes on a GPU, long on a CPU)
    model, schedule = dm.load_or_train_mnist_model(conditional=True, device=device)
else:
    ck = torch.load(ALT_CKPT, map_location=device, weights_only=True)
    model = dm.UNet(**ck["config"]).to(device)
    model.load_state_dict(ck["ema"])
    schedule = dm.NoiseSchedule(1000, "linear", device=device)
    print(f"{REF_CKPT.name} not found: using {ALT_CKPT.name} (same recipe, {ck['steps']} training steps)")
model.eval().requires_grad_(False)   # no training here; DPS differentiates w.r.t. the *input*, never the weights
T = schedule.T
print(f"U-Net with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, "
      f"{model.n_classes} classes (+ null label), T = {T}")

x_test, y_test = dm.load_mnist(train=False)
clf = dm.load_or_train_classifier(device=device).requires_grad_(False)


@torch.no_grad()
def classify(x):
    """Labels predicted by the evaluation classifier for images in [-1, 1] (returned on the CPU)."""
    return clf(x.clamp(-1, 1).to(device)).argmax(1).cpu()


def psnr(x, ref):
    """Peak signal-to-noise ratio in dB for images in [-1, 1] (peak-to-peak range 2)."""
    mse = ((x.float().cpu() - ref.float().cpu()) ** 2).mean()
    return (10 * torch.log10(4.0 / mse)).item()


def image_grid(rows, row_labels=None, col_labels=None, ax=None, title=None, size=0.62):
    """Draw rows of equally sized images as one labelled canvas.

    rows: list of tensors (n, C, H, W); grey images (C=1) are in [-1, 1], colour images (C=3) in [0, 1].
    """
    rgb = []
    for r in rows:
        r = r.detach().float().cpu()
        rgb.append(((r + 1) / 2).clamp(0, 1).repeat(1, 3, 1, 1) if r.shape[1] == 1 else r.clamp(0, 1))
    H, W, pad = rgb[0].shape[2], rgb[0].shape[3], 2
    n_cols = max(r.shape[0] for r in rgb)
    canvas = torch.ones(3, len(rgb) * (H + pad) + pad, n_cols * (W + pad) + pad)
    for i, r in enumerate(rgb):
        for j in range(r.shape[0]):
            canvas[:, pad + i * (H + pad):pad + i * (H + pad) + H, pad + j * (W + pad):pad + j * (W + pad) + W] = r[j]
    if ax is None:
        _, ax = plt.subplots(figsize=(size * n_cols + (1.2 if row_labels else 0.2), size * len(rgb) + 0.5))
    ax.imshow(canvas.permute(1, 2, 0).numpy(), interpolation="nearest")
    ax.set_xticks([pad + j * (W + pad) + W / 2 for j in range(n_cols)] if col_labels else [])
    ax.set_yticks([pad + i * (H + pad) + H / 2 for i in range(len(rgb))] if row_labels else [])
    if col_labels:
        ax.set_xticklabels(col_labels)
        ax.xaxis.tick_top()
    if row_labels:
        ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title)
    return ax


def heat_rgb(v, vmax):
    """Map a non-negative map (B, 1, H, W) to RGB with the 'magma' colormap (for uncertainty maps)."""
    v = (v.detach().float().cpu()[:, 0] / vmax).clamp(0, 1).numpy()
    return torch.as_tensor(plt.cm.magma(v)[..., :3]).permute(0, 3, 1, 2).float()


# %% [markdown]
# ### 1.1 Building blocks
#
# Every sampler in this notebook is a small variation of one update: the DDIM step of notebook 04 with
# stochasticity $\eta$, which moves from time $t$ to an earlier time $t'$ given the predictions $(\hat x_0,\hat\epsilon)$:
# $$x_{t'}=\sqrt{\bar\alpha_{t'}}\,\hat x_0+\sqrt{1-\bar\alpha_{t'}-s^2}\;\hat\epsilon+s\,z,\qquad
# s=\eta\sqrt{\frac{1-\bar\alpha_{t'}}{1-\bar\alpha_t}\Big(1-\frac{\bar\alpha_t}{\bar\alpha_{t'}}\Big)},\quad z\sim\mathcal N(0,I).$$
# With $\eta=1$ and $t'=t-1$ this is exactly DDPM's ancestral step. With $\eta=0$ it is deterministic. We also need
# the forward process between two times, $q(x_{t'}\mid x_t)=\mathcal N\big(\sqrt{\bar\alpha_{t'}/\bar\alpha_t}\,x_t,\;
# (1-\bar\alpha_{t'}/\bar\alpha_t)\,I\big)$ for $t'>t$, which "re-noises" a partially denoised image.
#
# **Gotcha: long strides under-disperse.** For $t'<t-1$ the $\eta=1$ step with a perfect denoiser still has the
# correct *mean* $\mathbb E[x_{t'}\mid x_t]$, because that mean is linear in $x_0$ and $\hat x_0=\mathbb E[x_0\mid x_t]$.
# But its variance $s^2$ is the variance of $q(x_{t'}\mid x_t,x_0)$ for a *known* $x_0$. The true reverse kernel
# $p(x_{t'}\mid x_t)=\int q(x_{t'}\mid x_t,x_0)\,p(x_0\mid x_t)\,dx_0$ also carries the uncertainty about $x_0$, an
# extra $c^2\,\mathrm{Cov}[x_0\mid x_t]$, where $c$ is the step's coefficient on $\hat x_0$. The missing term
# vanishes as the steps get finer, but with a few dozen strided steps the samples are too concentrated *even with a
# perfect denoiser* (Analytic-DPM, Bao et al. 2022, derives the optimal variance). We measure this in section 5.
#
# **Implementation detail.** Time is an integer $t\in\{0,\dots,T-1\}$, and we use $t=-1$ for clean data
# ($\bar\alpha_{-1}=1$). This removes the special cases at the end of the chain that cause many off-by-one bugs.
# `predict` is written without `torch.no_grad()` (unlike `dm.model_predictions`), because DPS must differentiate
# through it.

# %%
def abar_at(schedule, t):
    """alpha_bar_t, with the convention alpha_bar_{-1} = 1 ("time -1" is clean data)."""
    return schedule.alpha_bars[t] if t >= 0 else torch.ones((), device=schedule.device)


def predict(model, schedule, x, t, y=None, w=1.0):
    """(eps_hat, x0_hat) at integer time t, with optional classifier-free guidance of scale w (notebook 06)."""
    tt = torch.full((x.shape[0],), int(t), device=x.device, dtype=torch.long)
    if y is not None and w != 1.0:
        null = torch.full_like(y, model.n_classes)
        eps_c, eps_u = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([y, null])).chunk(2)
        eps = eps_u + w * (eps_c - eps_u)
    else:
        eps = model(x, tt, y)
    ab = schedule.alpha_bars[t]
    x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()       # Tweedie: x0_hat = E[x0 | x_t]
    return eps, x0


def ddim_step(schedule, x0, eps, t, t_next, eta=1.0, generator=None):
    """Move from time t to an earlier time t_next (-1 = clean data) given the predictions (x0_hat, eps_hat)."""
    ab, ab_next = abar_at(schedule, t), abar_at(schedule, t_next)
    s = eta * ((1 - ab_next) / (1 - ab) * (1 - ab / ab_next)).sqrt()
    x = ab_next.sqrt() * x0 + (1 - ab_next - s ** 2).clamp(min=0).sqrt() * eps
    if eta > 0 and t_next >= 0:
        x = x + s * torch.randn(x.shape, device=x.device, generator=generator)
    return x


def forward_jump(schedule, x, t, t_next, generator=None):
    """Sample q(x_{t_next} | x_t) for t_next > t: add exactly the noise the forward process adds in between."""
    a = abar_at(schedule, t_next) / abar_at(schedule, t)
    return a.sqrt() * x + (1 - a).sqrt() * torch.randn(x.shape, device=x.device, generator=generator)


def timesteps(t_start, n_steps):
    """About n_steps evenly spaced integer times from t_start down to 0 (descending)."""
    return [int(t) for t in np.unique(np.linspace(0, t_start, n_steps).round().astype(int))[::-1]]


@torch.no_grad()
def reverse_process(model, schedule, x, ts, y=None, w=1.0, eta=1.0, generator=None):
    """Denoise x, which sits at time ts[0], along the descending times ts and on to clean data."""
    for i, t in enumerate(ts):
        t_next = ts[i + 1] if i + 1 < len(ts) else -1
        eps, x0 = predict(model, schedule, x, t, y, w)
        x = ddim_step(schedule, x0, eps, t, t_next, eta, generator)
    return x


# %% [markdown]
# **Check.** Our blocks must reproduce `dmtools` (notebooks 03–04): the same predictions, with and without guidance,
# and the same deterministic ($\eta=0$) and stochastic ($\eta=1$) samplers when given the same random numbers.

# %%
n_chk = 5 if FAST else 10                 # sampler steps for the check
y_chk = torch.arange(6, device=device)
x_T_chk = torch.randn(6, 1, 28, 28, device=device, generator=make_gen(0))
with torch.no_grad():
    eps_ours, x0_ours = predict(model, schedule, x_T_chk, 500, y_chk, w=2.0)
eps_ref, x0_ref = dm.model_predictions(model, x_T_chk, 500, schedule, y=y_chk, guidance_scale=2.0)
checks = {
    "predict (eps, CFG w=2)": (eps_ours, eps_ref),
    "predict (x0_hat, CFG w=2)": (x0_ours, x0_ref),
    "sampler, eta=0 (DDIM)": (
        reverse_process(model, schedule, x_T_chk, timesteps(T - 1, n_chk), y=y_chk, eta=0.0),
        dm.ddim_sample(model, schedule, x_T_chk.shape, n_steps=n_chk, y=y_chk, x_T=x_T_chk)),
    "sampler, eta=1 (ancestral)": (
        reverse_process(model, schedule, x_T_chk, timesteps(T - 1, n_chk), y=y_chk, eta=1.0, generator=make_gen(1)),
        dm.ddim_sample(model, schedule, x_T_chk.shape, n_steps=n_chk, eta=1.0, y=y_chk, x_T=x_T_chk,
                       generator=make_gen(1))),
}
for name, (a, b) in checks.items():
    print(f"{name:<28s} max |ours - dmtools| = {(a - b).abs().max().item():.1e}")
    assert torch.allclose(a, b, atol=1e-3), name

# %% [markdown]
# The denoiser is the object most of this notebook builds on, so let us look at it. Below, one test digit is noised to
# increasing $t$ (top) and passed through the unconditional denoiser (bottom).

# %%
x_demo = x_test[:1].to(device)
t_show = [50, 150, 300, 450, 600, 800]
noise = torch.randn(x_demo.shape, device=device, generator=make_gen(2))
x_ts = torch.cat([schedule.q_sample(x_demo, torch.tensor([t], device=device), noise) for t in t_show])
with torch.no_grad():
    x0_hats = torch.cat([predict(model, schedule, x_ts[i:i + 1], t)[1] for i, t in enumerate(t_show)])
image_grid([x_ts, x0_hats], row_labels=["$x_t$", r"$\hat x_0(x_t)$"], size=0.9,
           col_labels=[f"t={t}\nSNR={schedule.snr(t).item():.2g}" for t in t_show],
           title=r"The denoiser $\hat x_0(x_t)\approx\mathbb{E}[x_0\mid x_t]$ of one test digit")
plt.tight_layout()
plt.show()

# %% [markdown]
# At small $t$ the estimate is essentially the digit. At large $t$, $\hat x_0$ is a blurry, grey superposition: the
# *average* of all digits that could have produced this $x_t$. It is a posterior mean, not a sample. Keep this
# picture in mind: DPS (section 4) uses this blurry estimate as a stand-in for the clean image.

# %% [markdown]
# ## 2 · Inpainting
#
# A binary mask $m$ marks the known pixels ($m=1$) and the missing ones ($m=0$). We observe $y=m\odot x_0$ exactly
# and want samples of the missing pixels given the known ones, $x_0^{u}\sim p(x_0^{u}\mid x_0^{k}=y)$.
#
# ### 2.1 The replacement method
#
# The simplest idea (used for imputation by Song et al. 2021) runs the ordinary reverse process and, after every step,
# overwrites the known pixels with a forward-noised copy of the observation *at the new noise level*:
# $$x^{u}_{t-1}\sim p_\theta(x_{t-1}\mid x_t)\ \text{(keep its unknown pixels)},\qquad
# x^{k}_{t-1}\sim q(x_{t-1}\mid x_0=y)=\mathcal N\big(\sqrt{\bar\alpha_{t-1}}\,y,\,(1-\bar\alpha_{t-1})I\big),$$
# $$x_{t-1}=m\odot x^{k}_{t-1}+(1-m)\odot x^{u}_{t-1}.$$
# The known pixels always have exactly the right statistics, and the network "sees" them while it denoises the rest.
#
# **Why it is not exact.** The unknown pixels should follow reverse dynamics driven by the *conditional* score
# $\nabla_{x^u}\log p_t(x^u_t\mid x^k_0=y)$. The network instead provides
# $\nabla_{x^u}\log p_t(x^u_t,x^k_t)=\nabla_{x^u}\log p_t(x^u_t\mid x^k_t)$: it conditions on the **noisy** known
# pixels $x^k_t$, not on $y$ itself. At high noise, $x^k_t$ says little about $y$, so the unknown region commits to a
# coarse structure that ignores most of the observation. By the time the known pixels are sharp, that structure can
# barely change. The typical symptoms are seams at the mask border and completions that look plausible locally but are
# incoherent globally, and they get worse as the mask grows. More steps do not help, because the error lies in the
# conditional the dynamics target, not in the discretisation.
#
# ### 2.2 RePaint: resample to harmonise
#
# Lugmayr et al. (2022) give the unknown region time to adapt to the known one at every noise level. After going
# down $j$ steps, the sampler **jumps back up** $j$ steps by adding fresh forward noise to the *whole* image,
# $x_{t+j}\sim q(x_{t+j}\mid x_t)$, and denoises again (with replacement). It climbs back and re-denoises $r-1$ times,
# so every level on the resampling grid is traversed $r$ times in total ($r=1$ is plain replacement). The price is
# roughly $r$ times more network evaluations (NFE).
#
# The intuition is MCMC-like. Without a constraint, noising up with the exact forward kernel and denoising back down
# with the exact reverse kernel leaves $p_t$ invariant, so each round trip is a valid move that lets the unknown
# pixels "mix" while the clamped known pixels keep pulling them. **But it is not an exact sampler of the
# conditional.** Every descent still uses the replacement kernel, which conditions on the noisy $x^k_t$ rather than
# on $y$, so $p_t(x_t\mid y)$ is not exactly invariant. RePaint is a heuristic that *reduces* the replacement bias,
# with no guarantee that it vanishes. Asymptotically exact alternatives exist, e.g. sequential Monte Carlo with
# twisted proposals (Wu et al. 2023), at a higher cost.
#
# **Gotchas** (each one silently produces bad results rather than an error):
# - noise the known pixels to the level of the *destination* of each step (`t_next`), and copy them exactly at the
#   end ($t=-1$);
# - the jump back noises the *whole* image, so known and unknown pixels stay at the same noise level;
# - be consistent about the mask convention: here $m=1$ means *known* everywhere. Flipping it inverts the task.

# %%
def repaint_path(n, jump, n_resample):
    """Sequence of grid indices visited by RePaint, from n-1 (pure noise) down to -1 (clean data).

    Moving down = one denoising step, moving up = one forward-noising step. Every `jump` grid steps the path
    climbs back `jump` steps, n_resample - 1 times. n_resample = 1 gives the plain replacement method.
    """
    resamples_left = {k: n_resample - 1 for k in range(0, n - jump, jump)}
    k, path = n - 1, [n - 1]
    while k >= 0:
        k -= 1
        path.append(k)
        if resamples_left.get(k, 0) > 0:
            resamples_left[k] -= 1
            for _ in range(jump):
                k += 1
                path.append(k)
    return path


@torch.no_grad()
def inpaint(model, schedule, x_known, mask, n_steps, jump=5, n_resample=1, y=None, w=1.0, generator=None):
    """Sample the unknown pixels (mask == 0) given the known ones (mask == 1). Returns (samples, NFE).

    n_resample = 1 is the replacement method, n_resample > 1 is RePaint. mask broadcasts against x_known.
    """
    grid = timesteps(schedule.T - 1, n_steps)[::-1]             # ascending times; grid index -1 = clean data

    def time_of(k):
        return grid[k] if k >= 0 else -1

    x = torch.randn(x_known.shape, device=x_known.device, generator=generator)
    path, nfe = repaint_path(len(grid), jump, n_resample), 0
    for k, k_next in zip(path[:-1], path[1:]):
        t, t_next = time_of(k), time_of(k_next)
        if k_next < k:   # one denoising step on the whole image, then overwrite the known pixels
            eps, x0 = predict(model, schedule, x, t, y, w)
            nfe += 1 if (y is None or w == 1.0) else 2
            x = ddim_step(schedule, x0, eps, t, t_next, eta=1.0, generator=generator)
            known = forward_jump(schedule, x_known, -1, t_next, generator) if t_next >= 0 else x_known
            x = mask * known + (1 - mask) * x
        else:            # RePaint's jump back: re-noise the whole image by one grid step
            x = forward_jump(schedule, x, t, t_next, generator)
    return x, nfe


print("RePaint path (n=8, jump=2, r=3):", repaint_path(8, 2, 3))
print("replacement   (n=8)            :", repaint_path(8, 2, 1))

# %% [markdown]
# Reading the RePaint path: it descends to grid index 4, climbs back to 6, descends to 4 again, and so on. Every
# multiple of `jump` below the top (here 4, 2 and 0) is reached $r=3$ times (with $r-1=2$ climbs back) before the
# walk continues downwards.
#
# ### 2.3 The exact lab: inpainting a 2-D Gaussian mixture
#
# To separate *method* error from *model* error we use the labelled 2-D mixture of notebook 06 and a **perfect
# model**. For a GMM, every noised marginal $q_t$ is again a GMM, so the Bayes-optimal noise predictor
# $\epsilon^\star(x_t,t)=-\sqrt{1-\bar\alpha_t}\,\nabla\log q_t(x_t)$ is known exactly. We wrap it with the same
# signature as the U-Net and feed it to the *same* `inpaint` function. The task is to observe the first coordinate,
# $x_1=c$, and sample the second.
#
# The exact answer is available too. Within component $k$ (mean $\mu$, covariance $\Sigma$), Gaussian conditioning
# gives $x_2\mid x_1=c\sim\mathcal N\big(\mu_2+\tfrac{\Sigma_{21}}{\Sigma_{11}}(c-\mu_1),\;\Sigma_{22}-\tfrac{\Sigma_{21}^2}{\Sigma_{11}}\big)$.
# The component weights are reweighted by how well each component explains the observation,
# $\pi_k\propto w_k\,\mathcal N(c;\mu_{k,1},\Sigma_{k,11})$. For a *noisy* observation $y=x_1+n$,
# $n\sim\mathcal N(0,\sigma_y^2)$, simply replace $\Sigma_{11}$ by $\Sigma_{11}+\sigma_y^2$ (we need that in section 4).

# %%
gmm, comp_class = dm.labelled_gmm()
sched_cpu = dm.NoiseSchedule(1000, "linear")   # the 2-D lab runs on the CPU (GaussianMixture works in float64)


class ExactEps(torch.nn.Module):
    """Bayes-optimal noise predictor of a GMM, eps*(x_t, t) = -sqrt(1 - abar_t) * grad log q_t(x_t).

    It has the signature of a trained network, so every sampler can be run with a *perfect* model.
    """
    n_classes = None

    def __init__(self, gmm, schedule):
        super().__init__()
        self.gmm, self.schedule, self._marginals = gmm, schedule, {}

    def forward(self, x, t, y=None):
        t = int(t[0])                    # our samplers use a single time per batch
        ab = float(self.schedule.alpha_bars[t])
        if t not in self._marginals:     # q_t is again a GMM (notebook 01); cache it per time step
            self._marginals[t] = self.gmm.noised(ab)
        return -math.sqrt(1 - ab) * self._marginals[t].score(x)


def gmm_x2_given_x1(gmm, c, noise_var=0.0):
    """Exact p(x2 | x1 = c) of a 2-D GMM, or p(x2 | y = c) if y = x1 + N(0, noise_var).

    Returns a 1-D mixture as (weights, means, variances).
    """
    mu, S = gmm.means, gmm.covs
    s11 = S[:, 0, 0] + noise_var
    log_w = torch.log(gmm.weights) - 0.5 * (c - mu[:, 0]) ** 2 / s11 - 0.5 * torch.log(s11)
    return torch.softmax(log_w, 0), mu[:, 1] + S[:, 1, 0] / s11 * (c - mu[:, 0]), S[:, 1, 1] - S[:, 1, 0] ** 2 / s11


def gmm1d_pdf(z, weights, means, variances):
    return (weights[:, None] * torch.exp(-0.5 * (z[None] - means[:, None]) ** 2 / variances[:, None])
            / torch.sqrt(2 * math.pi * variances[:, None])).sum(0)


def gmm1d_sample(n, weights, means, variances, seed):
    g = torch.Generator().manual_seed(seed)
    k = torch.multinomial(weights, n, replacement=True, generator=g)
    return means[k] + variances[k].sqrt() * torch.randn(n, dtype=torch.float64, generator=g)


def w1(a, b):
    """1-D Wasserstein-1 distance between two samples of equal size (mean gap between sorted values)."""
    return (torch.sort(a.double().flatten())[0] - torch.sort(b.double().flatten())[0]).abs().mean().item()


def implausible_mass(x2, weights, means, variances, rel=0.05):
    """Fraction of samples where the exact density is below `rel` times its peak (regions it nearly rules out)."""
    grid = torch.linspace(-5, 6, 4000, dtype=torch.float64)
    peak = gmm1d_pdf(grid, weights, means, variances).max()
    return (gmm1d_pdf(x2.double().flatten(), weights, means, variances) < rel * peak).double().mean().item()


exact_eps = ExactEps(gmm, sched_cpu)
C_OBS = -1.0                                         # we observe x1 = -1 and infer x2
cw, cm, cv = gmm_x2_given_x1(gmm, C_OBS)
exact_x2 = gmm1d_sample(N_TOY, cw, cm, cv, seed=1)

# Check 1: the closed form against brute force (sample the joint, keep points with x1 within 0.02 of C_OBS).
joint = gmm.sample(50_000 if FAST else 400_000, generator=torch.Generator().manual_seed(0))
near = joint[(joint[:, 0] - C_OBS).abs() < 0.02, 1]
err_mc = w1(near, gmm1d_sample(len(near), cw, cm, cv, seed=2))
print(f"rejection sampling: {len(near)} accepted, W1 to the closed form = {err_mc:.3f}")
assert err_mc < 0.15
# Check 2: our 1-D W1 agrees with dmtools' sliced Wasserstein (in 1-D every projection is +-1).
m_ = min(len(near), len(exact_x2))
assert abs(w1(near[:m_], exact_x2[:m_]) - dm.sliced_wasserstein(near[:m_, None], exact_x2[:m_, None])) < 1e-4

# %% [markdown]
# Now we run replacement and RePaint with the perfect model and compare their $x_2$ samples with the exact conditional.
# Two numbers measure the error. The 1-D Wasserstein distance $W_1$ to exact samples is the overall error; two exact
# samples of the same size are not at $W_1=0$ either, so we also compute that sampling-noise floor. The
# **implausible mass** is the fraction of samples that land where the exact conditional density is below 5% of its
# peak, i.e. in regions the observation nearly rules out. Exact samples give its baseline value.

# %%
x_obs_toy = torch.zeros(N_TOY, 2)
x_obs_toy[:, 0] = C_OBS
mask_toy = torch.tensor([1.0, 0.0])                  # x1 known, x2 unknown
toy_runs = [("replacement", n_steps_toy, 1) for n_steps_toy in ([25, 100] if FAST else [25, 100, 400, 1000])] + \
           [("RePaint", TOY_STEPS, r) for r in ([2, 5] if FAST else [2, 5, 10, 20])]
toy_results = {}
t0 = time.time()
for method, n_steps_toy, r in toy_runs:
    out, nfe = inpaint(exact_eps, sched_cpu, x_obs_toy, mask_toy, n_steps=n_steps_toy, jump=5, n_resample=r,
                       generator=torch.Generator().manual_seed(3))
    res = dict(x2=out[:, 1], nfe=nfe, w1=w1(out[:, 1], exact_x2), bad=implausible_mass(out[:, 1], cw, cm, cv))
    toy_results[(method, n_steps_toy, r)] = res
    print(f"{method:<12s} n={n_steps_toy:5d} r={r:3d}: NFE = {nfe:5d}   W1 to exact = {res['w1']:.3f}   "
          f"implausible mass = {res['bad']:.3f}")
noise_floor = w1(gmm1d_sample(N_TOY, cw, cm, cv, seed=4), exact_x2)
print(f"exact samples: W1 = {noise_floor:.3f} (sampling-noise floor), implausible mass = "
      f"{implausible_mass(exact_x2, cw, cm, cv):.3f}   ({time.time() - t0:.0f}s)")

# %%
REPL_SHOW = ("replacement", 100 if FAST else 400, 1)   # about the same NFE as ...
REPAINT_SHOW = ("RePaint", TOY_STEPS, 5)               # ... RePaint with r = 5
fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
pts, comp = gmm.sample(3000, return_labels=True, generator=torch.Generator().manual_seed(5))
cls = torch.as_tensor(comp_class)[comp]
for c, colour in zip(range(3), plt.cm.Set2.colors):
    axes[0].scatter(pts[cls == c, 0], pts[cls == c, 1], s=2, alpha=0.5, color=colour, label=f"class {c}")
axes[0].axvline(C_OBS, color=COL["exact"], lw=1.5, label=f"observed $x_1={C_OBS:g}$")
axes[0].set(title="2-D mixture (labelled_gmm)", xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
axes[0].legend(markerscale=4, loc="lower right", fontsize=7)
x2_grid = torch.linspace(-3, 3.5, 400, dtype=torch.float64)
for ax, key in zip(axes[1:3], [REPL_SHOW, REPAINT_SHOW]):
    res = toy_results[key]
    ax.hist(res["x2"].numpy(), bins=70, range=(-3, 3.5), density=True, color=COL[key[0]], alpha=0.6,
            label=f"{key[0]} (perfect model)")
    ax.plot(x2_grid, gmm1d_pdf(x2_grid, cw, cm, cv), color=COL["exact"], lw=1.5, label=r"exact $p(x_2\mid x_1=c)$")
    ax.set(title=f"{key[0]}: NFE = {res['nfe']},  $W_1$ = {res['w1']:.2f}\nimplausible mass = {res['bad']:.2f}",
           xlabel="$x_2$", ylabel="density")
    ax.legend(loc="upper left")
for method in ["replacement", "RePaint"]:
    keys = [k for k in toy_runs if k[0] == method]
    axes[3].plot([toy_results[k]["nfe"] for k in keys], [toy_results[k]["w1"] for k in keys], "o-",
                 color=COL[method], label=method + (" (more steps)" if method == "replacement" else " (more resampling)"))
axes[3].axhline(noise_floor, ls="--", color=COL["baseline"], label="sampling-noise floor")
axes[3].set(xscale="log", xlabel="network evaluations (NFE)", ylabel="$W_1$ to the exact conditional",
            title="error vs. compute", ylim=(0, None))
axes[3].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Even with a **perfect** score, replacement puts much of its mass on a wrong mode near $x_2\approx-0.4$, which the
# exact conditional almost rules out. Its error barely changes with 40× more steps (red curve), confirming that the
# bias is in the method, not the discretisation. The wrong mode comes from the components at $x_1=0$ and $x_1=-2.2$: at
# high noise, the noisy copy of $x_1=-1$ cannot rule them out, and $x_2$ commits to them early. RePaint at about the
# same cost moves only *part* of that mass back (compare the implausible mass of the two histograms). More resampling
# keeps reducing the error (blue curve), but slowly: even the largest $r$, at about 20× the NFE of a plain chain, stays
# far above the noise floor with a visible wrong mode. RePaint is better than replacement, but it is an approximation
# too, and with a perfect model its remaining error is pure method bias. (A smaller part of the error at $n=100$ grid
# steps comes from the under-dispersion of strided steps, section 1.1.)
#
# ### 2.4 Inpainting MNIST digits
#
# Now the real model. We use three masks: the bottom half missing, a 12×12 centre square missing, and 85% of the
# pixels missing at random. Three samplers see the same inputs: replacement with the usual `STEPS` steps; replacement
# with as many steps as RePaint spends NFE, so that compute is not the difference; and RePaint ($r=5$). The
# classifier tells us whether a completion is still recognisably the original digit. This is a rough proxy for
# "coherent with the known pixels", because a large mask can make a few digits genuinely ambiguous. We score
# `N_EVAL` digits per mask and show the first six.

# %%
def make_masks():
    """Three inpainting masks of shape (1, 1, 28, 28); 1 = known pixel."""
    bottom = torch.ones(1, 1, 28, 28)
    bottom[..., 14:, :] = 0
    centre = torch.ones(1, 1, 28, 28)
    centre[..., 8:20, 8:20] = 0
    sparse = (torch.rand(1, 1, 28, 28, generator=torch.Generator().manual_seed(0)) < 0.15).float()
    return {"bottom half": bottom, "centre square": centre, "85% random": sparse}


def show_masked(x, mask):
    """RGB view of a masked image: known pixels in grey-scale, missing pixels in blue."""
    rgb = ((x.cpu() + 1) / 2).repeat(1, 3, 1, 1)
    blue = torch.tensor([0.25, 0.45, 0.85]).view(1, 3, 1, 1)
    m = mask.cpu()
    return m * rgb + (1 - m) * blue


MASKS = make_masks()
INP_IDX = torch.tensor([0, 1, 3, 4, 7, 8, 40, 41, 42, 43, 44, 45])[:N_EVAL]   # the figure shows 7 2 0 4 9 5
n_per_mask = len(INP_IDX)
x_inp = x_test[INP_IDX].repeat(len(MASKS), 1, 1, 1).to(device)  # every mask gets the same digits
lab_inp = y_test[INP_IDX].repeat(len(MASKS))
mask_inp = torch.cat([m.expand(n_per_mask, -1, -1, -1) for m in MASKS.values()]).to(device)
R_RESAMPLE = 2 if FAST else 5
inp_out, inp_nfe = {}, {}
t0 = time.time()
inp_out["RePaint"], inp_nfe["RePaint"] = inpaint(model, schedule, x_inp, mask_inp, n_steps=STEPS, jump=5,
                                                 n_resample=R_RESAMPLE, generator=make_gen(10))
for name, n_steps_inp in [("replacement", STEPS), ("replacement, equal NFE", inp_nfe["RePaint"])]:
    inp_out[name], inp_nfe[name] = inpaint(model, schedule, x_inp, mask_inp, n_steps=n_steps_inp,
                                           generator=make_gen(10))
INP_METHODS = ["replacement", "replacement, equal NFE", "RePaint"]
print(f"{len(x_inp)} inputs x 3 samplers in {time.time() - t0:.0f}s; NFE: " +
      ", ".join(f"{m} {inp_nfe[m]}" for m in INP_METHODS))
# 1 = still classified as the original digit; one row per mask
correct = {m: (classify(inp_out[m]) == lab_inp).float().view(len(MASKS), n_per_mask) for m in INP_METHODS}


def mean_se(v):
    """Mean and standard error of a 0/1 vector, as text."""
    p = v.mean().item()
    return f"{p:.2f} ± {math.sqrt(p * (1 - p) / len(v)):.2f}"


print(f"\nfraction classified as the original digit (± standard error; {n_per_mask} digits per mask)")
print(f"{'mask':<15s}" + "".join(f"{m:>25s}" for m in INP_METHODS))
for i, name in enumerate(MASKS):
    print(f"{name:<15s}" + "".join(f"{mean_se(correct[m][i]):>25s}" for m in INP_METHODS))
print(f"{'all masks':<15s}" + "".join(f"{mean_se(correct[m].flatten()):>25s}" for m in INP_METHODS))

# %%
fig, axes = plt.subplots(1, len(MASKS), figsize=(15, 4.4))
for i, (ax, name) in enumerate(zip(axes, MASKS)):
    sl = slice(i * n_per_mask, i * n_per_mask + N_IMG)            # the first N_IMG digits of each mask
    image_grid([x_inp[sl], show_masked(x_inp[sl], mask_inp[sl])] + [inp_out[m][sl] for m in INP_METHODS],
               ax=ax, title=name,
               row_labels=["original", "observed"] + [f"{m.split(',')[0]}\n({inp_nfe[m]} NFE)" for m in INP_METHODS]
               if i == 0 else None)
plt.tight_layout()
plt.show()

# %% [markdown]
# Look at how the completions connect to the known pixels. Replacement often produces strokes that do not continue
# the observed ones (kinks and crossings at the border of the mask, an extra loop), and on the sparse mask it mostly
# draws scribbles that ignore the scattered known pixels. Giving it as many steps as RePaint spends NFE does not cure
# this, just as in the 2-D lab: the problem is the method, not the step count. RePaint's completions are coherent
# digits more often, most clearly on the sparse mask, and the pooled row of the table agrees. Read the per-mask rows
# with their standard errors in mind: with `N_EVAL` digits per mask, differences of one or two digits are noise. RePaint
# is not seamless either. Some of its completions still have strokes that stop or kink at the mask border, as the
# 2-D lab predicts for $r=5$. Note also that "coherent" does not mean "the original digit": the upper half of a 2 or
# a 9 is genuinely compatible with other digits, and a correct posterior sampler *should* sometimes complete it
# differently. The sparse mask is the hardest case at this step budget. Section 4.3 attacks a similar problem with
# DPS.
#
# ### 2.5 One input, many completions
#
# Inpainting is a *posterior sampling* problem, so different random seeds should give different but equally plausible
# completions when the observation is ambiguous. Their pixel-wise standard deviation is an uncertainty map, with
# exactly zero uncertainty on the known pixels.

# %%
AMB_IDX = torch.tensor([18, 9])                 # a 3 and a 9 with their bottom halves removed
n_comp = 2 if FAST else 6
x_amb = x_test[AMB_IDX].repeat_interleave(n_comp, 0).to(device)
mask_amb = MASKS["bottom half"].to(device)
t0 = time.time()
comps, _ = inpaint(model, schedule, x_amb, mask_amb, n_steps=STEPS, jump=5, n_resample=R_RESAMPLE,
                   generator=make_gen(11))
comps = comps.view(len(AMB_IDX), n_comp, 1, 28, 28).cpu()
std_map = comps.std(1, unbiased=False)
print(f"{len(x_amb)} completions in {time.time() - t0:.0f}s; labels assigned by the classifier:")
for i in range(len(AMB_IDX)):
    print(f"  digit {y_test[AMB_IDX[i]].item()}: {classify(comps[i]).tolist()}")
rows = [torch.cat([show_masked(x_test[AMB_IDX[i]:AMB_IDX[i] + 1], mask_amb), ((comps[i] + 1) / 2).repeat(1, 3, 1, 1),
                   heat_rgb(std_map[i:i + 1], vmax=1.0)]) for i in range(len(AMB_IDX))]
image_grid(rows, col_labels=["observed"] + [f"#{k + 1}" for k in range(n_comp)] + ["pixel std"], size=0.8,
           title="RePaint completions of the same input (bottom half missing); std map: black = 0, yellow = 1")
plt.tight_layout()
plt.show()

# %% [markdown]
# The completions differ exactly where the observation leaves room. The uncertainty map is zero on the known top half
# and largest where the strokes of the missing half could go in several ways (for example, whether a loop closes).
# This is a posterior, not a single "best guess". Averaging the completions would give a blurry estimate of the
# posterior mean, a point we make precise in section 5.
#
# ## 3 · SDEdit: editing by "noise, then denoise"
#
# Often the "measurement" is not a precise observation but a rough **guide**: a sketch, a collage or a damaged
# photo. SDEdit (Meng et al. 2022) handles this with no likelihood at all. It noises the guide to an intermediate
# time $t_0$ and runs the ordinary reverse process from there:
# $$x_{t_0}=\sqrt{\bar\alpha_{t_0}}\,x_{\text{guide}}+\sqrt{1-\bar\alpha_{t_0}}\,\epsilon,\qquad x_0\sim p_\theta(x_0\mid x_{t_0})\ \text{(reverse process from } t_0).$$
# **Why it works.** Noise at level $t_0$ drowns every image feature whose signal is weaker than the noise. In natural
# images (and digits), fine details carry little power and disappear first, while the coarse layout survives the
# longest (Dieleman's "spectral autoregression" view). After noising, the guide's unrealistic details (pen
# strokes, blocks, specks) are gone, its coarse layout remains, and the reverse process fills in realistic details
# consistent with that layout. The **strength** $s=t_0/T$ trades the two off. A low $s$ stays faithful to the guide
# but cannot remove its artefacts. A high $s$ gives realistic images that ignore the guide, and at $s=1$ it is
# ordinary sampling.
#
# **Implementation detail.** Use the *same step density* as a full run, so that $n=s\cdot N$ steps cover
# $[0,t_0]$. SDEdit is usually run with the stochastic sampler ($\eta=1$). With $\eta=0$ the output is a deterministic
# function of the guide and the noise draw.

# %%
@torch.no_grad()
def sdedit(model, schedule, x_guide, strength, n_steps_full, y=None, w=1.0, generator=None):
    """SDEdit: noise the guide to t0 = strength * (T-1), then run the stochastic reverse process from t0."""
    t0 = int(round(strength * (schedule.T - 1)))
    x_t0 = forward_jump(schedule, x_guide, -1, t0, generator)        # sqrt(abar) x + sqrt(1 - abar) eps
    n_steps = max(2, int(round(strength * n_steps_full)))
    return reverse_process(model, schedule, x_t0, timesteps(t0, n_steps), y=y, w=w, eta=1.0, generator=generator)


def draw_strokes(strokes, size=28, half_width=1.0, value=0.5):
    """A crude 'sketch': straight segments [(x0, y0, x1, y1), ...] drawn with a grey pen into a [-1, 1] image."""
    yy, xx = torch.meshgrid(torch.arange(size, dtype=torch.float32), torch.arange(size, dtype=torch.float32),
                            indexing="ij")
    p = torch.stack([xx, yy], -1)
    ink = torch.zeros(size, size)
    for x0, y0, x1, y1 in strokes:
        a, b = torch.tensor([x0, y0], dtype=torch.float32), torch.tensor([x1, y1], dtype=torch.float32)
        s = ((p - a) @ (b - a) / ((b - a) @ (b - a))).clamp(0, 1)      # projection onto the segment
        dist = (p - (a + s[..., None] * (b - a))).norm(dim=-1)
        ink = torch.maximum(ink, (half_width + 0.5 - dist).clamp(0, 1))  # anti-aliased pen
    return (-1 + (1 + value) * ink)[None, None]


sketches = torch.cat([draw_strokes([(7, 6, 21, 6), (21, 6, 11, 23)]),                  # a "7"
                      draw_strokes([(10, 5, 7, 15), (7, 15, 21, 15), (17, 5, 17, 23)])])  # a "4"
STRENGTHS_SK = [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
n_seeds = 2 if FAST else 3
t0 = time.time()
rows = []
for sk in sketches:                    # same generator seed for both sketches -> same noise for "seed k"
    guide = sk[None].repeat(n_seeds, 1, 1, 1).to(device)
    edits = [sdedit(model, schedule, guide, s, STEPS, generator=make_gen(20)) for s in STRENGTHS_SK]
    rows += [torch.cat([guide[i:i + 1]] + [e[i:i + 1] for e in edits]) for i in range(n_seeds)]
print(f"SDEdit of {2 * n_seeds} sketch copies at {len(STRENGTHS_SK)} strengths in {time.time() - t0:.0f}s")
image_grid(rows, col_labels=["guide"] + [f"s = {s:g}" for s in STRENGTHS_SK], size=0.8,
           row_labels=[f"sketch {d}, seed {k + 1}" for d in ["7", "4"] for k in range(n_seeds)],
           title="SDEdit of two hand-built sketches")
plt.tight_layout()
plt.show()

# %% [markdown]
# At the lowest strength the output keeps the ruler-straight strokes of the drawing (only their brightness starts to
# change): faithful, but not yet MNIST-like. A little more noise turns them into white, slightly curved pen strokes
# that keep the sketch's layout and identity. From the middle
# strengths on, the seeds start to disagree and the identity can change. At high strength the output no longer
# depends on the guide at all. The same seed then produces nearly the same digit for *both* sketches (compare the last
# columns of the "7" and "4" rows with equal seeds), because the guide is drowned out.
#
# ### 3.1 The realism–faithfulness trade-off, measured
#
# For a quantitative picture we damage test digits heavily (2×2 pixelation plus 12% salt-and-pepper specks), apply
# SDEdit at a range of strengths, and track three quantities:
# - **faithfulness to the content**: does the classifier still recognise the *original* digit?
# - **faithfulness to the pixels**: RMSE between output and guide;
# - **realism**, measured by an **atypicality score**: the diffusion model's own noise-prediction error on the image,
#   averaged over a few *low* noise levels and divided by the same quantity for real test digits. These errors are
#   terms of the training objective, a reweighted variational bound on $-\log p(x)$ (notebook 02). A value of 1 means
#   "as typical at fine scales as real digits", larger means less typical. Only low-noise terms of the bound are used,
#   so global shape and identity are not judged (a well-drawn wrong digit scores 1). It is the model judging images,
#   so it measures typicality under *this* prior rather than human-perceived realism.

# %%
def corrupt(x, generator):
    """Heavy, non-Gaussian damage: 2x2 pixelation plus 12% salt-and-pepper specks."""
    blocky = F.interpolate(F.avg_pool2d(x, 2), scale_factor=2, mode="nearest")
    u = torch.rand(x.shape, generator=generator)
    specks = torch.where(torch.rand(x.shape, generator=generator) < 0.5, -1.0, 1.0)
    return torch.where(u < 0.12, specks, blocky)


@torch.no_grad()
def denoising_error(x, ts=(50, 150, 300), n_draws=2):
    """Mean squared noise-prediction error of the (unconditional) model on images x at a few noise levels."""
    x = x.clamp(-1, 1).to(device)
    xs = x.repeat(len(ts) * n_draws, 1, 1, 1)
    t = torch.tensor(ts, device=device).repeat_interleave(len(x) * n_draws)
    eps = torch.randn(xs.shape, device=device, generator=make_gen(99))
    return ((model(schedule.q_sample(xs, t, eps), t) - eps) ** 2).mean().item()


err_real = denoising_error(x_test[7000:7032])                          # reference: real test digits


def atypicality(x):
    """Fine-scale denoising error relative to real digits: 1 = as typical as real data, larger = less typical."""
    return denoising_error(x) / err_real


N_SD = 4 if FAST else 16
x_clean_sd, lab_sd = x_test[100:100 + N_SD], y_test[100:100 + N_SD]
x_bad = corrupt(x_clean_sd, torch.Generator().manual_seed(21)).to(device)
STRENGTHS = [0.2, 0.6] if FAST else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
sd_out, sd_stats = {}, {"accuracy": [], "rmse_guide": [], "atypicality": []}
t0 = time.time()
for s in [0.0] + STRENGTHS:
    out = x_bad if s == 0 else sdedit(model, schedule, x_bad, s, STEPS, generator=make_gen(22))
    sd_out[s] = out.cpu()
    sd_stats["accuracy"].append((classify(out) == lab_sd).float().mean().item())
    sd_stats["rmse_guide"].append(((out - x_bad) ** 2).mean().sqrt().item())
    sd_stats["atypicality"].append(atypicality(out))
print(f"done in {time.time() - t0:.0f}s;  the clean digits themselves: atypicality {atypicality(x_clean_sd):.2f}, "
      f"accuracy {(classify(x_clean_sd) == lab_sd).float().mean():.2f}")
for s, a, r, g in zip([0.0] + STRENGTHS, sd_stats["accuracy"], sd_stats["rmse_guide"], sd_stats["atypicality"]):
    print(f"strength {s:4.2f}: accuracy {a:.2f}   RMSE to guide {r:.2f}   atypicality {g:5.2f}")

# %%
show_s = [0.0] + STRENGTHS
n_ex = min(3, N_SD)
image_grid([torch.cat([x_clean_sd[i:i + 1]] + [sd_out[s][i:i + 1] for s in show_s]) for i in range(n_ex)],
           size=0.8, col_labels=["clean\n(unseen)", "guide\n(s = 0)"] + [f"s = {s:g}" for s in STRENGTHS],
           title="SDEdit of corrupted test digits")
plt.tight_layout()
plt.show()

fig, ax1 = plt.subplots(figsize=(6.5, 3.6))
ax1.plot(show_s, sd_stats["accuracy"], "o-", color="tab:blue", label="classified as the original digit")
ax1.plot(show_s, sd_stats["rmse_guide"], "s-", color="tab:gray", label="RMSE to the guide (pixel faithfulness)")
ax1.set(xlabel="SDEdit strength $s=t_0/T$", ylabel="accuracy / RMSE", ylim=(0, 1.05))
ax2 = ax1.twinx()
ax2.plot(show_s, sd_stats["atypicality"], "^-", color="tab:orange", label="atypicality (right axis, 1 = real data)")
ax2.axhline(1.0, color="tab:orange", ls=":", lw=1)
ax2.set_ylabel("atypicality (log scale, lower = more realistic)")
ax2.set_yscale("log")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="center right")
ax1.set_title("SDEdit: artefacts vanish at low strength, the content at high strength")
plt.tight_layout()
plt.show()

# %% [markdown]
# The atypicality falls to the level of real data already at small strengths: pixelation and specks are
# fine-scale artefacts, the first things the noise erases. Recognisability of the *original* digit holds up at first
# (it can even improve, since the edit removes damage that confused the classifier) and then falls towards chance as
# the strength approaches 1 and the content is forgotten. Pixel distance to the guide grows steadily. The useful
# operating range is where the corruption is gone but the content is not: low strengths for this kind of
# fine-grained damage. How much strength a guide needs depends on how far it is from real data (the sketches above
# needed more).
#
# ### 3.2 Class-conditional SDEdit: turning a 3 into an 8
#
# The reverse process from $t_0$ does not have to use the same conditioning as the guide. Noise a "3" and denoise it
# with label 8 and classifier-free guidance ($w=3$, notebook 06). The coarse layout (size, slant, position) comes from
# the guide, and the identity comes from the label.

# %%
threes = x_test[y_test == 3][:N_IMG // 2 + 2].to(device)
y8 = torch.full((len(threes),), 8, device=device)
S_38 = [0.3, 0.45, 0.6, 0.75]
t0 = time.time()
to8 = {s: sdedit(model, schedule, threes, s, STEPS, y=y8, w=3.0, generator=make_gen(30)) for s in S_38}
print(f"done in {time.time() - t0:.0f}s;  fraction classified as 8: " +
      ", ".join(f"s={s:g}: {(classify(o) == 8).float().mean():.2f}" for s, o in to8.items()))
image_grid([threes] + [to8[s] for s in S_38], row_labels=["guide (a 3)"] + [f"s = {s:g}, label 8" for s in S_38],
           size=0.8, title="Class-conditional SDEdit, 3 → 8 (CFG w = 3)")
plt.tight_layout()
plt.show()

# %% [markdown]
# At the lowest strength the label only partly wins: some outputs are still 3s or 3/8 hybrids. From the middle
# strengths on, the outputs become 8s that inherit the slant, size and stroke thickness of the particular 3. This is the mechanism behind
# "image + text prompt" editing in large text-to-image models.
#
# ## 4 · Diffusion Posterior Sampling for linear inverse problems
#
# Many measurement processes are linear with Gaussian noise, $y=Ax_0+n$, $n\sim\mathcal N(0,\sigma_y^2I)$. Examples
# are downsampling (super-resolution), blurring, missing pixels and tomography. Here the likelihood
# $p(y\mid x_0)=\mathcal N(y;Ax_0,\sigma_y^2I)$ is known exactly, so we would like to use the full posterior score
# from section 1.
#
# **The DPS approximation** (Chung et al. 2023). The intractable term is
# $p_t(y\mid x_t)=\int\mathcal N(y;Ax_0,\sigma_y^2I)\,p(x_0\mid x_t)\,dx_0$. DPS replaces the denoising posterior
# $p(x_0\mid x_t)$ by a point mass at its mean, which the network gives us via Tweedie:
# $$p_t(y\mid x_t)\;\approx\;\mathcal N\big(y;\,A\hat x_0(x_t),\,\sigma_y^2I\big)
# \quad\Longrightarrow\quad
# \nabla_{x_t}\log p_t(y\mid x_t)\approx-\frac{1}{2\sigma_y^2}\nabla_{x_t}\big\|y-A\hat x_0(x_t)\big\|^2
# =\frac{1}{\sigma_y^2}\Big(\frac{\partial\hat x_0}{\partial x_t}\Big)^{\!\top}A^\top\big(y-A\hat x_0(x_t)\big).$$
# - The Jacobian $\partial\hat x_0/\partial x_t$ requires **backpropagation through the network** at every step: one
#   vector–Jacobian product, about 2–3× the cost of a plain step, plus activation memory.
# - The approximation is good at small $t$, where $p(x_0\mid x_t)$ is concentrated. It is crude at large $t$, where
#   $\hat x_0$ is the blurry average shown in section 1. Because the likelihood is a nonlinear function of $x_0$,
#   $\mathbb E[\mathcal N(y;Ax_0,\cdot)]\neq\mathcal N(y;A\,\mathbb E[x_0],\cdot)$ in general (Chung et al. call the
#   difference the *Jensen gap*), so the average of two wrong answers can be perfectly consistent with $y$.
#
# **The step size.** Taken literally, the weight $1/(2\sigma_y^2)$ is huge for small $\sigma_y$ and makes the sampler
# unstable unless the steps are tiny. DPS therefore uses a normalised step,
# $$x_{t-1}=x'_{t-1}-\zeta\,\nabla_{x_t}\big\|y-A\hat x_0(x_t)\big\|,$$
# where $x'_{t-1}$ is the ordinary ancestral step. This is the gradient of the *norm*, i.e. the paper's
# $\zeta_t=\zeta'/\|y-A\hat x_0\|$ applied to the squared norm, with $\zeta=2\zeta'$.
#
# **What $\zeta$ means.** With $r=y-A\hat x_0$, the normalised correction is
# $-\zeta\nabla\|r\|=-\frac{\zeta}{2\|r\|}\nabla\|r\|^2$, while the literal likelihood term is
# $-\frac{1}{2\sigma_y^2}\nabla\|r\|^2$. The practical rule therefore **contains no $\sigma_y$**. The weight of the data
# is set by $\zeta$ (and the current residual), so $\zeta$ acts as an inverse *likelihood temperature*. It must be
# re-tuned whenever $\sigma_y$, the operator or the number of steps changes. Near the end of sampling, where
# $\hat x_0(x_t)\approx x_t$, each step moves the sample a distance of about $\zeta$ along the residual direction. For a
# pixel mask this changes $\|r\|$ by about $\zeta$ per step, so the final residual ends up somewhere between 0 and
# $\zeta$ (about $\zeta/2$), *whatever* $\sigma_y$ is. To land at the noise level, pick
# $\zeta\approx2\sigma_y\sqrt{\dim y}$. For other operators the factor differs. Section 4.4 checks this.
#
# **Gotchas:** wrap the gradient in `torch.enable_grad()`, because samplers usually run under `no_grad`. Detach
# $x_t$ after each step so the graph does not grow across steps. Use a *per-sample* norm (summing per-sample losses
# gives per-sample gradients because the U-Net treats batch elements independently; it uses GroupNorm, not BatchNorm).
# The gradient of $\|r\|$ is undefined at $r=0$, so add a tiny $\varepsilon$ inside the square root.

# %%
def dps_sample(model, schedule, A, y_meas, shape, n_steps, zeta, weighting="dps", sigma_y=None, generator=None,
               x_T=None):
    """Diffusion posterior sampling for y = A(x0) + noise (Chung et al. 2023).

    Each step = an ordinary ancestral step + a correction from the gradient of the measurement error of x0_hat(x_t),
    obtained by backpropagating through the network.
      weighting="dps"  : x_{t-1} = x'_{t-1} - zeta * grad ||y - A x0_hat||       (the paper's practical rule)
      weighting="bayes": add zeta * grad log N(y; A x0_hat, sigma_y^2 I) to the score (zeta = 1 is the literal weight)
    zeta is a float or a tensor of shape (B,) (one step size per sample). x_T optionally fixes the starting noise.
    """
    dev = y_meas.device
    zeta = torch.as_tensor(zeta, dtype=torch.float32, device=dev)
    zeta = zeta.view(-1, *([1] * (len(shape) - 1))) if zeta.ndim else zeta
    ts = timesteps(schedule.T - 1, n_steps)
    x = x_T.to(dev) if x_T is not None else torch.randn(shape, device=dev, generator=generator)
    for i, t in enumerate(ts):
        t_next = ts[i + 1] if i + 1 < len(ts) else -1
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            eps, x0 = predict(model, schedule, x_in, t)
            sq_err = (y_meas - A(x0)).flatten(1).pow(2).sum(1)          # per-sample ||y - A x0_hat||^2
            if weighting == "dps":
                loss = (sq_err + 1e-12).sqrt().sum()
            else:
                loss = (sq_err / (2 * sigma_y ** 2)).sum()
            grad = torch.autograd.grad(loss, x_in)[0]
        eps, x0 = eps.detach(), x0.detach()
        if weighting == "dps":
            x = ddim_step(schedule, x0, eps, t, t_next, 1.0, generator) - zeta * grad
        else:   # posterior score = prior score + likelihood score, as in classifier guidance (notebook 06)
            ab = schedule.alpha_bars[t]
            eps = eps + zeta * (1 - ab).sqrt() * grad
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            x = ddim_step(schedule, x0, eps, t, t_next, 1.0, generator)
    return x.detach()


# %% [markdown]
# **Check: the gradient through the U-Net.** The backpropagated gradient of $\|y-A\hat x_0(x_t)\|$ must match a
# finite-difference directional derivative along a random unit direction $v$. (The CPU copy of the 2-D lab's
# schedule, `sched_cpu`, is defined in section 2.3; it is the same $T=1000$ linear schedule.)

# %%
def A_sr(x):
    """4x super-resolution operator: average 4x4 blocks, 28x28 -> 7x7."""
    return F.avg_pool2d(x, 4)


# On a float32 CPU copy: GPUs may use reduced-precision (TF32) convolutions, too coarse for finite differences.
model_cpu = copy.deepcopy(model).cpu()
x_c = torch.randn(1, 1, 28, 28, generator=torch.Generator().manual_seed(40))
y_c = A_sr(x_test[:1])


def meas_err(x, t=300):
    return (y_c - A_sr(predict(model_cpu, sched_cpu, x, t)[1])).flatten(1).norm(dim=1).sum()


with torch.enable_grad():
    x_in = x_c.clone().requires_grad_(True)
    g_auto = torch.autograd.grad(meas_err(x_in), x_in)[0]
v = torch.randn(x_c.shape, generator=torch.Generator().manual_seed(41))
v = v / v.norm()
h = 1e-2
with torch.no_grad():
    d_fd = ((meas_err(x_c + h * v) - meas_err(x_c - h * v)) / (2 * h)).item()
d_auto = (g_auto * v).sum().item()
print(f"directional derivative: autograd {d_auto:+.5f}, central finite difference {d_fd:+.5f}")
assert abs(d_auto - d_fd) < 0.02 * abs(d_auto) + 1e-3

# %% [markdown]
# ### 4.1 The exact lab again: how good is the DPS approximation?
#
# Back to the 2-D mixture with its perfect score. Now the measurement is noisy, $y=x_1+n$ with $\sigma_y=0.1$, and we
# observe $y=-1$. The exact posterior over $x_2$ is the formula of section 2.3 with $\Sigma_{11}+\sigma_y^2$. We run
# DPS twice. First with the literal Bayes weight $1/(2\sigma_y^2)$ and many small steps, which isolates the
# $\hat x_0$ approximation itself. Second with the practical normalised rule and 100 steps.

# %%
SIGMA_Y_TOY = 0.1
pw, pm, pv = gmm_x2_given_x1(gmm, C_OBS, SIGMA_Y_TOY ** 2)
N_DPS_TOY = N_TOY // 2                    # DPS backpropagates through the score at every step: use fewer particles
exact_post = gmm1d_sample(N_DPS_TOY, pw, pm, pv, seed=6)
y_toy = torch.full((N_DPS_TOY, 1), C_OBS)


def A_first(x):
    """Observe the first coordinate."""
    return x[:, :1]


def gmm_x1_given_y(gmm, y, noise_var):
    """Exact p(x1 | y) for y = x1 + N(0, noise_var): per component a Gaussian update, weights as in gmm_x2_given_x1."""
    s11 = gmm.covs[:, 0, 0]
    weights = gmm_x2_given_x1(gmm, y, noise_var)[0]
    gain = s11 / (s11 + noise_var)
    return weights, gmm.means[:, 0] + gain * (y - gmm.means[:, 0]), s11 * (1 - gain)


# How tightly should x1 fit the observation? The exact posterior answers it.
exact_x1 = gmm1d_sample(100_000, *gmm_x1_given_y(gmm, C_OBS, SIGMA_Y_TOY ** 2), seed=8)
print(f"exact posterior: mean |y - x1| = {(exact_x1 - C_OBS).abs().mean():.3f}")


dps_toy_runs = {"DPS, Bayes weight (zeta = 1)": dict(weighting="bayes", zeta=1.0, n_steps=10 * TOY_STEPS),
                "DPS, normalised rule (zeta = 0.3)": dict(weighting="dps", zeta=0.3, n_steps=TOY_STEPS)}
dps_toy = {}
t0 = time.time()
for name, kw in dps_toy_runs.items():
    out = dps_sample(exact_eps, sched_cpu, A_first, y_toy, (N_DPS_TOY, 2), sigma_y=SIGMA_Y_TOY,
                     generator=torch.Generator().manual_seed(7), **kw)
    dps_toy[name] = out
    print(f"{name:<34s} ({kw['n_steps']:4d} steps): W1(x2) to exact = {w1(out[:, 1], exact_post):.3f},  "
          f"mean |y - x1| = {(out[:, 0] - C_OBS).abs().mean():.3f}")
print(f"(replacement inpainting, which assumes sigma_y = 0, had W1 = {toy_results[REPL_SHOW]['w1']:.3f}; "
      f"{time.time() - t0:.0f}s)")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
x2_grid = torch.linspace(-3, 3.5, 400, dtype=torch.float64)
for ax, (name, out) in zip(axes, dps_toy.items()):
    ax.hist(out[:, 1].numpy(), bins=70, range=(-3, 3.5), density=True, color=COL["DPS"], alpha=0.6, label=name)
    ax.plot(x2_grid, gmm1d_pdf(x2_grid, pw, pm, pv), color=COL["exact"], lw=1.5, label=r"exact $p(x_2\mid y)$")
    ax.set(title=f"{name}\n$W_1$ = {w1(out[:, 1], exact_post):.2f}", xlabel="$x_2$")
    ax.legend(loc="upper left")
axes[0].set_ylabel("density")
plt.tight_layout()
plt.show()

# %% [markdown]
# Even with a **perfect prior** and the literal Bayes weight, DPS keeps a sizeable spurious mode near
# $x_2\approx-0.4$: the bias comes from the approximation $p(y\mid x_t)\approx p(y\mid\hat x_0)$ itself. The
# mechanism is the Jensen gap. At high noise, $\hat x_0(x_t)$ averages the wrong components at $x_1=0$ and $x_1=-2.2$,
# and that *average* is perfectly consistent with $y=-1$, so the likelihood gradient does not push the particle away.
# The practical normalised rule lands somewhat closer here, but it is biased too: the spurious mode survives and the
# main mode is slightly displaced. The observed coordinate tells the same story from the other side. Compare each
# rule's mean $|y-x_1|$ with the exact posterior's value printed above. The literal Bayes weight pulls $x_1$ *tighter*
# to $y$ than the posterior does: the point estimate $\hat x_0$ is over-confident, so it over-fits the observation.
# The normalised rule with $\zeta=0.3$ is looser than the posterior: it trusts the data less than Bayes would. Tuning
# $\zeta$ moves the error around; it does not make DPS exact. Despite this bias, DPS is very effective in practice,
# because it needs nothing but a differentiable $A$.
#
# ### 4.2 4× super-resolution of MNIST (7×7 → 28×28)
#
# $A$ averages 4×4 blocks and $\sigma_y=0.05$. For each measurement we draw several posterior samples. We compare
# them with bicubic upsampling, a naive baseline that knows nothing about digits. The measurement residual
# $\|y-Ax\|$ is reported relative to its expected value for the *true* image, $\sigma_y\sqrt{\dim y}$. For an *exact*
# posterior sampler this ratio should be about 1, not 0. Given $y$, the truth $x$ and a posterior sample $x'$ are two
# draws from the same distribution $p(x\mid y)$, so $\mathbb E\|y-Ax'\|^2=\mathbb E\|y-Ax\|^2=\sigma_y^2\dim y$. A value
# well below 1 means the sample fits the noise; well above 1, it ignores the data.

# %%
SIGMA_Y = 0.05
ZETA_SR = 2.0
SR_IDX = torch.tensor([4, 8, 9])[: (2 if FAST else 3)]           # digits 4 5 9
x_sr = x_test[SR_IDX].to(device)
y_sr = A_sr(x_sr) + SIGMA_Y * torch.randn(A_sr(x_sr).shape, device=device, generator=make_gen(42))
t0 = time.time()
post_sr = dps_sample(model, schedule, A_sr, y_sr.repeat_interleave(N_POST, 0), (len(SR_IDX) * N_POST, 1, 28, 28),
                     DPS_STEPS, zeta=ZETA_SR, generator=make_gen(43)).cpu()
print(f"DPS: {len(post_sr)} samples x {DPS_STEPS} steps in {time.time() - t0:.0f}s")
bicubic = F.interpolate(y_sr.cpu(), scale_factor=4, mode="bicubic", align_corners=False).clamp(-1, 1)


def report(name, x_hat, x_true, y, A, noise_norm, reps=1):
    """Print measurement residual (relative to noise), PSNR and classifier accuracy of reconstructions."""
    y, x_true = y.cpu().repeat_interleave(reps, 0), x_true.cpu().repeat_interleave(reps, 0)
    resid = (y - A(x_hat.cpu())).flatten(1).norm(dim=1) / noise_norm
    acc = (classify(x_hat) == classify(x_true)).float().mean()
    p = psnr(x_hat, x_true)
    p_str = f"{p:5.1f} dB" if math.isfinite(p) else "    -   "
    print(f"{name:<24s} residual/noise = {resid.mean():5.2f}   PSNR = {p_str}   same label as truth = {acc:.2f}")


noise_sr = SIGMA_Y * math.sqrt(y_sr[0].numel())
report("ground truth", x_sr, x_sr, y_sr, A_sr, noise_sr)
report("bicubic upsampling", bicubic, x_sr, y_sr, A_sr, noise_sr)
report("DPS (single samples)", post_sr, x_sr, y_sr, A_sr, noise_sr, reps=N_POST)
post_sr = post_sr.view(len(SR_IDX), N_POST, 1, 28, 28)
report("DPS (mean of samples)", post_sr.mean(1), x_sr, y_sr, A_sr, noise_sr)

# %%
up = lambda im: F.interpolate(im.cpu(), scale_factor=4, mode="nearest")   # show 7x7 measurements at 28x28
rows = [torch.cat([((x_sr[i:i + 1].cpu() + 1) / 2).repeat(1, 3, 1, 1), ((up(y_sr[i:i + 1]) + 1) / 2).repeat(1, 3, 1, 1),
                   ((bicubic[i:i + 1] + 1) / 2).repeat(1, 3, 1, 1), ((post_sr[i] + 1) / 2).repeat(1, 3, 1, 1),
                   heat_rgb(post_sr[i].std(0, unbiased=False)[None], vmax=1.0)]) for i in range(len(SR_IDX))]
image_grid(rows, size=0.85, col_labels=["truth", "y (7×7)", "bicubic"] + [f"DPS #{k + 1}" for k in range(N_POST)]
           + ["posterior\nstd"], title="4× super-resolution: DPS posterior samples vs. bicubic upsampling")
plt.tight_layout()
plt.show()

# %% [markdown]
# Bicubic upsampling is a blurry, low-contrast version of the measurement. The DPS samples are sharp digits that
# agree with the 7×7 measurement up to the noise, as the residual table shows. They differ from each other in
# stroke details that the block averages cannot determine, and the posterior-std map puts that uncertainty on the
# stroke boundaries. A high PSNR is not the goal of posterior sampling: a single sample is usually *further* from the
# truth in mean squared error than the posterior mean (section 5 explains why).
#
# ### 4.3 Recovering a digit from 15% of its pixels
#
# The same code handles a completely different operator: $A=$ "keep a random 15% of the pixels" (a fixed random mask
# $M$, so $Ax=M\odot x$), with noise $\sigma_y=0.05$. The naive baseline fills every missing pixel with a Gaussian-
# weighted average of the nearby known pixels (normalised convolution), falling back to the background where no
# pixel is known.

# %%
M_PX = (torch.rand(1, 1, 28, 28, generator=torch.Generator().manual_seed(50)) < 0.15).float().to(device)
ZETA_PX = 1.0


def A_px(x):
    """Keep a fixed random 15% of the pixels (zeros elsewhere)."""
    return x * M_PX.to(x.device)


def normalized_conv_fill(y, mask, sigma=1.5, prior_weight=0.02):
    """Naive interpolation: Gaussian-weighted average of the known pixels (background -1 where none are near)."""
    r = torch.arange(-4, 5, dtype=torch.float32)
    k1 = torch.exp(-0.5 * (r / sigma) ** 2)
    k = (k1[:, None] * k1[None, :])[None, None]
    num = F.conv2d(y.cpu() * mask.cpu(), k, padding=4) - prior_weight     # prior pull towards background (-1)
    den = F.conv2d(mask.cpu().expand_as(y.cpu()), k, padding=4) + prior_weight
    return (num / den).clamp(-1, 1)


PX_IDX = torch.tensor([0, 6, 11])[: (2 if FAST else 3)]           # digits 7 4 6
x_px = x_test[PX_IDX].to(device)
y_px = A_px(x_px + SIGMA_Y * torch.randn(x_px.shape, device=device, generator=make_gen(51)))
t0 = time.time()
post_px = dps_sample(model, schedule, A_px, y_px.repeat_interleave(N_POST, 0), (len(PX_IDX) * N_POST, 1, 28, 28),
                     DPS_STEPS, zeta=ZETA_PX, generator=make_gen(52)).cpu()
print(f"DPS: {len(post_px)} samples x {DPS_STEPS} steps in {time.time() - t0:.0f}s")
naive_px = normalized_conv_fill(y_px, M_PX)
noise_px = SIGMA_Y * math.sqrt(M_PX.sum().item())
report("ground truth", x_px, x_px, y_px, A_px, noise_px)
report("normalised convolution", naive_px, x_px, y_px, A_px, noise_px)
report("DPS (single samples)", post_px, x_px, y_px, A_px, noise_px, reps=N_POST)
post_px = post_px.view(len(PX_IDX), N_POST, 1, 28, 28)

rows = [torch.cat([((x_px[i:i + 1].cpu() + 1) / 2).repeat(1, 3, 1, 1), show_masked(y_px[i:i + 1], M_PX),
                   ((naive_px[i:i + 1] + 1) / 2).repeat(1, 3, 1, 1), ((post_px[i] + 1) / 2).repeat(1, 3, 1, 1),
                   heat_rgb(post_px[i].std(0, unbiased=False)[None], vmax=1.0)]) for i in range(len(PX_IDX))]
image_grid(rows, size=0.85, col_labels=["truth", "y (15% px)", "naive fill"] + [f"DPS #{k + 1}" for k in range(N_POST)]
           + ["posterior\nstd"], title="Sparse-pixel recovery: DPS vs. normalised-convolution interpolation")
plt.tight_layout()
plt.show()

# %% [markdown]
# With 85% of the pixels missing, local interpolation produces disconnected blobs. DPS produces complete digits that
# pass through the observed pixels. Where the observations leave room, the samples disagree, and they sometimes even
# disagree about which digit it is: that is honest posterior uncertainty. (Section 2's RePaint solves the noise-free
# version of this problem by a different route: hard replacement requires $\sigma_y=0$. Exercise 5 compares the two.)
#
# ### 4.4 The step size $\zeta$
#
# $\zeta$ is DPS's most important knob. Since `dps_sample` accepts one $\zeta$ per sample, one batch can sweep it. We
# use two sparse-pixel measurements and record the residual relative to the noise level. To isolate $\zeta$, every
# $\zeta$ sees the *same* noisy measurement and the *same* starting noise $x_T$ for a given digit (common random
# numbers). Only the per-step sampler noise differs. The dotted line is the rule of thumb from section 4: a final
# residual of about $\zeta/2$.

# %%
ZETAS = [0.0, 1.0] if FAST else [0.0, 0.3, 1.0, 3.0, 10.0]
SWEEP_IDX = torch.tensor([3, 7])                                    # digits 0 9
x_one = x_test[SWEEP_IDX].to(device)
y_one = A_px(x_one + SIGMA_Y * torch.randn(x_one.shape, device=device, generator=make_gen(60)))  # one y per digit
x_T_one = torch.randn(x_one.shape, device=device, generator=make_gen(61))                         # one x_T per digit
y_sw = y_one.repeat_interleave(len(ZETAS), 0)                       # ... shared by every zeta
zeta_vec = torch.tensor(ZETAS * len(SWEEP_IDX), device=device)
t0 = time.time()
sweep = dps_sample(model, schedule, A_px, y_sw, y_sw.shape, DPS_STEPS, zeta=zeta_vec, generator=make_gen(62),
                   x_T=x_T_one.repeat_interleave(len(ZETAS), 0)).cpu()
resid_sw = ((y_sw.cpu() - A_px(sweep)).flatten(1).norm(dim=1) / noise_px).view(len(SWEEP_IDX), len(ZETAS))
print(f"{len(sweep)} samples in {time.time() - t0:.0f}s;  rule of thumb for a mask: zeta = 2 sigma_y sqrt(m) = "
      f"{2 * noise_px:.2f};  residual / noise level:")
for i, idx in enumerate(SWEEP_IDX):
    print(f"  digit {y_test[idx].item()}: " + ", ".join(f"ζ={z:g}: {r:.2f}" for z, r in zip(ZETAS, resid_sw[i])))

fig = plt.figure(figsize=(12, 2.9))
ax_img = fig.add_axes([0.06, 0.05, 0.52, 0.85])
image_grid([sweep.view(len(SWEEP_IDX), len(ZETAS), 1, 28, 28)[i] for i in range(len(SWEEP_IDX))], ax=ax_img,
           col_labels=[f"ζ = {z:g}" for z in ZETAS], row_labels=[f"digit {y_test[i].item()}" for i in SWEEP_IDX])
ax_c = fig.add_axes([0.64, 0.17, 0.34, 0.7])
for i, idx in enumerate(SWEEP_IDX):
    ax_c.plot([max(z, 0.05) for z in ZETAS], resid_sw[i].numpy(), "o-", color=COL["DPS"], alpha=0.5 + 0.5 * i,
              label=f"digit {y_test[idx].item()}")
ax_c.axhline(1.0, ls="--", color=COL["exact"], lw=1, label="noise level (truth)")
z_line = np.geomspace(0.3, max(ZETAS), 50)
ax_c.plot(z_line, z_line / (2 * noise_px), ":", color=COL["baseline"], lw=1.5, label="rule of thumb ζ / 2")
ax_c.set(xscale="log", xlabel="ζ  (ζ = 0 plotted at 0.05)", ylabel="‖y − Ax‖ / (σ_y √m)", yscale="log",
         title="measurement consistency vs. step size")
ax_c.legend()
plt.show()

# %% [markdown]
# With $\zeta=0$ DPS is plain unconditional sampling: some digit, unrelated to $y$, with a residual far above the noise
# level. Too small a $\zeta$ is not enough to pull the sample to the measurement. A moderate $\zeta$, near the rule of
# thumb printed above, gives clean digits with a residual near or even *below* the noise level. Below means the sample
# has started to fit the noise, which an exact posterior sampler would not do (section 4.2). Once $\zeta$ is large
# enough to reach the data, the residual follows the dotted $\zeta/2$ line rather than the noise level: the last steps
# move the sample by about $\zeta$, whatever $\sigma_y$ is. At the largest $\zeta$ the image degrades, up to complete
# breakdown, because a fixed-length step that is fine early in sampling is far too large late in sampling, when $x_t$
# is almost clean. The best $\zeta$ depends on the operator, the noise level and the number of steps. In practice it
# is tuned on a few validation measurements, aiming for a residual of about the noise level.
#
# ## 5 · Denoising: the one inverse problem diffusion models solve exactly
#
# Suppose $y=x_0+\sigma n$ with $n\sim\mathcal N(0,I)$ and a known $\sigma$. Multiplying by $\sqrt{\bar\alpha}$ gives
# $\sqrt{\bar\alpha}\,y=\sqrt{\bar\alpha}\,x_0+\sqrt{\bar\alpha}\,\sigma\,n$. If we choose $t^\star$ such that
# $$\sqrt{\bar\alpha_{t^\star}}\;\sigma=\sqrt{1-\bar\alpha_{t^\star}}\quad\Longleftrightarrow\quad\sigma=\sqrt{\frac{1-\bar\alpha_{t^\star}}{\bar\alpha_{t^\star}}},$$
# then $\sqrt{\bar\alpha_{t^\star}}\,y$ has **exactly** the distribution of $x_{t^\star}$ given $x_0$. So
# $p(x_0\mid y)=p(x_0\mid x_{t^\star}=\sqrt{\bar\alpha_{t^\star}}\,y)$, and no approximation of the likelihood is needed:
# - **(a) Tweedie / MMSE:** $\hat x_0(\sqrt{\bar\alpha_{t^\star}}y,t^\star)=\mathbb E[x_0\mid y]$, with one network
#   evaluation. It is the estimate with the smallest mean squared error.
# - **(b) Posterior sampling:** run the reverse process from $t^\star$. Each run gives a sample from $p(x_0\mid y)$:
#   exactly for a perfect model in the limit of fine steps, approximately in practice (section 1.1's gotcha).
#
# **A sanity check from probability.** Let $X$ be the truth and $X'$ a posterior sample. Given $y$ they are two
# independent draws from the same posterior, so
# $\mathbb E\|X'-X\|^2=2\,\mathrm{tr}\,\mathrm{Cov}[X\mid y]=2\cdot\text{MMSE}$. A perfect posterior sampler has exactly
# **twice** the mean squared error of the Tweedie estimate, and averaging $K$ samples brings it down to
# $(1+1/K)\cdot\text{MMSE}$. The sample, however, *looks* like a real digit, while the posterior mean is an average
# of digits and is blurry. This is the **perception–distortion trade-off** (Blau & Michaeli 2018): no estimator can
# be both minimum-MSE and distributed like real data.
#
# **Gotchas:** $\sigma$ must be in data units (images in $[-1,1]$, so $\sigma=0.5$ is a quarter of the range). Do
# **not** clip the noisy image to the data range before rescaling, because clipping changes the noise distribution and
# breaks the exact correspondence. $t^\star$ is the nearest integer time, a tiny mismatch. Real sensor noise is rarely
# Gaussian with a known $\sigma$.

# %%
def t_star_for(sigma):
    """The integer time whose noise-to-signal ratio sqrt((1 - abar) / abar) is closest to sigma."""
    nsr = ((1 - schedule.alpha_bars) / schedule.alpha_bars).sqrt()
    return int((nsr - sigma).abs().argmin())


def gradient_energy(x):
    """Mean squared difference between neighbouring pixels: blurring an edge spreads it out and lowers this."""
    x = x.float()
    return (((x[..., 1:, :] - x[..., :-1, :]) ** 2).mean() + ((x[..., :, 1:] - x[..., :, :-1]) ** 2).mean()).item()


SIGMAS = [0.5] if FAST else [0.5, 1.0, 2.0, 3.0]
N_DN, K_DN = (2, 2) if FAST else (6, 6)
x_dn = x_test[10:10 + N_DN].to(device)
ge_clean = gradient_energy(x_dn)
dn = {}


def mse_to_clean(a):
    return ((a.cpu() - x_dn.cpu()) ** 2).mean().item()


DN_STRIDE = 50 if FAST else 10      # one reverse step per DN_STRIDE time units (= 100 steps over [0, T))
t0 = time.time()
for sigma in SIGMAS:
    ts_ = t_star_for(sigma)
    ab = schedule.alpha_bars[ts_]
    y_noisy = x_dn + sigma * torch.randn(x_dn.shape, device=device, generator=make_gen(70))   # NOT clipped
    x_t = ab.sqrt() * y_noisy
    with torch.no_grad():
        tweedie = predict(model, schedule, x_t, ts_)[1]
    n_steps = max(2, ts_ // DN_STRIDE)
    samples = reverse_process(model, schedule, x_t.repeat(K_DN, 1, 1, 1), timesteps(ts_, n_steps),
                              generator=make_gen(71)).view(K_DN, N_DN, 1, 28, 28)
    dn[sigma] = dict(t=ts_, n_steps=n_steps, y=y_noisy.cpu(), tweedie=tweedie.cpu(), samples=samples.cpu(),
                     mse_tweedie=mse_to_clean(tweedie), mse_sample=mse_to_clean(samples),
                     mse_mean=mse_to_clean(samples.mean(0)),
                     spread=samples.cpu().var(0, unbiased=True).mean().item(),   # sample variance around their mean
                     ge_tweedie=gradient_energy(tweedie) / ge_clean, ge_sample=gradient_energy(samples) / ge_clean)
    d = dn[sigma]
    print(f"sigma={sigma:3.1f}: t*={ts_:3d} (sqrt((1-abar)/abar) = {((1 - ab) / ab).sqrt().item():.3f}), "
          f"{n_steps} steps | PSNR: noisy {psnr(y_noisy, x_dn):5.1f}, Tweedie {psnr(tweedie, x_dn):5.1f}, sample "
          f"{10 * math.log10(4 / d['mse_sample']):5.1f}, mean of {K_DN} samples "
          f"{10 * math.log10(4 / d['mse_mean']):5.1f} dB")
    print(f"           relative to MSE(Tweedie): MSE(sample) {d['mse_sample'] / d['mse_tweedie']:.2f} (theory 2), "
          f"MSE(mean of {K_DN}) {d['mse_mean'] / d['mse_tweedie']:.2f} (theory {1 + 1 / K_DN:.2f}), "
          f"spread of the samples {d['spread'] / d['mse_tweedie']:.2f} (theory 1)")
print(f"({time.time() - t0:.0f}s)")

# %% [markdown]
# The PSNR ordering matches the theory: the Tweedie estimate (posterior mean) beats a single posterior sample. The
# three ratios, however, miss the theory in the same direction at every $\sigma$. A single sample's MSE is below
# twice Tweedie's. The mean of $K$ samples is about as good as Tweedie instead of $(1+1/K)$ times worse. And the
# samples' spread around their own mean is below Tweedie's error. All three say one thing: **the samples are less
# diverse than the posterior they should represent.** Using only a few digits adds noise to these numbers, but it
# would not push all of them the same way. Two systematic causes remain: the imperfect model, and the strided
# sampler of section 1.1. The second can be isolated exactly. With a Gaussian prior $\mathcal N(0,v_0)$ the denoiser
# is known in closed form and every step is linear-Gaussian, so the sampler's output given $x_{t^\star}$ is
# "gain $\times\,x_{t^\star}$ + Gaussian noise", and we can compare both parts with the exact posterior.

# %%
def strided_spread(v0, t_start, n_steps):
    """Our eta = 1 sampler with the EXACT denoiser of a Gaussian prior N(0, v0), started from a fixed x_{t_start}.

    The output is gain * x_{t_start} + Gaussian noise. Returns (gain, noise variance), each divided by its exact
    posterior value (the coefficient of E[x0 | x_t] and Var[x0 | x_t]).
    """
    gain, var = 1.0, 0.0
    ts = timesteps(t_start, n_steps)
    for i, t in enumerate(ts):
        t_next = ts[i + 1] if i + 1 < len(ts) else -1
        ab, ab_next = float(abar_at(schedule, t)), float(abar_at(schedule, t_next))
        k = math.sqrt(ab) * v0 / (ab * v0 + 1 - ab)               # exact denoiser: x0_hat = k * x_t
        e = (1 - math.sqrt(ab) * k) / math.sqrt(1 - ab)            # the matching eps_hat = e * x_t
        s2 = (1 - ab_next) / (1 - ab) * (1 - ab / ab_next)         # eta = 1 step variance, as in ddim_step
        a = math.sqrt(ab_next) * k + math.sqrt(max(1 - ab_next - s2, 0.0)) * e
        gain, var = a * gain, a * a * var + s2
    ab0 = float(abar_at(schedule, t_start))
    return gain / (math.sqrt(ab0) * v0 / (ab0 * v0 + 1 - ab0)), var / (v0 * (1 - ab0) / (ab0 * v0 + 1 - ab0))


t_s = t_star_for(1.0)
n_used = max(2, t_s // DN_STRIDE)
print(f"Gaussian prior, exact denoiser, reverse process from t* = {t_s} (sigma = 1):")
print("  sampler variance / exact posterior variance (the mean is exact in every case)")
for v0 in [1.0, 0.25]:
    row = []
    for n_s in [n_used, 2 * n_used, 4 * n_used, t_s + 1]:
        gain_ratio, var_ratio = strided_spread(v0, t_s, n_s)
        assert abs(gain_ratio - 1) < 1e-4, gain_ratio              # the conditional MEAN is right at any stride
        row.append(f"{n_s:4d} steps: {var_ratio:.2f}")
    print(f"  prior variance {v0:4.2f} | " + "   ".join(row) + f"   (used above: {n_used} steps)")

# %% [markdown]
# Even with a perfect denoiser, the step count used above leaves the samples with noticeably too little variance,
# and the deficit shrinks as the steps get finer. This accounts for part of the measured shortfall; the imperfect
# model accounts for the rest. (Exercise 4 lets you check it on MNIST by refining the steps.) For the
# perception–distortion picture below this is a detail, since the samples are still far more diverse than the single
# Tweedie estimate. Now the images and the sharpness side of the trade-off:

# %%
S_SHOW = SIGMAS[-2] if len(SIGMAS) > 1 else SIGMAS[0]
d = dn[S_SHOW]
show_n = min(4, N_DN)
image_grid([x_dn[:show_n], d["y"][:show_n], d["tweedie"][:show_n], d["samples"][0, :show_n], d["samples"][1, :show_n],
            d["samples"][:, :show_n].mean(0)], size=0.85,
           row_labels=["clean", f"noisy (σ = {S_SHOW:g})", "Tweedie (1 NFE)", "posterior sample 1",
                       "posterior sample 2", f"mean of {K_DN} samples"],
           title=f"Denoising at σ = {S_SHOW:g} (reverse process from t* = {d['t']})")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(5.5, 3.6))
for key, col, mk in [("tweedie", COL["Tweedie"], "o"), ("sample", COL["sample"], "s")]:
    ax.plot([dn[s][f"mse_{key}"] for s in SIGMAS], [dn[s][f"ge_{key}"] for s in SIGMAS], mk + "-", color=col,
            label="Tweedie (posterior mean)" if key == "tweedie" else "posterior sample")
    for s in SIGMAS:
        ax.annotate(f"σ={s:g}", (dn[s][f"mse_{key}"], dn[s][f"ge_{key}"]), textcoords="offset points",
                    xytext=(4, 4), fontsize=7)
ax.axhline(1.0, ls="--", color=COL["exact"], lw=1, label="sharpness of clean digits")
ax.set(xlabel="distortion: MSE to the clean image", ylabel="sharpness: gradient energy / clean",
       title="Perception–distortion trade-off", ylim=(0, 1.2),
       xlim=(0, 1.15 * max(dn[s]["mse_sample"] for s in SIGMAS)))
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# At every noise level the Tweedie estimate has the lower distortion (further left) but is too smooth: its gradient
# energy falls below that of real digits, more and more as $\sigma$ grows. Posterior samples keep close to real-digit
# sharpness but pay with substantially more MSE, because they commit to *one* plausible digit. At large $\sigma$ that
# digit may be a hallucination: a plausible stroke, or even a different digit, that was not in the original. Neither
# answer is wrong. Use Tweedie when you will average or measure, and samples when the output must look real.
#
# ## 6 · DDIM inversion: a deterministic latent space
#
# With $\eta=0$, DDIM is a discretised **ODE** (the probability-flow ODE of notebook 04). It maps each noise image
# $x_T$ to one image $x_0$, and running the same discretisation *forwards* maps an image back to "its" noise. The only
# approximation is that each forward step evaluates $\epsilon_\theta$ at the start of the step (the point we already
# know) instead of at its end:
# $$x_{t_{i+1}}=\sqrt{\bar\alpha_{t_{i+1}}}\,\hat x_0+\sqrt{1-\bar\alpha_{t_{i+1}}}\;\epsilon_\theta(x_{t_i},t_i),\qquad
# \hat x_0=\frac{x_{t_i}-\sqrt{1-\bar\alpha_{t_i}}\,\epsilon_\theta(x_{t_i},t_i)}{\sqrt{\bar\alpha_{t_i}}}.$$
# This error shrinks with more steps. `dm.ddim_invert` and `dm.ddim_sample(eta=0)` (notebook 04) implement the pair.
# Inversion gives every real image a **latent code** in which we can edit, as long as we decode with the same model,
# steps and conditioning.
#
# ### 6.1 Reconstruction

# %%
N_INV = 2 if FAST else 6
x_inv, lab_inv = x_test[20:20 + N_INV].to(device), y_test[20:20 + N_INV].to(device)
INV_STEPS = [5, 10] if FAST else [10, 20, 50]
inv = {}
t0 = time.time()
for n_inv in INV_STEPS:
    z_inv = dm.ddim_invert(model, schedule, x_inv, n_steps=n_inv, y=lab_inv)
    rec = dm.ddim_sample(model, schedule, x_inv.shape, n_steps=n_inv, x_T=z_inv, y=lab_inv)
    inv[n_inv] = dict(z=z_inv, rec=rec)
    print(f"{n_inv:3d} steps: reconstruction RMSE = {((rec - x_inv) ** 2).mean().sqrt():.3f}, PSNR = "
          f"{psnr(rec, x_inv):.1f} dB | latent: mean {z_inv.mean():+.2f}, std {z_inv.std():.2f}, "
          f"norm {z_inv.flatten(1).norm(dim=1).mean():.1f} (sqrt(d) = {math.sqrt(784):.0f})")
print(f"({time.time() - t0:.0f}s)")
n_hi, n_lo = INV_STEPS[-1], INV_STEPS[0]


# Is the inverted latent "typical" Gaussian noise? Compare its local statistics with a fresh N(0, I) sample.
white = torch.randn(inv[n_hi]["z"].shape, device=device, generator=make_gen(80))
ink = x_inv > 0                                                # the digit's stroke pixels
border = torch.ones(28, 28, dtype=torch.bool, device=device)
border[4:24, 4:24] = False                                     # the outer 4-pixel frame, always black in MNIST
for name, zz in [("inverted latents", inv[n_hi]["z"]), ("white noise N(0, I)", white)]:
    r = torch.corrcoef(torch.stack([zz.abs().flatten(), ink.float().flatten()]))[0, 1].item()
    print(f"{name:<20s}: std in the border frame {zz[..., border].std():.2f}, std on the strokes {zz[ink].std():.2f}, "
          f"corr(|z|, stroke mask) {r:+.2f}")
image_grid([x_inv, inv[n_hi]["z"].clamp(-2.5, 2.5) / 2.5, inv[n_hi]["rec"], inv[n_lo]["rec"]], size=0.85,
           row_labels=["original", f"inverted $x_T$ ({n_hi} steps)", f"reconstruction ({n_hi})",
                       f"reconstruction ({n_lo})"],
           title="DDIM inversion and reconstruction (conditioned on the true label)")
plt.tight_layout()
plt.show()

# %% [markdown]
# With enough steps the round trip reproduces the digit closely. With few steps the first-order approximation shows up
# as visible reconstruction errors. The latents have roughly the right summary statistics (mean ≈ 0, standard
# deviation ≈ 1, norm ≈ $\sqrt d$), but they are visibly **not** white noise: the digit's outline still shows, and the
# variance is not uniform. The printed statistics confirm it. Compared with fresh noise, the latent varies less in the
# always-black border and more on the strokes, and its magnitude correlates with where the ink is. A plausible reason
# is that the border is black in every MNIST digit, so the denoiser is nearly certain about it at every noise level,
# and the deterministic ODE pulls such pixels towards the mean of the noise instead of spreading them like typical
# noise. DDIM inversion finds *a* latent that decodes to the image, not a *typical* sample of $\mathcal N(0,I)$.
# Edits that assume typical Gaussian latents still work reasonably well on such codes (below), but they are less
# predictable.
#
# ### 6.2 Interpolation: why *slerp*, not *lerp*
#
# A standard Gaussian in $d=784$ dimensions concentrates on a thin shell: $\|\epsilon\|\approx\sqrt d$, with a
# standard deviation of only about $1/\sqrt2$. Two independent latents are nearly orthogonal, so their linear midpoint
# $\tfrac12(z_0+z_1)$ has norm $\approx\sqrt{d/2}\approx0.71\sqrt d$. That is far inside the shell, a "too quiet"
# noise image the model essentially never saw at $t=T$. Spherical interpolation moves along the great circle instead
# and keeps the norm:
# $$\mathrm{slerp}(z_0,z_1;\lambda)=\frac{\sin((1-\lambda)\theta)}{\sin\theta}\,z_0+\frac{\sin(\lambda\theta)}{\sin\theta}\,z_1,\qquad
# \cos\theta=\frac{\langle z_0,z_1\rangle}{\|z_0\|\,\|z_1\|}.$$

# %%
def slerp(z0, z1, lam):
    """Spherical interpolation between two batches of latents, per sample."""
    a, b = z0.flatten(1), z1.flatten(1)
    cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))
    theta = torch.arccos(cos.clamp(-1 + 1e-7, 1 - 1e-7))[:, None]
    out = (torch.sin((1 - lam) * theta) * a + torch.sin(lam * theta) * b) / torch.sin(theta)
    return out.view_as(z0)


def lerp(z0, z1, lam):
    return (1 - lam) * z0 + lam * z1


PAIRS = [(0, 1), (3, 4)]                                     # 7 -> 2 and 0 -> 4
LAMS = torch.linspace(0, 1, 5 if FAST else 8).tolist()
ends = x_test[[i for p in PAIRS for i in p]].to(device)
z_ends = dm.ddim_invert(model, schedule, ends, n_steps=STEPS)  # unconditional inversion (y = None)
z0s, z1s = z_ends[0::2], z_ends[1::2]
# Checks: slerp hits both end points, and keeps the norm where lerp does not.
assert torch.allclose(slerp(z0s, z1s, 0.0), z0s, atol=1e-4) and torch.allclose(slerp(z0s, z1s, 1.0), z1s, atol=1e-4)
for name, f in [("lerp", lerp), ("slerp", slerp)]:
    print(f"{name:>5s}: |z| along the path = " +
          " ".join(f"{f(z0s, z1s, l).flatten(1).norm(dim=1).mean():.1f}" for l in LAMS))
paths = {name: torch.cat([f(z0s, z1s, l) for l in LAMS]) for name, f in [("lerp", lerp), ("slerp", slerp)]}
t0 = time.time()
decoded = {name: dm.ddim_sample(model, schedule, zs.shape, n_steps=STEPS, x_T=zs).cpu() for name, zs in paths.items()}
print(f"decoded {2 * len(paths['lerp'])} latents in {time.time() - t0:.0f}s")
rows, labels = [], []
for p in range(len(PAIRS)):
    for name in ["lerp", "slerp"]:
        rows.append(decoded[name][p::len(PAIRS)])
        labels.append(f"{name}: {y_test[PAIRS[p][0]].item()} → {y_test[PAIRS[p][1]].item()}")
image_grid(rows, row_labels=labels, col_labels=[f"λ={l:.2f}" for l in LAMS], size=0.85,
           title="Interpolating DDIM latents")
plt.tight_layout()
plt.show()

# %% [markdown]
# The norms confirm the geometry: lerp dips well below $\sqrt d$ in the middle of the path, slerp stays on the shell.
# Decoded, the lerp midpoints tend to come out thinner and fainter, because the model reads a low-energy latent as an
# image with less ink. Slerp keeps normal stroke weight all the way. The path still has to cross between the two digit
# classes somewhere, so expect a few hybrid or third-class frames with either method.
#
# ### 6.3 Class swap: keep the style, change the label
#
# Invert a digit **with its own label** and decode the same latent **with a different label**. The latent keeps what
# the label does not explain (position, slant, thickness), and the label sets the identity. We reuse the inverted
# latents from 6.1.

# %%
SWAP_IDX = [0, 4, 5] if N_INV >= 6 else list(range(min(3, N_INV)))   # contrasting styles: a bold slanted 9,
n_swap = len(SWAP_IDX)                                                # a thin upright 4 and a wide, round 0
z_sw = inv[n_hi]["z"][SWAP_IDX]
targets = torch.arange(10, device=device)
t0 = time.time()
swapped = dm.ddim_sample(model, schedule, (10 * n_swap, 1, 28, 28), n_steps=n_hi,
                         x_T=z_sw.repeat_interleave(10, 0), y=targets.repeat(n_swap)).cpu()
acc_swap = (classify(swapped) == targets.cpu().repeat(n_swap)).float().mean()
print(f"{len(swapped)} decodes in {time.time() - t0:.0f}s; classified as the target label: {acc_swap:.2f}")
rows = [torch.cat([x_inv[i:i + 1].cpu(), swapped[10 * r:10 * (r + 1)]]) for r, i in enumerate(SWAP_IDX)]
image_grid(rows, col_labels=["original"] + [f"→ {k}" for k in range(10)], size=0.85,
           row_labels=[f"inverted as {lab_inv[i].item()}" for i in SWAP_IDX],
           title="Class swap: invert with the true label, decode with labels 0–9")
plt.tight_layout()
plt.show()

# %% [markdown]
# Each row shares the *style* of its original, such as the slant, the stroke width and the position in the frame,
# across all ten classes. The column with the original label reproduces the input. Some target classes fight the
# latent more than others (a wide, round style does not suit a "1"), and a guidance scale above 1 at decoding time
# would make the label win more often, at some cost in faithfulness.
#
# ## Exercises
#
# **Exercise 1 (when is replacement good enough?).** In the 2-D lab (section 2.3), change `C_OBS` to `0.0` and then
# to `1.8`. Before running, look at the mixture and predict how large replacement's error will be, compared with
# $x_1=-1$ and with the printed noise floor. Then vary the RePaint `jump` length (1, 5, 20) at a fixed `n_resample`.
# <details><summary>Hint / solution</summary>
# Replacement errs when, at the noise levels where $x_2$ commits to a mode, the noisy $x_1$ cannot rule out components
# that carry prior weight, are excluded by the exact conditional, and imply a different $x_2$ (section 2.1). At
# $x_1=-1$ the exact conditional sits almost entirely on $(-1,2)$, a light component (weight 1/9). Heavier or nearby
# mass, the class-1 component at $(0,-0.3)$ (weight 1/6) and the class-0 component at $(-2.2,-0.6)$, has $x_2$ far
# below, so the error is large. At $x_1=0$ the exact conditional is almost all on the heavy component $(0,-0.3)$, but
# its neighbours $(-1,2)$ and $(0.9,0.6)$ still attract mass: a clearly smaller error that is still far above the
# floor. At $x_1=1.8$ the nearby components are the ones in the conditional anyway (the class-2 cluster, which is
# isolated from the rest), so only their relative weights come out wrong. This gives the smallest error of the three,
# but it is still well above the floor. RePaint helps in all three cases. Jump length barely matters in this lab: 1 and
# 5 are about equal, and 20 was slightly worse in our runs. What reduces the error is the number of resampling passes
# $r$, and that costs NFE.
# </details>
#
# **Exercise 2 (your own sketch).** Use `draw_strokes` to draw a "1", a "0" (several short segments) or a "2", and find
# the smallest SDEdit strength that gives a realistic digit of the intended class. Repeat with `value=1.0,
# half_width=1.5` (bright, thick strokes). Predict first: will that strength go up or down?
# <details><summary>Hint / solution</summary>
# Bright, thick strokes are much closer to MNIST statistics, so less noise is needed to hide the remaining
# differences: the minimal strength goes down, and the output stays more faithful. The default grey strokes need more
# noise, because the model must "repaint" the stroke intensity and width, which are moderately coarse features.
# </details>
#
# **Exercise 3 (DPS knobs).** For super-resolution (4.2), raise `SIGMA_Y` to 0.2 (both when making `y_sr` and when
# judging the residual) and keep `ZETA_SR`. A noisier measurement is less informative, so the true posterior is wider.
# Does DPS's diversity grow accordingly? Check the residual/noise ratio and the posterior-std map. Then lower `ZETA_SR`
# until the residual is about the noise level, and look again. Finally, run the 2-D lab's `weighting="bayes"` rule on
# MNIST super-resolution with `zeta=1.0, sigma_y=0.05` and 100 steps.
# <details><summary>Hint / solution</summary>
# At fixed $\zeta$ the diversity grows only a little, and the residual/noise ratio drops below 1. The normalised rule
# does not know $\sigma_y$ (section 4), so it keeps pulling the sample towards $y$ and fits the extra noise instead of
# widening. Lowering $\zeta$ (roughly halving it) brings the ratio back to about 1. Now the std map brightens clearly
# and identities can start to vary: $\zeta$ is a likelihood temperature that must follow $\sigma_y$. With the literal
# Bayes weight and $\sigma_y=0.05$, the correction $\propto 1/\sigma_y^2=400$ is far too large for 100 strided steps,
# and the samples blow up to NaN. This is why DPS normalises the step.
# </details>
#
# **Exercise 4 (denoising knobs).** (a) In section 5, denoise $\sigma=1.0$ images but choose $t^\star$ for
# $\sigma=0.5$ (too little) and for $\sigma=2.0$ (too much). Predict what each output looks like, then check, and
# compare the PSNR of the Tweedie estimates. (b) Set `DN_STRIDE = 2` (five times finer steps, and five times slower)
# and re-run the denoising cell. Predict how the three printed ratios move.
# <details><summary>Hint / solution</summary>
# (a) Underestimating $\sigma$ means the model assumes less noise than there is: residual noise survives, or is turned
# into spurious little strokes. Overestimating means it assumes more noise: the Tweedie estimate gets over-smoothed
# (blurrier, more "average digit"), and the samples hallucinate more freely. The PSNR peaks near the true $\sigma$.
# (b) Finer steps remove most of the sampler's under-dispersion (the closed-form check says how much), so the spread
# ratio and MSE(sample)/MSE(Tweedie) should move towards 1 and 2. What remains is the model's own error. With six
# digits the change can be partly hidden by noise; raise `N_DN` for cleaner numbers.
# </details>
#
# **Exercise 5 (inpainting with DPS).** Solve the bottom-half and 85%-random inpainting tasks of section 2.4 with
# `dps_sample`, using `A = lambda x: x * mask` and `y = mask * x_true` (no noise; the normalised rule does not use
# $\sigma_y$ anyway). Compare with RePaint on the classifier metric and on the exact agreement with the known pixels,
# $\|m\odot(x-y)\|$.
# <details><summary>Hint / solution</summary>
# RePaint copies the known pixels exactly at the last step, so its agreement is perfect by construction. DPS only
# *pushes* towards them. With a fixed $\zeta$ its final residual is of order $\zeta/2$ (section 4), small but
# non-zero: a smaller $\zeta$ fits more tightly at the end but pulls less early on. In exchange DPS works for any differentiable $A$,
# where RePaint needs $A$ to be a pixel mask. Which one completes digits more coherently depends on $\zeta$ and the
# number of steps; on a GPU try both with 250+ steps.
# </details>
#
# **Exercise 6 (inversion under guidance).** Invert and reconstruct the digits of 6.1 with `guidance_scale=3.0`
# in both `dm.ddim_invert` and `dm.ddim_sample`. How does the reconstruction error change?
# <details><summary>Hint / solution</summary>
# It grows noticeably. The inversion assumes $\epsilon(x_{t_{i+1}})\approx\epsilon(x_{t_i})$. Guidance amplifies how
# fast the (guided) $\epsilon$ changes along the path, so the first-order error per step is larger and accumulates.
# This is a known practical problem for real-image editing with text-to-image models. Null-text inversion (Mokady et
# al. 2023) fixes it by optimising the unconditional embedding per step.
# </details>
#
# ## Key takeaways
#
# - A trained diffusion model is a **reusable prior**. Its denoiser $\hat x_0=\mathbb E[x_0\mid x_t]$ and its scores
#   can be combined with a measurement at sampling time, with no retraining.
# - Posterior sampling needs $\nabla\log p_t(y\mid x_t)$, which is intractable in general. Methods differ in how they
#   approximate or side-step it. The 2-D lab shows that replacement, RePaint (less so) and DPS are all biased
#   **even with a perfect prior**: their errors are method errors, not model errors.
# - **Inpainting:** replacement is cheap but conditions on noisy known pixels, which gives seams and incoherence, and
#   more steps do not fix it. **RePaint**'s noise-up/denoise-down resampling reduces the bias at a cost of about
#   $r\times$ more NFE, but slowly, with no exactness guarantee.
# - **SDEdit:** noise the guide to $t_0$ and denoise. The strength $t_0/T$ trades faithfulness against realism.
#   Changing the conditioning during the reverse pass edits the content while keeping the layout.
# - **DPS:** $p(y\mid x_t)\approx p(y\mid\hat x_0(x_t))$ and a gradient through the network make *any*
#   differentiable linear (or nonlinear) operator usable. The step size $\zeta$ is the key knob. In the normalised
#   rule it replaces $\sigma_y$ (a likelihood temperature) and must be re-tuned when the noise level changes. Aim for
#   a residual of about the noise level. Each step costs a backward pass.
# - **Denoising** with known Gaussian noise is exact: rescale and start at $t^\star$. The posterior mean (Tweedie)
#   minimises MSE but is blurry; posterior samples are sharp but have about twice the MSE (perception–distortion).
#   Coarse ancestral samplers under-disperse even with a perfect model, so their samples fall short of that factor.
# - **DDIM inversion** gives real images latent codes with roughly Gaussian summary statistics (but visibly atypical
#   structure) that support reconstruction, slerp interpolation and class swaps. More steps mean better inversion,
#   and guidance makes it worse.
#
# ## Further reading
#
# * Lugmayr, Danelljan, Romero, Yu, Timofte, Van Gool (2022). *RePaint: Inpainting using Denoising Diffusion
#   Probabilistic Models.* CVPR. arXiv:2201.09865.
# * Wu, Trippe, Naesseth, Blei, Cunningham (2023). *Practical and Asymptotically Exact Conditional Sampling in
#   Diffusion Models.* NeurIPS. arXiv:2306.17775. Twisted sequential Monte Carlo: conditional sampling from a
#   diffusion prior that becomes exact as the number of particles grows.
# * Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2021). *Score-Based Generative Modeling through Stochastic
#   Differential Equations.* ICLR. arXiv:2011.13456. Includes imputation by replacement and solving inverse problems
#   with a score model.
# * Meng, He, Song, Song, Wu, Zhu, Ermon (2022). *SDEdit: Guided Image Synthesis and Editing with Stochastic
#   Differential Equations.* ICLR. arXiv:2108.01073.
# * Chung, Kim, Mccann, Klasky, Ye (2023). *Diffusion Posterior Sampling for General Noisy Inverse Problems.* ICLR.
#   arXiv:2209.14687.
# * Kawar, Elad, Ermon, Song (2022). *Denoising Diffusion Restoration Models.* NeurIPS. And Song, Vahdat, Mardani,
#   Kautz (2023). *Pseudoinverse-Guided Diffusion Models for Inverse Problems.* ICLR. These use better approximations
#   of $p(y\mid x_t)$ for linear problems.
# * Efron (2011). *Tweedie's formula and selection bias.* Journal of the American Statistical Association.
# * Bao, Li, Zhu, Zhang (2022). *Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion
#   Probabilistic Models.* ICLR. arXiv:2201.06503. Why the reverse variance must include the uncertainty of
#   $\hat x_0$, which matters most for samplers with few steps.
# * Blau & Michaeli (2018). *The Perception-Distortion Tradeoff.* CVPR.
# * Song, Meng, Ermon (2021). *Denoising Diffusion Implicit Models.* ICLR. arXiv:2010.02502. Deterministic sampling,
#   inversion and latent interpolation.
# * Mokady, Hertz, Aberman, Pritch, Cohen-Or (2023). *Null-text Inversion for Editing Real Images using Guided
#   Diffusion Models.* CVPR.
# * Ho & Salimans (2022). *Classifier-Free Diffusion Guidance.* arXiv:2207.12598.
# * Kadkhodaie, Guth, Simoncelli, Mallat (2024). *Generalization in diffusion models arises from geometry-adaptive
#   harmonic representations.* ICLR. arXiv:2310.02557. What kind of prior a trained denoiser actually is.
# * Sander Dieleman, *Diffusion is spectral autoregression* (sander.ai, 2024), on why noise erases fine details first
#   (the intuition behind SDEdit), and *Guidance: a cheat code for diffusion models* (sander.ai, 2022).
# * Yang Song, *Generative Modeling by Estimating Gradients of the Data Distribution* (blog post, 2021). Includes
#   inverse problems with score models.
