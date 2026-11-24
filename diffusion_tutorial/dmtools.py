"""
dmtools -- shared helpers for the diffusion-model tutorial.

Every algorithm in this file is *derived and written out by hand* in one of the
notebooks first. This module only collects the final versions so that later
notebooks can `import dmtools` instead of re-typing a U-Net or a sampler.

Where each piece is introduced:

    NoiseSchedule (linear and cosine schedules),
      GaussianMixture.noised / .noised_ve / .responsibilities ... 01_forward_process
    SinusoidalEmbedding, MLPDenoiser, diffusion_loss, EMA,
      train_diffusion (the loop), ddpm_sample, model_predictions,
      mmd_rbf, sliced_wasserstein, make_toy ..................... 02_ddpm_from_scratch
    GaussianMixture.score, Tweedie's formula .................... 03_score_matching
    ddim_sample, ddim_invert .................................... 04_samplers
    UNet, the full training recipe (AdamW, warm-up, bf16 `amp`),
      load_mnist, MNISTClassifier, load_or_train_* .............. 05_images_unet
    classifier-free guidance (guidance_scale), labelled_gmm ..... 06_guidance
    (flow matching lives entirely in 08_flow_matching)

Conventions used everywhere in the tutorial
-------------------------------------------
* Discrete time t is an integer array index in {0, ..., T-1}, one less than
  the maths' t in {1, ..., T}: `schedule.alpha_bars[t]` is
  prod_{s=0..t} (1 - betas[s]), i.e. the maths' abar_{t+1}, and
  `alpha_bars_prev[0]` = 1 belongs to the data.  t = T-1 is (almost) pure noise.
* Forward process ("variance preserving"):
      x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps,   eps ~ N(0, I)
* Every model has the signature `model(x, t, y=None)`; `t` is a float or long
  tensor of shape (B,) and `y` an optional long tensor of class labels.
  For classifier-free guidance the label `n_classes` means "no label" (null).
* Prediction targets ("parameterisations"):  'eps' (noise), 'x0' (clean data),
  'v' (v = sqrt(abar) * eps - sqrt(1-abar) * x0).
* guidance_scale w follows the diffusers convention:
      eps = eps_uncond + w * (eps_cond - eps_uncond)
  so w = 0 is unconditional, w = 1 is plain conditional, w > 1 is "guided"
  (Ho & Salimans 2022 write 1 + w for this w).
* Flow matching (notebook 08 only) uses continuous t in [0, 1] with data at
  t = 0 and noise at t = 1: x_t = (1 - t) x_0 + t eps. Its MLPs are fed 1000 t.
* Images live in [-1, 1].
* Exact (float64) work happens on the CPU: GaussianMixture keeps its
  parameters as CPU float64 tensors, and NoiseSchedule stores float32 copies on
  its device. For a float64 copy write `schedule.alpha_bars.cpu().double()`
  (in that order: Apple's MPS backend has no float64).
"""

from __future__ import annotations

import contextlib
import copy
import math
import os
import time
import warnings
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"

__all__ = [
    "get_device", "seed_everything", "count_params", "fast_mode",
    "make_toy", "GaussianMixture", "default_gmm", "labelled_gmm",
    "NoiseSchedule", "SinusoidalEmbedding", "MLPDenoiser", "UNet", "EMA",
    "diffusion_loss", "train_diffusion", "model_predictions",
    "ddpm_sample", "ddim_sample", "ddim_invert",
    "mmd_rbf", "sliced_wasserstein",
    "load_mnist", "MNISTClassifier", "load_or_train_classifier",
    "load_or_train_mnist_model",
    "show_images", "scatter2d", "plot_vector_field", "plot_trajectories",
]


# ----------------------------------------------------------------------------
# Misc utilities
# ----------------------------------------------------------------------------

def get_device(verbose: bool = False) -> torch.device:
    """Best available device: CUDA GPU > Apple-silicon GPU (MPS) > CPU.

    Set the environment variable DM_DEVICE to override the choice, e.g. DM_DEVICE=cpu (in a notebook, run
    `%env DM_DEVICE=cpu` before the setup cell).
    """
    forced = os.environ.get("DM_DEVICE", "").strip()
    if forced:
        dev = torch.device(forced)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    if verbose:
        print(f"Using device: {dev}")
    return dev


def seed_everything(seed: int = 0) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    """Number of *trainable* parameters (requires_grad=True).

    Returns 0 for a frozen copy such as `EMA.model`; count those with
    `sum(p.numel() for p in model.parameters())`.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fast_mode() -> bool:
    """True when the environment variable DM_FAST=1 is set.

    Notebooks use this to shrink training runs to a smoke test (used by the
    automated checks that execute every notebook). Leave it unset to learn.
    """
    return os.environ.get("DM_FAST", "0") not in ("", "0", "false", "False")


# ----------------------------------------------------------------------------
# 2-D toy data
# ----------------------------------------------------------------------------

def make_toy(name: str, n: int, seed: Optional[int] = None, noise: float = None) -> torch.Tensor:
    """Sample `n` points from a named 2-D toy distribution (roughly unit scale).

    names: 'swiss_roll', 'moons', 'eight_gaussians', 'checkerboard', 'rings', 'gmm'
    """
    rng = np.random.default_rng(seed)
    if name == "swiss_roll":
        t = 1.5 * np.pi * (1 + 2 * rng.random(n))
        x = np.stack([t * np.cos(t), t * np.sin(t)], 1) / 5.0
        x += (0.05 if noise is None else noise) * rng.standard_normal((n, 2))
    elif name == "moons":
        n1 = n // 2
        n2 = n - n1
        a = np.pi * rng.random(n1)
        b = np.pi * rng.random(n2)
        x = np.concatenate([
            np.stack([np.cos(a), np.sin(a)], 1),
            np.stack([1 - np.cos(b), 0.5 - np.sin(b)], 1),
        ])
        x = (x - np.array([0.5, 0.25])) * 1.5
        x += (0.06 if noise is None else noise) * rng.standard_normal((n, 2))
        x = x[rng.permutation(n)]
    elif name == "eight_gaussians":
        ang = 2 * np.pi * rng.integers(0, 8, n) / 8
        x = 2.0 * np.stack([np.cos(ang), np.sin(ang)], 1)
        x += (0.1 if noise is None else noise) * rng.standard_normal((n, 2))
    elif name == "checkerboard":
        x1 = rng.random(n) * 4 - 2
        x2 = rng.random(n) - rng.integers(0, 2, n) * 2.0
        x2 = x2 + (np.floor(x1) % 2)
        x = np.stack([x1, x2], 1)
    elif name == "rings":
        r = rng.choice([0.6, 1.3, 2.0], size=n)
        ang = 2 * np.pi * rng.random(n)
        x = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
        x += (0.06 if noise is None else noise) * rng.standard_normal((n, 2))
    elif name == "gmm":
        g = torch.Generator().manual_seed(int(rng.integers(0, 2**31 - 1)))
        return default_gmm().sample(n, generator=g)
    else:
        raise ValueError(f"unknown toy dataset {name!r}")
    return torch.as_tensor(x, dtype=torch.float32)


class GaussianMixture:
    r"""A mixture of Gaussians in d dimensions with *exact* density and score.

    This is the tutorial's "ground truth" laboratory: because a Gaussian
    convolved with a Gaussian is Gaussian, the noised marginals q_t(x_t) of a
    GMM are again GMMs, so we know the exact score \nabla log q_t at every t.

    Parameters live on the CPU in float64. Inputs may be on any device:
    `component_log_probs` returns a CPU float64 tensor, `log_prob` and
    `responsibilities` return float32 on the input's device (for float64 work use
    `torch.logsumexp(gmm.component_log_probs(x), 1)`), and `score` returns the
    input's dtype and device.
    """

    def __init__(self, weights, means, covs):
        self.weights = torch.as_tensor(weights, dtype=torch.float64)
        self.weights = self.weights / self.weights.sum()
        self.means = torch.as_tensor(means, dtype=torch.float64)
        K, d = self.means.shape
        covs = torch.as_tensor(covs, dtype=torch.float64)
        if covs.ndim == 0:
            covs = covs * torch.eye(d).expand(K, d, d).clone()
        elif covs.ndim == 1:  # one isotropic variance per component
            covs = covs[:, None, None] * torch.eye(d)
        self.covs = covs
        self.K, self.d = K, d
        self._chol = torch.linalg.cholesky(self.covs)
        self._prec = torch.cholesky_inverse(self._chol)
        self._logdet = 2 * torch.log(torch.diagonal(self._chol, dim1=-2, dim2=-1)).sum(-1)

    # -- sampling -------------------------------------------------------------
    def sample(self, n: int, return_labels: bool = False, generator=None):
        k = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        z = torch.randn(n, self.d, dtype=torch.float64, generator=generator)
        x = self.means[k] + torch.einsum("nij,nj->ni", self._chol[k], z)
        x = x.float()
        return (x, k) if return_labels else x

    # -- density ---------------------------------------------------------------
    def component_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """log w_k + log N(x; mu_k, Sigma_k), shape (n, K), CPU float64."""
        x = x.to("cpu", torch.float64)
        diff = x[:, None, :] - self.means[None]                     # (n,K,d)
        maha = torch.einsum("nki,kij,nkj->nk", diff, self._prec, diff)
        log_norm = -0.5 * (self.d * math.log(2 * math.pi) + self._logdet)
        return torch.log(self.weights)[None] + log_norm[None] - 0.5 * maha

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(self.component_log_probs(x), dim=1).float().to(x.device)

    def responsibilities(self, x: torch.Tensor) -> torch.Tensor:
        """Posterior p(k | x) over mixture components, shape (n, K)."""
        return torch.softmax(self.component_log_probs(x), dim=1).float().to(x.device)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        r"""Exact score \nabla_x log p(x) = sum_k r_k(x) Sigma_k^{-1} (mu_k - x)."""
        x64 = x.to("cpu", torch.float64)
        r = torch.softmax(self.component_log_probs(x64), dim=1)     # (n,K)
        diff = self.means[None] - x64[:, None, :]                   # (n,K,d)
        per_comp = torch.einsum("kij,nkj->nki", self._prec, diff)   # (n,K,d)
        return (r[..., None] * per_comp).sum(1).to(x.dtype).to(x.device)

    # -- noising ---------------------------------------------------------------
    def noised(self, alpha_bar: float) -> "GaussianMixture":
        r"""Marginal of x_t = sqrt(abar) x_0 + sqrt(1-abar) eps  (variance preserving)."""
        a = float(alpha_bar)
        I = torch.eye(self.d, dtype=torch.float64)
        return GaussianMixture(self.weights, math.sqrt(a) * self.means, a * self.covs + (1 - a) * I)

    def noised_ve(self, sigma: float) -> "GaussianMixture":
        r"""Marginal of x_sigma = x_0 + sigma * eps  (variance exploding)."""
        I = torch.eye(self.d, dtype=torch.float64)
        return GaussianMixture(self.weights, self.means, self.covs + float(sigma) ** 2 * I)

    def subset(self, idx: Sequence[int]) -> "GaussianMixture":
        idx = torch.as_tensor(list(idx))
        return GaussianMixture(self.weights[idx], self.means[idx], self.covs[idx])

    def __repr__(self):
        return f"GaussianMixture(K={self.K}, d={self.d})"


def default_gmm() -> GaussianMixture:
    """A 2-D, 8-component mixture with unequal weights and anisotropic shapes."""
    ang = torch.arange(8, dtype=torch.float64) * (2 * math.pi / 8)
    means = torch.stack([2.0 * torch.cos(ang), 2.0 * torch.sin(ang)], 1)
    means[::2] *= 0.6  # alternate inner/outer ring -> a "star" shape
    weights = torch.tensor([3, 1, 3, 1, 3, 1, 3, 1], dtype=torch.float64)
    covs = []
    for k in range(8):
        R = torch.tensor([[math.cos(ang[k]), -math.sin(ang[k])], [math.sin(ang[k]), math.cos(ang[k])]],
                         dtype=torch.float64)
        D = torch.diag(torch.tensor([0.08, 0.01] if k % 2 else [0.02, 0.02], dtype=torch.float64))
        covs.append(R @ D @ R.T)
    return GaussianMixture(weights, means, torch.stack(covs))


def labelled_gmm():
    """A 2-D GMM whose components are grouped into 3 classes (for guidance demos).

    Returns (gmm, comp_class) where comp_class[k] is the class of component k.
    Class-conditional distributions are `gmm.subset(np.where(comp_class == c)[0])`.
    """
    means = torch.tensor([
        [-2.0, 1.0], [-1.0, 2.0], [-2.2, -0.6],   # class 0 (left / top-left)
        [0.0, -0.3], [0.9, 0.6],                   # class 1 (centre, overlaps class 2)
        [2.0, 1.2], [1.6, -1.6], [2.4, -0.2],     # class 2 (right)
    ], dtype=torch.float64)
    comp_class = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    weights = torch.ones(len(means), dtype=torch.float64)
    for c in range(3):  # equal class priors
        m = torch.as_tensor(comp_class == c)
        weights[m] = 1.0 / m.sum()
    covs = torch.full((len(means),), 0.12, dtype=torch.float64)
    return GaussianMixture(weights, means, covs), comp_class


# ----------------------------------------------------------------------------
# Noise schedule (discrete time, variance preserving)
# ----------------------------------------------------------------------------

class NoiseSchedule:
    r"""Precomputes every coefficient of the DDPM forward / reverse process.

    kind='linear': beta_t linearly spaced in [beta_start, beta_end] (Ho et al. 2020)
    kind='cosine': abar_t = f(t)/f(0), f(t) = cos^2(((t/T + s)/(1 + s)) * pi/2), with
                   maths t = 1..T (Nichol & Dhariwal 2021)

    The linear betas are rescaled by 1000/T so that every T approximates the same
    continuous process. This breaks down for small T: at T <= 20 some beta_t >= 1,
    which is not a valid schedule (a warning is raised). To view the T=1000
    process at fewer steps, subsample its alpha_bars instead ("respacing", see
    notebooks 01 and 02). The cosine schedule clamps beta_t at 0.999, so for small T
    its last alpha_bar is tiny and 1/sqrt(alpha_bar) is large.

    All coefficients are computed in float64 and stored as float32 on `device`.
    """

    def __init__(self, T: int = 1000, kind: str = "linear", beta_start: float = 1e-4,
                 beta_end: float = 0.02, cosine_s: float = 0.008, device=None):
        self.T, self.kind = T, kind
        if kind == "linear":
            # scale so that any T gives the same continuous-time process as T=1000
            scale = 1000.0 / T
            betas = torch.linspace(scale * beta_start, scale * beta_end, T, dtype=torch.float64)
        elif kind == "cosine":
            steps = torch.arange(T + 1, dtype=torch.float64) / T
            f = torch.cos((steps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2
            abar = f / f[0]
            betas = (1 - abar[1:] / abar[:-1]).clamp(0, 0.999)
        else:
            raise ValueError(kind)
        if (betas >= 1).any():
            warnings.warn(f"NoiseSchedule(T={T}, kind={kind!r}): max beta = {betas.max().item():.2f} >= 1, so "
                          "alpha_bar hits 0 or goes negative (NaNs downstream). Use a larger T, or respace the "
                          "T=1000 schedule.", stacklevel=2)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, 0)
        abar_prev = torch.cat([torch.ones(1, dtype=torch.float64), abar[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = abar
        self.alpha_bars_prev = abar_prev
        self.sqrt_alpha_bars = abar.sqrt()
        self.sqrt_one_minus_alpha_bars = (1 - abar).sqrt()
        # q(x_{t-1} | x_t, x_0) = N(coef_x0 * x_0 + coef_xt * x_t, posterior_variance)
        self.posterior_variance = betas * (1 - abar_prev) / (1 - abar)
        self.posterior_coef_x0 = betas * abar_prev.sqrt() / (1 - abar)
        self.posterior_coef_xt = (1 - abar_prev) * alphas.sqrt() / (1 - abar)
        self.log_snr = torch.log(abar) - torch.log1p(-abar)
        self.to(device or "cpu")

    def to(self, device):
        self.device = torch.device(device)
        for k, v in list(vars(self).items()):
            if torch.is_tensor(v):
                # float32 copies for use inside the networks / samplers
                setattr(self, k, v.to(self.device, torch.float32))
        return self

    @staticmethod
    def _bcast(coef: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """coef[t] reshaped to broadcast against x (B, ...)."""
        return coef[t.long()].view(-1, *([1] * (x.ndim - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """Sample x_t ~ q(x_t | x_0) in closed form."""
        if noise is None:
            noise = torch.randn_like(x0)
        return self._bcast(self.sqrt_alpha_bars, t, x0) * x0 + \
            self._bcast(self.sqrt_one_minus_alpha_bars, t, x0) * noise

    def snr(self, t):
        ab = self.alpha_bars[t.long() if torch.is_tensor(t) else t]
        return ab / (1 - ab)

    def __repr__(self):
        return f"NoiseSchedule(T={self.T}, kind={self.kind!r})"


# ----------------------------------------------------------------------------
# Networks
# ----------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Transformer-style embedding of a scalar time t -> R^dim."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim, self.max_period = dim, max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) *
                          torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class MLPDenoiser(nn.Module):
    """Residual MLP eps_theta(x, t[, y]) for low-dimensional (e.g. 2-D) data."""

    def __init__(self, data_dim: int = 2, hidden: int = 256, n_blocks: int = 3,
                 emb_dim: int = 64, n_classes: Optional[int] = None):
        super().__init__()
        self.data_dim, self.n_classes = data_dim, n_classes
        self.time_emb = nn.Sequential(SinusoidalEmbedding(emb_dim), nn.Linear(emb_dim, hidden),
                                      nn.SiLU(), nn.Linear(hidden, hidden))
        # one extra row = the "null" label used for classifier-free guidance
        self.class_emb = nn.Embedding(n_classes + 1, hidden) if n_classes else None
        self.inp = nn.Linear(data_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden))
            for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, data_dim))

    def forward(self, x, t, y=None):
        emb = self.time_emb(t)
        if self.class_emb is not None:
            if y is None:
                y = torch.full((x.shape[0],), self.n_classes, device=x.device, dtype=torch.long)
            emb = emb + self.class_emb(y)
        h = self.inp(x)
        for blk in self.blocks:
            h = h + blk(h + emb)
        return self.out(h)


def _groups(ch: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0 and ch // g >= 4:
            return g
    return 1


class ResBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv, with the time/class embedding injected by FiLM."""

    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)              # -> (scale, shift)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)                      # block starts as identity
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Multi-head self-attention over spatial positions (used at low resolution)."""

    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(_groups(ch), ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, emb=None):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
        h = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        h = h.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(h)


class UNet(nn.Module):
    """A small DDPM-style U-Net for 28x28 (or any size divisible by 2**(levels-1)) images.

    in_ch       : image channels
    base_ch     : channels at full resolution
    ch_mults    : channel multiplier per resolution level (each level halves H, W)
    n_res       : residual blocks per level
    attn_levels : levels (0 = full res) that get self-attention
    n_classes   : if given, the model is class-conditional with an extra null label
    """

    def __init__(self, in_ch=1, base_ch=32, ch_mults=(1, 2, 2), n_res=2, attn_levels=(2,),
                 dropout=0.1, n_classes: Optional[int] = None, out_ch: Optional[int] = None):
        super().__init__()
        self.n_classes = n_classes
        self.config = dict(in_ch=in_ch, base_ch=base_ch, ch_mults=tuple(ch_mults), n_res=n_res,
                           attn_levels=tuple(attn_levels), dropout=dropout, n_classes=n_classes,
                           out_ch=out_ch)
        emb_dim = 4 * base_ch
        self.time_emb = nn.Sequential(SinusoidalEmbedding(base_ch), nn.Linear(base_ch, emb_dim),
                                      nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.class_emb = nn.Embedding(n_classes + 1, emb_dim) if n_classes else None

        self.inp = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.down = nn.ModuleList()
        skip_chs, ch = [base_ch], base_ch
        for lvl, m in enumerate(ch_mults):
            for _ in range(n_res):
                blocks = [ResBlock(ch, base_ch * m, emb_dim, dropout)]
                ch = base_ch * m
                if lvl in attn_levels:
                    blocks.append(AttnBlock(ch))
                self.down.append(nn.ModuleList(blocks))
                skip_chs.append(ch)
            if lvl != len(ch_mults) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                skip_chs.append(ch)
        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_dim, dropout), AttnBlock(ch),
                                  ResBlock(ch, ch, emb_dim, dropout)])
        self.up = nn.ModuleList()
        for lvl, m in reversed(list(enumerate(ch_mults))):
            for i in range(n_res + 1):
                blocks = [ResBlock(ch + skip_chs.pop(), base_ch * m, emb_dim, dropout)]
                ch = base_ch * m
                if lvl in attn_levels:
                    blocks.append(AttnBlock(ch))
                if lvl != 0 and i == n_res:
                    blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
                    blocks.append(nn.Conv2d(ch, ch, 3, padding=1))
                self.up.append(nn.ModuleList(blocks))
        self.out = nn.Sequential(nn.GroupNorm(_groups(ch), ch), nn.SiLU(),
                                 nn.Conv2d(ch, out_ch or in_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    @staticmethod
    def _run(blocks, h, emb):
        for b in blocks:
            h = b(h, emb) if isinstance(b, (ResBlock, AttnBlock)) else b(h)
        return h

    def forward(self, x, t, y=None):
        emb = self.time_emb(t)
        if self.class_emb is not None:
            if y is None:
                y = torch.full((x.shape[0],), self.n_classes, device=x.device, dtype=torch.long)
            emb = emb + self.class_emb(y)
        h = self.inp(x)
        hs = [h]
        for blocks in self.down:
            h = self._run(blocks, h, emb)
            hs.append(h)
        h = self._run(self.mid, h, emb)
        for blocks in self.up:
            h = torch.cat([h, hs.pop()], dim=1)
            h = self._run(blocks, h, emb)
        return self.out(h)


class EMA:
    """Exponential moving average of model weights -- the weights you sample with.

    Usage:  ema = EMA(model); ...; opt.step(); ema.update(model); ...; ema.model(x, t)

    Warm-up: update n uses d_n = min(decay, (1 + n) / (10 + n)), so decay=0.999 is
    reached only after ~9000 updates. In shorter runs the warm-up sets the
    effective averaging window (d ~ 0.998 after 4000 updates), not 1/(1 - decay).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        # warm-up: average over the first few steps uniformly, then use `decay`
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def _pred_to_target(pred: str, x0, noise, sa, s1a):
    if pred == "eps":
        return noise
    if pred == "x0":
        return x0
    if pred == "v":
        return sa * noise - s1a * x0
    raise ValueError(pred)


def diffusion_loss(model, x0, schedule: NoiseSchedule, y=None, pred: str = "eps",
                   p_uncond: float = 0.0, t=None):
    """The 'simple' DDPM loss  E_{t, x0, eps} || target - model(x_t, t) ||^2."""
    B = x0.shape[0]
    if t is None:
        t = torch.randint(0, schedule.T, (B,), device=x0.device)
    noise = torch.randn_like(x0)
    xt = schedule.q_sample(x0, t, noise)
    if y is not None and p_uncond > 0:
        drop = torch.rand(B, device=x0.device) < p_uncond
        y = torch.where(drop, torch.full_like(y, model.n_classes), y)
    out = model(xt, t, y)
    sa = schedule._bcast(schedule.sqrt_alpha_bars, t, x0)
    s1a = schedule._bcast(schedule.sqrt_one_minus_alpha_bars, t, x0)
    target = _pred_to_target(pred, x0, noise, sa, s1a)
    return F.mse_loss(out, target)


def train_diffusion(model, data: torch.Tensor, schedule: NoiseSchedule, *, labels=None,
                    steps: int = 2000, batch_size: int = 128, lr: float = 2e-4,
                    pred: str = "eps", p_uncond: float = 0.0, ema_decay: float = 0.999,
                    grad_clip: float = 1.0, warmup: int = 100, weight_decay: float = 0.0,
                    log_every: int = 0, device=None, progress: bool = True, amp=None):
    """Generic training loop used for both the 2-D MLP and the image U-Net.

    Returns a dict with 'losses' (per step), 'ema' (EMA wrapper; use ema.model
    for sampling) and 'seconds'.

    amp: mixed precision. None = automatic (bfloat16 autocast on CUDA GPUs that
    support it, off elsewhere); True = bfloat16 autocast on any device (also
    speeds up recent CPUs ~2x); False = full float32.
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device
    if amp is None:
        amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model.to(device).train()
    data = data.to(device)
    labels = labels.to(device) if labels is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warmup)))
    ema = EMA(model, ema_decay)
    losses = []
    it = range(steps)
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(it, desc="training", leave=False)
            it = bar
        except ImportError:
            pass
    t0 = time.time()
    for step in it:
        idx = torch.randint(0, data.shape[0], (batch_size,), device=device)
        y = labels[idx] if labels is not None else None
        ctx = torch.autocast(device.type, dtype=torch.bfloat16) if amp else contextlib.nullcontext()
        with ctx:
            loss = diffusion_loss(model, data[idx], schedule, y=y, pred=pred, p_uncond=p_uncond)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()
        ema.update(model)
        losses.append(loss.item())
        if bar is not None and step % 50 == 0:
            bar.set_postfix(loss=f"{np.mean(losses[-50:]):.4f}")
        if log_every and (step + 1) % log_every == 0:
            print(f"step {step + 1:6d} | loss {np.mean(losses[-log_every:]):.4f}")
    model.eval()
    return {"losses": np.array(losses), "ema": ema, "seconds": time.time() - t0}


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------

@torch.no_grad()
def model_predictions(model, x, t, schedule: NoiseSchedule, y=None, guidance_scale: float = 1.0,
                      pred: str = "eps", clip_x0: Optional[float] = None):
    """Run the network and return (eps_hat, x0_hat), whatever it was trained to predict.

    Handles classifier-free guidance: if `y` is given and guidance_scale != 1,
    the model is evaluated with and without the label and the two noise
    predictions are extrapolated. Network outputs are cast to x's dtype first, so
    under bfloat16 autocast the guidance extrapolation happens in float32.
    """
    if not torch.is_tensor(t) or t.ndim == 0:
        t = torch.full((x.shape[0],), int(t), device=x.device, dtype=torch.long)
    sa = schedule._bcast(schedule.sqrt_alpha_bars, t, x)
    s1a = schedule._bcast(schedule.sqrt_one_minus_alpha_bars, t, x)

    def to_eps(out, xx):
        if pred == "eps":
            return out
        if pred == "x0":
            return (xx - sa * out) / s1a
        if pred == "v":
            return s1a * xx + sa * out
        raise ValueError(pred)

    if y is not None and guidance_scale != 1.0:
        null = torch.full_like(y, model.n_classes)
        out = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null]))
        out_c, out_u = out.to(x.dtype).chunk(2)
        eps_c, eps_u = to_eps(out_c, x), to_eps(out_u, x)
        eps = eps_u + guidance_scale * (eps_c - eps_u)
    else:
        eps = to_eps(model(x, t, y).to(x.dtype), x)
    x0 = (x - s1a * eps) / sa
    if clip_x0 is not None:
        x0 = x0.clamp(-clip_x0, clip_x0)
        eps = (x - sa * x0) / s1a  # keep eps consistent with the clipped x0
    return eps, x0


@torch.no_grad()
def ddpm_sample(model, schedule: NoiseSchedule, shape, *, y=None, guidance_scale: float = 1.0,
                pred: str = "eps", clip_x0: Optional[float] = None, x_T=None,
                return_traj: bool = False, traj_every: int = 1, generator=None, device=None):
    """Ancestral sampling: x_{t-1} ~ N(mu_theta(x_t, t), beta_tilde_t I), t = T-1..0.

    Returns x_0 (and, if return_traj, a list of (t, x_t, x0_hat) snapshots).
    """
    device = device or schedule.device
    x = x_T.to(device) if x_T is not None else torch.randn(shape, device=device, generator=generator)
    traj = []
    for t in range(schedule.T - 1, -1, -1):
        eps, x0 = model_predictions(model, x, t, schedule, y, guidance_scale, pred, clip_x0)
        mean = schedule.posterior_coef_x0[t] * x0 + schedule.posterior_coef_xt[t] * x
        if t > 0:
            noise = torch.randn(x.shape, device=device, generator=generator)
            x_new = mean + schedule.posterior_variance[t].sqrt() * noise
        else:
            x_new = mean
        if return_traj and (t % traj_every == 0 or t == schedule.T - 1):
            traj.append((t, x.detach().cpu(), x0.detach().cpu()))
        x = x_new
    return (x, traj) if return_traj else x


def _ddim_timesteps(T: int, n_steps: int):
    """The 'linspace' grid {T-1, ..., 0} (descending). Its last network call is at t=0,
    so n_steps calls give n_steps-1 real denoising moves; diffusers calls this spacing
    'linspace' and also offers 'leading' and 'trailing' (see notebooks 04 and 09)."""
    if n_steps <= 1:
        return [T - 1]
    ts = np.linspace(0, T - 1, n_steps).round().astype(int)
    return list(np.unique(ts))[::-1]  # descending, unique


@torch.no_grad()
def ddim_sample(model, schedule: NoiseSchedule, shape, *, n_steps: int = 50, eta: float = 0.0,
                y=None, guidance_scale: float = 1.0, pred: str = "eps",
                clip_x0: Optional[float] = None, x_T=None, return_traj: bool = False,
                generator=None, device=None):
    """DDIM (Song et al. 2021). eta=0: deterministic; eta=1: DDPM-like stochasticity.

    With fewer steps than T, eta=1 is not exactly DDPM: its per-step variance is
    the posterior variance of the *strided* chain, and even with a perfect model
    the samples come out slightly under-dispersed (see notebook 07 and Bao et al.
    2022, Analytic-DPM)."""
    device = device or schedule.device
    x = x_T.to(device) if x_T is not None else torch.randn(shape, device=device, generator=generator)
    ts = _ddim_timesteps(schedule.T, n_steps)
    traj = []
    for i, t in enumerate(ts):
        t_prev = ts[i + 1] if i + 1 < len(ts) else -1
        ab = schedule.alpha_bars[t]
        ab_prev = schedule.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        eps, x0 = model_predictions(model, x, t, schedule, y, guidance_scale, pred, clip_x0)
        sigma = eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).sqrt()
        dir_xt = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        x_new = ab_prev.sqrt() * x0 + dir_xt
        if eta > 0 and t_prev >= 0:
            x_new = x_new + sigma * torch.randn(x.shape, device=device, generator=generator)
        if return_traj:
            traj.append((t, x.detach().cpu(), x0.detach().cpu()))
        x = x_new
    return (x, traj) if return_traj else x


@torch.no_grad()
def ddim_invert(model, schedule: NoiseSchedule, x0, *, n_steps: int = 50, y=None,
                guidance_scale: float = 1.0, pred: str = "eps"):
    """Run the deterministic DDIM ODE *forwards* (data -> noise). Returns x_T."""
    ts = _ddim_timesteps(schedule.T, n_steps)[::-1]  # ascending
    x = x0.clone()
    t_prev = -1
    for t in ts:
        ab_prev = schedule.alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x.device)
        ab = schedule.alpha_bars[t]
        # approximate eps(x_t, t) with eps evaluated at the previous (less noisy) point
        eps, _ = model_predictions(model, x, max(t_prev, 0), schedule, y, guidance_scale, pred)
        x0_hat = (x - (1 - ab_prev).sqrt() * eps) / ab_prev.sqrt()
        x = ab.sqrt() * x0_hat + (1 - ab).sqrt() * eps
        t_prev = t
    return x


# ----------------------------------------------------------------------------
# Sample-quality metrics for low-dimensional data
# ----------------------------------------------------------------------------

def mmd_rbf(x: torch.Tensor, y: torch.Tensor, bandwidths=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
            max_n: int = 4000) -> float:
    """(Biased) squared maximum mean discrepancy with a sum of RBF kernels. 0 = identical.

    Computed on the CPU in float64 (inputs may live on any device). Uses the first
    `max_n` points of each set; the (2n x 2n) kernel matrices need ~0.5 GB at n=4000,
    so pass a smaller max_n on a small machine.
    """
    x, y = x[:max_n].detach().to("cpu", torch.float64), y[:max_n].detach().to("cpu", torch.float64)
    z = torch.cat([x, y])
    d2 = torch.cdist(z, z).pow(2)
    k = sum(torch.exp(-d2 / (2 * h ** 2)) for h in bandwidths)
    n = x.shape[0]
    return (k[:n, :n].mean() + k[n:, n:].mean() - 2 * k[:n, n:].mean()).item()


def sliced_wasserstein(x: torch.Tensor, y: torch.Tensor, n_proj: int = 256, seed: int = 0) -> float:
    """Average 1-D Wasserstein-1 distance over random projections."""
    g = torch.Generator().manual_seed(seed)
    n = min(len(x), len(y))
    x, y = x[:n].float().cpu(), y[:n].float().cpu()
    dirs = torch.randn(x.shape[1], n_proj, generator=g)
    dirs = dirs / dirs.norm(dim=0, keepdim=True)
    px, _ = torch.sort(x @ dirs, dim=0)
    py, _ = torch.sort(y @ dirs, dim=0)
    return (px - py).abs().mean().item()


# ----------------------------------------------------------------------------
# MNIST
# ----------------------------------------------------------------------------

def load_mnist(train: bool = True, root=None):
    """MNIST as float tensors in [-1, 1] of shape (N, 1, 28, 28), plus long labels."""
    from torchvision import datasets
    root = Path(root or DATA_DIR)
    ds = datasets.MNIST(root=str(root), train=train, download=True)
    x = ds.data.float().div(127.5).sub(1.0).unsqueeze(1)
    return x, ds.targets.clone().long()


class MNISTClassifier(nn.Module):
    """A small CNN (~98.5% test accuracy) used to *evaluate* generated digits."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def features(self, x):
        return self.net[:-2](x)


def load_or_train_classifier(device=None, epochs: int = 2, force: bool = False, verbose=True):
    device = device or get_device()
    path = CKPT_DIR / "mnist_classifier.pt"
    clf = MNISTClassifier().to(device)
    if path.exists() and not force:
        clf.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return clf.eval()
    x, y = load_mnist(train=True)
    xt, yt = load_mnist(train=False)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        clf.train()
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), 128):
            idx = perm[i:i + 128]
            loss = F.cross_entropy(clf(x[idx].to(device)), y[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    clf.eval()
    with torch.no_grad():
        acc = sum((clf(xt[i:i + 1000].to(device)).argmax(1).cpu() == yt[i:i + 1000]).sum().item()
                  for i in range(0, len(xt), 1000)) / len(xt)
    if verbose:
        print(f"MNIST classifier test accuracy: {acc:.4f}")
    CKPT_DIR.mkdir(exist_ok=True)
    torch.save(clf.state_dict(), path)
    return clf


# A deliberately small U-Net (~1.2M parameters) so it trains on a laptop CPU.
MNIST_RECIPE = dict(base_ch=32, ch_mults=(1, 2, 2), n_res=1, attn_levels=(2,), dropout=0.0)


def load_or_train_mnist_model(conditional: bool = True, steps: int = None, device=None,
                              force: bool = False, batch_size: int = 64, lr: float = 2e-4,
                              p_uncond: float = 0.1, amp=None, verbose: bool = True):
    """Return (ema_model, schedule) for MNIST, training (and caching) it if needed.

    checkpoints/mnist_unet_uncond.pt (notebooks 05 and 09) and
    checkpoints/mnist_unet_cond.pt (class-conditional with 10% label dropout for
    classifier-free guidance, notebooks 06-07) are made by `pretrain_checkpoints.py`:
    the recipe of notebook 05 (`MNIST_RECIPE`, linear schedule, T=1000,
    eps-prediction, EMA 0.999) trained for 12 000 steps at batch 64. If a file is
    missing it is trained here with the same settings and cached (a few minutes on
    a GPU; on a CPU about an hour with amp=True, longer in float32).
    """
    device = device or get_device()
    name = "mnist_unet_cond.pt" if conditional else "mnist_unet_uncond.pt"
    path = CKPT_DIR / name
    schedule = NoiseSchedule(1000, "linear", device=device)
    if path.exists() and not force:
        # our own checkpoint file (plain dict of tensors/ints/tuples), so weights_only=False is safe
        ck = torch.load(path, map_location=device, weights_only=False)
        model = UNet(**ck["config"]).to(device)
        model.load_state_dict(ck["ema"])
        if verbose:
            print(f"loaded {path.name} (trained {ck.get('steps', '?')} steps)")
        return model.eval(), schedule
    if steps is None:
        steps = 300 if fast_mode() else 12000
    x, y = load_mnist(train=True)
    model = UNet(n_classes=10 if conditional else None, **MNIST_RECIPE).to(device)
    if verbose:
        print(f"training {name}: {count_params(model)/1e6:.2f}M params, {steps} steps on {device} ...")
    out = train_diffusion(model, x, schedule, labels=y if conditional else None, steps=steps,
                          batch_size=batch_size, lr=lr, p_uncond=p_uncond if conditional else 0.0,
                          device=device, amp=amp)
    if verbose:
        print(f"done in {out['seconds']/60:.1f} min, final loss {out['losses'][-200:].mean():.4f}")
    if not fast_mode():
        CKPT_DIR.mkdir(exist_ok=True)
        torch.save({"config": out["ema"].model.config, "ema": out["ema"].model.state_dict(),
                    "steps": steps, "batch_size": batch_size, "p_uncond": p_uncond if conditional else 0.0,
                    "losses": torch.as_tensor(out["losses"], dtype=torch.float32)}, path)
    return out["ema"].model.eval(), schedule


# ----------------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------------

def show_images(x: torch.Tensor, nrow: int = 8, title: str = None, ax=None, figsize=None,
                vmin=-1.0, vmax=1.0):
    """Show a batch (B, C, H, W) of images in [-1, 1] as a grid."""
    import matplotlib.pyplot as plt
    x = x.detach().float().cpu().clamp(vmin, vmax)
    B, C, H, W = x.shape
    ncol = min(nrow, B)
    nrows = math.ceil(B / ncol)
    pad = 2
    grid = torch.full((C, nrows * (H + pad) + pad, ncol * (W + pad) + pad), vmax)
    for i in range(B):
        r, c = divmod(i, ncol)
        grid[:, pad + r * (H + pad): pad + r * (H + pad) + H, pad + c * (W + pad): pad + c * (W + pad) + W] = x[i]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (ncol * 0.6 + 0.4, nrows * 0.6 + 0.4))
    img = grid.permute(1, 2, 0).numpy()
    if C == 1:
        ax.imshow(img[..., 0], cmap="gray", vmin=vmin, vmax=vmax)
    else:
        ax.imshow((img - vmin) / (vmax - vmin))
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax


def scatter2d(x: torch.Tensor, ax=None, title: str = None, lim: float = 3.0, s: float = 2,
              alpha: float = 0.5, c=None, **kw):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    x = x.detach().cpu().numpy()
    ax.scatter(x[:, 0], x[:, 1], s=s, alpha=alpha, c=c, linewidths=0, **kw)
    if lim:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax


def plot_vector_field(fn: Callable[[torch.Tensor], torch.Tensor], ax=None, lim: float = 3.0,
                      n: int = 21, normalize: bool = False, **kw):
    """Quiver plot of a function R^2 -> R^2 (e.g. a score) on a grid."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    g = torch.linspace(-lim, lim, n)
    X, Y = torch.meshgrid(g, g, indexing="xy")
    pts = torch.stack([X.flatten(), Y.flatten()], 1)
    with torch.no_grad():
        v = fn(pts).detach().cpu()
    if normalize:
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
    ax.quiver(pts[:, 0], pts[:, 1], v[:, 0], v[:, 1], **kw)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    return ax


def plot_trajectories(traj: torch.Tensor, ax=None, n_show: int = 50, lim: float = 3.0,
                      color="k", alpha: float = 0.3, lw: float = 0.6):
    """Plot particle paths. traj: (S, N, 2) tensor of S snapshots of N points."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    traj = torch.as_tensor(traj).detach().cpu().numpy()
    for i in range(min(n_show, traj.shape[1])):
        ax.plot(traj[:, i, 0], traj[:, i, 1], color=color, alpha=alpha, lw=lw)
    ax.scatter(traj[0, :n_show, 0], traj[0, :n_show, 1], s=6, c="tab:blue", zorder=3, label="start")
    ax.scatter(traj[-1, :n_show, 0], traj[-1, :n_show, 1], s=6, c="tab:red", zorder=3, label="end")
    if lim:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    return ax
