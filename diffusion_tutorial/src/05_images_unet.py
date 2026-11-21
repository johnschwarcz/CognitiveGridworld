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
# # 05 · Scaling to images: a U-Net diffusion model on MNIST
#
# Notebooks 01–04 developed diffusion models on 2-D toy data, where a small MLP is a good enough denoiser and
# exact scores are available. Here we take the *same* objective and the *same* samplers to 784-dimensional
# images. The maths does not change; the network and the engineering around it do. This notebook is about those
# practical details: the architecture, the training recipe, the samplers on images, and how to tell whether the samples are any good.
#
# **You will learn:**
# - why an MLP is a poor denoiser for images, and which inductive biases a **U-Net** builds in;
# - to build the U-Net's parts yourself: residual blocks with **FiLM** time conditioning, **GroupNorm**,
#   **self-attention**, down/up-sampling, **zero-initialised** output layers;
# - the training recipe used in practice: data scaling, AdamW with warm-up, gradient clipping, **EMA** weights,
#   bfloat16 mixed precision, checkpointing;
# - DDPM vs DDIM on images, **x0-clipping**, and why sampling is **coarse-to-fine** ("spectral autoregression");
# - how to evaluate samples: nearest neighbours (memorisation), class coverage, and **Fréchet distances / FID**
#   with their pitfalls.
#
# **Prerequisites:** 01 (forward process, schedules), 02 (DDPM loss and ancestral sampling),
# 03 (scores, Tweedie's formula), 04 (DDIM).
#
# **Runtime:** CPU ≈ 15–20 min (more than half of it is the in-notebook training run); CUDA GPU ≈ 10 min with
# the GPU default of 20 000 training steps (a few minutes if you lower `TRAIN_STEPS`).

# %%
import contextlib
import copy
import math
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg

import dmtools as dm

device = dm.get_device(verbose=True)
FAST = dm.fast_mode()  # automated smoke-test switch, leave as is
dm.seed_everything(0)
plt.rcParams.update({"figure.dpi": 100, "font.size": 9, "axes.titlesize": 10,
                     "axes.labelsize": 9, "legend.fontsize": 8})

# bfloat16 mixed precision: ~2x faster on CUDA GPUs and on recent x86 CPUs (AVX-512/AMX).
# Off on Apple MPS. If it turns out *slower* on your CPU, set USE_AMP = False.
USE_AMP = device.type == "cpu" or (device.type == "cuda" and torch.cuda.is_bf16_supported())


def autocast():
    """Mixed-precision context for forward passes (a no-op when USE_AMP is False)."""
    if not USE_AMP:
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=torch.bfloat16)


GPU = device.type == "cuda"
TRAIN_STEPS = 20 if FAST else (20_000 if GPU else 1500)  # GPU: ~20k steps gives clean digits
BATCH, LR, WARMUP = 64, 2e-4, (5 if FAST else 200)
N_EVAL = 32 if FAST else (2000 if GPU else 512)   # DDIM-50 samples per model for the evaluation
N_DDPM = 8 if FAST else (512 if GPU else 128)     # DDPM-1000 samples (1000 network calls each!)
C_REAL, C_DDPM, C_DDIM, C_MINE = "0.35", "tab:blue", "tab:green", "tab:orange"  # colours used throughout
print(f"FAST={FAST}  USE_AMP={USE_AMP}  TRAIN_STEPS={TRAIN_STEPS}  N_EVAL={N_EVAL}  N_DDPM={N_DDPM}")

# %% [markdown]
# ## 1 · From 2-D to 784-D: what does an image denoiser need?
#
# Nothing in the *maths* of notebooks 01–04 depends on the dimension. An MNIST digit is a vector
# $x_0\in\mathbb{R}^{1\times 28\times 28}\cong\mathbb{R}^{784}$, the forward process is the same, and so is the loss
#
# $$\mathcal{L}(\theta)=\mathbb{E}_{t,\,x_0,\,\epsilon}\Big\|\epsilon-\epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\;t\big)\Big\|^2 .$$
#
# What changes is *what the network has to represent*. By Tweedie's formula (notebook 03), predicting the noise is
# the same as predicting the posterior mean of the clean image,
#
# $$\hat x_0(x_t,t)=\mathbb{E}[x_0\mid x_t]=\frac{x_t-\sqrt{1-\bar\alpha_t}\;\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}},$$
#
# so $\epsilon_\theta$ must be a Bayes-optimal **denoiser for images at every noise level**, a map
# $\mathbb{R}^{784}\to\mathbb{R}^{784}$. Images have structure that a generic MLP would have to learn from scratch:
#
# 1. **Locality and translation equivariance.** A stroke is a stroke wherever it appears. If the data distribution
#    is (roughly) shift-invariant, so is the posterior, and the optimal denoiser is shift-*equivariant*: denoising a
#    shifted image gives the shifted denoised image. A convolution has this built in through weight sharing, using a
#    handful of parameters. A dense layer has $784^2$ independent weights and must learn the same stroke detector
#    separately at every position.
# 2. **Multi-scale structure.** At high noise only coarse structure survives (which digit, roughly where), and the
#    optimal denoiser has to average over large regions. At low noise it must restore fine local detail. Down-sampling
#    gives large receptive fields cheaply; skip connections carry the fine detail around the bottleneck. That is
#    the **U-Net** (Ronneberger et al. 2015).
# 3. **Global context.** Whether a blob in the top-left is the top of a 7 or of a 9 depends on the whole image.
#    **Self-attention** lets every position look at every other position. It costs $(HW)^2$, which is affordable at
#    $7\times7=49$ positions but not at 784.
#
# Let's check claim 1 numerically: a convolution commutes with shifts and a dense layer does not.

# %%
x_train, y_train = dm.load_mnist(train=True)
x_test, y_test = dm.load_mnist(train=False)
print(f"train {tuple(x_train.shape)}, test {tuple(x_test.shape)}")


def shift_right(img, k):
    """Circularly shift a batch of images (B, C, H, W) by k pixels to the right."""
    return torch.roll(img, shifts=k, dims=-1)


torch.manual_seed(0)
conv = nn.Conv2d(1, 1, 3, padding=1, padding_mode="circular")  # circular padding -> exact equivariance
dense = nn.Linear(784, 784)


def apply_dense(z):
    return dense(z.flatten(1)).view_as(z)


img = x_test[:8]
with torch.no_grad():
    for name, layer, module in [("3x3 conv", conv, conv), ("dense layer", apply_dense, dense)]:
        err = (layer(shift_right(img, 3)) - shift_right(layer(img), 3)).abs().max().item()
        scale = layer(img).abs().max().item()
        print(f"{name:12s} params {dm.count_params(module):>8,d} | "
              f"max |f(shift x) - shift f(x)| = {err:.1e}   (output scale {scale:.2f})")

# %% [markdown]
# The convolution is exactly shift-equivariant (up to float rounding) with 10 parameters. The dense layer is not,
# even with 600k parameters. The real U-Net is only *approximately* equivariant: it uses zero padding (borders
# break the symmetry) and stride-2 down-sampling, which is equivariant only to shifts by multiples of the stride.
# Inductive biases are soft constraints, which is usually what you want.
#
# ## 2 · Building the U-Net, one component at a time
#
# This is the architecture we will build (the configuration `dm.MNIST_RECIPE`, about 1.2M parameters):
#
# ```
#  x_t (1×28×28)                                                              ε̂ (1×28×28)
#    └► conv ─► [Res] ──────────────────────────────────────────────► [Res][Res] ─► GN·SiLU·conv ─┘
#  28×28, 32 ch  │ stride-2 conv                         nearest-up+conv ▲
#  14×14, 64 ch  └► [Res] ───────────────────────────────────► [Res][Res]
#                     │ stride-2 conv            nearest-up+conv ▲
#  7×7,  64 ch        └► [Res+Attn] ─► middle: Res·Attn·Res ─► [Res+Attn][Res+Attn]
#
#  t ─► sinusoidal features ─► MLP ─► e_t  ──(FiLM: injected into every Res block)
# ```
#
# Every encoder output is pushed on a stack and **concatenated** into the decoder at the same resolution (the
# horizontal arrows; there are 6 such skips). We now write each piece ourselves and check it against the version in
# `dmtools`.
#
# ### 2.1 Residual block with FiLM time conditioning, and GroupNorm
#
# One network has to denoise at all 1000 noise levels. The right operation at $t=999$ (heavy smoothing,
# guessing the global shape) is very different from the one at $t=0$ (almost the identity). So the time
# embedding $e_t=\mathrm{MLP}(\text{sinusoidal}(t))$ is allowed to **rescale and shift every channel** of every
# residual block. This is FiLM, feature-wise linear modulation (Perez et al. 2018):
#
# $$h=\mathrm{Conv}_1\big(\mathrm{SiLU}(\mathrm{GN}(x))\big),\qquad (\gamma_t,\beta_t)=W\,\mathrm{SiLU}(e_t)+b\;\in\mathbb{R}^{C}\times\mathbb{R}^{C},$$
# $$h\leftarrow\mathrm{GN}(h)\odot(1+\gamma_t)+\beta_t,\qquad
#   \mathrm{out}=\mathrm{skip}(x)+\mathrm{Conv}_2\big(\mathrm{SiLU}(h)\big).$$
#
# Implementation details that matter:
# * **FiLM after normalisation.** GroupNorm removes each channel's scale and offset; FiLM then sets new,
#   $t$-dependent ones. Writing $(1+\gamma)$ instead of $\gamma$ means "no modulation" corresponds to $\gamma=0$.
# * **Zero-initialised last conv.** With $\mathrm{Conv}_2=0$ at initialisation, each block starts as its skip path,
#   so the deep network starts out as a shallow, well-conditioned function and the residual branches switch on
#   during training. The U-Net's final output conv is zero-initialised too (section 2.4).
# * **GroupNorm, not BatchNorm.** In a diffusion minibatch every sample has a *different* noise level.
#   BatchNorm would normalise a clean-ish image with statistics pooled from pure-noise images, make each output
#   depend on the other samples in the batch, and behave differently at sampling time, when the whole batch shares one
#   $t$. GroupNorm (Wu & He 2018) normalises each sample on its own, over groups of channels.

# %%
def num_groups(ch):
    """GroupNorm groups: the largest of 32, 16, ..., 2 that leaves >= 4 channels per group (else 1)."""
    return next((g for g in (32, 16, 8, 4, 2) if ch % g == 0 and ch // g >= 4), 1)


class MyResBlock(nn.Module):
    """GN -> SiLU -> conv -> GN -> FiLM(e_t) -> SiLU -> conv, plus a (1x1-conv) skip path."""

    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)            # e_t -> per-channel (scale, shift)
        self.norm2 = nn.GroupNorm(num_groups(out_ch), out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)                    # residual branch starts at exactly 0
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        # (B, 2*out_ch) -> (B, 2*out_ch, 1, 1): broadcast the same scale/shift over all pixels of a channel
        scale, shift = self.emb(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift              # FiLM
        h = self.conv2(self.drop(F.silu(h)))
        return h + self.skip(x)


torch.manual_seed(0)
emb_dim = 128
time_mlp = nn.Sequential(dm.SinusoidalEmbedding(32), nn.Linear(32, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
block = MyResBlock(32, 64, emb_dim)
h_in = torch.randn(4, 32, 14, 14)
e_t = time_mlp(torch.tensor([0, 250, 500, 999]))

with torch.no_grad():
    print("at init, block(x) == skip(x) exactly:", torch.equal(block(h_in, e_t), block.skip(h_in)))
    nn.init.normal_(block.conv2.weight, std=0.05)          # pretend training switched the branch on
    ref_block = dm.ResBlock(32, 64, emb_dim)
    ref_block.load_state_dict(block.state_dict())           # same parameter names -> same function
    print(f"max |MyResBlock - dm.ResBlock| = {(block(h_in, e_t) - ref_block(h_in, e_t)).abs().max():.1e}")
    # FiLM at work: the same input image gives different features at different t
    same_x = h_in[:1].expand(4, -1, -1, -1)
    out_t = block(same_x, e_t)
    print("same input, t = 0/250/500/999 -> feature std per t:", [f"{v:.3f}" for v in out_t.std(dim=(1, 2, 3))])

    # GroupNorm vs BatchNorm: does sample 0's output depend on the *other* samples in the batch?
    gn, bn = nn.GroupNorm(8, 32), nn.BatchNorm2d(32)        # bn in training mode = batch statistics
    batch_a = torch.randn(8, 32, 7, 7)
    batch_b = batch_a.clone()
    batch_b[1:] = 5 * torch.randn(7, 32, 7, 7)               # same sample 0, very different batch-mates
    for name, norm in [("GroupNorm", gn), ("BatchNorm", bn)]:
        d = (norm(batch_a)[0] - norm(batch_b)[0]).abs().max().item()
        print(f"{name:9s}: change in sample 0's output when the rest of the batch changes = {d:.2e}")

# %% [markdown]
# All three checks pass. The fresh block is exactly its skip path. Our block and `dm.ResBlock` compute the same
# function. FiLM makes the same input produce different features at different $t$. And GroupNorm, unlike
# BatchNorm, treats every sample independently. We will rely on that last property in section 6, where we
# evaluate many copies of one image in a single batch.
#
# ### 2.2 Self-attention at low resolution
#
# Treat the $N=HW$ positions of a feature map as tokens $h_i\in\mathbb{R}^C$. With per-head queries, keys and values
# $q_i=W_qh_i,\;k_i=W_kh_i,\;v_i=W_vh_i\in\mathbb{R}^d$:
#
# $$a_{ij}=\frac{\exp(q_i^\top k_j/\sqrt d)}{\sum_{j'}\exp(q_i^\top k_{j'}/\sqrt d)},\qquad
#   \mathrm{out}_i=h_i+W_o\sum_j a_{ij}\,v_j .$$
#
# Each output position mixes information from **all** positions, with weights that depend on the content. This
# gives a global receptive field in a single layer. The cost is the $N\times N$ matrix $a$ per head and image.
# $W_o$ is zero-initialised, so the block also starts as the identity.

# %%
class MyAttnBlock(nn.Module):
    """Multi-head self-attention over the H*W positions of a feature map, with a residual connection."""

    def __init__(self, ch, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(num_groups(ch), ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)      # a 1x1 conv = the same linear map at every position
        self.proj = nn.Conv2d(ch, ch, 1)         # W_o
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, emb=None):
        B, C, H, W = x.shape
        d = C // self.heads
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.heads, d, H * W).unbind(1)  # each (B, heads, d, N)
        attn = (q.transpose(-1, -2) @ k / math.sqrt(d)).softmax(dim=-1)  # (B, heads, N, N), rows sum to 1
        out = attn @ v.transpose(-1, -2)                                  # (B, heads, N, d)
        return x + self.proj(out.transpose(-1, -2).reshape(B, C, H, W))


torch.manual_seed(0)
attn_block = MyAttnBlock(64)
nn.init.normal_(attn_block.proj.weight, std=0.1)   # un-zero W_o so that the comparison is not trivial
ref_attn = dm.AttnBlock(64)                         # uses the fused F.scaled_dot_product_attention
ref_attn.load_state_dict(attn_block.state_dict())
feat = torch.randn(2, 64, 7, 7, requires_grad=True)
out = attn_block(feat)
print(f"max |MyAttnBlock - dm.AttnBlock| = {(out - ref_attn(feat)).abs().max():.1e}")

# Receptive field of ONE layer: how much of the output at the centre (3, 3) comes from far-away positions?
for name, layer in [("3x3 conv", nn.Conv2d(64, 64, 3, padding=1)), ("attention", attn_block)]:
    g, = torch.autograd.grad(layer(feat)[0, :, 3, 3].sum(), feat)
    influence = g[0].abs().sum(0)                       # (7, 7): sensitivity to each input position
    outside = 1 - (influence[2:5, 2:5].sum() / influence.sum()).item()
    print(f"{name:9s}: {(influence > 0).sum().item():2d} of 49 input positions influence output (3,3); "
          f"{outside:4.0%} of the influence comes from outside its 3x3 neighbourhood")

for hw in (28, 14, 7):
    print(f"attention matrix at {hw:2d}x{hw:<2d}: {(hw * hw) ** 2:>7,d} entries per head and image")

# %% [markdown]
# Our explicit softmax attention agrees with PyTorch's fused kernel. A 3×3 conv sees 9 positions. One attention
# layer draws a substantial part of its output from the whole 7×7 map. (Strictly, GroupNorm alone already couples
# all positions a little, through the shared mean and variance, which is why we measure influence *mass*.) The cost explains why attention is used only at the lowest resolution: the attention
# matrix at 28×28 is 256 times larger than at 7×7. Large image models (Stable Diffusion, DiT) use attention more
# widely, but always on down-sampled feature maps or patch tokens, for the same reason.
#
# ### 2.3 Down- and up-sampling
#
# * **Down:** a 3×3 convolution with stride 2. It is learned, and cheaper than conv + pooling.
# * **Up:** nearest-neighbour up-sampling followed by a 3×3 convolution. Transposed convolutions tend to produce
#   checkerboard artefacts.
# * **Gotcha:** every down-sampling halves $H$ and $W$, and the decoder must land on exactly the encoder's sizes
#   for the skip concatenations. The image size must therefore be divisible by $2^{\text{levels}-1}$: 28 → 14 → 7
#   works for 3 levels, and 30 → 15 → 8 does not (8 up-samples to 16 ≠ 15). Neither would a 4th level for 28
#   (7 → 4 → 8 ≠ 7).

# %%
down = nn.Conv2d(32, 32, 3, stride=2, padding=1)
up = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(32, 32, 3, padding=1))
with torch.no_grad():
    for size in (28, 30):
        h = torch.randn(1, 32, size, size)
        hs = [h]
        for _ in range(2):
            hs.append(down(hs[-1]))
        back = up(up(hs[-1]))
        sizes = " -> ".join(str(z.shape[-1]) for z in hs) + " -> up -> " + str(up(hs[-1]).shape[-1]) + \
            " -> up -> " + str(back.shape[-1])
        ok = all(up(hs[i + 1]).shape == hs[i].shape for i in range(2))
        print(f"{size}x{size}: {sizes}   decoder matches encoder at every level: {ok}")

# %% [markdown]
# ### 2.4 The assembled U-Net
#
# `dm.UNet` is these pieces assembled: its `ResBlock` and `AttnBlock` are exactly the classes we just checked
# against our own. It adds the time-embedding MLP, an input conv, the stack of skip connections, and a
# **zero-initialised output conv**. Rather than treat it as a black box, we re-run its forward pass step by step
# and print every tensor shape. Then we check that our step-by-step pass reproduces `model(x, t)`.

# %%
unet = dm.UNet(**dm.MNIST_RECIPE)
print("recipe:", dm.MNIST_RECIPE)
print(f"total parameters: {dm.count_params(unet):,}")
for part in ["time_emb", "inp", "down", "mid", "up", "out"]:
    print(f"   {part:9s}{dm.count_params(getattr(unet, part)):>9,d}")


def describe(blocks):
    """Short name of a list of layers, e.g. 'ResBlock+AttnBlock+Upsample+Conv'."""
    names = []
    for b in blocks:
        name = type(b).__name__.replace("Conv2d", "Conv")
        if isinstance(b, nn.Conv2d) and b.stride == (2, 2):
            name = "Conv(stride 2)"
        names.append(name)
    return "+".join(names)


def trace_unet(model, x, t):
    """Re-run dm.UNet.forward step by step, printing shapes and the skip stack. Returns the output."""
    emb = model.time_emb(t)                       # (B, 4*base_ch): sinusoidal features -> MLP
    print(f"{'input x':56s}{tuple(x.shape)}\n{'time embedding e_t':56s}{tuple(emb.shape)}")
    h = model.inp(x)
    skips = [h]
    print(f"{'inp conv':56s}{tuple(h.shape)}  -> push skip 0")
    for blocks in model.down:
        h = model._run(blocks, h, emb)
        skips.append(h)
        print(f"{'down  ' + describe(blocks):56s}{tuple(h.shape)}  -> push skip {len(skips) - 1}")
    h = model._run(model.mid, h, emb)
    print(f"{'mid   ' + describe(model.mid):56s}{tuple(h.shape)}")
    for blocks in model.up:
        k = len(skips) - 1
        h = torch.cat([h, skips.pop()], dim=1)    # concatenate along channels
        h = model._run(blocks, h, emb)
        print(f"{f'up    cat skip {k} -> ' + describe(blocks):56s}{tuple(h.shape)}")
    out = model.out(h)
    print(f"{'out   GroupNorm+SiLU+Conv':56s}{tuple(out.shape)}")
    return out


x_dummy, t_dummy = torch.randn(2, 1, 28, 28), torch.tensor([10, 900])
with torch.no_grad():
    print("fresh U-Net output is exactly zero:", bool((unet(x_dummy, t_dummy) == 0).all()), "\n")
    check_net = copy.deepcopy(unet)
    for p in check_net.parameters():              # un-zero the zero-initialised layers for a real test
        if not p.any():
            p.normal_(0, 0.02)
    traced = trace_unet(check_net, x_dummy, t_dummy)
    print(f"\nmax |trace - model(x, t)| = {(traced - check_net(x_dummy, t_dummy)).abs().max():.1e}")

# %% [markdown]
# Reading the trace: the encoder goes 28 → 14 → 7 while channels go 32 → 64. The decoder pops the skips in reverse
# order, so every concatenation pairs features of the same resolution. Attention runs only on the 7×7 maps.
#
# A fresh U-Net outputs **exactly zero**, so it initially predicts $\hat\epsilon=0$ and the initial loss is
# $\mathbb{E}\|\epsilon\|^2/784 = 1$. This does not stop learning. The gradient with respect to the
# zero-initialised weights is (activations × upstream error), which is non-zero, so they move on the first step and
# the signal starts to flow. We will see the loss start at 1 in section 4.
#
# ## 3 · Data: MNIST in $[-1, 1]$
#
# `dm.load_mnist` returns images scaled to $[-1,1]$ (background $=-1$, ink $=+1$). Why this matters:
#
# * **The noise schedule assumes data of unit scale.** The forward process ends at
#   $x_{T-1}=\sqrt{\bar\alpha_{T-1}}\,x_0+\sqrt{1-\bar\alpha_{T-1}}\,\epsilon$, and the sampler *starts* from
#   $\mathcal{N}(0,I)$. That is only consistent if $\sqrt{\bar\alpha_{T-1}}\,x_0$ is negligible, which holds when
#   $x_0=O(1)$. The signal-to-noise ratio is $\mathrm{SNR}(t)=\bar\alpha_t\,\mathrm{Var}[x_0]/(1-\bar\alpha_t)$,
#   so rescaling the data shifts *which* timesteps carry information.
# * **A known, bounded range** lets us clip $\hat x_0$ to $[-1,1]$ during sampling (section 5).
#
# "Centred" here means centred *range*. The mean pixel is not 0: most pixels are background. What matters is that the
# scale is $O(1)$ and the support is symmetric and known.

# %%
schedule = dm.NoiseSchedule(T=1000, kind="linear", device=device)
abar = schedule.alpha_bars.cpu().double()
print(f"pixels in [{x_train.min():.0f}, {x_train.max():.0f}], mean {x_train.mean():.3f}, std {x_train.std():.3f}\n")

x01 = (x_train[:10_000].double() + 1) / 2   # the same images in [0, 1]
print(f"{'scaling':>9s} | {'pixel std':>9s} | {'SNR(t) drops below 1 at':>23s} | "
      f"{'max signal left in x_T-1':>24s}")
scalings = {"[-1, 1]": 2 * x01 - 1, "[0, 1]": x01, "[0, 255]": 255 * x01}
for name, x in scalings.items():
    snr = abar * x.var() / (1 - abar)
    below = torch.nonzero(snr < 1)
    t_cross = f"t = {below[0].item()}" if len(below) else "never"
    left = (abar[-1].sqrt() * x).abs().amax(dim=(1, 2, 3)).mean().item()
    print(f"{name:>9s} | {x.std().item():9.3f} | {t_cross:>23s} | {left:24.3f}")

# %%
ts_show = [0, 50, 100, 200, 300, 500, 700, 999]
x0_show = x_train[:5].to(device)
eps_show = torch.randn_like(x0_show)              # the same noise at every t, so each row is one trajectory
fig, axes = plt.subplots(len(x0_show), len(ts_show), figsize=(8, 5.2))
for j, t in enumerate(ts_show):
    xt = schedule.q_sample(x0_show, torch.full((len(x0_show),), t, device=device), eps_show).cpu()
    for i in range(len(x0_show)):
        axes[i, j].imshow(xt[i, 0], cmap="gray", vmin=-1, vmax=1)
        axes[i, j].set_axis_off()
    axes[0, j].set_title(f"t={t}\n$\\sqrt{{\\bar\\alpha}}$={abar[t].sqrt():.2f}", fontsize=8)
fig.suptitle("Forward process on MNIST (display clipped to [-1, 1])")
plt.tight_layout()
plt.show()

# what x_{T-1} looks like if you forget to rescale the data
fig, axes = plt.subplots(1, 3, figsize=(8, 2.2))
for ax, (name, x) in zip(axes, scalings.items()):
    xT = abar[-1].sqrt() * x[:4] + (1 - abar[-1]).sqrt() * torch.randn_like(x[:4])
    dm.show_images(xT.float(), nrow=4, ax=ax, vmin=-2.5, vmax=2.5)
    ax.set_title(f"$x_{{T-1}}$ for data in {name}")
plt.tight_layout()
plt.show()

# %% [markdown]
# With data in $[-1,1]$, structure fades gradually: strokes are clear up to $t\approx100$, the digit is barely
# visible at $t\approx300$, and from $t\approx500$ on the eye sees only noise. (Some low-frequency information
# survives longer than the eye suggests, see section 6.) The second figure shows the scaling gotcha. Data in $[0,255]$
# is still plainly visible at the last timestep, so the model would never be trained on the pure noise the sampler
# starts from. Data in $[0,1]$ has a small variance, so SNR = 1 is reached much earlier, and fewer of the
# 1000 steps are spent at noise levels where the model still sees structure.
#
# ## 4 · Training, with every practical detail
#
# The objective is the "simple" DDPM loss from notebook 02 (`dm.diffusion_loss`): draw a minibatch, draw
# **an independent $t\sim\mathcal{U}\{0,\dots,T-1\}$ for every sample**, noise, predict $\epsilon$, take the MSE. The rest is engineering:
#
# | ingredient | value here | why |
# |---|---|---|
# | optimiser | AdamW, lr $2\cdot10^{-4}$, no weight decay | the standard DDPM setting for small U-Nets. Adam's per-parameter scaling copes with very different gradient scales across layers |
# | warm-up | lr ramps up linearly over the first 200 steps | Adam's second-moment estimates are poor at first, and large early steps on freshly switched-on residual branches can destabilise training |
# | gradient clipping | global norm ≤ 1 | caps the influence of unusually large gradients (unlucky batches, early training) on Adam's moment estimates. Adam divides by a running RMS of the gradients, so a *uniform* rescaling changes little; clipping only bites on outliers. Cheap insurance |
# | batch size | 64 | same as the reference checkpoint. Larger batches mostly reduce the loss noise from random $t$ |
# | random $t$ per sample | uniform | every minibatch is an unbiased estimate of the average over all noise levels. Stratified or low-discrepancy $t$ (VDM, Kingma et al. 2021) reduces the variance further |
# | **EMA** of weights | decay 0.999, used for **sampling** | see below |
# | mixed precision | bfloat16 autocast (`USE_AMP`) | convs and matmuls in bf16, while weights, optimiser state, norms and the loss stay in float32. bf16 has float32's exponent range, so unlike float16 it needs no loss scaling. `dm.train_diffusion(amp=...)` does the same |
# | checkpointing | model + EMA + optimiser + step | resuming without the optimiser state resets Adam's moments and causes a loss spike |
# | augmentation | none | augmentations must leave the data distribution unchanged, or the model learns to generate augmented data. A horizontally flipped 2 is not a digit |
#
# **Why sample with EMA weights?** With a constant learning rate, SGD does not converge to a point. It keeps jittering
# around the optimum, with a spread set by the learning rate and the gradient noise. The exponential moving average
#
# $$\theta^{\mathrm{EMA}}_n=d\,\theta^{\mathrm{EMA}}_{n-1}+(1-d)\,\theta_n=(1-d)\sum_{k\le n}d^{\,n-k}\theta_k+d^{\,n}\theta_0$$
#
# averages the last $\approx 1/(1-d)$ iterates (Polyak–Ruppert averaging), which cancels much of that jitter.
# Diffusion sampling applies the network hundreds of times, so small weight noise compounds into visible artefacts.
# EMA weights are therefore what everyone samples with, while the raw weights are what the optimiser keeps
# training. **Gotcha:** the $d^{\,n}\theta_0$ term means that early on a plain EMA is dominated by the random
# initialisation. `dm.EMA` uses the warm-up $d_n=\min\{d,\,(1+n)/(10+n)\}$, whose horizon $\approx(10+n)/9$ grows
# with $n$. It only reaches $d=0.999$ after about 9000 steps, so in a short run the EMA averages over just the last
# few hundred steps.
#
# **Another image gotcha: broadcasting.** Per-sample schedule coefficients have shape `(B,)` and must be reshaped
# to `(B, 1, 1, 1)` before they multiply a `(B, C, H, W)` batch (`NoiseSchedule._bcast` does this). A `(B, 1)`
# coefficient can broadcast silently into the wrong shape when `B` happens to equal `H` or `W`.

# %%
def save_checkpoint(path, model, ema, opt, step, losses):
    """Everything needed to resume training *and* to sample (EMA weights + config).
    Only tensors and plain Python types, so that it loads with torch.load(..., weights_only=True)."""
    torch.save({"config": model.config, "model": model.state_dict(), "ema": ema.model.state_dict(),
                "opt": opt.state_dict(), "step": step, "losses": torch.tensor(losses)}, path)


dm.seed_everything(0)
model = dm.UNet(**dm.MNIST_RECIPE).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP))  # linear warm-up
ema = dm.EMA(model, decay=0.999)
data = x_train.to(device)
log = {"loss": [], "grad_norm": []}
LOG_EVERY = max(1, TRAIN_STEPS // 6)

model.train()
t_start = time.time()
for step in range(TRAIN_STEPS):
    x0 = data[torch.randint(0, len(data), (BATCH,), device=device)]
    with autocast():
        loss = dm.diffusion_loss(model, x0, schedule)       # random t per sample, eps-prediction MSE
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # returns the norm *before* clipping
    opt.step()
    lr_sched.step()
    ema.update(model)                                       # after every optimiser step
    log["loss"].append(loss.item())
    log["grad_norm"].append(grad_norm.item())
    if step == 0 or (step + 1) % LOG_EVERY == 0:
        print(f"step {step + 1:5d} | loss (last {LOG_EVERY}) {np.mean(log['loss'][-LOG_EVERY:]):.4f} | "
              f"lr {lr_sched.get_last_lr()[0]:.1e} | {time.time() - t_start:5.0f}s")
model.eval()
train_sec = time.time() - t_start
print(f"\n{TRAIN_STEPS} steps in {train_sec / 60:.1f} min ({TRAIN_STEPS / train_sec:.1f} steps/s on {device})")
print(f"first-step loss {log['loss'][0]:.3f}  (zero-initialised output => E||eps||^2/784 = 1)")
assert abs(log["loss"][0] - 1) < 0.15

# checkpoint (FAST smoke tests write to a temporary file instead of checkpoints/)
CKPT_PATH = (Path(tempfile.gettempdir()) / "05_my_unet_fast.pt") if FAST else dm.CKPT_DIR / "05_my_unet.pt"
CKPT_PATH.parent.mkdir(exist_ok=True)
save_checkpoint(CKPT_PATH, model, ema, opt, TRAIN_STEPS, log["loss"])
ck = torch.load(CKPT_PATH, map_location=device, weights_only=True)  # never unpickle arbitrary objects
reloaded = dm.UNet(**ck["config"]).to(device).eval()
reloaded.load_state_dict(ck["ema"])
with torch.no_grad():
    same = torch.equal(reloaded(x_dummy.to(device), t_dummy.to(device)), ema.model(x_dummy.to(device), t_dummy.to(device)))
print(f"saved {CKPT_PATH.name}; reloaded EMA model reproduces the in-memory one exactly: {same}")

# %%
def running_mean(v, w):
    v = np.asarray(v)
    w = max(1, min(w, len(v)))
    return np.convolve(v, np.ones(w) / w, mode="valid")


steps_axis = np.arange(1, TRAIN_STEPS + 1)
w = 50
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
axes[0].plot(steps_axis, log["loss"], color=C_MINE, alpha=0.25, lw=0.6, label="per step")
axes[0].plot(steps_axis[len(steps_axis) - len(running_mean(log["loss"], w)):], running_mean(log["loss"], w),
             color=C_MINE, lw=1.8, label=f"running mean ({w} steps)")
axes[0].set(yscale="log", xlabel="training step", ylabel="MSE on ε", title="Training loss")
axes[0].legend()
axes[1].plot(steps_axis, log["grad_norm"], color="tab:purple", alpha=0.5, lw=0.7)
axes[1].axhline(1.0, color="k", ls="--", lw=1, label="clipping threshold")
axes[1].set(yscale="log", xlabel="training step", ylabel="‖∇θ L‖ before clipping", title="Gradient norm")
axes[1].legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# The loss starts at 1 (the zero-initialised output, as predicted), drops steeply within the first few hundred
# steps, and then declines slowly. That long slow tail is where sample quality is made. The per-step loss stays
# noisy forever. The noise is not a sign of a bad learning rate: each minibatch has a different mix of $t$, and the
# loss depends strongly on $t$ (we will measure this below). The gradient-norm panel shows how often clipping is
# active. It is mostly a guard against early or rare spikes.
#
# **Samples from the model we just trained.** We use DDIM with 50 steps (notebook 04) and the same starting noise
# for the raw and the EMA weights. On a CPU budget of 1500 steps, expect rough results: digit-like strokes and
# blobs, some recognisable digits, many malformed ones. That is honest for this little compute. On a GPU, set
# `TRAIN_STEPS = 20_000` (the default there) and the samples become clean.

# %%
SAMPLE_BS = 256


@torch.no_grad()
def sample_ddim(net, n, steps=50, seed=0, clip=1.0, eta=0.0):
    """n DDIM samples (deterministic for eta=0: `seed` fixes x_T), in batches, under autocast. Returns CPU."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, n, SAMPLE_BS):
        x_T = torch.randn(min(SAMPLE_BS, n - i), 1, 28, 28, generator=g).to(device)
        with autocast():
            x = dm.ddim_sample(net, schedule, x_T.shape, n_steps=steps, eta=eta, x_T=x_T, clip_x0=clip)
        out.append(x.float().cpu())
    return torch.cat(out)


n_raw = min(N_DDPM, N_EVAL)
mine_raw = sample_ddim(model, n_raw, seed=10)         # raw (optimiser) weights
t0 = time.time()
mine_ema = sample_ddim(ema.model, N_EVAL, seed=10)    # EMA weights; the first n_raw share x_T with mine_raw
print(f"DDIM-50: {N_EVAL} samples in {time.time() - t0:.1f}s")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
dm.show_images(mine_raw[:32], nrow=8, ax=axes[0], title=f"in-notebook model, raw weights ({TRAIN_STEPS} steps)")
dm.show_images(mine_ema[:32], nrow=8, ax=axes[1], title=f"in-notebook model, EMA weights ({TRAIN_STEPS} steps)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Compare the two grids position by position: the starting noise is the same. In a short run the EMA horizon is
# only a few hundred steps (see the warm-up formula), so the difference is modest. We will quantify it with
# Fréchet distances in section 7. In long runs, where the EMA averages over thousands of steps, the EMA weights are
# clearly better.
#
# **From here on we use a longer-trained reference model:** the same recipe, the same code, 12 000 steps at batch
# 64 (8× our budget). `dm.load_or_train_mnist_model` loads it from `checkpoints/`, or trains and caches it if it is
# missing (slow on a CPU).

# %%
# The reference checkpoint stores its loss curve as a numpy array, and torch >= 2.6 loads files with
# weights_only=True, which refuses numpy objects unless they are allow-listed. These are our own files, so:
if hasattr(torch.serialization, "add_safe_globals"):
    np_core = getattr(np, "_core", None) or np.core
    torch.serialization.add_safe_globals([np_core.multiarray._reconstruct, np.ndarray, np.dtype])

ref_model, ref_schedule = dm.load_or_train_mnist_model(conditional=False, device=device, batch_size=BATCH,
                                                       amp=USE_AMP)
assert torch.allclose(ref_schedule.alpha_bars, schedule.alpha_bars)  # same noise schedule as ours
print(f"reference model: {dm.count_params(ref_model):,} parameters, config {ref_model.config}")


@torch.no_grad()
def loss_per_t(net, x0, ts, seed=0):
    """Mean eps-MSE at each fixed t, with the same noise for every network (common random numbers).
    Runs in float32 on purpose: at large t the eps-error is ~1e-5, below bfloat16's resolution."""
    vals = []
    for t in ts:
        torch.manual_seed(seed)
        vals.append(dm.diffusion_loss(net, x0, schedule, t=torch.full((len(x0),), int(t), device=device)).item())
    return np.array(vals)


ts_loss = np.unique(np.r_[0, np.geomspace(1, 999, 12 if FAST else 20).astype(int)])
x_eval = x_test[: (32 if FAST else 128)].to(device)
snr_t = (schedule.alpha_bars / (1 - schedule.alpha_bars)).cpu().numpy()[ts_loss]
curves = {"in-notebook (EMA)": (loss_per_t(ema.model, x_eval, ts_loss), C_MINE),
          "reference (12k steps)": (loss_per_t(ref_model, x_eval, ts_loss), C_DDPM)}
prior_var = x_train.var(dim=0).mean().item()   # MSE of the best *constant* guess, E[x0], per pixel

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for name, (eps_mse, c) in curves.items():
    axes[0].plot(ts_loss, eps_mse, "o-", ms=3, color=c, label=name)
    axes[1].plot(ts_loss, eps_mse / snr_t, "o-", ms=3, color=c, label=name)   # x0-MSE = eps-MSE / SNR(t)
axes[0].set(xscale="symlog", yscale="log", xlabel="t", ylabel="MSE on ε  (what we train)",
            title="Loss per noise level")
axes[1].axhline(prior_var, color=C_REAL, ls="--", lw=1, label="predict the mean image (prior variance)")
axes[1].set(xscale="symlog", yscale="log", xlabel="t", ylabel="implied MSE on x̂0 per pixel",
            title="Same errors, measured on the clean image")
for ax in axes:
    ax.legend()
plt.tight_layout()
plt.show()
for name, (eps_mse, _) in curves.items():
    print(f"{name:22s}: x0-MSE at t=999 = {eps_mse[-1] / snr_t[-1]:.3f}   (prior variance {prior_var:.3f})")

# %% [markdown]
# Two views of the same errors. Since $x_0-\hat x_0=\sqrt{(1-\bar\alpha_t)/\bar\alpha_t}\,(\hat\epsilon-\epsilon)$,
# the $\epsilon$-MSE is the $x_0$-MSE weighted by $\mathrm{SNR}(t)=\bar\alpha_t/(1-\bar\alpha_t)$.
#
# * **Left:** the $\epsilon$-loss is largest at small $t$, where telling the tiny noise apart from fine image detail is
#   hard, and almost zero at large $t$, where $x_t\approx\epsilon$. So uniform-$t$ training spends most of its
#   gradient signal on low and intermediate noise levels. This **loss weighting** is a design choice, and
#   later work tunes it (Kingma et al. 2021; Karras et al. 2022).
# * **Right:** in image space the error grows with $t$ and saturates at the **prior variance**, the error of guessing
#   the average digit. That is the Bayesian answer: once $x_t$ carries no information, the posterior mean is the
#   prior mean. The printout checks this numerically.
# * The two models differ mostly at small and intermediate $t$, where the details are made. That is where the
#   extra training of the reference model went.
#
# ## 5 · Sampling images: DDPM vs DDIM, the denoising chain, x0-clipping
#
# Both samplers come from earlier notebooks: ancestral DDPM sampling with 1000 network evaluations (NFE, 02), and
# deterministic DDIM with 50 (04). We run both with the reference model and time them.

# %%
dm.seed_everything(1)
t0 = time.time()
with autocast():
    ddpm_x, ddpm_traj = dm.ddpm_sample(ref_model, schedule, (N_DDPM, 1, 28, 28), clip_x0=1.0,
                                       return_traj=True, traj_every=50)
ddpm_x = ddpm_x.float().cpu()
ddpm_sec = (time.time() - t0) / N_DDPM
t0 = time.time()
ddim_x = sample_ddim(ref_model, N_EVAL, steps=50, seed=2)
ddim_sec = (time.time() - t0) / N_EVAL
print(f"DDPM-1000: {N_DDPM} samples, {1000 * ddpm_sec:.0f} ms per sample")
print(f"DDIM-50  : {N_EVAL} samples, {1000 * ddim_sec:.0f} ms per sample  ({ddpm_sec / ddim_sec:.0f}x faster)")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
n_show = min(36, N_DDPM)
dm.show_images(ddpm_x[:n_show], nrow=6, ax=axes[0], title=f"DDPM, 1000 NFE ({1000 * ddpm_sec:.0f} ms/sample)")
dm.show_images(ddim_x[:36], nrow=6, ax=axes[1], title=f"DDIM, 50 NFE ({1000 * ddim_sec:.0f} ms/sample)")
plt.tight_layout()
plt.show()

# %% [markdown]
# Both grids should look like handwritten digits of similar quality, with DDIM at 1/20 of the cost.
# Section 7 compares them quantitatively. Most samples should be recognisable. Expect a few malformed
# ones from a 1.2M-parameter model trained for 12k steps on a CPU.
#
# **Numerical precision.** We sampled under bfloat16 autocast. The schedule coefficients and the sampler arithmetic
# stay in float32. This matters, because $\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$
# multiplies errors in $\hat\epsilon$ by up to $1/\sqrt{\bar\alpha_{T-1}}\approx156$. How much does bf16 change the
# samples? Run DDIM from the same $x_T$ in both precisions:

# %%
if USE_AMP:
    x_T = torch.randn(4 if FAST else 16, 1, 28, 28, generator=torch.Generator().manual_seed(3)).to(device)
    s32, tr32 = dm.ddim_sample(ref_model, schedule, x_T.shape, n_steps=50, x_T=x_T, return_traj=True)
    with autocast():
        s16, tr16 = dm.ddim_sample(ref_model, schedule, x_T.shape, n_steps=50, x_T=x_T, return_traj=True)
    diff = (s32.cpu() - s16.float().cpu()).abs()
    print(f"final samples, float32 vs bfloat16 from the same x_T: mean |diff| {diff.mean():.4f}, "
          f"max |diff| {diff.max():.3f}   (pixel range is 2)")
    for k in (0, len(tr32) // 2, len(tr32) - 1):
        d0 = (tr32[k][2] - tr16[k][2].float()).abs().mean().item()
        amp = (1 / schedule.sqrt_alpha_bars[tr32[k][0]]).item()
        print(f"x0_hat at t={tr32[k][0]:3d}: mean |diff| {d0:.4f}   (error amplification 1/sqrt(abar) = {amp:6.1f})")
else:
    print("USE_AMP is False: everything runs in float32, nothing to compare.")

# %% [markdown]
# The final samples barely change: the mean pixel difference is a small fraction of the $[-1,1]$ range, so for
# sampling bf16 is a nearly free speed-up. (A deterministic ODE sampler can still move a few pixels noticeably. Use
# float32 when you need exact reproducibility, for example for DDIM inversion.) The intermediate $\hat x_0$
# is a different story. At $t=999$ bf16 rounding of $\hat\epsilon$ is amplified about 156-fold, so the *prediction*
# is visibly corrupted even though the sample is not: in the DDPM/DDIM update, $\hat x_0$ enters with a weight of
# order $\sqrt{\bar\alpha}$, which cancels the amplification. Whenever we *look at* $\hat x_0$ at large $t$ below, we
# therefore recompute it in float32 from the stored states $x_t$. That costs one extra network call per snapshot.
#
# ### The denoising chain is coarse-to-fine
#
# Each DDPM step also gives the model's current best guess $\hat x_0(x_t,t)=\mathbb{E}[x_0\mid x_t]$. Below are
# $x_t$ and $\hat x_0$ along the DDPM chain for three of the samples above.

# %%
@torch.no_grad()
def x0_hat_fp32(net, x, t, bs=256):
    """Recompute x0_hat(x_t, t) in float32 (clipped to [-1, 1]) for a batch of stored states, on the CPU."""
    return torch.cat([dm.model_predictions(net, x[i:i + bs].to(device), t, schedule, clip_x0=1.0)[1].cpu()
                      for i in range(0, len(x), bs)])


# replace the bf16 read-outs stored in the trajectory by float32 ones: (t, x_t, x0_hat)
ddpm_traj = [(t, xt, x0_hat_fp32(ref_model, xt, t)) for t, xt, _ in ddpm_traj]
traj_t = [t for t, _, _ in ddpm_traj]
cols = [999, 800, 600, 500, 400, 300, 200, 100, 50, 0]
fig, axes = plt.subplots(6, len(cols), figsize=(10, 6.6))
for r in range(3):
    for j, tc in enumerate(cols):
        _, xt, x0h = ddpm_traj[traj_t.index(tc)]
        for k, (im, lab) in enumerate([(xt, "$x_t$"), (x0h, "$\\hat x_0$")]):
            ax = axes[2 * r + k, j]
            ax.imshow(im[r, 0], cmap="gray", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"#{r} {lab}", fontsize=9)
        axes[0, j].set_title(f"t={tc}", fontsize=9)
fig.suptitle("DDPM chain: noisy state $x_t$ and the model's current guess $\\hat x_0 = E[x_0|x_t]$")
plt.tight_layout()
plt.show()

# %% [markdown]
# Read the $\hat x_0$ rows from left to right. At $t=999$ the guess is a blurry "average digit", the prior mean.
# Through the middle of the chain a blurry but recognisable digit appears: the **global shape is decided early**.
# The last few hundred steps only sharpen strokes. We can measure when the identity of the digit is fixed by
# classifying $\hat x_0$ at each step and checking agreement with the class of the final sample:

# %%
clf = dm.load_or_train_classifier(device=device)


@torch.no_grad()
def classify(x, bs=1000):
    """Classifier probabilities for images in [-1, 1] (any device), returned on the CPU."""
    return torch.cat([clf(x[i:i + bs].to(device)).softmax(1).cpu() for i in range(0, len(x), bs)])


final_cls = classify(ddpm_x).argmax(1)
agree = [(classify(x0h.clamp(-1, 1)).argmax(1) == final_cls).float().mean().item() for _, _, x0h in ddpm_traj]
fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.plot(traj_t, agree, "o-", color=C_DDPM, ms=3)
ax.axhline(0.1, color=C_REAL, ls=":", lw=1, label="chance (10 classes)")
ax.set(xlabel="t  (axis reversed: sampling runs left to right)", ylabel="P[class of $\\hat x_0(t)$ = final class]",
       title="When is the digit's identity decided?", ylim=(0, 1.05))
ax.invert_xaxis()
ax.legend()
plt.tight_layout()
plt.show()
t_half = next((t for t, a in zip(traj_t, agree) if a >= 0.5), None)
print(f"class of x0_hat agrees with the final class for >= 50% of samples from t = {t_half} on")

# %% [markdown]
# The curve rises from near chance to almost 1 well before the end of the chain. The class is settled around the
# middle of the chain. The remaining steps change details, not identity. One caveat: the classifier was trained on
# sharp digits, so on very blurry $\hat x_0$ its answer is only a rough proxy for "what the model has committed to".
# Section 6 explains *why* generation is coarse-to-fine.
#
# ### x0-clipping
#
# The network predicts $\hat\epsilon$, and we convert it to $\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$.
# At large $t$ the factor $1/\sqrt{\bar\alpha_t}$ is huge, so small errors in $\hat\epsilon$ produce $\hat x_0$ far
# outside the data range. Since we *know* $x_0\in[-1,1]$, clipping $\hat x_0$ to that range is a free projection
# onto the data support. `dm.model_predictions(..., clip_x0=1.0)` clips, then recomputes $\hat\epsilon$ from the
# clipped $\hat x_0$ so that the pair stays consistent with $x_t$. Let's see how far out the raw predictions go along
# an unclipped DDIM trajectory, and what clipping does to few-step sampling:

# %%
n_clip = 16 if FAST else 64
x_T = torch.randn(n_clip, 1, 28, 28, generator=torch.Generator().manual_seed(4)).to(device)
_, traj_nc = dm.ddim_sample(ref_model, schedule, x_T.shape, n_steps=50, x_T=x_T, return_traj=True)  # float32
t_nc = [t for t, _, _ in traj_nc]
frac_out = [(x0h.abs() > 1).float().mean().item() for _, _, x0h in traj_nc]
max_abs = [x0h.abs().amax(dim=(1, 2, 3)).median().item() for _, _, x0h in traj_nc]

few = {}
for net_name, net in [("reference", ref_model), ("in-notebook", ema.model)]:
    for clip in (None, 1.0):
        few[(net_name, clip)] = sample_ddim(net, n_clip, steps=10, seed=5, clip=clip)

fig = plt.figure(figsize=(11, 5.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1, 1])
ax = fig.add_subplot(gs[0, 0])
ax.plot(t_nc, max_abs, "o-", ms=2, color=C_DDIM, label="median over samples of max |x̂0|")
ax.axhline(1, color="k", ls="--", lw=1)
ax.set(yscale="log", xlabel="t", ylabel="max |x̂0| per image", title="Unclipped DDIM-50: range of $\\hat x_0$")
ax.invert_xaxis()
ax.legend()
ax = fig.add_subplot(gs[1, 0])
ax.plot(t_nc, frac_out, "o-", ms=2, color=C_DDIM)
ax.set(xlabel="t", ylabel="fraction of pixels outside [-1, 1]", ylim=(0, None))
ax.invert_xaxis()
for i, net_name in enumerate(["reference", "in-notebook"]):
    for j, clip in enumerate((None, 1.0)):
        ax = fig.add_subplot(gs[i, 1 + j])
        dm.show_images(few[(net_name, clip)][:24], nrow=6, ax=ax,
                       title=f"{net_name}, DDIM-10, {'no clipping' if clip is None else 'clip x̂0 to [-1,1]'}")
plt.tight_layout()
plt.show()

for (net_name, clip), xs in few.items():
    p = classify(xs)
    print(f"{net_name:11s} DDIM-10 {'clip   ' if clip else 'no clip'}: mean classifier confidence "
          f"{p.max(1).values.mean():.3f} | mean |x| of background-ish pixels "
          f"{xs[xs < 0].abs().mean():.3f} (ideal 1.0)")

# %% [markdown]
# Early in the chain the unclipped $\hat x_0$ has values far outside $[-1,1]$ and many pixels out of range. As
# $t\to0$ it settles inside the data range. With only 10 sampling steps, every step jumps across a large range of
# noise levels and relies heavily on $\hat x_0$, so these errors leak into the samples. Compare the two columns,
# and the printed confidences and background values, for the reference model and for the under-trained one.
# Clipping is a cheap fix that is on by default in most DDPM code.
#
# **Dynamic thresholding** (Imagen, Saharia et al. 2022) is a refinement. Per sample, let $s$ be the 99.5th
# percentile of $|\hat x_0|$. If $s>1$, clip to $[-s,s]$ and divide by $s$. Static clipping saturates many pixels
# when guidance pushes $\hat x_0$ far out (notebook 06); dynamic thresholding rescales instead. It is an exercise
# below.
#
# ## 6 · Why coarse-to-fine? Diffusion as spectral autoregression
#
# Sander Dieleman's observation (2024): look at images in the **Fourier domain**. Natural images, and digits, have a
# power spectrum $P(f)$ that falls steeply with spatial frequency $f$. Gaussian noise is white: flat power at every
# frequency. With an orthonormal FFT $\mathcal{F}$, noise and signal are independent, so
#
# $$P_{x_t}(f)=\bar\alpha_t\,P_{x_0}(f)+(1-\bar\alpha_t),\qquad \mathrm{SNR}(f,t)=\frac{\bar\alpha_t\,P_{x_0}(f)}{1-\bar\alpha_t}.$$
#
# As $t$ grows, the noise floor rises uniformly and **swamps the weak high frequencies first**. The low frequencies
# survive longest. Run backwards, the reverse process must therefore produce low frequencies (global shape) first
# and high frequencies (stroke edges) last. It is an approximate autoregression from coarse to fine.
#
# The link to Bayesian denoising is direct. For a *Gaussian* image prior, the posterior mean is the **Wiener
# filter**, which shrinks every frequency by $\mathrm{SNR}/(1+\mathrm{SNR})$. Its relative error at frequency $f$ is
#
# $$\frac{\mathbb{E}\,|\mathcal{F}(\hat x_0-x_0)(f)|^2}{P_{x_0}(f)}=\frac{1}{1+\mathrm{SNR}(f,t)},$$
#
# so a frequency is "decided" (error below 1/2) once its SNR exceeds 1. Below we (a) measure spectra of noised
# MNIST, (b) turn them into this Wiener prediction, and (c) compare it with the actual reverse chain of our model.
# In (c), the final sample plays the role of $x_0$ and each $\hat x_0(t)$ is measured against it.

# %%
H = 28
fy, fx = torch.fft.fftfreq(H), torch.fft.fftfreq(H)
R_BIN = (torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2) * H).round().long()   # radius in cycles per image
FREQS = np.arange(1, H // 2 + 1)                                                # skip DC, stop at Nyquist (14)


def power_spectrum(x):
    """Radially averaged power spectrum of a batch of images (B, 1, H, W), orthonormal FFT, batch-averaged."""
    P = torch.fft.fft2(x.double().cpu(), norm="ortho").abs().pow(2).mean(dim=(0, 1))   # (H, W)
    return np.array([P[R_BIN == k].mean().item() for k in FREQS])


x_spec = x_train[:2000]
P0 = power_spectrum(x_spec)
# checks: white noise is flat at 1, and noised data obeys P_xt = abar * P_x0 + (1 - abar)
P_noise = power_spectrum(torch.randn(2000, 1, H, H))
t_chk = 300
xt_chk = schedule.q_sample(x_spec.to(device), torch.full((len(x_spec),), t_chk, device=device)).cpu()
pred = abar[t_chk].item() * P0 + (1 - abar[t_chk].item())
print(f"white noise power: min {P_noise.min():.3f}, max {P_noise.max():.3f} (should be ~1)")
print(f"P_xt vs abar*P_x0 + (1-abar) at t={t_chk}: max relative error "
      f"{np.max(np.abs(power_spectrum(xt_chk) - pred) / pred):.3f}")

# %%
t_lines = [20, 100, 200, 300, 450, 700]
t_grid = np.arange(0, 1000, 10)
ab_grid = abar[t_grid].numpy()
snr_ft = ab_grid[None, :] * P0[:, None] / (1 - ab_grid[None, :])    # (freq, t)
wiener_err = 1 / (1 + snr_ft)

# measured: per-frequency error of the chain's x0_hat(t) relative to the final sample
P_final = power_spectrum(ddpm_x)
chain_err = np.stack([power_spectrum(x0h - ddpm_x) / P_final for _, _, x0h in ddpm_traj], axis=1)  # (freq, snaps)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
cmap = plt.get_cmap("viridis")
for i, t in enumerate(t_lines):
    c = cmap(i / (len(t_lines) - 1))
    xt = schedule.q_sample(x_spec[:500].to(device), torch.full((500,), t, device=device)).cpu()
    axes[0].plot(FREQS, power_spectrum(xt), "-", color=c, lw=1.8, label=f"t={t}")
    axes[0].axhline(1 - abar[t].item(), color=c, ls=":", lw=1)
axes[0].plot(FREQS, P0, "k--", lw=1, label="clean $x_0$")
axes[0].set(xscale="log", yscale="log", ylim=(3e-3, None), xlabel="spatial frequency (cycles / image)", ylabel="power",
            title="(a) spectra of $x_t$ (dotted: noise floor $1-\\bar\\alpha_t$)")
axes[0].legend(ncol=2)
extent = [0, 999, FREQS[0] - 0.5, FREQS[-1] + 0.5]
for ax, err, ts_, title in [(axes[1], wiener_err, t_grid, "(b) Wiener prediction $1/(1+\\mathrm{SNR}(f,t))$"),
                            (axes[2], chain_err, np.array(traj_t), "(c) measured: reverse DDPM chain")]:
    order = np.argsort(ts_)
    im = ax.imshow(np.clip(err[:, order], 0, 1), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1,
                   extent=[ts_.min(), ts_.max(), extent[2], extent[3]])
    ax.contour(t_grid, FREQS, snr_ft, levels=[1.0], colors="cyan", linewidths=1.5)
    ax.set(xlabel="t", ylabel="frequency (cycles / image)", title=title)
    ax.set_xlim(999, 0)   # sampling direction: left to right
fig.colorbar(im, ax=axes[1:], label="relative error of $\\hat x_0$ at that frequency")
plt.show()

# %% [markdown]
# (a) The clean spectrum (dashed) falls by orders of magnitude from low to high frequency. Each noisy spectrum
# follows it until it hits its noise floor (dotted), then goes flat. As $t$ grows the floor rises and the
# crossing moves to lower frequencies. (b) In the Wiener picture, the region where a frequency is still
# undetermined (bright) sweeps from high to low frequencies as $t$ grows. The cyan line is $\mathrm{SNR}(f,t)=1$.
# (c) The trained model's reverse chain shows the same structure: low frequencies of $\hat x_0$ settle at large $t$
# (early in sampling), high frequencies only near the end, roughly along the same SNR = 1 line. MNIST is far from
# Gaussian, so (b) is only a guide. Where (c) is darker than (b), the model uses non-Gaussian structure (for
# example "strokes have sharp edges") to infer frequencies that a linear filter could not yet recover.
#
# This is the spectral explanation of the coarse-to-fine chain in section 5. It also explains several
# practical facts: why noise schedules matter so much (they decide how many steps are spent on each frequency
# band), why higher-resolution images need *shifted* schedules (more pixels add more high-frequency bands), and why
# sampling with very few steps loses fine detail first.
#
# ### What does the denoiser look at? Its Jacobian
#
# A second view of the same idea, closer to Bayesian inference. Differentiate the posterior mean with respect to
# the observation. Write $x_t=a\,x_0+s\,\epsilon$ with $a=\sqrt{\bar\alpha_t}$, $s^2=1-\bar\alpha_t$. Then
# $p(x_0\mid x_t)\propto p(x_0)\exp\!\big(a\,x_0^\top x_t/s^2-a^2\|x_0\|^2/2s^2\big)$ is an exponential family in
# $x_t$, and differentiating its mean gives its covariance ("second-order Tweedie"):
#
# $$\frac{\partial\,\mathbb{E}[x_0\mid x_t]}{\partial x_t}=\frac{a}{s^2}\Big(\mathbb{E}[x_0x_0^\top\mid x_t]-\mathbb{E}[x_0\mid x_t]\,\mathbb{E}[x_0\mid x_t]^\top\Big)=\frac{\sqrt{\bar\alpha_t}}{1-\bar\alpha_t}\,\mathrm{Cov}[x_0\mid x_t].$$
#
# So row $p$ of the denoiser's Jacobian is an **adaptive filter**: the weights with which pixel $p$ of $\hat x_0$
# averages the noisy input. It is proportional to the posterior covariance of pixel $p$ with every other pixel.
# Two predictions follow. The filter should widen as $t$ grows, which needs the U-Net's large receptive field.
# And the Jacobian should be **symmetric** (positive semi-definite, too), although nothing in the architecture
# enforces that.

# %%
def x0_hat_fn(net, t):
    """x -> x0_hat(x, t) for a fixed integer t (float32, differentiable)."""
    sa, s1a = schedule.sqrt_alpha_bars[t], schedule.sqrt_one_minus_alpha_bars[t]
    return lambda x: (x - s1a * net(x, torch.full((x.shape[0],), t, device=x.device))) / sa


def jacobian_rows(f, x, pixels, chunk=48):
    """Rows d f(x)[p] / d x for flat pixel indices p, using one copy of x per row in a single batch.
    This is valid only because samples in a batch do not interact (GroupNorm, not BatchNorm)."""
    rows = []
    for i in range(0, len(pixels), chunk):
        pix = pixels[i:i + chunk]
        X = x.repeat(len(pix), 1, 1, 1).requires_grad_(True)
        out = f(X).flatten(1)
        g, = torch.autograd.grad(out[torch.arange(len(pix), device=out.device), pix.to(out.device)].sum(), X)
        rows.append(g.flatten(1))
    return torch.cat(rows)


idx = int(torch.nonzero(x_test[:, 0, 14, 14] > 0.5)[0])   # a test digit with ink at the centre pixel
x_clean = x_test[idx:idx + 1].to(device)
p_centre = torch.tensor([14 * 28 + 14])
t_jac = [20, 150, 350, 600, 850]
noise = torch.randn(x_clean.shape, generator=torch.Generator().manual_seed(6)).to(device)
fig, axes = plt.subplots(2, len(t_jac), figsize=(10, 4.4))
for j, t in enumerate(t_jac):
    xt = schedule.q_sample(x_clean, torch.full((1,), t, device=device), noise)
    J_row = jacobian_rows(x0_hat_fn(ref_model, t), xt, p_centre)[0].view(28, 28).cpu()
    lim = J_row.abs().max().item()
    axes[0, j].imshow(xt[0, 0].cpu(), cmap="gray", vmin=-1, vmax=1)
    axes[0, j].set_title(f"$x_t$, t={t}")
    axes[1, j].imshow(J_row, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[1, j].set_title(f"∂x̂0[centre]/∂x_t  (max {lim:.2f})", fontsize=8)
    for r in range(2):
        axes[r, j].plot(14, 14, "+", color="gold", ms=8, mew=1.5)
        axes[r, j].set_axis_off()
fig.suptitle("Row of the denoiser's Jacobian for the centre pixel (+): the adaptive filter")
plt.tight_layout()
plt.show()

# symmetry check on the central 12x12 block of pixels (a 144 x 144 sub-matrix of the 784 x 784 Jacobian)
block_pix = torch.tensor([r * 28 + c for r in range(8, 20) for c in range(8, 20)])
t_sym = 300
xt = schedule.q_sample(x_clean, torch.full((1,), t_sym, device=device), noise)
J = jacobian_rows(x0_hat_fn(ref_model, t_sym), xt, block_pix)[:, block_pix].double().cpu()
asym = ((J - J.T).norm() / J.norm()).item()
eig = torch.linalg.eigvalsh((J + J.T) / 2)
A_rand = torch.randn(144, 144, dtype=torch.float64)
print(f"t={t_sym}: ||J - J^T|| / ||J|| = {asym:.3f}   (random matrix: {((A_rand - A_rand.T).norm() / A_rand.norm()):.2f})")
print(f"eigenvalues of the symmetric part: {eig.min():.3f} ... {eig.max():.3f}; "
      f"negative mass {eig.clamp(max=0).abs().sum() / eig.abs().sum():.1%}")

# %% [markdown]
# At small $t$ the filter is a narrow spot around the pixel, so the model barely smooths. As $t$ grows it spreads
# along the strokes and then over the whole digit, while its magnitude shrinks: the posterior relies less on the
# observation and more on the prior. At large $t$ one output pixel depends on input pixels far away, which a few 3×3
# convolutions could never do. This is the "global context" requirement from section 1, made visible.
#
# The symmetry check compares the Jacobian with its transpose. It is far more symmetric than a generic matrix,
# and its symmetric part is (nearly) positive semi-definite. The network was never told that its output is
# a posterior mean, i.e. that $\hat\epsilon$ is the gradient of a log-density (notebook 03), yet it learns
# approximately that structure. Kadkhodaie et al. (2024) build an analysis of generalisation on exactly these
# Jacobians.
#
# ## 7 · Evaluating image samples
#
# On 2-D toy data we could compare samples with the true density (MMD, sliced Wasserstein). For images there is no
# ground truth density, so practitioners use a toolbox of imperfect checks. We use three of them.
#
# ### 7a · Memorisation: nearest neighbours in the training set
#
# A model that copies training images would have perfect "quality". For each generated sample, find its nearest
# training image in pixel space. As a baseline, do the same for **held-out test images**: they are genuinely new
# digits from the same distribution. If generated samples sit much closer to the training set than test images
# do, the model memorises.

# %%
@torch.no_grad()
def nearest_train(x, k=3, chunk=256):
    """L2 (pixel-space) k nearest neighbours of each image in the training set. Returns (dists, indices)."""
    xtr = x_train.flatten(1).to(device)
    dists, idxs = [], []
    for i in range(0, len(x), chunk):
        d = torch.cdist(x[i:i + chunk].flatten(1).to(device), xtr)
        v, j = d.topk(k, dim=1, largest=False)
        dists.append(v.cpu())
        idxs.append(j.cpu())
    return torch.cat(dists), torch.cat(idxs)


nn_d_gen, nn_i_gen = nearest_train(ddim_x)
nn_d_test, _ = nearest_train(x_test[:N_EVAL])
n_rows = 6
fig = plt.figure(figsize=(11, 4.6))
gs = fig.add_gridspec(n_rows, 6, width_ratios=[1, 1, 1, 1, 0.3, 6])
for r in range(n_rows):
    ims = [ddim_x[r, 0]] + [x_train[nn_i_gen[r, k], 0] for k in range(3)]
    for c, im in enumerate(ims):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(im, cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if c > 0:
            ax.set_xlabel(f"{nn_d_gen[r, c - 1]:.1f}", fontsize=7, labelpad=1)
        if r == 0:
            ax.set_title("sample" if c == 0 else f"NN {c}", fontsize=8)
ax = fig.add_subplot(gs[:, 5])
bins = np.linspace(0, max(nn_d_gen[:, 0].max(), nn_d_test[:, 0].max()).item(), 30)
ax.hist(nn_d_test[:, 0].numpy(), bins=bins, alpha=0.6, color=C_REAL, label="held-out test images")
ax.hist(nn_d_gen[:, 0].numpy(), bins=bins, alpha=0.6, color=C_DDIM, label="generated (DDIM-50)")
ax.set(xlabel="L2 distance to nearest training image", ylabel="count",
       title="Distance to the training set (copying would pile up at 0)")
ax.legend()
plt.tight_layout()
plt.show()
print(f"median NN distance: generated {nn_d_gen[:, 0].median():.2f} | held-out test {nn_d_test[:, 0].median():.2f}"
      f" | min generated {nn_d_gen[:, 0].min():.2f}")

# %% [markdown]
# The nearest neighbours (distances under each image) are similar digits written differently, not copies. The
# two histograms overlap, and no generated sample sits at distance ≈ 0. With 60 000 training images and a small
# model, this is the expected regime: the model *generalises*. Kadkhodaie et al. (2024) show the transition. With
# few training images, the same architecture memorises (an exercise below). Two caveats. Pixel-space L2 misses
# near-copies that are shifted by a pixel. And "no copies" is not the same as "good samples".
#
# ### 7b · Mode coverage: which digits does the model generate?
#
# The unconditional model should generate all ten digits in roughly their training proportions. A model that
# dropped a class could still produce beautiful samples. We classify the samples with the pre-trained MNIST
# classifier and also record its confidence: low confidence indicates ambiguous or malformed digits.

# %%
sets = {"test images": (x_test[:N_EVAL], C_REAL), "reference DDPM-1000": (ddpm_x, C_DDPM),
        "reference DDIM-50": (ddim_x, C_DDIM), "in-notebook DDIM-50": (mine_ema, C_MINE)}
train_freq = np.bincount(y_train.numpy(), minlength=10) / len(y_train)
fig, ax = plt.subplots(figsize=(9, 3.4))
width = 0.8 / len(sets)
for i, (name, (xs, c)) in enumerate(sets.items()):
    p = classify(xs)
    freq = np.bincount(p.argmax(1).numpy(), minlength=10) / len(xs)
    ax.bar(np.arange(10) + (i - 1.5) * width, freq, width, color=c, label=f"{name} (n={len(xs)})")
    conf = p.max(1).values
    print(f"{name:22s}: mean confidence {conf.mean():.3f} | confident (p>0.9): {(conf > 0.9).float().mean():.1%}"
          f" | largest class share {freq.max():.1%}, smallest {freq.min():.1%}")
ax.plot(np.arange(10), train_freq, "k_", ms=14, mew=2, label="training-set frequency")
ax.set(xticks=range(10), xlabel="predicted digit", ylabel="fraction of samples", title="Class histogram")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
plt.tight_layout()
plt.show()

# %% [markdown]
# All ten digits appear, at frequencies near the training proportions, with sampling noise on top. With a few
# hundred samples, each bar has a standard error of about $\sqrt{0.1\cdot0.9/n}$, a few percentage points. The
# printed confidences separate fidelity from coverage. Compare each model's fraction of confidently classified
# samples with the test-image value: the gap measures ambiguous or malformed digits. The short-trained in-notebook
# model should show the largest gap.
#
# ### 7c · Fréchet distance in feature space (the idea behind FID)
#
# **FID** (Heusel et al. 2017) summarises "how far is the sample distribution from the data distribution" in one
# number. Embed real and generated images with a fixed, pre-trained network $\phi$. Fit a Gaussian to each set of
# features, and compute the Fréchet (2-Wasserstein) distance between the two Gaussians, which has a closed form:
#
# $$\mathrm{FD}=\|\mu_r-\mu_g\|^2+\mathrm{Tr}\Big(\Sigma_r+\Sigma_g-2\big(\Sigma_r\Sigma_g\big)^{1/2}\Big).$$
#
# Why features rather than pixels? Pixel-space statistics are dominated by low-level variation (stroke thickness,
# small shifts) that says little about whether an image *looks right*. The features of a network trained for
# recognition capture semantic and perceptual structure. FID proper uses the 2048-d pool features of an
# ImageNet-trained Inception-v3. For MNIST those are a poor fit (natural-image features applied to digits), so we
# use the 128-d penultimate layer of our MNIST classifier (`MNISTClassifier.features`). This is the
# same idea, sometimes called a "Fréchet classifier distance".
#
# A trace identity makes the matrix square root stable: $\mathrm{Tr}(\Sigma_r\Sigma_g)^{1/2}$ equals the sum of
# square roots of the eigenvalues of the symmetric PSD matrix $\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2}$ (a similar matrix).

# %%
def gaussian_stats(feats):
    f = feats.double().cpu().numpy()
    return f.mean(0), np.cov(f, rowvar=False)


def frechet_distance(mu1, S1, mu2, S2):
    """||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^(1/2)), via eigenvalues of S1^(1/2) S2 S1^(1/2)."""
    w, V = np.linalg.eigh(S1)
    sqrt_S1 = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    tr_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh(sqrt_S1 @ S2 @ sqrt_S1), 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(S1) + np.trace(S2) - 2 * tr_sqrt)


# checks: (1) diagonal covariances have the closed form sum (sqrt(s1) - sqrt(s2))^2; (2) agree with scipy's sqrtm
rng = np.random.default_rng(0)
m1, m2, s1, s2 = rng.normal(size=5), rng.normal(size=5), rng.uniform(0.1, 2, 5), rng.uniform(0.1, 2, 5)
closed = ((m1 - m2) ** 2).sum() + ((np.sqrt(s1) - np.sqrt(s2)) ** 2).sum()
A, B = rng.normal(size=(6, 6)), rng.normal(size=(6, 6))
S1, S2 = A @ A.T, B @ B.T
via_scipy = np.trace(S1 + S2 - 2 * linalg.sqrtm(S1 @ S2).real)
print(f"diagonal case: ours {frechet_distance(m1, np.diag(s1), m2, np.diag(s2)):.6f} vs closed form {closed:.6f}")
print(f"general case : ours {frechet_distance(np.zeros(6), S1, np.zeros(6), S2):.6f} vs scipy.sqrtm {via_scipy:.6f}")


@torch.no_grad()
def features(x, bs=1000):
    return torch.cat([clf.features(x[i:i + bs].to(device)).cpu() for i in range(0, len(x), bs)])


n_ref = 2000 if FAST else 10_000
mu_ref, S_ref = gaussian_stats(features(x_train[:n_ref]))   # reference statistics from training images
f_test = features(x_test)

# %% [markdown]
# **Pitfall 1: FD is biased upwards at small sample size.** Even two samples from the *same* distribution have
# FD > 0, because the covariance estimated from $n$ points is noisy. The bias shrinks roughly like $1/n$. Below we
# compute this "floor" for held-out test images at several $n$, then compare all models **at the same $n$**.

# %%
def fd_of_subsets(feats, n, max_rep=10, seed=0):
    """FD against the training reference for up to max_rep disjoint random subsets of size n."""
    perm = np.random.default_rng(seed).permutation(len(feats))
    reps = min(max_rep, len(feats) // n)
    return np.array([frechet_distance(*gaussian_stats(feats[perm[i * n:(i + 1) * n]]), mu_ref, S_ref)
                     for i in range(reps)])


ns = [8, 32] if FAST else [64, 128, 256, 512, 1024, 2500, 5000]
floor = {n: fd_of_subsets(f_test, n) for n in ns}

N_FD = N_DDPM   # the DDPM run is the most expensive set, so its size fixes the common n
results = {
    "test images (floor)": (fd_of_subsets(f_test, N_FD), C_REAL),
    "reference DDPM-1000": (fd_of_subsets(features(ddpm_x), N_FD), C_DDPM),
    "reference DDIM-50": (fd_of_subsets(features(ddim_x), N_FD), C_DDIM),
    "in-notebook EMA, DDIM-50": (fd_of_subsets(features(mine_ema), N_FD), C_MINE),
    "in-notebook raw, DDIM-50": (fd_of_subsets(features(mine_raw), N_FD), "tab:red"),
}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.3]})
axes[0].errorbar(ns, [floor[n].mean() for n in ns], yerr=[floor[n].std() for n in ns], fmt="o-", color=C_REAL,
                 capsize=3)
axes[0].set(xscale="log", yscale="log", xlabel="number of samples n",
            ylabel="FD(test subset, train reference)", title="Pitfall: FD of real data vs n")
names = list(results)
means = [results[k][0].mean() for k in names]
stds = [results[k][0].std() if len(results[k][0]) > 1 else 0 for k in names]
axes[1].barh(range(len(names)), means, xerr=stds, color=[results[k][1] for k in names], capsize=3)
axes[1].set_yticks(range(len(names)))
axes[1].set_yticklabels(names)
axes[1].invert_yaxis()
axes[1].set(xlabel=f"Fréchet distance in classifier features (n = {N_FD} per set)",
            title="Lower is better (error bars: sd over disjoint subsets)")
plt.tight_layout()
plt.show()
for k in names:
    v = results[k][0]
    print(f"{k:26s}: FD = {v.mean():7.2f}" + (f" ± {v.std():.2f}  ({len(v)} subsets)" if len(v) > 1 else "  (1 set)"))

# %% [markdown]
# **Left:** even real data has a clearly non-zero FD at small $n$, and the value keeps falling as $n$ grows. FD values
# computed at different $n$ (or with different feature networks, image resizing, or libraries) are **not
# comparable**. Published FIDs use 50 000 samples and a fixed Inception implementation for this reason.
#
# **Right**, everything at the same $n$: the real-data floor is lowest, the reference model comes next, and the
# short-trained in-notebook model is clearly worse. Compare the DDPM-1000 and DDIM-50 bars against their error
# bars to see whether 20× fewer steps cost measurable quality here, and the raw and EMA bars to see what averaging
# bought in our short run. With only $n=$ `N_FD` samples per set, small differences are within noise. On a GPU,
# raise `N_EVAL`/`N_DDPM` to make these comparisons sharp.
#
# **Further pitfalls of FID** to keep in mind:
# * **It is blind to memorisation.** Returning the training set verbatim gives a near-floor FD. Pair it with a
#   nearest-neighbour check (7a).
# * **It mixes fidelity and diversity into one number.** Precision/recall-style metrics (e.g. Kynkäänniemi et al.
#   2019) separate "are samples realistic?" from "is the whole distribution covered?".
# * **It assumes Gaussian features and inherits the biases of the feature network.** Inception features emphasise
#   ImageNet textures and object categories, and FID can disagree with human judgement. Always look at samples too.
# * **Implementation details move the number**: resizing filters, JPEG compression, float vs uint8 images.
#
# ## Exercises
#
# **Exercise 1 (architecture ablation).** Predict first, then test. (a) Replace the U-Net by an MLP on flattened
# images with *more* parameters (wrapper below), train it with the loop of section 4 for the same number of steps,
# and compare `loss_per_t` and DDIM samples. (b) Remove all attention from the U-Net:
# `net = dm.UNet(**{**dm.MNIST_RECIPE, "attn_levels": ()}); net.mid[1] = nn.Identity()` (the middle block always
# contains one `AttnBlock`). Which change hurts more at this resolution, and why?
#
# ```python
# class FlatMLP(nn.Module):                 # ~11M parameters vs the U-Net's 1.2M
#     def __init__(self):
#         super().__init__()
#         self.mlp = dm.MLPDenoiser(data_dim=784, hidden=1024, n_blocks=4)
#     def forward(self, x, t, y=None):
#         return self.mlp(x.flatten(1), t).view_as(x)
# ```
#
# <details><summary>Hint / solution</summary>
#
# Expect the MLP to be clearly worse at equal training steps despite ~10× more parameters: noisier, blotchier
# samples and a higher loss at small and intermediate $t$, where local stroke detail must be restored at every
# position. It has to learn the same local operation 784 times over (section 1). Removing attention should cost
# much less at 28×28, because two stride-2 down-samplings plus a stack of 3×3 convs already give the bottleneck a
# receptive field covering the whole digit. Attention earns its keep at higher resolutions and for long-range
# consistency (Dhariwal & Nichol 2021 ablate this). Plot `loss_per_t(new_model, x_eval, ts_loss)` next to the curves
# of section 4. A GPU makes this quick.
# </details>
#
# **Exercise 2 (EMA decay).** Before running anything: in a 1500-step run, what changes if you use
# `dm.EMA(model, decay=0.9999)` instead of 0.999? And with `decay=0.99`?
#
# <details><summary>Hint / solution</summary>
#
# Look at the warm-up formula $d_n=\min\{d,(1+n)/(10+n)\}$. $(1+n)/(10+n)<0.999$ for all $n<8990$, so for
# a 1500-step run 0.999 and 0.9999 are **identical**. The cap never binds. With 0.99 the cap binds after
# $n\approx890$ steps and the average covers only the last ~100 steps. The difference matters only in long runs,
# which is why large models use decays like 0.9999 together with long training. EDM2 (Karras et al. 2024) goes
# further and chooses the EMA length *after* training.
# </details>
#
# **Exercise 3 (dynamic thresholding).** Write your own DDIM loop (copy `dm.ddim_sample`, ~15 lines) that applies
# dynamic thresholding to $\hat x_0$. Compare it with static clipping and with no clipping for DDIM-10 on the
# in-notebook model, using the grids and the classifier confidence from section 5.
#
# <details><summary>Hint / solution</summary>
#
# ```python
# def dynamic_threshold(x0, p=0.995):
#     s = torch.quantile(x0.abs().flatten(1), p, dim=1).clamp(min=1.0).view(-1, 1, 1, 1)
#     return x0.clamp(-s, s) / s
# # inside the loop, after eps, x0 = dm.model_predictions(net, x, t, schedule):
# #     x0 = dynamic_threshold(x0); eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
# ```
# On MNIST, static and dynamic thresholding behave similarly, because almost all pixels are near ±1 anyway. The
# difference shows up with strong guidance (notebook 06), where static clipping produces saturated, "burnt" images.
# </details>
#
# **Exercise 4 (spectra of generated images).** Compare `power_spectrum(ddim_x)` and `power_spectrum(mine_ema)` with
# the spectrum of real digits `P0` on a log-log plot. Which frequencies does the short-trained model get wrong, and
# is that consistent with the per-$t$ loss curves?
#
# <details><summary>Hint / solution</summary>
#
# `plt.loglog(FREQS, P0); plt.loglog(FREQS, power_spectrum(ddim_x)); plt.loglog(FREQS, power_spectrum(mine_ema))`.
# High frequencies are generated last, at small $t$, and the in-notebook model's deficit is largest at small $t$
# (section 4). So expect its spectrum to deviate most at high frequencies: excess power from residual noise, or a
# deficit from blurry strokes. Low frequencies (overall layout and ink amount) are usually already close.
# </details>
#
# **Exercise 5 (memorisation on a tiny training set).** Train the same U-Net on only the first 64 training
# images (`data = x_train[:64].to(device)`) for a few thousand steps (GPU recommended), then rerun the
# nearest-neighbour check of 7a. What happens to the distance histogram? And what happens to the FD, measured
# against those 64 images and against the full training set?
#
# <details><summary>Hint / solution</summary>
#
# With 64 images the model has enough capacity to learn the *empirical* distribution, a mixture of 64 deltas, whose
# exact denoiser is a softmax-weighted average of training images (notebook 03). Samples then reproduce training
# images: the NN histogram collapses towards 0. FD against those 64 images becomes small, which shows pitfall 1 of
# FID: it rewards copying. Kadkhodaie et al. (2024) map out this memorisation-to-generalisation transition as a
# function of training-set size.
# </details>
#
# ## Key takeaways
#
# * Going from 2-D to images changes the **network**, not the maths. A U-Net builds in locality and translation
#   equivariance (convolutions), multi-scale processing (down/up-sampling with skip connections) and global context
#   (attention at low resolution).
# * The noise level enters every residual block through **FiLM** (per-channel scale and shift after GroupNorm).
#   **GroupNorm**, not BatchNorm, because the samples in a batch sit at different noise levels. **Zero-initialised**
#   residual branches and outputs make the fresh network a well-behaved identity and zero map. The initial loss is
#   therefore exactly 1.
# * The training recipe: data scaled to $[-1,1]$, AdamW at 2e-4 with warm-up, gradient clipping, uniform random $t$,
#   bf16 autocast, checkpoints that include optimiser and EMA state. **Sample with EMA weights.**
# * The $\epsilon$-loss is an SNR-weighted $x_0$-loss. It is largest at small $t$ and noisy because of random $t$. The
#   posterior-mean view explains its limits, for example that at $t\to T$ the $x_0$-error equals the prior variance.
# * DDIM-50 gives samples comparable to DDPM-1000 at 1/20 of the cost. x0-clipping is a free projection onto the
#   known data range that matters most for few-step or under-trained sampling.
# * Sampling is **coarse-to-fine** because the noise floor swamps the weak high frequencies first. In the Wiener /
#   posterior-mean picture, a frequency is decided once its SNR exceeds 1. The denoiser's Jacobian is an adaptive
#   filter that widens with $t$, proportional to the posterior covariance.
# * Evaluate with several imperfect tools: nearest neighbours (memorisation), class histograms (coverage),
#   Fréchet distances in a suitable feature space (always at equal $n$), and your own eyes.
#
# ## Further reading
#
# * Ho, Jain & Abbeel (2020). *Denoising Diffusion Probabilistic Models.* arXiv:2006.11239 — the U-Net recipe used here.
# * Ronneberger, Fischer & Brox (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* arXiv:1505.04597.
# * Perez et al. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer.* arXiv:1709.07871.
# * Wu & He (2018). *Group Normalization.* arXiv:1803.08494.
# * Nichol & Dhariwal (2021). *Improved Denoising Diffusion Probabilistic Models.* arXiv:2102.09672.
# * Dhariwal & Nichol (2021). *Diffusion Models Beat GANs on Image Synthesis.* arXiv:2105.05233 — architecture ablations (attention at several resolutions, adaptive GroupNorm).
# * Song, Meng & Ermon (2021). *Denoising Diffusion Implicit Models.* arXiv:2010.02502.
# * Kingma, Salimans, Poole & Ho (2021). *Variational Diffusion Models.* arXiv:2107.00630 — SNR view of loss weighting, low-discrepancy sampling of $t$.
# * Karras et al. (2022). *Elucidating the Design Space of Diffusion-Based Generative Models (EDM).* arXiv:2206.00364.
# * Karras et al. (2024). *Analyzing and Improving the Training Dynamics of Diffusion Models (EDM2).* arXiv:2312.02696 — post-hoc EMA.
# * Saharia et al. (2022). *Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Imagen).* arXiv:2205.11487 — dynamic thresholding.
# * Sander Dieleman (2024). *Diffusion is spectral autoregression.* Blog post, sander.ai.
# * Kadkhodaie, Guth, Simoncelli & Mallat (2024). *Generalization in diffusion models arises from geometry-adaptive harmonic representations.* arXiv:2310.02557.
# * Efron (2011). *Tweedie's formula and selection bias.* JASA.
# * Heusel et al. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.* arXiv:1706.08500 — introduces FID.
# * Kynkäänniemi et al. (2019). *Improved Precision and Recall Metric for Assessing Generative Models.* arXiv:1904.06991.
