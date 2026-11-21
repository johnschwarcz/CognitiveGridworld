# Diffusion models, from first principles to practice

A hands-on course of ten Jupyter notebooks. It starts from "add a little Gaussian noise" and ends with Stable
Diffusion, flow matching and diffusion policies for control. Every algorithm is **derived, then written out by hand in
a few lines of PyTorch, then checked numerically**: against a closed form, against an exact ground truth, or against
the shared library [`dmtools.py`](dmtools.py) that later notebooks import.

Everything in the core path runs on a laptop CPU. A GPU makes notebooks 05–09 faster and is needed only for the
optional real Stable Diffusion part of notebook 09.

## Why this tutorial is built the way it is

- **There is always a ground truth.** Notebooks 01–04, 06–08 and 10 do most of their work in 2-D, on Gaussian
  mixtures whose noised densities, scores, velocities and posteriors are known exactly
  (`dmtools.GaussianMixture`). This lets you separate *model error* (the network learned the wrong score) from
  *sampler error* (the ODE/SDE solver took too few steps) from *finite-sample noise*. Most tutorials cannot make
  that distinction.
- **Bayesian inference throughout.** The denoiser is a posterior mean, E[x₀ | xₜ] (Tweedie's formula). The reverse
  step is a Kalman-like update. Guidance is Bayes' rule for scores. Inpainting and super-resolution are posterior
  sampling. A diffusion policy samples p(action chunk | state) instead of regressing to its mean. The notebooks make
  these readings explicit and check them numerically.
- **Real use cases.** These include MNIST image generation with a U-Net, classifier-free guidance, inpainting with
  RePaint, SDEdit, diffusion posterior sampling for inverse problems, DDIM inversion and editing, latent diffusion
  with a hand-built miniature Stable Diffusion, `diffusers` schedulers, and diffusion policies with receding-horizon
  control and guided planning.
- **Gotchas are shown, not just listed.** Examples include the off-by-one between the maths index and the array
  index, bf16 error amplification at large t, the EMA warm-up, the trailing vs leading timestep grid, guidance
  overshoot, and the bias of Fréchet distances at small n.

## Roadmap

| # | Notebook | What you will be able to do | CPU time* |
|---|----------|-----------------------------|-----------|
| 01 | [The forward process](01_forward_process.ipynb) | Derive q(xₜ\|x₀) and the reverse posterior q(xₜ₋₁\|xₜ,x₀). Compare linear and cosine schedules through their SNR curves. See *what* noise destroys first (fine detail before coarse structure), why the reverse step is Gaussian only for small βₜ, and how VP and VE relate. No networks. | ~2 min |
| 02 | [DDPM from scratch (2-D)](02_ddpm_from_scratch.ipynb) | Go from the ELBO to L_simple. Write the time embedding, an MLP denoiser, the training loop, EMA and ancestral sampling by hand. Compare with a Bayes-optimal oracle denoiser. Run ε/x₀/v ablations, change T, and evaluate samples with MMD. | ~6 min |
| 03 | [The score-function view](03_score_matching.ipynb) | Scores, Langevin dynamics and its failure modes, annealed Langevin, denoising score matching (= DDPM), Tweedie's formula, learned vs exact scores, and why the *exact* score of a training set only memorises. | ~3 min |
| 04 | [Samplers](04_samplers.ipynb) | The VP-SDE, the reverse SDE and the probability-flow ODE. DDPM, Euler, Heun and DDIM, with their convergence orders. DDIM derived as Euler in the right coordinates, and DDIM inversion. Karras/EDM step placement, exact likelihoods via the ODE, and practical sampler guidance. | ~4 min |
| 05 | [A U-Net on MNIST](05_images_unet.ipynb) | Build the U-Net piece by piece (FiLM ResBlocks, GroupNorm, attention) and train it with the full recipe (AdamW, warm-up, clipping, EMA, bf16). Sample with DDPM and DDIM, and see why x₀-clipping is needed. Cover diffusion as spectral autoregression, the denoiser Jacobian as an adaptive filter, and evaluation (nearest neighbours, Fréchet distance and its pitfalls). | ~15 min |
| 06 | [Guidance](06_guidance.ipynb) | Conditional models, Bayes' rule for scores, and classifier guidance with an exact noise-aware classifier. Derive classifier-free guidance, see why guidance does *not* sample p(x)p(y\|x)ʷ, trace the fidelity–diversity trade-off on MNIST, and try guidance intervals, CFG-rescale and negative prompts. | ~10 min |
| 07 | [Applications](07_applications.ipynb) | Use a trained model as a prior: inpainting (replacement vs RePaint, against an exact 2-D posterior), SDEdit, diffusion posterior sampling for super-resolution and sparse recovery, denoising, the perception–distortion trade-off, and DDIM inversion for interpolation and class swaps. | ~5 min |
| 08 | [Flow matching](08_flow_matching.ipynb) | Go from continuity equation to the CFM loss, and compute the exact velocity field of a mixture. Train a flow-matching model, and see why it is diffusion in disguise (DDIM = Euler). Straighten trajectories with rectified flow (reflow) and minibatch OT couplings, then apply CFG to velocities. | ~7 min |
| 09 | [Latent diffusion and Stable Diffusion](09_latent_and_stable_diffusion.ipynb) | **A:** a KL autoencoder, the latent scale factor as a log-SNR shift, a miniature class-conditional "Stable Diffusion", pixel vs latent cost, img2img `strength`, and our model under `diffusers` schedulers. **B (GPU, optional):** take real SD apart: a hand-written sampling loop, schedulers, guidance, negative prompts, img2img, inpainting, memory tricks, and one-step SD-Turbo. | ~15 min (A) |
| 10 | [Diffusion for control](10_diffusion_for_control.ipynb) | Diffusion policies vs behaviour cloning on a bimodal expert, the noisiest denoiser as the BC regressor, closed-loop receding-horizon control, planning as inpainting (Diffuser), and test-time guidance around a new obstacle. | ~4 min |

\* Measured on a 4-core CPU with nothing else running. A CUDA GPU cuts notebooks 05–09 to a few minutes each. The
2-D notebooks use tiny networks and gain little from a GPU. Each notebook states its own runtime at the top.

Each notebook ends with **exercises** (with hints and solutions), **key takeaways** and **further reading**.

### Suggested paths

- **The full course:** 01 → 10 in order. Notebooks 01–04 are the theory core, 05–07 the image practice, and 08–10 the
  modern extensions.
- **"I want to understand it properly, fast":** 01 §3, §5 and §7, then 02, 03, 04, then 06 §1–3.
- **"I want to generate images":** 02 (skim), 05, 06, 07, 09.
- **"I work on RL and control":** 01 (skim), 02, 04 §1–3, 06 §1–3, 10, then 08.
- **Bayesian and neuroscience angles:** 01 §6 (a Bayesian observer of noised data), 01 §7 (the Kalman analogy), 03 §6
  (Tweedie), 05 §6 (spectral autoregression, the denoiser as an adaptive filter), 07 §1 (a model as a prior), 08 §4
  (posterior-mean paths as evidence accumulation).

## Setup

```bash
cd diffusion_tutorial
python -m venv .venv && source .venv/bin/activate      # or a conda env with Python >= 3.10
pip install -r requirements.txt                          # torch, numpy, scipy, matplotlib, scikit-learn, jupyterlab, diffusers, ...
jupyter lab
```

- Install PyTorch with CUDA support in the usual way (see pytorch.org) if you have an NVIDIA GPU. The notebooks pick
  `cuda`, then Apple's `mps`, then `cpu` automatically (`dmtools.get_device()`).
- MNIST (11 MB) is downloaded by torchvision into `data/` the first time it is needed.
- **Smoke-test mode:** `DM_FAST=1 jupyter lab` (or `DM_FAST=1 python tools/build.py --execute`) shrinks every
  training run and sample count so that each notebook runs in a minute or two. Results are then poor, but every cell runs.
  Use it to check your environment, or when you only want to read the code paths.
- **Real Stable Diffusion (notebook 09 Part B):** `pip install "transformers>=5.23" "accelerate>=1.16"`, use a GPU
  with ≥ 6 GB of memory and internet access to the Hugging Face Hub, and set `RUN_SD = True` in the notebook.

## What is in this folder

```
diffusion_tutorial/
├── 01_forward_process.ipynb … 10_diffusion_for_control.ipynb   executed notebooks (outputs included)
├── src/01_forward_process.py …                                 the same notebooks as plain Python ("# %%" cells)
├── dmtools.py              shared library: everything the notebooks write by hand, collected for re-use
├── checkpoints/            pre-trained MNIST models (see below); notebooks cache further models here
├── pretrain_checkpoints.py re-creates the MNIST checkpoints
├── tools/build.py          src/*.py -> *.ipynb (with jupytext), optionally executing them
├── tools/extract_figures.py dumps the figures and text outputs of an executed notebook
└── requirements.txt
```

**Editing.** The `.py` files in `src/` are the sources of truth, in jupytext's "percent" format. You can open them
directly in VS Code or PyCharm as runnable cells, or diff them in git. After editing one, rebuild its notebook with
`python tools/build.py 05` (convert only) or `python tools/build.py 05 --execute` (convert and run). You can also
just edit the `.ipynb` directly if you do not care about keeping the two in sync.

### Checkpoints

Three small models ship in `checkpoints/` (about 11 MB in total), so that the image notebooks do not have to train
for an hour on a CPU:

| file | what | used by |
|------|------|---------|
| `mnist_classifier.pt` | a small CNN (~98.5% test accuracy), used only as an *evaluator* | 05, 06, 07, 09 |
| `mnist_unet_uncond.pt` | unconditional U-Net (`dm.MNIST_RECIPE`, 1.2 M parameters), EMA weights | 05, 09 |
| `mnist_unet_cond.pt` | class-conditional U-Net, 10% label dropout for classifier-free guidance | 06, 07 |

Both U-Nets use the recipe of notebook 05 (linear schedule with T = 1000, ε-prediction, AdamW at 2e-4 with warm-up,
EMA 0.999), trained for 12 000 steps at batch 64 with bf16 autocast on a CPU. To re-create them, run
`python pretrain_checkpoints.py` (a few minutes per model on a GPU, about an hour on a CPU). If a file is missing, the
notebooks train it on first use and cache it. Notebook 05 also saves the model it trains itself (`05_my_unet.pt`, to
demonstrate checkpointing and resuming). Notebook 09 caches its autoencoder and latent model (`09_latent_mnist.pt`)
and reloads them on later runs; set `RETRAIN = True` there to train them again.

## `dmtools` at a glance

```python
import dmtools as dm
device = dm.get_device()

# data and exact ground truth
gmm = dm.default_gmm()                          # 2-D, 8-component mixture: .sample, .log_prob, .score, .noised(abar)
x0 = gmm.sample(10_000)
x, y = dm.load_mnist(train=True)                # images in [-1, 1], shape (N, 1, 28, 28)

# forward process
sched = dm.NoiseSchedule(T=1000, kind="linear", device=device)   # .alpha_bars, .q_sample(x0, t), .posterior_*
xt = sched.q_sample(x0.to(device), torch.full((len(x0),), 500, device=device))

# models and training
net = dm.MLPDenoiser(data_dim=2)                # or dm.UNet(**dm.MNIST_RECIPE, n_classes=10)
out = dm.train_diffusion(net, x0, sched, steps=3000, pred="eps")  # AdamW + warm-up + clipping + EMA (+ bf16 on GPU)
model = out["ema"].model

# sampling (eps-, x0- or v-prediction; classifier-free guidance with guidance_scale; x0 clipping)
samples = dm.ddim_sample(model, sched, (2000, 2), n_steps=50)    # or dm.ddpm_sample(...), dm.ddim_invert(...)
print(dm.mmd_rbf(samples, gmm.sample(2000)))    # sample-quality metric for low-dimensional data
```

Conventions (also in the docstring of `dmtools.py`):

- The array index `t` runs over {0, …, T−1}, so the code's `t` is the maths' t − 1.
- `x_t = sqrt(abar_t) x_0 + sqrt(1 − abar_t) ε`.
- Every model is called as `model(x, t, y=None)`. The label `n_classes` means "no label".
- Guidance follows `eps = eps_u + w (eps_c − eps_u)`, so w = 0 is unconditional and w = 1 is plain conditional.
- Flow matching (notebook 08) uses continuous t ∈ [0, 1], with data at t = 0 and noise at t = 1.

## Cheat sheet

Here t = 1…T, αₜ = 1 − βₜ and $\bar\alpha_t=\prod_{s\le t}\alpha_s$.

| | |
|---|---|
| forward step | $q(x_t\mid x_{t-1})=\mathcal N\big(\sqrt{\alpha_t}\,x_{t-1},\ \beta_t I\big)$ |
| any t in one shot | $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\quad \varepsilon\sim\mathcal N(0,I)$; $\ \mathrm{SNR}(t)=\bar\alpha_t/(1-\bar\alpha_t)$ |
| reverse posterior | $q(x_{t-1}\mid x_t,x_0)=\mathcal N\big(\tfrac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+\tfrac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,\ \tilde\beta_t I\big)$, $\ \tilde\beta_t=\tfrac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ |
| training loss | $L_\text{simple}=\mathbb E_{t,x_0,\varepsilon}\,\lVert\varepsilon-\varepsilon_\theta(x_t,t)\rVert^2$ (minimiser: $\varepsilon_\theta=\mathbb E[\varepsilon\mid x_t]$) |
| other targets | $x_0$-prediction; $v=\sqrt{\bar\alpha_t}\,\varepsilon-\sqrt{1-\bar\alpha_t}\,x_0$ |
| score | $\nabla_x\log q_t(x_t)=-\,\mathbb E[\varepsilon\mid x_t]/\sqrt{1-\bar\alpha_t}\approx-\varepsilon_\theta(x_t,t)/\sqrt{1-\bar\alpha_t}$ |
| Tweedie (denoiser) | $\hat x_0=\mathbb E[x_0\mid x_t]=\big(x_t+(1-\bar\alpha_t)\nabla\log q_t(x_t)\big)/\sqrt{\bar\alpha_t}=\big(x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon_\theta\big)/\sqrt{\bar\alpha_t}$ |
| DDPM step | $x_{t-1}=\tfrac{1}{\sqrt{\alpha_t}}\big(x_t-\tfrac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta\big)+\sigma_t z$, $\ \sigma_t^2\in\{\beta_t,\tilde\beta_t\}$ |
| DDIM step (η = 0) | $x_{s}=\sqrt{\bar\alpha_{s}}\,\hat x_0+\sqrt{1-\bar\alpha_{s}}\,\varepsilon_\theta$ for any earlier $s<t$ |
| VP-SDE | $dx=-\tfrac12\beta(t)x\,dt+\sqrt{\beta(t)}\,dw$; reverse: $dx=\big[-\tfrac12\beta x-\beta\nabla\log p_t\big]dt+\sqrt\beta\,d\bar w$ |
| probability-flow ODE | $\dfrac{dx}{dt}=-\tfrac12\beta(t)\big[x+\nabla\log p_t(x)\big]$ (same marginals, deterministic) |
| classifier guidance | $\nabla\log p_t(x\mid y)=\nabla\log p_t(x)+w\,\nabla\log p_t(y\mid x)$ |
| classifier-free guidance | $\tilde\varepsilon=\varepsilon_\theta(x_t,\varnothing)+w\big(\varepsilon_\theta(x_t,y)-\varepsilon_\theta(x_t,\varnothing)\big)$ |
| flow matching | $x_t=(1-t)x_0+t\varepsilon$, $\ L=\mathbb E\lVert v_\theta(x_t,t)-(\varepsilon-x_0)\rVert^2$, sample $\dot x=v_\theta$ from $t=1$ to $0$ |
| DPS (inverse problems) | after each reverse step: $x_{t-1}\leftarrow x_{t-1}-\zeta\,\nabla_{x_t}\lVert y-A\,\hat x_0(x_t)\rVert$ |

## References

The foundations:

- Sohl-Dickstein et al. (2015), *Deep unsupervised learning using nonequilibrium thermodynamics*, arXiv:1503.03585
- Ho, Jain & Abbeel (2020), *Denoising diffusion probabilistic models*, arXiv:2006.11239
- Song & Ermon (2019), *Generative modeling by estimating gradients of the data distribution*, arXiv:1907.05600
- Song et al. (2021), *Score-based generative modeling through stochastic differential equations*, arXiv:2011.13456
- Song, Meng & Ermon (2021), *Denoising diffusion implicit models*, arXiv:2010.02502
- Karras et al. (2022), *Elucidating the design space of diffusion-based generative models*, arXiv:2206.00364

Guidance, latents and applications:

- Nichol & Dhariwal (2021), *Improved DDPM*, arXiv:2102.09672; Dhariwal & Nichol (2021), *Diffusion models beat GANs
  on image synthesis*, arXiv:2105.05233
- Ho & Salimans (2022), *Classifier-free diffusion guidance*, arXiv:2207.12598
- Rombach et al. (2022), *High-resolution image synthesis with latent diffusion models*, arXiv:2112.10752
- Meng et al. (2022), *SDEdit*, arXiv:2108.01073; Lugmayr et al. (2022), *RePaint*, arXiv:2201.09865; Chung et al.
  (2023), *Diffusion posterior sampling for general noisy inverse problems*, arXiv:2209.14687

Flow matching:

- Lipman et al. (2023), *Flow matching for generative modeling*, arXiv:2210.02747
- Liu, Gong & Liu (2023), *Flow straight and fast: rectified flow*, arXiv:2209.03003
- Albergo & Vanden-Eijnden (2023), *Building normalizing flows with stochastic interpolants*, arXiv:2209.15571
- Tong et al. (TMLR 2024; arXiv 2023), *Improving and generalizing flow-based generative models with minibatch optimal transport*,
  arXiv:2302.00482

Architectures and control:

- Peebles & Xie (2023), *Scalable diffusion models with transformers (DiT)*, arXiv:2212.09748
- Esser et al. (2024), *Scaling rectified flow transformers for high-resolution image synthesis (SD3)*,
  arXiv:2403.03206
- Janner et al. (2022), *Planning with diffusion for flexible behavior synthesis (Diffuser)*, arXiv:2205.09991
- Chi et al. (2023), *Diffusion Policy*, arXiv:2303.04137

Denoisers as priors (for the Bayesian and neuroscience reader):

- Kadkhodaie & Simoncelli (2021), *Solving linear inverse problems using the prior implicit in a denoiser*,
  arXiv:2007.13640
- Kadkhodaie et al. (2024), *Generalization in diffusion models arises from geometry-adaptive harmonic
  representations*, arXiv:2310.02557

Overviews:

- Luo (2022), *Understanding diffusion models: a unified perspective*, arXiv:2208.11970
- Lilian Weng, *What are diffusion models?* (blog, 2021); Sander Dieleman's blog (*Perspectives on diffusion*,
  *Diffusion is spectral autoregression*, and others)

Each notebook's own *Further reading* section gives more targeted references.
