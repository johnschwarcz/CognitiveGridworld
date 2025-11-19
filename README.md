# CognitiveGridworld

**CognitiveGridworld** is a Python testbed for studying how models can generalise from limited experience.  It provides a configurable grid‑world environment, Bayesian inference routines and neural architectures for learning representations and policies.  The code can be used either as a cognitive simulator (to sample behaviours under different assumptions about the environment) or to train reinforcement‑learning (RL) agents that must learn to act from sparse observations.

## Overview

The repository is organised into several modules.  The root contains only a small README and a `packages.txt` file for reproducing the exact Conda environment.  Almost all of the functionality lives under the `main` folder, which is further divided into:

- **`env`** – Environment generators, Bayesian inference and preprocessing utilities.  These files define the probabilistic dynamics of the gridworld, generate contexts and observations, and allocate all of the required arrays in advance for efficiency.
- **`env_plotting`** – A set of helper functions for visualising flows and model performance.  These scripts build flow‑fields, animated plots and static figures for papers and presentations.
- **`model`** – Neural network architectures and controllers.  The `Model_architecture` class extends PyTorch modules to include classifiers, generators and controllers.  It provides methods for initialising hidden embeddings, LSTM‑based classifiers and generators that map latent states into observable distributions.
- **`REPLICATE_RESULTS`** – Scripts for replicating experiments from the associated research.  These include data collectors, plotting utilities and training scripts for RL controllers.  For example, `train_RL_networks.py` launches a full RL experiment with specified hyper‑parameters, and `train_sanity_networks.py` runs smaller sanity‑check experiments with or without reservoir connections.
- **`utils.py`** – A small utility file with helper functions used throughout the project.

The top‑level module `CognitiveGridworld.py` ties everything together.  When you instantiate `CognitiveGridworld`, the environment is preprocessed, state embeddings are generated, and either a neural model or a Bayesian simulator is run.  When plotting is enabled, a set of diagnostic plots (likelihood surface, Bayesian performance and trajectories) is automatically produced.

## Installation

The project is built for Python 3.8+ and relies on standard scientific computing libraries such as NumPy, PyTorch, Matplotlib and tqdm.  There is no `requirements.txt`, but two options are available:

1. **Conda environment** – the repository includes a `packages.txt` file that captures the exact versions of every package used in the original experiments.  You can reproduce the environment with:

   ```sh
   conda create --name coggridworld --file packages.txt
   conda activate coggridworld


2. **Pip installation** – alternatively, install the core dependencies manually:

   ```sh
   pip install numpy matplotlib tqdm torch
   ```

Some experiments require a CUDA‑enabled GPU for training neural networks.  If CUDA is not available, the code will automatically fall back to the CPU.

## Quick start

The easiest way to get started is to run the environment from Python.  The following example creates a small gridworld with default settings:

```python
from main.CognitiveGridworld import CognitiveGridworld

# initialise the environment and run a short simulation
cg = CognitiveGridworld(episodes=10,
                        state_num=500,
                        obs_num=5,
                        ctx_num=2,
                        show_plots=True)
```

Upon instantiation the simulator will preprocess the environment, generate state embeddings and run either Bayesian inference or the chosen model.  If `show_plots=True` the likelihood surface, Bayesian performance and a sample trajectory will be displayed.

### Training reinforcement‑learning agents

For full RL experiments, use the scripts in the `REPLICATE_RESULTS` folder:

* **Train RL networks**: run `train_RL_networks.py` to train a neural controller in RL mode.  The script specifies parameters such as the number of episodes, batch size and latent dimensionality.  You can adjust these values inside the script to match your compute budget.

  ```sh
  python main/REPLICATE_RESULTS/train_RL_networks.py
  ```

* **Sanity‑check experiments**: `train_sanity_networks.py` runs smaller experiments with and without reservoir networks.  It instantiates `CognitiveGridworld` several times with different settings for context dimension and reservoir usage.

  ```sh
  python main/REPLICATE_RESULTS/train_sanity_networks.py
  ```

* **Controller comparison**: scripts such as `compare_controllers.py` and `compare_RL updated.py` compare the performance of trained controllers against Bayesian agents and naive baselines.  They load previously saved environments and plot metrics such as reward curves, policy landscapes and generalisation performance.

### Environment customisation

When initialising `CognitiveGridworld`, you can pass a dictionary of keyword arguments to customise the simulation.  Some notable options include:

* `episodes`: total number of learning episodes (default 1).
* `state_num`: number of hidden states in the grid (default 5000).
* `obs_num`: number of possible observations per state (default 5).
* `ctx_num`: number of contexts (latent factors affecting dynamics).
* `realization_num`: number of realisations of the latent variables.
* `step_num`: number of steps per episode.
* `hid_dim`: size of the hidden layer used in neural models (default 1000).
* `learn_embeddings`: whether to learn internal embeddings (`True`) or use fixed embeddings (`False`).
* `reservoir`: toggles the use of a reservoir network in sanity experiments.
* `mode`: selects between Bayesian inference (default), RL (`"RL"`) or sanity mode (`"SANITY"`).  In RL mode, the classifier and controller networks are trained jointly; in sanity mode, only simple sanity networks are trained.

Most scripts set these options explicitly; you can edit the script files to try different configurations.

## Directory structure

```text
CognitiveGridworld/
├── README.md                 ← original (minimal) README supplied by the author
├── packages.txt              ← Conda environment specification
└── main/
    ├── CognitiveGridworld.py ← top‑level class orchestrating environment and model
    ├── env/                  ← environment definitions, generators and preprocessing
    ├── env_plotting/         ← plotting helpers and flow field visualisations
    ├── model/                ← neural architectures and controllers
    ├── REPLICATE_RESULTS/    ← scripts to reproduce experiments and generate plots
    └── utils.py              ← utility functions
```

