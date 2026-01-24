# CognitiveGridworld

**CognitiveGridworld** is a stationary POMDP for studying compositional generalization in latent space.

## Installation

The project is built for Python 3.8+ and relies on libraries such as NumPy and PyTorch. To reproduce the environment do:

```sh
   conda create --name CG --file packages.txt
   conda activate CG
```

## Quick start

The easiest way to get started is with `example.py`. 

`example.py` is an example entrypoint which will preprocess an environment and loop through episodes. 

Default values of all tune-able hyperparameters can be found in `main/CognitiveGridworld.py`. 

When initializing a `CognitiveGridworld`, you can pass a dictionary of keyword arguments.  Notable options include:
* `episodes`: total number of learning episodes for networks.
* `state_num`: size of the State Space.
* `obs_num`: dimensionality of the Observation Space.
* `ctx_num`: number of active states per contexts.
* `realization_num`: number of potential realizations of an active state.
* `step_num`: number of inference steps per episode.
* `hid_dim`: size of the hidden layer used in neural models.
* `learn_embeddings`: whether to learn the embeddings (`True`) or provide the true embeddings (`False`).
* `reservoir`: toggles the use of a reservoir network.
* `show_plots`: If `True` some useful diagnostic plots will be displayed.
* `cuda`: which GPU to utilize for (CUDA‑enabled) training. If CUDA is not available, the code will automatically fall back to the CPU.
* `mode`: selects between no network (`None`) (only Bayes) and 2 training (if `training=True`) modes RL (`"RL"`) and supervised (`"SANITY"`). In RL mode, a classifier and generator are trained jointly; in sanity mode, only a classifer is trained.

## Directory structure

```text
CognitiveGridworld/
├── README.md                 
├── packages.txt              ← Conda environment specification
├── example.py                ← Example entrypoint
└── main/
    ├── CognitiveGridworld.py ← top‑level class containing default hyperparameters
    ├── env/                  ← Preprocessing / generating trials / running Bayesian and network simulations
    ├── env_plotting/         ← plotting helpers
    ├── model/                ← neural networks
    ├── REPLICATE_RESULTS/    ← Entrypoints for training networks / analyzing data / generating plots
    └── utils.py              ← utility functions
```

## Customization
* ...`_Customization.py` files are located in env/ & model/.
* ...`_Customization.py` files are checked before default functions are run.
* ...`_Customization.py` files are designed for modification to environments, Bayesian observers and neural networks.

## Overview

The repository is organised into several modules. The `main` folder is divided into:

- **`env`** – For generating an environment and its corresponding Bayesian agents.  These files construct the joint likelihood, samples contexts and generates observations.
- **`env_plotting`** – Some helpful visualizations for sanity checking and monitoring performance.
- **`model`** – Network architectures.  The `Model_architecture` class includes a classifier, generator & controller. 
- **`REPLICATE_RESULTS`** – Scripts for replicating experiments from the associated research paper, including `train_...py` scripts for training and scripts for collecting and visualizing data.
- **`utils.py`** – A small utility file with helper functions used throughout the project.

When you instantiate a `CognitiveGridworld.py` module:
-  The environment is preprocessed
-  The embedding space are generated
-  Episodes (contexts, realizations and observations) are generated
-  Bayesian & Network agents are run with diagnostic plots every 'checkpoint_every' episodes.
