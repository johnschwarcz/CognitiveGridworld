# CognitiveGridworld

**CognitiveGridworld** is a Python testbed for studying compositional generalization in latent space.  It provides a configurable grid‑world environment which is not directly observed, but instead generates stochastic observations. 
The code can be used to study how Semantic Interaction Information depends on the environment and to evaluate algorithms designed to generalize from goal-directed experience.

## Overview

The repository is organised into several modules.  The root contains a `packages.txt` file for reproducing the exact Conda environment. The `main` folder is divided into:

- **`env`** – For generating an environment and its corresponding Bayesian agents.  These files construct the joint likelihood, samples contexts and generates observations.
- **`env_plotting`** – Some helpful visualizations for sanity checking and monitoring performance.
- **`model`** – Network architectures.  The `Model_architecture` class includes a classifier, generator & controller. 
- **`REPLICATE_RESULTS`** – Scripts for replicating experiments from the associated research paper, including `train_...py` scripts for training and scripts for collecting and visualizing data.
- **`utils.py`** – A small utility file with helper functions used throughout the project.

The top‑level module `CognitiveGridworld.py` ties everything together.  When you instantiate `CognitiveGridworld`:
-  The environment is preprocessed
-  The embedding space are generated
-  Episodes (contexts, realizations and observations) are generated
-  Bayesian & Network agents are run with diagnostic plots every 'checkpoint_every' episodes.

## Installation

The project is built for Python 3.8+ and relies on standard scientific computing libraries such as NumPy, PyTorch, Matplotlib and tqdm. 

**Conda environment** – the repository includes a `packages.txt` file that captures the exact versions of every package used in the original experiments.  You can reproduce the environment with:

```sh
   conda create --name coggridworld --file packages.txt
   conda activate coggridworld
```
The code utilizes a CUDA‑enabled GPU for training neural networks.  If CUDA is not available, the code will automatically fall back to the CPU.

## Quick start

The easiest way to get started is to run the environment from Python.The following example creates a small environment with default settings:

```sh
from main.CognitiveGridworld import CognitiveGridworld

# initialise the environment and run a short simulation
cg = CognitiveGridworld(episodes=10,
                        state_num=50,
                        obs_num=3,
                        ctx_num=2,
                        show_plots=True)
```

Upon instantiation the simulator will preprocess the environment, generate state embeddings and loop through episodes. If `show_plots=True` the likelihood, Bayesian performance and a sample trajectory will be displayed.

A standard example environment is located in example.py. 

### Environment

When initializing `CognitiveGridworld`, you can pass a dictionary of keyword arguments to customise the simulation.  Notable options include:

* `episodes`: total number of learning episodes for networks.
* `state_num`: size of the State Space.
* `obs_num`: dimensionality of the Observation Space.
* `ctx_num`: number of active states per contexts.
* `realization_num`: number of potential realizations of an active state.
* `step_num`: number of inference steps per episode.
* `hid_dim`: size of the hidden layer used in neural models.
* `learn_embeddings`: whether to learn the embeddings (`True`) or provide the true embeddings (`False`).
* `reservoir`: toggles the use of a reservoir network.
* `mode`: selects between no network (`None`) (only Bayes) and 2 training (if `training=True`) modes RL (`"RL"`) and supervised (`"SANITY"`). In RL mode, a classifier and generator are trained jointly; in sanity mode, only a classifer is trained.

The default values of all tune-able hyperparameters can be found in `main/CognitiveGridworld.py`.
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
Customization is built into the foundation of the CognitiveGridworld. 
* ...`_Customization.py` files are located in env/ & model/.
* ...`_Customization.py` files are checked before default functions are run.
* ...`_Customization.py` files are designed for modification to environments, Bayesian observers and neural networks.

