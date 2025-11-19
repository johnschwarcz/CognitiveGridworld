# CognitiveGridworld

**CognitiveGridworld** is a Python testbed for studying compositional generalization in latent space.  It provides a configurable grid‑world environment which is not directly observed, but instead generates stochastic observations. 
The code can be used to study how Semantic Interaction Information depends on the environment and to evaluate algorithms designed to generalize from goal-directed experience.

## Overview

The repository is organised into several modules.  The root contains a `packages.txt` file for reproducing the exact Conda environment. The `main` folder is divided into:

- **`env`** – For generating an environment and its corresponding Bayesian agents.  These files construct the joint likelihood, samples contexts and generates observations.
- **`env_plotting`** – Some helpful visualizations for sanity checking and monitoring performance.
- **`model`** – Neural network architectures.  The `Model_architecture` class includes a classifier, generator and controller. 
- **`REPLICATE_RESULTS`** – Scripts for replicating experiments from the associated research paper. These include `train_...py` scripts for training, and scripts for collecting and visualizing data.
- **`utils.py`** – A small utility file with helper functions used throughout the project.

The top‑level module `CognitiveGridworld.py` ties everything together.  When you instantiate `CognitiveGridworld`:
- **(1)** The environment is preprocessed
- **(2)** The embedding space are generated
- **(3.1)** Episodes (contexts, realizations and observations) are generated
- **(3.2)** Bayesian and Neural Network agents can be run. 
- **(3.3)** When plotting is enabled, a set of diagnostic plots is automatically produced every 'checkpoint_every' episodes.

## Installation

The project is built for Python 3.8+ and relies on standard scientific computing libraries such as NumPy, PyTorch, Matplotlib and tqdm. 

**Conda environment** – the repository includes a `packages.txt` file that captures the exact versions of every package used in the original experiments.  You can reproduce the environment with:

   ```sh
   conda create --name coggridworld --file packages.txt
   conda activate coggridworld


The code utilizes a CUDA‑enabled GPU for training neural networks.  If CUDA is not available, the code will automatically fall back to the CPU.

## Quick start

The easiest way to get started is to run the environment from Python.The following example creates a small environment with default settings:

```python
from main.CognitiveGridworld import CognitiveGridworld

# initialise the environment and run a short simulation
cg = CognitiveGridworld(episodes=10,
                        state_num=50,
                        obs_num=3,
                        ctx_num=2,
                        show_plots=True)
```

Upon instantiation the simulator will preprocess the environment, generate state embeddings and loop through episodes. If `show_plots=True` the likelihood surface, Bayesian performance and a sample trajectory will be displayed.
All tune-able hyperparameters can be found in main/CognitiveGridworld.py. A standard example environment is located in main/__init__.py. 

### Environment customisation

When initializing `CognitiveGridworld`, you can pass a dictionary of keyword arguments to customise the simulation.  Some notable options include:

* `episodes`: total number of learning episodes for networks.
* `state_num`: size of the State Space.
* `obs_num`: dimensionality of the Observation Space.
* `ctx_num`: number of active states per contexts.
* `realization_num`: number of potential realizations of an active state.
* `step_num`: number of inference steps per episode.
* `hid_dim`: size of the hidden layer used in neural models.
* `learn_embeddings`: whether to learn internal embeddings (`True`) or provide the true embeddings (`False`).
* `reservoir`: toggles the use of a reservoir network.
* `mode`: selects between using only Bayesian observers / no network ('None') and 2 training modes RL (`"RL"`) and supervised (`"SANITY"`). In RL mode, a classifier and generator are trained jointly; in sanity mode, only a classifer is trained.

Most scripts set these options explicitly; you can edit the script files to try different configurations.

## Directory structure

```text
CognitiveGridworld/
├── README.md                 
├── packages.txt              ← Conda environment specification
└── main/
    ├── CognitiveGridworld.py ← top‑level class orchestrating environment and model
    ├── env/                  ← environment definitions, generators and preprocessing
    ├── env_plotting/         ← plotting helpers and flow field visualisations
    ├── model/                ← neural architectures and controllers
    ├── REPLICATE_RESULTS/    ← scripts to reproduce experiments and generate plots
    └── utils.py              ← utility functions
```

