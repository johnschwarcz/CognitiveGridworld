"""
Script 1 — Retrain all models required for thesis figures.

Run from the project root:
    python3 main/REPLICATION/thesis/retrain_thesis_models.py

Covers (in order):
  1. Core sanity models    — ctx 1/2 (hid=1000) and ctx 3 (hid=5000)
                             used by: net_acc.py, plot_training.py, belief_compare.py,
                                      FR_vs_acc.py, MI_vs_acc.py
  2. Scaling reservoir      — reservoir_ctx_2 at 2k/5k/10k hidden dims
                             used by: plot_training.py (thesis_fig4 panel 2)
For params grid models, run train_models_multi.py directly (parallel execution).
"""

import os
import sys
import inspect

path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_root = os.path.abspath(os.path.join(path, '..', '..', '..'))
sys.path.insert(0, project_root)

from main.CognitiveGridworld import CognitiveGridworld


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Core sanity models
# Source: main/REPLICATION/train_sanity_networks.py (exact parameters)
# ─────────────────────────────────────────────────────────────────────────────

def train_sanity_models():
    cuda = 1
    realization_num = 10
    batch_num = 8000
    step_num = 30
    state_num = 500
    obs_num = 5

    # --- ctx 1 and ctx 2 models (hid_dim=1000) ---

    CognitiveGridworld(**{
        'mode': "SANITY", 'cuda': cuda, 'episodes': 50000,
        'realization_num': realization_num, 'hid_dim': 1000, 'obs_num': obs_num,
        'training': True, 'batch_num': batch_num, 'step_num': step_num,
        'state_num': state_num, 'learn_embeddings': False, 'show_plots': False,
        'reservoir': True, 'classifier_LR': .001,
        'ctx_num': 2, 'save_env': "/sanity/reservoir_ctx_2"
    })

    CognitiveGridworld(**{
        'mode': "SANITY", 'cuda': cuda, 'episodes': 25000,
        'realization_num': realization_num, 'hid_dim': 1000, 'obs_num': obs_num,
        'training': True, 'batch_num': batch_num, 'step_num': step_num,
        'state_num': state_num, 'learn_embeddings': False, 'show_plots': False,
        'reservoir': True, 'classifier_LR': .001,
        'ctx_num': 1, 'save_env': "/sanity/reservoir_ctx_1"
    })

    CognitiveGridworld(**{
        'mode': "SANITY", 'cuda': cuda, 'episodes': 50000,
        'realization_num': realization_num, 'hid_dim': 1000, 'obs_num': obs_num,
        'training': True, 'batch_num': batch_num, 'step_num': step_num,
        'state_num': state_num, 'learn_embeddings': False, 'show_plots': False,
        'reservoir': False, 'classifier_LR': .001,
        'ctx_num': 2, 'save_env': "/sanity/fully_trained_ctx_2"
    })

    CognitiveGridworld(**{
        'mode': "SANITY", 'cuda': cuda, 'episodes': 25000,
        'realization_num': realization_num, 'hid_dim': 1000, 'obs_num': obs_num,
        'training': True, 'batch_num': batch_num, 'step_num': step_num,
        'state_num': state_num, 'learn_embeddings': False, 'show_plots': False,
        'reservoir': False, 'classifier_LR': .001,
        'ctx_num': 1, 'save_env': "/sanity/fully_trained_ctx_1"
    })

    # --- ctx 3 models (hid_dim=5000, more episodes, smaller batch) ---

    CognitiveGridworld(**{
        'mode': "SANITY", 'cuda': cuda, 'episodes': 150000, 'plot_every': 10,
        'realization_num': realization_num, 'hid_dim': 5000, 'obs_num': obs_num,
        'training': True, 'batch_num': 3000, 'step_num': step_num,
        'state_num': state_num, 'learn_embeddings': False, 'show_plots': False,
        'reservoir': True, 'classifier_LR': .0005,
        'ctx_num': 3, 'save_env': "/sanity/reservoir_ctx_3"
    })

    CognitiveGridworld(**{
        'mode': "SANITY", 'cuda': cuda, 'episodes': 150000, 'plot_every': 10,
        'realization_num': realization_num, 'hid_dim': 5000, 'obs_num': obs_num,
        'training': True, 'batch_num': 3000, 'step_num': step_num,
        'state_num': state_num, 'learn_embeddings': False, 'show_plots': False,
        'reservoir': False, 'classifier_LR': .0005,
        'ctx_num': 3, 'save_env': "/sanity/fully_trained_ctx_3"
    })


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Scaling reservoir variants
# Source: plot_training.py loads reservoir_ctx_2_{k}k with hid_dim=1000*k
# Same training config as reservoir_ctx_2 but with scaled hidden dimension.
# ─────────────────────────────────────────────────────────────────────────────

def train_scaling_variants():
    cuda = 1
    realization_num = 10
    obs_num = 5
    state_num = 500
    batch_num = 2000 # 8000
    step_num = 10 # 30

    for k in [10, 5, 2, 1]:
        CognitiveGridworld(**{
            'mode': "SANITY", 'cuda': cuda, 'episodes': 50000,
            'realization_num': realization_num, 'hid_dim': 1000 * k, 'obs_num': obs_num,
            'training': True, 'batch_num': batch_num, 'step_num': step_num,
            'state_num': state_num, 'learn_embeddings': False, 'show_plots': False,
            'reservoir': True, 'classifier_LR': .001, 'skip_training_analyses': True,
            'ctx_num': 2, 'save_env': f"/sanity/reservoir_ctx_2_{k}k_{step_num}step"
        })

    CognitiveGridworld(**{
        'mode': "SANITY", 'cuda': cuda, 'episodes': 50000,
        'realization_num': realization_num, 'hid_dim': 1000, 'obs_num': obs_num,
        'training': True, 'batch_num': batch_num, 'step_num': step_num,
        'state_num': state_num, 'learn_embeddings': False, 'show_plots': False,
        'reservoir': False, 'classifier_LR': .001, 'skip_training_analyses': True,
        'ctx_num': 2, 'save_env': f"/sanity/fully_trained_ctx_2_1k_{step_num}step"
    })


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1: Core sanity models")
    print("=" * 60)
    # train_sanity_models()

    print("=" * 60)
    print("Phase 2: Scaling reservoir variants")
    print("=" * 60)
    train_scaling_variants()
