"""
Script 2 — Retrain models with clearly identifiable configurations.

Run from the project root:
    python3 main/REPLICATION/thesis/retrain_other_models.py

Covers:
  1. RL_2 model             — trained agent for the RL environment
                             source: main/REPLICATION/train_RL_networks.py
  2. RL_state_num_reps      — RL sensitivity sweep over state-space size
                             source: main/REPLICATION/compare_RL_training.py
  3. _e5 checkpoint models  — ctx_2 sanity models with checkpoint_every=5
                             source: main/REPLICATION/supplementary_plotter.py
  4. Controller pkl files   — joint / offline_net / online_net controllers
                             source: main/REPLICATION/train_controllers.py
                             NOTE: controllers load 'RL_2' (retrained here);
                             if train_controllers.py still uses load_env='RL',
                             update that path to 'RL_2' before running it.
"""

import os
import sys
import inspect
import pickle

path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_root = os.path.abspath(os.path.join(path, '..', '..', '..'))
sys.path.insert(0, project_root)

from main.CognitiveGridworld import CognitiveGridworld


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: RL_2 model
# Source: main/REPLICATION/train_RL_networks.py (exact parameters)
# ─────────────────────────────────────────────────────────────────────────────

def train_RL_2():
    CognitiveGridworld(**{
        'mode': "RL", 'cuda': 1, 'episodes': 50000,
        'realization_num': 10, 'hid_dim': 1000, 'obs_num': 5,
        'training': True, 'batch_num': 20000, 'step_num': 30, 'state_num': 500,
        'save_env': 'RL_2', 'classifier_LR': .0001, 'ctx_num': 2, 'show_plots': False,
        'generator_LR': .0001, 'classifier_ent_bonus': .01, 'learn_embeddings': True
    })


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: RL_state_num_reps sensitivity sweep
# Source: main/REPLICATION/compare_RL_training.py (exact parameters, do="train")
# Saves to: DATA/RL_state_num_reps/{state_num}_0_net.pth  (repetition 0)
# ─────────────────────────────────────────────────────────────────────────────

def train_RL_state_num_reps():
    state_nums = [100, 125, 150, 200, 250, 300, 400, 500, 1000]
    repetitions = 1   # compare_RL_training.py sets repetitions=1

    for r in range(repetitions):
        for state_num in state_nums:
            print(f"Training state_num={state_num}, rep={r}")
            CognitiveGridworld(**{
                'mode': "RL", 'cuda': 1, 'episodes': 50000, 
                'checkpoint_every': 5000, 'show_plots': False,
                'realization_num': 10, 'hid_dim': 1000, 'obs_num': 5,
                # 'training': True, 'batch_num': 20000, 'step_num': 30,
                'training': True, 'batch_num': 10000, 'step_num': 30,
                'state_num': state_num, 
                'save_env': f'/RL_state_num_reps/{state_num}_{r}',
                'classifier_LR': .0001, 'ctx_num': 2,
                'generator_LR': .0001, 'learn_embeddings': True
            })


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: _e5 checkpoint models
# Source: main/REPLICATION/supplementary_plotter.py lines 312–335
# Same architecture as the core ctx_2 models but saved with checkpoint_every=5.
# ─────────────────────────────────────────────────────────────────────────────

def train_e5_variants():
    cuda = 0
    realization_num = 10
    hid_dim = 1000
    obs_num = 5
    state_num = 500
    batch_num = 8000
    step_num = 30

    CognitiveGridworld(**{
        'mode': "SANITY", 'cuda': cuda, 'episodes': 50000, 'checkpoint_every': 5,
        'realization_num': realization_num, 'hid_dim': hid_dim, 'obs_num': obs_num,
        'show_plots': False, 'training': True,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num,
        'learn_embeddings': False, 'reservoir': False, 'classifier_LR': .001,
        'ctx_num': 2, 'save_env': "/sanity/fully_trained_ctx_2_e5"
    })

    CognitiveGridworld(**{
        'mode': "SANITY", 'cuda': cuda, 'episodes': 50000, 'checkpoint_every': 5,
        'realization_num': realization_num, 'hid_dim': hid_dim, 'obs_num': obs_num,
        'show_plots': False, 'training': True,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num,
        'learn_embeddings': False, 'reservoir': True, 'classifier_LR': .001,
        'ctx_num': 2, 'save_env': "/sanity/reservoir_ctx_2_e5"
    })


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Controller pkl files
# Source: main/REPLICATION/train_controllers.py (exact parameters).
# Original script uses load_env='RL' (DATA/RL_net.pth); this version uses
# 'RL_2' which is retrained in Phase 1 above.  Update train_controllers.py
# accordingly if you run that script directly instead.
# ─────────────────────────────────────────────────────────────────────────────

def train_controllers():
    controller_folder = os.path.join(project_root, "main", "DATA", "controller")
    os.makedirs(controller_folder, exist_ok=True)

    base_kwargs = {
        'mode': "RL", 'cuda': 1, 'load_env': 'RL_2', 'show_plots': False,
        'control_ent_bonus': .05, 'episodes': 1, 'ctx_num': 2,
        'realization_num': 10, 'batch_num': 20000, 'training': False,
        'hid_dim': 1000, 'obs_num': 5, 'state_num': 500, 'step_num': 30,
        'controller_LR': .005, 'learn_embeddings': True
    }
    # base_kwargs = {
    #     'mode': "RL", 'cuda': 1, 'load_env': 'RL_2', 'show_plots': False,
    #     'control_ent_bonus': .05, 'episodes': 1, 'ctx_num': 2,
    #     'realization_num': 10, 'batch_num': 100, 'training': False,
    #     'hid_dim': 1000, 'obs_num': 5, 'state_num': 500, 'step_num': 30,
    #     'controller_LR': .005, 'learn_embeddings': True
    # }
    
    
    eps = 2000
    reps = 20
      
 
    joint = CognitiveGridworld(**base_kwargs)
    joint.train_controller(eps=eps, reps=reps, offline_teacher='joint')
    with open(os.path.join(controller_folder, "joint.pkl"), 'wb') as f:
        pickle.dump(joint.controller_training_logs, f)
    del(joint)

    offline_net = CognitiveGridworld(**base_kwargs)
    offline_net.train_controller(eps=eps, reps=reps, offline_teacher='generator')
    with open(os.path.join(controller_folder, "offline_net.pkl"), 'wb') as f:
        pickle.dump(offline_net.controller_training_logs, f)
    del(offline_net)

    online_net = CognitiveGridworld(**{**base_kwargs,
        'episodes': 2, 'training': True, 'generator_LR': .001})
    online_net.train_controller(eps=eps, reps=reps)
    with open(os.path.join(controller_folder, "online_net.pkl"), 'wb') as f:
        pickle.dump(online_net.controller_training_logs, f)
    

if __name__ == "__main__":
    # print("=" * 60)
    # print("Phase 1: RL_2 model")
    # print("=" * 60)
    # train_RL_2()

    # print("=" * 60)
    # print("Phase 4: Controller pkl files (joint / offline_net / online_net)")
    # print("=" * 60)
    # train_controllers()

    print("=" * 60)
    print("Phase 2: RL_state_num_reps (9 state sizes × 1 rep)")
    print("=" * 60)
    train_RL_state_num_reps()

    # print("=" * 60)
    # print("Phase 3: _e5 checkpoint variants (2 models)")
    # print("=" * 60)
    # train_e5_variants()

    print("Done.")
