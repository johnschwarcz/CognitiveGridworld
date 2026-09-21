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
"""

import os
import sys
import inspect
import pickle
import numpy as np 
import multiprocessing as mp
from joblib import Parallel, delayed

# path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# project_root = os.path.abspath(os.path.join(path, '..', '..', '..'))
# sys.path.insert(0, project_root)

def find_project_root(start=None):
    d = os.path.abspath(start or os.getcwd())
    while True:
        if os.path.exists(os.path.join(d, 'main', 'CognitiveGridworld.py')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("project root not found")
        d = parent

project_root = find_project_root()
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

# def train_RL_state_num_reps():
#     # state_nums = [1000, 500, 250, 200, 150, 100, 50]
#     state_nums = [800, 400, 200, 100, 50]
#     # state_nums = np.flip(state_nums)
#     repetitions = 5
#     start_from = 0 

#     for r in range(start_from, repetitions):
#         for state_num in state_nums:
#             print(f"Training state_num={state_num}, rep={r}")
#             CognitiveGridworld(**{
#                 'mode': "RL", 'cuda': 1, 'training': True, 'show_plots': False, 
#                 'realization_num': 10, 'hid_dim': 1000, 'obs_num': 5, 'step_num': 30, 
               
#                 'episodes': 150000, 'batch_num': 10000, 'checkpoint_every': 5000, 
#                 'classifier_LR': .0001, 'generator_LR': .0001, 'classifier_ent_bonus': .1,

#                  # 'episodes': 50000, 'batch_num': 10000, 'classifier_LR': .0005, 'generator_LR': .0005,
#                #'episodes': 50000, 'batch_num': 20000, 'classifier_LR': .0001, 'generator_LR': .0001,
                
#                 'state_num': state_num, 'ctx_num': 2, 'learn_embeddings': True,
#                 'save_env': f'/RL_state_num_reps/{state_num}_{r}',
#             })



def _train_single_repetition_worker(task_args):
    r, state_nums, cuda, episodes, batch_num, checkpoint_every = task_args
    print(f"\n[Worker Rep {r}] Starting repetition {r} on GPU {cuda}...", flush=True)

    for state_num in state_nums:
        print(f"[Worker Rep {r}] Training state_num={state_num}, rep={r} on GPU {cuda}", flush=True)
        CognitiveGridworld(**{
            'mode': "RL",
            'cuda': cuda,
            'training': True,
            'show_plots': False,
            'realization_num': 10,
            'hid_dim': 1000,
            'obs_num': 5,
            'step_num': 30,
            'episodes': episodes,
            'batch_num': batch_num,
            'checkpoint_every': checkpoint_every,
            'classifier_LR': .0001,
            'generator_LR': .0001,
            'classifier_ent_bonus': .1,
            'state_num': state_num,
            'ctx_num': 2,
            'learn_embeddings': True,
            'save_env': f'/RL_state_num_reps/{state_num}_{r}',
        })

    print(f"[Worker Rep {r}] Completed all state_nums for repetition {r}.\n", flush=True)
    return r


def train_RL_state_num_reps():
    n_jobs = 3
    repetitions = 5
    start_from = 0
    cuda = 1
    episodes = 150000 # 50000
    batch_num = 5000 # 20000
    checkpoint_every = 5000
    state_nums = [800, 400, 200, 100, 50]

    reps = list(range(start_from, repetitions))
    task_payloads = [
        (r, state_nums, cuda, episodes, batch_num, checkpoint_every) for r in reps
    ]

    if n_jobs <= 1:
        for payload in task_payloads:
            _train_single_repetition_worker(payload)
    else:
        active_workers = min(n_jobs, len(reps))
        print(f"Running {len(reps)} repetition(s) across {active_workers} parallel worker(s) on GPU {cuda}...")
        
        # joblib's loky backend natively handles IPython/VS Code Interactive environments 
        # without crashing on spawn or dropping exceptions.
        Parallel(n_jobs=active_workers, backend="loky")(
            delayed(_train_single_repetition_worker)(payload) for payload in task_payloads
        )



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
# ─────────────────────────────────────────────────────────────────────────────

def train_controllers():
    controller_folder = os.path.join(project_root, "main", "DATA", "controller")
    os.makedirs(controller_folder, exist_ok=True)

    # base_kwargs = {
    #     'mode': "RL", 'cuda': 1, 'load_env': 'RL_2', 'show_plots': False,
    #     'control_ent_bonus': .05, 'episodes': 1, 'ctx_num': 2,
    #     'realization_num': 10, 'batch_num': 20000, 'training': False,
    #     'hid_dim': 1000, 'obs_num': 5, 'state_num': 500, 'step_num': 30,
    #     'controller_LR': .005, 'learn_embeddings': True
    # }
    
    cuda = 0
    ent = .05 
    batch_num = 20000 
    controller_LR = .005
    generator_LR = .001
    controller_eps = 500 # 2000
    reps = 20
      
    base_kwargs = {
        'mode': "RL", 'cuda': cuda, 'load_env': 'RL_2', 'show_plots': False,
        'control_ent_bonus': ent, 'episodes': 1, 'ctx_num': 2,
        'realization_num': 10, 'batch_num': batch_num, 'training': False,
        'hid_dim': 1000, 'obs_num': 5, 'state_num': 500, 'step_num': 30,
        'controller_LR': controller_LR, 'learn_embeddings': True
    }
     
    joint = CognitiveGridworld(**base_kwargs)
    joint.train_controller(eps=controller_eps, reps=reps, offline_teacher='joint')
    with open(os.path.join(controller_folder, "joint.pkl"), 'wb') as f:
        pickle.dump(joint.controller_training_logs, f)
    del(joint)

    offline_net = CognitiveGridworld(**base_kwargs)
    offline_net.train_controller(eps=controller_eps, reps=reps, offline_teacher='generator')
    with open(os.path.join(controller_folder, "offline_net.pkl"), 'wb') as f:
        pickle.dump(offline_net.controller_training_logs, f)
    del(offline_net)

    # online_net = CognitiveGridworld(**{**base_kwargs,
    #     'episodes': 1, 'training': True, 'generator_LR': generator_LR})
    online_net = CognitiveGridworld(**{'mode': "RL", 'cuda': cuda, 'show_plots': False,
        'control_ent_bonus': ent,  'episodes': 1, 'ctx_num': 2,  'realization_num': 10,
        'batch_num': batch_num, 'training': True, 'hid_dim': 1000,  'obs_num': 5,
        'state_num': 500, 'step_num': 30, 'controller_LR': controller_LR,
        'generator_LR' : generator_LR, 'learn_embeddings': True})

    online_net.train_controller(eps=controller_eps, reps=reps)
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
    print("Phase 2: RL_state_num_reps")
    print("=" * 60)
    train_RL_state_num_reps()

    # print("=" * 60)
    # print("Phase 3: _e5 checkpoint variants (2 models)")
    # print("=" * 60)
    # train_e5_variants()

    print("Done.")
