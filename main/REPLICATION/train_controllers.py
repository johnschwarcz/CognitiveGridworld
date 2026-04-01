import numpy as np
import torch
import os
import sys
import inspect
import pickle 

path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld
 
if __name__ == "__main__":
    cuda = 1
    reps = 20
    obs_num = 5
    step_num = 30
    hid_dim = 1000
    state_num = 500
    batch_num = 20000
    realization_num = 10
    controller_LR = .005
    generator_LR = .001
    ent =  .05
    eps = 2000
    folder = "main//DATA//controller"


    joint = CognitiveGridworld(**{'mode': "RL", 'cuda': cuda, 'load_env': 'RL', 'show_plots': False,
        'control_ent_bonus': ent, 'episodes': 1, 'ctx_num': 2,  'realization_num': realization_num, 'batch_num': batch_num, 'training': False,
        'hid_dim': hid_dim,  'obs_num': obs_num, 'state_num': state_num, 'step_num': step_num,
        'controller_LR': controller_LR, 'learn_embeddings': True})

    joint.train_controller(eps = eps, reps = reps, offline_teacher = 'joint')
    with open(os.path.join(folder, "joint.pkl"), 'wb') as f:
        pickle.dump(joint.controller_training_logs, f)

    offline_net = CognitiveGridworld(**{'mode': "RL", 'cuda': cuda, 'load_env': 'RL', 'show_plots': False,
        'control_ent_bonus': ent, 'episodes': 1, 'ctx_num': 2,  'realization_num': realization_num, 'batch_num': batch_num, 'training': False,
        'hid_dim': hid_dim,  'obs_num': obs_num, 'state_num': state_num, 'step_num': step_num,
        'controller_LR': controller_LR, 'learn_embeddings': True}) 

    offline_net.train_controller(eps = eps, reps = reps, offline_teacher = 'generator')
    with open(os.path.join(folder, "offline_net.pkl"), 'wb') as f:
        pickle.dump(offline_net.controller_training_logs, f)
    
    online_net = CognitiveGridworld(**{'mode': "RL", 'cuda': cuda, 'show_plots': False, 'control_ent_bonus': ent,
        'episodes': 2, 'ctx_num': 2,  'realization_num': realization_num, 'batch_num': batch_num, 'training': True,
        'hid_dim': hid_dim,  'obs_num': obs_num, 'state_num': state_num, 'step_num': step_num,
        'controller_LR': controller_LR, 'generator_LR' : generator_LR, 'learn_embeddings': True})
  
    online_net.train_controller(eps = eps, reps = reps)
    with open(os.path.join(folder, "online_net.pkl"), 'wb') as f:
        pickle.dump(online_net.controller_training_logs, f)
