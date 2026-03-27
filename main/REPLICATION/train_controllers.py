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
    cuda = 0

    reps = 20 # 10
    obs_num = 5
    step_num = 30
    hid_dim = 1000
    state_num = 500
    batch_num = 20000
    realization_num = 10
    controller_LR = .001
    generator_LR = .0001
    eps = 10000


    online_net = CognitiveGridworld(**{'mode': "RL", 'cuda': cuda, 'show_plots': False,
        'episodes': 2, 'ctx_num': 2,  'realization_num': realization_num, 'batch_num': batch_num, 'training': True,
        'hid_dim': hid_dim,  'obs_num': obs_num, 'state_num': state_num, 'step_num': step_num,
        'controller_LR': controller_LR, 'generator_LR' : generator_LR, 'learn_embeddings': True})
  
    online_net.train_controller(eps = eps, reps = reps)
    with open(os.path.join("main//DATA//controller", "online_net.pkl"), 'wb') as f:
        pickle.dump(online_net.controller_training_logs, f)

    joint, offline_net = [ CognitiveGridworld(**{'mode': "RL", 'cuda': cuda, 'load_env': 'RL', 'show_plots': False,
        'episodes': 2, 'ctx_num': 2,  'realization_num': realization_num, 'batch_num': batch_num, 'training': False,
        'hid_dim': hid_dim,  'obs_num': obs_num, 'state_num': state_num, 'step_num': step_num,
        'controller_LR': controller_LR, 'learn_embeddings': True}) for _ in range(2)]

    offline_net.train_controller(eps = eps, reps = reps, offline_teacher = 'generator')
    with open(os.path.join("main//DATA//controller", "offline_net.pkl"), 'wb') as f:
        pickle.dump(offline_net.controller_training_logs, f)
          
    joint.train_controller(eps = eps, reps = reps, offline_teacher = 'joint')
    with open(os.path.join("main//DATA//controller", "joint.pkl"), 'wb') as f:
        pickle.dump(joint.controller_training_logs, f)