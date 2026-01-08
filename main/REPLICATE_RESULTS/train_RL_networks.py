import numpy as np; import torch; import os; import sys; import inspect
path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld 

if __name__ == "__main__":
    cuda = 0
    realization_num = 10
    batch_num = 8000
    step_num = 30
    hid_dim = 1000
    state_num = 500
    obs_num = 5
    self = CognitiveGridworld(**{'mode': "RL", 'cuda': cuda, 'episodes': 125000,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'training': True,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'save_env': f'RL',
        'classifier_LR': .0005, 'ctx_num': 2, 'generator_LR':.0005, 'learn_embeddings': True})

    # self = CognitiveGridworld(**{'mode': "RL", 'cuda': cuda, 'episodes': 200000, 'checkpoint_every': 5000,
    #     'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'training': True,
    #     'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'save_env': f'RL_ctx_2',
    #     'classifier_LR': .0005, 'ctx_num': 2, 'generator_LR':.0005, 'learn_embeddings': True})

    # FOR LONG SLOW TRAINING TO HIGHER ACCURACY
    # self = CognitiveGridworld(**{'mode': "RL", 'cuda': cuda, 'episodes': 200000,
    #     'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num, 'checkpoint_every': 5000,
    #     'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'save_env': "RL_exp", 'training': True,
    # 'classifier_LR': .0001, 'ctx_num': 2, 'generator_LR':.0001, 'classifier_ent_bonus': .1, 'learn_embeddings': True})
