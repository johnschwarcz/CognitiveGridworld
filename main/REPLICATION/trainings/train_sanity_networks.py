import numpy as np; import torch; import os; import sys; import inspect
path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld 

if __name__ == "__main__":
    cuda = 1
    realization_num = 10
    batch_num = 8000
    step_num = 30
    hid_dim = 1000
    state_num = 500
    obs_num = 5

    self = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 50000,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num, 'training': True,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': True, 'classifier_LR': .001, 'ctx_num': 2, 'save_env': "/sanity/reservoir_ctx_2"})

    self = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 25000,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'training': True,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': True, 'classifier_LR': .001, 'ctx_num': 1, 'save_env': "/sanity/reservoir_ctx_1"})

    self = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 50000,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num, 'training': True,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': False, 'classifier_LR': .001, 'ctx_num': 2, 'save_env': "/sanity/fully_trained_ctx_2"})
        
    self = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 25000,
        'realization_num': realization_num,  'hid_dim': hid_dim, 'obs_num': obs_num,'training': True,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': False, 'classifier_LR': .001, 'ctx_num': 1, 'save_env': "/sanity/fully_trained_ctx_1"})

    batch_num = 3000
    hid_dim = 5000

    self = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 150000, 'plot_every': 10,
        'realization_num': 10,  'hid_dim': hid_dim,  'obs_num': 5, 'training': True,
        'batch_num': batch_num, 'step_num': 30, 'state_num': 500, 'learn_embeddings': False,
        'reservoir': True, 'classifier_LR': .0005, 'ctx_num': 3, 'save_env': "/sanity/reservoir_ctx_3"})

    self = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 150000, 'plot_every': 10,
        'realization_num': 10,  'hid_dim': hid_dim,  'obs_num': 5, 'training': True,
        'batch_num': batch_num, 'step_num': 30, 'state_num': 500, 'learn_embeddings': False,
        'reservoir': False, 'classifier_LR': .0005, 'ctx_num': 3, 'save_env': "/sanity/fully_trained_ctx_3"})


