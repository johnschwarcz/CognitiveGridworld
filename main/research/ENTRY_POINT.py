import numpy as np; import torch; import os; import sys; import inspect; from matplotlib.colors import PowerNorm
from tqdm import tqdm; 
import matplotlib.pyplot as plt; 
from matplotlib.colors import Normalize; from matplotlib.cm import ScalarMappable
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
    step_num = 30
    hid_dim = 1000
    state_num = 500
    obs_num = 5
    batch_num = 8000
    episodes = 1

    env = "/sanity/reservoir_ctx_2_e5"
    echo = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes, 'checkpoint_every': 5, 
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False,
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': True, 'classifier_LR': .001, 'ctx_num': 2, 
        'training': False, 'load_env': env})

    env = "/sanity/fully_trained_ctx_2_e5"
    trained = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes, 'checkpoint_every': 5,
        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num,'show_plots': False, 
        'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
        'reservoir': False, 'classifier_LR': .001, 'ctx_num': 2, 
        'training': False, 'load_env': env})
