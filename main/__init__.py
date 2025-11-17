# %matplotlib auto
# %matplotlib inline
import numpy as np; import torch; import os; import sys; import inspect
path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/Main')
sys.path.insert(0, path + '/Main/model')
from main.CognitiveGridworld import CognitiveGridworld

if __name__ == "__main__":
    mode = None  # [None, "SANITY", "RL"]  
    cuda = 0

    self = CognitiveGridworld(**{
        'episodes': 1,
        'state_num': 500, 
        'batch_num': 5000, 
        'step_num': 30, 
        'obs_num': 5, 
        'ctx_num': 2, 
        'KQ_dim': 30, 
        'realization_num': 10,
        'likelihood_temp': 2,
        'checkpoint_every': 500,    # start tests at every "checkpoint_every" episodes
        'showtime': .1,             # show print_time decorated function runtime if runtime > showtime min
        'show_plots': True,

        'mode': mode,
        'hid_dim': 1000,
        'classifier_LR': .0005, 
        'controller_LR': .005, 
        'generator_LR': .001,
        'learn_embeddings': True,   # if True, embedding space must be learned
        'reservoir': False,
        'training': True,
        'save_env': None,
        'load_env': None,
        'cuda': cuda})

