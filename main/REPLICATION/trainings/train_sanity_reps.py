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
    batch_num = 2000 # 8000
    step_num = 10 # 30
    episodes = 50000
    state_num = 500
    obs_num = 5

    for r in range(5):
        self = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes,
            'realization_num': realization_num,  'hid_dim': 10000,  'obs_num': obs_num, 'training': True,
            'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
            'reservoir': True, 'classifier_LR': .001, 
            'ctx_num': 2, 'save_env': f"/sanity_reps/reservoir_matched_ctx_2_rep{r}"})

        self = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': episodes,
            'realization_num': realization_num,  'hid_dim': 200,  'obs_num': obs_num, 'training': True,
            'batch_num': batch_num, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': False,
            'reservoir': False, 'classifier_LR': .001, 'early_stopping': True,
            'ctx_num': 2, 'save_env': f"/sanity_reps/fully_trained_ctx_2_rep{r}"})