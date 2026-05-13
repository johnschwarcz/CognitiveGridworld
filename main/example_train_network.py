import os; import sys; import inspect
path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld 

if __name__ == "__main__":
    cuda = 1
    batch_num = 500
    hid_dim = 2000

    self = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 150000, 'plot_every': 10,
        'realization_num': 10,  'hid_dim': hid_dim,  'obs_num': 5, 'training': True,
        'batch_num': batch_num, 'step_num': 30, 'state_num': 500, 'learn_embeddings': False,
        'reservoir': True, 'classifier_LR': .0005, 'ctx_num': 2, 'save_env': "/sanity/reservoir_ctx_2_2k"})
    
    del(self)
    