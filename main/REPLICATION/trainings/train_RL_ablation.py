import os
import sys
import inspect
path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
def find_project_root(start=None):
    d = os.path.abspath(start or path)
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

COMMON = { 'ctx_num': 2, 'obs_num': 5, 'step_num': 30, 'hid_dim': 1000, 'state_num': 500, 
          'realization_num': 10, 'learn_embeddings': True, 'training': True, 'show_plots': False,
           'episodes': 100000, 'batch_num': 5000, 'classifier_LR': .0005, 'generator_LR': .0005, 'classifier_ent_bonus': .01}


if __name__ == "__main__":
    cuda = 1
    for r in range(5):
        CognitiveGridworld(**COMMON, mode="RL", embedding_grad="classifier", embedding_reg=False,
                           cuda=cuda, save_env=f'RL_ablation_reps/RL_ablation_rep{r}')
        CognitiveGridworld(**COMMON, mode="RL", cuda=cuda, save_env=f'RL_ablation_reps/RL_exp_rep{r}')
