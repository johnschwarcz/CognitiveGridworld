import os
import sys
import inspect
path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
project_root = os.path.abspath(os.path.join(path, '..', '..'))
sys.path.insert(0, project_root)

from main.CognitiveGridworld import CognitiveGridworld

COMMON = { 'ctx_num': 2, 'obs_num': 5, 'step_num': 30, 'hid_dim': 1000, 'state_num': 500, 
          'realization_num': 10, 'learn_embeddings': True, 'training': True, 'show_plots': False,
           'episodes': 40000, 'batch_num': 20000, 'classifier_LR': .0005, 'generator_LR': .0005, 'classifier_ent_bonus': .01}


if __name__ == "__main__":
    cuda = 1
    for r in range(5):
        CognitiveGridworld(**COMMON, mode="ablation", cuda=cuda, save_env=f'RL_ablation_reps/RL_ablation_rep{r}')
        CognitiveGridworld(**COMMON, mode="RL", cuda=cuda, save_env=f'RL_ablation_reps/RL_exp_rep{r}')
