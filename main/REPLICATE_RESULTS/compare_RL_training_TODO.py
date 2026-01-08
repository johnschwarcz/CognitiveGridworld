import numpy as np; import torch; import os; import sys; import inspect; import pylab as plt
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
    step_num = 30
    hid_dim = 1000
    obs_num = 5
    # state_nums = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]
    state_nums = [100, 125, 150, 175, 200, 250, 300, 400, 600, 1000]
    do = "test" # ["train", "test"]
    repetitions = 5

    if do == "train":
        for state_num in state_nums:
        
            # for rep in range(repetitions):
            R = 2 if state_num == 100 else 0
            for rep in range(R, repetitions):
                print(f"Training state num: {state_num}, repetition: {rep}")
                self = CognitiveGridworld(**{'mode': "RL", 'cuda': cuda, 'episodes': 10000, 'plot_every': 5,
                'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num, 'training': True,
                'batch_num': 8000, 'step_num': step_num, 'state_num': state_num, 'save_env': f'/RL_state_num_reps/{state_num}_{rep}',
                'classifier_LR': .0005, 'ctx_num': 2, 'generator_LR':.0005, 'learn_embeddings': True})

    if do == "test":
        trains, tests = [np.empty(len(state_nums), dtype = object) for _ in range(2)]
        for i, state_num in enumerate(state_nums):
            CG = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1,
                'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num, 'show_plots' : False,
                'batch_num': 50, 'step_num': step_num, 'state_num': state_num, 'learn_embeddings': True,
                'ctx_num': 2, 'load_env': f"RL_state_num_{state_num}"})
            trains[i] = CG.train_acc_through_training[-1]
            tests[i] = CG.test_acc_through_training[-1]

        fig, ax = plt.subplots(1, 1, figsize = (10, 4), tight_layout = True)
        plt.plot(state_nums, [t[-1] for t in trains], '-o')
        plt.plot(state_nums, [t[-1] for t in tests], '-o')
        plt.xticks(ticks  = state_nums, labels = state_nums)
        plt.show()