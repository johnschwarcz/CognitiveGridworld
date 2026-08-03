import os
import sys
import inspect
import itertools
from concurrent.futures import ProcessPoolExecutor

# Setup paths
path = inspect.getfile(inspect.currentframe())
path = os.path.dirname(os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')

from main.CognitiveGridworld import CognitiveGridworld 

def run_single_experiment(params):
    """
    Worker function executed by individual processes.
    Unpacks parameters and runs a single instance of CognitiveGridworld.
    """
    likelihood_temp, likelihood_freq, cuda, batch_num, hid_dim, step_num = params
    
    print(f"Starting: likelihood_temp={likelihood_temp}, likelihood_freq={likelihood_freq}")
    
    # Instantiate and train the model
    model = CognitiveGridworld(**{
        'mode': "SANITY", 
        'episodes': 500000, 
        'show_plots': False,
        'realization_num': 10,  
        'hid_dim': hid_dim,  
        'obs_num': 5, 
        'training': True, 
        'skip_training_analyses': True,
        'batch_num': batch_num, 
        'step_num': step_num, 
        'state_num': 500, 
        'learn_embeddings': False,
        'classifier_LR': .001, 
        'ctx_num': 2,
        'likelihood_temp': likelihood_temp, 
        'likelihood_freq': likelihood_freq,
        'early_stopping': True,
        'cuda': cuda, 
        'show_plots': False,
        # 'reservoir': True, 
        # 'save_env': f"/params/reservoir_LT{likelihood_temp}_LF{likelihood_freq}"

        'reservoir': False, 
        'save_env': f"/params/fully_trained_LT{likelihood_temp}_LF{likelihood_freq}"
    })
    
    print(f"Finished: likelihood_temp={likelihood_temp}, likelihood_freq={likelihood_freq}")
    return f"Success: LT_{likelihood_temp}_LF_{likelihood_freq}"

if __name__ == "__main__":
    # Fixed hyperparameters
    cuda = 0
    batch_num = 5000
    hid_dim = 1000 # 2000
    step_num = 30
    max_workers = 2

    # Search space
    likelihood_temps = [3, 2, 1]
    likelihood_freqs = [2, 1.5, 1] # [2, 1, .5]
    
    parameter_grid = [
        (lt, lf, cuda, batch_num, hid_dim, step_num) 
        for lt, lf in itertools.product(likelihood_temps, likelihood_freqs)
    ]
    
    # Set max_workers to control concurrent processes (e.g., 2 or 3 depending on VRAM)
    print(f"Launching parameter search with {max_workers} concurrent processes...")
    
    # Execute the grid in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(run_single_experiment, parameter_grid))
        
    print("\nAll experiments complete!")
    for result in results:
        print(result)