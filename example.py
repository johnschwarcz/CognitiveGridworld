from main import CognitiveGridworld
# %matplotlib auto
# %matplotlib inline

if __name__ == "__main__":
    mode = None  # [None, "SANITY", "RL"]  
    cuda = 0

    self = CognitiveGridworld(**{
        'episodes': 1000,
        'state_num': 500, 
        'batch_num': 8000, 
        'step_num': 30, 
        'obs_num': 5, 
        'ctx_num': 2, 
        'KQ_dim': 30, 
        'realization_num': 10,
        'likelihood_temp': 2,
        'checkpoint_every': 500,    # test at every "checkpoint_every" episodes
        'showtime': .1,             # show print_time decorated function runtime if runtime > showtime min
        'show_plots': True,
        'plot_every': 5,            # plot at every "plot_every" checkpoints

        'mode': mode,
        'hid_dim': 1000,
        'classifier_LR': .0005, 
        'controller_LR': .005, 
        'generator_LR': .001,
        'learn_embeddings': True,   # if True, embedding space must be learned
        'reservoir': False,
        'training': False,
        'save_env': None,
        'load_env': None,
        'cuda': cuda})

