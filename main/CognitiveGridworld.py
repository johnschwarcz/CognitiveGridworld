import numpy as np; import torch; import os; import sys; import inspect
path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/env')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
sys.path.insert(0, path + '/main/plotting')
from main.env.env_model_manager import Env_model_manager

class CognitiveGridworld(Env_model_manager):
    def __init__(self, **init):
        self.__dict__.update(init)
        gpu = "cuda:" + str(init.get('cuda', 0))
        gpu_available = torch.cuda.is_available()
        self.device_type = gpu if gpu_available else "cpu"
        self.device = torch.device(self.device_type)
        project_root = os.path.dirname(os.path.abspath(__file__))
        self.DATA_path = os.path.join(project_root, 'DATA/')
    
        self.episodes = init.get('episodes', 1)
        self.showtime = init.get('showtime', .2)
        self.state_num = init.get('state_num', 5000)
        self.show_plots = init.get('show_plots', True)
        self.load_warnings = init.get('load_warnings', False)
        self.checkpoint_every = init.get('checkpoint_every', 500)
        self.plot_every = init.get('plot_every', 5)
        self.test_states = init.get('test_states', self.state_num // 10)

        self.likelihood_temp = init.get('likelihood_temp', 2)
        self.realization_num = init.get('realization_num', 10)
        self.batch_num = init.get('batch_num', 1000)
        self.step_num = init.get('step_num', 100)
        self.ctx_num = init.get('ctx_num', 2)
        self.obs_num = init.get('obs_num', 5)
        self.KQ_dim = init.get('KQ_dim', 30)

        self.control_ent_bonus_decay = init.get('control_ent_bonus_decay', 0.999)
        self.classifier_ent_bonus = init.get('classifier_ent_bonus', 0.1)
        self.classifier_LR = init.get('classifier_LR', 0.0005)
        self.controller_LR = init.get('controller_LR', 0.001)
        self.generator_LR = init.get('generator_LR', 0.001)
        self.hid_dim = init.get('hid_dim', 1000)

        self.learn_embeddings = init.get('learn_embeddings', True)
        self.reservoir = init.get('reservoir', False)
        self.training = init.get('training', False)
        self.load_env = init.get('load_env', None)
        self.save_env = init.get('save_env', None)

        self.trigger_simulation()

    def trigger_simulation(self):
        self.preprocess_env()
        self.EC_gen_state_embeddings()
        if self.mode is not None:
            self.run_model()
        else:
            self.run_generators()
            self.run_inference()
        if self.show_plots:
            self.main_plotters()

    def main_plotters(self):
        self.plot_likelihood()
        self.plot_bayes_perf()
        self.plot_trial()