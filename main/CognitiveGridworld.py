import torch
import os
import sys
import inspect

path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
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
        self.plot_every = init.get('plot_every', 2)
        self.test_states = init.get('test_states', self.state_num // 10)
        self.subsample_states = init.get('subsample_states', None)

        self.likelihood_temp = init.get('likelihood_temp', 2)
        self.likelihood_freq = init.get('likelihood_freq', 1)
        self.realization_num = init.get('realization_num', 10)
        self.batch_num = init.get('batch_num', 1000)
        self.step_num = init.get('step_num', 100)
        self.ctx_num = init.get('ctx_num', 2)
        self.obs_num = init.get('obs_num', 5)
        self.KQ_dim = init.get('KQ_dim', 30)

        self.skip_training_analyses = init.get('skip_training_analyses', False)
        self.early_stopping = init.get('early_stopping', False)

        self.control_ent_bonus = init.get('control_ent_bonus', .05)
        self.classifier_ent_bonus = init.get('classifier_ent_bonus', 0.1)
        self.classifier_LR = init.get('classifier_LR', 0.0005)
        self.controller_LR = init.get('controller_LR', 0.001)
        self.generator_LR = init.get('generator_LR', 0.001)
        self.readout_depth = init.get('readout_depth', 1)
        self.readin_depth = init.get('readin_depth', 1)
        self.output_joint = init.get('output_joint', False)
        self.gpu_inference = init.get('gpu_inference', False)
        self.embedding_grad = init.get('embedding_grad', 'generator')   # 'generator' | 'classifier' | 'both'
        self.embedding_reg = init.get('embedding_reg', True)
        self.hid_dim = init.get('hid_dim', 1000)
        
        # --- lazy/rich RNN mode ("lazyrich"), Clark, Bordelon, Zavatone-Veth & Pehlevan,
        self.gamma = init.get('gamma', 1.0)                     # output coupling; ->0+ lazy/reservoir, large = rich
        self.rnn_gain = init.get('rnn_gain', 1.5)               # g, recurrent gain of Eq. (1)
        self.rnn_beta = init.get('rnn_beta', 1e6)               # beta, inverse temperature of the Langevin flow
        self.rnn_tau = init.get('rnn_tau', 1.0)                 # tau, single-neuron time constant
        self.rnn_dt = init.get('rnn_dt', 0.4)                   # Euler step; only rnn_dt/rnn_tau matters
        self.rnn_x0 = init.get('rnn_x0', 0.0)                   # x^0, initial preactivation
        self.rnn_input_scale = init.get('rnn_input_scale', None)    # None -> 1/sqrt(D_in), as in Sec. SI.6
        self.rnn_grad_clip = init.get('rnn_grad_clip', 10.0)    # gradient-norm clip (Table SI.2)
        self.rnn_langevin = init.get('rnn_langevin', True)      # inject the sqrt(2/beta) noise of Eq. (4)
        self.rnn_nonlinearity = init.get('rnn_nonlinearity', 'tanh')   # phi(.)

        self.learn_embeddings = init.get('learn_embeddings', True)
        self.reservoir = init.get('reservoir', False)
        self.training = init.get('training', False)
        self.load_env = init.get('load_env', None)
        self.save_env = init.get('save_env', None)
        self.external_logger = init.get('external_logger', None)
        self.custom_log = {}

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