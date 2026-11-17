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
        self.readin_residual = init.get('readin_residual', False)   # FF only: h <- h + relu(W h + b) / sqrt(L) around each of the L N x N readin layers
        # FF only (2026-10-15): time-injected layers (main/model/time_ff.py): layer k gets the mean of the first T+1-k
        # observations through its own readin; frozen N x N layers at nn.Linear's default draws x ff_frozen_gain, rebuilt
        # from seeds on every use with ff_regenerate (too many weights to store at N >= 10000)
        self.ff_time_inject = init.get('ff_time_inject', False)
        self.ff_frozen_gain = init.get('ff_frozen_gain', 1.0)
        self.ff_regenerate = init.get('ff_regenerate', False)
        self.ff_shared_readin = init.get('ff_shared_readin', False)       # one readin for all the time-injected layers
        self.ff_inject_first_only = init.get('ff_inject_first_only', False)   # control: input at layer 1 only (standard depth-T FF)
        self.ff_layer_gains = init.get('ff_layer_gains', False)           # one trainable gain per time-injected layer's input
        self.output_joint = init.get('output_joint', False)
        self.gpu_inference = init.get('gpu_inference', False)
        self.embedding_grad = init.get('embedding_grad', 'generator')   # 'generator' | 'classifier' | 'both'
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
        self.rnn_skip = init.get('rnn_skip', False)             # memoryless bypass: phi(U I(t)) gets its own readout next to the RNN's

        self.learn_embeddings = init.get('learn_embeddings', True)
        self.reservoir = init.get('reservoir', False)
        # Wide reservoirs (2026-10-13). reservoir_fp16: keep the frozen LSTM weights in fp16 only (bit-identical under
        # autocast, which makes that same fp16 copy on every call; half the memory). reservoir_dedup: hand-written frozen
        # LSTM whose input projection runs once per distinct readin input (main/model/frozen_lstm.py; exact; implies
        # fp16 weights). micro_batch: SANITY episodes in slices of this many episodes, gradients accumulated to the
        # whole batch's before one optimizer step (exact at loss power 1).
        self.reservoir_fp16 = init.get('reservoir_fp16', False)
        self.reservoir_dedup = init.get('reservoir_dedup', False)
        self.micro_batch = init.get('micro_batch', None)
        self.training = init.get('training', False)
        self.load_env = init.get('load_env', None)
        self.save_env = init.get('save_env', None)
        self.external_logger = init.get('external_logger', None)
        self.oracle = self.mode == "oracle"
        # the Generator's Z-flow fix (stream LayerNorms, readout-aligned Z init, skip, orthogonal downscale init) without the
        # rest of oracle mode, e.g. under the original feedback (mode RL); defaults to oracle mode
        self.gen_zflow = init.get('gen_zflow', self.oracle)
        # training episodes only (2026-11): None | "low" | "high" | "half" -- joint realizations drawn from the cells where
        # the Naive observer converges to the true goal value ("low" FR), where it does not ("high"), or a fixed random half
        self.fr_select = init.get('fr_select', None)
        # offline imagination (2026-11-13; model_backward.dream_sample / dream_backward): on training iterations (modes RL /
        # oracle) the Classifier also learns, with oracle_loss's cross-entropy, from a batch its own Generator dreams on the
        # same environments; dream_weight scales that loss (0 = off: the code path never runs), dream_start = training
        # iterations without it, dream_mask_test = leave held-out (test-state) variables out of the dream loss (None = in
        # mode RL only, where RL feedback never trains them; oracle_loss supervises them), dream_shuffle = targets taken
        # from another dream episode (no-information control), dream_seed = seed of the dream's private torch RNG stream
        self.dream_weight = init.get('dream_weight', 0.0)
        self.dream_start = init.get('dream_start', 0)
        self.dream_mask_test = init.get('dream_mask_test', None)
        self.dream_shuffle = init.get('dream_shuffle', False)
        self.dream_seed = init.get('dream_seed', 0)
        # teaching a fresh Classifier from a frozen trained net (2026-11-13; model_architecture.teach_setup, model_backward.
        # frozen_backward; modes RL / oracle): freeze_generator = the embeddings and the Generator get no updates (out of
        # every optimizer, requires_grad off; only the Classifier steps), classifier_reinit = None, or the seed of fresh
        # Classifier weights drawn after load_env (needs freeze_generator), dream_only = the Classifier's only loss is the
        # dream loss (needs dream_weight > 0), oracle_teach = its loss is oracle_loss on the real batch, test-state
        # variables masked by dream_mask_test's rule. All off: the code path never runs
        self.classifier_reinit = init.get('classifier_reinit', None)
        self.freeze_generator = init.get('freeze_generator', False)
        self.dream_only = init.get('dream_only', False)
        self.oracle_teach = init.get('oracle_teach', False)
        # speed (2026-11-13, opt-in; results bitwise identical): training iterations keep the network's outputs and losses on
        # the GPU and compute their numpy copies only when read; the training slots of the logs are written once per block
        # (Env_model_manager.forward_backward_deferred, Env_model_data_manager.flush_train_log)
        self.defer_train_outputs = init.get('defer_train_outputs', False)
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