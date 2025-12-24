import torch; import numpy as np; import pylab as plt; from tqdm import tqdm
from main.utils import tnp; from main.env.env_preprocessing import Env_preprocessing
from main.model.model_architecture import Model_architecture as Model

class Env_control_manager(Env_preprocessing):

    def train_controller(self, eps, reps = 1, offline_teacher = None):
        self.controller_training_episodes = eps; self.reps = reps
        self.offline_teacher = offline_teacher;  self.test_set = True
        self.controller_training_logs = {'reward': np.zeros((self.reps, self.controller_training_episodes)),
                                         'example_policy': np.zeros((self.reps, self.controller_training_episodes, *self.ctx_dims)),
                                         'prefence_landscape': np.zeros((self.reps, *self.ctx_dims)), 'optimality': np.zeros(self.reps),}
        for self.rep in tqdm(range(reps),  desc = f"{offline_teacher or "ONLINE"}"):
            self.EC_gen_context()
            self.EC_gen_likelihoods()
            self.get_opt_preference()
            self.init_controller()
            self.controller_loop()

    def get_opt_preference(self):
        self.preferences = (np.random.rand(self.obs_num) > 0.5)
        P = self.preferences.reshape(1, self.obs_num, *([1]*self.ctx_num))
        PL = P * np.log(self.joint_likelihood) + (1-P) * np.log(1-self.joint_likelihood)

        self.prefence_landscape = np.exp( (PL).sum(1) / self.obs_num)
        self.controller_training_logs['prefence_landscape'][self.rep] = self.prefence_landscape.copy()[0]
        peak = self.prefence_landscape.max(axis = tuple(range(-self.ctx_num, 0)), keepdims = True).mean()
        self.controller_training_logs['optimality'][self.rep] = peak

    def init_controller(self):
        self.EC_gen_realizations()
        self.EC_gen_observations()
        self.prep_model() 
        JL, NL = tnp([self.joint_likelihood, self.naive_likelihood], 'torch', self.device)
        self.ctx_vals = self.model.init_controller(self.preferences, self.offline_teacher, JL, NL)

    def controller_loop(self):
        for e in range(self.controller_training_episodes):
            if self.offline_teacher is None:
                self.EC_gen_observations()
                O = self.obs_flat.mean(1)
                self.ctx_vals, argmax_policy, example_policy  = self.model.forward_controller(O)
            else:
                _, argmax_policy, example_policy  = self.model.forward_controller()

            reward = self.prefence_landscape[(self.batch_range,)+tuple(argmax_policy.T)].mean()   
            self.controller_training_logs['example_policy'][self.rep, e] = example_policy
            self.controller_training_logs['reward'][self.rep, e] = reward 