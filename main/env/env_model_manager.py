import torch; import numpy as np; import pylab as plt; from tqdm import tqdm
from main.utils import tnp; from .env_model_data_manager import Env_model_data_manager
from main.model.model_architecture import Model_architecture as Model

class Env_model_manager(Env_model_data_manager):

    def run_model(self):        
        self.model = Model(**vars(self))
        if type(self.load_env) == str:
            self.load() # Loads both env and model

        self.prep_data_manager()
        self.episode_loop()

        if self.show_plots:
            self.plot_model_perf()
        if self.save_env is not None:
            self.save() # Saves both env and model

    def episode_loop(self):
        for self.e in tqdm(range(self.episodes)):   
            self.log_ind = self.e % self.checkpoint_every
            self.test_set = (self.log_ind < self.test_eps) or (self.training == False)
            if self.log_ind == 0:
                if self.show_plots * (self.e > self.checkpoint_every):
                   self.plot_model_perf()
                self.preprocess_env()
            self.run_generators()

            if self.test_set or self.mode == "SANITY":
                self.run_inference()    
            self.prep_model()
            self.forward_backward()                                
            self.log_model_perf()
        
    def prep_model(self):               
        args_for_model = {
                        'joint_goal_belief': self.joint_goal_belief,                          
                        'obs_flat'         : self.obs_flat,
                        'goal_ind'         : self.goal_ind,
                        'goal_value'       : self.goal_value,
                        'ctx_inds'         : self.ctx_inds,
                        'ctx_vals'         : self.ctx_vals}
        keys, vals = zip(*args_for_model.items())
        args_for_model = dict(zip(keys, tnp(list(vals),'torch',self.device)))
        self.model.__dict__.update(args_for_model)
        self.model.ctx_vals = self.model.ctx_vals.long()
        self.model.ctx_inds = self.model.ctx_inds.long()
        self.model.goal_ind = self.model.goal_ind.long()

    def forward_backward(self):
        if self.test_set:
            with torch.no_grad():
                classifier_belief_flat, self.model_goal_belief = self.model.forward_pass()        
        else:  
            classifier_belief_flat, self.model_goal_belief = self.model.forward_pass()        
            self.classifier_loss, self.generator_loss = self.model.backward_pass()
        self.model_est, _, self.model_acc, self.model_TP, self.model_mse = self.get_goal_performance(classifier_belief_flat)   