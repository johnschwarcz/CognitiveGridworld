import numpy as np; import torch; import torch.nn as nn; import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from main.utils import tnp; from main.model.model_controller import Model_controller 

class Model_architecture(Model_controller):
    def __init__(self, **kwargs):
        super(Model_architecture, self).__init__()
        self.init_env_vars(kwargs)
        self.MC_init_internal_embeddings()
        self.MC_init_classifier()
        self.MC_init_generator()
        super().to(self.device)
        print("running on:", self.device)

    def init_env_vars(self, kwargs):
        self.__dict__.update(kwargs)
        self.ctx_range, self.batch_range, self.step_range, self.roll_V_range, self.realization_range, self.roll_realization_range, self.obs_range, self.all_K, self.all_Q =\
            tnp([self.ctx_range, self.batch_range, self.step_range, self.roll_V_range, self.realization_range, self.roll_realization_range, self.obs_range, self.all_K, self.all_Q], 'torch', self.device)
        (self.realization_range,self.ctx_range,self.batch_range,self.step_range,self.obs_range) = (x.long() for x in (self.realization_range,self.ctx_range,self.batch_range,self.step_range,self.obs_range))
        self.classifier_belief_flat = torch.ones(self.BSCR_dims,device = self.device)/self.realization_num
        self.generator_loss = self.classifier_loss = torch.zeros(1, device = self.device)
        self.batch_range_ = self.batch_range[:, None]
        self.step_range_ = self.step_range[None,:]
        self.ctx_range_ = self.ctx_range[None,:]
        self.CR = self.ctx_range_.expand(self.batch_num,-1)
        self.BR = self.batch_range_.expand(-1,self.ctx_num)

    def default_internal_embeddings(self):
        if self.learn_embeddings:
            print("LEARNING embedding space")
            self.all_K, self.all_Q = [torch.randn(*self.state_obs_dims, self.hid_dim).to(self.device) for _ in range(2)]
            self.all_K = nn.Parameter(self.all_K / torch.norm(self.all_K, dim=-1, keepdim=True))
            self.all_Q = nn.Parameter(self.all_Q / torch.norm(self.all_Q, dim=-1, keepdim=True))
        else:
            print("GIVEN embedding space")

    def default_classifier(self):
        inp_dims = (self.Z_num + 1) * self.obs_num
        self.classifier_readin = nn.Linear(inp_dims, self.hid_dim)   
        self.LSTM = nn.LSTM(self.hid_dim, self.hid_dim, batch_first = True)  
        self.classifier_readout = nn.Linear(self.hid_dim, self.realization_num * self.ctx_num)
        self.STM, self.LTM = [nn.Parameter(torch.randn(1, 1, self.hid_dim)) for _ in range(2)]
        params = [self.STM, self.LTM] + list(self.classifier_readin.parameters()) + list(self.classifier_readout.parameters())
        if not self.reservoir:
            params += list(self.LSTM.parameters()) 
        self.classifier_optim = optim.Adam([{'params': params,'lr': self.classifier_LR}]) 

    def default_generator(self):
        
        if self.learn_embeddings:
            self.Z_to_pobs = nn.Linear(self.Z_num , self.hid_dim)            
            self.sample_to_emb = nn.Embedding(self.realization_num, self.hid_dim)
            self.sample_to_hid = nn.Linear(self.hid_dim * self.ctx_num, self.hid_dim)

            self.conf_to_hid = nn.Linear(self.ctx_num, self.hid_dim)
            self.hid_to_hid = nn.Linear(self.hid_dim, self.hid_dim)
            self.hid_to_hid2 = nn.Linear(self.hid_dim, self.hid_dim)
            self.K_downscale = nn.Linear(self.hid_dim, self.KQ_dim)
            self.Q_downscale = nn.Linear(self.hid_dim, self.KQ_dim)
            self.hid_to_pobs = nn.Linear(self.hid_dim, 1)

            params = [{'params': [self.all_K, self.all_Q] +
                        list(self.Z_to_pobs.parameters()) +
                        list(self.sample_to_emb.parameters()) +          
                        list(self.sample_to_hid.parameters()) +          
                        list(self.hid_to_pobs.parameters()) +
                        list(self.K_downscale.parameters()) +
                        list(self.Q_downscale.parameters()) +
                        list(self.hid_to_hid.parameters()) +
                        list(self.hid_to_hid2.parameters()) +          
                        list(self.conf_to_hid.parameters()), 'lr': self.generator_LR}]                
            self.generator_optim = optim.Adam(params)

    def get_gradient_norm(self, layer, s = 0):
        for p in layer.parameters():
            s = s + p.grad.detach().pow(2).sum()
        return s.sqrt().item()
