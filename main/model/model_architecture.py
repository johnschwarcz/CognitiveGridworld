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
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.device_type != "cpu")
        super().to(self.device)
        print("running on:", self.device)

    def init_env_vars(self, kwargs):
        self.__dict__.update(kwargs)
        self.ctx_range, self.batch_range, self.step_range, self.roll_V_range, self.realization_range, self.roll_realization_range, self.obs_range, self.all_K, self.all_Q =\
            tnp([self.ctx_range, self.batch_range, self.step_range, self.roll_V_range, self.realization_range, self.roll_realization_range, self.obs_range, self.all_K, self.all_Q], 'torch', self.device)
        (self.realization_range,self.ctx_range,self.batch_range,self.step_range,self.obs_range) = (x.long() for x in (self.realization_range,self.ctx_range,self.batch_range,self.step_range,self.obs_range))
        self.generator_loss = self.classifier_loss = self.readin_grad = self.readout_grad = self.input_flat = self.update_flat = torch.zeros(1, device = self.device)
        self.classifier_belief_flat = torch.ones(self.BSCR_dims,device = self.device)/self.realization_num
        self.batch_range_ = self.batch_range[:, None]
        self.step_range_ = self.step_range[None,:]
        self.ctx_range_ = self.ctx_range[None,:]
        self.CR = self.ctx_range_.expand(self.batch_num,-1)
        self.BR = self.batch_range_.expand(-1,self.ctx_num)

    def default_internal_embeddings(self):
        if self.learn_embeddings:
            self.all_K, self.all_Q = [torch.randn(*self.state_obs_dims, self.hid_dim).to(self.device) for _ in range(2)]
            self.all_K = nn.Parameter(self.all_K / torch.norm(self.all_K, dim=-1, keepdim=True))
            self.all_Q = nn.Parameter(self.all_Q / torch.norm(self.all_Q, dim=-1, keepdim=True))

    def default_classifier(self):
        if self.mode == "lazyrich":
            self.lazyrich_classifier()
            return
        inp_dims = (self.Z_num + 1) * self.obs_num
        out_dims = self.realization_num * self.ctx_num
        self.classifier_readin = self.mlp_stack(inp_dims, self.hid_dim, self.readin_depth)   
        self.classifier_readout = self.mlp_stack(self.hid_dim, out_dims, self.readout_depth)
        params = list(self.classifier_readin.parameters()) + list(self.classifier_readout.parameters())
        if self.mode != "FF":
            self.LSTM = nn.LSTM(self.hid_dim, self.hid_dim, batch_first = True)  
            self.STM, self.LTM = [nn.Parameter(torch.randn(1, 1, self.hid_dim)) for _ in range(2)]
            params = params + [self.STM, self.LTM]
            if not self.reservoir:
                params += list(self.LSTM.parameters()) 

        self.classifier_optim = optim.Adam([{'params': params,'lr': self.classifier_LR}]) 

    def mlp_stack(self, in_dim, out_dim, depth):
        if depth == 1:
            return nn.Linear(in_dim, out_dim)
        layers = []
        for d in range(depth):
            layers.append(nn.Linear(in_dim if d == 0 else self.hid_dim, out_dim if d == depth - 1 else self.hid_dim))
            if d < depth - 1:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def trainable_params(self):
        return sum(p.numel() for g in self.classifier_optim.param_groups for p in g['params'] if p.requires_grad)

    def default_generator(self):
        
        if self.learn_embeddings:
            self.K_downscale = nn.Linear(self.hid_dim, self.KQ_dim)
            self.Q_downscale = nn.Linear(self.hid_dim, self.KQ_dim)
            embedding_params = ([self.all_K, self.all_Q] +
                                list(self.K_downscale.parameters()) +
                                list(self.Q_downscale.parameters()))
            if self.mode == "ablation":
                self.classifier_optim.add_param_group({'params': embedding_params, 'lr': self.generator_LR})
                return

            self.Z_to_pobs = nn.Linear(self.Z_num , self.hid_dim)            
            self.sample_to_emb = nn.Embedding(self.realization_num, self.hid_dim)
            self.sample_to_hid = nn.Linear(self.hid_dim * self.ctx_num, self.hid_dim)
            self.gen_hid2hid = nn.Linear(self.hid_dim, self.hid_dim)
            self.conf_to_hid = nn.Linear(self.ctx_num, self.hid_dim)
            self.hid_to_pobs = nn.Linear(self.hid_dim, 1)

            params = [{'params': embedding_params +
                        list(self.Z_to_pobs.parameters()) +
                        list(self.sample_to_emb.parameters()) +          
                        list(self.sample_to_hid.parameters()) +  
                        list(self.gen_hid2hid.parameters()) +        
                        list(self.hid_to_pobs.parameters()) +
                        list(self.conf_to_hid.parameters()), 'lr': self.generator_LR}]                
            self.generator_optim = optim.Adam(params)

    ########################################################################################################
    """ lazy/rich RNN — Clark, Bordelon, Zavatone-Veth & Pehlevan, bioRxiv 2026.03.02.708943 """
    ########################################################################################################

    def lazyrich_classifier(self):
        """
        Vanilla rate RNN in the muP-style parameterization of Eqs. (1)-(3):

            tau dx_i/dt = -x_i + (g/sqrt(N)) sum_j J_ij phi_j + sum_a U_ia I_a      (1)
            y_a         = (1/(N gamma)) sum_i V_ia phi_i                            (2)
            E(Theta)    = N gamma^2 L_task + (1/2 beta) ||Theta||^2                 (3)

        J, U, V are the raw O(1) parameters, N(0,1) at init (Tables SI.1/SI.2); every N- and
        gamma-dependent factor sits in the forward pass, not the weights. That is what makes
        the ridge of Eq. (3) a unit-variance Gaussian prior and one global step size correct
        for all three matrices.

        gamma is the knob: at gamma -> 0+ the 1/(N gamma) prefactor absorbs any misalignment
        and J is left untouched (a reservoir); large gamma demands alignment that only
        restructuring can supply. `reservoir=True` freezes J and U at the same output scale --
        an exact no-restructuring control.

        B&P (JSTAT 2023) write the same parameterization with N gamma^2 in the learning rate
        rather than the energy: their gamma_0 is this gamma, and classifier_LR * N * gamma^2
        == their eta_0 gamma_0^2 N, so classifier_LR is their base learning rate eta_0 (both
        printed below). Their App. N proves the family of such bookkeepings is equivalent,
        with gamma = gamma_0 sqrt(N) the unique scaling admitting feature learning.
        """
        N = self.hid_dim
        self.lazyrich_D_in = (self.Z_num + 1) * self.obs_num
        self.lazyrich_D_out = self.realization_num * self.ctx_num

        if self.gamma <= 0:
            raise ValueError("gamma must be > 0 (the paper's limit is gamma -> 0+); "
                             "for the exact reservoir use reservoir=True.")
        self.rnn_alpha = self.rnn_dt / self.rnn_tau                   # only the ratio matters (Sec. SI.6)
        if not 0 < self.rnn_alpha <= 1:
            raise ValueError(f"rnn_dt/rnn_tau = {self.rnn_alpha} must lie in (0, 1] for a stable Euler step.")
        self.rnn_J_scale = self.rnn_gain / np.sqrt(N)                 # g/sqrt(N) in Eq. (1)
        self.rnn_V_scale = 1.0 / (N * self.gamma)                     # 1/(N gamma) in Eq. (2)
        self.lazyrich_energy_scale = N * self.gamma ** 2              # N gamma^2 in Eq. (3)
        if self.rnn_input_scale is None:
            self.rnn_input_scale = 1.0 / np.sqrt(self.lazyrich_D_in)  # keeps the input drive O(1), as in Sec. SI.6

        self.rnn_J = nn.Parameter(torch.randn(N, N))                  # Theta = {J, U, V}
        self.rnn_U = nn.Parameter(torch.randn(N, self.lazyrich_D_in))
        self.rnn_V = nn.Parameter(torch.randn(N, self.lazyrich_D_out))
        self.register_buffer("rnn_J_init", self.rnn_J.detach().clone())

        if self.reservoir:
            self.rnn_J.requires_grad_(False)
            self.rnn_U.requires_grad_(False)
        self.lazyrich_params = [p for p in (self.rnn_J, self.rnn_U, self.rnn_V) if p.requires_grad]

        # Langevin gradient flow, Eq. (4). The ridge gradient (1/beta) Theta is supplied by
        # weight_decay; the matching sqrt(2 lr / beta) kick is injected in lazyrich_update().
        self.classifier_optim = optim.SGD(self.lazyrich_params, lr = self.classifier_LR, weight_decay = 1.0 / self.rnn_beta)
        self.lazyrich_noise_std = np.sqrt(2.0 * self.classifier_LR / self.rnn_beta)
        self.lazyrich_steps = 0
        self.lazyrich_clipped = 0
        # Equilibration diagnostic (Clark et al. Sec. SI.6): snapshot J halfway through so the
        # second half of training can be checked for a spectrum that has stopped moving.
        self.register_buffer("rnn_J_half", torch.zeros_like(self.rnn_J), persistent = False)
        self.lazyrich_snapshot_at = max(1, int(self.episodes) // 2)

        # classifier_LR is Bordelon & Pehlevan's base learning rate eta_0: the multiplier this
        # applies to grad L is lr * N * gamma^2, which is exactly their eta = eta_0 gamma_0^2 N.
        print(f"lazyrich: N={N} g={self.rnn_gain} gamma={self.gamma} beta={self.rnn_beta:g} "
              f"dt/tau={self.rnn_alpha} | init readout scale 1/(sqrt(N) gamma) = "
              f"{1/(np.sqrt(N)*self.gamma):.3f} | eta_0={self.classifier_LR} -> "
              f"eta_0 N gamma^2 = {self.classifier_LR * N * self.gamma**2:.1f}"
              + ("  [reservoir: J, U frozen]" if self.reservoir else ""))

    def lazyrich_restructuring(self):
        """||J - J_init||_F / ||J_init||_F. Read against lazyrich_noise_floor(): Langevin noise
        diffuses the weights even when learning does nothing, so this has a gamma-independent
        floor of noise_std*sqrt(steps) and only the excess is restructuring (the paper's
        representational-drift point). lazyrich_spectrum() is sharper -- outlier eigenvalues
        are self-averaging, so isotropic noise does not move them."""
        with torch.no_grad():
            return (self.rnn_J - self.rnn_J_init).norm().div(self.rnn_J_init.norm()).item()

    def lazyrich_clip_rate(self):
        """Fraction of updates where rnn_grad_clip bound. Must be ~0: ||grad E|| scales as gamma,
        so a binding clip flattens the lazy/rich axis toward one effective richness."""
        return self.lazyrich_clipped / max(self.lazyrich_steps, 1)

    def lazyrich_noise_floor(self):
        """Relative drift expected from the injected Langevin noise alone, with no learning."""
        if not self.rnn_langevin:
            return 0.0
        return self.lazyrich_noise_std * np.sqrt(max(self.lazyrich_steps, 0))

    def lazyrich_equilibration(self):
        """Clark Sec. SI.6's criterion for how long to train: compare the spectrum of
        (g/sqrt(N))J halfway through against the end. Once it stops moving the sampler is at
        equilibrium and gamma, not the episode count, sets the restructuring. Returns (weight
        drift, relative change in max|lambda|) over the second half; the second is the
        meaningful one, being self-averaging."""
        with torch.no_grad():
            if not self.rnn_J_half.any():
                return float("nan"), float("nan")
            dJ = (self.rnn_J - self.rnn_J_half).norm().div(self.rnn_J_half.norm()).item()
            half = np.abs(torch.linalg.eigvals(
                self.rnn_J_scale * self.rnn_J_half.detach().float().cpu()).numpy()).max()
            now = np.abs(self.lazyrich_spectrum()).max()
            return dJ, abs(now - half) / max(half, 1e-12)

    def lazyrich_spectrum(self, at_init = False):
        """Eigenvalues of (g/sqrt(N)) J (Figs. 3G, 4D): a circular-law bulk of radius g, out of
        which increasing gamma pulls complex-conjugate outliers carrying the learned dynamics."""
        with torch.no_grad():
            J = self.rnn_J_init if at_init else self.rnn_J
            return torch.linalg.eigvals(self.rnn_J_scale * J.detach().float().cpu()).numpy()

    def get_gradient_norm(self, layer, s = 0):
        for p in layer.parameters():
            s = s + p.grad.detach().pow(2).sum()
        return s.sqrt().item()
