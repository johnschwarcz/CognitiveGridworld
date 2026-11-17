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
        if self.mode == "lazyrich":
            self.rnn_J_init = self.rnn_J_init.cpu()     # only the CPU-side diagnostics read it
        print("running on:", self.device)

    def init_env_vars(self, kwargs):
        self.__dict__.update(kwargs)
        self.gen_zflow = kwargs.get('gen_zflow', kwargs.get('mode') == 'oracle')
        if getattr(self, "dream_weight", 0.0) > 0:
            assert self.mode in ("RL", "oracle") and self.learn_embeddings and not getattr(self, "micro_batch", None), \
                "dream needs a Generator and the RNN Classifier (mode RL or oracle, learn_embeddings, no micro_batch)"
        if getattr(self, "freeze_generator", False) or getattr(self, "classifier_reinit", None) is not None or \
                getattr(self, "dream_only", False) or getattr(self, "oracle_teach", False):
            assert self.mode in ("RL", "oracle") and self.learn_embeddings and not getattr(self, "micro_batch", None), \
                "teaching needs a Generator and the RNN Classifier (mode RL or oracle, learn_embeddings, no micro_batch)"
            assert getattr(self, "freeze_generator", False), "classifier_reinit / dream_only / oracle_teach need freeze_generator"
            assert not (getattr(self, "dream_only", False) and getattr(self, "oracle_teach", False)), "one teacher at a time"
            assert not getattr(self, "dream_only", False) or getattr(self, "dream_weight", 0.0) > 0, "dream_only needs dream_weight > 0"
        if getattr(self, "z_raw", False):
            assert self.mode in ("RL", "oracle") and self.learn_embeddings and not getattr(self, "readin_shared", False) \
                and not getattr(self, "reservoir_dedup", False), \
                "z_raw needs the RNN Classifier's default read-in and learned embeddings (mode RL or oracle)"
        if getattr(self, "z_full", False):
            assert getattr(self, "z_raw", False) and self.readin_depth == 1 and self.mode != "FF", \
                "z_full (no downscale, no norm loss) extends z_raw and needs the one-layer RNN read-in"
        if getattr(self, "z_attn", False):
            assert getattr(self, "z_full", False), "z_attn (a self-attention layer over the embeddings) extends z_full"
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
        inp_dims = (self.model_Z_num() + 1) * self.obs_num
        out_dims = self.R_to_the_ctx if self.output_joint else self.realization_num * self.ctx_num
        if self.mode == "FF" and getattr(self, "ff_time_inject", False):
            from main.model.time_ff import TimeInjectedFF
            self.classifier_readin = TimeInjectedFF(inp_dims, self.hid_dim, self.step_num, self.ff_frozen_gain,
                                                    self.ff_regenerate, self.device, shared = getattr(self, "ff_shared_readin", False),
                                                    first_only = getattr(self, "ff_inject_first_only", False),
                                                    layer_gains = getattr(self, "ff_layer_gains", False))
        elif getattr(self, "readin_shared", False):       # one function for every channel's (bit, Z): main/model/shared_readin.py
            from main.model.shared_readin import SharedChannelReadin
            self.classifier_readin = SharedChannelReadin(self.obs_num, self.Z_num, self.hid_dim,
                                                         getattr(self, "readin_channels", None))
        else:
            self.classifier_readin = self.mlp_stack(inp_dims, self.hid_dim, self.readin_depth)   
        self.classifier_readout = self.mlp_stack(self.hid_dim, out_dims, self.readout_depth)
        params = list(self.classifier_readin.parameters()) + list(self.classifier_readout.parameters())
        if self.mode != "FF":
            self.LSTM = nn.LSTM(self.hid_dim, self.hid_dim, batch_first = True)  
            self.STM, self.LTM = [nn.Parameter(torch.randn(1, 1, self.hid_dim)) for _ in range(2)]
            params = params + [self.STM, self.LTM]
            if not self.reservoir:
                params += list(self.LSTM.parameters()) 
            else:
                for p in self.LSTM.parameters():
                    p.requires_grad_(False)
                if getattr(self, "reservoir_fp16", False) or getattr(self, "reservoir_dedup", False):
                    self.LSTM.half()        # frozen, so no fp32 master is needed; autocast makes this same copy per call

        self.classifier_optim = optim.Adam([{'params': params,'lr': self.classifier_LR}])

    def model_Z_num(self):
        """Features per channel of the environment code both modules read (active_Z's last axis). Default: Z_num, the K.Q
        dot products of the ordered pairs of active variables (the per-channel form the environment builds its likelihood
        from). z_raw (2026-11-14, John: "remove the bottleneck from both the classifier and the generator"): the active
        variables' down-scaled K and Q vectors themselves, concatenated per channel (2 * ctx_num * KQ_dim features), so each
        module has to learn the interaction itself. z_full (2026-11-14, John: "remove the downscale and the normalization
        loss in this 'no world structure' setting"): the full embeddings instead (no K / Q down-scaling to the environment's
        KQ_dim, no norm loss): 2 * ctx_num * hid_dim features."""
        if getattr(self, "z_raw", False):
            return 2 * self.ctx_num * (self.hid_dim if getattr(self, "z_full", False) else self.KQ_dim)
        return self.Z_num

    def oracle_readin(self, W, B, norm_dims):
        with torch.no_grad():
            W.copy_((B * (W.norm() / B.norm())).to(W.dtype))
        return nn.ModuleList([nn.LayerNorm(d) for d in norm_dims])

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

    def teach_setup(self):
        """Teaching a Classifier from a frozen trained net (2026-11-13), called by Env_model_manager.run_model after
        load_env (whose load() replaces all_K / all_Q with new Parameters, so it must come after it). classifier_reinit: the
        Classifier's modules (read-in, LSTM, STM / LTM, read-out) are rebuilt by default_classifier() from a private torch
        RNG stream seeded with classifier_reinit (fork_rng: the global stream, hence every episode draw, is that of the same
        run without the re-init). freeze_generator: every parameter outside the Classifier's own group (all_K, all_Q, both
        downscales, the Generator, its stream norms) gets requires_grad off, and the Classifier's optimizer is rebuilt on its
        own parameters only (fresh Adam at classifier_LR; load() restores no optimizer state either). frozen_backward never
        touches generator_optim / combined_optim."""
        if self.classifier_reinit is not None:
            cuda = self.device.type == "cuda"
            with torch.random.fork_rng(devices = [self.device] if cuda else []):
                torch.manual_seed(int(self.classifier_reinit))
                if cuda:
                    with torch.cuda.device(self.device):
                        torch.cuda.manual_seed(int(self.classifier_reinit))
                self.default_classifier()                  # new modules (CPU) and a new classifier_optim on them
            super().to(self.device)                        # in place: the optimizer's Parameters stay the same objects
        if self.freeze_generator:
            own = list(self.classifier_optim.param_groups[0]['params'])      # group 0 = the Classifier (default_classifier)
            ids = {id(p) for p in own}
            for p in self.parameters():
                if id(p) not in ids:
                    p.requires_grad_(False)
            self.classifier_optim = optim.Adam([{'params': own, 'lr': self.classifier_LR}])

    def default_generator(self):
        
        if self.learn_embeddings and getattr(self, "z_full", False):
            embedding_params = [self.all_K, self.all_Q]                   # z_full: no down-scaling layers
            if getattr(self, "z_attn", False):
                # z_attn (2026-11-14, John: "make the first layer from embeddings a transformer attention layer"): one
                # pre-LayerNorm multi-head self-attention layer with a residual connection, over the C active variables'
                # tokens [K_v(o); Q_v(o)] (2 * hid_dim), applied to every channel with the same weights. It is part of the
                # shared environment code, so it is trained by whichever loss trains the embeddings (embedding_grad).
                d = 2 * self.hid_dim
                self.z_attn_norm = nn.LayerNorm(d)
                self.z_attn = nn.MultiheadAttention(d, getattr(self, "z_attn_heads", 16), batch_first = True)
                embedding_params += list(self.z_attn_norm.parameters()) + list(self.z_attn.parameters())
        elif self.learn_embeddings:
            self.K_downscale = nn.Linear(self.hid_dim, self.KQ_dim)
            self.Q_downscale = nn.Linear(self.hid_dim, self.KQ_dim)

            if self.gen_zflow:
                for L in (self.K_downscale, self.Q_downscale):
                    nn.init.orthogonal_(L.weight, gain=(self.hid_dim / self.KQ_dim)**0.5)
                    nn.init.zeros_(L.bias)

            embedding_params = ([self.all_K, self.all_Q] +
                                list(self.K_downscale.parameters()) +
                                list(self.Q_downscale.parameters()))
        if self.learn_embeddings:
            self.Z_to_pobs = nn.Linear(self.model_Z_num() , self.hid_dim)
            self.sample_to_emb = nn.Embedding(self.realization_num, self.hid_dim)
            self.sample_to_hid = nn.Linear(self.hid_dim * self.ctx_num, self.hid_dim)
            self.gen_hid2hid = nn.Linear(self.hid_dim, self.hid_dim)
            self.conf_to_hid = nn.Linear(self.ctx_num, self.hid_dim)
            self.hid_to_pobs = nn.Linear(self.hid_dim, 1)

            if self.gen_zflow and getattr(self, "z_raw", False):
                # z_raw: no structured start for the Z read-in (the directions below assume one code per K.Q pair term);
                # the same layer norms, Z_to_pobs keeps its default init
                self.gen_stream_norm = nn.ModuleList([nn.LayerNorm(d) for d in [self.hid_dim] * 3])
            elif self.gen_zflow:
                Wo = self.hid_to_pobs.weight[0].double()
                dirs = (Wo, self.gen_hid2hid.weight.double().T @ Wo)
                # C > 2 (2026-11-12): the likelihood's logit is a sum of pairwise terms of the C = 2 form, so each pair's two
                # codes get the two directions (Z_num = 2 at C = 2: unchanged; 1 at C = 1)
                B = torch.stack([dirs[j % 2] for j in range(self.Z_num)], dim = 1)
                self.gen_stream_norm = self.oracle_readin(self.Z_to_pobs.weight, B, [self.hid_dim] * 3)

            generator_params = ([] if self.embedding_grad == "classifier" else embedding_params)
            if self.embedding_grad == "classifier":
                self.classifier_optim.add_param_group({'params': embedding_params, 'lr': self.generator_LR})

            params = [{'params': generator_params +
                        list(self.Z_to_pobs.parameters()) +
                        list(self.sample_to_emb.parameters()) +          
                        list(self.sample_to_hid.parameters()) +  
                        list(self.gen_hid2hid.parameters()) +        
                        list(self.hid_to_pobs.parameters()) +
                        list(self.conf_to_hid.parameters()) +
                        (list(self.gen_stream_norm.parameters()) if self.gen_zflow else []),
                        'lr': self.generator_LR}]                
            self.generator_optim = optim.Adam(params)
            if self.embedding_grad == "both":
                # one graph, one step: the shared all_K -> active_Z subtree cannot take two backwards
                self.combined_optim = optim.Adam(
                    [{'params': [p for g in self.classifier_optim.param_groups for p in g['params']],
                      'lr': self.classifier_LR},
                     {'params': [p for g in self.generator_optim.param_groups for p in g['params']],
                      'lr': self.generator_LR}])

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
        restructuring can supply.

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
                             "use a small gamma for the lazy regime.")
        self.rnn_alpha = self.rnn_dt / self.rnn_tau                   # only the ratio matters (Sec. SI.6)
        if not 0 < self.rnn_alpha <= 1:
            raise ValueError(f"rnn_dt/rnn_tau = {self.rnn_alpha} must lie in (0, 1] for a stable Euler step.")
        self.rnn_J_scale = self.rnn_gain / np.sqrt(N)                 # g/sqrt(N) in Eq. (1)
        self.rnn_V_scale = 1.0 / (N * self.gamma)                     # 1/(N gamma) in Eq. (2)
        self.lazyrich_energy_scale = N * self.gamma ** 2              # N gamma^2 in Eq. (3)
        if self.rnn_input_scale is None:
            self.rnn_input_scale = 1.0 / np.sqrt(self.lazyrich_D_in)  # keeps the input drive O(1), as in Sec. SI.6

        self.rnn_J = nn.Parameter(torch.randn(N, N))                  # Theta = {J, U, V}
        if self.reservoir:                                            # J frozen at init: out of the optimizer, its ridge and the noise
            self.rnn_J.requires_grad_(False)
        self.rnn_U = nn.Parameter(torch.randn(N, self.lazyrich_D_in))
        self.rnn_V = nn.Parameter(torch.randn(N, self.lazyrich_D_out))
        if self.rnn_skip:                                             # readout of the memoryless bypass, scaled like V
            self.rnn_V_skip = nn.Parameter(torch.randn(N, self.lazyrich_D_out))
        self.register_buffer("rnn_J_init", self.rnn_J.detach().clone())
        self.lazyrich_params = [p for p in (self.rnn_J, self.rnn_U, self.rnn_V, getattr(self, "rnn_V_skip", None))
                                if p is not None and p.requires_grad]

        # Langevin gradient flow, Eq. (4). The ridge gradient (1/beta) Theta is supplied by
        # weight_decay; the matching sqrt(2 lr / beta) kick is injected in lazyrich_update().
        self.classifier_optim = optim.SGD(self.lazyrich_params, lr = self.classifier_LR, weight_decay = 1.0 / self.rnn_beta)
        self.lazyrich_noise_std = np.sqrt(2.0 * self.classifier_LR / self.rnn_beta)
        self.lazyrich_steps = 0
        self.lazyrich_clipped = 0
        # Equilibration diagnostic (Clark et al. Sec. SI.6): snapshot J halfway through so the
        # second half of training can be checked for a spectrum that has stopped moving.
        self.rnn_J_half = None                                        # CPU copy, taken in lazyrich_update()
        self.lazyrich_snapshot_at = max(1, int(self.episodes) // 2)

        # classifier_LR is Bordelon & Pehlevan's base learning rate eta_0: the multiplier this
        # applies to grad L is lr * N * gamma^2, which is exactly their eta = eta_0 gamma_0^2 N.
        print(f"lazyrich: N={N} g={self.rnn_gain} gamma={self.gamma} beta={self.rnn_beta:g} "
              f"dt/tau={self.rnn_alpha} | init readout scale 1/(sqrt(N) gamma) = "
              f"{1/(np.sqrt(N)*self.gamma):.3f} | eta_0={self.classifier_LR} -> "
              f"eta_0 N gamma^2 = {self.classifier_LR * N * self.gamma**2:.1f}"
              f"{' | skip' if self.rnn_skip else ''}{' | J frozen (reservoir)' if self.reservoir else ''}")

    def lazyrich_restructuring(self):
        """||J - J_init||_F / ||J_init||_F. Read against lazyrich_noise_floor(): Langevin noise
        diffuses the weights even when learning does nothing, so this has a gamma-independent
        floor of noise_std*sqrt(steps) and only the excess is restructuring (the paper's
        representational-drift point). lazyrich_spectrum() is sharper -- outlier eigenvalues
        are self-averaging, so isotropic noise does not move them."""
        with torch.no_grad():
            J0 = self.rnn_J_init
            return (self.rnn_J.detach().to(J0.device) - J0).norm().div(J0.norm()).item()

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
            if self.rnn_J_half is None:
                return float("nan"), float("nan")
            dJ = (self.rnn_J.detach().cpu() - self.rnn_J_half).norm().div(self.rnn_J_half.norm()).item()
            half = np.abs(torch.linalg.eigvals(
                self.rnn_J_scale * self.rnn_J_half.float()).numpy()).max()
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
