import torch; from torch.distributions import Categorical; from main.utils import tnp
import torch.nn.functional as F
from contextlib import contextmanager
from .model_backward import Model_backward
from .frozen_lstm import frozen_lstm_dedup

class _SharedCastMatmul(torch.autograd.Function):
    """phi @ J.T in fp16 with one copy Jh of J shared by every time step. Plain autocast re-casts
    J.T at each step and keeps every copy for the backward pass: T x N^2 of GPU memory, ~22 GB at
    N = 20000. The gradient still reaches J step by step in fp32, as through autocast's own casts,
    so results are bit-identical to phi @ J.T under autocast."""
    @staticmethod
    def forward(ctx, phi, J, Jh):
        ph = phi.to(Jh.dtype)
        ctx.save_for_backward(ph, Jh)
        return ph @ Jh.T

    @staticmethod
    def backward(ctx, g):
        ph, Jh = ctx.saved_tensors
        grad_J = (g.T @ ph).float() if ctx.needs_input_grad[1] else None   # None when J is frozen (reservoir)
        return (g @ Jh).float(), grad_J, None      # grad_J laid out like J, so accumulating it stays contiguous

class Model_forward(Model_backward):
    def forward_compute(self):
        """forward_pass without its outputs: everything stays on the GPU"""
        self.MC_to_interactions()
        self.MC_to_classification()
        self.MC_to_pobs()

    def forward_pass(self):
        self.forward_compute()
        return tnp([self.classifier_belief_flat, self.classifier_goal_belief, self.input_flat, self.update_flat], 'np')

    def default_interactions(self):
        self.Z_count = 0
        self.active_K = self.all_K[self.ctx_inds, :]
        self.active_Q = self.all_Q[self.ctx_inds, :]
        if self.learn_embeddings and not getattr(self, "z_full", False):     # z_full: the full embeddings, no down-scaling
            self.active_K = self.K_downscale(self.active_K)
            self.active_Q = self.Q_downscale(self.active_Q)

        if getattr(self, "z_raw", False):    # no K.Q bottleneck (model_architecture.model_Z_num): per channel, the active
            B = self.batch_num               # variables' K vectors then their Q vectors, (B, O, 2 * C * KQ_dim), float32
            if getattr(self, "z_attn", False):
                # z_attn (model_architecture.default_generator): per channel, the C active variables are tokens [K_v; Q_v]
                # (2 * hid); h = x + MHA(LN(x)), the same layer for every channel; the code is [h_1; ...; h_C] per channel
                x = torch.cat((self.active_K, self.active_Q), -1).transpose(1, 2)            # (B, O, C, 2 * hid)
                x = x.reshape(B * self.obs_num, self.ctx_num, -1)
                y = self.z_attn_norm(x)
                h = x + self.z_attn(y, y, y, need_weights = False)[0]
                self.active_Z = h.reshape(B, self.obs_num, -1).float()                     # (B, O, C * 2 * hid)
                return
            K = self.active_K.transpose(1, 2).reshape(B, self.obs_num, -1)          # (B, C, O, KQ) -> (B, O, C * KQ)
            Q = self.active_Q.transpose(1, 2).reshape(B, self.obs_num, -1)
            self.active_Z = torch.cat((K, Q), -1).float()
            return
        self.active_Z = torch.zeros(self.Z_dims, device = self.device)
        if self.ctx_num == 1:
            self.KQ_to_Z(0,0)
        else:
            # Python ints, not the elements of the CUDA tensor ctx_range: the same slices, without 2 GPU syncs per pair
            # (range() and indexing turned each 0-d CUDA tensor into an int with .item()) (2026-11-13, speed)
            for self.k in range(self.ctx_num):
                for self.q in range(self.k+1, self.ctx_num):     
                    self.KQ_to_Z(self.k, self.q)
                    self.KQ_to_Z(self.q, self.k)      

    def classifier_Z(self):
        return self.active_Z if self.embedding_grad in ("classifier", "both") else self.active_Z.detach()

    def generator_Z(self):
        return self.active_Z if self.embedding_grad in ("generator", "both") else self.active_Z.detach()

    def KQ_to_Z(self, ctx_1, ctx_2):
        K = self.active_K[:,ctx_1]
        Q = self.active_Q[:,ctx_2]
        Z = (K * Q).sum(-1)
        self.active_Z[:,:, self.Z_count] = Z
        self.Z_count += 1 

    def default_classification(self):
        if self.mode == "FF":
            self.FF_classification()
        elif self.mode == "lazyrich":
            self.lazyrich_classification()
        else:
            self.RNN_classification()

    def FF_classification(self):
        Z = self.classifier_Z()
        if getattr(self, "ff_time_inject", False):              # main/model/time_ff.py: one layer per time step
            steps = torch.arange(1, self.step_num + 1, device = self.device, dtype = self.obs_flat.dtype)
            prefix_means = self.obs_flat.cumsum(1) / steps[None, :, None]
            hid = self.classifier_readin(prefix_means, Z.reshape(self.batch_num, -1))
            logits = self.classifier_readout(hid.unsqueeze(1)).expand(-1, self.step_num, -1)
            self.postprocess_belief(logits, accumulate = False)
            return
        Z = Z.reshape(self.batch_num, 1, -1)
        O = self.obs_flat.mean(1, keepdims = True)
        if self.readin_residual:
            hid = self.residual_readin(torch.cat((O, Z), dim = -1))
        else:
            hid = torch.relu(self.classifier_to_hid(O, Z))
        logits = self.classifier_readout(hid)
        logits = logits.expand(-1, self.step_num, -1)
        self.postprocess_belief(logits, accumulate = False)
    
    def RNN_classification(self):
        if getattr(self, "reservoir_dedup", False):
            update, RNN_input = self.dedup_LSTM()
        else:
            RNN_input = torch.relu(self.RNN_readin())
            stm = self.STM.expand(-1, self.batch_num, -1).contiguous()
            ltm = self.LTM.expand(-1, self.batch_num, -1).contiguous()
            update, _ = self.LSTM(RNN_input, (stm, ltm))
        logits = self.classifier_readout(update)
        self.postprocess_belief(logits)

        if (self.mode == "SANITY") and self.testing and self.keep_test_activations():   # mode first: testing is a GPU tensor
            self.update_flat = update.detach()        
            self.input_flat = (RNN_input() if callable(RNN_input) else RNN_input).detach()
        else:
            self.update_flat = torch.zeros(1, device = self.device)    
            self.input_flat = torch.zeros(1, device = self.device)

    def keep_test_activations(self):
        """The (B,T,N) test-episode copies of the LSTM input and output feed only the training analyses; on the wide-reservoir
        paths (micro_batch, reservoir_dedup) they are skipped when those analyses are (9.6 GB each at N=20000, B=8000)."""
        wide = getattr(self, "micro_batch", None) or getattr(self, "reservoir_dedup", False)
        return not (wide and getattr(self, "skip_training_analyses", False))

    def dedup_LSTM(self):
        """The frozen LSTM of reservoir_dedup (main/model/frozen_lstm.py). The readin input at step t is (obs bits at t, the
        episode's Z), so (episode, obs pattern) identifies it; the trainable readin and the input projection run once per
        distinct pair. returns: the (B,T,N) LSTM output and a callable giving the (B,T,N) readin output if needed."""
        O = self.obs_flat                                                       # (B, T, obs_num), exactly 0/1
        n_codes = 2 ** self.obs_num
        bits = 2 ** torch.arange(self.obs_num, device = self.device)
        key = torch.arange(self.batch_num, device = self.device)[:, None] * n_codes + (O.round().long() * bits).sum(-1)
        uk, inv = torch.unique(key, return_inverse = True)                     # inv (B, T): which distinct input
        O_u = ((uk[:, None] % n_codes) // bits % 2).to(O.dtype)                 # the obs bits of each distinct input
        Z_u = self.classifier_Z().reshape(self.batch_num, -1)[uk // n_codes]
        A_u = torch.relu(self.classifier_to_hid(O_u, Z_u))                      # (U, N)
        L = self.LSTM
        acc = torch.promote_types(L.weight_hh_l0.dtype, torch.float32)
        bias = (L.bias_ih_l0.to(acc) + L.bias_hh_l0.to(acc)).to(L.weight_hh_l0.dtype)
        update = frozen_lstm_dedup(A_u, inv.t().contiguous(), self.STM.reshape(1, -1), self.LTM.reshape(1, -1),
                                   L.weight_ih_l0, L.weight_hh_l0, bias)
        return update, (lambda: A_u[inv])

    @contextmanager
    def batch_slice(self, sl):
        """Point every per-episode tensor, and the batch size, at episodes sl; yields sl's share of the batch."""
        keys = ("obs_flat", "joint_goal_belief", "goal_ind", "goal_value", "ctx_inds", "ctx_vals", "active_Z")
        saved = {k: getattr(self, k) for k in keys + ("batch_range", "batch_num", "BSCR_dims")}
        n = sl.stop - sl.start
        for k in keys:
            setattr(self, k, saved[k][sl])
        self.batch_num, self.batch_range = n, saved["batch_range"][:n]
        self.BSCR_dims = (n, *tuple(saved["BSCR_dims"])[1:])
        try:
            yield n / saved["batch_num"]
        finally:
            self.__dict__.update(saved)

    def forward_pass_micro(self, train, outputs = True):
        """A SANITY episode in slices of micro_batch episodes, so memory follows the slice, not the batch. Training: each
        slice's SANITY_loss, weighted by its share of the batch, is backpropagated, so the gradients accumulate to the
        whole batch's; then one optimizer step. Exact at loss power 1, where the loss is a plain mean. At other powers
        DKL_loss's detached minimum is the slice's: near-exact (<= 0.5% of the gradient weight at initialization with
        slices of 100, ~1e-17 once trained; measured 2026-10-13). The logged loss sums the slices' shares.
        returns: forward_pass()'s outputs, then (training) backward_pass()'s; outputs = False (training only): the same
        slices, losses and step, with nothing gathered or copied to the CPU, and returns None"""
        assert self.mode == "SANITY" and not self.learn_embeddings, "micro_batch supports mode SANITY without learned embeddings"
        assert train or outputs, "outputs = False steps the optimizer: training only"
        self.MC_to_interactions()
        if train:
            self.classifier_optim.zero_grad()
        B, parts, loss = self.batch_num, [], 0.
        keep = bool(self.testing) and self.keep_test_activations()
        for start in range(0, B, self.micro_batch):
            with self.batch_slice(slice(start, min(start + self.micro_batch, B))) as share:
                self.MC_to_classification()
                if outputs:
                    part = [self.classifier_belief_flat, self.classifier_goal_belief.detach(), self.classifier_goal_selection, self.ACC]
                    if keep:
                        part += [self.input_flat, self.update_flat]
                    parts.append(part)
                if train:
                    self.SANITY_loss()
                    self.scaler.scale(self.classifier_loss * share).backward()
                    loss = loss + self.classifier_loss.detach() * share
        if not outputs:
            self.MC_to_pobs()
            self.scaler.step(self.classifier_optim)
            self.scaler.update()
            return
        whole = [torch.cat(x) for x in zip(*parts)]
        self.classifier_belief_flat, self.classifier_goal_belief, self.classifier_goal_selection, self.ACC = whole[:4]
        self.classifier_belief = self.classifier_belief_flat
        if keep:
            self.input_flat, self.update_flat = whole[4:]
        self.MC_to_pobs()
        out = tnp([self.classifier_belief_flat, self.classifier_goal_belief, self.input_flat, self.update_flat], 'np')
        if not train:
            return out
        self.classifier_loss = loss
        self.readin_grad = self.get_gradient_norm(self.classifier_readin)
        self.readout_grad = self.get_gradient_norm(self.classifier_readout)
        self.scaler.step(self.classifier_optim)
        self.scaler.update()
        return [*out, *self.backward_return()]

    def RNN_readin(self):
        if getattr(self, "readin_shared", False):
            return self.classifier_readin.episode(self.obs_flat, self.classifier_Z())
        O = self.obs_flat
        Z = self.classifier_Z()
        if getattr(self, "readin_perm", False):     # each episode's channels in a random order, bits and codes moved together
            perm = torch.rand(self.batch_num, self.obs_num, device=Z.device).argsort(1)
            O = O.gather(2, perm[:, None, :].expand(-1, O.shape[1], -1))
            Z = Z.gather(1, perm[:, :, None].expand(-1, -1, Z.shape[2]))
        if getattr(self, "z_full", False):
            # z_full: the (B, T, O + O * 4 * hid) input is never built; the one-layer read-in is split into its bit and code
            # columns (the same linear map: W [bits, Z] + b = W_bits bits + W_Z Z + b) and its Z part runs once per episode
            W, b = self.classifier_readin.weight, self.classifier_readin.bias
            n_bits = O.shape[-1]
            z_part = Z.reshape(self.batch_num, -1) @ W[:, n_bits:].t() + b                 # (B, hid)
            return O @ W[:, :n_bits].t() + z_part[:, None, :]
        Z = Z.reshape(self.batch_num, 1, -1)
        Z = Z.expand(-1, self.step_num, -1)
        return self.classifier_to_hid(O, Z)

    def residual_readin(self, x):
        """readin_residual: h = relu(W_0 x + b_0), then h <- h + relu(W_l h + b_l) / sqrt(L) for each of the L N x N layers
        of the readin stack, so every layer (trained or frozen) is a residual block and the input reaches the readout
        unattenuated. The 1/sqrt(L) branch scale keeps the residual sum O(1): unscaled, 29 default-init blocks grow it to
        rms ~116, the readout saturates (max prob 1.0) and training locks at chance (tested 2026-10-12)."""
        readin = self.classifier_readin
        linears = [l for l in (readin if isinstance(readin, torch.nn.Sequential) else [readin]) if isinstance(l, torch.nn.Linear)]
        h = torch.relu(linears[0](x))
        scale = max(len(linears) - 1, 1) ** -.5
        for layer in linears[1:]:
            h = h + scale * torch.relu(layer(h))
        return h

    def classifier_to_hid(self, O, Z):
        return self.classifier_readin(torch.cat((O, Z), dim = -1))

    def postprocess_belief(self, logits, accumulate = True):
        if accumulate:
            logits = logits.cumsum(1)
        belief = self.logits_to_belief(logits)
        self.classifier_belief = belief
        self.classifier_belief_flat = belief.detach()
        self.classifier_goal_belief = belief[self.batch_range, :, self.goal_ind]
        self.classifier_goal_selection = Categorical(self.classifier_goal_belief, validate_args = self.categorical_checks()).sample() # SAMPLES FROM MARGINAL BELIEF
        self.ACC = (self.classifier_goal_selection == self.goal_value[:,None]).float() 

    def categorical_checks(self):
        """None: torch's default argument checks (a GPU sync each); False under defer_train_outputs (same samples)"""
        return False if getattr(self, "defer_train_outputs", False) else None

    def logits_to_belief(self, logits):
        if self.output_joint:
            belief = torch.softmax(logits, -1)
            B, S, R, C = self.batch_num, self.step_num, self.realization_num, self.ctx_num
            return torch.stack([belief.reshape(B, S, R**c, R, R**(C - 1 - c)).sum((2, 4)) for c in range(C)], dim = 2)            

        logits = logits.reshape(self.BSCR_dims)
        return torch.softmax(logits, -1)
        
    def default_pobs(self, training_controller = False):
        if self.learn_embeddings or training_controller: 
            if training_controller or self.oracle:
                conf = torch.ones(*self.batch_ctx_dims, device = self.device)
                sample = self.controller_actions if training_controller else self.ctx_vals
            else:
                CBF = self.classifier_belief_flat[:, -1]
                sample = torch.distributions.Categorical(probs=CBF, validate_args = self.categorical_checks()).sample()   # SAMPLES FROM JOINT BELIEF
                sample[self.batch_range, self.goal_ind] = self.classifier_goal_selection[:, -1]
                conf = CBF[self.BR, self.CR, sample]
                conf[self.batch_range, self.goal_ind] = self.ACC[:,-1]
 
            self.get_prediction(sample, conf)
                
    def get_prediction(self, sample, conf):
        sample_emb = self.sample_to_emb(sample).reshape(self.batch_num, -1)
        s = self.sample_to_hid(sample_emb).unsqueeze(1)
        c = self.conf_to_hid(conf).unsqueeze(1)
        z = self.Z_to_pobs(self.generator_Z())
        if self.gen_zflow:
            s, c, z = self.get_layer_norm(self.gen_stream_norm, (s, c, z))

        pre = s + c + z
        x = self.gen_hid2hid(torch.relu(pre))
        x = self.hid_to_pobs(torch.relu(pre + x if self.gen_zflow else x))
        self.pred_pobs = torch.sigmoid(x).squeeze()

    def get_layer_norm(self, norm, inps):
        return (n(v) for n, v in zip(norm, inps))

    ########################################################################################################
    """ lazy/rich RNN — Clark, Bordelon, Zavatone-Veth & Pehlevan, bioRxiv 2026.03.02.708943 """
    ########################################################################################################

    def lazyrich_phi(self, x):
        if self.rnn_nonlinearity == "tanh":     return torch.tanh(x)
        if self.rnn_nonlinearity == "relu":     return torch.relu(x)
        if self.rnn_nonlinearity == "linear":   return x
        raise ValueError(f"unknown rnn_nonlinearity: {self.rnn_nonlinearity}")

    def lazyrich_classification(self):
        """
        Euler integration of Eqs. (SI.1)-(SI.3) with alpha = dt/tau:

            x(t_1)  = x^0
            x(t_n)  = (1-alpha) x(t_{n-1}) + alpha [ (g/sqrt(N)) J phi(x(t_{n-1})) + U I(t_{n-1}) ]
            y(t_n)  = (1/(N gamma)) V^T phi(x(t_n))

        Step t consumes observation t and reads out the state it produces, so y[t] sees
        observations 0..t -- the same causal alignment as the LSTM path, keeping accuracy, TP
        and MSE comparable across modes. The readout emits belief *increments*, cumsum-ed
        before the softmax exactly as in postprocess_belief.

        The state stays fp32 under autocast: gamma's whole point is a small but coherent
        perturbation to J, and the recurrence is where that would be lost.
        """
        # Drop the last test episode's (B,T,N) diagnostics before this forward allocates its own;
        # forward_pass has already handed numpy copies to the env.
        self.update_flat = self.input_flat = self.lazyrich_preact = None
        I = self.lazyrich_readin()                                              # (B, T, D_in)
        drive = self.rnn_input_scale * torch.einsum('bti,ni->btn', I, self.rnn_U)

        testing = bool(self.testing)
        phis, preacts = [], []
        for x, phi in self.lazyrich_states(drive, self.rnn_J):
            phis.append(phi)
            if testing:
                preacts.append(x)                                               # only kept for the diagnostics below

        phi_flat = torch.stack(phis, 1)                                         # (B, T, N)
        update = self.rnn_V_scale * (phi_flat @ self.rnn_V)                     # Eq. (2)
        if self.rnn_skip:
            # Bypass past the recurrence: the same input drive U I(t), no J and no memory, with
            # its own readout on the same 1/(N gamma) scale. It alone can represent naive Bayes.
            update = update + self.rnn_V_scale * (self.lazyrich_phi(drive) @ self.rnn_V_skip)
        self.postprocess_belief(update)

        if testing:
            # phi is what the DMFT's C(t,t') = (1/N) sum_i phi_i(t) phi_i(t') is built from, so
            # it is the right analogue of the LSTM hidden state for the existing PR/PCA code.
            # .float(): a (B,T,N) fp16 array overflows any downstream variance/PCA reduction.
            self.update_flat = phi_flat.detach().float()
            self.input_flat = drive.detach().float()
            self.lazyrich_preact = torch.stack(preacts, 1).detach().float()     # x(t), for Figs. 3J / 4G
        else:
            self.update_flat = torch.zeros(1, device = self.device)
            self.input_flat = torch.zeros(1, device = self.device)

    def lazyrich_states(self, drive, J):
        """Euler steps of Eq. (SI.2) from x^0 under recurrent weights J: yields (x(t), phi(x(t))).
        Under autocast J is cast to fp16 once per forward and shared by every step (_SharedCastMatmul)."""
        amp = torch.is_autocast_enabled(self.device.type)
        Jh = J.detach().to(torch.get_autocast_dtype(self.device.type)) if amp else None
        x = torch.full((self.batch_num, self.hid_dim), float(self.rnn_x0), device = self.device)
        phi = self.lazyrich_phi(x)                                              # phi(x^0), not read out
        for t in range(self.step_num):
            rec = _SharedCastMatmul.apply(phi, J, Jh) if amp else phi @ J.T
            x = (1 - self.rnn_alpha) * x + self.rnn_alpha * (
                self.rnn_J_scale * rec + drive[:, t])
            phi = self.lazyrich_phi(x)
            yield x, phi

    def lazyrich_readin(self):
        """Raw I_a(t): observation bits plus the (time-constant) interaction scalars Z. No
        learned projection or nonlinearity -- U is the only thing between task and preactivation."""
        Z = self.classifier_Z().reshape(self.batch_num, 1, -1).expand(-1, self.step_num, -1)
        return torch.cat((self.obs_flat, Z), dim = -1)
