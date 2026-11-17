import torch; import numpy as np; import pylab as plt; from tqdm import tqdm; from contextlib import nullcontext
from main.utils import tnp; from main.env.env_model_data_manager import Env_model_data_manager
from main.model.model_architecture import Model_architecture as Model

class Env_model_manager(Env_model_data_manager):

    def run_model(self, disable_tqdm = False):        
        self.model = Model(**vars(self))
        if type(self.load_env) == str:
            self.load() # Loads both env and model
        if getattr(self, "freeze_generator", False):
            assert type(self.load_env) == str and getattr(self, "loaded", False), "freeze_generator needs a loaded net (load_env)"
            self.model.teach_setup() # after load(): fresh / frozen modules (model_architecture.teach_setup)
        self.prep_data_manager()
        self.episode_loop(disable_tqdm)
        self.flush_train_log()
        self.custom_log = {k: np.stack(v) for k, v in self.custom_log.items()}
        if self.save_env is not None:
            self.save() # Saves both env and model

    def episode_loop(self, disable_tqdm):
        pbar = tqdm(range(self.episodes), disable=disable_tqdm)
        for self.e in pbar:   
            self.preprocess_episode()
            self.run_episode()
            if self.test_set:
                curr_joint = self.joint_acc.mean(0)[-1]
                curr_naive = self.naive_acc.mean(0)[-1]
                curr_test = self.test_accs[self.test_e - 1, -1]
                pbar.set_postfix_str(f"acc {curr_test:.3f} | joint {curr_joint:.3f} | naive {curr_naive:.3f}")
                if self.early_stopping:
                    if curr_test > curr_joint:
                        print("Stopping early: ", curr_test, " > ", curr_joint)
                        return 

    def preprocess_episode(self):
        self.log_ind = self.e % self.checkpoint_every
        self.test_set = (self.log_ind == 0) or (self.training == False)
        if self.test_set or (self.e == self.episodes - 1):
            self.flush_train_log()          # defer_train_outputs: the block's last training iteration, before the test
        if (self.e == self.episodes - 1) or (self.log_ind == 0):
            if self.show_plots * (self.e > 1) * (self.test_e % self.plot_every == 0):
                self.plot_model_perf()
            self.preprocess_env()      

    def run_episode(self):  
        self.run_generators()
        if (self.skip_inference is False) or self.test_set:
            self.run_inference() 
        self.prep_model()
        self.forward_backward()
        self.log_model()
        self.run_external_logger()
        
    def prep_model(self):               
        args_for_model = {
            'joint_goal_belief': self.joint_goal_belief,
            'testing': np.array([self.test_set]),
            'obs_flat': self.obs_flat,
            'goal_ind': self.goal_ind,
            'goal_value': self.goal_value,
            'ctx_inds': self.ctx_inds,
            'ctx_vals': self.ctx_vals}
        # (2026-11-13, speed) a training iteration that skips inference hands over the same, unchanged array as the
        # previous iteration (the last test's belief): the model keeps the copy it already has (same values, no transfer)
        jgb, sent = args_for_model['joint_goal_belief'], self.__dict__.get("_sent_joint_goal_belief")
        if (sent is not None and sent[0] is jgb and self.model.__dict__.get("joint_goal_belief") is sent[1]
                and not self.test_set and self.skip_inference):
            del args_for_model['joint_goal_belief']
        keys, vals = zip(*args_for_model.items())
        args_for_model = dict(zip(keys, tnp(list(vals),'torch',self.device)))
        self.model.__dict__.update(args_for_model)
        if 'joint_goal_belief' in args_for_model:
            self._sent_joint_goal_belief = (jgb, args_for_model['joint_goal_belief'])
        self.model.ctx_vals = self.model.ctx_vals.long()
        self.model.ctx_inds = self.model.ctx_inds.long()
        self.model.goal_ind = self.model.goal_ind.long()

    def uses_micro_batches(self):
        return self.mode == "SANITY" and bool(self.micro_batch) and self.micro_batch < self.batch_num

    def forward_backward(self):
        micro = self.uses_micro_batches()
        with torch.amp.autocast('cuda', enabled = self.device_type != "cpu"):
            if self.test_set:
                with torch.no_grad():
                    forward = self.model.forward_pass_micro(train = False) if micro else self.model.forward_pass()
                    self.model_belief, self.model_goal_belief, self.model_input_flat, self.model_update_flat = forward
            elif micro:                                  # slices, gradients accumulated, one step (model_forward.py)
                (self.model_belief, self.model_goal_belief, self.model_input_flat, self.model_update_flat,
                 self.classifier_loss, self.generator_loss, self.readin_grad, self.readout_grad) = self.model.forward_pass_micro(train = True)
            elif getattr(self, "defer_train_outputs", False):
                return self.forward_backward_deferred()
            else:                                        # dream_only: the real batch is for the logs only, no graph
                with torch.no_grad() if getattr(self.model, "dream_only", False) else nullcontext():
                    self.model_belief, self.model_goal_belief, self.model_input_flat, self.model_update_flat = self.model.forward_pass()
                self.classifier_loss, self.generator_loss, self.readin_grad, self.readout_grad = self.model.backward_pass()
                
        self.model_est, _, self.model_acc, self.model_TP, self.model_mse = self.get_goal_performance(self.model_belief)

    def forward_backward_deferred(self):
        """defer_train_outputs (2026-11-13, speed; opt-in): a training iteration's forward and update as in forward_backward
        (same calls, same order), but its outputs stay on the GPU: model_belief ... model_mse (the device -> host copies and
        numpy get_goal_performance) and classifier_loss ... readout_grad (backward_return's copies) are computed on first
        read (Env_helpers.set_lazy), bit for bit as before, from this iteration's tensors and goals; log_model writes the
        training slots of the logs once per block (flush_train_log), so a reader of those slots between two tests sees them
        filled only up to the previous block. Also turns off torch.distributions' argument checks of the two Categorical
        samples (model_forward; same samples), so invalid probabilities no longer raise there. Removes 10 of a training
        iteration's 18 GPU syncs and ~4 ms (batch 1,000) / ~48 ms (10,000) of numpy (get_goal_performance). The previous
        iteration's pending values (never read: this iteration replaces them anyway) are dropped first, so its GPU tensors
        are not held through this forward and backward."""
        self.drop_lazy(self._DEFERRED_OUTPUTS + self._DEFERRED_LOSSES)
        m = self.model
        with torch.no_grad() if getattr(m, "dream_only", False) else nullcontext():
            m.forward_compute()
        outs = (m.classifier_belief_flat, m.classifier_goal_belief, m.input_flat, m.update_flat)
        m.raw_return = True
        try:
            losses = m.backward_pass()
        finally:
            m.raw_return = False
        goal_ind, goal_value = self.goal_ind, self.goal_value
        self.set_lazy(self._DEFERRED_OUTPUTS, lambda env: env.deferred_outputs(outs, goal_ind, goal_value))
        self.set_lazy(self._DEFERRED_LOSSES, lambda env: dict(zip(env._DEFERRED_LOSSES, tnp(list(losses), 'np'))))

    _DEFERRED_OUTPUTS = ("model_belief", "model_goal_belief", "model_input_flat", "model_update_flat",
                         "model_est", "model_acc", "model_TP", "model_mse")
    _DEFERRED_LOSSES = ("classifier_loss", "generator_loss", "readin_grad", "readout_grad")

    def deferred_outputs(self, outs, goal_ind, goal_value):
        """forward_backward's numpy outputs of a deferred training iteration: the same copies, the same get_goal_performance"""
        b, gb, i, u = tnp(list(outs), 'np')
        est, _, acc, TP, mse = self.get_goal_performance(b, goal_ind = goal_ind, goal_value = goal_value)
        return dict(model_belief = b, model_goal_belief = gb, model_input_flat = i, model_update_flat = u,
                    model_est = est, model_acc = acc, model_TP = TP, model_mse = mse)
