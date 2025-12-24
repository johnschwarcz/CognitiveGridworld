import torch; import numpy as np; import pylab as plt; from tqdm import tqdm
from main.utils import tnp; from main.env.env_control_manager import Env_control_manager
from main.model.model_architecture import Model_architecture as Model

class Env_model_data_manager(Env_control_manager):

    def prep_data_manager(self):
        self.test_e = 0
        s = (self.episodes, self.batch_num, self.Z_num * self.obs_num)
        self.test_true_Z, self.test_model_Z = [np.zeros(s) for _ in range(2)]
        self.test_accs, self.train_accs \
            = [np.zeros((self.episodes, self.step_num)) for _ in range(2)]
        self.test_model_SII, self.classifier_loss_log, self.generator_loss_log \
            = [np.zeros(self.episodes) for _ in range(3)]
     
    def log_model_perf(self):
        avg_model_acc = self.model_acc.mean(0)
        self.model_perf_log[self.log_ind, :, 0] = avg_model_acc
        self.model_perf_log[self.log_ind, :, 1] = self.model_TP.mean(0)
        self.model_perf_log[self.log_ind, :, 2] = self.model_mse.mean(0)
        self.generator_loss_log[self.e] = self.model.generator_loss
        self.classifier_loss_log[self.e] = self.model.classifier_loss

        if self.test_set:
            self.test_model_SII[self.test_e] = self.get_SII()
            self.test_model_Z[self.test_e, :] = tnp(self.model.active_Z, 'np').reshape(self.batch_num, -1)
            self.test_true_Z[self.test_e, :] = self.joint_Z.reshape(self.batch_num, -1) 
            self.test_accs[self.test_e, :] = avg_model_acc
            self.test_e += 1
        else:
            self.train_accs[self.test_e:, :] = avg_model_acc         # overwrite until test

    def get_SII(self, eps = 1e-8):
        J = np.clip(self.joint_goal_belief, a_min= eps, a_max = float(1) - eps)
        N = np.clip(self.naive_goal_belief, a_min= eps, a_max = float(1) - eps)
        M = np.clip(self.model_goal_belief, a_min= eps, a_max = float(1) - eps)
        JN_DKL = (J * np.log(J/N)).sum(-1).mean()
        JM_DKL = (J * np.log(J/M)).sum(-1).mean()
        return 1 - JM_DKL/JN_DKL

    def save(self):
        save_path = self.DATA_path + self.save_env + "_net.pth"
        save_dict = {                         
                "model_SII_through_training": self.test_model_SII[:self.test_e], 
                "model_Z_through_training":   self.test_model_Z[:self.test_e], 
                "true_Z_through_training":    self.test_true_Z[:self.test_e], 
                "test_acc_through_training":  self.test_accs[:self.test_e], 
                "train_acc_through_training": self.train_accs[:self.test_e], 
                "gen_loss_through_training":  self.generator_loss_log, 
                "pol_loss_through_training":  self.classifier_loss_log, 
                "K": self.all_K, "Q": self.all_Q,
                "model_K": tnp(self.model.all_K,'np'), 
                "model_Q": tnp(self.model.all_Q,'np'),
                "weights": self.model.state_dict()}
        try:
            torch.save(save_dict, save_path)
            print("ENV SAVED")
        except:
            print("SAVING FAILED")
            torch.save(save_dict, save_path)

    def load(self):
        try:
            load_dict = torch.load(self.DATA_path + self.load_env + "_net.pth", weights_only=False)
            self.all_K = load_dict["K"]
            self.all_Q = load_dict["Q"]
            self.model.load_state_dict(load_dict["weights"])
            self.model.all_K = torch.nn.Parameter(tnp(load_dict["model_K"], 'torch', self.device))
            self.model.all_Q = torch.nn.Parameter(tnp(load_dict["model_Q"], 'torch', self.device))
            self.true_Z_through_training = load_dict["true_Z_through_training"]
            self.model_Z_through_training = load_dict["model_Z_through_training"]
            self.model_SII_through_training = load_dict["model_SII_through_training"]
            self.train_acc_through_training = load_dict["train_acc_through_training"]
            self.test_acc_through_training = load_dict["test_acc_through_training"]
            self.gen_loss_through_training = load_dict["gen_loss_through_training"]
            self.pol_loss_through_training = load_dict["pol_loss_through_training"]
            print("ENV LOADED")
        except:
            print("LOADING FAILED")
            self.model.load_state_dict(load_dict["weights"])