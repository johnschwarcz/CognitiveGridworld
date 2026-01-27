import torch; import numpy as np; import pylab as plt; from tqdm import tqdm;
from sklearn.decomposition import PCA; from sklearn.linear_model import LogisticRegression
from main.utils import tnp; from main.env.env_control_manager import Env_control_manager
from main.model.model_architecture import Model_architecture as Model

class Env_model_data_manager(Env_control_manager):

    def prep_data_manager(self):
        self.test_e = 0

        self.test_model_update_dim, self.test_model_input_dim = [np.zeros((self.episodes, self.hid_dim)) for _ in range(2)]
        self.test_model_update_dim_per_step, self.test_model_input_dim_per_step = [np.zeros((self.episodes, self.hid_dim, self.step_num)) for _ in range(2)]
        self.classifier_loss_log, self.generator_loss_log, self.test_SII_score, self.test_SII_coef, self.readin_grad_log, self.readout_grad_log, \
                                                                                                       = [np.zeros(self.episodes) for _ in range(6)]
        self.test_net_joint_DKL, self.test_net_naive_DKL, self.test_accs, self.train_accs, self.test_TPs, self.train_TPs, self.test_mses, self.train_mses \
                                                                                            = [np.zeros((self.episodes, self.step_num)) for _ in range(8)]

    def log_model(self):
        if self.test_set:
            self.test_accs[self.test_e, :] = self.model_acc.mean(0)
            self.test_TPs[self.test_e, :] = self.model_TP.mean(0)
            self.test_mses[self.test_e, :] = self.model_mse.mean(0)
            if self.training and (self.mode == "SANITY"):
                self.perform_sanity_training_analyses()
            self.test_e += 1
        else:
            # Overwrites until test
            self.classifier_loss_log[self.test_e] = self.classifier_loss
            self.generator_loss_log[self.test_e] = self.generator_loss
            self.readout_grad_log[self.test_e] = self.readout_grad
            self.readin_grad_log[self.test_e] = self.readin_grad
            self.train_accs[self.test_e, :] = self.model_acc.mean(0)
            self.train_TPs[self.test_e, :] = self.model_TP.mean(0)
            self.train_mses[self.test_e, :] = self.model_mse.mean(0)
        
    def perform_sanity_training_analyses(self):
            self.test_net_joint_DKL[self.test_e, :] = self.DKL(self.model_goal_belief, self.joint_goal_belief)
            self.test_net_naive_DKL[self.test_e, :] = self.DKL(self.model_goal_belief, self.naive_goal_belief)

            # Get SII prediction of accuracy
            SII = self.DKL(self.joint_goal_belief, self.naive_goal_belief, avg_over_batch=False, sym = True)
            X = SII[:, -1].reshape(-1, 1)
            Y = self.model_acc[:, -1]

            reg = LogisticRegression().fit(X, Y) # predict final step acc from final step SII
            self.test_SII_score[self.test_e] = reg.score(X, Y)
            self.test_SII_coef[self.test_e] = reg.coef_[0][0]

            # Get total dimensionality of LSTM input and output 
            update_pca = PCA(n_components = self.hid_dim).fit(self.model_update_flat.reshape(-1, self.hid_dim))
            input_pca = PCA(n_components = self.hid_dim).fit(self.model_input_flat.reshape(-1, self.hid_dim))
            self.test_model_update_dim[self.test_e] = update_pca.explained_variance_
            self.test_model_input_dim[self.test_e] = input_pca.explained_variance_

            # Get per step dimensionality of LSTM input and output 
            for t in self.step_range:
                update_pca = PCA(n_components = self.hid_dim).fit(self.model_update_flat[:, t])
                input_pca = PCA(n_components = self.hid_dim).fit(self.model_input_flat[:, t])
                self.test_model_update_dim_per_step[self.test_e, :, t] = update_pca.explained_variance_
                self.test_model_input_dim_per_step[self.test_e, :, t] = input_pca.explained_variance_

    def save(self):
        save_path = self.DATA_path + self.save_env + "_net.pth"
        save_dict = {                         
                "readin_grad_log_through_training": self.readin_grad_log[:self.test_e],
                "readout_grad_log_through_training": self.readout_grad_log[:self.test_e],
                "test_model_input_dim_through_training": self.test_model_input_dim[:self.test_e], 
                "test_model_update_dim_through_training": self.test_model_update_dim[:self.test_e], 
                "test_model_update_dim_per_step_through_training": self.test_model_update_dim_per_step[:self.test_e], 
                "test_model_input_dim_per_step_through_training": self.test_model_input_dim_per_step[:self.test_e],
                "test_net_joint_DKL_through_training": self.test_net_joint_DKL[:self.test_e], 
                "test_net_naive_DKL_through_training": self.test_net_naive_DKL[:self.test_e], 
                "test_SII_score_through_training": self.test_SII_score[:self.test_e],
                "test_SII_coef_through_training": self.test_SII_coef[:self.test_e],
                "test_acc_through_training":  self.test_accs[:self.test_e], 
                "train_acc_through_training": self.train_accs[:self.test_e], 
                "gen_loss_through_training":  self.generator_loss_log[:self.test_e], 
                "pol_loss_through_training":  self.classifier_loss_log[:self.test_e], 

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

            self.readin_grad_log_through_training = self.skippable_load(load_dict, "readin_grad_log_through_training")
            self.readout_grad_log_through_training = self.skippable_load(load_dict, "readout_grad_log_through_training")
            self.test_model_input_dim_through_training = self.skippable_load(load_dict, "test_model_input_dim_through_training")
            self.test_model_update_dim_through_training = self.skippable_load(load_dict, "test_model_update_dim_through_training")
            self.test_model_input_dim_per_step_through_training = self.skippable_load(load_dict, "test_model_input_dim_per_step_through_training")
            self.test_model_update_dim_per_step_through_training = self.skippable_load(load_dict, "test_model_update_dim_per_step_through_training")
            self.test_SII_coef_through_training = self.skippable_load(load_dict, "test_SII_coef_through_training")
            self.test_SII_score_through_training = self.skippable_load(load_dict, "test_SII_score_through_training")
            self.test_net_joint_DKL_through_training = self.skippable_load(load_dict, "test_net_joint_DKL_through_training")
            self.test_net_naive_DKL_through_training = self.skippable_load(load_dict, "test_net_naive_DKL_through_training")
            self.train_acc_through_training = self.skippable_load(load_dict, "train_acc_through_training")
            self.test_acc_through_training = self.skippable_load(load_dict, "test_acc_through_training")
            self.gen_loss_through_training = self.skippable_load(load_dict, "gen_loss_through_training")
            self.pol_loss_through_training = self.skippable_load(load_dict, "pol_loss_through_training")

            print("ENV LOADED")
        except:
            print("LOADING FAILED")
            self.model.load_state_dict(load_dict["weights"])

    def skippable_load(self, load_dict, field):
        if field in load_dict:
            return load_dict[field]
        else:
            if self.load_warnings:
                print(f"{field} not found.")
            return None