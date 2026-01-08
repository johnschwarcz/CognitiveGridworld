import pylab as plt; import numpy as np
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from .plotting_anim import Plotting_anim
        
class Plotters(Plotting_anim):
    
    def plot_likelihood(self, batch=None, h = 3, w = 10, obs = None, naive = True):
        """ Plot likelihoods for a given batch """
        self.sample_obs(obs)
        self.sample_batch(batch)  
        self.rows = int(self.ctx_num * (self.ctx_num - 1) / 2)
        self.plot_likelihood_helper(h, w, self.joint_likelihood, title = "joint")
        if naive:
            self.plot_likelihood_helper(h, w, self.naive_likelihood, title = "naive")
        
    def plot_trial(self, h = 7, w = 10, batch=None):
        """ Plot observations & belief with ground truth """
        fig, ax = plt.subplots(self.ctx_num + 1, 2, figsize=(w, h),  gridspec_kw=dict(width_ratios=[30, 1]))
        self.sample_batch(batch)
        self.plot_trial_helper(ax)
        self.plot_trial_postprocess(fig, ax)
        
    def plot_bayes_perf(self, h = 5, w = 12, samples = 200, y1 = 1, y2 = None, y3 = None):
        """ Plot P(correct goal realization) for joint and naive inference """
        fig, ax = plt.subplots(1, 3, figsize=(w, h))
        self.plot_agent_perf(ax, samples)
        self.plot_performance_postprocess(fig, ax)

    def plot_model_perf(self):
        fig, ax = plt.subplots(1,3, figsize = (12,4), tight_layout = True)
        bayes_perfs = [self.agent_accs, self.agent_TPs, self.agent_mses]
        model_train_perfs = [self.train_accs, self.train_TPs, self.train_mses]
        model_test_perfs = [self.test_accs, self.test_TPs, self.test_mses]
        titles = ["acc", "pgoal", "mse"]

        for p, title in enumerate(titles):
            bayes_perf = bayes_perfs[p]
            ax[p].plot(model_train_perfs[p][self.test_e-1], c = 'C0', linewidth = 5, label = "model train")
            ax[p].plot(bayes_perf[0].mean(0), c = 'C1', linewidth = 5, label = "joint")
            ax[p].plot(bayes_perf[1].mean(0), c = 'C2', linewidth = 5, label = "naive")
            ax[p].plot(model_test_perfs[p][self.test_e-1], c = 'r', linewidth = 5, label = "model test")       
            ax[p].set_title(title)
            ax[p].legend()
        plt.show()      
        
        fig, ax = plt.subplots(1, 3, figsize = (12,4))
        ax[0].plot(self.generator_loss_log[1:self.test_e])
        ax[1].plot(self.classifier_loss_log[1:self.test_e])
        ax[2].plot(self.test_accs[1:self.test_e, -1])
        plt.show()      