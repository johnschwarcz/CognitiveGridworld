# %matplotlib auto
# %matplotlib inline
import numpy as np; import torch; import pylab as plt; import os; import sys; import inspect
path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld

if __name__ == "__main__":
    mode = None  # [None, "SANITY", "RL"]  
    cuda = 0

    self = CognitiveGridworld(**{
        'episodes': 1,
        'state_num': 500, 
        'batch_num': 15000, 
        'step_num': 30, 
        'obs_num': 5, 
        'ctx_num': 2, 
        'KQ_dim': 30, 
        'realization_num': 10,
        'likelihood_temp': 2,
        'show_plots': True,

        'mode': mode,
        'hid_dim': 1000,
        'classifier_LR': .0005, 
        'controller_LR': .005, 
        'generator_LR': .001,
        'learn_embeddings': True,   
        'reservoir': False,
        'save_env': None,
        'load_env': None,
        'cuda': cuda})

    JGB = self.joint_goal_belief
    NGB = self.naive_goal_belief
    Jent = (-JGB * np.log(JGB)).sum(-1)
    Nent = (-NGB * np.log(NGB)).sum(-1)
    ent_diff = Jent - Nent
    Jlogit = np.log(self.joint_TP / (1-self.joint_TP))
    Nlogit = np.log(self.naive_TP / (1-self.naive_TP))
    belief_diff = Jlogit - Nlogit

    """ PLOTTING """
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    plt.hist(Jent[:,0], bins = 55); plt.hist(Nent[:,0], alpha = .5, bins = 55); plt.show()
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    plt.hist(Jent[:,-1], bins = 55); plt.hist(Nent[:,-1], alpha = .5, bins = 55); plt.show()

    fig = plt.figure(figsize=(8, 10)); views = ((90,-90), (0, -90), (0, 0), (5, -60))
    for i, (elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(1, 4, i, projection='3d')
        for t in range(self.step_num):
            c = plt.cm.coolwarm(t / max(1, self.step_num - 1))
            ax.plot(t, Nent[:, t], Nlogit[:, t], 'o', color=c, alpha=.05, ms=3)
        ax.set_xlabel("t"); ax.set_ylabel("Nent"); ax.set_zlabel("NLLR");    ax.view_init(elev=elev, azim=azim)
        ax.set_zlim(-20,20)

        ax = fig.add_subplot(2, 4, i, projection='3d')
        for t in range(self.step_num):
            c = plt.cm.coolwarm(t / max(1, self.step_num - 1))
            ax.plot(t, Jent[:, t], Jlogit[:, t], 'o', color=c, alpha= .05, ms=3)
        ax.set_xlabel("t"); ax.set_ylabel("Jent"); ax.set_zlabel("JLLR");    ax.view_init(elev=elev, azim=azim)
        ax.set_zlim(-20,20)
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, self.step_num - 1))
    sm.set_array([]);plt.tight_layout();plt.show()

    t = np.arange(self.step_num, dtype=float)
    n = Jent.shape[0]
    Jent_m = Jent.mean(0); Jent_se = Jent.std(0, ddof=1)
    Nent_m = Nent.mean(0); Nent_se = Nent.std(0, ddof=1) 
    Jlog_m = Jlogit.mean(0); Jlog_se = Jlogit.std(0, ddof=1) 
    Nlog_m = Nlogit.mean(0); Nlog_se = Nlogit.std(0, ddof=1)

    fig, ax = plt.subplots(1, 3, figsize=(8, 3), tight_layout=True)
    ax[0].plot(t, Jent_m); ax[0].fill_between(t, Jent_m - Jent_se, Jent_m + Jent_se, alpha=.4)
    ax[0].plot(t, Nent_m); ax[0].fill_between(t, Nent_m - Nent_se, Nent_m + Nent_se, alpha=.4)
    ax[0].set_title("Entropy"); ax[0].set_xlabel("t"); ax[0].set_ylabel("entropy")
    ax[1].plot(t, Jlog_m); ax[1].fill_between(t, Jlog_m - Jlog_se, Jlog_m + Jlog_se, alpha=.4)
    ax[1].plot(t, Nlog_m); ax[1].fill_between(t, Nlog_m - Nlog_se, Nlog_m + Nlog_se, alpha=.4)
    ax[1].set_title("True Positive logit"); ax[1].set_xlabel("t"); ax[1].set_ylabel("log(TP/(1-TP))")
    ax[2].plot(t, Jlog_m)
    ax[2].plot(t, Nlog_m)
    ax[2].set_title("True Positive logit"); ax[2].set_xlabel("t"); ax[2].set_ylabel("log(TP/(1-TP))")
    plt.show()


