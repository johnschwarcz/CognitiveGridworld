import numpy as np; import torch; import os; import sys; import inspect
import pylab as plt; from matplotlib.colors import PowerNorm
from tqdm import tqdm
path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld
from main.REPLICATE_RESULTS.collection_plotters import Collection_Plotters

class Sanity_Collector(Collection_Plotters):
    def __init__(self, cuda, state_num, batch_num, realization_num, step_num, ctx_num, obs_num):
        self.realization_num = realization_num
        self.state_num = state_num
        self.batch_num = batch_num
        self.step_num = step_num
        self.ctx_num = ctx_num
        self.obs_num = obs_num
        self.cuda = cuda

    def zeros(self, dims, num, astype = None):
        if astype is None:
            return [np.zeros(dims) for _ in range(num)]
        return [np.zeros(dims, dtype = astype) for _ in range(num)]

    def DKL(self, x, y, eps = 1e-15):
        x = np.clip(x, a_min = eps, a_max = float(1) - eps)
        y = np.clip(y, a_min = eps, a_max = float(1) - eps)
        return (x * np.log(x/y)).sum(-1).mean(0)

    def collect_bayes(self, rep = 1):
        self.collect(rep = rep)

    def collect_net(self, mode, rep = 1, bayes_on_cond = 0):
        self.collect(WITH_net = True, mode = mode, rep = rep, bayes_on_cond = bayes_on_cond)
        
    def collect(self, WITH_net = False, mode = None, rep = 1, bayes_on_cond = 0):
        #######################################
        """ COLLECT DATA """
        #######################################        
        self.WITH_net = WITH_net
        self.pairs = 1 + WITH_net # if net, collect reservoir and fully trained 
        self.WITHOUT_net = not WITH_net
        self.linestyles = ['-', '--'] if WITH_net else ['-']
        self.labels = ["joint inference", "independent inference"] 
        net_envs = ["/sanity/fully_trained","/sanity/reservoir"]
        if WITH_net:
            self.labels += net_envs

        self.accs, self.goal_TP = self.zeros((self.pairs, 2, self.ctx_num, self.batch_num, self.step_num), 2)
        self.goal_value, self.goal_ind = self.zeros((self.pairs, 2, self.ctx_num, self.batch_num), 2, astype = int)
        self.belief = np.zeros((self.pairs, 2, self.ctx_num, self.batch_num, self.step_num, self.realization_num)) # [0- Bayes, 1- Net] | [0- joint/trained, 1- naive/reservoir]
        self.net_joint_DKL, self.net_naive_DKL, self.joint_net_DKL, self.naive_net_DKL =  self.zeros((2, self.ctx_num, self.step_num), 4) # [0- Trained, 1- Reservoir]
        self.joint_naive_DKL, self.naive_joint_DKL = self.zeros((self.ctx_num, self.step_num), 2)

        for ctx in tqdm(range(self.ctx_num)):
            for cond in range(self.pairs):
                if WITH_net:
                    load = net_envs[cond] + f"_ctx_{ctx+1}"
                    print(f"loading {load}")
                else:
                    load = False

                for r in range(rep):
                    agent = CognitiveGridworld(**{'mode': mode, 'cuda': self.cuda, 'ctx_num':  ctx + 1, 'state_num': self.state_num,
                        'load_env': load, 'training': False, 'batch_num': self.batch_num, 'obs_num': self.obs_num, 'show_plots' : False,
                        'step_num': self.step_num, 'episodes': 1, 'realization_num': self.realization_num, 'learn_embeddings': False})    

                    # collect bayes  
                    bayes_i = 0       
                    joint_belief = agent.joint_goal_belief.copy()
                    naive_belief = agent.naive_goal_belief.copy()
                    if cond == bayes_on_cond:   
                        joint_i = 0
                        self.belief[bayes_i,joint_i,ctx] = joint_belief
                        self.goal_TP[bayes_i, joint_i, ctx] = agent.joint_TP
                        self.goal_ind[bayes_i,joint_i,ctx] = agent.goal_ind.astype(int)
                        self.goal_value[bayes_i,joint_i,ctx] = agent.goal_value.astype(int)

                        naive_i = 1
                        self.belief[bayes_i,naive_i,ctx] = naive_belief
                        self.goal_TP[bayes_i, naive_i, ctx] = agent.naive_TP
                        self.goal_ind[bayes_i,naive_i,ctx] = agent.goal_ind.astype(int)
                        self.goal_value[bayes_i,naive_i,ctx] = agent.goal_value.astype(int)

                        self.accs[bayes_i,joint_i,ctx] += agent.joint_acc / rep
                        self.accs[bayes_i,naive_i,ctx] += agent.naive_acc / rep  
                        self.joint_naive_DKL[ctx] += self.DKL(joint_belief, naive_belief) / rep
                        self.naive_joint_DKL[ctx] += self.DKL(naive_belief, joint_belief) / rep

                    if ctx == 1:
                        self.expert_likelihood = agent.joint_likelihood.copy()
                        self.naive_likelihood = agent.naive_likelihood.copy()

                    # collect network
                    net_i = 1
                    if self.WITH_net: 
                        net_belief = agent.model.classifier_goal_belief.cpu().numpy()
                        self.goal_TP[net_i, cond, ctx] = agent.model_TP
                        self.belief[net_i,cond,ctx] = net_belief
                        self.goal_ind[net_i,cond,ctx] = agent.goal_ind.astype(int)
                        self.goal_value[net_i,cond,ctx] = agent.goal_value.astype(int)

                        self.accs[net_i,cond,ctx] += agent.model_acc / rep
                        self.joint_net_DKL[cond,ctx] += self.DKL(joint_belief, net_belief) / rep
                        self.naive_net_DKL[cond,ctx] += self.DKL(naive_belief, net_belief) / rep
                        self.net_joint_DKL[cond,ctx] += self.DKL(net_belief, joint_belief) / rep
                        self.net_naive_DKL[cond,ctx] += self.DKL(net_belief, naive_belief) / rep

                    del(agent)