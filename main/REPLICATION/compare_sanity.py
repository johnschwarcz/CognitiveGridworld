import numpy as np; import torch; import os; import sys; import inspect
import pylab as plt; from matplotlib.colors import PowerNorm
path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.REPLICATION.Collector import Sanity_Collector
from main.REPLICATION.collection_plotters import Collection_Plotters

if __name__ == "__main__":
    cuda = 0
    obs_num = 5 
    state_num = 500  
    realization_num = 10 

    batch_num = 10000
    step_num = 100
    ctx_num = 4
    rep = 50
    joint_v_naive = Sanity_Collector(cuda, state_num, batch_num, realization_num, step_num, ctx_num, obs_num)
    joint_v_naive.collect_bayes(rep = rep)
    joint_v_naive.plot_likelihood()
    joint_v_naive.plot_belief()
    joint_v_naive.plot_perf()

    batch_num = 10000
    step_num = 30
    ctx_num = 2
    rep = 10
    train_v_random = Sanity_Collector(cuda, state_num, batch_num, realization_num, step_num, ctx_num, obs_num)
    train_v_random.collect_net(mode="SANITY", rep = rep, hid_dims = [1000, 1000])
    train_v_random.plot_likelihood()
    train_v_random.plot_belief()
    train_v_random.plot_perf()
    train_v_random.plot_density_curves(bins=100, ymax = .07, sigma = 3)
    
    batch_num = 1000
    step_num = 30
    ctx_num = 3
    rep = 100
    train_v_random = Sanity_Collector(cuda, state_num, batch_num, realization_num, step_num, ctx_num, obs_num)
    train_v_random.collect_net(mode="SANITY", rep = rep, hid_dims = [1000, 1000, 5000])
    train_v_random.plot_likelihood()
    train_v_random.plot_belief()
    train_v_random.plot_perf()
    train_v_random.plot_density_curves(bins=100, ymax = .07, sigma = 3)