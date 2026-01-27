import numpy as np; import torch; import os; import sys; import inspect; import pylab as plt
path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld 

if __name__ == "__main__":
    cuda = 1
    realization_num = 10
    step_num = 30
    hid_dim = 1000
    obs_num = 5
    # state_nums = [100, 125, 150, 175, 200, 250, 300, 400, 600, 1000]
    state_nums = [100, 150, 200, 250, 500, 1000]
    episodes = 125000
    do = "test" # ["train", "test"]
    repetitions = 5

    if do == "train":
        for r in range(repetitions):
            for state_num in state_nums:
                print(f"Training state num: {state_num}, repetition: {r}")
                self = CognitiveGridworld(**{'mode': "RL", 'cuda': cuda, 'episodes': episodes, 'plot_every': 5, 'checkpoint_every': 125000,
                'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num, 'training': True,
                'batch_num': 8000, 'step_num': step_num, 'state_num': state_num, 'save_env': f'/RL_state_num_reps/{state_num}_{r}',
                'classifier_LR': .0005, 'ctx_num': 2, 'generator_LR':.0005, 'learn_embeddings': True})

    if do == "test":
        tr = np.empty((len(state_nums), repetitions), dtype = object)
        te  = np.empty((len(state_nums), repetitions), dtype = object)

        for i, state_num in enumerate(state_nums):
            for r in range(repetitions):
                self = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1,
                    'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num, 'show_plots': False,
                    'batch_num': 50, 'step_num': step_num, 'state_num': int(state_num), 'learn_embeddings': True,
                    'ctx_num': 2, 'load_env': f'/RL_state_num_reps/{int(state_num)}_{r}', 'training': False})

                tr[i, r] = self.train_acc_through_training
                te[i, r]  = self.test_acc_through_training

        trains = np.array([[tr[i,r] for r in range(repetitions)] for i in range(len(state_nums))]) # shape: state_nums x repetitions x episodes x steps
        tests = np.array([[te[i,r] for r in range(repetitions)] for i in range(len(state_nums))])
        trains = trains[:,:, 1:]
        tests = tests[:,:, 1:]
        eps = trains.shape[2] 

        n_tr = trains.sum(1)
        n_te = tests.sum(1)
        mu_tr = trains.mean(1)
        mu_te = tests.mean(1)
        se_tr =trains.std(1) / np.sqrt(n_tr)
        se_te = tests.std(1) / np.sqrt(n_te)


        # PLOTTING  
        n_states=len(state_nums); n_reps=repetitions; n_eps=eps; n_steps=30
        trains=trains.reshape(n_states,n_reps,n_eps,n_steps); tests=tests.reshape(n_states,n_reps,n_eps,n_steps)

        fig,ax=plt.subplots(1,3,figsize=(20,5),tight_layout=True)
        xax=np.asarray(state_nums)
        cmap=plt.get_cmap("coolwarm"); colors=cmap(np.linspace(0,1,n_states))

        for e in (0,n_eps-1):
            TR=trains[:,:,e,-1].mean(1); TE=tests[:,:,e,-1].mean(1)
            ax[0].plot(xax,TR,c="C0",alpha=.5,lw=5); ax[0].plot(xax,TE,c="C1",alpha=.5,lw=5)
            y=trains[:,:,e,-1].std(1,ddof=1)/np.sqrt(n_reps) if n_reps>1 else np.zeros_like(TR)
            ax[0].fill_between(xax,TR-y,TR+y,color="C0",alpha=.3)
            y=tests[:,:,e,-1].std(1,ddof=1)/np.sqrt(n_reps) if n_reps>1 else np.zeros_like(TE)
            ax[0].fill_between(xax,TE-y,TE+y,color="C1",alpha=.3)

        ax[0].set_title("accuracy through episodes"); ax[0].set_xticks(state_nums)
        ax[0].set_xlabel("state num"); ax[0].set_ylabel("accuracy (last step)")

        alphs=np.linspace(.2,1,n_states); ms=np.linspace(10,80,n_eps)
        sizes_ax2=np.tile(np.linspace(5,50,n_eps),n_reps)

        for s in range(n_states):
            mt=trains.mean(1)[s,:,-1]; me=tests.mean(1)[s,:,-1]; t=np.arange(n_eps)
            ax[1].plot(t,mt,"-",c="C0",alpha=alphs[s],lw=1); ax[1].plot(t,me,"-",c="C1",alpha=alphs[s],lw=1)
            ax[1].scatter(t,mt,s=ms,c="C0",alpha=alphs[s]); ax[1].scatter(t,me,s=ms,c="C1",alpha=alphs[s])
            tr=trains.mean(-1)[s].reshape(-1); te=tests.mean(-1)[s].reshape(-1)
            ax[2].scatter(tr,te,c=colors[s][None,:],alpha=.5,s=sizes_ax2,edgecolors="k",linewidth=.3)

        ax[1].set_xlabel("episode"); ax[1].set_title("Learning Curve")
        ax[2].set_xlabel("train accuracy"); ax[2].set_ylabel("test accuracy"); ax[2].set_title("Train vs Test (Blue->Red = State num)")
        ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[2].set_xscale("logit"); ax[2].set_yscale("logit")
        plt.show()
