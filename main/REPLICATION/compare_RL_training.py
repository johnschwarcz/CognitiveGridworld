import numpy as np; import torch; import os; import sys; import inspect; import pylab as plt; import seaborn as sns; import matplotlib as mpl; import matplotlib.gridspec as gs
from matplotlib.patches import Patch; import matplotlib.gridspec as gridspec

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
    state_nums = [100, 125, 150, 200, 250, 300, 400, 500, 1000]
    episodes = 50000
    do = "test" # ["train", "test"]
    repetitions = 1

    if do == "train":
        for r in range(0, repetitions):
            for state_num in state_nums:
                print(f"Training state num: {state_num}, repetition: {r}")
                self = CognitiveGridworld(**{'mode': "RL", 'cuda': cuda, 'episodes': episodes, 'plot_every': 10, 'checkpoint_every': episodes//10,
                'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num, 'training': True,
                'batch_num': 20000, 'step_num': step_num, 'state_num': state_num, 'save_env': f'/RL_state_num_reps/{state_num}_{r}',
                'classifier_LR': .0001, 'ctx_num': 2, 'generator_LR':.0001, 'learn_embeddings': True})
                del(self)
    if do == "test":
        tr = np.empty((len(state_nums), repetitions), dtype = object)
        te  = np.empty((len(state_nums), repetitions), dtype = object)

        for i, state_num in enumerate(state_nums):
            for r in range(repetitions):
                self = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1,
                    'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num, 'show_plots': False,
                    'batch_num': 5, 'step_num': step_num, 'state_num': int(state_num), 'learn_embeddings': True,
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

        # --- Original 2D Plots ---
        fig,ax=plt.subplots(1,3,figsize=(16, 4), constrained_layout=True)
        xax=np.asarray(state_nums)
        cmap=plt.get_cmap("coolwarm"); colors=cmap(np.linspace(0,1,n_states))

        TR=trains[:,:,-1,-1].mean(1)
        TE=tests[:,:,-1,-1].mean(1)
        ax[0].plot(xax,TR,'o', c="C0",alpha=.5,lw=5)
        ax[0].plot(xax,TE,'o', c="C1",alpha=.5,lw=5)

        ax[0].set_title("accuracy through episodes"); ax[0].set_xticks(state_nums)
        ax[0].set_xlabel("state num"); ax[0].set_ylabel("accuracy (last step)")

        alphs=np.linspace(.2,1,n_states); ms=np.linspace(10,80,n_eps)
        sizes_ax2=np.tile(np.linspace(5,50,n_eps),n_reps)

        for s in range(n_states):
            mt=trains.mean(1)[s,:,-1]; me=tests.mean(1)[s,:,-1]; t=np.arange(n_eps)
            ax[1].plot(t,mt,"-",c="C0",alpha=alphs[s],lw=1); ax[1].plot(t,me,"-",c="C1",alpha=alphs[s],lw=1)
            ax[1].scatter(t,mt,s=ms,c="C0",alpha=alphs[s]); ax[1].scatter(t,me,s=ms,c="C1",alpha=alphs[s])
            tr_mean=trains.mean(-1)[s].reshape(-1); te_mean=tests.mean(-1)[s].reshape(-1)
            ax[2].scatter(tr_mean,te_mean,c=colors[s][None,:],alpha=.5,s=sizes_ax2,edgecolors="k",linewidth=.3)

        ax[1].set_xlabel("episode"); ax[1].set_title("Learning Curve")
        ax[2].set_xlabel("train accuracy"); ax[2].set_ylabel("test accuracy"); ax[2].set_title("Train vs Test (Blue->Red = State num)")
        ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[2].set_xscale("logit"); ax[2].set_yscale("logit")
        
        # Calculate Z matrices for subsequent plots
        X, Y = np.meshgrid(np.arange(n_eps), state_nums)
        Z_train = trains.mean(1)[:, :, -1]
        Z_test = tests.mean(1)[:, :, -1]

        """ version A """
        # --- POLISHED FIGURE: REFORMATTED SPLIT PANELS ---
        fig_split = plt.figure(figsize=(24, 10))
        
        # Add overarching block titles
        fig_split.text(0.33, 0.96, "Learning curves per $\mathcal{S}$", ha='center', va='top', fontsize=20, fontweight='bold')
        fig_split.text(0.83, 0.96, "Overlaid learning curves", ha='center', va='top', fontsize=20, fontweight='bold')
        
        # 2-Column Master Grid: [Sub-panels (Left), Conglomerates (Right)]
        gs_super = gs.GridSpec(1, 2, figure=fig_split, width_ratios=[2, 1], wspace=0.02)
        
        cmap_eps = plt.get_cmap("plasma")
        colors_eps = cmap_eps(np.linspace(0, 1, n_eps))
        cmap_states =  plt.get_cmap("plasma") # plt.get_cmap("coolwarm")
        colors_states = cmap_states(np.linspace(0, 1, n_eps + 6)) # cmap_states(np.linspace(0, 1, n_eps))
        
        # Setup Palettes for Left Block: Unsaturated "cool" (desaturated cyan-to-purple)
        cmap_cool = plt.get_cmap("cool")
        colors_sub = [sns.desaturate(cmap_cool(x), 0.4) for x in np.linspace(0, 1, n_states)]
        
        # --- Left Block (Sub-panels) ---
        gs_sub = gs_super[0].subgridspec(2, n_states, wspace=0.0, hspace=0.05)
        ax_curves_sub = np.empty((2, n_states), dtype=object)
        
        x_pct = np.array([10,20,30,40,50,60,70,80,90,100])
        
        for s in range(n_states):
            if s == 0:
                ax_curves_sub[0, s] = fig_split.add_subplot(gs_sub[0, s])
                ax_curves_sub[1, s] = fig_split.add_subplot(gs_sub[1, s], sharex=ax_curves_sub[0, s])
            else:
                ax_curves_sub[0, s] = fig_split.add_subplot(gs_sub[0, s], sharey=ax_curves_sub[0, 0])
                ax_curves_sub[1, s] = fig_split.add_subplot(gs_sub[1, s], sharey=ax_curves_sub[1, 0], sharex=ax_curves_sub[0, s])
                
            line_color = colors_sub[s]
            fill_color = colors_sub[s]
            
            # Train - separated plot and scatter to map edgecolors to the state colors
            ax_curves_sub[0, s].plot(x_pct, Z_train[s, :], '-', color=line_color, alpha=1, zorder=10, lw = 2)
            # ax_curves_sub[0, s].scatter(x_pct, Z_train[s, :], c=[line_color]*n_eps, edgecolors=colors_states, linewidth=1.5, s=40, zorder=11)
            ax_curves_sub[0, s].scatter(x_pct, Z_train[s, :], c=line_color, s=50, zorder=11)
            ax_curves_sub[0, s].fill_between(x_pct, Z_train[s, :], 0.05, color=fill_color, alpha=0.15, zorder=-s)

            # Test - separated plot and scatter
            ax_curves_sub[1, s].plot(x_pct, Z_test[s, :], '-', color=line_color, alpha=1, zorder=10, lw = 2)
            # ax_curves_sub[1, s].scatter(x_pct, Z_test[s, :], c=[line_color]*n_eps, edgecolors=colors_states, linewidth=1.5, s=40, zorder=11)
            ax_curves_sub[1, s].scatter(x_pct, Z_test[s, :],  c=line_color, s=50, zorder=11)
            ax_curves_sub[1, s].fill_between(x_pct, Z_test[s, :], 0.05, color=fill_color, alpha=0.15, zorder=-s)

            # Formats Both Rows
            for row in range(2):
                ax_curves_sub[1, s].text(0.5, 0.95, f"$\mathcal{{S}} =$ {state_nums[s]}", 
                                         transform=ax_curves_sub[1, s].transAxes, 
                                         ha='center', va='top', fontsize=14, fontweight='bold', color=colors_sub[s],
                                         bbox=dict(facecolor='white', alpha=1, edgecolor='none', pad=2), zorder=20)

                ax_curves_sub[row, s].set_ylim([0.08, 0.8])
                ax_curves_sub[row, s].grid(True, linestyle='--', alpha=0.4, zorder=0)
                ax_curves_sub[row, s].spines['top'].set_visible(False)
                ax_curves_sub[row, s].spines['right'].set_visible(False)
                
                # Update all tick label fonts to 14
                ax_curves_sub[row, s].tick_params(axis='both', which='major', labelsize=14)
                
                # Turn off y-ticks for all but the first column
                if s > 0:
                    ax_curves_sub[row, s].tick_params(labelleft=False)
                
                # Clearer Y-Labels describing the sets
                if s == 0:
                    phase_str = "Training" if row == 0 else "Testing"
                    ax_curves_sub[row, s].set_ylabel(f"{phase_str} Set Accuracy", fontsize=16)
                
                if s == 4:
                    ax_curves_sub[1, s].xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=100, decimals=0))
                    ax_curves_sub[1, s].set_xticks([10,50, 100], labels = ["0%","50%","100%"])
                    ax_curves_sub[1, s].set_xlabel("Training Progress", fontsize=16)
                    ax_curves_sub[1, s].tick_params(labelbottom=True)
                ax_curves_sub[row, s].set_xlim(10, 100)

            # Remove bottom x-ticks for all columns EXCEPT the 4th
            if s != 4:
                ax_curves_sub[0, s].tick_params(labelbottom=False)
                ax_curves_sub[1, s].tick_params(labelbottom=False)
        ax_curves_sub[0, 4].tick_params(labelbottom=False)

        # --- Right Block (Conglomerates) ---
        gs_conglom = gs_super[1].subgridspec(2, 1, hspace=0.05)
        ax_conglom_top = fig_split.add_subplot(gs_conglom[0, 0])
        ax_conglom_bot = fig_split.add_subplot(gs_conglom[1, 0], sharex=ax_conglom_top)
        ax_curves_conglom = [ax_conglom_top, ax_conglom_bot]
        state_nums_fill = state_nums.copy()
        state_nums_fill[0] = state_nums[0] * .95 
        state_nums_fill[-1] = state_nums[-1] * 1.05 
        
        for i, phase in enumerate(["Training", "Testing"]):
            
            # Draw vertical lines connecting the points within the same state (S)
            for s in range(n_states):
                y_data_s = Z_train[s, :] if i==0 else Z_test[s, :]
                # ax_curves_conglom[i].plot([state_nums[s]] * n_eps, y_data_s, '-', color=colors_sub[s], lw=1, alpha=1, zorder=9)
            
            for e in range(n_eps):
                line_color = colors_states[e]
                fill_color = colors_eps[e]
                y_data = Z_train[:, e] if i==0 else Z_test[:, e]
                
                # Plot the horizontal curve alone without markers
                ax_curves_conglom[i].plot(state_nums, y_data, ':', color=line_color, lw = 2, alpha=1, zorder=10)
                
                # Complex scattering: Overlay scatter points colored dynamically by State (S)
                # ax_curves_conglom[i].scatter(state_nums, y_data, c=line_color, edgecolor=colors_sub, linewidth=2, s=50, zorder=11)
                # ax_curves_conglom[i].scatter(state_nums, y_data, c=colors_sub, edgecolor=line_color, linewidth=1, s=50, zorder=11)
                ax_curves_conglom[i].scatter(state_nums, y_data, c=line_color, s=50, zorder=11)
                # ax_curves_conglom[i].scatter(state_nums, y_data, c=colors_sub, s=50, zorder=11)
                
                # Fill between
                ax_curves_conglom[i].fill_between(state_nums, y_data, 0.01, color=fill_color, alpha=0.25, zorder=-e)
            
            ax_curves_conglom[i].set_xscale("log")
            ax_curves_conglom[i].set_xlim(state_nums[0], state_nums[-1]) 
            ax_curves_conglom[i].minorticks_off()
            ax_curves_conglom[i].set_ylim([0.08, 0.8])
            ax_curves_conglom[i].grid(True, linestyle='--', alpha=0.4, zorder=0)
            ax_curves_conglom[i].spines['top'].set_visible(False)
            ax_curves_conglom[i].spines['right'].set_visible(False)
            
            # Increase x-tick fonts to 14, hide y-ticks and labels entirely
            ax_curves_conglom[i].tick_params(axis='x', which='major', labelsize=14)
            ax_curves_conglom[i].tick_params(axis='y', left=False, labelleft=False)

        ax_curves_conglom[0].tick_params(labelbottom=False)
        ax_curves_conglom[1].set_xticks(state_nums)
        ax_curves_conglom[1].set_xticklabels([str(s) for s in state_nums], fontsize = 14)
        
        ax_curves_conglom[1].set_xlabel("$\mathcal{S}$", fontsize=16)

        # Custom Legend (Dynamically generating all steps from 100% to 0%)
        legend_elements = [
            mpl.lines.Line2D([0], [0], color=colors_states[e], ls='-', lw=2, label=f'{int(x_pct[e])}%')
            for e in reversed(range(n_eps))
        ]

        # Apply global margins
        fig_split.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.90)
        
        # Legend removed from ax_curves_conglom[0] 
        # Added full legend to ax_curves_conglom[1] with 2 columns to save space
        ax_curves_conglom[1].legend(handles=legend_elements, loc='upper left', frameon=True, ncol=1, fontsize=12, title = "Training Progress")
        plt.savefig("RL_training_comparison.svg", dpi=300)
        plt.show()

        """ VERSION B """

        # --- POLISHED FIGURE: COMBINED PANELS & 3D LANDSCAPE ---
        fig = plt.figure(figsize=(15, 15))

        # Main Title
        fig.suptitle("Learning curves per $\mathcal{S}$", fontsize=24, fontweight='bold', y=0.96)

        # Main GridSpec: 2 Rows (Top: Split Panels, Bottom: Overlaid 2D + 3D)
        # Increased height_ratio for the bottom row to give the 3D plot more vertical room
        gs_main = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.25)

        # Set up Colormaps
        cmap_eps = plt.get_cmap("plasma")
        colors_eps = cmap_eps(np.linspace(0, 1, n_eps))
        cmap_states = plt.get_cmap("plasma") 
        colors_states = cmap_states(np.linspace(0, 1, n_eps + 6)) 

        cmap_cool = plt.get_cmap("cool")
        colors_sub = [sns.desaturate(cmap_cool(x), 0.4) for x in np.linspace(0, 1, n_states)]

        x_pct = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        # ==========================================
        # TOP BLOCK: SPLIT PANELS
        # ==========================================
        gs_top = gs_main[0].subgridspec(2, n_states, wspace=0.0, hspace=0.05)
        ax_curves_sub = np.empty((2, n_states), dtype=object)

        for s in range(n_states):
            if s == 0:
                ax_curves_sub[0, s] = fig.add_subplot(gs_top[0, s])
                ax_curves_sub[1, s] = fig.add_subplot(gs_top[1, s], sharex=ax_curves_sub[0, s])
            else:
                ax_curves_sub[0, s] = fig.add_subplot(gs_top[0, s], sharey=ax_curves_sub[0, 0])
                ax_curves_sub[1, s] = fig.add_subplot(gs_top[1, s], sharey=ax_curves_sub[1, 0], sharex=ax_curves_sub[0, s])
                
            line_color = colors_sub[s]
            fill_color = colors_sub[s]
            
            # Train 
            ax_curves_sub[0, s].plot(x_pct, Z_train[s, :], '-', color=line_color, alpha=1, zorder=10, lw=2)
            ax_curves_sub[0, s].scatter(x_pct, Z_train[s, :], color=line_color, s=50, zorder=11)
            ax_curves_sub[0, s].fill_between(x_pct, Z_train[s, :], 0.05, color=fill_color, alpha=0.15, zorder=-s)

            # Test
            ax_curves_sub[1, s].plot(x_pct, Z_test[s, :], '-', color=line_color, alpha=1, zorder=10, lw=2)
            ax_curves_sub[1, s].scatter(x_pct, Z_test[s, :], color=line_color, s=50, zorder=11)
            ax_curves_sub[1, s].fill_between(x_pct, Z_test[s, :], 0.05, color=fill_color, alpha=0.15, zorder=-s)

            # Formats Both Rows
            for row in range(2):
                ax_curves_sub[1, s].text(0.5, 0.95, f"$\mathcal{{S}} =$ {state_nums[s]}", 
                                        transform=ax_curves_sub[1, s].transAxes, 
                                        ha='center', va='top', fontsize=14, fontweight='bold', color=colors_sub[s],
                                        bbox=dict(facecolor='white', alpha=1, edgecolor='none', pad=2), zorder=20)

                ax_curves_sub[row, s].set_ylim([0.08, 0.8])
                ax_curves_sub[row, s].grid(True, linestyle='--', alpha=0.4, zorder=0)
                ax_curves_sub[row, s].spines['top'].set_visible(False)
                ax_curves_sub[row, s].spines['right'].set_visible(False)
                
                ax_curves_sub[row, s].tick_params(axis='both', which='major', labelsize=14)
                
                if s > 0:
                    ax_curves_sub[row, s].tick_params(labelleft=False)
                
                if s == 0:
                    phase_str = "Training" if row == 0 else "Testing"
                    ax_curves_sub[row, s].set_ylabel(f"{phase_str} Accuracy", fontsize=16)
                
                # Format the middle X-axis for centering
                mid_idx = n_states // 2
                if s == mid_idx:
                    ax_curves_sub[1, s].xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=100, decimals=0))
                    ax_curves_sub[1, s].set_xticks([10, 50, 100], labels=["0%", "50%", "100%"])
                    ax_curves_sub[1, s].set_xlabel("Training Progress", fontsize=16)
                    ax_curves_sub[1, s].tick_params(labelbottom=True)
                ax_curves_sub[row, s].set_xlim(10, 100)

            if s != mid_idx:
                ax_curves_sub[0, s].tick_params(labelbottom=False)
                ax_curves_sub[1, s].tick_params(labelbottom=False)
        ax_curves_sub[0, mid_idx].tick_params(labelbottom=False)

        # ==========================================
        # BOTTOM BLOCK: OVERLAID 2D & 3D LANDSCAPE
        # ==========================================
        # Adjusted width_ratios to give the 3D plot significantly more horizontal width
        gs_bottom = gs_main[1].subgridspec(1, 2, width_ratios=[1, 1.3], wspace=0.)

        # --- Bottom Left (Conglomerates) ---
        gs_conglom = gs_bottom[0].subgridspec(2, 1, hspace=0.05)
        ax_conglom_top = fig.add_subplot(gs_conglom[0, 0])
        ax_conglom_bot = fig.add_subplot(gs_conglom[1, 0], sharex=ax_conglom_top)
        ax_curves_conglom = [ax_conglom_top, ax_conglom_bot]

        ax_conglom_top.set_title("Overlaid learning curves", fontsize=18, fontweight='bold', pad=15)

        for i, phase in enumerate(["Training", "Testing"]):
            for e in range(n_eps):
                line_color = colors_states[e]
                fill_color = colors_eps[e]
                y_data = Z_train[:, e] if i == 0 else Z_test[:, e]
                
                ax_curves_conglom[i].plot(state_nums, y_data, ':', color=line_color, lw=2, alpha=1, zorder=10)
                ax_curves_conglom[i].scatter(state_nums, y_data, color=line_color, s=50, zorder=11)
                ax_curves_conglom[i].fill_between(state_nums, y_data, 0.01, color=fill_color, alpha=0.25, zorder=-e)
            
            ax_curves_conglom[i].set_xscale("log")
            ax_curves_conglom[i].set_xlim(state_nums[0], state_nums[-1]) 
            ax_curves_conglom[i].minorticks_off()
            ax_curves_conglom[i].set_ylim([0.08, 0.8])
            ax_curves_conglom[i].grid(True, linestyle='--', alpha=0.4, zorder=0)
            ax_curves_conglom[i].spines['top'].set_visible(False)
            ax_curves_conglom[i].spines['right'].set_visible(False)
            
            ax_curves_conglom[i].tick_params(axis='x', which='major', labelsize=14)
            
            # Re-enabled y-ticks and y-labels for the bottom left panels
            ax_curves_conglom[i].tick_params(axis='y', which='major', left=True, labelleft=True, labelsize=14)
            phase_str = "Train" if i == 0 else "Test"
            ax_curves_conglom[i].set_ylabel(f"{phase} Accuracy", fontsize=15)

        ax_curves_conglom[0].tick_params(labelbottom=False)
        ax_curves_conglom[1].set_xticks(state_nums)
        ax_curves_conglom[1].set_xticklabels([str(s) for s in state_nums], fontsize=14)
        ax_curves_conglom[1].set_xlabel("$\mathcal{S}$", fontsize=16)

        legend_elements = [
            mpl.lines.Line2D([0], [0], color=colors_states[e], ls='-', lw=2, label=f'{int(x_pct[e])}%')
            for e in reversed(range(n_eps))
        ]
        ax_curves_conglom[1].legend(handles=legend_elements, loc='upper left', frameon=True, ncol=1, fontsize=10, title="Training Progress")

        # --- Bottom Right (3D Landscape) ---
        ax_3d = fig.add_subplot(gs_bottom[1], projection='3d')
        # Adjusted box aspect to widen the 3D plot and make it feel more expansive
        ax_3d.set_box_aspect((2 / 2, 2 / 2, 0.7 / 2))

        X_3d, Y_log_3d = np.meshgrid(x_pct, np.log10(state_nums))

        # Surfaces
        surf_train = ax_3d.plot_surface(X_3d, Y_log_3d, Z_train, color='darkgreen', 
                                        edgecolor='darkgreen', ls=':', alpha=0.5, zorder=3, antialiased=True)
        surf_test = ax_3d.plot_surface(X_3d, Y_log_3d, Z_test, color='darkred', 
                                    edgecolor='none', alpha=1, zorder=-2, antialiased=True)

    ax_3d.set_xlabel("Training Progress (%)", fontsize=16, labelpad=8)
    ax_3d.set_ylabel("$\mathcal{S}$", fontsize=16, labelpad=5) 
    ax_3d.set_zlabel("Accuracy", fontsize=16, labelpad=5)

    ax_3d.set_zlim(0.08, 0.8)

    ax_3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax_3d.view_init(elev=30, azim=-200) 
    ax_3d.set_yticks(np.log10(state_nums))
    ax_3d.set_yticklabels([str(s) for s in state_nums]) 

    ax_3d.invert_yaxis()
    ax_3d.tick_params(axis='both', which='major', labelsize=10)

    legend_elements_3d = [
        Patch(facecolor='darkgreen', alpha=0.7, label='Training Set'),
        Patch(facecolor='darkred', alpha=1.0, label='Testing Set'),
    ]
    # Shifted legend slightly to not clash with the expanded 3D plot
    ax_3d.legend(handles=legend_elements_3d, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncols=2, fontsize=16, frameon=True)

    # Layout padding and global adjustment
    # Increased left margin slightly (0.04 -> 0.05) to accommodate the new y-labels on the bottom left
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.92)

    plt.savefig("Combined_RL_training_landscape.svg", dpi=300)
    plt.show()



    # ==========================================
    # VERSION C: POLISHED FIGURE (1x2 Split Top, Split 3D Bottom)
    # ==========================================
    fig = plt.figure(figsize=(24, 20))

    gs_main = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.25)

    cmap_eps = plt.get_cmap("plasma")
    colors_eps = cmap_eps(np.linspace(0, 1, n_eps))
    cmap_states = plt.get_cmap("plasma") 
    colors_states = cmap_states(np.linspace(0, 1, n_eps + 6)) 
    cmap_cool = plt.get_cmap("cool")
    colors_sub = [sns.desaturate(cmap_cool(x), 0.4) for x in np.linspace(0, 1, n_states)]

    # --- TOP ROW: Sub-panels + Conglomerates ---
    gs_super = gs_main[0].subgridspec(1, 2, width_ratios=[2, 1], wspace=0.08)

    fig.text(0.33, 0.93, "Learning curves per $\mathcal{S}$", ha='center', va='top', fontsize=22, fontweight='bold')
    fig.text(0.83, 0.93, "Overlaid learning curves", ha='center', va='top', fontsize=22, fontweight='bold')

    # Left Block (Sub-panels)
    gs_sub = gs_super[0].subgridspec(2, n_states, wspace=0.0, hspace=0.05)
    ax_curves_sub = np.empty((2, n_states), dtype=object)

    for s in range(n_states):
        if s == 0:
            ax_curves_sub[0, s] = fig.add_subplot(gs_sub[0, s])
            ax_curves_sub[1, s] = fig.add_subplot(gs_sub[1, s], sharex=ax_curves_sub[0, s])
        else:
            ax_curves_sub[0, s] = fig.add_subplot(gs_sub[0, s], sharey=ax_curves_sub[0, 0])
            ax_curves_sub[1, s] = fig.add_subplot(gs_sub[1, s], sharey=ax_curves_sub[1, 0], sharex=ax_curves_sub[0, s])
            
        line_color = colors_sub[s]
        fill_color = colors_sub[s]
        
        # Train 
        ax_curves_sub[0, s].plot(x_pct, Z_train[s, :], '-', color=line_color, alpha=1, zorder=10, lw=2)
        ax_curves_sub[0, s].scatter(x_pct, Z_train[s, :], c=line_color, s=50, zorder=11)
        ax_curves_sub[0, s].fill_between(x_pct, Z_train[s, :], 0.05, color=fill_color, alpha=0.15, zorder=-s)

        # Test 
        ax_curves_sub[1, s].plot(x_pct, Z_test[s, :], '-', color=line_color, alpha=1, zorder=10, lw=2)
        ax_curves_sub[1, s].scatter(x_pct, Z_test[s, :], c=line_color, s=50, zorder=11)
        ax_curves_sub[1, s].fill_between(x_pct, Z_test[s, :], 0.05, color=fill_color, alpha=0.15, zorder=-s)

        for row in range(2):
            ax_curves_sub[1, s].text(0.5, 0.95, f"$\mathcal{{S}} =$ {state_nums[s]}", 
                                        transform=ax_curves_sub[1, s].transAxes, 
                                        ha='center', va='top', fontsize=14, fontweight='bold', color=colors_sub[s],
                                        bbox=dict(facecolor='white', alpha=1, edgecolor='none', pad=2), zorder=20)

            ax_curves_sub[row, s].set_ylim([0.08, 0.8])
            ax_curves_sub[row, s].grid(True, linestyle='--', alpha=0.4, zorder=0)
            ax_curves_sub[row, s].spines['top'].set_visible(False)
            ax_curves_sub[row, s].spines['right'].set_visible(False)
            ax_curves_sub[row, s].tick_params(axis='both', which='major', labelsize=14)
            
            if s > 0:
                ax_curves_sub[row, s].tick_params(labelleft=False)
            
            if s == 0:
                phase_str = "Training" if row == 0 else "Testing"
                ax_curves_sub[row, s].set_ylabel(f"{phase_str} Set Accuracy", fontsize=16)
            
            mid_idx = n_states // 2
            if s == mid_idx: 
                ax_curves_sub[1, s].xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=100, decimals=0))
                ax_curves_sub[1, s].set_xticks([x_pct[0], x_pct[len(x_pct)//2], x_pct[-1]], labels=["0%", "50%", "100%"])
                ax_curves_sub[1, s].set_xlabel("Training Progress", fontsize=16)
                ax_curves_sub[1, s].tick_params(labelbottom=True)
            
            ax_curves_sub[row, s].set_xlim(x_pct[0], x_pct[-1])

        if s != mid_idx:
            ax_curves_sub[0, s].tick_params(labelbottom=False)
            ax_curves_sub[1, s].tick_params(labelbottom=False)
    ax_curves_sub[0, mid_idx].tick_params(labelbottom=False)

    # Right Block (Conglomerates)
    gs_conglom = gs_super[1].subgridspec(2, 1, hspace=0.05)
    ax_conglom_top = fig.add_subplot(gs_conglom[0, 0])
    ax_conglom_bot = fig.add_subplot(gs_conglom[1, 0], sharex=ax_conglom_top)
    ax_curves_conglom = [ax_conglom_top, ax_conglom_bot]

    for i, phase in enumerate(["Training", "Testing"]):
        for e in range(n_eps):
            line_color = colors_states[e]
            fill_color = colors_eps[e]
            y_data = Z_train[:, e] if i==0 else Z_test[:, e]
            
            ax_curves_conglom[i].plot(state_nums, y_data, ':', color=line_color, lw=2, alpha=1, zorder=10)
            ax_curves_conglom[i].scatter(state_nums, y_data, c=line_color, s=50, zorder=11)
            ax_curves_conglom[i].fill_between(state_nums, y_data, 0.01, color=fill_color, alpha=0.25, zorder=-e)
        
        ax_curves_conglom[i].set_xscale("log")
        ax_curves_conglom[i].set_xlim(state_nums[0], state_nums[-1]) 
        ax_curves_conglom[i].minorticks_off()
        ax_curves_conglom[i].set_ylim([0.08, 0.8])
        ax_curves_conglom[i].grid(True, linestyle='--', alpha=0.4, zorder=0)
        ax_curves_conglom[i].spines['top'].set_visible(False)
        ax_curves_conglom[i].spines['right'].set_visible(False)
        
        ax_curves_conglom[i].tick_params(axis='x', which='major', labelsize=14)
        ax_curves_conglom[i].tick_params(axis='y', left=True, labelleft=True, labelsize=14)
        
        phase_str = "Train" if i == 0 else "Test"
        ax_curves_conglom[i].set_ylabel(f"{phase_str} Accuracy", fontsize=15)

    ax_curves_conglom[0].tick_params(labelbottom=False)
    ax_curves_conglom[1].set_xticks(state_nums)
    ax_curves_conglom[1].set_xticklabels([str(s) for s in state_nums], fontsize = 14)
    ax_curves_conglom[1].set_xlabel("$\mathcal{S}$", fontsize=16)

    legend_elements = [
        mpl.lines.Line2D([0], [0], color=colors_states[e], ls='-', lw=2, label=f'{int(x_pct[e])}%')
        for e in reversed(range(n_eps))
    ]
    ax_curves_conglom[1].legend(handles=legend_elements, loc='upper left', frameon=True, ncol=1, fontsize=14, title="Training Progress")

    # --- BOTTOM ROW: Split 3D Panels ---
    gs_bottom = gs_main[1].subgridspec(1, 2, wspace=0.05)
    X_3d, Y_log_3d = np.meshgrid(x_pct, np.log10(state_nums))

    # Bottom Left (3D Train)
    ax_3d_train = fig.add_subplot(gs_bottom[0, 0], projection='3d')
    ax_3d_train.set_box_aspect((1/2, 1, 1/2))
    
    # Draw translucent surface without edges
    surf_train = ax_3d_train.plot_surface(X_3d, Y_log_3d, Z_train, color='#D9E8D9', shade = False,
                                            edgecolor='none', alpha=1, zorder=1, antialiased=True)
    surf_train = ax_3d_train.plot_surface(X_3d, Y_log_3d, Z_train, color='#D9E8D9', shade = True,
                                            edgecolor='none', alpha=.25, zorder=2, antialiased=True)    
    # Overlay custom colored lines to match 2D plots
    for s in range(n_states):
        ax_3d_train.plot(X_3d[s, :], Y_log_3d[s, :], Z_train[s, :], color=colors_sub[s],  lw=2, ls = '-', zorder=3)
    for e in range(n_eps):
        ax_3d_train.plot(X_3d[:, e], Y_log_3d[:, e], Z_train[:, e], color=colors_states[e],  lw=2, ls = ':', zorder=3)

    ax_3d_train.set_title("Training Set", fontsize=20, fontweight='bold', pad=-5)
    ax_3d_train.set_xlabel("Training Progress (%)", fontsize=14, labelpad=5)
    ax_3d_train.set_ylabel("$\mathcal{S}$", fontsize=14, labelpad=5) 
    ax_3d_train.set_zlabel("Accuracy", fontsize=14, labelpad=5)

    ax_3d_train.set_zlim(0.08, 0.8)
    ax_3d_train.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3d_train.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3d_train.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax_3d_train.view_init(elev=30, azim=-210) 
    ax_3d_train.set_yticks(np.log10(state_nums))
    ax_3d_train.set_yticklabels([str(s) for s in state_nums]) 
    ax_3d_train.invert_yaxis()
    ax_3d_train.tick_params(axis='both', which='major', labelsize=10)
    # Bottom Right (3D Test)
    ax_3d_test = fig.add_subplot(gs_bottom[0, 1], projection='3d')
    ax_3d_test.set_box_aspect((1/2, 1, 1/2))
    
    # Draw translucent surface without edges
    surf_test = ax_3d_test.plot_surface(X_3d, Y_log_3d, Z_test, color='#EDD9D9', 
                                        edgecolor='none', alpha=1, zorder=1, antialiased=True, shade = False)
    surf_test = ax_3d_test.plot_surface(X_3d, Y_log_3d, Z_test, color='#EDD9D9', 
                                        edgecolor='none', alpha=.25, zorder=2, antialiased=True, shade = True)                                        
    # Overlay custom colored lines to match 2D plots
    for s in range(n_states):
        ax_3d_test.plot(X_3d[s, :], Y_log_3d[s, :], Z_test[s, :], color=colors_sub[s], lw=2, ls = '-', zorder=3)
    for e in range(n_eps):
        ax_3d_test.plot(X_3d[:, e], Y_log_3d[:, e], Z_test[:, e], color=colors_states[e], lw=2, ls=':', zorder=3)

    ax_3d_test.set_title("Testing Set", fontsize=20, fontweight='bold', pad=-5)
    ax_3d_test.set_xlabel("Training Progress (%)", fontsize=14, labelpad=5)
    ax_3d_test.set_ylabel("$\mathcal{S}$", fontsize=14, labelpad=5) 
    ax_3d_test.set_zlabel("Accuracy", fontsize=14, labelpad=5)

    ax_3d_test.set_zlim(0.08, 0.8)
    ax_3d_test.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3d_test.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_3d_test.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax_3d_test.view_init(elev=30, azim=-210) 
    ax_3d_test.set_yticks(np.log10(state_nums))
    ax_3d_test.set_yticklabels([str(s) for s in state_nums]) 
    ax_3d_test.invert_yaxis()
    ax_3d_test.tick_params(axis='both', which='major', labelsize=10)

    for ax in [ax_3d_train, ax_3d_test]:
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo["grid"].update({"linestyle": ":", "color": "lightgray", "alpha": 0.})
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92)
    plt.savefig("Version_C_RL_Landscape.svg", dpi=300)
    plt.show()


