import numpy as np; import torch; import os; import sys; import inspect; import pylab as plt; import seaborn as sns; import matplotlib as mpl; import matplotlib.gridspec as gs
from matplotlib.patches import Patch; import matplotlib.gridspec as gridspec
import re; from collections import defaultdict

path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld 
from main.utils import fig_path

if __name__ == "__main__":
    cuda = 1
    realization_num = 10
    step_num = 30
    hid_dim = 1000
    obs_num = 5
    episodes = 50000
    do = "test" # ["train", "test"]

    # --- Dynamic Discovery of state_nums and varying repetitions ---
    # Updated to include the 'DATA' subfolder as shown in the file explorer
    possible_dirs = [
        os.path.join(path, 'main', 'DATA', 'RL_state_num_reps'),
        os.path.join(path, 'DATA', 'RL_state_num_reps'),
        path + '/RL_state_num_reps',
        os.path.join(path, 'main', 'RL_state_num_reps'),
        './RL_state_num_reps',
        '/RL_state_num_reps'
    ]
    
    env_dir = None
    for d in possible_dirs:
        if os.path.exists(d):
            env_dir = d
            break
            
    state_reps_map = defaultdict(list)
    if env_dir is not None:
        for fname in os.listdir(env_dir):
            # Accounts for the '_net.pth' suffix in the file names
            match = re.search(r'^(\d+)_(\d+)', fname)
            if match:
                s_num = int(match.group(1))
                r_num = int(match.group(2))
                if r_num not in state_reps_map[s_num]:
                    state_reps_map[s_num].append(r_num)
                    
    state_nums = sorted(list(state_reps_map.keys()))
    for s in state_nums:
        state_reps_map[s].sort()

    if do == "train":
        # Fallback values for explicitly generating new training subsets
        train_state_nums = [1000, 500, 400, 300, 250, 200, 150, 125, 100]
        train_repetitions = 1
        for r in range(0, train_repetitions):
            for state_num in train_state_nums:
                print(f"Training state num: {state_num}, repetition: {r}")
                self = CognitiveGridworld(**{'mode': "RL", 'cuda': cuda, 'episodes': episodes, 'plot_every': 10, 'checkpoint_every': episodes//10,
                'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num, 'training': True,
                'batch_num': 20000, 'step_num': step_num, 'state_num': state_num, 'save_env': f'/RL_state_num_reps/{state_num}_{r}',
                'classifier_LR': .0001, 'ctx_num': 2, 'generator_LR':.0001, 'learn_embeddings': True})
                del(self)

    if do == "test":
        with torch.no_grad():
            if not state_nums:
                # Replaced sys.exit(1) with an Exception to avoid crashing the IPython kernel
                raise FileNotFoundError(f"No saved environments found. Checked the following directories: {possible_dirs}")

            max_reps = max(len(reps) for reps in state_reps_map.values())
            tr_list = []
            te_list = []

            for i, state_num in enumerate(state_nums):
                s_tr = []
                s_te = []
                for r in state_reps_map[state_num]:
                    print(f"Testing state num: {state_num}, repetition: {r}")
                    self = CognitiveGridworld(**{'mode': "SANITY", 'cuda': cuda, 'episodes': 1,
                        'realization_num': realization_num,  'hid_dim': hid_dim,  'obs_num': obs_num, 'show_plots': False,
                        'batch_num': 5, 'step_num': step_num, 'state_num': int(state_num), 'learn_embeddings': True,
                        'ctx_num': 2, 'load_env': f'/RL_state_num_reps/{int(state_num)}_{r}', 'training': False})

                    s_tr.append(self.train_acc_through_training)
                    s_te.append(self.test_acc_through_training)
                    del(self)
                tr_list.append(s_tr)
                te_list.append(s_te)

        # Get episode/step dimensions dynamically from the first valid record
        eps_shape = tr_list[0][0].shape 
        
        # Create NaN-padded arrays to handle variable repetitions gracefully
        trains = np.full((len(state_nums), max_reps, eps_shape[0], eps_shape[1]), np.nan)
        tests = np.full((len(state_nums), max_reps, eps_shape[0], eps_shape[1]), np.nan)

        for i in range(len(state_nums)):
            for j, val in enumerate(tr_list[i]):
                trains[i, j] = val
                tests[i, j] = te_list[i][j]

        trains = trains[:,:, 1:]
        tests = tests[:,:, 1:]
        eps = trains.shape[2] 

        # Valid counts per state_num to adjust standard error for varied repetitions
        n_tr = np.sum(~np.isnan(trains[:, :, 0, 0]), axis=1)
        n_te = np.sum(~np.isnan(tests[:, :, 0, 0]), axis=1)
        
        with np.errstate(invalid='ignore', divide='ignore'):
            mu_tr = np.nanmean(trains, axis=1)
            mu_te = np.nanmean(tests, axis=1)
            se_tr = np.nanstd(trains, axis=1) / np.sqrt(n_tr)[:, None, None]
            se_te = np.nanstd(tests, axis=1) / np.sqrt(n_te)[:, None, None]

        # PLOTTING  
        n_states = len(state_nums)
        n_reps = max_reps
        n_eps = eps
        n_steps = trains.shape[3]
        
        # Arrays are already correctly shaped, ensuring dimensional safety mapping to code
        trains = trains.reshape(n_states, n_reps, n_eps, n_steps)
        tests = tests.reshape(n_states, n_reps, n_eps, n_steps)

        # --- Original 2D Plots ---
        fig,ax=plt.subplots(1,3,figsize=(16, 4), constrained_layout=True)
        xax=np.asarray(state_nums)
        cmap=plt.get_cmap("coolwarm"); colors=cmap(np.linspace(0,1,n_states))

        # Replaced standard .mean() with np.nanmean() over axis 1 (reps)
        TR = np.nanmean(trains[:, :, -1, -1], axis=1)
        TE = np.nanmean(tests[:, :, -1, -1], axis=1)
        ax[0].plot(xax,TR,'o', c="C0",alpha=.5,lw=5)
        ax[0].plot(xax,TE,'o', c="C1",alpha=.5,lw=5)

        ax[0].set_title("accuracy through episodes"); ax[0].set_xticks(state_nums)
        ax[0].set_xlabel("state num"); ax[0].set_ylabel("accuracy (last step)")

        alphs=np.linspace(.2,1,n_states); ms=np.linspace(10,80,n_eps)
        sizes_ax2=np.tile(np.linspace(5,50,n_eps),n_reps)

        for s in range(n_states):
            mt = np.nanmean(trains, axis=1)[s,:,-1]
            me = np.nanmean(tests, axis=1)[s,:,-1]
            t = np.arange(n_eps)
            ax[1].plot(t,mt,"-",c="C0",alpha=alphs[s],lw=1); ax[1].plot(t,me,"-",c="C1",alpha=alphs[s],lw=1)
            ax[1].scatter(t,mt,s=ms,c="C0",alpha=alphs[s]); ax[1].scatter(t,me,s=ms,c="C1",alpha=alphs[s])
            
            # Flatten handles the padded NaNs perfectly; scatter naturally ignores missing points
            tr_mean = np.nanmean(trains, axis=-1)[s].flatten()
            te_mean = np.nanmean(tests, axis=-1)[s].flatten()
            ax[2].scatter(tr_mean,te_mean,c=colors[s][None,:],alpha=.5,s=sizes_ax2,edgecolors="k",linewidth=.3)

        ax[1].set_xlabel("episode"); ax[1].set_title("Learning Curve")
        ax[2].set_xlabel("train accuracy"); ax[2].set_ylabel("test accuracy"); ax[2].set_title("Train vs Test (Blue->Red = State num)")
        ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[2].set_xscale("logit"); ax[2].set_yscale("logit")
        
        # Calculate Z matrices for subsequent plots using nanmean
        X, Y = np.meshgrid(np.arange(n_eps), state_nums)
        Z_train = np.nanmean(trains, axis=1)[:, :, -1]
        Z_test = np.nanmean(tests, axis=1)[:, :, -1]

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
        cmap_states =  plt.get_cmap("plasma")
        colors_states = cmap_states(np.linspace(0, 1, n_eps + 6))
        
        # Setup Palettes for Left Block: Unsaturated "cool" (desaturated cyan-to-purple)
        cmap_cool = plt.get_cmap("cool")
        colors_sub = [sns.desaturate(cmap_cool(x), 0.4) for x in np.linspace(0, 1, n_states)]
        
        # --- Left Block (Sub-panels) ---
        gs_sub = gs_super[0].subgridspec(2, n_states, wspace=0.0, hspace=0.05)
        ax_curves_sub = np.empty((2, n_states), dtype=object)
        
        # Dynamic x_pct
        x_pct = np.linspace(100/n_eps, 100, n_eps)
        
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
            ax_curves_sub[0, s].scatter(x_pct, Z_train[s, :], c=line_color, s=50, zorder=11)
            ax_curves_sub[0, s].fill_between(x_pct, Z_train[s, :], 0.05, color=fill_color, alpha=0.15, zorder=-s)

            # Test - separated plot and scatter
            ax_curves_sub[1, s].plot(x_pct, Z_test[s, :], '-', color=line_color, alpha=1, zorder=10, lw = 2)
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
                
                mid_idx = n_states // 2
                if s == mid_idx:
                    ax_curves_sub[1, s].xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=100, decimals=0))
                    ax_curves_sub[1, s].set_xticks([x_pct[0], x_pct[n_eps//2], x_pct[-1]], labels = ["0%","50%","100%"])
                    ax_curves_sub[1, s].set_xlabel("Training Progress", fontsize=16)
                    ax_curves_sub[1, s].tick_params(labelbottom=True)
                
                # Dynamically set x limits
                ax_curves_sub[row, s].set_xlim(x_pct[0], x_pct[-1])

            # Remove bottom x-ticks for all columns EXCEPT the mid index
            if s != mid_idx:
                ax_curves_sub[0, s].tick_params(labelbottom=False)
                ax_curves_sub[1, s].tick_params(labelbottom=False)
        ax_curves_sub[0, mid_idx].tick_params(labelbottom=False)

        # --- Right Block (Conglomerates) ---
        gs_conglom = gs_super[1].subgridspec(2, 1, hspace=0.05)
        ax_conglom_top = fig_split.add_subplot(gs_conglom[0, 0])
        ax_conglom_bot = fig_split.add_subplot(gs_conglom[1, 0], sharex=ax_conglom_top)
        ax_curves_conglom = [ax_conglom_top, ax_conglom_bot]
        state_nums_fill = state_nums.copy()
        state_nums_fill[0] = state_nums[0] * .95 
        state_nums_fill[-1] = state_nums[-1] * 1.05 
        
        for i, phase in enumerate(["Training", "Testing"]):
            
            for e in range(n_eps):
                line_color = colors_states[e]
                fill_color = colors_eps[e]
                y_data = Z_train[:, e] if i==0 else Z_test[:, e]
                
                # Plot the horizontal curve alone without markers
                ax_curves_conglom[i].plot(state_nums, y_data, ':', color=line_color, lw = 2, alpha=1, zorder=10)
                ax_curves_conglom[i].scatter(state_nums, y_data, c=line_color, s=50, zorder=11)
                
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
        
        ax_curves_conglom[1].legend(handles=legend_elements, loc='upper left', frameon=True, ncol=1, fontsize=12, title = "Training Progress")
        plt.savefig(fig_path("RL_training_comparison.svg"), dpi=300)
        plt.show()

        """ VERSION B """

        # --- POLISHED FIGURE: COMBINED PANELS & 3D LANDSCAPE ---
        fig = plt.figure(figsize=(15, 15))

        # Main Title
        fig.suptitle("Learning curves per $\mathcal{S}$", fontsize=24, fontweight='bold', y=0.96)

        # Main GridSpec: 2 Rows (Top: Split Panels, Bottom: Overlaid 2D + 3D)
        gs_main = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.25)

        # Set up Colormaps
        cmap_eps = plt.get_cmap("plasma")
        colors_eps = cmap_eps(np.linspace(0, 1, n_eps))
        cmap_states = plt.get_cmap("plasma") 
        colors_states = cmap_states(np.linspace(0, 1, n_eps + 6)) 

        cmap_cool = plt.get_cmap("cool")
        colors_sub = [sns.desaturate(cmap_cool(x), 0.4) for x in np.linspace(0, 1, n_states)]

        # Dynamic x_pct
        x_pct = np.linspace(100/n_eps, 100, n_eps)

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
                    ax_curves_sub[1, s].set_xticks([x_pct[0], x_pct[n_eps//2], x_pct[-1]], labels=["0%", "50%", "100%"])
                    ax_curves_sub[1, s].set_xlabel("Training Progress", fontsize=16)
                    ax_curves_sub[1, s].tick_params(labelbottom=True)
                ax_curves_sub[row, s].set_xlim(x_pct[0], x_pct[-1])

            if s != mid_idx:
                ax_curves_sub[0, s].tick_params(labelbottom=False)
                ax_curves_sub[1, s].tick_params(labelbottom=False)
        ax_curves_sub[0, mid_idx].tick_params(labelbottom=False)

        # ==========================================
        # BOTTOM BLOCK: OVERLAID 2D & 3D LANDSCAPE
        # ==========================================
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
        ax_3d.legend(handles=legend_elements_3d, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncols=2, fontsize=16, frameon=True)

        fig.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.92)

        plt.savefig(fig_path("Combined_RL_training_landscape.svg"), dpi=300)
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
        plt.savefig(fig_path("Version_C_RL_Landscape.svg"), dpi=300)
        plt.show()