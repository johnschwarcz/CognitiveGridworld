import os
import sys
import inspect
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib import gridspec

# Setup paths
path = inspect.getfile(inspect.currentframe())
path = os.path.dirname(os.path.abspath(path))
print("root:", path)
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')

from main.CognitiveGridworld import CognitiveGridworld 

def extract_data(likelihood_temps, likelihood_freqs, base_params, target_zs=[0.17, 0.37], tolerance=0.03):
    """
    Iterates sequentially through parameter configurations, loads the models,
    runs 1 episode of inference, and extracts the accuracy arrays along with
    the generative likelihood structures for visualization.
    Returns a dictionary keyed by (lt, lf).
    """
    results = {}
    grid_size = len(target_zs)

    for lt in likelihood_temps:
        for lf in likelihood_freqs:           
            # Paths for both saved environments
            res_path = f"/params3/reservoir_LT{lt}_LF{lf}"
            ft_path = f"/params3/fully_trained_LT{lt}_LF{lf}"
            N = 1000
            
            # 1. Load and evaluate Reservoir Model
            res_args = base_params.copy()
            res_args.update({
                'likelihood_temp': lt,
                'likelihood_freq': lf,
                'reservoir': True,
                'load_env': res_path,
                'hid_dim': N,
            })
            res_model = CognitiveGridworld(**res_args)
            
            # 2. Load and evaluate Fully Trained Model
            ft_args = base_params.copy()
            ft_args.update({
                'likelihood_temp': lt,
                'likelihood_freq': lf,
                'reservoir': False,
                'load_env': ft_path,
                'hid_dim': N,
            })
            ft_model = CognitiveGridworld(**ft_args)                
            
            # 3. Extract Generating Functions (Likelihoods)
            # The likelihood structure is a property of the environment, identical for both agents.
            lik_dict = None
            if hasattr(ft_model, 'joint_Z') and getattr(ft_model, 'ctx_num', 0) >= 2:
                Z0 = ft_model.joint_Z[:, :, 0].flatten()
                Z1 = ft_model.joint_Z[:, :, 1].flatten()

                nl_flat = ft_model.naive_likelihood.reshape(-1, ft_model.ctx_num, ft_model.realization_num)
                L0 = nl_flat[:, 0, :]
                L1 = nl_flat[:, 1, :]
                jl_flat = ft_model.joint_likelihood.reshape(-1, ft_model.realization_num, ft_model.realization_num)

                lik_dict = {}
                for zi in range(grid_size):
                    for zj in range(grid_size):
                        mask_x = np.abs(Z0 - target_zs[zi]) <= tolerance
                        mask_y = np.abs(Z1 - target_zs[zj]) <= tolerance
                        mask = mask_x & mask_y

                        if mask.any():
                            lik_dict[(zi, zj)] = {
                                'L0': L0[mask].mean(axis=0),
                                'L1': L1[mask].mean(axis=0),
                                'mat': jl_flat[mask].mean(axis=0)
                            }
                        else:
                            lik_dict[(zi, zj)] = None

            # Store the extracted, averaged data in the results dictionary
            results[(lt, lf)] = {
                'res_model_acc': res_model.model_acc.mean(0),
                'ft_model_acc': ft_model.model_acc.mean(0),
                'joint_acc': res_model.joint_acc.mean(0),
                'naive_acc': res_model.naive_acc.mean(0),
                'joint_std': res_model.joint_acc.std(0),
                'naive_std': res_model.naive_acc.std(0),
                # 'res_train_acc': res_model.train_acc_through_training[:, -1],
                # 'ft_train_acc': ft_model.train_acc_through_training[:, -1],
                'res_train_acc': res_model.test_acc_through_training[:, -1],
                'ft_train_acc': ft_model.test_acc_through_training[:, -1],                
                'likelihoods': lik_dict
            }
            
    return results

def plot_results(likelihood_temps, likelihood_freqs, data):
    """
    Takes the extracted data dictionary and plots the 3x3 grid for inference steps.
    """
    print("Generating inference visualization...")
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
    for i, lt in enumerate(likelihood_temps):
        for j, lf in enumerate(likelihood_freqs):
            ax = axes[i, j]            
            
            # Retrieve data for this configuration
            config_data = data[(lt, lf)]
            res_model_acc = config_data['res_model_acc']
            ft_model_acc = config_data['ft_model_acc']
            joint_acc = config_data['joint_acc']
            naive_acc = config_data['naive_acc']

            # --- PLOTTING ---
            # Network performances
            ax.plot(res_model_acc, label='Echo State', color='r', linewidth=1, ls ='None', marker = 'o', ms = 4)
            ax.plot(ft_model_acc, label='Fully Trained', color='g', linewidth=1, linestyle='None', marker = 'o', ms = 4)
            
            # Bayes baselines
            ax.plot(joint_acc, label='Exact', color='g', linewidth=1.5, ls = '--')
            ax.plot(naive_acc, label='Factorized', color='r', linewidth=1.5, ls = '--')

            # Subplot formatting
            ax.set_title(f"Temperature = {lt} | Frequency = {lf}", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            # Label outer boundaries
            if i == 2:
                ax.set_xlabel("Inference Steps")
            if j == 0:
                ax.set_ylabel("Accuracy")
            # Place legend only in the first subplot to keep layout tidy
            if i == 0 and j == 0:
                ax.legend(loc='best', fontsize=12)
                
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

def plot_training_results(likelihood_temps, likelihood_freqs, data, batch_num):
    """
    Takes the extracted data dictionary and plots the 3x3 grid showing
    accuracy at the final step of inference throughout the training process.
    """
    print("Generating training trajectory visualization...")
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
    for i, lt in enumerate(likelihood_temps):
        for j, lf in enumerate(likelihood_freqs):
            ax = axes[i, j]            
            
            # Retrieve data for this configuration
            config_data = data[(lt, lf)]
            res_train_acc = config_data['res_train_acc']
            ft_train_acc = config_data['ft_train_acc']
            
            # Isolate the final inference step for the bayesian baselines
            joint_acc_final = config_data['joint_acc'][-1]
            naive_acc_final = config_data['naive_acc'][-1]
            joint_se = config_data['joint_std'][-1] / np.sqrt(batch_num)
            naive_se = config_data['naive_std'][-1] / np.sqrt(batch_num)

            # --- PLOTTING ---
            # Network performances throughout training
            ax.plot(res_train_acc, label='Echo State', color='r', linewidth=1.5, alpha = 1)
            ax.plot(ft_train_acc, label='Fully Trained', color='g', linewidth=1.5, alpha = 1)
            
            # Bayes baselines (horizontal lines across all training epochs)
            ax.axhline(y=joint_acc_final, label='Exact', color='g', linewidth=3, ls=':', alpha = 1)
            ax.axhline(y=naive_acc_final, label='Factorized', color='r', linewidth=3, ls=':', alpha = 1)

            # Subplot formatting
            ax.set_title(f"Temperature = {lt} | Frequency = {lf}", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Label outer boundaries
            if i == 2:
                ax.set_xlabel("Training Epochs")
            if j == 0:
                ax.set_ylabel("Final Step Accuracy")
                
            # Place legend only in the first subplot to keep layout tidy
            if i == 0 and j == 0:
                ax.legend(loc='best', fontsize=12)
            ax.set_ylim(.05, 1.05)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

def plot_likelihood_configurations(likelihood_temps, likelihood_freqs, data, base_params, target_zs=[0.17, 0.37]):
    """
    Plots a 3x3 grid of parameter configurations, where each cell contains a nested 
    target_zs x target_zs grid of generative likelihood functions (Joint and Marginals).
    """
    grid_size = len(target_zs)
    fig = plt.figure(figsize=(15, 15))
    
    # Define master 3x3 grid
    outer_grid = fig.add_gridspec(3, 3, wspace=0.3, hspace=0.35, left=0.05, right=0.95, bottom=0.05, top=0.92)   
    r_num = base_params.get('realization_num', 10)
    x_vals = np.arange(r_num)
    scale = r_num - 1
    
    for i, lt in enumerate(likelihood_temps):
        for j, lf in enumerate(likelihood_freqs):
            config_data = data[(lt, lf)].get('likelihoods')
            
            # Setup a background axis just for the title of this 3x3 cell
            ax_title = fig.add_subplot(outer_grid[i, j])
            ax_title.axis('off')
            ax_title.set_title(f"Temperature = {lt} | Frequency = {lf}", fontsize=12, pad=15)
            
            if config_data is None:
                ax_title.text(0.5, 0.5, "Data Unavailable", ha='center', va='center')
                continue
            
            # Subdivide the outer cell into a grid for the target Zs
            inner_outer_grid = outer_grid[i, j].subgridspec(grid_size, grid_size, wspace=0.15, hspace=0.1)
            
            # Determine global maximum for this specific LT/LF config to scale marginals uniformly
            global_max = 0
            for zi in range(grid_size):
                for zj in range(grid_size):
                    cell = config_data.get((zi, zj))
                    if cell is not None:
                        global_max = max(global_max, cell['L0'].max(), cell['L1'].max())
            if global_max == 0: global_max = 1 
            
            for zi in range(grid_size):
                for zj in range(grid_size):
                    row = grid_size - 1 - zj  
                    col = zi                  
                    
                    # Split inner cell into joint + marginals
                    inner_grid = inner_outer_grid[row, col].subgridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.0, hspace=0.0)
                    
                    ax_joint = fig.add_subplot(inner_grid[1, 0])
                    ax_marg_x = fig.add_subplot(inner_grid[0, 0], sharex=ax_joint)
                    ax_marg_y = fig.add_subplot(inner_grid[1, 1], sharey=ax_joint)
                    
                    ax_marg_x.axis('off')
                    ax_marg_y.axis('off')
                    
                    cell_data = config_data.get((zi, zj))
                    if cell_data is not None:
                        mean_L0 = cell_data['L0']
                        mean_L1 = cell_data['L1']
                        mean_mat = cell_data['mat']
                        
                        # Plot X marginal
                        ax_marg_x.plot(x_vals, mean_L0, color='C0', lw=1.5)
                        ax_marg_x.fill_between(x_vals, mean_L0, color='C0', alpha=0.3)
                        ax_marg_x.set_ylim(0, global_max * 1.05)
                        
                        # Plot Y marginal
                        ax_marg_y.plot(mean_L1, x_vals, color='C1', lw=1.5)
                        ax_marg_y.fill_betweenx(x_vals, mean_L1, color='C1', alpha=0.3)
                        ax_marg_y.set_xlim(0, global_max * 1.05)
                        
                        # Plot Joint Likelihood Heatmap
                        ax_joint.imshow(mean_mat.T, cmap='viridis', origin='lower', vmin=0, aspect='auto')
                    
                    # Formatting
                    ax_joint.set_xlim(-0.5, scale + 0.5)
                    ax_joint.set_ylim(-0.5, scale + 0.5)
                    ax_joint.set_xticks([])
                    ax_joint.set_yticks([])
                    for spine in ax_joint.spines.values():
                        spine.set_color('gray')
                        spine.set_alpha(0.3)
                        
    plt.show()

if __name__ == "__main__":
    # Search space
    likelihood_temps = [1, 2, 3]
    likelihood_freqs = [.5, 1, 2]
    batch_num = 20000 # 25000     
    # Base configuration for 1-episode inference run
    base_params = {
        'mode': "SANITY", 
        'episodes': 1,
        'show_plots': False,
        'realization_num': 10,  
        'obs_num': 5, 
        'training': False, 
        'skip_training_analyses': True,
        'batch_num': batch_num,
        'step_num': 30, # 10, 
        'state_num': 500, 
        'learn_embeddings': False,
        'classifier_LR': .001, 
        'ctx_num': 2,
        'early_stopping': True,
        'cuda': 0
    }
    
    # 1. Extract the data (heavy lifting + likelihoods)
    # extracted_data = extract_data(likelihood_temps, likelihood_freqs, base_params, target_zs=[0.17, 0.37])
    
    # 2. Plot the original inference step data
    plot_results(likelihood_temps, likelihood_freqs, extracted_data)

    # 3. Plot the new training trajectory data
    plot_training_results(likelihood_temps, likelihood_freqs, extracted_data, batch_num)
    
    # 4. Plot the generative likelihood structures across configurations
    plot_likelihood_configurations(likelihood_temps, likelihood_freqs, extracted_data, base_params, target_zs=[0.17, 0.37])