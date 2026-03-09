"""Combined representational manifolds (Sum of Marginal Entropies, 5 Rows)."""
import numpy as np
import os
import sys
import inspect
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/bayes')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld

# ═══════════════════════════════════════════════════════════════════
# Plotting Functions
# ═══════════════════════════════════════════════════════════════════

def compute_r2(X, y):
    return Ridge(alpha=1.0).fit(X, y).score(X, y)

def calc_entropy(p):
    """Helper to calculate Shannon entropy along axis 1."""
    p = np.clip(p, 1e-20, 1.0)
    p /= np.sum(p, axis=1, keepdims=True)
    return -np.sum(p * np.log(p), axis=1)

def plot_manifold_evolution(model_pairs, R_val, timesteps=[0, 5, 10, -1]):
    """Plots the PCA manifold across models, allowing for multiple timesteps."""
    models_to_plot = ["Trained", "Joint", "Naive", "Echo"]
    n_models, n_steps = len(models_to_plot), len(timesteps)
    n_cols = (n_models * n_steps) + (n_steps - 1)
    
    # Compact width ratios with a 0.3 spacer
    width_ratios = ([1] * n_models + [0.3]) * n_steps
    width_ratios = width_ratios[:-1]
            
    fig, axes = plt.subplots(5, n_cols, figsize=(8 * n_steps, 12), 
                             gridspec_kw={'width_ratios': width_ratios}, 
                             constrained_layout=True)
    
    title_str = " $\\rightarrow$ ".join([f"Step (t={t})" for t in timesteps])
    fig.suptitle(f"Representational Manifold Evolution: {title_str}", fontsize=20, fontweight='bold')
    
    # Turn off axes for the empty spacer columns
    for row in axes:
        for step in range(n_steps - 1):
            row[(step + 1) * n_models + step].axis('off')

    for t_idx_loop, t_idx in enumerate(timesteps):
        for m_idx, mname in enumerate(models_to_plot):
            if mname not in model_pairs: 
                continue
                    
            bel, model = model_pairs[mname]
            ctx = np.asarray(model.ctx_vals)
            r1, r2 = ctx[:, 0].astype(int), ctx[:, 1].astype(int)
            
            X_slice = bel[:, t_idx]
            
            # Flatten for PCA
            X_flat = X_slice.reshape(len(X_slice), -1)           
            
            # Ensure 3D shape for entropy extraction, handling both flat and unflattened inputs
            X_3d = X_slice.reshape(len(X_slice), 2, -1) if X_slice.ndim == 2 else X_slice
            r_ent_sum = calc_entropy(X_3d[:, 0]) + calc_entropy(X_3d[:, 1])
             
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_flat)
            var_exp = pca.explained_variance_ratio_
            
            col = (t_idx_loop * (n_models + 1)) + m_idx
            
            row_data = [
                (r1, 'viridis', 'R1'),
                (r2, 'viridis', 'R2'),
                (r1 + r2, 'inferno', 'SUM'),
                (r1 - r2, 'inferno', 'DIFF'),
                (r_ent_sum, 'plasma', 'ENTROPY')
            ]
            
            for row, (data, cmap, ylabel_title) in enumerate(row_data):
                ax = axes[row, col]
                ax.scatter(X_pca[:, 0], X_pca[:, 1], c=data, cmap=cmap, s=150, alpha=0.1, edgecolor='none')
                
                r2_score = compute_r2(X_pca, data)
                r2_text = f"$R^2$: {r2_score:.2f}"
                
                ax.text(0.95, 0.05, r2_text, transform=ax.transAxes, 
                        ha='right', va='bottom', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=np.clip(1.5 * r2_score, 0.2, 1), edgecolor='k'))
                
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(True, linestyle='--', alpha=0.3)
                
                if row == 0:
                    ax.set_title(f"{mname} | Step (t={t_idx})\nPC1: {var_exp[0]:.1%}, PC2: {var_exp[1]:.1%}", fontweight='bold', fontsize=10)
                if col == 0:
                    ax.set_ylabel(ylabel_title, fontweight='bold', fontsize=12)
                    
    plt.show()

# ═══════════════════════════════════════════════════════════════════
# Execution
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    common = dict(mode="SANITY", cuda=0, episodes=1, checkpoint_every=5,
                  realization_num=10, hid_dim=1000, obs_num=5, show_plots=False,
                  batch_num=15000, step_num=30, state_num=500,
                  learn_embeddings=False, classifier_LR=.001, ctx_num=2, training=False)

    # Uncomment and instantiate models here if not already loaded:
    # echo    = CognitiveGridworld(**{**common, 'reservoir': True,  'load_env': "/sanity/reservoir_ctx_2_e5"})
    # trained = CognitiveGridworld(**{**common, 'reservoir': False, 'load_env': "/sanity/fully_trained_ctx_2_e5"})

    model_pairs = {
        "Trained": (trained.model_belief_flat, trained),
        "Joint":   (trained.joint_belief, trained),
        "Naive":   (echo.naive_belief, echo),
        "Echo":    (echo.model_belief_flat, echo),
    }

    plot_manifold_evolution(model_pairs, trained.realization_num, timesteps=[0, 5, 10, -1])