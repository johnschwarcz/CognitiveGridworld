"""Four-condition oracle pilot: which gradient path can learn the embeddings?

    python main/REPLICATION/trainings/train_oracle_pilot.py <condition> <cuda>

Conditions differ only in which loss reaches the compositional embeddings, and whether
L_reg is applied to them.
"""
import os, sys, time
import numpy as np
_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d) and not os.path.exists(os.path.join(_d, 'main', 'CognitiveGridworld.py')):
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from main.env.env_model_data_manager import Env_model_data_manager
from main.CognitiveGridworld import CognitiveGridworld

CONDITIONS = {
    "default":   dict(embedding_grad="generator",  embedding_reg=True),
    "clf_noreg": dict(embedding_grad="classifier", embedding_reg=False),
    "clf_reg":   dict(embedding_grad="classifier", embedding_reg=True),
    "both":      dict(embedding_grad="both",       embedding_reg=True),
}
COMMON = dict(mode="oracle", show_plots=False, ctx_num=2, obs_num=5, realization_num=10,
              state_num=500, hid_dim=1000, batch_num=10000, step_num=30,
              learn_embeddings=True, training=True, classifier_LR=1e-3, generator_LR=1e-3,
              classifier_ent_bonus=.01, episodes=50000, checkpoint_every=250,
              gpu_inference=True)

_orig, T0 = Env_model_data_manager.log_model, time.time()
def log_model(self):
    _orig(self)
    if self.test_set and self.test_e > 1:
        i = self.test_e - 1
        ep = i * self.checkpoint_every
        print(f"ep {ep:>6} ({ep*self.batch_num/1e6:>6.0f}M)  train {self.train_accs[i,-1]:.4f}  "
              f"test {self.test_accs[i,-1]:.4f}  gen_loss "
              f"{float(np.ravel(self.generator_loss_log[i])[0]):.4f}  [{(time.time()-T0)/60:.0f}m]",
              flush=True)
Env_model_data_manager.log_model = log_model

if __name__ == "__main__":
    name, cuda = sys.argv[1], int(sys.argv[2])
    print(f"=== {name} | {CONDITIONS[name]} | cuda {cuda} | "
          f"batch {COMMON['batch_num']} x {COMMON['episodes']} eps = "
          f"{COMMON['batch_num']*COMMON['episodes']/1e6:.0f}M samples ===", flush=True)
    CognitiveGridworld(**COMMON, **CONDITIONS[name], cuda=cuda,
                       save_env=f"/oracle_pilot/{name}")
    print(f"=== {name} DONE in {(time.time()-T0)/3600:.1f} h ===", flush=True)
