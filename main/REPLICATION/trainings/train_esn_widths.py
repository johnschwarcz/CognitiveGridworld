"""ESNs at other widths, trained with the paper's Echo N = 1k protocol (train_sanity_networks.py ->
sanity/reservoir_ctx_2: SANITY mode, 50,000 episodes, batch 8000, 30 steps, 500 variables, 10 values, 5 observations,
two variables, lr 1e-3, learn_embeddings False, the default loss, a test batch every 500 episodes), for Figure 9's
left panel (John 10-31: "start a fresh 2K ESN", after asking for ESNs "with the same amount of training"). Only the
width changes, plus four switches that change no update:
  gpu_inference           the same exact posterior, on the GPU (2.7x faster at width 100)
  micro_batch             the batch-8000 update in slices with gradients accumulated (exact at loss power 1, near-exact
                          at the default power: model_forward.forward_pass_micro); needed for memory, as BPTT runs
                          through the frozen LSTM to the trained read-in
  show_plots              False
  skip_training_analyses  the agent's own per-test analyses (two 1000-component PCAs of 240,000 x N activations: ~5 min
                          per test episode at N = 2000, ~8 h per run); Figure 9 needs only test_acc_through_training,
                          and the paper's train_sanity_reps.py skips them too (the first 2k launch, 22:17, was stopped
                          at 22:25 for this)
Seeded with SEED for reproducibility (the paper's runs were unseeded).
-> main/DATA/experimental/esn_widths/reservoir_ctx_2_<N>N_net.pth (save() format, with test_acc_through_training)
Run from the repository root, detached:
    setsid nohup /home/johns/anaconda3/envs/eureka/bin/python main/REPLICATION/trainings/train_esn_widths.py \\
        <N> <gpu> [micro_batch, default 1000] > main/DATA/experimental/esn_widths/reservoir_ctx_2_<N>N.log 2>&1 &"""
import os
import sys
import random
import numpy as np
import torch

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")
sys.path[:0] = [ROOT] + [os.path.join(ROOT, *p) for p in (["main"], ["main", "bayes"], ["main", "model"])]
from main.CognitiveGridworld import CognitiveGridworld                            # noqa: E402

SEED = 20261031


if __name__ == "__main__":
    N, gpu = int(sys.argv[1]), int(sys.argv[2])
    micro = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    os.makedirs(os.path.join(ROOT, "main", "DATA", "experimental", "esn_widths"), exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    CognitiveGridworld(**{"mode": "SANITY", "cuda": gpu, "episodes": 50000, "realization_num": 10, "hid_dim": N,
                          "obs_num": 5, "training": True, "batch_num": 8000, "step_num": 30, "state_num": 500,
                          "learn_embeddings": False, "reservoir": True, "classifier_LR": .001, "ctx_num": 2,
                          "gpu_inference": True, "micro_batch": micro, "show_plots": False,
                          "skip_training_analyses": True,
                          "save_env": f"/experimental/esn_widths/reservoir_ctx_2_{N}N"})
