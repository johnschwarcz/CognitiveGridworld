"""Re-run the first 105 episodes of the training-dynamics pair (seed 20261029) with a checkpoint at every test episode
(every 5) to 100: train_training_dynamics.main with schedule(dense_until=100); output <net>_s20261029_ep105/."""
import os, sys
ROOT = "/home/johns/Desktop/CognitiveGridworld"
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "main", "REPLICATION", "trainings")]
import train_training_dynamics as T
_orig = T.schedule
T.schedule = lambda episodes, dense_until=0: _orig(episodes, dense_until=100)
T.main(sys.argv[1], int(sys.argv[2]), 105, 20261029)
