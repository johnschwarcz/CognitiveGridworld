# Reproducing the data and figures

The repository holds code, figures and a small data bundle. The large data lives outside git. Most data is never needed
whole: rebuild only what one analysis needs.

## What is where

| tier | size | where | how to get it |
|---|---|---|---|
| code, figures (`main/REPLICATION/figure_bin`, PNG/SVG) | — | git | clone |
| light: figure-level summary caches | ~96 MB | git, `main/DATA_light/` | `python main/data_tools.py light-restore` |
| archive: weights, training-run logs, the probe set (a 38 GB selection, below) | 38 GB | private Hugging Face dataset `johnschwarcz/cognitive-gridworld-data` | `python main/data_tools.py fetch '<pattern>'` (needs a read token: `export HF_TOKEN=...`, ask John) |
| caches: everything else in `main/DATA` | ~105 GB | not stored | regenerate (below) |

`main/DATA_MANIFEST.md` summarises every folder of `main/DATA` (size, tier, the modules that mention it).
`main/DATA_MANIFEST.tsv` lists every file with its size and tier.

## Setup

1. Create the environment with `conda env create -f environment.yml` (exact versions). Alternatively,
   `pip install -r requirements-core.txt` gives Python 3.12, torch 2.5.1, numpy 1.26.4, scipy 1.15.1 and matplotlib 3.10.0.
2. `export CGW_ROOT=/path/to/your/clone`. Modules default to `/home/johns/Desktop/CognitiveGridworld`.
   `CGW_PY` overrides the python that a few modules use to launch workers.
3. Run `python main/data_tools.py light-restore` to copy the light bundle into `main/DATA`.
4. GPU modules take one GPU each: `CUDA_VISIBLE_DEVICES=0` (or `1`).
5. Every analysis module's docstring states its inputs, outputs, commands and runtime.

## Minimal regeneration for one analysis

* `python main/data_tools.py plan <name>` prints, for a traced command:
  * the archive files to fetch;
  * the light files it uses;
  * the commands to run first, in order, with runtimes.

  Per-checkpoint families generalise from one traced checkpoint. Example, `plan ipcs_figure`
  (td_tt_interaction_pcs.png):
  1. fetch 4.3 GB: the probe set and the 114 checkpoints of the trained run;
  2. run `td_tt_interaction_pcs.py one <E>` for each checkpoint (~3 s each on a GPU);
  3. run `figure` (~2 s).
* Traces are stored in `main/deps/`. Currently there are three: `ipcs_49995`, `lesfix_49995` and `ipcs_figure`.
* To add a trace, run `python main/data_tools.py trace <name> <script.py> [args]` on a machine that has the inputs. The
  outputs must not exist yet, because resumable harvests skip existing files. The trace records exactly which
  `main/DATA` files the run reads and writes.
* A file listed under "no trace writes yet" either has a producer that has not been traced, or is optional. For example,
  td_tt_interaction_pcs checks itself against `analysis/<run>/ck_*.npz` only when that file exists.

## Costs in the training-dynamics family (one GPU)

* Weights: each run saved 114 checkpoints through training (`ckpt_ep*_net.pth`). A checkpoint is about 35 MB for the
  N=1000 LSTM, so about 4 GB per run.
* Per-checkpoint harvests (forward passes over the weights):

  | harvest | per checkpoint |
  |---|---|
  | `analysis/<run>/ck_*.npz` (td_analysis; 40 GB in total; the slowest) | ~65 s |
  | interaction directions | ~3 s |
  | flips, flip ratios | ~4 s |
  | fixed-direction lesions | ~25 s |
  | first lesion pass (109 conditions) | ~11 min |

* Summary caches (`combined`, `ratios`, `numbers`) take seconds to minutes on a CPU. Figures take seconds.
* Rebuilding all caches takes about a day of GPU time. A single analysis usually needs one run's checkpoints and minutes
  to a couple of hours.

## Tested (2026-11-03)

A fresh copy of the repository was given only the inputs `plan` lists: the probe set, one checkpoint and one 26 MB
reference cache. With `CGW_ROOT` pointing at the copy, it regenerated two caches:

| cache | max relative difference from the original |
|---|---|
| `through_training/interaction_pcs/trained_49995.npz` | 4e-17 |
| `through_training/lesions/trained/fixed_49995.npz` | 4e-16 |

## What the archive holds (`archive_selection` in main/data_tools.py; 5,573 files, 38.4 GB)

Weights through training (all 114 checkpoints) are kept only where through-training analyses use them:
* the Trained RNN and the ESN;
* the width and loss-power replications: N = 50, 200, 100 × 5 seeds, and loss power 1.

Every other run keeps only its final weights: the joint-loss runs, the MLP and the last-step variants. Only the last
optimizer state is kept per run. All training logs, the probe set and the other experiments' networks are kept.

Not uploaded, and kept only on John's machine:
* the very wide networks (`ff_deep/widths`, 28 GB; `lazyrich`, 7 GB);
* the other runs' intermediate checkpoints (30 GB);
* older optimizer states.

## Why the full local archive is ~107 GB

| what | size |
|---|---|
| weights through training (1,892 checkpoints of the training-dynamics runs, 114 per run) | 44.6 GB |
| wide feed-forward networks (`ff_deep/widths`, N = 10,000 units; final, resume and optimizer files) | 28.3 GB |
| large reservoirs and sanity runs (`lazyrich`, `sanity*`) | 17.7 GB |
| optimizer states (11 per training-dynamics run) | 3.6 GB |
| `final_net` files (weights plus training logs) | 1.9 GB |
| other runs | 4.9 GB |
| training-run logs and tests | 5.4 GB |
| probe set | 0.3 GB |
