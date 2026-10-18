"""Deep feedforward networks (mode "FF") on the sufficient statistic: what does depth buy?

    python main/REPLICATION/trainings/train_ff_deep.py [sweep | main | all | smoke | lstm | variants | d9_frozen | widths | widths_fixed | small | deep30 | deep30_res | frozen_widths | reservoir_compare | d4_w20000 | reservoir_1000 | matched_a | reservoirs_wide | tff_series | small_widths]

FF sees each trial's observation means over all T steps plus the true Z, an exactly sufficient
statistic for the final-step joint posterior, so only final-step accuracy is meaningful.
Width 1000, readin_depth 1..9 with readout_depth 1, standard PyTorch init. The FF loss is the
power-0.5 symmetric KL to the joint posterior at the final step (the repo default); a power-1 arm is
run alongside, set in this script only (FF_loss's default is patched inside each run's own process).

sweep  every (power, depth, learning rate) for SWEEP_EPISODES at seed SWEEP_SEED, nothing saved; a
       rate per (power, depth) is then chosen by choose_lr's rule and written to chosen_lr.json.
main   every (power, depth) x REPS for MAIN_EPISODES at its chosen rate (seed = rep), each saved as
       main/DATA/experimental/ff_deep/d{d}_p{power}_lr{lr}_rep{r}_net.pth. Power-0.5 depths whose
       every rate collapsed are skipped: a collapsed net has exactly zero gradients and cannot recover.
all    sweep, then main.          smoke    every depth at a tiny size, into ff_deep/_smoke.
lstm   control for the power: the LSTM of sanity_default_reps (mode SANITY, same settings; loss on all
       steps) trained with power-1 loss instead of its default 0.5, one rep for MAIN_EPISODES at lr 1e-3,
       saved as ff_deep/lstm_p1_lr0.001_rep0_net.pth and logged like the FF runs.
variants  depth 4, power 1, into ff_deep/d4_variants: (a) each non-empty subset of its three N x N layers frozen
       at initialization (weights and biases; Adam skips parameters without gradients), lr 1e-3 as chosen
       for depth 4, seeds = the trained depth-4 reps' so every condition starts from the same weights and
       world; (b) width N = WIDE fully trained, its rate picked by the same sweep and rule as before. Both
       VARIANT_REPS x MAIN_EPISODES, all 8 at once. Frozen runs log frozen_drift (max |W - W_init| over frozen layers, must be 0).
d9_frozen  depth 9, power 1, lr 1e-3, seed 0 (the fully trained depth-9 rep 0's weights and world), into ff_deep/d9_frozen:
       k = 1..8 of its N x N layers frozen from the input end or from the output end (d9_frozen_sets), MAIN_EPISODES.
       Depth 9 with k frozen has as many trainable parameters as a fully trained depth 9 - k.
widths  into ff_deep/widths, power 1, seed 0 (one world with every run above), MAIN_EPISODES: FF at widths 1000 and
       NEW_WIDTHS, each depth 1-9 fully trained plus depth 4's FROZEN and depth 9's d9_frozen_sets; and LSTMs at
       LSTM_WIDTHS (trainable parameters matching the width-1000 FF depths 2-8) and 1000. Every (width, depth) and LSTM
       width gets its own rate from the usual sweep and rule (FF over LRS_WIDTHS, which adds 3e-3; LSTMs over
       LSTM_LRS); frozen runs use their depth's. Existing equivalent results are copied in and skipped; width-1000 runs
       are redone only where the chosen rate is not 1e-3. Sweep, then main, via run_scheduled on the GPUs given
       (default GPU 0; launched 2026-10-11 on GPU 0 only, John's choice), stopped during the sweep.
widths_fixed  the same main runs (NEW_WIDTHS and LSTM_WIDTHS; width 1000 exists) without the sweep, all at FIXED_LR,
       into ff_deep/widths (2026-10-11, finally NEW_WIDTHS = [2000] and LSTM widths 600 and 400, on both GPUs).
deep30  depth 30 with all 29 N x N layers frozen at widths 1000, 2000, 4000 (power 1, FIXED_LR, seed 0, MAIN_EPISODES), into
       ff_deep/widths; `deep30 <gpus>` (default GPU 0). The no-skip control for a frozen stack with skip connections.
reservoir_compare  reservoirs (LSTM frozen, loss power 0.5 as in sanity_default_reps) at N = 200 and 100, and FF depth 4
       with all three N x N layers frozen at widths 100, 500, 5000, 10000; power 1 for FF, FIXED_LR, seed 0,
       MAIN_EPISODES (20k), into ff_deep/widths. `reservoir_compare <gpus>` (default GPU 0).
d4_w20000  FF depth 4, all three N x N layers frozen, width 20000 (completes that series); `d4_w20000 <gpus>` (default GPU 0).
matched_a  FF at widths 200 and 2000 (depths 1-4, all N x N layers frozen) and reservoirs at loss power 1 (as the FF nets)
       at N = 100, 200, 500, batch 8000, MAIN_EPISODES, seed 0; `matched_a <gpus> <budget MB>` (default GPU 0, 13000).
small_widths  N = 10, 20, 50 for every family of ff_deep_frozen_widths: FF depths 1-3 (every N x N layer frozen),
       reservoirs and fully trained LSTMs (power 1), depth 30 with all 29 layers frozen x sqrt(6) (tff1; the time-injected
       nets left the analysis, John 2026-10-16); batch 8000, MAIN_EPISODES, seed 0; `small_widths <gpus> <budget MB>`.
tff_series  time-injected FF (one layer per time step, layer k fed the mean of the first 31 - k observations; 29 frozen
       layers = default draws x sqrt(6)), power 1, batch 8000, MAIN_EPISODES, seed 0; `tff_series <gpus> <budget MB> <N,...>
       [tff | tffs | tff1] [free floor MB, default 3000]`: tffs one shared readin; tff1 the control without time injection (input at layer 1 only: the
       standard depth-30 FF with the same frozen layers, 2026-10-16).
reservoirs_wide  reservoirs at loss power 1 on the wide path (fp16 frozen weights, deduplicating frozen LSTM, micro-batches
       with accumulated gradients; same model and batch-8000 updates), batch 8000, MAIN_EPISODES, seed 0;
       `reservoirs_wide <gpus> <budget MB> <N:micro,...> [lr, default FIXED_LR]`.
reservoir_1000  reservoir at N = 1000 with the same settings as reservoir_compare (power 0.5, FIXED_LR, seed 0, batch 8000,
       MAIN_EPISODES), so all reservoirs in ff_deep_frozen_widths share the FF nets' data; `reservoir_1000 <gpus>`.
frozen_widths  depths 1-3 at FROZEN_WIDTHS with every N x N layer frozen (only readin and readout train; depth 1 has
       none), power 1, FIXED_LR, seed 0, MAIN_EPISODES, into ff_deep/widths; existing runs are skipped (depth 2 at
       width 1000 is there already; depth 1 at width 1000 is ff_deep/d1_p1_lr0.001_rep0). `frozen_widths <gpus>`
       (default GPU 1, John's choice, 2026-10-12).
deep30_res  the width-4000 deep30 net with residual skips (readin_residual: h <- h + relu(W h + b) / sqrt(29) around every
       N x N layer, the 29 frozen ones included), name suffix _res; `deep30_res <gpus>` (default GPU 1, John's choice).
small  LSTM width 200 (0.33M parameters) on GPU 0, and depth 2 with its one N x N layer frozen at widths 1000 and 2000
       on GPU 1 (depth 2's only frozen version: input end, output end and all frozen coincide); power 1, FIXED_LR,
       seed 0, MAIN_EPISODES, into ff_deep/widths (2026-10-12).

Every run is its own process (a hard crash kills only that run) and writes log_<name>.txt,
<name>_curve.npz (rewritten at every test checkpoint) and <name>_summary.json when it ends. Runs whose
summary exists are skipped, so a restart resumes. progress.log lists runs as they finish.
PARALLEL runs at a time, all on GPU 0 (GPU 1 is reserved), except where a mode says otherwise.
"""
import os
import sys
import json
import inspect
import contextlib
import traceback
import time
import math
import queue
import glob
import shutil
import threading
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

root = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
while not os.path.exists(os.path.join(root, 'main', 'CognitiveGridworld.py')):
    root, prev = os.path.dirname(root), root
    if root == prev:
        raise RuntimeError("project root not found")
sys.path.insert(0, root)

SCRIPT = os.path.abspath(inspect.getfile(inspect.currentframe()))
PY = sys.executable
OUT = os.path.join(root, 'main', 'DATA', 'experimental', 'ff_deep')
DEPTHS = list(range(1, 10))
POWERS = [0.5, 1]
LRS = {0.5: [1e-3, 3e-4, 1e-4, 3e-5, 1e-5], 1: [1e-3, 3e-4, 1e-4]}
REPS = [0, 1]
SWEEP_SEED = 100                    # separate from the main reps, so no rep is the run its rate was picked on
SWEEP_EPISODES, SWEEP_CKPT = 1500, 250
MAIN_EPISODES, MAIN_CKPT = 20000, 500
PARALLEL = 6                        # all on GPU 0; ~1.3 GB each; the CPU-side environment is the bottleneck
VARIANTS = os.path.join(OUT, "d4_variants")
# depth 4's N x N layers (1-3; 0 is the 15 x N readin), frozen from the input end and from the output end;
# [2] and [1, 3] were run at first and stopped part way (2026-10-08): John kept only the two contiguous families
FROZEN = [[1], [1, 2], [1, 2, 3], [3], [2, 3]]
WIDE = 4000
VARIANT_REPS = [0]                  # one rep per variant: the depth-4 reps agreed closely
D9_DIR = os.path.join(OUT, "d9_frozen")
WIDTHS_DIR = os.path.join(OUT, "widths")
NEW_WIDTHS = [2000]                 # was [2000, 3000, 4000]; John cut it to width 2000 only (2026-10-11)
LSTM_LRS = [3e-3, 1e-3, 3e-4]       # smaller LSTMs may want larger rates than the width-1000 LSTM's 1e-3
FIXED_LR = 1e-3                     # widths_fixed: one rate for every width, depth and LSTM size
TFF_GAIN = math.sqrt(6)         # time-injected FF: the frozen layers' default draws x sqrt(6) (Kaiming-uniform; 2026-10-15)
TFF_REGENERATE_FROM = 2000      # widths whose 29 frozen layers are rebuilt from seeds instead of stored (identical
                                # weights to storing them, far less memory: GPU 0 is shared, 2026-10-15)
FROZEN_WIDTHS = [100, 500, 1000, 5000, 10000, 20000]
LRS_WIDTHS = [3e-3, 1e-3, 3e-4, 1e-4]   # FF grid in widths mode: width 1000 picked 1e-3, the top of LRS[1], at every depth


def ff_params(depth, n, k = 0):
    """Trainable parameters of an FF of this depth and width with k of its N x N layers frozen."""
    return 16 * n + (depth - 1 - k) * (n * n + n) + 20 * n + 20


def lstm_params(n):
    """Trainable parameters of the LSTM (mode SANITY) at width n: readin 15 -> n, the LSTM's two 4n x n matrices and
    two 4n biases, readout n -> 20, and the learned initial state (h0, c0)."""
    return 16 * n + 8 * n * n + 8 * n + 20 * n + 20 + 2 * n


# first matched to the width-1000 FF depths 2-8 ([357, 502, 613, 708, 791, 866, 935]); John cut it to two smaller LSTMs
# beside the existing width-1000 one (2026-10-11)
LSTM_WIDTHS = [600, 400, 200]      # 200 added 2026-10-12 (small mode)


def d9_frozen_sets():
    """Depth 9's N x N layers are 1-8 (0 is the 15 x N readin). k of them frozen from the input end, {1..k}, and from
    the output end, {9-k..8}, for k = 1..8 (k = 8 is the same set either way): 15 sets, interleaved by k."""
    sets = []
    for k in range(1, 9):
        a, b = list(range(1, k + 1)), list(range(9 - k, 9))
        sets += [a] if a == b else [a, b]
    return sets

BASE = dict(
    mode = "FF", training = True, show_plots = False, learn_embeddings = False, reservoir = False,
    skip_training_analyses = True, gpu_inference = True, state_num = 500, ctx_num = 2, obs_num = 5,
    realization_num = 10, step_num = 30, hid_dim = 1000, batch_num = 8000, readout_depth = 1, cuda = 0)


def run_name(depth, power, lr, seed, hid = 1000, frozen = (), residual = False, reservoir = False):
    """depth "lstm" is the LSTM control (mode SANITY) rather than an FF depth; hid and frozen appear only when not default."""
    return ((depth if depth in ("lstm", "tff", "tffs", "tff1") else f"d{depth}") + (f"_N{hid}" if hid != 1000 else "") + f"_p{power:g}_lr{lr:g}"
            + (frozen_tag(frozen) if frozen else "") + ("_res" if residual else "") + ("_reservoir" if reservoir else "") + f"_rep{seed}")


def frozen_tag(frozen):
    """_frz<digits> for single-digit layer indices (as all runs up to depth 9 are named); a contiguous range with any
    index >= 10 as _frz<first>-<last>, since digit strings would be ambiguous there."""
    frozen = list(frozen)
    if max(frozen) < 10:
        return f"_frz{''.join(map(str, frozen))}"
    assert frozen == list(range(frozen[0], frozen[-1] + 1)), "only contiguous sets are named above depth 10"
    return f"_frz{frozen[0]}-{frozen[-1]}"


def write_atomic(path, save):
    """save(file object) writes to a temp file, which then replaces path: a kill mid-write keeps the old file."""
    tmp = path + ".tmp"
    with open(tmp, "wb" if path.endswith(".npz") else "w") as fh:
        save(fh)
    os.replace(tmp, path)


def train(depth, power, lr, seed, episodes, checkpoint_every, folder, save, batch_num = None, hid = 1000, frozen = (), residual = False,
          reservoir = False, micro_batch = None):
    """One run, in its own process. Writes the summary json whatever happens. micro_batch (reservoirs): the wide path,
    i.e. fp16-only frozen weights, the deduplicating frozen LSTM and batches run in slices of micro_batch episodes with
    gradients accumulated (main/model/frozen_lstm.py, model_forward.py forward_pass_micro): the same model and the same
    batch-8000 updates (fp64-tested 2026-10-13), so the run name does not change."""
    import torch
    from main.CognitiveGridworld import CognitiveGridworld
    from main.model.model_backward import Model_backward
    Model_backward.FF_loss.__defaults__ = (1e-8, power)       # this process only
    Model_backward.SANITY_loss.__defaults__ = (1e-8, power)

    def first_and_last_linear(model):
        layers = [l for stack in (model.classifier_readin, model.classifier_readout)
                  for l in stack.modules() if isinstance(l, torch.nn.Linear)]
        return layers[0], layers[-1]

    def grad_norm(layer):
        return float(layer.weight.grad.norm()) if layer.weight.grad is not None else np.nan

    name = run_name(depth, power, lr, seed, hid, frozen, residual, reservoir)
    os.makedirs(folder, exist_ok = True)
    rows = []
    frozen_init, info = {}, {}

    def freeze(model):
        """Keep the chosen readin layers at their initial weights and biases: no gradient, so Adam never updates them."""
        readin = model.classifier_readin              # a single Linear at depth 1 and in the LSTM
        linears = [l for l in (readin if isinstance(readin, torch.nn.Sequential) else [readin]) if isinstance(l, torch.nn.Linear)]
        for i in frozen:
            for prm in linears[i].parameters():
                prm.requires_grad_(False)
            # snapshot on the CPU: on the GPU a width-20000 layer's copy plus the drift check's temporaries cost ~4.8 GB
            frozen_init[i] = (linears[i], linears[i].weight.detach().cpu().clone(), linears[i].bias.detach().cpu().clone())
        info["trainable"] = sum(prm.numel() for prm in model.parameters() if prm.requires_grad)

    def record(self):
        """Called after every episode; logs only at test checkpoints and must return {}."""
        if not info:
            freeze(self.model)                      # first call: after episode 0, a test episode, before any update
        if not self.test_set:
            return {}
        q = np.asarray(self.model_goal_belief, float)[:, -1]                       # (B, R), final step
        qc = np.clip(q, 1e-12, 1)
        P, N = (np.clip(np.asarray(b, float)[:, -1], 1e-12, 1) for b in (self.joint_goal_belief, self.naive_goal_belief))
        top, goal = q.argmax(-1), np.asarray(self.goal_ind)
        share = [np.bincount(top[goal == c], minlength = q.shape[-1]).max() / max((goal == c).sum(), 1)
                 for c in range(self.ctx_num)]                                      # per goal context: an input-blind net answers one value per context
        first, last = first_and_last_linear(self.model)
        rows.append(dict(
            ep = self.e, acc = self.test_accs[self.test_e - 1, -1],
            joint = self.joint_acc.mean(0)[-1], naive = self.naive_acc.mean(0)[-1],
            loss = float(np.ravel(getattr(self, "classifier_loss", np.nan))[0]),     # last training episode
            maxprob = q.max(-1).mean(), top_share = max(share),
            kl_joint = (P * np.log(P / qc)).sum(-1).mean(), kl_naive = (N * np.log(N / qc)).sum(-1).mean(),
            kl_net_joint = (qc * np.log(qc / P)).sum(-1).mean(),                    # the direction the LSTM logs hold
            grad_first = grad_norm(first), grad_last = grad_norm(last),
            frozen_drift = max((max(float((l.weight.detach().cpu() - w0).abs().max()), float((l.bias.detach().cpu() - b0).abs().max()))
                                for l, w0, b0 in frozen_init.values()), default = 0.)))
        write_atomic(os.path.join(folder, f"{name}_curve.npz"),
                     lambda fh: np.savez(fh, **{k: np.array([r[k] for r in rows]) for k in rows[0]}))
        sys.stdout.flush(); sys.stderr.flush()
        return {}

    torch.manual_seed(seed); np.random.seed(seed)
    status = "ok"
    with open(os.path.join(folder, f"log_{name}.txt"), "w", buffering = 1) as f, \
         contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        print(json.dumps(dict(name = name, depth = depth, power = power, lr = lr, seed = seed, episodes = episodes,
                              hid = hid, frozen = list(frozen), residual = residual, reservoir = reservoir, micro_batch = micro_batch,
                              gpu = os.environ.get("CUDA_VISIBLE_DEVICES"))), flush = True)
        try:
            CognitiveGridworld(**dict(BASE, batch_num = batch_num or BASE["batch_num"], hid_dim = hid, readin_residual = residual,
                                      reservoir = reservoir,
                                      **(dict(micro_batch = micro_batch, reservoir_dedup = True, reservoir_fp16 = True) if micro_batch else {}),
                                      **(dict(mode = "SANITY", readin_depth = 1) if depth == "lstm" else
                                         dict(readin_depth = 1, ff_time_inject = True, ff_frozen_gain = TFF_GAIN,
                                              ff_regenerate = hid >= TFF_REGENERATE_FROM, ff_shared_readin = depth == "tffs",
                                              ff_inject_first_only = depth == "tff1")
                                         if depth in ("tff", "tffs", "tff1") else
                                         dict(readin_depth = depth))),
                               classifier_LR = lr, episodes = episodes,
                               checkpoint_every = checkpoint_every, external_logger = record,
                               save_env = os.path.relpath(os.path.join(folder, name), os.path.join(root, 'main', 'DATA'))
                                          if save else None)
        except Exception:
            traceback.print_exc()
            status = "crashed"
    last3 = rows[-3:]
    gpu_mem_mb = round(torch.cuda.max_memory_reserved() / 2 ** 20) if torch.cuda.is_available() else None
    summary = dict(name = name, depth = depth, power = power, lr = lr, seed = seed, hid = hid, frozen = list(frozen), residual = residual,
                   reservoir = reservoir, micro_batch = micro_batch,
                   trainable = info.get("trainable"), frozen_drift = max([r["frozen_drift"] for r in rows], default = None),
                   status = status, checkpoints = len(rows), gpu_mem_mb = gpu_mem_mb,
                   last_ep = int(rows[-1]["ep"]) if rows else None,
                   acc_last3 = float(np.mean([r["acc"] for r in last3])) if last3 else None,
                   maxprob_last3 = float(np.mean([r["maxprob"] for r in last3])) if last3 else None,
                   top_share_last3 = float(np.mean([r["top_share"] for r in last3])) if last3 else None,
                   naive = float(np.mean([r["naive"] for r in rows])) if rows else None,
                   joint = float(np.mean([r["joint"] for r in rows])) if rows else None)
    write_atomic(os.path.join(folder, f"{name}_summary.json"), lambda fh: json.dump(summary, fh, indent = 1))
    return summary


def run_one(job, parallel = PARALLEL, gpu = "0"):
    """Run one job in a fresh process unless its summary already exists; never raises."""
    depth, power, lr, seed, episodes, ckpt, folder, save = job[:8]
    hid, frozen = (job[9], job[10]) if len(job) > 9 else (1000, ())
    residual = job[11] if len(job) > 11 else False
    reservoir = job[12] if len(job) > 12 else False
    path = os.path.join(folder, f"{run_name(depth, power, lr, seed, hid, frozen, residual, reservoir)}_summary.json")
    if os.path.exists(path):
        return dict(json.load(open(path)), skipped = True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = gpu, MPLBACKEND = "Agg",
               OMP_NUM_THREADS = str(max(1, (os.cpu_count() or 8) // parallel)))
    rc = subprocess.run([PY, SCRIPT, "one", json.dumps(job)], env = env, cwd = root,
                        stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL).returncode
    if os.path.exists(path):
        return json.load(open(path))
    return dict(name = run_name(depth, power, lr, seed, hid, frozen, residual, reservoir), depth = depth, power = power, lr = lr, seed = seed,
                hid = hid, frozen = list(frozen), status = f"died rc={rc}")


def run_many(jobs, tag, at_once = PARALLEL, slots = None, later = None):
    """Run jobs, one per GPU slot. slots: GPU ids, one entry per concurrent run (default: at_once on GPU 0). later:
    (more slots, a function that blocks until they may be used), added by a background thread."""
    os.makedirs(OUT, exist_ok = True)
    out = []
    slots = slots or ["0"] * min(at_once, len(jobs))
    free = queue.Queue()
    for g in slots:
        free.put(g)
    if later:
        def release(more, wait):
            wait()
            for g in more:
                free.put(g)
        threading.Thread(target = release, args = later, daemon = True).start()
    workers = min(len(jobs), len(slots) + (len(later[0]) if later else 0))

    def work(job):
        g = free.get()
        try:
            return run_one(job, workers, g)
        finally:
            free.put(g)

    with open(os.path.join(OUT, "progress.log"), "a", buffering = 1) as log:
        log.write(f"=== {tag}: {len(jobs)} runs, GPU slots {dict(Counter(slots))}"
                  + (f", then {dict(Counter(later[0]))} more" if later else "") + "\n")
        with ThreadPoolExecutor(workers) as pool:
            for fut in as_completed([pool.submit(work, j) for j in jobs]):
                out.append(fut.result())
                log.write(json.dumps(out[-1]) + "\n")
        log.write(f"=== {tag}: done\n")
    return out


def mem_mb(job):
    """Estimated peak GPU memory of a job in MB (as nvidia-smi reports it), for run_scheduled. Fitted 2026-10-09 to
    probes at batch 8000 (FF: depths 1/4/9 at widths 1000-4000, frozen sets; LSTM at widths 357 and 935), x 1.15:
    FF ~ 1150 + 16 B per trainable and 8 B per frozen parameter (weight + the frozen_init copy) + 2 B per batch x
    width x depth activation; LSTM ~ 280 + 9.8 MB per unit of width (its 30 steps of gates and states dominate).
    Against 60-episode probes (memory plateaus after a few steps) the margin is ~5-20%."""
    depth, hid, frozen = job[0], (job[9] if len(job) > 9 else 1000), (job[10] if len(job) > 10 else [])
    batch = job[8] or BASE["batch_num"]
    micro = job[13] if len(job) > 13 else None
    if depth in ("tff", "tffs", "tff1"):   # frozen layers: 29 stored (fp32 + the fp16 copy kept for backward) or one rebuilt at a time
        L = BASE["step_num"] - 1                   # (fp16 + an fp32 draw chunk); plus every layer's fp16 ReLU output
        weights = 2 * hid * hid + 4 * 4096 * hid if hid >= TFF_REGENERATE_FROM else L * 6 * hid * hid
        return 1.15 * (1150 + (weights + BASE["step_num"] * 2 * batch * hid) / 2 ** 20)
    if depth == "lstm" and micro:                  # wide path: fp16 weights + one slice's activations; measured 3356 MB at N=2000,
        return 1.15 * (1000 + (16 * hid * hid + 1300 * hid * min(micro, batch)) / 2 ** 20)   # slices of 1000 (2026-10-13)
    if depth == "lstm":                           # at most widths; small ones peak at ~2 GB whatever the width (N=100: 2024 MB,
        return max(1.15 * (280 + 9.8 * hid * batch / 8000), 2600)   # N=200: 1816 MB allocated, + CUDA context; 2026-10-16)
    k = len(frozen)
    return 1.15 * (1150 + (ff_params(depth, hid, k) * 16 + k * (hid * hid + hid) * 8 + batch * hid * depth * 2) / 2 ** 20)


def gpu_cost(job):
    """Relative GPU work of a job (episodes x batch-width^2 units), to start the heaviest jobs first."""
    depth, hid, frozen = job[0], (job[9] if len(job) > 9 else 1000), (job[10] if len(job) > 10 else [])
    if depth in ("tff", "tffs", "tff1"):
        return job[4] * hid * hid * 4 * (BASE["step_num"] - 1)
    if depth == "lstm":
        return job[4] * 1440 * hid * hid                 # 30 steps x (2n x 4n) per step, forward and backward
    k = len(frozen)
    return job[4] * hid * hid * (6 * (depth - 1 - k) + 4 * k)


def job_name(job):
    depth, power, lr, seed = job[:4]
    return run_name(depth, power, lr, seed, *((job[9], job[10]) if len(job) > 9 else ()), *job[11:13])


def retryable(job, res):
    """A run that died, failed to start, or crashed on CUDA out-of-memory (external GPU use can cause it) gets one more
    try; a crash for any other reason (e.g. NaN from divergence at its rate) is an outcome and is kept."""
    status = str(res.get("status", ""))
    if status.startswith(("died", "orchestrator error")):
        return True
    if status == "crashed":
        log = os.path.join(job[6], f"log_{job_name(job)}.txt")
        return os.path.exists(log) and ("out of memory" in open(log, errors = "ignore").read().lower())
    return False


def gpu_free_mb(gpu):
    """Free memory of a GPU as nvidia-smi reports it, in MiB: counts every process, not only this scheduler's."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu)],
                         capture_output = True, text = True).stdout.strip()
    return float(out.splitlines()[0]) if out else 0.


def run_scheduled(jobs, tag, gpus = ("0", "1"), budget_mb = 21000, per_gpu = 8, free_floor_mb = None):
    """Start jobs in list order, each on the GPU with the least estimated memory in use among those where it fits
    (mem_mb within budget_mb, fewer than per_gpu running); a job that fits nowhere waits for a running one to end.
    free_floor_mb: also require that the GPU's actual free memory, less the estimates of jobs started in the last 3 min
    (not yet allocated), stays at least this much above the job's estimate; re-checked every minute, so jobs start as other
    schedulers' runs end (2026-10-15: two launches next to a run whose memory had grown came within 1 GB of the limit).
    A retryable failure is requeued once (a crashed run's summary is kept as <name>_summary_try1.json)."""
    os.makedirs(OUT, exist_ok = True)
    cond = threading.Condition()
    used, running = {g: 0 for g in gpus}, {g: 0 for g in gpus}
    out, threads, retried = [], [], set()
    pending = list(jobs)
    recent = []                                          # (start time, gpu, estimate MB) of jobs that may still be allocating
    log = open(os.path.join(OUT, "progress.log"), "a", buffering = 1)
    log.write(f"=== {tag}: {len(jobs)} runs on GPUs {list(gpus)}, up to {per_gpu} each within {budget_mb} MB\n")

    def runner(job, g, mb):
        try:
            res = run_one(job, len(gpus) * per_gpu, g)
        except BaseException as e:                       # run_one should not raise; never leave the slot taken
            res = dict(name = job_name(job), status = f"orchestrator error {e!r}")
        again = job_name(job) not in retried and retryable(job, res)
        if again:
            retried.add(job_name(job))
            summary = os.path.join(job[6], f"{job_name(job)}_summary.json")
            if os.path.exists(summary):
                os.replace(summary, summary.replace("_summary.json", "_summary_try1.json"))
        with cond:
            used[g] -= mb
            running[g] -= 1
            if again:
                pending.append(job)
            else:
                out.append(res)
            try:
                log.write(json.dumps(res) + ("   <- requeued once" if again else "") + "\n")
            except Exception:
                pass
            cond.notify_all()

    with cond:
        while pending or any(running.values()):
            for i, job in enumerate(pending):
                mb = math.ceil(min(mem_mb(job), budget_mb))       # integers: no float residue blocks a job
                recent[:] = [r for r in recent if time.time() - r[0] < 180]
                fits = [g for g in gpus if running[g] < per_gpu and used[g] + mb <= budget_mb
                        and (free_floor_mb is None or gpu_free_mb(g) - sum(r[2] for r in recent if r[1] == g) - mem_mb(job) >= free_floor_mb)]
                if fits:
                    g = min(fits, key = lambda g: used[g])
                    used[g] += mb
                    running[g] += 1
                    recent.append((time.time(), g, mem_mb(job)))
                    threads.append(threading.Thread(target = runner, args = (job, g, mb)))
                    threads[-1].start()
                    pending.pop(i)
                    break
            else:                                        # woken by any finishing run; the timeout is a safety net
                cond.wait(timeout = 60 if free_floor_mb is not None else 600)
    for t in threads:
        t.join()
    log.write(f"=== {tag}: done\n")
    log.close()
    return out


def collapsed(s):
    return s["maxprob_last3"] is not None and s["maxprob_last3"] > 0.95 and s["acc_last3"] < 0.15


def choose_lr(summaries, depth, power, hid = 1000, grid = None):
    """Rule fixed before the sweep was run. Score = final-step test accuracy over the last 3 checkpoints.
    Excluded: runs that crashed or died, and collapsed ones (mean max probability > 0.95 with score < 0.15).
    Among the rest take the best score; if other rates score within 0.01 of it, take the SMALLEST of those
    (a larger rate that looks as good at 1,500 episodes can still collapse later in a long run).
    If every rate is excluded: the best-scoring run that did not crash, flagged; if all crashed, the
    smallest rate, flagged. (Added 2026-10-09, not part of the rule: `note` lists rates that crashed, died or are
    missing from grid, and a pick at the top of grid.)"""
    runs = [s for s in summaries if s["depth"] == depth and s["power"] == power and s.get("hid", 1000) == hid]
    alive = [s for s in runs if s["status"] == "ok" and s.get("acc_last3") is not None]
    valid = [s for s in alive if not collapsed(s)]
    note = [f"{s['lr']:g} {s['status']}" for s in runs if s not in alive]
    note += [f"{g:g} missing" for g in (grid or []) if g not in {s["lr"] for s in runs}]
    if valid:
        best = max(s["acc_last3"] for s in valid)
        pick = min((s for s in valid if s["acc_last3"] >= best - 0.01), key = lambda s: s["lr"])
        if grid and pick["lr"] == max(grid):
            note.append("pick at the top of the grid")
        return dict(lr = pick["lr"], score = pick["acc_last3"], flag = "", note = "; ".join(note))
    if alive:
        pick = max(alive, key = lambda s: s["acc_last3"])
        return dict(lr = pick["lr"], score = pick["acc_last3"], flag = "every rate collapsed", note = "; ".join(note))
    return dict(lr = min(s["lr"] for s in runs), score = None, flag = "every rate crashed", note = "; ".join(note))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "one":                                   # internal: one run in this process
        train(*json.loads(sys.argv[2]))
        sys.exit()
    if what == "smoke":                                 # end-to-end check at a tiny size
        folder = os.path.join(OUT, "_smoke")
        res = run_many([(d, p, 1e-3, 0, 12, 5, folder, True, 200) for p in POWERS for d in DEPTHS], "smoke")
        print(json.dumps(sorted(res, key = lambda s: (s["power"], s["depth"])), indent = 1))
        print({(d, p): choose_lr(res, d, p) for p in POWERS for d in DEPTHS})
        sys.exit()
    if what == "variants":
        sweep = run_many([(4, 1, lr, SWEEP_SEED, SWEEP_EPISODES, SWEEP_CKPT, os.path.join(VARIANTS, "lr_sweep"), False, None, WIDE, [])
                          for lr in LRS[1]], f"d4 N{WIDE} lr sweep")
        wide = dict(depth = 4, power = 1, hid = WIDE, **choose_lr(sweep, 4, 1))
        with open(os.path.join(VARIANTS, "chosen_lr.json"), "w") as f:
            json.dump(dict(rule = choose_lr.__doc__, chosen = [wide], sweep = sweep), f, indent = 1)
        run_many([(4, 1, wide["lr"], r, MAIN_EPISODES, MAIN_CKPT, VARIANTS, True, None, WIDE, []) for r in VARIANT_REPS]     # slowest first
                 + [(4, 1, 1e-3, r, MAIN_EPISODES, MAIN_CKPT, VARIANTS, True, None, 1000, fr) for r in VARIANT_REPS for fr in FROZEN],
                 "d4 variants", at_once = 8)                     # 8 fit: 24 cores, ~13 GB of GPU 0
        sys.exit()
    if what == "d9_frozen":
        # default: 8 now on GPU 1 (freed by John for this), 7 on GPU 0 once the d4 variants end.
        # `d9_frozen <first> <gpu> <slots>`: only jobs[first:], now, on that GPU (used 10-08 to start jobs 8-14 on GPU 0
        # without waiting, after the first launcher was stopped with its 8 runs left going)
        jobs = [(9, 1, 1e-3, 0, MAIN_EPISODES, MAIN_CKPT, D9_DIR, True, None, 1000, fr) for fr in d9_frozen_sets()]
        if len(sys.argv) > 2:
            first, gpu, n = int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
            run_many(jobs[first:], f"d9 frozen sweep, jobs {first}-{len(jobs) - 1}", slots = [gpu] * n)
            sys.exit()
        def d4_variants_done():
            while "=== d4 variants: done" not in open(os.path.join(OUT, "progress.log")).read():
                time.sleep(60)
        run_many(jobs, "d9 frozen sweep", slots = ["1"] * 8, later = (["0"] * 7, d4_variants_done))
        sys.exit()
    if what == "reservoir_compare":                     # John, 2026-10-13: matched comparison at 20k episodes
        # reservoirs (LSTM frozen, power 0.5 as in sanity_default_reps) at N = 200 and 100, and FF depth 4 with all three
        # N x N layers frozen at the widths the depth 1-3 series has (1000 and 2000 exist); seed 0, MAIN_EPISODES
        jobs = [("lstm", .5, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, [], False, True) for n in (200, 100)]
        jobs += [(4, 1, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, [1, 2, 3]) for n in (100, 500, 5000, 10000)]
        run_scheduled(sorted(jobs, key = gpu_cost, reverse = True), "reservoir_compare: reservoirs N 200/100, FF d4 frozen widths",
                      gpus = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("0",))
        sys.exit()
    if what == "small_widths":                          # John, 2026-10-16: N = 10, 20, 50 for everything in the figure
        ns = (10, 20, 50)                               # LSTMs first: they need the largest holes (~2.6 GB each)
        jobs = [("lstm", 1, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, [], False, True) for n in ns]   # reservoirs
        jobs += [("lstm", 1, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, []) for n in ns]             # fully trained
        jobs += [("tff1", 1, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, []) for n in ns]        # depth 30, x sqrt(6)
        jobs += [(d, 1, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, list(range(1, d))) for d in (1, 2, 3) for n in ns]   # depth 4 dropped (John 10-18)
        run_scheduled(jobs, "small_widths: N 10/20/50 for every family", gpus = tuple(sys.argv[2].split(",")),
                      budget_mb = int(sys.argv[3]), free_floor_mb = 3000)
        sys.exit()
    if what == "tff_series":                            # John, 2026-10-15: time-injected FF (main/model/time_ff.py)
        # one layer per time step, layer k fed the mean of the first 31 - k observations through its own readin; the 29
        # frozen N x N layers are the default draws x TFF_GAIN; power 1, FIXED_LR, batch 8000, MAIN_EPISODES, seed 0, in
        # the order given: `tff_series <gpus> <budget MB> <N,...>`
        depth = sys.argv[5] if len(sys.argv) > 5 else "tff"     # "tffs": one shared readin; "tff1": input at layer 1 only
        jobs = [(depth, 1, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, int(n), []) for n in sys.argv[4].split(",")]
        floor = int(sys.argv[6]) if len(sys.argv) > 6 else 3000      # [free floor MB]: lower to start beside a big run
        run_scheduled(jobs, f"tff_series {sys.argv[4]} {depth}" + (f" floor {floor}" if floor != 3000 else ""),
                      gpus = tuple(sys.argv[2].split(",")), budget_mb = int(sys.argv[3]), free_floor_mb = floor)
        sys.exit()
    if what == "reservoirs_wide":                       # John, 2026-10-13: matched reservoirs on the wide path
        # reservoirs at loss power 1 (as the FF nets), batch 8000, MAIN_EPISODES, seed 0, each N with its micro_batch:
        # `reservoirs_wide <gpus> <budget MB> <N:micro,...>`, e.g. 2000:1000,5000:500
        spec = [tuple(int(v) for v in x.split(":")) for x in sys.argv[4].split(",")]
        lr = float(sys.argv[5]) if len(sys.argv) > 5 else FIXED_LR    # [lr]: 5e-4 at N=10000 (John 10-18): with Adam the readout's
        # step on the 30-step cumsummed logits grows ~ lr x N; at 1e-3 the N=10000 reservoir overshoots in its first ~500 updates
        jobs = [("lstm", 1, lr, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, [], False, True, m) for n, m in spec]
        run_scheduled(jobs, f"reservoirs_wide {sys.argv[4]}" + (f" lr {lr:g}" if lr != FIXED_LR else ""),
                      gpus = tuple(sys.argv[2].split(",")), budget_mb = int(sys.argv[3]))
        sys.exit()
    if what == "matched_a":                             # John, 2026-10-13: matched widths, phase A (current code)
        # FF at widths 200 and 2000 (depths 1-4, every N x N layer frozen; existing runs are skipped) and reservoirs at loss
        # power 1, as the FF nets, at N = 100, 200, 500; all batch 8000, MAIN_EPISODES, seed 0. GPU 0 also holds the
        # depth-4 width-20000 run (~10 GB), hence the smaller memory budget (argv[3], MB of mem_mb estimates).
        jobs = [(d, 1, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, list(range(1, d))) for n in (200, 2000) for d in (1, 2, 3, 4)]
        jobs += [("lstm", 1, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, [], False, True) for n in (100, 200, 500)]
        run_scheduled(sorted(jobs, key = mem_mb, reverse = True), "matched_a: FF widths 200/2000, reservoirs power 1 N 100/200/500",
                      gpus = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("0",),
                      budget_mb = int(sys.argv[3]) if len(sys.argv) > 3 else 13000)
        sys.exit()
    if what == "reservoir_1000":                        # John, 2026-10-13: a reservoir at width 1000 on the FF nets' data
        # (20k episodes x batch 8000, seed 0, power 0.5), so every reservoir in ff_deep_frozen_widths has the same data
        jobs = [("lstm", .5, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, 1000, [], False, True)]
        run_scheduled(jobs, "reservoir_1000: reservoir N 1000, matched data",
                      gpus = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("0",))
        sys.exit()
    if what == "d4_w20000":                             # John, 2026-10-13: completes the depth-4 frozen-width series
        jobs = [(4, 1, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, 20000, [1, 2, 3])]
        run_scheduled(jobs, "d4_w20000: FF depth 4, all 3 N x N layers frozen, width 20000",
                      gpus = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("0",))
        sys.exit()
    if what == "frozen_widths":                         # John, 2026-10-12: only readin and readout trained, depth 1-3
        jobs = [(d, 1, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, list(range(1, d)))
                for n in FROZEN_WIDTHS for d in (1, 2, 3) if (d, n) != (1, 1000)]      # depth 1 at width 1000 is ff_deep/d1_p1_lr0.001_rep0
        run_scheduled(sorted(jobs, key = gpu_cost, reverse = True), "frozen_widths: depths 1-3, every N x N layer frozen",
                      gpus = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("1",))
        sys.exit()
    if what == "deep30_res":                            # John, 2026-10-12: deep30 width 4000 with residual skips, on GPU 1
        jobs = [(30, 1, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, 4000, list(range(1, 30)), True)]
        run_scheduled(jobs, "deep30_res: width 4000, all 29 frozen, residual skips", gpus = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("1",))
        sys.exit()
    if what == "deep30":                                # John, 2026-10-12: control for a skip-connection test
        jobs = [(30, 1, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, list(range(1, 30))) for n in (4000, 2000, 1000)]
        run_scheduled(jobs, "deep30: all 29 N x N layers frozen", gpus = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("0",))
        sys.exit()
    if what == "small":                                 # John, 2026-10-12: one GPU each
        lstm = [("lstm", 1, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, 200, [])]
        d2 = [(2, 1, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, [1]) for n in (1000, 2000)]
        threads = [threading.Thread(target = run_scheduled, args = (jobs, tag), kwargs = dict(gpus = (gpu,)))
                   for jobs, tag, gpu in ((lstm, "small: LSTM width 200", "0"), (d2, "small: depth 2, layer frozen", "1"))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        sys.exit()
    if what == "widths_fixed":                          # `widths_fixed [gpu ids, comma-separated; default 0]`
        # the widths main runs without the sweep, every run at FIXED_LR (John, 2026-10-11): at power 1 the sweeps never
        # clearly moved the rate (1e-3 won at every width-1000 depth; width 4000 depth 4 was a 0.005 tie with 3e-4).
        # Width-1000 runs and the width-1000 LSTM exist at this rate already.
        gpus = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("0",)
        main_jobs = []
        for n in NEW_WIDTHS:
            main_jobs += [(d, 1, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, []) for d in DEPTHS]
            main_jobs += [(4, 1, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, fr) for fr in FROZEN]
            main_jobs += [(9, 1, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, fr) for fr in d9_frozen_sets()]
        main_jobs += [("lstm", 1, FIXED_LR, 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, []) for n in LSTM_WIDTHS]
        run_scheduled(sorted(main_jobs, key = gpu_cost, reverse = True), "widths main, fixed lr", gpus = gpus)
        sys.exit()
    if what == "widths":                                # `widths [gpu ids, comma-separated; default 0]`
        gpus = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("0",)
        sweep_dir = os.path.join(WIDTHS_DIR, "lr_sweep")
        os.makedirs(sweep_dir, exist_ok = True)
        # results that already exist with identical settings are copied in, so resume skips them instead of rerunning:
        # the width-1000 sweep (seed 100, LRS[1]) and depth 4 at width 4000 (its sweep and seed-0 main, d4_variants)
        for src in (glob.glob(os.path.join(OUT, "lr_sweep", "d*_p1_lr*_rep100_summary.json"))
                    + glob.glob(os.path.join(VARIANTS, "lr_sweep", f"d4_N{WIDE}_p1_lr*_rep100_summary.json"))):
            dst = os.path.join(sweep_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        for src in glob.glob(os.path.join(VARIANTS, f"*d4_N{WIDE}_p1_lr0.0003_rep0*")):
            dst = os.path.join(WIDTHS_DIR, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        widths = [1000] + NEW_WIDTHS
        keys = [(d, n) for n in widths for d in DEPTHS] + [("lstm", n) for n in LSTM_WIDTHS + [1000]]
        grid = lambda d: LSTM_LRS if d == "lstm" else LRS_WIDTHS
        sweep_jobs = [(d, 1, lr, SWEEP_SEED, SWEEP_EPISODES, SWEEP_CKPT, sweep_dir, False, None, n, []) for d, n in keys for lr in grid(d)]
        sweep = run_scheduled(sorted(sweep_jobs, key = gpu_cost, reverse = True), "widths lr sweep", gpus = gpus)
        chosen = {key: choose_lr(sweep, key[0], 1, key[1], grid(key[0])) for key in keys}
        with open(os.path.join(WIDTHS_DIR, "chosen_lr.json"), "w") as f:
            json.dump(dict(rule = choose_lr.__doc__, chosen = [dict(depth = d, hid = n, power = 1, **chosen[d, n]) for d, n in keys],
                           sweep = sweep), f, indent = 1)
        with open(os.path.join(OUT, "progress.log"), "a") as log:
            for (d, n), c in chosen.items():
                if c["flag"] or c["note"]:
                    log.write(f"widths: depth {d} width {n}: lr {c['lr']:g} {c['flag']} {c['note']}\n")
        # width-1000 runs at 1e-3 already exist (ff_deep, d4_variants, d9_frozen, lstm_p1_lr0.001_rep0): only a width-1000
        # (depth, LSTM) whose chosen rate changed gets new main runs, into widths/ like the others
        new = lambda key: not (key[1] == 1000 and chosen[key]["lr"] == 1e-3)
        main_jobs = []
        for n in widths:
            main_jobs += [(d, 1, chosen[d, n]["lr"], 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, [])
                          for d in DEPTHS if new((d, n))]
            if new((4, n)):
                main_jobs += [(4, 1, chosen[4, n]["lr"], 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, fr) for fr in FROZEN]
            if new((9, n)):
                main_jobs += [(9, 1, chosen[9, n]["lr"], 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, fr) for fr in d9_frozen_sets()]
        main_jobs += [("lstm", 1, chosen["lstm", n]["lr"], 0, MAIN_EPISODES, MAIN_CKPT, WIDTHS_DIR, True, None, n, [])
                      for n in LSTM_WIDTHS + [1000] if new(("lstm", n))]
        run_scheduled(sorted(main_jobs, key = gpu_cost, reverse = True), "widths main", gpus = gpus)
        sys.exit()
    if what == "lstm":
        run_many([("lstm", 1, 1e-3, 0, MAIN_EPISODES, MAIN_CKPT, OUT, True)], "lstm power 1")
        sys.exit()
    sweep_dir = os.path.join(OUT, "lr_sweep")
    if what in ("sweep", "all"):
        sweep = run_many([(d, p, lr, SWEEP_SEED, SWEEP_EPISODES, SWEEP_CKPT, sweep_dir, False)
                          for p in POWERS for d in DEPTHS for lr in LRS[p]], "lr sweep")
        chosen = [dict(depth = d, power = p, **choose_lr(sweep, d, p)) for p in POWERS for d in DEPTHS]
        with open(os.path.join(OUT, "chosen_lr.json"), "w") as f:
            json.dump(dict(rule = choose_lr.__doc__, chosen = chosen, sweep = sweep), f, indent = 1)
    if what in ("main", "all"):
        with open(os.path.join(OUT, "chosen_lr.json")) as f:
            chosen = json.load(f)["chosen"]
        keep = [c for c in chosen if not (c["power"] == 0.5 and c["flag"])]
        with open(os.path.join(OUT, "progress.log"), "a") as log:
            for c in chosen:
                if c not in keep:
                    log.write(f"skipping main runs for depth {c['depth']} power {c['power']}: {c['flag']}\n")
        run_many([(c["depth"], c["power"], c["lr"], r, MAIN_EPISODES, MAIN_CKPT, OUT, True)
                  for r in REPS for c in keep], "main")
