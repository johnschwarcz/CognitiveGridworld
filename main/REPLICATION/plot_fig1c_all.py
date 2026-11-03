"""Regenerate the fig1c figures and every cache they read, in one process: how the factorized observers fold the latent
grid (two summary figures), belief trajectories of particles by their factorization regret (FR), and four movies of
the beliefs over time -> main/REPLICATION/figure_bin/ (or --out DIR).

    target   output                                  drawn (by)                                                   min
    summary  fig1c_summary_fold.png                  distances between particles' estimates; convergence to the   0.1
             fig1c_summary_updates.png               true cell by the cell's FR; the update sizes of the two
                                                     beliefs and their joint movement (figure_summary)
    traj_fr  fig1c_traj_FR_groups_T30.png            50 particles in the lowest and the highest 10% of FR, in     0.1
                                                     each agent's step-30 plane (figure_traj_fr)
    pcs      fig1c_movie_pcs.gif                     the beliefs at every step, each step in its own PC plane,    1.7
                                                     and the mean paths of four cells (figure_pcs)
    envdist  fig1c_movie_fixed_envdist.gif           the beliefs and 39 environments' grids in each agent's       2.8
                                                     T = 30 plane (figure_envdist)
    sweep    fig1c_movie_hybrid_sweep_combined.gif   a hybrid observer with the joint evidence at the least or    1.1
                                                     the most entangled stream-steps (figure_sweep)
    split    fig1c_movie_split_traj.gif              pairs of streams that stay together, then split, and their   1.3
                                                     environments' trajectories (figure_split)
Groups: stills = summary traj_fr; movies = pcs envdist sweep split; all = both (the default). Minutes: all six in one
process on the CPU of a shared 24-core machine with the caches present, 7.1 min in all (about one core, 2.3 GB at
most); every output byte-identical to the kept file in figure_bin (27 Oct 2026). Each figure sets its own complete
matplotlib state (plotting_helpers.style), so the order in which the figures run does not matter.
Not made here: fig1c_evidence_geometry.png (the evidence's geometry, a separate direction), the measures movies and
their still (fig1c_movie_measures*.gif, fig1c_measures_rings.png; exploratory) and the master figure (fig1_master.*).
Agents: Joint = the Optimal (exact joint) Bayes observer, Naive = the Naive (factorized) Bayes observer, Trained = the
trained LSTM (Trained RNN), Echo = the ESN (the reservoir network). A goal-ordered belief is 20 numbers: the goal
variable's marginal over its values r = 0..9, then the contextual variable's; raw order is the harvest's context order
[context 0 | context 1]. A cell is (r_g, r_c); one stream (a particle) is one episode's 30 observations of one cell.

Data: main/DATA/experimental/fig1c/ (DATA is gitignored); the figures read nothing else. A missing cache is built when a
figure first needs it (ensure), after the caches it reads; builders write through a temporary file, never half a file.
    cache            file(s)                          builder                inputs                                  min
    record           record.npz                       build_record           the gradflow harvests (8.6 GB each)       0
    episodes         episodes.npz                     build_episodes         record, the trained checkpoint's K, Q   0.2
    beliefs          beliefs.npz                      build_beliefs          record, both checkpoints                  5
    beliefs_steps    beliefs_steps.npz                build_beliefs_steps    record, both checkpoints                3.5
    alias_traj       alias_traj.npz                   build_alias_traj       record, episodes, both checkpoints      1.5
    scan_envs        scan_envs.npz                    build_scan_envs        record, episodes                          0
    scan             scan_meta.npz, scan_<agent>.npz  build_scan             scan_envs, record, episodes, ckpts    20-45
    distances_fr     distances_fr.npz                 build_distances_fr     scan                                    0.1
    distances_pairs  distances_pairs.npz              build_distances_pairs  record, episodes, both checkpoints        6
    updates          updates_50k.npz                  build_updates          record, episodes, beliefs_steps, ckpts    5
    split_pairs      split_pairs.npz                  build_split_pairs      scan                                    0.5
    sweep100         sweep100.npz                     build_sweep100         record, episodes, scan_envs               3
    (optional)       traj_fr_draw_{low,high}.npz      -                      fr_group_episodes checks its draws on them
(minutes: CPU at 2 threads; distances_pairs in 6 worker processes of 1 thread.) What each file holds:
    record.npz           the primary (below): ep_ctx_inds (50000, 2), ep_goal_ind (50000,), ep_ctx_vals (50000, 2) as
                         uint8; ep_obs_flat (50000, 30, 5) bool; ep_active_Z (50000, 5, 2) float32, the network's
                         context input as the GPU computed it (a CPU recomputation differs by one rounding step in 2/3
                         of its entries); the ESN harvest's first 20,000 episodes (echo_ep_ctx_vals, echo_ep_obs_flat,
                         echo_ep_active_Z), which feed only beliefs.npz's echo and ctx_echo. record() decodes them to
                         the harvest's dtypes
    episodes.npz         the environment side of the 50,000 episodes as the harvest stored it: ep_joint_likelihood
                         (50000, 5, 10, 10) f64, P(o_i = 1 | r0, r1); ep_joint_belief, ep_naive_belief
                         (50000, 30, 2, 10) f64, raw order; rev (50000, 30) f32, the entanglement of each step
    beliefs.npz          trained, echo, echo_on_trained (20000, 2, 2, 10) f32 at t = 1 and 30, raw order; ctx_trained,
                         ctx_echo (20000, 2); steps
    beliefs_steps.npz    Trained, Echo (20000, 30, 20) f32 goal-ordered, every step
    alias_traj.npz       harvest episode ENV's four CELLS x 2000 streams (seed 1): Joint, Naive, Trained, Echo
                         (8000, 30, 20) f32 goal-ordered; cell_of, cells, rev (8000, 30)
    scan_envs.npz        eps, goal (80,): the scan's environments (harvest episodes)
    scan_*.npz           the alias scan: the first 40 of scan_envs x 100 goal-ordered cells x 50 streams (seed
                         1000 + e); scan_<agent>.npz: beliefs (40, 5000, 30, 20) f16, answer (40, 5000, 30, 2) int8;
                         scan_meta.npz: eps, goal, table_id, seed, pert_seed, sigma, M, cells, cell_of, L, Lpert, obs,
                         rev, FR, naive_inf(_gap), pert_inf(_gap), recreate_maxdiff, alias_answer_mismatch
    distances_fr.npz     per scan environment (e = 2 left out), cell and step: the mean distance of the posterior means
                         from the true cell, binned by the cell's FR at T: dist_t{10,20}, fr_median{10,20}, FRb, agents
    distances_pairs.npz  110 environments x 100 cells x 10 streams (seed 7,000,000 + episode): the distance between the
                         estimates of every pair of streams, averaged per true distance: ymean (110, 4, 2, 30, 51), n,
                         xs, eps, agents, estimators
    updates_50k.npz      every distinct-variable episode's per-step update of the posterior means (u_<agent>) and of the
                         beliefs' total variation (tv_<agent>); rev, fr_t, env, rg, rc, check_trained, check_echo
    split_pairs.npz      the scan's stream pairs that stay together and then split, and the three shown
    sweep100.npz         the hybrid observer's cell means at T with the joint evidence at the p% least / most entangled
                         stream-steps, 100 environments x 100 cells x 50 streams: cm, thr, fr, eps, seeds, sweep
The primary is record.npz: the random draws of the trained network's gradflow harvest (main/DATA/cache/
gradflow_trained.npz: 50,000 test episodes from numpy's global generator, which was never seeded, so no seed redraws
them). With the two checkpoints (main/DATA/sanity/fully_trained_ctx_2_e5_net.pth, reservoir_ctx_2_e5_net.pth) and
fixed seeds it determines every other file; build_record extracts it from the harvests when they are at hand.
The files were seeded on 27 Oct 2026 with byte copies (checked with cmp) of the research caches the kept figures were
rendered from (in main/DATA/experimental/: the research caches' folder, fig1c_compression/ and its figdata/round4b,
round6_hidden and traj_fr_groups); record.npz was extracted from the harvests and episodes.npz built here, both
bit-identical to the harvest's members.

Seeds: every draw but the harvest's is seeded (numpy default_rng), so a rebuild draws no new environment or stream:
the environments are the harvest's episodes (record.npz), and these seeds fix everything drawn from them
    scan_envs        42                    80 of the distinct (likelihood table, goal) among the first 20,000
    scan             1000 + e              environment e's 100 cells x 50 streams (stream s: cell s // 50)
    scan, Lpert      100 + e               the perturbed tables (read by no figure)
    alias_traj       1                     episode ENV = 1434 (goal variable 31, contextual 2): CELLS x 2000 streams
    distances_pairs  7,000,000 + episode   110 environments, evenly spaced over the distinct pairs (no draw)
    sweep100         2026; 1000 + i,       the order in which new pairs are taken; their streams
                     1080 + j
    traj_fr          0                     per FR group, 50 variable pairs, then one episode of each
    split, row 3     0                     the trajectories' drawing order
The split pairs are selected without a draw; shown: scan environment 15 (episode 566; goal variable 13, contextual
20; streams 507 and 4885, cells (1, 0) and (9, 7); split step 6), 39 (episode 1795; 10, 19; streams 491, 4500; (0, 9),
(9, 0); 8) and 20 (episode 1034; 7, 44; streams 23, 4445; (0, 0), (8, 8); 12). The PCA planes are deterministic
(scikit-learn's covariance_eigh solver for these shapes). --seeds writes every draw and every environment it produced
(episode, goal and contextual variables, stream seed) to seeds.json in the data folder; a copy is kept beside this
script, fig1c_seeds.json.

Regenerating: a full regeneration from record.npz alone (--data DIR, DIR holding only record.npz; or --rebuild-all in
place) took 37 min of wall time on the CPU at the script's defaults on 27 Oct 2026: scan 14.4, distances_pairs 7.0
(6 workers), updates 2.6, sweep100 1.9, beliefs 1.5, beliefs_steps 1.4, alias_traj 0.6, split_pairs 0.4, then the
six targets ~11. Against the seeded files:
    byte for byte  beliefs, scan_meta, scan_Joint, scan_Naive, distances_pairs, sweep100 (whole files); episodes,
                   scan_envs (eps, goal), alias_traj's Joint, Naive, rev, cells; updates_50k's Bayes members;
                   split_pairs' selection (every pair, and the three shown); --seeds (the environments)
    ~1e-15         distances_fr's FRb (MKL's rounding depends on the thread count)
    1e-6..6e-5     the networks wherever the seeded file replayed them on GPU 0 or at another thread count:
                   beliefs_steps (<= 1.8e-5; answers differ in 4 of 1.2 M stream-steps), alias_traj (<= 7.5e-6; 0),
                   updates_50k (<= 7.5e-5), distances_fr's network distances (<= 6.3e-5); the scan's networks within
                   one float16 step (4.9e-4; answers differ in 40 of 12 M)
The figures drawn from the regenerated caches: sweep byte for byte; summary_fold 605 and summary_updates 56 pixels
(<= 2/255); traj_FR 2078 pixels (<= 2/255); pcs <= 196, envdist <= 368, split <= 4754 pixels per frame (<= 0.25%): the
network panels' points and lines move, and so do a few entries of the GIFs' median-cut palettes (by one unit), which
re-maps anti-aliased pixels in the other panels by <= 30/255.
The replays are op for op the gradflow harvest's hand-written fp32 LSTM (lstm_input, forward_states, then the read-out;
belief_t = per-context softmax of the summed read-out logits in float64), so at equal thread counts they equal the
original replay code bit for bit; weights are read straight from the checkpoints.
Threads: the plane fits (belief_geometry, step_planes) always run at FIT_THREADS = 3 BLAS threads (fit_threads): their
last bits change with MKL's thread count, and with them a few anti-aliased pixels of the split movie, whose kept file
was made at 3 threads. Everything else follows OMP_NUM_THREADS (run as a script: 8 when it is not set); the replays use
the builders' torch thread counts.

Run from the repository root with the eureka environment's python (the default python cannot parse the repo; headless,
CPU only):
    python main/REPLICATION/plot_fig1c_all.py [target ...] [--out DIR] [--data DIR] [--rebuild cache [...] |
                                              --rebuild-all] [--list]
        target         summary, traj_fr, pcs, envdist, sweep, split, or a group: stills, movies, all (default: all)
        --out          write the figures into DIR instead of main/REPLICATION/figure_bin (to compare with the kept ones)
        --data         read and build the caches in DIR instead of main/DATA/experimental/fig1c; a folder holding only
                       record.npz regenerates every cache from the primary, leaving the seeded files untouched
        --rebuild      rebuild the named caches (at least one) before making the targets; the caches that depend on
                       them are not rebuilt unless named
        --rebuild-all  rebuild every cache but the record (overwrites the seeded files: the network caches then differ
                       from them by ~1e-5, see above, and so do a few pixels of the network figures)
        --list         print the targets and the caches with their status, then exit
        --seeds [PATH] write the seeds and the environments they produced as JSON (default: seeds.json in the data
                       folder), then exit
From python, with main/REPLICATION on sys.path:
    import plot_fig1c_all as P; P.figure_split(out_dir="..."); P.ensure("scan"); P.list_status()"""
import contextlib
import ctypes
import io
import os
import struct
import sys
import time
import traceback
import warnings
import zipfile

if __name__ == "__main__":                                                        # a script: headless, and BLAS
    os.environ.setdefault("MPLBACKEND", "Agg")                                    # threads capped unless set (read
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):    # when numpy loads)
        os.environ.setdefault(_v, os.environ.get("OMP_NUM_THREADS", "8"))
import numpy as np                                                                # noqa: E402


def _root():
    for start in (globals().get("__file__"), globals().get("__vsc_ipynb_file__"), os.getcwd()):
        d = os.path.abspath(start) if start and os.path.exists(start) else None
        while d:
            if os.path.exists(os.path.join(d, "main", "CognitiveGridworld.py")):
                return d
            d = None if os.path.dirname(d) == d else os.path.dirname(d)
    raise RuntimeError("project root not found")


ROOT = _root()
sys.path[:0] = [p for p in (ROOT, os.path.join(ROOT, "main", "REPLICATION")) if p not in sys.path]
import matplotlib.pyplot as plt                                                   # noqa: E402
import matplotlib.patheffects as pe                                               # noqa: E402
from matplotlib.collections import LineCollection                                 # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, SymLogNorm, to_rgb  # noqa: E402
from matplotlib.lines import Line2D                                               # noqa: E402
from PIL import Image                                                             # noqa: E402
import plotting_helpers as ph                                                     # noqa: E402


# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

DATA_DIR = os.path.join(ROOT, "main", "DATA", "experimental", "fig1c")
CHECKPOINTS = {"trained": os.path.join(ROOT, "main", "DATA", "sanity", "fully_trained_ctx_2_e5_net.pth"),
               "echo": os.path.join(ROOT, "main", "DATA", "sanity", "reservoir_ctx_2_e5_net.pth")}
HARVESTS = {"trained": os.path.join(ROOT, "main", "DATA", "cache", "gradflow_trained.npz"),
            "echo": os.path.join(ROOT, "main", "DATA", "cache", "gradflow_echo.npz")}
NET_NAME = {"trained": "Trained", "echo": "Echo"}
AGENTS4 = ("Joint", "Trained", "Naive", "Echo")                                   # the order of the stored agent axes
N_ALL, N_EP, N_ECHO, BATCH = 50000, 20000, 20000, 1000                            # harvest episodes; those of the
                                                                                  # planes and beliefs; ESN-harvest
                                                                                  # episodes kept; streams per batch
N_T, N_R, N_OBS = 30, 10, 5                                                       # steps, values, observables
STEPS = (0, 29)                                                                   # t = 1 and T = 30 (indices)
TS = tuple(s + 1 for s in STEPS)
ENV = 1434                                                                        # alias_traj: harvest episode ENV
CELLS = ((0, 2), (5, 8), (9, 3), (9, 7))                                          # and its four cells (r_g, r_c)
ALIAS_M, ALIAS_SEED = 2000, 1                                                     # streams per cell, seed
SCAN_N_ENV, SCAN_M, SCAN_SEED0, SCAN_PSEED0 = 40, 50, 1000, 100                   # environments, streams per cell,
                                                                                  # seeds
SCAN_SIGMA, SCAN_CLIP, FR_EPS = .8, 1e-6, 1e-15                                   # the perturbed table; FR's clip
SCAN_CELLS = [(g, c) for g in range(N_R) for c in range(N_R)]                     # goal-ordered, k = 10 r_g + r_c
SCAN_AGENTS = ("Joint", "Naive", "Trained", "Echo")
SCAN_FILES = ("scan_meta.npz",) + tuple(f"scan_{a}.npz" for a in SCAN_AGENTS)
SCAN_LEFT_OUT = 2                                                                 # e = 2: an exactly symmetric table
PAIR_N_ENV, PAIR_M, PAIR_SEED0 = 110, 10, 7_000_000
PAIR_CELLS = np.array([(rg, rc) for rg in range(N_R) for rc in range(N_R)])
PAIR_SQ = np.unique([a * a + b * b for a in range(N_R) for b in range(N_R)])      # squared true distances (51)
PAIR_SQ_IDX = np.full(PAIR_SQ.max() + 1, -1)
PAIR_SQ_IDX[PAIR_SQ] = np.arange(len(PAIR_SQ))
SWEEP = np.arange(0, 101, 2)                                                      # % of an environment's stream-steps
SWEEP_N_ENV, SWEEP_SEED, SWEEP_M = 100, 2026, 50
SPLIT_WINDOWS = ((11, 22), (8, 10), (6, 7))                                       # split_pairs: t_s windows shown
FR_GROUP_SUB, FR_GROUP_Q, FR_GROUP_N, FR_GROUP_SEED = (0, 3, 6, 9), .10, 50, 0     # fr_group_episodes' draw
FIT_THREADS = 3                                                                   # BLAS threads of the plane fits
FIG_DIR = os.path.join(ROOT, "main", "REPLICATION", "figure_bin")                 # main.utils.fig_path's folder
_MEMO = {}                                                                        # values read once per process


# ═══════════════════════════════════════════════════════════════════
# Data: files (paths, lazy members, atomic writes; ensure builds what is missing)
# ═══════════════════════════════════════════════════════════════════

def data_path(fname, folder=None):
    """a file's path in the data folder (or in folder)"""
    return os.path.join(folder or DATA_DIR, fname)


def npz_member(path, key):
    """one member of an .npz: a read-only memory map when it is stored uncompressed (np.savez), without reading the
    rest of the archive; else loaded"""
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(key + ".npy")
    if info.compress_type != 0:
        with np.load(path) as f:
            return f[key]
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        h = fh.read(30)
        assert h[:4] == b"PK\x03\x04", f"{path}: no local header for {key}"
        n, m = struct.unpack("<HH", h[26:30])
        fh.seek(info.header_offset + 30 + n + m)
        version = np.lib.format.read_magic(fh)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read(fh)
        offset = fh.tell()
    if shape == ():
        return np.memmap(path, dtype=dtype, mode="r", shape=(1,), offset=offset)[0]
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset, order="F" if fortran else "C")


def _save(path, compressed=False, **arrays):
    """np.savez(_compressed) to path through a temporary file in the same folder (os.replace: never half-written)"""
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(folder, f".{os.path.basename(path)[:-4]}.{os.getpid()}.tmp.npz")
    try:
        (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"saved {path}", flush=True)
    return path


def ensure(name, rebuild=False):
    """the path of cache name (a list of paths for the scan), built first when it is missing (or rebuild); its
    dependencies are built when they are missing; after a build the values memoized from the caches are dropped"""
    files, builder, deps = CACHES[name]
    paths = [data_path(f) for f in files]
    if rebuild or not all(os.path.exists(p) for p in paths):
        for d in deps:
            ensure(d)
        t0 = time.time()
        print(f"building {name} ({', '.join(files)})", flush=True)
        builder()
        _MEMO.clear()                                                             # values read from the old files
        print(f"built {name} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return paths[0] if len(paths) == 1 else paths


def load_cache(name):
    """np.load of a single-file cache (built when missing)"""
    return np.load(ensure(name))


# ═══════════════════════════════════════════════════════════════════
# Data: the episode record, the checkpoints and the environment side of the episodes
# ═══════════════════════════════════════════════════════════════════

RECORD_INT = ("ep_ctx_inds", "ep_goal_ind", "ep_ctx_vals", "echo_ep_ctx_vals")    # uint8 in the file, int64 decoded
RECORD_OBS = ("ep_obs_flat", "echo_ep_obs_flat")                                  # bool in the file, float32 decoded


def build_record(folder=None):
    """record.npz from the gradflow harvests (main/DATA/cache/gradflow_{trained,echo}.npz, members read alone): the
    trained harvest's ep_ctx_inds, ep_goal_ind, ep_ctx_vals, ep_obs_flat, ep_active_Z for all 50,000 episodes and the
    ESN harvest's ep_ctx_vals, ep_obs_flat, ep_active_Z for its first N_ECHO; indices and values as uint8, observations
    as bool, Z as stored; every decoded member is checked equal to the harvest's. About 1 s, 3.2 MB."""
    for net, p in HARVESTS.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} is missing: record.npz is extracted from it, and its episodes were drawn "
                                    "from an unseeded generator, so nothing else can recreate them")
    src = {"trained": (HARVESTS["trained"], N_ALL, ""), "echo": (HARVESTS["echo"], N_ECHO, "echo_")}
    keys = {"trained": ("ep_ctx_inds", "ep_goal_ind", "ep_ctx_vals", "ep_obs_flat", "ep_active_Z"),
            "echo": ("ep_ctx_vals", "ep_obs_flat", "ep_active_Z")}
    out = {}
    for net, (p, n, pre) in src.items():
        for k in keys[net]:
            v = np.array(npz_member(p, k)[:n])
            if pre + k in RECORD_INT:
                enc = v.astype(np.uint8)
            elif pre + k in RECORD_OBS:
                enc = v.astype(bool)
            else:
                enc = v
            assert np.array_equal(_decode(pre + k, enc), v) and _decode(pre + k, enc).dtype == v.dtype, k
            out[pre + k] = enc
    out["n_episodes"] = np.array([N_ALL, N_ECHO])
    return _save(data_path("record.npz", folder), compressed=True, **out)


def _decode(key, v):
    if key in RECORD_INT:
        return v.astype(np.int64)
    if key in RECORD_OBS:
        return v.astype(np.float32)
    return v


def record():
    """the episode record in the harvest's dtypes (indices and values int64, observations float32 0 / 1, Z float32),
    read-only, read once per process"""
    if "record" not in _MEMO:
        with np.load(ensure("record")) as f:
            rec = {k: _decode(k, f[k]) for k in f.files}
        for v in rec.values():
            v.flags.writeable = False
        _MEMO["record"] = rec
    return _MEMO["record"]


def checkpoint(net):
    """a network's checkpoint (main/DATA/sanity/*_net.pth, on the CPU): its state embeddings K, Q (float64) and the
    float32 weights of the read-in, LSTM, initial state (STM, LTM) and read-out"""
    if ("ckpt", net) not in _MEMO:
        import torch
        d = torch.load(CHECKPOINTS[net], map_location="cpu", weights_only=False)
        _MEMO[("ckpt", net)] = {"K": d["K"], "Q": d["Q"], "W": {k: v.detach().float() for k, v in d["weights"].items()}}
    return _MEMO[("ckpt", net)]


def _gridworld(batch):
    """a CognitiveGridworld environment without a network, for batches of `batch` episodes, with the trained
    checkpoint's state embeddings (the harvest's); its constructor draws a dummy batch from numpy's global generator,
    whose state is restored"""
    from main.CognitiveGridworld import CognitiveGridworld
    state = np.random.get_state()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            env = CognitiveGridworld(mode=None, show_plots=False, episodes=1, batch_num=batch, step_num=N_T,
                                     state_num=500, realization_num=N_R, obs_num=N_OBS, ctx_num=2, hid_dim=1000,
                                     learn_embeddings=False)
    finally:
        np.random.set_state(state)
    ck = checkpoint("trained")
    env.all_K, env.all_Q = ck["K"], ck["Q"]
    return env


def _approx_lik(b, eps=1e-99):
    """the history-conditioned update of beliefs b (n, T, ..., R): b_1, then b_t / b_{t-1} normalized"""
    b = np.maximum(b.astype(np.float64), eps)
    px = np.zeros_like(b)
    px[:, 0], r = b[:, 0], b[:, 1:] / b[:, :-1]
    px[:, 1:] = r / r.sum(-1, keepdims=True)
    return px


def _step_dkl(p, q, eps=1e-99):
    """0.5 [KL(p || q) + KL(q || p)] over the last axis, both floored and normalized"""
    p, q = np.maximum(p.astype(np.float64), eps), np.maximum(q.astype(np.float64), eps)
    p, q = p / p.sum(-1, keepdims=True), q / q.sum(-1, keepdims=True)
    return 0.5 * np.sum(p * (np.log(p) - np.log(q)) + q * (np.log(q) - np.log(p)), -1)


def env_side(rows, env=None):
    """the environment side of the record's episodes rows (a range): the likelihood tables and both Bayes observers
    from main.CognitiveGridworld (numpy, float64), entanglement rev as the gradflow harvest computes it (the mean over
    both contexts of 0.5 Jeffreys(the Naive's step likelihood, the Optimal's history-conditioned update)) -> the
    episodes.npz members for these rows"""
    rows = np.arange(rows.start, rows.stop) if isinstance(rows, range) else np.asarray(rows)
    env = env if env is not None else _gridworld(len(rows))
    assert env.batch_num == len(rows)
    rec = record()
    env.ctx_inds, env.goal_ind, env.ctx_vals = (rec[k][rows].astype(int) for k in ("ep_ctx_inds", "ep_goal_ind",
                                                                                  "ep_ctx_vals"))
    env.goal_value = env.ctx_vals[env.batch_range, env.goal_ind]
    env.default_likelihoods()
    env.obs_flat = rec["ep_obs_flat"][rows].astype(bool)
    with contextlib.redirect_stdout(io.StringIO()):
        env.run_inference()
    rev = _step_dkl(env.naive_px / env.naive_px.sum(-1, keepdims=True),
                    _approx_lik(env.joint_belief.astype(np.float64))).mean(-1)
    return {"ep_joint_likelihood": env.joint_likelihood.astype(np.float64),
            "ep_joint_belief": env.joint_belief.astype(np.float64),
            "ep_naive_belief": env.naive_belief.astype(np.float64), "rev": rev.astype(np.float32)}


def build_episodes(folder=None):
    """episodes.npz: env_side for all 50,000 episodes in the harvest's batches of 1000 (uncompressed, 686 MB, so
    members memory-map); bit-identical to the harvest's ep_joint_likelihood, ep_joint_belief, ep_naive_belief and rev.
    CPU, numpy, about 15 s"""
    env, out = _gridworld(BATCH), None
    for b0 in range(0, N_ALL, BATCH):
        part = env_side(range(b0, b0 + BATCH), env)
        if out is None:
            out = {k: np.empty((N_ALL,) + v.shape[1:], v.dtype) for k, v in part.items()}
        for k, v in part.items():
            out[k][b0:b0 + BATCH] = v
    return _save(data_path("episodes.npz", folder), **out)


def episodes(key):
    """one member of episodes.npz (built when missing) as a read-only memory map over all 50,000 episodes"""
    return npz_member(ensure("episodes"), key)


# ═══════════════════════════════════════════════════════════════════
# Data: network replays (gradflow_analysis' hand-written LSTM on the checkpoints' weights)
# ═══════════════════════════════════════════════════════════════════

def _logits(W, obs, Z):
    """one batch's read-out logits (B, T, 2 R): read-in relu(W_in [o_t, Z] + b), the LSTM written out (gate order i,
    f, g, o; initial state STM, LTM), read-out per step; exactly the ops of gradflow_analysis' lstm_input /
    forward_states and the model's read-out, in fp32"""
    import torch
    F = torch.nn.functional
    B, T = obs.shape[:2]
    Zx = Z.float().reshape(B, 1, -1).expand(-1, T, -1)
    x = torch.relu(F.linear(torch.cat((obs.float(), Zx), -1), W["classifier_readin.weight"],
                            W["classifier_readin.bias"]))
    W_ih, W_hh = W["LSTM.weight_ih_l0"], W["LSTM.weight_hh_l0"]
    b = W["LSTM.bias_ih_l0"] + W["LSTM.bias_hh_l0"]
    H = x.shape[-1]
    u = W["STM"].reshape(1, H).expand(B, H)
    c = W["LTM"].reshape(1, H).expand(B, H)
    xa = x @ W_ih.T + b
    U = torch.empty(B, T, H)
    for s in range(T):
        i, f, g, o = (xa[:, s] + u @ W_hh.T).chunk(4, -1)
        i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
        c = f * c + i * g
        tc = c.tanh()
        u = o * tc
        U[:, s] = u
    return F.linear(U, W["classifier_readout.weight"], W["classifier_readout.bias"])


def replay_beliefs(net, obs, Z, steps=range(N_T), threads=None, batch=BATCH):
    """network net's ("trained" or "echo") beliefs on streams obs (n, T, 5) with context input Z (n, 5, 2): per context
    the softmax over the values of the summed read-out logits r_1 + ... + r_t (float64, as the training loss) at the
    steps -> (n, len(steps), 2, R) float32, raw context order. In batches of `batch` streams (results depend on the
    batch's composition, the device and the torch thread count at the float32 rounding level); threads: torch threads
    for this call (default: as set)"""
    import torch
    W = checkpoint(net)["W"]
    old, out = torch.get_num_threads(), []
    if threads:
        torch.set_num_threads(threads)
    try:
        with torch.no_grad():
            for b0 in range(0, len(obs), batch):
                o = torch.from_numpy(np.array(obs[b0:b0 + batch], np.float32))
                z = torch.from_numpy(np.array(Z[b0:b0 + batch], np.float32))
                r = _logits(W, o, z)
                q = r.double().cumsum(1).reshape(len(o), N_T, 2, N_R).softmax(-1)
                out.append(q[:, list(steps)].float().numpy())
    finally:
        torch.set_num_threads(old)
    return np.concatenate(out)


def _goal_ordered(b, goal):
    """raw-order beliefs (n, T, 2, R) -> goal-ordered (n, T, 2R); goal: per stream (n,) or one int"""
    if np.ndim(goal) == 0:
        return np.concatenate([b[:, :, goal], b[:, :, 1 - goal]], -1)
    ar = np.arange(len(b))
    return np.concatenate([b[ar, :, goal], b[ar, :, 1 - goal]], -1)


def build_beliefs(threads=12, folder=None):
    """beliefs.npz: at t = 1 and T = 30 on the first N_EP episodes, the trained network and the ESN on the trained
    harvest's episodes (trained, echo_on_trained) and the ESN on its own harvest's (echo); ctx_trained, ctx_echo. CPU;
    the seeded file was made at 12 torch threads, so 12 reproduce it bit for bit (other counts: within ~2e-6); 60 replay
    batches (about 5 min at 2 threads)"""
    rec, n = record(), N_EP
    obs, Z = rec["ep_obs_flat"][:n], rec["ep_active_Z"][:n]
    res = {"trained": replay_beliefs("trained", obs, Z, STEPS, threads),
           "echo": replay_beliefs("echo", rec["echo_ep_obs_flat"][:n], rec["echo_ep_active_Z"][:n], STEPS, threads),
           "echo_on_trained": replay_beliefs("echo", obs, Z, STEPS, threads),
           "ctx_trained": np.array(rec["ep_ctx_vals"][:n]), "ctx_echo": np.array(rec["echo_ep_ctx_vals"][:n]),
           "steps": np.array(STEPS)}
    return _save(data_path("beliefs.npz", folder), **res)


def build_beliefs_steps(threads=None, folder=None):
    """beliefs_steps.npz: both networks at every step on the trained harvest's first N_EP episodes, goal-ordered
    float32 (the seeded file was replayed on GPU 0: a CPU rebuild differs by up to ~1e-5, no answer changes); 40 replay
    batches (about 3.5 min at 2 threads)"""
    rec, n = record(), N_EP
    goal = rec["ep_goal_ind"][:n].astype(int)
    out = {}
    for name, net in (("Trained", "trained"), ("Echo", "echo")):
        b = replay_beliefs(net, rec["ep_obs_flat"][:n], rec["ep_active_Z"][:n], range(N_T), threads)
        out[name] = _goal_ordered(b, goal).astype(np.float32)
    return _save(data_path("beliefs_steps.npz", folder), **out)


# ═══════════════════════════════════════════════════════════════════
# Data: exact observers on simulated streams of one environment
# ═══════════════════════════════════════════════════════════════════

def cell_streams(L, goal, cells, M, seed):
    """M streams per goal-ordered cell (r_g, r_c) of the table L (5, R, R; raw order L[i, r0, r1]) with goal context
    goal: o_{t,i} = 1 with probability L[i, r0, r1], (r0, r1) = (r_c, r_g) if goal == 1 else (r_g, r_c), from
    default_rng(seed).random((n, T, 5)) -> obs (n, T, 5) bool, cell_of (n,) (streams contiguous per cell)"""
    cell_of = np.repeat(np.arange(len(cells)), M)
    r01 = [(rc, rg) if goal == 1 else (rg, rc) for rg, rc in cells]
    obs = np.random.default_rng(seed).random((len(cell_of), N_T, len(L))) < \
        np.stack([L[:, a, b] for a, b in r01])[cell_of][:, None]
    return obs, cell_of


def exact_bayes(obs, L):
    """the exact observers on streams obs (n, T, 5) under the table L, float64, raw context order: the joint
    log-likelihood terms lJ (n, T, R, R); the Optimal's marginals (n, T, 2, R) (posterior over the 100 cells, uniform
    prior); the Naive's per-step log-likelihoods lp (n, T, 2, R) (each context's table averaged over the other value)"""
    o = obs.astype(np.float64)
    lJ = np.einsum("nti,ijk->ntjk", o, np.log(L)) + np.einsum("nti,ijk->ntjk", 1 - o, np.log1p(-L))
    cum = np.cumsum(lJ, 1)
    post = np.exp(cum - cum.max((-1, -2), keepdims=True))
    post /= post.sum((-1, -2), keepdims=True)
    joint = np.stack([post.sum(-1), post.sum(-2)], 2)
    Lb = (L.mean(2), L.mean(1))
    lp = np.stack([o @ np.log(Lb[k]) + (1 - o) @ np.log1p(-Lb[k]) for k in (0, 1)], 2)
    return lJ, joint, lp


def _entanglement(px, joint):
    """rev (n, T): the mean over both contexts of 0.5 Jeffreys(the Naive's step likelihood px, the Optimal's
    history-conditioned update)"""
    return _step_dkl(px / px.sum(-1, keepdims=True),
                     np.stack([_approx_lik(joint[:, :, k]) for k in (0, 1)], 2)).mean(-1)


def _env_table(EP):
    """harvest episode EP's likelihood table (5, R, R) float64 and goal context"""
    return np.array(episodes("ep_joint_likelihood")[EP], np.float64), int(record()["ep_goal_ind"][EP])


def simulate_cells(EP, cells, M, seed, nets=("trained", "echo"), threads=None, net_streams=None):
    """M streams per cell of environment EP (harvest episode EP: its table, goal context and context input Z), seed:
    the Optimal and the Naive (multiplied step likelihoods) exactly, the networks replayed on the same streams (the
    first net_streams only, if given) -> ({agent: (n, T, 20) goal-ordered float32}, cell_of, rev (n, T) float32, and
    for the scan obs (n, T, 5) bool, the Optimal's float64 marginals (n, T, 2, R))"""
    L, g = _env_table(EP)
    obs, cell_of = cell_streams(L, g, cells, M, seed)
    _, joint, lp = exact_bayes(obs, L)
    px = np.exp(lp)
    naive = np.cumprod(px, 1)
    rev = _entanglement(px, joint)
    B = {"Joint": joint, "Naive": naive / naive.sum(-1, keepdims=True)}
    k = len(cell_of) if net_streams is None else net_streams
    Z = np.repeat(np.asarray(record()["ep_active_Z"][EP:EP + 1]), k, 0)
    for net in nets:
        B[NET_NAME[net]] = replay_beliefs(net, obs[:k].astype(np.float32), Z, range(N_T), threads)
    out = {a: _goal_ordered(b, g).astype(np.float32) for a, b in B.items()}
    return out, cell_of, rev.astype(np.float32), obs, joint


def build_alias_traj(threads=None, folder=None):
    """alias_traj.npz: simulate_cells for ENV's CELLS, ALIAS_M streams each, seed ALIAS_SEED (the seeded file was
    replayed on GPU 0; a CPU rebuild: Joint, Naive, rev, cell_of bit for bit, networks within ~1e-5, no answer
    changes); 16 replay batches (about 1.5 min at 2 threads)"""
    out, cell_of, rev, _, _ = simulate_cells(ENV, CELLS, ALIAS_M, ALIAS_SEED, threads=threads)
    return _save(data_path("alias_traj.npz", folder), **out, cell_of=cell_of, cells=np.array(CELLS), rev=rev)


# ═══════════════════════════════════════════════════════════════════
# Data: the alias scan (40 environments x 100 cells x 50 streams)
# ═══════════════════════════════════════════════════════════════════

def scan_environments(n=80, seed=42):
    """the scan's environments: the first harvest episode of every distinct (likelihood table rounded to 6 decimals,
    goal context) among the first N_EP, n of them drawn by default_rng(seed) without replacement, sorted -> eps, goal"""
    Lall = np.array(episodes("ep_joint_likelihood")[:N_EP], np.float64)
    G = record()["ep_goal_ind"][:N_EP].astype(int)
    _, env = np.unique(np.round(Lall.reshape(N_EP, -1), 6), axis=0, return_inverse=True)
    env = env.ravel()
    first = {}
    for ep in range(N_EP):
        first.setdefault((env[ep], G[ep]), ep)
    rng = np.random.default_rng(seed)
    eps = sorted(rng.choice(sorted(first.values()), n, replace=False).tolist())
    return np.array(eps), G[eps]


def build_scan_envs(folder=None):
    """scan_envs.npz: eps, goal of scan_environments() (1 s; bit for bit)"""
    eps, goal = scan_environments()
    return _save(data_path("scan_envs.npz", folder), eps=eps, goal=goal)


def joint_marginals(obs, L):
    """the Optimal's marginals (n, T, 2, R) on streams obs, raw order, float64"""
    return exact_bayes(obs, L)[1]


def naive_marginals(obs, L):
    """the Naive's marginals in log space (n, T, 2, R), raw order, float64 (the scan's Naive; simulate_cells
    multiplies the step likelihoods instead, as the alias cache)"""
    o = obs.astype(np.float64)
    Lb = (L.mean(2), L.mean(1))
    cum = np.cumsum(np.stack([o @ np.log(Lb[k]) + (1 - o) @ np.log1p(-Lb[k]) for k in (0, 1)], 2), 1)
    b = np.exp(cum - cum.max(-1, keepdims=True))
    return b / b.sum(-1, keepdims=True)


def pert_table(L, e):
    """the perturbed-table control's model: sigmoid(logit(clip(L, CLIP, 1 - CLIP)) + SIGMA eps), eps ~ N(0, 1) per
    entry from default_rng(SCAN_PSEED0 + e)"""
    Lc = np.clip(L, SCAN_CLIP, 1 - SCAN_CLIP)
    eps = np.random.default_rng(SCAN_PSEED0 + e).standard_normal(L.shape)
    return 1 / (1 + np.exp(-(np.log(Lc) - np.log1p(-Lc) + SCAN_SIGMA * eps)))


def fr_sym(bj, bn, eps=FR_EPS):
    """factorization regret (the run's SII): [KL(bj || bn) + KL(bn || bj)] / 2 over the last axis, both clipped to
    [eps, 1 - eps], nats"""
    x, y = np.clip(bj, eps, 1 - eps), np.clip(bn, eps, 1 - eps)
    return .5 * ((x * np.log(x / y)).sum(-1) + (y * np.log(y / x)).sum(-1))


def scan_answers(V):
    """goal-ordered beliefs (..., 20) -> each marginal's argmax (..., 2) int8"""
    return np.stack([V[..., :N_R].argmax(-1), V[..., N_R:].argmax(-1)], -1).astype(np.int8)


def _cell_probs(L, goal, cells=SCAN_CELLS):
    r01 = [(rc, rg) if goal == 1 else (rg, rc) for rg, rc in cells]
    return np.stack([L[:, a, b] for a, b in r01])


def _top2(s):
    o = np.sort(s, -1)
    return s.argmax(-1), o[..., -1] - o[..., -2]


def naive_asymptote(L, goal, cells=SCAN_CELLS):
    """the Naive's asymptotic answer per cell (its KL projection) and the per-step log-likelihood gap to the runner-up
    -> (len(cells), 2) goal-ordered, (len(cells), 2) nats"""
    P = _cell_probs(L, goal, cells)
    Lb = (L.mean(2), L.mean(1))
    ans, gap = zip(*(_top2(P @ np.log(q) + (1 - P) @ np.log1p(-q)) for q in (Lb[goal], Lb[1 - goal])))
    return np.stack(ans, 1), np.stack(gap, 1)


def pert_asymptote(L, Lp, goal, cells=SCAN_CELLS):
    """the perturbed-table observer's asymptotic cell per true cell and its gap -> (len(cells), 2), (len(cells),)"""
    P, F = _cell_probs(L, goal, cells), Lp.reshape(len(Lp), -1)
    k, gap = _top2(P @ np.log(F) + (1 - P) @ np.log1p(-F))
    a, b = k // N_R, k % N_R
    return np.stack([b, a] if goal == 1 else [a, b], 1), gap


def scan_env(e, EP, threads=None):
    """scan environment e (harvest episode EP): SCAN_M streams per cell, seed SCAN_SEED0 + e; the Optimal and the
    (log-space) Naive exactly, the networks replayed, rev, FR, the asymptotes and the cross-checks -> the environment's
    rows of every scan_meta / scan_<agent> member (float16 beliefs, answers from full precision)"""
    L, goal = _env_table(EP)
    out, cell_of, rev, obs, J = simulate_cells(EP, SCAN_CELLS, SCAN_M, SCAN_SEED0 + e, threads=threads)
    Nv = naive_marginals(obs, L)
    Jg, Ng = _goal_ordered(J, goal), _goal_ordered(Nv, goal)
    dJ, dN = np.abs(Jg - out["Joint"]).max(), np.abs(Ng - out["Naive"]).max()
    assert dJ < 1e-5 and dN < 1e-5, f"scan environment {e}: the two Optimal / Naive computations differ"
    Lp = pert_table(L, e)
    n_inf, n_gap = naive_asymptote(L, goal)
    p_inf, p_gap = pert_asymptote(L, Lp, goal)
    full = {"Joint": Jg, "Naive": Ng, "Trained": out["Trained"], "Echo": out["Echo"]}
    part = {"obs": obs, "rev": rev, "FR": fr_sym(J[:, :, goal], Nv[:, :, goal]).astype(np.float32), "Lpert": Lp,
            "naive_inf": n_inf, "naive_inf_gap": n_gap, "pert_inf": p_inf, "pert_inf_gap": p_gap,
            "recreate_maxdiff": [float(dJ), float(dN)],
            "alias_answer_mismatch": [int((scan_answers(out[a]) != scan_answers(full[a])).any(-1).sum())
                                      for a in ("Joint", "Naive")]}
    for a, V in full.items():
        part[f"b_{a}"], part[f"a_{a}"] = V.astype(np.float16), scan_answers(V)
    return part


def build_scan(threads=None, folder=None):
    """scan_meta.npz and scan_{Joint,Naive,Trained,Echo}.npz (keys as the seeded files; the perturbed-table agent's
    beliefs, scan_Pert.npz, are not rebuilt, its table and asymptotes are): scan_env for the first SCAN_N_ENV of
    scan_envs. The seeded networks were replayed on GPU 0 (a CPU rebuild: Bayes, obs, rev, FR bit for bit; network
    float16 beliefs within one float16 step, answers differ in 0-1 of an environment's 150,000 stream-steps). 400
    replay batches plus the exact observers: 30-65 s per environment at 2 threads (20-45 min), peak ~2.5 GB"""
    with load_cache("scan_envs") as S:
        eps, goal_s = S["eps"][:SCAN_N_ENV].astype(int), S["goal"][:SCAN_N_ENV]
    G = record()["ep_goal_ind"].astype(int)
    assert np.array_equal(G[eps], goal_s)
    parts = []
    for e, EP in enumerate(eps):
        t0 = time.time()
        parts.append(scan_env(e, int(EP), threads))
        print(f"scan environment {e + 1}/{len(eps)} (episode {EP}): {time.time() - t0:.0f} s", flush=True)
    Lall = np.array(episodes("ep_joint_likelihood")[:N_EP], np.float64)
    _, tid = np.unique(np.round(Lall.reshape(N_EP, -1), 6), axis=0, return_inverse=True)
    meta = dict(eps=eps, goal=G[eps], table_id=tid.ravel()[eps], seed=SCAN_SEED0 + np.arange(len(eps)),
                pert_seed=SCAN_PSEED0 + np.arange(len(eps)), sigma=np.float64(SCAN_SIGMA), M=np.int64(SCAN_M),
                cells=np.array(SCAN_CELLS), cell_of=np.repeat(np.arange(len(SCAN_CELLS)), SCAN_M),
                L=np.stack([_env_table(int(EP))[0] for EP in eps]), Lpert=np.stack([p["Lpert"] for p in parts]))
    for k, dt in (("obs", bool), ("rev", np.float32), ("FR", np.float32), ("naive_inf", np.int8),
                  ("naive_inf_gap", np.float64), ("pert_inf", np.int8), ("pert_inf_gap", np.float64)):
        meta[k] = np.stack([p[k] for p in parts]).astype(dt)
    meta["recreate_maxdiff"] = np.array([p["recreate_maxdiff"] for p in parts])
    meta["alias_answer_mismatch"] = np.array([p["alias_answer_mismatch"] for p in parts])
    paths = [_save(data_path("scan_meta.npz", folder), **meta)]
    for a in SCAN_AGENTS:
        paths.append(_save(data_path(f"scan_{a}.npz", folder), beliefs=np.stack([p[f"b_{a}"] for p in parts]),
                           answer=np.stack([p[f"a_{a}"] for p in parts])))
    return paths


def scan_beliefs(agent):
    """the scan's goal-ordered beliefs of agent, (40, 5000, 30, 20) float16, as a read-only memory map"""
    return npz_member(ensure("scan")[1 + SCAN_AGENTS.index(agent)], "beliefs")


def scan_meta(key):
    """one member of scan_meta.npz (a memory map)"""
    return npz_member(ensure("scan")[0], key)


def scan_cell_means(agent, envs):
    """each cell's mean belief over its SCAN_M streams in the scan environments envs -> (len(envs), 100, T, 20)
    float32 (float16 beliefs cast to float32, then averaged)"""
    X = scan_beliefs(agent)[list(envs)].astype(np.float32)
    return X.reshape(len(envs), len(SCAN_CELLS), SCAN_M, N_T, 2 * N_R).mean(2)


# ═══════════════════════════════════════════════════════════════════
# Data: builders from the scan (distances by FR, split pairs)
# ═══════════════════════════════════════════════════════════════════

def _fr_both_T(L, goal, cells, obs):
    """per stream: FR_g + FR_c after all T steps, exact from the counts (L raw order, cells goal-ordered)"""
    r0, r1 = (cells[:, 1], cells[:, 0]) if goal == 1 else (cells[:, 0], cells[:, 1])
    P = np.clip(L[:, r0, r1].T, 1e-12, 1 - 1e-12)
    n = obs.sum(1).astype(np.float64)
    ll = n @ np.log(P).T + (N_T - n) @ np.log(1 - P).T
    post = np.exp(ll - ll.max(1, keepdims=True))
    post = (post / post.sum(1, keepdims=True)).reshape(-1, 10, 10)
    fr = 0.
    for opt, Pv in ((post.sum(2), P.reshape(10, 10, 5).mean(1)), (post.sum(1), P.reshape(10, 10, 5).mean(0))):
        lv = n @ np.log(Pv).T + (N_T - n) @ np.log(1 - Pv).T
        q = np.exp(lv - lv.max(1, keepdims=True))
        fr = fr + ph.jeffreys_half(opt, q / q.sum(1, keepdims=True), clip_top=False)
    return fr


def build_distances_fr(folder=None):
    """distances_fr.npz: per scan environment (e = 2 left out), cell and step, the mean over the cell's streams of the
    distance of the posterior means (E[r_g], E[r_c]) (float16 beliefs, each marginal renormalized) from the true cell;
    each cell's FR_g + FR_c at T (exact, float64), the cells ranked by it within the environment into 10 and 20
    equal-count bins (stable sort). Bit for bit except FRb / fr_median (~1e-14: MKL's rounding of the count-table
    products depends on its thread count; the seeded file's is unknown, 1 and 2 threads both differ); about 5 s"""
    with np.load(ensure("scan")[0]) as m:
        cells, cell_of, L, goal, obs = m["cells"], m["cell_of"], m["L"], m["goal"], m["obs"]
    n_env, n = obs.shape[:2]
    assert (cell_of == np.arange(n) // (n // 100)).all()
    pos = cells[cell_of].astype(np.float64)
    Rf = np.arange(10, dtype=np.float64)
    DISP = np.zeros((len(AGENTS4), n_env, 100, N_T))
    for a_, a in enumerate(AGENTS4):
        B = scan_beliefs(a)
        for e in range(n_env):
            b = np.asarray(B[e]).astype(np.float64).reshape(n, N_T, 2, 10)
            E = (b * Rf).sum(-1) / b.sum(-1)
            DISP[a_, e] = np.linalg.norm(E - pos[:, None], axis=-1).reshape(100, -1, N_T).mean(1)
    FRb = np.stack([_fr_both_T(L[e], int(goal[e]), cells, obs[e]).reshape(100, -1).mean(1) for e in range(n_env)])
    keep = np.arange(n_env) != SCAN_LEFT_OUT
    DISP, FRb = DISP[:, keep], FRb[keep]
    out = dict(FRb=FRb, agents=np.array(AGENTS4))
    for nbin in (10, 20):
        b = np.zeros_like(FRb, dtype=int)
        for e in range(len(FRb)):
            b[e, np.argsort(FRb[e], kind="stable")] = np.arange(100) * nbin // 100
        out[f"dist_t{nbin}"] = np.stack([np.stack([DISP[:, e, b[e] == q].mean(1) for q in range(nbin)], 1)
                                         for e in range(len(FRb))], 1)
        out[f"fr_median{nbin}"] = np.median([[np.median(FRb[e][b[e] == q]) for q in range(nbin)]
                                             for e in range(len(FRb))], 0)
    return _save(data_path("distances_fr.npz", folder), **out)


def split_pair_data():
    """the scan's stream pairs that split (selected in the Optimal): two streams of different cells in one
    environment (e = 2 left out), true cells >= 4 apart, posterior means within 0.6 from step 2 through t_s (within 0.4
    at t_s; t_s = 5..22) and >= 3 further apart two steps later, each pair once at its last t_s; per pair the jump, each
    stream's final distance to its true cell per agent and the split step's entanglement percentile; the three shown
    (distinct environments and cells, the largest jump in each SPLIT_WINDOWS window) -> the split_pairs.npz members"""
    from scipy.spatial import cKDTree
    with np.load(ensure("scan")[0]) as m:
        cells, cell_of, rev = m["cells"], m["cell_of"], m["rev"]
    envs = [e for e in range(SCAN_N_ENV) if e != SCAN_LEFT_OUT]
    R = np.arange(10)                                                             # int64: b * R is float64, as stored
    X, raw = {}, {}
    for a in AGENTS4:
        raw[a] = scan_beliefs(a)
        b = raw[a].astype(np.float32).reshape(*raw[a].shape[:3], 2, 10)
        X[a] = (b * R).sum(-1) / b.sum(-1)
        del b
    c_pos = cells[cell_of].astype(np.float32)
    cand = {}
    for e in envs:
        XJ = X["Joint"][e]
        for ts in range(5, 23):
            dd, jj = cKDTree(XJ[:, ts - 1]).query(XJ[:, ts - 1], k=25)
            i, j = np.repeat(np.arange(5000), 25), jj.ravel()
            ok = (cell_of[i] != cell_of[j]) & (dd.ravel() < .4) & (i < j)
            i, j = i[ok], j[ok]
            d = np.linalg.norm(XJ[i] - XJ[j], axis=-1)
            sel = ((d[:, 1:ts].max(1) < .6) & (d[:, ts + 1] - d[:, ts - 1] >= 3)
                   & (np.linalg.norm(c_pos[i] - c_pos[j], axis=1) >= 4))
            for i_, j_, d_ in zip(i[sel], j[sel], d[sel]):
                cand[(e, i_, j_)] = (ts, d_[ts + 1] - d_[ts - 1])
    keys = sorted(cand)
    e_, i_, j_ = (np.array([k[n] for k in keys]) for n in range(3))
    ts_ = np.array([cand[k][0] for k in keys])
    jump = np.array([cand[k][1] for k in keys])
    err = np.stack([np.stack([np.linalg.norm(X[a][e_, s, -1] - c_pos[s], axis=-1) for s in (i_, j_)], 1)
                    for a in AGENTS4])
    ent = (rev[e_, i_, ts_] + rev[e_, j_, ts_]) / 2
    pct = np.array([np.mean(rev[e][:, t] < v) for e, t, v in zip(e_, ts_, ent)])
    shown, used_env, used_cell = [], set(), set()
    for lo, hi in SPLIT_WINDOWS:
        for k in np.argsort(-jump):
            if lo <= ts_[k] <= hi and e_[k] not in used_env and not ({cell_of[i_[k]], cell_of[j_[k]]} & used_cell):
                shown.append((e_[k], i_[k], j_[k], ts_[k]))
                used_env.add(e_[k]), used_cell.update({cell_of[i_[k]], cell_of[j_[k]]})
                break
    shown.sort(key=lambda s: s[3])
    se, si, sj, sts = (np.array([s[n] for s in shown]) for n in range(4))
    P = np.stack([np.stack([X[a][se, s] for s in (si, sj)], 1) for a in AGENTS4])
    B = np.stack([np.stack([raw[a][se, s].astype(np.float32) for s in (si, sj)], 1) for a in AGENTS4])
    return dict(env=e_, i=i_, j=j_, ts=ts_, jump=jump, err=err, pct=pct, shown_env=se, shown_i=si, shown_j=sj,
                shown_ts=sts, P=P, B=B, c_i=c_pos[si], c_j=c_pos[sj], n_env=len(envs), agents=np.array(AGENTS4))


def build_split_pairs(folder=None):
    """split_pairs.npz (deflate): split_pair_data(); bit for bit (the file byte for byte) from the stored scan, about
    30 s, 2.8 GB peak"""
    return _save(data_path("split_pairs.npz", folder), compressed=True, **split_pair_data())


# ═══════════════════════════════════════════════════════════════════
# Data: distances between pairs of particles (110 environments, network replays)
# ═══════════════════════════════════════════════════════════════════

def pair_environments(n_env=PAIR_N_ENV):
    """one harvest episode per distinct unordered pair of different active variables (the first in the harvest),
    n_env of them evenly spaced -> episode indices"""
    ci = np.sort(record()["ep_ctx_inds"].astype(int), 1)
    seen, eps = set(), []
    for ep, (a, b) in enumerate(ci):
        if a != b and (a, b) not in seen:
            seen.add((a, b)), eps.append(ep)
    all_eps = np.array(eps)
    return all_eps[np.unique(np.linspace(0, len(all_eps) - 1, n_env).round().astype(int))]


def _pair_sums(B, g, cell_of):
    """sums of the estimate distance per discrete true distance and the pair counts: (4 agents, 2 estimators
    [posterior means, argmax], T, 51), (51,)"""
    Rf = np.arange(10, dtype=np.float64)
    pos = PAIR_CELLS[cell_of]
    iu = np.triu_indices(len(cell_of), 1)
    q = PAIR_SQ_IDX[((pos[iu[0]] - pos[iu[1]]) ** 2).sum(1)]
    cnt = np.bincount(q, minlength=len(PAIR_SQ)).astype(np.float64)
    S = np.zeros((len(AGENTS4), 2, N_T, len(PAIR_SQ)))
    for a_, a in enumerate(AGENTS4):
        b = B[a].reshape(len(cell_of), N_T, 2, 10)
        b = b / b.sum(-1, keepdims=True)
        b = np.stack([b[:, :, g], b[:, :, 1 - g]], 2)
        for e_, E in enumerate(((b * Rf).sum(-1), b.argmax(-1).astype(np.float64))):
            for s in range(N_T):
                S[a_, e_, s] = np.bincount(q, weights=np.sqrt(((E[iu[0], s] - E[iu[1], s]) ** 2).sum(1)),
                                           minlength=len(PAIR_SQ))
    return S, cnt


def pair_env(EP, threads=1):
    """environment EP's pair sums: PAIR_M streams per cell (seed PAIR_SEED0 + EP), the Optimal and the Naive exact, the
    networks replayed (one batch of 1000 streams each) -> _pair_sums"""
    L, g = _env_table(EP)
    obs, cell_of = cell_streams(L, g, PAIR_CELLS, PAIR_M, PAIR_SEED0 + EP)
    _, joint, lp = exact_bayes(obs, L)
    naive = np.cumprod(np.exp(lp), 1)
    B = {"Joint": joint.reshape(-1, N_T, 20), "Naive": (naive / naive.sum(-1, keepdims=True)).reshape(-1, N_T, 20)}
    Z = np.repeat(np.asarray(record()["ep_active_Z"][EP:EP + 1]), len(cell_of), 0)
    for net in ("trained", "echo"):
        B[NET_NAME[net]] = replay_beliefs(net, obs.astype(np.float32), Z, range(N_T), threads).reshape(-1, N_T, 20)
    return _pair_sums(B, g, cell_of)


def _pairs_worker(args):
    eps, threads = args
    res = [pair_env(int(EP), threads) for EP in eps]
    return np.array([r[0] for r in res]), res[0][1]


def build_distances_pairs(n_env=PAIR_N_ENV, workers=6, threads=1, folder=None):
    """distances_pairs.npz: pair_env for pair_environments(n_env), in `workers` forked processes of `threads` torch
    threads each (1 reproduces the seeded file bit for bit; other counts change the networks' float32 rounding and can
    flip argmax ties). 15-20 s per environment at 1 thread: about 30 CPU-min, ~6 min with 6 workers"""
    eps = pair_environments(n_env)
    record(), checkpoint("trained"), checkpoint("echo"), episodes("ep_joint_likelihood")   # read before forking
    parts = [(p, threads) for p in np.array_split(eps, workers) if len(p)]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(len(parts)) as pool:
            res = pool.map(_pairs_worker, parts)
    else:
        res = [_pairs_worker(p) for p in parts]
    S, cnt = np.concatenate([r[0] for r in res]), res[0][1]
    return _save(data_path("distances_pairs.npz", folder), ymean=(S / cnt).astype(np.float32), n=cnt,
                 xs=np.sqrt(PAIR_SQ), eps=eps, agents=np.array(AGENTS4), estimators=np.array(["mean", "argmax"]))


# ═══════════════════════════════════════════════════════════════════
# Data: per-step updates on all 50,000 episodes
# ═══════════════════════════════════════════════════════════════════

def update_beliefs(rows, threads=6):
    """goal-ordered float32 beliefs (len(rows), T, 20) of the four agents on harvest episodes rows (a range): the
    Optimal and the Naive from episodes.npz; the networks from beliefs_steps.npz for episodes < N_EP and replayed (in
    the harvest's batches of 1000, which rows must align to there) beyond"""
    rows = np.arange(rows.start, rows.stop)
    goal = record()["ep_goal_ind"][rows].astype(int)
    B = {a: _goal_ordered(np.array(episodes(k)[rows]), goal).astype(np.float32)
         for a, k in (("Joint", "ep_joint_belief"), ("Naive", "ep_naive_belief"))}
    lo, hi = rows[rows < N_EP], rows[rows >= N_EP]
    if len(hi):
        assert hi[0] % BATCH == 0, "replayed episodes must start a batch"
    for name, net in (("Trained", "trained"), ("Echo", "echo")):
        parts = []
        if len(lo):
            parts.append(np.array(npz_member(ensure("beliefs_steps"), name)[lo]))
        if len(hi):
            rec = record()
            b = replay_beliefs(net, rec["ep_obs_flat"][hi], rec["ep_active_Z"][hi], range(N_T), threads)
            parts.append(_goal_ordered(b, goal[rows >= N_EP]).astype(np.float32))
        B[name] = np.concatenate(parts)
    return B


def update_arrays(B, keep):
    """fr_t and the per-step updates u_<agent> (posterior means, E_0 = 4.5) and tv_<agent> (total variation, b_0
    uniform) of the kept episodes, from goal-ordered float32 beliefs B {agent: (n, T, 20)}"""
    out, R = {}, N_R
    J, Nv = B["Joint"].astype(np.float64), B["Naive"].astype(np.float64)
    fr = ph.jeffreys_half(J[..., :R], Nv[..., :R]) + ph.jeffreys_half(J[..., R:], Nv[..., R:])    # FR_g + FR_c
    out["fr_t"] = fr[keep].astype(np.float32)
    Rv = np.arange(R)
    for a in AGENTS4:
        b = B[a][keep].astype(np.float64).reshape(keep.sum(), N_T, 2, R)
        b /= b.sum(-1, keepdims=True)
        E = np.concatenate([np.full((len(b), 1, 2), (R - 1) / 2), (b * Rv).sum(-1)], 1)
        out["u_" + a] = np.diff(E, axis=1).astype(np.float32)
        b_ = np.concatenate([np.full((len(b), 1, 2, R), 1 / R), b], 1)
        out["tv_" + a] = (.5 * np.abs(np.diff(b_, axis=1)).sum(-1)).astype(np.float32)
    return out


def build_updates(threads=6, folder=None):
    """updates_50k.npz (deflate): every distinct-variable episode's per-step updates of the four agents (update_beliefs
    for all 50,000 episodes; the networks' episodes 20,000-49,999 replayed at `threads`: the seeded file used 6), rev,
    fr_t, env (index of the variable pair), rg, rc; check_trained / check_echo: max |diff| of a replay of episodes
    0-999 against beliefs_steps.npz. 62 replay batches plus the numpy and the compression: about 5 min at 2 threads,
    peak ~2 GB"""
    rec = record()
    goal, ar = rec["ep_goal_ind"][:N_ALL].astype(int), np.arange(N_ALL)
    B = update_beliefs(range(0, N_ALL), threads)
    checks = {}
    for name, net in (("Trained", "trained"), ("Echo", "echo")):
        b0 = replay_beliefs(net, rec["ep_obs_flat"][:BATCH], rec["ep_active_Z"][:BATCH], range(N_T), threads)
        checks[name] = float(np.abs(_goal_ordered(b0, goal[:BATCH]) - B[name][:BATCH]).max())
    ci = np.sort(rec["ep_ctx_inds"][:N_ALL].astype(int), 1)
    keep = ci[:, 0] != ci[:, 1]
    ctx = rec["ep_ctx_vals"][:N_ALL].astype(int)
    out = dict(env=np.unique(ci[keep], axis=0, return_inverse=True)[1].ravel(), rg=ctx[ar, goal][keep],
               rc=ctx[ar, 1 - goal][keep], rev=np.array(episodes("rev")[:N_ALL])[keep].astype(np.float32),
               agents=np.array(AGENTS4), check_trained=checks["Trained"], check_echo=checks["Echo"])
    out.update(update_arrays(B, keep))
    return _save(data_path("updates_50k.npz", folder), compressed=True, **out)


# ═══════════════════════════════════════════════════════════════════
# Data: the hybrid observer's threshold sweep on 100 environments
# ═══════════════════════════════════════════════════════════════════

def sweep_environments(n_env=SWEEP_N_ENV, seed=SWEEP_SEED):
    """harvest episodes and observation seeds: scan_envs' 80 (seed 1000 + its index, so its first 40 are the scan's
    environments) without same-variable environments and repeated variable pairs, then random harvest episodes with
    new pairs (default_rng(seed) permutation; seeds 1080 + j) up to n_env"""
    with load_cache("scan_envs") as f:
        pool = f["eps"]
    ci = np.sort(record()["ep_ctx_inds"], 1)
    eps, seeds, seen = [], [], set()
    for i, ep in enumerate(pool):
        pair = tuple(ci[ep])
        if pair[0] != pair[1] and pair not in seen:
            eps.append(int(ep)), seeds.append(1000 + i), seen.add(pair)
    rng = np.random.default_rng(seed)
    j = 0
    for ep in rng.permutation(len(ci)):
        if len(eps) >= n_env:
            break
        pair = tuple(ci[ep])
        if pair[0] != pair[1] and pair not in seen:
            eps.append(int(ep)), seeds.append(1080 + j), seen.add(pair)
            j += 1
    return np.array(eps), np.array(seeds)


def sweep_env(EP, seed):
    """environment EP's sweep: 100 cells x SWEEP_M streams (seed), the hybrid observer (a joint belief that takes
    the joint likelihood at the selected stream-steps and the Naive's factorized one elsewhere) with the joint evidence
    at the SWEEP % least (w = 0) / most (w = 1) entangled of the environment's stream-steps (nested sets) -> cell means
    of its goal-ordered beliefs at T (2, len(SWEEP), 100, 20) float32, the entanglement at each selection's edge
    (2, len(SWEEP)), each cell's goal-variable FR at T (100,)"""
    K, M, T, R = len(SCAN_CELLS), SWEEP_M, N_T, N_R
    L, g = _env_table(EP)
    obs, cell_of = cell_streams(L, g, SCAN_CELLS, M, int(seed))
    lJ, joint, lp = exact_bayes(obs, L)
    lN = lp[:, :, 0][..., :, None] + lp[:, :, 1][..., None, :]
    n = len(lJ)
    px = np.exp(lp)
    naive = np.cumprod(px, 1)
    naive /= naive.sum(-1, keepdims=True)
    rev = _entanglement(px, joint)
    frc = fr_sym(joint[:, -1, g], naive[:, -1, g]).reshape(K, M).mean(1)
    Dflat = (lJ - lN).reshape(n * T, R, R)
    base = lN.sum(1)
    flat = rev.ravel()
    desc = np.argsort(-flat)
    cm = np.zeros((2, len(SWEEP), K, 20), np.float32)
    thr = np.full((2, len(SWEEP)), np.nan)
    for w, order in enumerate((desc[::-1], desc)):
        Ssum, done = base.copy(), 0
        for j, pct in enumerate(SWEEP):
            nsel = int(round(n * T * pct / 100))
            new = order[done:nsel]
            np.add.at(Ssum, new // T, Dflat[new])
            done = nsel
            if nsel:
                thr[w, j] = flat[order[nsel - 1]]
            q = np.exp(Ssum - Ssum.max((-1, -2), keepdims=True))
            q /= q.sum((-1, -2), keepdims=True)
            m = np.stack([q.sum(-1), q.sum(-2)], 1)
            b = np.concatenate([m[:, g], m[:, 1 - g]], -1)
            if pct == 100:
                assert np.abs(b - np.concatenate([joint[:, -1, g], joint[:, -1, 1 - g]], -1)).max() < 1e-9
            cm[w, j] = b.reshape(K, M, 20).mean(1)
    return cm, thr, frc


def build_sweep100(n_env=SWEEP_N_ENV, folder=None):
    """sweep100.npz: sweep_env for sweep_environments(n_env): cm (2, levels, E, 100, 20), thr (2, levels, E),
    fr (E, 100), eps, seeds, sweep. numpy only, bit for bit; 1.6 s per environment (~3 min)"""
    eps, seeds = sweep_environments(n_env)
    cm = np.zeros((2, len(SWEEP), len(eps), len(SCAN_CELLS), 20), np.float32)
    thr = np.full((2, len(SWEEP), len(eps)), np.nan)
    frc = np.zeros((len(eps), len(SCAN_CELLS)))
    for ie, (EP, sd) in enumerate(zip(eps, seeds)):
        cm[:, :, ie], thr[:, :, ie], frc[ie] = sweep_env(int(EP), sd)
    return _save(data_path("sweep100.npz", folder), cm=cm, thr=thr, fr=frc, eps=eps, seeds=seeds, sweep=SWEEP)


# ═══════════════════════════════════════════════════════════════════
# Data: the registry (name: files, builder, the caches it reads)
# ═══════════════════════════════════════════════════════════════════

CACHES = {                                                     # name: (files, builder, the caches it reads)
    "record": (("record.npz",), build_record, ()),
    "episodes": (("episodes.npz",), build_episodes, ("record",)),
    "beliefs": (("beliefs.npz",), build_beliefs, ("record",)),
    "beliefs_steps": (("beliefs_steps.npz",), build_beliefs_steps, ("record",)),
    "alias_traj": (("alias_traj.npz",), build_alias_traj, ("record", "episodes")),
    "scan_envs": (("scan_envs.npz",), build_scan_envs, ("record", "episodes")),
    "scan": (SCAN_FILES, build_scan, ("record", "episodes", "scan_envs")),
    "distances_fr": (("distances_fr.npz",), build_distances_fr, ("scan",)),
    "distances_pairs": (("distances_pairs.npz",), build_distances_pairs, ("record", "episodes")),
    "updates": (("updates_50k.npz",), build_updates, ("record", "episodes", "beliefs_steps")),
    "split_pairs": (("split_pairs.npz",), build_split_pairs, ("scan",)),
    "sweep100": (("sweep100.npz",), build_sweep100, ("record", "episodes", "scan_envs")),
}


# ═══════════════════════════════════════════════════════════════════
# Analysis shared by the figures: belief planes, targets, beliefs at every step
# ═══════════════════════════════════════════════════════════════════

def _mkl():
    """numpy's MKL runtime library (ctypes) when its BLAS is MKL with thread-local counts, else None"""
    if "mkl" not in _MEMO:
        from threadpoolctl import threadpool_info
        lib = [d["filepath"] for d in threadpool_info() if d.get("internal_api") == "mkl"]
        mkl = ctypes.CDLL(lib[0]) if lib else None
        _MEMO["mkl"] = mkl if mkl is not None and hasattr(mkl, "MKL_Set_Num_Threads_Local") else None
    return _MEMO["mkl"]


@contextlib.contextmanager
def fit_threads(n=FIT_THREADS):
    """n BLAS threads for the calling thread while the block runs, whatever the environment or earlier calls set: the
    plane fits' last bits (PCA, ridge, correlations; up to ~1e-13) depend on MKL's thread count, and with them a few
    anti-aliased pixels of the split movie. MKL: its thread-local count, the previous one restored after (MKL's global
    count stops applying to the main thread once a LAPACK call such as eigh has run in the process); another BLAS:
    threadpoolctl's limit"""
    mkl = _mkl()
    if mkl is None:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=n, user_api="blas"):
            yield
        return
    old = mkl.MKL_Set_Num_Threads_Local(n)
    try:
        yield
    finally:
        mkl.MKL_Set_Num_Threads_Local(old)


def align_plane(X, w_g, w_c):
    """turn (rotate, and mirror where needed) a 2-PC plane so that the r_g decoder points right and the r_c decoder up:
    the decoders' bisector to 45 degrees, r_g on its clockwise side (orthogonal: distances, R^2 and the decoders' angle
    unchanged) -> X Q, Q^T w_g, Q^T w_c"""
    u_g, u_c = w_g / np.linalg.norm(w_g), w_c / np.linalg.norm(w_c)
    M = np.eye(2) if u_g[0] * u_c[1] - u_g[1] * u_c[0] >= 0 else np.diag([1., -1.])
    b = M @ (u_g + u_c)
    th = np.pi / 4 - np.arctan2(b[1], b[0])
    Q = (np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ M).T
    return X @ Q, Q.T @ w_g, Q.T @ w_c


def belief_geometry():
    """each agent's own PCs of its goal-ordered beliefs at t = 1 and T = 30 on the first N_EP episodes (the networks
    from beliefs.npz, the ESN on the trained network's episodes; the Bayes observers from episodes.npz): the 2 of PCs
    1-3 holding the most latent variance (single-PC R^2 of r_g plus r_c), ridge (alpha 1) decoders of r_g and r_c from
    them, the plane turned by align_plane -> {(agent, t): dict(X (N, 2), rg, rc, pcs, r2_g, r2_c, w_g, w_c, angle,
    proj = (PCA mean, the 2 components, Q))}; computed once per process, the fits at FIT_THREADS BLAS threads"""
    if "belief_geometry" in _MEMO:
        return _MEMO["belief_geometry"]
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    with np.load(ensure("beliefs")) as f:
        B = dict(f)
    N = len(B["trained"])
    rec = record()
    goal, ctx = rec["ep_goal_ind"][:N].astype(int), rec["ep_ctx_vals"][:N].astype(int)
    assert np.array_equal(ctx, B["ctx_trained"])
    ar = np.arange(N)
    rg, rc = ctx[ar, goal], ctx[ar, 1 - goal]
    agents = {"Trained": B["trained"], "Echo": B["echo_on_trained"],
              "Joint": np.array(episodes("ep_joint_belief")[:N][:, list(STEPS)]),
              "Naive": np.array(episodes("ep_naive_belief")[:N][:, list(STEPS)])}
    out = {}

    def latent_pcs(P3):
        lat = [sum(np.corrcoef(P3[:, q], y)[0, 1] ** 2 for y in (rg, rc)) for q in range(3)]
        return sorted(int(q) for q in np.argsort(lat)[-2:])

    with fit_threads():
        for i, t in enumerate(TS):
            for name, bel in agents.items():
                b = np.asarray(bel[:, i], np.float64)
                V = np.concatenate([b[ar, goal], b[ar, 1 - goal]], -1)
                pca = PCA(n_components=3).fit(V)
                P3 = pca.transform(V)
                pcs = latent_pcs(P3)
                X = P3[:, pcs]
                d = {"X": X, "rg": rg, "rc": rc, "pcs": tuple(q + 1 for q in pcs)}
                for k, y in (("g", rg), ("c", rc)):
                    m = Ridge(alpha=1.0).fit(X, y)
                    d[f"r2_{k}"], d[f"w_{k}"] = m.score(X, y), m.coef_
                d["angle"] = np.degrees(np.arccos(np.clip(d["w_g"] @ d["w_c"] / np.linalg.norm(d["w_g"]) /
                                                          np.linalg.norm(d["w_c"]), -1, 1)))
                Q = align_plane(np.eye(2), d["w_g"], d["w_c"])[0]
                d["X"], d["w_g"], d["w_c"] = X @ Q, Q.T @ d["w_g"], Q.T @ d["w_c"]
                d["proj"] = (pca.mean_, pca.components_[pcs], Q)
                out[(name, t)] = d
    _MEMO["belief_geometry"] = out
    return out


def project_plane(D, V):
    """beliefs V (..., 20) into a belief_geometry plane D, turned as it is drawn: (v - mean) C^T Q -> (..., 2)"""
    m, C, Q = D["proj"]
    return (np.asarray(V, np.float64) - m) @ C.T @ Q


def targets(cells):
    """where a correct belief about each cell sits: the Optimal's mean T = 30 belief (goal-ordered) over the first
    N_EP episodes in which it is right about both values of that cell -> (len(cells), 20), the episodes per cell"""
    rec = record()
    ar = np.arange(N_EP)
    goal, ctx = rec["ep_goal_ind"][:N_EP].astype(int), rec["ep_ctx_vals"][:N_EP].astype(int)
    b = np.asarray(episodes("ep_joint_belief")[:N_EP, STEPS[1]], np.float64)
    V = np.concatenate([b[ar, goal], b[ar, 1 - goal]], -1)
    rg, rc = ctx[ar, goal], ctx[ar, 1 - goal]
    ok = (V[:, :N_R].argmax(-1) == rg) & (V[:, N_R:].argmax(-1) == rc)
    sel = [ok & (rg == g) & (rc == c) for g, c in cells]
    return np.array([V[s].mean(0) for s in sel]), [int(s.sum()) for s in sel]


def beliefs_every_step():
    """each agent's goal-ordered beliefs (N_EP, T, 20) float32 at every step on the first N_EP episodes (the networks
    from beliefs_steps.npz, the Bayes observers from episodes.npz), and each episode's r_g, r_c; read once per
    process, read-only"""
    if "beliefs_every_step" not in _MEMO:
        with np.load(ensure("beliefs_steps")) as f:
            BA = dict(f)
        rec = record()
        goal, ctx = rec["ep_goal_ind"][:N_EP].astype(int), rec["ep_ctx_vals"][:N_EP].astype(int)
        ar = np.arange(N_EP)
        out = {"Trained": BA["Trained"], "Echo": BA["Echo"]}
        for a, k in (("Joint", "ep_joint_belief"), ("Naive", "ep_naive_belief")):
            b = episodes(k)[:N_EP]
            out[a] = np.concatenate([b[ar, :, goal], b[ar, :, 1 - goal]], -1).astype(np.float32)
        rg, rc = ctx[ar, goal], ctx[ar, 1 - goal]
        for v in (*out.values(), rg, rc):
            v.flags.writeable = False
        _MEMO["beliefs_every_step"] = (out, rg, rc)
    return _MEMO["beliefs_every_step"]


def step_planes(V, rg, rc):
    """each step's plane by belief_geometry's rule: the PCA of the step's beliefs V[:, t], the 2 of PCs 1-3 holding
    the most latent variance, ridge decoders of r_g and r_c, turned by align_plane -> mean m (T, 20), orthonormal axes
    U (T, 20, 2) (a belief v is drawn at (v - m) U), pcs (T, 2) 1-based, lat (T, 5) single-PC latent R^2 of PCs 1-5,
    var (T, 5) their variance, r2 (T, 2) the plane's decoder R^2 of r_g, r_c. The fits at FIT_THREADS BLAS threads"""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    S = {k: [] for k in ("m", "U", "pcs", "lat", "var", "r2")}
    with fit_threads():
        for t in range(V.shape[1]):
            X = np.asarray(V[:, t], np.float64)
            pca = PCA(n_components=5).fit(X)
            P = pca.transform(X)
            lat = np.array([sum(np.corrcoef(P[:, q], y)[0, 1] ** 2 for y in (rg, rc)) for q in range(5)])
            sel = sorted(int(q) for q in np.argsort(lat[:3])[-2:])
            fits = [Ridge(alpha=1.0).fit(P[:, sel], y) for y in (rg, rc)]
            Q = align_plane(np.eye(2), fits[0].coef_, fits[1].coef_)[0]
            r2 = [m_.score(P[:, sel], y) for m_, y in zip(fits, (rg, rc))]
            for k, v in (("m", pca.mean_), ("U", pca.components_[sel].T @ Q), ("pcs", [q + 1 for q in sel]),
                         ("lat", lat), ("var", pca.explained_variance_), ("r2", r2)):
                S[k].append(v)
    return {k: np.array(v) for k, v in S.items()}


def alias_trajectories():
    """alias_traj.npz -> ({agent: (n, T, 20)}, cell_of, cells [(r_g, r_c)], rev (n, T))"""
    with np.load(ensure("alias_traj")) as f:
        cells = [tuple(int(v) for v in c) for c in f["cells"]]
        assert cells == list(CELLS), f"alias_traj.npz holds cells {cells}, not CELLS"
        return {a: f[a] for a in ("Trained", "Echo", "Joint", "Naive")}, f["cell_of"], cells, f["rev"]


def fr_group_episodes(group, BV=None, rg=None, rc=None):
    """the FR_GROUP_N episodes of FR group 'low' or 'high': among the first N_EP, distinct-variable episodes with both
    values in FR_GROUP_SUB and FR_g + FR_c at T (Jeffreys/2 between the Optimal's and the Naive's marginals, clipped to
    [1e-15, 1 - 1e-15]) in the lowest / highest FR_GROUP_Q of them; FR_GROUP_N variable pairs drawn without replacement
    (default_rng(FR_GROUP_SEED)), one of each pair's episodes uniformly -> episodes, the pairs available, the FR cut.
    BV, rg, rc: beliefs_every_step()'s, if at hand. Checked against traj_fr_draw_<group>.npz when it is there"""
    if BV is None:
        BV, rg, rc = beliefs_every_step()
    n = N_EP
    ci = np.sort(record()["ep_ctx_inds"][:n].astype(int), 1)
    J, N = BV["Joint"][:, -1].astype(np.float64), BV["Naive"][:, -1].astype(np.float64)
    frT = ph.jeffreys_half(J[:, :10], N[:, :10]) + ph.jeffreys_half(J[:, 10:], N[:, 10:])
    elig = (ci[:, 0] != ci[:, 1]) & np.isin(rg[:n], FR_GROUP_SUB) & np.isin(rc[:n], FR_GROUP_SUB)
    q = np.quantile(frT[elig], FR_GROUP_Q if group == "low" else 1 - FR_GROUP_Q)
    elig &= (frT <= q) if group == "low" else (frT >= q)
    ok = np.flatnonzero(elig)
    pairs, inv = np.unique(ci[ok], axis=0, return_inverse=True)
    inv = inv.ravel()
    rng = np.random.default_rng(FR_GROUP_SEED)
    pick = rng.choice(len(pairs), FR_GROUP_N, replace=False)
    eps = np.array([rng.choice(ok[inv == p]) for p in pick])
    ref = data_path(f"traj_fr_draw_{group}.npz")
    if os.path.exists(ref):
        assert np.array_equal(eps, np.load(ref)["episodes"]), f"the {group}-FR draw differs from {ref}"
    return eps, len(pairs), float(q)


def plane_at(S, V, tau):
    """beliefs V (..., 20) drawn in step planes S (step_planes) at step index tau (0 .. T - 1, continuous): (v - m)
    polar(U), m and U the Catmull-Rom blends of the steps' means and axes -> (..., 2)"""
    return (np.asarray(V, np.float64) - ph.blend(S["m"], tau)) @ ph.polar(ph.blend(S["U"], tau))


# ═══════════════════════════════════════════════════════════════════
# Figures: where they are saved
# ═══════════════════════════════════════════════════════════════════

def figure_path(name, out_dir=None):
    """where figure name is saved: in out_dir (created when missing), else in main/REPLICATION/figure_bin through
    main.utils.fig_path (looked up when called, so a patched fig_path applies)"""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, name)
    import main.utils as mu
    return mu.fig_path(name)


# ═══════════════════════════════════════════════════════════════════
# Figure: summary (fig1c_summary_fold.png, fig1c_summary_updates.png)
# ═══════════════════════════════════════════════════════════════════

SUMMARY_FILES = ("fig1c_summary_fold.png", "fig1c_summary_updates.png")
FOLD_NBIN, FOLD_ZOOM = 20, 1.2                                                    # fold b: FR bins per environment
                                                                                  # (5 cells each), the box's zoom
FOLD_LABEL_XZ = ((-22, 6.15), (13, 3.3))                                          # fold b: (step axis, height) of
                                                                                  # the two halves' labels
FOLD_FR_CMAP = LinearSegmentedColormap.from_list("fr", ["#8c8c8c", "#f29b5e", "#d7301f", "#67000d"])
                                                                                  # fold b: the cell's FR bin
UPD_PAIRS = (("Joint", "Naive"), ("Trained", "Echo"))                             # joint agent - factorized agent
UPD_LIM, UPD_BIN, UPD_LINTHRESH, UPD_MIN_STEP = 5., .25, 5e-5, .25                # map side and bin (grid units),
                                                                                  # linear range of the colour (share),
                                                                                  # steps shorter than this left out
JM_NB = 20                                                                        # joint movement: quantile bins
JM_STYLE = {"Trained": dict(lw=2.6, alpha=.45), "Joint": dict(lw=1.3), "Naive": dict(lw=1.3), "Echo": dict(lw=1.3)}


def fold_pairs_block(fig, x0, y0, sq, gx, gy, Z, formula):
    """fold panel a in fig: the four agents' pair-distance panels (sq inches square) as a 2 x 2 block, gx and gy
    inches apart, its lower-left corner at (x0, y0) inches; the header (name over formula) and the letter above it,
    the step colour bar to its right. Z: distances_pairs' ymean, xs, agents"""
    fs = ph.FIG_FONTS
    Y, xs, ag = Z["ymean"].astype(np.float64), Z["xs"], list(Z["agents"])            # Y (E, 4, 2, T, K)
    nE = len(Y)
    vir = plt.get_cmap("viridis")
    nrow = ncol = 2
    top = y0 + nrow * sq + (nrow - 1) * gy                                           # the block's top
    for k, a in enumerate(AGENTS4):
        r_, c_k = divmod(k, ncol)
        ax = ph.ax_in(fig, x0 + c_k * (sq + gx), y0 + (nrow - 1 - r_) * (sq + gy), sq, sq)
        Ya = Y[:, ag.index(a), 0]                                                    # posterior means (E, T, K)
        M, SE = Ya.mean(0), Ya.std(0, ddof=1) / np.sqrt(nE)
        for s_ in range(N_T):
            c_ = vir(s_ / (N_T - 1))
            ax.vlines(xs, M[s_] - SE[s_], M[s_] + SE[s_], color=c_, lw=.5, alpha=.7, zorder=2)
            ax.plot(xs, M[s_], c=c_, lw=1.0, zorder=3)
        ax.plot([0, 13], [0, 13], c="k", ls="--", lw=.9, zorder=4)
        ax.set_xlim(0, 13), ax.set_ylim(0, 13)
        ax.set_xticks([0, 4, 8, 12]), ax.set_yticks([0, 4, 8, 12])
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(ph.AGENT_NAME[a], fontsize=fs["label"] - .5, color=ph.AGENT_COLOR[a] if a != "Naive" else "0.35",
                     pad=3)
        ax.set_xlabel("True distance", labelpad=1)
        if c_k == 0:
            ax.set_ylabel("Estimate distance", labelpad=1)
        if k == 0:                                                                   # the header, after the first panel
            hdr = ph.ax_in(fig, x0, top + .38, ncol * sq + (ncol - 1) * gx, .01)
            hdr.axis("off")
            ph.title2(hdr, "Distance between two particles: estimates (posterior means) vs targets", formula)
            ph.letter(fig, hdr, "a", dx=.62)
    cax = ph.ax_in(fig, x0 + ncol * sq + (ncol - 1) * gx + .2, y0 + .1, .1, top - y0 - .2)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=vir, norm=plt.Normalize(1, N_T)), cax=cax, ticks=[1, 10, 20, 30])
    cb.ax.tick_params(labelsize=fs["tick"] - 1, length=2, pad=1)
    cb.set_label("step $t$", fontsize=fs["tick"] - .5, labelpad=2)


def fold_fr_3d(fig, rect, Z):
    """fold panel b's 3-D axes in fig at rect (x, y, w, h; inches): the distance from the true cell over t, one line
    per agent and FR bin, time mirrored at t = 1 (Optimal and Trained RNN to the right, Naive and ESN to the left),
    the FR axis labelled with the bins' median FR. Z: distances_fr's agents, dist_t20, fr_median20 -> the axes"""
    fs = ph.FIG_FONTS
    nbin = FOLD_NBIN
    ag, D_t, frm = list(Z["agents"]), Z[f"dist_t{nbin}"], Z[f"fr_median{nbin}"]
    M = D_t.mean(1)                                                                  # (4, nbin, T)
    col = [FOLD_FR_CMAP(q / (nbin - 1)) for q in range(nbin)]
    side = {"Joint": 1, "Trained": 1, "Naive": -1, "Echo": -1}                       # +: right half, -: left half
    theory, net = dict(lw=2.2, alpha=.85, ls=(0, (2.8, 1.4))), dict(lw=1.0, ls="-")
    dark = lambda c: tuple(np.array(c[:3]) * .75) + (1.,)                            # networks slightly darker
    sty = {"Joint": theory, "Naive": theory, "Trained": net, "Echo": net}
    ticks = [1, round(1 + (nbin - 1) / 3), round(1 + 2 * (nbin - 1) / 3), nbin]
    tlab = ["%.2g" % frm[k - 1] if frm[k - 1] < 10 else "%.0f" % frm[k - 1] for k in ticks]
    ax = ph.ax_in(fig, *rect, projection="3d")
    ax.computed_zorder = False
    x = np.arange(1, N_T + 1) - 1.                                                   # 0 at t = 1
    for a in ("Naive", "Joint", "Echo", "Trained"):                                  # theory first, networks on top
        for q in range(nbin):
            o_ = slice(None, None, side[a])                                          # left half drawn from t = 30
            ax.plot((side[a] * x)[o_], np.full(N_T, q + 1), M[ag.index(a), q][o_], clip_on=False,  # 3-D axes clip
                    c=dark(col[q]) if a in ("Trained", "Echo") else col[q],                        # to a square box
                    zorder=(3 if a in ("Naive", "Joint") else 5), **sty[a])
    ax.plot([0, 0], [.5, nbin + .5], [0, 0], c="0.55", lw=.8, zorder=2)             # the t = 1 seam on the floor
    ax.set_xlim(-29.5, 29.5), ax.set_ylim(.5, nbin + .5), ax.set_zlim(0, 6)
    ax.set_xticks([-29, -19, -9, 0, 9, 19, 29], ["30", "20", "10", "1", "10", "20", "30"])
    ax.set_zticks([0, 2, 4, 6])
    ax.set_yticks(ticks, tlab)
    ax.tick_params(labelsize=fs["tick"] - 1, pad=0)
    ax.set_xlabel(r"Step $t$ (mirrored at $t = 1$)", fontsize=fs["tick"] + .5, labelpad=0)
    ax.set_ylabel("Cell's FR (nats; %d bins)" % nbin, fontsize=fs["tick"] + .5, labelpad=-1)
    ax.set_zlabel("Distance from the true cell", fontsize=fs["tick"] + .5, labelpad=-3)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((1, 1, 1, 0)), pane.set_edgecolor("0.85")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update(color="0.9", linewidth=.4)
    ax.view_init(elev=22, azim=-58)
    ax.set_box_aspect((8, 4, 3), zoom=FOLD_ZOOM)
    kw = dict(fontsize=fs["label"], ha="center", va="bottom", zorder=10)
    ax.text(FOLD_LABEL_XZ[0][0], nbin + .6, FOLD_LABEL_XZ[0][1], "Naive and ESN", color="0.3", **kw)
    ax.text(FOLD_LABEL_XZ[1][0], nbin + .6, FOLD_LABEL_XZ[1][1], "Optimal and Trained RNN", color="k", **kw)
    return ax


def fold_fit_3d(target, Z, passes=3, dpi=60):
    """the rect (x, y, w, h; inches) for fold_fr_3d that makes the 3-D panel's drawn content (curves, box, ticks,
    labels: its inked pixels, measured on a large scratch canvas at dpi so nothing is clipped) fill target (x0, y0,
    x1, y1; inches), centred; up to `passes` rounds until the scale changes by less than 1%"""
    rw, rh = 8., 5.
    Tw, Th = target[2] - target[0], target[3] - target[1]
    for _ in range(passes):
        tf = plt.figure(figsize=(3 * rw, 3 * rh), dpi=dpi)
        fold_fr_3d(tf, (rw, rh, rw, rh), Z)
        tf.canvas.draw()
        ink = (np.asarray(tf.canvas.buffer_rgba())[..., :3] < 250).any(-1)
        plt.close(tf)
        rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
        cx0, cx1 = cols[0] / dpi - rw, (cols[-1] + 1) / dpi - rw                    # from the axes' left edge
        cy0, cy1 = 2 * rh - (rows[-1] + 1) / dpi, 2 * rh - rows[0] / dpi            # from the axes' bottom edge
        sc = min(Tw / (cx1 - cx0), Th / (cy1 - cy0))
        rect = (target[0] + (Tw - sc * (cx1 - cx0)) / 2 - sc * cx0, target[1] + (Th - sc * (cy1 - cy0)) / 2 - sc * cy0,
                sc * rw, sc * rh)
        if abs(sc - 1) < .01:
            break
        rw, rh = sc * rw, sc * rh
    return rect


def update_densities(size):
    """each agent's share of its steps per UPD_BIN x UPD_BIN bin of (|u_g(t)|, |u_c(t)|) on [0, UPD_LIM]^2 among its
    steps of length >= UPD_MIN_STEP, the bins' counts and all its points; size {agent: |u| (n, T, 2) float64}
    -> {agent: (shares, counts, points (n T, 2))}, the bin edges"""
    edges = np.arange(0, UPD_LIM + UPD_BIN / 2, UPD_BIN)
    out = {}
    for a in AGENTS4:
        u = size[a].reshape(-1, 2)
        keep = np.hypot(u[:, 0], u[:, 1]) >= UPD_MIN_STEP
        n, _, _ = np.histogram2d(u[keep, 0], u[keep, 1], [edges, edges])
        out[a] = (n / keep.sum(), n, u)
    return out, edges


def update_maps(fig, rects, cax_rect, size):
    """updates panels a, b in fig: the difference maps of update_densities, Optimal - Naive and Trained RNN - ESN
    (symmetric log colour), each with the share of its long steps near the diagonal above it; rects the two panels'
    (x, y, w, h) and cax_rect the shared colour bar's, in inches; size as update_densities"""
    fs = ph.FIG_FONTS
    d, edges = update_densities(size)
    vmax = max(np.abs(d[a1][0] - d[a2][0]).max() for a1, a2 in UPD_PAIRS)
    norm = SymLogNorm(UPD_LINTHRESH, vmin=-vmax, vmax=vmax, base=10)
    for c, (a1, a2) in enumerate(UPD_PAIRS):
        ax = ph.ax_in(fig, *rects[c])
        D = d[a1][0] - d[a2][0]
        D = np.where((d[a1][1] + d[a2][1]) > 0, D, np.nan)                         # black: no step of either
        im = ax.imshow(D.T, origin="lower", extent=[0, edges[-1], 0, edges[-1]], cmap=ph.DIFF_CMAP, norm=norm,
                       interpolation="nearest")
        ax.set_facecolor("black")
        ax.plot([0, UPD_LIM], [0, UPD_LIM], c="0.65", lw=.7, ls=(0, (3, 2)))
        ax.set_xlim(0, UPD_LIM), ax.set_ylim(0, UPD_LIM), ax.set_aspect("equal")
        ax.set_xticks(range(0, int(UPD_LIM) + 1)), ax.set_yticks(range(0, int(UPD_LIM) + 1))
        ax.set_xlabel(r"$|\Delta x| = |E_t[r_g] - E_{t-1}[r_g]|$", labelpad=1)
        if c == 0:
            ax.set_ylabel(r"$|\Delta y| = |E_t[r_c] - E_{t-1}[r_c]|$", labelpad=1)
        else:
            ax.set_yticklabels([])
        ax.set_title(f"{ph.AGENT_NAME[a1]} $-$ {ph.AGENT_NAME[a2]}", fontsize=fs["label"] + .5, pad=30)
        txt = []
        for a in (a1, a2):                                                         # long steps near the diagonal
            u = d[a][2]
            big = np.hypot(u[:, 0], u[:, 1]) > 2
            diag = big & (np.minimum(u[:, 0], u[:, 1]) > .5 * np.maximum(u[:, 0], u[:, 1]))
            txt.append("%s %.0f%%" % (ph.AGENT_NAME[a], 100 * diag.sum() / max(big.sum(), 1)))
        ax.text(.5, 1.015, "steps longer than 2 near the diagonal:\n" + ", ".join(txt), transform=ax.transAxes,
                ha="center", va="bottom", fontsize=fs["tick"], color="0.25", linespacing=1.25)
    cx = ph.ax_in(fig, *cax_rect)
    ticks = [t for t in (-1e-1, -1e-2, -1e-3, -1e-4, 0, 1e-4, 1e-3, 1e-2, 1e-1) if abs(t) <= vmax]
    cb = fig.colorbar(im, cax=cx, ticks=ticks)
    cb.ax.set_yticklabels([("%g" % t) for t in ticks])
    cb.set_label("share of steps:\njoint $-$ factorized agent", labelpad=4)
    cb.ax.text(0, 1.03, "joint agent\nmore often", transform=cb.ax.transAxes, ha="left", va="bottom",
               fontsize=fs["tick"], color="#c0392b", linespacing=1.1)
    cb.ax.text(0, -.03, "factorized agent\nmore often", transform=cb.ax.transAxes, ha="left", va="top",
               fontsize=fs["tick"], color="#2a6dff", linespacing=1.1)
    cb.outline.set_visible(False)


def _jm_bins(x):
    """JM_NB equal-count bins of x (quantile edges) -> bin index per entry"""
    e = np.quantile(x, np.linspace(0, 1, JM_NB + 1))
    return np.clip(np.searchsorted(e, x, side="right") - 1, 0, JM_NB - 1)


def _jm_cov(a, b):
    """the covariance of a and b along axis 0 (n - 1 normalized)"""
    return ((a - a.mean(0)) * (b - b.mean(0))).sum(0) / (len(a) - 1)


def joint_movement(size, fr_t, rev):
    """the joint movement Cov(|u_g(t)|, |u_c(t)|) per quantile bin (JM_NB,) for each grouping and agent: "fr" by FR_t
    and "ent" by the entanglement rev (bins cut anew at every step, the covariance per step and bin, then its mean
    over the steps), "S" by the path length S = sum_t ||u(t)|| (each agent binned by its own S; the covariance per
    step and bin, then the mean over the steps); size {agent: |u| (n, T, 2) float64}, fr_t and rev (n, T) float64
    -> C {(grouping, agent): (JM_NB,)}, med {("S", agent): the bins' median S}"""
    C, med = {}, {}
    for key, X in (("fr", fr_t), ("ent", rev)):
        q = np.stack([_jm_bins(X[:, t]) for t in range(N_T)], 1)                  # (n, T), cut anew each step
        for a in AGENTS4:
            C[key, a] = np.array([np.mean([_jm_cov(size[a][q[:, t] == k, t, 0], size[a][q[:, t] == k, t, 1])
                                           for t in range(N_T)]) for k in range(JM_NB)])
    for a in AGENTS4:
        S = np.hypot(size[a][..., 0], size[a][..., 1]).sum(1)                    # path length per trajectory
        q = _jm_bins(S)
        C["S", a] = np.array([_jm_cov(size[a][q == k, :, 0], size[a][q == k, :, 1]).mean() for k in range(JM_NB)])
        med["S", a] = np.array([np.median(S[q == k]) for k in range(JM_NB)])
    return C, med


def joint_movement_panels(fig, x0, y0, pw, pht, gx, C, med, letters, legend_xy):
    """updates panels c-e in fig: the joint movement by FR_t, by the entanglement and by the path length (one line
    per agent), in a row of pw x pht inches panels gx apart from (x0, y0) inches, each named (no formula) with its
    letter on the name's line; the agents' legend (upper centre, one row) at legend_xy inches"""
    W, H = fig.get_size_inches()
    fs = ph.FIG_FONTS
    cols = (("fr", "Factorization regret at the step", "$FR_t$, quantile bin at each step"),
            ("ent", "Entanglement of the step", r"$\mathcal{E}_t$, quantile bin at each step"),
            ("S", "Size of the belief's movement", "path length $S$ (bin median, grid units)"))
    xs = np.arange(1, JM_NB + 1)
    name_y = 6                                                                     # the name's baseline (pt)
    for c_, (key, cname, xlab) in enumerate(cols):
        ax = ph.ax_in(fig, x0 + c_ * (pw + gx), y0, pw, pht)
        for a in AGENTS4:
            xv = med[key, a] if key == "S" else xs
            ax.plot(xv, C[key, a], c=ph.AGENT_COLOR[a], marker="o", ms=2.2, **JM_STYLE[a])
        ax.axhline(0, c="0.8", lw=.6, zorder=0)
        if key == "S":
            ax.set_xscale("log")
            lo_, hi_ = min(med[key, a][0] for a in AGENTS4), max(med[key, a][-1] for a in AGENTS4)
            cand = [m * 10. ** k for k in range(-4, 3) for m in (1, 2, 5)]
            ax.set_xticks([v for v in cand if lo_ * .95 <= v <= hi_ * 1.05])
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "%g" % v))
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
        else:
            ax.set_xlim(.5, JM_NB + .5), ax.set_xticks([1, 10, 20])
        ax.set_xlabel(xlab, labelpad=1)
        if c_ == 0:
            ax.set_ylabel(r"joint movement (grid units$^2$)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.annotate(cname, (.5, 1), xycoords="axes fraction", xytext=(0, name_y), textcoords="offset points",
                    ha="center", va="baseline", fontsize=fs["label"] + .5)
        x_, yy, w, h = ax.get_position().bounds
        fig.text(x_ - .62 / W, yy + h + name_y / 72 / H, r"$\mathbf{%s}$" % letters[c_], fontsize=fs["label"] + 2,
                 va="baseline")
    hand = [Line2D([], [], c=ph.AGENT_COLOR[a], label=ph.AGENT_NAME[a], **JM_STYLE[a]) for a in AGENTS4]
    fig.legend(handles=hand, loc="upper center", bbox_to_anchor=(legend_xy[0] / W, legend_xy[1] / H), ncol=4,
               frameon=False, fontsize=fs["tick"], handlelength=1.8, columnspacing=1.4)


def figure_summary_fold(out_dir=None):
    """fig1c_summary_fold.png (18.0 x 7.5 in, 300 dpi) -> its path. One row: a as a 2 x 2 block, b beside it, scaled
    to fill the rest (its drawn content measured on a scratch canvas: fold_fit_3d).
    a  Distance between two particles: estimates (posterior means) vs targets. 110 environments (one harvest episode
       per distinct pair of active variables, evenly spaced over the 1225 pairs: its table, goal variable and Z),
       each with all 100 cells x 10 streams (seed 7,000,000 + episode), the four agents on the same streams (the
       Optimal and the Naive exact, the Trained RNN and the ESN replayed). For every pair of streams of an
       environment: x = the Euclidean distance between their true cells (51 discrete values), y = the Euclidean
       distance between the agent's estimates of the two particles, r^ = (E[r_g], E[r_c]) (grid units); one curve
       per step t (viridis, colour bar), mean over environments, SEM bars over environments; dashed: the target
       y = x. Data: distances_pairs (ymean[:, agent, 0], xs).
    b  How a cell's factorization regret shapes convergence to the true cell (3-D; no colour bar: the FR axis
       carries the values, the header names the line styles). Distance: the mean over the cell's 50 particles of
       the Euclidean distance of the posterior means from the true cell (grid units; per particle, so split clouds
       do not cancel), mean over the alias scan's 39 environments (40 environments x 100 cells x 50 streams; e = 2,
       an exactly symmetric table, left out). Cells binned by their FR within each environment, 20 bins of 5
       cells; a cell's FR: the mean over its streams of FR_g + FR_c at T = 30, the Jeffreys/2 divergence between the
       Optimal's and the Naive's marginal of each variable (nats); the FR axis is labelled with the median over
       environments of each bin's median FR.
       The Optimal and the Trained RNN run from t = 1 to 30 to the right, the Naive and the ESN from t = 1 to 30 to
       the left, so each bin's curves meet at t = 1, where all agents start alike. Colour: the cell's FR bin (grey
       to dark red); dashed: theory (Naive left, Optimal right); thin solid, slightly darkened: the networks (ESN
       left, Trained RNN right), drawn on top. Data: distances_fr (dist_t20, fr_median20)."""
    with load_cache("distances_pairs") as Z:
        pairs = {k: Z[k] for k in ("ymean", "xs", "agents")}
    with load_cache("distances_fr") as Z:
        fr = {k: Z[k] for k in ("agents", f"dist_t{FOLD_NBIN}", f"fr_median{FOLD_NBIN}")}
    with ph.style("fig"):
        sq, gx, gy, xa, ya = 2.55, .45, .85, .8, .62
        top = ya + 2 * sq + gy                                                        # a's block
        hdr_y = top + .38                                                             # both headers' line
        xb = xa + 2 * sq + gx + .2 + .1 + .75                                         # after a's step colour bar
        W, H = xb + 10.6, hdr_y + 17 / 72 + .35
        fig = plt.figure(figsize=(W, H))
        fold_pairs_block(fig, xa, ya, sq, gx, gy, pairs,
                         r"$\|\hat r_i - \hat r_j\|$ against the true $\|r_i - r_j\|$, $\hat r = (E[r_g], E[r_c])$; "
                         r"one curve per step; SEM bars")
        fold_fr_3d(fig, fold_fit_3d((xb, .08, W - .08, hdr_y - .12), fr), fr)
        hdr = ph.ax_in(fig, xb + .45, hdr_y, W - xb - .55, .01)
        hdr.axis("off")
        ph.title2(hdr, "How a cell's factorization regret shapes convergence to the true cell",
                  r"$\langle \|(E[r_g], E[r_c]) - (r_g, r_c)\| \rangle_{\mathrm{particles}}$; "
                  r"mirrored at $t = 1$; dashed: theory, solid: networks")
        ph.letter(fig, hdr, "b", dx=.62)
        path = figure_path(SUMMARY_FILES[0], out_dir)
        fig.savefig(path, dpi=300)
        plt.close(fig)
    return path


def figure_summary_updates(out_dir=None):
    """fig1c_summary_updates.png (18.85 x 4.8 in, 300 dpi) -> its path. One row of five; the curves are taller than
    the maps by the maps' notes, so every panel's title and letter sit on one line. Data: updates, every step
    t = 1..30 of the trained network's gradflow harvest (all 50,000 episodes: 48,968 trajectories of
    distinct-variable environments, 1,225 environments; the four agents on the same episodes): the update of the
    posterior means u_v(t) = E_t[r_v] - E_{t-1}[r_v] (E_0 = 4.5; grid units) of the goal (v = g) and the contextual
    variable (v = c).
    a, b  How the two beliefs move: each agent's share of its (trajectory, step) points per 0.25 x 0.25 bin of
          (|dx|, |dy|) = (|u_g(t)|, |u_c(t)|) (steps shorter than 0.25 left out), the joint minus the factorized
          agent: Optimal - Naive (a), Trained RNN - ESN (b). Red: the joint agent's steps land there more often,
          blue: the factorized agent's; black: equally, or no step of either. Symmetric log colour scale in absolute
          shares per bin (linear within 5e-5). A point's distance from the origin is the step's length, its angle
          how the step divides between the two variables (dashed diagonal: equally). Above each map: the share of
          each agent's steps longer than 2 that lie near the diagonal (the smaller size over half the larger).
    c-e   Joint movement of the two beliefs: the covariance of the two update sizes, Cov(|u_g(t)|, |u_c(t)|), over a
          group of points, large only when both beliefs move a lot in the same steps (the correlation times both
          sizes' spreads; grid units^2); 20 quantile bins each:
          c  by FR_t, the factorization regret at the step (FR_g + FR_c, Jeffreys/2 between the Optimal's and the
             Naive's marginals), and d  by E_t, the step's entanglement (Jeffreys/2 between the Naive's likelihood
             of the step and the Optimal's update b_t / b_{t-1} of each marginal, mean over the two variables): bins
             cut anew at every step, the covariance per step and bin, then its mean over the 30 steps;
          e  by the belief's path length S = sum_t ||u(t)|| (each agent binned by its own S; x: the bin's median,
             log scale): the covariance per step and bin, then its mean over the steps."""
    with load_cache("updates") as Z:
        size = {a: np.abs(Z["u_" + a].astype(np.float64)) for a in AGENTS4}       # |u_v(t)| (n, T, 2)
        fr_t, rev = Z["fr_t"].astype(np.float64), Z["rev"].astype(np.float64)
    C, med = joint_movement(size, fr_t, rev)
    with ph.style("fig"):
        fs = ph.FIG_FONTS
        P, y0 = 2.6, .62                                                              # map side, bottom of every panel
        Pc = P + 30 / 72 - 6 / 72                                                     # curves: titles on the maps' line
        xa, gap_ab, wc, gxc = .85, .35, 2.75, .85
        xb = xa + P + gap_ab
        xcb = xb + P + .3                                                             # the maps' colour bar
        xc = xcb + 1.95                                                               # the first curve
        W = xc + 3 * wc + 2 * gxc + .25
        top = y0 + Pc + 6 / 72                                                        # the titles' baseline
        hdr_y = top + .55
        H = hdr_y + 17 / 72 + .38
        fig = plt.figure(figsize=(W, H))
        update_maps(fig, [(xa, y0, P, P), (xb, y0, P, P)], (xcb, y0 + .45, .13, P - .9), size)
        for x, off, s_ in ((xa, .62, "a"), (xb, .3, "b")):
            fig.text((x - off) / W, top / H, r"$\mathbf{%s}$" % s_, fontsize=fs["label"] + 2, va="baseline")
        joint_movement_panels(fig, xc, y0, wc, Pc, gxc, C, med, "cde", (xc + 1.5 * wc + gxc, top + .43))
        for x0, w, name, formula in (
                (xa, xb + P - xa, "How the two beliefs move",
                 r"share of steps per bin of $(|\Delta x|, |\Delta y|)$, joint $-$ factorized agent; every step of "
                 + f"{len(fr_t):,} trajectories"),
                (xc, 3 * wc + 2 * gxc, "Joint movement of the two beliefs",
                 r"by step: $\mathrm{Cov}(|u_g(t)|, |u_c(t)|)$, mean over $t$")):
            hdr = ph.ax_in(fig, x0, hdr_y, w, .01)
            hdr.axis("off")
            ph.title2(hdr, name, formula)
        path = figure_path(SUMMARY_FILES[1], out_dir)
        fig.savefig(path, dpi=300)
        plt.close(fig)
    return path


def figure_summary(out_dir=None):
    """The factorized observers' fold, in two summary figures -> the paths of fig1c_summary_fold.png and
    fig1c_summary_updates.png, in main/REPLICATION/figure_bin (or out_dir):
        fig1c_summary_fold.png (one row: a as a 2 x 2 block, b beside it)                   figure_summary_fold
            a     distance between two particles, estimates (posterior means) vs targets
            b     how a cell's factorization regret shapes convergence to the true cell (3-D)
        fig1c_summary_updates.png (one row of five)                                         figure_summary_updates
            a, b  update sizes of the two beliefs, joint minus factorized agent
            c-e   joint movement of the two beliefs by FR, entanglement and path length
    The captions are in the two functions' docstrings (the figures carry none). Data: the caches distances_pairs,
    distances_fr and updates (built first when missing; the data layer says how). Style: plotting_helpers'
    "fig" (cmr10 text, Computer Modern math), set inside each function, whatever ran before in the process.
    About 10 s with the caches. Run from the repository root with the eureka environment's python:
        python main/REPLICATION/plot_fig1c_all.py summary [--out DIR]"""
    return figure_summary_fold(out_dir), figure_summary_updates(out_dir)


# ═══════════════════════════════════════════════════════════════════
# Figures: the step-plane movies' panel layout (inches; pcs, the traj_fr stills, envdist's markers)
# ═══════════════════════════════════════════════════════════════════

MOVIE_AGENTS = (("Trained", "Echo"), ("Joint", "Naive"))                          # the 2 x 2 order
MOVIE_W, MOVIE_DPI = 7.05, 150                                                    # width (in), resolution
MOVIE_BS, MOVIE_X0, MOVIE_XGAP, MOVIE_YNAME, MOVIE_YTOP = 2.4, .15, .25, .44, .55  # panel side, left edge, column gap,
                                                                                  # names' band, clock band
MOVIE_STRIP, MOVIE_STRIP_GAP = .30, .30                                           # the PC strip under a panel
MOVIE_LEG_X = MOVIE_X0 + 2 * MOVIE_BS + MOVIE_XGAP + .32                          # the key column's left edge
MOVIE_FS = dict(t=13, name=11, lab=9, note=7.5, tick=7.5)                         # type sizes (pt)
MOVIE_SCALE = MOVIE_BS / 1.11                                                     # panel side over the stills' 1.11 in
MOVIE_PX = 1                                                                      # cloud points: PX x PX pixel squares,
                                                                                  # not anti-aliased (anti-aliased
                                                                                  # sub-pixel motion changes every
                                                                                  # pixel: GIFs 2.5x larger)
MOVIE_MS_START, MOVIE_MS_END, MOVIE_MS_HOME = 9, 10.5, 18                         # path start, head, target (pt)
MOVIE_SUB = 8                                                                     # path points per step
MOVIE_ACCENT = "#d62728"                                                          # the pair shown, the cursor
CELL_MARKS = ("o", "s", "^", "*")                                                 # per cell of alias_traj
CELL_MARK_SCALE = {"*": 1.45}                                                     # panels: a star of the same nominal
CELL_MARK_SCALE_KEY = {"*": 1.15}                                                 # size looks smaller; the key: less


def movie_strip(fig, x, y, S):
    """the step planes S under a panel whose lower left is (x, y) (inches): the single-PC latent R^2 (r_g + r_c) of
    each step's PCs 1-3 (white 0 -> black 1), the pair shown (dots), a cursor at the frame's step -> the cursor"""
    H = fig.get_figheight()
    ax = fig.add_axes([(x + .32) / MOVIE_W, (y - .07 - MOVIE_STRIP) / H, (MOVIE_BS - .32) / MOVIE_W, MOVIE_STRIP / H])
    ax.imshow(S["lat"][:, :3].T, cmap="Greys", vmin=0, vmax=1, aspect="auto", interpolation="nearest",
              extent=(.5, N_T + .5, 3.5, .5))
    ax.scatter(np.repeat(np.arange(1, N_T + 1), 2), S["pcs"].ravel(), s=2.5, c=MOVIE_ACCENT, lw=0, zorder=4)
    ax.set_xlim(.5, N_T + .5), ax.set_ylim(3.5, .5)
    ax.set_yticks([1, 2, 3], ["PC1", "PC2", "PC3"])
    ax.set_xticks([1, 10, 20, 30])
    ax.tick_params(length=1.5, pad=1, labelsize=MOVIE_FS["tick"] - 1)
    for s_ in ax.spines.values():
        s_.set_linewidth(.5)
    return ax.axvline(1, c="k", lw=.9, zorder=5)


# ═══════════════════════════════════════════════════════════════════
# Figure: belief trajectories in the lowest and the highest FR (fig1c_traj_FR_groups_T30.png)
# ═══════════════════════════════════════════════════════════════════

TRAJ_FR_STY = dict(lw=.5, alpha=.8, ms=3.4, home_ms=5.2, home_mew=1., faint="0.82", faint_alpha=.5)
                    # trajectories (pt, alpha), particles and targets (marker size and edge, pt), the undrawn cells


def traj_fr_group(group, BV, rg, rc, S, home):
    """one FR group ('low' or 'high'): its draw (fr_group_episodes) and per agent the step planes S[agent], the
    particles' beliefs V (50, T, 20) float64, their Catmull-Rom paths P20 (50, len(tt), 20) at MOVIE_SUB points per
    step tt, the 16 cells' targets home (16, 20) -> per-agent data, and the colours and the draw. BV, rg, rc:
    beliefs_every_step()"""
    eps, n_avail, fr_cut = fr_group_episodes(group, BV, rg, rc)
    cell = np.stack([rg[eps], rc[eps]], 1).astype(int)
    hc = np.array([(g, c) for g in FR_GROUP_SUB for c in FR_GROUP_SUB])
    tt = np.linspace(0, N_T - 1, (N_T - 1) * MOVIE_SUB + 1)
    data = {}
    for a in ("Trained", "Echo", "Joint", "Naive"):
        V = np.asarray(BV[a][eps], np.float64)
        data[a] = dict(S=S[a], V=V, tt=tt, home=home, P20=np.stack([ph.blend(V, tau, axis=1) for tau in tt], 1))
    col = ph.bivariate(cell[:, 0], cell[:, 1])
    return data, dict(col=col, rgba=np.concatenate([col, np.full((len(col), 1), TRAJ_FR_STY["alpha"])], 1),
                      group=group, home_col=ph.bivariate(hc[:, 0], hc[:, 1]), n=len(eps), n_avail=n_avail,
                      fr_cut=fr_cut)


def _traj_fr_at(D, tau):
    """frame tau in its own plane (plane_at): every particle's trajectory so far ((n, k + 1, 2): its path's points up
    to tau, then its position at tau), its position (n, 2), the targets (16, 2)"""
    k = int(np.searchsorted(D["tt"], tau, side="right"))
    head = plane_at(D["S"], ph.blend(D["V"], tau, axis=1), tau)
    trail = np.concatenate([plane_at(D["S"], D["P20"][:, :k], tau), head[:, None]], 1)
    return trail, head, plane_at(D["S"], D["home"], tau)


def _traj_fr_box(D, taus):
    """one square data box holding every frame's trajectories and targets (frames taus; 6% margin)"""
    ext = []
    for tau in taus:
        tr, hd, hm = _traj_fr_at(D, tau)
        pts = np.concatenate([tr.reshape(-1, 2), hm])
        ext += [pts.min(0), pts.max(0)]
    return ph.square_limits(np.stack(ext), 1.06)


def _traj_fr_key(fig, H, m):
    """the right column: the latent space (the 16 drawn cells lit, the others washed out), the marks, the draw"""
    STY, FS, W = TRAJ_FR_STY, MOVIE_FS, MOVIE_W
    x, top, ks = MOVIE_LEG_X, H - .62, 1.15
    fig.text((x + ks / 2) / W, top / H, "Latent space", ha="center", va="baseline", fontsize=FS["lab"])
    ax = fig.add_axes([(x + .12) / W, (top - .1 - ks) / H, ks / W, ks / H])
    gg, cc = np.meshgrid(np.arange(N_R), np.arange(N_R))
    lit = np.zeros((N_R, N_R), bool)
    lit[np.ix_(list(FR_GROUP_SUB), list(FR_GROUP_SUB))] = True
    fa = np.array(to_rgb(STY["faint"]))
    ph.latent_key(ax, np.where(lit[..., None], ph.bivariate(gg, cc), STY["faint_alpha"] * fa + 1 - STY["faint_alpha"]),
                  R=N_R)
    ax.tick_params(length=1.5, pad=1)
    ax.set_xlabel("$r_g$", labelpad=-4)
    ax.set_ylabel("$r_c$", labelpad=-4)
    y = top - ks - .62
    y += .12
    c_mid = tuple(ph.bivariate(5, 4))
    lab = dict(ha="left", va="center", fontsize=FS["note"] + .5, color="0.25")
    fig.add_artist(Line2D([(x + .2) / W], [y / H], transform=fig.transFigure, marker="o", ms=STY["ms"] * 1.6,
                          mfc=c_mid, mec="none", ls="none"))
    fig.text((x + .42) / W, y / H, "a particle now", **lab)
    y -= .24
    fig.add_artist(Line2D([(x + .05) / W, (x + .35) / W], [y / H, (y + .05) / H], transform=fig.transFigure,
                          color=c_mid, lw=STY["lw"] * 2, alpha=.8, solid_capstyle="round"))
    fig.text((x + .42) / W, y / H, "its trajectory so far", **lab)
    y -= .24
    fig.add_artist(Line2D([(x + .2) / W], [y / H], transform=fig.transFigure, marker="o", ms=STY["home_ms"] * 1.5,
                          mfc="none", mec=c_mid, mew=STY["home_mew"] * 1.5, ls="none"))
    fig.text((x + .42) / W, y / H, "the cell's target", **lab)
    g = m["group"]
    who = (f"{FR_GROUP_N} environments drawn at\nrandom (of {m['n_avail']:,}), one\nrandom episode each: {m['n']}\n"
           "particles, each coloured by\nits true cell $(r_g, r_c)$" +
           ",\nboth values in $\\{%s\\}$\n(16 cells; their targets)" % ", ".join(str(v) for v in FR_GROUP_SUB) +
           "\n\n%s 10%% of FR at $T = 30$\n(FR %s %.2g nats)" % ({"low": "the lowest", "high": "the highest"}[g],
                                                                {"low": "$\\leq$", "high": "$\\geq$"}[g], m["fr_cut"]))
    plane = ("\n\neach step's own PC plane,\nas in the pcs movie (PCs\nof 20,000 episodes' beliefs:\none plane for all "
             "environ-\nments); all trajectories\nre-drawn in the current plane")
    fig.text(x / W, (y - .26) / H, who + plane + "\n\ntarget: the Optimal's mean\ncorrect $T = 30$ belief\nabout the "
             "cell\n\nstrips: latent $R^2$ of\neach step's PCs 1-3\n(white 0, black 1); dots:\nthe pair shown",
             ha="left", va="top", fontsize=FS["note"], color="0.3", linespacing=1.25)


def traj_fr_still(data, m):
    """one FR group's still (traj_fr_group's data and m), the t = 30 frame of its movie -> RGB image (PIL), 1057 x
    1132 px"""
    T, W, BS, X0, XGAP, FS, STY = N_T, MOVIE_W, MOVIE_BS, MOVIE_X0, MOVIE_XGAP, MOVIE_FS, TRAJ_FR_STY
    taus = [tau for tau, _ in ph.schedule(T=T)][::6] + [T - 1.]                  # the movie's frames (its boxes)
    with ph.style("movie"):
        row_h = MOVIE_YNAME + BS + (MOVIE_STRIP + MOVIE_STRIP_GAP)               # (H = 7.55 in)
        H = MOVIE_YTOP + 2 * row_h + .12
        fig = plt.figure(figsize=(W, H), dpi=MOVIE_DPI, facecolor="white")
        try:
            for r, agents in enumerate(MOVIE_AGENTS):
                y = H - MOVIE_YTOP - MOVIE_YNAME - BS - r * row_h
                for p, a in enumerate(agents):
                    D, x = data[a], X0 + p * (BS + XGAP)
                    ax = ph.ax_in(fig, x, y, BS, BS)
                    xl, yl = _traj_fr_box(D, taus)
                    ax.set_xlim(*xl), ax.set_ylim(*yl)
                    ax.set_aspect("equal", adjustable="box")
                    ax.axis("off")
                    fig.text((x + BS / 2) / W, (y + BS + .22) / H, ph.AGENT_NAME_BAYES[a], ha="center",
                             va="baseline", fontsize=FS["name"], color=ph.AGENT_COLOR[a])
                    fig.text((x + BS / 2) / W, (y + BS + .07) / H,
                             "PCs %d-%d of step $t$" % tuple(D["S"]["pcs"][T - 1]), ha="center", va="baseline",
                             fontsize=FS["note"], color="0.45")
                    movie_strip(fig, x, y, D["S"]).set_xdata([T, T])
                    tr, hd, hm = _traj_fr_at(D, T - 1.)
                    ax.add_collection(LineCollection(list(tr), colors=m["rgba"], linewidths=STY["lw"], zorder=1,
                                                     capstyle="round", joinstyle="round", antialiased=True))
                    ax.scatter(*hm.T, s=STY["home_ms"] ** 2, facecolors="none", edgecolors=m["home_col"],
                               linewidths=STY["home_mew"], zorder=2)
                    ax.scatter(*hd.T, s=STY["ms"] ** 2, c=m["col"], lw=0, zorder=3, antialiased=True)
            ph.clock_bar(fig, X0, H - .33, X0 + .95, X0 + 2 * BS + XGAP, T=T, fs_t=FS["t"],
                         fs_tick=FS["tick"])(T - 1.)
            _traj_fr_key(fig, H, m)
            im = Image.fromarray(ph.grab(fig))
        finally:
            plt.close(fig)
    return im


def figure_traj_fr(out_dir=None):
    """Belief trajectories of 50 particles, each from a different environment, in the lowest and in the highest 10%
    of factorization regret (FR) at T = 30 -> fig1c_traj_FR_groups_T30.png in out_dir (default main/REPLICATION/
    figure_bin; 2114 x 1132 px) -> [its path].

    Two stills side by side, left the lowest 10% of FR, right the highest; each is the t = 30 frame of an FR group's
    movie in the pcs movie's layout (7.05 x 7.55 in at 150 dpi: 1057 x 1132 px).
    Draw (fr_group_episodes): the trained harvest's first 20,000 episodes (each one stream of one cell of one
    environment; the episodes the step planes are fitted on). Eligible: distinct-variable environments, both values
    in {0, 3, 6, 9} (16 cells), and FR at T = 30 (FR_g + FR_c, each half the Jeffreys divergence between the Optimal's
    and the Naive's marginal of the variable, both clipped to [1e-15, 1 - 1e-15]; nats) in the lowest / highest 10% of
    the eligible episodes; then 50 environments drawn uniformly without replacement (default_rng(0)) and one of each
    environment's eligible episodes uniformly.
        panels     2 x 2: Trained RNN and ESN over Optimal Bayes and Naive Bayes, each in its own step-30 plane, the
                   pcs movie's (step_planes: of the PCA of the agent's goal-ordered beliefs at the step on all 20,000
                   episodes, the 2 of PCs 1-3 holding the most latent variance, turned so that the ridge decoder of
                   r_g points right and that of r_c up; one plane for all environments). Each particle's whole
                   trajectory re-drawn in that plane (Catmull-Rom between steps, 8 points per step; 0.5 pt lines at
                   alpha 0.8) and a filled circle at t = 30, in its true cell's colour (the bivariate key); hollow
                   circles: the 16 cells' targets (targets: the Optimal's mean T = 30 belief over the episodes in which
                   it is right about both values of the cell). Each panel's data box is its movie's: a square holding
                   the trajectories and targets of every 6th frame of the movie's schedule and of its last frame, with
                   a 6% margin
        strips     under each panel, the step planes: the single-PC latent R^2 (r_g + r_c) of each step's PCs 1-3
                   (white 0, black 1), the pair shown (red dots), the cursor at t = 30; over the panel, the pair shown
        furniture  the clock on top (t = 30, the bar of the 30 steps full); on the right, the latent space with the
                   16 drawn cells lit, the key of the marks and a note on the draw (with the group's FR cut, nats)
    Data, through the data layer (each file built when missing): beliefs_steps.npz (the networks at every step, the
    ESN replayed on the trained network's episodes), episodes.npz (the Bayes observers' beliefs, which also give FR
    and the targets), record.npz (each episode's variables and values); the draws are checked against
    traj_fr_draw_{low,high}.npz when it is there. Style: plotting_helpers.style("movie"), whatever ran before. About
    3 s on the CPU with the data in place.
    Run from the repository root with the eureka environment's python:
        python main/REPLICATION/plot_fig1c_all.py traj_fr [--out DIR]"""
    name = "fig1c_traj_FR_groups_T30.png"
    BV, rg, rc = beliefs_every_step()
    rg, rc = np.asarray(rg), np.asarray(rc)
    S = {a: step_planes(BV[a], rg, rc) for a in ("Trained", "Echo", "Joint", "Naive")}
    home, _ = targets([(g, c) for g in FR_GROUP_SUB for c in FR_GROUP_SUB])
    a, b = (traj_fr_still(*traj_fr_group(g, BV, rg, rc, S, home)) for g in ("low", "high"))
    out = Image.new("RGB", (a.size[0] + b.size[0], a.size[1]), "white")
    out.paste(a, (0, 0))
    out.paste(b, (a.size[0], 0))
    path = figure_path(name, out_dir)
    out.save(path)
    return [path]


# ═══════════════════════════════════════════════════════════════════
# Figure: the step-plane movie (fig1c_movie_pcs.gif)
# ═══════════════════════════════════════════════════════════════════

def _pcs_data():
    """per agent: its beliefs at every step V (N_EP, T, 20), their colours, its step planes S, the four cells' mean
    belief paths M (MOVIE_SUB points per step, 20-dim) at step indices tt, the targets (20-dim) -> data, cells"""
    G = belief_geometry()
    BV, rg, rc = beliefs_every_step()
    AV, a_of, cells, _ = alias_trajectories()
    home_v, _ = targets(cells)
    tt = np.linspace(0, N_T - 1, (N_T - 1) * MOVIE_SUB + 1)
    out = {}
    for a in BV:
        S = step_planes(BV[a], rg, rc)
        for tau, t in ((0, 1), (N_T - 1, N_T)):                                   # the first / last frames are
            assert np.abs(plane_at(S, BV[a][:, tau], tau) - G[(a, t)]["X"]).max() < 2e-3, \
                f"{a}: step {t} is not belief_geometry's cloud"                   # belief_geometry's clouds
        M = [np.stack([ph.blend(AV[a][a_of == j].mean(0), tau) for tau in tt]) for j in range(len(cells))]
        out[a] = dict(S=S, V=BV[a], col=ph.bivariate(rg, rc), home=home_v, tt=tt, M=M)
    return out, cells


def _pcs_trails(D, tau):
    """the cells' mean belief paths seen in the plane of frame tau -> [(tt, (len(tt), 2))]"""
    return [(D["tt"], plane_at(D["S"], M_, tau)) for M_ in D["M"]]


def _pcs_limits(D, taus):
    """one square data box holding the cloud, the paths and the targets of every frame taus (8% margin)"""
    ext = []
    for tau in taus:
        pts = np.concatenate([plane_at(D["S"], ph.blend(D["V"], tau, axis=1), tau), plane_at(D["S"], D["home"], tau)]
                             + [P for _, P in _pcs_trails(D, tau)])
        ext += [pts.min(0), pts.max(0)]
    return ph.square_limits(np.stack(ext), 1.08)


def _pcs_key(fig, cells, H):
    """the right column: the latent space with the cells' markers, the key of the markers and of the paths, the note"""
    x, top, ks, W = MOVIE_LEG_X, H - .62, 1.15, MOVIE_W
    fig.text((x + ks / 2) / W, top / H, "Latent space", ha="center", va="baseline", fontsize=MOVIE_FS["lab"])
    ax = fig.add_axes([(x + .12) / W, (top - .1 - ks) / H, ks / W, ks / H])
    ph.latent_key(ax, R=N_R)
    for (rg_, rc_), m in zip(cells, CELL_MARKS):
        ax.scatter(rg_, rc_, marker=m, s=(.62 * MOVIE_MS_HOME * CELL_MARK_SCALE_KEY.get(m, 1)) ** 2, facecolor="white",
                   edgecolor="k", lw=.8, clip_on=False, zorder=3)
    ax.tick_params(length=1.5, pad=1)
    ax.set_xlabel("$r_g$", labelpad=-4)
    ax.set_ylabel("$r_c$", labelpad=-4)
    y = top - ks - .62
    for mk, ms, mew, fc, lab in (("o", MOVIE_MS_START, .8, "white", "start ($t = 1$)"),
                                 ("o", MOVIE_MS_END, .8, "0.55", "now"), ("o", MOVIE_MS_HOME, 1.1, "white", "target")):
        fig.add_artist(Line2D([(x + .2) / W], [y / H], transform=fig.transFigure, marker=mk, ms=ms, mfc=fc,
                              mec="k", mew=mew, ls="none"))
        fig.text((x + .45) / W, y / H, lab, ha="left", va="center", fontsize=MOVIE_FS["lab"], color="0.25")
        y -= .3
    ex = np.linspace(x + .05, x + 1.05, 40)                                       # a path: t = 1 -> T
    ev = np.log1p((np.arange(39) + .5) / 39 * (N_T - 1)) / np.log(N_T)
    widths, greys = ph.tail_style(ev, MOVIE_SCALE)
    fig.add_artist(LineCollection(np.stack([np.stack([ex[:-1], ex[1:]], 1) / W, np.full((39, 2), y / H)], -1),
                                  transform=fig.transFigure, colors=greys, linewidths=widths, capstyle="round"))
    fig.text(ex[0] / W, (y - .2) / H, "$t = 1$", ha="left", va="baseline", fontsize=MOVIE_FS["tick"], color="0.4")
    fig.text(ex[-1] / W, (y - .2) / H, "$T$", ha="right", va="baseline", fontsize=MOVIE_FS["tick"], color="0.4")
    note = (f"{N_EP:,} episodes,\ncoloured by $(r_g, r_c)$\n\n"
            "each step's own PC plane\n(the 2 of PCs 1-3 holding\nthe most latent variance)\n\n"
            "strips: latent $R^2$ of\n"
            "each step's PCs 1-3\n(white 0, black 1); dots:\nthe pair shown"
            "\n\ncurves: a cell's mean\nbelief over %d streams\nof one environment" % ALIAS_M)
    fig.text(x / W, (y - .42) / H, note, ha="left", va="top", fontsize=MOVIE_FS["note"], color="0.3", linespacing=1.25)


def _pcs_build(data, cells):
    """the figure and its moving artists -> fig, update(tau) (tau: the step index, 0 .. T - 1, continuous)"""
    taus = [tau for tau, _ in ph.schedule(T=N_T)][::6] + [N_T - 1.]
    W, BS = MOVIE_W, MOVIE_BS
    row_h = MOVIE_YNAME + BS + (MOVIE_STRIP + MOVIE_STRIP_GAP)
    H = MOVIE_YTOP + 2 * row_h + .12
    fig = plt.figure(figsize=(W, H), dpi=MOVIE_DPI, facecolor="white")
    moving, cursors, notes = [], [], []
    for r, agents in enumerate(MOVIE_AGENTS):
        y = H - MOVIE_YTOP - MOVIE_YNAME - BS - r * row_h
        for p, a in enumerate(agents):
            D = data[a]
            x = MOVIE_X0 + p * (BS + MOVIE_XGAP)
            ax = ph.ax_in(fig, x, y, BS, BS)
            xl, yl = _pcs_limits(D, taus)
            ax.set_xlim(*xl), ax.set_ylim(*yl)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
            fig.text((x + BS / 2) / W, (y + BS + .22) / H, ph.AGENT_NAME_BAYES[a], ha="center", va="baseline",
                     fontsize=MOVIE_FS["name"], color=ph.AGENT_COLOR[a])
            plane = fig.text((x + BS / 2) / W, (y + BS + .07) / H, "", ha="center", va="baseline",
                             fontsize=MOVIE_FS["note"], color="0.45")
            cursors.append(movie_strip(fig, x, y, D["S"]))
            notes.append((plane, D["S"]["pcs"]))
            sc = ax.scatter(*plane_at(D["S"], D["V"][:, 0], 0.).T, lw=0, zorder=1, marker="s",
                            s=(MOVIE_PX * 72 / MOVIE_DPI) ** 2, antialiased=False, c=D["col"])
            homes, halos, lines, starts, heads = [], [], [], [], []
            for j, (rg_, rc_) in enumerate(cells):
                mk = CELL_MARKS[j]
                sf = CELL_MARK_SCALE.get(mk, 1)
                homes.append(ax.plot([], [], marker=mk, ms=MOVIE_MS_HOME * sf, mfc="white", mec="k", mew=1.1,
                                     ls="none", zorder=6.8, clip_on=False,
                                     path_effects=[pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])[0])
                halos.append(ax.add_collection(LineCollection([], colors="white", capstyle="round", zorder=6.9)))
                lines.append(ax.add_collection(LineCollection([], capstyle="round", zorder=7)))
                starts.append(ax.plot([], [], marker=mk, ms=MOVIE_MS_START * sf, mfc="white", mec="k", mew=.8,
                                      ls="none", zorder=8 if sf > 1 else 8.1, clip_on=False)[0])
                heads.append(ax.plot([], [], marker=mk, ms=MOVIE_MS_END * sf, mfc=ph.bivariate(rg_, rc_), mec="k",
                                     mew=.8, ls="none", zorder=8.2 if sf > 1 else 8.3, clip_on=False)[0])
            moving.append((D, sc, homes, halos, lines, starts, heads))
    set_clock = ph.clock_bar(fig, MOVIE_X0, H - .33, MOVIE_X0 + .95, MOVIE_X0 + 2 * BS + MOVIE_XGAP, T=N_T,
                             fs_t=MOVIE_FS["t"], fs_tick=MOVIE_FS["tick"])
    _pcs_key(fig, cells, H)

    def update(tau):
        for D, sc, homes, halos, lines, starts, heads in moving:
            sc.set_offsets(plane_at(D["S"], ph.blend(D["V"], tau, axis=1), tau))
            Hm = plane_at(D["S"], D["home"], tau)
            for j, (tt, P) in enumerate(_pcs_trails(D, tau)):
                homes[j].set_data([Hm[j, 0]], [Hm[j, 1]])
                starts[j].set_data([P[0, 0]], [P[0, 1]])
                seg, e, q = ph.tail(tt, P, tau, T=N_T)
                wd, greys = ph.tail_style(e, MOVIE_SCALE)
                halos[j].set_segments(seg), halos[j].set_linewidths(wd + 1.4)
                lines[j].set_segments(seg), lines[j].set_linewidths(wd)
                lines[j].set_colors(greys)
                heads[j].set_data([q[0]], [q[1]])
        for cur in cursors:
            cur.set_xdata([tau + 1, tau + 1])
        for txt, pcs in notes:                                                    # the plane of the nearest step
            txt.set_text("PCs %d-%d of step $t$" % tuple(pcs[int(np.floor(tau + .5))]))
        set_clock(tau)

    return fig, update


def figure_pcs(out_dir=None):
    """The step-plane movie: the four agents' beliefs at every step t = 1 .. T = 30, each step seen in its own plane,
    with the mean belief paths of four cells of one environment growing through time -> fig1c_movie_pcs.gif in out_dir
    (default main/REPLICATION/figure_bin; 7.05 x 7.55 in at 150 dpi: 1057 x 1132 px, 376 frames, 16.7 s) -> [its path].
        panels     2 x 2: Trained RNN and ESN over Optimal Bayes and Naive Bayes. Each panel shows the agent's beliefs
                   on the trained harvest's first 20,000 episodes at the frame's step, one 1-pixel square per episode
                   in the colour of its cell (r_g, r_c) (plotting_helpers.bivariate), in that step's own plane: of the
                   PCA of the agent's goal-ordered beliefs (20 numbers) at the step, the 2 of PCs 1-3 that hold the
                   most latent variance (single-PC R^2 of r_g plus r_c), turned so that the ridge decoder of r_g points
                   right and that of r_c up (step_planes, the rule of belief_geometry; the first and last frames are
                   belief_geometry's t = 1 and T = 30 clouds, checked to 2e-3). Between steps the beliefs, the planes'
                   means and the planes' axes are Catmull-Rom blends through the steps, the axes made orthonormal
                   again, so the drawn plane never shrinks a distance. Each panel's data box is fixed: a square holding
                   the cloud, paths and targets of every 6th frame and of the last, its side 1.08 times their larger
                   extent
        paths      the four cells (0, 2), (5, 8), (9, 3), (9, 7) of one environment (harvest episode 1434): each cell's
                   mean belief over its 2000 simulated streams, from t = 1 to the frame's step, the whole path seen in
                   the frame's own plane (a trace through the changing planes would draw their turn as motion); light
                   and thin at t = 1 to black and thick at T, in log t; from a white marker (t = 1) to a marker in the
                   cell's colour (now); a large white marker at the cell's target, the Optimal's mean T = 30 belief
                   over the episodes (of the 20,000) in which it is right about both values of that cell (targets)
        strips     under each panel, the step planes: the single-PC latent R^2 of each step's PCs 1-3 (white 0, black
                   1), the pair shown (red dots) and a cursor at the frame's step; over the panel, the pair at the
                   nearest step
        furniture  the clock on top (t, and a bar of the 30 steps); on the right, the latent space with the cells'
                   markers, the key of the markers and the paths, and a note
    Timing (plotting_helpers.schedule, 25 fps): 0.5 s at t = 1; 15 s of motion, the clock running as sqrt(t) (time per
    step ~ 1 / sqrt(t + 1): beliefs move most early on, ~12% of the panel from t = 1 to 2, < 1% late), eased in from
    rest over 0.7 s; 1.2 s at T. One 255-colour palette for every frame, no dithering, so still regions do not flicker:
    PIL's median cut of four sample frames and of swatches of the key colours (the names, the greys, the cells'
    colours).
    Data, through the data layer: beliefs_steps.npz (the networks at every step, the ESN replayed on the trained
    network's episodes), episodes.npz (the Bayes observers), record.npz (each episode's goal variable and values),
    alias_traj.npz (the paths), beliefs.npz (belief_geometry, for the check of the first and last frames).
    Style: plotting_helpers.style("movie"), whatever ran before. About 2.5 min on the CPU, 1.7 GB.
    Run from the repository root with the eureka environment's python:
        python main/REPLICATION/plot_fig1c_all.py pcs [--out DIR]"""
    t0, name = time.time(), "fig1c_movie_pcs.gif"
    data, cells = _pcs_data()
    sched = ph.schedule(T=N_T)
    frames = []
    with ph.style("movie"):
        fig, update = _pcs_build(data, cells)
        try:
            protos = []
            for tau in (0., 4., 9., N_T - 1.):                                    # the palette: four frames and
                update(tau)                                                       # swatches of the key colours (the
                protos.append(ph.grab(fig))                                       # names, the greys, the cells')
            keys = [to_rgb(c) for c in (*ph.AGENT_COLOR.values(), "0.25", "0.3", "0.35", "0.4", "0.45", "0.92",
                                        "white")]
            keys += [tuple(ph.bivariate(g, c)) for g, c in cells]
            pal = ph.median_cut_palette(protos, keys)
            for i, (tau, _) in enumerate(sched):
                update(tau)
                frames.append(ph.quantize(ph.grab(fig), pal))
                if i % 50 == 0:
                    print(f"pcs: frame {i + 1} of {len(sched)}", flush=True)
        finally:
            plt.close(fig)
    path = ph.save_gif(frames, [ms for _, ms in sched], figure_path(name, out_dir))
    print(f"saved {path}: {len(frames)} frames, {sum(ms for _, ms in sched) / 1000:.1f} s, {frames[0].size[0]} x "
          f"{frames[0].size[1]} px, {os.path.getsize(path) / 1e6:.1f} MB, made in {(time.time() - t0) / 60:.1f} min")
    return [path]


# ═══════════════════════════════════════════════════════════════════
# Figure: beliefs and every environment's grid in each agent's T = 30 plane (fig1c_movie_fixed_envdist.gif)
# ═══════════════════════════════════════════════════════════════════

def envdist_data():
    """what the envdist movie draws, per agent (Joint, Trained, Naive, Echo): plane, its T = 30 plane
    (belief_geometry); V (N_EP, T, 20) float32, its goal-ordered beliefs at every step on the first N_EP episodes
    (beliefs_every_step) and col (N_EP, 3), their cells' colours; home (4, 2), the four alias cells' targets in the
    plane; paths, each alias cell's mean belief (over its streams) at the step indices tt by blend, projected point by
    point -> 4 x (len(tt), 2); Mn (39, 100, T, 20) float32, the scan's cell means (each cell's mean belief over its
    SCAN_M streams) in the 39 environments, and grid (39, 100, T, 2), each step's projected; grid_home (100, 2), the
    100 cells' targets in the plane. Shared: tt (8 points per step), cells (the four), edges (the 100 cells' grid
    edges), envs (the scan's environments, e = SCAN_LEFT_OUT left out)"""
    T = N_T
    G = belief_geometry()
    BV, rg, rc = beliefs_every_step()
    AV, a_of, cells, _ = alias_trajectories()
    home_v, _ = targets(cells)
    grid_cells = np.array(scan_meta("cells"))
    assert np.array_equal(scan_meta("cell_of"), np.repeat(np.arange(len(grid_cells)), SCAN_M))
    grid_home_v, _ = targets([tuple(int(v) for v in c) for c in grid_cells])
    envs = tuple(e for e in range(SCAN_N_ENV) if e != SCAN_LEFT_OUT)
    tt = np.linspace(0, T - 1, (T - 1) * 8 + 1)
    col = ph.bivariate(rg, rc)
    out = dict(tt=tt, cells=cells, edges=ph.grid_edges(grid_cells), envs=envs)
    for a in AGENTS4:
        plane = G[(a, T)]
        assert np.abs(project_plane(plane, BV[a][:, T - 1]) - plane["X"]).max() < 2e-3, \
            f"{a}: step T is not the plane's own cloud"                          # beliefs_steps vs beliefs.npz
        paths = []
        for j in range(len(cells)):
            m = AV[a][a_of == j].mean(0)                                          # (T, 20) float32
            M = np.stack([ph.blend(m, tau) for tau in tt])
            paths.append(np.stack([project_plane(plane, v) for v in M]))
        Mn = scan_cell_means(a, envs)
        out[a] = dict(plane=plane, V=BV[a], col=col, home=project_plane(plane, home_v), paths=paths, Mn=Mn,
                      grid=np.stack([project_plane(plane, Mn[:, :, t]) for t in range(T)], 2),
                      grid_home=project_plane(plane, grid_home_v))
    return out


def figure_envdist(out_dir=None):
    """The beliefs and every environment's grid in each agent's own T = 30 plane, as one 2 x 4 movie ->
    fig1c_movie_fixed_envdist.gif in out_dir (default main/REPLICATION/figure_bin; 1655 x 782 px, 376 frames, 16.7 s)
    -> [its path].

    Columns: Optimal Bayes, Trained RNN, Naive Bayes, ESN. Every panel is the agent's own T = 30 PC plane
    (belief_geometry: the 2 of PCs 1-3 of its T = 30 beliefs holding the most latent variance, turned so that the r_g
    decoder points right and the r_c decoder up; its PCs are named under the agent's name), one projection in both rows
    and in every frame, at one scale per agent (one square box holds both rows' contents at every step).
        row 1     the beliefs of the trained harvest's first 20,000 episodes (the ESN replayed on the same episodes),
                  each a 1-pixel square coloured by its true cell (r_g, r_c) (the bivariate key); the last frame's
                  clouds are belief_geometry's T = 30 clouds. Over them the mean belief paths of the four alias cells
                  (alias_traj: harvest episode 1434, cells (0, 2), (5, 8), (9, 3), (9, 7); a cell's mean belief over
                  its 2000 streams, 8 points per step), drawn up to the frame, light and thin at t = 1 and black and
                  thick at T (fixed in absolute time); their markers (circle, square, triangle, star): start (t = 1),
                  now (in the cell's colour) and target, where a correct belief about the cell sits (targets: the
                  Optimal's mean T = 30 belief over the episodes in which it gets both values right)
        row 2     the grids of the alias scan's 39 environments (e = 2, an exactly symmetric table, left out): each
                  cell's mean belief over its 50 streams, joined to its grid neighbours; each edge coloured by its
                  distance from the environments' average grid (black), the mean over its two cells of their distance
                  from their place in the average grid, in cells of the targets' grid (its mean edge length), 0-3;
                  dashed grey: the targets' grid (the 100 cells' targets)
        column 5  the keys: the latent space with the four cells' markers (beside row 1); the markers and paths of
                  row 1, the colour bar and lines of row 2 (beside row 2)
    Timing: schedule() at 25 frames per second (a 0.5 s hold at t = 1; the steps over 15 s, time per step ~ 1 / sqrt(t
    + 1), eased in from rest; a 1.2 s hold at T); between steps the beliefs and the cell means by the Catmull-Rom blend
    through the steps.
    Data (envdist_data): record.npz and beliefs.npz (the planes), beliefs_steps.npz and episodes.npz (the beliefs at
    every step, the targets), alias_traj.npz (the paths), scan_meta.npz and scan_<agent>.npz (the grids) of the data
    folder.
    Style: plotting_helpers.style("movie") at 100 dpi, whatever ran before. GIF palette: the agents' colours, the 100
    cells' colours, 26 greys and the distance colour ramp (full and at the grids' alpha over white) pinned, the rest a
    median cut of 4 frames; PIL's matching (the white background comes out (253, 251, 250)). About 3-5 min on the CPU.
    Run from the repository root with the eureka environment's python:
        python main/REPLICATION/plot_fig1c_all.py envdist [--out DIR]"""
    name = "fig1c_movie_fixed_envdist.gif"
    T = N_T
    DPI = 100
    BS, XG, X0 = 2.95, .3, .55                                                   # panel side, column gap, left edge
    Y_CLOCK, Y_NAME, Y_GAP = .8, .62, .2                                         # top band, names band, row gap
    KX, KW = X0 + 4 * BS + 3 * XG + .5, 2.55                                     # the keys' column: left edge, width
    W = KX + KW + .25
    H = Y_CLOCK + Y_NAME + BS + Y_GAP + BS + .3
    FS = dict(t=15, name=13.5, plane=10, row=11.5, lab=12, key=11, tick=10)
    ENV_LW, ENV_ALPHA, DIST_MAX = .5, .7, 3.                                     # the grids' lines; the colour scale
    data = envdist_data()
    tt, cells, edges, envs = data["tt"], data["cells"], data["edges"], data["envs"]
    sched = ph.schedule()
    scale = BS / 1.11                                                            # the side over the stills' 1.11
    with ph.style("movie"):
        fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor="white")
        dist_norm = plt.Normalize(0, DIST_MAX)
        y1 = H - Y_CLOCK - Y_NAME - BS
        y2 = y1 - Y_GAP - BS
        taus = [tau for tau, _ in sched][::6] + [T - 1.]
        cols = []
        for c_, a in enumerate(AGENTS4):                                         # the columns
            D = data[a]
            x = X0 + c_ * (BS + XG)
            # one square box per agent for both rows: the cloud (every 6th frame), the paths and the targets (8%
            # margin), then every environment's grid at every step and the targets' grid (2%)
            ext = []
            for tau in taus:
                pts = np.concatenate([project_plane(D["plane"], ph.blend(D["V"], tau, axis=1)), D["home"]]
                                     + D["paths"])
                ext += [pts.min(0), pts.max(0)]
            box = np.array(ph.square_limits(np.stack(ext), 1.08)).T                      # [[x0, y0], [x1, y1]]
            xl, yl = ph.square_limits(np.concatenate([box, D["grid"].reshape(-1, 2), D["grid_home"]]), 1.02)
            fig.text((x + BS / 2) / W, (y1 + BS + .3) / H, ph.AGENT_NAME_BAYES[a], ha="center", va="baseline",
                     fontsize=FS["name"], color=ph.AGENT_COLOR[a])
            fig.text((x + BS / 2) / W, (y1 + BS + .1) / H, "$T = 30$ PCs %d-%d" % D["plane"]["pcs"], ha="center",
                     va="baseline", fontsize=FS["plane"], color="0.45")
            # row 1: the beliefs, the cells' paths and markers
            a1 = ph.ax_in(fig, x, y1, BS, BS)
            a1.set_xlim(*xl), a1.set_ylim(*yl), a1.set_aspect("equal", adjustable="box"), a1.axis("off")
            sc = a1.scatter(*project_plane(D["plane"], D["V"][:, 0]).T, lw=0, zorder=1, marker="s",
                            s=(MOVIE_PX * 72 / DPI) ** 2, antialiased=False, c=D["col"])
            halos, lines, heads = [], [], []
            for j, (rg_, rc_) in enumerate(cells):
                mk = CELL_MARKS[j]
                sf = CELL_MARK_SCALE.get(mk, 1)
                a1.plot([D["home"][j, 0]], [D["home"][j, 1]], marker=mk, ms=MOVIE_MS_HOME * sf, mfc="white", mec="k",
                        mew=1.1, ls="none", zorder=6.8, clip_on=False,
                        path_effects=[pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])
                halos.append(a1.add_collection(LineCollection([], colors="white", capstyle="round", zorder=6.9)))
                lines.append(a1.add_collection(LineCollection([], capstyle="round", zorder=7)))
                a1.plot([D["paths"][j][0, 0]], [D["paths"][j][0, 1]], marker=mk, ms=MOVIE_MS_START * sf, mfc="white",
                        mec="k", mew=.8, ls="none", zorder=8 if sf > 1 else 8.1, clip_on=False)
                heads.append(a1.plot([], [], marker=mk, ms=MOVIE_MS_END * sf, mfc=ph.bivariate(rg_, rc_), mec="k",
                                     mew=.8, ls="none", zorder=8.2 if sf > 1 else 8.3, clip_on=False)[0])
            # row 2: every environment's grid
            a2 = ph.ax_in(fig, x, y2, BS, BS)
            a2.set_xlim(*xl), a2.set_ylim(*yl), a2.set_aspect("equal", adjustable="box")
            a2.set_xticks([]), a2.set_yticks([])
            for sp in a2.spines.values():
                sp.set_color("0.75"), sp.set_linewidth(.6)
            hm = D["grid_home"]
            a2.add_collection(LineCollection(hm[edges], colors="0.78", linewidths=.55, linestyles=(0, (3, 2)),
                                             zorder=0))
            unit = np.linalg.norm(hm[edges[:, 0]] - hm[edges[:, 1]], axis=1).mean()   # one cell of the targets' grid
            env_lc = []
            for k in range(len(envs)):
                lc = LineCollection(D["grid"][k, :, 0][edges], cmap=ph.FR_CMAP, norm=dist_norm, linewidths=ENV_LW,
                                    alpha=ENV_ALPHA, zorder=2)
                lc.set_array(np.zeros(len(edges)))
                env_lc.append(a2.add_collection(lc))
            avg = a2.add_collection(LineCollection(D["grid"][:, :, 0].mean(0)[edges], colors="k", linewidths=1.2,
                                                   zorder=4))
            cols.append(dict(D=D, sc=sc, halos=halos, lines=lines, heads=heads, env_lc=env_lc, avg=avg, unit=unit))
        for y, txt in ((y1, f"beliefs of {N_EP:,} episodes"), (y2, f"grids of {len(envs)} environments")):
            fig.text((X0 - .22) / W, (y + BS / 2) / H, txt, rotation=90, ha="center", va="center", fontsize=FS["row"],
                     color="0.3")
        clock = ph.clock_bar(fig, X0, H - .42, X0 + 1.15, X0 + 4 * BS + 3 * XG, T, h=.12, dy=.3, fs_t=FS["t"],
                             fs_tick=FS["tick"])
        # column 5, beside row 1: the latent space, large
        ks = 2.35
        kx = KX + (KW - ks) / 2 + .12
        fig.text((kx + ks / 2) / W, (y1 + .3 + ks + .1) / H, "Latent space", ha="center", va="bottom",
                 fontsize=FS["lab"])
        ax = ph.ax_in(fig, kx, y1 + .3, ks, ks)
        ph.latent_key(ax, R=N_R)
        for (rg_, rc_), m in zip(cells, CELL_MARKS):
            ax.scatter(rg_, rc_, marker=m, s=(.8 * MOVIE_MS_HOME * CELL_MARK_SCALE_KEY.get(m, 1)) ** 2,
                       facecolor="white", edgecolor="k", lw=1., clip_on=False, zorder=3)
        ax.tick_params(length=2, pad=1.5, labelsize=FS["tick"])
        ax.set_xlabel("$r_g$", labelpad=-6, fontsize=FS["lab"])
        ax.set_ylabel("$r_c$", labelpad=-6, fontsize=FS["lab"])
        # beside row 2: the markers and paths of row 1
        x = KX + .05
        y = y2 + BS - .12
        for ms, mew, fc, lab in ((MOVIE_MS_START, .8, "white", "start ($t = 1$)"), (MOVIE_MS_END, .8, "0.55", "now"),
                                 (MOVIE_MS_HOME, 1.1, "white", "target")):
            fig.add_artist(Line2D([(x + .2) / W], [y / H], transform=fig.transFigure, marker="o", ms=ms * .85, mfc=fc,
                                  mec="k", mew=mew, ls="none"))
            fig.text((x + .5) / W, y / H, lab, ha="left", va="center", fontsize=FS["key"], color="0.2")
            y -= .34
        ex = np.linspace(x + .02, x + .4, 30)
        ev = np.log1p((np.arange(29) + .5) / 29 * (T - 1)) / np.log(T)               # a path's log time, t = 1 -> T
        wd, greys = ph.tail_style(ev, scale)
        fig.add_artist(LineCollection(np.stack([np.stack([ex[:-1], ex[1:]], 1) / W, np.full((29, 2), y / H)], -1),
                                      transform=fig.transFigure, colors=greys, linewidths=wd, capstyle="round"))
        fig.text((x + .5) / W, y / H, "a cell's mean path", ha="left", va="center", fontsize=FS["key"], color="0.2")
        # beside row 2: the colour bar and lines of row 2
        yb = y - .72
        cax = ph.ax_in(fig, x + .05, yb, KW - .25, .14)
        cb = fig.colorbar(plt.cm.ScalarMappable(cmap=ph.FR_CMAP, norm=dist_norm), cax=cax, orientation="horizontal",
                          extend="max")
        cb.ax.tick_params(labelsize=FS["tick"], length=2, pad=1)
        cb.outline.set_visible(False)
        fig.text((x + .05) / W, (yb + .24) / H, "distance from the average (cells)", ha="left", va="bottom",
                 fontsize=FS["key"], color="0.2")
        y = yb - .55
        for ls, c, lw, lab in (("-", "k", 1.4, "average grid"), ((0, (3, 2)), "0.6", 1., "targets' grid")):
            fig.add_artist(Line2D([(x + .02) / W, (x + .4) / W], [y / H] * 2, transform=fig.transFigure, c=c, lw=lw,
                                  ls=ls))
            fig.text((x + .5) / W, y / H, lab, ha="left", va="center", fontsize=FS["key"], color="0.2")
            y -= .32

        def update(tau):
            for col in cols:
                D = col["D"]
                col["sc"].set_offsets(project_plane(D["plane"], ph.blend(D["V"], tau, axis=1)))
                for j, P in enumerate(D["paths"]):
                    seg, e, q = ph.tail(tt, P, tau, T)
                    wd, greys = ph.tail_style(e, scale)
                    col["halos"][j].set_segments(seg), col["halos"][j].set_linewidths(wd + 1.4)
                    col["lines"][j].set_segments(seg), col["lines"][j].set_linewidths(wd)
                    col["lines"][j].set_colors(greys)
                    col["heads"][j].set_data([q[0]], [q[1]])
                Mt = ph.blend(D["Mn"], tau, axis=2)                                   # (E, 100, 20)
                Pt = np.stack([project_plane(D["plane"], Mt[k]) for k in range(len(Mt))])
                pa = Pt.mean(0)                                                       # the average grid
                col["avg"].set_segments(pa[edges])
                for k, lc in enumerate(col["env_lc"]):
                    dv = np.linalg.norm(Pt[k] - pa, axis=1)
                    lc.set_segments(Pt[k][edges])
                    lc.set_array((dv[edges[:, 0]] + dv[edges[:, 1]]) / 2 / col["unit"])
            clock(tau)

        protos = []
        for tau in (0., 4., 9., T - 1.):
            update(tau)
            protos.append(ph.grab(fig))
        exact = [to_rgb(c) for c in ph.AGENT_COLOR.values()]
        exact += [tuple(np.asarray(ph.bivariate(g, c), np.float64)[:3]) for g in range(N_R) for c in range(N_R)]
        exact += [(v, v, v) for v in np.linspace(0, 1, 26)]
        ramp = [np.asarray(ph.FR_CMAP(v)[:3]) for v in np.linspace(0, 1, 24)]
        exact += [tuple(c) for c in ramp] + [tuple(ENV_ALPHA * c + (1 - ENV_ALPHA)) for c in ramp]
        pal, pcols, n_exact = ph.exact_palette(protos, exact)
        print(f"palette: {n_exact} exact colours + {len(pcols) - n_exact} from the frames", flush=True)
        frames = []
        for i, (tau, _) in enumerate(sched):
            update(tau)
            frames.append(ph.quantize(ph.grab(fig), pal))
            if i % 50 == 0:
                print(f"frame {i + 1} of {len(sched)}", flush=True)
        plt.close(fig)
    path = ph.save_gif(frames, [ms for _, ms in sched], figure_path(name, out_dir))
    print(f"saved {path}: {len(frames)} frames, {sum(ms for _, ms in sched) / 1000:.1f} s, {frames[0].size[0]} x "
          f"{frames[0].size[1]} px, {os.path.getsize(path) / 1e6:.1f} MB", flush=True)
    return [path]


# ═══════════════════════════════════════════════════════════════════
# Figure: the hybrid observer's sweep (fig1c_movie_hybrid_sweep_combined.gif)
# ═══════════════════════════════════════════════════════════════════

SWEEP_TEXT = dict(                                       # every text of the sweep movie ({}: filled in when drawn)
    clock="joint evidence at {p:.0f}% of the stream-steps ({n_env} envs x 50 streams per cell)",
    row_top="the environments' average grid",
    row_bottom="each environment's grid (black: their average; dashed grey: the true lattice)",
    head_left="least entangled steps first",
    head_right="most entangled steps first",
    thr_left="joint where entanglement < {thr:.2f} nats",
    thr_right="joint where entanglement > {thr:.2f} nats",
    naive="no joint evidence yet (the Naive)",
    cbar_top="FR of the edge's cells at T (nats)",
    cbar_bottom="distance from the average grid (cells)",
    curve_title="unfolding",
    curve_x="steps (%)",
    curve_y="grid error")


def figure_sweep(out_dir=None):
    """Where the unfolding comes from, as one movie: a hybrid observer that takes the joint evidence at a growing share
    of the stream-steps, the least or the most entangled first -> fig1c_movie_hybrid_sweep_combined.gif in out_dir
    (default main/REPLICATION/figure_bin; 1311 x 885 px, 201 frames, 17.1 s) -> [its path].

    The hybrid observer keeps a joint belief over (r_g, r_c) (uniform prior). At the selected stream-steps it multiplies
    in the step's joint likelihood, as the Optimal does; at all other steps the Naive's factorized likelihood (each
    variable's table averaged over the other variable). All steps joint is the Optimal, none is the Naive. In each of
    the 100 environments (100 cells x 50 streams x 30 steps), the stream-steps are ranked by their entanglement (the
    mean over both variables of half the Jeffreys divergence between the Naive's step likelihood and the Optimal's
    history-conditioned update, nats), and the joint evidence goes to the p% least entangled (left column, blue) or the
    p% most entangled (right column, red), p = 0, 2, ..., 100 (each set holds the one before). The threshold in each
    column's title is the entanglement at the selection's edge, median over the environments. The frames move linearly
    from one p to the next in 4 frames of 70 ms, with holds of 1.2 s at 0% and 2 s at 100%.
        top row     the environments' average grid: each cell's posterior means (E[r_g], E[r_c]) at T = 30 (of its
                    mean belief over its 50 streams), averaged over the environments and joined to its grid
                    neighbours; each edge coloured by the FR of its two cells (their geometric mean; a cell's FR: half
                    the Jeffreys divergence between the Optimal's and the Naive's goal-variable marginals at T, mean
                    over its streams and the environments, floored at 1e-6; nats, log scale .05-30, the same in every
                    frame); dashed grey: the true lattice (r_g, r_c)
        bottom row  each environment's grid, each edge coloured by its distance from the environments' average grid
                    (black): the mean over its two cells of their distance from their place in the average grid
                    (grid units, 0-3)
        right       unfolding: the grid error, the mean distance of the average grid's 100 cells from the true lattice
                    (grid units), against p; faint: the whole curve, bold up to the frame, dot: the frame
    Data: sweep100.npz of the data layer (cm, the cells' mean beliefs at T (2, 51, 100, 100, 20); thr, the
    entanglement at each selection's edge (2, 51, 100); fr (100, 100); sweep, p), built from the record by
    build_sweep100 when it is missing (numpy only, about 3 min).
    Style: matplotlib's defaults (DejaVu Sans) at 100 dpi, whatever ran before. GIF palette: white, black, the column
    colours, 24 greys and the FR colour ramp at 32 points (also mixed 65:35 with white) pinned, the rest a median cut of
    4 frames; PIL's matching (the white background comes out (253, 253, 252)). 1-2 min on the CPU.
    Run from the repository root with the eureka environment's python:
        python main/REPLICATION/plot_fig1c_all.py sweep [--out DIR]"""
    name = "fig1c_movie_hybrid_sweep_combined.gif"
    TX, dist_max = SWEEP_TEXT, 3.
    with load_cache("sweep100") as Z:
        cm, thr, sw, fr = Z["cm"], Z["thr"], Z["sweep"].astype(float), Z["fr"]
    nE = cm.shape[2]
    cells = np.array(SCAN_CELLS)                                                 # the cache's cell order
    fr = np.maximum(fr.mean(0), 1e-6)
    edges = ph.grid_edges(cells)
    P = ph.posterior_means(cm, normalize=None)                                   # (2, levels, nE, 100, 2)
    Pa = P.mean(2)
    d = np.linalg.norm(Pa - cells, axis=-1).mean(-1)                             # the grid error (2, levels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                          # p = 0: nothing selected, nan
        th = np.nanmedian(thr, -1)
    frn, dn = LogNorm(.05, 30), plt.Normalize(0, dist_max)
    ecol = np.sqrt(fr[edges[:, 0]] * fr[edges[:, 1]])
    FSZ = dict(clock=15, row=12, title=11.5, lab=10.5, tick=9.5)
    pw, gap, rg, cw = 3.0, .45, .8, 4.3                                          # inches
    xcb = .3 + 2 * pw + gap + .3
    xcv = xcb + .16 + .75 + .55
    W = xcv + cw + .3
    H = 1.5 + pw + rg + pw + .55
    COL = ("#3f88c5", "#d7301f")
    HEAD, TTL = (TX["head_left"], TX["head_right"]), (TX["thr_left"], TX["thr_right"])
    y_top = H - 1.5 - pw
    y_bot = y_top - rg - pw
    with ph.style("default"):
        fig = plt.figure(figsize=(W, H), facecolor="white", dpi=100)
        art = []
        for w in range(2):
            x = .3 + w * (pw + gap)
            a1 = ph.ax_in(fig, x, y_top, pw, pw)
            a2 = ph.ax_in(fig, x, y_bot, pw, pw)
            for ax in (a1, a2):
                ax.add_collection(LineCollection(cells[edges].astype(float), colors="0.85", linewidths=.5,
                                                 linestyles=(0, (3, 2)), zorder=0))
                ax.set_xlim(-.9, 9.9), ax.set_ylim(-.9, 9.9), ax.set_aspect("equal")
                ax.set_xticks([]), ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color(COL[w]), sp.set_linewidth(1.4)
            lca = LineCollection(Pa[w, 0][edges], cmap=ph.FR_CMAP, norm=frn, linewidths=1.6, zorder=3)
            lca.set_array(ecol)
            a1.add_collection(lca)
            ttl = a1.set_title("", fontsize=FSZ["title"], color=COL[w], pad=6, linespacing=1.3)
            envs = []
            for e in range(nE):
                lc = LineCollection(P[w, 0, e][edges], cmap=ph.FR_CMAP, norm=dn, linewidths=.3, alpha=.65, zorder=2)
                lc.set_array(np.zeros(len(edges)))
                envs.append(a2.add_collection(lc))
            avg = a2.add_collection(LineCollection(Pa[w, 0][edges], colors="k", linewidths=1.2, zorder=4))
            art.append((w, lca, ttl, envs, avg))
        fig.text(.3 / W, (y_top + pw + .75) / H, TX["row_top"].format(n_env=nE),
                 fontsize=FSZ["row"], color="0.2", va="bottom")
        fig.text(.3 / W, (y_bot + pw + .12) / H, TX["row_bottom"].format(n_env=nE),
                 fontsize=FSZ["row"], color="0.2", va="bottom")
        for y, nm, lab, ext in ((y_top, frn, TX["cbar_top"], "both"), (y_bot, dn, TX["cbar_bottom"], "max")):
            cbx = ph.ax_in(fig, xcb, y, .16, pw)
            cb = fig.colorbar(plt.cm.ScalarMappable(cmap=ph.FR_CMAP, norm=nm), cax=cbx, extend=ext)
            cb.set_label(lab, fontsize=FSZ["lab"])
            cb.ax.tick_params(labelsize=FSZ["tick"])
            cb.outline.set_visible(False)
        ax = ph.ax_in(fig, xcv, y_bot + .5, cw, y_top + pw - .4 - y_bot - .5)   # the grid error
        ax.set_title(TX["curve_title"], fontsize=FSZ["row"], pad=8)
        for w in range(2):
            ax.plot(sw, d[w], c=COL[w], lw=1.3, alpha=.35)
        trace = [ax.plot([], [], c=COL[w], lw=2.4)[0] for w in range(2)]
        dots = [ax.plot([], [], "o", c=COL[w], ms=7)[0] for w in range(2)]
        ax.set_xlim(-1, 101), ax.set_ylim(0, d.max() * 1.06), ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel(TX["curve_x"], fontsize=FSZ["lab"], labelpad=2)
        ax.set_ylabel(TX["curve_y"], fontsize=FSZ["lab"])
        ax.tick_params(labelsize=FSZ["tick"])
        ax.spines[["top", "right"]].set_visible(False)
        clock = fig.text(.3 / W, (H - .35) / H, "", fontsize=FSZ["clock"], va="center")

        def update(x):                                    # x: position in sweep levels (0 .. 50), linear in between
            j = min(int(np.floor(x)), len(sw) - 2)
            f = x - j
            p_now = (1 - f) * sw[j] + f * sw[j + 1]
            k = int(np.searchsorted(sw, p_now, side="right"))
            for w, lca, ttl, envs, avg in art:
                Pe = (1 - f) * P[w, j] + f * P[w, j + 1]
                pa = Pe.mean(0)
                lca.set_segments(pa[edges])
                avg.set_segments(pa[edges])
                for e, lc in enumerate(envs):
                    dv = np.linalg.norm(Pe[e] - pa, axis=1)
                    lc.set_segments(Pe[e][edges])
                    lc.set_array((dv[edges[:, 0]] + dv[edges[:, 1]]) / 2)
                t_now = (1 - f) * th[w, j] + f * th[w, j + 1] if np.isfinite(th[w, j]) else th[w, j + 1]
                ttl.set_text(HEAD[w] + "\n" + (TTL[w].format(thr=t_now) if p_now > 0 else TX["naive"]))
                dd = (1 - f) * d[w, j] + f * d[w, j + 1]
                trace[w].set_data(np.r_[sw[:k], p_now], np.r_[d[w, :k], dd])
                dots[w].set_data([p_now], [dd])
            clock.set_text(TX["clock"].format(p=p_now, n_env=nE))

        xs = np.linspace(0, len(sw) - 1, 4 * (len(sw) - 1) + 1)
        sched = [(xs[0], 1200)] + [(x, 70) for x in xs[1:-1]] + [(xs[-1], 2000)]
        protos = []
        for x in (0., 6., 20., len(sw) - 1.):
            update(x)
            protos.append(ph.grab(fig))
        exact = [(1., 1., 1.), (0., 0., 0.)] + [to_rgb(c) for c in COL] + [(v, v, v) for v in np.linspace(0, 1, 24)]
        ramp = [np.asarray(ph.FR_CMAP(v)[:3]) for v in np.linspace(0, 1, 32)]
        exact += [tuple(c) for c in ramp] + [tuple(.65 * c + .35) for c in ramp]
        pal = ph.exact_palette(protos, exact)[0]
        frames = []
        for i, (x, _) in enumerate(sched):
            update(x)
            frames.append(ph.quantize(ph.grab(fig), pal))
            if i % 50 == 0:
                print(f"frame {i + 1} of {len(sched)}", flush=True)
        plt.close(fig)
    path = ph.save_gif(frames, [ms for _, ms in sched], figure_path(name, out_dir))
    print(f"saved {path}: {len(frames)} frames, {frames[0].size[0]} x {frames[0].size[1]} px, "
          f"{os.path.getsize(path) / 1e6:.1f} MB", flush=True)
    return [path]


# ═══════════════════════════════════════════════════════════════════
# Figure: pairs that split, and their environments (fig1c_movie_split_traj.gif)
# ═══════════════════════════════════════════════════════════════════

SPLIT_DPI, SPLIT_FPS = 110, 25
SPLIT_PW, SPLIT_XG, SPLIT_X0 = 2.7, .3, .8                                       # panel side, column gap, left edge
SPLIT_Y_CLOCK, SPLIT_Y_NAME, SPLIT_GAP12, SPLIT_DH, SPLIT_GAP23 = .8, .66, .55, 1.62, .62   # bands, gaps (inches)
SPLIT_KX, SPLIT_KW = SPLIT_X0 + 4 * SPLIT_PW + 3 * SPLIT_XG + .45, 2.75        # the keys' column
SPLIT_W = SPLIT_KX + SPLIT_KW + .15
SPLIT_H = SPLIT_Y_CLOCK + SPLIT_Y_NAME + SPLIT_PW + SPLIT_GAP12 + SPLIT_DH + SPLIT_GAP23 + SPLIT_PW + .3
SPLIT_JUMP, SPLIT_JUMP_DASH = 2., (0, (3.2, 2.2))                                # a jump: one step > 2 cells, dashed
SPLIT_LW_PATH = 1.4                                                              # row 1: the paths
SPLIT_LW_S, SPLIT_YS = 1.8, 7.2                                                  # row 2: lines, y range +-YS
SPLIT_PRE_ALPHA = .3                                                             # rows 1-2: before the split
SPLIT_SUB = 8                                                                    # row 3: path points per step
SPLIT_FS = dict(t=15, name=13.5, stat=10, row=11.5, lab=11.5, key=10.5, tick=9.5)
SPLIT_TRAJ = dict(lw=.3, alpha=.5, ms=2.4, home_ms=4.4, home_mew=.6, aa=True)  # row 3: thin anti-aliased lines
SPLIT_MARKERS = ("o", "^", "D")                                                  # rows 1-2: the pairs' shapes


def figure_split(out_dir=None):
    """Pairs of streams that stay together and then split, and the trajectories of their environments, as one movie
    -> fig1c_movie_split_traj.gif in out_dir (default main/REPLICATION/figure_bin; 1743 x 1094 px at 110 dpi, 398
    frames of 40 ms at 25 per second, 317 once Pillow merges the identical ones, 15.9 s) -> [its path].

    Data: the alias scan (39 environments, e = 2 left out; 100 cells x 50 streams x 30 steps; all four agents on the
    same streams). A stream's position is its posterior means x = (E[r_g], E[r_c]) (cells). The pairs come from
    split_pair_data, selected in the Optimal: two streams of different cells in one environment, true cells at least 4
    apart, within 0.6 of each other from step 2 through t_s (within 0.4 at t_s; t_s = 5..22) and at least 3 further
    apart two steps later, each pair once at its last t_s (743 pairs in 39 environments); shown are three, in
    distinct environments and cells, the largest jump in each window of t_s (11-22, 8-10, 6-7): scan environments 15,
    39 and 20, t_s = 6, 8, 12.
    Columns: Optimal Bayes, Trained RNN, Naive Bayes, ESN; a fifth column holds the keys. Over each column, the share of
    the 743 pairs in which both streams end within 1 cell of their true cells at T.
        row 1  the three pairs: each stream's path of posterior means, faint before its pair's split step, coloured by
               its true cell (the bivariate key of row 3); each pair its own marker shape; stream i filled, stream j
               outlined; small: the streams now, large: their true cells; a step that moves a stream more than 2
               cells is dashed (a jump: the posterior mean crosses between two modes, the belief never travels the
               path; a stream can jump at several steps)
        row 2  each stream's signed position in its pair (cells): (x - xbar) . e, xbar the pair's centroid (the mean
               of its two streams now) and e the unit vector from j's true cell to i's: 0 while the two are together,
               + (i) and - (j) once they split, ending at +-d/2 (half the cells' distance) when both reach their
               cells; the two lines mirror each other by construction. (Measured from the cells' fixed midpoint
               instead, these pairs sit together at j's cell before the split and i jumps across.) One solid line per
               stream in its colour (faint before the split, as row 1); the streams now: the pair's marker, i filled,
               j outlined (as row 1); black on the axis: the split step
        row 3  the trajectories of the same three environments: one stream per cell from each (the first of the
               cell's 50; 300 particles, one fixed random drawing order), each coloured by its true cell, in the
               agent's fixed T = 30 PC plane (belief_geometry); circles: the cells' targets (targets)
    Timing: a hold of 0.8 s at t = 1, then 0.6 s per step around each of the three splits (the six steps from
    t = t_s - 2 to t_s + 4), 0.3 s per step elsewhere, a hold of 2.5 s at T; the paths move linearly between steps in
    rows 1-2, by Catmull-Rom splines in row 3.
    Data: split_pairs.npz of the data layer (P, the shown streams' posterior means (4, 3, 2, 30, 2); shown_env,
    shown_ts, c_i, c_j, err), built from the scan by build_split_pairs when it is missing (about 30 s); the scan's
    float16 beliefs and its cells, cell_of; beliefs.npz, the record and episodes.npz (belief_geometry's planes,
    targets).
    Style: plotting_helpers' "fig" (cmr10, math in Computer Modern) at 110 dpi, whatever ran before. GIF palette:
    white, black, 24 greys, the agents' colours and tints, the faint paths' colours and the 100 cells' colours pinned,
    the rest a median cut of 4 frames; every pixel to its nearest palette colour exactly (white stays white).
    Threads: belief_geometry's planes change in their last bits (up to 1e-13) with MKL's thread count, and so do a
    few of row 3's anti-aliased pixels; the kept file was made at 3 threads, so the planes are fitted at FIT_THREADS =
    3 whatever the environment (fit_threads; fitted at 2, one pixel differs in each of 3 of its 317 frames). About
    2 min on the CPU (one core), 2.1 GB.
    Run from the repository root with the eureka environment's python:
        python main/REPLICATION/plot_fig1c_all.py split [--out DIR]"""
    name = "fig1c_movie_split_traj.gif"
    T, W, H, FS = N_T, SPLIT_W, SPLIT_H, SPLIT_FS
    PW, XG, X0, DH, YS = SPLIT_PW, SPLIT_XG, SPLIT_X0, SPLIT_DH, SPLIT_YS
    with load_cache("split_pairs") as Z:
        P, sts, envs = Z["P"], Z["shown_ts"], [int(e) for e in Z["shown_env"]]
        c_i, c_j, err = Z["c_i"], Z["c_j"], Z["err"]
    n = len(sts)
    pmk = SPLIT_MARKERS[:n]
    tcol = [[np.asarray(ph.bivariate(int(round(c[0])), int(round(c[1]))), np.float64)[:3] for c in cs]
            for cs in (c_i, c_j)]                                                # each stream's colour: its true cell
    half = np.linalg.norm(c_i - c_j, axis=-1) / 2                                # row 2: each pair's half distance
    e_ax = (c_i - c_j) / (2 * half[:, None])                                     # and axis (j's cell -> i's)
    both = (err <= 1).all(-1).mean(1)
    TD = _split_traj_data(envs)
    taus = ph.split_schedule(sts, T=T, fps=SPLIT_FPS)
    with ph.style("fig"):
        fig = plt.figure(figsize=(W, H), dpi=SPLIT_DPI, facecolor="white")
        y1 = H - SPLIT_Y_CLOCK - SPLIT_Y_NAME - PW
        y2 = y1 - SPLIT_GAP12 - DH
        y3 = y2 - SPLIT_GAP23 - PW
        g = np.arange(N_R)
        lim_taus = taus[::10] + [T - 1.]
        cols = []
        for c_, a in enumerate(AGENTS4):                                         # the columns: split_pairs' agents
            x = X0 + c_ * (PW + XG)
            fig.text((x + PW / 2) / W, (y1 + PW + .36) / H, ph.AGENT_NAME_BAYES[a], ha="center", va="baseline",
                     fontsize=FS["name"], color=ph.AGENT_COLOR[a])
            fig.text((x + PW / 2) / W, (y1 + PW + .12) / H, "both streams end at their cells: %d%%" % round(
                100 * both[c_]), ha="center", va="baseline", fontsize=FS["stat"], color="0.35")
            # row 1: the pairs
            a1 = ph.ax_in(fig, x, y1, PW, PW)
            a1.scatter(*np.meshgrid(g, g), s=2, c="0.82", lw=0, zorder=0)
            a1.set_xlim(-.6, 9.6), a1.set_ylim(-.6, 9.6), a1.set_aspect("equal")
            a1.set_xticks([0, 3, 6, 9]), a1.set_yticks([0, 3, 6, 9])
            a1.tick_params(labelsize=FS["tick"])
            a1.set_xlabel(r"$E[r_g]$", fontsize=FS["lab"], labelpad=1)
            if c_ == 0:
                a1.set_ylabel(r"$E[r_c]$", fontsize=FS["lab"], labelpad=1)
            else:
                a1.set_yticklabels([])
            lines = []
            for p in range(n):
                ci, cj = tcol[0][p], tcol[1][p]
                a1.plot(*c_i[p], pmk[p], ms=11.5, color=ci, mec="0.2", mew=.6, zorder=2)        # true cells: large,
                a1.plot(*c_j[p], pmk[p], ms=11.5, mfc="white", mec=cj, mew=2.2, zorder=2)       # i filled, j outlined
                seg = [a1.add_collection(LineCollection([], linewidths=SPLIT_LW_PATH, capstyle="round",
                                                        zorder=3 + .5 * k)) for k in (0, 1)]    # each stream's path
                hi, = a1.plot([], [], pmk[p], ms=6.5, color=ci, mec="0.15", mew=.6, zorder=5)   # the streams now:
                hj, = a1.plot([], [], pmk[p], ms=6.5, mfc="white", mec=cj, mew=1.8, zorder=5)   # small
                lines.append((seg, hi, hj, (ci, cj)))
            # row 2: each stream's signed position in its pair
            a2 = ph.ax_in(fig, x, y2, PW, DH)
            a2.axhline(0, c="0.6", lw=.8, ls=":", zorder=0)
            dl, dh = [], []
            for p in range(n):
                ci, cj = tcol[0][p], tcol[1][p]
                a2.plot([sts[p] + 1], [-YS], pmk[p], ms=6.5, color="k", clip_on=False, zorder=6)  # the split step
                ls_ = []
                for al in (SPLIT_PRE_ALPHA, 1.):                                                # before / from t_s
                    for k in (0, 1):                                                            # (opaque tints)
                        ls_.append(a2.plot([], [], c=al * tcol[k][p] + 1 - al, lw=SPLIT_LW_S, solid_capstyle="round",
                                           zorder=3 + .1 * p + .05 * k)[0])
                dl.append(ls_)                                                                  # pre i, j; post i, j
                dh.append((a2.plot([], [], pmk[p], ms=6.5, color=ci, mec="0.15", mew=.6, zorder=7)[0],
                           a2.plot([], [], pmk[p], ms=6.5, mfc="white", mec=cj, mew=1.8, zorder=7)[0]))
            cur = a2.axvline(1, c="0.7", lw=.7, zorder=0)
            a2.set_xlim(.5, T + .5), a2.set_ylim(-YS, YS)
            a2.set_yticks([-5, 0, 5]), a2.set_xticks([1, 10, 20, 30])
            a2.tick_params(labelsize=FS["tick"])
            a2.spines[["top", "right"]].set_visible(False)
            a2.set_xlabel("step $t$", fontsize=FS["lab"], labelpad=1)
            if c_ > 0:
                a2.set_yticklabels([])
            # row 3: the environments' trajectories
            Dt = TD[a]
            a3 = ph.ax_in(fig, x, y3, PW, PW)
            xl, yl = _split_traj_limits(Dt, lim_taus)
            a3.set_xlim(*xl), a3.set_ylim(*yl), a3.set_aspect("equal", adjustable="box"), a3.axis("off")
            tr, hd, hm = _split_traj_frame(Dt, 0.)
            lc = a3.add_collection(LineCollection(list(tr), colors=Dt["rgba"], linewidths=SPLIT_TRAJ["lw"], zorder=1,
                                                  capstyle="round", joinstyle="round", antialiased=SPLIT_TRAJ["aa"]))
            ht = a3.scatter(*hm.T, s=SPLIT_TRAJ["home_ms"] ** 2, facecolors="none", edgecolors=Dt["home_col"],
                            linewidths=SPLIT_TRAJ["home_mew"], zorder=2)
            pt = a3.scatter(*hd.T, s=SPLIT_TRAJ["ms"] ** 2, c=Dt["col"], lw=0, zorder=3, antialiased=False)
            cols.append(dict(c_=c_, lines=lines, dl=dl, dh=dh, cur=cur, Dt=Dt, lc=lc, ht=ht, pt=pt))
        for y, h, txt in ((y1, PW, "pairs that split"), (y2, DH, "position in the pair"),
                          (y3, PW, "their environments")):
            fig.text(.2 / W, (y + h / 2) / H, txt, rotation=90, ha="center", va="center", fontsize=FS["row"],
                     color="0.3")
        set_clock = ph.clock_bar(fig, X0, H - .42, X0 + 1.15, X0 + 4 * PW + 3 * XG, T=T, h=.12, dy=.3,
                                 fs_t=FS["t"], fs_tick=FS["tick"])
        _split_keys(fig, pmk, y1, y2, y3, envs)

        def update(tau):
            s = int(np.floor(tau))
            for C in cols:
                a_ = C["c_"]
                for p, (seg, hi, hj, cc) in enumerate(C["lines"]):
                    path, head = _split_at(P[a_, p], tau)                          # (2, s + 2, 2): steps 0..s, tau
                    k0 = sts[p] - 1                                                # index of step t_s
                    for k, lc_ in enumerate(seg):
                        _split_style(lc_, path[k], P[a_, p, k], k0, cc[k])
                    hi.set_data([head[0, 0]], [head[0, 1]]), hj.set_data([head[1, 0]], [head[1, 1]])
                    sv = (path - path.mean(0)) @ e_ax[p]                           # (2, s + 2): steps 0..s, tau
                    xs_ = np.r_[np.arange(1, s + 2), tau + 1]
                    m_ = len(xs_)
                    for h_, sl in enumerate((slice(0, min(m_, k0 + 1)), slice(min(k0, m_), m_))):  # before / from t_s
                        for k in (0, 1):
                            C["dl"][p][2 * h_ + k].set_data(xs_[sl], sv[k, sl])
                    for k in (0, 1):
                        C["dh"][p][k].set_data([xs_[-1]], [sv[k, -1]])
                C["cur"].set_xdata([tau + 1])
                tr_, hd_, hm_ = _split_traj_frame(C["Dt"], tau)
                C["lc"].set_segments(list(tr_))
                C["ht"].set_offsets(hm_)
                C["pt"].set_offsets(hd_)
            set_clock(tau, s + 1)

        protos = []
        for tau in (0., 6., 12., T - 1.):
            update(tau)
            protos.append(ph.grab(fig))
        cells = np.array([(rg, rc) for rg in range(N_R) for rc in range(N_R)])
        keys = [(1., 1., 1.), (0., 0., 0.)] + [(v, v, v) for v in np.linspace(0, 1, 24)]
        keys += [to_rgb(c) for c in ph.AGENT_COLOR.values()]
        keys += [tuple(al * np.array(to_rgb(ph.AGENT_COLOR[k])) + 1 - al) for k in ("Trained", "Echo")
                 for al in (.2, .4, .6, .8)]
        keys += [tuple(SPLIT_PRE_ALPHA * c + 1 - SPLIT_PRE_ALPHA) for cs in tcol for c in cs]   # the faint paths
        keys += [tuple(np.asarray(c, np.float64)[:3]) for c in ph.bivariate(cells[:, 0], cells[:, 1])]
        _, pal_cols, n_exact = ph.exact_palette(protos, keys, n=255)
        quant = ph.ExactQuantizer(pal_cols)
        print(f"palette: {n_exact} exact colours + {len(pal_cols) - n_exact} from the frames", flush=True)
        frames = []
        for i, tau in enumerate(taus):
            update(tau)
            frames.append(quant(ph.grab(fig)))
            if i % 100 == 0:
                print(f"frame {i + 1} of {len(taus)}", flush=True)
        plt.close(fig)
    path = ph.save_gif(frames, 1000 // SPLIT_FPS, figure_path(name, out_dir))
    print(f"saved {path}: {len(frames)} frames, {len(frames) / SPLIT_FPS:.1f} s, {frames[0].size[0]} x "
          f"{frames[0].size[1]} px, {os.path.getsize(path) / 1e6:.1f} MB", flush=True)
    return [path]


def _split_traj_data(envs):
    """row 3, per agent in its fixed T = 30 plane (belief_geometry): one stream per cell (the cell's first of SCAN_M) of
    each scan environment in envs, in one fixed random order (default_rng(0)) -> {agent: dict(V2, the positions at
    every step (n, T, 2); tt, the path's step indices, SPLIT_SUB per step; P2, the Catmull-Rom paths at tt
    (n, len(tt), 2); col, rgba, the true cells' colours (alpha SPLIT_TRAJ["alpha"]); home2, the 100 cells' targets
    (100, 2); home_col, their colours)}"""
    G = belief_geometry()
    cells_all = np.asarray(scan_meta("cells"))
    cell_of = np.asarray(scan_meta("cell_of"))
    assert (cell_of == np.repeat(np.arange(len(cells_all)), SCAN_M)).all()
    js = np.arange(len(cells_all)) * SCAN_M                                      # each cell's first stream
    ev, js = np.repeat(np.asarray(envs, int), len(js)), np.tile(js, len(envs))
    p = np.random.default_rng(0).permutation(len(js))
    ev, js = ev[p], js[p]
    cell = cells_all[cell_of[js]]
    home, _ = targets([tuple(int(v) for v in c) for c in cells_all])
    tt = np.linspace(0, N_T - 1, (N_T - 1) * SPLIT_SUB + 1)
    col = np.asarray(ph.bivariate(cell[:, 0], cell[:, 1]))
    out = {}
    for a in AGENTS4:
        D30 = G[(a, N_T)]
        V = scan_beliefs(a)[ev, js].astype(np.float64)
        out[a] = dict(V2=project_plane(D30, V), tt=tt,
                      P2=project_plane(D30, np.stack([ph.blend(V, tau, axis=1) for tau in tt], 1)),
                      col=col, rgba=np.concatenate([col, np.full((len(col), 1), SPLIT_TRAJ["alpha"])], 1),
                      home2=project_plane(D30, np.asarray(home)),
                      home_col=np.asarray(ph.bivariate(cells_all[:, 0], cells_all[:, 1])))
    return out


def _split_traj_frame(D, tau):
    """row 3 at step index tau: every particle's trajectory so far (its path's points up to tau, then its position at
    tau), its position, the targets; all in the fixed plane"""
    k = int(np.searchsorted(D["tt"], tau, side="right"))
    head = ph.blend(D["V2"], tau, axis=1)                                        # (Catmull-Rom is linear: blending
    trail = np.concatenate([D["P2"][:, :k], head[:, None]], 1)                   # the projections = projecting)
    return trail, head, D["home2"]


def _split_traj_limits(D, taus):
    """row 3's square data box: everything drawn at the step indices taus, 4% margin"""
    ext = []
    for tau in taus:
        tr, _, hm = _split_traj_frame(D, tau)
        pts = np.concatenate([tr.reshape(-1, 2), hm])
        ext += [pts.min(0), pts.max(0)]
    return ph.square_limits(np.array(ext), 1.04)


def _split_at(Y, tau):
    """positions Y (..., T, 2) up to step index tau: the steps 0 .. floor(tau) and the point at tau (linear between
    steps) -> (..., floor(tau) + 2, 2), the point at tau"""
    s = int(np.floor(tau))
    f = tau - s
    head = Y[..., s, :] if f == 0 else (1 - f) * Y[..., s, :] + f * Y[..., s + 1, :]
    return np.concatenate([Y[..., :s + 1, :], head[..., None, :]], -2), head


def _split_style(lc, path, steps, k0, col):
    """one stream's path up to the frame (path: its points at steps 0 .. s and at tau; steps: all its T points) as
    segments, each styled by its step: faint before the split (ending at or before step index k0), dashed if the
    full step moves the stream more than SPLIT_JUMP cells"""
    segs = np.stack([path[:-1], path[1:]], 1)
    k = np.arange(len(segs))
    disp = np.r_[np.linalg.norm(np.diff(steps, axis=0), axis=-1), 0.]          # (the point at tau after the last
    jump = disp[np.minimum(k, len(disp) - 1)] > SPLIT_JUMP                       # step repeats it: no move)
    rgba = np.tile(np.r_[col, 1.], (len(segs), 1))
    rgba[k < k0, 3] = SPLIT_PRE_ALPHA
    lc.set_segments(segs)
    lc.set_color(rgba)
    lc.set_linestyle([SPLIT_JUMP_DASH if j_ else "-" for j_ in jump])


def _split_keys(fig, pmk, y1, y2, y3, envs):
    """the fifth column, beside each row: rows 1-2, the pairs' shapes (their environments), the two streams (fill: i,
    outline: j), now and true cell (small, large), before the split, a jump; row 2, zero and the split step; row 3,
    the colours (true cells, rows 1 and 3) and the target"""
    W, H, FS, PW, x = SPLIT_W, SPLIT_H, SPLIT_FS, SPLIT_PW, SPLIT_KX

    def entry(y, lab, line=None, marker=None):
        if line is not None:
            fig.add_artist(Line2D([(x + .02) / W, (x + .4) / W], [y / H] * 2, transform=fig.transFigure, **line))
        if marker is not None:
            fig.add_artist(Line2D([(x + (.3 if line is not None else .21)) / W], [y / H], transform=fig.transFigure,
                                  ls="none", **marker))
        fig.text((x + .5) / W, y / H, lab, ha="left", va="center", fontsize=FS["key"], color="0.2")

    fill, edge = "0.72", "0.25"                                                  # the key's stand-ins for i's, j's
    y = y1 + PW - .05                                                            # beside row 1
    fig.text(x / W, y / H, "Rows 1-2: three pairs", ha="left", va="top", fontsize=FS["lab"], color="0.15")
    y -= .45
    for p, mk in enumerate(pmk):
        entry(y, "environment %d" % envs[p], marker=dict(marker=mk, ms=7, color="0.55", mec="0.15", mew=.6))
        y -= .3
    y -= .08
    entry(y, "stream $i$ (fill)", line=dict(c=fill, lw=SPLIT_LW_PATH),
          marker=dict(marker="o", ms=6.5, color=fill, mec="0.15", mew=.6))
    y -= .3
    entry(y, "stream $j$ (outline)", line=dict(c=edge, lw=SPLIT_LW_PATH),
          marker=dict(marker="o", ms=6.5, mfc="white", mec=edge, mew=1.8))
    y -= .3
    for xm, ms in ((.1, 6.5), (.32, 11.5)):                                      # small: now; large: true cell
        fig.add_artist(Line2D([(x + xm) / W], [y / H], transform=fig.transFigure, ls="none", marker="o", ms=ms,
                              mfc="white", mec="0.45", mew=1.3))
    fig.text((x + .5) / W, y / H, "small: now, large: true cell", ha="left", va="center", fontsize=FS["key"],
             color="0.2")
    y -= .3
    entry(y, "before the split", line=dict(c="0.3", lw=SPLIT_LW_PATH, alpha=SPLIT_PRE_ALPHA))
    y -= .3
    entry(y, "jump (one step $>$ %g cells)" % SPLIT_JUMP, line=dict(c="0.3", lw=SPLIT_LW_PATH, ls=SPLIT_JUMP_DASH))
    y = y2 + SPLIT_DH / 2 + .15                                                  # beside row 2 (its middle)
    entry(y, "0: together; $i$ $+$, $j$ $-$", line=dict(c="0.6", lw=.9, ls=":"))
    y -= .3
    entry(y, "split step", marker=dict(marker="o", ms=6.5, color="k"))
    top = y3 + PW + .3                                                           # beside row 3
    fig.text(x / W, top / H, "Row 3: one stream per cell", ha="left", va="top", fontsize=FS["lab"], color="0.15")
    ks = 1.55
    ax = ph.ax_in(fig, x + .3, top - .68 - ks, ks, ks)
    ph.latent_key(ax, R=N_R)
    ax.tick_params(length=2, pad=1, labelsize=FS["tick"])
    ax.set_xlabel("$r_g$", labelpad=-5, fontsize=FS["lab"])
    ax.set_ylabel("$r_c$", labelpad=-5, fontsize=FS["lab"])
    ax.set_title("true cell colour (rows 1, 3)", fontsize=FS["key"], color="0.2", pad=3)
    entry(top - .68 - ks - .45, "target", marker=dict(marker="o", ms=6, mfc="none", mec="0.35", mew=1))


# ═══════════════════════════════════════════════════════════════════
# Seeds: every random draw behind the figures, and the environments it produced
# ═══════════════════════════════════════════════════════════════════

def seed_manifest():
    """every random draw behind the figures (what, generator, seed) and, in full, the environments behind the
    environment-specific figures: each as its harvest episode and its goal and contextual variables (indices among the
    500), with the seed of its observation streams -> a JSON-ready dict"""
    rec = record()
    ci, gi, cv = (rec[k].astype(int) for k in ("ep_ctx_inds", "ep_goal_ind", "ep_ctx_vals"))

    def env(ep):
        g = gi[ep]
        return {"episode": int(ep), "goal_variable": int(ci[ep, g]), "contextual_variable": int(ci[ep, 1 - g])}

    def cell(s):                                                                  # a scan stream's goal-ordered cell
        return [int(s) // SCAN_M // N_R, int(s) // SCAN_M % N_R]

    with load_cache("scan_envs") as z:
        scan_eps = z["eps"].astype(int)
    scan = [dict(scan_env=i, **env(ep), in_scan=i < SCAN_N_ENV, left_out=i == SCAN_LEFT_OUT,
                 stream_seed=SCAN_SEED0 + i if i < SCAN_N_ENV else None) for i, ep in enumerate(scan_eps)]
    with load_cache("split_pairs") as z:
        split = [dict(scan_env=int(e), **env(scan_eps[e]), stream_seed=SCAN_SEED0 + int(e), split_step=int(t),
                      stream_i=int(i), cell_i=cell(i), stream_j=int(j), cell_j=cell(j))
                 for e, i, j, t in zip(z["shown_env"], z["shown_i"], z["shown_j"], z["shown_ts"])]
    sweep_eps, sweep_seeds = sweep_environments()
    traj = {}
    for group in ("low", "high"):
        eps, n_pairs, cut = fr_group_episodes(group)
        traj[group] = {"fr_cut": cut, "pairs_available": int(n_pairs),
                       "episodes": [dict(**env(ep), true_cell=[int(cv[ep, gi[ep]]), int(cv[ep, 1 - gi[ep]])])
                                    for ep in eps]}
    U = "numpy.random.default_rng"
    draws = [
        {"what": "the harvest's 50,000 test episodes: active variables, goal, true values, observations (record.npz)",
         "generator": "numpy's global generator inside CognitiveGridworld, never seeded", "seed": None,
         "note": "the only unseeded draw: kept as record.npz, the primary; it cannot be redrawn"},
        {"what": "scan_envs: 80 environments, the first harvest episode of each distinct (likelihood table, goal) "
                 "among the first 20,000, drawn without replacement, sorted", "generator": U, "seed": 42},
        {"what": "scan: environment e's 100 cells x 50 streams x 30 steps x 5 observations, o = (U < p) row by row "
                 "(stream s: cell s // 50, goal-ordered k = 10 r_g + r_c)", "generator": U,
         "seed": "SCAN_SEED0 + e = 1000 + e (e = 0..39)"},
        {"what": "scan: the perturbed tables Lpert (stored, read by no figure)", "generator": U, "seed": "100 + e"},
        {"what": "alias_traj: harvest episode ENV = 1434, its four CELLS x 2000 streams", "generator": U, "seed": 1},
        {"what": "distances_pairs: 110 environments (no draw: evenly spaced over the distinct variable pairs) x 100 "
                 "cells x 10 streams", "generator": U, "seed": "7,000,000 + episode"},
        {"what": "sweep100: scan_envs' 80 (stream seeds 1000 + index), then the harvest's episodes in a random order "
                 "for new variable pairs (stream seeds 1080 + j) up to 100", "generator": U,
         "seed": "2026 (the order); the stream seeds as listed"},
        {"what": "traj_fr: per FR group, 50 variable pairs without replacement among the eligible ones, then one "
                 "episode of each, uniformly", "generator": U, "seed": FR_GROUP_SEED},
        {"what": "split movie, row 3: the drawing order of the 300 trajectories", "generator": U, "seed": 0},
        {"what": "split pairs", "generator": None, "seed": None,
         "note": "no draw: selected deterministically from the scan's Optimal beliefs (split_pair_data)"},
        {"what": "the PCA planes (belief_geometry, step_planes)", "generator": None, "seed": None,
         "note": "deterministic: scikit-learn's covariance_eigh solver for these shapes"}]
    return {"about": "fig1c: every random draw and the environments it produced (plot_fig1c_all.py --seeds)",
            "draws": draws,
            "environments": {
                "scan": scan,
                "split_shown": split,
                "alias_traj": dict(**env(ENV), cells=[list(c) for c in CELLS], streams_per_cell=ALIAS_M,
                                   seed=ALIAS_SEED),
                "traj_fr": traj,
                "sweep100": [dict(**env(ep), stream_seed=int(sd)) for ep, sd in zip(sweep_eps, sweep_seeds)],
                "distances_pairs": [dict(**env(ep), stream_seed=PAIR_SEED0 + int(ep))
                                    for ep in pair_environments()]}}


def write_seeds(path=None):
    """seed_manifest() as JSON at path (default: seeds.json in the data folder) -> path"""
    import json
    path = path or data_path("seeds.json")
    man = seed_manifest()
    with open(path, "w") as f:
        json.dump(man, f, indent=1)
    E = man["environments"]
    print(f"saved {path}: {len(man['draws'])} draws; environments: scan {len(E['scan'])}, split shown "
          f"{len(E['split_shown'])}, traj_fr {sum(len(v['episodes']) for v in E['traj_fr'].values())}, sweep100 "
          f"{len(E['sweep100'])}, distances_pairs {len(E['distances_pairs'])}", flush=True)
    return path


# ═══════════════════════════════════════════════════════════════════
# Command line
# ═══════════════════════════════════════════════════════════════════

TARGETS = {                                         # name: (outputs, the function that makes them, the caches it reads)
    "summary": (SUMMARY_FILES, figure_summary, ("distances_pairs", "distances_fr", "updates")),
    "traj_fr": (("fig1c_traj_FR_groups_T30.png",), figure_traj_fr, ("record", "episodes", "beliefs_steps")),
    "pcs": (("fig1c_movie_pcs.gif",), figure_pcs, ("record", "episodes", "beliefs", "beliefs_steps", "alias_traj")),
    "envdist": (("fig1c_movie_fixed_envdist.gif",), figure_envdist,
                ("record", "episodes", "beliefs", "beliefs_steps", "alias_traj", "scan")),
    "sweep": (("fig1c_movie_hybrid_sweep_combined.gif",), figure_sweep, ("sweep100",)),
    "split": (("fig1c_movie_split_traj.gif",), figure_split, ("record", "episodes", "beliefs", "scan", "split_pairs")),
}
GROUPS = {"stills": ("summary", "traj_fr"), "movies": ("pcs", "envdist", "sweep", "split")}
GROUPS["all"] = GROUPS["stills"] + GROUPS["movies"]


def list_status(out_dir=None):
    """print the targets (their outputs in out_dir, else figure_bin: present or missing; the caches they read) and the
    caches (present with their size, or missing; builder and inputs)"""
    folder = out_dir or FIG_DIR
    print(f"targets (outputs in {folder}):")
    for name, (outs, fn, reads) in TARGETS.items():
        print(f"    {name:8s} {fn.__name__:16s} reads {', '.join(reads)}")
        for f in outs:
            print(f"             {f:40s} {'present' if os.path.exists(os.path.join(folder, f)) else 'missing'}")
    print("groups: " + "; ".join(f"{g} = {' '.join(v)}" for g, v in GROUPS.items()))
    print(f"caches (in {DATA_DIR}):")
    for name, (files, builder, deps) in CACHES.items():
        paths = [data_path(f) for f in files]
        have = [p for p in paths if os.path.exists(p)]
        status = ("present %9.1f MB" % (sum(os.path.getsize(p) for p in have) / 1e6) if len(have) == len(paths)
                  else f"missing {len(paths) - len(have)} of {len(paths)} file(s)")
        shown = ", ".join(files) if len(files) < 3 else f"{files[0]} and {len(files) - 1} more"
        print(f"    {name:16s} {status:22s} {shown:34s} <- {builder.__name__}"
              f"({', '.join(deps) if deps else 'the gradflow harvests'})")


def main(argv=None):
    """the command line (see the module docstring) -> exit status, 0 when every output was written"""
    import argparse
    p = argparse.ArgumentParser(prog="plot_fig1c_all.py", description=__doc__.split("\n\n")[0],
                                epilog="targets: " + ", ".join(TARGETS) + "; groups: " + ", ".join(GROUPS)
                                       + ". caches: " + ", ".join(CACHES) + ".")
    p.add_argument("target", nargs="*", help="targets or groups (default: all)")
    p.add_argument("--out", metavar="DIR", help="write the figures into DIR instead of main/REPLICATION/figure_bin")
    p.add_argument("--data", metavar="DIR",
                   help="read and build the caches in DIR instead of main/DATA/experimental/fig1c (a regeneration "
                        "from the primary: DIR holding record.npz alone)")
    p.add_argument("--rebuild", nargs="+", default=[], metavar="cache",
                   help="rebuild these caches before making the targets (their dependents only if named too)")
    p.add_argument("--rebuild-all", action="store_true",
                   help="rebuild every cache but the record first (about 40 min; overwrites the seeded files)")
    p.add_argument("--list", action="store_true", help="print the targets and the caches with their status, then exit")
    p.add_argument("--seeds", nargs="?", const="", metavar="PATH",
                   help="write every random draw's seed and the environments it produced as JSON (default: seeds.json "
                        "in the data folder), then exit")
    a = p.parse_args(argv)
    if a.data:
        global DATA_DIR
        DATA_DIR = os.path.abspath(a.data)
    words = list(a.target) + [w for w in a.rebuild if w not in CACHES]          # (targets may follow --rebuild)
    bad = [w for w in words if w not in TARGETS and w not in GROUPS]
    if bad:
        p.error(f"unknown target(s) {bad} (caches are named only after --rebuild)")
    named = [w for w in a.rebuild if w in CACHES]
    if a.rebuild and not named:
        p.error("--rebuild names no cache (see --list); to rebuild them all, use --rebuild-all")
    out = os.path.abspath(a.out) if a.out else None
    if a.list:
        list_status(out)
        return 0
    if a.seeds is not None:
        write_seeds(os.path.abspath(a.seeds) if a.seeds else None)
        return 0
    if named or a.rebuild_all:
        todo = [c for c in CACHES if c != "record"] if a.rebuild_all else [c for c in CACHES if c in named]
        print(f"rebuilding {', '.join(todo)} in {DATA_DIR}", flush=True)
        for name in todo:                                                         # in dependency order
            ensure(name, rebuild=True)
    names = list(dict.fromkeys(t for w in (words or ["all"]) for t in GROUPS.get(w, (w,))))
    print(f"making {', '.join(names)} -> {out or FIG_DIR}", flush=True)
    report, failed = [], False
    for name in names:
        outs, fn, _ = TARGETS[name]
        print(f"\n=== {name}", flush=True)
        t0 = time.time()
        try:
            paths = list(fn(out))
        except Exception:
            traceback.print_exc()
            paths = []
        ok = len(paths) == len(outs) and all(os.path.exists(q) and os.path.getmtime(q) >= t0 - 1 for q in paths)
        failed |= not ok
        report.append(f"{name:8s} {(time.time() - t0) / 60:5.1f} min  " + (", ".join(map(os.path.basename, paths))
                                                                          if ok else "FAILED: " + ", ".join(outs)))
    print("\n" + "\n".join(report), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
