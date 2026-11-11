"""Which micro_batch is fastest for a wide reservoir on the GPU it is about to use (John 2026-11-05: "have them first check if a
larger microbatch size would speed things up"). Slices change memory and speed, not the update: exact on the joint loss, near-exact
on the goal loss (model_forward.forward_pass_micro). `bench_micro.py <net> [sizes, default 1000,2000] [episodes, default 16]`
with CUDA_VISIBLE_DEVICES set to one GPU: each size runs in its own process for a few real episodes, in a temporary folder that is
removed, and the median training step is compared. A size that runs out of memory or peaks above 88% of the GPU is skipped. Last line printed: BEST <size>; a
larger size is taken only if it is at least 3% faster than the smallest."""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np

ROOT = os.environ.get("CGW_ROOT", "/home/johns/Desktop/CognitiveGridworld")


def worker(out, net, micro, episodes):
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    spec = importlib.util.spec_from_file_location("ttd", os.path.join(ROOT, "main/REPLICATION/trainings/train_training_dynamics.py"))
    ttd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ttd)
    import torch
    ttd.OUT = out
    os.environ["TD_MICRO"] = micro
    steps = []
    step = ttd.Lean.forward_backward

    def timed(self):
        t = time.time()
        r = step(self)
        torch.cuda.synchronize()
        if not self.test_set:
            steps.append(time.time() - t)
        return r
    ttd.Lean.forward_backward = timed
    ttd.main(net, 0, episodes=int(episodes), seed=12345)
    print("RESULT", float(np.median(steps[2:])), torch.cuda.max_memory_reserved() // 2**20)


def main():
    net = sys.argv[1]
    sizes = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1000,2000").split(",")]   # 1000 the prior (John 11-05: about 12 h per run faster than 250 on his other machine)
    episodes = sys.argv[3] if len(sys.argv) > 3 else "16"
    speed = {}
    total = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"], capture_output=True, text=True).stdout.split()[0])
    limit = .88 * total                                                           # a multi-day run has no resume: leave room
    for m in sizes:
        tmp = tempfile.mkdtemp(dir=os.path.join(ROOT, "main", "DATA", "experimental"), prefix="_bench_")
        try:
            r = subprocess.run([sys.executable, __file__, "worker", tmp, net, str(m), episodes], capture_output=True, text=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
        if not line:
            print(f"micro {m}: failed ({'out of memory' if 'OutOfMemory' in r.stderr else r.stderr[-200:]})", flush=True)
            continue
        step, peak = float(line[0].split()[1]), int(line[0].split()[2])
        print(f"micro {m}: {step:.3f} s per training step, peak {peak} MB" + ("" if peak <= limit else ", too near the GPU's limit: skipped"), flush=True)
        if peak <= limit:
            speed[m] = step
    base = speed.get(min(sizes))
    best = min(speed, key=speed.get) if speed else min(sizes)
    if base is not None and speed[best] > .97 * base:
        best = min(sizes)
    print("BEST", best)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(*sys.argv[2:6])
    else:
        main()
