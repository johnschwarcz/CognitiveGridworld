"""Var(z)-decoupling test (11-14; train_bayes_generator.py tasks 123-252, Joint's answers, 8,000 episodes): does the small-d_E
failure come from Var(z) = 1/d_E? Arms: none = unscaled, env = the environment's z x sqrt(d_E/30), both = also the
Generator's z_hat (at d_E 30 the factor is 1, so its "none" runs stand for every arm).

Transfer = normalized novel accuracy n = (plugin - zblind) / (joint - zblind) >= the cut at the run's last test, all three
logged per test (zblind = Bayes with the likelihood averaged over z: a Generator that ignores z_hat, the floor). Per cell:
mean n over seeds (the number of seeds in brackets). First |S| = the smallest |S| from which every larger |S| of the row
transfers. Two cuts are reported: .2, the floor-dependent rule adopted on 11-14 after the first results (John: a threshold
relative to the floor, which is .10 to .20 depending on the world), and .5, the cut fixed before any run.

Pre-registered (11-14, before any run), first |S| with n >= .5 under scaling:
  H1  |S|* grows with d_E at fixed Var(z) (BBP/spanning bound ~3 d_E): d_E 10 -> 50, 20 -> 75, 30 -> 100, 90 -> 500
      (d_E 90 at |S| 250 is knife-edge: the fitted constant puts its bound at 230-306)
  H2  a Var(z)-only barrier (~90-100) plus spanning (~2.2 d_E): 10 -> 100, 20 -> 100, 30 -> 100, 90 -> 250
      (d_E 90 at |S| 125 stays partial, as unscaled)
Both: d_E 10 transfers by |S| 125 and d_E 20 moves left of 250; otherwise the phase-folding account fails. The model arm is
expected to match the env arm within one |S| step (the theory is about the target, not the Generator's initial z-gain).

    python zscale_table.py [--pull] [--logs DIR] [--episodes 8000]
"""
import os, re, glob, json, argparse, subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LOGS = os.path.join(ROOT, "main", "DATA", "experimental", "bayes_gen_zscale")
S_GRID, DIMS, ARMS = (50, 75, 100, 125, 250, 500, 1000), (10, 20, 30, 90), ("none", "env", "both")
PRED = {"H1": {10: 50, 20: 75, 30: 100, 90: 500}, "H2": {10: 100, 20: 100, 30: 100, 90: 250}}
CUTS = ((.2, "adopted 11-14, floor-dependent"), (.5, "pre-registered"))


def pull(logs):
    """the runs live in ~/cgw_zscale on the cluster, apart from ~/cgw (zscale_progress.py pulls the same way)"""
    os.makedirs(logs, exist_ok=True)
    subprocess.run(["bash", os.path.join(ROOT, "main", "REPLICATION", "trainings", "cluster.sh"), "pull",
                    "cgw_zscale/runs/bayes_gen/./joint_S*_d*_s*_z*.jsonl", logs])


def load(logs, episodes):
    """{(arm, d_E, |S|): [(seed, n, novel plugin, train plugin), ...]} from the finished runs"""
    cells = {}
    for f in glob.glob(os.path.join(logs, "joint_S*_d*_s*_z*.jsonl")):
        m = re.search(r"joint_S(\d+)_d(\d+)_s(\d+)_z(none|env|both)\.jsonl$", f)
        recs = [json.loads(l) for l in open(f) if l.strip()] if m else []
        if not recs or recs[-1]["ep"] < episodes:
            continue
        r = recs[-1]
        n = (r["novel_plugin"] - r["novel_zblind"]) / max(r["novel_joint"] - r["novel_zblind"], 1e-6)
        cells.setdefault((m[4], int(m[2]), int(m[1])), []).append((int(m[3]), n, r["novel_plugin"], r["train_plugin"]))
    return cells


def first_transfer(means, cut):
    """smallest tested |S| from which every larger tested |S| has mean n >= cut (None: never)"""
    tested = [S for S in S_GRID if S in means]
    for i, S in enumerate(tested):
        if all(means[T] >= cut for T in tested[i:]):
            return S
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--logs", default=LOGS)
    ap.add_argument("--episodes", type=int, default=8000)
    a = ap.parse_args()
    if a.pull:
        pull(a.logs)
    cells = load(a.logs, a.episodes)
    print(f"{sum(map(len, cells.values()))} finished runs in {a.logs}\n")
    means = {}
    for arm in ARMS:
        print(f"arm {arm}: mean n (seeds)" + "".join(f"{S:>11d}" for S in S_GRID))
        for d in DIMS:
            src = "none" if d == 30 else arm                                   # factor 1 at d_E 30
            means[arm, d], row = {}, ""
            for S in S_GRID:
                v = cells.get((src, d, S), [])
                if v:
                    means[arm, d][S] = float(np.mean([x[1] for x in v]))
                row += f"{means[arm, d][S]:8.2f}({len(v)})" if v else f"{'-':>11s}"
            print(f"  d_E {d:3d}{' ' * 14}{row}")
        print()
    for cut, what in CUTS:
        firsts = {(arm, d): first_transfer(means[arm, d], cut) if means[arm, d] else "-" for arm in ARMS for d in DIMS}
        print(f"cut n >= {cut:g} ({what}): first transferring |S| by d_E " + "; ".join(
            f"{arm} " + "/".join(str(firsts[arm, d]) for d in DIMS) for arm in ARMS))
        for arm in ("env", "both"):
            for h, pred in PRED.items():
                hits = [d for d in DIMS if firsts[arm, d] == pred[d]]
                print(f"  {arm:4s} vs {h}: first |S| matches in {len(hits)}/4 rows {hits}  (predicted {pred})")
        core = [firsts["env", 10] not in (None, "-") and firsts["env", 10] <= 125,
                firsts["env", 20] not in (None, "-") and firsts["env", 20] < 250]
        print(f"  core (env arm): d_E 10 transfers by 125: {core[0]}; d_E 20 moves left of 250: {core[1]}\n")


if __name__ == "__main__":
    main()
