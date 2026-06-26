"""
Plot search progress from a run log.

Reads a logs/<problem>_<ts>.log and plots, per iteration:
  * best score ever seen   (cumulative max of all valid rollout scores)
  * avg of the top-3 scores ever seen  (cumulative)

Both curves are computed from the SAME pool of valid scores (the `value=` lines:
the bootstrap seeds plus every valid rollout), so the top-3 average is always
<= the best. Scores are read at the log's 4-decimal precision.

Usage:
    python plot_progress.py [path/to/run.log] [out.png]
"""

import re
import sys

import matplotlib
matplotlib.use("Agg")            # headless: write a file, no display needed
import matplotlib.pyplot as plt

DEFAULT_LOG = "/work/mohammad/BanditEvolve/logs/circle_packing_20260625_202129.log"
SOTA = 2.635983                  # known best sum of radii for n=26 (config target)

ITER_START = re.compile(r"^\[iter (\d+)\].*parents picked")   # start of iter N's rollouts
ITER_BEST = re.compile(r"^\[iter (\d+)\] best=")             # iter N completed (summary)
VALUE = re.compile(r"value=([0-9]+\.[0-9]+)")                # a valid rollout / seed score


def parse(path):
    """Return (per_iter, completed):
      per_iter[i] = list of valid scores produced during iter i (seeds -> -1)
      completed   = set of iters that printed a `best=` summary line
    """
    per_iter = {}
    completed = set()
    cur = -1                     # seeds / bootstrap land before iter 0
    with open(path) as f:
        for line in f:
            m = ITER_START.match(line)
            if m:
                cur = int(m.group(1))
                continue
            b = ITER_BEST.match(line)
            if b:
                completed.add(int(b.group(1)))
                continue          # summary line carries best=/raw=, not a score
            for v in VALUE.findall(line):
                per_iter.setdefault(cur, []).append(float(v))
    return per_iter, completed


def progress(per_iter, completed):
    """Cumulative best and top-3 average over completed iterations."""
    pool = list(per_iter.get(-1, []))        # seed scores = the pre-iter-0 baseline
    xs, best, top3 = [], [], []
    for it in sorted(completed):
        pool.extend(per_iter.get(it, []))
        if not pool:
            continue
        top = sorted(pool, reverse=True)
        xs.append(it)
        best.append(top[0])
        top3.append(sum(top[:3]) / min(3, len(top)))
    return xs, best, top3


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG
    out = sys.argv[2] if len(sys.argv) > 2 else "progress.png"

    per_iter, completed = parse(path)
    xs, best, top3 = progress(per_iter, completed)
    if not xs:
        print(f"no completed iterations parsed from {path}")
        return

    n_scores = sum(len(v) for v in per_iter.values())
    print(f"parsed {len(xs)} iters / {n_scores} valid scores from {path}")
    print(f"final best ever    = {best[-1]:.4f}")
    print(f"final top-3 avg    = {top3[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(xs, best, "-o", ms=3, lw=1.6, label="best ever seen")
    ax.plot(xs, top3, "-s", ms=3, lw=1.6, label="avg of top-3 ever seen")
    ax.axhline(SOTA, ls="--", lw=1.4, color="lightcoral", label=f"SOTA = {SOTA}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("sum of radii")
    ax.set_title(f"Search progress  ({path.split('/')[-1]})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
